from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import (
    DEFAULT_GROUP_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_N_SPLITS,
    DEFAULT_SEED,
    DEFAULT_SPATIAL_BLOCK_ORIGIN_X_M,
    DEFAULT_SPATIAL_BLOCK_ORIGIN_Y_M,
    DEFAULT_SPATIAL_MAX_FOLD_SIZE_IMBALANCE,
    DEFAULT_SPATIAL_MIN_BLOCKS_PER_FOLD,
    DEFAULT_SPATIAL_SOURCE_CRS,
    DEFAULT_SPATIAL_TARGET_CRS,
    DEFAULT_SPATIAL_VARIOGRAM_MAX_PAIRS,
    DEFAULT_SPATIAL_VARIOGRAM_MIN_PAIRS_PER_LAG,
    DEFAULT_SPATIAL_VARIOGRAM_N_LAGS,
    DEFAULT_SPATIAL_VARIOGRAM_SAMPLE_SIZE,
    DEFAULT_SPATIAL_X_COL,
    DEFAULT_SPATIAL_Y_COL,
    DEFAULT_SPLIT_STRATEGY,
    DEFAULT_YEAR_COL,
    SPLIT_STRATEGY_STRATIFIED_GROUP_KFOLD,
    SPLIT_STRATEGY_SPATIAL_BLOCK_CV,
    SUPPORTED_SPLIT_STRATEGIES,
)
from .data_utils import (
    DEFAULT_INPUT_NODATA_VALUES,
    PreparedModelingFrame,
    load_table,
    normalize_sample_id,
    prepare_dataframe,
    prepare_modeling_frame,
)


QTP_ALBERS_PROJ4 = "+proj=aea +lat_1=27.5 +lat_2=37.5 +lat_0=0 +lon_0=90 +datum=WGS84 +units=m +no_defs"


@dataclass(frozen=True)
class FoldArtifacts:
    fold_index: int
    train_rows: np.ndarray
    val_rows: np.ndarray


@dataclass(frozen=True)
class FoldManifestBuildResult:
    manifest: pd.DataFrame
    prepared: PreparedModelingFrame
    dropped_rows_audit: pd.DataFrame
    split_metadata: dict[str, Any]


@dataclass(frozen=True)
class SpatialBlockPlan:
    block_ids: np.ndarray
    x_projected_m: np.ndarray
    y_projected_m: np.ndarray
    block_size_m: float
    metadata: dict[str, Any]


def normalize_split_strategy(value: Any) -> str:
    text = str(value or DEFAULT_SPLIT_STRATEGY).strip().lower()
    aliases = {
        "stratified_group": "stratified_group_kfold",
        "spatial": SPLIT_STRATEGY_SPATIAL_BLOCK_CV,
        "spatial_block": SPLIT_STRATEGY_SPATIAL_BLOCK_CV,
    }
    normalized = aliases.get(text, text)
    if normalized not in SUPPORTED_SPLIT_STRATEGIES:
        raise ValueError(
            f"Unsupported split_strategy `{value}`. Expected one of: {sorted(SUPPORTED_SPLIT_STRATEGIES)}"
        )
    return normalized


def build_stratify_key(
    df: pd.DataFrame,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.Series:
    years = df[year_col].astype(str).str.strip()
    labels = df[label_col].astype(str).str.strip()
    return years + "__" + labels


def build_group_array(df: pd.DataFrame, group_col: str = DEFAULT_GROUP_COL) -> np.ndarray:
    return normalize_sample_id(df[group_col]).to_numpy()


def _make_stratified_group_folds(
    prepared: pd.DataFrame,
    *,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    year_col: str,
    label_col: str,
    grouping_name: str,
) -> list[FoldArtifacts]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Spatial/group CV requires at least {n_splits} distinct {grouping_name} groups, "
            f"but only {len(unique_groups)} are available."
        )

    stratify_key = build_stratify_key(prepared, year_col=year_col, label_col=label_col)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    artifacts: list[FoldArtifacts] = []

    try:
        for fold_index, (_, val_idx) in enumerate(splitter.split(prepared, stratify_key, groups)):
            train_idx = np.setdiff1d(np.arange(len(prepared)), val_idx, assume_unique=False)
            artifacts.append(
                FoldArtifacts(
                    fold_index=fold_index,
                    train_rows=train_idx,
                    val_rows=np.asarray(val_idx, dtype=int),
                )
            )
    except ValueError as exc:
        raise ValueError(
            f"Unable to construct {n_splits}-fold {grouping_name} cross-validation with "
            "year-plus-label stratification. Check the number of occupied groups and strata."
        ) from exc

    return artifacts


def make_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> list[FoldArtifacts]:
    prepared = prepare_dataframe(df, group_col=group_col, year_col=year_col, label_col=label_col)
    groups = build_group_array(prepared, group_col=group_col)
    return _make_stratified_group_folds(
        prepared,
        groups=groups,
        n_splits=n_splits,
        seed=seed,
        year_col=year_col,
        label_col=label_col,
        grouping_name="sample",
    )


def _resolve_spatial_crs(value: Any) -> str:
    text = str(value or DEFAULT_SPATIAL_TARGET_CRS).strip()
    if text.lower() in {"qtp_albers", "qtp-albers", "albers_qtp"}:
        return QTP_ALBERS_PROJ4
    return text


def _project_spatial_coordinates(
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    source_crs: Any,
    target_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from pyproj import Transformer
    except ImportError as exc:  # pragma: no cover - dependency is declared in pyproject.toml
        raise ImportError("Spatial Block CV requires the `pyproj` dependency.") from exc

    x_numeric = pd.to_numeric(x_values, errors="coerce").to_numpy(dtype=float)
    y_numeric = pd.to_numeric(y_values, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_numeric) & np.isfinite(y_numeric)
    if not valid.all():
        raise ValueError(
            "Spatial Block CV requires finite coordinates for every modeling row; "
            f"invalid_rows={int((~valid).sum())}."
        )

    transformer = Transformer.from_crs(
        str(source_crs or DEFAULT_SPATIAL_SOURCE_CRS),
        _resolve_spatial_crs(target_crs),
        always_xy=True,
    )
    x_projected, y_projected = transformer.transform(x_numeric, y_numeric)
    x_projected = np.asarray(x_projected, dtype=float)
    y_projected = np.asarray(y_projected, dtype=float)
    projected_valid = np.isfinite(x_projected) & np.isfinite(y_projected)
    if not projected_valid.all():
        raise ValueError("Spatial coordinate projection produced non-finite values.")
    return x_projected, y_projected


def _is_categorical_variogram_feature(column: str) -> bool:
    normalized = str(column).strip().lower()
    return (
        normalized in {"aspect", "vegetation_type"}
        or normalized.startswith("aspect__")
        or normalized.startswith("vegetation_type__")
        or normalized.endswith("__missing")
        or normalized.endswith("_missing")
    )


def _resolve_variogram_feature_columns(
    df: pd.DataFrame,
    *,
    config: dict[str, Any],
    feature_cols: list[str] | None,
    excluded_columns: set[str],
) -> tuple[list[str], list[str]]:
    configured = config.get("spatial_variogram_feature_cols")
    if configured is None:
        candidates = list(feature_cols or [])
        if not candidates:
            candidates = [
                column
                for column in df.columns
                if column not in excluded_columns and pd.api.types.is_numeric_dtype(df[column])
            ]
    elif isinstance(configured, (list, tuple)):
        candidates = [str(column) for column in configured]
    else:
        raise ValueError("`spatial_variogram_feature_cols` must be a list when provided.")

    missing = [column for column in candidates if column not in df.columns]
    if missing:
        raise ValueError(f"Spatial variogram feature columns are missing from data: {missing}")

    excluded = []
    resolved = []
    for column in candidates:
        if column in excluded_columns or _is_categorical_variogram_feature(column):
            excluded.append(column)
            continue
        if not pd.api.types.is_numeric_dtype(df[column]):
            excluded.append(column)
            continue
        resolved.append(column)
    return resolved, excluded


def _resolve_variogram_nodata_values(config: dict[str, Any]) -> list[float]:
    configured = config.get("input_nodata_values", DEFAULT_INPUT_NODATA_VALUES)
    if configured is None:
        return []
    raw_values = list(configured) if isinstance(configured, (list, tuple, set)) else [configured]
    resolved: list[float] = []
    for value in raw_values:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            raise ValueError(f"Configured input nodata value is not numeric: {value!r}")
        resolved.append(float(parsed))
    return list(dict.fromkeys(resolved))


def _mask_variogram_nodata(values: pd.Series, config: dict[str, Any]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    for nodata_value in _resolve_variogram_nodata_values(config):
        numeric = numeric.mask(numeric.eq(nodata_value), np.nan)
    return numeric


def estimate_empirical_variogram_range(
    x_projected_m: np.ndarray,
    y_projected_m: np.ndarray,
    values: np.ndarray,
    *,
    n_lags: int = DEFAULT_SPATIAL_VARIOGRAM_N_LAGS,
    max_pairs: int = DEFAULT_SPATIAL_VARIOGRAM_MAX_PAIRS,
    min_pairs_per_lag: int = DEFAULT_SPATIAL_VARIOGRAM_MIN_PAIRS_PER_LAG,
    max_distance_m: float | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[float | None, dict[str, Any]]:
    """Estimate a practical spatial range from an empirical semivariogram.

    The range is the first lag reaching 95% of a robust tail sill estimate;
    pair sampling and lag support limits keep the calculation bounded.
    """
    if int(n_lags) < 3:
        raise ValueError("`spatial_variogram_n_lags` must be >= 3.")
    if int(max_pairs) < 3:
        raise ValueError("`spatial_variogram_max_pairs` must be >= 3.")
    if int(min_pairs_per_lag) < 1:
        raise ValueError("`spatial_variogram_min_pairs_per_lag` must be >= 1.")

    x = np.asarray(x_projected_m, dtype=float).reshape(-1)
    y = np.asarray(y_projected_m, dtype=float).reshape(-1)
    z = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    if not (len(x) == len(y) == len(z)):
        raise ValueError("Variogram coordinate and value arrays must have the same length.")

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[valid], y[valid], z[valid]
    n_observations = int(len(z))
    base_metadata: dict[str, Any] = {
        "n_observations": n_observations,
        "n_lags": int(n_lags),
        "max_pairs": int(max_pairs),
        "min_pairs_per_lag": int(min_pairs_per_lag),
    }
    if n_observations < 4 or not np.isfinite(np.var(z)) or float(np.var(z)) <= 1e-12:
        return None, {**base_metadata, "status": "insufficient_variation"}

    pair_count = n_observations * (n_observations - 1) // 2
    rng = np.random.default_rng(int(seed))
    if pair_count <= int(max_pairs):
        left, right = np.triu_indices(n_observations, k=1)
    else:
        left = rng.integers(0, n_observations, size=int(max_pairs), endpoint=False)
        right = rng.integers(0, n_observations, size=int(max_pairs), endpoint=False)
        valid_pairs = left != right
        left, right = left[valid_pairs], right[valid_pairs]

    distances = np.hypot(x[left] - x[right], y[left] - y[right])
    semivariances = 0.5 * np.square(z[left] - z[right])
    finite_pairs = np.isfinite(distances) & np.isfinite(semivariances) & (distances > 0.0)
    distances = distances[finite_pairs]
    semivariances = semivariances[finite_pairs]
    if len(distances) < max(3, int(n_lags)):
        return None, {**base_metadata, "status": "insufficient_pairs", "n_pairs": int(len(distances))}

    if max_distance_m is None:
        max_distance = float(np.nanquantile(distances, 0.9))
    else:
        max_distance = float(max_distance_m)
    if not np.isfinite(max_distance) or max_distance <= 0.0:
        return None, {**base_metadata, "status": "invalid_max_distance", "n_pairs": int(len(distances))}

    within_range = distances <= max_distance
    distances = distances[within_range]
    semivariances = semivariances[within_range]
    if len(distances) < max(3, int(n_lags)):
        return None, {**base_metadata, "status": "insufficient_pairs_in_range", "n_pairs": int(len(distances))}

    lag_edges = np.linspace(0.0, max_distance, int(n_lags) + 1)
    lag_indices = np.searchsorted(lag_edges, distances, side="right") - 1
    lag_indices = np.clip(lag_indices, 0, int(n_lags) - 1)
    counts = np.bincount(lag_indices, minlength=int(n_lags))
    sums = np.bincount(lag_indices, weights=semivariances, minlength=int(n_lags))
    usable = counts >= int(min_pairs_per_lag)
    if int(usable.sum()) < 3:
        return None, {
            **base_metadata,
            "status": "insufficient_lag_support",
            "n_pairs": int(len(distances)),
            "usable_lags": int(usable.sum()),
        }

    lag_centres = 0.5 * (lag_edges[:-1] + lag_edges[1:])
    gamma = sums[usable] / counts[usable]
    lag_centres = lag_centres[usable]
    finite_gamma = np.isfinite(gamma) & np.isfinite(lag_centres)
    gamma = gamma[finite_gamma]
    lag_centres = lag_centres[finite_gamma]
    if len(gamma) < 3:
        return None, {**base_metadata, "status": "insufficient_finite_lags"}

    tail_start = max(0, len(gamma) - max(3, int(np.ceil(len(gamma) * 0.25))))
    sill = float(np.nanmedian(gamma[tail_start:]))
    if not np.isfinite(sill) or sill <= 1e-12:
        return None, {**base_metadata, "status": "invalid_sill"}

    cumulative_gamma = np.maximum.accumulate(gamma)
    target = 0.95 * sill
    reached = np.flatnonzero(cumulative_gamma >= target)
    range_index = int(reached[0]) if len(reached) else len(lag_centres) - 1
    range_m = float(lag_centres[range_index])
    return range_m, {
        **base_metadata,
        "status": "ok",
        "n_pairs": int(len(distances)),
        "usable_lags": int(len(gamma)),
        "max_distance_m": max_distance,
        "sill": sill,
        "target": target,
        "range_m": range_m,
    }


def build_spatial_block_plan(
    df: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    group_col: str = DEFAULT_GROUP_COL,
    feature_cols: list[str] | None = None,
) -> SpatialBlockPlan:
    """Project coordinates, estimate a block size, and assign site-level blocks."""
    cfg = dict(config or {})
    x_col = str(cfg.get("spatial_x_col", DEFAULT_SPATIAL_X_COL))
    y_col = str(cfg.get("spatial_y_col", DEFAULT_SPATIAL_Y_COL))
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            "Spatial Block CV requires coordinate columns "
            f"`{x_col}` and `{y_col}` in the modeling table."
        )
    if group_col not in df.columns:
        raise ValueError(f"Spatial Block CV requires group column `{group_col}`.")

    x_projected, y_projected = _project_spatial_coordinates(
        df[x_col],
        df[y_col],
        source_crs=cfg.get("spatial_source_crs", DEFAULT_SPATIAL_SOURCE_CRS),
        target_crs=cfg.get("spatial_target_crs", DEFAULT_SPATIAL_TARGET_CRS),
    )

    normalized_groups = normalize_sample_id(df[group_col]).to_numpy()
    site_frame = pd.DataFrame(
        {
            group_col: normalized_groups,
            "__spatial_x_m": x_projected,
            "__spatial_y_m": y_projected,
        },
        index=df.index,
    )
    excluded_columns = {
        group_col,
        str(cfg.get("year_col", DEFAULT_YEAR_COL)),
        str(cfg.get("label_col", DEFAULT_LABEL_COL)),
        str(cfg.get("rank_col", "Label_Rank")),
        str(cfg.get("cluster_col", "Cluster")),
        "Cluster",
        "Basin_cluster",
        "Occurrence_Year",
        x_col,
        y_col,
    }
    variogram_cols, excluded_variogram_cols = _resolve_variogram_feature_columns(
        df,
        config=cfg,
        feature_cols=feature_cols,
        excluded_columns=excluded_columns,
    )
    for column in variogram_cols:
        site_frame[column] = _mask_variogram_nodata(df[column], cfg).to_numpy(dtype=float)

    aggregation = {"__spatial_x_m": "mean", "__spatial_y_m": "mean"}
    aggregation.update({column: "mean" for column in variogram_cols})
    site_frame = site_frame.groupby(group_col, sort=True, as_index=False).agg(aggregation)
    site_lookup = site_frame.set_index(group_col)
    site_x_projected = site_lookup.loc[normalized_groups, "__spatial_x_m"].to_numpy(dtype=float)
    site_y_projected = site_lookup.loc[normalized_groups, "__spatial_y_m"].to_numpy(dtype=float)

    sample_size = int(cfg.get("spatial_variogram_sample_size", DEFAULT_SPATIAL_VARIOGRAM_SAMPLE_SIZE))
    if sample_size < 4:
        raise ValueError("`spatial_variogram_sample_size` must be >= 4.")
    if len(site_frame) > sample_size:
        sample_rng = np.random.default_rng(int(cfg.get("seed", DEFAULT_SEED)))
        sample_indices = sample_rng.choice(len(site_frame), size=sample_size, replace=False)
        variogram_frame = site_frame.iloc[np.sort(sample_indices)].reset_index(drop=True)
    else:
        variogram_frame = site_frame.reset_index(drop=True)

    variogram_ranges: dict[str, float] = {}
    variogram_diagnostics: dict[str, dict[str, Any]] = {}
    for column in variogram_cols:
        range_m, diagnostics = estimate_empirical_variogram_range(
            variogram_frame["__spatial_x_m"].to_numpy(dtype=float),
            variogram_frame["__spatial_y_m"].to_numpy(dtype=float),
            variogram_frame[column].to_numpy(dtype=float),
            n_lags=int(cfg.get("spatial_variogram_n_lags", DEFAULT_SPATIAL_VARIOGRAM_N_LAGS)),
            max_pairs=int(cfg.get("spatial_variogram_max_pairs", DEFAULT_SPATIAL_VARIOGRAM_MAX_PAIRS)),
            min_pairs_per_lag=int(
                cfg.get(
                    "spatial_variogram_min_pairs_per_lag",
                    DEFAULT_SPATIAL_VARIOGRAM_MIN_PAIRS_PER_LAG,
                )
            ),
            max_distance_m=(
                float(cfg["spatial_variogram_max_distance_m"])
                if cfg.get("spatial_variogram_max_distance_m") not in (None, "")
                else None
            ),
            seed=int(cfg.get("seed", DEFAULT_SEED)),
        )
        variogram_diagnostics[column] = diagnostics
        if range_m is not None and np.isfinite(range_m) and range_m > 0.0:
            variogram_ranges[column] = float(range_m)

    configured_block_size = cfg.get("spatial_block_size_m")
    if configured_block_size not in (None, ""):
        block_size_m = float(configured_block_size)
        block_size_source = "configured"
    else:
        if not variogram_ranges:
            raise ValueError(
                "Unable to estimate a positive spatial block size from the empirical variograms. "
                "Provide usable continuous variogram features or set `spatial_block_size_m` explicitly."
            )
        block_size_m = float(np.median(list(variogram_ranges.values())))
        block_size_source = "variogram_median"
    if not np.isfinite(block_size_m) or block_size_m <= 0.0:
        raise ValueError("`spatial_block_size_m` and variogram-derived block size must be finite and > 0.")

    origin_x_m = float(cfg.get("spatial_block_origin_x_m", DEFAULT_SPATIAL_BLOCK_ORIGIN_X_M))
    origin_y_m = float(cfg.get("spatial_block_origin_y_m", DEFAULT_SPATIAL_BLOCK_ORIGIN_Y_M))
    # Assign the block from one site-level coordinate for every row. This keeps
    # repeated multi-year observations of a Sample_ID in the same spatial block,
    # even if their stored coordinates are not bit-for-bit identical.
    block_columns = np.floor((site_x_projected - origin_x_m) / block_size_m).astype(np.int64)
    block_rows = np.floor((site_y_projected - origin_y_m) / block_size_m).astype(np.int64)
    block_ids = np.asarray(
        [f"b{column}_{row}" for column, row in zip(block_columns, block_rows)],
        dtype=object,
    )

    metadata: dict[str, Any] = {
        "split_strategy": SPLIT_STRATEGY_SPATIAL_BLOCK_CV,
        "spatial_x_col": x_col,
        "spatial_y_col": y_col,
        "spatial_source_crs": str(cfg.get("spatial_source_crs", DEFAULT_SPATIAL_SOURCE_CRS)),
        "spatial_target_crs": str(cfg.get("spatial_target_crs", DEFAULT_SPATIAL_TARGET_CRS)),
        "spatial_block_size_m": block_size_m,
        "spatial_block_size_source": block_size_source,
        "spatial_block_origin_x_m": origin_x_m,
        "spatial_block_origin_y_m": origin_y_m,
        "spatial_coordinate_aggregation": "mean_by_group",
        "spatial_block_count": int(pd.Series(block_ids).nunique()),
        "spatial_site_count": int(len(site_frame)),
        "spatial_variogram_sample_size": int(len(variogram_frame)),
        "spatial_variogram_input_nodata_values": _resolve_variogram_nodata_values(cfg),
        "spatial_variogram_feature_cols": list(variogram_cols),
        "spatial_variogram_excluded_cols": list(excluded_variogram_cols),
        "spatial_variogram_ranges_m": variogram_ranges,
        "spatial_variogram_diagnostics": variogram_diagnostics,
    }
    return SpatialBlockPlan(
        block_ids=block_ids,
        x_projected_m=site_x_projected,
        y_projected_m=site_y_projected,
        block_size_m=block_size_m,
        metadata=metadata,
    )


def _spatial_block_ids_from_coordinates(
    x_projected_m: np.ndarray,
    y_projected_m: np.ndarray,
    *,
    block_size_m: float,
    origin_x_m: float,
    origin_y_m: float,
) -> np.ndarray:
    if not np.isfinite(block_size_m) or block_size_m <= 0.0:
        raise ValueError("Spatial block size must be finite and > 0.")
    block_columns = np.floor((np.asarray(x_projected_m, dtype=float) - origin_x_m) / block_size_m).astype(np.int64)
    block_rows = np.floor((np.asarray(y_projected_m, dtype=float) - origin_y_m) / block_size_m).astype(np.int64)
    return np.asarray(
        [f"b{column}_{row}" for column, row in zip(block_columns, block_rows)],
        dtype=object,
    )


def _make_balanced_spatial_block_folds(
    prepared: pd.DataFrame,
    *,
    block_ids: np.ndarray,
    n_splits: int,
    seed: int,
    year_col: str,
    label_col: str,
) -> tuple[list[FoldArtifacts], dict[str, Any]]:
    """Assign whole spatial blocks while balancing year/label strata.

    ``StratifiedGroupKFold`` is intentionally retained for the random strategy,
    but it can assign a few large spatial blocks to very uneven folds.  A
    deterministic greedy assignment gives spatial blocks the same indivisibility
    guarantee while making the fold-size and stratum targets explicit.
    """
    groups = np.asarray(block_ids, dtype=object).reshape(-1)
    if len(groups) != len(prepared):
        raise ValueError("Spatial block IDs must have one value per prepared row.")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Spatial block CV requires at least {n_splits} distinct spatial blocks, "
            f"but only {len(unique_groups)} are available."
        )

    stratify_key = build_stratify_key(prepared, year_col=year_col, label_col=label_col).to_numpy(dtype=str)
    unique_strata, stratum_inverse = np.unique(stratify_key, return_inverse=True)
    group_counts = np.zeros((len(unique_groups), len(unique_strata)), dtype=np.int64)
    np.add.at(group_counts, (inverse, stratum_inverse), 1)
    group_totals = group_counts.sum(axis=1)
    target_strata = group_counts.sum(axis=0).astype(float) / float(n_splits)
    target_total = float(len(prepared)) / float(n_splits)

    # Largest blocks are placed first.  A seeded tie-break keeps repeated runs
    # deterministic without making the result depend on object ordering.
    rng = np.random.default_rng(int(seed))
    tie_break = rng.random(len(unique_groups))
    order = np.lexsort((tie_break, -group_totals))
    fold_strata = np.zeros((n_splits, len(unique_strata)), dtype=np.int64)
    fold_totals = np.zeros(n_splits, dtype=np.int64)
    group_fold = np.full(len(unique_groups), -1, dtype=np.int64)
    stratum_denominator = np.maximum(target_strata, 1.0)

    # Seed every fold with one block before optimizing the remaining blocks.
    # Without this warm start, minimizing the absolute prospective stratum
    # error can prefer already-populated folds while empty folds still have a
    # large target gap; that degenerates into fewer than ``n_splits`` usable
    # validation folds on regular grids.
    for fold_index, group_index in enumerate(order[:n_splits]):
        group_fold[group_index] = fold_index
        fold_strata[fold_index] += group_counts[group_index]
        fold_totals[fold_index] += group_totals[group_index]

    for group_index in order[n_splits:]:
        scores: list[float] = []
        for fold_index in range(n_splits):
            # Score the resulting *whole assignment*, rather than only the
            # candidate fold.  The latter systematically prefers folds that
            # are already populated (they are closer to the target in
            # absolute terms) and can strand empty folds on regular grids.
            prospective_fold_strata = fold_strata.copy()
            prospective_fold_strata[fold_index] += group_counts[group_index]
            prospective_totals = fold_totals.copy()
            prospective_totals[fold_index] += group_totals[group_index]
            stratum_error = np.mean(
                np.square(
                    (prospective_fold_strata.astype(float) - target_strata[None, :])
                    / stratum_denominator[None, :]
                )
            )
            size_error = np.mean(
                np.square(
                    (prospective_totals.astype(float) - target_total)
                    / max(target_total, 1.0)
                )
            )
            scores.append(float(stratum_error + 0.25 * size_error))
        best_score = min(scores)
        best_folds = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-15))
        selected_fold = int(rng.choice(best_folds))
        group_fold[group_index] = selected_fold
        fold_strata[selected_fold] += group_counts[group_index]
        fold_totals[selected_fold] += group_totals[group_index]

    artifacts: list[FoldArtifacts] = []
    for fold_index in range(n_splits):
        val_idx = np.flatnonzero(group_fold[inverse] == fold_index).astype(int)
        train_idx = np.setdiff1d(np.arange(len(prepared)), val_idx, assume_unique=False)
        artifacts.append(
            FoldArtifacts(
                fold_index=fold_index,
                train_rows=train_idx,
                val_rows=val_idx,
            )
        )

    fold_size_imbalance = float(
        (fold_totals.max() - fold_totals.min()) / max(float(fold_totals.mean()), 1.0)
    )
    quality = {
        "spatial_fold_row_counts": [int(value) for value in fold_totals.tolist()],
        "spatial_fold_block_counts": [
            int((group_fold == fold_index).sum()) for fold_index in range(n_splits)
        ],
        "spatial_fold_size_imbalance": fold_size_imbalance,
        "spatial_fold_stratum_counts": {
            str(stratum): [int(value) for value in fold_strata[:, stratum_index].tolist()]
            for stratum_index, stratum in enumerate(unique_strata)
        },
    }
    return artifacts, quality


def _select_spatial_block_plan_and_folds(
    prepared: pd.DataFrame,
    *,
    initial_plan: SpatialBlockPlan,
    config: dict[str, Any] | None,
    n_splits: int,
    seed: int,
    group_col: str,
    year_col: str,
    label_col: str,
) -> tuple[SpatialBlockPlan, list[FoldArtifacts]]:
    """Choose the largest variogram-derived block size with usable fold balance.

    A practical range can exceed the width of a regional study area.  In that
    case a fixed grid produces only a handful of giant blocks and makes a
    nominal 5-fold manifest unusable.  When the size was auto-derived, search
    downward from the variogram range and retain the largest candidate that
    yields enough blocks and balanced year/label strata.  Explicit sizes are
    honored exactly and their diagnostics remain visible in the manifest audit.
    """
    cfg = dict(config or {})
    configured_size = cfg.get("spatial_block_size_m")
    min_blocks_per_fold = int(
        cfg.get("spatial_min_blocks_per_fold", DEFAULT_SPATIAL_MIN_BLOCKS_PER_FOLD)
    )
    max_imbalance = float(
        cfg.get("spatial_max_fold_size_imbalance", DEFAULT_SPATIAL_MAX_FOLD_SIZE_IMBALANCE)
    )
    if min_blocks_per_fold < 1:
        raise ValueError("`spatial_min_blocks_per_fold` must be >= 1.")
    if not np.isfinite(max_imbalance) or max_imbalance < 0.0:
        raise ValueError("`spatial_max_fold_size_imbalance` must be finite and >= 0.")

    origin_x_m = float(initial_plan.metadata["spatial_block_origin_x_m"])
    origin_y_m = float(initial_plan.metadata["spatial_block_origin_y_m"])
    candidate_sizes = [float(initial_plan.block_size_m)]
    if configured_size in (None, ""):
        # Search at 1% resolution.  This preserves the variogram range as the
        # upper bound while allowing the study-area geometry to determine the
        # largest practically balanced block edge.
        candidate_sizes = [
            float(initial_plan.block_size_m) * factor
            for factor in np.arange(1.0, 0.049, -0.01)
        ]

    selected: tuple[float, np.ndarray, list[FoldArtifacts], dict[str, Any]] | None = None
    best_effort: tuple[float, np.ndarray, list[FoldArtifacts], dict[str, Any], float] | None = None
    for candidate_size in candidate_sizes:
        candidate_ids = _spatial_block_ids_from_coordinates(
            initial_plan.x_projected_m,
            initial_plan.y_projected_m,
            block_size_m=candidate_size,
            origin_x_m=origin_x_m,
            origin_y_m=origin_y_m,
        )
        candidate_folds, quality = _make_balanced_spatial_block_folds(
            prepared,
            block_ids=candidate_ids,
            n_splits=n_splits,
            seed=seed,
            year_col=year_col,
            label_col=label_col,
        )
        block_count = int(pd.Series(candidate_ids).nunique())
        quality_score = float(quality["spatial_fold_size_imbalance"])
        if block_count < n_splits * min_blocks_per_fold:
            quality_score += float(n_splits * min_blocks_per_fold - block_count) / max(
                float(n_splits * min_blocks_per_fold), 1.0
            )
        if best_effort is None or quality_score < best_effort[4]:
            best_effort = (candidate_size, candidate_ids, candidate_folds, quality, quality_score)

        fold_has_binary_labels = all(
            {0, 1}.issubset(
                set(
                    prepared.iloc[fold.val_rows][label_col]
                    .astype(int)
                    .tolist()
                )
            )
            for fold in candidate_folds
        )
        feasible = (
            block_count >= n_splits * min_blocks_per_fold
            and quality["spatial_fold_size_imbalance"] <= max_imbalance
            and fold_has_binary_labels
        )
        if feasible:
            selected = (candidate_size, candidate_ids, candidate_folds, quality)
            break

    if selected is None:
        if best_effort is None:
            raise ValueError("Unable to construct any spatial block fold candidate.")
        candidate_size, candidate_ids, candidate_folds, quality, _ = best_effort
        raise ValueError(
            "Unable to construct balanced spatial block folds: "
            f"best_size_m={candidate_size:.3f}, "
            f"fold_size_imbalance={quality['spatial_fold_size_imbalance']:.3f}, "
            f"required_max={max_imbalance:.3f}. "
            "Set `spatial_block_size_m` explicitly or relax spatial fold constraints."
        )

    candidate_size, candidate_ids, candidate_folds, quality = selected
    selected_metadata = dict(initial_plan.metadata)
    raw_range = float(initial_plan.block_size_m)
    if configured_size in (None, "") and not np.isclose(candidate_size, raw_range):
        selected_metadata["spatial_block_size_source"] = "variogram_median_backoff"
    selected_metadata.update(
        {
            "spatial_variogram_practical_range_m": raw_range,
            "spatial_block_size_m": float(candidate_size),
            "spatial_block_size_backoff_factor": float(candidate_size / raw_range),
            "spatial_block_count": int(pd.Series(candidate_ids).nunique()),
            "spatial_min_blocks_per_fold": int(min_blocks_per_fold),
            "spatial_max_fold_size_imbalance": float(max_imbalance),
            **quality,
        }
    )
    return (
        SpatialBlockPlan(
            block_ids=candidate_ids,
            x_projected_m=initial_plan.x_projected_m,
            y_projected_m=initial_plan.y_projected_m,
            block_size_m=float(candidate_size),
            metadata=selected_metadata,
        ),
        candidate_folds,
    )


def make_spatial_block_folds(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
    *,
    config: dict[str, Any] | None = None,
    spatial_plan: SpatialBlockPlan | None = None,
) -> list[FoldArtifacts]:
    """Create stratified folds whose grouping units are spatial blocks."""
    prepared = prepare_dataframe(df, group_col=group_col, year_col=year_col, label_col=label_col)
    plan = spatial_plan or build_spatial_block_plan(
        prepared,
        config=config,
        group_col=group_col,
        feature_cols=(config or {}).get("feature_cols"),
    )
    selection_config = dict(config or {})
    if spatial_plan is not None and selection_config.get("spatial_block_size_m") in (None, ""):
        # A caller that supplies a precomputed plan has already made the block
        # size choice; do not silently back it off just because no config map
        # was passed alongside the plan.
        selection_config["spatial_block_size_m"] = float(plan.block_size_m)
    _, folds = _select_spatial_block_plan_and_folds(
        prepared,
        initial_plan=plan,
        config=selection_config,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        year_col=year_col,
        label_col=label_col,
    )
    return folds


def build_fold_manifest(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
    # Keep the programmatic helper backward-compatible; configured pipeline
    # routes explicitly default to Spatial Block CV.
    split_strategy: str = SPLIT_STRATEGY_STRATIFIED_GROUP_KFOLD,
    spatial_config: dict[str, Any] | None = None,
    variogram_feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    manifest, _ = _build_fold_manifest(
        df,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        year_col=year_col,
        label_col=label_col,
        split_strategy=split_strategy,
        spatial_config=spatial_config,
        variogram_feature_cols=variogram_feature_cols,
    )
    return manifest


def _build_fold_manifest(
    df: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    group_col: str,
    year_col: str,
    label_col: str,
    split_strategy: str,
    spatial_config: dict[str, Any] | None = None,
    variogram_feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prepared = prepare_dataframe(df, group_col=group_col, year_col=year_col, label_col=label_col)
    normalized_strategy = normalize_split_strategy(split_strategy)
    spatial_plan: SpatialBlockPlan | None = None
    if normalized_strategy == SPLIT_STRATEGY_SPATIAL_BLOCK_CV:
        spatial_plan = build_spatial_block_plan(
            prepared,
            config=spatial_config,
            group_col=group_col,
            feature_cols=variogram_feature_cols,
        )
        spatial_plan, folds = _select_spatial_block_plan_and_folds(
            prepared,
            initial_plan=spatial_plan,
            config=spatial_config,
            n_splits=n_splits,
            seed=seed,
            group_col=group_col,
            year_col=year_col,
            label_col=label_col,
        )
        split_metadata = {
            **spatial_plan.metadata,
            "n_splits": int(n_splits),
            "seed": int(seed),
            "group_col": group_col,
            "year_col": year_col,
            "label_col": label_col,
        }
    else:
        folds = make_stratified_group_folds(
            prepared,
            n_splits=n_splits,
            seed=seed,
            group_col=group_col,
            year_col=year_col,
            label_col=label_col,
        )
        split_metadata = {
            "split_strategy": normalized_strategy,
            "n_splits": int(n_splits),
            "seed": int(seed),
            "group_col": group_col,
            "year_col": year_col,
            "label_col": label_col,
        }

    rows: list[dict[str, object]] = []
    group_values = prepared[group_col].to_numpy()
    year_values = prepared[year_col].to_numpy()
    label_values = prepared[label_col].to_numpy()
    for fold in folds:
        for split_name, indices in (("train", fold.train_rows), ("val", fold.val_rows)):
            for idx in indices:
                row_position = int(idx)
                row: dict[str, object] = {
                    "fold": fold.fold_index,
                    "split": split_name,
                    "row_index": int(prepared.index[row_position]),
                    group_col: group_values[row_position],
                    year_col: year_values[row_position],
                    label_col: label_values[row_position],
                    "split_strategy": normalized_strategy,
                }
                if spatial_plan is not None:
                    row.update(
                        {
                            "spatial_block_id": str(spatial_plan.block_ids[row_position]),
                            "spatial_x_m": float(spatial_plan.x_projected_m[row_position]),
                            "spatial_y_m": float(spatial_plan.y_projected_m[row_position]),
                            "spatial_block_size_m": float(spatial_plan.block_size_m),
                        }
                    )
                rows.append(row)

    return pd.DataFrame(rows), split_metadata


def build_fold_manifest_from_config(config: dict[str, Any]) -> FoldManifestBuildResult:
    source = load_table(config["data_path"])
    prepared = prepare_modeling_frame(config, df=source)
    manifest, split_metadata = _build_fold_manifest(
        prepared.frame,
        n_splits=int(config.get("n_splits", DEFAULT_N_SPLITS)),
        seed=int(config.get("seed", DEFAULT_SEED)),
        group_col=prepared.group_col,
        year_col=prepared.year_col,
        label_col=prepared.label_col,
        split_strategy=normalize_split_strategy(config.get("split_strategy", DEFAULT_SPLIT_STRATEGY)),
        spatial_config=config,
        variogram_feature_cols=list(prepared.feature_cols),
    )
    dropped_rows_audit = build_dropped_rows_audit_frame(source, prepared)
    return FoldManifestBuildResult(
        manifest=manifest,
        prepared=prepared,
        dropped_rows_audit=dropped_rows_audit,
        split_metadata=split_metadata,
    )


def build_dropped_rows_audit_frame(source: pd.DataFrame, prepared: PreparedModelingFrame) -> pd.DataFrame:
    reason_columns = [
        "drop_reason_missing_required",
        "drop_reason_missing_required_raw_na",
        "drop_reason_missing_required_input_nodata",
        "drop_reason_invalid_numeric",
    ]
    if not prepared.dropped_row_indices:
        return pd.DataFrame(columns=["row_index", *source.columns.tolist(), *reason_columns])

    audit = source.loc[list(prepared.dropped_row_indices)].copy()
    audit.insert(0, "row_index", audit.index.astype(int))
    missing_required = set(prepared.dropped_row_indices_missing_required)
    missing_required_raw_na = set(prepared.dropped_row_indices_missing_required_raw_na)
    missing_required_input_nodata = set(prepared.dropped_row_indices_missing_required_input_nodata)
    invalid_numeric = set(prepared.dropped_row_indices_invalid_numeric)
    row_index_series = audit["row_index"]
    audit["drop_reason_missing_required"] = row_index_series.isin(missing_required)
    audit["drop_reason_missing_required_raw_na"] = row_index_series.isin(missing_required_raw_na)
    audit["drop_reason_missing_required_input_nodata"] = row_index_series.isin(missing_required_input_nodata)
    audit["drop_reason_invalid_numeric"] = row_index_series.isin(invalid_numeric)
    return audit.reset_index(drop=True)


def build_manifest_cleaning_summary(
    prepared: PreparedModelingFrame,
    manifest: pd.DataFrame,
    *,
    data_path: str | Path | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary = prepared.metadata()
    resolved_split_metadata = dict(split_metadata or {})
    summary.update(
        {
            "route": prepared.route,
            "data_path": str(data_path) if data_path is not None else None,
            "clean_row_count": int(len(prepared.frame)),
            "manifest_row_count": int(len(manifest)),
            "manifest_fold_count": int(manifest["fold"].nunique()) if not manifest.empty and "fold" in manifest.columns else 0,
            "manifest_split_counts": (
                manifest["split"].astype(str).value_counts().sort_index().to_dict()
                if not manifest.empty and "split" in manifest.columns
                else {}
            ),
            "split_metadata": resolved_split_metadata,
        }
    )
    if "split_strategy" in resolved_split_metadata:
        summary["split_strategy"] = resolved_split_metadata["split_strategy"]
    if "spatial_block_size_m" in resolved_split_metadata:
        summary["spatial_block_size_m"] = resolved_split_metadata["spatial_block_size_m"]
    return summary


def derive_manifest_audit_paths(manifest_path: str | Path) -> dict[str, Path]:
    manifest = Path(manifest_path)
    stem = manifest.stem
    return {
        "summary_path": manifest.with_name(f"{stem}_cleaning_summary.json"),
        "dropped_rows_path": manifest.with_name(f"{stem}_dropped_rows.csv"),
    }


def save_manifest_cleaning_audit(
    manifest_path: str | Path,
    prepared: PreparedModelingFrame,
    dropped_rows_audit: pd.DataFrame,
    *,
    data_path: str | Path | None = None,
    manifest: pd.DataFrame | None = None,
    split_metadata: dict[str, Any] | None = None,
) -> dict[str, Path]:
    audit_paths = derive_manifest_audit_paths(manifest_path)
    summary = build_manifest_cleaning_summary(
        prepared,
        manifest if manifest is not None else pd.DataFrame(),
        data_path=data_path,
        split_metadata=split_metadata,
    )
    summary_path = audit_paths["summary_path"]
    dropped_rows_path = audit_paths["dropped_rows_path"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    dropped_rows_audit.to_csv(dropped_rows_path, index=False)
    return audit_paths


def summarize_manifest(
    manifest: pd.DataFrame,
    fold_col: str = "fold",
    split_col: str = "split",
) -> pd.DataFrame:
    grouped = manifest.groupby([fold_col, split_col], dropna=False).size().reset_index(name="count")
    return grouped.sort_values([fold_col, split_col]).reset_index(drop=True)


def save_manifest(manifest: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, index=False)
    return output


def load_fold_manifest_split(
    manifest_path: str | Path,
    fold_index: int = 0,
    fold_col: str = "fold",
    split_col: str = "split",
    row_index_col: str = "row_index",
) -> dict[str, np.ndarray]:
    manifest = pd.read_csv(Path(manifest_path))
    if fold_col not in manifest.columns or split_col not in manifest.columns or row_index_col not in manifest.columns:
        raise ValueError(
            f"Fold manifest must include `{fold_col}`, `{split_col}`, and `{row_index_col}` columns."
        )
    fold_df = manifest[manifest[fold_col] == fold_index].copy()
    if fold_df.empty:
        raise ValueError(f"Fold {fold_index} not found in manifest: {manifest_path}")

    split_map: dict[str, np.ndarray] = {}
    for split_name, group_df in fold_df.groupby(split_col):
        split_map[str(split_name)] = group_df[row_index_col].astype(int).to_numpy()
    return split_map


def list_manifest_fold_indices(
    manifest_path: str | Path,
    fold_col: str = "fold",
) -> list[int]:
    manifest = pd.read_csv(Path(manifest_path))
    if fold_col not in manifest.columns:
        raise ValueError(f"Fold manifest must include `{fold_col}` column.")
    fold_values = pd.to_numeric(manifest[fold_col], errors="coerce").dropna().astype(int).unique().tolist()
    fold_indices = sorted(int(value) for value in fold_values)
    if not fold_indices:
        raise ValueError(f"No folds found in manifest: {manifest_path}")
    return fold_indices


def resolve_requested_fold_indices(
    config: dict[str, Any],
    manifest_path: str | Path | None = None,
) -> list[int] | None:
    raw_fold_indices = config.get("fold_indices")
    if isinstance(raw_fold_indices, (list, tuple, set)) and raw_fold_indices:
        return sorted({int(value) for value in raw_fold_indices})

    raw_fold_index = config.get("fold_index")
    if raw_fold_index is not None:
        return [int(raw_fold_index)]

    candidate_manifest = manifest_path or config.get("split_manifest_path")
    if isinstance(candidate_manifest, (str, Path)):
        manifest = Path(candidate_manifest)
        if manifest.exists():
            return list_manifest_fold_indices(manifest)

    return None


def load_npz_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path)) as data:
        return {key: np.asarray(value, dtype=int) for key, value in data.items()}
