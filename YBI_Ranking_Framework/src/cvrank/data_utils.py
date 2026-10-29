from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import copy
import json
import os
import random
import warnings

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - optional for split/config-only commands
    torch = None

from .constants import (
    DEFAULT_GROUP_COL,
    DEFAULT_LABEL_COL,
    DEFAULT_RANK_COL,
    DEFAULT_YEAR_COL,
    FEATURE_COLS,
    RANK_LOSS_MODE_DISABLED,
)
from .rank_loss_mode import get_rank_loss_mode

DEFAULT_INPUT_NODATA_VALUES = (-9999.0,)
FEATURE_PREPARATION_ARTIFACT_VERSION = 1
DEFAULT_CATEGORICAL_MISSING_TOKEN = "missing"
DEFAULT_ONE_HOT_FEATURES = frozenset({"Vegetation_type"})
DEFAULT_MEDIAN_IMPUTED_FEATURES = frozenset({"Soil", "GIC"})


@dataclass(frozen=True)
class PreparedModelingFrame:
    frame: pd.DataFrame
    route: str
    group_col: str
    label_col: str
    rank_col: str | None
    cluster_col: str | None
    year_col: str
    raw_feature_cols: tuple[str, ...]
    feature_cols: tuple[str, ...]
    required_columns: tuple[str, ...]
    raw_row_count: int
    dropped_row_indices: tuple[int, ...]
    dropped_rows_missing_required: int
    dropped_rows_missing_required_raw_na: int
    dropped_rows_missing_required_input_nodata: int
    dropped_rows_invalid_numeric: int
    dropped_row_indices_missing_required: tuple[int, ...]
    dropped_row_indices_missing_required_raw_na: tuple[int, ...]
    dropped_row_indices_missing_required_input_nodata: tuple[int, ...]
    dropped_row_indices_invalid_numeric: tuple[int, ...]
    rank_input_status: str
    rank_aux_effective: bool
    input_nodata_values: tuple[float, ...]
    data_quality_warnings: tuple[str, ...]
    feature_preparation_artifact: dict[str, Any]

    @property
    def dropped_rows(self) -> int:
        return len(self.dropped_row_indices)

    @property
    def dropped_row_indices_preview(self) -> tuple[int, ...]:
        return self.dropped_row_indices[:10]

    def metadata(self) -> dict[str, Any]:
        return {
            "raw_row_count": self.raw_row_count,
            "rank_input_status": self.rank_input_status,
            "rank_aux_effective": self.rank_aux_effective,
            "data_quality_warnings": list(self.data_quality_warnings),
            "dropped_rows_missing_required": self.dropped_rows_missing_required,
            "dropped_rows_missing_required_raw_na": self.dropped_rows_missing_required_raw_na,
            "dropped_rows_missing_required_input_nodata": self.dropped_rows_missing_required_input_nodata,
            "dropped_rows_invalid_numeric": self.dropped_rows_invalid_numeric,
            "dropped_row_indices_missing_required_preview": list(self.dropped_row_indices_missing_required[:10]),
            "dropped_row_indices_missing_required_raw_na_preview": list(
                self.dropped_row_indices_missing_required_raw_na[:10]
            ),
            "dropped_row_indices_missing_required_input_nodata_preview": list(
                self.dropped_row_indices_missing_required_input_nodata[:10]
            ),
            "dropped_row_indices_invalid_numeric_preview": list(self.dropped_row_indices_invalid_numeric[:10]),
            "input_nodata_values": list(self.input_nodata_values),
            "dropped_row_indices_preview": list(self.dropped_row_indices_preview),
            "raw_feature_cols": list(self.raw_feature_cols),
            "feature_cols": list(self.feature_cols),
            "feature_preparation_artifact": copy.deepcopy(self.feature_preparation_artifact),
        }


def load_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path))


def ensure_required_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_sample_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\s+", "_", regex=True).str.lower()


def prepare_dataframe(
    df: pd.DataFrame,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    ensure_required_columns(df, [group_col, year_col, label_col])
    out = df.copy()
    out[group_col] = normalize_sample_id(out[group_col])
    out[year_col] = out[year_col].astype(str).str.strip()
    out[label_col] = out[label_col].astype(str).str.strip()
    return out


def resolve_feature_columns(route: str, df: pd.DataFrame | None = None) -> list[str]:
    return list(FEATURE_COLS) if df is None else [col for col in FEATURE_COLS if col in df.columns]


def _normalize_route(route: Any) -> str:
    return str(route or "dbscan").strip().lower()


def _resolve_label_col(config: dict[str, Any], df: pd.DataFrame) -> str:
    label_col = str(config.get("label_col", DEFAULT_LABEL_COL))
    if label_col not in df.columns:
        raise ValueError(f"Label column `{label_col}` not found in data.")
    return label_col


def _resolve_rank_input(
    route: str,
    config: dict[str, Any],
    df: pd.DataFrame,
) -> tuple[str | None, str, bool, list[str]]:
    requested_use_rank_aux = bool(config.get("use_rank_aux", True))
    if not requested_use_rank_aux or get_rank_loss_mode(config) == RANK_LOSS_MODE_DISABLED:
        return None, "disabled", False, []

    rank_col = str(config.get("rank_col", DEFAULT_RANK_COL))
    if rank_col in df.columns:
        return rank_col, "available", True, []

    message = (
        f"Route `{route}` is missing rank column `{rank_col}`; "
        "disabling rank auxiliary inputs for this run."
    )
    return None, "missing_degraded", False, [message]


def _resolve_group_col(config: dict[str, Any], df: pd.DataFrame) -> str:
    group_col = str(config.get("group_col", DEFAULT_GROUP_COL))
    if group_col not in df.columns:
        raise ValueError(f"Group column `{group_col}` not found in data.")
    return group_col


def _resolve_cluster_col(route: str, config: dict[str, Any], df: pd.DataFrame) -> str | None:
    default = "Basin_cluster" if route == "basin" else "Cluster"
    cluster_col = str(config.get("cluster_col", default))
    if cluster_col in df.columns:
        return cluster_col
    return None


def _resolve_feature_cols(route: str, config: dict[str, Any], df: pd.DataFrame) -> list[str]:
    configured = config.get("feature_cols")
    if isinstance(configured, (list, tuple)) and configured:
        cols = [str(col) for col in configured]
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing configured feature columns: {missing}")
        return cols
    return resolve_feature_columns(route, df)


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]:
    return {column: pd.to_numeric(df[column], errors="coerce") for column in columns}


def _invalid_numeric_value_mask(raw: pd.Series, numeric: pd.Series) -> pd.Series:
    """Mark present values that cannot be represented as finite model inputs."""
    return raw.notna() & (numeric.isna() | ~np.isfinite(numeric))


def _resolve_input_nodata_values(config: dict[str, Any]) -> tuple[float, ...]:
    configured = config.get("input_nodata_values", DEFAULT_INPUT_NODATA_VALUES)
    if configured is None:
        return ()
    if isinstance(configured, (list, tuple, set)):
        raw_values = list(configured)
    else:
        raw_values = [configured]
    resolved: list[float] = []
    for value in raw_values:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            raise ValueError(f"Configured input nodata value is not numeric: {value!r}")
        resolved.append(float(parsed))
    return tuple(dict.fromkeys(resolved))


def _mask_input_nodata_values(
    numeric_series: pd.Series,
    input_nodata_values: tuple[float, ...],
) -> tuple[pd.Series, pd.Series]:
    nodata_mask = pd.Series(False, index=numeric_series.index)
    if not input_nodata_values:
        return numeric_series, nodata_mask
    for nodata_value in input_nodata_values:
        nodata_mask |= numeric_series.eq(nodata_value)
    return numeric_series.mask(nodata_mask, np.nan), nodata_mask


def _sanitize_token_for_column(token: str) -> str:
    return (
        str(token)
        .strip()
        .replace("-", "neg_")
        .replace(".", "p")
        .replace(" ", "_")
        .replace("/", "_")
    )


def _canonicalize_categorical_token(value: Any) -> str:
    numeric = float(value)
    if not np.isfinite(numeric):
        return DEFAULT_CATEGORICAL_MISSING_TOKEN
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-9:
        return str(int(rounded))
    return format(numeric, ".12g")


def _build_categorical_output_column_name(feature_name: str, token: str) -> str:
    return f"{feature_name}__{_sanitize_token_for_column(token)}"


def _fit_feature_preparation_artifact(
    source: pd.DataFrame,
    raw_feature_cols: list[str],
    input_nodata_values: tuple[float, ...],
) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "version": FEATURE_PREPARATION_ARTIFACT_VERSION,
        "raw_feature_cols": list(raw_feature_cols),
        "input_nodata_values": list(input_nodata_values),
        "categorical": {},
        "numeric_imputation": {},
        "final_feature_cols": [],
    }
    final_feature_cols: list[str] = []

    for feature_name in raw_feature_cols:
        if feature_name in DEFAULT_ONE_HOT_FEATURES:
            numeric_values = pd.to_numeric(source[feature_name], errors="coerce")
            masked_values, _ = _mask_input_nodata_values(numeric_values, input_nodata_values)
            observed_tokens = [
                _canonicalize_categorical_token(value)
                for value in masked_values.dropna().tolist()
            ]
            categories = sorted(set(observed_tokens), key=lambda item: (item == DEFAULT_CATEGORICAL_MISSING_TOKEN, item))
            if DEFAULT_CATEGORICAL_MISSING_TOKEN not in categories:
                categories.append(DEFAULT_CATEGORICAL_MISSING_TOKEN)
            output_columns = [_build_categorical_output_column_name(feature_name, token) for token in categories]
            artifact["categorical"][feature_name] = {
                "kind": "one_hot",
                "categories": categories,
                "missing_token": DEFAULT_CATEGORICAL_MISSING_TOKEN,
                "output_columns": output_columns,
            }
            final_feature_cols.extend(output_columns)
            continue

        if feature_name in DEFAULT_MEDIAN_IMPUTED_FEATURES:
            missing_indicator_col = f"{feature_name}_missing"
            artifact["numeric_imputation"][feature_name] = {
                "strategy": "median",
                "missing_indicator_col": missing_indicator_col,
                "fill_value": None,
            }
            final_feature_cols.extend([feature_name, missing_indicator_col])
            continue

        final_feature_cols.append(feature_name)

    artifact["final_feature_cols"] = final_feature_cols
    return artifact


def _resolve_feature_preparation_artifact(
    source: pd.DataFrame,
    raw_feature_cols: list[str],
    input_nodata_values: tuple[float, ...],
    feature_preparation_artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if feature_preparation_artifact is None:
        return _fit_feature_preparation_artifact(source, raw_feature_cols, input_nodata_values)

    artifact = copy.deepcopy(feature_preparation_artifact)
    expected_raw_features = list(artifact.get("raw_feature_cols", []))
    if expected_raw_features and expected_raw_features != list(raw_feature_cols):
        raise ValueError(
            "Feature preparation artifact raw_feature_cols do not match requested feature columns: "
            f"{expected_raw_features} != {list(raw_feature_cols)}"
        )
    artifact.setdefault("raw_feature_cols", list(raw_feature_cols))
    artifact.setdefault("input_nodata_values", list(input_nodata_values))
    artifact.setdefault("categorical", {})
    artifact.setdefault("numeric_imputation", {})
    artifact.setdefault("final_feature_cols", list(raw_feature_cols))
    return artifact


def load_feature_preparation_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path)
    with artifact_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_feature_preparation_artifact(path: str | Path, artifact: dict[str, Any]) -> Path:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8") as fh:
        json.dump(artifact, fh, ensure_ascii=False, indent=2, default=str)
    return artifact_path


def attach_numeric_imputation_values(
    artifact: dict[str, Any],
    numeric_imputation_values: dict[str, float],
) -> dict[str, Any]:
    updated = copy.deepcopy(artifact)
    for feature_name, fill_value in numeric_imputation_values.items():
        if feature_name not in updated.get("numeric_imputation", {}):
            continue
        updated["numeric_imputation"][feature_name]["fill_value"] = float(fill_value)
    return updated


def apply_feature_preparation_artifact(
    source: pd.DataFrame,
    raw_feature_cols: list[str],
    input_nodata_values: tuple[float, ...],
    artifact: dict[str, Any],
    *,
    fill_numeric_imputation: bool,
) -> tuple[pd.DataFrame, tuple[str, ...], pd.Series, pd.Series, pd.Series]:
    transformed = pd.DataFrame(index=source.index)
    invalid_numeric_mask = pd.Series(False, index=source.index)
    missing_required_raw_na_mask = pd.Series(False, index=source.index)
    missing_required_input_nodata_mask = pd.Series(False, index=source.index)

    categorical_features = artifact.get("categorical", {})
    numeric_imputation_features = artifact.get("numeric_imputation", {})

    for feature_name in raw_feature_cols:
        raw_numeric = pd.to_numeric(source[feature_name], errors="coerce")
        invalid_numeric_mask |= _invalid_numeric_value_mask(source[feature_name], raw_numeric)
        normalized_feature, nodata_mask = _mask_input_nodata_values(raw_numeric, input_nodata_values)

        if feature_name in categorical_features:
            categorical_meta = categorical_features[feature_name]
            categories = list(categorical_meta["categories"])
            missing_token = str(categorical_meta.get("missing_token", DEFAULT_CATEGORICAL_MISSING_TOKEN))
            output_columns = list(categorical_meta["output_columns"])

            category_tokens = pd.Series(missing_token, index=source.index, dtype="object")
            valid_mask = normalized_feature.notna()
            if valid_mask.any():
                category_tokens.loc[valid_mask] = normalized_feature.loc[valid_mask].map(_canonicalize_categorical_token)

            for token, output_column in zip(categories, output_columns):
                transformed[output_column] = category_tokens.eq(token).astype(float)
            continue

        transformed[feature_name] = normalized_feature

        if feature_name in numeric_imputation_features:
            missing_indicator_col = str(numeric_imputation_features[feature_name]["missing_indicator_col"])
            transformed[missing_indicator_col] = transformed[feature_name].isna().astype(float)
            if fill_numeric_imputation:
                fill_value = numeric_imputation_features[feature_name].get("fill_value")
                if fill_value is None or not np.isfinite(fill_value):
                    raise ValueError(f"Missing fitted imputation value for feature `{feature_name}`.")
                transformed[feature_name] = transformed[feature_name].fillna(float(fill_value))
            continue

        missing_required_raw_na_mask |= source[feature_name].isna()
        missing_required_input_nodata_mask |= (~source[feature_name].isna()) & nodata_mask

    final_feature_cols = tuple(artifact.get("final_feature_cols", list(transformed.columns)))
    missing_final_cols = [column for column in final_feature_cols if column not in transformed.columns]
    if missing_final_cols:
        raise ValueError(f"Feature preparation artifact is missing generated columns: {missing_final_cols}")
    transformed = transformed.loc[:, list(final_feature_cols)].copy()
    return (
        transformed,
        final_feature_cols,
        invalid_numeric_mask,
        missing_required_raw_na_mask,
        missing_required_input_nodata_mask,
    )


def prepare_modeling_frame(
    config: dict[str, Any],
    df: pd.DataFrame | None = None,
    *,
    include_rank_in_required: bool | None = None,
    feature_preparation_artifact: dict[str, Any] | None = None,
    fill_numeric_imputation: bool = False,
) -> PreparedModelingFrame:
    route = _normalize_route(config.get("route", "dbscan"))
    source = load_table(config["data_path"]) if df is None else df.copy()

    label_col = _resolve_label_col(config, source)
    group_col = _resolve_group_col(config, source)
    year_col = str(config.get("year_col", DEFAULT_YEAR_COL))
    if year_col not in source.columns:
        raise ValueError(f"Year column `{year_col}` not found in data.")
    rank_col, rank_input_status, rank_aux_effective, warning_messages = _resolve_rank_input(route, config, source)
    if include_rank_in_required is None:
        include_rank = rank_aux_effective
    else:
        include_rank = bool(include_rank_in_required) and rank_col is not None
    cluster_col = _resolve_cluster_col(route, config, source)
    use_cluster_in_rank_loss = bool(
        rank_aux_effective and config.get("use_cluster_in_rank_loss", False)
    )
    if use_cluster_in_rank_loss and cluster_col is None:
        configured_cluster_col = str(
            config.get("cluster_col", "Basin_cluster" if route == "basin" else "Cluster")
        )
        raise ValueError(
            "use_cluster_in_rank_loss=true requires cluster column "
            f"`{configured_cluster_col}`, but it was not found in the data."
        )
    if not use_cluster_in_rank_loss:
        cluster_col = None
    raw_feature_cols = _resolve_feature_cols(route, config, source)

    required = [group_col, year_col, label_col, *raw_feature_cols]
    if include_rank:
        required.append(rank_col)
    if cluster_col is not None:
        required.append(cluster_col)
    required = list(dict.fromkeys(required))

    ensure_required_columns(source, required)
    input_nodata_values = _resolve_input_nodata_values(config)

    numeric_columns = [label_col, year_col, *raw_feature_cols]
    if include_rank:
        numeric_columns.append(rank_col)
    if cluster_col is not None:
        numeric_columns.append(cluster_col)
    numeric_columns = list(dict.fromkeys(numeric_columns))
    coerced_columns_raw = _coerce_numeric_columns(source, numeric_columns)

    feature_preparation = _resolve_feature_preparation_artifact(
        source,
        raw_feature_cols,
        input_nodata_values,
        feature_preparation_artifact=feature_preparation_artifact,
    )
    (
        transformed_features,
        final_feature_cols,
        feature_invalid_numeric_mask,
        feature_missing_required_raw_na_mask,
        feature_missing_required_input_nodata_mask,
    ) = apply_feature_preparation_artifact(
        source,
        raw_feature_cols,
        input_nodata_values,
        feature_preparation,
        fill_numeric_imputation=fill_numeric_imputation,
    )

    required_value_columns = [group_col, year_col, label_col]
    if include_rank and rank_col is not None:
        required_value_columns.append(rank_col)
    if cluster_col is not None:
        required_value_columns.append(cluster_col)
    required_value_columns = list(dict.fromkeys(required_value_columns))
    missing_required_raw_na_mask = (
        source[required_value_columns].isna().any(axis=1)
        | feature_missing_required_raw_na_mask
    )
    missing_required_input_nodata_mask = feature_missing_required_input_nodata_mask.copy()

    invalid_numeric_mask = feature_invalid_numeric_mask.copy()
    for column in [label_col, year_col]:
        invalid_numeric_mask |= _invalid_numeric_value_mask(source[column], coerced_columns_raw[column])
    if include_rank and rank_col is not None:
        invalid_numeric_mask |= _invalid_numeric_value_mask(source[rank_col], coerced_columns_raw[rank_col])
    if cluster_col is not None:
        invalid_numeric_mask |= _invalid_numeric_value_mask(source[cluster_col], coerced_columns_raw[cluster_col])

    missing_required_mask = missing_required_raw_na_mask | missing_required_input_nodata_mask
    invalid_numeric_only_mask = (~missing_required_mask) & invalid_numeric_mask
    kept_mask = ~(missing_required_mask | invalid_numeric_only_mask)
    dropped_row_indices = tuple(int(idx) for idx in source.index[~kept_mask].tolist())
    dropped_row_indices_missing_required = tuple(int(idx) for idx in source.index[missing_required_mask].tolist())
    dropped_row_indices_missing_required_raw_na = tuple(int(idx) for idx in source.index[missing_required_raw_na_mask].tolist())
    dropped_row_indices_missing_required_input_nodata = tuple(
        int(idx) for idx in source.index[missing_required_input_nodata_mask].tolist()
    )
    dropped_row_indices_invalid_numeric = tuple(int(idx) for idx in source.index[invalid_numeric_only_mask].tolist())

    dropped_rows_missing_required = int(missing_required_mask.sum())
    dropped_rows_missing_required_raw_na = int(missing_required_raw_na_mask.sum())
    dropped_rows_missing_required_input_nodata = int(missing_required_input_nodata_mask.sum())
    dropped_rows_invalid_numeric = int(invalid_numeric_only_mask.sum())
    if dropped_rows_missing_required_raw_na > 0:
        warning_messages.append(
            "Dropped "
            f"{dropped_rows_missing_required_raw_na} rows because required inputs contain missing values."
        )
    if dropped_rows_missing_required_input_nodata > 0:
        warning_messages.append(
            "Dropped "
            f"{dropped_rows_missing_required_input_nodata} rows because feature inputs match configured nodata values "
            f"{list(input_nodata_values)}."
        )
    if dropped_rows_invalid_numeric > 0:
        warning_messages.append(
            "Dropped "
            f"{dropped_rows_invalid_numeric} rows because required numeric inputs could not be converted to finite values."
        )

    for message in warning_messages:
        warnings.warn(message, UserWarning, stacklevel=2)

    clean = source.loc[kept_mask].copy()
    required_numeric_columns = [label_col, year_col]
    if include_rank and rank_col is not None:
        required_numeric_columns.append(rank_col)
    if cluster_col is not None:
        required_numeric_columns.append(cluster_col)
    remaining_invalid = {
        column: int(
            (
                coerced_columns_raw[column].loc[kept_mask].isna()
                | ~np.isfinite(coerced_columns_raw[column].loc[kept_mask])
            ).sum()
        )
        for column in dict.fromkeys(required_numeric_columns)
    }
    remaining_invalid = {column: count for column, count in remaining_invalid.items() if count > 0}
    if remaining_invalid:
        raise RuntimeError(
            "Internal data-cleaning invariant violated: required numeric columns still contain "
            f"missing or non-finite values: {remaining_invalid}"
        )
    clean[group_col] = normalize_sample_id(clean[group_col])
    clean[label_col] = coerced_columns_raw[label_col].loc[kept_mask]
    if include_rank and rank_col is not None:
        clean[rank_col] = coerced_columns_raw[rank_col].loc[kept_mask]
    if cluster_col is not None:
        clean[cluster_col] = coerced_columns_raw[cluster_col].loc[kept_mask]
    clean[year_col] = coerced_columns_raw[year_col].loc[kept_mask].astype(int)
    for feature_col in final_feature_cols:
        clean[feature_col] = transformed_features.loc[kept_mask, feature_col]

    return PreparedModelingFrame(
        frame=clean,
        route=route,
        group_col=group_col,
        label_col=label_col,
        rank_col=rank_col,
        cluster_col=cluster_col,
        year_col=year_col,
        raw_feature_cols=tuple(raw_feature_cols),
        feature_cols=tuple(final_feature_cols),
        required_columns=tuple(required),
        raw_row_count=int(len(source)),
        dropped_row_indices=dropped_row_indices,
        dropped_rows_missing_required=dropped_rows_missing_required,
        dropped_rows_missing_required_raw_na=dropped_rows_missing_required_raw_na,
        dropped_rows_missing_required_input_nodata=dropped_rows_missing_required_input_nodata,
        dropped_rows_invalid_numeric=dropped_rows_invalid_numeric,
        dropped_row_indices_missing_required=dropped_row_indices_missing_required,
        dropped_row_indices_missing_required_raw_na=dropped_row_indices_missing_required_raw_na,
        dropped_row_indices_missing_required_input_nodata=dropped_row_indices_missing_required_input_nodata,
        dropped_row_indices_invalid_numeric=dropped_row_indices_invalid_numeric,
        rank_input_status=rank_input_status,
        rank_aux_effective=rank_aux_effective,
        input_nodata_values=input_nodata_values,
        data_quality_warnings=tuple(warning_messages),
        feature_preparation_artifact=feature_preparation,
    )


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
