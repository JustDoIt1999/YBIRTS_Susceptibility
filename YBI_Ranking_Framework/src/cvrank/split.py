from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .constants import DEFAULT_GROUP_COL, DEFAULT_LABEL_COL, DEFAULT_N_SPLITS, DEFAULT_SEED, DEFAULT_YEAR_COL
from .data_utils import PreparedModelingFrame, load_table, normalize_sample_id, prepare_dataframe, prepare_modeling_frame


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


def make_stratified_group_folds(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> list[FoldArtifacts]:
    prepared = prepare_dataframe(df, group_col=group_col, year_col=year_col, label_col=label_col)
    stratify_key = build_stratify_key(prepared, year_col=year_col, label_col=label_col)
    groups = build_group_array(prepared, group_col=group_col)

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    artifacts: list[FoldArtifacts] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(prepared, stratify_key, groups)):
        artifacts.append(
            FoldArtifacts(
                fold_index=fold_index,
                train_rows=train_idx,
                val_rows=np.asarray(val_idx, dtype=int),
            )
        )

    return artifacts


def build_fold_manifest(
    df: pd.DataFrame,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    group_col: str = DEFAULT_GROUP_COL,
    year_col: str = DEFAULT_YEAR_COL,
    label_col: str = DEFAULT_LABEL_COL,
) -> pd.DataFrame:
    prepared = prepare_dataframe(df, group_col=group_col, year_col=year_col, label_col=label_col)
    folds = make_stratified_group_folds(
        prepared,
        n_splits=n_splits,
        seed=seed,
        group_col=group_col,
        year_col=year_col,
        label_col=label_col,
    )

    parts: list[pd.DataFrame] = []
    columns = [group_col, year_col, label_col]
    for fold in folds:
        for split_name, indices in (("train", fold.train_rows), ("val", fold.val_rows)):
            part = prepared.iloc[indices][columns].copy()
            part.insert(0, "row_index", part.index.to_numpy(dtype=int))
            part.insert(0, "split", split_name)
            part.insert(0, "fold", fold.fold_index)
            parts.append(part)

    return pd.concat(parts, ignore_index=True)


def build_fold_manifest_from_config(config: dict[str, Any]) -> FoldManifestBuildResult:
    source = load_table(config["data_path"])
    prepared = prepare_modeling_frame(config, df=source)
    manifest = build_fold_manifest(
        prepared.frame,
        n_splits=int(config.get("n_splits", DEFAULT_N_SPLITS)),
        seed=int(config.get("seed", DEFAULT_SEED)),
        group_col=prepared.group_col,
        year_col=prepared.year_col,
        label_col=prepared.label_col,
    )
    dropped_rows_audit = build_dropped_rows_audit_frame(source, prepared)
    return FoldManifestBuildResult(
        manifest=manifest,
        prepared=prepared,
        dropped_rows_audit=dropped_rows_audit,
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
) -> dict[str, Any]:
    summary = prepared.metadata()
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
        }
    )
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
) -> dict[str, Path]:
    audit_paths = derive_manifest_audit_paths(manifest_path)
    summary = build_manifest_cleaning_summary(
        prepared,
        manifest if manifest is not None else pd.DataFrame(),
        data_path=data_path,
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
