from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .batch_size import get_batch_size_tag, tag_directory_path, tag_name
from .constants import DEFAULT_BATCH_SIZE, DEFAULT_GROUP_COL, DEFAULT_SEED
from .data_utils import PreparedModelingFrame, attach_numeric_imputation_values, prepare_modeling_frame
from .model import CVRankNetModel, normalize_batch_norm_mode, normalize_dropout_mode
from .split import load_fold_manifest_split, load_npz_split


ALLOWED_DEVICE_VALUES = {"auto", "cpu", "cuda", "mps"}


@dataclass(frozen=True)
class PreparedSplit:
    train_idx: np.ndarray
    eval_idx: np.ndarray
    split_mode: str
    split_name: str
    fold_index: int | None = None


class RTSTabularDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        cls_labels: np.ndarray,
        rank_labels: np.ndarray,
        years: np.ndarray,
        clusters: np.ndarray,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.cls_labels = torch.tensor(cls_labels, dtype=torch.float32).view(-1, 1)
        self.rank_labels = torch.tensor(rank_labels, dtype=torch.float32).view(-1, 1)
        self.years = torch.tensor(years, dtype=torch.float32).view(-1, 1)
        self.clusters = torch.tensor(clusters, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.features[idx],
            self.cls_labels[idx],
            self.rank_labels[idx],
            self.years[idx],
            self.clusters[idx],
        )


def normalize_device(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text not in ALLOWED_DEVICE_VALUES:
        raise ValueError(f"Invalid device {value!r}. Expected one of: {sorted(ALLOWED_DEVICE_VALUES)}")
    return text


def _select_auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_device(config: dict[str, Any] | None = None) -> torch.device:
    requested: str | None = None
    if isinstance(config, dict):
        raw_device = config.get("device")
        if raw_device not in (None, ""):
            requested = normalize_device(raw_device)
        elif "use_cuda" in config:
            requested = "cuda" if bool(config.get("use_cuda", False)) else "cpu"
    if requested is None:
        requested = "cpu"

    if requested == "auto":
        return _select_auto_device()
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn("Requested device `cuda` is unavailable; falling back to CPU.", UserWarning, stacklevel=2)
        return torch.device("cpu")
    if requested == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        warnings.warn("Requested device `mps` is unavailable; falling back to CPU.", UserWarning, stacklevel=2)
        return torch.device("cpu")
    return torch.device("cpu")


def get_device_metadata(config: dict[str, Any] | None = None) -> dict[str, str]:
    resolved = resolve_torch_device(config)
    requested = "cpu"
    if isinstance(config, dict):
        raw_device = config.get("device")
        if raw_device not in (None, ""):
            requested = normalize_device(raw_device)
        elif "use_cuda" in config:
            requested = "cuda" if bool(config.get("use_cuda", False)) else "cpu"
    return {
        "device_requested": requested,
        "device_resolved": resolved.type,
    }


def prepare_frame(config: dict[str, Any]) -> tuple[pd.DataFrame, str, str, str | None, str | None, str, list[str]]:
    prepared = prepare_frame_details(config)
    return (
        prepared.frame,
        prepared.route,
        prepared.label_col,
        prepared.rank_col,
        prepared.cluster_col,
        prepared.year_col,
        list(prepared.feature_cols),
    )


def prepare_frame_details(
    config: dict[str, Any],
    *,
    include_rank_in_required: bool | None = None,
    feature_preparation_artifact: dict[str, Any] | None = None,
    fill_numeric_imputation: bool = False,
) -> PreparedModelingFrame:
    return prepare_modeling_frame(
        config,
        include_rank_in_required=include_rank_in_required,
        feature_preparation_artifact=feature_preparation_artifact,
        fill_numeric_imputation=fill_numeric_imputation,
    )


def prepare_frame_metadata(prepared: PreparedModelingFrame) -> dict[str, Any]:
    return prepared.metadata()


def merge_prepared_metadata(payload: dict[str, Any], prepared: PreparedModelingFrame) -> dict[str, Any]:
    return {**payload, **prepare_frame_metadata(prepared)}


def resolve_split_from_config(
    df: pd.DataFrame,
    config: dict[str, Any],
    group_col: str = DEFAULT_GROUP_COL,
    label_col: str = "Label_Cls",
) -> PreparedSplit:
    fold_index_raw = config.get("fold_index")
    fold_index = int(fold_index_raw) if fold_index_raw is not None else None

    split_manifest_path = config.get("split_manifest_path")
    if isinstance(split_manifest_path, str) and split_manifest_path.strip() and os.path.exists(split_manifest_path):
        if fold_index is None:
            raise ValueError("Single-fold split resolution requires explicit `fold_index`.")
        split_map = load_fold_manifest_split(split_manifest_path, fold_index=fold_index)
        train_idx = split_map.get("train")
        eval_idx = split_map.get("val")
        split_name = "val"
        if eval_idx is None or len(eval_idx) == 0:
            eval_idx = split_map.get("test")
            split_name = "test"
        if train_idx is None or eval_idx is None:
            raise ValueError(f"Fold manifest must contain train and val/test rows: {split_manifest_path}")
        return PreparedSplit(train_idx=train_idx, eval_idx=eval_idx, split_mode="fold_manifest", split_name=split_name, fold_index=fold_index)

    fixed_split_path = config.get("fixed_split_path")
    if isinstance(fixed_split_path, str) and fixed_split_path.strip() and os.path.exists(fixed_split_path):
        split_map = load_npz_split(fixed_split_path)
        train_idx = split_map.get("train_idx")
        eval_idx = split_map.get("test_idx")
        if train_idx is None or eval_idx is None:
            raise ValueError(f"Fixed split file must include train_idx and test_idx: {fixed_split_path}")
        return PreparedSplit(train_idx=train_idx, eval_idx=eval_idx, split_mode="fixed_npz", split_name="test", fold_index=None)

    groups = df[group_col].astype(str).to_numpy()
    y = df[label_col].to_numpy()
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=float(config.get("test_size", 0.2)),
        random_state=int(config.get("seed", DEFAULT_SEED)),
    )
    train_idx, eval_idx = next(gss.split(np.zeros((len(df), 1)), y, groups=groups))
    return PreparedSplit(
        train_idx=np.asarray(train_idx, dtype=int),
        eval_idx=np.asarray(eval_idx, dtype=int),
        split_mode="random_group_shuffle",
        split_name="test",
        fold_index=None,
    )


def default_output_root(config: dict[str, Any]) -> Path:
    output_root = Path(config.get("output_root", "./data/output"))
    return tag_directory_path(output_root, get_batch_size_tag(config))


def default_model_path(config: dict[str, Any]) -> Path:
    default_path = default_output_root(config) / tag_name("model.pth", get_batch_size_tag(config))
    return Path(config.get("model_path") or default_path)


def default_scaler_path(config: dict[str, Any]) -> Path:
    default_path = default_output_root(config) / tag_name("scaler.pkl", get_batch_size_tag(config))
    return Path(config.get("scaler_path") or default_path)


def default_metrics_path(config: dict[str, Any]) -> Path:
    default_path = default_output_root(config) / tag_name("metrics.json", get_batch_size_tag(config))
    return Path(config.get("metrics_path") or default_path)


def default_validation_metrics_path(config: dict[str, Any]) -> Path:
    default_path = default_output_root(config) / tag_name("validation_metrics.json", get_batch_size_tag(config))
    return Path(config.get("validation_metrics_path") or default_path)


def default_history_path(config: dict[str, Any]) -> Path:
    metrics_path = default_metrics_path(config)
    return Path(config.get("history_path") or metrics_path.with_name(metrics_path.stem + "_history.csv"))


def default_feature_preparation_path(config: dict[str, Any]) -> Path:
    scaler_path = default_scaler_path(config)
    default_path = scaler_path.with_name(scaler_path.stem + "_feature_preparation.json")
    return Path(config.get("feature_preparation_path") or default_path)


def default_fold_runs_manifest_path(config: dict[str, Any]) -> Path:
    default_path = default_output_root(config) / tag_name("fold_runs_manifest.json", get_batch_size_tag(config))
    return Path(config.get("fold_runs_manifest_path") or default_path)


def load_fold_runs_manifest(config_or_path: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(config_or_path, dict):
        manifest_path = default_fold_runs_manifest_path(config_or_path)
    else:
        manifest_path = Path(config_or_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Fold runs manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def try_load_fold_runs_manifest(config: dict[str, Any]) -> dict[str, Any] | None:
    manifest_path = default_fold_runs_manifest_path(config)
    if not manifest_path.exists():
        return None
    return load_fold_runs_manifest(manifest_path)


def build_fold_training_config(config: dict[str, Any], route: str, fold_index: int) -> dict[str, Any]:
    batch_size_tag = get_batch_size_tag(config)
    fold_root = default_output_root(config) / "folds" / f"fold_{fold_index}"
    model_dir = fold_root / "model"
    prefix = f"{route}_fold_{fold_index}"
    tagged_prefix = tag_name(prefix, batch_size_tag)

    fold_config = dict(config)
    fold_config.pop("fold_indices", None)
    fold_config["fold_index"] = int(fold_index)
    fold_config["output_root"] = str(fold_root)
    fold_config["model_path"] = str(model_dir / f"{tagged_prefix}_model.pth")
    fold_config["scaler_path"] = str(model_dir / f"{tagged_prefix}_scaler.pkl")
    fold_config["metrics_path"] = str(model_dir / f"{tagged_prefix}_train_metrics.json")
    fold_config["validation_metrics_path"] = str(model_dir / f"{tagged_prefix}_validation_metrics.json")
    fold_config["history_path"] = str(model_dir / f"{tagged_prefix}_train_history.csv")
    fold_config["feature_preparation_path"] = str(model_dir / f"{tagged_prefix}_feature_preparation.json")
    fold_config["fold_runs_manifest_path"] = str(default_fold_runs_manifest_path(config))
    return fold_config


def concat_output_batches(outputs_list: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not outputs_list:
        return {
            "y_prob": np.asarray([], dtype=float),
            "y_true": np.asarray([], dtype=float),
            "y_rank": np.asarray([], dtype=float),
            "years": np.asarray([], dtype=float),
            "clusters": np.asarray([], dtype=float),
        }
    keys = ["y_prob", "y_true", "y_rank", "years", "clusters"]
    merged: dict[str, np.ndarray] = {}
    for key in keys:
        merged[key] = np.concatenate([np.asarray(item.get(key, []), dtype=float) for item in outputs_list]).astype(float)
    return merged


def numeric_mean_std(values: list[Any]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=0))


def build_model(input_dim: int, config: dict[str, Any]) -> CVRankNetModel:
    hidden_dims = config.get("hidden_dims", (128, 64, 32))
    if isinstance(hidden_dims, list):
        hidden_dims = tuple(int(v) for v in hidden_dims)
    dropout = float(config.get("dropout", 0.2))
    dropout_mode = normalize_dropout_mode(config.get("dropout_mode", "all"))
    batch_norm_mode = normalize_batch_norm_mode(config.get("batch_norm_mode", "all"))
    return CVRankNetModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        dropout_mode=dropout_mode,
        batch_norm_mode=batch_norm_mode,
    )


def prepare_loaders(
    df: pd.DataFrame,
    config: dict[str, Any],
    feature_cols: list[str],
    label_col: str,
    rank_col: str | None,
    cluster_col: str | None,
    year_col: str,
    split: PreparedSplit,
    scaler: StandardScaler | None = None,
) -> tuple[DataLoader, DataLoader, StandardScaler, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    y_cls = df[label_col].fillna(-1).to_numpy(dtype=float)
    y_rank = df[rank_col].fillna(0).to_numpy(dtype=float) if rank_col is not None else np.zeros(len(df), dtype=float)
    years = df[year_col].to_numpy(dtype=float) if year_col in df.columns else np.zeros(len(df), dtype=float)
    clusters = df[cluster_col].to_numpy(dtype=float) if cluster_col is not None else np.zeros(len(df), dtype=float)

    index_lookup = pd.Series(np.arange(len(df), dtype=int), index=df.index)

    def _resolve_split_positions(split_indices: np.ndarray, split_label: str) -> np.ndarray:
        requested = np.asarray(split_indices, dtype=int)
        mapped = index_lookup.reindex(requested)
        if mapped.notna().all():
            return mapped.astype(int).to_numpy()
        if split.split_mode != "fold_manifest" and requested.size > 0 and requested.min() >= 0 and requested.max() < len(df):
            return requested.astype(int)
        missing = requested[mapped.isna().to_numpy()]
        preview = ", ".join(str(int(idx)) for idx in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(
            f"{split.split_mode} {split_label} indices reference rows missing from prepared dataframe: "
            f"count={len(missing)}, row_index_preview=[{preview}{suffix}]"
        )

    train_idx = _resolve_split_positions(split.train_idx, "train")
    eval_idx = _resolve_split_positions(split.eval_idx, split.split_name)
    if train_idx.size == 0 or eval_idx.size == 0:
        raise ValueError("Split indices cannot be empty.")

    working_df = df.copy()
    fitted_feature_preparation: dict[str, Any] | None = None
    input_artifact = config.get("feature_preparation_artifact")
    if isinstance(input_artifact, dict):
        numeric_imputation_values: dict[str, float] = {}
        for feature_name, meta in input_artifact.get("numeric_imputation", {}).items():
            if feature_name not in working_df.columns:
                raise ValueError(f"Imputed feature `{feature_name}` not found in dataframe.")
            fill_value = meta.get("fill_value")
            if fill_value is None:
                train_values = pd.to_numeric(working_df.iloc[train_idx][feature_name], errors="coerce")
                finite_values = train_values[np.isfinite(train_values)]
                if finite_values.size == 0:
                    raise ValueError(f"Cannot fit median imputer for feature `{feature_name}`: no finite train values.")
                fill_value = float(np.median(finite_values))
            else:
                fill_value = float(fill_value)
            working_df[feature_name] = pd.to_numeric(working_df[feature_name], errors="coerce").fillna(fill_value)
            numeric_imputation_values[feature_name] = float(fill_value)
        fitted_feature_preparation = attach_numeric_imputation_values(input_artifact, numeric_imputation_values)

    x = working_df[feature_cols].to_numpy(dtype=float)
    x_train, x_eval = x[train_idx], x[eval_idx]
    active_scaler = scaler if scaler is not None else StandardScaler()
    if scaler is None:
        x_train_scaled = active_scaler.fit_transform(x_train)
    else:
        x_train_scaled = active_scaler.transform(x_train)
    x_eval_scaled = active_scaler.transform(x_eval)

    train_ds = RTSTabularDataset(x_train_scaled, y_cls[train_idx], y_rank[train_idx], years[train_idx], clusters[train_idx])
    eval_ds = RTSTabularDataset(x_eval_scaled, y_cls[eval_idx], y_rank[eval_idx], years[eval_idx], clusters[eval_idx])

    requested_batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    batch_norm_mode = normalize_batch_norm_mode(config.get("batch_norm_mode", "all"))
    if len(train_ds) < 2 and batch_norm_mode != "none":
        raise ValueError("Training split is too small for an MLP that uses BatchNorm.")
    effective_batch_size = max(2, min(requested_batch_size, len(train_ds)))
    drop_last = len(train_ds) > effective_batch_size and (len(train_ds) % effective_batch_size == 1)
    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, drop_last=drop_last)
    eval_loader = DataLoader(eval_ds, batch_size=max(1, min(effective_batch_size, len(eval_ds))), shuffle=False)
    return train_loader, eval_loader, active_scaler, y_cls, y_rank, years, clusters, x, fitted_feature_preparation


def has_split_manifest(config: dict[str, Any]) -> bool:
    split_manifest_path = config.get("split_manifest_path")
    if not isinstance(split_manifest_path, (str, Path)):
        return False
    manifest_path = Path(split_manifest_path)
    return bool(str(manifest_path).strip()) and manifest_path.exists()


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return out
