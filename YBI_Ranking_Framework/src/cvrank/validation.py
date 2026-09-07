from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .shared import (
    default_feature_preparation_path as _default_feature_preparation_path,
    default_fold_runs_manifest_path as _default_fold_runs_manifest_path,
    load_fold_runs_manifest,
)
from .batch_size import get_batch_size_tag, tag_name
from .constants import RANK_LOSS_MODE_DISABLED
from .data_utils import load_feature_preparation_artifact
from .rank_loss_mode import resolve_effective_rank_loss_mode
from .shared import (
    build_model,
    get_device_metadata,
    has_split_manifest,
    prepare_frame_details,
    prepare_frame_metadata,
    prepare_loaders,
    resolve_torch_device,
    resolve_split_from_config,
)
from .split import resolve_requested_fold_indices

EMPTY_ROUTE_METRICS = {
    "auc": float("nan"),
    "f1": float("nan"),
    "accuracy": float("nan"),
    "kappa": float("nan"),
    "precision": float("nan"),
    "recall": float("nan"),
}


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    valid_mask = np.isin(y_true, [0, 1])
    y_true = np.asarray(y_true)[valid_mask]
    y_prob = np.asarray(y_prob)[valid_mask]
    if y_true.size == 0:
        return {
            "auc": float("nan"),
            "f1": float("nan"),
            "accuracy": float("nan"),
            "kappa": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
        }
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, y_prob)) if np.unique(y_true).size >= 2 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def summarize_metric_dicts(metric_dicts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not metric_dicts:
        return {"mean": {}, "std": {}}

    numeric_values: dict[str, list[float]] = {}
    for metric_dict in metric_dicts:
        for key, value in metric_dict.items():
            if isinstance(value, (int, float, np.floating, np.integer, bool)):
                numeric_values.setdefault(key, []).append(float(value))

    mean_values: dict[str, float] = {}
    std_values: dict[str, float] = {}
    for key, values in numeric_values.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            mean_values[key] = float("nan")
            std_values[key] = float("nan")
            continue
        mean_values[key] = float(np.nanmean(arr))
        std_values[key] = float(np.nanstd(arr, ddof=0))
    return {"mean": mean_values, "std": std_values}




def compute_route_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    *,
    y_rank: np.ndarray | None = None,
    years: np.ndarray | None = None,
    clusters: np.ndarray | None = None,
    include_rank_metrics: bool = True,
    use_cluster_in_rank_loss: bool = False,
) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=float).reshape(-1)
    return compute_binary_metrics(y_true, y_prob, threshold=threshold)


def collect_loader_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    ranks: list[np.ndarray] = []
    years: list[np.ndarray] = []
    clusters: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x_b, y_cls_b, y_rank_b, year_b, cluster_b = batch
            x_b = x_b.to(device)
            _, prob = model(x_b)
            probs.append(prob.detach().cpu().numpy().reshape(-1))
            labels.append(y_cls_b.detach().cpu().numpy().reshape(-1))
            ranks.append(y_rank_b.detach().cpu().numpy().reshape(-1))
            years.append(year_b.detach().cpu().numpy().reshape(-1))
            clusters.append(cluster_b.detach().cpu().numpy().reshape(-1))

    if not probs:
        return {
            "y_prob": np.asarray([], dtype=float),
            "y_true": np.asarray([], dtype=float),
            "y_rank": np.asarray([], dtype=float),
            "years": np.asarray([], dtype=float),
            "clusters": np.asarray([], dtype=float),
        }

    return {
        "y_prob": np.concatenate(probs).astype(float),
        "y_true": np.concatenate(labels).astype(float),
        "y_rank": np.concatenate(ranks).astype(float),
        "years": np.concatenate(years).astype(float),
        "clusters": np.concatenate(clusters).astype(float),
    }


def compute_metrics_from_outputs(
    outputs: dict[str, np.ndarray],
    *,
    threshold: float = 0.5,
    include_rank_metrics: bool = True,
    use_cluster_in_rank_loss: bool = False,
) -> dict[str, Any]:
    y_prob = np.asarray(outputs.get("y_prob", []), dtype=float)
    y_true = np.asarray(outputs.get("y_true", []), dtype=float)
    y_rank = np.asarray(outputs.get("y_rank", []), dtype=float)
    years = np.asarray(outputs.get("years", []), dtype=float)
    clusters = np.asarray(outputs.get("clusters", []), dtype=float)
    if y_prob.size == 0:
        return dict(EMPTY_ROUTE_METRICS)
    return compute_route_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
        y_rank=y_rank,
        years=years,
        clusters=clusters,
        include_rank_metrics=include_rank_metrics,
        use_cluster_in_rank_loss=use_cluster_in_rank_loss,
    )


def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    threshold: float = 0.5,
    include_rank_metrics: bool = True,
    use_cluster_in_rank_loss: bool = False,
) -> dict[str, Any]:
    outputs = collect_loader_outputs(model, loader, device)
    return compute_metrics_from_outputs(
        outputs,
        threshold=threshold,
        include_rank_metrics=include_rank_metrics,
        use_cluster_in_rank_loss=use_cluster_in_rank_loss,
    )


def _load_optional_feature_preparation_artifact(path_value: str | Path | None) -> tuple[dict[str, Any] | None, Path | None]:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return None, None
    artifact_path = Path(path_value).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"Feature preparation artifact not found: {artifact_path}")
    return load_feature_preparation_artifact(artifact_path), artifact_path


def _load_scaler_artifact(path_value: str | Path | None, *, required: bool = False) -> tuple[Any | None, Path | None]:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        if required:
            raise ValueError("A saved scaler artifact is required for multi-fold OOF validation.")
        return None, None
    scaler_path = Path(path_value).resolve()
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler artifact not found: {scaler_path}")
    try:
        with scaler_path.open("rb") as fh:
            scaler = pickle.load(fh)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise ValueError(f"Unable to load scaler artifact: {scaler_path}") from exc
    if not hasattr(scaler, "transform"):
        raise ValueError(f"Scaler artifact does not provide transform(): {scaler_path}")
    return scaler, scaler_path


def run_validation(config: dict[str, Any]) -> dict[str, Any]:
    from .data_utils import seed_everything

    seed_everything(int(config.get("seed", 42)))
    requested_folds = resolve_requested_fold_indices(config, config.get("split_manifest_path"))
    multi_fold_validation = bool(has_split_manifest(config) and requested_folds and len(requested_folds) > 1)

    base_feature_preparation: dict[str, Any] | None = None
    base_feature_preparation_path: Path | None = None
    fold_entries: dict[int, dict[str, Any]] = {}
    if multi_fold_validation:
        fold_manifest = load_fold_runs_manifest(config)
        fold_entries = {int(item["fold_index"]): item for item in fold_manifest.get("folds", [])}
        missing_folds = [fold_index for fold_index in requested_folds if fold_index not in fold_entries]
        if missing_folds:
            raise ValueError(f"Fold runs manifest is missing trained artifacts for folds: {missing_folds}")
        base_feature_preparation, base_feature_preparation_path = _load_optional_feature_preparation_artifact(
            fold_entries[int(requested_folds[0])].get("feature_preparation_path")
        )
        if base_feature_preparation is None or base_feature_preparation_path is None:
            raise ValueError("Multi-fold OOF validation requires a feature preparation artifact for every fold.")
    else:
        resolved_feature_preparation_path = config.get("feature_preparation_path")
        if resolved_feature_preparation_path is None and _default_feature_preparation_path(config).exists():
            resolved_feature_preparation_path = _default_feature_preparation_path(config)
        base_feature_preparation, base_feature_preparation_path = _load_optional_feature_preparation_artifact(
            resolved_feature_preparation_path
        )

    prepared = prepare_frame_details(config, feature_preparation_artifact=base_feature_preparation)
    df = prepared.frame
    route = prepared.route
    label_col = prepared.label_col
    rank_col = prepared.rank_col
    cluster_col = prepared.cluster_col
    year_col = prepared.year_col
    feature_cols = list(prepared.feature_cols)
    prepared_metadata = prepare_frame_metadata(prepared)
    use_rank_aux = bool(prepared.rank_aux_effective) and rank_col is not None
    rank_loss_mode = resolve_effective_rank_loss_mode(config, use_rank_aux=use_rank_aux, rank_col=rank_col)
    rank_aux_effective = rank_loss_mode != RANK_LOSS_MODE_DISABLED and rank_col is not None
    threshold = float(config.get("threshold", 0.5))
    use_cluster_in_rank_loss = bool(config.get("use_cluster_in_rank_loss", False))
    group_col = str(config.get("group_col", "Sample_ID"))
    batch_size_tag = get_batch_size_tag(config)
    validation_metrics_path = Path(
        config.get("validation_metrics_path")
        or Path(config.get("output_root", "./data/output")) / tag_name("validation_metrics.json", batch_size_tag)
    )
    validation_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    if multi_fold_validation:
        device = resolve_torch_device(config)
        fold_metrics: list[dict[str, Any]] = []
        oof_detail_frames: list[pd.DataFrame] = []

        for fold_index in requested_folds:
            fold_artifact, fold_artifact_path = _load_optional_feature_preparation_artifact(
                fold_entries[int(fold_index)].get("feature_preparation_path")
            )
            if fold_artifact is None or fold_artifact_path is None:
                raise ValueError(f"Fold {fold_index} is missing its feature preparation artifact.")
            fold_feature_cols = list(fold_artifact.get("final_feature_cols", feature_cols))
            if fold_feature_cols != feature_cols:
                raise ValueError(
                    f"Fold {fold_index} feature schema does not match prepared validation frame: "
                    f"{fold_feature_cols} != {feature_cols}"
                )
            fold_config = dict(config)
            fold_config["fold_index"] = int(fold_index)
            if fold_artifact is not None:
                fold_config["feature_preparation_artifact"] = fold_artifact
            split = resolve_split_from_config(df, fold_config, group_col=group_col, label_col=label_col)
            prepared_for_member = prepare_frame_details(
                config,
                feature_preparation_artifact=fold_artifact,
                fill_numeric_imputation=fold_artifact is not None,
            )
            current_df = prepared_for_member.frame
            current_feature_cols = list(prepared_for_member.feature_cols)
            if current_feature_cols != feature_cols:
                raise ValueError(
                    f"Fold {fold_index} feature schema does not match prepared validation frame: "
                    f"{current_feature_cols} != {feature_cols}"
                )
            if not current_df.index.equals(df.index):
                raise ValueError(f"Fold {fold_index} prepared validation rows do not align with base frame.")

            model_raw = fold_entries[fold_index].get("model_path")
            if not isinstance(model_raw, (str, Path)) or not str(model_raw).strip():
                raise ValueError(f"Fold {fold_index} manifest does not provide a model_path.")
            model_path = Path(model_raw).resolve()
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            fold_scaler, scaler_path = _load_scaler_artifact(
                fold_entries[fold_index].get("scaler_path"),
                required=True,
            )

            _, eval_loader, _, _, _, _, _, _, _ = prepare_loaders(
                df,
                fold_config,
                feature_cols,
                label_col,
                rank_col,
                cluster_col,
                year_col,
                split,
                scaler=fold_scaler,
            )

            model = build_model(input_dim=len(feature_cols), config=fold_config).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()

            outputs = collect_loader_outputs(model, eval_loader, device)
            metrics = compute_metrics_from_outputs(
                outputs,
                threshold=threshold,
                include_rank_metrics=rank_aux_effective,
                use_cluster_in_rank_loss=use_cluster_in_rank_loss,
            )
            fold_metrics.append(
                {
                    "fold_index": int(fold_index),
                    "metrics": metrics,
                    "model_path": str(model_path),
                    "scaler_path": str(scaler_path),
                    "feature_preparation_path": str(fold_artifact_path) if fold_artifact_path is not None else None,
                }
            )

            raw_eval_idx = pd.Index(np.asarray(split.eval_idx, dtype=int))
            kept_eval_idx = raw_eval_idx[raw_eval_idx.isin(df.index)]
            if len(kept_eval_idx) != len(outputs["y_prob"]):
                raise ValueError(
                    "OOF output length does not match resolved evaluation indices "
                    f"for fold {fold_index}: {len(outputs['y_prob'])} != {len(kept_eval_idx)}"
                )

            eval_frame = df.reindex(kept_eval_idx)
            detail_frame = pd.DataFrame(
                {
                    "fold_index": int(fold_index),
                    "row_index": kept_eval_idx.to_numpy(dtype=int, copy=False),
                    group_col: eval_frame[group_col].to_numpy(),
                    "y_true": outputs["y_true"],
                    "y_prob": outputs["y_prob"],
                    "y_rank": outputs["y_rank"],
                    year_col: outputs["years"],
                }
            )
            if cluster_col is not None:
                detail_frame[cluster_col] = outputs["clusters"]
            oof_detail_frames.append(detail_frame)

        oof_details = pd.concat(oof_detail_frames, ignore_index=True)
        expected_row_indices = pd.Index(np.asarray(df.index, dtype=int))
        observed_row_indices = pd.Index(oof_details["row_index"].to_numpy(dtype=int))
        if (
            not expected_row_indices.is_unique
            or not observed_row_indices.is_unique
            or len(observed_row_indices) != len(expected_row_indices)
            or set(observed_row_indices.tolist()) != set(expected_row_indices.tolist())
        ):
            raise ValueError(
                "OOF coverage must contain every cleaned modeling row exactly once; "
                f"expected={len(expected_row_indices)}, observed={len(observed_row_indices)}."
            )
        if group_col in oof_details.columns:
            fold_counts_by_group = oof_details.groupby(group_col, dropna=False)["fold_index"].nunique()
            if not fold_counts_by_group.empty and int(fold_counts_by_group.max()) > 1:
                raise ValueError("OOF coverage assigns at least one group to multiple validation folds.")
        oof_details["pred_label"] = (oof_details["y_prob"].to_numpy(dtype=float) >= threshold).astype(int)
        oof_details["is_error"] = (
            oof_details["pred_label"].to_numpy(dtype=int) != oof_details["y_true"].to_numpy(dtype=int)
        ).astype(int)
        oof_details["abs_error"] = np.abs(
            oof_details["y_true"].to_numpy(dtype=float) - oof_details["y_prob"].to_numpy(dtype=float)
        )
        oof_details_path = validation_metrics_path.with_name(validation_metrics_path.stem + "_oof.csv")
        oof_details.to_csv(oof_details_path, index=False)
        oof_metrics = compute_metrics_from_outputs(
            {
                "y_prob": oof_details["y_prob"].to_numpy(dtype=float),
                "y_true": oof_details["y_true"].to_numpy(dtype=float),
                "y_rank": oof_details["y_rank"].to_numpy(dtype=float),
                "years": oof_details[year_col].to_numpy(dtype=float),
                "clusters": (
                    oof_details[cluster_col].to_numpy(dtype=float)
                    if cluster_col is not None and cluster_col in oof_details.columns
                    else np.zeros(len(oof_details), dtype=float)
                ),
            },
            threshold=threshold,
            include_rank_metrics=rank_aux_effective,
            use_cluster_in_rank_loss=use_cluster_in_rank_loss,
        )
        payload = {
            "route": route,
            "split_mode": "fold_manifest_oof",
            "split_name": "val",
            "fold_indices": requested_folds,
            "lambda_rank": float(config.get("lambda_rank", config.get("alpha", 1.0))),
            "rank_loss_mode": rank_loss_mode,
            "batch_size": int(config.get("batch_size")) if config.get("batch_size") is not None else None,
            "batch_size_tag": config.get("batch_size_tag"),
            "batch_size_source": config.get("batch_size_source"),
            "label_col": label_col,
            "rank_col": rank_col,
            "cluster_col": cluster_col,
            "year_col": year_col,
            "feature_count": len(feature_cols),
            "fold_metrics": fold_metrics,
            "fold_metric_summary": summarize_metric_dicts([item["metrics"] for item in fold_metrics]),
            "oof_metrics": oof_metrics,
            "oof_row_count": int(len(oof_details)),
            "artifact_source": fold_manifest.get("artifact_source"),
            "fold_runs_manifest_path": str(_default_fold_runs_manifest_path(config)),
            "oof_details_path": str(oof_details_path),
            "feature_preparation_path": str(base_feature_preparation_path) if base_feature_preparation_path is not None else None,
            **prepared_metadata,
            "rank_aux_effective": rank_aux_effective,
            **get_device_metadata(config),
        }
    else:
        single_config = dict(config)
        if requested_folds and len(requested_folds) == 1 and single_config.get("fold_index") is None:
            single_config["fold_index"] = int(requested_folds[0])

        split = resolve_split_from_config(df, single_config, group_col=group_col, label_col=label_col)
        if base_feature_preparation is not None:
            single_config["feature_preparation_artifact"] = base_feature_preparation
            expected_feature_cols = list(base_feature_preparation.get("final_feature_cols", feature_cols))
            if expected_feature_cols != feature_cols:
                raise ValueError(
                    "Configured feature preparation artifact does not match prepared validation frame: "
                    f"{expected_feature_cols} != {feature_cols}"
                )
        saved_scaler, saved_scaler_path = _load_scaler_artifact(single_config.get("scaler_path"))
        _, eval_loader, _, _, _, _, _, _, _ = prepare_loaders(
            df,
            single_config,
            feature_cols,
            label_col,
            rank_col,
            cluster_col,
            year_col,
            split,
            scaler=saved_scaler,
        )

        model_path = Path(single_config["model_path"]).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        device = resolve_torch_device(config)
        model = build_model(input_dim=len(feature_cols), config=single_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        metrics = evaluate_loader(
            model,
            eval_loader,
            device,
            threshold=threshold,
            include_rank_metrics=rank_aux_effective,
            use_cluster_in_rank_loss=use_cluster_in_rank_loss,
        )
        payload = {
            "route": route,
            "split_mode": split.split_mode,
            "split_name": split.split_name,
            "fold_index": split.fold_index,
            "lambda_rank": float(config.get("lambda_rank", config.get("alpha", 1.0))),
            "rank_loss_mode": rank_loss_mode,
            "batch_size": int(config.get("batch_size")) if config.get("batch_size") is not None else None,
            "effective_batch_size": int(eval_loader.batch_size or 0),
            "batch_size_tag": config.get("batch_size_tag"),
            "batch_size_source": config.get("batch_size_source"),
            "label_col": label_col,
            "rank_col": rank_col,
            "cluster_col": cluster_col,
            "year_col": year_col,
            "feature_count": len(feature_cols),
            "metrics": metrics,
            "feature_preparation_path": str(base_feature_preparation_path) if base_feature_preparation_path is not None else None,
            **prepared_metadata,
            "rank_aux_effective": rank_aux_effective,
            **get_device_metadata(config),
        }

    with validation_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    print(f"Validation metrics saved: {validation_metrics_path}")
    return payload


__all__ = [
    "EMPTY_ROUTE_METRICS",
    "collect_loader_outputs",
    "compute_binary_metrics",
    "compute_metrics_from_outputs",
    "compute_route_metrics",
    "evaluate_loader",
    "run_validation",
    "summarize_metric_dicts",
]
