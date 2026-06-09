from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .batch_size import get_batch_size_tag
from .constants import DEFAULT_BATCH_SIZE, DEFAULT_GROUP_COL, DEFAULT_SEED, RANK_LOSS_MODE_DISABLED
from .data_utils import save_feature_preparation_artifact, seed_everything
from .losses import (
    compute_rank_loss,
    masked_bce_with_logits,
    ndcg_aware_lambdarank_loss,
)
from .rank_loss_mode import resolve_effective_rank_loss_mode
from .selection import get_lambda_rank
from .shared import (
    PreparedSplit,
    build_model as _build_model,
    build_fold_training_config as _build_fold_training_config,
    default_feature_preparation_path as _default_feature_preparation_path,
    default_fold_runs_manifest_path as _default_fold_runs_manifest_path,
    default_history_path as _default_history_path,
    default_metrics_path as _default_metrics_path,
    default_model_path as _default_model_path,
    default_output_root as _default_output_root,
    default_scaler_path as _default_scaler_path,
    default_validation_metrics_path as _default_validation_metrics_path,
    get_device_metadata,
    has_split_manifest as _has_split_manifest,
    load_fold_runs_manifest,
    numeric_mean_std as _numeric_mean_std,
    prepare_frame as _prepare_frame,
    prepare_frame_details as _prepare_frame_details,
    prepare_frame_metadata as _prepare_frame_metadata,
    prepare_loaders as _prepare_loaders,
    resolve_torch_device,
    resolve_split_from_config as _resolve_split_from_config,
    save_json,
    try_load_fold_runs_manifest as _try_load_fold_runs_manifest,
)
from .split import resolve_requested_fold_indices
from .validation import collect_loader_outputs, compute_metrics_from_outputs, evaluate_loader, summarize_metric_dicts



def _compat_epoch_eval_call(model, loader, device):  # noqa: ANN001
    return evaluate_loader(model, loader, device, include_rank_metrics=False)


def _format_lambda_rank_for_logging(lambda_rank: float, rank_aux_enabled: bool) -> str:
    if not rank_aux_enabled:
        return "ignored"
    return str(lambda_rank)


def _run_training_single_fold(
    config: dict[str, Any],
    df: pd.DataFrame,
    route: str,
    label_col: str,
    rank_col: str | None,
    cluster_col: str | None,
    year_col: str,
    feature_cols: list[str],
    split: PreparedSplit,
    prepared_metadata: dict[str, Any],
    feature_preparation_artifact: dict[str, Any],
) -> dict[str, Any]:
    seed_everything(int(config.get("seed", DEFAULT_SEED)))

    loader_config = dict(config)
    loader_config["feature_preparation_artifact"] = feature_preparation_artifact
    train_loader, eval_loader, scaler, _, _, _, _, _, fitted_feature_preparation = _prepare_loaders(
        df,
        loader_config,
        feature_cols,
        label_col,
        rank_col,
        cluster_col,
        year_col,
        split,
    )

    device = resolve_torch_device(config)
    device_metadata = get_device_metadata(config)
    model = _build_model(input_dim=len(feature_cols), config=config).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )

    lambda_rank = get_lambda_rank(config)
    lambda_source = "lambda_rank" if "lambda_rank" in config else ("alpha" if "alpha" in config else "default")
    use_rank_aux = bool(prepared_metadata.get("rank_aux_effective", False))
    use_cluster_in_rank_loss = bool(config.get("use_cluster_in_rank_loss", False))
    rank_loss_mode = resolve_effective_rank_loss_mode(
        config,
        use_rank_aux=use_rank_aux,
        rank_col=rank_col,
    )
    rank_aux_enabled = rank_loss_mode != RANK_LOSS_MODE_DISABLED
    threshold = float(config.get("threshold", 0.5))
    epochs = int(config.get("epochs", 100))
    patience = int(config.get("patience", 15))
    requested_batch_size = int(config.get("batch_size", DEFAULT_BATCH_SIZE))
    batch_size_tag = get_batch_size_tag(config)
    batch_size_source = config.get("batch_size_source")

    history: list[dict[str, Any]] = []
    best_epoch = -1
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None

    print(
        "Start training: "
        f"route={route} | split_mode={split.split_mode} | eval_split={split.split_name} | "
        f"fold_index={split.fold_index} | batch_size={train_loader.batch_size} | "
        f"rank_aux={rank_aux_enabled} | lambda_rank={_format_lambda_rank_for_logging(lambda_rank, rank_aux_enabled)} | "
        f"rank_loss_mode={rank_loss_mode} | "
        f"device={device.type}"
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls = 0.0
        epoch_rank = 0.0

        for x_b, y_cls_b, y_rank_b, year_b, cluster_b in train_loader:
            x_b = x_b.to(device)
            y_cls_b = y_cls_b.to(device)
            y_rank_b = y_rank_b.to(device)
            year_b = year_b.to(device)
            cluster_b = cluster_b.to(device)

            optimizer.zero_grad()
            logits, _ = model(x_b)
            loss_cls = masked_bce_with_logits(logits, y_cls_b)
            if rank_aux_enabled and rank_col is not None:
                loss_rank, _ = compute_rank_loss(
                    rank_loss_mode,
                    logits,
                    y_rank_b,
                    year_b,
                    cluster_b,
                    use_cluster_in_rank_loss=use_cluster_in_rank_loss,
                )
                loss = loss_cls + lambda_rank * loss_rank
                epoch_rank += float(loss_rank.item())
            else:
                loss = loss_cls
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            epoch_cls += float(loss_cls.item())

        val_eval = _compat_epoch_eval_call(model, eval_loader, device)

        avg_loss = epoch_loss / max(len(train_loader), 1)
        avg_cls = epoch_cls / max(len(train_loader), 1)
        avg_rank = epoch_rank / max(len(train_loader), 1) if rank_aux_enabled and rank_col is not None else 0.0

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": float(avg_loss),
            "train_cls_loss": float(avg_cls),
            "train_rank_loss": float(avg_rank),
            "lambda_rank": float(lambda_rank),
            "rank_loss_mode": rank_loss_mode,
            "train_auc": float("nan"),
            "train_f1": float("nan"),
            "val_auc": val_eval.get("auc"),
            "val_f1": val_eval.get("f1"),
            "val_loss_proxy": float(1.0 - (val_eval.get("auc") if np.isfinite(val_eval.get("auc", np.nan)) else 0.0)),
        }
        history.append(epoch_record)

        current_val_loss = float(epoch_record["val_loss_proxy"])
        improved = current_val_loss < best_val_loss
        if improved:
            best_val_loss = current_val_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            print(
                f"Epoch [{epoch + 1}/{epochs}] | "
                f"loss={avg_loss:.4f} | "
                f"val_auc={val_eval.get('auc', float('nan')):.4f} | val_f1={val_eval.get('f1', float('nan')):.4f}"
            )

        if not improved and patience > 0 and (epoch + 1 - best_epoch) >= patience:
            print(f"Early stopping at epoch {epoch + 1} (best epoch={best_epoch}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_outputs = collect_loader_outputs(model, eval_loader, device)
    final_eval = compute_metrics_from_outputs(
        eval_outputs,
        threshold=threshold,
        include_rank_metrics=rank_aux_enabled,
        use_cluster_in_rank_loss=use_cluster_in_rank_loss,
    )

    model_path = _default_model_path(config)
    scaler_path = _default_scaler_path(config)
    metrics_path = _default_metrics_path(config)
    validation_metrics_path = _default_validation_metrics_path(config)
    history_path = _default_history_path(config)
    feature_preparation_path = _default_feature_preparation_path(config)

    for path in [model_path, scaler_path, metrics_path, validation_metrics_path, history_path, feature_preparation_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_path)
    with scaler_path.open("wb") as fh:
        pickle.dump(scaler, fh)
    if fitted_feature_preparation is None:
        raise RuntimeError("Training did not produce a fitted feature preparation artifact.")
    save_feature_preparation_artifact(feature_preparation_path, fitted_feature_preparation)

    history_df = pd.DataFrame(history)
    history_df.to_csv(history_path, index=False)

    metrics = {
        "route": route,
        "split_mode": split.split_mode,
        "split_name": split.split_name,
        "fold_index": split.fold_index,
        "feature_count": len(feature_cols),
        "feature_cols": feature_cols,
        "label_col": label_col,
        "rank_col": rank_col,
        "cluster_col": cluster_col,
        "year_col": year_col,
        "use_rank_aux": rank_aux_enabled and rank_col is not None,
        "rank_aux_effective": rank_aux_enabled,
        "use_cluster_in_rank_loss": use_cluster_in_rank_loss,
        "lambda_rank": float(lambda_rank),
        "rank_loss_mode": rank_loss_mode,
        "lambda_source": lambda_source,
        "batch_size": requested_batch_size,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": int(train_loader.batch_size or 0),
        "batch_size_tag": batch_size_tag,
        "batch_size_source": batch_size_source,
        "epochs_requested": epochs,
        "best_epoch": best_epoch,
        "best_val_loss_proxy": float(best_val_loss),
        "best_eval": final_eval,
        "metrics_path": str(metrics_path),
        "history_path": str(history_path),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "validation_metrics_path": str(validation_metrics_path),
        "feature_preparation_path": str(feature_preparation_path),
        **prepared_metadata,
        "feature_preparation_artifact": fitted_feature_preparation,
        **device_metadata,
    }
    save_json(metrics_path, metrics)

    validation_payload = {
        "route": route,
        "split_mode": split.split_mode,
        "split_name": split.split_name,
        "fold_index": split.fold_index,
        "lambda_rank": float(lambda_rank),
        "rank_loss_mode": rank_loss_mode,
        "label_col": label_col,
        "rank_col": rank_col,
        "cluster_col": cluster_col,
        "year_col": year_col,
        "feature_count": len(feature_cols),
        "batch_size": requested_batch_size,
        "requested_batch_size": requested_batch_size,
        "effective_batch_size": int(eval_loader.batch_size or 0),
        "batch_size_tag": batch_size_tag,
        "batch_size_source": batch_size_source,
        "metrics": final_eval,
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_preparation_path": str(feature_preparation_path),
        **prepared_metadata,
        "feature_preparation_artifact": fitted_feature_preparation,
        "rank_aux_effective": rank_aux_enabled,
        **device_metadata,
    }
    save_json(validation_metrics_path, validation_payload)

    print(f"Model saved: {model_path}")
    print(f"Scaler saved: {scaler_path}")
    print(f"Feature preparation saved: {feature_preparation_path}")
    print(f"Metrics saved: {metrics_path}")
    print(f"Validation metrics saved: {validation_metrics_path}")
    print(f"History saved: {history_path}")
    return {
        "metrics": metrics,
        "validation_payload": validation_payload,
        "history_records": history,
        "eval_outputs": eval_outputs,
    }


def run_training(config: dict[str, Any]) -> dict[str, Any]:
    prepared = _prepare_frame_details(config)
    df = prepared.frame
    route = prepared.route
    label_col = prepared.label_col
    rank_col = prepared.rank_col
    cluster_col = prepared.cluster_col
    year_col = prepared.year_col
    feature_cols = list(prepared.feature_cols)
    prepared_metadata = _prepare_frame_metadata(prepared)
    group_col = str(config.get("group_col", DEFAULT_GROUP_COL))
    requested_folds = resolve_requested_fold_indices(config, config.get("split_manifest_path"))
    split_manifest_present = _has_split_manifest(config)

    if split_manifest_present and requested_folds and len(requested_folds) > 1:
        output_root = _default_output_root(config)
        output_root.mkdir(parents=True, exist_ok=True)
        metrics_path = _default_metrics_path(config)
        validation_metrics_path = _default_validation_metrics_path(config)
        history_path = _default_history_path(config)
        fold_runs_manifest_path = _default_fold_runs_manifest_path(config)

        lambda_rank = get_lambda_rank(config)
        lambda_source = "lambda_rank" if "lambda_rank" in config else ("alpha" if "alpha" in config else "default")
        use_rank_aux = bool(prepared_metadata.get("rank_aux_effective", False))
        use_cluster_in_rank_loss = bool(config.get("use_cluster_in_rank_loss", False))
        rank_loss_mode = resolve_effective_rank_loss_mode(config, use_rank_aux=use_rank_aux, rank_col=rank_col)
        rank_aux_enabled = rank_loss_mode != RANK_LOSS_MODE_DISABLED

        print(
            "Start multi-fold training: "
            f"route={route} | folds={requested_folds} | "
            f"lambda_rank={_format_lambda_rank_for_logging(lambda_rank, rank_aux_enabled)} | "
            f"rank_loss_mode={rank_loss_mode}"
        )

        fold_entries: list[dict[str, Any]] = []
        fold_metrics: list[dict[str, Any]] = []
        fold_validation_metrics: list[dict[str, Any]] = []
        combined_history_rows: list[dict[str, Any]] = []
        for fold_index in requested_folds:
            fold_config = _build_fold_training_config(config, route, fold_index)
            split = _resolve_split_from_config(df, fold_config, group_col=group_col, label_col=label_col)
            result = _run_training_single_fold(
                fold_config,
                df,
                route,
                label_col,
                rank_col,
                cluster_col,
                year_col,
                feature_cols,
                split,
                prepared_metadata,
                prepared.feature_preparation_artifact,
            )
            metrics = result["metrics"]
            validation_payload = result["validation_payload"]
            fold_metrics.append(metrics)
            fold_validation_metrics.append({"fold_index": fold_index, "metrics": validation_payload["metrics"]})
            combined_history_rows.extend({"fold_index": fold_index, **row} for row in result["history_records"])
            fold_entries.append(
                {
                    "fold_index": fold_index,
                    "output_root": fold_config["output_root"],
                    "model_path": metrics["model_path"],
                    "scaler_path": metrics["scaler_path"],
                    "metrics_path": metrics["metrics_path"],
                    "validation_metrics_path": metrics["validation_metrics_path"],
                    "history_path": metrics["history_path"],
                    "feature_preparation_path": metrics["feature_preparation_path"],
                    "best_epoch": metrics["best_epoch"],
                    "best_val_loss_proxy": metrics["best_val_loss_proxy"],
                    "best_eval": metrics["best_eval"],
                }
            )

        history_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(combined_history_rows).to_csv(history_path, index=False)

        per_fold_metric_summary = summarize_metric_dicts([item["metrics"] for item in fold_validation_metrics])
        validation_payload = {
            "route": route,
            "split_mode": "fold_manifest_oof",
            "split_name": "val",
            "fold_indices": requested_folds,
            "lambda_rank": float(lambda_rank),
            "rank_loss_mode": rank_loss_mode,
            "batch_size": int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
            "batch_size_tag": get_batch_size_tag(config),
            "batch_size_source": config.get("batch_size_source"),
            "label_col": label_col,
            "rank_col": rank_col,
            "cluster_col": cluster_col,
            "year_col": year_col,
            "feature_count": len(feature_cols),
            "fold_metrics": fold_validation_metrics,
            "fold_metric_summary": per_fold_metric_summary,
            "fold_runs_manifest_path": str(fold_runs_manifest_path),
            **prepared_metadata,
            "rank_aux_effective": rank_aux_enabled,
            **get_device_metadata(config),
        }
        save_json(validation_metrics_path, validation_payload)

        fold_runs_manifest = {
            "route": route,
            "output_root": str(output_root),
            "fold_indices": requested_folds,
            "n_folds": int(len(requested_folds)),
            "batch_size": int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
            "batch_size_tag": get_batch_size_tag(config),
            "batch_size_source": config.get("batch_size_source"),
            "fold_runs_manifest_path": str(fold_runs_manifest_path),
            "metrics_path": str(metrics_path),
            "validation_metrics_path": str(validation_metrics_path),
            "history_path": str(history_path),
            "folds": fold_entries,
        }
        save_json(fold_runs_manifest_path, fold_runs_manifest)

        best_eval_summary = summarize_metric_dicts([item["best_eval"] for item in fold_metrics])
        best_epoch_mean, best_epoch_std = _numeric_mean_std([item["best_epoch"] for item in fold_metrics])
        best_val_loss_mean, best_val_loss_std = _numeric_mean_std([item["best_val_loss_proxy"] for item in fold_metrics])

        aggregate_metrics = {
            "route": route,
            "split_mode": "fold_manifest_multi",
            "split_name": "val",
            "fold_indices": requested_folds,
            "feature_count": len(feature_cols),
            "feature_cols": feature_cols,
            "label_col": label_col,
            "rank_col": rank_col,
            "cluster_col": cluster_col,
            "year_col": year_col,
            "use_rank_aux": rank_aux_enabled and rank_col is not None,
            "use_cluster_in_rank_loss": use_cluster_in_rank_loss,
            "lambda_rank": float(lambda_rank),
            "rank_loss_mode": rank_loss_mode,
            "lambda_source": lambda_source,
            "batch_size": int(config.get("batch_size", DEFAULT_BATCH_SIZE)),
            "effective_batch_sizes": [int(item.get("effective_batch_size", 0)) for item in fold_metrics],
            "batch_size_tag": get_batch_size_tag(config),
            "batch_size_source": config.get("batch_size_source"),
            "epochs_requested": int(config.get("epochs", 100)),
            "best_epoch_mean": best_epoch_mean,
            "best_epoch_std": best_epoch_std,
            "best_val_loss_proxy_mean": best_val_loss_mean,
            "best_val_loss_proxy_std": best_val_loss_std,
            "best_eval_summary": best_eval_summary,
            "metrics_path": str(metrics_path),
            "history_path": str(history_path),
            "validation_metrics_path": str(validation_metrics_path),
            "fold_runs_manifest_path": str(fold_runs_manifest_path),
            "fold_results": fold_entries,
            **prepared_metadata,
            "rank_aux_effective": rank_aux_enabled,
            **get_device_metadata(config),
        }
        save_json(metrics_path, aggregate_metrics)

        print(f"Fold runs manifest saved: {fold_runs_manifest_path}")
        print(f"Aggregate metrics saved: {metrics_path}")
        print(f"Aggregate validation metrics saved: {validation_metrics_path}")
        print(f"Aggregate history saved: {history_path}")
        return aggregate_metrics

    single_config = dict(config)
    if requested_folds and len(requested_folds) == 1 and single_config.get("fold_index") is None:
        single_config["fold_index"] = int(requested_folds[0])

    split = _resolve_split_from_config(df, single_config, group_col=group_col, label_col=label_col)
    result = _run_training_single_fold(
        single_config,
        df,
        route,
        label_col,
        rank_col,
        cluster_col,
        year_col,
        feature_cols,
        split,
        prepared_metadata,
        prepared.feature_preparation_artifact,
    )
    return result["metrics"]


__all__ = [
    "PreparedSplit",
    "_build_fold_training_config",
    "_build_model",
    "_default_feature_preparation_path",
    "_default_fold_runs_manifest_path",
    "_default_history_path",
    "_default_metrics_path",
    "_default_model_path",
    "_default_output_root",
    "_default_scaler_path",
    "_default_validation_metrics_path",
    "_has_split_manifest",
    "_numeric_mean_std",
    "_prepare_frame",
    "_prepare_frame_details",
    "_prepare_frame_metadata",
    "_prepare_loaders",
    "_resolve_split_from_config",
    "_try_load_fold_runs_manifest",
    "compute_rank_loss",
    "load_fold_runs_manifest",
    "masked_bce_with_logits",
    "ndcg_aware_lambdarank_loss",
    "run_training",
]
