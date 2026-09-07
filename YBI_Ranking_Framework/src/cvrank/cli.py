from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from .batch_size import apply_batch_size_override, apply_batch_size_path_tagging
from .config import (
    load_yaml_config,
    resolve_config_paths,
    resolve_path,
)
from .data_utils import load_table
from .rank_loss_mode import apply_rank_loss_mode_override, normalize_rank_loss_mode
from .output_suffix import (
    apply_output_suffix_override,
    apply_output_suffix_path_tagging,
    normalize_output_suffix,
)
from .run_fingerprint import FileHashCache, describe_file
from .selection import run_lambda_selection
from .shared import build_fold_training_config, numeric_mean_std, save_json
from .split import build_fold_manifest_from_config, save_manifest, save_manifest_cleaning_audit, summarize_manifest
from .split import resolve_requested_fold_indices
from .sweep import run_lambda_sweep
from .training import run_training
from .validation import run_validation, summarize_metric_dicts


PipelineCallable = Callable[[dict[str, Any]], Any]


RUNTIME_CONFIG_COMMANDS = {
    "train",
    "validate",
    "sweep-lambda",
    "select-lambda",
}

NON_RUNTIME_COMMANDS = {
    "show-config",
    "make-folds",
    "summarize",
}

RUNTIME_METADATA_KEYS = {
    "rank_input_status",
    "rank_aux_effective",
    "data_quality_warnings",
    "raw_row_count",
    "clean_row_count",
    "dropped_rows_missing_required",
    "dropped_rows_missing_required_raw_na",
    "dropped_rows_missing_required_input_nodata",
    "dropped_rows_invalid_numeric",
    "dropped_row_indices_preview",
    "fold_resolution_source",
    "device_requested",
    "device_resolved",
}


def collect_environment_snapshot(base_dir: str | Path | None = None) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "base_dir": str(Path(base_dir or ".").resolve()),
    }


def save_run_metadata(base_dir: str | Path, command: str, payload: dict[str, Any]) -> Path:
    root = Path(base_dir).resolve()
    out_dir = root / "artifacts" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"{stamp}_{command}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return path


@dataclass
class PipelineRunner:
    command: str
    apply_batch_size_tagging: bool = True
    apply_rank_loss_override: bool = True

    def resolve_runtime_overrides(
        self,
        config: dict[str, Any],
        *,
        batch_size: int | None = None,
        rank_loss_mode: str | None = None,
        output_suffix: str | None = None,
    ) -> dict[str, Any]:
        resolved = apply_output_suffix_override(config, output_suffix)
        resolved = apply_batch_size_override(resolved, batch_size)
        if self.apply_rank_loss_override:
            resolved = apply_rank_loss_mode_override(resolved, rank_loss_mode)
        resolved = apply_output_suffix_path_tagging(resolved)
        if self.apply_batch_size_tagging:
            resolved = apply_batch_size_path_tagging(resolved)
        return resolved


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")


def _add_batch_size_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional batch size override for this run. When set, outputs are grouped under a bs_<N> tag.",
    )


def _add_rank_loss_mode_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rank-loss-mode",
        type=normalize_rank_loss_mode,
        metavar="{lambdarank,disabled}",
        default=None,
        help=(
            "Optional rank loss mode override. Supported values: lambdarank, disabled "
            "(aliases: lambda, ndcg_aware_lambdarank, none, off). "
            "disabled means the classification baseline: only ranking loss is disabled; "
            "model training and projection remain enabled."
        ),
    )


def _add_output_suffix_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-suffix",
        type=normalize_output_suffix,
        default=None,
        help="Optional output root suffix. Example: archive -> output_dbscan_archive.",
    )


_PROMOTED_PREPARED_METADATA_KEYS = (
    "raw_row_count",
    "rank_input_status",
    "rank_aux_effective",
    "data_quality_warnings",
    "dropped_rows_missing_required",
    "dropped_rows_missing_required_raw_na",
    "dropped_rows_missing_required_input_nodata",
    "dropped_rows_invalid_numeric",
    "dropped_row_indices_missing_required_preview",
    "dropped_row_indices_missing_required_raw_na_preview",
    "dropped_row_indices_missing_required_input_nodata_preview",
    "dropped_row_indices_invalid_numeric_preview",
    "input_nodata_values",
    "dropped_row_indices_preview",
    "raw_feature_cols",
    "feature_cols",
    "feature_preparation_artifact",
)

_PROMOTED_DEVICE_METADATA_KEYS = ("device_requested", "device_resolved")


def _collect_present_keys(payload: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: payload[key] for key in keys if key in payload}


def _default_sweep_runs_csv_from_summary(summary_csv: Path) -> Path:
    name = summary_csv.name
    if "summary" in name:
        return summary_csv.with_name(name.replace("summary", "runs", 1))
    return summary_csv.with_name("lambda_sweep_runs.csv")


def _has_csv_value(value: Any) -> bool:
    if value is None:
        return False
    try:
        if bool(pd.isna(value)):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _load_selected_sweep_rows(selected: dict[str, Any], train_cfg: dict[str, Any]) -> dict[int, dict[str, Any]] | None:
    summary_csv_raw = selected.get("source_summary_csv")
    if summary_csv_raw in (None, ""):
        return None

    runs_csv = _default_sweep_runs_csv_from_summary(Path(str(summary_csv_raw)))
    if not runs_csv.exists():
        return None

    try:
        runs_df = pd.read_csv(runs_csv)
    except (OSError, ValueError, pd.errors.ParserError):
        return None
    if runs_df.empty:
        return None

    route = str(selected.get("route", "")).strip().lower()
    if route == "":
        return None

    rank_loss_mode = normalize_rank_loss_mode(
        selected.get("rank_loss_mode", train_cfg.get("rank_loss_mode", "ndcg_aware_lambdarank"))
    )
    lambda_rank = float(selected["lambda_rank"])
    batch_size_tag = train_cfg.get("batch_size_tag")

    filtered = runs_df[runs_df["route"].astype(str).str.lower() == route].copy()
    if "rank_loss_mode" in filtered.columns:
        filtered = filtered[filtered["rank_loss_mode"].astype(str).map(normalize_rank_loss_mode) == rank_loss_mode].copy()
    if filtered.empty:
        return None

    lambda_series = pd.to_numeric(filtered["lambda_rank"], errors="coerce")
    filtered = filtered[(lambda_series - lambda_rank).abs() <= 1e-12].copy()
    if filtered.empty:
        return None

    if batch_size_tag not in (None, ""):
        # Older sweep manifests may not have populated ``batch_size_tag`` even
        # though they do contain the numeric batch size.  Prefer the explicit
        # tag, then fall back to the numeric value; never silently accept a
        # different batch size.
        tagged = pd.DataFrame()
        if "batch_size_tag" in filtered.columns:
            batch_series = filtered["batch_size_tag"].astype(str).str.strip()
            tagged = filtered[batch_series == str(batch_size_tag)].copy()
        if not tagged.empty:
            filtered = tagged
        else:
            expected_batch_size = pd.to_numeric(pd.Series([train_cfg.get("batch_size")]), errors="coerce").iloc[0]
            if pd.isna(expected_batch_size) or "batch_size" not in filtered.columns:
                return None
            numeric_batch = pd.to_numeric(filtered["batch_size"], errors="coerce")
            filtered = filtered[numeric_batch == float(expected_batch_size)].copy()
            if filtered.empty:
                return None

    rows_by_fold: dict[int, dict[str, Any]] = {}
    for row in filtered.to_dict(orient="records"):
        try:
            fold_index = int(row["fold_index"])
        except (TypeError, ValueError, KeyError):
            continue
        rows_by_fold[fold_index] = row
    return rows_by_fold or None


def _copy_promoted_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Required sweep artifact not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _promote_selected_training_from_sweep(selected: dict[str, Any], train_cfg: dict[str, Any]) -> dict[str, Any] | None:
    requested_folds = resolve_requested_fold_indices(train_cfg, train_cfg.get("split_manifest_path"))
    if not requested_folds or len(requested_folds) <= 1:
        return None

    sweep_rows = _load_selected_sweep_rows(selected, train_cfg)
    if not sweep_rows:
        return None
    if any(int(fold_index) not in sweep_rows for fold_index in requested_folds):
        return None

    route = str(selected["route"]).strip().lower()
    selected_run_root = Path(str(selected["selected_run_root"]))
    selected_run_root.mkdir(parents=True, exist_ok=True)

    fold_entries: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    fold_validation_metrics: list[dict[str, Any]] = []
    combined_history_rows: list[dict[str, Any]] = []
    base_metrics_payload: dict[str, Any] | None = None
    hash_cache = FileHashCache()

    for fold_index in requested_folds:
        row = sweep_rows[int(fold_index)]
        source_metrics_path = Path(str(row["metrics_path"])).resolve()
        source_history_path = Path(str(row["history_path"])).resolve()
        source_model_path = Path(str(row["model_path"])).resolve()
        source_scaler_path = Path(str(row["scaler_path"])).resolve()

        if not source_metrics_path.exists() or not source_history_path.exists() or not source_model_path.exists() or not source_scaler_path.exists():
            return None

        with source_metrics_path.open("r", encoding="utf-8") as fh:
            source_metrics = json.load(fh)

        source_validation_raw = source_metrics.get("validation_metrics_path")
        source_feature_preparation_raw = source_metrics.get("feature_preparation_path")
        if source_validation_raw in (None, "") or source_feature_preparation_raw in (None, ""):
            return None

        source_validation_path = Path(str(source_validation_raw)).resolve()
        source_feature_preparation_path = Path(str(source_feature_preparation_raw)).resolve()
        if not source_validation_path.exists() or not source_feature_preparation_path.exists():
            return None

        source_completion_raw = row.get("run_completion_path")
        source_completion_path = (
            Path(str(source_completion_raw)).resolve()
            if _has_csv_value(source_completion_raw)
            else None
        )
        if source_completion_path is not None:
            if not source_completion_path.exists():
                return None
            try:
                with source_completion_path.open("r", encoding="utf-8") as fh:
                    source_completion = json.load(fh)
            except (OSError, json.JSONDecodeError):
                return None
            if not isinstance(source_completion, dict) or source_completion.get("status") != "complete":
                return None

        with source_validation_path.open("r", encoding="utf-8") as fh:
            source_validation = json.load(fh)
        history_df = pd.read_csv(source_history_path)

        fold_cfg = build_fold_training_config(train_cfg, route, int(fold_index))
        dest_model_path = Path(str(fold_cfg["model_path"]))
        dest_scaler_path = Path(str(fold_cfg["scaler_path"]))
        dest_metrics_path = Path(str(fold_cfg["metrics_path"]))
        dest_validation_path = Path(str(fold_cfg["validation_metrics_path"]))
        dest_history_path = Path(str(fold_cfg["history_path"]))
        dest_feature_preparation_path = Path(str(fold_cfg["feature_preparation_path"]))

        _copy_promoted_file(source_model_path, dest_model_path)
        _copy_promoted_file(source_scaler_path, dest_scaler_path)
        _copy_promoted_file(source_feature_preparation_path, dest_feature_preparation_path)

        dest_history_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_promoted_file(source_history_path, dest_history_path)
        combined_history_rows.extend({"fold_index": int(fold_index), **row_data} for row_data in history_df.to_dict(orient="records"))

        source_artifact_paths = {
            "model": source_model_path,
            "scaler": source_scaler_path,
            "feature_preparation": source_feature_preparation_path,
            "metrics": source_metrics_path,
            "validation_metrics": source_validation_path,
            "history": source_history_path,
        }
        source_artifacts = {
            key: describe_file(path, hash_cache)
            for key, path in source_artifact_paths.items()
        }
        if source_completion_path is not None:
            source_artifacts["run_completion"] = describe_file(source_completion_path, hash_cache)

        promoted_metrics = dict(source_metrics)
        promoted_metrics.update(
            {
                "fold_index": int(fold_index),
                "metrics_path": str(dest_metrics_path),
                "history_path": str(dest_history_path),
                "model_path": str(dest_model_path),
                "scaler_path": str(dest_scaler_path),
                "validation_metrics_path": str(dest_validation_path),
                "feature_preparation_path": str(dest_feature_preparation_path),
            }
        )
        save_json(dest_metrics_path, promoted_metrics)

        promoted_validation = dict(source_validation)
        promoted_validation.update(
            {
                "fold_index": int(fold_index),
                "model_path": str(dest_model_path),
                "scaler_path": str(dest_scaler_path),
                "feature_preparation_path": str(dest_feature_preparation_path),
            }
        )
        save_json(dest_validation_path, promoted_validation)

        fold_metrics.append(promoted_metrics)
        fold_validation_metrics.append({"fold_index": int(fold_index), "metrics": promoted_validation["metrics"]})
        fold_entries.append(
            {
                "fold_index": int(fold_index),
                "output_root": fold_cfg["output_root"],
                "model_path": promoted_metrics["model_path"],
                "scaler_path": promoted_metrics["scaler_path"],
                "metrics_path": promoted_metrics["metrics_path"],
                "validation_metrics_path": promoted_metrics["validation_metrics_path"],
                "history_path": promoted_metrics["history_path"],
                "feature_preparation_path": promoted_metrics["feature_preparation_path"],
                "artifact_source": "lambda_sweep",
                "source_run_root": (
                    str(Path(str(row.get("run_dir"))).resolve())
                    if _has_csv_value(row.get("run_dir"))
                    else str(source_model_path.parent.parent)
                ),
                "source_run_completion_path": str(source_completion_path) if source_completion_path is not None else None,
                "source_artifacts": source_artifacts,
                "destination_artifacts": {
                    "model": describe_file(dest_model_path, hash_cache),
                    "scaler": describe_file(dest_scaler_path, hash_cache),
                    "feature_preparation": describe_file(dest_feature_preparation_path, hash_cache),
                    "history": describe_file(dest_history_path, hash_cache),
                },
                "best_epoch": promoted_metrics["best_epoch"],
                "best_val_loss_proxy": promoted_metrics["best_val_loss_proxy"],
                "best_eval": promoted_metrics["best_eval"],
            }
        )
        if base_metrics_payload is None:
            base_metrics_payload = promoted_metrics

    if base_metrics_payload is None:
        return None

    history_path = Path(str(train_cfg["history_path"]))
    metrics_path = Path(str(train_cfg["metrics_path"]))
    validation_metrics_path = Path(str(train_cfg["validation_metrics_path"]))
    fold_runs_manifest_path = Path(str(train_cfg["fold_runs_manifest_path"]))

    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(combined_history_rows).to_csv(history_path, index=False)

    per_fold_metric_summary = summarize_metric_dicts([item["metrics"] for item in fold_validation_metrics])
    validation_payload = {
        "route": route,
        "split_mode": "fold_manifest_oof",
        "split_name": "val",
        "fold_indices": [int(fold_index) for fold_index in requested_folds],
        "lambda_rank": float(selected["lambda_rank"]),
        "rank_loss_mode": str(selected["rank_loss_mode"]),
        "batch_size": int(train_cfg.get("batch_size")) if train_cfg.get("batch_size") is not None else base_metrics_payload.get("batch_size"),
        "batch_size_tag": train_cfg.get("batch_size_tag"),
        "batch_size_source": train_cfg.get("batch_size_source"),
        "label_col": base_metrics_payload.get("label_col"),
        "rank_col": base_metrics_payload.get("rank_col"),
        "cluster_col": base_metrics_payload.get("cluster_col"),
        "year_col": base_metrics_payload.get("year_col"),
        "feature_count": int(base_metrics_payload.get("feature_count", 0)),
        "fold_metrics": fold_validation_metrics,
        "fold_metric_summary": per_fold_metric_summary,
        "fold_runs_manifest_path": str(fold_runs_manifest_path),
        **_collect_present_keys(base_metrics_payload, _PROMOTED_PREPARED_METADATA_KEYS),
        "rank_aux_effective": bool(base_metrics_payload.get("rank_aux_effective", base_metrics_payload.get("use_rank_aux", False))),
        **_collect_present_keys(base_metrics_payload, _PROMOTED_DEVICE_METADATA_KEYS),
    }
    save_json(validation_metrics_path, validation_payload)

    fold_runs_manifest = {
        "route": route,
        "output_root": str(selected_run_root),
        "fold_indices": [int(fold_index) for fold_index in requested_folds],
        "n_folds": int(len(requested_folds)),
        "batch_size": int(train_cfg.get("batch_size")) if train_cfg.get("batch_size") is not None else base_metrics_payload.get("batch_size"),
        "batch_size_tag": train_cfg.get("batch_size_tag"),
        "batch_size_source": train_cfg.get("batch_size_source"),
        "fold_runs_manifest_path": str(fold_runs_manifest_path),
        "metrics_path": str(metrics_path),
        "validation_metrics_path": str(validation_metrics_path),
        "history_path": str(history_path),
        "artifact_source": "lambda_sweep",
        "source_summary_csv": str(selected.get("source_summary_csv")) if selected.get("source_summary_csv") else None,
        "source_runs_csv": (
            str(_default_sweep_runs_csv_from_summary(Path(str(selected["source_summary_csv"]))))
            if selected.get("source_summary_csv")
            else None
        ),
        "lambda_rank": float(selected["lambda_rank"]),
        "rank_loss_mode": str(selected["rank_loss_mode"]),
        "folds": fold_entries,
    }
    save_json(fold_runs_manifest_path, fold_runs_manifest)

    best_eval_summary = summarize_metric_dicts([item["best_eval"] for item in fold_metrics])
    best_epoch_mean, best_epoch_std = numeric_mean_std([item["best_epoch"] for item in fold_metrics])
    best_val_loss_mean, best_val_loss_std = numeric_mean_std([item["best_val_loss_proxy"] for item in fold_metrics])
    aggregate_metrics = {
        "route": route,
        "split_mode": "fold_manifest_multi",
        "split_name": "val",
        "fold_indices": [int(fold_index) for fold_index in requested_folds],
        "feature_count": int(base_metrics_payload.get("feature_count", 0)),
        "feature_cols": list(base_metrics_payload.get("feature_cols", [])),
        "label_col": base_metrics_payload.get("label_col"),
        "rank_col": base_metrics_payload.get("rank_col"),
        "cluster_col": base_metrics_payload.get("cluster_col"),
        "year_col": base_metrics_payload.get("year_col"),
        "use_rank_aux": bool(base_metrics_payload.get("use_rank_aux", False)),
        "use_cluster_in_rank_loss": bool(base_metrics_payload.get("use_cluster_in_rank_loss", False)),
        "lambda_rank": float(selected["lambda_rank"]),
        "rank_loss_mode": str(selected["rank_loss_mode"]),
        "lambda_source": str(base_metrics_payload.get("lambda_source", "sweep_promoted")),
        "batch_size": int(train_cfg.get("batch_size")) if train_cfg.get("batch_size") is not None else base_metrics_payload.get("batch_size"),
        "effective_batch_sizes": [int(item.get("effective_batch_size", 0) or 0) for item in fold_metrics],
        "batch_size_tag": train_cfg.get("batch_size_tag"),
        "batch_size_source": train_cfg.get("batch_size_source"),
        "epochs_requested": int(base_metrics_payload.get("epochs_requested", 0) or 0),
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
        **_collect_present_keys(base_metrics_payload, _PROMOTED_PREPARED_METADATA_KEYS),
        "rank_aux_effective": bool(base_metrics_payload.get("rank_aux_effective", base_metrics_payload.get("use_rank_aux", False))),
        **_collect_present_keys(base_metrics_payload, _PROMOTED_DEVICE_METADATA_KEYS),
    }
    save_json(metrics_path, aggregate_metrics)

    return {
        "reused_existing_training": True,
        "promoted_from_sweep": True,
        "post_selection_action": "use_sweep_artifacts",
        "selected_run_root": str(selected_run_root),
        "fold_runs_manifest_path": str(fold_runs_manifest_path),
        "metrics_path": str(metrics_path),
        "validation_metrics_path": str(validation_metrics_path),
        "history_path": str(history_path),
        "source_summary_csv": str(selected.get("source_summary_csv")) if selected.get("source_summary_csv") else None,
        "source_runs_csv": str(_default_sweep_runs_csv_from_summary(Path(str(selected["source_summary_csv"])))) if selected.get("source_summary_csv") else None,
    }


def _run_select_and_train(config: dict[str, Any], *, retrain: bool = False) -> dict[str, Any]:
    """Select λ and either materialize sweep artifacts or explicitly retrain.

    The CLI's default post-selection action is artifact reuse.  Fresh model
    fitting is opt-in via ``--retrain`` and must never be reached through a
    fallback when sweep artifacts are missing.
    """
    selection_result = run_lambda_selection(config)

    train_results: list[dict[str, Any]] = []
    for selected in selection_result.get("selected_configs", []):
        train_cfg = dict(selected["config"])
        if retrain:
            # This is the only code path that is allowed to fit new models.
            train_result = run_training(train_cfg)
            training_reused = False
            promoted_from_sweep = False
        else:
            promoted_result = _promote_selected_training_from_sweep(selected, train_cfg)
            if promoted_result is None:
                route = str(selected.get("route", "")).strip().lower()
                lambda_rank = selected.get("lambda_rank")
                raise RuntimeError(
                    "Selected sweep artifacts are incomplete for "
                    f"route={route!r}, lambda_rank={lambda_rank!r}; refusing implicit retraining. "
                    "Use `select-lambda --retrain` to explicitly train fresh models."
                )
            train_result = promoted_result
            training_reused = True
            promoted_from_sweep = True
        train_results.append(
            {
                "route": selected["route"],
                "lambda_rank": float(selected["lambda_rank"]),
                "config_path": selected["selected_config_path"],
                "training_reused": training_reused,
                "promoted_from_sweep": promoted_from_sweep,
                "post_selection_action": "retrain" if retrain else "use_sweep_artifacts",
                "selected_config_unchanged": bool(selected.get("selected_config_unchanged")),
                "train_result": train_result,
            }
        )
    selection_result["training_results"] = train_results
    selection_result["post_selection_action"] = "retrain" if retrain else "use_sweep_artifacts"
    return selection_result


def _run_command_handler(args: argparse.Namespace, cfg: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    if args.command == "train":
        return _run_with_metadata(base_dir, "train", cfg, lambda: run_training(cfg))
    if args.command == "validate":
        return _run_with_metadata(base_dir, "validate", cfg, lambda: run_validation(cfg))
    if args.command == "sweep-lambda":
        sweep_cfg = dict(cfg)
        sweep_cfg["reuse_existing"] = bool(args.reuse_existing)
        return _run_with_metadata(base_dir, "sweep_lambda", sweep_cfg, lambda: run_lambda_sweep(sweep_cfg))
    if args.command == "select-lambda":
        select_cfg = dict(cfg)
        select_cfg = apply_batch_size_path_tagging(select_cfg)
        retrain = bool(args.retrain)
        select_cfg["post_selection_action"] = "retrain" if retrain else "use_sweep_artifacts"
        return _run_with_metadata(
            base_dir,
            "select_lambda",
            select_cfg,
            lambda: _run_select_and_train(select_cfg, retrain=retrain),
        )
    raise ValueError(f"Unknown command: {args.command}")


def _run_show_config(args: argparse.Namespace) -> tuple[Any, bool]:
    cfg = _load_resolved_config_from_args(args)
    if bool(getattr(args, "with_validation", False)):
        payload = {
            "validation": {"ok": True, "message": "schema validation passed"},
            "config": cfg,
        }
        return payload, True
    return cfg, True


def _run_make_folds(args: argparse.Namespace, base_dir: Path) -> tuple[Any, bool]:
    cfg = _load_resolved_config_from_args(args)
    base_dir = Path(cfg.get("base_dir", ".")).resolve()

    def _action() -> dict[str, Any]:
        result = build_fold_manifest_from_config(cfg)
        output = resolve_path(args.output, base_dir=base_dir) if args.output else None
        if output is None:
            output = Path(cfg.get("split_manifest_path") or cfg["fixed_split_path"])
        manifest_path = save_manifest(result.manifest, output)
        audit_paths = save_manifest_cleaning_audit(
            manifest_path,
            result.prepared,
            result.dropped_rows_audit,
            data_path=cfg.get("data_path"),
            manifest=result.manifest,
            split_metadata=result.split_metadata,
        )
        preview = list(result.prepared.dropped_row_indices_preview)
        print(f"[make-folds] route={result.prepared.route} manifest={manifest_path}")
        print(
            "[make-folds] "
            f"clean_rows={len(result.prepared.frame)}/{result.prepared.raw_row_count} "
            f"dropped_rows={result.prepared.dropped_rows} "
            f"row_index_preview={preview}"
        )
        split_strategy = result.split_metadata.get("split_strategy", cfg.get("split_strategy"))
        block_size = result.split_metadata.get("spatial_block_size_m")
        split_detail = f"split_strategy={split_strategy}"
        if block_size is not None:
            split_detail += f" spatial_block_size_m={float(block_size):.3f}"
            split_detail += f" source={result.split_metadata.get('spatial_block_size_source', 'unknown')}"
        print(f"[make-folds] {split_detail}")
        print(
            "[make-folds] "
            f"cleaning_summary={audit_paths['summary_path']} "
            f"dropped_rows_csv={audit_paths['dropped_rows_path']}"
        )
        return {
            "output_path": str(manifest_path),
            "cleaning_summary_path": str(audit_paths["summary_path"]),
            "dropped_rows_path": str(audit_paths["dropped_rows_path"]),
            "clean_row_count": int(len(result.prepared.frame)),
            "split_metadata": result.split_metadata,
            **result.prepared.metadata(),
        }

    output_payload = _run_with_metadata(base_dir, "make_folds", cfg, _action)
    return str(output_payload["output_path"]), False


def _run_summarize(args: argparse.Namespace, base_dir: Path) -> tuple[Any, bool]:
    manifest = load_table(args.manifest)
    summary = summarize_manifest(manifest)
    output = resolve_path(args.output, base_dir=base_dir) if args.output else None
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        return str(output), False
    return summary.to_string(index=False), False


def _run_non_runtime_command(args: argparse.Namespace, base_dir: Path) -> tuple[Any, bool]:
    if args.command == "show-config":
        return _run_show_config(args)
    if args.command == "make-folds":
        return _run_make_folds(args, base_dir)
    if args.command == "summarize":
        return _run_summarize(args, base_dir)
    raise ValueError(f"Unknown non-runtime command: {args.command}")


def _apply_common_args(parser: argparse.ArgumentParser, arg_names: list[str]) -> None:
    builders: dict[str, Any] = {
        "config": _add_config_arg,
        "batch_size": _add_batch_size_arg,
        "rank_loss_mode": _add_rank_loss_mode_arg,
        "output_suffix": _add_output_suffix_arg,
    }
    for name in arg_names:
        builder = builders.get(name)
        if builder is None:
            raise ValueError(f"Unknown common parser arg: {name}")
        builder(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cvrank CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_folds = subparsers.add_parser("make-folds", help="Generate configured cross-validation manifests.")
    _apply_common_args(make_folds, ["config"])
    make_folds.add_argument("--output", type=str, default=None, help="Optional output CSV path.")

    summarize = subparsers.add_parser("summarize", help="Summarize a saved fold manifest.")
    summarize.add_argument("--manifest", type=str, required=True, help="Fold manifest CSV path.")
    summarize.add_argument("--output", type=str, default=None, help="Optional output CSV path.")

    show_cfg = subparsers.add_parser("show-config", help="Print resolved YAML config.")
    _apply_common_args(show_cfg, ["config"])
    show_cfg.add_argument(
        "--with-validation",
        action="store_true",
        help="Include schema validation status in output payload.",
    )

    train = subparsers.add_parser("train", help="Train the tabular model on one configured split.")
    _apply_common_args(train, ["config", "batch_size", "rank_loss_mode", "output_suffix"])

    validate = subparsers.add_parser("validate", help="Evaluate a trained checkpoint on the configured split.")
    _apply_common_args(validate, ["config", "batch_size", "rank_loss_mode", "output_suffix"])

    sweep_lambda = subparsers.add_parser(
        "sweep-lambda",
        help="Run lambda-grid sweep for rank-aware routes and collect ROC-AUC/F1 summaries.",
    )
    _apply_common_args(sweep_lambda, ["config", "batch_size", "rank_loss_mode", "output_suffix"])
    sweep_reuse_group = sweep_lambda.add_mutually_exclusive_group()
    sweep_reuse_group.add_argument(
        "--reuse-existing",
        dest="reuse_existing",
        action="store_true",
        help="Reuse only fingerprint-compatible, complete sweep runs.",
    )
    sweep_reuse_group.add_argument(
        "--no-reuse-existing",
        dest="reuse_existing",
        action="store_false",
        help="Force every requested sweep run to train again (default).",
    )
    sweep_lambda.set_defaults(reuse_existing=False)

    select_lambda = subparsers.add_parser(
        "select-lambda",
        help=(
            "Select the best lambda_rank and materialize the matching sweep artifacts "
            "(default); use --retrain to fit fresh final models."
        ),
    )
    _apply_common_args(select_lambda, ["config", "batch_size", "rank_loss_mode", "output_suffix"])
    select_lambda.add_argument(
        "--retrain",
        action="store_true",
        help="Explicitly retrain fresh final models after selecting lambda_rank.",
    )

    return parser


def _load_resolved_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    raw = load_yaml_config(args.config)
    return resolve_config_paths(raw)


def _run_with_metadata(base_dir: Path, command: str, payload: dict[str, Any], fn) -> Any:  # noqa: ANN001
    meta = {
        "command": command,
        "status": "started",
        "environment": collect_environment_snapshot(base_dir),
        "config": payload,
    }
    try:
        result = fn()
        if isinstance(result, dict):
            runtime_metadata = {key: result[key] for key in RUNTIME_METADATA_KEYS if key in result}
            if runtime_metadata:
                meta["runtime"] = runtime_metadata
        meta["status"] = "completed"
        return result
    except Exception as exc:
        meta["status"] = "failed"
        meta["error"] = repr(exc)
        raise
    finally:
        save_run_metadata(base_dir, command, meta)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    base_dir = Path.cwd().resolve()

    if args.command in NON_RUNTIME_COMMANDS:
        payload, as_json = _run_non_runtime_command(args, base_dir)
        if as_json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(payload)
        return

    if args.command in RUNTIME_CONFIG_COMMANDS:
        cfg = _load_resolved_config_from_args(args)
        base_dir = Path(cfg.get("base_dir", ".")).resolve()
        runner = PipelineRunner(command=args.command)
        cfg = runner.resolve_runtime_overrides(
            cfg,
            batch_size=getattr(args, "batch_size", None),
            rank_loss_mode=getattr(args, "rank_loss_mode", None) if hasattr(args, "rank_loss_mode") else None,
            output_suffix=getattr(args, "output_suffix", None) if hasattr(args, "output_suffix") else None,
        )

        result = _run_command_handler(args, cfg, base_dir)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        return

    raise ValueError(f"Unknown command: {args.command}")
