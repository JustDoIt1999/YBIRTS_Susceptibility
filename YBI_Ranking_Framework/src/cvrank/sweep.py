from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .batch_size import get_batch_size_tag, merge_batch_size_metadata, tag_name
from .config import load_yaml_config, resolve_config_paths, resolve_path
from .constants import DEFAULT_LAMBDA_SWEEP_GRID
from .rank_loss_mode import (
    apply_rank_loss_mode_override,
    apply_rank_loss_mode_path_tagging,
    get_explicit_rank_loss_mode,
    get_rank_loss_mode,
    merge_rank_loss_mode_metadata,
)
from .split import resolve_requested_fold_indices
from .training import run_training


def _resolve_route_config(route_cfg_path: Path) -> dict[str, Any]:
    return resolve_config_paths(load_yaml_config(route_cfg_path))

ALLOWED_SWEEP_ROUTES = {"dbscan", "no_cluster", "basin"}
RUN_IDENTITY_COLUMNS = [
    "route",
    "lambda_rank",
    "rank_loss_mode",
    "fold_index",
    "batch_size_tag",
]


def _resolve_sweep_grouping_rank_loss_mode(
    config: dict[str, Any],
    route_config_paths: list[Any],
    base_dir: Path,
) -> str | None:
    explicit_mode = get_explicit_rank_loss_mode(config)
    if explicit_mode is not None:
        return explicit_mode

    route_modes: set[str] = set()
    for route_cfg_path_raw in route_config_paths:
        route_cfg_path = resolve_path(route_cfg_path_raw, base_dir=base_dir)
        if route_cfg_path is None or not route_cfg_path.exists():
            continue
        route_cfg = _resolve_route_config(route_cfg_path)
        route_cfg = merge_rank_loss_mode_metadata(route_cfg, config)
        route_modes.add(get_rank_loss_mode(route_cfg))
        if len(route_modes) > 1:
            return None

    return next(iter(route_modes)) if len(route_modes) == 1 else None


def _lambda_label(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "p")


def _flatten_run_result(route: str, lambda_rank: float, fold_index: int, result: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    best_eval = result.get("best_eval") or {}
    return {
        "route": route,
        "lambda_rank": float(lambda_rank),
        "rank_loss_mode": result.get("rank_loss_mode"),
        "fold_index": int(fold_index),
        "split_mode": result.get("split_mode"),
        "split_name": result.get("split_name"),
        "best_epoch": result.get("best_epoch"),
        "best_val_loss_proxy": result.get("best_val_loss_proxy"),
        "roc_auc": best_eval.get("auc"),
        "f1": best_eval.get("f1"),
        "accuracy": best_eval.get("accuracy"),
        "kappa": best_eval.get("kappa"),
        "recall": best_eval.get("recall"),
        "precision": best_eval.get("precision"),
        "feature_count": result.get("feature_count"),
        "use_rank_aux": result.get("use_rank_aux"),
        "use_cluster_in_rank_loss": result.get("use_cluster_in_rank_loss"),
        "lambda_source": result.get("lambda_source"),
        "batch_size": result.get("batch_size"),
        "effective_batch_size": result.get("effective_batch_size"),
        "batch_size_tag": result.get("batch_size_tag"),
        "batch_size_source": result.get("batch_size_source"),
        "history_path": result.get("history_path"),
        "metrics_path": result.get("metrics_path"),
        "model_path": result.get("model_path"),
        "scaler_path": result.get("scaler_path"),
        "run_dir": str(run_dir),
    }


def _empty_runs_frame() -> pd.DataFrame:
    columns = list(_flatten_run_result("", 0.0, 0, {}, Path(".")))
    columns.append("run_config_path")
    return pd.DataFrame(columns=columns)


def _empty_run_row() -> dict[str, Any]:
    return {column: None for column in _empty_runs_frame().columns}


def _dedupe_lambda_grid(values: list[Any]) -> tuple[list[float], list[float]]:
    raw_values = [float(value) for value in values]
    deduped: list[float] = []
    dropped: list[float] = []
    seen: set[float] = set()
    for value in raw_values:
        if value in seen:
            dropped.append(value)
            continue
        seen.add(value)
        deduped.append(value)
    return raw_values, deduped, dropped


def _run_config_path(run_dir: Path) -> Path:
    return run_dir / "resolved_run_config.yaml"


def _sweep_result_path(run_dir: Path) -> Path:
    return run_dir / "sweep_result.json"


def _has_reusable_run_artifacts(run_dir: Path) -> bool:
    return _run_config_path(run_dir).exists() and _sweep_result_path(run_dir).exists()


def _has_partial_run_artifacts(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    return any(run_dir.iterdir())


def _normalize_existing_run_row(
    row: dict[str, Any],
    *,
    route: str,
    lambda_rank: float,
    fold_index: int,
    run_dir: Path,
    run_config_path: Path,
    rank_loss_mode: str,
    batch_size_tag: str | None,
) -> dict[str, Any]:
    normalized = _empty_run_row()
    normalized.update(dict(row))
    normalized["route"] = route
    normalized["lambda_rank"] = float(lambda_rank)
    normalized["fold_index"] = int(fold_index)
    normalized["run_dir"] = str(run_dir)
    normalized["run_config_path"] = str(run_config_path)
    normalized["rank_loss_mode"] = str(normalized.get("rank_loss_mode") or rank_loss_mode)
    normalized["batch_size_tag"] = normalized.get("batch_size_tag") or batch_size_tag
    return normalized


def _load_existing_run_row(
    run_dir: Path,
    *,
    route: str,
    lambda_rank: float,
    fold_index: int,
    rank_loss_mode: str,
    batch_size_tag: str | None,
) -> dict[str, Any] | None:
    if not _has_reusable_run_artifacts(run_dir):
        return None

    run_config_path = _run_config_path(run_dir)
    result_path = _sweep_result_path(run_dir)
    try:
        with result_path.open("r", encoding="utf-8") as fh:
            row = json.load(fh)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return None
    if not isinstance(row, dict):
        return None

    return _normalize_existing_run_row(
        row,
        route=route,
        lambda_rank=lambda_rank,
        fold_index=fold_index,
        run_dir=run_dir,
        run_config_path=run_config_path,
        rank_loss_mode=rank_loss_mode,
        batch_size_tag=batch_size_tag,
    )


def _normalize_optional_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _run_identity_key(
    *,
    route: str,
    lambda_rank: float,
    rank_loss_mode: str,
    fold_index: int,
    batch_size_tag: str | None,
) -> tuple[str, float, str, int, str | None]:
    return (
        str(route).strip().lower(),
        float(lambda_rank),
        str(rank_loss_mode or ""),
        int(fold_index),
        _normalize_optional_text(batch_size_tag),
    )


def _load_existing_runs_lookup(runs_csv: Path) -> dict[tuple[str, float, str, int, str | None], dict[str, Any]]:
    if not runs_csv.exists():
        return {}

    try:
        runs_df = pd.read_csv(runs_csv)
    except (OSError, ValueError, pd.errors.ParserError):
        return {}

    if runs_df.empty:
        return {}

    required_columns = {"route", "lambda_rank", "fold_index"}
    if not required_columns.issubset(runs_df.columns):
        return {}

    lookup: dict[tuple[str, float, str, int, str | None], dict[str, Any]] = {}
    for row in runs_df.to_dict(orient="records"):
        try:
            route = str(row.get("route", "")).strip().lower()
            lambda_rank = float(row.get("lambda_rank"))
            fold_index = int(row.get("fold_index"))
        except (TypeError, ValueError):
            continue

        rank_loss_mode = str(row.get("rank_loss_mode") or "")
        batch_size_tag = _normalize_optional_text(row.get("batch_size_tag"))
        run_dir_raw = row.get("run_dir")
        run_dir = Path(str(run_dir_raw)) if _normalize_optional_text(run_dir_raw) is not None else Path(".")
        run_config_raw = row.get("run_config_path")
        run_config_path = (
            Path(str(run_config_raw))
            if _normalize_optional_text(run_config_raw) is not None
            else _run_config_path(run_dir)
        )

        key = _run_identity_key(
            route=route,
            lambda_rank=lambda_rank,
            rank_loss_mode=rank_loss_mode,
            fold_index=fold_index,
            batch_size_tag=batch_size_tag,
        )
        lookup[key] = _normalize_existing_run_row(
            row,
            route=route,
            lambda_rank=lambda_rank,
            fold_index=fold_index,
            run_dir=run_dir,
            run_config_path=run_config_path,
            rank_loss_mode=rank_loss_mode,
            batch_size_tag=batch_size_tag,
        )

    return lookup


def _dedupe_run_rows(run_rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not run_rows:
        return _empty_runs_frame()

    runs_df = pd.DataFrame(run_rows)
    missing_columns = [column for column in RUN_IDENTITY_COLUMNS if column not in runs_df.columns]
    if missing_columns:
        raise ValueError(f"Sweep run rows missing identity columns: {missing_columns}")

    deduped_df = runs_df.drop_duplicates(subset=RUN_IDENTITY_COLUMNS, keep="last")
    return deduped_df.sort_values(["route", "rank_loss_mode", "lambda_rank", "fold_index"]).reset_index(drop=True)


def _resolve_sweep_fold_plan(config: dict[str, Any], route_config: dict[str, Any]) -> tuple[list[int], str]:
    requested = resolve_requested_fold_indices(config, config.get("split_manifest_path"))
    if requested:
        return requested, "explicit"

    route_requested = resolve_requested_fold_indices(route_config, route_config.get("split_manifest_path"))
    if route_requested:
        source = "manifest" if route_config.get("split_manifest_path") else "explicit"
        return route_requested, source

    raise ValueError(
        "Lambda sweep requires explicit `fold_index`/`fold_indices` or a readable `split_manifest_path`; "
        "implicit single-fold fallback is no longer supported."
    )


def _resolve_sweep_fold_indices(config: dict[str, object], route_config: dict[str, object]) -> list[int]:
    fold_indices, _ = _resolve_sweep_fold_plan(config, route_config)
    return fold_indices


def _build_run_config(
    base_config: dict[str, Any],
    sweep_root: Path,
    route: str,
    lambda_rank: float,
    fold_index: int,
) -> tuple[dict[str, Any], Path]:
    batch_size_tag = get_batch_size_tag(base_config)
    lambda_dir = f"lambda_{_lambda_label(lambda_rank)}"
    run_dir = sweep_root / route / lambda_dir / f"fold_{fold_index}"
    model_dir = run_dir / "model"
    prefix = f"{route}_lambda_{_lambda_label(lambda_rank)}_fold_{fold_index}"
    tagged_prefix = tag_name(prefix, batch_size_tag) if batch_size_tag else prefix

    run_config = deepcopy(base_config)
    run_config.pop("fold_indices", None)
    run_config["lambda_rank"] = float(lambda_rank)
    run_config["fold_index"] = int(fold_index)
    run_config["output_root"] = str(run_dir)
    run_config["model_path"] = str(model_dir / f"{tagged_prefix}_model.pth")
    run_config["scaler_path"] = str(model_dir / f"{tagged_prefix}_scaler.pkl")
    run_config["metrics_path"] = str(model_dir / f"{tagged_prefix}_train_metrics.json")
    run_config["history_path"] = str(model_dir / f"{tagged_prefix}_train_history.csv")
    run_config["validation_metrics_path"] = str(model_dir / f"{tagged_prefix}_validation_metrics.json")
    run_config = apply_rank_loss_mode_override(run_config, None)
    return run_config, run_dir


def run_lambda_sweep(config: dict[str, Any]) -> dict[str, Any]:
    base_dir = Path(config.get("base_dir", ".")).resolve()
    route_config_paths = config.get("route_config_paths")
    if not isinstance(route_config_paths, list) or not route_config_paths:
        raise ValueError("`route_config_paths` must be a non-empty list of route config paths.")

    path_tagging_config = dict(config)
    grouping_rank_loss_mode = _resolve_sweep_grouping_rank_loss_mode(config, route_config_paths, base_dir)
    if grouping_rank_loss_mode is not None:
        path_tagging_config["rank_loss_mode"] = grouping_rank_loss_mode
    path_tagging_config = apply_rank_loss_mode_path_tagging(path_tagging_config)

    sweep_root = resolve_path(path_tagging_config.get("output_root") or "./data/output/sweeps/lambda_rank", base_dir=base_dir)
    if sweep_root is None:
        sweep_root = (base_dir / "data" / "output" / "sweeps" / "lambda_rank").resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    runs_csv = resolve_path(path_tagging_config.get("runs_csv") or (sweep_root / "lambda_sweep_runs.csv"), base_dir=base_dir)
    summary_csv = resolve_path(path_tagging_config.get("summary_csv") or (sweep_root / "lambda_sweep_summary.csv"), base_dir=base_dir)
    if runs_csv is None or summary_csv is None:
        raise ValueError("Failed to resolve sweep output paths.")
    existing_runs_lookup = _load_existing_runs_lookup(runs_csv)

    raw_lambda_grid, lambda_grid, dropped_duplicate_lambda_values = _dedupe_lambda_grid(
        list(config.get("lambda_grid", DEFAULT_LAMBDA_SWEEP_GRID))
    )

    run_rows: list[dict[str, Any]] = []
    route_fold_indices: dict[str, list[int]] = {}
    route_fold_resolution_source: dict[str, str] = {}
    n_requested_runs = 0
    n_unique_requested_runs = 0
    n_reused_runs = 0
    n_executed_runs = 0
    n_recovered_or_rerun_runs = 0
    for route_cfg_path in route_config_paths:
        route_cfg_path = resolve_path(route_cfg_path, base_dir=base_dir)
        if route_cfg_path is None:
            raise ValueError("Invalid route config path in sweep config.")
        route_cfg = _resolve_route_config(route_cfg_path)
        route_cfg = merge_batch_size_metadata(route_cfg, config)
        route_cfg = merge_rank_loss_mode_metadata(route_cfg, config)
        route_cfg = apply_rank_loss_mode_override(route_cfg, None)
        route = str(route_cfg.get("route", "")).strip().lower()
        if route not in ALLOWED_SWEEP_ROUTES:
            raise ValueError(f"Route `{route}` is not allowed in lambda sweep. Allowed: {sorted(ALLOWED_SWEEP_ROUTES)}")

        fold_indices, fold_resolution_source = _resolve_sweep_fold_plan(config, route_cfg)
        route_fold_indices[route] = list(fold_indices)
        route_fold_resolution_source[route] = fold_resolution_source
        n_requested_runs += len(fold_indices) * len(raw_lambda_grid)
        n_unique_requested_runs += len(fold_indices) * len(lambda_grid)
        for fold_index in fold_indices:
            for lambda_rank in lambda_grid:
                run_config, run_dir = _build_run_config(route_cfg, sweep_root, route, lambda_rank, fold_index)
                existing_key = _run_identity_key(
                    route=route,
                    lambda_rank=lambda_rank,
                    rank_loss_mode=str(run_config.get("rank_loss_mode") or ""),
                    fold_index=fold_index,
                    batch_size_tag=run_config.get("batch_size_tag"),
                )
                existing_row = None
                if _has_reusable_run_artifacts(run_dir):
                    existing_row = existing_runs_lookup.get(existing_key)
                if existing_row is None:
                    existing_row = _load_existing_run_row(
                    run_dir,
                    route=route,
                    lambda_rank=lambda_rank,
                    fold_index=fold_index,
                    rank_loss_mode=str(run_config.get("rank_loss_mode") or ""),
                    batch_size_tag=run_config.get("batch_size_tag"),
                    )
                if existing_row is not None:
                    run_rows.append(existing_row)
                    n_reused_runs += 1
                    continue

                rerun_existing = _has_partial_run_artifacts(run_dir)
                run_dir.mkdir(parents=True, exist_ok=True)
                run_config_path = _run_config_path(run_dir)
                with run_config_path.open("w", encoding="utf-8") as fh:
                    yaml.safe_dump(run_config, fh, sort_keys=False, allow_unicode=True)

                result = run_training(run_config)
                row = _flatten_run_result(route, lambda_rank, fold_index, result, run_dir)
                row["run_config_path"] = str(run_config_path)
                run_rows.append(row)
                n_executed_runs += 1
                if rerun_existing:
                    n_recovered_or_rerun_runs += 1

                with _sweep_result_path(run_dir).open("w", encoding="utf-8") as fh:
                    json.dump(row, fh, ensure_ascii=False, indent=2, default=str)

    runs_df = _dedupe_run_rows(run_rows)

    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    runs_df.to_csv(runs_csv, index=False)

    if runs_df.empty:
        summary_df = pd.DataFrame(
            columns=[
                "route",
                "lambda_rank",
                "rank_loss_mode",
                "n_runs",
                "roc_auc_mean",
                "roc_auc_std",
                "f1_mean",
                "f1_std",
                "accuracy_mean",
                "accuracy_std",
                "kappa_mean",
                "kappa_std",
                "recall_mean",
                "recall_std",
                "precision_mean",
                "precision_std",
                "best_epoch_mean",
                "best_epoch_std",
            ]
        )
    else:
        summary_df = (
            runs_df.groupby(["route", "lambda_rank", "rank_loss_mode"], as_index=False)
            .agg(
                n_runs=("f1", "size"),
                roc_auc_mean=("roc_auc", "mean"),
                roc_auc_std=("roc_auc", "std"),
                f1_mean=("f1", "mean"),
                f1_std=("f1", "std"),
                accuracy_mean=("accuracy", "mean"),
                accuracy_std=("accuracy", "std"),
                kappa_mean=("kappa", "mean"),
                kappa_std=("kappa", "std"),
                recall_mean=("recall", "mean"),
                recall_std=("recall", "std"),
                precision_mean=("precision", "mean"),
                precision_std=("precision", "std"),
                best_epoch_mean=("best_epoch", "mean"),
                best_epoch_std=("best_epoch", "std"),
            )
            .sort_values(["route", "rank_loss_mode", "lambda_rank"])
            .reset_index(drop=True)
        )
    summary_df.to_csv(summary_csv, index=False)

    batch_size_tag = get_batch_size_tag(config)
    manifest = {
        "sweep_name": str(config.get("sweep_name", "lambda_sweep")),
        "base_dir": str(base_dir),
        "output_root": str(sweep_root),
        "lambda_grid": lambda_grid,
        "raw_lambda_grid": raw_lambda_grid,
        "deduped_lambda_grid": lambda_grid,
        "dropped_duplicate_lambda_values": dropped_duplicate_lambda_values,
        "fold_indices": sorted({fold_index for values in route_fold_indices.values() for fold_index in values}),
        "route_fold_indices": route_fold_indices,
        "fold_resolution_source": (
            next(iter(set(route_fold_resolution_source.values())))
            if len(set(route_fold_resolution_source.values())) == 1
            else "mixed"
        ),
        "route_fold_resolution_source": route_fold_resolution_source,
        "route_config_paths": [str(resolve_path(path, base_dir=base_dir) or path) for path in route_config_paths],
        "runs_csv": str(runs_csv),
        "summary_csv": str(summary_csv),
        "batch_size": int(config.get("batch_size")) if config.get("batch_size") is not None else None,
        "batch_size_tag": config.get("batch_size_tag"),
        "batch_size_source": config.get("batch_size_source"),
        "rank_loss_modes": sorted(set(runs_df["rank_loss_mode"].dropna().astype(str))) if not runs_df.empty else [],
        "n_runs": int(len(runs_df)),
        "n_requested_runs": int(n_requested_runs),
        "n_unique_requested_runs": int(n_unique_requested_runs),
        "n_reused_runs": int(n_reused_runs),
        "n_executed_runs": int(n_executed_runs),
        "n_recovered_or_rerun_runs": int(n_recovered_or_rerun_runs),
        "routes": sorted(set(runs_df["route"].astype(str))) if not runs_df.empty else [],
    }
    manifest_path = sweep_root / tag_name("lambda_sweep_manifest.json", batch_size_tag)
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)

    print(f"Lambda sweep runs saved: {runs_csv}")
    print(f"Lambda sweep summary saved: {summary_csv}")
    return manifest


__all__ = ["_resolve_sweep_fold_indices", "run_lambda_sweep"]
