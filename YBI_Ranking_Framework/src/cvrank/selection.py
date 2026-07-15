from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .artifact_layout import (
    ARTIFACT_LAYOUT_VERSION,
    build_selected_route_root,
    make_selected_model_identity,
)
from .batch_size import (
    BATCH_SIZE_SOURCE_CONFIG,
    BATCH_SIZE_SOURCE_KEY,
    BATCH_SIZE_TAG_KEY,
    make_batch_size_tag,
    merge_batch_size_metadata,
)
from .config import load_yaml_config, resolve_config_paths, resolve_path
from .output_suffix import get_output_suffix
from .rank_loss_mode import (
    SUPPORTED_RANK_LOSS_MODES,
    apply_rank_loss_mode_override,
    get_explicit_rank_loss_mode,
    get_rank_loss_mode,
    merge_rank_loss_mode_metadata,
    normalize_rank_loss_mode,
    tag_rank_loss_mode_directory,
    tag_rank_loss_mode_file_parent,
)
from .constants import RANK_LOSS_MODE_DISABLED


def _resolve_route_config(route_cfg_path: Path) -> dict[str, Any]:
    return resolve_config_paths(load_yaml_config(route_cfg_path))


ALLOWED_SELECTION_ROUTES = {"dbscan", "no_cluster", "basin"}
LAMBDA_RANK_KEY = "lambda_rank"
DEFAULT_LAMBDA_RANK = 1.0
SELECTION_COMPARISON_DECIMALS = 3
SELECTION_SORT_COLUMNS = [
    "roc_auc_mean",
    "f1_mean",
    "accuracy_mean",
    "kappa_mean",
    "lambda_rank",
]
SELECTION_SORT_ASCENDING = [False, False, False, False, False]
SELECTION_BASE_REQUIRED_COLUMNS = [
    "route",
    "lambda_rank",
    "rank_loss_mode",
    "roc_auc_mean",
    "accuracy_mean",
]
OPTIONAL_RANK_SORT_COLUMN = "pairwise_rank_accuracy_mean"
SELECTION_STRATEGY_S1 = "s1"
SELECTION_STRATEGY_S2 = "s2"
SELECTION_POLICY = "s2_first_adaptive_s1_fallback"
CLASSIFICATION_BASELINE_POLICY = "classification_baseline"
SELECTION_POLICY_KEY = "selection_policy"
SELECTION_FILTER_TYPE_NONE = "none"
SELECTION_FILTER_TYPE_BALANCED_BASELINE_GATE = "balanced_baseline_gate"

_TRAINING_FILE_KEYS = {
    "model_path",
    "scaler_path",
    "metrics_path",
    "validation_metrics_path",
    "history_path",
    "fold_runs_manifest_path",
}


def normalize_lambda_rank(value: Any) -> float:
    try:
        lambda_rank = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid lambda_rank: {value!r}") from exc
    if not math.isfinite(lambda_rank):
        raise ValueError(f"lambda_rank must be finite, got {lambda_rank}.")
    if lambda_rank < 0:
        raise ValueError(f"lambda_rank must be non-negative, got {lambda_rank}.")
    return lambda_rank


def get_lambda_rank(config: dict[str, Any] | None) -> float:
    if not isinstance(config, dict):
        return DEFAULT_LAMBDA_RANK
    raw_value = config.get(LAMBDA_RANK_KEY, config.get("alpha"))
    if raw_value in (None, ""):
        return DEFAULT_LAMBDA_RANK
    return normalize_lambda_rank(raw_value)


def apply_lambda_rank_override(config: dict[str, Any], requested_lambda_rank: Any | None) -> dict[str, Any]:
    updated = dict(config)
    if requested_lambda_rank is not None:
        updated[LAMBDA_RANK_KEY] = normalize_lambda_rank(requested_lambda_rank)
        return updated
    updated[LAMBDA_RANK_KEY] = get_lambda_rank(updated)
    return updated


def ensure_selection_policy_not_configurable(config: dict[str, Any]) -> None:
    legacy_keys = ("selection_strategy", "selection_strategy_tag", "selection_strategy_source")
    configured = [key for key in legacy_keys if config.get(key) not in (None, "")]
    if configured:
        raise ValueError(
            "Lambda selection uses the fixed adaptive policy "
            f"`{SELECTION_POLICY}`; remove unsupported configurable keys: {configured}."
        )


def ensure_required_columns(frame: pd.DataFrame, required: list[str]) -> None:
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Selection frame missing required columns: {missing}")


def to_float(value: Any) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return float("nan")
    return float(parsed)


def round_metric_value(value: Any) -> float:
    parsed = to_float(value)
    if math.isnan(parsed):
        return float("nan")
    return round(parsed, SELECTION_COMPARISON_DECIMALS)


def required_summary_columns() -> set[str]:
    required = set(SELECTION_BASE_REQUIRED_COLUMNS)
    required.add("f1_mean")
    required.add("kappa_mean")
    return required


def clean_candidate_frame(route_df: pd.DataFrame) -> pd.DataFrame:
    candidate = route_df.copy()
    if OPTIONAL_RANK_SORT_COLUMN not in candidate.columns:
        candidate[OPTIONAL_RANK_SORT_COLUMN] = float("nan")

    numeric_columns = {
        "lambda_rank",
        "roc_auc_mean",
        "accuracy_mean",
        "f1_mean",
        "kappa_mean",
        "recall_mean",
        "precision_mean",
        OPTIONAL_RANK_SORT_COLUMN,
    }
    for column in numeric_columns:
        if column not in candidate.columns:
            candidate[column] = float("nan")
        candidate[column] = pd.to_numeric(candidate[column], errors="coerce")

    candidate = candidate.dropna(subset=["roc_auc_mean", "accuracy_mean", "f1_mean", "kappa_mean", "lambda_rank"])
    if candidate.empty:
        raise ValueError("No valid lambda candidates remain after removing missing metrics.")
    return candidate


def sort_candidate_frame(candidate: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[bool]]:
    sorted_candidate = candidate.sort_values(
        SELECTION_SORT_COLUMNS,
        ascending=SELECTION_SORT_ASCENDING,
        kind="mergesort",
        na_position="last",
    )
    return sorted_candidate, list(SELECTION_SORT_COLUMNS), list(SELECTION_SORT_ASCENDING)


def resolve_zero_lambda_baseline(candidate: pd.DataFrame) -> pd.Series:
    baseline = candidate[candidate["lambda_rank"] == 0]
    if baseline.empty:
        raise ValueError("Selection requires a lambda=0 baseline candidate for each route and rank_loss_mode.")
    ranked_baseline, _, _ = sort_candidate_frame(baseline)
    return ranked_baseline.iloc[0]


def _build_selection_metadata(
    strategy: str,
    baseline: pd.Series,
    candidate: pd.DataFrame,
    nonzero_candidate: pd.DataFrame,
    eligible: pd.DataFrame,
    filtered: pd.DataFrame,
    filter_type: str,
) -> dict[str, Any]:
    return {
        SELECTION_POLICY_KEY: SELECTION_POLICY,
        "selection_stage": strategy,
        "selection_filter_type": filter_type,
        "baseline_lambda_rank": float(baseline["lambda_rank"]),
        "baseline_roc_auc_mean": float(baseline["roc_auc_mean"]),
        "baseline_accuracy_mean": float(baseline["accuracy_mean"]),
        "baseline_f1_mean": float(baseline["f1_mean"]),
        "baseline_kappa_mean": float(baseline["kappa_mean"]),
        "total_candidate_count": int(len(candidate)),
        "nonzero_candidate_count": int(len(nonzero_candidate)),
        "eligible_candidate_count": int(len(eligible)),
        "filtered_candidate_count": int(len(filtered)),
    }


def _resolve_nonzero_candidates(candidate: pd.DataFrame) -> pd.DataFrame:
    return candidate[candidate["lambda_rank"] != 0].copy()


def _select_s1_candidate(candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = resolve_zero_lambda_baseline(candidate)
    nonzero_candidate = _resolve_nonzero_candidates(candidate)
    eligible = candidate.copy()
    filtered = candidate.iloc[0:0].copy()
    metadata = _build_selection_metadata(
        SELECTION_STRATEGY_S1,
        baseline,
        candidate,
        nonzero_candidate,
        eligible,
        filtered,
        SELECTION_FILTER_TYPE_NONE,
    )
    return eligible, metadata


def _select_s2_candidate(candidate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    baseline = resolve_zero_lambda_baseline(candidate)
    nonzero_candidate = _resolve_nonzero_candidates(candidate)
    eligible_mask = (
        (nonzero_candidate["roc_auc_mean"] >= float(baseline["roc_auc_mean"]))
        & (nonzero_candidate["accuracy_mean"] >= float(baseline["accuracy_mean"]))
        & (nonzero_candidate["f1_mean"] >= float(baseline["f1_mean"]))
        & (nonzero_candidate["kappa_mean"] >= float(baseline["kappa_mean"]))
    )
    eligible = nonzero_candidate.loc[eligible_mask].copy()
    filtered = pd.concat(
        [candidate[candidate["lambda_rank"] == 0].copy(), nonzero_candidate.loc[~eligible_mask].copy()],
        ignore_index=True,
    )
    metadata = _build_selection_metadata(
        SELECTION_STRATEGY_S2,
        baseline,
        candidate,
        nonzero_candidate,
        eligible,
        filtered,
        SELECTION_FILTER_TYPE_BALANCED_BASELINE_GATE,
    )
    return eligible, metadata


def pick_top_candidate_with_metadata(
    frame: pd.DataFrame,
) -> tuple[pd.Series, list[str], list[bool], dict[str, Any]]:
    cleaned = clean_candidate_frame(frame)
    eligible, metadata = _select_s2_candidate(cleaned)
    metadata["selection_stage_actual"] = SELECTION_STRATEGY_S2
    metadata["selection_fallback_used"] = False
    if eligible.empty:
        print("[Selection] S2 matched no eligible nonzero lambda. Adaptive fallback to S1...")
        eligible, fallback_metadata = _select_s1_candidate(cleaned)
        metadata["selection_stage_actual"] = "s1_fallback"
        metadata["selection_fallback_used"] = True
        metadata["fallback_eligible_candidate_count"] = int(fallback_metadata["eligible_candidate_count"])
        metadata["fallback_filtered_candidate_count"] = int(fallback_metadata["filtered_candidate_count"])

    if eligible.empty:
        baseline = resolve_zero_lambda_baseline(cleaned)
        metadata["selection_used_lambda_zero_fallback"] = True
        return baseline, list(SELECTION_SORT_COLUMNS), list(SELECTION_SORT_ASCENDING), metadata

    ranked, sort_columns, sort_ascending = sort_candidate_frame(eligible)
    best_lambda_rank = float(ranked.iloc[0]["lambda_rank"])
    selected = cleaned[cleaned["lambda_rank"] == best_lambda_rank]
    if selected.empty:
        raise ValueError(f"Unable to map selected lambda_rank={best_lambda_rank} back to original metrics.")
    metadata["selection_used_lambda_zero_fallback"] = bool(best_lambda_rank == 0)
    return selected.iloc[0], sort_columns, sort_ascending, metadata


def pick_classification_baseline_candidate_with_metadata(
    frame: pd.DataFrame,
) -> tuple[pd.Series, list[str], list[bool], dict[str, Any]]:
    cleaned = clean_candidate_frame(frame)
    baseline_rows = cleaned[cleaned["lambda_rank"] == 0].copy()
    if baseline_rows.empty:
        raise ValueError("Classification-baseline selection requires a lambda=0 candidate.")
    ranked, sort_columns, sort_ascending = sort_candidate_frame(baseline_rows)
    baseline = ranked.iloc[0]
    nonzero = cleaned[cleaned["lambda_rank"] != 0].copy()
    metadata = {
        SELECTION_POLICY_KEY: CLASSIFICATION_BASELINE_POLICY,
        "selection_stage": CLASSIFICATION_BASELINE_POLICY,
        "selection_stage_actual": CLASSIFICATION_BASELINE_POLICY,
        "selection_filter_type": SELECTION_FILTER_TYPE_NONE,
        "selection_fallback_used": False,
        "selection_used_lambda_zero_fallback": False,
        "baseline_lambda_rank": 0.0,
        "baseline_roc_auc_mean": float(baseline["roc_auc_mean"]),
        "baseline_accuracy_mean": float(baseline["accuracy_mean"]),
        "baseline_f1_mean": float(baseline["f1_mean"]),
        "baseline_kappa_mean": float(baseline["kappa_mean"]),
        "total_candidate_count": int(len(cleaned)),
        "nonzero_candidate_count": int(len(nonzero)),
        "eligible_candidate_count": int(len(baseline_rows)),
        "filtered_candidate_count": int(len(nonzero)),
    }
    return baseline, sort_columns, sort_ascending, metadata


def pick_top_candidate(
    frame: pd.DataFrame,
) -> tuple[pd.Series, list[str], list[bool]]:
    best_row, sort_columns, sort_ascending, _metadata = pick_top_candidate_with_metadata(frame)
    return best_row, sort_columns, sort_ascending


def lambda_label(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "p")


def selection_workspace_root(selection_output_root: Path) -> Path:
    root = Path(selection_output_root).resolve()
    for candidate in (root, *root.parents):
        if candidate.name == "selected_lambda" or candidate.name.startswith("selected_lambda_"):
            return candidate.parent
    raise ValueError(
        "Unable to locate the CVRank output workspace from selection output root: "
        f"{selection_output_root}"
    )


def selection_dir_for_output_root(output_root: Path) -> Path:
    return output_root / "selection"


def build_final_training_config(
    base_config: dict[str, Any],
    selected_row,
    selection_output_root: Path,
    *,
    selection_policy: str = SELECTION_POLICY,
) -> tuple[dict[str, Any], Path, Path, Path, bool]:
    route = str(base_config.get("route", "")).strip().lower()
    lambda_rank = float(selected_row["lambda_rank"])
    final_config = deepcopy(base_config)
    final_config.pop("fold_index", None)
    final_config.pop("fold_indices", None)
    selected_rank_loss_mode = selected_row.get(
        "rank_loss_mode",
        base_config.get("rank_loss_mode", "ndcg_aware_lambdarank"),
    )
    final_config["rank_loss_mode"] = normalize_rank_loss_mode(selected_rank_loss_mode)
    if final_config["rank_loss_mode"] == RANK_LOSS_MODE_DISABLED and lambda_rank != 0:
        raise ValueError(
            "Classification-baseline training requires lambda_rank=0 when rank_loss_mode=disabled."
        )
    final_config = apply_rank_loss_mode_override(final_config, None)
    final_config = apply_lambda_rank_override(final_config, lambda_rank)
    identity = make_selected_model_identity(
        route=route,
        batch_size=final_config.get("batch_size"),
        rank_loss_mode=final_config["rank_loss_mode"],
        train_output_suffix=get_output_suffix(final_config),
    )
    final_config["batch_size"] = identity.batch_size
    final_config[BATCH_SIZE_TAG_KEY] = make_batch_size_tag(identity.batch_size)
    if final_config.get(BATCH_SIZE_SOURCE_KEY) in (None, ""):
        final_config[BATCH_SIZE_SOURCE_KEY] = BATCH_SIZE_SOURCE_CONFIG
    final_config["artifact_layout_version"] = ARTIFACT_LAYOUT_VERSION
    final_config["artifact_identity"] = {
        **identity.as_dict(),
        SELECTION_POLICY_KEY: selection_policy,
    }
    final_run_root = build_selected_route_root(
        selection_workspace_root(selection_output_root),
        identity,
    )
    model_dir = final_run_root / "model"
    prefix = f"{route}_lambda_{lambda_label(lambda_rank)}"
    final_config["output_root"] = str(final_run_root)
    final_config["model_path"] = str(model_dir / f"{prefix}_model.pth")
    final_config["scaler_path"] = str(model_dir / f"{prefix}_scaler.pkl")
    final_config["metrics_path"] = str(model_dir / f"{prefix}_train_metrics.json")
    final_config["validation_metrics_path"] = str(model_dir / f"{prefix}_validation_metrics.json")
    final_config["history_path"] = str(model_dir / f"{prefix}_train_history.csv")
    final_config["fold_runs_manifest_path"] = str(final_run_root / "fold_runs_manifest.json")
    final_config["selection_source"] = (
        "classification_baseline" if selection_policy == CLASSIFICATION_BASELINE_POLICY else "lambda_sweep_summary"
    )
    final_config[SELECTION_POLICY_KEY] = selection_policy
    selection_dir = selection_dir_for_output_root(final_run_root)

    final_config_path = selection_dir / f"{route}_best_train.yaml"
    selection_decision_path = selection_dir / f"{route}_selection_decision.json"
    final_config_path.parent.mkdir(parents=True, exist_ok=True)
    existing_config: dict[str, Any] | None = None
    if final_config_path.exists():
        with final_config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            existing_config = loaded

    selected_config_unchanged = existing_config == final_config
    if not selected_config_unchanged:
        with final_config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(final_config, fh, sort_keys=False, allow_unicode=True)

    return final_config, final_config_path, final_run_root, selection_decision_path, selected_config_unchanged


_build_final_training_config = build_final_training_config


def _series_float(series: pd.Series, key: str) -> float:
    if key not in series.index:
        return float("nan")
    return to_float(series.get(key))


def _series_metric(series: pd.Series, key: str) -> float:
    if key not in series.index:
        return float("nan")
    return round_metric_value(series.get(key))


def _resolve_selection_grouping_rank_loss_mode(config: dict[str, Any], base_dir: Path) -> str | None:
    explicit_mode = get_explicit_rank_loss_mode(config)
    if explicit_mode is not None:
        return explicit_mode

    route_config_paths = config.get("route_config_paths")
    if not isinstance(route_config_paths, list) or not route_config_paths:
        return None

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


def _require_summary_columns(summary_df: pd.DataFrame) -> None:
    ensure_required_columns(summary_df, sorted(required_summary_columns()))
    rank_loss_modes = summary_df["rank_loss_mode"].dropna().astype(str).str.strip().unique().tolist()
    if not rank_loss_modes:
        raise ValueError("Lambda sweep summary must include non-empty `rank_loss_mode` values.")
    invalid: list[str] = []
    for mode in rank_loss_modes:
        try:
            normalized = normalize_rank_loss_mode(mode)
        except ValueError:
            invalid.append(mode)
            continue
        if normalized not in SUPPORTED_RANK_LOSS_MODES:
            invalid.append(mode)
    if invalid:
        raise ValueError(
            "Lambda sweep summary contains incompatible rank loss modes "
            f"{invalid}; rerun `sweep-lambda` with a supported rank loss mode."
        )


def _pick_best_candidate(
    route_df: pd.DataFrame,
) -> tuple[pd.Series, list[str], list[bool]]:
    return pick_top_candidate(route_df)


def run_lambda_selection(config: dict[str, Any]) -> dict[str, Any]:
    ensure_selection_policy_not_configurable(config)
    config = dict(config)
    config[SELECTION_POLICY_KEY] = SELECTION_POLICY
    base_dir = Path(config.get("base_dir", ".")).resolve()
    path_tagging_config = dict(config)
    grouping_rank_loss_mode = _resolve_selection_grouping_rank_loss_mode(config, base_dir)
    if grouping_rank_loss_mode is not None:
        path_tagging_config["rank_loss_mode"] = grouping_rank_loss_mode
    summary_csv = resolve_path(config.get("summary_csv"), base_dir=base_dir)
    if summary_csv is None:
        raise ValueError("`summary_csv` is required for lambda selection.")
    tagged_summary_csv = tag_rank_loss_mode_file_parent(summary_csv, path_tagging_config)
    if tagged_summary_csv.exists():
        summary_csv = tagged_summary_csv
    if not summary_csv.exists():
        raise FileNotFoundError(f"Lambda sweep summary not found: {tagged_summary_csv}")

    output_root = resolve_path(config.get("output_root") or "./data/output/selected_lambda", base_dir=base_dir)
    if output_root is None:
        output_root = (base_dir / "data" / "output" / "selected_lambda").resolve()
    output_root = tag_rank_loss_mode_directory(output_root, path_tagging_config)
    output_root.mkdir(parents=True, exist_ok=True)

    route_config_paths = config.get("route_config_paths")
    if not isinstance(route_config_paths, list) or not route_config_paths:
        raise ValueError("`route_config_paths` must be a non-empty list of route config paths.")

    sweep_df = pd.read_csv(summary_csv)
    _require_summary_columns(sweep_df)

    selected_rows: list[dict[str, Any]] = []
    selected_configs: list[dict[str, Any]] = []
    config_paths: list[str] = []

    for route_cfg_path_raw in route_config_paths:
        route_cfg_path = resolve_path(route_cfg_path_raw, base_dir=base_dir)
        if route_cfg_path is None:
            raise ValueError(f"Invalid route config path: {route_cfg_path_raw}")
        if not route_cfg_path.exists():
            raise FileNotFoundError(f"Route config not found: {route_cfg_path}")

        route_cfg = _resolve_route_config(route_cfg_path)
        route_cfg = merge_batch_size_metadata(route_cfg, config)
        route_cfg = merge_rank_loss_mode_metadata(route_cfg, config)
        route_cfg = apply_rank_loss_mode_override(route_cfg, None)
        if get_output_suffix(config) is not None:
            route_cfg["output_suffix"] = get_output_suffix(config)
        ensure_selection_policy_not_configurable(route_cfg)
        route = str(route_cfg.get("route", "")).strip().lower()
        if route not in ALLOWED_SELECTION_ROUTES:
            raise ValueError(f"Route `{route}` is not allowed for lambda selection.")

        route_df = sweep_df[sweep_df["route"].astype(str).str.lower() == route].copy()
        if route_df.empty:
            raise ValueError(f"No sweep summary rows found for route `{route}` in {summary_csv}")
        route_df["rank_loss_mode"] = route_df["rank_loss_mode"].map(normalize_rank_loss_mode)

        target_rank_loss_mode = get_rank_loss_mode(route_cfg)
        route_df = route_df[route_df["rank_loss_mode"] == target_rank_loss_mode].copy()
        if route_df.empty:
            raise ValueError(
                f"No sweep summary rows found for route `{route}` with rank_loss_mode "
                f"`{target_rank_loss_mode}` in {summary_csv}"
            )

        effective_selection_policy = (
            CLASSIFICATION_BASELINE_POLICY
            if target_rank_loss_mode == RANK_LOSS_MODE_DISABLED
            else SELECTION_POLICY
        )
        route_cfg[SELECTION_POLICY_KEY] = effective_selection_policy
        if target_rank_loss_mode == RANK_LOSS_MODE_DISABLED:
            best_row, sort_columns, sort_ascending, decision_metadata = (
                pick_classification_baseline_candidate_with_metadata(route_df)
            )
        else:
            best_row, sort_columns, sort_ascending, decision_metadata = pick_top_candidate_with_metadata(route_df)
        final_config, final_config_path, final_run_root, selection_decision_path, selected_config_unchanged = build_final_training_config(
            route_cfg,
            best_row,
            output_root,
            selection_policy=effective_selection_policy,
        )

        selected_row = {
            "route": route,
            "lambda_rank": float(best_row["lambda_rank"]),
            "rank_loss_mode": str(best_row["rank_loss_mode"]),
            "roc_auc_mean": round_metric_value(best_row["roc_auc_mean"]),
            "roc_auc_std": _series_metric(best_row, "roc_auc_std"),
            "accuracy_mean": round_metric_value(best_row["accuracy_mean"]),
            "accuracy_std": _series_metric(best_row, "accuracy_std"),
            "f1_mean": round_metric_value(best_row["f1_mean"]),
            "f1_std": _series_metric(best_row, "f1_std"),
            "kappa_mean": _series_metric(best_row, "kappa_mean"),
            "kappa_std": _series_metric(best_row, "kappa_std"),
            "recall_mean": round_metric_value(best_row["recall_mean"]),
            "recall_std": _series_metric(best_row, "recall_std"),
            "precision_mean": round_metric_value(best_row["precision_mean"]),
            "precision_std": _series_metric(best_row, "precision_std"),
            "best_epoch_mean": _series_metric(best_row, "best_epoch_mean"),
            "best_epoch_std": _series_metric(best_row, "best_epoch_std"),
            "batch_size": int(final_config.get("batch_size")) if final_config.get("batch_size") is not None else None,
            "batch_size_tag": final_config.get("batch_size_tag"),
            "batch_size_source": final_config.get("batch_size_source"),
            SELECTION_POLICY_KEY: effective_selection_policy,
            "artifact_layout_version": final_config.get("artifact_layout_version"),
            "artifact_identity": final_config.get("artifact_identity"),
            "source_summary_csv": str(summary_csv),
            "route_config_path": str(route_cfg_path),
            "selected_config_path": str(final_config_path),
            "selected_run_root": str(final_run_root),
            "selection_decision_path": str(selection_decision_path),
            "selection_sort_columns_used": sort_columns,
            "selection_sort_ascending_used": sort_ascending,
            "selection_filter_type": str(decision_metadata["selection_filter_type"]),
            "selection_used_lambda_zero_fallback": bool(decision_metadata["selection_used_lambda_zero_fallback"]),
            "selection_total_candidate_count": int(decision_metadata["total_candidate_count"]),
            "selection_nonzero_candidate_count": int(decision_metadata["nonzero_candidate_count"]),
            "selection_eligible_candidate_count": int(decision_metadata["eligible_candidate_count"]),
            "selection_filtered_candidate_count": int(decision_metadata["filtered_candidate_count"]),
            "baseline_lambda_rank": float(decision_metadata["baseline_lambda_rank"]),
            "baseline_roc_auc_mean": round_metric_value(decision_metadata["baseline_roc_auc_mean"]),
            "baseline_accuracy_mean": round_metric_value(decision_metadata["baseline_accuracy_mean"]),
            "baseline_f1_mean": round_metric_value(decision_metadata["baseline_f1_mean"]),
            "baseline_kappa_mean": round_metric_value(decision_metadata["baseline_kappa_mean"]),
        }
        selected_row["selection_stage_actual"] = str(decision_metadata["selection_stage_actual"])
        selected_row["selection_fallback_used"] = bool(decision_metadata["selection_fallback_used"])
        if "fallback_eligible_candidate_count" in decision_metadata:
            selected_row["fallback_eligible_candidate_count"] = int(decision_metadata["fallback_eligible_candidate_count"])
            selected_row["fallback_filtered_candidate_count"] = int(decision_metadata["fallback_filtered_candidate_count"])

        selection_decision_path.parent.mkdir(parents=True, exist_ok=True)
        with selection_decision_path.open("w", encoding="utf-8") as fh:
            json.dump(selected_row, fh, ensure_ascii=False, indent=2, default=str)
        selected_rows.append(selected_row)
        selected_configs.append(
            {
                "route": route,
                "selected_config_path": str(final_config_path),
                "selected_run_root": str(final_run_root),
                "selection_decision_path": str(selection_decision_path),
                "source_summary_csv": str(summary_csv),
                "lambda_rank": float(best_row["lambda_rank"]),
                "rank_loss_mode": str(best_row["rank_loss_mode"]),
                "selected_config_unchanged": bool(selected_config_unchanged),
                "config": final_config,
            }
        )
        config_paths.append(str(final_config_path))

    selected_df = pd.DataFrame(selected_rows)
    summary_path = output_root / "selection_summary.csv"
    manifest_path = output_root / "selection_manifest.json"
    selected_df.to_csv(summary_path, index=False, float_format=f"%.{SELECTION_COMPARISON_DECIMALS}f")

    manifest = {
        "selection_name": str(config.get("selection_name", "lambda_rank_selection")),
        "base_dir": str(base_dir),
        "output_root": str(output_root),
        "summary_csv": str(summary_csv),
        "route_config_paths": [str(resolve_path(path, base_dir=base_dir) or path) for path in route_config_paths],
        "selected_config_paths": config_paths,
        "train_after_select": bool(config.get("train_after_select", False)),
        SELECTION_POLICY_KEY: (
            selected_rows[0][SELECTION_POLICY_KEY]
            if selected_rows and len({row[SELECTION_POLICY_KEY] for row in selected_rows}) == 1
            else "mode_dependent"
        ),
        "artifact_layout_version": ARTIFACT_LAYOUT_VERSION,
        "output_suffix": get_output_suffix(config),
        "selection_sort_columns": list(SELECTION_SORT_COLUMNS),
        "selection_sort_ascending": list(SELECTION_SORT_ASCENDING),
        "selection_fallback_sort_columns": list(SELECTION_SORT_COLUMNS),
        "selection_fallback_sort_ascending": list(SELECTION_SORT_ASCENDING),
        "selection_policy_definition": {
            "classification_baseline": {
                "rank_loss_mode": RANK_LOSS_MODE_DISABLED,
                "fixed_lambda_rank": 0.0,
                "uses_adaptive_lambda_selection": False,
            },
            "primary_stage": {
                "name": "S2 balanced baseline gate",
                "nonzero_only": True,
                "sort_columns": list(SELECTION_SORT_COLUMNS),
                "filter_type": SELECTION_FILTER_TYPE_BALANCED_BASELINE_GATE,
                "baseline_requirements": {
                    "roc_auc_mean_gte_lambda_zero": True,
                    "accuracy_mean_gte_lambda_zero": True,
                    "f1_mean_gte_lambda_zero": True,
                    "kappa_mean_gte_lambda_zero": True,
                },
            },
            "fallback_stage": {
                "name": "S1 AUC-first fallback",
                "trigger": "no eligible nonzero lambda after the primary stage",
                "nonzero_only": False,
                "sort_columns": list(SELECTION_SORT_COLUMNS),
                "filter_type": SELECTION_FILTER_TYPE_NONE,
            },
        },
        "batch_size": int(config.get("batch_size")) if config.get("batch_size") is not None else None,
        "batch_size_tag": config.get("batch_size_tag"),
        "batch_size_source": config.get("batch_size_source"),
        "n_selected_routes": int(len(selected_rows)),
        "selected_routes": selected_rows,
    }
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2, default=str)

    print(f"Lambda selection summary saved: {summary_path}")
    print(f"Lambda selection manifest saved: {manifest_path}")
    for item in selected_rows:
        print(
            f"Selected route={item['route']} lambda_rank={item['lambda_rank']} "
            f"policy={item[SELECTION_POLICY_KEY]} config={item['selected_config_path']}"
        )

    return {
        "selection_name": manifest["selection_name"],
        "summary_csv": str(summary_path),
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "selected_routes": selected_rows,
        "selected_configs": selected_configs,
    }


__all__ = [
    "CLASSIFICATION_BASELINE_POLICY",
    "SELECTION_POLICY",
    "SELECTION_SORT_ASCENDING",
    "SELECTION_SORT_COLUMNS",
    "_build_final_training_config",
    "_pick_best_candidate",
    "apply_lambda_rank_override",
    "ensure_selection_policy_not_configurable",
    "get_lambda_rank",
    "normalize_lambda_rank",
    "pick_classification_baseline_candidate_with_metadata",
    "run_lambda_selection",
]
