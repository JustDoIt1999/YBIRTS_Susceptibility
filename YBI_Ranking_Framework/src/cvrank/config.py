from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from .constants import DEFAULT_BATCH_SIZE, DEFAULT_N_SPLITS, DEFAULT_SEED, FEATURE_COLS
from .constants import RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK

def _project_output_root(project_root: str | Path) -> Path:
    return Path(project_root).resolve() / "data" / "output"


def _environmental_outputs_root(workspace_root: str | Path) -> Path:
    return Path(workspace_root).resolve() / "Dataset_Construction" / "outputs" / "03_environmental_data"


def _training_table_path(workspace_root: str | Path, route: str) -> Path:
    return _environmental_outputs_root(workspace_root) / "Total_features_sample.csv"


def _resolve_feature_cols_for_route(route: str) -> list[str]:
    return list(FEATURE_COLS)


@dataclass(frozen=True)
class RouteDefinition:
    route: str
    data_path: str
    output_root: str
    cluster_col: str | None
    year_col: str
    label_col: str
    rank_col: str
    use_rank_aux: bool
    use_cluster_in_rank_loss: bool
    feature_cols: tuple[str, ...]

    @property
    def output_root_name(self) -> str:
        return f"data/output/output_{self.route}"

    @property
    def split_manifest_name(self) -> str:
        return f"data/output/folds/{self.route}_5fold_manifest.csv"

    @property
    def fixed_split_name(self) -> str:
        return f"data/output/folds/{self.route}_train_test_split.npz"

ROUTE_REGISTRY: dict[str, RouteDefinition] = {
    "dbscan": RouteDefinition(
        route="dbscan",
        data_path="../Dataset_Construction/outputs/03_environmental_data/Total_features_sample.csv",
        output_root="./data/output/output_dbscan",
        cluster_col="Cluster",
        year_col="Feature_Year",
        label_col="Label_Cls",
        rank_col="Label_Rank",
        use_rank_aux=True,
        use_cluster_in_rank_loss=True,
        feature_cols=tuple(FEATURE_COLS),
    ),
    "no_cluster": RouteDefinition(
        route="no_cluster",
        data_path="../Dataset_Construction/outputs/03_environmental_data/Total_features_sample.csv",
        output_root="./data/output/output_no_cluster",
        cluster_col="Cluster",
        year_col="Feature_Year",
        label_col="Label_Cls",
        rank_col="Label_Rank",
        use_rank_aux=True,
        use_cluster_in_rank_loss=False,
        feature_cols=tuple(FEATURE_COLS),
    ),
    "basin": RouteDefinition(
        route="basin",
        data_path="../Dataset_Construction/outputs/03_environmental_data/Total_features_sample.csv",
        output_root="./data/output/output_basin",
        cluster_col="Basin_cluster",
        year_col="Feature_Year",
        label_col="Label_Cls",
        rank_col="Label_Rank",
        use_rank_aux=True,
        use_cluster_in_rank_loss=True,
        feature_cols=tuple(FEATURE_COLS),
    ),
}


def normalize_route(route: str | None) -> str:
    return str(route or "dbscan").strip().lower()


def get_route_definition(route: str | None) -> RouteDefinition:
    normalized = normalize_route(route)
    try:
        return ROUTE_REGISTRY[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(ROUTE_REGISTRY))
        raise ValueError(f"Unsupported route '{route}'. Expected one of: {supported}") from exc

REQUIRED_KEYS_BY_TASK: dict[str, list[str]] = {
    "train_validate": ["route", "data_path", "output_root", "label_col", "group_col"],
}

def validate_config_schema(config: dict[str, Any]) -> None:
    task = str(config.get("task", "")).strip().lower()
    required = REQUIRED_KEYS_BY_TASK.get(task, [])
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"Config schema validation failed for task `{task}`: missing keys {missing}")

    batch_size = config.get("batch_size")
    if batch_size is not None and int(batch_size) <= 0:
        raise ValueError("`batch_size` must be > 0")

    if config.get("n_splits") is not None and int(config["n_splits"]) < 2:
        raise ValueError("`n_splits` must be >= 2")


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("YAML config must be a mapping.")
    data = _resolve_layered_yaml(data, config_path.parent)
    data["_config_path"] = str(config_path)
    data["_config_dir"] = str(config_path.parent)
    return data


def _resolve_layered_yaml(data: dict[str, Any], config_dir: Path) -> dict[str, Any]:
    merged = dict(data)
    base_ref = merged.pop("base_config", None)
    overlays_ref = merged.pop("overlays", None)

    base_config: dict[str, Any] = {}
    if isinstance(base_ref, (str, Path)):
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (config_dir / base_path).resolve()
        with base_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Base config must be a mapping: {base_path}")
        base_config = loaded

    overlay_config: dict[str, Any] = {}
    if isinstance(overlays_ref, list):
        for item in overlays_ref:
            overlay_path = Path(item)
            if not overlay_path.is_absolute():
                overlay_path = (config_dir / overlay_path).resolve()
            with overlay_path.open("r", encoding="utf-8") as fh:
                loaded = yaml.safe_load(fh) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"Overlay config must be a mapping: {overlay_path}")
            overlay_config = _deep_merge(overlay_config, loaded)

    return merge_dicts(base_config, overlay_config, merged)


def merge_dicts(*dicts: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in dicts:
        merged = _deep_merge(merged, item)
    return merged


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_path(value: str | Path | None, base_dir: str | Path | None = None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    root = Path(base_dir or ".").resolve()
    return (root / path).resolve()


def build_common_defaults(base_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(base_dir or ".").resolve()
    return {
        "project_name": "cvrank",
        "base_dir": str(root),
        "seed": DEFAULT_SEED,
        "n_splits": DEFAULT_N_SPLITS,
        "group_col": "Sample_ID",
        "stratify_mode": "year_plus_label",
        "split_strategy": "stratified_group_kfold",
    }


def resolve_config_paths(config: dict[str, Any], base_dir: str | Path | None = None) -> dict[str, Any]:
    config_dir = Path(str(config.get("_config_dir") or ".")).resolve()
    root_candidate = Path(base_dir or config.get("base_dir") or ".")
    if root_candidate.is_absolute():
        root = root_candidate
    else:
        root = (config_dir / root_candidate).resolve()

    resolved = deepcopy(config)
    resolved.pop("_config_path", None)
    resolved.pop("_config_dir", None)
    # `artifacts_dir` used to look configurable, but run metadata always lives under
    # `<base_dir>/artifacts/runs`. Drop legacy values so `show-config` reflects reality.
    resolved.pop("artifacts_dir", None)
    resolved["base_dir"] = str(root)

    path_keys = [
        "data_dir",
        "data_path",
        "model_path",
        "scaler_path",
        "metrics_path",
        "validation_metrics_path",
        "history_path",
        "fold_runs_manifest_path",
        "split_manifest_path",
        "fixed_split_path",
        "output_dir",
        "output_root",
        "runs_csv",
        "summary_csv",
        "run_config_path",
    ]
    for key in path_keys:
        value = resolved.get(key)
        if isinstance(value, str) and value:
            resolved[key] = str(resolve_path(value, base_dir=root))

    if resolved.get("route") not in (None, ""):
        route = normalize_route(resolved.get("route"))
        route_def = get_route_definition(route)
        resolved["route"] = route
        resolved["feature_cols"] = _resolve_feature_cols_for_route(route) or list(route_def.feature_cols)

    if isinstance(resolved.get("route_config_paths"), list):
        resolved["route_config_paths"] = [
            str(resolve_path(item, base_dir=root) or item) if isinstance(item, (str, Path)) else item
            for item in resolved["route_config_paths"]
        ]
    validate_config_schema(resolved)
    return resolved


def _build_paths(base_dir: str | Path | None = None, output_root_name: str | Path | None = None) -> dict[str, Path]:
    root = Path(base_dir or ".").resolve()
    if output_root_name is None:
        output_root = root / get_route_definition("dbscan").output_root_name
    else:
        candidate = Path(output_root_name)
        output_root = candidate if candidate.is_absolute() else (root / candidate).resolve()
    return {
        "base_dir": root,
        "output_root": output_root,
        "model_dir": output_root / "model",
    }


def _build_train_validate_config(route: str, base_dir: str | Path | None = None, output_root_name: str | Path | None = None) -> dict[str, Any]:
    route = normalize_route(route)
    route_def = get_route_definition(route)
    root = Path(base_dir or ".").resolve()
    output_root_name = output_root_name or (_project_output_root(root) / f"output_{route}").resolve()
    paths = _build_paths(base_dir=root, output_root_name=output_root_name)
    model_prefix = route
    data_path = _training_table_path(root.parent, route)
    split_manifest_path = _project_output_root(root) / "folds" / f"{route}_5fold_manifest.csv"
    fixed_split_path = _project_output_root(root) / "folds" / f"{route}_train_test_split.npz"

    config: dict[str, Any] = {
        "project_name": "cvrank",
        "route": route,
        "task": "train_validate",
        "base_dir": str(root),
        "data_path": str(data_path),
        "output_root": str(paths["output_root"]),
        "model_path": str(paths["model_dir"] / f"{model_prefix}_model.pth"),
        "scaler_path": str(paths["model_dir"] / f"{model_prefix}_scaler.pkl"),
        "metrics_path": str(paths["model_dir"] / f"{model_prefix}_train_metrics.json"),
        "validation_metrics_path": str(paths["model_dir"] / f"{model_prefix}_validation_metrics.json"),
        "history_path": str(paths["model_dir"] / f"{model_prefix}_train_history.csv"),
        "fold_runs_manifest_path": str(paths["output_root"] / "fold_runs_manifest.json"),
        "split_manifest_path": str(split_manifest_path),
        "fixed_split_path": str(fixed_split_path),
        "seed": DEFAULT_SEED,
        "epochs": 100,
        "patience": 15,
        "batch_size": DEFAULT_BATCH_SIZE,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "lambda_rank": 1.0,
        "rank_loss_mode": RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
        "threshold": 0.5,
        "n_splits": DEFAULT_N_SPLITS,
        "group_col": "Sample_ID",
        "year_col": route_def.year_col,
        "label_col": route_def.label_col,
        "rank_col": route_def.rank_col,
        "cluster_col": route_def.cluster_col,
        "stratify_mode": "year_plus_label",
        "use_rank_aux": route_def.use_rank_aux,
        "use_cluster_in_rank_loss": route_def.use_cluster_in_rank_loss,
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
    }
    return config
