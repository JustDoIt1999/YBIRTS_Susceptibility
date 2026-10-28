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

from .constants import FEATURE_COLS

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
        resolved["feature_cols"] = list(route_def.feature_cols)

    if isinstance(resolved.get("route_config_paths"), list):
        resolved["route_config_paths"] = [
            str(resolve_path(item, base_dir=root) or item) if isinstance(item, (str, Path)) else item
            for item in resolved["route_config_paths"]
        ]
    validate_config_schema(resolved)
    return resolved
