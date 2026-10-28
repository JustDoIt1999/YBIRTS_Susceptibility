from __future__ import annotations

import json
import math
import pickle
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PREDICTION_ROOT / "configs"
CVRANK_ROOT = WORKSPACE_ROOT / "YBI_Ranking_Framework"
CVRANK_SRC = CVRANK_ROOT / "src"
CVRANK_OUTPUT_ROOT = CVRANK_ROOT / "data" / "output"
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(CVRANK_SRC) not in sys.path:
    sys.path.insert(0, str(CVRANK_SRC))
# These imports intentionally follow the workspace-path bootstrap so this module
# can resolve both repository packages when invoked through a direct script.
from Susceptibility_Mapping import gee_context as ee_ctx  # noqa: E402
from Susceptibility_Mapping.uncertainty_layers import (  # noqa: E402
    DEFAULT_UNCERTAINTY_EXPORT_METRICS,
    UNCERTAINTY_METRIC_NAMES,
    classify_by_thresholds,
    compute_fixed_width_thresholds,
    compute_jenks_thresholds,
    encode_confidence_classes,
    sample_finite_values,
    summarize_prediction_members as summarize_member_prediction_matrix,
)
from cvrank.artifact_layout import (  # noqa: E402
    build_legacy_untagged_batch_train_config_path,
    build_selected_train_config_path,
    make_selected_model_identity,
)
from cvrank.rank_loss_mode import (  # noqa: E402
    SUPPORTED_RANK_LOSS_MODES as CVRANK_SUPPORTED_RANK_LOSS_MODES,
    normalize_rank_loss_mode as normalize_cvrank_rank_loss_mode,
)

np: Any = None
pd: Any = None
rasterio: Any = None
torch: Any = None
yaml: Any = None
crs_factory: Any = None
from_origin: Any = None
window_factory: Any = None
tqdm: Any = None
TilePreparationResult = tuple[Any, Any, dict[str, Any]]
TilePreparationFuture = Future[TilePreparationResult]
MODEL_BACKEND_CVRANK = "cvrank"
DEFAULT_MODEL_BACKEND = MODEL_BACKEND_CVRANK
SUPPORTED_MODEL_BACKENDS = {MODEL_BACKEND_CVRANK}

RANK_ROUTES = {"dbscan", "no_cluster", "basin"}
IN_MEMORY_NODATA_VALUE = -9999.0
AUX_MASK_BAND = "aoi_mask"
PREDICTION_RESULT_BAND_NAME = "prediction"
CONFIDENCE_RESULT_BAND_NAME = "confidence_class"
SUPPORTED_RANK_LOSS_MODES = set(CVRANK_SUPPORTED_RANK_LOSS_MODES)
DEFAULT_CACHE_POLICY = "feature_tiles"
SUPPORTED_CACHE_POLICIES = {"none", "feature_tiles"}
RUN_STATE_VERSION = 3
MODEL_BUNDLE_IDENTITY_VERSION = 1
CACHE_FORMAT_VERSION = 2
DEFAULT_UNCERTAINTY_OUTPUT_SUBDIR = "uncertainty"
UNCERTAINTY_BACKFILL_PROGRESS_SUFFIX = "_uncertainty_backfill_progress.json"
UNCERTAINTY_BACKFILL_FAILED_SUFFIX = "_uncertainty_backfill_failed_tiles.json"
UNCERTAINTY_BACKFILL_RUNTIME_SUFFIX = "_uncertainty_backfill_runtime_summary.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def import_yaml_dependency() -> None:
    global yaml
    if yaml is not None:
        return
    try:
        import yaml as _yaml
    except ImportError as exc:
        raise ImportError("Missing pyyaml dependency, unable to read YAML configuration.") from exc
    yaml = _yaml


def import_runtime_dependencies() -> None:
    global np, pd, rasterio, torch, crs_factory, from_origin, window_factory, tqdm

    if np is not None:
        return
    try:
        import_yaml_dependency()
        import numpy as _np
        import pandas as _pd
        import rasterio as _rasterio
        from rasterio.crs import CRS as _CRS
        from rasterio.transform import from_origin as _from_origin
        from rasterio.windows import Window as _Window
        from tqdm.auto import tqdm as _tqdm
    except ImportError as exc:
        raise ImportError(
            "This script requires the current environment to have at least: numpy, pandas, rasterio, pyyaml, and tqdm."
        ) from exc

    np = _np
    pd = _pd
    rasterio = _rasterio
    crs_factory = _CRS
    from_origin = _from_origin
    window_factory = _Window
    tqdm = _tqdm
    try:
        import torch as _torch
    except ImportError:
        torch = None
    else:
        torch = _torch


def normalize_model_backend(value: Any | None) -> str:
    normalized = str(value or DEFAULT_MODEL_BACKEND).strip().lower()
    if normalized not in SUPPORTED_MODEL_BACKENDS:
        raise ValueError(
            f"Unsupported model_backend: {value!r}; supported={sorted(SUPPORTED_MODEL_BACKENDS)}"
        )
    return normalized


def load_training_config(
    config_path: str | Path,
    model_backend: Any | None,
) -> dict[str, Any]:
    from cvrank.config import load_yaml_config, resolve_config_paths

    model_backend = normalize_model_backend(model_backend)
    resolved_config = resolve_config_paths(load_yaml_config(config_path))
    resolved_config["_config_path"] = str(resolve_path(config_path))
    resolved_config["_resolved_base_dir"] = str(Path(resolved_config.get("base_dir") or Path(config_path).parent).resolve())
    resolved_config["model_backend"] = model_backend
    return resolved_config


def resolve_path(value: str | Path) -> Path:
    """Return an absolute, user-expanded path for public callers."""
    return Path(value).expanduser().resolve()


def _resolve_path_from_base(value: str | Path, base_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(base_dir).expanduser().resolve() / path).resolve()


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def compute_payload_hash(payload: Any) -> str:
    return sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_file_hash(path: str | Path) -> str:
    resolved_path = resolve_path(path)
    hasher = sha256()
    with resolved_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def describe_model_artifact(path: str | Path) -> dict[str, Any]:
    resolved_path = resolve_path(path)
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Required model artifact not found: {resolved_path}")
    stat = resolved_path.stat()
    return {
        "path": str(resolved_path),
        "size": int(stat.st_size),
        "sha256": compute_file_hash(resolved_path),
    }


_progress_display_active = False


def emit_runtime_event(event: str, **payload: Any) -> None:
    if _progress_display_active and str(event).startswith("tile_"):
        return
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


class TerminalProgressDisplay:
    def __init__(
        self,
        *,
        label: str,
        manifest: dict[str, Any],
        enabled: bool,
    ) -> None:
        self.enabled = bool(enabled) and sys.stderr.isatty()
        self.label = str(label)
        self._previous_active = False
        self._started_at = time.perf_counter()
        counts = output_progress_counts(manifest)
        self._initial_done = int(counts.get("done", 0))
        self.total_bar = None
        self.fetch_bar = None
        if not self.enabled:
            return
        import_runtime_dependencies()
        total = len(output_leaf_tile_ids(manifest))
        fetched = self._fetched_count(counts)
        self.total_bar = tqdm(
            total=total,
            initial=self._initial_done,
            desc=f"{self.label} infer",
            unit="tile",
            position=0,
            dynamic_ncols=True,
            leave=True,
        )
        self.fetch_bar = tqdm(
            total=total,
            initial=fetched,
            desc=f"{self.label} fetch",
            unit="tile",
            position=1,
            dynamic_ncols=True,
            leave=True,
        )
        self.update_from_manifest(manifest, force=True)

    @staticmethod
    def _fetched_count(counts: dict[str, int]) -> int:
        return int(counts.get("downloaded", 0)) + int(counts.get("inferring", 0)) + int(counts.get("done", 0))

    def __enter__(self) -> "TerminalProgressDisplay":
        global _progress_display_active
        self._previous_active = _progress_display_active
        if self.enabled:
            _progress_display_active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        global _progress_display_active
        if self.enabled:
            self.close()
        _progress_display_active = self._previous_active

    def update_from_manifest(self, manifest: dict[str, Any], *, force: bool = False) -> None:
        if not self.enabled or self.total_bar is None or self.fetch_bar is None:
            return
        counts = output_progress_counts(manifest)
        total = len(output_leaf_tile_ids(manifest))
        fetched = min(total, self._fetched_count(counts))
        done = min(total, int(counts.get("done", 0)))
        if force or self.total_bar.total != total:
            self.total_bar.total = total
            self.fetch_bar.total = total
        self.total_bar.n = done
        self.fetch_bar.n = fetched

        elapsed_seconds = max(time.perf_counter() - self._started_at, 1e-6)
        session_done = max(0, done - self._initial_done)
        rate_tph = session_done * 3600.0 / elapsed_seconds
        remaining_tiles = max(0, total - done)
        eta_hours = (remaining_tiles / rate_tph) if rate_tph > 0 else None

        total_postfix = {
            "downloading": int(counts.get("downloading", 0)),
            "inferring": int(counts.get("inferring", 0)),
            "failed": int(counts.get("failed", 0)),
            "tile/h": round(rate_tph, 1),
        }
        if eta_hours is not None:
            total_postfix["eta_h"] = round(eta_hours, 1)
        self.total_bar.set_postfix(total_postfix, refresh=False)
        self.fetch_bar.set_postfix(
            {
                "ready": fetched,
                "pending": int(counts.get("pending", 0)),
            },
            refresh=False,
        )
        self.total_bar.refresh()
        self.fetch_bar.refresh()

    def close(self) -> None:
        if not self.enabled:
            return
        assert self.total_bar is not None
        assert self.fetch_bar is not None
        self.total_bar.close()
        self.fetch_bar.close()


def save_json(path: str | Path, payload: Any) -> Path:
    resolved_path = resolve_path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
    return resolved_path


def load_json(path: str | Path) -> Any:
    resolved_path = resolve_path(path)
    with resolved_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_yaml_document(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    import_yaml_dependency()
    resolved_path = resolve_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as fh:
        payload = yaml.safe_load(fh) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"The YAML root element must be a mapping: {resolved_path}")
    return payload, resolved_path


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    cfg, resolved_path = load_yaml_document(config_path)
    base_dir_raw = cfg.get("base_dir")
    if base_dir_raw in (None, ""):
        base_dir = resolved_path.parent
    else:
        base_dir = Path(base_dir_raw).expanduser()
        if not base_dir.is_absolute():
            base_dir = (resolved_path.parent / base_dir).resolve()
    cfg["_config_path"] = str(resolved_path)
    cfg["_resolved_base_dir"] = str(base_dir.resolve())
    return cfg


def resolve_cfg_path(cfg: dict[str, Any], key: str) -> Path | None:
    raw = cfg.get(key)
    if raw in (None, ""):
        return None
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(cfg["_resolved_base_dir"]) / path).resolve()


def load_projection_job_config(
    config_path: str | Path,
    *,
    allowed_keys: set[str],
    path_keys: set[str],
) -> dict[str, Any]:
    payload, resolved_path = load_yaml_document(config_path)
    unknown_keys = sorted(set(payload) - allowed_keys)
    if unknown_keys:
        raise KeyError(f"Unrecognized fields found in YAML config: {unknown_keys}; config={resolved_path}")
    normalized = dict(payload)
    for key in path_keys:
        if normalized.get(key) not in (None, ""):
            normalized[key] = _resolve_path_from_base(normalized[key], resolved_path.parent)
    normalized["_job_config_path"] = str(resolved_path)
    return normalized


def resolve_raster_output_dir(job: dict[str, Any], default_path: str | Path | None = None) -> Path:
    candidate = job.get("raster_output_dir")
    if candidate in (None, ""):
        candidate = job.get("output_dir")
    if candidate in (None, ""):
        candidate = default_path or (PREDICTION_ROOT / "outputs")
    return resolve_path(candidate)


def resolve_json_output_dir(job: dict[str, Any], default_path: str | Path | None = None) -> Path:
    candidate = job.get("json_output_dir")
    if candidate in (None, ""):
        candidate = default_path or (PREDICTION_ROOT / "outputs")
    return resolve_path(candidate)


def should_write_uncertainty_outputs(job: dict[str, Any], model_members: list[dict[str, Any]]) -> bool:
    explicit = job.get("write_uncertainty_outputs")
    if explicit is not None:
        return bool(explicit) and len(model_members) > 1
    return False


def should_write_confidence_outputs(job: dict[str, Any], model_members: list[dict[str, Any]]) -> bool:
    explicit = job.get("write_confidence_outputs")
    if explicit is not None:
        return bool(explicit) and len(model_members) > 1
    return False


def should_backfill_uncertainty_only(job: dict[str, Any], model_members: list[dict[str, Any]]) -> bool:
    explicit = job.get("backfill_uncertainty_only")
    if explicit is None:
        return False
    if not bool(explicit):
        return False
    if not should_write_uncertainty_outputs(job, model_members):
        raise ValueError("backfill_uncertainty_only=true requires write_uncertainty_outputs=true.")
    if bool(job.get("download_only", False)):
        raise ValueError("backfill_uncertainty_only cannot be used together with download_only=true.")
    return True


def resolve_uncertainty_export_metrics(job: dict[str, Any]) -> tuple[str, ...]:
    raw = job.get("uncertainty_metrics")
    if raw in (None, "", []):
        requested = tuple(DEFAULT_UNCERTAINTY_EXPORT_METRICS)
    elif isinstance(raw, str):
        requested = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    else:
        requested = tuple(str(part).strip().lower() for part in raw if str(part).strip())
    unknown = sorted(set(requested) - (set(UNCERTAINTY_METRIC_NAMES) - {"mean"}))
    if unknown:
        raise ValueError(
            f"Unsupported uncertainty_metrics: {unknown}; supported={sorted(set(UNCERTAINTY_METRIC_NAMES) - {'mean'})}"
        )
    if not requested:
        raise ValueError("uncertainty_metrics cannot be empty when uncertainty outputs are enabled.")
    if bool(job.get("write_confidence_outputs", False)) and "std" not in requested:
        raise ValueError("write_confidence_outputs=true requires uncertainty_metrics to include 'std'.")
    return requested


def resolve_uncertainty_output_subdir(job: dict[str, Any]) -> str:
    value = str(job.get("uncertainty_output_subdir") or DEFAULT_UNCERTAINTY_OUTPUT_SUBDIR).strip().strip("/").strip("\\")
    return value or DEFAULT_UNCERTAINTY_OUTPUT_SUBDIR


def build_uncertainty_output_paths(
    *,
    output_path: Path,
    json_output_dir: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    subdir = resolve_uncertainty_output_subdir(job)
    raster_dir = output_path.parent / subdir
    json_dir = json_output_dir / subdir
    stem = output_path.stem
    metric_names = resolve_uncertainty_export_metrics(job)
    metric_paths = {
        metric_name: raster_dir / f"{stem}_{metric_name}.tif"
        for metric_name in metric_names
    }
    confidence_class_count = max(2, int(job.get("confidence_class_count", 5) or 5))
    return {
        "raster_dir": raster_dir,
        "json_dir": json_dir,
        "metric_paths": metric_paths,
        "confidence_path": raster_dir / f"{stem}_confidence_{confidence_class_count}x{confidence_class_count}.tif",
        "uncertainty_summary_path": json_dir / f"{stem}_uncertainty_summary.json",
        "confidence_summary_path": json_dir / f"{stem}_confidence_{confidence_class_count}x{confidence_class_count}_summary.json",
    }


def normalize_rank_loss_mode(rank_loss_mode: str | None) -> str:
    if not str(rank_loss_mode or "").strip():
        raise ValueError("rank_loss_mode cannot be empty; please specify it explicitly in the job YAML or CLI.")
    return normalize_cvrank_rank_loss_mode(rank_loss_mode)


def normalize_training_batch_size(batch_size: int | str | None) -> int:
    if batch_size in (None, ""):
        raise ValueError("batch_size cannot be empty; please specify it explicitly in the job YAML or CLI.")
    value = int(batch_size)
    if value <= 0:
        raise ValueError(f"batch_size must be a positive integer; got={batch_size}")
    return value


def build_selected_train_config_candidate_path(
    *,
    route: str,
    batch_size: int,
    rank_loss_mode: str,
    train_output_suffix: str | None = None,
) -> Path:
    identity = make_selected_model_identity(
        route=route,
        batch_size=batch_size,
        rank_loss_mode=rank_loss_mode,
        train_output_suffix=train_output_suffix,
    )
    return build_selected_train_config_path(CVRANK_OUTPUT_ROOT, identity).resolve()


def validate_selected_train_config_identity(
    config_path: Path,
    *,
    route: str,
    batch_size: int,
    rank_loss_mode: str,
    train_output_suffix: str | None,
) -> dict[str, Any]:
    import_yaml_dependency()
    with config_path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Selected train config must be a YAML mapping: {config_path}")

    requested_identity = make_selected_model_identity(
        route=route,
        batch_size=batch_size,
        rank_loss_mode=rank_loss_mode,
        train_output_suffix=train_output_suffix,
    )
    actual_identity = make_selected_model_identity(
        route=config.get("route"),
        batch_size=config.get("batch_size"),
        rank_loss_mode=config.get("rank_loss_mode"),
        train_output_suffix=config.get("output_suffix"),
    )
    if actual_identity != requested_identity:
        raise ValueError(
            "Selected train config identity mismatch.\n"
            f"requested={requested_identity.as_dict()}\n"
            f"resolved={actual_identity.as_dict()}\n"
            f"path={config_path}"
        )
    return {
        "artifact_identity": requested_identity.as_dict(),
        "artifact_layout_version": config.get("artifact_layout_version"),
        "selection_policy": config.get("selection_policy"),
    }


def resolve_selected_train_config(
    *,
    route: str | None,
    batch_size: int | str | None,
    rank_loss_mode: str | None,
    train_output_suffix: str | None = None,
) -> dict[str, Any]:
    normalized_route = str(route or "").strip().lower()
    if normalized_route not in RANK_ROUTES:
        raise ValueError(f"Unsupported route for train config resolution: {route}; supported={sorted(RANK_ROUTES)}")
    normalized_batch_size = normalize_training_batch_size(batch_size)
    normalized_rank_loss_mode = normalize_rank_loss_mode(rank_loss_mode)
    identity = make_selected_model_identity(
        route=normalized_route,
        batch_size=normalized_batch_size,
        rank_loss_mode=normalized_rank_loss_mode,
        train_output_suffix=train_output_suffix,
    )
    canonical_path = build_selected_train_config_path(CVRANK_OUTPUT_ROOT, identity).resolve()
    legacy_path = build_legacy_untagged_batch_train_config_path(CVRANK_OUTPUT_ROOT, identity).resolve()
    candidates = (
        ("canonical", canonical_path),
        ("legacy_untagged_batch", legacy_path),
    )
    for artifact_layout, config_path in candidates:
        if not config_path.exists():
            continue
        identity_metadata = validate_selected_train_config_identity(
            config_path,
            route=normalized_route,
            batch_size=normalized_batch_size,
            rank_loss_mode=normalized_rank_loss_mode,
            train_output_suffix=train_output_suffix,
        )
        return {
            "train_config_path": config_path,
            "artifact_layout": artifact_layout,
            **identity_metadata,
        }

    raise FileNotFoundError(
        "No matching training configuration found.\n"
        f"route={normalized_route}\n"
        f"batch_size={normalized_batch_size}\n"
        f"rank_loss_mode={normalized_rank_loss_mode}\n"
        f"train_output_suffix={identity.train_output_suffix}\n"
        f"expected_candidates={[str(path) for _layout, path in candidates]}"
    )


def resolve_selected_train_config_path(
    *,
    route: str | None,
    batch_size: int | str | None,
    rank_loss_mode: str | None,
    train_output_suffix: str | None = None,
) -> Path:
    resolved = resolve_selected_train_config(
        route=route,
        batch_size=batch_size,
        rank_loss_mode=rank_loss_mode,
        train_output_suffix=train_output_suffix,
    )
    return resolved["train_config_path"]


def finalize_train_config(job: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(job)
    resolved["model_backend"] = normalize_model_backend(resolved.get("model_backend"))
    if resolved.get("train_config_path") in (None, ""):
        resolved["rank_loss_mode"] = normalize_rank_loss_mode(resolved.get("rank_loss_mode"))
        resolved["batch_size"] = normalize_training_batch_size(resolved.get("batch_size"))
        resolved_train_config = resolve_selected_train_config(
            route=resolved.get("route"),
            batch_size=resolved.get("batch_size"),
            rank_loss_mode=resolved.get("rank_loss_mode"),
            train_output_suffix=resolved.get("train_output_suffix"),
        )
        resolved["train_config_path"] = resolved_train_config["train_config_path"]
        resolved["train_config_source"] = "auto_resolved"
        resolved["artifact_layout"] = resolved_train_config["artifact_layout"]
        resolved["artifact_layout_version"] = resolved_train_config.get("artifact_layout_version")
        resolved["artifact_identity"] = resolved_train_config["artifact_identity"]
        resolved["selection_policy"] = resolved_train_config.get("selection_policy")
    else:
        resolved["train_config_path"] = resolve_path(resolved["train_config_path"])
        resolved["train_config_source"] = "explicit"
    return resolved


def normalize_cache_policy(value: str | None) -> str:
    policy = str(value or DEFAULT_CACHE_POLICY).strip().lower()
    if policy not in SUPPORTED_CACHE_POLICIES:
        raise ValueError(f"Unsupported cache_policy: {value}; supported={sorted(SUPPORTED_CACHE_POLICIES)}")
    return policy


def normalize_positive_int(value: Any, *, field_name: str) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer; got={value!r}") from exc
    if normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer; got={value!r}")
    return normalized


def normalize_minimum_int(value: Any, *, field_name: str, minimum: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer >= {minimum}; got={value!r}") from exc
    if normalized < int(minimum):
        raise ValueError(f"{field_name} must be an integer >= {minimum}; got={value!r}")
    return normalized


def get_route_feature_cols(route: str) -> list[str]:
    route = str(route or "").strip().lower()
    if route in RANK_ROUTES:
        return list(ee_ctx.CURRENT_RANK_FEATURE_COLS)
    raise ValueError(f"Unsupported route for rank raster prediction: {route}")


def resolve_training_feature_cols(cfg: dict[str, Any], route: str) -> list[str]:
    configured = cfg.get("feature_cols")
    if isinstance(configured, list) and configured:
        return [str(item) for item in configured]
    return get_route_feature_cols(route)


def assert_training_schema_matches_features(cfg: dict[str, Any], feature_cols: list[str]) -> None:
    import_runtime_dependencies()
    data_path = resolve_cfg_path(cfg, "data_path")
    if data_path is None or not data_path.exists():
        print(f"Skip schema header check because data_path is unavailable: {data_path}")
        return
    header = pd.read_csv(data_path, nrows=0)
    missing = [col for col in feature_cols if col not in header.columns]
    if missing:
        raise ValueError(
            "The selected training configuration is inconsistent with the official 14-factor rank schema and cannot be used directly with this script.\n"
            f"config={cfg['_config_path']}\n"
            f"data_path={data_path}\n"
            f"missing_feature_columns={missing}"
        )


def _load_optional_feature_preparation_artifact(
    path_value: str | Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    if not isinstance(path_value, (str, Path)) or not str(path_value).strip():
        return None, None
    artifact_path = resolve_path(path_value)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Feature preparation artifact not found: {artifact_path}")
    from cvrank.data_utils import load_feature_preparation_artifact

    return load_feature_preparation_artifact(artifact_path), artifact_path


def resolve_model_member_artifacts(
    cfg: dict[str, Any],
) -> tuple[list[dict[str, Any]], Path | None]:
    from cvrank.shared import default_feature_preparation_path

    manifest_path = resolve_cfg_path(cfg, "fold_runs_manifest_path")
    members_meta: list[dict[str, Any]] = []
    if manifest_path is not None and manifest_path.exists():
        manifest = load_json(manifest_path)
        for item in manifest.get("folds", []):
            members_meta.append(
                {
                    "fold_index": item.get("fold_index"),
                    "model_path": resolve_path(item["model_path"]),
                    "scaler_path": resolve_path(item["scaler_path"]),
                    "feature_preparation_path": (
                        None
                        if item.get("feature_preparation_path") in (None, "")
                        else resolve_path(item["feature_preparation_path"])
                    ),
                }
            )
        if not members_meta:
            raise ValueError(f"Fold runs manifest contains no model members: {manifest_path}")
        return members_meta, manifest_path

    model_path = resolve_cfg_path(cfg, "model_path")
    scaler_path = resolve_cfg_path(cfg, "scaler_path")
    if model_path is None or scaler_path is None:
        raise ValueError("The selected config must provide model_path and scaler_path.")
    configured_feature_preparation_path = resolve_cfg_path(cfg, "feature_preparation_path")
    default_artifact_path = resolve_path(default_feature_preparation_path(cfg))
    if configured_feature_preparation_path is not None:
        feature_preparation_path = configured_feature_preparation_path
    elif default_artifact_path.exists():
        feature_preparation_path = default_artifact_path
    else:
        feature_preparation_path = None
    members_meta.append(
        {
            "fold_index": None,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "feature_preparation_path": feature_preparation_path,
        }
    )
    return members_meta, None


def build_model_bundle_identity(
    *,
    train_config_path: str | Path,
    model_backend: Any,
    members_meta: list[dict[str, Any]],
    fold_manifest_path: str | Path | None,
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for position, meta in enumerate(members_meta):
        feature_preparation_path = meta.get("feature_preparation_path")
        members.append(
            {
                "position": int(position),
                "fold_index": meta.get("fold_index"),
                "model": describe_model_artifact(meta["model_path"]),
                "scaler": describe_model_artifact(meta["scaler_path"]),
                "feature_preparation": (
                    None
                    if feature_preparation_path in (None, "")
                    else describe_model_artifact(feature_preparation_path)
                ),
            }
        )
    payload = {
        "schema_version": MODEL_BUNDLE_IDENTITY_VERSION,
        "model_backend": normalize_model_backend(model_backend),
        "train_config": describe_model_artifact(train_config_path),
        "fold_manifest": (
            None if fold_manifest_path is None else describe_model_artifact(fold_manifest_path)
        ),
        "members": members,
    }
    return {**payload, "bundle_hash": compute_payload_hash(payload)}


def verify_model_bundle_unchanged(
    identity: dict[str, Any],
    *,
    train_config_path: str | Path,
    model_backend: Any,
    members_meta: list[dict[str, Any]],
    fold_manifest_path: str | Path | None,
) -> None:
    current = build_model_bundle_identity(
        train_config_path=train_config_path,
        model_backend=model_backend,
        members_meta=members_meta,
        fold_manifest_path=fold_manifest_path,
    )
    if current["bundle_hash"] != identity["bundle_hash"]:
        raise RuntimeError(
            "Model artifacts changed while projection was loading them. Stop concurrent training or artifact replacement, then retry."
        )


def load_model_scaler_members(
    cfg: dict[str, Any],
    feature_cols: list[str],
    device: Any,
    members_meta: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    import_runtime_dependencies()
    import torch
    from cvrank.shared import build_model

    if members_meta is None:
        members_meta, _manifest_path = resolve_model_member_artifacts(
            cfg,
        )

    members: list[dict[str, Any]] = []
    for meta in members_meta:
        if not meta["model_path"].exists():
            raise FileNotFoundError(f"Model checkpoint not found: {meta['model_path']}")
        if not meta["scaler_path"].exists():
            raise FileNotFoundError(f"Scaler file not found: {meta['scaler_path']}")
        artifact, artifact_path = _load_optional_feature_preparation_artifact(
            meta.get("feature_preparation_path"),
        )
        member_feature_cols = list(artifact.get("final_feature_cols", feature_cols)) if artifact is not None else list(feature_cols)
        with meta["scaler_path"].open("rb") as fh:
            scaler = pickle.load(fh)
        model = build_model(input_dim=len(member_feature_cols), config=cfg).to(device)
        model.load_state_dict(torch.load(meta["model_path"], map_location=device))
        model.eval()
        members.append(
            {
                "fold_index": meta.get("fold_index"),
                "model": model,
                "scaler": scaler,
                "model_path": meta["model_path"],
                "scaler_path": meta["scaler_path"],
                "feature_cols": member_feature_cols,
                "feature_preparation_artifact": artifact,
                "feature_preparation_path": artifact_path,
            }
        )
    return members


def get_projected_grid(region_geom) -> tuple[float, float, float, float, int, int]:
    import_runtime_dependencies()
    target_proj = ee_ctx.ee.Projection(ee_ctx.TARGET_PROJECTION_WKT)
    coords = region_geom.bounds(ee_ctx.MAX_ERROR, target_proj).coordinates().getInfo()[0]
    xs = [float(pt[0]) for pt in coords]
    ys = [float(pt[1]) for pt in coords]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = int(math.ceil((xmax - xmin) / ee_ctx.TARGET_SCALE))
    height = int(math.ceil((ymax - ymin) / ee_ctx.TARGET_SCALE))
    return xmin, ymin, xmax, ymax, width, height


def iter_projected_tiles(xmin: float, ymax: float, width: int, height: int, tile_size: int):
    tile_size = int(tile_size)
    for row_off in range(0, height, tile_size):
        for col_off in range(0, width, tile_size):
            tile_width = min(tile_size, width - col_off)
            tile_height = min(tile_size, height - row_off)
            tile_xmin = xmin + col_off * ee_ctx.TARGET_SCALE
            tile_ymax = ymax - row_off * ee_ctx.TARGET_SCALE
            tile_ymin = tile_ymax - tile_height * ee_ctx.TARGET_SCALE
            yield {
                "row_idx": row_off // tile_size,
                "col_idx": col_off // tile_size,
                "row_off": row_off,
                "col_off": col_off,
                "width": tile_width,
                "height": tile_height,
                "xmin": tile_xmin,
                "ymax": tile_ymax,
                "ymin": tile_ymin,
                "tile_size": tile_size,
            }


def tile_id(tile: dict[str, Any]) -> str:
    return (
        f"r{int(tile['row_off']):05d}_c{int(tile['col_off']):05d}"
        f"_w{int(tile['width']):04d}_h{int(tile['height']):04d}_s{int(tile['tile_size']):04d}"
    )


def build_active_tile_lookup(
    region_geom,
    xmin: float,
    ymax: float,
    width: int,
    height: int,
    tile_size: int,
    retry: int = 4,
) -> dict[str, Any]:
    import_runtime_dependencies()
    tile_size = int(tile_size)
    tile_cols = int(math.ceil(width / tile_size))
    tile_rows = int(math.ceil(height / tile_size))
    coarse_scale = float(tile_size * ee_ctx.TARGET_SCALE)
    mask_band = "tile_mask"
    mask_image = (
        ee_ctx.ee.Image.constant(1)
        .rename(mask_band)
        .setDefaultProjection(crs=ee_ctx.TARGET_PROJECTION_WKT, scale=ee_ctx.TARGET_SCALE)
        .clip(region_geom)
        .unmask(0)  # pyright: ignore[reportArgumentType] -- Earth Engine accepts numeric fill values.
        .reduceResolution(
            reducer=ee_ctx.ee.Reducer.max(),
            bestEffort=True,
            maxPixels=min(65535, max(4096, tile_size * tile_size)),
        )
        .reproject(crs=ee_ctx.TARGET_PROJECTION_WKT, scale=coarse_scale)
    )
    request = {
        "expression": mask_image,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": [mask_band],
        "grid": {
            "dimensions": {
                "width": tile_cols,
                "height": tile_rows,
            },
            "affineTransform": {
                "scaleX": coarse_scale,
                "shearX": 0.0,
                "translateX": float(xmin),
                "shearY": 0.0,
                "scaleY": -coarse_scale,
                "translateY": float(ymax),
            },
            "crsWkt": ee_ctx.TARGET_PROJECTION_WKT,
        },
    }
    last_error = None
    for attempt in range(1, int(retry) + 1):
        try:
            struct_array = ee_ctx.ee.data.computePixels(  # pyright: ignore[reportPrivateImportUsage]
                request
            )
            mask_array = np.asarray(struct_array[mask_band], dtype=np.float32)
            expected_shape = (tile_rows, tile_cols)
            if mask_array.shape != expected_shape:
                raise ValueError(f"Tile occupancy mask shape mismatch: {mask_array.shape} != {expected_shape}")
            active_cells = np.argwhere(mask_array > 0.0)
            active_lookup = {(int(row_idx), int(col_idx)) for row_idx, col_idx in active_cells.tolist()}
            return {
                "tile_rows": tile_rows,
                "tile_cols": tile_cols,
                "active_lookup": active_lookup,
                "active_tile_count": len(active_lookup),
                "mask_shape": expected_shape,
            }
        except Exception as exc:
            last_error = exc
            if attempt >= int(retry):
                break
            time.sleep(float(2 ** (attempt - 1)))
    raise RuntimeError(f"Failed to build active tile lookup after {retry} attempts: {last_error}")


def load_or_create_active_tile_lookup_cache(
    *,
    cache_root: Path,
    provider_name: str,
    region_geom,
    xmin: float,
    ymax: float,
    width: int,
    height: int,
    tile_size: int,
    retry: int,
) -> tuple[dict[str, Any], Path]:
    runtime_ctx = ee_ctx.get_runtime_context()
    metadata_key = compute_payload_hash(
        {
            "provider_name": provider_name,
            "target_projection_wkt": ee_ctx.TARGET_PROJECTION_WKT,
            "target_scale": ee_ctx.TARGET_SCALE,
            "grid": {"width": int(width), "height": int(height), "xmin": float(xmin), "ymax": float(ymax)},
            "tile_size": int(tile_size),
            "aoi_asset_id": runtime_ctx.get("AOI_ASSET_ID"),
        }
    )
    cache_path = cache_root / "metadata" / provider_name / metadata_key / f"active_tiles_t{int(tile_size)}.json"
    if cache_path.exists():
        payload = load_json(cache_path)
        payload["active_lookup"] = {(int(row_idx), int(col_idx)) for row_idx, col_idx in payload.get("active_cells", [])}
        return payload, cache_path
    payload = build_active_tile_lookup(
        region_geom,
        xmin=xmin,
        ymax=ymax,
        width=width,
        height=height,
        tile_size=tile_size,
        retry=retry,
    )
    serializable = dict(payload)
    serializable["active_cells"] = [list(item) for item in sorted(payload["active_lookup"])]
    serializable.pop("active_lookup", None)
    save_json(cache_path, serializable)
    return payload, cache_path


def subdivide_tile(tile: dict[str, Any], fallback_tile_size: int) -> list[dict[str, Any]]:
    fallback_tile_size = int(fallback_tile_size)
    if fallback_tile_size <= 0:
        raise ValueError(f"fallback_tile_size must be positive; got={fallback_tile_size}")
    if fallback_tile_size >= int(tile["tile_size"]):
        return []

    children: list[dict[str, Any]] = []
    row_limit = int(tile["row_off"]) + int(tile["height"])
    col_limit = int(tile["col_off"]) + int(tile["width"])
    for row_off in range(int(tile["row_off"]), row_limit, fallback_tile_size):
        for col_off in range(int(tile["col_off"]), col_limit, fallback_tile_size):
            child_width = min(fallback_tile_size, col_limit - col_off)
            child_height = min(fallback_tile_size, row_limit - row_off)
            child_xmin = float(tile["xmin"]) + (col_off - int(tile["col_off"])) * ee_ctx.TARGET_SCALE
            child_ymax = float(tile["ymax"]) - (row_off - int(tile["row_off"])) * ee_ctx.TARGET_SCALE
            child_ymin = child_ymax - child_height * ee_ctx.TARGET_SCALE
            children.append(
                {
                    "row_idx": row_off // fallback_tile_size,
                    "col_idx": col_off // fallback_tile_size,
                    "row_off": row_off,
                    "col_off": col_off,
                    "width": child_width,
                    "height": child_height,
                    "xmin": child_xmin,
                    "ymax": child_ymax,
                    "ymin": child_ymin,
                    "tile_size": fallback_tile_size,
                }
            )
    return children


def download_tile_array(image, tile: dict[str, Any], band_ids: list[str], retry: int = 4) -> Any:
    import_runtime_dependencies()
    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": band_ids,
        "grid": {
            "dimensions": {
                "width": int(tile["width"]),
                "height": int(tile["height"]),
            },
            "affineTransform": {
                "scaleX": float(ee_ctx.TARGET_SCALE),
                "shearX": 0.0,
                "translateX": float(tile["xmin"]),
                "shearY": 0.0,
                "scaleY": -float(ee_ctx.TARGET_SCALE),
                "translateY": float(tile["ymax"]),
            },
            "crsWkt": ee_ctx.TARGET_PROJECTION_WKT,
        },
    }
    last_error = None
    for attempt in range(1, int(retry) + 1):
        try:
            struct_array = ee_ctx.ee.data.computePixels(  # pyright: ignore[reportPrivateImportUsage]
                request
            )
            array = np.dstack([struct_array[band] for band in band_ids]).astype(np.float32, copy=False)
            expected_shape = (int(tile["height"]), int(tile["width"]), len(band_ids))
            if array.shape != expected_shape:
                raise ValueError(f"Tile array shape mismatch: {array.shape} != {expected_shape}")
            return array
        except Exception as exc:
            last_error = exc
            if attempt >= int(retry):
                break
            time.sleep(float(2 ** (attempt - 1)))
    raise RuntimeError(f"Failed to download tile after {retry} attempts: {last_error}")


def predict_probability_block(
    feature_array: Any,
    aoi_mask: Any,
    feature_cols: list[str],
    model_members: list[dict[str, Any]],
    device: Any,
    inference_batch_size: int,
    nodata_value: float,
    *,
    uncertainty_metrics: tuple[str, ...] | list[str] | None = None,
    uncertainty_epsilon: float = 1e-6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import_runtime_dependencies()
    from cvrank.data_utils import apply_feature_preparation_artifact

    base_invalid_mask = aoi_mask < 0.5
    metric_names = ("mean", *(tuple(uncertainty_metrics or DEFAULT_UNCERTAINTY_EXPORT_METRICS)))
    flat_frame = pd.DataFrame(feature_array.reshape(-1, len(feature_cols)), columns=feature_cols)
    base_invalid_flat_mask = base_invalid_mask.reshape(-1)

    if (~base_invalid_flat_mask).sum() == 0:
        return {
            name: np.full(base_invalid_mask.shape, nodata_value, dtype=np.float32)
            for name in metric_names
        }, {"valid_pixels": 0}

    member_predictions: list[NDArray[Any]] = []
    valid_flat_mask: NDArray[Any] | None = None
    batch_size = max(1, int(inference_batch_size))

    for member in model_members:
        member_invalid_mask = pd.Series(base_invalid_flat_mask, index=flat_frame.index, dtype=bool)
        artifact = member["feature_preparation_artifact"]
        if artifact is not None:
            artifact_raw_feature_cols = list(artifact.get("raw_feature_cols", feature_cols))
            if artifact_raw_feature_cols != feature_cols:
                raise ValueError(
                    "Raster feature order does not match feature preparation artifact raw schema: "
                    f"{artifact_raw_feature_cols} != {feature_cols}"
                )
            transformed, final_feature_cols, invalid_numeric_mask, missing_raw_na_mask, missing_input_nodata_mask = (
                apply_feature_preparation_artifact(
                    flat_frame,
                    feature_cols,
                    tuple(float(value) for value in artifact.get("input_nodata_values", [])),
                    artifact,
                    fill_numeric_imputation=True,
                )
            )
            expected_feature_cols = list(member["feature_cols"])
            if list(final_feature_cols) != expected_feature_cols:
                raise ValueError(
                    "Raster feature preparation artifact final schema does not match model input schema: "
                    f"{list(final_feature_cols)} != {expected_feature_cols}"
                )
            member_invalid_mask |= invalid_numeric_mask | missing_raw_na_mask | missing_input_nodata_mask
            member_invalid_mask |= transformed.loc[:, expected_feature_cols].isna().any(axis=1)
            member_valid_flat_mask = (~member_invalid_mask).to_numpy(dtype=bool, copy=False)
            x_valid = transformed.loc[~member_invalid_mask, expected_feature_cols].to_numpy(dtype=np.float32, copy=False)
        else:
            numeric_frame = flat_frame.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")
            invalid_numeric_mask = (flat_frame.loc[:, feature_cols].notna() & numeric_frame.isna()).any(axis=1)
            missing_raw_na_mask = flat_frame.loc[:, feature_cols].isna().any(axis=1)
            missing_input_nodata_mask = (numeric_frame == IN_MEMORY_NODATA_VALUE).any(axis=1)
            member_invalid_mask |= invalid_numeric_mask | missing_raw_na_mask | missing_input_nodata_mask
            member_valid_flat_mask = (~member_invalid_mask).to_numpy(dtype=bool, copy=False)
            x_valid = numeric_frame.loc[~member_invalid_mask, feature_cols].to_numpy(dtype=np.float32, copy=False)

        if valid_flat_mask is None:
            valid_flat_mask = member_valid_flat_mask
        elif not np.array_equal(valid_flat_mask, member_valid_flat_mask):
            raise ValueError("Raster ensemble members disagree on which pixels are valid after preprocessing.")

        valid_count = int(member_valid_flat_mask.sum())
        if valid_count == 0:
            continue

        x_scaled = member["scaler"].transform(x_valid)
        fold_predictions: list[NDArray[Any]] = []
        for start in range(0, valid_count, batch_size):
            stop = min(start + batch_size, valid_count)
            batch_tensor = torch.as_tensor(x_scaled[start:stop], dtype=torch.float32, device=device)
            with torch.inference_mode():
                _, prob_tensor = member["model"](batch_tensor)
            fold_predictions.append(prob_tensor.detach().cpu().numpy().reshape(-1))
        current_pred = np.concatenate(fold_predictions, axis=0).astype(np.float64, copy=False)
        member_predictions.append(current_pred)

    if valid_flat_mask is None:
        raise RuntimeError("No raster ensemble members available for prediction.")

    valid_count = int(valid_flat_mask.sum())
    if not member_predictions or valid_count == 0:
        return {
            name: np.full(base_invalid_mask.shape, nodata_value, dtype=np.float32)
            for name in metric_names
        }, {"valid_pixels": 0}

    summary = summarize_member_prediction_matrix(
        np.vstack(member_predictions),
        epsilon=float(uncertainty_epsilon),
        metric_names=metric_names,
    )
    blocks: dict[str, Any] = {}
    for metric_name, metric_values in summary.items():
        metric_block = np.full(base_invalid_mask.shape, nodata_value, dtype=np.float32)
        metric_block.reshape(-1)[valid_flat_mask] = metric_values.astype(np.float32, copy=False)
        blocks[metric_name] = metric_block
    mean_pred = summary["mean"]
    return blocks, {
        "valid_pixels": valid_count,
        "pred_min": float(mean_pred.min()),
        "pred_max": float(mean_pred.max()),
        "pred_mean": float(mean_pred.mean()),
        "member_count": int(len(member_predictions)),
    }


def normalize_runtime_job(job: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(job)
    if normalized.get("tile_size_primary") in (None, ""):
        if normalized.get("tile_size") not in (None, ""):
            normalized["tile_size_primary"] = normalized["tile_size"]
        else:
            normalized["tile_size_primary"] = 128
    normalized["tile_size_primary"] = normalize_positive_int(
        normalized["tile_size_primary"],
        field_name="tile_size_primary",
    )

    if normalized.get("tile_size_fallback") in (None, ""):
        if normalized.get("tile_size") not in (None, ""):
            normalized["tile_size_fallback"] = normalized["tile_size"]
        else:
            normalized["tile_size_fallback"] = 64
    normalized["tile_size_fallback"] = normalize_positive_int(
        normalized["tile_size_fallback"],
        field_name="tile_size_fallback",
    )
    if normalized["tile_size_fallback"] > normalized["tile_size_primary"]:
        normalized["tile_size_fallback"] = normalized["tile_size_primary"]

    normalized["cache_root"] = resolve_path(normalized.get("cache_root") or (PREDICTION_ROOT / "cache"))
    enable_cache = normalized.get("enable_cache")
    if enable_cache is not None:
        enable_cache = bool(enable_cache)
        normalized["enable_cache"] = enable_cache
        if not enable_cache:
            normalized["cache_policy"] = "none"
        elif normalized.get("cache_policy") in (None, ""):
            normalized["cache_policy"] = DEFAULT_CACHE_POLICY
    normalized["cache_policy"] = normalize_cache_policy(normalized.get("cache_policy"))
    if normalized.get("enable_cache") is None:
        normalized["enable_cache"] = normalized["cache_policy"] != "none"
    normalized["resume"] = bool(normalized.get("resume", True))
    normalized["download_retry"] = normalize_minimum_int(
        normalized.get("download_retry", 4),
        field_name="download_retry",
        minimum=1,
    )
    normalized["write_progress_every_n_tiles"] = normalize_positive_int(
        normalized.get("write_progress_every_n_tiles", 10),
        field_name="write_progress_every_n_tiles",
    )
    normalized["download_workers"] = normalize_positive_int(
        normalized.get("download_workers", 2),
        field_name="download_workers",
    )
    normalized["prefetch_queue_size"] = normalize_positive_int(
        normalized.get("prefetch_queue_size", 2),
        field_name="prefetch_queue_size",
    )
    normalized["confidence_class_count"] = normalize_minimum_int(
        normalized.get("confidence_class_count", 5),
        field_name="confidence_class_count",
        minimum=2,
    )
    normalized["confidence_sampling_per_tile"] = normalize_positive_int(
        normalized.get("confidence_sampling_per_tile", 4096),
        field_name="confidence_sampling_per_tile",
    )
    normalized["confidence_jenks_max_samples"] = normalize_positive_int(
        normalized.get("confidence_jenks_max_samples", 10000),
        field_name="confidence_jenks_max_samples",
    )
    normalized["show_progress"] = bool(normalized.get("show_progress", True))
    return normalized


def resolve_prediction_output_dir(default_path: str | Path) -> Path:
    return resolve_path(default_path)


def resolve_default_inference_batch_size(device_name: str) -> int:
    normalized = str(device_name or "").strip().lower()
    if normalized.startswith("cuda"):
        return 32768
    if normalized == "mps":
        return 16384
    return 4096


def resolve_inference_batch_size(job: dict[str, Any], device_meta: dict[str, Any]) -> int:
    explicit = job.get("inference_batch_size")
    if explicit not in (None, ""):
        return max(1, int(explicit))
    return resolve_default_inference_batch_size(str(device_meta.get("device_resolved") or job.get("device") or "cpu"))


def serialize_tile(tile: dict[str, Any]) -> dict[str, Any]:
    keys = ["row_idx", "col_idx", "row_off", "col_off", "width", "height", "xmin", "ymax", "ymin", "tile_size"]
    return {key: tile[key] for key in keys if key in tile}


def build_tile_state(tile: dict[str, Any], *, parent_tile_id: str | None = None) -> dict[str, Any]:
    return {
        "tile": serialize_tile(tile),
        "status": "pending",
        "parent_tile_id": parent_tile_id,
        "attempt_count": 0,
        "error": None,
        "cache_path": None,
        "static_cache_path": None,
        "dynamic_cache_path": None,
        "stats": None,
        "started_at": None,
        "download_started_at": None,
        "download_completed_at": None,
        "inference_started_at": None,
        "completed_at": None,
        "status_updated_at": utc_now_iso(),
    }


def build_progress_manifest(
    *,
    provider_name: str,
    target: dict[str, Any],
    output_path: Path,
    dataset_cache_key: str,
    static_dataset_cache_key: str,
    dynamic_dataset_cache_key: str,
    run_config_hash: str,
    model_bundle_identity: dict[str, Any],
    primary_tiles: list[dict[str, Any]],
    job: dict[str, Any],
) -> dict[str, Any]:
    order = [tile_id(tile) for tile in primary_tiles]
    tiles = {tile_id(tile): build_tile_state(tile) for tile in primary_tiles}
    return {
        "version": RUN_STATE_VERSION,
        "provider_name": provider_name,
        "target_id": target["target_id"],
        "output_path": str(output_path),
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset_cache_key": dataset_cache_key,
        "static_dataset_cache_key": static_dataset_cache_key,
        "dynamic_dataset_cache_key": dynamic_dataset_cache_key,
        "run_config_hash": run_config_hash,
        "model_bundle_hash": model_bundle_identity["bundle_hash"],
        "model_bundle_identity": model_bundle_identity,
        "cache_policy": job["cache_policy"],
        "tile_size_primary": int(job["tile_size_primary"]),
        "tile_size_fallback": int(job["tile_size_fallback"]),
        "resume_enabled": bool(job["resume"]),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "tile_order": order,
        "tiles": tiles,
    }


def progress_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {
        "pending": 0,
        "downloading": 0,
        "downloaded": 0,
        "inferring": 0,
        "done": 0,
        "failed": 0,
        "fallback_expanded": 0,
    }
    for state in manifest.get("tiles", {}).values():
        status = str(state.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def manifest_parent_tile_ids(manifest: dict[str, Any]) -> set[str]:
    tiles = manifest.get("tiles", {})
    if not isinstance(tiles, dict):
        return set()
    return {
        str(state.get("parent_tile_id"))
        for state in tiles.values()
        if isinstance(state, dict)
        and state.get("parent_tile_id") not in (None, "")
        and str(state.get("parent_tile_id")) in tiles
    }


def output_leaf_tile_ids(manifest: dict[str, Any]) -> list[str]:
    tiles = manifest.get("tiles", {})
    if not isinstance(tiles, dict):
        return []
    parent_ids = manifest_parent_tile_ids(manifest)
    ordered_ids: list[str] = []
    seen: set[str] = set()
    for raw_tile_id in manifest.get("tile_order", []):
        tile_key = str(raw_tile_id)
        if tile_key in tiles and tile_key not in parent_ids and tile_key not in seen:
            ordered_ids.append(tile_key)
            seen.add(tile_key)
    for raw_tile_id in tiles:
        tile_key = str(raw_tile_id)
        if tile_key not in parent_ids and tile_key not in seen:
            ordered_ids.append(tile_key)
            seen.add(tile_key)
    return ordered_ids


def output_progress_counts(manifest: dict[str, Any]) -> dict[str, int]:
    counts = {
        "pending": 0,
        "downloading": 0,
        "downloaded": 0,
        "inferring": 0,
        "done": 0,
        "failed": 0,
    }
    tiles = manifest.get("tiles", {})
    for tile_key in output_leaf_tile_ids(manifest):
        state = tiles.get(tile_key, {})
        status = str(state.get("status") or "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def manifest_output_complete(manifest: dict[str, Any]) -> bool:
    tiles = manifest.get("tiles", {})
    if not isinstance(tiles, dict):
        return False
    parent_ids = manifest_parent_tile_ids(manifest)
    parents_complete = all(
        str(tiles.get(tile_key, {}).get("status")) == "fallback_expanded"
        for tile_key in parent_ids
    )
    leaves_complete = all(
        str(tiles.get(tile_key, {}).get("status")) == "done"
        for tile_key in output_leaf_tile_ids(manifest)
    )
    return parents_complete and leaves_complete


def save_progress_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now_iso()
    manifest["progress_counts"] = progress_counts(manifest)
    save_json(path, manifest)


def reset_manifest_for_missing_output(manifest: dict[str, Any]) -> dict[str, Any]:
    """Requeue every output-writing leaf while preserving fallback topology and feature caches."""
    updated = json.loads(json.dumps(manifest))
    reset_at = utc_now_iso()
    parent_ids = manifest_parent_tile_ids(updated)
    reset_leaf_count = 0
    preserved_parent_count = 0
    for tile_key, state in updated.get("tiles", {}).items():
        if str(tile_key) in parent_ids:
            state["status"] = "fallback_expanded"
            state["error"] = None
            state["status_updated_at"] = reset_at
            preserved_parent_count += 1
            continue
        state["status"] = "pending"
        state["attempt_count"] = 0
        state["error"] = None
        state["stats"] = None
        state["started_at"] = None
        state["download_started_at"] = None
        state["download_completed_at"] = None
        state["inference_started_at"] = None
        state["completed_at"] = None
        state["status_updated_at"] = reset_at
        reset_leaf_count += 1
    restart_event = {
        "reason": "output_missing",
        "reset_at": reset_at,
        "reset_leaf_tile_count": reset_leaf_count,
        "preserved_fallback_parent_count": preserved_parent_count,
    }
    restart_history = updated.get("restart_history")
    if not isinstance(restart_history, list):
        restart_history = []
    restart_history.append(restart_event)
    updated["restart_history"] = restart_history
    updated["last_restart"] = restart_event
    updated["updated_at"] = reset_at
    return updated


def reset_manifest_for_restart(manifest: dict[str, Any]) -> dict[str, Any]:
    """Backward-compatible alias for the output-missing restart behavior."""
    return reset_manifest_for_missing_output(manifest)


def load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return load_json(path)


def prepare_progress_manifest(
    *,
    manifest_path: Path,
    output_path: Path,
    provider_name: str,
    target: dict[str, Any],
    dataset_cache_key: str,
    static_dataset_cache_key: str,
    dynamic_dataset_cache_key: str,
    run_config_hash: str,
    model_bundle_identity: dict[str, Any],
    primary_tiles: list[dict[str, Any]],
    job: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    download_only = bool(job.get("download_only", False))
    existing = load_existing_manifest(manifest_path)
    if existing is None:
        manifest = build_progress_manifest(
            provider_name=provider_name,
            target=target,
            output_path=output_path,
            dataset_cache_key=dataset_cache_key,
            static_dataset_cache_key=static_dataset_cache_key,
            dynamic_dataset_cache_key=dynamic_dataset_cache_key,
            run_config_hash=run_config_hash,
            model_bundle_identity=model_bundle_identity,
            primary_tiles=primary_tiles,
            job=job,
        )
        save_progress_manifest(manifest_path, manifest)
        return manifest, "new"

    expected_bundle_hash = model_bundle_identity["bundle_hash"]
    existing_bundle_hash = existing.get("model_bundle_hash")
    legacy_model_identity = not download_only and existing_bundle_hash in (None, "")
    model_bundle_changed = not download_only and existing_bundle_hash != expected_bundle_hash
    if existing.get("run_config_hash") != run_config_hash or legacy_model_identity or model_bundle_changed:
        if not bool(job.get("overwrite", False)):
            reason = (
                "the existing manifest predates model-bundle verification"
                if legacy_model_identity
                else (
                    "the model checkpoint, scaler, preprocessing artifact, or fold manifest has changed"
                    if model_bundle_changed
                    else "the projection configuration has changed"
                )
            )
            raise RuntimeError(
                f"Cannot safely resume because {reason}. Existing pixels may come from different model artifacts. "
                "Please explicitly pass --overwrite to rebuild all output pixels; downloaded feature caches can still be reused."
            )
        manifest = build_progress_manifest(
            provider_name=provider_name,
            target=target,
            output_path=output_path,
            dataset_cache_key=dataset_cache_key,
            static_dataset_cache_key=static_dataset_cache_key,
            dynamic_dataset_cache_key=dynamic_dataset_cache_key,
            run_config_hash=run_config_hash,
            model_bundle_identity=model_bundle_identity,
            primary_tiles=primary_tiles,
            job=job,
        )
        save_progress_manifest(manifest_path, manifest)
        return manifest, "reinitialized"

    if bool(job.get("overwrite", False)):
        manifest = build_progress_manifest(
            provider_name=provider_name,
            target=target,
            output_path=output_path,
            dataset_cache_key=dataset_cache_key,
            static_dataset_cache_key=static_dataset_cache_key,
            dynamic_dataset_cache_key=dynamic_dataset_cache_key,
            run_config_hash=run_config_hash,
            model_bundle_identity=model_bundle_identity,
            primary_tiles=primary_tiles,
            job=job,
        )
        save_progress_manifest(manifest_path, manifest)
        return manifest, "overwrite_reset"

    if download_only:
        if not bool(job.get("resume", True)):
            raise FileExistsError(f"Download manifest already exists and resume=false: {manifest_path}")
        return existing, "resume"

    if not output_path.exists():
        manifest = reset_manifest_for_missing_output(existing)
        save_progress_manifest(manifest_path, manifest)
        return manifest, "output_missing_reset"

    if not bool(job.get("resume", True)):
        raise FileExistsError(f"Output already exists and resume=false: {output_path}")

    return existing, "resume"


def build_component_dataset_cache_key(
    *,
    target: dict[str, Any],
    feature_cols: list[str],
    grid: dict[str, Any],
    job: dict[str, Any],
    component: str,
    stack_signature: str,
) -> str:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "component": component,
        "stack_signature": stack_signature,
        "feature_cols": list(feature_cols),
        "target_projection_wkt": ee_ctx.TARGET_PROJECTION_WKT,
        "target_scale": ee_ctx.TARGET_SCALE,
        "grid": grid,
        "tile_size_primary": int(job["tile_size_primary"]),
        "tile_size_fallback": int(job["tile_size_fallback"]),
        "nodata_value": float(job.get("nodata_value", IN_MEMORY_NODATA_VALUE)),
    }
    return compute_payload_hash(payload)


def build_dataset_cache_key(
    *,
    provider_name: str,
    target: dict[str, Any],
    feature_cols: list[str],
    grid: dict[str, Any],
    job: dict[str, Any],
    static_dataset_cache_key: str,
    dynamic_dataset_cache_key: str,
) -> str:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "provider_name": provider_name,
        "target_id": target["target_id"],
        "stack_signature": target["stack_signature"],
        "feature_cols": list(feature_cols),
        "static_feature_cols": list(target["static_feature_cols"]),
        "dynamic_feature_cols": list(target["dynamic_feature_cols"]),
        "static_dataset_cache_key": static_dataset_cache_key,
        "dynamic_dataset_cache_key": dynamic_dataset_cache_key,
        "target_projection_wkt": ee_ctx.TARGET_PROJECTION_WKT,
        "target_scale": ee_ctx.TARGET_SCALE,
        "grid": grid,
        "tile_size_primary": int(job["tile_size_primary"]),
        "tile_size_fallback": int(job["tile_size_fallback"]),
        "nodata_value": float(job.get("nodata_value", IN_MEMORY_NODATA_VALUE)),
        "input_source": target.get("input_source", "asset"),
    }
    return compute_payload_hash(payload)


def build_run_config_hash(
    *,
    dataset_cache_key: str,
    static_dataset_cache_key: str,
    dynamic_dataset_cache_key: str,
    model_bundle_hash: str,
    job: dict[str, Any],
    provider_name: str,
    output_path: Path,
) -> str:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "dataset_cache_key": dataset_cache_key,
        "static_dataset_cache_key": static_dataset_cache_key,
        "dynamic_dataset_cache_key": dynamic_dataset_cache_key,
        "model_bundle_hash": model_bundle_hash,
        "model_backend": normalize_model_backend(job.get("model_backend")),
        "provider_name": provider_name,
        "output_path": str(output_path),
        "device": str(job.get("device") or "cpu"),
        "inference_batch_size": job.get("inference_batch_size"),
        "cache_policy": job["cache_policy"],
        "download_only": bool(job.get("download_only", False)),
    }
    return compute_payload_hash(payload)


def build_static_cache_path(cache_root: Path, dataset_cache_key: str, tile: dict[str, Any]) -> Path:
    return cache_root / "static_tiles" / dataset_cache_key / f"{tile_id(tile)}.npz"


def build_dynamic_cache_path(cache_root: Path, dataset_cache_key: str, tile: dict[str, Any]) -> Path:
    return cache_root / "dynamic_tiles" / dataset_cache_key / f"{tile_id(tile)}.npz"


def load_cached_static_tile(cache_path: Path) -> tuple[Any, Any] | None:
    import_runtime_dependencies()
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as payload:
        feature_array = payload["feature_array"].astype(np.float32, copy=False)
        aoi_mask = payload["aoi_mask"].astype(np.float32, copy=False)
    return feature_array, aoi_mask


def save_cached_static_tile(cache_path: Path, feature_array: Any, aoi_mask: Any) -> Path:
    import_runtime_dependencies()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feature_array=feature_array.astype(np.float32), aoi_mask=aoi_mask.astype(np.float32))
    return cache_path


def load_cached_dynamic_tile(cache_path: Path) -> Any | None:
    import_runtime_dependencies()
    if not cache_path.exists():
        return None
    with np.load(cache_path, allow_pickle=False) as payload:
        return payload["feature_array"].astype(np.float32, copy=False)


def save_cached_dynamic_tile(cache_path: Path, feature_array: Any) -> Path:
    import_runtime_dependencies()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feature_array=feature_array.astype(np.float32))
    return cache_path


def ensure_output_dataset(
    *,
    output_path: Path,
    profile: dict[str, Any],
    tags: dict[str, Any],
    overwrite: bool,
    resume_mode: bool,
    band_name: str = PREDICTION_RESULT_BAND_NAME,
) -> Any:
    import_runtime_dependencies()
    if output_path.exists() and resume_mode and not overwrite:
        return rasterio.open(output_path, "r+")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.set_band_description(1, str(band_name))
        dst.update_tags(**tags)
    return rasterio.open(output_path, "r+")


def prepare_uncertainty_output_datasets(
    *,
    output_paths: dict[str, Any],
    profile: dict[str, Any],
    output_tags: dict[str, Any],
    overwrite: bool,
    resume_mode: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    metric_datasets: dict[str, Any] = {}
    metric_path_strings: dict[str, str] = {}
    for metric_name, metric_path in output_paths["metric_paths"].items():
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_datasets[metric_name] = ensure_output_dataset(
            output_path=metric_path,
            profile=profile,
            tags={**output_tags, "metric_name": metric_name, "metric_family": "uncertainty"},
            overwrite=overwrite,
            resume_mode=resume_mode,
            band_name=metric_name,
        )
        metric_path_strings[metric_name] = str(metric_path)
    return metric_datasets, metric_path_strings


def close_output_datasets(datasets: dict[str, Any]) -> None:
    for dataset in datasets.values():
        try:
            dataset.close()
        except Exception:
            continue


def all_uncertainty_metric_outputs_exist(output_paths: dict[str, Any]) -> bool:
    return all(Path(path).exists() for path in output_paths["metric_paths"].values())


def collect_metric_samples_from_rasters(
    *,
    mean_path: Path,
    std_path: Path,
    manifest: dict[str, Any],
    target_sample_size: int,
    seed: int,
) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    import_runtime_dependencies()
    target_sample_size = max(1, int(target_sample_size))
    rng = np.random.default_rng(int(seed))
    done_tiles = [
        dict(manifest["tiles"][tile_key]["tile"])
        for tile_key in manifest.get("tile_order", [])
        if str(manifest["tiles"].get(tile_key, {}).get("status")) == "done"
    ]
    total_pixel_count = sum(int(tile["width"]) * int(tile["height"]) for tile in done_tiles)
    sampling_step = max(1, int(math.ceil(math.sqrt(max(1, total_pixel_count) / float(target_sample_size)))))
    skip_factor = sampling_step - 1
    mean_samples: list[NDArray[Any]] = []
    std_samples: list[NDArray[Any]] = []
    with rasterio.open(mean_path) as mean_src, rasterio.open(std_path) as std_src:
        for tile in done_tiles:
            window = window_factory(
                int(tile["col_off"]),
                int(tile["row_off"]),
                int(tile["width"]),
                int(tile["height"]),
            )
            mean_block = mean_src.read(1, window=window)
            std_block = std_src.read(1, window=window)
            mean_nodata = mean_src.nodata
            std_nodata = std_src.nodata
            valid_mask = np.isfinite(mean_block) & np.isfinite(std_block)
            if mean_nodata is not None:
                valid_mask &= mean_block != mean_nodata
            if std_nodata is not None:
                valid_mask &= std_block != std_nodata
            if not np.any(valid_mask):
                continue
            row_start = (-int(tile["row_off"])) % sampling_step
            col_start = (-int(tile["col_off"])) % sampling_step
            mean_sparse = mean_block[row_start::sampling_step, col_start::sampling_step]
            std_sparse = std_block[row_start::sampling_step, col_start::sampling_step]
            valid_sparse_mask = valid_mask[row_start::sampling_step, col_start::sampling_step]
            if not np.any(valid_sparse_mask):
                continue
            mean_samples.append(np.asarray(mean_sparse[valid_sparse_mask], dtype=np.float32).reshape(-1))
            std_samples.append(np.asarray(std_sparse[valid_sparse_mask], dtype=np.float32).reshape(-1))

    mean_sample = np.concatenate(mean_samples).astype(np.float32, copy=False) if mean_samples else np.asarray([], dtype=np.float32)
    std_sample = np.concatenate(std_samples).astype(np.float32, copy=False) if std_samples else np.asarray([], dtype=np.float32)
    if std_sample.size > target_sample_size:
        indices = rng.choice(std_sample.size, size=target_sample_size, replace=False)
        mean_sample = mean_sample[indices].astype(np.float32, copy=False)
        std_sample = std_sample[indices].astype(np.float32, copy=False)
    return (
        {
            "mean": mean_sample,
            "std": std_sample,
        },
        {
            "sampling_method": "systematic_skip",
            "sample_target_size": int(target_sample_size),
            "sampling_step": int(sampling_step),
            "skip_factor": int(skip_factor),
            "sample_count_mean": int(mean_sample.size),
            "sample_count_std": int(std_sample.size),
            "sample_seed": int(seed),
        },
    )


def build_confidence_raster(
    *,
    mean_path: Path,
    std_path: Path,
    confidence_path: Path,
    confidence_summary_path: Path,
    manifest: dict[str, Any],
    class_count: int,
    nodata_value: int,
    sample_per_tile: int,
    jenks_max_samples: int,
    sample_seed: int,
    overwrite: bool,
    resume_mode: bool,
) -> dict[str, Any]:
    import_runtime_dependencies()
    emit_runtime_event(
        "confidence_stage_start",
        mean_path=str(mean_path),
        std_path=str(std_path),
        confidence_path=str(confidence_path),
        class_count=int(class_count),
        sample_per_tile=int(sample_per_tile),
        jenks_max_samples=int(jenks_max_samples),
    )
    samples, sampling_meta = collect_metric_samples_from_rasters(
        mean_path=mean_path,
        std_path=std_path,
        manifest=manifest,
        target_sample_size=max(1, int(jenks_max_samples)),
        seed=int(sample_seed),
    )
    emit_runtime_event(
        "confidence_sampling_complete",
        sample_count_mean=int(samples["mean"].size),
        sample_count_std=int(samples["std"].size),
        sampling_method=str(sampling_meta["sampling_method"]),
        sample_target_size=int(sampling_meta["sample_target_size"]),
        sampling_step=int(sampling_meta["sampling_step"]),
        skip_factor=int(sampling_meta["skip_factor"]),
    )
    if samples["mean"].size == 0 or samples["std"].size == 0:
        summary = {
            "confidence_enabled": False,
            "reason": "no_finite_samples",
            "confidence_output_path": str(confidence_path),
            "confidence_class_count": int(class_count),
            "sample_count_mean": int(samples["mean"].size),
            "sample_count_std": int(samples["std"].size),
            "class_rows": [],
        }
        save_json(confidence_summary_path, summary)
        return summary

    mean_thresholds = compute_fixed_width_thresholds(start=0.0, stop=1.0, class_count=int(class_count))
    jenks_rng = np.random.default_rng(int(sample_seed))
    std_sample_for_jenks = sample_finite_values(
        samples["std"],
        max_values=min(int(samples["std"].size), max(1, int(jenks_max_samples))),
        rng=jenks_rng,
    )
    emit_runtime_event(
        "confidence_jenks_start",
        class_count=int(class_count),
        jenks_sample_count_std=int(std_sample_for_jenks.size),
    )
    std_thresholds = compute_jenks_thresholds(std_sample_for_jenks, class_count=int(class_count))
    emit_runtime_event(
        "confidence_jenks_complete",
        uncertainty_thresholds=[float(value) for value in std_thresholds.tolist()],
    )

    with rasterio.open(mean_path) as mean_src, rasterio.open(std_path) as std_src:
        confidence_profile = dict(mean_src.profile)
        confidence_profile["dtype"] = "int16"
        confidence_profile["nodata"] = int(nodata_value)
        confidence_path.parent.mkdir(parents=True, exist_ok=True)
        with ensure_output_dataset(
            output_path=confidence_path,
            profile=confidence_profile,
            tags={
                "metric_name": "confidence",
                "classification_method": "fixed_mean_plus_jenks_std",
                "susceptibility_source": str(mean_path),
                "uncertainty_source": str(std_path),
            },
            overwrite=overwrite,
            resume_mode=resume_mode,
            band_name=CONFIDENCE_RESULT_BAND_NAME,
        ) as confidence_dst:
            code_counts: dict[int, int] = {}
            mean_nodata = mean_src.nodata
            std_nodata = std_src.nodata
            processed_tiles = 0
            total_tiles = sum(
                1 for tile_key in manifest.get("tile_order", [])
                if str(manifest["tiles"].get(tile_key, {}).get("status")) == "done"
            )
            progress_interval = max(1, min(100, total_tiles // 10 if total_tiles > 0 else 1))
            for tile_key in manifest.get("tile_order", []):
                state = manifest["tiles"].get(tile_key, {})
                if str(state.get("status")) != "done":
                    continue
                tile = state["tile"]
                window = window_factory(
                    int(tile["col_off"]),
                    int(tile["row_off"]),
                    int(tile["width"]),
                    int(tile["height"]),
                )
                mean_block = mean_src.read(1, window=window)
                std_block = std_src.read(1, window=window)
                valid_mask = np.isfinite(mean_block) & np.isfinite(std_block)
                if mean_nodata is not None:
                    valid_mask &= mean_block != mean_nodata
                if std_nodata is not None:
                    valid_mask &= std_block != std_nodata
                susc_classes = classify_by_thresholds(mean_block, thresholds=mean_thresholds, valid_mask=valid_mask)
                unc_classes = classify_by_thresholds(std_block, thresholds=std_thresholds, valid_mask=valid_mask)
                confidence_codes = encode_confidence_classes(
                    susc_classes,
                    unc_classes,
                    nodata_value=int(nodata_value),
                )
                confidence_dst.write(confidence_codes.astype(np.int16, copy=False), 1, window=window)
                valid_codes = confidence_codes[confidence_codes != int(nodata_value)]
                if valid_codes.size > 0:
                    unique_codes, unique_counts = np.unique(valid_codes.reshape(-1), return_counts=True)
                    for code, count in zip(unique_codes.tolist(), unique_counts.tolist()):
                        code_counts[int(code)] = code_counts.get(int(code), 0) + int(count)
                processed_tiles += 1
                if processed_tiles == 1 or processed_tiles % progress_interval == 0 or processed_tiles == total_tiles:
                    emit_runtime_event(
                        "confidence_write_progress",
                        processed_tiles=int(processed_tiles),
                        total_tiles=int(total_tiles),
                    )

    class_rows = []
    total_pixels = sum(code_counts.values())
    if total_pixels > 0:
        for code in sorted(code_counts):
            count = int(code_counts[code])
            class_rows.append(
                {
                    "confidence_code": int(code),
                    "uncertainty_class": int(code // 10),
                    "susceptibility_class": int(code % 10),
                    "pixel_count": count,
                    "pixel_ratio": float(count / total_pixels),
                }
            )

    summary = {
        "confidence_enabled": True,
        "confidence_output_path": str(confidence_path),
        "confidence_class_count": int(class_count),
        "classification_method": "fixed_mean_plus_jenks_std",
        "sample_count_mean": int(samples["mean"].size),
        "sample_count_std": int(samples["std"].size),
        "jenks_sample_count_std": int(std_sample_for_jenks.size),
        "uncertainty_sampling": sampling_meta,
        "susceptibility_classification": {
            "method": "fixed_width",
            "range": [0.0, 1.0],
            "class_count": int(class_count),
        },
        "uncertainty_classification": {
            "method": "natural_breaks_jenks",
            "class_count": int(class_count),
            "source_metric": "std",
        },
        "susceptibility_thresholds": [float(value) for value in mean_thresholds.tolist()],
        "uncertainty_thresholds": [float(value) for value in std_thresholds.tolist()],
        "class_rows": class_rows,
    }
    save_json(confidence_summary_path, summary)
    emit_runtime_event(
        "confidence_stage_complete",
        confidence_output_path=str(confidence_path),
        confidence_summary_path=str(confidence_summary_path),
    )
    return summary


def mark_tile_failed(manifest: dict[str, Any], tile_key: str, error: str) -> None:
    state = manifest["tiles"][tile_key]
    state["status"] = "failed"
    state["error"] = error
    state["completed_at"] = utc_now_iso()
    state["status_updated_at"] = utc_now_iso()


def update_tile_status(
    manifest: dict[str, Any],
    tile_key: str,
    status: str,
    *,
    error: str | None = None,
    cache_path: str | None = None,
    static_cache_path: str | None = None,
    dynamic_cache_path: str | None = None,
    stats: dict[str, Any] | None = None,
) -> None:
    now = utc_now_iso()
    state = manifest["tiles"][tile_key]
    state["status"] = status
    state["status_updated_at"] = now
    if error is not None:
        state["error"] = error
    if cache_path is not None:
        state["cache_path"] = cache_path
    if static_cache_path is not None:
        state["static_cache_path"] = static_cache_path
    if dynamic_cache_path is not None:
        state["dynamic_cache_path"] = dynamic_cache_path
    if stats is not None:
        state["stats"] = stats
    if status == "downloading":
        if state.get("started_at") is None:
            state["started_at"] = now
        state["download_started_at"] = now
    elif status == "downloaded":
        if state.get("download_started_at") is None:
            state["download_started_at"] = now
        state["download_completed_at"] = now
    elif status == "inferring":
        if state.get("download_completed_at") is None:
            state["download_completed_at"] = now
        state["inference_started_at"] = now
    elif status == "done":
        if state.get("inference_started_at") is None:
            state["inference_started_at"] = now
        state["completed_at"] = now
    elif status == "pending":
        state["error"] = None


def assemble_feature_tile(
    *,
    static_feature_array: Any,
    dynamic_feature_array: Any,
    static_feature_cols: list[str],
    dynamic_feature_cols: list[str],
    feature_cols: list[str],
) -> Any:
    import_runtime_dependencies()
    if static_feature_array.shape[:2] != dynamic_feature_array.shape[:2]:
        raise ValueError(
            "Static and dynamic tile shapes do not match: "
            f"{static_feature_array.shape[:2]} != {dynamic_feature_array.shape[:2]}"
        )
    output = np.empty(
        (int(static_feature_array.shape[0]), int(static_feature_array.shape[1]), len(feature_cols)),
        dtype=np.float32,
    )
    static_lookup = {name: idx for idx, name in enumerate(static_feature_cols)}
    dynamic_lookup = {name: idx for idx, name in enumerate(dynamic_feature_cols)}
    for out_idx, band_name in enumerate(feature_cols):
        if band_name in static_lookup:
            output[:, :, out_idx] = static_feature_array[:, :, static_lookup[band_name]]
        elif band_name in dynamic_lookup:
            output[:, :, out_idx] = dynamic_feature_array[:, :, dynamic_lookup[band_name]]
        else:
            raise KeyError(f"Feature band `{band_name}` is missing from both static and dynamic tile caches.")
    return output


def prepare_tile_data(
    *,
    target: dict[str, Any],
    tile: dict[str, Any],
    feature_cols: list[str],
    static_cache_path: Path,
    dynamic_cache_path: Path,
    cache_policy: str,
    download_retry: int,
) -> tuple[Any, Any, dict[str, Any]]:
    current_tile_id = tile_id(tile)
    if cache_policy != "feature_tiles":
        emit_runtime_event(
            "tile_download_start",
            mode="combined",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(feature_cols) + 1,
        )
        started_at = time.perf_counter()
        tile_array = download_tile_array(
            target["download_image"],
            tile=tile,
            band_ids=feature_cols + [AUX_MASK_BAND],
            retry=download_retry,
        )
        emit_runtime_event(
            "tile_download_complete",
            mode="combined",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(feature_cols) + 1,
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            shape=list(tile_array.shape),
        )
        feature_array = tile_array[:, :, : len(feature_cols)]
        aoi_mask = tile_array[:, :, -1]
        return feature_array, aoi_mask, {
            "cache_enabled": False,
            "cache_hit": False,
            "static_cache_hit": False,
            "dynamic_cache_hit": False,
            "download_mode": "combined",
        }

    static_feature_cols = list(target["static_feature_cols"])
    dynamic_feature_cols = list(target["dynamic_feature_cols"])
    static_cached = load_cached_static_tile(static_cache_path)
    dynamic_cached = load_cached_dynamic_tile(dynamic_cache_path)

    static_cache_hit = static_cached is not None
    dynamic_cache_hit = dynamic_cached is not None

    if static_cached is None:
        emit_runtime_event(
            "tile_download_start",
            mode="static",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(static_feature_cols) + 1,
        )
        started_at = time.perf_counter()
        static_tile_array = download_tile_array(
            target["static_download_image"],
            tile=tile,
            band_ids=static_feature_cols + [AUX_MASK_BAND],
            retry=download_retry,
        )
        emit_runtime_event(
            "tile_download_complete",
            mode="static",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(static_feature_cols) + 1,
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            shape=list(static_tile_array.shape),
        )
        static_feature_array = static_tile_array[:, :, : len(static_feature_cols)]
        aoi_mask = static_tile_array[:, :, -1]
        save_cached_static_tile(static_cache_path, static_feature_array, aoi_mask)
        emit_runtime_event(
            "tile_cache_saved",
            mode="static",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            cache_path=str(static_cache_path),
        )
    else:
        static_feature_array, aoi_mask = static_cached
        emit_runtime_event(
            "tile_cache_hit",
            mode="static",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            cache_path=str(static_cache_path),
        )

    if dynamic_cached is None:
        emit_runtime_event(
            "tile_download_start",
            mode="dynamic",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(dynamic_feature_cols),
        )
        started_at = time.perf_counter()
        dynamic_tile_array = download_tile_array(
            target["dynamic_download_image"],
            tile=tile,
            band_ids=dynamic_feature_cols,
            retry=download_retry,
        )
        emit_runtime_event(
            "tile_download_complete",
            mode="dynamic",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            band_count=len(dynamic_feature_cols),
            elapsed_seconds=round(time.perf_counter() - started_at, 3),
            shape=list(dynamic_tile_array.shape),
        )
        dynamic_feature_array = dynamic_tile_array[:, :, : len(dynamic_feature_cols)]
        save_cached_dynamic_tile(dynamic_cache_path, dynamic_feature_array)
        emit_runtime_event(
            "tile_cache_saved",
            mode="dynamic",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            cache_path=str(dynamic_cache_path),
        )
    else:
        dynamic_feature_array = dynamic_cached
        emit_runtime_event(
            "tile_cache_hit",
            mode="dynamic",
            target_id=target.get("target_id"),
            tile_id=current_tile_id,
            cache_path=str(dynamic_cache_path),
        )

    feature_array = assemble_feature_tile(
        static_feature_array=static_feature_array,
        dynamic_feature_array=dynamic_feature_array,
        static_feature_cols=static_feature_cols,
        dynamic_feature_cols=dynamic_feature_cols,
        feature_cols=feature_cols,
    )
    return feature_array, aoi_mask, {
        "cache_enabled": True,
        "cache_hit": bool(static_cache_hit and dynamic_cache_hit),
        "static_cache_hit": bool(static_cache_hit),
        "dynamic_cache_hit": bool(dynamic_cache_hit),
        "download_mode": "split",
    }


def submit_tile_prepare_task(
    *,
    executor: ThreadPoolExecutor,
    target: dict[str, Any],
    tile: dict[str, Any],
    feature_cols: list[str],
    static_cache_path: Path,
    dynamic_cache_path: Path,
    cache_policy: str,
    download_retry: int,
) -> TilePreparationFuture:
    return executor.submit(
        prepare_tile_data,
        target=target,
        tile=tile,
        feature_cols=feature_cols,
        static_cache_path=static_cache_path,
        dynamic_cache_path=dynamic_cache_path,
        cache_policy=cache_policy,
        download_retry=download_retry,
    )


def fill_prefetch_window(
    *,
    executor: ThreadPoolExecutor,
    pending_futures: dict[str, TilePreparationFuture],
    manifest: dict[str, Any],
    manifest_path: Path,
    start_index: int,
    max_pending: int,
    cache_root: Path,
    static_dataset_cache_key: str,
    dynamic_dataset_cache_key: str,
    target: dict[str, Any],
    feature_cols: list[str],
    cache_policy: str,
    download_retry: int,
) -> None:
    if max_pending <= 0:
        return
    tile_order = manifest.get("tile_order", [])
    for scan_idx in range(int(start_index), len(tile_order)):
        if len(pending_futures) >= max_pending:
            break
        tile_key = tile_order[scan_idx]
        if tile_key in pending_futures:
            continue
        tile_state = manifest["tiles"][tile_key]
        if tile_state["status"] != "pending":
            continue
        tile = dict(tile_state["tile"])
        static_cache_path = build_static_cache_path(cache_root, static_dataset_cache_key, tile)
        dynamic_cache_path = build_dynamic_cache_path(cache_root, dynamic_dataset_cache_key, tile)
        static_cache_path_str = str(static_cache_path) if cache_policy == "feature_tiles" else None
        dynamic_cache_path_str = str(dynamic_cache_path) if cache_policy == "feature_tiles" else None
        update_tile_status(
            manifest,
            tile_key,
            "downloading",
            cache_path=None,
            static_cache_path=static_cache_path_str,
            dynamic_cache_path=dynamic_cache_path_str,
        )
        save_progress_manifest(manifest_path, manifest)
        pending_futures[tile_key] = submit_tile_prepare_task(
            executor=executor,
            target=target,
            tile=tile,
            feature_cols=feature_cols,
            static_cache_path=static_cache_path,
            dynamic_cache_path=dynamic_cache_path,
            cache_policy=cache_policy,
            download_retry=download_retry,
        )


def insert_fallback_tiles(
    manifest: dict[str, Any],
    *,
    parent_tile_id: str,
    fallback_tiles: list[dict[str, Any]],
    insert_after_index: int,
) -> None:
    manifest["tiles"][parent_tile_id]["status"] = "fallback_expanded"
    manifest["tiles"][parent_tile_id]["completed_at"] = utc_now_iso()
    new_ids: list[str] = []
    for tile in fallback_tiles:
        child_key = tile_id(tile)
        if child_key not in manifest["tiles"]:
            manifest["tiles"][child_key] = build_tile_state(tile, parent_tile_id=parent_tile_id)
        new_ids.append(child_key)
    manifest["tile_order"][insert_after_index + 1:insert_after_index + 1] = new_ids


def run_projection_target(
    *,
    provider_name: str,
    source_script: str,
    job: dict[str, Any],
    cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    target: dict[str, Any],
    feature_cols: list[str],
    primary_tiles: list[dict[str, Any]],
    profile: dict[str, Any],
    raster_output_dir: Path,
    json_output_dir: Path,
    cache_root: Path,
    device: Any,
    device_meta: dict[str, Any],
    model_members: list[dict[str, Any]],
    model_bundle_identity: dict[str, Any],
    runtime_summary: dict[str, Any],
    grid_summary: dict[str, Any],
) -> dict[str, Any]:
    import_runtime_dependencies()
    download_only = bool(job.get("download_only", False))
    output_path = raster_output_dir / target["output_name"]
    output_stem = output_path.stem
    uncertainty_enabled = bool(not download_only and should_write_uncertainty_outputs(job, model_members))
    uncertainty_export_metrics = resolve_uncertainty_export_metrics(job) if uncertainty_enabled else tuple()
    backfill_uncertainty_only = bool(
        not download_only
        and uncertainty_enabled
        and should_backfill_uncertainty_only(job, model_members)
    )
    confidence_enabled = bool(
        not download_only
        and uncertainty_enabled
        and should_write_confidence_outputs(job, model_members)
    )
    uncertainty_output_paths = build_uncertainty_output_paths(
        output_path=output_path,
        json_output_dir=json_output_dir,
        job=job,
    )
    if uncertainty_enabled:
        uncertainty_output_paths["raster_dir"].mkdir(parents=True, exist_ok=True)
        uncertainty_output_paths["json_dir"].mkdir(parents=True, exist_ok=True)
    primary_manifest_path = json_output_dir / f"{output_stem}_progress.json"
    manifest_path = primary_manifest_path
    failed_tiles_path = json_output_dir / f"{output_stem}_failed_tiles.json"
    runtime_summary_path = json_output_dir / f"{output_stem}_runtime_summary.json"
    if backfill_uncertainty_only:
        manifest_path = uncertainty_output_paths["json_dir"] / f"{output_stem}{UNCERTAINTY_BACKFILL_PROGRESS_SUFFIX}"
        failed_tiles_path = uncertainty_output_paths["json_dir"] / f"{output_stem}{UNCERTAINTY_BACKFILL_FAILED_SUFFIX}"
        runtime_summary_path = uncertainty_output_paths["json_dir"] / f"{output_stem}{UNCERTAINTY_BACKFILL_RUNTIME_SUFFIX}"
        if not output_path.exists():
            raise FileNotFoundError(
                "backfill_uncertainty_only=true requires an existing primary mean raster; "
                f"missing={output_path}"
            )
        primary_manifest = load_existing_manifest(primary_manifest_path)
        if primary_manifest is None:
            raise RuntimeError(
                "Cannot verify backfill_uncertainty_only against the primary mean raster because its progress manifest is missing. Rebuild the projection with --overwrite."
            )
        if primary_manifest.get("model_bundle_hash") != model_bundle_identity["bundle_hash"]:
            raise RuntimeError(
                "Cannot backfill uncertainty because the primary mean raster was not verified with the current model bundle. Rebuild the full projection with --overwrite."
            )

    static_dataset_cache_key = build_component_dataset_cache_key(
        target=target,
        feature_cols=target["static_feature_cols"],
        grid=grid_summary,
        job=job,
        component="static",
        stack_signature=target["static_stack_signature"],
    )
    dynamic_dataset_cache_key = build_component_dataset_cache_key(
        target=target,
        feature_cols=target["dynamic_feature_cols"],
        grid=grid_summary,
        job=job,
        component="dynamic",
        stack_signature=target["dynamic_stack_signature"],
    )
    dataset_cache_key = build_dataset_cache_key(
        provider_name=provider_name,
        target=target,
        feature_cols=feature_cols,
        grid=grid_summary,
        job=job,
        static_dataset_cache_key=static_dataset_cache_key,
        dynamic_dataset_cache_key=dynamic_dataset_cache_key,
    )
    run_config_hash = build_run_config_hash(
        dataset_cache_key=dataset_cache_key,
        static_dataset_cache_key=static_dataset_cache_key,
        dynamic_dataset_cache_key=dynamic_dataset_cache_key,
        model_bundle_hash=model_bundle_identity["bundle_hash"],
        job=job,
        provider_name=provider_name,
        output_path=output_path,
    )
    static_cache_dir = cache_root / "static_tiles" / static_dataset_cache_key
    dynamic_cache_dir = cache_root / "dynamic_tiles" / dynamic_dataset_cache_key
    manifest, manifest_mode = prepare_progress_manifest(
        manifest_path=manifest_path,
        output_path=output_path,
        provider_name=provider_name,
        target=target,
        dataset_cache_key=dataset_cache_key,
        static_dataset_cache_key=static_dataset_cache_key,
        dynamic_dataset_cache_key=dynamic_dataset_cache_key,
        run_config_hash=run_config_hash,
        model_bundle_identity=model_bundle_identity,
        primary_tiles=primary_tiles,
        job=job,
    )

    if not output_path.exists() and not bool(job.get("overwrite", False)):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        not download_only
        and not backfill_uncertainty_only
        and output_path.exists()
        and not bool(job.get("overwrite", False))
        and not bool(job.get("resume", True))
        and manifest_mode == "new"
    ):
        raise FileExistsError(f"Output already exists and overwrite=False: {output_path}")

    if manifest_output_complete(manifest) and (download_only or output_path.exists()):
        existing_uncertainty_metric_paths: dict[str, str] = {
            metric_name: str(metric_path)
            for metric_name, metric_path in uncertainty_output_paths["metric_paths"].items()
            if Path(metric_path).exists()
        }
        confidence_summary = None
        if confidence_enabled and all_uncertainty_metric_outputs_exist(uncertainty_output_paths):
            confidence_missing = not Path(uncertainty_output_paths["confidence_path"]).exists()
            confidence_summary_missing = not Path(uncertainty_output_paths["confidence_summary_path"]).exists()
            if confidence_missing or confidence_summary_missing:
                confidence_summary = build_confidence_raster(
                    mean_path=output_path,
                    std_path=uncertainty_output_paths["metric_paths"]["std"],
                    confidence_path=uncertainty_output_paths["confidence_path"],
                    confidence_summary_path=uncertainty_output_paths["confidence_summary_path"],
                    manifest=manifest,
                    class_count=max(2, int(job.get("confidence_class_count", 5) or 5)),
                    nodata_value=int(job.get("confidence_nodata_value", -9999) or -9999),
                    sample_per_tile=max(1, int(job.get("confidence_sampling_per_tile", 4096) or 4096)),
                    jenks_max_samples=max(1, int(job.get("confidence_jenks_max_samples", 10000) or 10000)),
                    sample_seed=int(job.get("confidence_sample_seed", 42) or 42),
                    overwrite=bool(job.get("overwrite", False)),
                    resume_mode=bool(job.get("resume", True)),
                )
        elif uncertainty_enabled and not all_uncertainty_metric_outputs_exist(uncertainty_output_paths):
            raise RuntimeError(
                "Primary raster prediction is already complete, but uncertainty rasters are missing. "
                "Re-run with --backfill-uncertainty-only to backfill uncertainty outputs without rewriting mean, "
                "or use --overwrite for a full rerun."
            )
        if uncertainty_enabled and not Path(uncertainty_output_paths["uncertainty_summary_path"]).exists():
            save_json(
                uncertainty_output_paths["uncertainty_summary_path"],
                {
                    "uncertainty_enabled": True,
                    "metric_output_paths": existing_uncertainty_metric_paths,
                    "uncertainty_metrics": list(uncertainty_export_metrics),
                    "confidence_enabled": confidence_enabled,
                    "confidence_output_path": (
                        None
                        if not confidence_enabled or not Path(uncertainty_output_paths["confidence_path"]).exists()
                        else str(uncertainty_output_paths["confidence_path"])
                    ),
                    "confidence_summary_path": (
                        None
                        if not confidence_enabled or not Path(uncertainty_output_paths["confidence_summary_path"]).exists()
                        else str(uncertainty_output_paths["confidence_summary_path"])
                    ),
                    "uncertainty_epsilon": float(job.get("uncertainty_epsilon", 1e-6) or 1e-6),
                    "confidence_class_count": int(job.get("confidence_class_count", 5) or 5),
                    "confidence_sampling_per_tile": int(job.get("confidence_sampling_per_tile", 4096) or 4096),
                    "confidence_jenks_max_samples": int(job.get("confidence_jenks_max_samples", 10000) or 10000),
                    "backfill_uncertainty_only": backfill_uncertainty_only,
                },
            )
        result = {
            **target["result_fields"],
            "output_path": None if download_only else str(output_path),
            "grid_width": int(grid_summary["width"]),
            "grid_height": int(grid_summary["height"]),
            "tile_count": len(manifest.get("tile_order", [])),
            "valid_pixels": None,
            "pred_min": None,
            "pred_max": None,
            "pred_mean": None,
            "ensemble_members": len(model_members),
            "model_bundle_hash": model_bundle_identity["bundle_hash"],
            "uncertainty_enabled": uncertainty_enabled,
            "confidence_enabled": confidence_enabled,
            "backfill_uncertainty_only": backfill_uncertainty_only,
            "uncertainty_metrics": list(uncertainty_export_metrics),
            "uncertainty_output_paths": existing_uncertainty_metric_paths,
            "confidence_output_path": (
                None
                if not confidence_enabled or not Path(uncertainty_output_paths["confidence_path"]).exists()
                else str(uncertainty_output_paths["confidence_path"])
            ),
            "uncertainty_summary_path": (
                None
                if not uncertainty_enabled or not Path(uncertainty_output_paths["uncertainty_summary_path"]).exists()
                else str(uncertainty_output_paths["uncertainty_summary_path"])
            ),
            "confidence_summary_path": (
                None
                if not confidence_enabled or not Path(uncertainty_output_paths["confidence_summary_path"]).exists()
                else str(uncertainty_output_paths["confidence_summary_path"])
            ),
            "download_only": download_only,
            "resume_mode": manifest_mode,
            **device_meta,
        }
        save_json(
            runtime_summary_path,
            {
                "provider_name": provider_name,
                "source_script": source_script,
                "target": target["result_fields"],
                "result": result,
                "manifest_path": str(manifest_path),
                "failed_tiles_path": str(failed_tiles_path),
                "dataset_cache_key": dataset_cache_key,
                "static_dataset_cache_key": static_dataset_cache_key,
                "dynamic_dataset_cache_key": dynamic_dataset_cache_key,
                "run_config_hash": run_config_hash,
                "model_bundle_hash": model_bundle_identity["bundle_hash"],
                "uncertainty": confidence_summary,
                "runtime_summary": runtime_summary,
                "grid_summary": grid_summary,
                "job": job,
            },
        )
        return result

    dst = None
    uncertainty_datasets: dict[str, Any] = {}
    uncertainty_metric_paths: dict[str, str] = {}
    if not download_only:
        if not backfill_uncertainty_only:
            dst = ensure_output_dataset(
                output_path=output_path,
                profile=profile,
                tags={**target["output_tags"], "source_script": source_script},
                overwrite=bool(job.get("overwrite", False)),
                resume_mode=bool(job.get("resume", True)),
            )
        if uncertainty_enabled:
            uncertainty_datasets, uncertainty_metric_paths = prepare_uncertainty_output_datasets(
                output_paths=uncertainty_output_paths,
                profile=profile,
                output_tags={**target["output_tags"], "source_script": source_script},
                overwrite=bool(job.get("overwrite", False)),
                resume_mode=bool(job.get("resume", True)),
            )
    valid_pixels = 0
    pred_min = float("inf")
    pred_max = float("-inf")
    pred_sum = 0.0
    completed_since_save = 0
    failed_tiles: list[dict[str, Any]] = []
    pending_futures: dict[str, TilePreparationFuture] = {}
    max_pending = max(1, int(job["prefetch_queue_size"]))
    try:
        with TerminalProgressDisplay(
            label=str(target.get("label") or target["target_id"]),
            manifest=manifest,
            enabled=bool(job.get("show_progress", True)),
        ) as progress_display:
            with ThreadPoolExecutor(max_workers=max(1, int(job["download_workers"]))) as download_pool:
                with (nullcontext() if download_only or dst is None else dst):
                    idx = 0
                    while idx < len(manifest["tile_order"]):
                        fill_prefetch_window(
                            executor=download_pool,
                            pending_futures=pending_futures,
                            manifest=manifest,
                            manifest_path=manifest_path,
                            start_index=idx,
                            max_pending=max_pending,
                            cache_root=cache_root,
                            static_dataset_cache_key=static_dataset_cache_key,
                            dynamic_dataset_cache_key=dynamic_dataset_cache_key,
                            target=target,
                            feature_cols=feature_cols,
                            cache_policy=job["cache_policy"],
                            download_retry=int(job["download_retry"]),
                        )
                        progress_display.update_from_manifest(manifest)

                        tile_key = manifest["tile_order"][idx]
                        tile_state = manifest["tiles"][tile_key]
                        if tile_state["status"] in {"done", "fallback_expanded"}:
                            pending_future = pending_futures.pop(tile_key, None)
                            if pending_future is not None and not pending_future.done():
                                pending_future.cancel()
                            progress_display.update_from_manifest(manifest)
                            idx += 1
                            continue

                        tile = dict(tile_state["tile"])
                        tile_state["attempt_count"] = int(tile_state.get("attempt_count", 0)) + 1
                        static_cache_path = build_static_cache_path(cache_root, static_dataset_cache_key, tile)
                        dynamic_cache_path = build_dynamic_cache_path(cache_root, dynamic_dataset_cache_key, tile)
                        static_cache_path_str = str(static_cache_path) if job["cache_policy"] == "feature_tiles" else None
                        dynamic_cache_path_str = str(dynamic_cache_path) if job["cache_policy"] == "feature_tiles" else None
                        future = pending_futures.pop(tile_key, None)
                        if future is None:
                            update_tile_status(
                                manifest,
                                tile_key,
                                "downloading",
                                cache_path=None,
                                static_cache_path=static_cache_path_str,
                                dynamic_cache_path=dynamic_cache_path_str,
                            )
                            save_progress_manifest(manifest_path, manifest)
                            progress_display.update_from_manifest(manifest)
                            future = submit_tile_prepare_task(
                                executor=download_pool,
                                target=target,
                                tile=tile,
                                feature_cols=feature_cols,
                                static_cache_path=static_cache_path,
                                dynamic_cache_path=dynamic_cache_path,
                                cache_policy=job["cache_policy"],
                                download_retry=int(job["download_retry"]),
                            )
                        try:
                            feature_array, aoi_mask, cache_meta = future.result()
                            update_tile_status(
                                manifest,
                                tile_key,
                                "downloaded",
                                cache_path=None,
                                static_cache_path=static_cache_path_str,
                                dynamic_cache_path=dynamic_cache_path_str,
                                stats=dict(cache_meta),
                            )
                            save_progress_manifest(manifest_path, manifest)
                            progress_display.update_from_manifest(manifest)
                        except Exception as exc:
                            if int(job["tile_size_fallback"]) < int(tile["tile_size"]):
                                fallback_tiles = subdivide_tile(tile, int(job["tile_size_fallback"]))
                                insert_fallback_tiles(
                                    manifest,
                                    parent_tile_id=tile_key,
                                    fallback_tiles=fallback_tiles,
                                    insert_after_index=idx,
                                )
                                save_progress_manifest(manifest_path, manifest)
                                progress_display.update_from_manifest(manifest, force=True)
                                idx += 1
                                continue
                            error_message = str(exc)
                            mark_tile_failed(manifest, tile_key, error_message)
                            failed_tiles.append({"tile_id": tile_key, "error": error_message, "tile": tile})
                            save_progress_manifest(manifest_path, manifest)
                            progress_display.update_from_manifest(manifest)
                            save_json(failed_tiles_path, failed_tiles)
                            raise RuntimeError(f"Tile failed after retry budget: {tile_key}; error={error_message}") from exc

                        if download_only:
                            update_tile_status(
                                manifest,
                                tile_key,
                                "done",
                                cache_path=None,
                                static_cache_path=static_cache_path_str,
                                dynamic_cache_path=dynamic_cache_path_str,
                                stats=dict(cache_meta),
                            )
                            save_progress_manifest(manifest_path, manifest)
                            progress_display.update_from_manifest(manifest)
                            completed_since_save += 1
                            if completed_since_save >= int(job["write_progress_every_n_tiles"]):
                                save_progress_manifest(manifest_path, manifest)
                                progress_display.update_from_manifest(manifest)
                                completed_since_save = 0
                            sleep_seconds = float(job.get("sleep_seconds_between_tiles", 0.0) or 0.0)
                            if sleep_seconds > 0:
                                time.sleep(sleep_seconds)
                            idx += 1
                            continue

                        update_tile_status(
                            manifest,
                            tile_key,
                            "inferring",
                            cache_path=None,
                            static_cache_path=static_cache_path_str,
                            dynamic_cache_path=dynamic_cache_path_str,
                            stats=dict(cache_meta),
                        )
                        save_progress_manifest(manifest_path, manifest)
                        progress_display.update_from_manifest(manifest)
                        prediction_blocks, stats = predict_probability_block(
                            feature_array,
                            aoi_mask,
                            feature_cols=feature_cols,
                            model_members=model_members,
                            device=device,
                            inference_batch_size=int(runtime_cfg["inference_batch_size"]),
                            nodata_value=float(profile["nodata"]),
                            uncertainty_metrics=uncertainty_export_metrics,
                            uncertainty_epsilon=float(job.get("uncertainty_epsilon", 1e-6) or 1e-6),
                        )
                        window = window_factory(
                            int(tile["col_off"]),
                            int(tile["row_off"]),
                            int(tile["width"]),
                            int(tile["height"]),
                        )
                        if dst is not None:
                            dst.write(prediction_blocks["mean"].astype(np.float32, copy=False), 1, window=window)
                        for metric_name, dataset in uncertainty_datasets.items():
                            dataset.write(prediction_blocks[metric_name].astype(np.float32, copy=False), 1, window=window)

                        final_stats = {
                            **stats,
                            **cache_meta,
                        }
                        update_tile_status(
                            manifest,
                            tile_key,
                            "done",
                            cache_path=None,
                            static_cache_path=static_cache_path_str,
                            dynamic_cache_path=dynamic_cache_path_str,
                            stats=final_stats,
                        )
                        save_progress_manifest(manifest_path, manifest)
                        progress_display.update_from_manifest(manifest)
                        if stats["valid_pixels"] > 0:
                            valid_pixels += int(stats["valid_pixels"])
                            pred_min = min(pred_min, float(stats["pred_min"]))
                            pred_max = max(pred_max, float(stats["pred_max"]))
                            pred_sum += float(stats["pred_mean"]) * int(stats["valid_pixels"])
                        completed_since_save += 1
                        if completed_since_save >= int(job["write_progress_every_n_tiles"]):
                            save_progress_manifest(manifest_path, manifest)
                            progress_display.update_from_manifest(manifest)
                            completed_since_save = 0
                        sleep_seconds = float(job.get("sleep_seconds_between_tiles", 0.0) or 0.0)
                        if sleep_seconds > 0:
                            time.sleep(sleep_seconds)
                        idx += 1
    finally:
        for pending_future in pending_futures.values():
            if not pending_future.done():
                pending_future.cancel()
        close_output_datasets(uncertainty_datasets)
        save_progress_manifest(manifest_path, manifest)
        save_json(failed_tiles_path, failed_tiles)

    confidence_summary = None
    if confidence_enabled and all_uncertainty_metric_outputs_exist(uncertainty_output_paths):
        confidence_summary = build_confidence_raster(
            mean_path=output_path,
            std_path=uncertainty_output_paths["metric_paths"]["std"],
            confidence_path=uncertainty_output_paths["confidence_path"],
            confidence_summary_path=uncertainty_output_paths["confidence_summary_path"],
            manifest=manifest,
            class_count=max(2, int(job.get("confidence_class_count", 5) or 5)),
            nodata_value=int(job.get("confidence_nodata_value", -9999) or -9999),
            sample_per_tile=max(1, int(job.get("confidence_sampling_per_tile", 4096) or 4096)),
            jenks_max_samples=max(1, int(job.get("confidence_jenks_max_samples", 10000) or 10000)),
            sample_seed=int(job.get("confidence_sample_seed", 42) or 42),
            overwrite=bool(job.get("overwrite", False)),
            resume_mode=bool(job.get("resume", True)),
        )

    if uncertainty_enabled:
        save_json(
            uncertainty_output_paths["uncertainty_summary_path"],
            {
                "uncertainty_enabled": True,
                "metric_output_paths": uncertainty_metric_paths,
                "uncertainty_metrics": list(uncertainty_export_metrics),
                "confidence_enabled": confidence_enabled,
                "confidence_output_path": (
                    None
                    if not confidence_enabled or not Path(uncertainty_output_paths["confidence_path"]).exists()
                    else str(uncertainty_output_paths["confidence_path"])
                ),
                "confidence_summary_path": (
                    None
                    if not confidence_enabled or not Path(uncertainty_output_paths["confidence_summary_path"]).exists()
                    else str(uncertainty_output_paths["confidence_summary_path"])
                ),
                "uncertainty_epsilon": float(job.get("uncertainty_epsilon", 1e-6) or 1e-6),
                "confidence_class_count": int(job.get("confidence_class_count", 5) or 5),
                "confidence_sampling_per_tile": int(job.get("confidence_sampling_per_tile", 4096) or 4096),
                "confidence_jenks_max_samples": int(job.get("confidence_jenks_max_samples", 10000) or 10000),
                "backfill_uncertainty_only": backfill_uncertainty_only,
            },
        )

    result = {
        **target["result_fields"],
        "output_path": None if download_only else str(output_path),
        "grid_width": int(grid_summary["width"]),
        "grid_height": int(grid_summary["height"]),
        "tile_count": len(manifest.get("tile_order", [])),
        "valid_pixels": None if download_only else int(valid_pixels),
        "pred_min": None if download_only or valid_pixels == 0 else float(pred_min),
        "pred_max": None if download_only or valid_pixels == 0 else float(pred_max),
        "pred_mean": None if download_only or valid_pixels == 0 else float(pred_sum / valid_pixels),
        "ensemble_members": len(model_members),
        "model_bundle_hash": model_bundle_identity["bundle_hash"],
        "uncertainty_enabled": uncertainty_enabled,
        "confidence_enabled": confidence_enabled,
        "backfill_uncertainty_only": backfill_uncertainty_only,
        "uncertainty_metrics": list(uncertainty_export_metrics),
        "uncertainty_output_paths": uncertainty_metric_paths,
        "confidence_output_path": (
            None
            if not confidence_enabled or not Path(uncertainty_output_paths["confidence_path"]).exists()
            else str(uncertainty_output_paths["confidence_path"])
        ),
        "uncertainty_summary_path": (
            None
            if not uncertainty_enabled or not Path(uncertainty_output_paths["uncertainty_summary_path"]).exists()
            else str(uncertainty_output_paths["uncertainty_summary_path"])
        ),
        "confidence_summary_path": (
            None
            if not confidence_enabled or not Path(uncertainty_output_paths["confidence_summary_path"]).exists()
            else str(uncertainty_output_paths["confidence_summary_path"])
        ),
        "download_only": download_only,
        "resume_mode": manifest_mode,
        "manifest_path": str(manifest_path),
        "failed_tiles_path": str(failed_tiles_path),
        "runtime_summary_path": str(runtime_summary_path),
        **device_meta,
    }
    save_json(
        runtime_summary_path,
        {
            "provider_name": provider_name,
            "source_script": source_script,
            "target": target["result_fields"],
            "result": result,
            "manifest_path": str(manifest_path),
            "failed_tiles_path": str(failed_tiles_path),
            "dataset_cache_key": dataset_cache_key,
            "static_dataset_cache_key": static_dataset_cache_key,
            "dynamic_dataset_cache_key": dynamic_dataset_cache_key,
            "run_config_hash": run_config_hash,
            "model_bundle_hash": model_bundle_identity["bundle_hash"],
            "static_cache_dir": str(static_cache_dir),
            "dynamic_cache_dir": str(dynamic_cache_dir),
            "uncertainty": confidence_summary,
            "runtime_summary": runtime_summary,
            "grid_summary": grid_summary,
            "job": {
                **job,
                "train_config_path": str(job["train_config_path"]),
                "cache_root": str(job["cache_root"]),
            },
        },
    )
    return result


def run_projection_targets(
    *,
    provider_name: str,
    source_script: str,
    job: dict[str, Any],
    cfg: dict[str, Any],
    feature_cols: list[str],
    runtime_summary: dict[str, Any],
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    import_runtime_dependencies()
    region = ee_ctx.get_runtime_context()["region"]
    download_only = bool(job.get("download_only", False))
    job = normalize_runtime_job(job if download_only else finalize_train_config(job))
    if not download_only:
        assert_training_schema_matches_features(cfg, feature_cols)

    runtime_cfg = dict(cfg)
    runtime_cfg["device"] = str(job.get("device") or "cpu")
    if download_only:
        device = None
        device_meta = {
            "device_requested": str(runtime_cfg["device"]),
            "device_resolved": None,
        }
        runtime_cfg["inference_batch_size"] = None
        model_members = []
        model_bundle_identity = {
            "schema_version": MODEL_BUNDLE_IDENTITY_VERSION,
            "mode": "download_only",
            "bundle_hash": "download_only",
        }
    else:
        from cvrank.shared import get_device_metadata, resolve_torch_device

        device = resolve_torch_device(runtime_cfg)
        device_meta = get_device_metadata(runtime_cfg, resolved=device)
        runtime_cfg["inference_batch_size"] = resolve_inference_batch_size(job, device_meta)
        members_meta, fold_manifest_path = resolve_model_member_artifacts(
            cfg,
        )
        model_bundle_identity = build_model_bundle_identity(
            train_config_path=job["train_config_path"],
            model_backend=job.get("model_backend"),
            members_meta=members_meta,
            fold_manifest_path=fold_manifest_path,
        )
        model_members = load_model_scaler_members(
            cfg,
            feature_cols,
            device,

            members_meta=members_meta,
        )
        verify_model_bundle_unchanged(
            model_bundle_identity,
            train_config_path=job["train_config_path"],
            model_backend=job.get("model_backend"),
            members_meta=members_meta,
            fold_manifest_path=fold_manifest_path,
        )

    if download_only:
        inference_batch_size_summary = None
    else:
        resolved_inference_batch_size = runtime_cfg.get("inference_batch_size")
        if resolved_inference_batch_size is None:
            raise RuntimeError("Inference mode requires a resolved inference_batch_size.")
        inference_batch_size_summary = int(resolved_inference_batch_size)

    raster_output_dir = resolve_raster_output_dir(job, PREDICTION_ROOT / "outputs" / "tif")
    json_output_dir = resolve_json_output_dir(job, PREDICTION_ROOT / "outputs" / "json")
    raster_output_dir.mkdir(parents=True, exist_ok=True)
    json_output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = resolve_path(job.get("cache_root", PREDICTION_ROOT / "cache"))
    cache_root.mkdir(parents=True, exist_ok=True)

    xmin, _ymin, _xmax, ymax, width, height = get_projected_grid(region)
    transform = from_origin(xmin, ymax, ee_ctx.TARGET_SCALE, ee_ctx.TARGET_SCALE)
    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "float32",
        "crs": crs_factory.from_wkt(ee_ctx.TARGET_PROJECTION_WKT),
        "transform": transform,
        "nodata": float(job.get("nodata_value", IN_MEMORY_NODATA_VALUE)),
        "compress": "lzw",
    }
    grid_summary = {
        "xmin": float(xmin),
        "ymax": float(ymax),
        "width": int(width),
        "height": int(height),
        "target_scale": float(ee_ctx.TARGET_SCALE),
    }

    active_tile_info, active_tile_cache_path = load_or_create_active_tile_lookup_cache(
        cache_root=cache_root,
        provider_name=provider_name,
        region_geom=region,
        xmin=xmin,
        ymax=ymax,
        width=width,
        height=height,
        tile_size=int(job["tile_size_primary"]),
        retry=int(job["download_retry"]),
    )
    primary_tiles_all = list(iter_projected_tiles(xmin, ymax, width, height, int(job["tile_size_primary"])))
    active_lookup = active_tile_info["active_lookup"]
    primary_tiles = [
        tile
        for tile in primary_tiles_all
        if (int(tile["row_idx"]), int(tile["col_idx"])) in active_lookup
    ]
    max_tiles = job.get("max_tiles")
    if max_tiles not in (None, ""):
        primary_tiles = primary_tiles[: int(max_tiles)]

    print(
        json.dumps(
            {
                "provider_name": provider_name,
                "targets": [target["target_id"] for target in targets],
                "feature_count": len(feature_cols),
                "feature_cols": feature_cols,
                "tile_size_primary": int(job["tile_size_primary"]),
                "tile_size_fallback": int(job["tile_size_fallback"]),
                "tile_count_raw": len(primary_tiles_all),
                "tile_count_active": active_tile_info["active_tile_count"],
                "tile_count": len(primary_tiles),
                "download_retry": int(job["download_retry"]),
                "cache_policy": job["cache_policy"],
                "download_only": download_only,
                "cache_format_version": CACHE_FORMAT_VERSION,
                "cache_root": str(cache_root),
                "resume": bool(job["resume"]),
                "active_tile_cache_path": str(active_tile_cache_path),
                "output_dir": str(raster_output_dir),
                "raster_output_dir": str(raster_output_dir),
                "json_output_dir": str(json_output_dir),
                "ensemble_members": len(model_members),
                "write_uncertainty_outputs": should_write_uncertainty_outputs(job, model_members),
                "write_confidence_outputs": should_write_confidence_outputs(job, model_members),
                "inference_batch_size": inference_batch_size_summary,
                "download_workers": int(job["download_workers"]),
                "prefetch_queue_size": int(job["prefetch_queue_size"]),
                "show_progress": bool(job.get("show_progress", True)),
                "model_backend": normalize_model_backend(job.get("model_backend")),
                "model_bundle_hash": model_bundle_identity["bundle_hash"],
                **device_meta,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    results: list[dict[str, Any]] = []
    total_targets = len(targets)
    for target_index, target in enumerate(targets, start=1):
        print(
            json.dumps(
                {
                    "event": "target_start",
                    "provider_name": provider_name,
                    "target_index": int(target_index),
                    "target_count": int(total_targets),
                    "target_id": target["target_id"],
                    "label": target.get("label"),
                    "output_name": target["output_name"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        result = run_projection_target(
            provider_name=provider_name,
            source_script=source_script,
            job=job,
            cfg=cfg,
            runtime_cfg=runtime_cfg,
            target=target,
            feature_cols=feature_cols,
            primary_tiles=primary_tiles,
            profile=profile,
            raster_output_dir=raster_output_dir,
            json_output_dir=json_output_dir,
            cache_root=cache_root,
            device=device,
            device_meta=device_meta,
            model_members=model_members,
            model_bundle_identity=model_bundle_identity,

            runtime_summary=runtime_summary,
            grid_summary=grid_summary,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "event": "target_complete",
                    "provider_name": provider_name,
                    "target_index": int(target_index),
                    "target_count": int(total_targets),
                    "target_id": target["target_id"],
                    "label": target.get("label"),
                    "output_path": result["output_path"],
                    "tile_count": result.get("tile_count"),
                    "valid_pixels": result.get("valid_pixels"),
                    "resume_mode": result.get("resume_mode"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return results
