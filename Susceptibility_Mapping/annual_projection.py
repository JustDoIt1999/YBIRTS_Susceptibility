from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

# These imports intentionally follow the workspace-path bootstrap so this file can
# be run both as a module and directly as a script.
from Susceptibility_Mapping import gee_context as ee_ctx  # noqa: E402
from Susceptibility_Mapping import projection_engine as engine  # noqa: E402


PREDICTION_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PREDICTION_ROOT / "configs"
DEFAULT_JOB_CONFIG_PATH = CONFIG_ROOT / "cvrank" / "annual_projection.yaml"
SUPPORTED_PREDICTION_YEARS = tuple(int(year) for year in ee_ctx.PREDICTION_ASSET_YEARS)
PREDICTION_INPUT_SOURCE = "asset"
DOWNLOAD_ONLY_OUTPUT_NAME_TEMPLATE = "annual_cache_{year}.tif"

DEFAULT_JOB = {
    "model_backend": engine.DEFAULT_MODEL_BACKEND,
    "train_config_path": None,
    "batch_size": None,
    "rank_loss_mode": None,
    "train_output_suffix": None,
    "route": None,
    "years": list(SUPPORTED_PREDICTION_YEARS),
    "output_dir": PREDICTION_ROOT / "outputs" / "tif" / "annual",
    "raster_output_dir": PREDICTION_ROOT / "outputs" / "tif" / "annual",
    "json_output_dir": PREDICTION_ROOT / "outputs" / "json" / "annual",
    "output_name_template": "{model_backend}_{route}_bs_{batch_size}_RTS_Susceptibility_{year}.tif",
    "overwrite": False,
    "device": "cpu",
    "tile_size_primary": 128,
    "tile_size_fallback": 64,
    "tile_size": None,
    "inference_batch_size": None,
    "ee_endpoint_mode": ee_ctx.DEFAULT_EE_ENDPOINT_MODE,
    "cache_root": PREDICTION_ROOT / "cache",
    "enable_cache": False,
    "resume": True,
    "cache_policy": "none",
    "download_workers": 2,
    "prefetch_queue_size": 2,
    "show_progress": True,
    "write_progress_every_n_tiles": 10,
    "max_tiles": None,
    "download_retry": 4,
    "sleep_seconds_between_tiles": 0.0,
    "nodata_value": engine.IN_MEMORY_NODATA_VALUE,
    "write_uncertainty_outputs": False,
    "uncertainty_metrics": list(engine.DEFAULT_UNCERTAINTY_EXPORT_METRICS),
    "write_confidence_outputs": False,
    "uncertainty_output_subdir": engine.DEFAULT_UNCERTAINTY_OUTPUT_SUBDIR,
    "uncertainty_epsilon": 1e-6,
    "confidence_class_count": 5,
    "confidence_sampling_per_tile": 4096,
    "confidence_jenks_max_samples": 10000,
    "confidence_sample_seed": 42,
    "confidence_nodata_value": -9999,
    "backfill_uncertainty_only": False,
    "download_only": False,
    "run": False,
}

ANNUAL_JOB_CONFIG_PATH_KEYS = {
    "train_config_path",
    "output_dir",
    "raster_output_dir",
    "json_output_dir",
    "cache_root",
}
ANNUAL_JOB_CONFIG_ALLOWED_KEYS = {
    "model_backend",
    "train_config_path",
    "batch_size",
    "rank_loss_mode",
    "train_output_suffix",
    "route",
    "years",
    "output_dir",
    "raster_output_dir",
    "json_output_dir",
    "output_name_template",
    "overwrite",
    "device",
    "ee_endpoint_mode",
    "tile_size_primary",
    "tile_size_fallback",
    "tile_size",
    "inference_batch_size",
    "cache_root",
    "enable_cache",
    "resume",
    "cache_policy",
    "download_workers",
    "prefetch_queue_size",
    "show_progress",
    "write_progress_every_n_tiles",
    "max_tiles",
    "download_retry",
    "sleep_seconds_between_tiles",
    "nodata_value",
    "write_uncertainty_outputs",
    "uncertainty_metrics",
    "write_confidence_outputs",
    "uncertainty_output_subdir",
    "uncertainty_epsilon",
    "confidence_class_count",
    "confidence_sampling_per_tile",
    "confidence_jenks_max_samples",
    "confidence_sample_seed",
    "confidence_nodata_value",
    "backfill_uncertainty_only",
    "download_only",
    "run",
}


def _apply_default_output_layout(
    job: dict[str, Any],
    *,
    raster_dir_explicit: bool,
    json_dir_explicit: bool,
) -> dict[str, Any]:
    updated = dict(job)
    if not raster_dir_explicit:
        base_raster_dir = updated.get("raster_output_dir") or updated.get("output_dir") or DEFAULT_JOB["raster_output_dir"]
        resolved_raster_dir = engine.resolve_prediction_output_dir(base_raster_dir)
        updated["raster_output_dir"] = resolved_raster_dir
        updated["output_dir"] = resolved_raster_dir
    if not json_dir_explicit:
        base_json_dir = updated.get("json_output_dir") or DEFAULT_JOB["json_output_dir"]
        updated["json_output_dir"] = engine.resolve_prediction_output_dir(base_json_dir)
    return updated


def build_prediction_stack(route: str, year: int):
    route = str(route or "").strip().lower()
    if route not in engine.RANK_ROUTES:
        raise ValueError(f"Unsupported route for annual raster prediction: {route}")
    selected_year = int(year)
    if selected_year not in SUPPORTED_PREDICTION_YEARS:
        raise ValueError(
            f"Prediction asset mode only supports years {list(SUPPORTED_PREDICTION_YEARS)}; got={selected_year}"
        )
    return ee_ctx.build_prediction_feature_year_stack_from_assets(selected_year).select(ee_ctx.CURRENT_RANK_FEATURE_COLS)


def resolve_annual_model_identity(job: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, int, str]:
    route = str(job.get("route") or cfg.get("route") or "").strip().lower()
    selected_batch_size_raw = job.get("batch_size")
    if selected_batch_size_raw in (None, ""):
        selected_batch_size_raw = cfg.get("batch_size")
    selected_batch_size = engine.normalize_training_batch_size(selected_batch_size_raw)
    selected_rank_loss_mode = engine.normalize_rank_loss_mode(job.get("rank_loss_mode") or cfg.get("rank_loss_mode"))
    return route, selected_batch_size, selected_rank_loss_mode


def resolve_annual_download_only_target_identity(job: dict[str, Any]) -> tuple[str, str]:
    return "annual_cache", DOWNLOAD_ONLY_OUTPUT_NAME_TEMPLATE


def partition_annual_feature_cols(feature_cols: list[str]) -> tuple[list[str], list[str]]:
    """Split the resolved model inputs into ordered static and dynamic bands."""
    normalized_feature_cols = [str(column) for column in feature_cols]
    if not normalized_feature_cols:
        raise ValueError("feature_cols must contain at least one annual projection feature")
    if len(normalized_feature_cols) != len(set(normalized_feature_cols)):
        raise ValueError("feature_cols must not contain duplicate annual projection features")
    static_feature_names = set(ee_ctx.CURRENT_STATIC_EXPORT_FEATURE_COLS)
    static_feature_cols = [column for column in normalized_feature_cols if column in static_feature_names]
    dynamic_feature_cols = [column for column in normalized_feature_cols if column not in static_feature_names]
    return static_feature_cols, dynamic_feature_cols


def build_job_preview(job: dict[str, Any]) -> dict[str, Any]:
    download_only = bool(job.get("download_only", False))
    job = engine.normalize_runtime_job(job if download_only else engine.finalize_train_config(job))
    cfg = {} if download_only else engine.load_training_config(job["train_config_path"], job.get("model_backend"))
    if download_only:
        route = None
        selected_batch_size = None
        selected_rank_loss_mode = None
        feature_cols_preview = list(ee_ctx.CURRENT_RANK_FEATURE_COLS)
        _, output_template = resolve_annual_download_only_target_identity(job)
    else:
        route, selected_batch_size, selected_rank_loss_mode = resolve_annual_model_identity(job, cfg)
        feature_cols_preview = engine.resolve_training_feature_cols(cfg, route)
        output_template = str(job.get("output_name_template", DEFAULT_JOB["output_name_template"]))
    output_names_preview = [
        output_template.format(
            route=route,
            batch_size=selected_batch_size,
            rank_loss_mode=selected_rank_loss_mode,
            year=int(year),
            model_backend=str(job.get("model_backend")),
            module=str(job.get("model_backend")),
        )
        for year in job.get("years", list(SUPPORTED_PREDICTION_YEARS))
    ]
    raster_output_dir = engine.resolve_raster_output_dir(job, DEFAULT_JOB["raster_output_dir"])
    json_output_dir = engine.resolve_json_output_dir(job, DEFAULT_JOB["json_output_dir"])
    preview = {
        "run_now": bool(job.get("run", False)),
        "job_config_path": str(job.get("_job_config_path")) if job.get("_job_config_path") else None,
        "route_preview": route or None,
        "model_backend": str(job.get("model_backend")),
        "batch_size_preview": selected_batch_size,
        "rank_loss_mode_preview": selected_rank_loss_mode,
        "train_output_suffix": job.get("train_output_suffix"),
        "selection_policy": None if download_only else cfg.get("selection_policy"),
        "train_config_source": str(job.get("train_config_source")),
        "targets_preview": [int(y) for y in job.get("years", list(SUPPORTED_PREDICTION_YEARS))],
        "input_source": PREDICTION_INPUT_SOURCE,
        "supported_prediction_years": list(SUPPORTED_PREDICTION_YEARS),
        "feature_count_preview": len(feature_cols_preview),
        "train_config_path": None
        if not job.get("train_config_path")
        else str(engine.resolve_path(job["train_config_path"])),
        "output_dir": str(raster_output_dir),
        "raster_output_dir": str(raster_output_dir),
        "json_output_dir": str(json_output_dir),
        "output_names_preview": output_names_preview,
        "cache_root": str(engine.resolve_path(job["cache_root"])),
        "enable_cache": bool(job["enable_cache"]),
        "tile_size_primary": int(job["tile_size_primary"]),
        "tile_size_fallback": int(job["tile_size_fallback"]),
        "ee_endpoint_mode": str(job["ee_endpoint_mode"]),
        "resume": bool(job["resume"]),
        "cache_policy": str(job["cache_policy"]),
        "download_workers": int(job["download_workers"]),
        "prefetch_queue_size": int(job["prefetch_queue_size"]),
        "show_progress": bool(job["show_progress"]),
        "write_progress_every_n_tiles": int(job["write_progress_every_n_tiles"]),
        "download_only": bool(job.get("download_only", False)),
        "write_uncertainty_outputs": bool(job.get("write_uncertainty_outputs", False)),
        "uncertainty_metrics": list(engine.resolve_uncertainty_export_metrics(job))
        if bool(job.get("write_uncertainty_outputs", False))
        else [],
        "write_confidence_outputs": bool(job.get("write_confidence_outputs", False)),
        "uncertainty_output_subdir": str(job.get("uncertainty_output_subdir") or engine.DEFAULT_UNCERTAINTY_OUTPUT_SUBDIR),
        "confidence_class_count": int(job.get("confidence_class_count", 5) or 5),
        "confidence_jenks_max_samples": int(job.get("confidence_jenks_max_samples", 10000) or 10000),
        "backfill_uncertainty_only": bool(job.get("backfill_uncertainty_only", False)),
        "note": (
            "The annual pipeline reads the official annual GEE assets. "
            "Specifically, 2016-2020 reads from factor_library_v2/dynamic_annual, "
            "and 2021-2022 reads from prediction_factor_library_v1/dynamic_annual. "
            "When download_only=True, it only downloads the annual tile cache without running inference, generating TIFs, or requiring model weights."
        ),
        "ee_runtime": ee_ctx.build_runtime_summary(ee_endpoint_mode=job["ee_endpoint_mode"]),
    }
    return preview


def build_annual_targets(
    job: dict[str, Any],
    route: str | None,
    selected_batch_size: int | None,
    selected_rank_loss_mode: str | None,
    *,
    feature_cols: list[str],
) -> list[dict[str, Any]]:
    runtime_ctx = ee_ctx.get_runtime_context(ee_endpoint_mode=job["ee_endpoint_mode"])
    region = runtime_ctx["region"]
    project_id = ee_ctx.runtime_project_id()
    glacier_fc = ee_ctx.ee.FeatureCollection(f"projects/{project_id}/assets/glacier")
    lake_fc = ee_ctx.ee.FeatureCollection(f"projects/{project_id}/assets/lake")

    exclude_mask = ee_ctx.ee.Image.constant(1).paint(glacier_fc, 0).paint(lake_fc, 0)

    aoi_mask_image = (
        ee_ctx.ee.Image.constant(1)
        .rename(engine.AUX_MASK_BAND)
        .clip(region)
        .unmask(0)  # pyright: ignore[reportArgumentType] -- Earth Engine accepts numeric fill values.
        .multiply(exclude_mask)
    )
    download_only = bool(job.get("download_only", False))
    if not download_only and route is None:
        raise ValueError("route is required when building annual inference targets")
    feature_cols = [str(column) for column in feature_cols]
    static_feature_cols, dynamic_feature_cols = partition_annual_feature_cols(feature_cols)
    static_stack = ee_ctx.load_static_factor_stack_from_assets().select(static_feature_cols)
    if download_only:
        download_label_prefix, output_template = resolve_annual_download_only_target_identity(job)
    else:
        download_label_prefix = str(route)
        output_template = str(job.get("output_name_template", DEFAULT_JOB["output_name_template"]))
    targets: list[dict[str, Any]] = []
    for year in [int(value) for value in job.get("years", list(SUPPORTED_PREDICTION_YEARS))]:
        dynamic_stack = ee_ctx.load_prediction_dynamic_year_stack_from_assets(year).select(dynamic_feature_cols)
        stack = static_stack.addBands(dynamic_stack).select(feature_cols)
        static_download_image = (
            static_stack.clip(region)
            .unmask(engine.IN_MEMORY_NODATA_VALUE)
            .addBands(aoi_mask_image)
            .select(static_feature_cols + [engine.AUX_MASK_BAND])
        )
        dynamic_download_image = (
            dynamic_stack.clip(region)
            .unmask(engine.IN_MEMORY_NODATA_VALUE)
            .select(dynamic_feature_cols)
        )
        download_image = (
            stack.clip(region)
            .unmask(engine.IN_MEMORY_NODATA_VALUE)
            .addBands(aoi_mask_image)
            .select(feature_cols + [engine.AUX_MASK_BAND])
        )
        targets.append(
            {
                "target_id": f"{download_label_prefix}_{year}",
                "label": f"{download_label_prefix} {year}",
                "download_image": download_image,
                "static_download_image": static_download_image,
                "dynamic_download_image": dynamic_download_image,
                "static_feature_cols": static_feature_cols,
                "dynamic_feature_cols": dynamic_feature_cols,
                "output_name": output_template.format(
                    route=route,
                    batch_size=selected_batch_size,
                    rank_loss_mode=selected_rank_loss_mode,
                    year=year,
                    model_backend=str(job.get("model_backend")),
                    module=str(job.get("model_backend")),
                ),
                "output_tags": {
                    "model_backend": str(job.get("model_backend")),
                    "route": route,
                    "batch_size": selected_batch_size,
                    "rank_loss_mode": selected_rank_loss_mode,
                    "selection_policy": None if download_only else job.get("selection_policy"),
                    "year": int(year),
                    "band_name": engine.PREDICTION_RESULT_BAND_NAME,
                    "input_source": PREDICTION_INPUT_SOURCE,
                    "download_only": download_only,
                },
                "result_fields": {
                    "route": route,
                    "batch_size": selected_batch_size,
                    "rank_loss_mode": selected_rank_loss_mode,
                    "selection_policy": None if download_only else job.get("selection_policy"),
                    "year": int(year),
                    "grid_width": None,
                    "grid_height": None,
                },
                "static_stack_signature": "shared_static_assets:v2",
                "dynamic_stack_signature": f"annual_prediction_dynamic_assets:{year}",
                "stack_signature": f"annual_prediction_assets:{year}" if download_only else f"annual_prediction_assets:{route}:{year}",
                "input_source": PREDICTION_INPUT_SOURCE,
            }
        )
    return targets


def run_annual_raster_prediction(job: dict[str, Any]) -> list[dict[str, Any]]:
    engine.import_runtime_dependencies()
    download_only = bool(job.get("download_only", False))
    job = engine.normalize_runtime_job(job if download_only else engine.finalize_train_config(job))
    cfg = {} if download_only else engine.load_training_config(job["train_config_path"], job.get("model_backend"))
    if download_only:
        route = None
        selected_batch_size = None
        selected_rank_loss_mode = None
        feature_cols = list(ee_ctx.CURRENT_RANK_FEATURE_COLS)
    else:
        route, selected_batch_size, selected_rank_loss_mode = resolve_annual_model_identity(job, cfg)
        feature_cols = engine.resolve_training_feature_cols(cfg, route)
        job["selection_policy"] = cfg.get("selection_policy")
    runtime_summary = ee_ctx.build_runtime_summary(ee_endpoint_mode=job["ee_endpoint_mode"])
    targets = build_annual_targets(
        job,
        route,
        selected_batch_size,
        selected_rank_loss_mode,
        feature_cols=feature_cols,
    )
    results = engine.run_projection_targets(
        provider_name="annual_projection",
        source_script="Susceptibility_Mapping/annual_projection.py",
        job=job,
        cfg=cfg,
        feature_cols=feature_cols,
        runtime_summary=runtime_summary,
        targets=targets,
    )
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run annual raster prediction with cache/resume aware projection engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--job-config", type=str, default=str(DEFAULT_JOB_CONFIG_PATH), help="Projection job YAML path.")
    parser.add_argument(
        "--model-backend",
        type=str,
        default=None,
        help=f"Model backend used for loading training artifacts: {sorted(engine.SUPPORTED_MODEL_BACKENDS)}.",
    )
    parser.add_argument("--train-config", type=str, default=None, help="Training config YAML path.")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size tag used to resolve train_config_path.")
    parser.add_argument(
        "--rank-loss-mode",
        type=str,
        default=None,
        help=(
            "Training rank loss mode used to resolve train_config_path. "
            "disabled selects the classification baseline; it disables ranking loss, not model inference."
        ),
    )
    parser.add_argument("--train-output-suffix", type=str, default=None, help="Optional suffix of the selected training output root.")
    parser.add_argument("--route", type=str, default=None, help="Override route.")
    parser.add_argument("--years", type=int, nargs="+", default=None, help=f"Prediction years. Supported values: {list(SUPPORTED_PREDICTION_YEARS)}")
    parser.add_argument("--tile-size-primary", type=int, default=None, help="Primary tile edge length in pixels.")
    parser.add_argument("--tile-size-fallback", type=int, default=None, help="Fallback tile edge length in pixels.")
    parser.add_argument("--tile-size", type=int, default=None, help="Legacy alias. Sets both primary and fallback tile size.")
    parser.add_argument("--inference-batch-size", type=int, default=None, help="Torch inference batch size per tile.")
    parser.add_argument("--ee-endpoint-mode", type=str, default=None, help="Earth Engine endpoint mode: auto / standard / highvolume.")
    parser.add_argument("--cache-root", type=str, default=None, help="Tile cache root directory.")
    parser.add_argument("--enable-cache", dest="enable_cache", action="store_true", default=None, help="Enable tile cache.")
    parser.add_argument("--disable-cache", dest="enable_cache", action="store_false", help="Disable tile cache.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=None, help="Resume from progress manifest.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume mode.")
    parser.add_argument("--cache-policy", type=str, default=None, help="Cache policy: none / feature_tiles.")
    parser.add_argument("--download-workers", type=int, default=None, help="Reserved worker count for tile download stage.")
    parser.add_argument("--prefetch-queue-size", type=int, default=None, help="Reserved prefetch queue size.")
    parser.add_argument("--show-progress", dest="show_progress", action="store_true", default=None, help="Show terminal progress bars during execution.")
    parser.add_argument("--hide-progress", dest="show_progress", action="store_false", help="Disable terminal progress bars.")
    parser.add_argument("--write-progress-every-n-tiles", type=int, default=None, help="Flush progress manifest every N tiles.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument("--raster-output-dir", type=str, default=None, help="GeoTIFF output directory.")
    parser.add_argument("--json-output-dir", type=str, default=None, help="JSON metadata output directory.")
    parser.add_argument("--output-name-template", type=str, default=None, help="Output name template with {route} and {year}.")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing output tif.")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu/cuda/mps/auto.")
    parser.add_argument("--max-tiles", type=int, default=None, help="Optional tile cap for smoke tests.")
    parser.add_argument("--download-retry", type=int, default=None, help="Retry count for computePixels.")
    parser.add_argument("--sleep-seconds-between-tiles", type=float, default=None, help="Optional sleep between tiles.")
    parser.add_argument("--nodata-value", type=float, default=None, help="Output nodata value.")
    parser.add_argument(
        "--write-uncertainty-outputs",
        dest="write_uncertainty_outputs",
        action="store_true",
        default=None,
        help="Write uncertainty rasters for fold ensembles; the formal default is std.",
    )
    parser.add_argument(
        "--no-write-uncertainty-outputs",
        dest="write_uncertainty_outputs",
        action="store_false",
        help="Disable uncertainty raster outputs.",
    )
    parser.add_argument(
        "--uncertainty-metrics",
        type=str,
        nargs="+",
        default=None,
        help="Subset of uncertainty rasters to export: std cv q05 q95 entropy. Default formal route exports std.",
    )
    parser.add_argument(
        "--write-confidence-outputs",
        dest="write_confidence_outputs",
        action="store_true",
        default=None,
        help="Write confidence-class raster based on mean susceptibility and std.",
    )
    parser.add_argument(
        "--no-write-confidence-outputs",
        dest="write_confidence_outputs",
        action="store_false",
        help="Disable confidence raster outputs.",
    )
    parser.add_argument("--uncertainty-output-subdir", type=str, default=None, help="Subdirectory name for uncertainty rasters and summaries.")
    parser.add_argument("--uncertainty-epsilon", type=float, default=None, help="Minimum mean threshold used when optional CV = std / mean is requested.")
    parser.add_argument("--confidence-class-count", type=int, default=None, help="Class count used for susceptibility and std cross-classification.")
    parser.add_argument("--confidence-sampling-per-tile", type=int, default=None, help="Maximum sampled pixels per tile for confidence threshold estimation.")
    parser.add_argument(
        "--confidence-jenks-max-samples",
        type=int,
        default=None,
        help="Maximum global std sample count used by Jenks natural-breaks thresholding.",
    )
    parser.add_argument("--confidence-sample-seed", type=int, default=None, help="Random seed used for confidence threshold sampling.")
    parser.add_argument("--confidence-nodata-value", type=int, default=None, help="Nodata value used by the confidence raster.")
    parser.add_argument(
        "--backfill-uncertainty-only",
        dest="backfill_uncertainty_only",
        action="store_true",
        default=None,
        help="Reuse the existing mean raster and only backfill uncertainty/confidence outputs.",
    )
    parser.add_argument(
        "--no-backfill-uncertainty-only",
        dest="backfill_uncertainty_only",
        action="store_false",
        help="Disable uncertainty-only backfill mode.",
    )
    parser.add_argument("--download-only", action="store_true", default=None, help="Only download annual static/dynamic tile caches; skip inference and GeoTIFF writing.")
    parser.add_argument("--run", action="store_true", default=None, help="Actually execute prediction. Without this flag only print the resolved job.")
    return parser


def build_job_from_args(args: argparse.Namespace) -> dict[str, Any]:
    job = dict(DEFAULT_JOB)
    job_config_path = engine.resolve_path(args.job_config)
    job.update(
        engine.load_projection_job_config(
            job_config_path,
            allowed_keys=ANNUAL_JOB_CONFIG_ALLOWED_KEYS,
            path_keys=ANNUAL_JOB_CONFIG_PATH_KEYS,
        )
    )
    if args.model_backend not in (None, ""):
        job["model_backend"] = engine.normalize_model_backend(args.model_backend)
    if args.train_config not in (None, ""):
        job["train_config_path"] = engine.resolve_path(args.train_config)
        if args.batch_size is None:
            job["batch_size"] = None
        if args.rank_loss_mode in (None, ""):
            job["rank_loss_mode"] = None
        if args.train_output_suffix in (None, ""):
            job["train_output_suffix"] = None
        if args.route in (None, ""):
            job["route"] = None
    if args.batch_size is not None:
        job["batch_size"] = int(args.batch_size)
    if args.rank_loss_mode not in (None, ""):
        job["rank_loss_mode"] = str(args.rank_loss_mode).strip().lower()
    if args.train_output_suffix not in (None, ""):
        job["train_output_suffix"] = str(args.train_output_suffix).strip()
    if args.route not in (None, ""):
        job["route"] = str(args.route).strip().lower()
    if args.years is not None:
        job["years"] = [int(year) for year in args.years]
    if args.tile_size is not None:
        job["tile_size"] = int(args.tile_size)
    if args.tile_size_primary is not None:
        job["tile_size_primary"] = int(args.tile_size_primary)
    if args.tile_size_fallback is not None:
        job["tile_size_fallback"] = int(args.tile_size_fallback)
    if args.inference_batch_size is not None:
        job["inference_batch_size"] = int(args.inference_batch_size)
    if args.ee_endpoint_mode not in (None, ""):
        job["ee_endpoint_mode"] = ee_ctx.normalize_ee_endpoint_mode(args.ee_endpoint_mode)
    if args.cache_root not in (None, ""):
        job["cache_root"] = engine.resolve_path(args.cache_root)
    if args.enable_cache is not None:
        job["enable_cache"] = bool(args.enable_cache)
    if args.resume is not None:
        job["resume"] = bool(args.resume)
    if args.cache_policy not in (None, ""):
        job["cache_policy"] = str(args.cache_policy).strip().lower()
    if args.download_workers is not None:
        job["download_workers"] = int(args.download_workers)
    if args.prefetch_queue_size is not None:
        job["prefetch_queue_size"] = int(args.prefetch_queue_size)
    if args.show_progress is not None:
        job["show_progress"] = bool(args.show_progress)
    if args.write_progress_every_n_tiles is not None:
        job["write_progress_every_n_tiles"] = int(args.write_progress_every_n_tiles)
    if args.output_dir not in (None, ""):
        job["output_dir"] = engine.resolve_path(args.output_dir)
        job["raster_output_dir"] = engine.resolve_path(args.output_dir)
    if args.raster_output_dir not in (None, ""):
        job["raster_output_dir"] = engine.resolve_path(args.raster_output_dir)
        job["output_dir"] = job["raster_output_dir"]
    if args.json_output_dir not in (None, ""):
        job["json_output_dir"] = engine.resolve_path(args.json_output_dir)
    if args.output_name_template not in (None, ""):
        job["output_name_template"] = str(args.output_name_template)
    if args.overwrite is not None:
        job["overwrite"] = bool(args.overwrite)
    if args.device not in (None, ""):
        job["device"] = str(args.device)
    if args.max_tiles is not None:
        job["max_tiles"] = int(args.max_tiles)
    if args.download_retry is not None:
        job["download_retry"] = int(args.download_retry)
    if args.sleep_seconds_between_tiles is not None:
        job["sleep_seconds_between_tiles"] = float(args.sleep_seconds_between_tiles)
    if args.nodata_value is not None:
        job["nodata_value"] = float(args.nodata_value)
    if args.write_uncertainty_outputs is not None:
        job["write_uncertainty_outputs"] = bool(args.write_uncertainty_outputs)
    if args.uncertainty_metrics is not None:
        job["uncertainty_metrics"] = [str(value).strip().lower() for value in args.uncertainty_metrics if str(value).strip()]
    if args.write_confidence_outputs is not None:
        job["write_confidence_outputs"] = bool(args.write_confidence_outputs)
    if args.uncertainty_output_subdir not in (None, ""):
        job["uncertainty_output_subdir"] = str(args.uncertainty_output_subdir)
    if args.uncertainty_epsilon is not None:
        job["uncertainty_epsilon"] = float(args.uncertainty_epsilon)
    if args.confidence_class_count is not None:
        job["confidence_class_count"] = int(args.confidence_class_count)
    if args.confidence_sampling_per_tile is not None:
        job["confidence_sampling_per_tile"] = int(args.confidence_sampling_per_tile)
    if args.confidence_jenks_max_samples is not None:
        job["confidence_jenks_max_samples"] = int(args.confidence_jenks_max_samples)
    if args.confidence_sample_seed is not None:
        job["confidence_sample_seed"] = int(args.confidence_sample_seed)
    if args.confidence_nodata_value is not None:
        job["confidence_nodata_value"] = int(args.confidence_nodata_value)
    if args.backfill_uncertainty_only is not None:
        job["backfill_uncertainty_only"] = bool(args.backfill_uncertainty_only)
    if args.download_only:
        job["download_only"] = True
        job["run"] = True
        job["enable_cache"] = True
        job["cache_policy"] = "feature_tiles"
        if args.route in (None, ""):
            job["route"] = None
        if args.batch_size is None:
            job["batch_size"] = None
        if args.rank_loss_mode in (None, ""):
            job["rank_loss_mode"] = None
        if args.train_config in (None, ""):
            job["train_config_path"] = None
    if args.run is not None:
        job["run"] = bool(args.run)
    job["model_backend"] = engine.normalize_model_backend(job.get("model_backend"))
    job["ee_endpoint_mode"] = ee_ctx.normalize_ee_endpoint_mode(job.get("ee_endpoint_mode"))
    job = _apply_default_output_layout(
        job,
        raster_dir_explicit=args.output_dir not in (None, "") or args.raster_output_dir not in (None, ""),
        json_dir_explicit=args.json_output_dir not in (None, ""),
    )
    if not bool(job.get("download_only", False)):
        job = engine.finalize_train_config(job)
    job["raster_output_dir"] = engine.resolve_raster_output_dir(job, DEFAULT_JOB["raster_output_dir"])
    job["output_dir"] = job["raster_output_dir"]
    job["json_output_dir"] = engine.resolve_json_output_dir(job, DEFAULT_JOB["json_output_dir"])
    job["cache_root"] = engine.resolve_path(job["cache_root"])
    job["years"] = [int(year) for year in job.get("years", list(SUPPORTED_PREDICTION_YEARS))]
    return job


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    job = build_job_from_args(args)

    preview = build_job_preview(job)
    print(json.dumps(preview, ensure_ascii=False, indent=2, default=str))
    if not bool(job.get("run", False)):
        return 0

    results = run_annual_raster_prediction(job)
    print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
