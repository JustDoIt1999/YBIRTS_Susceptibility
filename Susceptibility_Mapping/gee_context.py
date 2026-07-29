from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

load_dotenv()

PUBLIC_ASSET_PROJECT_ID = "summer-presence-475413-i4"
PUBLIC_ASSET_ROOT = f"projects/{PUBLIC_ASSET_PROJECT_ID}/assets"
EE_STANDARD_ENDPOINT_URL = "https://earthengine.googleapis.com"
EE_HIGH_VOLUME_ENDPOINT_URL = "https://earthengine-highvolume.googleapis.com"
SUPPORTED_EE_ENDPOINT_MODES = {"auto", "standard", "highvolume"}
DEFAULT_EE_ENDPOINT_MODE = str(os.environ.get("EE_ENDPOINT_MODE") or "auto").strip().lower() or "auto"

TARGET_SCALE = 100
TARGET_PROJECTION_WKT = (
    'PROJCS["Albers",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],'
    'PARAMETER["false_northing",0.0],PARAMETER["central_meridian",90.0],PARAMETER["standard_parallel_1",27.5],'
    'PARAMETER["standard_parallel_2",37.5],PARAMETER["latitude_of_origin",0.0],UNIT["Meter",1.0]]'
)
MAX_ERROR = 1000

CURRENT_STATIC_EXPORT_FEATURE_COLS = [
    "Slope",
    "Aspect",
    "TWI",
    "Local_Relief",
    "Soil",
    "Vegetation_type",
    "GIC",
]
CURRENT_RANK_FEATURE_COLS = [
    "Slope",
    "Aspect",
    "TWI",
    "Local_Relief",
    "Soil",
    "Vegetation_type",
    "GIC",
    "TDD",
    "SMDP",
    "Precipitation",
    "WSTmax",
    "WSRH",
    "NDVI",
    "Summer_SoilMoisture",
]
ASSET_DTYPE_BY_BAND = {
    "Slope": "float32",
    "Aspect": "int16",
    "TWI": "float32",
    "Local_Relief": "float32",
    "Soil": "float32",
    "Vegetation_type": "int16",
    "GIC": "float32",
    "TDD": "float32",
    "SMDP": "float32",
    "Precipitation": "float32",
    "WSTmax": "float32",
    "WSRH": "float32",
    "NDVI": "float32",
    "Summer_SoilMoisture": "float32",
}

STATIC_FACTOR_SPECS = [
    {"factor_name": "Slope", "band_name": "Slope"},
    {"factor_name": "Aspect", "band_name": "Aspect"},
    {"factor_name": "TWI", "band_name": "TWI"},
    {"factor_name": "Local_Relief", "band_name": "Local_Relief", "asset_name": "Local_Relief_5km"},
    {"factor_name": "Soil", "band_name": "Soil"},
    {"factor_name": "GIC", "band_name": "GIC"},
]
DYNAMIC_ANNUAL_FACTOR_SPECS = [
    {"asset_factor_name": "TDD", "asset_band_name": "TDD", "band_name": "TDD"},
    {"asset_factor_name": "SMDP", "asset_band_name": "SMDP", "band_name": "SMDP"},
    {"asset_factor_name": "Precipitation", "asset_band_name": "Precipitation", "band_name": "Precipitation"},
    {"asset_factor_name": "WSTmax", "asset_band_name": "WSTmax", "band_name": "WSTmax"},
    {"asset_factor_name": "WSRH", "asset_band_name": "WSRH", "band_name": "WSRH"},
    {"asset_factor_name": "NDVI", "asset_band_name": "NDVI", "band_name": "NDVI"},
    {
        "asset_factor_name": "Summer_SoilMoisture",
        "asset_band_name": "Summer_SoilMoisture",
        "band_name": "Summer_SoilMoisture",
    },
]
FACTOR_LIBRARY_ROOT = f"{PUBLIC_ASSET_ROOT}/environmental_factors/factor_library_v2"
FACTOR_LIBRARY_VERSION_TAG = "v2"
STATIC_FACTOR_ASSET_ROOT = f"{FACTOR_LIBRARY_ROOT}/static"
LEGACY_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT = f"{FACTOR_LIBRARY_ROOT}/dynamic_annual"
PREDICTION_FACTOR_LIBRARY_ROOT = f"{PUBLIC_ASSET_ROOT}/environmental_factors/factor_library_prediction_v1"
PREDICTION_FACTOR_LIBRARY_VERSION_TAG = "v1"
PREDICTION_SMDP_ASSET_SUFFIX = "_v1"
PREDICTION_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT = f"{PREDICTION_FACTOR_LIBRARY_ROOT}/dynamic_annual"
PREDICTION_ASSET_YEARS = [
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
]

_EE_INITIALIZED = False
_EE_ENDPOINT_MODE_REQUESTED = DEFAULT_EE_ENDPOINT_MODE
_EE_ENDPOINT_MODE_RESOLVED = "highvolume" if DEFAULT_EE_ENDPOINT_MODE == "auto" else DEFAULT_EE_ENDPOINT_MODE
_EE_ENDPOINT_URL = (
    EE_HIGH_VOLUME_ENDPOINT_URL
    if _EE_ENDPOINT_MODE_RESOLVED == "highvolume"
    else EE_STANDARD_ENDPOINT_URL
)
_RUNTIME_CONTEXT: dict[str, Any] | None = None
ee = None


def _env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def runtime_project_id() -> str:
    project_id = os.environ.get("EE_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError(
            "EE_PROJECT_ID is not set. This module reads public Earth Engine assets from "
            f"{PUBLIC_ASSET_ROOT}, but Earth Engine computations should run under your own "
            "Earth Engine-enabled Google Cloud project.\n\n"
            "Set it before running Step_03:\n"
            "  macOS/Linux: export EE_PROJECT_ID='your-project-id'\n"
            "  Windows PowerShell: $env:EE_PROJECT_ID='your-project-id'"
        )
    return project_id


def _ensure_ee_imported():
    global ee
    if ee is not None:
        return ee
    try:
        import ee as _ee
    except ImportError as exc:
        raise ImportError(
            "Missing earthengine-api (`ee`) dependency, unable to read GEE assets."
        ) from exc
    ee = _ee
    return ee


def normalize_ee_endpoint_mode(mode: str | None) -> str:
    normalized = str(mode or DEFAULT_EE_ENDPOINT_MODE).strip().lower()
    if normalized not in SUPPORTED_EE_ENDPOINT_MODES:
        raise ValueError(f"Unsupported ee_endpoint_mode: {mode}; supported={sorted(SUPPORTED_EE_ENDPOINT_MODES)}")
    return normalized


def resolve_ee_endpoint_mode(mode: str | None) -> str:
    normalized = normalize_ee_endpoint_mode(mode)
    if normalized == "auto":
        return "highvolume"
    return normalized


def ee_endpoint_url_for_mode(mode: str | None) -> str:
    resolved = resolve_ee_endpoint_mode(mode)
    if resolved == "highvolume":
        return EE_HIGH_VOLUME_ENDPOINT_URL
    return EE_STANDARD_ENDPOINT_URL


def initialize_ee(ee_endpoint_mode: str | None = None, *, allow_interactive_auth: bool | None = None) -> None:
    global _EE_INITIALIZED, _EE_ENDPOINT_MODE_REQUESTED, _EE_ENDPOINT_MODE_RESOLVED, _EE_ENDPOINT_URL, _RUNTIME_CONTEXT
    if ee_endpoint_mode is None and _EE_INITIALIZED:
        requested_mode = _EE_ENDPOINT_MODE_REQUESTED
    else:
        requested_mode = normalize_ee_endpoint_mode(ee_endpoint_mode)
    resolved_mode = resolve_ee_endpoint_mode(requested_mode)
    endpoint_url = ee_endpoint_url_for_mode(requested_mode)
    if _EE_INITIALIZED and _EE_ENDPOINT_MODE_RESOLVED == resolved_mode and _EE_ENDPOINT_URL == endpoint_url:
        return
    ee_module = _ensure_ee_imported()
    project_id = runtime_project_id()
    reset_fn = getattr(ee_module, "Reset", None)
    if callable(reset_fn):
        reset_fn()
    try:
        ee_module.Initialize(project=project_id, opt_url=endpoint_url)
    except Exception as exc:
        message = str(exc)
        if "oauth2.googleapis.com" in message or "ConnectTimeout" in message or "TransportError" in message:
            raise RuntimeError(
                "Earth Engine credentials refresh failed. The current environment cannot connect to the Google OAuth service `oauth2.googleapis.com`.\n"
                "This is typically not an issue with the model or script, but rather a network, proxy, or authentication state issue.\n"
                "Please verify that your terminal can access Google, then run `earthengine authenticate` again,\n"
                "or configure valid credentials using the `GOOGLE_APPLICATION_CREDENTIALS` environment variable."
            ) from exc
        interactive_auth_enabled = (
            _env_flag_enabled("EE_ALLOW_INTERACTIVE_AUTH")
            if allow_interactive_auth is None
            else bool(allow_interactive_auth)
        )
        if not interactive_auth_enabled:
            raise RuntimeError(
                "Earth Engine initialization failed and interactive authentication is disabled.\n"
                "Run `earthengine authenticate` in an interactive terminal, set valid credentials with "
                "`GOOGLE_APPLICATION_CREDENTIALS`, or set `EE_ALLOW_INTERACTIVE_AUTH=1` to allow this script "
                "to start an interactive authentication flow.\n"
                f"Details: {message}"
            ) from exc
        print("Earth Engine initialize failed, attempting interactive authentication...", exc)
        ee_module.Authenticate()
        ee_module.Initialize(project=project_id, opt_url=endpoint_url)
    _EE_INITIALIZED = True
    _EE_ENDPOINT_MODE_REQUESTED = requested_mode
    _EE_ENDPOINT_MODE_RESOLVED = resolved_mode
    _EE_ENDPOINT_URL = endpoint_url
    _RUNTIME_CONTEXT = None


def current_ee_runtime_config() -> dict[str, Any]:
    return {
        "ee_endpoint_mode_requested": _EE_ENDPOINT_MODE_REQUESTED,
        "ee_endpoint_mode_resolved": _EE_ENDPOINT_MODE_RESOLVED,
        "ee_endpoint_url": _EE_ENDPOINT_URL,
    }


def _is_dataset_construction_root(root: Path) -> bool:
    return (
        root.is_dir()
        and (root / "inputs").is_dir()
        and (root / "src" / "dataset_construction").is_dir()
    )


def resolve_data_root() -> Path:
    """Locate the current Dataset_Construction root without relying on the launch directory."""
    module_root = Path(__file__).resolve().parents[1] / "Dataset_Construction"
    if _is_dataset_construction_root(module_root):
        return module_root

    here = Path.cwd().resolve()
    for candidate in [here, *here.parents]:
        for root in (candidate, candidate / "Dataset_Construction"):
            if _is_dataset_construction_root(root):
                return root

    raise FileNotFoundError(
        "Could not locate the Dataset_Construction root directory. "
        "Expected to find both `inputs/` and `src/dataset_construction/`."
    )


def bbox_from_bounds_coords(coords) -> tuple[float, float, float, float]:
    ring = coords[0]
    xs = [pt[0] for pt in ring]
    ys = [pt[1] for pt in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def expected_asset_dtype(band_name: str) -> str:
    band_name = str(band_name).strip()
    if band_name not in ASSET_DTYPE_BY_BAND:
        raise KeyError(f"Undefined band dtype policy: {band_name}")
    return ASSET_DTYPE_BY_BAND[band_name]


def cast_single_band_for_asset(image: ee.Image, band_name: str) -> ee.Image:
    _ensure_ee_imported()
    band_name = str(band_name).strip()
    band_image = ee.Image(image).select([band_name])
    dtype = expected_asset_dtype(band_name)
    if dtype == "int16":
        return band_image.toInt16()
    if dtype == "float32":
        return band_image.toFloat()
    raise ValueError(f"Unsupported asset dtype: {dtype}; band={band_name}")


def cast_stack_for_asset(image: ee.Image, band_names: list[str]) -> ee.Image:
    _ensure_ee_imported()
    stack = None
    for band_name in band_names:
        typed_band = cast_single_band_for_asset(image, band_name)
        stack = typed_band if stack is None else stack.addBands(typed_band)
    return ee.Image(stack).select(band_names)


def asset_exists(asset_id: str) -> bool:
    initialize_ee()
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception as exc:
        message = str(exc).lower()
        not_found_markers = (
            "not found",
            "does not exist",
            "asset not found",
            "asset does not exist",
        )
        if any(marker in message for marker in not_found_markers):
            return False
        raise RuntimeError(f"Failed to check Earth Engine asset existence: {asset_id}") from exc


def factor_asset_root(group: str) -> str:
    normalized = str(group).strip().lower()
    if normalized == "static":
        return STATIC_FACTOR_ASSET_ROOT
    raise ValueError(f"Unsupported factor asset group: {group}")


def factor_asset_name(group: str, factor_name: str, year: int | None = None) -> str:
    normalized_group = str(group).strip().lower()
    normalized_factor = str(factor_name).strip()
    if normalized_group == "static":
        factor_spec = next(
            (item for item in STATIC_FACTOR_SPECS if item["factor_name"] == normalized_factor),
            None,
        )
        asset_name = str((factor_spec or {}).get("asset_name") or normalized_factor)
        return f"{asset_name}_{FACTOR_LIBRARY_VERSION_TAG}"
    raise ValueError(f"Unsupported factor asset group: {group}")


def factor_asset_id(group: str, factor_name: str, year: int | None = None) -> str:
    return f"{factor_asset_root(group).rstrip('/')}/{factor_asset_name(group, factor_name, year=year)}"


def load_factor_asset_image(asset_id: str, expected_band: str, *, label: str) -> ee.Image:
    initialize_ee()
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"{label} asset does not exist: {asset_id}")
    return ee.Image(asset_id).select([expected_band])


def _load_static_factor_stack_from_assets(feature_cols: list[str]) -> ee.Image:
    stack = None
    for band_name in feature_cols:
        image = load_factor_asset_image(
            factor_asset_id("static", band_name),
            band_name,
            label=f"static factor {band_name}",
        )
        stack = image if stack is None else stack.addBands(image)
    return cast_stack_for_asset(ee.Image(stack).select(feature_cols), feature_cols)


def load_static_factor_stack_from_assets() -> ee.Image:
    return _load_static_factor_stack_from_assets(list(CURRENT_STATIC_EXPORT_FEATURE_COLS))


def normalize_prediction_asset_years(years=None) -> list[int]:
    if years is None:
        return list(PREDICTION_ASSET_YEARS)
    normalized = sorted({int(year) for year in years})
    if not normalized:
        raise ValueError("Prediction asset years list cannot be empty.")
    unsupported = [year for year in normalized if year not in PREDICTION_ASSET_YEARS]
    if unsupported:
        raise ValueError(
            f"Prediction asset library only supports years {PREDICTION_ASSET_YEARS}; unsupported={unsupported}"
        )
    return normalized


def prediction_factor_asset_name(factor_name: str, year: int) -> str:
    normalized_factor = str(factor_name).strip()
    if 2016 <= int(year) <= 2020:
        return f"{normalized_factor}_{int(year)}_{FACTOR_LIBRARY_VERSION_TAG}"
    if normalized_factor == "SMDP":
        return f"{normalized_factor}_{int(year)}{PREDICTION_SMDP_ASSET_SUFFIX}"
    return f"{normalized_factor}_{int(year)}_{PREDICTION_FACTOR_LIBRARY_VERSION_TAG}"


def prediction_factor_asset_id(factor_name: str, year: int) -> str:
    selected_year = int(year)
    if 2016 <= selected_year <= 2020:
        root_dir = LEGACY_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT
    else:
        root_dir = PREDICTION_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT
    return f"{root_dir.rstrip('/')}/{prediction_factor_asset_name(factor_name, selected_year)}"


def load_prediction_dynamic_factor_from_assets(factor_name: str, year: int) -> ee.Image:
    selected_year = normalize_prediction_asset_years([year])[0]
    normalized_factor = str(factor_name).strip()
    factor_spec = next(
        (item for item in DYNAMIC_ANNUAL_FACTOR_SPECS if item["band_name"] == normalized_factor),
        None,
    )
    if factor_spec is None:
        raise ValueError(f"Unsupported prediction dynamic annual factor: {factor_name}")
    asset_sources = factor_spec.get("asset_sources") or [
        {
            "asset_factor_name": factor_spec.get("asset_factor_name") or factor_spec["band_name"],
            "asset_band_name": factor_spec.get("asset_band_name") or factor_spec["band_name"],
        }
    ]
    attempted_asset_ids: list[str] = []
    for source in asset_sources:
        asset_factor_name = str(source.get("asset_factor_name") or factor_spec["band_name"]).strip()
        asset_band_name = str(source.get("asset_band_name") or factor_spec["band_name"]).strip()
        asset_id = prediction_factor_asset_id(asset_factor_name, selected_year)
        attempted_asset_ids.append(asset_id)
        if not asset_exists(asset_id):
            continue
        image = load_factor_asset_image(
            asset_id,
            asset_band_name,
            label=f"prediction dynamic annual {asset_factor_name} {selected_year}",
        )
        if asset_band_name != factor_spec["band_name"]:
            image = image.rename(factor_spec["band_name"])
        return image
    raise FileNotFoundError(
        f"prediction dynamic annual {normalized_factor} {selected_year} asset does not exist; attempted={attempted_asset_ids}"
    )


def load_prediction_dynamic_year_stack_from_assets(year: int) -> ee.Image:
    selected_year = normalize_prediction_asset_years([year])[0]
    stack = None
    for spec in DYNAMIC_ANNUAL_FACTOR_SPECS:
        band_image = load_prediction_dynamic_factor_from_assets(spec["band_name"], selected_year)
        stack = band_image if stack is None else stack.addBands(band_image)
    band_names = [item["band_name"] for item in DYNAMIC_ANNUAL_FACTOR_SPECS]
    return ee.Image(stack).select(band_names)


def build_prediction_feature_year_stack_from_assets(year: int) -> ee.Image:
    initialize_ee()
    selected_year = normalize_prediction_asset_years([year])[0]
    raw_stack = (
        load_static_factor_stack_from_assets()
        .addBands(load_prediction_dynamic_year_stack_from_assets(selected_year))
        .select(CURRENT_RANK_FEATURE_COLS)
    )
    return cast_stack_for_asset(raw_stack, CURRENT_RANK_FEATURE_COLS)


def build_runtime_context(ee_endpoint_mode: str | None = None) -> dict[str, Any]:
    initialize_ee(ee_endpoint_mode=ee_endpoint_mode)
    repo = resolve_data_root()
    aoi_asset_id = f"{PUBLIC_ASSET_ROOT}/Qinghai-Tibet_Plateau"
    aoi = ee.FeatureCollection(aoi_asset_id)
    region = aoi.geometry()
    aoi_feature_count = int(aoi.size().getInfo())
    aoi_bbox = bbox_from_bounds_coords(region.bounds(100).coordinates().getInfo())
    return {
        "REPO": repo,
        "AOI_ASSET_ID": aoi_asset_id,
        "aoi": aoi,
        "region": region,
        "aoi_feature_count": aoi_feature_count,
        "aoi_bbox": aoi_bbox,
    }


def get_runtime_context(ee_endpoint_mode: str | None = None) -> dict[str, Any]:
    global _RUNTIME_CONTEXT
    requested_mode = normalize_ee_endpoint_mode(ee_endpoint_mode)
    resolved_mode = resolve_ee_endpoint_mode(requested_mode)
    if (
        _RUNTIME_CONTEXT is None
        or _EE_ENDPOINT_MODE_REQUESTED != requested_mode
        or _EE_ENDPOINT_MODE_RESOLVED != resolved_mode
    ):
        _RUNTIME_CONTEXT = build_runtime_context(ee_endpoint_mode=ee_endpoint_mode)
    return _RUNTIME_CONTEXT


def build_runtime_summary(ee_endpoint_mode: str | None = None) -> dict[str, Any]:
    ctx = get_runtime_context(ee_endpoint_mode=ee_endpoint_mode)
    return {
        "project_id": runtime_project_id(),
        "public_asset_project_id": PUBLIC_ASSET_PROJECT_ID,
        "public_asset_root": PUBLIC_ASSET_ROOT,
        "repo": str(ctx["REPO"]),
        "aoi_asset_id": ctx["AOI_ASSET_ID"],
        "aoi_feature_count": ctx["aoi_feature_count"],
        "aoi_bbox": ctx["aoi_bbox"],
        "target_scale": TARGET_SCALE,
        "rank_feature_count": len(CURRENT_RANK_FEATURE_COLS),
        "prediction_asset_years": list(PREDICTION_ASSET_YEARS),
        **current_ee_runtime_config(),
    }


__all__ = [
    "CURRENT_RANK_FEATURE_COLS",
    "DEFAULT_EE_ENDPOINT_MODE",
    "EE_HIGH_VOLUME_ENDPOINT_URL",
    "EE_STANDARD_ENDPOINT_URL",
    "MAX_ERROR",
    "PREDICTION_ASSET_YEARS",
    "PUBLIC_ASSET_PROJECT_ID",
    "PUBLIC_ASSET_ROOT",
    "TARGET_PROJECTION_WKT",
    "TARGET_SCALE",
    "asset_exists",
    "build_prediction_feature_year_stack_from_assets",
    "build_runtime_summary",
    "current_ee_runtime_config",
    "ee",
    "ee_endpoint_url_for_mode",
    "get_runtime_context",
    "initialize_ee",
    "normalize_ee_endpoint_mode",
    "resolve_ee_endpoint_mode",
    "runtime_project_id",
]
