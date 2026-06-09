#!/usr/bin/env python3
"""Step_01 environmental factor-library workflow."""

# # Environmental Rasters: geemap + Google Earth Engine

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import json

import ee

from dataset_construction.common import python_output_root, resolve_dataset_construction_root
from dataset_construction.ee_runtime import PUBLIC_ASSET_ROOT, asset_exists, initialize_ee, runtime_asset_root
from dataset_construction.factor_library import (
    DYNAMIC_ANNUAL_FACTOR_SPECS,
    STATIC_FACTOR_SPECS,
    cast_single_band_for_asset,
    cast_stack_for_asset,
)

PROJECT_ID = initialize_ee(context="environmental_factors workflow")
RUNTIME_ASSET_ROOT = runtime_asset_root(PROJECT_ID)
REPO = resolve_dataset_construction_root(Path.cwd())
PYTHON_ENV_OUTPUT_DIR = python_output_root(REPO) / "03_environmental_data"
FEATURE_MANIFEST_PATH = PYTHON_ENV_OUTPUT_DIR / "feature_manifest.json"

# ## 2. Workspace Paths and Study Area AOI (QTP Boundary)

AOI_ASSET_ID = f"{PUBLIC_ASSET_ROOT}/Qinghai-Tibet_Plateau"
aoi = ee.FeatureCollection(AOI_ASSET_ID)
region = aoi.geometry()

# ## 3. Parameters and Factor Separation

TARGET_YEAR = 2016
TARGET_SCALE = 100
TARGET_PROJECTION_WKT = (
    'PROJCS["Albers",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",90.0],PARAMETER["standard_parallel_1",27.5],PARAMETER["standard_parallel_2",37.5],PARAMETER["latitude_of_origin",0.0],UNIT["Meter",1.0]]'
)
TARGET_CRS = TARGET_PROJECTION_WKT
TARGET_CRS_TRANSFORM = [TARGET_SCALE, 0, 0, 0, -TARGET_SCALE, 0]
EXPORT_SCALE = 100


DEM_DATASET_ID = "CGIAR/SRTM90_V4"
FLOW_ACC_DATASET_ID = "WWF/HydroSHEDS/15ACC"
TWI_MIN_TAN_SLOPE = 1e-3
LOCAL_RELIEF_RADIUS_M = 5000
ASPECT_FLAT_THRESHOLD_DEG = 0.0
SOILGRIDS_CLAY_ASSET = "projects/soilgrids-isric/clay_mean"
SOILGRIDS_SILT_ASSET = "projects/soilgrids-isric/silt_mean"
SOILGRIDS_DEPTHS = [
    ("0-5cm", 5.0),
    ("5-15cm", 10.0),
    ("15-30cm", 15.0),
    ("30-60cm", 30.0),
    ("60-100cm", 40.0),
    ("100-200cm", 100.0),
]
SOILGRIDS_PERCENT_SCALE = 10.0
VEGETATION_TYPE_ASSET_ID = f"{PUBLIC_ASSET_ROOT}/environmental_factors/vegetation_type/QTP_Vegetation_type"
GIC_ASSET_ID = f"{PUBLIC_ASSET_ROOT}/environmental_factors/factor_library_v2/static/GIC_v2"
NDVI_DATASET_ID = "MODIS/061/MYD13Q1"
NDVI_SUMMER_START = "06-01"
NDVI_SUMMER_END_EXCLUSIVE = "09-01"
NDVI_SCALE_FACTOR = 0.0001
SOIL_MOISTURE_DATASET_005 = "NASA/SMAP/SPL3SMP_E/005"
SOIL_MOISTURE_DATASET_006 = "NASA/SMAP/SPL3SMP_E/006"
SOIL_MOISTURE_AM_BAND = "soil_moisture_am"
SOIL_MOISTURE_PM_BAND = "soil_moisture_pm"
SOIL_MOISTURE_SUMMER_START = "06-01"
SOIL_MOISTURE_SUMMER_END_EXCLUSIVE = "09-01"
PRECIP_DATASET_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
PRECIP_BAND_NAME = "total_precipitation_sum"
PRECIP_TO_MM = 1000.0
SMDP_DATASET_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
SMDP_BAND_NAME = "total_precipitation_sum"
SMDP_TO_MM = 1000.0
SMDP_SUMMER_START = "06-01"
SMDP_SUMMER_END_EXCLUSIVE = "09-01"
TDD_DATASET_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
WARM_SEASON_START = "06-01"
WARM_SEASON_END_EXCLUSIVE = "09-01"
HISTORICAL_CLIMATE_COLLECTION_ID = "ECMWF/ERA5_LAND/DAILY_AGGR"
HISTORICAL_TMEAN_BAND = "temperature_2m"
HISTORICAL_TMAX_BAND = "temperature_2m_max"
HISTORICAL_DEWPOINT_BAND = "dewpoint_temperature_2m"
TDD_BASE_TEMP_C = 0.0

# ### 3.1 DEM

dem_raw = ee.Image(DEM_DATASET_ID).select("elevation").clip(region)
terrain_projection = ee.Projection(TARGET_CRS, TARGET_CRS_TRANSFORM)
dem = dem_raw.resample("bilinear").reproject(terrain_projection).rename("DEM")
terrain = ee.Terrain.products(dem)

# ### 3.2 Slope

slope = terrain.select("slope").rename("Slope")

# ### 3.3 Aspect

def classify_aspect_image(
    aspect_deg_image: ee.Image,
    slope_image: ee.Image,
    mask_image: ee.Image,
    band_name: str = "Aspect",
) -> ee.Image:
    base = aspect_deg_image.multiply(0)
    aspect_nonflat = (
        base.add(2)
        .where(aspect_deg_image.gte(22.5).And(aspect_deg_image.lt(67.5)), 3)
        .where(aspect_deg_image.gte(67.5).And(aspect_deg_image.lt(112.5)), 4)
        .where(aspect_deg_image.gte(112.5).And(aspect_deg_image.lt(157.5)), 5)
        .where(aspect_deg_image.gte(157.5).And(aspect_deg_image.lt(202.5)), 6)
        .where(aspect_deg_image.gte(202.5).And(aspect_deg_image.lt(247.5)), 7)
        .where(aspect_deg_image.gte(247.5).And(aspect_deg_image.lt(292.5)), 8)
        .where(aspect_deg_image.gte(292.5).And(aspect_deg_image.lt(337.5)), 9)
    )
    non_flat = slope_image.gt(ASPECT_FLAT_THRESHOLD_DEG)
    return base.where(non_flat, aspect_nonflat).updateMask(mask_image).rename(band_name).toInt16()

aspect_deg = terrain.select("aspect").rename("Aspect_deg")
aspect = classify_aspect_image(aspect_deg, slope, dem.mask()).reproject(terrain_projection)

# ### 3.4 TWI

flow_acc = ee.Image(FLOW_ACC_DATASET_ID).select("b1").clip(region)
flow_acc_100m = flow_acc.resample("bilinear").reproject(terrain_projection)
cell_size_m = ee.Image.constant(TARGET_SCALE)
slope_for_twi = slope.reproject(terrain_projection)
slope_rad = slope_for_twi.multiply(3.141592653589793 / 180.0)
safe_tan_slope = slope_rad.tan().max(ee.Image.constant(TWI_MIN_TAN_SLOPE))
twi = flow_acc_100m.add(1).multiply(cell_size_m).divide(safe_tan_slope).log().rename("TWI")

# ### 3.5 Local_Relief

def build_local_relief_layer(
    dem_image: ee.Image,
    radius_m: float = LOCAL_RELIEF_RADIUS_M,
    band_name: str = "Local_Relief",
    projection: ee.Projection | None = terrain_projection,
) -> ee.Image:
    kernel = ee.Kernel.circle(radius=radius_m, units="meters", normalize=False)
    neighborhood_stats = dem_image.reduceNeighborhood(reducer=ee.Reducer.minMax(), kernel=kernel)
    relief_image = neighborhood_stats.select('.*_max').subtract(neighborhood_stats.select('.*_min')).updateMask(dem_image.mask()).rename(band_name)
    if projection is not None:
        relief_image = relief_image.reproject(projection)
    return relief_image


local_relief = build_local_relief_layer(dem, radius_m=LOCAL_RELIEF_RADIUS_M, band_name="Local_Relief")

# ### 3.6 Soil

def normalize_soilgrids_depth_label(depth_label: str) -> str:
    if depth_label.startswith("val_") and depth_label.endswith("_mean"):
        depth_label = depth_label[4:-5]
        return depth_label.replace("_", "-")
    return depth_label


def resolve_soilgrids_band_name(available_bands, variable_prefix: str, depth_label: str) -> str:
    normalized_depth = normalize_soilgrids_depth_label(depth_label)
    candidates = [
        f"{variable_prefix}_{normalized_depth}_mean",
        f"val_{normalized_depth.replace('-', '_')}_mean",
    ]
    for candidate in candidates:
        if candidate in available_bands:
            return candidate
    raise KeyError(
        f"{variable_prefix} has no available band at depth {normalized_depth}. Candidates: {candidates}; Available: {sorted(available_bands)}"
    )


def build_soil_fine_sediment_layer(
    region: ee.Geometry,
    projection: ee.Projection | None = terrain_projection,
):
    clay = ee.Image(SOILGRIDS_CLAY_ASSET).clip(region).resample("bilinear")
    silt = ee.Image(SOILGRIDS_SILT_ASSET).clip(region).resample("bilinear")
    clay_bands = set(clay.bandNames().getInfo())
    silt_bands = set(silt.bandNames().getInfo())

    weighted_sum = ee.Image.constant(0).toFloat()
    total_thickness_cm = 0.0
    band_pairs = []
    for depth_label, thickness_cm in SOILGRIDS_DEPTHS:
        clay_band = resolve_soilgrids_band_name(clay_bands, "clay", depth_label)
        silt_band = resolve_soilgrids_band_name(silt_bands, "silt", depth_label)
        band_pairs.append((depth_label, clay_band, silt_band))
        fine_pct = (
            clay.select(clay_band).divide(SOILGRIDS_PERCENT_SCALE)
            .add(silt.select(silt_band).divide(SOILGRIDS_PERCENT_SCALE))
        )
        weighted_sum = weighted_sum.add(fine_pct.multiply(thickness_cm))
        total_thickness_cm += thickness_cm

    soil = weighted_sum.divide(total_thickness_cm).rename("Soil")
    if projection is not None:
        soil = soil.reproject(projection)
    return soil, band_pairs


soil, _ = build_soil_fine_sediment_layer(region)

# ### 3.7 Vegetation_type

vegetation_type = (
    ee.Image(VEGETATION_TYPE_ASSET_ID)
    .clip(region)
    .rename("Vegetation_type")
    .reproject(terrain_projection)
    .toInt16()
)

# ### 3.8 GIC

gic = (
    ee.Image(GIC_ASSET_ID)
    .clip(region)
    .resample("bilinear")
    .reproject(terrain_projection)
    .rename("GIC")
)

# ### 3.9 NDVI / Summer_SoilMoisture

def build_ndvi_for_year(year: int, band_name: str = "NDVI") -> ee.Image:
    summer_start = f"{year}-{NDVI_SUMMER_START}"
    summer_end = f"{year}-{NDVI_SUMMER_END_EXCLUSIVE}"
    return (
        ee.ImageCollection(NDVI_DATASET_ID)
        .filterDate(summer_start, summer_end)
        .select("NDVI")
        .median()
        .multiply(NDVI_SCALE_FACTOR)
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )

def build_soil_moisture_collection(start_date: str, end_date: str) -> ee.ImageCollection:
    return ee.ImageCollection(SOIL_MOISTURE_DATASET_005).merge(ee.ImageCollection(SOIL_MOISTURE_DATASET_006)).filterDate(start_date, end_date)

def combine_daily_soil_moisture(image: ee.Image, band_name: str = "soil_moisture_daily") -> ee.Image:
    return (
        ee.ImageCollection.fromImages([
            image.select(SOIL_MOISTURE_AM_BAND).rename(band_name),
            image.select(SOIL_MOISTURE_PM_BAND).rename(band_name),
        ])
        .mean()
        .copyProperties(image, ["system:time_start"])
    )

def build_summer_soil_moisture_for_year(year: int, band_name: str = "Summer_SoilMoisture") -> ee.Image:
    summer_start = f"{year}-{SOIL_MOISTURE_SUMMER_START}"
    summer_end = f"{year}-{SOIL_MOISTURE_SUMMER_END_EXCLUSIVE}"
    return (
        build_soil_moisture_collection(summer_start, summer_end)
        .select([SOIL_MOISTURE_AM_BAND, SOIL_MOISTURE_PM_BAND])
        .map(lambda image: combine_daily_soil_moisture(image, band_name="soil_moisture_daily"))
        .mean()
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )

ndvi = build_ndvi_for_year(TARGET_YEAR, band_name="NDVI")
summer_soil_moisture = build_summer_soil_moisture_for_year(TARGET_YEAR, band_name="Summer_SoilMoisture")

# ### 3.10 Precipitation

def build_precip_for_year(year: int, band_name: str = "Precipitation") -> ee.Image:
    year_start = f"{year}-01-01"
    year_end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(PRECIP_DATASET_ID)
        .filterDate(year_start, year_end)
        .select(PRECIP_BAND_NAME)
        .sum()
        .multiply(PRECIP_TO_MM)
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )

precip = build_precip_for_year(TARGET_YEAR, band_name="Precipitation")

# ### 3.11 SMDP

def build_smdp_for_year(year: int, band_name: str = "SMDP") -> ee.Image:
    summer_start = f"{year}-{SMDP_SUMMER_START}"
    summer_end = f"{year}-{SMDP_SUMMER_END_EXCLUSIVE}"
    return (
        ee.ImageCollection(SMDP_DATASET_ID)
        .filterDate(summer_start, summer_end)
        .select(SMDP_BAND_NAME)
        .max()
        .multiply(SMDP_TO_MM)
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )

smdp = build_smdp_for_year(TARGET_YEAR, band_name="SMDP")

# ### 3.12 TDD

def positive_temperature_c(image: ee.Image) -> ee.Image:
    temp_c = image.subtract(273.15)
    return temp_c.max(ee.Image.constant(TDD_BASE_TEMP_C)).copyProperties(image, ["system:time_start"])


def build_tdd_for_year(year: int, band_name: str = "TDD") -> ee.Image:
    year_start = f"{year}-01-01"
    year_end = f"{year + 1}-01-01"
    return (
        ee.ImageCollection(TDD_DATASET_ID)
        .filterDate(year_start, year_end)
        .select("temperature_2m")
        .map(positive_temperature_c)
        .sum()
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )


tdd = build_tdd_for_year(TARGET_YEAR, band_name="TDD")

# ### 3.13 WSTmax

def warm_season_window(year: int) -> tuple[str, str]:
    return (f"{year}-{WARM_SEASON_START}", f"{year}-{WARM_SEASON_END_EXCLUSIVE}")


def get_historical_era5_collection(start_date: str, end_date: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(HISTORICAL_CLIMATE_COLLECTION_ID)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .select(
            [
                HISTORICAL_TMEAN_BAND,
                HISTORICAL_TMAX_BAND,
                HISTORICAL_DEWPOINT_BAND,
            ]
        )
    )


def build_tmax_for_year(year: int, band_name: str = "WSTmax") -> ee.Image:
    warm_start, warm_end = warm_season_window(year)
    return (
        get_historical_era5_collection(warm_start, warm_end)
        .select(HISTORICAL_TMAX_BAND)
        .max()
        .subtract(273.15)
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )


tmax = build_tmax_for_year(TARGET_YEAR, band_name="WSTmax")

# ### 3.14 WSRH

def calculate_relative_humidity_pct(temp_c: ee.Image, dewpoint_c: ee.Image) -> ee.Image:
    alpha = 17.625
    beta = 243.04
    saturation_ratio = temp_c.multiply(alpha).divide(temp_c.add(beta)).exp()
    actual_ratio = dewpoint_c.multiply(alpha).divide(dewpoint_c.add(beta)).exp()
    return actual_ratio.divide(saturation_ratio).multiply(100.0).clamp(0.0, 100.0)


def build_warm_season_rh_for_year(year: int, band_name: str = "WSRH") -> ee.Image:
    warm_start, warm_end = warm_season_window(year)

    def _daily_rh(image: ee.Image) -> ee.Image:
        temp_c = image.select(HISTORICAL_TMEAN_BAND).subtract(273.15)
        dewpoint_c = image.select(HISTORICAL_DEWPOINT_BAND).subtract(273.15)
        return calculate_relative_humidity_pct(temp_c, dewpoint_c).rename(band_name).copyProperties(image, ["system:time_start"])

    return (
        get_historical_era5_collection(warm_start, warm_end)
        .map(_daily_rh)
        .mean()
        .clip(region)
        .resample("bilinear")
        .reproject(terrain_projection)
        .rename(band_name)
    )


warm_season_rh_target_year = build_warm_season_rh_for_year(TARGET_YEAR, band_name="WSRH")

# ### 3.15 Band Stacking

annual_stack_uncached_cache = {}
CURRENT_STATIC_EXPORT_FEATURE_COLS = [
    "Slope",
    "Aspect",
    "TWI",
    "Local_Relief",
    "Soil",
    "Vegetation_type",
    "GIC",
]
EXPORT_COLUMN_ALIASES = {}
EXPORT_SOURCE_COLUMN_NAMES = {new_name: old_name for old_name, new_name in EXPORT_COLUMN_ALIASES.items()}
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
RANK_SOURCE_FEATURE_COLS = [EXPORT_SOURCE_COLUMN_NAMES.get(col, col) for col in CURRENT_RANK_FEATURE_COLS]


def build_static_feature_stack_uncached(relief_image: ee.Image | None = None) -> ee.Image:
    relief_source = ee.Image(relief_image) if relief_image is not None else local_relief
    raw_stack = (
        slope.addBands(aspect)
        .addBands(twi)
        .addBands(relief_source)
        .addBands(soil)
        .addBands(vegetation_type)
        .addBands(gic)
        .select(CURRENT_STATIC_EXPORT_FEATURE_COLS)
    )
    return cast_stack_for_asset(raw_stack, CURRENT_STATIC_EXPORT_FEATURE_COLS)


def build_feature_year_stack_uncached(year: int, static_stack_image: ee.Image | None = None) -> ee.Image:
    static_stack = ee.Image(static_stack_image) if static_stack_image is not None else build_static_feature_stack_uncached()
    raw_stack = (
        static_stack
        .addBands(build_tdd_for_year(year, band_name="TDD"))
        .addBands(build_smdp_for_year(year, band_name="SMDP"))
        .addBands(build_precip_for_year(year, band_name="Precipitation"))
        .addBands(build_tmax_for_year(year, band_name="WSTmax"))
        .addBands(build_warm_season_rh_for_year(year, band_name="WSRH"))
        .addBands(build_ndvi_for_year(year, band_name="NDVI"))
        .addBands(build_summer_soil_moisture_for_year(year, band_name="Summer_SoilMoisture"))
        .select(RANK_SOURCE_FEATURE_COLS, CURRENT_RANK_FEATURE_COLS)
    )
    return cast_stack_for_asset(raw_stack, CURRENT_RANK_FEATURE_COLS)


def get_feature_year_stack_uncached(year: int) -> ee.Image:
    year = int(year)
    if year not in annual_stack_uncached_cache:
        annual_stack_uncached_cache[year] = build_feature_year_stack_uncached(year)
    return annual_stack_uncached_cache[year]


stack = get_feature_year_stack_uncached(TARGET_YEAR)

# ## 4. Official Route: Single-factor Asset Library

from datetime import datetime, timezone

DEFAULT_ASSET_YEARS = [2016, 2017, 2018, 2019, 2020]
ASSET_TASK_ACTIVE_STATES = ("READY", "RUNNING", "CANCEL_REQUESTED")
ASSET_EXPORT_SCALE = EXPORT_SCALE
ASSET_EXPORT_MAX_PIXELS = 1e13
ASSET_EXPORT_CRS = TARGET_CRS
ASSET_EXPORT_CRS_TRANSFORM = list(TARGET_CRS_TRANSFORM)
ASSET_CATEGORICAL_BANDS = frozenset({"Aspect", "Vegetation_type"})

OPERATION_TO_TASK_STATE = {
    "PENDING": "READY",
    "RUNNING": "RUNNING",
    "SUCCEEDED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLING": "CANCEL_REQUESTED",
    "CANCELLED": "CANCELLED",
}



def _coerce_task_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _timestamp_to_millis(value) -> int:
    if value in (None, ""):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return 0
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return _coerce_task_int(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return int(parsed.timestamp() * 1000)
    return _coerce_task_int(value)


def _normalize_operation_state(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return OPERATION_TO_TASK_STATE.get(text, text)


def normalize_ee_task_record(raw_record: dict) -> dict:
    metadata = raw_record.get("metadata") if isinstance(raw_record.get("metadata"), dict) else {}
    if "done" in raw_record and metadata:
        error_message = None
        if raw_record.get("done") and isinstance(raw_record.get("error"), dict):
            error_message = raw_record["error"].get("message")
        operation_name = str(raw_record.get("name") or "")
        return {
            "id": operation_name.rsplit("/", 1)[-1] or operation_name,
            "name": operation_name,
            "description": metadata.get("description"),
            "state": _normalize_operation_state(metadata.get("state")),
            "task_type": metadata.get("type"),
            "creation_timestamp_ms": _timestamp_to_millis(metadata.get("createTime")),
            "start_timestamp_ms": _timestamp_to_millis(metadata.get("startTime")),
            "update_timestamp_ms": _timestamp_to_millis(metadata.get("updateTime")),
            "error_message": error_message,
            "source": "listOperations",
        }
    return {
        "id": raw_record.get("id") or raw_record.get("name"),
        "name": raw_record.get("name"),
        "description": raw_record.get("description") or metadata.get("description"),
        "state": _normalize_operation_state(raw_record.get("state") or metadata.get("state")),
        "task_type": raw_record.get("task_type") or raw_record.get("type") or metadata.get("type"),
        "creation_timestamp_ms": _coerce_task_int(
            raw_record.get("creation_timestamp_ms") or metadata.get("creation_timestamp_ms")
        ),
        "start_timestamp_ms": _coerce_task_int(
            raw_record.get("start_timestamp_ms") or metadata.get("start_timestamp_ms")
        ),
        "update_timestamp_ms": _coerce_task_int(
            raw_record.get("update_timestamp_ms") or metadata.get("update_timestamp_ms")
        ),
        "error_message": raw_record.get("error_message") or metadata.get("error_message"),
        "source": "getTaskList",
    }


def sort_key_for_task_record(task_record: dict) -> tuple[int, int, int]:
    return (
        int(task_record.get("update_timestamp_ms") or 0),
        int(task_record.get("start_timestamp_ms") or 0),
        int(task_record.get("creation_timestamp_ms") or 0),
    )


def fetch_task_snapshot_by_description(
    ee_data_module,
    relevant_descriptions: set[str],
    *,
    fallback_label: str | None = None,
) -> dict:
    source = None
    try:
        raw_records = ee_data_module.listOperations() or []
        source = "listOperations"
        error = None
    except Exception as exc:
        try:
            raw_records = ee_data_module.getTaskList() or []
            source = "getTaskList"
            error = None
        except Exception as fallback_exc:
            label = f"{fallback_label} " if fallback_label else ""
            raw_records = []
            error = f"{label}task query failed: primary={exc!r}; fallback={fallback_exc!r}"

    normalized_tasks = []
    for raw_record in raw_records:
        if not isinstance(raw_record, dict):
            continue
        task = normalize_ee_task_record(raw_record)
        if task.get("description") in relevant_descriptions:
            normalized_tasks.append(task)

    normalized_tasks.sort(key=sort_key_for_task_record, reverse=True)
    task_index = {description: [] for description in relevant_descriptions}
    for task in normalized_tasks:
        task_index.setdefault(task["description"], []).append(task)
    return {
        "ok": error is None,
        "error": error,
        "source": source,
        "tasks": normalized_tasks,
        "index": task_index,
    }




def build_export_pyramiding_policy(
    expected_bands: list[str] | tuple[str, ...],
    *,
    categorical_bands: set[str] | frozenset[str],
    default_policy: str = "mean",
    categorical_policy: str = "mode",
) -> dict[str, str]:
    policy = {".default": default_policy}
    for band_name in expected_bands:
        if str(band_name) in categorical_bands:
            policy[str(band_name)] = categorical_policy
    return policy

def normalize_asset_years(years=None) -> list[int]:
    if years is None:
        return list(DEFAULT_ASSET_YEARS)
    normalized = sorted({int(year) for year in years})
    if not normalized:
        raise ValueError("Years list cannot be empty.")
    return normalized


def asset_export_pyramiding_policy(expected_bands: list[str]) -> dict[str, str]:
    return build_export_pyramiding_policy(
        expected_bands,
        categorical_bands=ASSET_CATEGORICAL_BANDS,
    )


def attach_export_grid_metadata(
    image: ee.Image,
    *,
    asset_id: str,
    expected_bands: list[str],
    route: str,
    group: str | None = None,
    year: int | None = None,
) -> ee.Image:
    props = {
        "asset_id": asset_id,
        "asset_route": route,
        "expected_bands": ",".join(expected_bands),
        "grid_crs": ASSET_EXPORT_CRS,
        "grid_transform": ",".join(str(value) for value in (ASSET_EXPORT_CRS_TRANSFORM or [])),
        "grid_scale_m": float(TARGET_SCALE),
    }
    if group is not None:
        props["asset_group"] = group
    if year is not None:
        props["year"] = int(year)
        props["system:time_start"] = ee.Date.fromYMD(int(year), 1, 1).millis()
    return ee.Image(image).set(props)


def start_image_asset_export_task(
    image: ee.Image,
    *,
    asset_id: str,
    description: str,
    expected_bands: list[str],
    route: str,
    group: str | None = None,
    year: int | None = None,
) -> ee.batch.Task:
    export_image = attach_export_grid_metadata(
        image,
        asset_id=asset_id,
        expected_bands=expected_bands,
        route=route,
        group=group,
        year=year,
    )
    export_kwargs = {
        "image": export_image,
        "description": description,
        "assetId": asset_id,
        "region": region,
        "maxPixels": int(ASSET_EXPORT_MAX_PIXELS),
        "pyramidingPolicy": asset_export_pyramiding_policy(expected_bands),
    }
    if ASSET_EXPORT_CRS:
        export_kwargs["crs"] = ASSET_EXPORT_CRS
    if ASSET_EXPORT_CRS_TRANSFORM:
        export_kwargs["crsTransform"] = ASSET_EXPORT_CRS_TRANSFORM
    else:
        export_kwargs["scale"] = ASSET_EXPORT_SCALE
    task = ee.batch.Export.image.toAsset(**export_kwargs)
    task.start()
    return task


print(
    {
        "formal_asset_strategy": "factor_library_only",
        "factor_library_years": DEFAULT_ASSET_YEARS,
        "asset_export_crs_defined": bool(ASSET_EXPORT_CRS),
        "asset_export_transform_defined": bool(ASSET_EXPORT_CRS_TRANSFORM),
        "asset_categorical_bands": sorted(ASSET_CATEGORICAL_BANDS),
    }
)

FACTOR_LIBRARY_ROOT = f"{RUNTIME_ASSET_ROOT}/environmental_factors/factor_library_v2"
FACTOR_LIBRARY_VERSION_TAG = "v2"
SMDP_ASSET_VERSION_SUFFIX = "_v2"
STATIC_FACTOR_ASSET_ROOT = f"{FACTOR_LIBRARY_ROOT}/static"
DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT = f"{FACTOR_LIBRARY_ROOT}/dynamic_annual"
FACTOR_LIBRARY_TASK_PREFIX = "factorlib"
ASSET_LIBRARY_YEARS = list(DEFAULT_ASSET_YEARS)
FACTOR_LIBRARY_ACTIVE_TASK_STATES = ASSET_TASK_ACTIVE_STATES
FACTOR_LIBRARY_GROUP_SEQUENCE = ("static", "dynamic_annual")
FACTOR_LIBRARY_STORAGE_MODE = {
    "static": "single_image_asset",
    "dynamic_annual": "annual_image_assets_with_optional_collection_view",
}
FACTOR_LIBRARY_GROUP_TO_NOTEBOOK_CELL = {
    "static": "4.2",
    "dynamic_annual": "4.3",
}
FACTOR_LIBRARY_DYNAMIC_ANNUAL_BANDS = [item["band_name"] for item in DYNAMIC_ANNUAL_FACTOR_SPECS]


def factor_group_storage_mode(group: str) -> str:
    normalized = normalize_factor_library_group(group)
    return FACTOR_LIBRARY_STORAGE_MODE[normalized]
def factor_asset_root(group: str) -> str:
    normalized = str(group).strip().lower()
    if normalized == "static":
        return STATIC_FACTOR_ASSET_ROOT
    if normalized == "dynamic_annual":
        return DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT
    raise ValueError(f"Unsupported factor asset group: {group}")


def factor_group_specs(group: str) -> list[dict]:
    normalized = str(group).strip().lower()
    if normalized == "static":
        return [dict(item) for item in STATIC_FACTOR_SPECS]
    if normalized == "dynamic_annual":
        return [dict(item) for item in DYNAMIC_ANNUAL_FACTOR_SPECS]
    raise ValueError(f"Unsupported factor asset group: {group}")


def normalize_factor_library_group(group: str) -> str:
    normalized = str(group).strip().lower()
    if normalized not in FACTOR_LIBRARY_GROUP_SEQUENCE:
        raise ValueError(f"Unsupported factor asset group: {group}; allowed={list(FACTOR_LIBRARY_GROUP_SEQUENCE)}")
    return normalized


def factor_asset_name(group: str, factor_name: str, year: int | None = None) -> str:
    normalized_group = normalize_factor_library_group(group)
    normalized_factor = str(factor_name).strip()
    if normalized_group == "static":
        if normalized_factor == "Local_Relief":
            return f"Local_Relief_5km_{FACTOR_LIBRARY_VERSION_TAG}"
        return f"{normalized_factor}_{FACTOR_LIBRARY_VERSION_TAG}"
    if year is None:
        raise ValueError("dynamic_annual asset name requires year.")
    if normalized_factor == "SMDP":
        return f"{normalized_factor}_{int(year)}{SMDP_ASSET_VERSION_SUFFIX}"
    return f"{normalized_factor}_{int(year)}_{FACTOR_LIBRARY_VERSION_TAG}"


def factor_asset_id(group: str, factor_name: str, year: int | None = None) -> str:
    return f"{factor_asset_root(group).rstrip('/')}/{factor_asset_name(group, factor_name, year=year)}"


def factor_task_description(group: str, factor_name: str, year: int | None = None) -> str:
    normalized_group = normalize_factor_library_group(group)
    normalized_factor = str(factor_name).strip()
    if normalized_group == "static":
        return f"{FACTOR_LIBRARY_TASK_PREFIX}_static_{factor_asset_name('static', normalized_factor)}"
    if year is None:
        raise ValueError("dynamic_annual task description requires year.")
    suffix = SMDP_ASSET_VERSION_SUFFIX.lstrip("_") if normalized_factor == "SMDP" else FACTOR_LIBRARY_VERSION_TAG
    return f"{FACTOR_LIBRARY_TASK_PREFIX}_dynamic_annual_{normalized_factor}_{int(year)}_{suffix}"


def factor_library_root_parent_asset_id() -> str:
    return FACTOR_LIBRARY_ROOT.rsplit('/', 1)[0]


def ensure_folder_asset_exists(folder_asset_id: str, parent_asset_id: str) -> dict:
    if asset_exists(folder_asset_id):
        return {"asset_id": folder_asset_id, "status": "exists", "kind": "folder", "parent_asset_id": parent_asset_id}
    if not asset_exists(parent_asset_id):
        raise FileNotFoundError(f"Parent asset directory does not exist: {parent_asset_id}")
    try:
        ee.data.createFolder(folder_asset_id)
        status = "created"
    except Exception as exc:
        if asset_exists(folder_asset_id):
            status = "exists"
        else:
            raise RuntimeError(f"Failed to create asset directory: {folder_asset_id}; reason={exc}") from exc
    return {"asset_id": folder_asset_id, "status": status, "kind": "folder", "parent_asset_id": parent_asset_id}


def ensure_factor_library_asset_roots() -> list[dict]:
    created = []
    created.append(ensure_folder_asset_exists(FACTOR_LIBRARY_ROOT, factor_library_root_parent_asset_id()))
    created.append(ensure_folder_asset_exists(STATIC_FACTOR_ASSET_ROOT, FACTOR_LIBRARY_ROOT))
    created.append(ensure_folder_asset_exists(DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT, FACTOR_LIBRARY_ROOT))
    return created


def build_static_factor_image(factor_name: str) -> ee.Image:
    normalized_factor = str(factor_name).strip()
    if normalized_factor == "Slope":
        return cast_single_band_for_asset(slope.select(["Slope"]), "Slope")
    if normalized_factor == "Aspect":
        return cast_single_band_for_asset(aspect.select(["Aspect"]), "Aspect")
    if normalized_factor == "TWI":
        return cast_single_band_for_asset(twi.select(["TWI"]), "TWI")
    if normalized_factor == "Local_Relief":
        return cast_single_band_for_asset(local_relief.select(["Local_Relief"]), "Local_Relief")
    if normalized_factor == "Soil":
        return cast_single_band_for_asset(soil.select(["Soil"]), "Soil")
    if normalized_factor == "Vegetation_type":
        return cast_single_band_for_asset(vegetation_type.select(["Vegetation_type"]), "Vegetation_type")
    if normalized_factor == "GIC":
        return cast_single_band_for_asset(gic.select(["GIC"]), "GIC")
    raise ValueError(f"Unsupported static factor: {factor_name}")


def build_dynamic_annual_factor_image(factor_name: str, year: int) -> ee.Image:
    normalized_factor = str(factor_name).strip()
    year = int(year)
    if normalized_factor == "TDD":
        return cast_single_band_for_asset(build_tdd_for_year(year, band_name="TDD"), "TDD")
    if normalized_factor == "SMDP":
        return cast_single_band_for_asset(build_smdp_for_year(year, band_name="SMDP"), "SMDP")
    if normalized_factor == "Precipitation":
        return cast_single_band_for_asset(build_precip_for_year(year, band_name="Precipitation"), "Precipitation")
    if normalized_factor == "WSTmax":
        return cast_single_band_for_asset(build_tmax_for_year(year, band_name="WSTmax"), "WSTmax")
    if normalized_factor == "WSRH":
        return cast_single_band_for_asset(build_warm_season_rh_for_year(year, band_name="WSRH"), "WSRH")
    if normalized_factor == "NDVI":
        return cast_single_band_for_asset(build_ndvi_for_year(year, band_name="NDVI"), "NDVI")
    if normalized_factor == "Summer_SoilMoisture":
        return cast_single_band_for_asset(build_summer_soil_moisture_for_year(year, band_name="Summer_SoilMoisture"), "Summer_SoilMoisture")
    raise ValueError(f"Unsupported annual dynamic factor: {factor_name}")
def build_factor_asset_specs(group: str, years=None, include_image: bool = False) -> list[dict]:
    normalized_group = normalize_factor_library_group(group)
    selected_years = normalize_asset_years(years)
    specs = []
    for spec in factor_group_specs(normalized_group):
        factor_name = spec["factor_name"]
        band_name = spec["band_name"]
        if normalized_group == "static":
            row = {
                "group": normalized_group,
                "factor_name": factor_name,
                "band_name": band_name,
                "asset_id": factor_asset_id(normalized_group, factor_name),
                "task_description": factor_task_description(normalized_group, factor_name),
                "expected_bands": [band_name],
                "years_covered": list(selected_years),
            }
            if include_image:
                row["image"] = build_static_factor_image(factor_name)
            specs.append(row)
            continue

        for year in selected_years:
            row = {
                "group": normalized_group,
                "factor_name": factor_name,
                "band_name": band_name,
                "year": int(year),
                "asset_id": factor_asset_id(normalized_group, factor_name, year=int(year)),
                "task_description": factor_task_description(normalized_group, factor_name, year=int(year)),
                "expected_bands": [band_name],
                "years_covered": [int(year)],
            }
            if include_image:
                row["image"] = build_dynamic_annual_factor_image(factor_name, int(year))
            specs.append(row)
    return specs


def attach_factor_asset_export_metadata(image: ee.Image, spec: dict) -> ee.Image:
    props = {
        "factor_library_group": spec["group"],
        "factor_name": spec["factor_name"],
        "band_name": spec["band_name"],
        "asset_version": FACTOR_LIBRARY_VERSION_TAG,
        "storage_mode": factor_group_storage_mode(spec["group"]),
        "preferred_asset_type": "ee.Image",
    }
    if spec.get("year") is not None:
        year = int(spec["year"])
        props["year"] = year
        props["series_key"] = spec["factor_name"]
        props["collection_view_supported"] = True
        props["system:time_start"] = ee.Date.fromYMD(year, 1, 1).millis()
    return ee.Image(image).set(props)


def build_factor_asset_export_image(spec: dict) -> ee.Image:
    group = normalize_factor_library_group(spec["group"])
    factor_name = spec["factor_name"]
    if group == "static":
        image = build_static_factor_image(factor_name)
    else:
        image = build_dynamic_annual_factor_image(factor_name, int(spec["year"]))
    return attach_factor_asset_export_metadata(image, spec)


def fetch_factor_library_task_snapshot(years=None) -> dict:
    selected_years = normalize_asset_years(years)
    relevant_descriptions = {
        spec["task_description"]
        for group in FACTOR_LIBRARY_GROUP_SEQUENCE
        for spec in build_factor_asset_specs(group, years=selected_years, include_image=False)
    }
    return fetch_task_snapshot_by_description(
        ee.data,
        relevant_descriptions,
        fallback_label="factor_library",
    )


def active_factor_library_tasks_for_description(task_snapshot: dict, description: str | None) -> list[dict]:
    if not description:
        return []
    return [task for task in task_snapshot.get("index", {}).get(description, []) if task.get("state") in FACTOR_LIBRARY_ACTIVE_TASK_STATES]


def submit_factor_asset_group(group: str, years=None) -> dict:
    normalized_group = normalize_factor_library_group(group)
    selected_years = normalize_asset_years(years)
    folder_records = ensure_factor_library_asset_roots()
    task_snapshot = fetch_factor_library_task_snapshot(years=selected_years)
    if task_snapshot.get("error"):
        raise RuntimeError(
            "Failed to read the factor_library task list. To avoid duplicate submissions, this execution has been aborted."
            f" error={task_snapshot['error']}"
        )
    submitted_records = []
    for spec in build_factor_asset_specs(normalized_group, years=selected_years, include_image=False):
        if asset_exists(spec["asset_id"]):
            submitted_records.append(
                {
                    "group": normalized_group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "status": "skipped_existing",
                }
            )
            continue
        active_records = active_factor_library_tasks_for_description(task_snapshot, spec["task_description"])
        if active_records:
            submitted_records.append(
                {
                    "group": normalized_group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "status": "skipped_active",
                    "task_id": active_records[0].get("id"),
                    "task_state": active_records[0].get("state"),
                }
            )
            continue
        task = start_image_asset_export_task(
            build_factor_asset_export_image(spec),
            asset_id=spec["asset_id"],
            description=spec["task_description"],
            expected_bands=list(spec["expected_bands"]),
            route="factor_library",
            group=normalized_group,
            year=spec.get("year"),
        )
        task_status = task.status() or {}
        submitted_records.append(
            {
                "group": normalized_group,
                "factor_name": spec["factor_name"],
                "year": spec.get("year"),
                "asset_id": spec["asset_id"],
                "task_description": spec["task_description"],
                "status": task_status.get("state"),
                "task_id": task_status.get("id"),
            }
        )
    return {
        "group": normalized_group,
        "years": selected_years if normalized_group == "dynamic_annual" else None,
        "folder_records": folder_records,
        "submitted_records": submitted_records,
    }


def describe_factor_asset_library(years=None) -> dict:
    selected_years = normalize_asset_years(years)
    task_snapshot = fetch_factor_library_task_snapshot(years=selected_years)
    rows = []
    for group in FACTOR_LIBRARY_GROUP_SEQUENCE:
        for spec in build_factor_asset_specs(group, years=selected_years, include_image=False):
            active_records = active_factor_library_tasks_for_description(task_snapshot, spec["task_description"])
            latest_records = task_snapshot.get("index", {}).get(spec["task_description"], [])
            latest_record = latest_records[0] if latest_records else None
            exists = asset_exists(spec["asset_id"])
            if exists:
                status = "exists"
            elif active_records:
                status = f"task_{str(active_records[0].get('state')).lower()}"
            elif latest_record and latest_record.get("state") in {"FAILED", "CANCELLED"}:
                status = f"missing_last_{str(latest_record.get('state')).lower()}"
            else:
                status = "missing"
            rows.append(
                {
                    "group": group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "expected_bands": list(spec["expected_bands"]),
                    "status": status,
                    "exists": exists,
                    "active_task_state": active_records[0].get("state") if active_records else None,
                    "active_task_id": active_records[0].get("id") if active_records else None,
                    "latest_task_state": latest_record.get("state") if latest_record else None,
                    "latest_task_id": latest_record.get("id") if latest_record else None,
                }
            )
    summary = []
    for group in FACTOR_LIBRARY_GROUP_SEQUENCE:
        group_rows = [row for row in rows if row["group"] == group]
        summary.append(
            {
                "group": group,
                "total_assets": len(group_rows),
                "existing_assets": sum(bool(row["exists"]) for row in group_rows),
                "active_tasks": sum(bool(row["active_task_state"]) for row in group_rows),
                "pending_assets": sum((not row["exists"]) and (not row["active_task_state"]) for row in group_rows),
            }
        )
    recommended_cell = None
    recommended_message = "The single-factor asset library is complete."
    for item in summary:
        if item["pending_assets"] > 0:
            recommended_cell = FACTOR_LIBRARY_GROUP_TO_NOTEBOOK_CELL[item["group"]]
            recommended_message = f"Next step: Run cell {recommended_cell} to complete the {item['group']} asset group."
            break
    if recommended_cell is None:
        active_groups = [item for item in summary if item["active_tasks"] > 0]
        if active_groups:
            recommended_message = f"Active tasks are currently running: {[item['group'] for item in active_groups]}. Please wait for completion."
    return {
        "years": selected_years,
        "task_query_source": task_snapshot.get("source"),
        "task_query_error": task_snapshot.get("error"),
        "summary": summary,
        "status": rows,
        "active_tasks": [task for task in task_snapshot.get("tasks", []) if task.get("state") in FACTOR_LIBRARY_ACTIVE_TASK_STATES],
        "recommended_next_cell": recommended_cell,
        "recommended_next_message": recommended_message,
    }


def load_factor_asset_image(asset_id: str, expected_band: str, *, label: str) -> ee.Image:
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"{label} asset does not exist: {asset_id}")
    return ee.Image(asset_id).select([expected_band])


def load_factor_stack_from_assets(group: str, band_names: list[str], *, year: int | None = None) -> ee.Image:
    normalized_group = normalize_factor_library_group(group)
    stack = None
    for band_name in band_names:
        label = f"{normalized_group} factor {band_name}" if year is None else f"{normalized_group} factor {band_name} {int(year)}"
        image = load_factor_asset_image(
            factor_asset_id(normalized_group, band_name, year=year),
            band_name,
            label=label,
        )
        stack = image if stack is None else stack.addBands(image)
    return cast_stack_for_asset(ee.Image(stack).select(band_names), band_names)


def load_static_factor_stack_from_assets() -> ee.Image:
    return load_factor_stack_from_assets("static", CURRENT_STATIC_EXPORT_FEATURE_COLS)
def load_dynamic_year_stack_from_assets(year: int) -> ee.Image:
    return load_factor_stack_from_assets("dynamic_annual", FACTOR_LIBRARY_DYNAMIC_ANNUAL_BANDS, year=int(year)).select(
        FACTOR_LIBRARY_DYNAMIC_ANNUAL_BANDS,
        [EXPORT_COLUMN_ALIASES.get(name, name) for name in FACTOR_LIBRARY_DYNAMIC_ANNUAL_BANDS],
    )


def load_dynamic_factor_collection_from_assets(factor_name: str, years=None) -> ee.ImageCollection:
    normalized_factor = str(factor_name).strip()
    if normalized_factor not in FACTOR_LIBRARY_DYNAMIC_ANNUAL_BANDS:
        raise ValueError(f"Unsupported annual dynamic factor collection: {factor_name}")
    selected_years = normalize_asset_years(years if years is not None else ASSET_LIBRARY_YEARS)
    images = []
    for year in selected_years:
        image = load_factor_asset_image(
            factor_asset_id("dynamic_annual", normalized_factor, year=int(year)),
            normalized_factor,
            label=f"dynamic annual factor {normalized_factor} {int(year)}",
        ).set(
            {
                "factor_name": normalized_factor,
                "year": int(year),
                "system:time_start": ee.Date.fromYMD(int(year), 1, 1).millis(),
            }
        )
        images.append(image)
    return ee.ImageCollection.fromImages(images).sort("system:time_start")


def print_factor_asset_submit_summary(result: dict) -> None:
    group = result.get("group")
    years = result.get("years")
    submitted_records = result.get("submitted_records") or []
    status_counts = {}
    for record in submitted_records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"factor_library submit: {group}")
    if years:
        print(f"- years: {years}")
    print(f"- targets: {len(submitted_records)}")
    if status_counts:
        print(
            "- result summary: "
            + ", ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts))
        )
    else:
        print("- result summary: none")


def print_factor_library_status_summary(status_result: dict) -> None:
    years = status_result.get("years")
    summary = status_result.get("summary") or []
    task_query_error = status_result.get("task_query_error")

    print("factor_library status")
    print(f"- years: {years}")
    for item in summary:
        print(
            f"- {item['group']}: "
            f"existing={item['existing_assets']}/{item['total_assets']}, "
            f"active={item['active_tasks']}, "
            f"pending={item['pending_assets']}"
        )
    if task_query_error:
        print(f"- task query error: {task_query_error}")
    print("- collection policy: ImageCollection is only an optional read-only view for dynamic annual series, not the primary storage format.")
    print(f"- next: {status_result.get('recommended_next_message')}")

# ### 4.2 Submit Static Single-factor `v2` Assets

factor_library_static_result = submit_factor_asset_group("static")
print_factor_asset_submit_summary(factor_library_static_result)

# ### 4.3 Submit Annual Dynamic Single-factor `v2` Assets

ASSET_LIBRARY_YEARS = list(DEFAULT_ASSET_YEARS)
factor_library_annual_result = submit_factor_asset_group("dynamic_annual", years=ASSET_LIBRARY_YEARS)
print_factor_asset_submit_summary(factor_library_annual_result)

# ### 4.4 Single-factor Asset Library Status and Loading

factor_library_status = describe_factor_asset_library(years=ASSET_LIBRARY_YEARS)
print_factor_library_status_summary(factor_library_status)

PREDICTION_FACTOR_LIBRARY_ROOT = f"{RUNTIME_ASSET_ROOT}/environmental_factors/factor_library_prediction_v1"
PREDICTION_FACTOR_LIBRARY_VERSION_TAG = "v1"
PREDICTION_SMDP_ASSET_SUFFIX = "_v1"
PREDICTION_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT = f"{PREDICTION_FACTOR_LIBRARY_ROOT}/dynamic_annual"
PREDICTION_FACTOR_LIBRARY_TASK_PREFIX = "factorpred"
PREDICTION_ASSET_YEARS = [2021, 2022]
PREDICTION_FACTOR_LIBRARY_ACTIVE_TASK_STATES = ASSET_TASK_ACTIVE_STATES
PREDICTION_FACTOR_LIBRARY_GROUP_SEQUENCE = ("dynamic_annual",)
PREDICTION_FACTOR_LIBRARY_GROUP_TO_NOTEBOOK_CELL = {"dynamic_annual": "4.6"}
PREDICTION_DYNAMIC_ANNUAL_FACTOR_SPECS = [dict(item) for item in DYNAMIC_ANNUAL_FACTOR_SPECS]


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


def normalize_prediction_factor_library_group(group: str) -> str:
    normalized = str(group).strip().lower()
    if normalized not in PREDICTION_FACTOR_LIBRARY_GROUP_SEQUENCE:
        raise ValueError(
            f"Unsupported prediction asset group: {group}; allowed={list(PREDICTION_FACTOR_LIBRARY_GROUP_SEQUENCE)}"
        )
    return normalized


def prediction_factor_asset_root(group: str) -> str:
    normalized = normalize_prediction_factor_library_group(group)
    if normalized == "dynamic_annual":
        return PREDICTION_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT
    raise ValueError(f"Unsupported prediction asset group: {group}")


def prediction_factor_group_specs(group: str) -> list[dict]:
    normalized = normalize_prediction_factor_library_group(group)
    if normalized == "dynamic_annual":
        return [dict(item) for item in PREDICTION_DYNAMIC_ANNUAL_FACTOR_SPECS]
    raise ValueError(f"Unsupported prediction asset group: {group}")


def prediction_factor_asset_name(group: str, factor_name: str, year: int | None = None) -> str:
    normalize_prediction_factor_library_group(group)
    normalized_factor = str(factor_name).strip()
    if year is None:
        raise ValueError("Prediction annual dynamic asset name requires year.")
    if normalized_factor == "SMDP":
        return f"{normalized_factor}_{int(year)}{PREDICTION_SMDP_ASSET_SUFFIX}"
    return f"{normalized_factor}_{int(year)}_{PREDICTION_FACTOR_LIBRARY_VERSION_TAG}"


def prediction_factor_asset_id(group: str, factor_name: str, year: int | None = None) -> str:
    return (
        f"{prediction_factor_asset_root(group).rstrip('/')}"
        f"/{prediction_factor_asset_name(group, factor_name, year=year)}"
    )


def prediction_factor_task_description(group: str, factor_name: str, year: int | None = None) -> str:
    normalize_prediction_factor_library_group(group)
    normalized_factor = str(factor_name).strip()
    if year is None:
        raise ValueError("Prediction annual dynamic task description requires year.")
    suffix = PREDICTION_SMDP_ASSET_SUFFIX.lstrip("_") if normalized_factor == "SMDP" else PREDICTION_FACTOR_LIBRARY_VERSION_TAG
    return (
        f"{PREDICTION_FACTOR_LIBRARY_TASK_PREFIX}_dynamic_annual_"
        f"{normalized_factor}_{int(year)}_{suffix}"
    )


def prediction_factor_library_root_parent_asset_id() -> str:
    return PREDICTION_FACTOR_LIBRARY_ROOT.rsplit('/', 1)[0]


def ensure_prediction_factor_library_asset_roots() -> list[dict]:
    created = []
    created.append(
        ensure_folder_asset_exists(
            PREDICTION_FACTOR_LIBRARY_ROOT,
            prediction_factor_library_root_parent_asset_id(),
        )
    )
    created.append(
        ensure_folder_asset_exists(
            PREDICTION_DYNAMIC_ANNUAL_FACTOR_ASSET_ROOT,
            PREDICTION_FACTOR_LIBRARY_ROOT,
        )
    )
    return created


def build_prediction_factor_asset_specs(group: str = "dynamic_annual", years=None, include_image: bool = False) -> list[dict]:
    normalized_group = normalize_prediction_factor_library_group(group)
    selected_years = normalize_prediction_asset_years(years)
    specs = []
    for spec in prediction_factor_group_specs(normalized_group):
        factor_name = spec["factor_name"]
        band_name = spec["band_name"]
        for year in selected_years:
            row = {
                "group": normalized_group,
                "factor_name": factor_name,
                "band_name": band_name,
                "year": int(year),
                "asset_id": prediction_factor_asset_id(normalized_group, factor_name, year=int(year)),
                "task_description": prediction_factor_task_description(normalized_group, factor_name, year=int(year)),
                "expected_bands": [band_name],
                "years_covered": [int(year)],
            }
            if include_image:
                row["image"] = build_dynamic_annual_factor_image(factor_name, int(year))
            specs.append(row)
    return specs


def relevant_prediction_factor_task_descriptions(years=None) -> set[str]:
    return {
        spec["task_description"]
        for spec in build_prediction_factor_asset_specs("dynamic_annual", years=years, include_image=False)
    }


def fetch_prediction_factor_library_task_snapshot(years=None) -> dict:
    selected_years = normalize_prediction_asset_years(years)
    return fetch_task_snapshot_by_description(
        ee.data,
        relevant_prediction_factor_task_descriptions(selected_years),
        fallback_label="prediction_factor_library",
    )


def active_prediction_factor_library_tasks_for_description(task_snapshot: dict, description: str) -> list[dict]:
    return [
        task
        for task in task_snapshot.get("index", {}).get(description, [])
        if task.get("state") in PREDICTION_FACTOR_LIBRARY_ACTIVE_TASK_STATES
    ]


def build_prediction_factor_asset_export_image(spec: dict) -> ee.Image:
    return build_dynamic_annual_factor_image(spec["factor_name"], int(spec["year"]))


def submit_prediction_factor_asset_group(group: str = "dynamic_annual", years=None) -> dict:
    normalized_group = normalize_prediction_factor_library_group(group)
    selected_years = normalize_prediction_asset_years(years)
    folder_records = ensure_prediction_factor_library_asset_roots()
    task_snapshot = fetch_prediction_factor_library_task_snapshot(years=selected_years)
    if task_snapshot.get("error"):
        raise RuntimeError(
            "Failed to read the prediction_factor_library task list. To avoid duplicate submissions, this execution has been aborted."
            f" error={task_snapshot['error']}"
        )
    submitted_records = []
    for spec in build_prediction_factor_asset_specs(normalized_group, years=selected_years, include_image=False):
        if asset_exists(spec["asset_id"]):
            submitted_records.append(
                {
                    "group": normalized_group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "status": "skipped_existing",
                }
            )
            continue
        active_records = active_prediction_factor_library_tasks_for_description(task_snapshot, spec["task_description"])
        if active_records:
            submitted_records.append(
                {
                    "group": normalized_group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "status": "skipped_active",
                    "task_id": active_records[0].get("id"),
                    "task_state": active_records[0].get("state"),
                }
            )
            continue
        task = start_image_asset_export_task(
            build_prediction_factor_asset_export_image(spec),
            asset_id=spec["asset_id"],
            description=spec["task_description"],
            expected_bands=list(spec["expected_bands"]),
            route="prediction_factor_library",
            group=normalized_group,
            year=spec.get("year"),
        )
        task_status = task.status() or {}
        submitted_records.append(
            {
                "group": normalized_group,
                "factor_name": spec["factor_name"],
                "year": spec.get("year"),
                "asset_id": spec["asset_id"],
                "task_description": spec["task_description"],
                "status": task_status.get("state"),
                "task_id": task_status.get("id"),
            }
        )
    return {
        "group": normalized_group,
        "years": selected_years,
        "folder_records": folder_records,
        "submitted_records": submitted_records,
    }


def print_prediction_factor_asset_submit_summary(result: dict) -> None:
    group = result.get("group")
    years = result.get("years")
    submitted_records = result.get("submitted_records") or []
    status_counts = {}
    for record in submitted_records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    print(f"prediction_factor_library submit: {group}")
    print(f"- years: {years}")
    print(f"- targets: {len(submitted_records)}")
    if status_counts:
        print("- result summary: " + ", ".join(f"{key}={status_counts[key]}" for key in sorted(status_counts)))
    else:
        print("- result summary: none")


def describe_prediction_factor_asset_library(years=None) -> dict:
    selected_years = normalize_prediction_asset_years(years)
    task_snapshot = fetch_prediction_factor_library_task_snapshot(years=selected_years)
    rows = []
    for group in PREDICTION_FACTOR_LIBRARY_GROUP_SEQUENCE:
        for spec in build_prediction_factor_asset_specs(group, years=selected_years, include_image=False):
            active_records = active_prediction_factor_library_tasks_for_description(task_snapshot, spec["task_description"])
            latest_records = task_snapshot.get("index", {}).get(spec["task_description"], [])
            latest_record = latest_records[0] if latest_records else None
            exists = asset_exists(spec["asset_id"])
            if exists:
                status = "exists"
            elif active_records:
                status = f"task_{str(active_records[0].get('state')).lower()}"
            elif latest_record and latest_record.get("state") in {"FAILED", "CANCELLED"}:
                status = f"missing_last_{str(latest_record.get('state')).lower()}"
            else:
                status = "missing"
            rows.append(
                {
                    "group": group,
                    "factor_name": spec["factor_name"],
                    "year": spec.get("year"),
                    "asset_id": spec["asset_id"],
                    "task_description": spec["task_description"],
                    "expected_bands": list(spec["expected_bands"]),
                    "status": status,
                    "exists": exists,
                    "active_task_state": active_records[0].get("state") if active_records else None,
                    "active_task_id": active_records[0].get("id") if active_records else None,
                    "latest_task_state": latest_record.get("state") if latest_record else None,
                    "latest_task_id": latest_record.get("id") if latest_record else None,
                }
            )
    summary = []
    for group in PREDICTION_FACTOR_LIBRARY_GROUP_SEQUENCE:
        group_rows = [row for row in rows if row["group"] == group]
        summary.append(
            {
                "group": group,
                "total_assets": len(group_rows),
                "existing_assets": sum(bool(row["exists"]) for row in group_rows),
                "active_tasks": sum(bool(row["active_task_state"]) for row in group_rows),
                "pending_assets": sum((not row["exists"]) and (not row["active_task_state"]) for row in group_rows),
            }
        )
    recommended_cell = None
    recommended_message = "The prediction asset library is complete."
    for item in summary:
        if item["pending_assets"] > 0:
            recommended_cell = PREDICTION_FACTOR_LIBRARY_GROUP_TO_NOTEBOOK_CELL[item["group"]]
            recommended_message = f"Next step: Run cell {recommended_cell} to complete the prediction asset library."
            break
    if recommended_cell is None:
        active_groups = [item for item in summary if item["active_tasks"] > 0]
        if active_groups:
            recommended_message = f"Active tasks are currently running: {[item['group'] for item in active_groups]}. Please wait for completion."
    return {
        "years": selected_years,
        "task_query_source": task_snapshot.get("source"),
        "task_query_error": task_snapshot.get("error"),
        "summary": summary,
        "status": rows,
        "active_tasks": [
            task for task in task_snapshot.get("tasks", [])
            if task.get("state") in PREDICTION_FACTOR_LIBRARY_ACTIVE_TASK_STATES
        ],
        "recommended_next_cell": recommended_cell,
        "recommended_next_message": recommended_message,
    }


def print_prediction_factor_library_status_summary(status_result: dict) -> None:
    years = status_result.get("years")
    summary = status_result.get("summary") or []
    task_query_error = status_result.get("task_query_error")

    print("prediction_factor_library status")
    print(f"- years: {years}")
    for item in summary:
        print(
            f"- {item['group']}: "
            f"existing={item['existing_assets']}/{item['total_assets']}, "
            f"active={item['active_tasks']}, "
            f"pending={item['pending_assets']}"
        )
    if task_query_error:
        print(f"- task query error: {task_query_error}")
    print("- static source: factor_library_v2/static")
    print("- next: " + str(status_result.get("recommended_next_message")))


def load_prediction_dynamic_factor_from_assets(factor_name: str, year: int) -> ee.Image:
    normalized_factor = str(factor_name).strip()
    factor_spec = next(
        (item for item in PREDICTION_DYNAMIC_ANNUAL_FACTOR_SPECS if item["factor_name"] == normalized_factor),
        None,
    )
    if factor_spec is None:
        raise ValueError(f"Unsupported prediction annual dynamic factor: {factor_name}")
    selected_year = normalize_prediction_asset_years([year])[0]
    return load_factor_asset_image(
        prediction_factor_asset_id("dynamic_annual", normalized_factor, year=selected_year),
        factor_spec["band_name"],
        label=f"prediction dynamic annual {normalized_factor} {selected_year}",
    )


def load_prediction_dynamic_year_stack_from_assets(year: int) -> ee.Image:
    selected_year = normalize_prediction_asset_years([year])[0]
    stack = None
    for spec in PREDICTION_DYNAMIC_ANNUAL_FACTOR_SPECS:
        band_image = load_prediction_dynamic_factor_from_assets(spec["factor_name"], selected_year)
        stack = band_image if stack is None else stack.addBands(band_image)
    return ee.Image(stack).select([item["band_name"] for item in PREDICTION_DYNAMIC_ANNUAL_FACTOR_SPECS])


def write_feature_manifest() -> None:
    PYTHON_ENV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "runtime_project_id": PROJECT_ID,
        "runtime_asset_root": RUNTIME_ASSET_ROOT,
        "public_asset_root": PUBLIC_ASSET_ROOT,
        "factor_library_root": FACTOR_LIBRARY_ROOT,
        "prediction_factor_library_root": PREDICTION_FACTOR_LIBRARY_ROOT,
        "default_asset_years": list(DEFAULT_ASSET_YEARS),
        "prediction_asset_years": list(PREDICTION_ASSET_YEARS),
        "factor_library_status": factor_library_status,
        "prediction_factor_library_status": prediction_factor_library_status,
    }
    FEATURE_MANIFEST_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote: {FEATURE_MANIFEST_PATH}")

# ### 4.5 Submit 2021-2022 Prediction Annual Dynamic Single-factor Assets

prediction_factor_library_annual_result = submit_prediction_factor_asset_group(
    "dynamic_annual",
    years=PREDICTION_ASSET_YEARS,
)
print_prediction_factor_asset_submit_summary(prediction_factor_library_annual_result)

# ### 4.6 Prediction Asset Library Status

prediction_factor_library_status = describe_prediction_factor_asset_library(
    years=PREDICTION_ASSET_YEARS,
)
print_prediction_factor_library_status_summary(prediction_factor_library_status)
write_feature_manifest()
