#!/usr/bin/env python3
"""Step_01 Total_features_sample workflow."""

# # Build `Total_features_sample.csv`

# ## 1. GEE Authentication and Initialization

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()
from pathlib import Path
import json
import sqlite3
import pyproj
from shapely import wkb
from shapely.ops import transform as shapely_transform

import ee
try:
    import geemap
except ImportError:
    geemap = None
import numpy as np
import pandas as pd

from dataset_construction.common import python_output_root, resolve_dataset_construction_root
from dataset_construction.ee_runtime import asset_exists, initialize_ee, runtime_asset_root
from dataset_construction.factor_library import (
    DYNAMIC_ANNUAL_FACTOR_SPECS,
    RANK_DYNAMIC_FEATURE_COLUMNS,
    RANK_FEATURE_OUTPUT_COLUMNS,
    RANK_STATIC_FEATURE_COLUMNS,
    STATIC_FACTOR_SPECS,
    cast_stack_for_asset,
    factor_asset_id,
    factor_library_root,
)

PROJECT_ID = initialize_ee(context="build_total_features workflow")
RUNTIME_ASSET_ROOT = runtime_asset_root(PROJECT_ID)
REPO = resolve_dataset_construction_root(Path.cwd())
PYTHON_OUTPUT_ROOT = python_output_root(REPO)

# ## 2. Define Paths and Constants

SAMPLE_OUTPUT_DIR = PYTHON_OUTPUT_ROOT / "02_sample_tables"
ENV_OUTPUT_DIR = PYTHON_OUTPUT_ROOT / "03_environmental_data"
GOOGLE_DRIVE_LOCAL_DIR = ENV_OUTPUT_DIR / "Google_drive"
DIRECT_LOCAL_EXPORT_DIR = ENV_OUTPUT_DIR / "Direct_download"

SLOPE_UNITS_PATH = REPO / "inputs" / "QTP_SlopeUnits.gpkg"
RANK_SAMPLE_PATH = SAMPLE_OUTPUT_DIR / "RTS_Total_sample_rank.csv"
RANK_FINAL_PATH = ENV_OUTPUT_DIR / "Total_features_sample.csv"
CLOUD_MANIFEST_PATH = GOOGLE_DRIVE_LOCAL_DIR / "rank_cloud_export_manifest.csv"
DIRECT_LOCAL_MANIFEST_PATH = DIRECT_LOCAL_EXPORT_DIR / "rank_direct_download_manifest.csv"

TASK_EXPORT_FOLDER = "RTS_Susceptibility"
RANK_TASK_DESCRIPTION_PREFIX = "export_Total_features_sample"
DRIVE_READONLY_SCOPE = "https://www.googleapis.com/auth/drive.readonly"
SAMPLE_SCALE = 100
TABULAR_NODATA_VALUE = -9999.0
FEATURE_PHYSICAL_QUALITY_REPORT_PATH = ENV_OUTPUT_DIR / "feature_physical_quality_report.json"
# Bounds are deliberately limited to invariants implied by the feature
# definitions.  TWI and WSTmax use broad physical envelopes rather than
# distribution-derived clipping so legitimate regional extremes are retained.
FEATURE_PHYSICAL_BOUNDS = {
    "Slope": (0.0, 90.0),
    "Slope_std": (0.0, 90.0),
    "Northness": (-1.0, 1.0),
    "Eastness": (-1.0, 1.0),
    "TWI": (-100.0, 100.0),
    "Local_Relief": (0.0, None),
    "Soil": (0.0, 100.0),
    "GIC": (0.0, None),
    "Vegetation_type": (0.0, 14.0),
    "TDD": (0.0, None),
    "SMDP": (0.0, None),
    "Precipitation": (0.0, None),
    "WSTmax": (-90.0, 70.0),
    "WSRH": (0.0, 100.0),
    "NDVI": (-1.0, 1.0),
    "Summer_SoilMoisture": (0.0, 1.0),
}
DIRECT_LOCAL_TIMEOUT = 600
CLOUD_RANK_BATCH_ROWS = 5000
CLOUD_DISABLE_BATCH_SPLIT = True
DOWNLOAD_SEARCH_DIRS = [
    GOOGLE_DRIVE_LOCAL_DIR,
    Path.home() / "Downloads",
]
FACTOR_LIBRARY_ROOT = factor_library_root(RUNTIME_ASSET_ROOT)

RANK_META_COLUMNS = [
    "Sample_ID",
    "Longitude",
    "Latitude",
    "Feature_Year",
    "Cluster",
    "Basin_cluster",
    "Label_Cls",
    "Label_Rank",
]
RANK_OUTPUT_COLUMNS = [*RANK_META_COLUMNS, *RANK_FEATURE_OUTPUT_COLUMNS]


def build_feature_physical_quality_report(df: pd.DataFrame) -> dict:
    """Check invariant physical ranges without clipping legitimate values.

    The tabular sentinel is treated as an explicit missing value and is
    reported separately.  Any non-sentinel value outside a feature's broad
    physical support fails the quality gate, which prevents masked-pixel
    contamination from silently reaching the training table.
    """
    report = {
        "status": "pass",
        "row_count": int(len(df)),
        "nodata_value": TABULAR_NODATA_VALUE,
        "nodata_counts": {},
        "physical_invalid_by_feature": {},
        "physical_invalid_rows": 0,
        "physical_invalid_sample_preview": [],
    }
    invalid_masks = []
    for feature_name, (lower, upper) in FEATURE_PHYSICAL_BOUNDS.items():
        if feature_name not in df.columns:
            continue
        values = pd.to_numeric(df[feature_name], errors="coerce")
        nodata_mask = values.eq(TABULAR_NODATA_VALUE)
        report["nodata_counts"][feature_name] = int(nodata_mask.sum())
        invalid = values.isna() | ~np.isfinite(values)
        invalid &= ~nodata_mask
        if lower is not None:
            invalid |= values.lt(float(lower)) & ~nodata_mask
        if upper is not None:
            invalid |= values.gt(float(upper)) & ~nodata_mask
        invalid_masks.append(invalid)
        report["physical_invalid_by_feature"][feature_name] = int(invalid.sum())

    if invalid_masks:
        invalid_frame = pd.concat(invalid_masks, axis=1)
        any_invalid = invalid_frame.any(axis=1)
        report["physical_invalid_rows"] = int(any_invalid.sum())
        if any_invalid.any() and "Sample_ID" in df.columns:
            report["physical_invalid_sample_preview"] = (
                df.loc[any_invalid, "Sample_ID"].astype(str).drop_duplicates().head(10).tolist()
            )
    if report["physical_invalid_rows"]:
        report["status"] = "fail"
    return report

print(
    {
        "rank_sample_table": str(RANK_SAMPLE_PATH),
        "final_training_table": str(RANK_FINAL_PATH),
        "factor_library_root": FACTOR_LIBRARY_ROOT,
        "task_folder": TASK_EXPORT_FOLDER,
        "manifest": str(CLOUD_MANIFEST_PATH),
    }
)

# ## 3. GEE Image Stacking and Sampling Functions

def load_factor_asset_image(asset_id: str, expected_band: str, *, label: str) -> ee.Image:
    if not asset_exists(asset_id):
        raise FileNotFoundError(f"{label} asset does not exist: {asset_id}")
    return ee.Image(asset_id).select([expected_band])


def load_static_factor_stack_from_assets() -> ee.Image:
    stack = None
    from_vals = [0, 2, 3, 4, 5, 6, 7, 8, 9]
    cos_vals = [0.0, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071, 0.0, 0.7071]
    sin_vals = [0.0, 0.0, 0.7071, 1.0, 0.7071, 0.0, -0.7071, -1.0, -0.7071]

    for spec in STATIC_FACTOR_SPECS:
        actual_asset_factor = spec.get("source_factor", spec["factor_name"])
        actual_asset_band = actual_asset_factor if actual_asset_factor in ("Aspect", "Slope") else spec["band_name"]
        image = load_factor_asset_image(
            factor_asset_id(RUNTIME_ASSET_ROOT, "static", actual_asset_factor),
            actual_asset_band,
            label=f"static factor {spec['factor_name']}",
        )
        if spec["factor_name"] == "Northness":
            image = image.remap(from_vals, cos_vals, 0.0).rename(["Northness"]).toFloat()
        elif spec["factor_name"] == "Eastness":
            image = image.remap(from_vals, sin_vals, 0.0).rename(["Eastness"]).toFloat()
        elif spec["factor_name"] == "Slope_std":
            image = image.rename(["Slope_std"]).toFloat()

        stack = image if stack is None else stack.addBands(image)
    return cast_stack_for_asset(ee.Image(stack).select(RANK_STATIC_FEATURE_COLUMNS), RANK_STATIC_FEATURE_COLUMNS)


def load_dynamic_year_stack_from_assets(year: int) -> ee.Image:
    stack = None
    for spec in DYNAMIC_ANNUAL_FACTOR_SPECS:
        image = load_factor_asset_image(
            factor_asset_id(RUNTIME_ASSET_ROOT, "dynamic_annual", spec["factor_name"], year=int(year)),
            spec["band_name"],
            label=f"dynamic annual factor {spec['factor_name']} {int(year)}",
        )
        stack = image if stack is None else stack.addBands(image)
    return cast_stack_for_asset(ee.Image(stack).select(RANK_DYNAMIC_FEATURE_COLUMNS), RANK_DYNAMIC_FEATURE_COLUMNS)


def build_feature_year_stack_from_factor_library(year: int) -> ee.Image:
    raw_stack = (
        load_static_factor_stack_from_assets()
        .addBands(load_dynamic_year_stack_from_assets(int(year)))
        .select(RANK_FEATURE_OUTPUT_COLUMNS)
    )
    return cast_stack_for_asset(raw_stack, RANK_FEATURE_OUTPUT_COLUMNS)


def get_feature_year_stack(year: int) -> ee.Image:
    """Return a masked stack for polygon reduction.

    The region-reduction path must retain the source masks until
    ``reduceRegions`` has calculated the polygon statistics.  Filling masked
    pixels with ``-9999`` before a mean/stdDev reducer mixes NoData into valid
    pixels and creates synthetic values such as ``-2000`` or ``-9000``.  The
    table finalization path converts genuinely missing properties to the
    configured tabular sentinel after reduction.
    """
    return build_feature_year_stack_from_factor_library(int(year))

# ## 4. Local IO and Data Processing Helper Functions

def ensure_export_dirs() -> None:
    GOOGLE_DRIVE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    DIRECT_LOCAL_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    ENV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def require_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{table_name} missing required columns: {missing}")


def require_non_missing_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing_mask = df[required].isna().any(axis=1)
    if missing_mask.any():
        sample_preview = (
            df.loc[missing_mask, "Sample_ID"].astype(str).drop_duplicates().head(10).tolist()
            if "Sample_ID" in df.columns
            else df.index[missing_mask].astype(str).tolist()[:10]
        )
        raise ValueError(
            f"{table_name} required metadata columns contain missing values; missing_rows={int(missing_mask.sum())}; "
            f"sample_id_preview={sample_preview}"
        )


def cast_nullable_int_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(TABULAR_NODATA_VALUE).round().astype(int)
    return df


def cleanup_sampled_frame(df: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [col for col in ["longitude", "latitude", "system:index", ".geo", "geo"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df


def load_rank_points() -> pd.DataFrame:
    rank_raw = pd.read_csv(RANK_SAMPLE_PATH)
    required = [
        "Sample_ID",
        "Longitude",
        "Latitude",
        "Feature_Year",
        "Label_Cls",
        "Label_Rank",
        "Cluster",
        "Basin_cluster",
    ]
    require_columns(rank_raw, required, "RTS_Total_sample_rank.csv")
    require_non_missing_columns(rank_raw, required, "RTS_Total_sample_rank.csv")
    rank_points = rank_raw.copy()
    for col in ["Feature_Year", "Label_Cls", "Label_Rank", "Cluster", "Basin_cluster"]:
        rank_points[col] = pd.to_numeric(rank_points[col], errors="raise").astype(int)
    if rank_points.empty:
        raise ValueError("RTS_Total_sample_rank.csv is empty, cannot proceed with export.")
    return rank_points


def read_rank_years_from_samples() -> list[int]:
    rank_years = pd.read_csv(RANK_SAMPLE_PATH, usecols=["Feature_Year"])["Feature_Year"]
    years = sorted({int(year) for year in pd.to_numeric(rank_years, errors="coerce").dropna().tolist()})
    if not years:
        raise ValueError("No valid Feature_Year found in RTS_Total_sample_rank.csv.")
    return years


def required_factor_library_asset_ids(years: list[int]) -> list[str]:
    asset_ids = [
        factor_asset_id(RUNTIME_ASSET_ROOT, "static", spec.get("source_factor", spec["factor_name"]))
        for spec in STATIC_FACTOR_SPECS
    ]
    for year in years:
        asset_ids.extend(
            factor_asset_id(RUNTIME_ASSET_ROOT, "dynamic_annual", spec["factor_name"], year=int(year))
            for spec in DYNAMIC_ANNUAL_FACTOR_SPECS
        )
    deduped = []
    for asset_id in asset_ids:
        if asset_id not in deduped:
            deduped.append(asset_id)
    return deduped


def validate_required_factor_library_assets(years: list[int]) -> list[str]:
    required_asset_ids = required_factor_library_asset_ids(years)
    missing = [asset_id for asset_id in required_asset_ids if not asset_exists(asset_id)]
    if missing:
        raise FileNotFoundError(
            "Sample extraction depends on factor_library_v2 single-factor assets. Please complete the asset preparation in environmental_rasters_geemap.ipynb first."
            f" missing_assets={missing}"
        )
    return required_asset_ids


def load_slope_unit_geometries(sample_ids: list[str]) -> dict[str, object]:
    if not SLOPE_UNITS_PATH.exists():
        raise FileNotFoundError(f"Slope units GPKG not found: {SLOPE_UNITS_PATH}")
    conn = sqlite3.connect(str(SLOPE_UNITS_PATH))
    cursor = conn.cursor()

    transformer = pyproj.Transformer.from_crs(
        "+proj=aea +lat_1=27.5 +lat_2=37.5 +lat_0=0 +lon_0=90 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs",
        "EPSG:4326",
        always_xy=True
    )

    su_geoms = {}
    deduped_ids = sorted(set(sample_ids))
    chunk_size = 900
    for i in range(0, len(deduped_ids), chunk_size):
        chunk = deduped_ids[i:i + chunk_size]
        placeholders = ','.join('?' for _ in chunk)
        cursor.execute(f"SELECT su_id, geom FROM qtp_slope_units WHERE su_id IN ({placeholders})", chunk)
        for su_id, geom_blob in cursor.fetchall():
            flags = geom_blob[3]
            envelope_type = (flags >> 1) & 0x07
            env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
            header_len = 8 + env_sizes.get(envelope_type, 0)
            geom_albers = wkb.loads(geom_blob[header_len:])
            geom_wgs84 = shapely_transform(transformer.transform, geom_albers)
            su_geoms[su_id] = geom_wgs84

    conn.close()
    return su_geoms


def extract_slope_units_features_for_year(
    points_year: pd.DataFrame,
    year: int,
    image_stack: ee.Image,
    su_geoms: dict[str, object],
    batch_size: int = 250,
) -> pd.DataFrame:
    all_rows = []
    records = points_year.to_dict(orient="records")
    total = len(records)
    continuous_bands = [b for b in RANK_FEATURE_OUTPUT_COLUMNS if b != "Vegetation_type" and b != "Slope_std"]
    continuous_stack = image_stack.select(continuous_bands)
    slope_stack = image_stack.select(["Slope_std"])
    veg_stack = load_factor_asset_image(
        factor_asset_id(RUNTIME_ASSET_ROOT, "static", "Vegetation_type"),
        "Vegetation_type",
        label="Vegetation_type",
    ).toInt16()

    for start_idx in range(0, total, batch_size):
        chunk = records[start_idx:start_idx + batch_size]
        features = []
        for row in chunk:
            su_id = str(row["Sample_ID"])
            geom = su_geoms.get(su_id)
            if geom is not None:
                ee_geom = ee.Geometry(geom.__geo_interface__)
                props = {"Sample_ID": su_id, "Feature_Year": int(year)}
                features.append(ee.Feature(ee_geom, props))

        if not features:
            continue

        fc = ee.FeatureCollection(features)
        reduced_cont = continuous_stack.reduceRegions(collection=fc, reducer=ee.Reducer.mean(), scale=SAMPLE_SCALE)
        reduced_slope_std = slope_stack.reduceRegions(collection=fc, reducer=ee.Reducer.stdDev(), scale=SAMPLE_SCALE)
        reduced_veg = veg_stack.reduceRegions(collection=fc, reducer=ee.Reducer.mode(), scale=SAMPLE_SCALE)

        cont_info = None
        slope_std_info = None
        veg_info = None
        for attempt in range(1, 5):
            try:
                cont_info = reduced_cont.getInfo()
                slope_std_info = reduced_slope_std.getInfo()
                veg_info = reduced_veg.getInfo()
                break
            except Exception as exc:
                if attempt == 4:
                    raise RuntimeError(f"[{year}] GEE batch [{start_idx}:{start_idx+len(chunk)}] failed after 4 retries: {exc}") from exc
                import time
                time.sleep(3 * attempt)

        cont_props = {f["properties"]["Sample_ID"]: f["properties"] for f in cont_info.get("features", [])}
        slope_std_props = {f["properties"]["Sample_ID"]: f["properties"].get("stdDev", f["properties"].get("Slope_std")) for f in slope_std_info.get("features", [])}
        veg_props = {f["properties"]["Sample_ID"]: f["properties"].get("mode", f["properties"].get("Vegetation_type")) for f in veg_info.get("features", [])}

        for su_id, props in cont_props.items():
            if su_id in slope_std_props and slope_std_props[su_id] is not None:
                props["Slope_std"] = slope_std_props[su_id]
            if su_id in veg_props and veg_props[su_id] is not None:
                props["Vegetation_type"] = veg_props[su_id]
            all_rows.append(props)

    return pd.DataFrame(all_rows)


def start_table_export_task(
    collection: ee.FeatureCollection,
    *,
    description: str,
    file_name_prefix: str,
    selectors: list[str],
) -> ee.batch.Task:
    task = ee.batch.Export.table.toDrive(
        collection=collection,
        description=description,
        folder=TASK_EXPORT_FOLDER,
        fileNamePrefix=file_name_prefix,
        fileFormat="CSV",
        selectors=selectors,
    )
    task.start()
    return task


def get_task_status_snapshot(task: ee.batch.Task) -> dict:
    status = task.status() or {}
    return {
        "task_id": status.get("id"),
        "operation_name": status.get("name"),
        "task_state": status.get("state"),
        "creation_timestamp_ms": status.get("creation_timestamp_ms"),
        "update_timestamp_ms": status.get("update_timestamp_ms"),
        "start_timestamp_ms": status.get("start_timestamp_ms"),
        "task_type": status.get("task_type"),
    }


def build_rank_batches(points_df: pd.DataFrame, *, batch_size: int) -> list[dict]:
    batches = []
    for year, pts_year in points_df.groupby("Feature_Year", sort=True):
        pts_year = pts_year.sort_values(["Feature_Year", "Sample_ID"]).reset_index(drop=True)
        if CLOUD_DISABLE_BATCH_SPLIT:
            batches.append(
                {
                    "year": int(year),
                    "batch_index": 1,
                    "points_df": pts_year.copy(),
                    "expected_rows": len(pts_year),
                    "year_rows_total": len(pts_year),
                    "batch_size": len(pts_year),
                    "batch_start": 1,
                    "batch_end": len(pts_year),
                }
            )
            continue
        for batch_index, start in enumerate(range(0, len(pts_year), int(batch_size)), start=1):
            end = min(start + int(batch_size), len(pts_year))
            batches.append(
                {
                    "year": int(year),
                    "batch_index": batch_index,
                    "points_df": pts_year.iloc[start:end].copy(),
                    "expected_rows": end - start,
                    "year_rows_total": len(pts_year),
                    "batch_size": int(batch_size),
                    "batch_start": start + 1,
                    "batch_end": end,
                }
            )
    return batches


def rank_batch_suffix(batch: dict, *, batch_size: int | None = None) -> str:
    if CLOUD_DISABLE_BATCH_SPLIT:
        return str(int(batch["year"]))
    threshold = int(batch_size if batch_size is not None else batch.get("batch_size", CLOUD_RANK_BATCH_ROWS))
    if int(batch["year_rows_total"]) > threshold:
        return f"{int(batch['year'])}_part{int(batch['batch_index']):02d}"
    return str(int(batch["year"]))


def make_manifest_record(batch: dict, *, file_name: str, task_description: str, drive_prefix: str, task: ee.batch.Task) -> dict:
    return {
        "route": "rank",
        "year": int(batch["year"]),
        "batch_index": int(batch["batch_index"]),
        "expected_rows": int(batch["expected_rows"]),
        "file_name": file_name,
        "task_description": task_description,
        "drive_prefix": drive_prefix,
        **get_task_status_snapshot(task),
    }


def write_cloud_manifest(records: list[dict]) -> pd.DataFrame:
    ensure_export_dirs()
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No cloud manifest records to write.")
    frame = frame.sort_values(["year", "batch_index"]).reset_index(drop=True)
    frame.to_csv(CLOUD_MANIFEST_PATH, index=False)
    return frame


def read_cloud_manifest() -> pd.DataFrame:
    if not CLOUD_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"cloud manifest does not exist: {CLOUD_MANIFEST_PATH}")
    manifest = pd.read_csv(CLOUD_MANIFEST_PATH)
    if manifest.empty:
        raise ValueError(f"cloud manifest is empty: {CLOUD_MANIFEST_PATH}")
    for col in ["year", "batch_index", "expected_rows", "creation_timestamp_ms", "update_timestamp_ms", "start_timestamp_ms"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce").astype("Int64")
    return manifest


def latest_export_task_lookup() -> dict[str, dict]:
    latest = {}
    for task in ee.data.getTaskList():
        description = str(task.get("description") or "")
        if not description.startswith(RANK_TASK_DESCRIPTION_PREFIX):
            continue
        created = int(task.get("creation_timestamp_ms") or -1)
        previous = latest.get(description)
        previous_created = int(previous.get("creation_timestamp_ms") or -1) if previous else -1
        if previous is None or created >= previous_created:
            latest[description] = task
    return latest


def refresh_cloud_manifest_for_finalize(manifest: pd.DataFrame) -> pd.DataFrame:
    refreshed = manifest.copy()
    task_lookup = latest_export_task_lookup()
    for idx, row in refreshed.iterrows():
        task = task_lookup.get(str(row.get("task_description") or ""))
        if not task:
            continue
        refreshed.at[idx, "task_id"] = task.get("id")
        refreshed.at[idx, "operation_name"] = task.get("name")
        refreshed.at[idx, "task_state"] = task.get("state")
        refreshed.at[idx, "creation_timestamp_ms"] = task.get("creation_timestamp_ms")
        refreshed.at[idx, "update_timestamp_ms"] = task.get("update_timestamp_ms")
        refreshed.at[idx, "start_timestamp_ms"] = task.get("start_timestamp_ms")
        refreshed.at[idx, "task_type"] = task.get("task_type")
    refreshed.to_csv(CLOUD_MANIFEST_PATH, index=False)
    return read_cloud_manifest()


def build_drive_service():
    try:
        import google.auth
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise ImportError(
            "Automatic downloading requires `google-api-python-client` and `google-auth`; please install requirements.txt first."
        ) from exc

    credentials, _ = google.auth.default(scopes=[DRIVE_READONLY_SCOPE])
    if credentials is None:
        raise RuntimeError(
            "Could not retrieve Google Drive credentials. Please run `gcloud auth application-default login --scopes=https://www.googleapis.com/auth/drive.readonly` first."
        )
    if not credentials.valid and getattr(credentials, "refresh_token", None):
        credentials.refresh(Request())
    return build("drive", "v3", credentials=credentials, cache_discovery=False)


def resolve_drive_folder_id(service, *, folder_name: str, folder_id: str | None = None) -> str:
    if folder_id:
        return str(folder_id)
    escaped_folder_name = folder_name.replace("'", "\\'")
    query = (
        "mimeType='application/vnd.google-apps.folder' and trashed=false and "
        f"name='{escaped_folder_name}'"
    )
    response = service.files().list(
        q=query,
        pageSize=10,
        fields="files(id,name,modifiedTime)",
        orderBy="modifiedTime desc",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(f"Export folder not found in Google Drive: {folder_name}")
    return str(files[0]["id"])


def find_drive_file_metadata(service, *, folder_id: str, file_name: str) -> dict:
    escaped_name = file_name.replace("'", "\\'")
    query = f"trashed=false and name='{escaped_name}' and '{folder_id}' in parents"
    response = service.files().list(
        q=query,
        pageSize=10,
        fields="files(id,name,size,modifiedTime,parents)",
        orderBy="modifiedTime desc",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute()
    files = response.get("files", [])
    if not files:
        raise FileNotFoundError(f"Export file not found in Google Drive: {file_name}")
    return dict(files[0])


def download_drive_file(service, *, file_id: str, dest_path: Path, overwrite: bool = False) -> Path:
    try:
        from googleapiclient.http import MediaIoBaseDownload
    except ImportError as exc:
        raise ImportError(
            "Automatic downloading requires `google-api-python-client`; please install requirements.txt first."
        ) from exc

    dest_path = Path(dest_path)
    if dest_path.exists() and not overwrite:
        return dest_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
    with dest_path.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    return dest_path


def auto_download_rank_exports_from_drive(
    manifest: pd.DataFrame,
    *,
    download_dir: str | Path,
    folder_name: str,
    folder_id: str | None = None,
    overwrite: bool = False,
) -> dict:
    incomplete = manifest.loc[~manifest["task_state"].fillna("UNKNOWN").astype(str).eq("COMPLETED")].copy()
    if not incomplete.empty:
        preview = incomplete[["year", "task_description", "task_state"]].astype(str).to_dict("records")[:10]
        raise RuntimeError(
            "Some export tasks are still running, automatic download is currently unavailable; "
            f" pending={len(incomplete)}; preview={preview}"
        )

    service = build_drive_service()
    resolved_folder_id = resolve_drive_folder_id(service, folder_name=folder_name, folder_id=folder_id)
    download_root = Path(download_dir).expanduser()
    records = []
    for row in manifest.to_dict("records"):
        file_name = str(row["file_name"])
        metadata = find_drive_file_metadata(service, folder_id=resolved_folder_id, file_name=file_name)
        local_path = download_drive_file(
            service,
            file_id=str(metadata["id"]),
            dest_path=download_root / file_name,
            overwrite=overwrite,
        )
        records.append(
            {
                "file_name": file_name,
                "drive_file_id": str(metadata["id"]),
                "local_path": str(local_path),
                "downloaded": True,
            }
        )
    return {
        "folder_id": resolved_folder_id,
        "download_dir": str(download_root),
        "records": records,
    }


def resolve_download_dirs(explicit_dir: str | Path | None = None) -> list[Path]:
    candidate_dirs = []
    if explicit_dir is not None:
        candidate_dirs.append(Path(explicit_dir).expanduser())
    candidate_dirs.extend(DOWNLOAD_SEARCH_DIRS)
    deduped = []
    for candidate_dir in candidate_dirs:
        if candidate_dir not in deduped:
            deduped.append(candidate_dir)
    return deduped


def find_download_dir(expected_files: list[str], explicit_dir: str | Path | None = None) -> Path:
    candidate_dirs = resolve_download_dirs(explicit_dir)
    best_dir = None
    best_present = []
    for candidate_dir in candidate_dirs:
        present = [name for name in expected_files if (candidate_dir / name).is_file()]
        if len(present) == len(expected_files):
            return candidate_dir
        if len(present) > len(best_present):
            best_dir = candidate_dir
            best_present = present
    missing_preview = [name for name in expected_files if name not in best_present][:10]
    raise FileNotFoundError(
        "No complete downloaded files found; "
        f" searched_dirs={[str(path) for path in candidate_dirs]};"
        f" best_dir={str(best_dir) if best_dir is not None else None};"
        f" found={len(best_present)}/{len(expected_files)};"
        f" missing_preview={missing_preview}"
    )


def direct_local_file_name_for_year(year: int) -> str:
    return f"Total_features_sample_{int(year)}.csv"


def write_direct_local_manifest(records: list[dict]) -> pd.DataFrame:
    ensure_export_dirs()
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("No direct local manifest records to write.")
    frame = frame.sort_values(["year"]).reset_index(drop=True)
    frame.to_csv(DIRECT_LOCAL_MANIFEST_PATH, index=False)
    return frame


def read_direct_local_manifest() -> pd.DataFrame:
    if not DIRECT_LOCAL_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"direct local manifest does not exist: {DIRECT_LOCAL_MANIFEST_PATH}")
    manifest = pd.read_csv(DIRECT_LOCAL_MANIFEST_PATH)
    if manifest.empty:
        raise ValueError(f"direct local manifest is empty: {DIRECT_LOCAL_MANIFEST_PATH}")
    for col in ["year", "expected_rows", "downloaded_rows"]:
        if col in manifest.columns:
            manifest[col] = pd.to_numeric(manifest[col], errors="coerce").astype("Int64")
    return manifest


def export_rank_batches_direct_to_local(points_df: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ensure_export_dirs()
    normalized_years = sorted({int(year) for year in years})

    print("Loading Slope Unit geometries from GPKG...")
    all_sample_ids = points_df["Sample_ID"].astype(str).tolist()
    su_geoms = load_slope_unit_geometries(all_sample_ids)
    print(f"Loaded {len(su_geoms)} Slope Unit geometries.")

    def download_year_data(year: int) -> dict:
        points_year = (
            points_df.loc[points_df["Feature_Year"].astype(int) == int(year)]
            .sort_values(["Feature_Year", "Sample_ID"])
            .reset_index(drop=True)
        )
        if points_year.empty:
            raise ValueError(f"No sample rows found for Feature_Year={year}.")

        file_name = direct_local_file_name_for_year(int(year))
        output_path = DIRECT_LOCAL_EXPORT_DIR / file_name
        if output_path.exists() and output_path.stat().st_size > 100:
            try:
                existing_df = pd.read_csv(output_path)
                if len(existing_df) == len(points_year) and all(c in existing_df.columns for c in RANK_FEATURE_OUTPUT_COLUMNS):
                    print(f"[{year}] Existing complete feature file found with all {len(RANK_FEATURE_OUTPUT_COLUMNS)} columns: {file_name} ({len(existing_df)} rows), reusing...")
                    return {
                        "route": "rank",
                        "year": int(year),
                        "expected_rows": int(len(points_year)),
                        "downloaded_rows": int(len(existing_df)),
                        "file_name": file_name,
                        "local_path": str(output_path),
                    }
            except Exception:
                pass

        print(f"[{year}] Starting environmental feature extraction for {len(points_year)} slope units...")
        try:
            image_stack = get_feature_year_stack(int(year))
            sampled_df = extract_slope_units_features_for_year(
                points_year,
                int(year),
                image_stack,
                su_geoms,
                batch_size=150,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to extract sampled environmental factors for year={year}") from exc

        sampled_df = cleanup_sampled_frame(sampled_df)
        sampled_df.to_csv(output_path, index=False)
        print(f"[{year}] Feature extraction success! Saved {len(sampled_df)} rows to {file_name}")

        return {
            "route": "rank",
            "year": int(year),
            "expected_rows": int(len(points_year)),
            "downloaded_rows": int(len(sampled_df)),
            "file_name": file_name,
            "local_path": str(output_path),
        }

    records = []
    max_workers = min(len(normalized_years), 2)
    print(f"Starting feature downloads with {max_workers} worker(s) for years: {normalized_years}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_year = {executor.submit(download_year_data, year): year for year in normalized_years}
        for future in as_completed(future_to_year):
            year = future_to_year[future]
            try:
                record = future.result()
                records.append(record)
            except Exception as exc:
                print(f"[{year}] Thread execution failed: {exc}")
                raise exc

    return write_direct_local_manifest(records)

def load_downloaded_rank_frame(manifest: pd.DataFrame, *, explicit_dir: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    expected_files = manifest["file_name"].astype(str).tolist()
    selected_dir = find_download_dir(expected_files, explicit_dir=explicit_dir)
    frames = []
    for file_name in expected_files:
        file_path = selected_dir / file_name
        shard_df = pd.read_csv(file_path)
        frames.append(cleanup_sampled_frame(shard_df))
    if not frames:
        return pd.DataFrame(), selected_dir
    return pd.concat(frames, ignore_index=True), selected_dir


def merge_sampled_to_points(points_df: pd.DataFrame, sampled_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["Sample_ID", "Feature_Year"]
    feature_cols = RANK_FEATURE_OUTPUT_COLUMNS
    if sampled_df is None or sampled_df.empty:
        sampled_features = pd.DataFrame(columns=[*key_cols, *feature_cols])
    else:
        sampled = cleanup_sampled_frame(sampled_df.copy())
        missing_keys = [col for col in key_cols if col not in sampled.columns]
        if missing_keys:
            raise KeyError(f"rank sampling results missing primary key columns: {missing_keys}")
        for col in feature_cols:
            if col not in sampled.columns:
                sampled[col] = pd.NA
        sampled_features = sampled[[*key_cols, *feature_cols]].copy()
        duplicated = int(sampled_features.duplicated(subset=key_cols).sum())
        if duplicated:
            sampled_features = sampled_features.drop_duplicates(subset=key_cols, keep="first")
    merged = points_df.merge(sampled_features, on=key_cols, how="left", validate="1:1")
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(TABULAR_NODATA_VALUE)
    return merged


def finalize_rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    finalized = cast_nullable_int_columns(
        df.copy(),
        ["Feature_Year", "Label_Cls", "Label_Rank", "Cluster", "Basin_cluster", "Vegetation_type"],
    )
    for col in RANK_FEATURE_OUTPUT_COLUMNS:
        if col not in finalized.columns:
            finalized[col] = TABULAR_NODATA_VALUE
    return finalized[RANK_OUTPUT_COLUMNS].sort_values(["Feature_Year", "Sample_ID"]).reset_index(drop=True)


def count_missing_sample_rows(points_df: pd.DataFrame, sampled_df: pd.DataFrame) -> int:
    if sampled_df is None or sampled_df.empty:
        return len(points_df)
    key_cols = ["Sample_ID", "Feature_Year"]
    sampled = cleanup_sampled_frame(sampled_df.copy())
    if any(col not in sampled.columns for col in key_cols):
        return len(points_df)
    input_keys = points_df.loc[:, key_cols].astype(str).agg("::".join, axis=1)
    sampled_keys = set(sampled.loc[:, key_cols].astype(str).agg("::".join, axis=1))
    return int((~input_keys.isin(sampled_keys)).sum())

def run_build_total_features():
    ensure_export_dirs()
    rank_points = load_rank_points()
    rank_years = read_rank_years_from_samples()
    required_assets = validate_required_factor_library_assets(rank_years)
    direct_manifest = export_rank_batches_direct_to_local(rank_points, rank_years)

    print("rank direct local download")
    print(f"- sample rows: {len(rank_points)}")
    print(f"- years: {rank_years}")
    print(f"- required assets: {len(required_assets)}")
    print(f"- output dir: {DIRECT_LOCAL_EXPORT_DIR}")
    print(f"- manifest: {DIRECT_LOCAL_MANIFEST_PATH}")
    print(f"- files: {direct_manifest['file_name'].astype(str).tolist()}")

    direct_manifest = read_direct_local_manifest()
    sampled_df, selected_dir = load_downloaded_rank_frame(direct_manifest, explicit_dir=DIRECT_LOCAL_EXPORT_DIR)
    rank_points = load_rank_points()
    finalized = finalize_rank_frame(merge_sampled_to_points(rank_points, sampled_df))
    physical_quality = build_feature_physical_quality_report(finalized)
    FEATURE_PHYSICAL_QUALITY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_PHYSICAL_QUALITY_REPORT_PATH.write_text(
        json.dumps(physical_quality, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("feature physical quality")
    print(f"- status: {physical_quality['status']}")
    print(f"- nodata counts: {physical_quality['nodata_counts']}")
    print(f"- invalid rows: {physical_quality['physical_invalid_rows']}")
    if physical_quality["status"] != "pass":
        raise ValueError(
            "Feature physical quality gate failed; refusing to write the final training table. "
            f"invalid_by_feature={physical_quality['physical_invalid_by_feature']}"
        )
    RANK_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    finalized.to_csv(RANK_FINAL_PATH, index=False)

    print("rank direct local finalize")
    print(f"- manifest: {DIRECT_LOCAL_MANIFEST_PATH}")
    print(f"- download dir: {selected_dir}")
    print(f"- downloaded rows: {len(sampled_df)}")
    print(f"- missing sampled rows: {count_missing_sample_rows(rank_points, sampled_df)}")
    print(f"- written: {RANK_FINAL_PATH}")
    print(f"- output columns: {list(finalized.columns)}")


if __name__ == "__main__":
    import traceback
    try:
        run_build_total_features()
    except Exception as exc:
        traceback.print_exc()
        raise exc

#The alternative route is closed by default; When you need to use Google Drive to export, cancel the full-frame comment below.
# ensure_export_dirs()
# rank_points = load_rank_points()
# rank_years = read_rank_years_from_samples()
# required_assets = validate_required_factor_library_assets(rank_years)
# rank_batches = build_rank_batches(rank_points, batch_size=CLOUD_RANK_BATCH_ROWS)
# rank_images = {year: get_feature_year_stack(int(year)) for year in rank_years}
#
# manifest_records = []
# for batch in rank_batches:
#     suffix = rank_batch_suffix(batch, batch_size=CLOUD_RANK_BATCH_ROWS)
#     task_description = f"{RANK_TASK_DESCRIPTION_PREFIX}_{suffix}"
#     drive_prefix = f"{RANK_FINAL_PATH.stem}_{suffix}"
#     task = start_table_export_task(
#         sample_points_once_fc(batch["points_df"], rank_images[int(batch["year"])]),
#         description=task_description,
#         file_name_prefix=drive_prefix,
#         selectors=RANK_OUTPUT_COLUMNS,
#     )
#     manifest_records.append(
#         make_manifest_record(
#             batch,
#             file_name=f"{drive_prefix}.csv",
#             task_description=task_description,
#             drive_prefix=drive_prefix,
#             task=task,
#         )
#     )
#
# cloud_manifest = write_cloud_manifest(manifest_records)
# print("rank export submit")
# print(f"- sample rows: {len(rank_points)}")
# print(f"- years: {rank_years}")
# print(f"- batches: {len(rank_batches)}")
# print(f"- required assets: {len(required_assets)}")
# print(f"- manifest: {CLOUD_MANIFEST_PATH}")
# print(f"- files: {cloud_manifest['file_name'].astype(str).tolist()}")
# print(f"- task states: {cloud_manifest['task_state'].fillna('UNKNOWN').astype(str).tolist()}")

# cloud_manifest = refresh_cloud_manifest_for_finalize(read_cloud_manifest())
# sampled_df, selected_dir = load_downloaded_rank_frame(cloud_manifest, explicit_dir=GOOGLE_DRIVE_LOCAL_DIR)
# rank_points = load_rank_points()
# finalized = finalize_rank_frame(merge_sampled_to_points(rank_points, sampled_df))
# RANK_FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
# finalized.to_csv(RANK_FINAL_PATH, index=False)
#
# print("rank export finalize")
# print(f"- manifest: {CLOUD_MANIFEST_PATH}")
# print(f"- download dir: {selected_dir}")
# print(f"- downloaded rows: {len(sampled_df)}")
# print(f"- missing sampled rows: {count_missing_sample_rows(rank_points, sampled_df)}")
# print(f"- written: {RANK_FINAL_PATH}")
# print(f"- output columns: {list(finalized.columns)}")
