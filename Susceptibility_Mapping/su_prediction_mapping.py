"""Attach slope-unit prediction Parquet files to a copied GeoPackage.

The original slope-unit GeoPackage is never modified.  The script first copies
it to a temporary file beside the requested output, validates every prediction
table by ``fid`` and ``su_id``, adds one REAL field per result, and atomically
publishes the completed ``QTP_SlopeUnits_predictions.gpkg``.  The pipeline's
``-9999`` prediction nodata sentinel is written as SQL NULL in the output.
The command-line workflow defaults to the 2010 permafrost mask and only keeps
predictions for SUs whose polygon intersects mask class ``1``.  The retained
slope-unit geometry is not clipped; it remains the complete prediction unit.
By default, the published GeoPackage also removes non-permafrost SUs, so its
feature class is the actual prediction domain.  ``--keep-all-slope-units`` is
available for diagnostic exports that need the full source geometry with NULLs
outside the domain.  ``--permafrost-mode centroid`` is retained as a legacy
option for reproducing the former centroid rule.

Each ``--result`` value uses this format::

    PARQUET_PATH::OUTPUT_FIELD[::VALUE_COLUMN]

For example::

    --result predictions_bs128_2021.parquet::pred_bs128_2021

The default value column is ``prediction``.  Parquet input is streamed in
batches and staged in SQLite, so memory use does not grow with the number of
slope units.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import time
import uuid
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_GPKG = WORKSPACE_ROOT / "Dataset_Construction" / "inputs" / "QTP_SlopeUnits.gpkg"
DEFAULT_OUTPUT_GPKG = (
    WORKSPACE_ROOT
    / "Susceptibility_Mapping"
    / "output"
    / "gpkg"
    / "QTP_SlopeUnits_permafrost_predictions.gpkg"
)
DEFAULT_TABLE_NAME = "qtp_slope_units"
DEFAULT_VALUE_COLUMN = "prediction"
DEFAULT_BATCH_ROWS = 100_000
PREDICTION_NODATA_VALUE = -9999.0
DEFAULT_PERMAFROST_MASK = WORKSPACE_ROOT / "Dataset_Construction" / "inputs" / "permafrost_qtp_2010.tif"
DEFAULT_PERMAFROST_MASK_VALUE = 1
DEFAULT_PERMAFROST_MODE = "intersects"
PERMAFROST_MODES = ("centroid", "intersects")
PERMAFROST_DOMAIN_TABLE = "permafrost_prediction_domain"
MAX_ARCGIS_FIELD_LENGTH = 63
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# The source GeoPackage stores the QTP Albers CRS as a GDAL WKT1 string.  That
# string is valid for GDAL/PROJ, but ArcGIS Pro is stricter about WKT1
# projection and parameter names (for example, it expects ``Albers`` and
# ``Central_Meridian`` rather than ``Albers_Conic_Equal_Area`` and
# ``longitude_of_center``).  Keep the same mathematical CRS and write the
# ArcGIS/ESRI WKT1 representation only to the generated output copy.
ARCGIS_QTP_ALBERS_WKT = (
    'PROJCS["QTP_Albers_Equal_Area",'
    'GEOGCS["GCS_WGS_1984",'
    'DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],'
    'UNIT["Degree",0.0174532925199433]],'
    'PROJECTION["Albers"],'
    'PARAMETER["False_Easting",0.0],'
    'PARAMETER["False_Northing",0.0],'
    'PARAMETER["Central_Meridian",90.0],'
    'PARAMETER["Standard_Parallel_1",27.5],'
    'PARAMETER["Standard_Parallel_2",37.5],'
    'PARAMETER["Latitude_Of_Origin",0.0],'
    'UNIT["Meter",1.0]]'
)


class MappingError(RuntimeError):
    """Raised when a prediction result cannot be safely mapped."""


@dataclass(frozen=True)
class ResultSpec:
    parquet_path: Path
    output_field: str
    value_column: str


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str), flush=True)


def quote_identifier(value: str, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not IDENTIFIER_RE.fullmatch(normalized):
        raise ValueError(
            f"Invalid {label} {normalized!r}; use letters, digits, and underscores, "
            "and do not start with a digit."
        )
    if len(normalized) > MAX_ARCGIS_FIELD_LENGTH:
        raise ValueError(
            f"{label} {normalized!r} is {len(normalized)} characters; "
            f"keep it at or below {MAX_ARCGIS_FIELD_LENGTH} for ArcGIS compatibility."
        )
    return f'"{normalized}"'


def infer_output_field(path: Path) -> str:
    stem = path.stem
    match = re.search(r"(?i)bs[_-]?(\d+)[_-](20\d{2})", stem)
    if match:
        return f"pred_bs{int(match.group(1))}_{int(match.group(2))}"
    candidate = re.sub(r"[^A-Za-z0-9_]+", "_", stem).strip("_")
    if not candidate:
        raise ValueError(f"Cannot infer an output field from {path}; provide PATH::OUTPUT_FIELD.")
    if candidate[0].isdigit():
        candidate = f"pred_{candidate}"
    if len(candidate) > MAX_ARCGIS_FIELD_LENGTH:
        raise ValueError(
            f"Inferred output field {candidate!r} is too long; provide PATH::OUTPUT_FIELD explicitly."
        )
    return candidate


def parse_result_spec(raw: str, *, default_value_column: str) -> ResultSpec:
    parts = [part.strip() for part in str(raw).split("::")]
    if not parts or not parts[0] or len(parts) > 3:
        raise ValueError(
            f"Invalid --result value {raw!r}; expected PARQUET_PATH::OUTPUT_FIELD[::VALUE_COLUMN]."
        )
    path = Path(parts[0]).expanduser().resolve()
    output_field = parts[1] if len(parts) >= 2 and parts[1] else infer_output_field(path)
    value_column = parts[2] if len(parts) >= 3 and parts[2] else str(default_value_column).strip()
    quote_identifier(output_field, label="output field")
    if not value_column:
        raise ValueError(f"Prediction value column is empty for {path}")
    return ResultSpec(path, output_field, value_column)


def resolve_column_name(names: Iterable[str], requested: str, *, path: Path, role: str) -> str:
    available = [str(name) for name in names]
    if requested in available:
        return requested
    matches = [name for name in available if name.casefold() == str(requested).casefold()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise MappingError(f"Ambiguous {role} column {requested!r} in {path}: {matches}")
    raise MappingError(
        f"Missing {role} column {requested!r} in {path}; available columns={available}"
    )


def normalize_fid(value: Any, *, path: Path, row_number: int) -> int:
    if value is None or isinstance(value, bool):
        raise MappingError(f"Invalid fid at {path}, row {row_number}: {value!r}")
    try:
        numeric = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MappingError(f"Invalid fid at {path}, row {row_number}: {value!r}") from exc
    if numeric <= 0:
        raise MappingError(f"fid must be positive at {path}, row {row_number}: {numeric}")
    if isinstance(value, float) and (not math.isfinite(value) or value != numeric):
        raise MappingError(f"fid is not an integer at {path}, row {row_number}: {value!r}")
    return numeric


def normalize_su_id(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        if value.is_integer():
            return str(int(value))
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_prediction(value: Any, *, path: Path, row_number: int) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MappingError(
            f"Prediction value is not numeric at {path}, row {row_number}: {value!r}"
        ) from exc
    if not math.isfinite(numeric) or numeric == PREDICTION_NODATA_VALUE:
        return None
    return numeric


def normalize_text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan", "null"):
        return None
    return text


def register_gpkg_spatial_compatibility_functions(conn: sqlite3.Connection) -> None:
    """Allow attribute-only updates on GPKGs with OGR RTree triggers.

    GeoPackage files written by GDAL commonly include RTree triggers whose
    ``AFTER UPDATE`` predicates call spatial functions even when only a new
    attribute column is being updated.  Python's stdlib SQLite does not load
    a spatial extension, but an attribute-only update never changes geometry;
    the RTree predicates are therefore false because ``fid`` is unchanged.
    Registering conservative compatibility functions keeps the original
    triggers and spatial index intact.
    """

    def st_is_empty(value: Any) -> int:
        return int(value is None or (isinstance(value, (bytes, bytearray, memoryview)) and len(value) == 0))

    def st_coordinate(value: Any) -> float:
        # These functions are only reached while evaluating an unchanged
        # geometry trigger predicate.  Their result cannot affect the update.
        return 0.0

    conn.create_function("ST_IsEmpty", 1, st_is_empty)
    for name in ("ST_MinX", "ST_MaxX", "ST_MinY", "ST_MaxY"):
        conn.create_function(name, 1, st_coordinate)


def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def table_columns(conn: sqlite3.Connection, table_name: str) -> dict[str, dict[str, Any]]:
    quoted_table = quote_identifier(table_name, label="table name")
    rows = conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    return {
        str(row[1]): {
            "cid": int(row[0]),
            "type": str(row[2]),
            "notnull": bool(row[3]),
            "default": row[4],
            "primary_key": bool(row[5]),
        }
        for row in rows
    }


def validate_source_gpkg(path: Path, table_name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Source GeoPackage does not exist: {path}")
    with closing(connect_read_only(path)) as conn:
        integrity = str(conn.execute("PRAGMA quick_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise MappingError(f"Source GeoPackage failed PRAGMA quick_check: {integrity}")
        columns = table_columns(conn, table_name)
        missing = sorted({"fid", "su_id", "geom"} - set(columns))
        if missing:
            raise MappingError(f"Source table {table_name!r} is missing columns: {missing}")
        if not columns["fid"]["primary_key"]:
            raise MappingError(f"{table_name}.fid must be the primary key")
        geometry_row = conn.execute(
            "SELECT column_name, geometry_type_name, srs_id "
            "FROM gpkg_geometry_columns WHERE table_name = ?",
            (table_name,),
        ).fetchone()
        if geometry_row is None:
            raise MappingError(f"{table_name!r} is not registered in gpkg_geometry_columns")
        row_count, min_fid, max_fid = conn.execute(
            f"SELECT COUNT(*), MIN(\"fid\"), MAX(\"fid\") FROM {quote_identifier(table_name, label='table name')}"
        ).fetchone()
    return {
        "table_name": table_name,
        "row_count": int(row_count),
        "min_fid": int(min_fid),
        "max_fid": int(max_fid),
        "geometry_column": str(geometry_row[0]),
        "geometry_type": str(geometry_row[1]),
        "srs_id": int(geometry_row[2]),
    }


def normalize_arcgis_compatible_crs(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    source_srs_id: int,
) -> dict[str, Any]:
    """Rewrite the output copy's QTP CRS using ArcGIS-compatible ESRI WKT1.

    This deliberately changes CRS metadata only.  It does not transform or
    rewrite any geometry coordinates.  Before changing the definition, the
    stored CRS is compared with the expected QTP Albers projection so that a
    wrong CRS cannot be silently assigned to the layer.

    The function is a no-op for a test/minimal database without
    ``gpkg_spatial_ref_sys`` or for a different, non-QTP CRS.  A real QTP
    record with an invalid or inconsistent definition fails explicitly.
    """

    started = time.perf_counter()
    base_summary = {
        "enabled": False,
        "applied": False,
        "source_srs_id": int(source_srs_id),
        "target_wkt_format": "WKT1_ESRI",
        "coordinate_values_changed": False,
    }
    spatial_ref_table = conn.execute(
        "SELECT 1 FROM sqlite_master "
        "WHERE type = 'table' AND name = 'gpkg_spatial_ref_sys'"
    ).fetchone()
    if spatial_ref_table is None:
        return {
            **base_summary,
            "reason": "gpkg_spatial_ref_sys is absent; CRS metadata was not changed",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    srs_row = conn.execute(
        "SELECT srs_name, organization, organization_coordsys_id, definition, description "
        "FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (int(source_srs_id),),
    ).fetchone()
    if srs_row is None:
        raise MappingError(
            f"Could not resolve output CRS metadata for srs_id={source_srs_id}"
        )

    source_definition = str(srs_row[3] or "").strip()
    if not source_definition:
        raise MappingError(
            f"CRS definition is empty for srs_id={source_srs_id}; "
            "refusing to publish an ArcGIS mapping layer."
        )

    from pyproj import CRS

    try:
        source_crs = CRS.from_wkt(source_definition)
        target_crs = CRS.from_wkt(ARCGIS_QTP_ALBERS_WKT)
    except Exception as exc:
        if int(source_srs_id) == 100000:
            raise MappingError(
                f"QTP custom CRS srs_id={source_srs_id} cannot be parsed; "
                "refusing to rewrite CRS metadata."
            ) from exc
        return {
            **base_summary,
            "reason": (
                f"srs_id={source_srs_id} is not parseable as WKT and is not the "
                "known QTP custom CRS; metadata was not changed"
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    try:
        same_crs = bool(source_crs.equals(target_crs, ignore_axis_order=True))
    except TypeError:  # pragma: no cover - compatibility with older pyproj
        same_crs = bool(source_crs.equals(target_crs))
    if not same_crs:
        if int(source_srs_id) == 100000:
            raise MappingError(
                f"srs_id={source_srs_id} does not describe the expected QTP Albers "
                "projection; refusing to assign a different CRS definition."
            )
        return {
            **base_summary,
            "reason": (
                f"srs_id={source_srs_id} is a different CRS; its original metadata "
                "was preserved"
            ),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    # Keep the custom CRS identifier and organization.  In particular, do not
    # invent an EPSG/ESRI authority code for this project-specific projection.
    # ArcGIS receives the parseable ESRI WKT1 definition while the GPKG still
    # correctly identifies the CRS as custom.
    new_srs_name = "QTP_Albers_Equal_Area"
    new_description = (
        "QTP Albers Equal Area custom CRS; ESRI WKT1 representation for ArcGIS Pro"
    )
    changed = any(
        (
            str(srs_row[0] or "") != new_srs_name,
            str(srs_row[3] or "").strip() != ARCGIS_QTP_ALBERS_WKT,
            str(srs_row[4] or "") != new_description,
        )
    )
    if changed:
        conn.execute(
            "UPDATE gpkg_spatial_ref_sys "
            "SET srs_name = ?, definition = ?, description = ? "
            "WHERE srs_id = ?",
            (new_srs_name, ARCGIS_QTP_ALBERS_WKT, new_description, int(source_srs_id)),
        )
        conn.commit()

    return {
        **base_summary,
        "enabled": True,
        "applied": bool(changed),
        "srs_id": int(source_srs_id),
        "srs_name": new_srs_name,
        "organization": str(srs_row[1] or ""),
        "organization_coordsys_id": (
            None if srs_row[2] is None else int(srs_row[2])
        ),
        "definition": ARCGIS_QTP_ALBERS_WKT,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


def build_permafrost_domain_table(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    source_srs_id: int,
    mask_path: Path,
    mask_value: int,
    mode: str,
    source_row_count: int,
    batch_rows: int,
) -> tuple[str, dict[str, Any]]:
    """Stage the SU prediction domain from the permafrost mask.

    ``centroid`` tests the raster class at each SU centroid.  ``intersects``
    vectorizes the requested raster class once and retains a complete SU when
    its polygon intersects that mask geometry.  The latter is the default so
    boundary SUs are not lost merely because their centroid falls outside the
    mask.
    """

    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in PERMAFROST_MODES:
        raise ValueError(
            f"Unsupported permafrost domain mode {mode!r}; "
            f"choose one of {', '.join(PERMAFROST_MODES)}."
        )
    mask_path = mask_path.resolve()
    if not mask_path.is_file():
        raise FileNotFoundError(f"Permafrost mask does not exist: {mask_path}")

    import numpy as np
    # Import Shapely before Rasterio.  Both packages ship GEOS-related DLLs on
    # Windows, and importing Rasterio first can make Shapely's extension fail
    # to load in some environments.
    from shapely import wkb
    from shapely.geometry import shape as shapely_shape
    from shapely.ops import transform as shapely_transform
    from shapely.ops import unary_union
    from shapely.prepared import prep
    import rasterio
    from rasterio.features import shapes as raster_shapes
    from pyproj import CRS, Transformer

    gpkg_envelope_bytes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}

    def decode_gpkg_geometry(blob: bytes):
        if not isinstance(blob, (bytes, bytearray, memoryview)) or bytes(blob)[:2] != b"GP":
            raise ValueError("Expected a GeoPackage geometry blob with GP header.")
        raw = bytes(blob)
        envelope_type = (int(raw[3]) >> 1) & 0x07
        header_len = 8 + gpkg_envelope_bytes.get(envelope_type, 0)
        return wkb.loads(raw[header_len:])

    srs_row = conn.execute(
        "SELECT definition FROM gpkg_spatial_ref_sys WHERE srs_id = ?",
        (int(source_srs_id),),
    ).fetchone()
    if srs_row is None or not str(srs_row[0] or "").strip():
        raise MappingError(f"Could not resolve source CRS definition for srs_id={source_srs_id}")
    source_crs = CRS.from_wkt(str(srs_row[0]))

    domain_table = PERMAFROST_DOMAIN_TABLE
    quoted_domain = quote_identifier(domain_table, label="temporary domain table name")
    conn.execute(f"DROP TABLE IF EXISTS {quoted_domain}")
    conn.execute(f"CREATE TEMP TABLE {quoted_domain} (fid INTEGER PRIMARY KEY)")

    started = time.perf_counter()
    total_rows = 0
    in_domain_rows = 0
    out_of_mask_rows = 0
    empty_geometry_rows = 0
    mask_component_count: int | None = None
    insert_sql = f"INSERT INTO {quoted_domain} (fid) VALUES (?)"
    with rasterio.open(mask_path) as mask_src:
        if mask_src.crs is None:
            raise MappingError(f"Permafrost mask has no CRS: {mask_path}")
        mask_array = mask_src.read(1)
        mask_crs = CRS.from_user_input(mask_src.crs)
        mask_height, mask_width = mask_array.shape

        prepared_mask_geometry = None
        if normalized_mode == "intersects":
            mask_geometries = [
                shapely_shape(geometry_mapping)
                for geometry_mapping, value in raster_shapes(
                    mask_array,
                    mask=(mask_array == int(mask_value)),
                    transform=mask_src.transform,
                    connectivity=8,
                )
                if int(value) == int(mask_value)
            ]
            mask_component_count = len(mask_geometries)
            if not mask_geometries:
                raise MappingError(
                    f"Permafrost mask contains no cells with value {mask_value}: {mask_path}"
                )
            mask_geometry = unary_union(mask_geometries)
            if mask_geometry.is_empty:
                raise MappingError(
                    f"Permafrost mask geometry is empty for value {mask_value}: {mask_path}"
                )
            mask_to_source = Transformer.from_crs(mask_crs, source_crs, always_xy=True)
            prepared_mask_geometry = prep(
                shapely_transform(mask_to_source.transform, mask_geometry)
            )
        else:
            transformer = Transformer.from_crs(source_crs, mask_crs, always_xy=True)
            inverse_transform = ~mask_src.transform

        cursor = conn.execute(
            f"SELECT \"fid\", \"geom\" FROM {quote_identifier(table_name, label='table name')} ORDER BY \"fid\""
        )
        while True:
            fetched = cursor.fetchmany(max(1, int(batch_rows)))
            if not fetched:
                break
            fids: list[int] = []
            is_in_domain = np.zeros(len(fetched), dtype=bool)
            source_x = np.full(len(fetched), np.nan, dtype=np.float64)
            source_y = np.full(len(fetched), np.nan, dtype=np.float64)
            for index, (fid_raw, geom_blob) in enumerate(fetched):
                fid = int(fid_raw)
                fids.append(fid)
                try:
                    geometry = decode_gpkg_geometry(geom_blob)
                except Exception as exc:
                    raise MappingError(f"Could not decode geometry for fid={fid}") from exc
                if geometry.is_empty:
                    empty_geometry_rows += 1
                    continue
                if normalized_mode == "intersects":
                    try:
                        is_in_domain[index] = bool(
                            prepared_mask_geometry.intersects(geometry)
                        )
                    except Exception as exc:
                        raise MappingError(
                            f"Could not test permafrost intersection for fid={fid}"
                        ) from exc
                    continue
                centroid = geometry.centroid
                if centroid.is_empty or not (np.isfinite(centroid.x) and np.isfinite(centroid.y)):
                    empty_geometry_rows += 1
                    continue
                source_x[index] = float(centroid.x)
                source_y[index] = float(centroid.y)

            if normalized_mode == "centroid":
                coordinate_valid = np.isfinite(source_x) & np.isfinite(source_y)
                mask_x = np.full(len(fetched), np.nan, dtype=np.float64)
                mask_y = np.full(len(fetched), np.nan, dtype=np.float64)
                if coordinate_valid.any():
                    transformed_x, transformed_y = transformer.transform(
                        source_x[coordinate_valid], source_y[coordinate_valid]
                    )
                    mask_x[coordinate_valid] = np.asarray(transformed_x, dtype=np.float64)
                    mask_y[coordinate_valid] = np.asarray(transformed_y, dtype=np.float64)

                col_float = (
                    inverse_transform.a * mask_x
                    + inverse_transform.b * mask_y
                    + inverse_transform.c
                )
                row_float = (
                    inverse_transform.d * mask_x
                    + inverse_transform.e * mask_y
                    + inverse_transform.f
                )
                cols = np.floor(col_float).astype(np.int64)
                rows = np.floor(row_float).astype(np.int64)
                in_bounds = (
                    coordinate_valid
                    & (rows >= 0)
                    & (rows < mask_height)
                    & (cols >= 0)
                    & (cols < mask_width)
                )
                out_of_mask_rows += int((coordinate_valid & ~in_bounds).sum())
                is_in_domain[in_bounds] = (
                    mask_array[rows[in_bounds], cols[in_bounds]] == int(mask_value)
                )
            domain_fids = [(fid,) for fid, selected in zip(fids, is_in_domain) if selected]
            if domain_fids:
                conn.executemany(insert_sql, domain_fids)
            total_rows += len(fetched)
            in_domain_rows += len(domain_fids)
            conn.commit()
            if total_rows == len(fetched) or total_rows % (max(1, int(batch_rows)) * 5) == 0:
                emit(
                    "permafrost_domain_progress",
                    rows=total_rows,
                    rows_total=int(source_row_count),
                    in_domain_rows=in_domain_rows,
                )

        if total_rows != int(source_row_count):
            raise MappingError(
                f"Permafrost domain scan row count mismatch: {total_rows} != {source_row_count}"
            )

        mask_crs_text = str(mask_src.crs)
        mask_shape = [int(mask_height), int(mask_width)]

    domain_summary = {
        "enabled": True,
        "mode": normalized_mode,
        "domain_rule": (
            "polygon_intersects_mask_class"
            if normalized_mode == "intersects"
            else "centroid_pixel_matches_mask_class"
        ),
        "mask_path": str(mask_path),
        "mask_value": int(mask_value),
        "mask_crs": mask_crs_text,
        "mask_shape": mask_shape,
        "mask_component_count": mask_component_count,
        "source_srs_id": int(source_srs_id),
        "source_rows": int(total_rows),
        "in_domain_rows": int(in_domain_rows),
        "out_of_domain_rows": int(total_rows - in_domain_rows),
        "centroid_out_of_mask_rows": (
            int(out_of_mask_rows) if normalized_mode == "centroid" else None
        ),
        "intersection_tested_rows": (
            int(total_rows - empty_geometry_rows)
            if normalized_mode == "intersects"
            else None
        ),
        "empty_geometry_rows": int(empty_geometry_rows),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    return domain_table, domain_summary


def validate_result_specs(specs: list[ResultSpec]) -> None:
    if not specs:
        raise ValueError("At least one --result is required")
    seen_fields: dict[str, Path] = {}
    for spec in specs:
        if not spec.parquet_path.is_file():
            raise FileNotFoundError(f"Prediction Parquet does not exist: {spec.parquet_path}")
        folded = spec.output_field.casefold()
        if folded in seen_fields:
            raise ValueError(
                f"Duplicate output field {spec.output_field!r}: "
                f"{seen_fields[folded]} and {spec.parquet_path}"
            )
        seen_fields[folded] = spec.parquet_path


def add_prediction_column(
    conn: sqlite3.Connection, table_name: str, field_name: str, field_type: str = "REAL"
) -> bool:
    field_type_upper = str(field_type).strip().upper()
    existing = {name.casefold(): name for name in table_columns(conn, table_name)}
    if field_name.casefold() in existing:
        actual = existing[field_name.casefold()]
        declared = table_columns(conn, table_name)[actual]["type"].upper()
        if field_type_upper == "REAL":
            valid_types = ("REAL", "FLOAT", "DOUBLE", "NUMERIC")
        else:
            valid_types = ("TEXT", "VARCHAR", "CHAR", "STRING")
        if not any(token in declared for token in valid_types):
            raise MappingError(
                f"Existing output field {actual!r} has incompatible SQLite type {declared!r} for {field_type_upper}"
            )
        return False
    conn.execute(
        f"ALTER TABLE {quote_identifier(table_name, label='table name')} "
        f"ADD COLUMN {quote_identifier(field_name, label='output field')} {field_type_upper}"
    )
    conn.commit()
    return True


def stage_parquet(
    conn: sqlite3.Connection,
    spec: ResultSpec,
    *,
    fid_column: str,
    su_id_column: str,
    skip_su_id_validation: bool,
    batch_rows: int,
) -> tuple[str, dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(spec.parquet_path)
    names = parquet.schema_arrow.names
    fid_name = resolve_column_name(names, fid_column, path=spec.parquet_path, role="fid")
    value_name = resolve_column_name(
        names,
        spec.value_column,
        path=spec.parquet_path,
        role="prediction value",
    )
    value_field = parquet.schema_arrow.field(value_name)
    is_text = pa.types.is_string(value_field.type) or pa.types.is_large_string(value_field.type)
    sql_type = "TEXT" if is_text else "REAL"

    su_id_name = None
    if not skip_su_id_validation:
        su_id_name = resolve_column_name(names, su_id_column, path=spec.parquet_path, role="su_id")

    requested_columns = [fid_name]
    if su_id_name is not None and su_id_name not in requested_columns:
        requested_columns.append(su_id_name)
    if value_name not in requested_columns:
        requested_columns.append(value_name)

    stage_name = f"stage_su_prediction_{uuid.uuid4().hex}"
    quoted_stage = quote_identifier(stage_name, label="temporary table name")
    conn.execute(
        f"CREATE TEMP TABLE {quoted_stage} ("
        f"fid INTEGER PRIMARY KEY, su_id TEXT, prediction {sql_type})"
    )
    insert_sql = f"INSERT INTO {quoted_stage} (fid, su_id, prediction) VALUES (?, ?, ?)"
    input_rows = 0
    finite_values = 0
    null_values = 0
    su_id_nulls = 0
    value_min = float("inf")
    value_max = float("-inf")
    started = time.perf_counter()

    for record_batch in parquet.iter_batches(
        batch_size=max(1, int(batch_rows)),
        columns=requested_columns,
        use_threads=True,
    ):
        payload = record_batch.to_pydict()
        fids = payload[fid_name]
        su_ids = payload[su_id_name] if su_id_name is not None else [None] * len(fids)
        predictions = payload[value_name]
        rows: list[tuple[int, str | None, Any]] = []
        for index, (fid_raw, su_id_raw, prediction_raw) in enumerate(
            zip(fids, su_ids, predictions),
            start=input_rows + 1,
        ):
            fid = normalize_fid(fid_raw, path=spec.parquet_path, row_number=index)
            su_id = normalize_su_id(su_id_raw)
            if is_text:
                prediction = normalize_text_value(prediction_raw)
                if prediction is None:
                    null_values += 1
                else:
                    finite_values += 1
            else:
                prediction = normalize_prediction(
                    prediction_raw,
                    path=spec.parquet_path,
                    row_number=index,
                )
                if prediction is None:
                    null_values += 1
                else:
                    finite_values += 1
                    value_min = min(value_min, prediction)
                    value_max = max(value_max, prediction)
            rows.append((fid, su_id, prediction))
        try:
            conn.executemany(insert_sql, rows)
            conn.commit()
        except sqlite3.IntegrityError as exc:
            conn.rollback()
            raise MappingError(
                f"Duplicate or invalid fid detected in {spec.parquet_path}; "
                "each slope unit must appear at most once."
            ) from exc
        input_rows += len(rows)
        if input_rows == len(rows) or input_rows % (max(1, int(batch_rows)) * 10) == 0:
            emit(
                "prediction_stage_progress",
                parquet_path=str(spec.parquet_path),
                output_field=spec.output_field,
                rows=input_rows,
            )

    return stage_name, {
        "parquet_path": str(spec.parquet_path),
        "output_field": spec.output_field,
        "value_column": value_name,
        "fid_column": fid_name,
        "su_id_column": su_id_name,
        "input_rows": int(input_rows),
        "finite_values": int(finite_values),
        "null_values": int(null_values),
        "su_id_nulls": int(su_id_nulls),
        "prediction_min": None if (is_text or finite_values == 0) else float(value_min),
        "prediction_max": None if (is_text or finite_values == 0) else float(value_max),
        "field_type": sql_type,
        "stage_seconds": round(time.perf_counter() - started, 3),
    }


def attach_staged_result(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    field_name: str,
    stage_name: str,
    source_row_count: int,
    skip_su_id_validation: bool,
    require_complete: bool,
    domain_table: str | None = None,
) -> dict[str, Any]:
    quoted_table = quote_identifier(table_name, label="table name")
    quoted_field = quote_identifier(field_name, label="output field")
    quoted_stage = quote_identifier(stage_name, label="temporary table name")
    staged_rows = int(conn.execute(f"SELECT COUNT(*) FROM {quoted_stage}").fetchone()[0])
    matched_rows = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {quoted_stage} AS s "
            f"INNER JOIN {quoted_table} AS g ON g.\"fid\" = s.fid"
        ).fetchone()[0]
    )
    unknown_fids = staged_rows - matched_rows
    if unknown_fids:
        examples = [
            int(row[0])
            for row in conn.execute(
                f"SELECT s.fid FROM {quoted_stage} AS s "
                f"LEFT JOIN {quoted_table} AS g ON g.\"fid\" = s.fid "
                "WHERE g.\"fid\" IS NULL LIMIT 5"
            ).fetchall()
        ]
        raise MappingError(
            f"Prediction table contains {unknown_fids} fid values absent from the GeoPackage; "
            f"examples={examples}"
        )

    su_id_mismatches = 0
    if not skip_su_id_validation:
        su_id_mismatches = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {quoted_stage} AS s "
                f"INNER JOIN {quoted_table} AS g ON g.\"fid\" = s.fid "
                "WHERE s.su_id IS NULL OR g.\"su_id\" IS NULL "
                "OR CAST(g.\"su_id\" AS TEXT) <> s.su_id"
            ).fetchone()[0]
        )
        if su_id_mismatches:
            examples = conn.execute(
                f"SELECT s.fid, s.su_id, g.\"su_id\" FROM {quoted_stage} AS s "
                f"INNER JOIN {quoted_table} AS g ON g.\"fid\" = s.fid "
                "WHERE s.su_id IS NULL OR g.\"su_id\" IS NULL "
                "OR CAST(g.\"su_id\" AS TEXT) <> s.su_id LIMIT 5"
            ).fetchall()
            raise MappingError(
                f"Prediction table has {su_id_mismatches} fid/su_id mismatches; examples={examples}"
            )

    missing_source_rows = int(source_row_count) - matched_rows
    if require_complete and missing_source_rows:
        raise MappingError(
            f"Prediction table is incomplete: {matched_rows}/{source_row_count} slope units present; "
            f"missing={missing_source_rows}. Remove --require-complete to retain NULLs for missing units."
        )

    domain_matched_rows = matched_rows
    domain_filter_sql = ""
    if domain_table:
        quoted_domain = quote_identifier(domain_table, label="temporary domain table name")
        domain_matched_rows = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {quoted_stage} AS s "
                f"INNER JOIN {quoted_domain} AS d ON d.fid = s.fid"
            ).fetchone()[0]
        )
        domain_filter_sql = (
            f" AND EXISTS (SELECT 1 FROM {quoted_domain} AS d WHERE d.fid = g.\"fid\")"
        )

    conn.execute(f"UPDATE {quoted_table} SET {quoted_field} = NULL")
    conn.execute(
        f"UPDATE {quoted_table} AS g SET {quoted_field} = ("
        f"SELECT s.prediction FROM {quoted_stage} AS s WHERE s.fid = g.\"fid\") "
        f"WHERE g.\"fid\" IN (SELECT fid FROM {quoted_stage})"
        f"{domain_filter_sql}"
    )
    conn.commit()
    output_non_null = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {quoted_table} WHERE {quoted_field} IS NOT NULL"
        ).fetchone()[0]
    )
    return {
        "staged_rows": staged_rows,
        "matched_rows": matched_rows,
        "unknown_fids": unknown_fids,
        "su_id_mismatches": su_id_mismatches,
        "missing_source_rows": missing_source_rows,
        "coverage_ratio": float(matched_rows / source_row_count) if source_row_count else 0.0,
        "domain_matched_rows": int(domain_matched_rows),
        "domain_excluded_rows": int(matched_rows - domain_matched_rows),
        "output_non_null": output_non_null,
    }


def update_gpkg_contents_extent(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    geometry_column: str,
) -> dict[str, float] | None:
    """Refresh the registered layer extent from its RTree, when available."""

    rtree_name = f"rtree_{table_name}_{geometry_column}"
    rtree_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (rtree_name,),
    ).fetchone()
    if rtree_exists is None:
        return None
    min_x, min_y, max_x, max_y = conn.execute(
        f"SELECT MIN(minx), MIN(miny), MAX(maxx), MAX(maxy) "
        f"FROM {quote_identifier(rtree_name, label='RTree table name')}"
    ).fetchone()
    if any(value is None for value in (min_x, min_y, max_x, max_y)):
        return None
    extent = {
        "min_x": float(min_x),
        "min_y": float(min_y),
        "max_x": float(max_x),
        "max_y": float(max_y),
    }
    contents_exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'gpkg_contents'"
    ).fetchone()
    if contents_exists is not None:
        conn.execute(
            "UPDATE gpkg_contents SET min_x = ?, min_y = ?, max_x = ?, max_y = ? "
            "WHERE table_name = ?",
            (
                extent["min_x"],
                extent["min_y"],
                extent["max_x"],
                extent["max_y"],
                table_name,
            ),
        )
        conn.commit()
    return extent


def filter_output_to_prediction_domain(
    conn: sqlite3.Connection,
    *,
    table_name: str,
    geometry_column: str,
    domain_table: str,
    source_row_count: int,
) -> dict[str, Any]:
    """Keep only domain SUs in the output copy and refresh its extent/index."""

    quoted_table = quote_identifier(table_name, label="table name")
    quoted_domain = quote_identifier(domain_table, label="temporary domain table name")
    before_rows = int(conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0])
    if before_rows != int(source_row_count):
        raise MappingError(
            "The output table changed before domain filtering: "
            f"expected={source_row_count}, actual={before_rows}"
        )
    domain_rows = int(
        conn.execute(
            f"SELECT COUNT(*) FROM {quoted_table} AS g "
            f"INNER JOIN {quoted_domain} AS d ON d.fid = g.\"fid\""
        ).fetchone()[0]
    )
    conn.execute(
        f"DELETE FROM {quoted_table} AS g "
        f"WHERE NOT EXISTS (SELECT 1 FROM {quoted_domain} AS d WHERE d.fid = g.\"fid\")"
    )
    conn.commit()
    after_rows = int(conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0])
    if after_rows != domain_rows:
        raise MappingError(
            "Domain filtering produced an unexpected row count: "
            f"expected={domain_rows}, actual={after_rows}"
        )
    extent = update_gpkg_contents_extent(
        conn,
        table_name=table_name,
        geometry_column=geometry_column,
    )
    return {
        "enabled": True,
        "source_rows": int(before_rows),
        "retained_rows": int(after_rows),
        "excluded_rows": int(before_rows - after_rows),
        "extent": extent,
    }


def cleanup_temp_database(path: Path) -> None:
    for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm"), Path(f"{path}-journal")):
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            pass


def map_predictions(
    *,
    source_gpkg: Path,
    output_gpkg: Path,
    table_name: str,
    specs: list[ResultSpec],
    fid_column: str,
    su_id_column: str,
    skip_su_id_validation: bool,
    require_complete: bool,
    batch_rows: int,
    overwrite: bool,
    permafrost_mask: Path | None = None,
    permafrost_mask_value: int = DEFAULT_PERMAFROST_MASK_VALUE,
    permafrost_mode: str = DEFAULT_PERMAFROST_MODE,
    arcgis_compatible_crs: bool = True,
    restrict_to_prediction_domain: bool = True,
    compact_output: bool = True,
) -> dict[str, Any]:
    source_gpkg = source_gpkg.resolve()
    output_gpkg = output_gpkg.resolve()
    if source_gpkg == output_gpkg:
        raise ValueError("Source and output GeoPackage paths must be different")
    if restrict_to_prediction_domain and permafrost_mask is None:
        raise ValueError(
            "restrict_to_prediction_domain=True requires the permafrost mask; "
            "pass a mask path or explicitly disable domain-only output."
        )
    validate_result_specs(specs)
    source_summary = validate_source_gpkg(source_gpkg, table_name)
    if output_gpkg.exists() and not overwrite:
        raise FileExistsError(
            f"Output GeoPackage already exists: {output_gpkg}. Pass --overwrite to replace that output only."
        )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    temp_gpkg = output_gpkg.with_name(f".{output_gpkg.name}.{uuid.uuid4().hex}.tmp")
    started = time.perf_counter()
    emit(
        "gpkg_copy_start",
        source_gpkg=str(source_gpkg),
        temporary_gpkg=str(temp_gpkg),
        source_bytes=source_gpkg.stat().st_size,
    )
    try:
        shutil.copy2(source_gpkg, temp_gpkg)
        emit("gpkg_copy_complete", elapsed_seconds=round(time.perf_counter() - started, 3))
        result_summaries: list[dict[str, Any]] = []
        domain_table: str | None = None
        domain_summary: dict[str, Any] = {
            "enabled": False,
            "mode": None,
            "mask_path": None,
        }
        arcgis_crs_summary: dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
            "coordinate_values_changed": False,
        }
        domain_filter_summary: dict[str, Any] = {
            "enabled": False,
            "retained_rows": None,
            "excluded_rows": 0,
            "extent": None,
        }
        with closing(sqlite3.connect(temp_gpkg)) as conn:
            register_gpkg_spatial_compatibility_functions(conn)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA temp_store = FILE")
            conn.execute("PRAGMA synchronous = NORMAL")
            if permafrost_mask is not None:
                # PyArrow, Shapely, and Rasterio each load native DLLs on
                # Windows.  Import PyArrow before the GIS libraries used by
                # domain construction so later Parquet staging remains safe.
                import pyarrow  # noqa: F401

                domain_table, domain_summary = build_permafrost_domain_table(
                    conn,
                    table_name=table_name,
                    source_srs_id=int(source_summary["srs_id"]),
                    mask_path=permafrost_mask,
                    mask_value=int(permafrost_mask_value),
                    mode=str(permafrost_mode),
                    source_row_count=int(source_summary["row_count"]),
                    batch_rows=batch_rows,
                )
                emit("permafrost_domain_complete", **domain_summary)
            if arcgis_compatible_crs:
                arcgis_crs_summary = normalize_arcgis_compatible_crs(
                    conn,
                    table_name=table_name,
                    source_srs_id=int(source_summary["srs_id"]),
                )
                emit("arcgis_crs_normalized", **arcgis_crs_summary)
            for index, spec in enumerate(specs, start=1):
                emit(
                    "prediction_attach_start",
                    item=index,
                    items_total=len(specs),
                    parquet_path=str(spec.parquet_path),
                    output_field=spec.output_field,
                    value_column=spec.value_column,
                )
                stage_name = ""
                field_added = False
                try:
                    stage_name, staged_summary = stage_parquet(
                        conn,
                        spec,
                        fid_column=fid_column,
                        su_id_column=su_id_column,
                        skip_su_id_validation=skip_su_id_validation,
                        batch_rows=batch_rows,
                    )
                    field_added = add_prediction_column(
                        conn,
                        table_name,
                        spec.output_field,
                        field_type=staged_summary.get("field_type", "REAL"),
                    )
                    attached_summary = attach_staged_result(
                        conn,
                        table_name=table_name,
                        field_name=spec.output_field,
                        stage_name=stage_name,
                        source_row_count=int(source_summary["row_count"]),
                        skip_su_id_validation=skip_su_id_validation,
                        require_complete=require_complete,
                        domain_table=domain_table,
                    )
                finally:
                    if stage_name:
                        conn.execute(
                            f"DROP TABLE IF EXISTS {quote_identifier(stage_name, label='temporary table name')}"
                        )
                        conn.commit()
                item_summary = {
                    **staged_summary,
                    **attached_summary,
                    "field_added": bool(field_added),
                }
                result_summaries.append(item_summary)
                emit("prediction_attach_complete", **item_summary)

            if restrict_to_prediction_domain:
                if domain_table is None:
                    raise MappingError(
                        "Domain-only output was requested, but no permafrost domain table exists."
                    )
                domain_filter_summary = filter_output_to_prediction_domain(
                    conn,
                    table_name=table_name,
                    geometry_column=str(source_summary["geometry_column"]),
                    domain_table=domain_table,
                    source_row_count=int(source_summary["row_count"]),
                )
                emit("prediction_domain_filter_complete", **domain_filter_summary)
                conn.execute(
                    f"DROP TABLE IF EXISTS {quote_identifier(domain_table, label='temporary domain table name')}"
                )
                conn.commit()
                if compact_output:
                    compact_started = time.perf_counter()
                    emit("gpkg_compact_start")
                    conn.execute("VACUUM")
                    emit(
                        "gpkg_compact_complete",
                        elapsed_seconds=round(time.perf_counter() - compact_started, 3),
                    )

            timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            conn.execute(
                "UPDATE gpkg_contents SET last_change = ? WHERE table_name = ?",
                (timestamp, table_name),
            )
            conn.commit()
            integrity = str(conn.execute("PRAGMA quick_check").fetchone()[0])
            if integrity.lower() != "ok":
                raise MappingError(f"Generated GeoPackage failed PRAGMA quick_check: {integrity}")
            output_row_count = int(
                conn.execute(
                    f"SELECT COUNT(*) FROM {quote_identifier(table_name, label='table name')}"
                ).fetchone()[0]
            )
            expected_output_row_count = int(source_summary["row_count"])
            if restrict_to_prediction_domain:
                expected_output_row_count = int(domain_summary["in_domain_rows"])
            if output_row_count != expected_output_row_count:
                raise MappingError(
                    "Output slope-unit row count changed unexpectedly: "
                    f"expected={expected_output_row_count}, output={output_row_count}"
                )

        os.replace(temp_gpkg, output_gpkg)
        summary = {
            "source_gpkg": str(source_gpkg),
            "output_gpkg": str(output_gpkg),
            "table_name": table_name,
            "source": source_summary,
            "arcgis_crs": arcgis_crs_summary,
            "prediction_domain": domain_summary,
            "prediction_domain_filter": domain_filter_summary,
            "prediction_fields": [spec.output_field for spec in specs],
            "results": result_summaries,
            "output_bytes": output_gpkg.stat().st_size,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "original_source_modified": False,
            "integrity_check": "ok",
            "output_row_count": int(output_row_count),
            "output_domain_only": bool(restrict_to_prediction_domain),
        }
        return summary
    except Exception:
        cleanup_temp_database(temp_gpkg)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gpkg", type=Path, default=DEFAULT_SOURCE_GPKG)
    parser.add_argument("--output-gpkg", type=Path, default=DEFAULT_OUTPUT_GPKG)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        metavar="PARQUET::FIELD[::VALUE_COLUMN]",
        help=(
            "Prediction result to attach. Repeat for all batch-size/year combinations. "
            "If FIELD is omitted it is inferred from a bsNNN_YYYY filename."
        ),
    )
    parser.add_argument("--value-column", default=DEFAULT_VALUE_COLUMN)
    parser.add_argument("--fid-column", default="fid")
    parser.add_argument("--su-id-column", default="su_id")
    parser.add_argument("--skip-su-id-validation", action="store_true")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless every source slope unit has a prediction row.",
    )
    parser.add_argument("--batch-rows", type=int, default=DEFAULT_BATCH_ROWS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--permafrost-mask",
        type=Path,
        default=DEFAULT_PERMAFROST_MASK,
        help=(
            "2010 permafrost raster used to restrict the prediction domain. "
            "The default keeps complete SUs whose polygon intersects mask class 1."
        ),
    )
    parser.add_argument(
        "--permafrost-mask-value",
        type=int,
        default=DEFAULT_PERMAFROST_MASK_VALUE,
        help="Raster class treated as permafrost; defaults to 1 to match PA sampling.",
    )
    parser.add_argument(
        "--permafrost-mode",
        choices=PERMAFROST_MODES,
        default=DEFAULT_PERMAFROST_MODE,
        help=(
            "SU domain rule. 'intersects' keeps complete SUs with any polygon "
            "intersection with mask class 1; 'centroid' is the legacy rule."
        ),
    )
    parser.add_argument(
        "--no-permafrost-mask",
        action="store_true",
        help="Disable the default permafrost-only prediction domain (diagnostic use only).",
    )
    parser.add_argument(
        "--no-arcgis-compatible-crs",
        action="store_true",
        help=(
            "Keep the source CRS WKT unchanged (diagnostic use only; the default "
            "writes the QTP custom CRS in ESRI WKT1 form for ArcGIS Pro)."
        ),
    )
    parser.add_argument(
        "--keep-all-slope-units",
        action="store_true",
        help=(
            "Keep non-permafrost slope units in the output with NULL prediction "
            "fields (diagnostic use only; domain-only output is the default)."
        ),
    )
    parser.add_argument(
        "--no-compact-output",
        action="store_true",
        help=(
            "Skip VACUUM after domain filtering. The output remains domain-only "
            "but may retain unused SQLite pages."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional summary path; defaults beside the output GPKG.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if int(args.batch_rows) <= 0:
        raise ValueError("--batch-rows must be positive")
    specs = [
        parse_result_spec(raw, default_value_column=str(args.value_column))
        for raw in args.result
    ]
    summary = map_predictions(
        source_gpkg=args.source_gpkg,
        output_gpkg=args.output_gpkg,
        table_name=str(args.table_name),
        specs=specs,
        fid_column=str(args.fid_column),
        su_id_column=str(args.su_id_column),
        skip_su_id_validation=bool(args.skip_su_id_validation),
        require_complete=bool(args.require_complete),
        batch_rows=int(args.batch_rows),
        overwrite=bool(args.overwrite),
        permafrost_mask=None if args.no_permafrost_mask else args.permafrost_mask,
        permafrost_mask_value=int(args.permafrost_mask_value),
        permafrost_mode=str(args.permafrost_mode),
        arcgis_compatible_crs=not bool(args.no_arcgis_compatible_crs),
        restrict_to_prediction_domain=not bool(args.keep_all_slope_units),
        compact_output=not bool(args.no_compact_output),
    )
    summary_path = (
        args.summary_json.resolve()
        if args.summary_json is not None
        else args.output_gpkg.resolve().with_suffix(".mapping_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("mapping_complete", summary_json=str(summary_path), **summary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("mapping_failed", error_type=type(exc).__name__, error=str(exc))
        raise
