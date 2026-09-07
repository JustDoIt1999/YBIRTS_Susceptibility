"""Small local prototype for slope-unit zonal statistics.

This deliberately does not run model inference.  It uses one completed raster
tile from the existing feature cache, queries only intersecting slope units
through the GeoPackage RTree, rasterizes their ``fid`` values, and computes
grouped statistics with NumPy.  The output is marked as tile-partial because a
slope unit can cross tile boundaries.

The cached static channel named ``Slope_std`` is the source Slope pixel layer
under the current derived-feature cache contract.  Its SU-level stdDev is
computed here; it is not read as a pre-existing raster statistic.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkb
from rasterio.features import rasterize
from rasterio.transform import Affine


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = WORKSPACE_ROOT / "Susceptibility_Mapping" / "output" / "json" / "annual" / "annual_cache_2021_progress.json"
DEFAULT_SLOPE_UNITS = WORKSPACE_ROOT / "Dataset_Construction" / "inputs" / "QTP_SlopeUnits.gpkg"
DEFAULT_OUTPUT = WORKSPACE_ROOT / "Susceptibility_Mapping" / "output" / "parquet" / "prototype" / "su_zonal_2021_tile0.parquet"

STATIC_CACHE_COLS = [
    "Slope",
    "Slope_std",  # source Slope pixels; SU-level stdDev is computed below
    "Northness",
    "Eastness",
    "TWI",
    "Local_Relief",
    "Soil",
    "Vegetation_type",
    "GIC",
]
DYNAMIC_CACHE_COLS = [
    "TDD",
    "SMDP",
    "Precipitation",
    "WSTmax",
    "WSRH",
    "NDVI",
    "Summer_SoilMoisture",
]
NODATA = -9999.0
GPKG_ENVELOPE_BYTES = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--tile-index", type=int, default=0, help="Index in manifest.tile_order for a completed tile.")
    parser.add_argument("--slope-units", type=Path, default=DEFAULT_SLOPE_UNITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def decode_gpkg_geometry(blob: bytes):
    if blob[:2] != b"GP":
        raise ValueError("Expected a GeoPackage geometry blob with GP header.")
    envelope_type = (int(blob[3]) >> 1) & 0x07
    header_len = 8 + GPKG_ENVELOPE_BYTES.get(envelope_type, 0)
    return wkb.loads(blob[header_len:])


def query_tile_slope_units(gpkg_path: Path, tile: dict[str, Any]) -> list[dict[str, Any]]:
    xmin = float(tile["xmin"])
    xmax = xmin + int(tile["width"]) * 100.0
    ymin = float(tile["ymin"])
    ymax = float(tile["ymax"])
    sql = """
        SELECT s.fid, s.su_id, s.basin_id, s.area_km2, s.geom
        FROM rtree_qtp_slope_units_geom AS r
        JOIN qtp_slope_units AS s ON s.fid = r.id
        WHERE r.maxx >= ? AND r.minx <= ?
          AND r.maxy >= ? AND r.miny <= ?
        ORDER BY s.fid
    """
    rows: list[dict[str, Any]] = []
    uri = f"file:{gpkg_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        for fid, su_id, basin_id, area_km2, geom_blob in conn.execute(sql, (xmin, xmax, ymin, ymax)):
            geom = decode_gpkg_geometry(geom_blob)
            geom_minx, geom_miny, geom_maxx, geom_maxy = geom.bounds
            rows.append(
                {
                    "fid": int(fid),
                    "su_id": str(su_id),
                    "basin_id": str(basin_id),
                    "area_km2": float(area_km2) if area_km2 is not None else np.nan,
                    "geometry": geom,
                    "fully_contained": bool(
                        geom_minx >= xmin
                        and geom_miny >= ymin
                        and geom_maxx <= xmax
                        and geom_maxy <= ymax
                    ),
                }
            )
    return rows


def load_npz(path: Path, *, expect_mask: bool) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as payload:
        if "feature_array" not in payload:
            raise KeyError(f"Missing feature_array in {path}")
        feature_array = payload["feature_array"].astype(np.float32, copy=False)
        mask = payload["aoi_mask"].astype(np.float32, copy=False) if expect_mask else None
    return feature_array, mask


def grouped_mean(values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_domain & np.isfinite(values) & (values != NODATA)
    ids, inverse = np.unique(labels[valid], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.int64)
    sums = np.bincount(inverse, weights=values[valid], minlength=len(ids))
    return ids, counts, sums / np.maximum(counts, 1)


def grouped_std(values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_domain & np.isfinite(values) & (values != NODATA)
    ids, inverse = np.unique(labels[valid], return_inverse=True)
    counts = np.bincount(inverse, minlength=len(ids)).astype(np.int64)
    sums = np.bincount(inverse, weights=values[valid], minlength=len(ids))
    sums_sq = np.bincount(inverse, weights=np.square(values[valid], dtype=np.float64), minlength=len(ids))
    means = sums / np.maximum(counts, 1)
    variance = np.maximum(sums_sq / np.maximum(counts, 1) - np.square(means), 0.0)
    result = np.sqrt(variance)
    result[counts < 2] = np.nan
    return ids, counts, result


def grouped_mode(values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = valid_domain & np.isfinite(values) & (values != NODATA)
    ids, inverse = np.unique(labels[valid], return_inverse=True)
    integer_values = np.rint(values[valid]).astype(np.int64)
    categories = np.unique(integer_values)
    counts = np.zeros((len(ids), len(categories)), dtype=np.int32)
    for category_index, category in enumerate(categories):
        counts[:, category_index] = np.bincount(
            inverse[integer_values == category],
            minlength=len(ids),
        )
    selected = np.argmax(counts, axis=1)
    return ids, counts.sum(axis=1, dtype=np.int64), categories[selected].astype(np.int32)


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    tile_order = list(manifest.get("tile_order") or [])
    if not tile_order:
        raise ValueError(f"Manifest has no tile_order: {args.manifest}")
    if args.tile_index < 0 or args.tile_index >= len(tile_order):
        raise IndexError(f"tile-index={args.tile_index} outside 0..{len(tile_order) - 1}")

    tile_id = str(tile_order[args.tile_index])
    tile_state = manifest["tiles"][tile_id]
    tile = dict(tile_state["tile"])
    if str(tile_state.get("status")) != "done":
        raise RuntimeError(f"Selected tile is not complete: {tile_id} status={tile_state.get('status')}")
    static_path = Path(str(tile_state["static_cache_path"])).resolve()
    dynamic_path = Path(str(tile_state["dynamic_cache_path"])).resolve()
    static_array, aoi_mask = load_npz(static_path, expect_mask=True)
    dynamic_array, _ = load_npz(dynamic_path, expect_mask=False)
    if static_array.shape[:2] != dynamic_array.shape[:2]:
        raise ValueError(f"Static/dynamic shape mismatch: {static_array.shape} vs {dynamic_array.shape}")
    if static_array.shape[2] != len(STATIC_CACHE_COLS):
        raise ValueError(f"Unexpected static channel count: {static_array.shape[2]} != {len(STATIC_CACHE_COLS)}")
    if dynamic_array.shape[2] != len(DYNAMIC_CACHE_COLS):
        raise ValueError(f"Unexpected dynamic channel count: {dynamic_array.shape[2]} != {len(DYNAMIC_CACHE_COLS)}")

    units = query_tile_slope_units(args.slope_units.resolve(), tile)
    if not units:
        raise RuntimeError(f"No slope units intersect tile {tile_id}")
    transform = Affine(
        100.0,
        0.0,
        float(tile["xmin"]),
        0.0,
        -100.0,
        float(tile["ymax"]),
    )
    labels = rasterize(
        ((unit["geometry"], unit["fid"]) for unit in units),
        out_shape=static_array.shape[:2],
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )
    valid_domain = (labels > 0) & (aoi_mask > 0)
    frame = pd.DataFrame(
        [
            {
                "fid": unit["fid"],
                "su_id": unit["su_id"],
                "basin_id": unit["basin_id"],
                "area_km2": unit["area_km2"],
                "tile_id": tile_id,
                "fully_contained_in_tile": unit["fully_contained"],
            }
            for unit in units
        ]
    ).set_index("fid")
    frame["pixel_count"] = pd.Series(np.bincount(labels[valid_domain].astype(np.int64), minlength=int(frame.index.max()) + 1), dtype="Int64").reindex(frame.index).fillna(0).astype("int64").to_numpy()

    static_index = {name: index for index, name in enumerate(STATIC_CACHE_COLS)}
    dynamic_index = {name: index for index, name in enumerate(DYNAMIC_CACHE_COLS)}
    for name in ["Slope", "Northness", "Eastness", "TWI", "Local_Relief", "Soil", "GIC"]:
        ids, counts, means = grouped_mean(static_array[:, :, static_index[name]], labels, valid_domain)
        frame.loc[ids, name] = means
        frame.loc[ids, f"{name}_pixel_count"] = counts

    ids, counts, slope_std = grouped_std(static_array[:, :, static_index["Slope_std"]], labels, valid_domain)
    frame.loc[ids, "Slope_std"] = slope_std
    frame.loc[ids, "Slope_std_pixel_count"] = counts

    ids, counts, vegetation_mode = grouped_mode(static_array[:, :, static_index["Vegetation_type"]], labels, valid_domain)
    frame.loc[ids, "Vegetation_type"] = vegetation_mode
    frame.loc[ids, "Vegetation_type_pixel_count"] = counts

    for name in DYNAMIC_CACHE_COLS:
        ids, counts, means = grouped_mean(dynamic_array[:, :, dynamic_index[name]], labels, valid_domain)
        frame.loc[ids, name] = means
        frame.loc[ids, f"{name}_pixel_count"] = counts

    frame = frame.reset_index()
    frame["stats_scope"] = "tile_partial"
    frame["feature_year"] = int(args.manifest.stem.split("_")[-2]) if args.manifest.stem.startswith("annual_cache_") else 2021
    frame["static_cache_path"] = str(static_path)
    frame["dynamic_cache_path"] = str(dynamic_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False)

    summary = {
        "output": str(args.output.resolve()),
        "tile_id": tile_id,
        "tile_shape": list(static_array.shape[:2]),
        "intersecting_slope_units": int(len(units)),
        "fully_contained_slope_units": int(sum(unit["fully_contained"] for unit in units)),
        "crossing_slope_units": int(sum(not unit["fully_contained"] for unit in units)),
        "rasterized_slope_units": int(np.unique(labels[labels > 0]).size),
        "valid_pixels": int(valid_domain.sum()),
        "output_rows": int(len(frame)),
        "static_cache_mb": round(static_path.stat().st_size / 1024**2, 3),
        "dynamic_cache_mb": round(dynamic_path.stat().st_size / 1024**2, 3),
        "stats_scope": "tile_partial",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
