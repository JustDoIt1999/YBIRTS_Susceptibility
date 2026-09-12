"""Streaming slope-unit feature extraction from cached raster tiles.

The script keeps slope units as the feature/prediction unit while retaining
tiles only as an I/O partition.  It queries the GeoPackage RTree per tile,
rasterizes polygon ``fid`` labels, accumulates grouped statistics in disk-backed
NumPy arrays, and writes chunked Parquet tables.

The current cache contract stores the source Slope pixels in the static channel
named ``Slope_std``.  The SU-level ``Slope_std`` is calculated from those source
pixels here.  This is intentionally separate from pixel-level model inference.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from shapely import wkb
from rasterio.features import rasterize
from rasterio.transform import Affine

from Susceptibility_Mapping.su_zonal_prototype import (
    NODATA,
    grouped_mean,
    grouped_mode,
    grouped_std,
    load_manifest,
    load_npz,
    query_tile_slope_units,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATIC_MANIFEST = WORKSPACE_ROOT / "Susceptibility_Mapping" / "output" / "json" / "annual" / "annual_cache_2021_progress.json"
DEFAULT_DYNAMIC_MANIFEST = DEFAULT_STATIC_MANIFEST
DEFAULT_SLOPE_UNITS = WORKSPACE_ROOT / "Dataset_Construction" / "inputs" / "QTP_SlopeUnits.gpkg"
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "Susceptibility_Mapping" / "output" / "parquet" / "su_features"
DEFAULT_WORK_ROOT = WORKSPACE_ROOT / "Susceptibility_Mapping" / "tmp" / "su_accumulators"

STATIC_CACHE_COLS = [
    "Slope",
    "Slope_std",  # source Slope pixels; SU-level stdDev is computed below
    "Northness",
    "TWI",
    "Soil",
    "Vegetation_type",
    "GIC",
]
STATIC_MEAN_COLS = ["Slope", "Northness", "TWI", "Soil", "GIC"]
DYNAMIC_COLS = [
    "TDD",
    "SMDP",
    "Precipitation",
    "Summer_SoilMoisture",
    "NDVI",
]
MAX_VEGETATION_CLASS = 32
METADATA_CHUNK_SIZE = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("static", "dynamic"), required=True)
    parser.add_argument("--year", type=int, default=None, help="Dynamic feature year; required for --phase dynamic.")
    parser.add_argument("--static-manifest", type=Path, default=DEFAULT_STATIC_MANIFEST)
    parser.add_argument("--dynamic-manifest", type=Path, default=DEFAULT_DYNAMIC_MANIFEST)
    parser.add_argument("--slope-units", type=Path, default=DEFAULT_SLOPE_UNITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--max-tiles", type=int, default=None, help="Process only the first available tiles; for smoke tests.")
    parser.add_argument("--allow-partial", action="store_true", help="Use available cached tiles instead of requiring all manifest tiles.")
    parser.add_argument("--metadata-chunk-size", type=int, default=METADATA_CHUNK_SIZE)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing accumulator/output for this phase.")
    return parser.parse_args()


def gpkg_row_count_and_max_fid(gpkg_path: Path) -> tuple[int, int]:
    uri = f"file:{gpkg_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        count, max_fid = conn.execute("SELECT COUNT(*), MAX(fid) FROM qtp_slope_units").fetchone()
    if count is None or max_fid is None or int(count) <= 0:
        raise ValueError(f"Slope-unit table is empty: {gpkg_path}")
    return int(count), int(max_fid)


def choose_tiles(
    *,
    static_manifest: dict[str, Any],
    dynamic_manifest: dict[str, Any] | None,
    phase: str,
    allow_partial: bool,
    max_tiles: int | None,
) -> list[tuple[str, dict[str, Any], Path, Path | None]]:
    from Susceptibility_Mapping.projection_engine import output_leaf_tile_ids

    target_manifest = dynamic_manifest if (phase == "dynamic" and dynamic_manifest is not None) else static_manifest
    tiles_dict = target_manifest.get("tiles") or {}
    leaf_tile_ids = output_leaf_tile_ids(target_manifest)

    selected: list[tuple[str, dict[str, Any], Path, Path | None]] = []
    missing: list[str] = []
    for tile_id in leaf_tile_ids:
        tile_id = str(tile_id)
        tile_state = tiles_dict.get(tile_id) or {}
        static_raw = tile_state.get("static_cache_path")
        static_path = Path(str(static_raw)).resolve() if static_raw else None
        dynamic_path: Path | None = None
        if phase == "dynamic":
            dynamic_raw = tile_state.get("dynamic_cache_path")
            dynamic_path = Path(str(dynamic_raw)).resolve() if dynamic_raw else None
        required_paths = [static_path] if phase == "static" else [static_path, dynamic_path]
        if all(path is not None and path.exists() for path in required_paths):
            selected.append((tile_id, dict(tile_state.get("tile") or {}), static_path, dynamic_path))
        else:
            missing.append(tile_id)

    if not allow_partial and missing:
        raise RuntimeError(
            f"Missing cached inputs for {len(missing)} manifest tiles. "
            "Finish the tile cache first or pass --allow-partial for a diagnostic run."
        )
    if max_tiles not in (None, ""):
        selected = selected[: max(0, int(max_tiles))]
    if not selected:
        raise RuntimeError("No usable cached tiles were found for the requested phase.")
    return selected


class DiskAccumulator:
    """Disk-backed grouped-statistics accumulator indexed by GeoPackage fid."""

    def __init__(
        self,
        *,
        work_dir: Path,
        max_fid: int,
        mean_cols: list[str],
        include_static_extras: bool,
        overwrite: bool,
    ) -> None:
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_fid = int(max_fid)
        self.n_slots = self.max_fid + 1
        self.mean_cols = list(mean_cols)
        self.include_static_extras = bool(include_static_extras)
        self._arrays: dict[str, np.memmap] = {}
        self._spec_path = self.work_dir / "accumulator_spec.json"
        spec = {
            "max_fid": self.max_fid,
            "mean_cols": self.mean_cols,
            "include_static_extras": self.include_static_extras,
            "vegetation_classes": MAX_VEGETATION_CLASS,
        }
        if self._spec_path.exists() and not overwrite:
            existing = json.loads(self._spec_path.read_text(encoding="utf-8"))
            if existing != spec:
                raise RuntimeError(f"Accumulator specification mismatch: {self.work_dir}")
            mode = "r+"
        else:
            self._spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
            mode = "w+"

        self.sums = self._open("sums", np.float64, (self.n_slots, len(self.mean_cols)), mode)
        self.counts = self._open("counts", np.uint32, (self.n_slots, len(self.mean_cols)), mode)
        self.pixel_count = self._open("pixel_count", np.uint32, (self.n_slots,), mode)
        self.tile_count = self._open("tile_count", np.uint32, (self.n_slots,), mode)
        if self.include_static_extras:
            self.slope_sum = self._open("slope_sum", np.float64, (self.n_slots,), mode)
            self.slope_sum_sq = self._open("slope_sum_sq", np.float64, (self.n_slots,), mode)
            self.slope_count = self._open("slope_count", np.uint32, (self.n_slots,), mode)
            self.vegetation_counts = self._open(
                "vegetation_counts",
                np.uint32,
                (self.n_slots, MAX_VEGETATION_CLASS),
                mode,
            )

    def _open(self, name: str, dtype: Any, shape: tuple[int, ...], mode: str) -> np.memmap:
        path = self.work_dir / f"{name}.bin"
        array = np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        self._arrays[name] = array
        return array

    def update_base(self, labels: np.ndarray, valid_domain: np.ndarray) -> None:
        ids, inverse = np.unique(labels[valid_domain], return_inverse=True)
        counts = np.bincount(inverse, minlength=len(ids)).astype(np.uint32)
        self.pixel_count[ids] += counts
        self.tile_count[ids] += 1

    def update_mean(self, values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray, column_index: int) -> None:
        ids, counts, means = grouped_mean(values, labels, valid_domain)
        if len(ids) == 0:
            return
        sums = means.astype(np.float64) * counts.astype(np.float64)
        self.sums[ids, column_index] += sums
        self.counts[ids, column_index] += counts.astype(np.uint32)

    def update_slope_std(self, values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray) -> None:
        ids, counts, _ = grouped_std(values, labels, valid_domain)
        valid = valid_domain & np.isfinite(values) & (values != NODATA)
        ids, inverse = np.unique(labels[valid], return_inverse=True)
        sums = np.bincount(inverse, weights=values[valid], minlength=len(ids))
        sums_sq = np.bincount(inverse, weights=np.square(values[valid], dtype=np.float64), minlength=len(ids))
        self.slope_count[ids] += counts.astype(np.uint32)
        self.slope_sum[ids] += sums
        self.slope_sum_sq[ids] += sums_sq

    def update_vegetation(self, values: np.ndarray, labels: np.ndarray, valid_domain: np.ndarray) -> None:
        valid = valid_domain & np.isfinite(values) & (values != NODATA)
        if not valid.any():
            return
        ids, inverse = np.unique(labels[valid], return_inverse=True)
        categories = np.rint(values[valid]).astype(np.int64)
        for category in np.unique(categories):
            if category < 0 or category >= MAX_VEGETATION_CLASS:
                continue
            counts = np.bincount(inverse[categories == category], minlength=len(ids)).astype(np.uint32)
            self.vegetation_counts[ids, int(category)] += counts

    def flush(self) -> None:
        for array in self._arrays.values():
            array.flush()

    def close(self) -> None:
        self.flush()
        self._arrays.clear()

    def means_for(self, fids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sums = np.asarray(self.sums[fids], dtype=np.float64)
        counts = np.asarray(self.counts[fids], dtype=np.uint32)
        output = np.full(sums.shape, np.nan, dtype=np.float32)
        np.divide(sums, counts, out=output, where=counts > 0)
        return output, counts


def iter_metadata_chunks(gpkg_path: Path, chunk_size: int) -> Iterable[pd.DataFrame]:
    uri = f"file:{gpkg_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        cursor = conn.execute("SELECT fid, su_id, basin_id, area_km2 FROM qtp_slope_units ORDER BY fid")
        while True:
            rows = cursor.fetchmany(int(chunk_size))
            if not rows:
                break
            yield pd.DataFrame(rows, columns=["fid", "su_id", "basin_id", "area_km2"])


def write_output(
    *,
    accumulator: DiskAccumulator,
    gpkg_path: Path,
    output_path: Path,
    phase: str,
    year: int | None,
    processed_tile_count: int,
    expected_tile_count: int,
    chunk_size: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    writer = None
    total_rows = 0
    status_counts: dict[str, int] = {}
    try:
        for metadata in iter_metadata_chunks(gpkg_path, chunk_size):
            fids = metadata["fid"].to_numpy(dtype=np.int64)
            frame = metadata.copy()
            mean_values, mean_counts = accumulator.means_for(fids)
            for index, name in enumerate(accumulator.mean_cols):
                frame[name] = mean_values[:, index]
                frame[f"{name}_pixel_count"] = mean_counts[:, index]

            common_pixel_count = np.asarray(accumulator.pixel_count[fids], dtype=np.uint32)
            tile_count = np.asarray(accumulator.tile_count[fids], dtype=np.uint32)
            frame["pixel_count"] = common_pixel_count
            frame["tile_touch_count"] = tile_count

            if phase == "static":
                slope_count = np.asarray(accumulator.slope_count[fids], dtype=np.uint32)
                slope_sum = np.asarray(accumulator.slope_sum[fids], dtype=np.float64)
                slope_sum_sq = np.asarray(accumulator.slope_sum_sq[fids], dtype=np.float64)
                slope_std = np.full(len(fids), np.nan, dtype=np.float32)
                valid = slope_count >= 2
                slope_mean = np.zeros(len(fids), dtype=np.float64)
                np.divide(slope_sum, slope_count, out=slope_mean, where=slope_count > 0)
                variance = np.zeros(len(fids), dtype=np.float64)
                np.divide(slope_sum_sq, slope_count, out=variance, where=slope_count > 0)
                variance -= np.square(slope_mean)
                slope_std[valid] = np.sqrt(np.maximum(variance[valid], 0.0)).astype(np.float32)
                frame["Slope_std"] = slope_std
                frame["Slope_std_pixel_count"] = slope_count
                vegetation_counts = np.asarray(accumulator.vegetation_counts[fids], dtype=np.uint32)
                vegetation_total = vegetation_counts.sum(axis=1, dtype=np.uint64)
                vegetation_mode = np.full(len(fids), np.nan, dtype=np.float32)
                veg_valid = vegetation_total > 0
                vegetation_mode[veg_valid] = np.argmax(vegetation_counts[veg_valid], axis=1).astype(np.float32)
                frame["Vegetation_type"] = vegetation_mode
                frame["Vegetation_type_pixel_count"] = vegetation_total.astype(np.uint32)
                missing_required = frame[STATIC_MEAN_COLS].isna().any(axis=1).to_numpy()
                status = np.full(len(fids), "ok", dtype=object)
                status[common_pixel_count == 0] = "no_valid_pixel"
                status[(common_pixel_count > 0) & (slope_count < 2)] = "insufficient_slope_pixels"
                status[(common_pixel_count > 0) & (slope_count >= 2) & missing_required] = "missing_static_feature"
            else:
                missing_required = frame[accumulator.mean_cols].isna().any(axis=1).to_numpy()
                status = np.full(len(fids), "ok", dtype=object)
                status[common_pixel_count == 0] = "no_valid_pixel"
                status[(common_pixel_count > 0) & missing_required] = "missing_dynamic_feature"

            frame["stats_status"] = status
            frame["stats_scope"] = "complete_active_tiles" if processed_tile_count == expected_tile_count else "partial_active_tiles"
            frame["processed_tile_count"] = int(processed_tile_count)
            frame["expected_tile_count"] = int(expected_tile_count)
            if year is not None:
                frame["feature_year"] = int(year)
            total_rows += len(frame)
            for value in status.tolist():
                status_counts[str(value)] = status_counts.get(str(value), 0) + 1

            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(output_path), table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    return {
        "output_path": str(output_path.resolve()),
        "rows": int(total_rows),
        "status_counts": status_counts,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    phase = str(args.phase)
    if phase == "dynamic" and args.year is None:
        raise ValueError("--year is required for --phase dynamic")
    static_manifest_path = args.static_manifest.resolve()
    static_manifest = load_manifest(static_manifest_path)
    dynamic_manifest = load_manifest(args.dynamic_manifest.resolve()) if phase == "dynamic" else None
    units_count, max_fid = gpkg_row_count_and_max_fid(args.slope_units.resolve())
    tiles = choose_tiles(
        static_manifest=static_manifest,
        dynamic_manifest=dynamic_manifest,
        phase=phase,
        allow_partial=bool(args.allow_partial),
        max_tiles=args.max_tiles,
    )
    cache_tag = "static" if phase == "static" else f"dynamic_{int(args.year)}"
    work_dir = args.work_root.resolve() / cache_tag
    output_name = "static_features.parquet" if phase == "static" else f"dynamic_features_{int(args.year)}.parquet"
    output_path = args.output_root.resolve() / output_name
    if work_dir.exists() and not args.overwrite:
        raise RuntimeError(f"Accumulator already exists; pass --overwrite or use a new work root: {work_dir}")
    accumulator = DiskAccumulator(
        work_dir=work_dir,
        max_fid=max_fid,
        mean_cols=STATIC_MEAN_COLS if phase == "static" else DYNAMIC_COLS,
        include_static_extras=phase == "static",
        overwrite=bool(args.overwrite),
    )
    started = time.perf_counter()
    for tile_number, (tile_id, tile, static_path, dynamic_path) in enumerate(tiles, start=1):
        static_array, aoi_mask = load_npz(static_path, expect_mask=True)
        if phase == "dynamic":
            if dynamic_path is None:
                raise RuntimeError(f"Missing dynamic cache path for {tile_id}")
            dynamic_array, _ = load_npz(dynamic_path, expect_mask=False)
            if dynamic_array.shape[:2] != static_array.shape[:2]:
                raise ValueError(f"Static/dynamic shape mismatch for {tile_id}")
        units = query_tile_slope_units(args.slope_units.resolve(), tile)
        if not units:
            if tile_number == 1 or tile_number % 25 == 0 or tile_number == len(tiles):
                elapsed = max(time.perf_counter() - started, 1e-9)
                print(json.dumps({
                    "event": "tile_complete",
                    "phase": phase,
                    "tile": tile_number,
                    "tiles_total": len(tiles),
                    "tile_id": tile_id,
                    "intersecting_slope_units": 0,
                    "valid_pixels": 0,
                    "elapsed_seconds": round(elapsed, 3),
                    "tiles_per_second": round(tile_number / elapsed, 3),
                }, ensure_ascii=False))
            continue
        transform = Affine(100.0, 0.0, float(tile["xmin"]), 0.0, -100.0, float(tile["ymax"]))
        labels = rasterize(
            ((unit["geometry"], unit["fid"]) for unit in units),
            out_shape=static_array.shape[:2],
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
        valid_domain = (labels > 0) & (aoi_mask > 0)
        accumulator.update_base(labels, valid_domain)
        if phase == "static":
            static_index = {name: index for index, name in enumerate(STATIC_CACHE_COLS)}
            for index, name in enumerate(STATIC_MEAN_COLS):
                accumulator.update_mean(static_array[:, :, static_index[name]], labels, valid_domain, index)
            accumulator.update_slope_std(static_array[:, :, static_index["Slope_std"]], labels, valid_domain)
            accumulator.update_vegetation(static_array[:, :, static_index["Vegetation_type"]], labels, valid_domain)
        else:
            dynamic_index = {name: index for index, name in enumerate(DYNAMIC_COLS)}
            for index, name in enumerate(DYNAMIC_COLS):
                accumulator.update_mean(dynamic_array[:, :, dynamic_index[name]], labels, valid_domain, index)
        accumulator.flush()
        if tile_number == 1 or tile_number % 25 == 0 or tile_number == len(tiles):
            elapsed = max(time.perf_counter() - started, 1e-9)
            print(json.dumps({
                "event": "tile_complete",
                "phase": phase,
                "tile": tile_number,
                "tiles_total": len(tiles),
                "tile_id": tile_id,
                "intersecting_slope_units": len(units),
                "valid_pixels": int(valid_domain.sum()),
                "elapsed_seconds": round(elapsed, 3),
                "tiles_per_second": round(tile_number / elapsed, 3),
            }, ensure_ascii=False))

    result = write_output(
        accumulator=accumulator,
        gpkg_path=args.slope_units.resolve(),
        output_path=output_path,
        phase=phase,
        year=int(args.year) if args.year is not None else None,
        processed_tile_count=len(tiles),
        expected_tile_count=len((static_manifest.get("tile_order") or [])) if args.max_tiles in (None, "") else len(tiles),
        chunk_size=max(1, int(args.metadata_chunk_size)),
    )
    accumulator.close()
    elapsed = max(time.perf_counter() - started, 1e-9)
    summary = {
        "phase": phase,
        "year": int(args.year) if args.year is not None else None,
        "slope_unit_count": units_count,
        "max_fid": max_fid,
        "processed_tiles": len(tiles),
        "manifest_tiles": len(static_manifest.get("tile_order") or []),
        "elapsed_seconds": round(elapsed, 3),
        "output": result,
        "work_dir": str(work_dir),
        "allow_partial": bool(args.allow_partial),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    run(parse_args())
