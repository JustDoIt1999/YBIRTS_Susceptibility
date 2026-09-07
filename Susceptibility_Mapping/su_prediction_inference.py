"""Run the trained CVRank ensemble on slope-unit feature Parquet tables.

Slope units are the prediction units in this module.  Raster tiles have
already been reduced to one row per slope unit by ``su_feature_extractor``;
this script only streams those rows in bounded chunks, applies the exact
feature-preparation artifacts saved with each fold, and writes one prediction
Parquet per batch size and year.

The output contract is intentionally compatible with
``su_prediction_mapping.py``::

    pred_bs128_2021.parquet
    pred_bs256_2021.parquet
    ...

Each output contains ``fid``, ``su_id``, ``prediction`` and ensemble
uncertainty fields.  Missing model inputs are written as NaN and are reported
in ``prediction_status``; the later GPKG connection layer converts these to
NULL attributes for GIS use.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_ROOT = WORKSPACE_ROOT / "Susceptibility_Mapping"
DEFAULT_STATIC_FEATURES = PREDICTION_ROOT / "output" / "parquet" / "su_features" / "static_features.parquet"
DEFAULT_DYNAMIC_TEMPLATE = (
    PREDICTION_ROOT / "output" / "parquet" / "su_features" / "dynamic_features_{year}.parquet"
)
DEFAULT_OUTPUT_ROOT = PREDICTION_ROOT / "output" / "parquet" / "su_predictions"
DEFAULT_ROUTE = "dbscan"
DEFAULT_RANK_LOSS_MODE = "ndcg_aware_lambdarank"
DEFAULT_BATCH_SIZES = (128, 256, 512, 1024)
DEFAULT_YEARS = (2021, 2022)
DEFAULT_ROW_BATCH_SIZE = 50_000
DEFAULT_INFERENCE_BATCH_SIZE = 4_096

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from Susceptibility_Mapping import gee_context as ee_ctx  # noqa: E402
from Susceptibility_Mapping import projection_engine as engine  # noqa: E402


class PredictionError(RuntimeError):
    """Raised when SU prediction inputs or model artifacts are inconsistent."""


@dataclass
class LoadedModelBundle:
    batch_size: int
    config_path: Path
    config: dict[str, Any]
    backend_adapter: dict[str, Any]
    device: Any
    device_meta: dict[str, Any]
    feature_cols: list[str]
    members: list[dict[str, Any]]
    bundle_identity: dict[str, Any]


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False, default=str), flush=True)


def parse_int_list(values: list[str] | tuple[str, ...], *, label: str) -> list[int]:
    try:
        parsed = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain integers: {values!r}") from exc
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError(f"{label} must contain positive integers: {values!r}")
    return list(dict.fromkeys(parsed))


def load_model_bundle(
    *,
    batch_size: int,
    route: str,
    rank_loss_mode: str,
    train_output_suffix: str | None,
    model_backend: str,
    device_name: str,
) -> LoadedModelBundle:
    feature_cols = list(ee_ctx.CURRENT_RANK_FEATURE_COLS)
    config_path = engine.resolve_selected_train_config_path(
        route=route,
        batch_size=batch_size,
        rank_loss_mode=rank_loss_mode,
        train_output_suffix=train_output_suffix,
    )
    config = engine.load_training_config(config_path, model_backend)
    configured_feature_cols = [str(value) for value in config.get("feature_cols") or feature_cols]
    if configured_feature_cols != feature_cols:
        raise PredictionError(
            "The selected model feature schema does not match the SU feature table schema: "
            f"model={configured_feature_cols}; su={feature_cols}"
        )
    config["device"] = str(device_name)
    backend_adapter = engine.ensure_inference_dependencies(model_backend)
    device = backend_adapter["resolve_torch_device"](config)
    device_meta = backend_adapter["get_device_metadata"](config)
    members_meta, fold_manifest_path = engine.resolve_model_member_artifacts(
        config,
        backend_adapter=backend_adapter,
    )
    bundle_identity = engine.build_model_bundle_identity(
        train_config_path=config_path,
        model_backend=model_backend,
        members_meta=members_meta,
        fold_manifest_path=fold_manifest_path,
    )
    members = engine.load_model_scaler_members(
        config,
        feature_cols,
        device,
        backend_adapter=backend_adapter,
        members_meta=members_meta,
    )
    engine.verify_model_bundle_unchanged(
        bundle_identity,
        train_config_path=config_path,
        model_backend=model_backend,
        members_meta=members_meta,
        fold_manifest_path=fold_manifest_path,
    )
    emit(
        "model_bundle_loaded",
        batch_size=int(batch_size),
        config_path=str(config_path),
        fold_count=len(members),
        model_bundle_hash=bundle_identity["bundle_hash"],
        device=device_meta,
    )
    return LoadedModelBundle(
        batch_size=int(batch_size),
        config_path=config_path,
        config=config,
        backend_adapter=backend_adapter,
        device=device,
        device_meta=device_meta,
        feature_cols=feature_cols,
        members=members,
        bundle_identity=bundle_identity,
    )


def column_index(batch: Any, name: str, *, path: Path) -> int:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise PredictionError(f"Missing column {name!r} in {path}")
    return int(index)


def batch_to_frame(batch: Any, columns: list[str], *, path: Path) -> Any:
    engine.import_runtime_dependencies()
    data: dict[str, Any] = {}
    for name in columns:
        array = batch.column(column_index(batch, name, path=path))
        try:
            data[name] = array.to_numpy(zero_copy_only=False)
        except (TypeError, ValueError):
            data[name] = array.to_pylist()
    return engine.pd.DataFrame(data, columns=columns)


def get_batch_values(batch: Any, name: str, *, path: Path) -> list[Any]:
    return batch.column(column_index(batch, name, path=path)).to_pylist()


def iter_aligned_feature_batches(
    *,
    static_path: Path,
    dynamic_path: Path,
    feature_cols: list[str],
    row_batch_size: int,
    max_rows: int | None,
    max_batches: int | None,
) -> Iterator[tuple[Any, Any, int]]:
    import pyarrow.parquet as pq

    static_pf = pq.ParquetFile(static_path)
    dynamic_pf = pq.ParquetFile(dynamic_path)
    required_static = ["fid", "su_id", "stats_status", *feature_cols[:9]]
    required_dynamic = ["fid", "stats_status", *feature_cols[9:]]
    static_names = set(static_pf.schema_arrow.names)
    dynamic_names = set(dynamic_pf.schema_arrow.names)
    missing_static = sorted(set(required_static) - static_names)
    missing_dynamic = sorted(set(required_dynamic) - dynamic_names)
    if missing_static or missing_dynamic:
        raise PredictionError(
            f"Missing SU feature columns: static={missing_static}, dynamic={missing_dynamic}"
        )
    if static_pf.metadata.num_rows != dynamic_pf.metadata.num_rows:
        raise PredictionError(
            f"Static/dynamic row count mismatch: {static_pf.metadata.num_rows} != {dynamic_pf.metadata.num_rows}"
        )

    static_iter = static_pf.iter_batches(
        batch_size=max(1, int(row_batch_size)),
        columns=required_static,
        use_threads=True,
    )
    dynamic_iter = dynamic_pf.iter_batches(
        batch_size=max(1, int(row_batch_size)),
        columns=required_dynamic,
        use_threads=True,
    )
    rows_seen = 0
    batches_seen = 0
    while True:
        try:
            static_batch = next(static_iter)
        except StopIteration:
            try:
                next(dynamic_iter)
            except StopIteration:
                return
            raise PredictionError("Dynamic feature Parquet contains rows after static Parquet ended")
        try:
            dynamic_batch = next(dynamic_iter)
        except StopIteration as exc:
            raise PredictionError("Static feature Parquet contains rows after dynamic Parquet ended") from exc
        static_fids = static_batch.column(column_index(static_batch, "fid", path=static_path)).to_numpy(
            zero_copy_only=False
        )
        dynamic_fids = dynamic_batch.column(column_index(dynamic_batch, "fid", path=dynamic_path)).to_numpy(
            zero_copy_only=False
        )
        if len(static_fids) != len(dynamic_fids) or not (static_fids == dynamic_fids).all():
            raise PredictionError(
                f"Static/dynamic fid alignment failed around row {rows_seen + 1}: "
                f"static_first={static_fids[:3].tolist()}, dynamic_first={dynamic_fids[:3].tolist()}"
            )
        if max_rows not in (None, ""):
            remaining = int(max_rows) - rows_seen
            if remaining <= 0:
                return
            if len(static_fids) > remaining:
                static_batch = static_batch.slice(0, remaining)
                dynamic_batch = dynamic_batch.slice(0, remaining)
        rows = int(static_batch.num_rows)
        if rows <= 0:
            return
        rows_seen += rows
        batches_seen += 1
        yield static_batch, dynamic_batch, rows
        if max_batches not in (None, "") and batches_seen >= int(max_batches):
            return
        if max_rows not in (None, "") and rows_seen >= int(max_rows):
            return


def predict_su_batch(
    *,
    static_batch: Any,
    dynamic_batch: Any,
    static_path: Path,
    dynamic_path: Path,
    bundle: LoadedModelBundle,
    uncertainty_metrics: tuple[str, ...],
) -> dict[str, Any]:
    engine.import_runtime_dependencies()
    static_cols = bundle.feature_cols[:9]
    dynamic_cols = bundle.feature_cols[9:]
    static_frame = batch_to_frame(static_batch, static_cols, path=static_path)
    dynamic_frame = batch_to_frame(dynamic_batch, dynamic_cols, path=dynamic_path)
    features = engine.pd.concat([static_frame, dynamic_frame], axis=1).loc[:, bundle.feature_cols]
    feature_array = features.to_numpy(dtype=engine.np.float32, copy=True).reshape(-1, 1, len(bundle.feature_cols))
    aoi_mask = engine.np.ones((len(features), 1), dtype=engine.np.float32)
    blocks, prediction_summary = engine.predict_probability_block(
        feature_array,
        aoi_mask,
        bundle.feature_cols,
        bundle.members,
        bundle.device,
        DEFAULT_INFERENCE_BATCH_SIZE,
        engine.IN_MEMORY_NODATA_VALUE,
        apply_feature_preparation=bundle.backend_adapter["apply_feature_preparation_artifact"],
        uncertainty_metrics=uncertainty_metrics,
    )
    values: dict[str, Any] = {
        "fid": static_batch.column(column_index(static_batch, "fid", path=static_path)).to_numpy(
            zero_copy_only=False
        ).astype("int64", copy=False),
        "su_id": get_batch_values(static_batch, "su_id", path=static_path),
        "prediction": blocks["mean"].reshape(-1).astype(engine.np.float32, copy=False),
        "prediction_status": [],
        "static_stats_status": get_batch_values(static_batch, "stats_status", path=static_path),
        "dynamic_stats_status": get_batch_values(dynamic_batch, "stats_status", path=dynamic_path),
    }
    for metric_name in uncertainty_metrics:
        if metric_name in blocks:
            values[f"prediction_{metric_name}"] = blocks[metric_name].reshape(-1).astype(
                engine.np.float32, copy=False
            )
    prediction_array = values["prediction"]
    values["prediction_status"] = [
        "ok" if math.isfinite(float(value)) and float(value) != engine.IN_MEMORY_NODATA_VALUE else "invalid_model_input"
        for value in prediction_array
    ]
    return {"values": values, "summary": prediction_summary}


def output_path_for(output_root: Path, batch_size: int, year: int) -> Path:
    return output_root / f"pred_bs{int(batch_size)}_{int(year)}.parquet"


def run_one_output(
    *,
    static_path: Path,
    dynamic_path: Path,
    output_path: Path,
    bundle: LoadedModelBundle,
    year: int,
    row_batch_size: int,
    max_rows: int | None,
    max_batches: int | None,
    uncertainty_metrics: tuple[str, ...],
    overwrite: bool,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not static_path.is_file():
        raise FileNotFoundError(f"Static feature Parquet not found: {static_path}")
    if not dynamic_path.is_file():
        raise FileNotFoundError(f"Dynamic feature Parquet not found: {dynamic_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Prediction output exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
    writer = None
    started = time.perf_counter()
    rows_written = 0
    status_counts: Counter[str] = Counter()
    pred_min = float("inf")
    pred_max = float("-inf")
    pred_sum = 0.0
    try:
        emit(
            "su_prediction_start",
            batch_size=bundle.batch_size,
            year=int(year),
            static_path=str(static_path),
            dynamic_path=str(dynamic_path),
            output_path=str(output_path),
            row_batch_size=int(row_batch_size),
        )
        for batch_number, (static_batch, dynamic_batch, row_count) in enumerate(
            iter_aligned_feature_batches(
                static_path=static_path,
                dynamic_path=dynamic_path,
                feature_cols=bundle.feature_cols,
                row_batch_size=row_batch_size,
                max_rows=max_rows,
                max_batches=max_batches,
            ),
            start=1,
        ):
            predicted = predict_su_batch(
                static_batch=static_batch,
                dynamic_batch=dynamic_batch,
                static_path=static_path,
                dynamic_path=dynamic_path,
                bundle=bundle,
                uncertainty_metrics=uncertainty_metrics,
            )
            data = predicted["values"]
            table = pa.Table.from_pydict(data)
            metadata = dict(table.schema.metadata or {})
            metadata[b"su_prediction_year"] = str(int(year)).encode("utf-8")
            metadata[b"su_prediction_batch_size"] = str(int(bundle.batch_size)).encode("utf-8")
            metadata[b"su_prediction_model_bundle_hash"] = str(bundle.bundle_identity["bundle_hash"]).encode("utf-8")
            metadata[b"su_prediction_feature_cols"] = json.dumps(bundle.feature_cols).encode("utf-8")
            table = table.replace_schema_metadata(metadata)
            if writer is None:
                writer = pq.ParquetWriter(str(temp_path), table.schema, compression="zstd")
            writer.write_table(table)
            predictions = data["prediction"]
            finite_mask = engine.np.isfinite(predictions) & (predictions != engine.IN_MEMORY_NODATA_VALUE)
            finite = predictions[finite_mask]
            if finite.size:
                pred_min = min(pred_min, float(finite.min()))
                pred_max = max(pred_max, float(finite.max()))
                pred_sum += float(finite.astype(engine.np.float64).sum())
            status_counts.update(str(value) for value in data["prediction_status"])
            rows_written += int(row_count)
            if batch_number == 1 or batch_number % 10 == 0:
                emit(
                    "su_prediction_progress",
                    batch_size=bundle.batch_size,
                    year=int(year),
                    batch=batch_number,
                    rows=rows_written,
                    predicted_rows=int(sum(status_counts.get("ok", 0) for _ in [0])),
                )
        if writer is None:
            raise PredictionError(f"No rows were written for {output_path}")
        writer.close()
        writer = None
        os.replace(temp_path, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        try:
            if temp_path.exists():
                temp_path.unlink()
        except OSError:
            pass
        raise

    finite_count = int(status_counts.get("ok", 0))
    summary = {
        "output_path": str(output_path.resolve()),
        "batch_size": int(bundle.batch_size),
        "year": int(year),
        "rows": int(rows_written),
        "predicted_rows": finite_count,
        "status_counts": dict(status_counts),
        "prediction_min": None if finite_count == 0 else float(pred_min),
        "prediction_max": None if finite_count == 0 else float(pred_max),
        "prediction_mean": None if finite_count == 0 else float(pred_sum / finite_count),
        "model_config_path": str(bundle.config_path),
        "model_bundle_hash": bundle.bundle_identity["bundle_hash"],
        "fold_count": len(bundle.members),
        "device": bundle.device_meta,
        "row_batch_size": int(row_batch_size),
        "max_rows": None if max_rows in (None, "") else int(max_rows),
        "max_batches": None if max_batches in (None, "") else int(max_batches),
        "complete_input": max_rows in (None, "") and max_batches in (None, ""),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("su_prediction_complete", **summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-features", type=Path, default=DEFAULT_STATIC_FEATURES)
    parser.add_argument("--dynamic-template", type=str, default=str(DEFAULT_DYNAMIC_TEMPLATE))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--years", type=int, nargs="+", default=list(DEFAULT_YEARS))
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES))
    parser.add_argument("--route", default=DEFAULT_ROUTE)
    parser.add_argument("--rank-loss-mode", default=DEFAULT_RANK_LOSS_MODE)
    parser.add_argument("--train-output-suffix", default=None)
    parser.add_argument("--model-backend", default="cvrank")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--row-batch-size", type=int, default=DEFAULT_ROW_BATCH_SIZE)
    parser.add_argument("--inference-batch-size", type=int, default=DEFAULT_INFERENCE_BATCH_SIZE)
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for a smoke test.")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches for a smoke test.")
    parser.add_argument(
        "--uncertainty-metrics",
        nargs="+",
        default=["std", "q05", "q95"],
        choices=["std", "cv", "q05", "q95", "entropy"],
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    engine.import_runtime_dependencies()
    years = parse_int_list(args.years, label="--years")
    batch_sizes = parse_int_list(args.batch_sizes, label="--batch-sizes")
    if args.row_batch_size <= 0 or args.inference_batch_size <= 0:
        raise ValueError("--row-batch-size and --inference-batch-size must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when supplied")
    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive when supplied")

    # The shared engine function reads the configured inference batch size from
    # the runtime job.  The SU loop passes the explicit value through this
    # module-level setting so the chunk size and model batch size are separate.
    global DEFAULT_INFERENCE_BATCH_SIZE
    DEFAULT_INFERENCE_BATCH_SIZE = int(args.inference_batch_size)

    output_root = args.output_root.resolve()
    static_path = args.static_features.resolve()
    summaries: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        bundle = load_model_bundle(
            batch_size=batch_size,
            route=str(args.route),
            rank_loss_mode=str(args.rank_loss_mode),
            train_output_suffix=args.train_output_suffix,
            model_backend=str(args.model_backend),
            device_name=str(args.device),
        )
        for year in years:
            dynamic_path = Path(str(args.dynamic_template).format(year=int(year))).expanduser().resolve()
            output_path = output_path_for(output_root, batch_size, year)
            summaries.append(
                run_one_output(
                    static_path=static_path,
                    dynamic_path=dynamic_path,
                    output_path=output_path,
                    bundle=bundle,
                    year=year,
                    row_batch_size=int(args.row_batch_size),
                    max_rows=args.max_rows,
                    max_batches=args.max_batches,
                    uncertainty_metrics=tuple(str(value) for value in args.uncertainty_metrics),
                    overwrite=bool(args.overwrite),
                )
            )
    manifest = {
        "static_features": str(static_path),
        "dynamic_template": str(args.dynamic_template),
        "output_root": str(output_root),
        "route": str(args.route),
        "rank_loss_mode": str(args.rank_loss_mode),
        "batch_sizes": batch_sizes,
        "years": years,
        "results": summaries,
    }
    manifest_path = output_root / "su_prediction_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    emit("su_prediction_run_complete", manifest_path=str(manifest_path), outputs=len(summaries))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("su_prediction_failed", error_type=type(exc).__name__, error=str(exc))
        raise
