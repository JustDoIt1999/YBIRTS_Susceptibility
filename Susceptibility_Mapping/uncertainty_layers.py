from __future__ import annotations

from typing import Any

import numpy as np


UNCERTAINTY_METRIC_NAMES = ("mean", "std", "cv", "q05", "q95", "entropy")
DEFAULT_UNCERTAINTY_EXPORT_METRICS = ("std",)


def summarize_prediction_members(
    member_predictions: np.ndarray,
    *,
    epsilon: float = 1e-6,
    metric_names: tuple[str, ...] | list[str] | None = None,
) -> dict[str, np.ndarray]:
    matrix = np.asarray(member_predictions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("member_predictions must be a non-empty 2D array.")
    requested = tuple(metric_names or UNCERTAINTY_METRIC_NAMES)
    unknown = sorted(set(requested) - set(UNCERTAINTY_METRIC_NAMES))
    if unknown:
        raise ValueError(f"Unsupported uncertainty metrics: {unknown}")
    if "mean" not in requested:
        requested = ("mean", *requested)

    mean = np.mean(matrix, axis=0)
    out: dict[str, np.ndarray] = {
        "mean": mean.astype(np.float32, copy=False),
    }
    std: np.ndarray | None = None
    if "std" in requested or "cv" in requested:
        std = np.std(matrix, axis=0, ddof=0)
    if "std" in requested:
        out["std"] = std.astype(np.float32, copy=False)
    if "cv" in requested:
        assert std is not None
        cv = np.divide(std, mean, out=np.full_like(std, np.nan), where=mean > float(epsilon))
        out["cv"] = cv.astype(np.float32, copy=False)
    if "q05" in requested:
        q05 = np.quantile(matrix, 0.05, axis=0)
        out["q05"] = q05.astype(np.float32, copy=False)
    if "q95" in requested:
        q95 = np.quantile(matrix, 0.95, axis=0)
        out["q95"] = q95.astype(np.float32, copy=False)
    if "entropy" in requested:
        clipped = np.clip(mean, epsilon, 1.0 - epsilon)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        out["entropy"] = entropy.astype(np.float32, copy=False)
    return out


def sample_finite_values(
    values: np.ndarray,
    *,
    max_values: int,
    rng: np.random.Generator,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size <= int(max_values):
        return finite.astype(np.float32, copy=False)
    indices = rng.choice(finite.size, size=int(max_values), replace=False)
    return finite[indices].astype(np.float32, copy=False)


def compute_quantile_thresholds(sample: np.ndarray, *, class_count: int) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot compute class thresholds from an empty sample.")
    if int(class_count) < 2:
        raise ValueError("class_count must be at least 2.")
    quantiles = np.linspace(0.0, 1.0, int(class_count) + 1, dtype=np.float64)[1:-1]
    thresholds = np.quantile(finite, quantiles)
    return np.asarray(thresholds, dtype=np.float32)


def compute_fixed_width_thresholds(
    *,
    start: float,
    stop: float,
    class_count: int,
) -> np.ndarray:
    if int(class_count) < 2:
        raise ValueError("class_count must be at least 2.")
    edges = np.linspace(float(start), float(stop), int(class_count) + 1, dtype=np.float64)
    return np.asarray(edges[1:-1], dtype=np.float32)


def compute_jenks_thresholds(sample: np.ndarray, *, class_count: int) -> np.ndarray:
    values = np.asarray(sample, dtype=np.float64).reshape(-1)
    finite = np.sort(values[np.isfinite(values)])
    n = int(finite.size)
    if n == 0:
        raise ValueError("Cannot compute Jenks thresholds from an empty sample.")
    if int(class_count) < 2:
        raise ValueError("class_count must be at least 2.")
    class_count = min(int(class_count), n)
    if class_count <= 1:
        return np.asarray([], dtype=np.float32)

    lower = np.zeros((n + 1, class_count + 1), dtype=np.int32)
    variance = np.full((n + 1, class_count + 1), np.inf, dtype=np.float64)
    for cls in range(1, class_count + 1):
        lower[1, cls] = 1
        variance[1, cls] = 0.0
    variance[0, 1:] = 0.0

    for end in range(1, n + 1):
        # Accumulate in the same reverse order as the scalar recurrence.
        values = finite[:end][::-1]
        sums = np.cumsum(values)
        squared_sums = np.cumsum(values * values)
        segment_variances = squared_sums - sums * sums / np.arange(1, end + 1)
        if end > 1:
            candidates = segment_variances[:-1, None] + variance[end - 1:0:-1, 1:class_count]
            best_spans = np.argmin(candidates, axis=0)
            best_variances = candidates[best_spans, np.arange(class_count - 1)]
            improved = best_variances < variance[end, 2:]
            # argmin chooses the shortest span on ties, matching the scalar recurrence's strict '<'.
            lower[end, 2:][improved] = end - best_spans[improved]
            variance[end, 2:][improved] = best_variances[improved]
        lower[end, 1] = 1
        variance[end, 1] = segment_variances[-1]

    thresholds = np.zeros(class_count - 1, dtype=np.float64)
    end = n
    for cls in range(class_count, 1, -1):
        idx = lower[end, cls]
        thresholds[cls - 2] = finite[idx - 2] if idx > 1 else finite[0]
        end = idx - 1
    return np.asarray(thresholds, dtype=np.float32)


def classify_by_thresholds(
    values: np.ndarray,
    *,
    thresholds: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    thresholds_arr = np.asarray(thresholds, dtype=np.float32).reshape(-1)
    if valid_mask is None:
        mask = np.isfinite(arr)
    else:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != arr.shape:
            raise ValueError("valid_mask must have the same shape as values.")
    classes = np.zeros(arr.shape, dtype=np.int16)
    if np.any(mask):
        classes[mask] = np.searchsorted(thresholds_arr, arr[mask], side="right").astype(np.int16) + 1
    return classes


def encode_confidence_classes(
    susceptibility_classes: np.ndarray,
    uncertainty_classes: np.ndarray,
    *,
    nodata_value: int = -9999,
) -> np.ndarray:
    susc = np.asarray(susceptibility_classes, dtype=np.int16)
    unc = np.asarray(uncertainty_classes, dtype=np.int16)
    if susc.shape != unc.shape:
        raise ValueError("susceptibility_classes and uncertainty_classes must have the same shape.")
    out = np.full(susc.shape, int(nodata_value), dtype=np.int16)
    valid = (susc > 0) & (unc > 0)
    out[valid] = (unc[valid] * 10 + susc[valid]).astype(np.int16, copy=False)
    return out


def summarize_confidence_codes(codes: np.ndarray, *, nodata_value: int = -9999) -> list[dict[str, Any]]:
    arr = np.asarray(codes, dtype=np.int32).reshape(-1)
    valid = arr[arr != int(nodata_value)]
    if valid.size == 0:
        return []
    unique, counts = np.unique(valid, return_counts=True)
    total = int(counts.sum())
    rows: list[dict[str, Any]] = []
    for code, count in zip(unique.tolist(), counts.tolist()):
        rows.append(
            {
                "confidence_code": int(code),
                "uncertainty_class": int(code // 10),
                "susceptibility_class": int(code % 10),
                "pixel_count": int(count),
                "pixel_ratio": float(count / total),
            }
        )
    return rows


__all__ = [
    "UNCERTAINTY_METRIC_NAMES",
    "DEFAULT_UNCERTAINTY_EXPORT_METRICS",
    "classify_by_thresholds",
    "compute_fixed_width_thresholds",
    "compute_jenks_thresholds",
    "compute_quantile_thresholds",
    "encode_confidence_classes",
    "sample_finite_values",
    "summarize_confidence_codes",
    "summarize_prediction_members",
]
