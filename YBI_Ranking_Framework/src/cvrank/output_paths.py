"""Shared output fields and root rewriting for configuration path tags."""
from __future__ import annotations

from pathlib import Path
from typing import Any


OUTPUT_DIRECTORY_KEYS = frozenset({"output_root", "output_dir"})
OUTPUT_FILE_KEYS = frozenset({
    "model_path",
    "scaler_path",
    "metrics_path",
    "validation_metrics_path",
    "history_path",
    "fold_runs_manifest_path",
    "runs_csv",
    "summary_csv",
    "run_config_path",
})


def rewrite_under_root(path: Path, old_root: Path | None, new_root: Path | None) -> tuple[Path, bool]:
    if old_root is None or new_root is None:
        return path, False
    try:
        relative = path.relative_to(old_root)
    except ValueError:
        return path, False
    return new_root / relative, True


def lambda_label(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.replace(".", "p")
