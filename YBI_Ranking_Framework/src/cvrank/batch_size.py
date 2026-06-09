from __future__ import annotations

from pathlib import Path
from typing import Any


BATCH_SIZE_TAG_KEY = "batch_size_tag"
BATCH_SIZE_SOURCE_KEY = "batch_size_source"
BATCH_SIZE_SOURCE_CONFIG = "config"
BATCH_SIZE_SOURCE_CLI_OVERRIDE = "cli_override"

_DIRECTORY_KEYS = {
    "output_root",
    "output_dir",
}
_FILE_KEYS = {
    "model_path",
    "scaler_path",
    "metrics_path",
    "validation_metrics_path",
    "history_path",
    "fold_runs_manifest_path",
    "runs_csv",
    "summary_csv",
    "run_config_path",
}
_NAME_ONLY_KEYS = {
    "distribution_filename",
    "pr_roc_filename",
}
_INSERT_BEFORE_MARKERS = (
    "_validation_metrics",
    "_train_metrics",
    "_train_history",
    "_best_train",
    "_raster_manifest",
    "_model",
    "_scaler",
)


def normalize_batch_size(value: Any) -> int:
    try:
        batch_size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid batch size: {value!r}") from exc
    if batch_size <= 0:
        raise ValueError(f"Batch size must be a positive integer, got {batch_size}.")
    return batch_size


def make_batch_size_tag(value: Any) -> str:
    return f"bs_{normalize_batch_size(value)}"


def get_batch_size_tag(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    value = config.get(BATCH_SIZE_TAG_KEY)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_existing_batch_tag_suffix(stem: str, batch_tag: str) -> bool:
    token = f"_{batch_tag}_"
    suffixes = (
        f"_{batch_tag}",
        f"_{batch_tag}_strategy",
    )
    return any(stem.endswith(suffix) for suffix in suffixes) or token in stem


def tag_name(name: str, tag: str | None) -> str:
    clean_tag = str(tag or "").strip()
    if not clean_tag:
        return name

    path = Path(name)
    suffixes = path.suffixes
    suffix = "".join(suffixes)
    stem = name[: -len(suffix)] if suffix else name
    if _has_existing_batch_tag_suffix(stem, clean_tag):
        return name

    for marker in _INSERT_BEFORE_MARKERS:
        if not stem.endswith(marker):
            continue
        prefix = stem[: -len(marker)]
        if _has_existing_batch_tag_suffix(prefix, clean_tag):
            return name
        if prefix:
            return f"{prefix}_{clean_tag}{marker}{suffix}"

    return f"{stem}_{clean_tag}{suffix}"



def tag_directory_path(path: str | Path, tag: str | None) -> Path:
    clean_tag = str(tag or "").strip()
    base = Path(path)
    if not clean_tag or clean_tag in base.parts:
        return base
    return base / clean_tag


def tag_file_path(path: str | Path, tag: str | None) -> Path:
    clean_tag = str(tag or "").strip()
    base = Path(path)
    if not clean_tag:
        return base
    return base.with_name(tag_name(base.name, clean_tag))


def merge_batch_size_metadata(target: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(target)
    if isinstance(source, dict):
        for key in ("batch_size", BATCH_SIZE_TAG_KEY, BATCH_SIZE_SOURCE_KEY):
            if key in source and source.get(key) not in (None, ""):
                merged[key] = source[key]
    if merged.get(BATCH_SIZE_TAG_KEY) and not merged.get(BATCH_SIZE_SOURCE_KEY):
        merged[BATCH_SIZE_SOURCE_KEY] = BATCH_SIZE_SOURCE_CONFIG
    return merged


def apply_batch_size_override(config: dict[str, Any], requested_batch_size: Any | None) -> dict[str, Any]:
    updated = dict(config)
    if requested_batch_size is None:
        if updated.get(BATCH_SIZE_TAG_KEY) and not updated.get(BATCH_SIZE_SOURCE_KEY):
            updated[BATCH_SIZE_SOURCE_KEY] = BATCH_SIZE_SOURCE_CONFIG
        return updated

    batch_size = normalize_batch_size(requested_batch_size)
    updated["batch_size"] = batch_size
    updated[BATCH_SIZE_TAG_KEY] = make_batch_size_tag(batch_size)
    updated[BATCH_SIZE_SOURCE_KEY] = BATCH_SIZE_SOURCE_CLI_OVERRIDE
    return updated


def _is_relative_to(path: Path, parent: Path | None) -> bool:
    if parent is None:
        return False
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _rewrite_under_root(path: Path, old_root: Path | None, new_root: Path | None) -> tuple[Path, bool]:
    if old_root is None or new_root is None or not _is_relative_to(path, old_root):
        return path, False
    return new_root / path.relative_to(old_root), True


def apply_batch_size_path_tagging(config: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(config)
    batch_size_tag = get_batch_size_tag(tagged)
    if batch_size_tag is None:
        return tagged

    old_output_root = Path(str(tagged["output_root"])) if tagged.get("output_root") else None
    new_output_root = tag_directory_path(old_output_root, batch_size_tag) if old_output_root is not None else None

    if new_output_root is not None:
        tagged["output_root"] = str(new_output_root)

    for key in _DIRECTORY_KEYS - {"output_root"}:
        value = tagged.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            continue
        path = Path(value)
        rewritten, changed = _rewrite_under_root(path, old_output_root, new_output_root)
        if not changed:
            rewritten = tag_directory_path(path, batch_size_tag)
        tagged[key] = str(rewritten)

    for key in _FILE_KEYS:
        value = tagged.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            continue
        path = Path(value)
        rewritten, changed = _rewrite_under_root(path, old_output_root, new_output_root)

        if not changed:
            rewritten = tag_directory_path(path.parent, batch_size_tag) / tag_file_path(path.name, batch_size_tag)
        else:
            rewritten = tag_file_path(rewritten, batch_size_tag)
        tagged[key] = str(rewritten)

    for key in _NAME_ONLY_KEYS:
        value = tagged.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        tagged[key] = tag_name(value, batch_size_tag)

    if tagged.get(BATCH_SIZE_TAG_KEY) and not tagged.get(BATCH_SIZE_SOURCE_KEY):
        tagged[BATCH_SIZE_SOURCE_KEY] = BATCH_SIZE_SOURCE_CONFIG
    return tagged
