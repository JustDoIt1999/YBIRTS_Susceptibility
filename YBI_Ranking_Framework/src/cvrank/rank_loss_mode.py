from __future__ import annotations

from pathlib import Path
from typing import Any

from .batch_size import tag_directory_path
from .constants import (
    RANK_LOSS_MODE_DISABLED,
    RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
)
from .output_paths import (
    OUTPUT_DIRECTORY_KEYS as _DIRECTORY_KEYS,
    OUTPUT_FILE_KEYS as _FILE_KEYS,
    rewrite_under_root,
)


RANK_LOSS_MODE_KEY = "rank_loss_mode"
DEFAULT_RANK_LOSS_MODE = RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK
ALLOWED_RANK_LOSS_MODES = {
    RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
}
SUPPORTED_RANK_LOSS_MODES = ALLOWED_RANK_LOSS_MODES | {RANK_LOSS_MODE_DISABLED}

_ALIASES = {
    "lambdarank": RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
    "lambda": RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
    "ndcg_aware_lambdarank": RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK,
    "disabled": RANK_LOSS_MODE_DISABLED,
    "none": RANK_LOSS_MODE_DISABLED,
    "off": RANK_LOSS_MODE_DISABLED,
}

def normalize_rank_loss_mode(value: Any) -> str:
    text = str(value or "").strip().lower()
    normalized = _ALIASES.get(text)
    if normalized is None:
        raise ValueError(
            "Invalid rank loss mode "
            f"{value!r}. Expected one of: {sorted(SUPPORTED_RANK_LOSS_MODES)} "
            "(aliases: 'lambdarank', 'lambda', 'ndcg_aware_lambdarank', 'none', 'off')."
        )
    return normalized


def get_rank_loss_mode(config: dict[str, Any] | None) -> str:
    if not isinstance(config, dict):
        return DEFAULT_RANK_LOSS_MODE
    value = config.get(RANK_LOSS_MODE_KEY)
    if value in (None, ""):
        return DEFAULT_RANK_LOSS_MODE
    return normalize_rank_loss_mode(value)


def get_explicit_rank_loss_mode(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    value = config.get(RANK_LOSS_MODE_KEY)
    if value in (None, ""):
        return None
    return normalize_rank_loss_mode(value)


def get_rank_loss_mode_path_tag(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    return get_rank_loss_mode(config)


def tag_rank_loss_mode_directory(path: str | Path, config: dict[str, Any] | None) -> Path:
    return tag_directory_path(path, get_rank_loss_mode_path_tag(config))


def tag_rank_loss_mode_file_parent(path: str | Path, config: dict[str, Any] | None) -> Path:
    base = Path(path)
    tag = get_rank_loss_mode_path_tag(config)
    if tag is None or tag in base.parts:
        return base
    return tag_directory_path(base.parent, tag) / base.name


def apply_rank_loss_mode_override(config: dict[str, Any], requested_mode: Any | None) -> dict[str, Any]:
    updated = dict(config)
    if requested_mode is not None:
        updated[RANK_LOSS_MODE_KEY] = normalize_rank_loss_mode(requested_mode)
        return updated
    updated[RANK_LOSS_MODE_KEY] = get_rank_loss_mode(updated)
    return updated


def apply_rank_loss_mode_path_tagging(config: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(config)
    rank_loss_mode_tag = get_rank_loss_mode_path_tag(tagged)
    if rank_loss_mode_tag is None:
        return tagged

    old_output_root = Path(str(tagged["output_root"])) if tagged.get("output_root") else None
    new_output_root = (
        tag_directory_path(old_output_root, rank_loss_mode_tag) if old_output_root is not None else None
    )

    if new_output_root is not None:
        tagged["output_root"] = str(new_output_root)

    for key in _DIRECTORY_KEYS - {"output_root"}:
        value = tagged.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            continue
        path = Path(value)
        rewritten, changed = rewrite_under_root(path, old_output_root, new_output_root)
        if changed:
            tagged[key] = str(rewritten)

    for key in _FILE_KEYS:
        value = tagged.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            continue
        path = Path(value)
        rewritten, changed = rewrite_under_root(path, old_output_root, new_output_root)
        if changed:
            tagged[key] = str(rewritten)

    return tagged


def merge_rank_loss_mode_metadata(target: dict[str, Any], source: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(target)
    if not isinstance(source, dict):
        return merged
    value = source.get(RANK_LOSS_MODE_KEY)
    if value not in (None, ""):
        merged[RANK_LOSS_MODE_KEY] = normalize_rank_loss_mode(value)
    return merged


def resolve_effective_rank_loss_mode(
    config: dict[str, Any] | None,
    *,
    use_rank_aux: bool,
    rank_col: str | None,
) -> str:
    if not use_rank_aux or rank_col is None:
        return RANK_LOSS_MODE_DISABLED
    mode = get_rank_loss_mode(config)
    if mode == RANK_LOSS_MODE_DISABLED:
        return RANK_LOSS_MODE_DISABLED
    return mode
