from __future__ import annotations

import re
from pathlib import Path
from typing import Any


OUTPUT_SUFFIX_KEY = "output_suffix"

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
_SUFFIX_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_GLOBAL_COMPONENT_NAMES = {
    "selected_lambda",
    "lambda_rank",
    "workflow_configs",
}


def normalize_output_suffix(value: Any | None) -> str | None:
    text = str(value or "").strip()
    if text == "":
        return None
    if not _SUFFIX_PATTERN.fullmatch(text):
        raise ValueError(
            "Invalid output suffix "
            f"{value!r}. Expected letters, numbers, underscores, or hyphens."
        )
    return text


def get_output_suffix(config: dict[str, Any] | None) -> str | None:
    if not isinstance(config, dict):
        return None
    return normalize_output_suffix(config.get(OUTPUT_SUFFIX_KEY))


def apply_output_suffix_override(config: dict[str, Any], requested_suffix: Any | None) -> dict[str, Any]:
    updated = dict(config)
    clean_suffix = normalize_output_suffix(requested_suffix)
    if clean_suffix is not None:
        updated[OUTPUT_SUFFIX_KEY] = clean_suffix
    return updated


def append_suffix_to_name(name: str, suffix: Any | None) -> str:
    clean_suffix = normalize_output_suffix(suffix)
    if clean_suffix is None:
        return name
    token = f"_{clean_suffix}"
    if name.endswith(token):
        return name
    return f"{name}{token}"


def extract_component_suffix(name: str, base_name: str) -> str | None:
    if name == base_name:
        return None
    prefix = f"{base_name}_"
    if not name.startswith(prefix):
        return None
    remainder = name[len(prefix) :]
    return normalize_output_suffix(remainder)


def _component_names_for_config(config: dict[str, Any]) -> set[str]:
    names = set(_GLOBAL_COMPONENT_NAMES)
    route = str(config.get("route", "") or "").strip().lower()
    if route:
        names.add(f"output_{route}")
        names.add(f"output_{route}_selected")
    return names


def tag_path_output_suffix(path: str | Path, suffix: Any | None, component_names: set[str]) -> Path:
    clean_suffix = normalize_output_suffix(suffix)
    base = Path(path)
    if clean_suffix is None:
        return base

    rewritten_parts: list[str] = []
    for part in base.parts:
        rewritten = part
        for base_name in component_names:
            existing_suffix = extract_component_suffix(part, base_name)
            if part == base_name or existing_suffix is not None:
                rewritten = append_suffix_to_name(base_name, clean_suffix)
                break
        rewritten_parts.append(rewritten)
    return Path(*rewritten_parts)


def apply_output_suffix_path_tagging(config: dict[str, Any]) -> dict[str, Any]:
    tagged = dict(config)
    output_suffix = get_output_suffix(tagged)
    if output_suffix is None:
        return tagged

    component_names = _component_names_for_config(tagged)
    for key in _DIRECTORY_KEYS | _FILE_KEYS:
        value = tagged.get(key)
        if not isinstance(value, (str, Path)) or not str(value).strip():
            continue
        tagged[key] = str(tag_path_output_suffix(value, output_suffix, component_names))
    return tagged


__all__ = [
    "OUTPUT_SUFFIX_KEY",
    "append_suffix_to_name",
    "apply_output_suffix_override",
    "apply_output_suffix_path_tagging",
    "extract_component_suffix",
    "get_output_suffix",
    "normalize_output_suffix",
    "tag_path_output_suffix",
]
