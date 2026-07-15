from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


RUN_FINGERPRINT_SCHEMA_VERSION = 1
RUN_COMPLETION_SCHEMA_VERSION = 1
RUN_COMPLETION_FILENAME = "run_completion.json"

_CONTROL_CONFIG_KEYS = {
    "reuse_existing",
    "skip_existing",
}
_INPUT_PATH_KEYS = (
    "data_path",
    "split_manifest_path",
    "fixed_split_path",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_ready(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_ready(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value.resolve())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


class FileHashCache:
    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, int], str] = {}

    def sha256(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        stat = resolved.stat()
        cache_key = (str(resolved), int(stat.st_size), int(stat.st_mtime_ns))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        digest = hashlib.sha256()
        with resolved.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
        self._cache[cache_key] = value
        return value


def describe_file(path: str | Path, hash_cache: FileHashCache) -> dict[str, Any]:
    resolved = Path(path).resolve()
    description: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.is_file(),
    }
    if not description["exists"]:
        return description
    stat = resolved.stat()
    description.update(
        {
            "size": int(stat.st_size),
            "sha256": hash_cache.sha256(resolved),
        }
    )
    return description


def semantic_run_config(run_config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in run_config.items()
        if key not in _CONTROL_CONFIG_KEYS
    }


def describe_run_inputs(
    run_config: dict[str, Any],
    hash_cache: FileHashCache,
) -> dict[str, dict[str, Any]]:
    inputs: dict[str, dict[str, Any]] = {}
    for key in _INPUT_PATH_KEYS:
        raw_path = run_config.get(key)
        if raw_path in (None, ""):
            continue
        inputs[key] = describe_file(raw_path, hash_cache)
    data_description = inputs.get("data_path")
    if data_description is None or not data_description.get("exists"):
        raise FileNotFoundError(
            f"Training data required for sweep fingerprinting was not found: {run_config.get('data_path')}"
        )
    return inputs


def build_training_code_identity(
    hash_cache: FileHashCache,
    package_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(package_root or Path(__file__).resolve().parent).resolve()
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(root)),
                "sha256": hash_cache.sha256(path),
            }
        )
    return {
        "package_root": str(root),
        "files": files,
        "sha256": sha256_value(files),
    }


def build_run_fingerprint(
    run_config: dict[str, Any],
    *,
    hash_cache: FileHashCache,
    code_identity: dict[str, Any],
) -> dict[str, Any]:
    canonical_config = semantic_run_config(run_config)
    config_sha256 = sha256_value(canonical_config)
    inputs = describe_run_inputs(run_config, hash_cache)
    inputs_sha256 = sha256_value(inputs)
    code_sha256 = str(code_identity["sha256"])
    components = {
        "schema_version": RUN_FINGERPRINT_SCHEMA_VERSION,
        "config_sha256": config_sha256,
        "inputs_sha256": inputs_sha256,
        "code_sha256": code_sha256,
    }
    return {
        **components,
        "run_fingerprint": sha256_value(components),
        "inputs": inputs,
        "code": code_identity,
    }


def compare_run_fingerprints(
    saved: dict[str, Any] | None,
    current: dict[str, Any],
) -> tuple[bool, str]:
    if not isinstance(saved, dict):
        return False, "legacy_unverified"
    if saved.get("schema_version") != RUN_FINGERPRINT_SCHEMA_VERSION:
        return False, "fingerprint_schema_changed"
    if saved.get("config_sha256") != current.get("config_sha256"):
        return False, "config_changed"
    if saved.get("inputs_sha256") != current.get("inputs_sha256"):
        return False, "input_changed"
    if saved.get("code_sha256") != current.get("code_sha256"):
        return False, "code_changed"
    if saved.get("run_fingerprint") != current.get("run_fingerprint"):
        return False, "fingerprint_changed"
    return True, "compatible"


__all__ = [
    "FileHashCache",
    "RUN_COMPLETION_FILENAME",
    "RUN_COMPLETION_SCHEMA_VERSION",
    "RUN_FINGERPRINT_SCHEMA_VERSION",
    "build_run_fingerprint",
    "build_training_code_identity",
    "canonical_json_bytes",
    "compare_run_fingerprints",
    "describe_file",
    "semantic_run_config",
    "sha256_value",
]
