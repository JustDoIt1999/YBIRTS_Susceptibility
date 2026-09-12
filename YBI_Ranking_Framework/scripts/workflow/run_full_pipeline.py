#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    print("Error: PyYAML package is not installed. Please activate your environment first.", file=sys.stderr)
    sys.exit(1)


CVRANK_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = CVRANK_ROOT / "src"
DEFAULT_WORKFLOW_CONFIG = CVRANK_ROOT / "scripts" / "workflow" / "full_pipeline.yaml"
DEFAULT_RANK_LOSS_MODE = "ndcg_aware_lambdarank"
SUPPORTED_WORKFLOW_KEYS = {"skip_existing", "batch_size", "rank_loss_mode"}
ROUTES = ("dbscan", "no_cluster", "basin")


def workflow_config_path_from_argv(default_path: Path, argv: list[str]) -> Path:
    config_path = default_path
    index = 0
    while index < len(argv):
        if argv[index] == "--config":
            if index + 1 >= len(argv):
                raise ValueError("Missing value for --config")
            config_path = Path(argv[index + 1]).resolve()
            index += 2
            continue
        index += 1
    return config_path


def trim_workflow_value(value: Any) -> str:
    text = str("" if value is None else value).strip()
    if len(text) >= 2 and ((text[0] == text[-1] == '"') or (text[0] == text[-1] == "'")):
        return text[1:-1]
    return text


def workflow_lower(value: Any) -> str:
    return trim_workflow_value(value).lower()


def normalize_workflow_bool(key_name: str, raw_value: Any, default_value: bool = False) -> bool:
    lower_value = workflow_lower(raw_value)
    if lower_value == "":
        return bool(default_value)
    if lower_value in {"true", "yes", "y", "on", "1"}:
        return True
    if lower_value in {"false", "no", "n", "off", "0", "null", "~"}:
        return False
    raise ValueError(f"Invalid boolean for `{key_name}`: {raw_value}")


def normalize_workflow_optional_int(key_name: str, raw_value: Any) -> str:
    value = trim_workflow_value(raw_value)
    lower_value = value.lower()
    if lower_value in {"", "null", "~"}:
        return ""
    if not value.isdigit():
        raise ValueError(f"Invalid integer for `{key_name}`: {raw_value}")
    if int(value) <= 0:
        raise ValueError(f"`{key_name}` must be > 0, got {value}.")
    return value


def normalize_workflow_rank_loss_mode(raw_value: Any) -> str:
    lower_value = workflow_lower(raw_value)
    if lower_value in {"", "null", "~"}:
        return ""
    if lower_value in {"lambdarank", "lambda", "ndcg_aware_lambdarank"}:
        return DEFAULT_RANK_LOSS_MODE
    if lower_value in {"disabled", "none", "off"}:
        return "disabled"
    raise ValueError(f"Invalid `rank_loss_mode`: {raw_value}")


def parse_workflow_yaml(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Workflow config must be a mapping: {config_path}")
    invalid_keys = sorted(set(payload) - SUPPORTED_WORKFLOW_KEYS)
    if invalid_keys:
        joined = " ".join(invalid_keys)
        raise ValueError(f"Unsupported keys for workflow 'full' in {config_path}: {joined}")
    return payload


def load_workflow_config(config_path: Path) -> dict[str, Any]:
    raw = parse_workflow_yaml(config_path)
    return {
        "skip_existing": normalize_workflow_bool("skip_existing", raw.get("skip_existing"), False),
        "batch_size": normalize_workflow_optional_int("batch_size", raw.get("batch_size")),
        "rank_loss_mode": normalize_workflow_rank_loss_mode(raw.get("rank_loss_mode")),
    }


def resolve_cvrank_python() -> str:
    python_bin = os.environ.get("PYTHON_BIN", "").strip()
    if python_bin:
        return python_bin
    if sys.executable and Path(sys.executable).exists():
        return sys.executable
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        for candidate in (Path(conda_prefix) / "bin" / "python", Path(conda_prefix) / "python.exe"):
            if candidate.exists():
                return str(candidate)
    resolved = shutil.which("python")
    if resolved:
        return resolved
    raise FileNotFoundError("Unable to resolve a Python interpreter for cvrank.")


def append_output_suffix_name(name: str, suffix: str) -> str:
    if not suffix:
        return name
    return f"{name}_{suffix}"


def selection_output_root(batch_size: str, rank_loss_mode: str, output_suffix: str) -> Path:
    root = CVRANK_ROOT / "data" / "output" / append_output_suffix_name("selected_lambda", output_suffix)
    if batch_size:
        root = root / f"bs_{batch_size}"
    return root / (rank_loss_mode or DEFAULT_RANK_LOSS_MODE)


def sweep_reuse_flag(skip_existing: bool) -> str:
    return "--reuse-existing" if skip_existing else "--no-reuse-existing"


def print_effective_options(workflow_config: Path, skip_existing: bool, batch_size: str, rank_loss_mode: str, output_suffix: str) -> None:
    effective_batch_size = batch_size or "<route-config-default>"
    effective_rank_loss_mode = rank_loss_mode or "<route-config-default>"
    rank_loss_mode_display = (
        f"{effective_rank_loss_mode} [classification baseline; ranking loss disabled]"
        if rank_loss_mode == "disabled"
        else effective_rank_loss_mode
    )
    effective_output_suffix = output_suffix or "<none>"
    print(f"Workflow config: {workflow_config}")
    print("Effective workflow options:")
    print(f"  skip_existing={str(skip_existing).lower()}")
    print(f"  batch_size={effective_batch_size}")
    print(f"  rank_loss_mode={rank_loss_mode_display}")
    print(f"  output_suffix={effective_output_suffix}")


def run_step(title: str, command: list[str]) -> None:
    print()
    print(f"==> {title}")
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{SRC_DIR}{os.pathsep}{pythonpath}" if pythonpath else str(SRC_DIR)
    try:
        subprocess.run(command, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Error occurred during step: {title}. Exit code: {exc.returncode}", file=sys.stderr)
        raise SystemExit(exc.returncode) from None


def all_paths_exist(paths: list[str]) -> bool:
    return all(Path(path).exists() for path in paths)


def all_globs_match(patterns: list[str]) -> bool:
    return all(glob.glob(pattern) for pattern in patterns)


def selected_training_artifacts_complete(selection_manifest_path: Path, expected_routes: tuple[str, ...]) -> bool:
    if not selection_manifest_path.exists():
        return False
    try:
        manifest = json.loads(selection_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    # The full workflow now materializes the selected sweep artifacts by
    # default.  Do not treat a legacy or explicitly retrained manifest as a
    # valid reusable result.
    if manifest.get("post_selection_action") != "use_sweep_artifacts":
        return False
    selected_routes = manifest.get("selected_routes")
    if not isinstance(selected_routes, list):
        return False
    route_records = {
        str(record.get("route", "")).strip().lower(): record
        for record in selected_routes
        if isinstance(record, dict)
    }
    for route in expected_routes:
        record = route_records.get(route)
        if record is None:
            return False
        config_path = Path(str(record.get("selected_config_path") or ""))
        if not config_path.exists():
            return False
        try:
            train_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            return False
        if not isinstance(train_config, dict):
            return False
        required_keys = (
            "metrics_path",
            "validation_metrics_path",
            "history_path",
            "fold_runs_manifest_path",
        )
        required_paths = [Path(str(train_config.get(key) or "")) for key in required_keys]
        if not all(path.exists() and path.is_file() for path in required_paths):
            return False
        try:
            fold_manifest = json.loads(required_paths[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(fold_manifest, dict) or fold_manifest.get("artifact_source") != "lambda_sweep":
            return False
        fold_records = fold_manifest.get("folds")
        if not isinstance(fold_records, list) or not fold_records or not all(isinstance(record, dict) for record in fold_records):
            return False
        if any(record.get("artifact_source") != "lambda_sweep" for record in fold_records):
            return False
    return True


def run_step_if_needed(
    title: str,
    *,
    skip_existing: bool,
    success_paths: list[str] | None = None,
    success_globs: list[str] | None = None,
    command: list[str],
) -> None:
    success_paths = success_paths or []
    success_globs = success_globs or []
    marker = ""

    if skip_existing:
        paths_ok = True
        globs_ok = True
        if success_paths:
            paths_ok = all_paths_exist(success_paths)
            if paths_ok:
                marker = success_paths[0]
        if success_globs:
            globs_ok = all_globs_match(success_globs)
            if globs_ok and not marker:
                matches = glob.glob(success_globs[0])
                if matches:
                    marker = matches[0]
        if (success_paths or success_globs) and paths_ok and globs_ok:
            print()
            print(f"==> skip ({title})")
            if marker:
                print(f"    already exists: {marker}")
            return

    run_step(title, command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the full YBI_Ranking_Framework pipeline (make-folds, sweep-lambda, select-lambda)."
    )
    parser.add_argument("--config", type=str, default=None, help="Workflow-level YAML defaults.")
    parser.add_argument(
        "--route",
        type=str,
        choices=["dbscan", "no_cluster", "basin", "all"],
        default="all",
        help="Specify a single route to run, or 'all' to run all configured routes.",
    )
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true", help="Skip steps whose key output artifacts already exist.")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Force re-running all steps for this run.")
    parser.add_argument("--batch-size", type=str, default=None, help="Override batch size.")
    parser.add_argument("--clear-batch-size", action="store_true", help="Ignore any workflow-level batch size override and fall back to the route config default.")
    parser.add_argument("--rank-loss-mode", type=str, default=None, help="Override rank loss mode.")
    parser.add_argument("--clear-rank-loss-mode", action="store_true", help="Ignore any workflow-level rank loss mode override and use the route config default.")
    parser.add_argument("--output-suffix", type=str, default=None, help="Append a suffix to output roots.")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        workflow_config = workflow_config_path_from_argv(DEFAULT_WORKFLOW_CONFIG, argv)
        defaults = load_workflow_config(workflow_config)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parser = build_parser()
    parser.set_defaults(
        skip_existing=defaults["skip_existing"],
        batch_size=defaults["batch_size"] or None,
        rank_loss_mode=defaults["rank_loss_mode"] or None,
    )
    args = parser.parse_args(argv)
    routes_to_run = (args.route,) if args.route != "all" else ROUTES

    batch_size = "" if args.clear_batch_size else str(args.batch_size or "")
    rank_loss_mode = "" if args.clear_rank_loss_mode else str(args.rank_loss_mode or "")
    output_suffix = str(args.output_suffix or "")
    if rank_loss_mode:
        try:
            rank_loss_mode = normalize_workflow_rank_loss_mode(rank_loss_mode)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    print_effective_options(
        workflow_config,
        bool(args.skip_existing),
        batch_size,
        rank_loss_mode,
        output_suffix,
    )

    python_exec = resolve_cvrank_python()

    extra_args: list[str] = []
    sweep_extra_args: list[str] = []
    select_extra_args: list[str] = []
    if batch_size:
        extra_args.extend(["--batch-size", batch_size])
        sweep_extra_args.extend(["--batch-size", batch_size])
        select_extra_args.extend(["--batch-size", batch_size])
    if output_suffix:
        extra_args.extend(["--output-suffix", output_suffix])
        sweep_extra_args.extend(["--output-suffix", output_suffix])
        select_extra_args.extend(["--output-suffix", output_suffix])
    if rank_loss_mode:
        sweep_extra_args.extend(["--rank-loss-mode", rank_loss_mode])
        select_extra_args.extend(["--rank-loss-mode", rank_loss_mode])

    for route in routes_to_run:
        manifest_path = CVRANK_ROOT / "data" / "output" / "folds" / f"{route}_5fold_manifest.csv"
        route_config = CVRANK_ROOT / "configs" / "routes" / f"{route}.yaml"
        command = [python_exec, "-m", "cvrank", "make-folds", "--config", str(route_config)]
        run_step_if_needed(
            f"make-folds ({route})",
            skip_existing=bool(args.skip_existing),
            success_paths=[str(manifest_path)],
            command=command,
        )

    sweep_config_path = CVRANK_ROOT / "configs" / "lambda" / "lambda_sweep.yaml"
    select_config_path = CVRANK_ROOT / "configs" / "lambda" / "lambda_select.yaml"
    temp_sweep_yaml = None
    temp_select_yaml = None

    try:
        if args.route != "all":
            with sweep_config_path.open("r", encoding="utf-8") as fh:
                sweep_data = yaml.safe_load(fh) or {}
            sweep_data["route_config_paths"] = [
                p for p in sweep_data.get("route_config_paths", [])
                if f"/{args.route}.yaml" in p or f"\\{args.route}.yaml" in p
            ]
            temp_sweep_yaml = sweep_config_path.with_name(f"lambda_sweep_{args.route}_temp.yaml")
            with temp_sweep_yaml.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(sweep_data, fh)
            
            with select_config_path.open("r", encoding="utf-8") as fh:
                select_data = yaml.safe_load(fh) or {}
            select_data["route_config_paths"] = [
                p for p in select_data.get("route_config_paths", [])
                if f"/{args.route}.yaml" in p or f"\\{args.route}.yaml" in p
            ]
            temp_select_yaml = select_config_path.with_name(f"lambda_select_{args.route}_temp.yaml")
            with temp_select_yaml.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(select_data, fh)

            sweep_config_to_run = temp_sweep_yaml
            select_config_to_run = temp_select_yaml
        else:
            sweep_config_to_run = sweep_config_path
            select_config_to_run = select_config_path

        sweep_command = [
            python_exec,
            "-m",
            "cvrank",
            "sweep-lambda",
            "--config",
            str(sweep_config_to_run),
            sweep_reuse_flag(bool(args.skip_existing)),
            *sweep_extra_args,
        ]
        run_step("sweep-lambda", sweep_command)

        selection_root = selection_output_root(batch_size, rank_loss_mode, output_suffix)
        select_command = [
            python_exec,
            "-m",
            "cvrank",
            "select-lambda",
            "--config",
            str(select_config_to_run),
            *select_extra_args,
        ]

        selection_manifest_path = selection_root / "selection_manifest.json"
        if bool(args.skip_existing) and selected_training_artifacts_complete(selection_manifest_path, routes_to_run):
            print()
            print("==> skip (select-lambda)")
            print(f"    complete manifest: {selection_manifest_path}")
        else:
            run_step("select-lambda", select_command)
    finally:
        if temp_sweep_yaml is not None and temp_sweep_yaml.exists():
            temp_sweep_yaml.unlink()
        if temp_select_yaml is not None and temp_select_yaml.exists():
            temp_select_yaml.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
