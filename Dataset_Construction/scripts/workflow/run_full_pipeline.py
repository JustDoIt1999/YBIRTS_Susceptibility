#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

WORKFLOW_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = WORKFLOW_DIR.parent
DATA_PROCESSING_DIR = SCRIPTS_DIR.parent
REPO_ROOT = DATA_PROCESSING_DIR.parent
DEFAULT_CONFIG_PATH = WORKFLOW_DIR / "full_pipeline.yaml"


@dataclass(frozen=True)
class WorkflowStep:
    number: int
    name: str
    notebook: Path
    module: str
    outputs: tuple[Path, ...]
    requires_gee: bool = False


STEPS: tuple[WorkflowStep, ...] = (
    WorkflowStep(
        number=1,
        name="pa_sampling",
        notebook=DATA_PROCESSING_DIR / "Jupyter_notebook" / "01_pa_sampling" / "pa_sampling.ipynb",
        module="dataset_construction.pa_sampling",
        outputs=(
            DATA_PROCESSING_DIR / "outputs" / "01_pa_sampling" / "positive_su.csv",
            DATA_PROCESSING_DIR / "outputs" / "01_pa_sampling" / "pseudo_absence_random.csv",
        ),
    ),
    WorkflowStep(
        number=2,
        name="sample_tables",
        notebook=DATA_PROCESSING_DIR / "Jupyter_notebook" / "02_sample_tables" / "RTS_Total_sample.ipynb",
        module="dataset_construction.sample_tables",
        outputs=(
            DATA_PROCESSING_DIR / "outputs" / "02_sample_tables" / "RTS_Total_sample.csv",
            DATA_PROCESSING_DIR / "outputs" / "02_sample_tables" / "RTS_Total_sample_rank.csv",
        ),
    ),
    WorkflowStep(
        number=3,
        name="environmental_factors",
        notebook=DATA_PROCESSING_DIR / "Jupyter_notebook" / "03_environmental_data" / "environmental_rasters_geemap.ipynb",
        module="dataset_construction.environmental_factors",
        outputs=(DATA_PROCESSING_DIR / "outputs" / "03_environmental_data" / "feature_manifest.json",),
        requires_gee=True,
    ),
    WorkflowStep(
        number=4,
        name="build_total_features",
        notebook=DATA_PROCESSING_DIR / "Jupyter_notebook" / "03_environmental_data" / "build_total_features_sample.ipynb",
        module="dataset_construction.build_total_features",
        outputs=(DATA_PROCESSING_DIR / "outputs" / "03_environmental_data" / "Total_features_sample.csv",),
        requires_gee=True,
    ),
)

STEP_ALIASES = {
    "1": 1,
    "pa_sampling": 1,
    "pa-sampling": 1,
    "2": 2,
    "sample_tables": 2,
    "sample-tables": 2,
    "3": 3,
    "environmental_factors": 3,
    "environmental-factors": 3,
    "factor_library": 3,
    "factor-library": 3,
    "4": 4,
    "build_total_features": 4,
    "build-total-features": 4,
    "build_total_features_sample": 4,
    "build-total-features-sample": 4,
}
ALLOWED_CONFIG_KEYS = frozenset({"skip_existing", "from_step", "to_step", "direct_download"})


def parse_bool(value: str | bool | None) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "on", "1"}:
        return True
    if normalized in {"false", "no", "off", "0"}:
        return False
    return None


def parse_scalar(value: str) -> object:
    text = value.strip()
    if not text:
        return ""
    if text.startswith(("'", '"')) and text.endswith(("'", '"')) and len(text) >= 2:
        return text[1:-1]
    maybe_bool = parse_bool(text)
    if maybe_bool is not None:
        return maybe_bool
    if text.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(text)
    except ValueError:
        return text


def load_simple_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    config: dict[str, object] = {}
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {lineno}: {raw_line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        if key not in ALLOWED_CONFIG_KEYS:
            raise ValueError(f"Unsupported workflow config key: {key}")
        config[key] = parse_scalar(raw_value)
    return config


def resolve_step_number(value: object | None, *, flag_name: str) -> int | None:
    if value is None or value == "":
        return None
    normalized = str(value).strip().lower()
    if normalized not in STEP_ALIASES:
        valid = ", ".join(step.name for step in STEPS)
        raise ValueError(f"Unsupported {flag_name}: {value}. Valid values: 1-4, {valid}")
    return STEP_ALIASES[normalized]


def resolve_python_bin() -> str:
    env_value = os.environ.get("PYTHON_BIN", "").strip()
    if env_value:
        return env_value
    executable = Path(sys.executable)
    if executable.exists():
        return str(executable)
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        candidates = [
            Path(conda_prefix) / "bin" / "python",
            Path(conda_prefix) / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
    return shutil.which("python3") or shutil.which("python") or "python"


def require_gee_auth(python_bin: str) -> None:
    script = """
import os
import sys

project = os.environ.get("EE_PROJECT_ID", "").strip()
if not project:
    print("ERROR: EE_PROJECT_ID is not set.", file=sys.stderr)
    print(
        "This workflow reads public Earth Engine assets, but execution must use your own "
        "Earth Engine-enabled Google Cloud project.",
        file=sys.stderr,
    )
    print("Example: export EE_PROJECT_ID='your-project-id'", file=sys.stderr)
    raise SystemExit(2)
try:
    import ee
except Exception as exc:  # pragma: no cover
    print("ERROR: Earth Engine Python API is unavailable in the workflow environment.", file=sys.stderr)
    print(f"Details: {exc}", file=sys.stderr)
    print("Run this workflow from the geemap/Earth Engine environment, then retry.", file=sys.stderr)
    raise SystemExit(3)

try:
    ee.Initialize(project=project)
except Exception as exc:
    print("ERROR: Earth Engine authentication/initialization failed before workflow execution.", file=sys.stderr)
    print(f"Project: {project}", file=sys.stderr)
    print(f"Details: {exc}", file=sys.stderr)
    print("Authenticate manually in the target environment first, then rerun the workflow.", file=sys.stderr)
    print("Example:", file=sys.stderr)
    print(f"  {sys.executable} -c \\\"import ee; ee.Authenticate(); ee.Initialize(project='{project}')\\\"", file=sys.stderr)
    raise SystemExit(4)
"""
    completed = run_python_helper(python_bin, script)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def run_python_helper(python_bin: str, script: str) -> subprocess.CompletedProcess[str]:
    command = [python_bin, "-c", textwrap.dedent(script)]
    return subprocess.run(command, cwd=REPO_ROOT, text=True)


def wait_for_runtime_factor_library_assets(python_bin: str, *, expect_export_tasks: bool) -> None:
    sample_path = DATA_PROCESSING_DIR / "outputs" / "02_sample_tables" / "RTS_Total_sample_rank.csv"
    timeout_seconds = int(os.environ.get("STEP01_FACTOR_LIBRARY_WAIT_TIMEOUT_SECONDS", "14400"))
    poll_seconds = int(os.environ.get("STEP01_FACTOR_LIBRARY_POLL_SECONDS", "30"))
    script = """
import csv
import os
import sys
import time

SAMPLE_PATH = __SAMPLE_PATH__
EXPECT_EXPORT_TASKS = __EXPECT_EXPORT_TASKS__
TIMEOUT_SECONDS = __TIMEOUT_SECONDS__
POLL_SECONDS = __POLL_SECONDS__
NOT_FOUND_MARKERS = ("not found", "does not exist", "asset not found", "asset does not exist")
ACTIVE_STATES = {{"READY", "RUNNING", "CANCEL_REQUESTED"}}
FAILED_STATES = {{"FAILED", "CANCELLED"}}
STATE_MAP = {{
    "PENDING": "READY",
    "RUNNING": "RUNNING",
    "SUCCEEDED": "COMPLETED",
    "FAILED": "FAILED",
    "CANCELLING": "CANCEL_REQUESTED",
    "CANCELLED": "CANCELLED",
}}
STATIC_FACTORS = [
    ("Slope", None),
    ("Aspect", None),
    ("TWI", None),
    ("Local_Relief", "Local_Relief_5km"),
    ("Soil", None),
    ("GIC", None),
    ("Vegetation_type", None),
]
DYNAMIC_FACTORS = ["TDD", "SMDP", "Precipitation", "WSTmax", "WSRH", "NDVI", "Summer_SoilMoisture"]

project = os.environ.get("EE_PROJECT_ID", "").strip()
if not project:
    print("ERROR: EE_PROJECT_ID is not set.", file=sys.stderr)
    raise SystemExit(2)

if not os.path.exists(SAMPLE_PATH):
    print(f"ERROR: missing sample-year source table: {{SAMPLE_PATH}}", file=sys.stderr)
    raise SystemExit(3)

try:
    import ee
except Exception as exc:
    print("ERROR: Earth Engine Python API is unavailable in the workflow environment.", file=sys.stderr)
    print(f"Details: {{exc}}", file=sys.stderr)
    raise SystemExit(4)

ee.Initialize(project=project)

runtime_asset_root = f"projects/{{project}}/assets"
factor_library_root = f"{{runtime_asset_root}}/environmental_factors/factor_library_v2"
static_root = f"{{factor_library_root}}/static"
dynamic_root = f"{{factor_library_root}}/dynamic_annual"

with open(SAMPLE_PATH, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    years = sorted({int(row["Feature_Year"]) for row in reader if row.get("Feature_Year") not in (None, "")})
if not years:
    print("ERROR: no valid Feature_Year values were found in RTS_Total_sample_rank.csv", file=sys.stderr)
    raise SystemExit(5)

def asset_name(group: str, factor_name: str, year: int | None = None) -> str:
    if group == "static":
        lookup = next((item for item in STATIC_FACTORS if item[0] == factor_name), None)
        if lookup is None:
            raise ValueError(f"Unsupported static factor: {{factor_name}}")
        base_name = lookup[1] or factor_name
        return f"{{base_name}}_v2"
    if year is None:
        raise ValueError("dynamic_annual asset name requires year")
    if factor_name == "SMDP":
        return f"{{factor_name}}_{{int(year)}}_v2"
    return f"{{factor_name}}_{{int(year)}}_v2"

def asset_id(group: str, factor_name: str, year: int | None = None) -> str:
    root = static_root if group == "static" else dynamic_root
    return f"{{root}}/{{asset_name(group, factor_name, year=year)}}"

def task_description(group: str, factor_name: str, year: int | None = None) -> str:
    if group == "static":
        return f"factorlib_static_{{asset_name('static', factor_name)}}"
    if year is None:
        raise ValueError("dynamic_annual task description requires year")
    suffix = "v2"
    return f"factorlib_dynamic_annual_{{factor_name}}_{{int(year)}}_{{suffix}}"

required_assets = []
for factor_name, _override in STATIC_FACTORS:
    required_assets.append({{
        "group": "static",
        "factor_name": factor_name,
        "year": None,
        "asset_id": asset_id("static", factor_name),
        "task_description": task_description("static", factor_name),
    }})
for year in years:
    for factor_name in DYNAMIC_FACTORS:
        required_assets.append({{
            "group": "dynamic_annual",
            "factor_name": factor_name,
            "year": int(year),
            "asset_id": asset_id("dynamic_annual", factor_name, year=int(year)),
            "task_description": task_description("dynamic_annual", factor_name, year=int(year)),
        }})

def asset_exists(candidate: str) -> bool:
    try:
        ee.data.getAsset(candidate)
        return True
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in NOT_FOUND_MARKERS):
            return False
        raise RuntimeError(f"Failed to check Earth Engine asset existence: {{candidate}}") from exc

def normalize_state(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    return STATE_MAP.get(text, text)

def task_sort_key(record: dict) -> tuple[int, int, int]:
    return (
        int(record.get("update_timestamp_ms") or 0),
        int(record.get("start_timestamp_ms") or 0),
        int(record.get("creation_timestamp_ms") or 0),
    )

def task_int(value) -> int:
    try:
        return int(value)
    except Exception:
        return 0

def fetch_task_index() -> dict[str, list[dict]]:
    descriptions = {{item["task_description"] for item in required_assets}}
    normalized = []
    errors = []
    for fn_name in ("listOperations", "getTaskList"):
        fn = getattr(ee.data, fn_name, None)
        if not callable(fn):
            continue
        try:
            records = fn() or []
        except Exception as exc:
            errors.append(f"{{fn_name}}={{exc!r}}")
            continue
        for raw in records:
            if not isinstance(raw, dict):
                continue
            metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {{}}
            if "done" in raw and metadata:
                description = metadata.get("description")
                state = normalize_state(metadata.get("state"))
                normalized.append({{
                    "description": description,
                    "state": state,
                    "task_id": str(raw.get("name") or "").rsplit("/", 1)[-1],
                    "creation_timestamp_ms": task_int(metadata.get("createTime")),
                    "start_timestamp_ms": task_int(metadata.get("startTime")),
                    "update_timestamp_ms": task_int(metadata.get("updateTime")),
                }})
            else:
                description = raw.get("description") or metadata.get("description")
                state = normalize_state(raw.get("state") or metadata.get("state"))
                normalized.append({{
                    "description": description,
                    "state": state,
                    "task_id": raw.get("id") or raw.get("name"),
                    "creation_timestamp_ms": task_int(raw.get("creation_timestamp_ms") or metadata.get("creation_timestamp_ms")),
                    "start_timestamp_ms": task_int(raw.get("start_timestamp_ms") or metadata.get("start_timestamp_ms")),
                    "update_timestamp_ms": task_int(raw.get("update_timestamp_ms") or metadata.get("update_timestamp_ms")),
                }})
        break
    index = {{desc: [] for desc in descriptions}}
    for record in normalized:
        description = record.get("description")
        if description in index:
            index[description].append(record)
    for desc in index:
        index[desc].sort(key=task_sort_key, reverse=True)
    return index

start = time.time()
attempt = 0
while True:
    attempt += 1
    missing_assets = []
    for item in required_assets:
        if not asset_exists(item["asset_id"]):
            missing_assets.append(item)
    ready_count = len(required_assets) - len(missing_assets)
    if not missing_assets:
        print(
            "Runtime factor-library assets ready for build_total_features: "
            f"{{ready_count}}/{{len(required_assets)}} assets; years={{years}}; root={{factor_library_root}}"
        )
        raise SystemExit(0)

    task_index = fetch_task_index()
    failed = []
    active_count = 0
    observed_count = 0
    for item in missing_assets:
        related = task_index.get(item["task_description"], [])
        if related:
            observed_count += 1
        latest = related[0] if related else None
        state = latest.get("state") if latest else None
        if state in ACTIVE_STATES:
            active_count += 1
        if state in FAILED_STATES:
            failed.append({{
                "asset_id": item["asset_id"],
                "task_description": item["task_description"],
                "task_state": state,
                "task_id": latest.get("task_id"),
            }})

    if failed:
        print("ERROR: required factor-library export tasks failed before build_total_features could start.", file=sys.stderr)
        for row in failed[:10]:
            print(
                f"  - asset={{row['asset_id']}} task={{row['task_description']}} state={{row['task_state']}} id={{row['task_id']}}",
                file=sys.stderr,
            )
        raise SystemExit(6)

    elapsed = int(time.time() - start)
    if not EXPECT_EXPORT_TASKS and attempt == 1 and observed_count == 0:
        print(
            "ERROR: required runtime factor-library assets are missing and no matching export tasks were found. "
            "Run environmental_rasters_geemap.ipynb or workflow step 3 first.",
            file=sys.stderr,
        )
        preview = [item["asset_id"] for item in missing_assets[:10]]
        print(f"Missing asset preview: {{preview}}", file=sys.stderr)
        raise SystemExit(7)

    if elapsed >= TIMEOUT_SECONDS:
        print(
            "ERROR: timed out while waiting for runtime factor-library assets required by build_total_features.",
            file=sys.stderr,
        )
        print(
            f"Waited {{elapsed}}s; ready={{ready_count}}/{{len(required_assets)}}; "
            f"missing={{len(missing_assets)}}; active_tasks={{active_count}}",
            file=sys.stderr,
        )
        preview = [item["asset_id"] for item in missing_assets[:10]]
        print(f"Missing asset preview: {{preview}}", file=sys.stderr)
        raise SystemExit(8)

    print(
        "[wait] runtime factor-library assets: "
        f"ready={{ready_count}}/{{len(required_assets)}} missing={{len(missing_assets)}} "
        f"active_tasks={{active_count}} observed_tasks={{observed_count}} elapsed={{elapsed}}s"
    )
    time.sleep(max(POLL_SECONDS, 1))
"""
    script = (
        script.replace("__SAMPLE_PATH__", repr(str(sample_path)))
        .replace("__EXPECT_EXPORT_TASKS__", repr(bool(expect_export_tasks)))
        .replace("__TIMEOUT_SECONDS__", repr(timeout_seconds))
        .replace("__POLL_SECONDS__", repr(poll_seconds))
        .replace("{{", "{")
        .replace("}}", "}")
    )
    completed = run_python_helper(python_bin, script)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, end="", file=sys.stderr)
        raise SystemExit(completed.returncode)


def outputs_exist(step: WorkflowStep) -> bool:
    return all(path.exists() for path in step.outputs)


def run_step_script(step: WorkflowStep, python_bin: str) -> int:
    src_dir = DATA_PROCESSING_DIR / "src"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    env["PYTHONPATH"] = str(src_dir) if not existing_pythonpath else os.pathsep.join([str(src_dir), existing_pythonpath])
    command = [python_bin, "-m", step.module]
    print(f"[run] step {step.number}: {step.name}")
    print(f"      notebook: {step.notebook.relative_to(REPO_ROOT)}")
    print(f"      module: {step.module}")
    return subprocess.run(command, cwd=REPO_ROOT, env=env).returncode


def select_steps(from_step: int | None, to_step: int | None) -> list[WorkflowStep]:
    start = from_step or 1
    end = to_step or len(STEPS)
    if start > end:
        raise ValueError(f"Invalid step range: from_step={start}, to_step={end}")
    return [step for step in STEPS if start <= step.number <= end]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the formal Dataset_Construction Python workflow end-to-end.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Workflow config path. Default: scripts/workflow/full_pipeline.yaml",
    )
    parser.add_argument("--skip-existing", dest="skip_existing", action="store_true")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--from-step", help="Start from step number or step name.")
    parser.add_argument("--to-step", help="Stop after step number or step name.")
    parser.add_argument("--direct-download", dest="direct_download", action="store_true")
    parser.add_argument("--no-direct-download", dest="direct_download", action="store_false")
    parser.set_defaults(skip_existing=None, direct_download=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_simple_yaml(config_path)

    skip_existing = config.get("skip_existing", False)
    direct_download = config.get("direct_download", True)
    from_step_value = config.get("from_step")
    to_step_value = config.get("to_step")

    if args.skip_existing is not None:
        skip_existing = args.skip_existing
    if args.direct_download is not None:
        direct_download = args.direct_download
    if args.from_step is not None:
        from_step_value = args.from_step
    if args.to_step is not None:
        to_step_value = args.to_step

    skip_existing = bool(skip_existing)
    direct_download = bool(direct_download)
    from_step = resolve_step_number(from_step_value, flag_name="from_step")
    to_step = resolve_step_number(to_step_value, flag_name="to_step")
    selected_steps = select_steps(from_step, to_step)

    if not direct_download and any(step.number == 4 for step in selected_steps):
        print(
            "ERROR: This automated workflow only supports the preferred direct-download route "
            "for build_total_features_sample.ipynb.",
            file=sys.stderr,
        )
        print(
            "If you need the Google Drive backup route, open "
            "Jupyter_notebook/03_environmental_data/build_total_features_sample.ipynb manually.",
            file=sys.stderr,
        )
        return 2

    python_bin = resolve_python_bin()
    print("Dataset_Construction workflow")
    print(f"- repo: {REPO_ROOT}")
    print(f"- config: {config_path}")
    print(f"- python: {python_bin}")
    print(f"- skip_existing: {skip_existing}")
    print(f"- direct_download: {direct_download}")
    print(f"- step_range: {selected_steps[0].number} -> {selected_steps[-1].number}")
    print("- preview notebook is excluded from this workflow")

    gee_preflight_done = False
    completed_steps: set[int] = set()
    for step in selected_steps:
        if skip_existing and outputs_exist(step):
            print(f"[skip] step {step.number}: {step.name}")
            for output in step.outputs:
                print(f"       existing: {output.relative_to(REPO_ROOT)}")
            continue
        if step.requires_gee and not gee_preflight_done:
            print("[check] Earth Engine authentication preflight")
            require_gee_auth(python_bin)
            gee_preflight_done = True
        if step.number == 4:
            print("[wait] required runtime factor-library assets for build_total_features")
            wait_for_runtime_factor_library_assets(
                python_bin,
                expect_export_tasks=3 in completed_steps,
            )
        exit_code = run_step_script(step, python_bin)
        if exit_code != 0:
            print(f"[fail] step {step.number}: {step.name} exited with code {exit_code}", file=sys.stderr)
            return exit_code
        completed_steps.add(step.number)

    print("[done] Dataset_Construction workflow completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
