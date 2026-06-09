from __future__ import annotations

import os

import ee

PUBLIC_ASSET_PROJECT_ID = "summer-presence-475413-i4"
PUBLIC_ASSET_ROOT = f"projects/{PUBLIC_ASSET_PROJECT_ID}/assets"
NOT_FOUND_MARKERS = (
    "not found",
    "does not exist",
    "asset not found",
    "asset does not exist",
)


def env_flag_enabled(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def runtime_project_id(*, required: bool = True, context: str = "workflow") -> str:
    project_id = os.environ.get("EE_PROJECT_ID", "").strip()
    if project_id or not required:
        return project_id
    raise RuntimeError(
        "EE_PROJECT_ID is not set. This Step_01 "
        f"{context} reads public Earth Engine assets from PUBLIC_ASSET_ROOT, but "
        "computations and exports must run under your own Earth Engine-enabled Google Cloud project.\n\n"
        "Set it before running:\n"
        "  macOS/Linux: export EE_PROJECT_ID='your-project-id'\n"
        "  Windows PowerShell: $env:EE_PROJECT_ID='your-project-id'"
    )


def runtime_asset_root(project_id: str | None = None) -> str:
    selected_project = project_id or runtime_project_id()
    return f"projects/{selected_project}/assets"


def initialize_ee(
    *,
    project_id: str | None = None,
    allow_interactive_auth: bool | None = None,
    context: str = "workflow",
) -> str:
    selected_project = project_id or runtime_project_id(context=context)
    interactive_allowed = (
        env_flag_enabled("EE_ALLOW_INTERACTIVE_AUTH")
        if allow_interactive_auth is None
        else bool(allow_interactive_auth)
    )
    try:
        ee.Initialize(project=selected_project)
    except Exception as exc:
        if not interactive_allowed:
            raise RuntimeError(
                "Earth Engine initialization failed and interactive authentication is disabled.\n"
                "Run `earthengine authenticate`, set GOOGLE_APPLICATION_CREDENTIALS, or set "
                "`EE_ALLOW_INTERACTIVE_AUTH=1` before running this workflow interactively.\n\n"
                f"Details: {exc}"
            ) from exc
        print("Earth Engine initialization failed. Attempting interactive authentication...")
        ee.Authenticate()
        ee.Initialize(project=selected_project)
    return selected_project


def asset_exists(asset_id: str) -> bool:
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception as exc:
        message = str(exc).lower()
        if any(marker in message for marker in NOT_FOUND_MARKERS):
            return False
        raise RuntimeError(f"Failed to check Earth Engine asset existence: {asset_id}") from exc
