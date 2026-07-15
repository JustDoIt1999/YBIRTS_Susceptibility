from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .batch_size import normalize_batch_size
from .output_suffix import append_suffix_to_name, normalize_output_suffix
from .rank_loss_mode import normalize_rank_loss_mode


ARTIFACT_LAYOUT_VERSION = 1
SELECTED_MODEL_ROUTES = {"dbscan", "no_cluster", "basin"}


@dataclass(frozen=True)
class SelectedModelIdentity:
    route: str
    batch_size: int
    rank_loss_mode: str
    train_output_suffix: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_selected_model_identity(
    *,
    route: Any,
    batch_size: Any,
    rank_loss_mode: Any,
    train_output_suffix: Any | None = None,
) -> SelectedModelIdentity:
    normalized_route = str(route or "").strip().lower()
    if normalized_route not in SELECTED_MODEL_ROUTES:
        raise ValueError(
            f"Unsupported selected-model route: {route!r}; "
            f"supported={sorted(SELECTED_MODEL_ROUTES)}"
        )
    return SelectedModelIdentity(
        route=normalized_route,
        batch_size=normalize_batch_size(batch_size),
        rank_loss_mode=normalize_rank_loss_mode(rank_loss_mode),
        train_output_suffix=normalize_output_suffix(train_output_suffix),
    )


def build_selected_route_root(
    cvrank_output_root: str | Path,
    identity: SelectedModelIdentity,
) -> Path:
    route_component = append_suffix_to_name(
        f"output_{identity.route}_selected",
        identity.train_output_suffix,
    )
    return (
        Path(cvrank_output_root)
        / route_component
        / f"bs_{identity.batch_size}"
        / identity.rank_loss_mode
    )


def build_selected_train_config_path(
    cvrank_output_root: str | Path,
    identity: SelectedModelIdentity,
) -> Path:
    return (
        build_selected_route_root(cvrank_output_root, identity)
        / "selection"
        / f"{identity.route}_best_train.yaml"
    )


def build_legacy_untagged_batch_route_root(
    cvrank_output_root: str | Path,
    identity: SelectedModelIdentity,
) -> Path:
    route_component = append_suffix_to_name(
        f"output_{identity.route}_selected",
        identity.train_output_suffix,
    )
    return Path(cvrank_output_root) / route_component / identity.rank_loss_mode


def build_legacy_untagged_batch_train_config_path(
    cvrank_output_root: str | Path,
    identity: SelectedModelIdentity,
) -> Path:
    return (
        build_legacy_untagged_batch_route_root(cvrank_output_root, identity)
        / "selection"
        / f"{identity.route}_best_train.yaml"
    )


__all__ = [
    "ARTIFACT_LAYOUT_VERSION",
    "SELECTED_MODEL_ROUTES",
    "SelectedModelIdentity",
    "build_legacy_untagged_batch_route_root",
    "build_legacy_untagged_batch_train_config_path",
    "build_selected_route_root",
    "build_selected_train_config_path",
    "make_selected_model_identity",
]
