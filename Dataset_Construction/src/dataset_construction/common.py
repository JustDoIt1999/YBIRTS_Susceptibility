from __future__ import annotations

from pathlib import Path


def resolve_dataset_construction_root(
    start: Path | str = ".",
    *,
    required_relative_paths: tuple[tuple[str, ...], ...] = (
        ("inputs",),
        ("src", "dataset_construction"),
    ),
) -> Path:
    start_path = Path(start).resolve()
    candidates = [start_path, *start_path.parents]
    for candidate in candidates:
        for root in (candidate, candidate / "Dataset_Construction"):
            if all((root.joinpath(*parts)).exists() for parts in required_relative_paths):
                return root
    required_text = ", ".join("/".join(parts) for parts in required_relative_paths)
    raise FileNotFoundError(
        "Could not locate the Dataset_Construction root directory. "
        f"Expected to find: {required_text}"
    )


def python_output_root(root: Path) -> Path:
    return root / "outputs"


def notebook_root(root: Path) -> Path:
    return root / "Jupyter_notebook"
