import numpy as np
import pandas as pd

from cvrank.split import (
    build_fold_manifest,
    build_spatial_block_plan,
    estimate_empirical_variogram_range,
)


def _grid_frame(n_side: int = 8, spacing_m: float = 1_000.0) -> pd.DataFrame:
    x_grid, y_grid = np.meshgrid(
        np.arange(n_side, dtype=float) * spacing_m,
        np.arange(n_side, dtype=float) * spacing_m,
    )
    x_values = x_grid.ravel()
    y_values = y_grid.ravel()
    rows = []
    for site_index, (x_value, y_value) in enumerate(zip(x_values, y_values)):
        for year_offset in (0, 1):
            rows.append(
                {
                    "Sample_ID": f"site_{site_index}",
                    "Feature_Year": 2020 + year_offset,
                    "Label_Cls": (site_index + year_offset) % 2,
                    "Longitude": x_value + (-1.0 if year_offset == 0 else 1.0),
                    "Latitude": y_value + (-1.0 if year_offset == 0 else 1.0),
                    "Signal": np.sin(x_value / 3_000.0) + np.cos(y_value / 3_000.0),
                    "Aspect__N": float(site_index % 2 == 0),
                    "Soil_missing": float(year_offset == 1),
                }
            )
    return pd.DataFrame(rows)


def _variogram_config() -> dict[str, object]:
    return {
        "spatial_x_col": "Longitude",
        "spatial_y_col": "Latitude",
        "spatial_source_crs": "EPSG:3857",
        "spatial_target_crs": "EPSG:3857",
        "spatial_variogram_feature_cols": ["Signal", "Aspect__N", "Soil_missing"],
        "spatial_variogram_n_lags": 10,
        "spatial_variogram_max_pairs": 5_000,
        "spatial_variogram_min_pairs_per_lag": 5,
        "spatial_variogram_sample_size": 100,
        "seed": 42,
    }


def test_empirical_variogram_returns_a_positive_range():
    frame = _grid_frame(n_side=8)
    value_range, diagnostics = estimate_empirical_variogram_range(
        frame["Longitude"].to_numpy(),
        frame["Latitude"].to_numpy(),
        frame["Signal"].to_numpy(),
        n_lags=10,
        max_pairs=5_000,
        min_pairs_per_lag=5,
        seed=42,
    )

    assert diagnostics["status"] == "ok"
    assert value_range is not None
    assert value_range > 0


def test_spatial_block_size_defaults_to_variogram_median_and_excludes_categorical_inputs():
    frame = _grid_frame()
    plan = build_spatial_block_plan(
        frame,
        config=_variogram_config(),
        feature_cols=["Signal", "Aspect__N", "Soil_missing"],
    )

    assert plan.block_size_m > 0
    assert plan.metadata["spatial_block_size_source"] == "variogram_median"
    assert plan.metadata["spatial_variogram_feature_cols"] == ["Signal"]
    assert set(plan.metadata["spatial_variogram_excluded_cols"]) == {"Aspect__N", "Soil_missing"}


def test_spatial_manifest_keeps_sites_and_blocks_together():
    frame = _grid_frame(n_side=6)
    config = {**_variogram_config(), "spatial_block_size_m": 1_000.0}
    manifest = build_fold_manifest(
        frame,
        n_splits=5,
        seed=42,
        split_strategy="spatial_block_cv",
        spatial_config=config,
        variogram_feature_cols=["Signal", "Aspect__N", "Soil_missing"],
    )

    assert manifest["split_strategy"].eq("spatial_block_cv").all()
    assert "spatial_block_id" in manifest.columns
    assert manifest.groupby("Sample_ID")["spatial_block_id"].nunique().max() == 1
    assert manifest.groupby(["fold", "spatial_block_id"])["split"].nunique().max() == 1
    validation_block_fold_counts = (
        manifest[manifest["split"] == "val"]
        .groupby("spatial_block_id")["fold"]
        .nunique()
    )
    assert validation_block_fold_counts.min() == 1
    assert validation_block_fold_counts.max() == 1
    assert len(manifest) == len(frame) * 5


def test_random_group_strategy_remains_available():
    frame = _grid_frame(n_side=6)
    manifest = build_fold_manifest(
        frame,
        n_splits=5,
        seed=42,
    )

    assert manifest["split_strategy"].eq("stratified_group_kfold").all()
    assert "spatial_block_id" not in manifest.columns
    assert manifest.groupby("Sample_ID")["fold"].nunique().max() == 5

