#!/usr/bin/env python3
"""Step_01 pseudo-absence sampling workflow."""

# # Pseudo-absence Sampling Workflow

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import contains_xy, make_valid

from dataset_construction.common import python_output_root, resolve_dataset_construction_root

print('geopandas', gpd.__version__)
print('numpy', np.__version__)

# ## Parameters

DATA_ROOT = resolve_dataset_construction_root(Path.cwd())
PYTHON_OUTPUT_DIR = python_output_root(DATA_ROOT) / '01_pa_sampling'
WORKING_CRS_WKT = '''PROJCS["Albers",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",90.0],PARAMETER["standard_parallel_1",27.5],PARAMETER["standard_parallel_2",37.5],PARAMETER["latitude_of_origin",0.0],UNIT["Meter",1.0]]'''
OUTPUT_CRS_WKT = '''GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'''
POINTS_PATH = DATA_ROOT / 'inputs' / 'positive_samples.csv'
BOUNDARY_PATH = DATA_ROOT / 'inputs' / 'study_area' / 'QTP_boundary.shp'
OUTPUT_CSV = PYTHON_OUTPUT_DIR / 'pseudo_absence_random.csv'
OUTPUT_PNG = PYTHON_OUTPUT_DIR / 'pa_distribution.png'

ID_COL = 'RTS_ID'
YEAR_COL = 'Year'
LABEL_COL = 'Label_Cls'
POSITIVE_LABEL = 1
PA_TARGET_MODE = 'match_presence'
PA_TARGET_N_PER_YEAR = None
MIN_SEP_PRESENCE_M = 1000.0
MIN_SEP_PA_M = 1.0
BATCH_FACTOR = 100
MAX_PASSES = 10000
SEED = 20260330
SAVE_PNG = True
OVERWRITE_OUTPUT = True

print(DATA_ROOT)
print(OUTPUT_CSV)

# ## Load Study Area and Presence (Positive) Samples

required_cols = {ID_COL, 'Longitude', 'Latitude', YEAR_COL, LABEL_COL}
points_df = pd.read_csv(POINTS_PATH)
missing = required_cols.difference(points_df.columns)
if missing:
    raise KeyError(f'Missing columns in {POINTS_PATH.name}: {sorted(missing)}')

points_df = points_df.loc[points_df[LABEL_COL].eq(POSITIVE_LABEL), [ID_COL, 'Longitude', 'Latitude', YEAR_COL, LABEL_COL]].copy()
points_df[YEAR_COL] = points_df[YEAR_COL].astype(int)

presence_gdf = gpd.GeoDataFrame(
    points_df,
    geometry=gpd.points_from_xy(points_df['Longitude'], points_df['Latitude']),
    crs=OUTPUT_CRS_WKT,
).to_crs(WORKING_CRS_WKT)

boundary_gdf = gpd.read_file(BOUNDARY_PATH)
if boundary_gdf.crs is None:
    boundary_gdf = boundary_gdf.set_crs(WORKING_CRS_WKT)
boundary_gdf = boundary_gdf.to_crs(WORKING_CRS_WKT)
boundary_geom = make_valid(boundary_gdf.geometry.union_all())

presence_counts = presence_gdf.groupby(YEAR_COL).size().rename('n_presence').sort_index()
presence_counts

# ## Sampling Functions

def geometry_random_points(geometry, n: int, rng: np.random.Generator) -> np.ndarray:
    minx, miny, maxx, maxy = geometry.bounds
    accepted = []
    remaining = n
    while remaining > 0:
        batch = max(1000, remaining * 3)
        xs = rng.uniform(minx, maxx, batch)
        ys = rng.uniform(miny, maxy, batch)
        mask = contains_xy(geometry, xs, ys)
        if np.any(mask):
            pts = np.column_stack([xs[mask], ys[mask]])
            accepted.append(pts)
            remaining -= len(pts)
    return np.vstack(accepted)[:n]



def filter_by_min_distance(candidates: np.ndarray, references: np.ndarray, min_distance: float) -> np.ndarray:
    if len(candidates) == 0 or min_distance <= 0 or len(references) == 0:
        return candidates
    tree = cKDTree(references)
    nearest_dist, _ = tree.query(candidates, k=1)
    return candidates[nearest_dist >= min_distance]



def drop_close_pairs(candidates: np.ndarray, min_distance: float) -> np.ndarray:
    if len(candidates) <= 1 or min_distance <= 0:
        return candidates
    tree = cKDTree(candidates)
    pairs = tree.query_pairs(r=min_distance, output_type='ndarray')
    if len(pairs) == 0:
        return candidates
    drop_idx = set()
    for left, right in pairs:
        if left not in drop_idx:
            drop_idx.add(int(right))
    keep = np.ones(len(candidates), dtype=bool)
    keep[list(drop_idx)] = False
    return candidates[keep]



def sample_pa_one_year(
    geometry,
    n_target: int,
    presence_coords: np.ndarray,
    accum_coords: np.ndarray,
    *,
    min_sep_presence_m: float,
    min_sep_pa_m: float,
    batch_factor: int,
    max_passes: int,
    seed: int,
    year: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed + int(year))
    remain = int(n_target)
    passes = 0
    accepted_chunks: list[np.ndarray] = []
    local_accum = accum_coords.copy()

    while remain > 0 and passes < max_passes:
        passes += 1
        batch_size = max(1, min(int(remain * batch_factor), 10000))
        cand = geometry_random_points(geometry, batch_size, rng)
        cand = filter_by_min_distance(cand, presence_coords, min_sep_presence_m)
        cand = filter_by_min_distance(cand, local_accum, min_sep_pa_m)
        cand = drop_close_pairs(cand, min_sep_pa_m)
        if len(cand) == 0:
            continue
        take = cand[:remain]
        accepted_chunks.append(take)
        remain -= len(take)
        local_accum = np.vstack([local_accum, take]) if len(local_accum) else take

    if remain > 0:
        raise RuntimeError(
            f'Year {year} still needs {remain} pseudo-absence points after {max_passes} passes. '
            'Increase MAX_PASSES or reduce the minimum-distance constraints.'
        )

    return np.vstack(accepted_chunks)

# ## Run Sampling

presence_coords = np.column_stack([presence_gdf.geometry.x.to_numpy(), presence_gdf.geometry.y.to_numpy()])

if PA_TARGET_MODE == 'match_presence':
    targets = presence_counts.to_dict()
elif PA_TARGET_MODE == 'fixed_per_year':
    if PA_TARGET_N_PER_YEAR is None:
        raise ValueError('PA_TARGET_N_PER_YEAR must be provided when PA_TARGET_MODE == fixed_per_year')
    targets = {int(year): int(PA_TARGET_N_PER_YEAR) for year in presence_counts.index}
else:
    raise ValueError(f'Unsupported PA_TARGET_MODE: {PA_TARGET_MODE}')

accum_coords = np.empty((0, 2), dtype=float)
pa_parts = []

for year, n_target in targets.items():
    sampled = sample_pa_one_year(
        boundary_geom,
        n_target=int(n_target),
        presence_coords=presence_coords,
        accum_coords=accum_coords,
        min_sep_presence_m=MIN_SEP_PRESENCE_M,
        min_sep_pa_m=MIN_SEP_PA_M,
        batch_factor=BATCH_FACTOR,
        max_passes=MAX_PASSES,
        seed=SEED,
        year=int(year),
    )
    accum_coords = np.vstack([accum_coords, sampled]) if len(accum_coords) else sampled
    pa_parts.append(
        pd.DataFrame(
            {
                'Year': int(year),
                'PA_index': np.arange(1, len(sampled) + 1, dtype=int),
                'X_Albers': sampled[:, 0],
                'Y_Albers': sampled[:, 1],
            }
        )
    )

pa_working_df = pd.concat(pa_parts, ignore_index=True)
pa_working_df['PA_ID'] = [f"PA_{year}_{idx:04d}" for year, idx in zip(pa_working_df['Year'], pa_working_df['PA_index'])]
pa_working_df['Strategy'] = 'random'

pa_working_gdf = gpd.GeoDataFrame(
    pa_working_df,
    geometry=gpd.points_from_xy(pa_working_df['X_Albers'], pa_working_df['Y_Albers']),
    crs=WORKING_CRS_WKT,
)
pa_wgs84_gdf = pa_working_gdf.to_crs(OUTPUT_CRS_WKT)
pa_working_df['Longitude'] = pa_wgs84_gdf.geometry.x
pa_working_df['Latitude'] = pa_wgs84_gdf.geometry.y

pa_export_df = pa_working_df[['PA_ID', 'Longitude', 'Latitude', 'Year', 'Strategy']].copy()
pa_export_df.head()

# ## Quality Control

pa_coords = pa_working_df[['X_Albers', 'Y_Albers']].to_numpy()
pres_tree = cKDTree(presence_coords)
pa_tree = cKDTree(pa_coords)

min_dist_to_presence = pres_tree.query(pa_coords, k=1)[0].min()
min_dist_between_pa = pa_tree.query(pa_coords, k=2)[0][:, 1].min() if len(pa_coords) > 1 else np.nan

qc_summary = pd.DataFrame(
    {
        'target_n': pd.Series(targets),
        'generated_n': pa_export_df.groupby('Year').size(),
    }
)
qc_summary['match'] = qc_summary['target_n'].eq(qc_summary['generated_n'])

print(qc_summary)
print(f'min distance to any presence: {min_dist_to_presence:.3f} m')
print(f'min distance between PA points: {min_dist_between_pa:.3f} m')

if not qc_summary['match'].all():
    raise AssertionError('Yearly pseudo-absence counts do not match targets.')
if min_dist_to_presence < MIN_SEP_PRESENCE_M:
    raise AssertionError('Presence distance constraint violated.')
if MIN_SEP_PA_M > 0 and min_dist_between_pa < MIN_SEP_PA_M:
    raise AssertionError('PA-to-PA distance constraint violated.')

# ## Export Results and Optional Comparison

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
if OUTPUT_CSV.exists() and not OVERWRITE_OUTPUT:
    raise FileExistsError(f'{OUTPUT_CSV} already exists. Set OVERWRITE_OUTPUT = True to replace it.')

pa_export_df.to_csv(OUTPUT_CSV, index=False)
print(f'Wrote: {OUTPUT_CSV}')

if SAVE_PNG:
    import seaborn as sns
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    boundary_wgs84 = boundary_gdf.to_crs(OUTPUT_CRS_WKT)
    presence_wgs84 = presence_gdf.to_crs(OUTPUT_CRS_WKT)
    pa_plot_gdf = pa_wgs84_gdf.copy()
    year_order = sorted(pa_plot_gdf[YEAR_COL].unique())
    ggplot_like_palette = ['#F8766D', '#00BFC4', '#7CAE00', '#C77CFF', '#E6AB02', '#00BA38', '#619CFF']
    year_colors = {
        int(year): ggplot_like_palette[idx % len(ggplot_like_palette)]
        for idx, year in enumerate(year_order)
    }

    sns.set_theme(
        style='whitegrid',
        context='notebook',
        rc={
            'axes.facecolor': '#fbfbf8',
            'figure.facecolor': '#ffffff',
            'grid.color': '#dfdfdf',
            'grid.linewidth': 0.7,
            'axes.edgecolor': '#d0d0d0',
            'axes.labelcolor': '#3a3a3a',
            'xtick.color': '#666666',
            'ytick.color': '#666666',
        },
    )

    minx, miny, maxx, maxy = boundary_wgs84.total_bounds
    x_pad = (maxx - minx) * 0.03
    y_pad = (maxy - miny) * 0.05
    mid_lat = (miny + maxy) / 2

    fig, ax = plt.subplots(figsize=(12.2, 7.3), dpi=200)
    boundary_wgs84.plot(
        ax=ax,
        facecolor='#fcfbf6',
        edgecolor='#9a9a9a',
        linewidth=0.85,
        zorder=1,
    )

    for year in year_order:
        subset = pa_plot_gdf.loc[pa_plot_gdf[YEAR_COL].eq(year)]
        ax.scatter(
            subset.geometry.x,
            subset.geometry.y,
            s=3,
            color=year_colors[int(year)],
            alpha=0.72,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

    ax.scatter(
        presence_wgs84.geometry.x,
        presence_wgs84.geometry.y,
        s=4.2,
        color='#202020',
        alpha=0.55,
        linewidths=0,
        rasterized=True,
        zorder=3,
    )

    ax.set_aspect(1 / np.cos(np.deg2rad(mid_lat)))
    ax.set_xlim(minx - x_pad, maxx + x_pad)
    ax.set_ylim(miny - y_pad, maxy + y_pad)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('')
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0f}\N{DEGREE SIGN}E'))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0f}\N{DEGREE SIGN}N'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#d0d0d0')
    ax.spines['bottom'].set_color('#d0d0d0')
    ax.grid(True, color='#dfdfdf', linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.08, right=0.80, top=0.83, bottom=0.12)
    fig.text(
        0.08,
        0.89,
        'Pseudo-absence vs. presence (random, by year)',
        fontsize=13,
        fontweight='bold',
        ha='left',
        va='bottom',
    )
    fig.text(
        0.08,
        0.865,
        f'Presences n={len(presence_wgs84)} | PA n={len(pa_plot_gdf)}',
        fontsize=10.5,
        color='#3a3a3a',
        ha='left',
        va='bottom',
    )

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle='none',
            marker='o',
            markersize=4.8,
            markerfacecolor=year_colors[int(year)],
            markeredgewidth=0,
            alpha=0.72,
            label=str(year),
        )
        for year in year_order
    ]
    ax.legend(
        handles=legend_handles,
        title='PA year',
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        frameon=False,
        labelspacing=0.9,
        handletextpad=0.6,
        borderaxespad=0.0,
    )

    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    plt.close(fig)
    print(f'Wrote: {OUTPUT_PNG}')
