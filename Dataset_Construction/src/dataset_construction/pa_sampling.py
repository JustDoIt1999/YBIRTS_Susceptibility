#!/usr/bin/env python3
"""Step_01 Slope Unit pseudo-absence sampling workflow."""

# # Slope Unit Pseudo-absence Sampling Workflow

from __future__ import annotations

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely import wkb
from shapely.geometry import Point
import pyproj
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dataset_construction.common import python_output_root, resolve_dataset_construction_root

print('numpy', np.__version__)
print('pandas', pd.__version__)

# ## Parameters

DATA_ROOT = resolve_dataset_construction_root(Path.cwd())
PYTHON_OUTPUT_DIR = python_output_root(DATA_ROOT) / '01_pa_sampling'
WORKING_CRS_WKT = '''PROJCS["Albers",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Albers"],PARAMETER["false_easting",0.0],PARAMETER["false_northing",0.0],PARAMETER["central_meridian",90.0],PARAMETER["standard_parallel_1",27.5],PARAMETER["standard_parallel_2",37.5],PARAMETER["latitude_of_origin",0.0],UNIT["Meter",1.0]]'''
OUTPUT_CRS_WKT = '''GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'''
PROJ4_WORKING = "+proj=aea +lat_1=27.5 +lat_2=37.5 +lat_0=0 +lon_0=90 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"

POINTS_PATH = DATA_ROOT / 'inputs' / 'positive_samples.csv'
SLOPE_UNITS_PATH = DATA_ROOT / 'inputs' / 'QTP_SlopeUnits.gpkg'
PERMAFROST_TIF_PATH = DATA_ROOT / 'inputs' / 'permafrost_qtp_2010.tif'
PERMAFROST_GEE_ASSET = 'projects/summer-presence-475413-i4/assets/permafrost_distribution_map_2010_QTP'
BOUNDARY_PATH = DATA_ROOT / 'inputs' / 'study_area' / 'QTP_boundary.shp'

OUTPUT_POSITIVE_CSV = PYTHON_OUTPUT_DIR / 'positive_su.csv'
OUTPUT_CSV = PYTHON_OUTPUT_DIR / 'pseudo_absence_random.csv'
OUTPUT_PNG = PYTHON_OUTPUT_DIR / 'pa_distribution.png'

ID_COL = 'RTS_ID'
YEAR_COL = 'Year'
LABEL_COL = 'Label_Cls'
POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0
PA_TARGET_MODE = 'match_presence'
PA_TARGET_N_PER_YEAR = None
MIN_SEP_PRESENCE_M = 1000.0
MIN_SEP_PA_M = 1000.0
MAX_LAKE_RATIO = 0.5
MIN_SLOPE_MEAN = 2.0
HARD_NEGATIVE_RATIO = 0.5        # 50% hard negatives from positive basins
GRID_SIZE_M = 50000.0            # 50 km spatial grid for balanced background
MAX_PER_GRID_CELL = 4            # Maximum background SUs per 50km cell to enforce spatial spread
SEED = 20260330
SAVE_PNG = True
OVERWRITE_OUTPUT = True

print(f"Data root: {DATA_ROOT}")
print(f"Positive samples path: {POINTS_PATH}")
print(f"Slope units path: {SLOPE_UNITS_PATH}")
print(f"Permafrost mask path: {PERMAFROST_TIF_PATH}")
print(f"Output positive SU CSV: {OUTPUT_POSITIVE_CSV}")
print(f"Output pseudo absence CSV: {OUTPUT_CSV}")

# ## 1. Load Positive Points and Match to Slope Units

if not SLOPE_UNITS_PATH.exists():
    raise FileNotFoundError(f"Slope units GPKG not found: {SLOPE_UNITS_PATH}")

required_cols = {ID_COL, 'Longitude', 'Latitude', YEAR_COL, LABEL_COL}
points_df = pd.read_csv(POINTS_PATH)
missing = required_cols.difference(points_df.columns)
if missing:
    raise KeyError(f'Missing columns in {POINTS_PATH.name}: {sorted(missing)}')

points_df = points_df.loc[points_df[LABEL_COL].eq(POSITIVE_LABEL)].copy()
points_df[YEAR_COL] = points_df[YEAR_COL].astype(int)

# Reproject points from WGS84 to Working Albers
transformer = pyproj.Transformer.from_crs("EPSG:4326", PROJ4_WORKING, always_xy=True)
inv_transformer = pyproj.Transformer.from_crs(PROJ4_WORKING, "EPSG:4326", always_xy=True)

xs, ys = transformer.transform(points_df['Longitude'].values, points_df['Latitude'].values)
points_df['x'] = xs
points_df['y'] = ys

print(f"Loaded {len(points_df)} positive presence points. Matching against Slope Units...")

conn = sqlite3.connect(str(SLOPE_UNITS_PATH))
cursor = conn.cursor()

def parse_gpkg_geom(geom_blob: bytes):
    flags = geom_blob[3]
    envelope_type = (flags >> 1) & 0x07
    env_sizes = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}
    header_len = 8 + env_sizes.get(envelope_type, 0)
    return wkb.loads(geom_blob[header_len:])


def load_or_fetch_permafrost_mask(tif_path: Path, asset_id: str) -> tuple[np.ndarray, float, float, float, float]:
    if not tif_path.exists():
        import urllib.request
        import ee
        from dataset_construction.ee_runtime import initialize_ee
        initialize_ee(context="download_permafrost_mask")
        img = ee.Image(asset_id).select(['b1'])
        url = img.getDownloadURL({
            'scale': 1000,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF'
        })
        print(f"Downloading permafrost GeoTIFF from GEE to {tif_path}...")
        tif_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, tif_path)
        print(f"Downloaded permafrost GeoTIFF: {tif_path.stat().st_size} bytes")

    arr = tifffile.imread(tif_path)
    with tifffile.TiffFile(tif_path) as tif:
        meta = tif.geotiff_metadata or {}
        model_transform = meta.get('ModelTransformation', [[0.008983152841195215, 0.0, 0.0, 73.40134186540611], [0.0, -0.008983152841195215, 0.0, 39.87621546206556]])
        x0 = float(model_transform[0][3])
        dx = float(model_transform[0][0])
        y0 = float(model_transform[1][3])
        dy = abs(float(model_transform[1][1]))
    return arr, x0, y0, dx, dy


def is_in_permafrost(lon: float, lat: float, mask_arr: np.ndarray, x0: float, y0: float, dx: float, dy: float) -> bool:
    col = int(np.floor((lon - x0) / dx))
    row = int(np.floor((y0 - lat) / dy))
    if 0 <= row < mask_arr.shape[0] and 0 <= col < mask_arr.shape[1]:
        return mask_arr[row, col] == 1  # 1 represents permafrost zone
    return False

matched_records = []
unmatched_count = 0

for idx, row in points_df.iterrows():
    x, y = row['x'], row['y']
    cursor.execute("""
        SELECT id FROM rtree_qtp_slope_units_geom 
        WHERE minx <= ? AND maxx >= ? AND miny <= ? AND maxy >= ?
    """, (x, x, y, y))
    candidates = cursor.fetchall()
    found = False
    if candidates:
        placeholders = ','.join('?' for _ in [c[0] for c in candidates])
        cursor.execute(f"SELECT fid, su_id, basin_id, geom FROM qtp_slope_units WHERE fid IN ({placeholders})", [c[0] for c in candidates])
        p = Point(x, y)
        for fid, su_id, basin_id, geom_blob in cursor.fetchall():
            geom = parse_gpkg_geom(geom_blob)
            if geom.contains(p) or geom.touches(p):
                matched_records.append({
                    'fid': fid,
                    'su_id': su_id,
                    'basin_id': basin_id,
                    'Year': int(row[YEAR_COL]),
                    'Cluster': int(row.get('Cluster', 0)),
                    'Basin_cluster': int(row.get('Basin_cluster', 0)),
                    'point_x': x,
                    'point_y': y,
                })
                found = True
                break
    if not found:
        unmatched_count += 1

print(f"Matched points: {len(matched_records)}/{len(points_df)} ({len(matched_records)/len(points_df)*100:.2f}%)")
if unmatched_count > 0:
    print(f"Notice: {unmatched_count} points did not intersect any SU (likely outside boundary or on lake water).")

pos_matched_df = pd.DataFrame(matched_records)

# Aggregate multiple points within the same Slope Unit:
# 1. Earliest initiation year min(Year)
# 2. Inherit Cluster and Basin_cluster (100% homogeneous within each SU)
pos_su_agg = pos_matched_df.groupby('su_id').agg({
    'fid': 'first',
    'basin_id': 'first',
    'Year': 'min',
    'Cluster': 'first',
    'Basin_cluster': 'first',
}).reset_index()

pos_fids = set(pos_su_agg['fid'].tolist())
print(f"Total unique positive Slope Units: {len(pos_su_agg)}")

# Extract polygon centroids for all positive Slope Units
pos_fid_list = list(pos_fids)
placeholders = ','.join('?' for _ in pos_fid_list)
cursor.execute(f"SELECT fid, geom FROM qtp_slope_units WHERE fid IN ({placeholders})", pos_fid_list)

pos_centroids = {}
for fid, geom_blob in cursor.fetchall():
    geom = parse_gpkg_geom(geom_blob)
    c = geom.centroid
    pos_centroids[fid] = (c.x, c.y)

pos_su_agg['cx'] = pos_su_agg['fid'].map(lambda f: pos_centroids[f][0])
pos_su_agg['cy'] = pos_su_agg['fid'].map(lambda f: pos_centroids[f][1])

# Convert centroids to WGS84
pos_lons, pos_lats = inv_transformer.transform(pos_su_agg['cx'].values, pos_su_agg['cy'].values)
pos_su_agg['Longitude'] = pos_lons
pos_su_agg['Latitude'] = pos_lats
pos_su_agg['Label'] = POSITIVE_LABEL
pos_su_agg['Label_Cls'] = POSITIVE_LABEL
pos_su_agg['Sample_ID'] = pos_su_agg['su_id']

presence_counts = pos_su_agg.groupby(YEAR_COL).size().rename('n_presence').sort_index()
print(f"Positive Slope Unit yearly counts:\n{presence_counts}")

pos_coords = np.column_stack([pos_su_agg['cx'].values, pos_su_agg['cy'].values])
pos_tree = cKDTree(pos_coords)

# ## 2. Load Permafrost Mask and Prepare Candidate Pools

permafrost_mask, pf_x0, pf_y0, pf_dx, pf_dy = load_or_fetch_permafrost_mask(
    PERMAFROST_TIF_PATH, PERMAFROST_GEE_ASSET
)
total_pf_pixels = int((permafrost_mask == 1).sum())
print(f"Permafrost mask loaded: shape={permafrost_mask.shape}, permafrost pixels={total_pf_pixels}")

pos_basin_list = sorted(set(pos_su_agg['basin_id'].unique()))
print(f"Number of unique basins with RTS presence: {len(pos_basin_list)}")

placeholders = ','.join('?' for _ in pos_basin_list)
cursor.execute(f"SELECT fid FROM qtp_slope_units WHERE basin_id IN ({placeholders})", pos_basin_list)
hard_candidate_fids = np.array([r[0] for r in cursor.fetchall() if r[0] not in pos_fids])
print(f"Total hard negative candidate SUs in positive basins: {len(hard_candidate_fids)}")

cursor.execute("SELECT MAX(fid) FROM qtp_slope_units")
max_fid = cursor.fetchone()[0]
print(f"Total slope units in GPKG: {max_fid}")

# ## 3. Sample Negative Slope Units (Hybrid: Hard Negatives + Spatially Balanced Background)

if PA_TARGET_MODE == 'match_presence':
    targets = presence_counts.to_dict()
elif PA_TARGET_MODE == 'fixed_per_year':
    if PA_TARGET_N_PER_YEAR is None:
        raise ValueError('PA_TARGET_N_PER_YEAR must be provided when PA_TARGET_MODE == fixed_per_year')
    targets = {int(year): int(PA_TARGET_N_PER_YEAR) for year in presence_counts.index}
else:
    raise ValueError(f'Unsupported PA_TARGET_MODE: {PA_TARGET_MODE}')

rng = np.random.default_rng(SEED)
selected_pa_records = []
selected_pa_fids = set()
selected_pa_coords: list[list[float]] = []
grid_cell_counts: dict[tuple[int, int], int] = {}

for year, target_n in targets.items():
    target_hard = int(target_n * HARD_NEGATIVE_RATIO)
    target_bg = target_n - target_hard
    print(f"\n--- Sampling for Year {year}: Total={target_n} (Hard={target_hard}, Spatially Balanced BG={target_bg}) ---")

    # Part A: Sample Hard Negatives (From Positive Basins within Permafrost Zone)
    hard_collected = 0
    passes = 0
    while hard_collected < target_hard and passes < 500:
        passes += 1
        needed = target_hard - hard_collected
        batch_size = max(50, needed * 4)
        sample_pool = [f for f in hard_candidate_fids if f not in selected_pa_fids]
        if not sample_pool:
            break
        chosen_fids = rng.choice(sample_pool, size=min(batch_size, len(sample_pool)), replace=False).tolist()

        placeholders = ','.join('?' for _ in chosen_fids)
        cursor.execute(f"""
            SELECT fid, su_id, basin_id, slope_mean, lake_ratio, geom 
            FROM qtp_slope_units 
            WHERE fid IN ({placeholders})
        """, chosen_fids)

        for fid, su_id, basin_id, slope_mean, lake_ratio, geom_blob in cursor.fetchall():
            if fid in selected_pa_fids:
                continue
            if (lake_ratio is not None and lake_ratio > MAX_LAKE_RATIO) or (slope_mean is not None and slope_mean < MIN_SLOPE_MEAN):
                continue

            geom = parse_gpkg_geom(geom_blob)
            c = geom.centroid
            cx, cy = c.x, c.y
            lon, lat = inv_transformer.transform(cx, cy)

            # Check permafrost mask (Class 1)
            if not is_in_permafrost(lon, lat, permafrost_mask, pf_x0, pf_y0, pf_dx, pf_dy):
                continue

            # Check distance to presence SUs
            dist_to_pos = pos_tree.query([cx, cy], k=1)[0]
            if dist_to_pos < MIN_SEP_PRESENCE_M:
                continue

            # Check distance to already selected PA SUs
            if selected_pa_coords:
                pa_tree = cKDTree(np.array(selected_pa_coords))
                dist_to_pa = pa_tree.query([cx, cy], k=1)[0]
                if dist_to_pa < MIN_SEP_PA_M:
                    continue

            selected_pa_fids.add(fid)
            selected_pa_coords.append([cx, cy])
            selected_pa_records.append({
                'PA_ID': su_id,
                'Sample_ID': su_id,
                'su_id': su_id,
                'basin_id': basin_id,
                'Longitude': lon,
                'Latitude': lat,
                'Year': int(year),
                'Label': NEGATIVE_LABEL,
                'Label_Cls': NEGATIVE_LABEL,
                'Cluster': 0,
                'Basin_cluster': 0,
                'Strategy': 'hard_negative',
                'cx': cx,
                'cy': cy,
            })
            hard_collected += 1
            if hard_collected >= target_hard:
                break

    print(f"[{year}] Collected {hard_collected}/{target_hard} Hard Negative SUs in positive basins.")

    # Part B: Sample Spatially Balanced Background Negatives (Enforcing Spatial Spread across Permafrost Grid)
    bg_collected = 0
    passes = 0
    while bg_collected < target_bg and passes < 1000:
        passes += 1
        needed = target_bg - bg_collected
        batch_size = max(100, needed * 5)
        candidate_fids = rng.integers(1, max_fid + 1, size=batch_size).tolist()
        candidate_fids = [f for f in candidate_fids if f not in pos_fids and f not in selected_pa_fids]
        if not candidate_fids:
            continue

        placeholders = ','.join('?' for _ in candidate_fids)
        cursor.execute(f"""
            SELECT fid, su_id, basin_id, slope_mean, lake_ratio, geom 
            FROM qtp_slope_units 
            WHERE fid IN ({placeholders})
        """, candidate_fids)

        for fid, su_id, basin_id, slope_mean, lake_ratio, geom_blob in cursor.fetchall():
            if fid in selected_pa_fids:
                continue
            if (lake_ratio is not None and lake_ratio > MAX_LAKE_RATIO) or (slope_mean is not None and slope_mean < MIN_SLOPE_MEAN):
                continue

            geom = parse_gpkg_geom(geom_blob)
            c = geom.centroid
            cx, cy = c.x, c.y
            lon, lat = inv_transformer.transform(cx, cy)

            # Check permafrost mask (Class 1)
            if not is_in_permafrost(lon, lat, permafrost_mask, pf_x0, pf_y0, pf_dx, pf_dy):
                continue

            # Grid cell quota check to ensure spatial spread
            gx = int(np.floor(cx / GRID_SIZE_M))
            gy = int(np.floor(cy / GRID_SIZE_M))
            cell_key = (gx, gy)
            if grid_cell_counts.get(cell_key, 0) >= MAX_PER_GRID_CELL:
                continue

            # Check distance to presence SUs
            dist_to_pos = pos_tree.query([cx, cy], k=1)[0]
            if dist_to_pos < MIN_SEP_PRESENCE_M:
                continue

            # Check distance to already selected PA SUs
            if selected_pa_coords:
                pa_tree = cKDTree(np.array(selected_pa_coords))
                dist_to_pa = pa_tree.query([cx, cy], k=1)[0]
                if dist_to_pa < MIN_SEP_PA_M:
                    continue

            selected_pa_fids.add(fid)
            selected_pa_coords.append([cx, cy])
            grid_cell_counts[cell_key] = grid_cell_counts.get(cell_key, 0) + 1
            selected_pa_records.append({
                'PA_ID': su_id,
                'Sample_ID': su_id,
                'su_id': su_id,
                'basin_id': basin_id,
                'Longitude': lon,
                'Latitude': lat,
                'Year': int(year),
                'Label': NEGATIVE_LABEL,
                'Label_Cls': NEGATIVE_LABEL,
                'Cluster': 0,
                'Basin_cluster': 0,
                'Strategy': 'spatial_balanced',
                'cx': cx,
                'cy': cy,
            })
            bg_collected += 1
            if bg_collected >= target_bg:
                break

    print(f"[{year}] Collected {bg_collected}/{target_bg} Spatially Balanced Background SUs across permafrost grid.")

conn.close()

pa_df = pd.DataFrame(selected_pa_records)
print(f"\nSuccessfully sampled {len(pa_df)} negative Slope Units across all years.")
print("Strategy distribution:\n", pa_df['Strategy'].value_counts())
print(f"Negative SUs cover {pa_df['basin_id'].nunique()} unique basins across QTP.")

# ## 3. Quality Control

pa_coords = np.column_stack([pa_df['cx'].values, pa_df['cy'].values])
min_dist_to_pos = pos_tree.query(pa_coords, k=1)[0].min()
min_dist_between_pa = cKDTree(pa_coords).query(pa_coords, k=2)[0][:, 1].min() if len(pa_coords) > 1 else np.nan

qc_summary = pd.DataFrame({
    'target_n': pd.Series(targets),
    'generated_n': pa_df.groupby(YEAR_COL).size(),
})
qc_summary['match'] = qc_summary['target_n'].eq(qc_summary['generated_n'])

print(qc_summary)
print(f'Minimum distance to any positive SU: {min_dist_to_pos:.2f} m (threshold >= {MIN_SEP_PRESENCE_M} m)')
print(f'Minimum distance between negative SUs: {min_dist_between_pa:.2f} m (threshold >= {MIN_SEP_PA_M} m)')

if not qc_summary['match'].all():
    raise AssertionError('Yearly pseudo-absence counts do not match presence targets.')
if min_dist_to_pos < MIN_SEP_PRESENCE_M:
    raise AssertionError('Presence distance constraint violated.')
if MIN_SEP_PA_M > 0 and min_dist_between_pa < MIN_SEP_PA_M:
    raise AssertionError('PA-to-PA distance constraint violated.')

# ## 4. Export Results

PYTHON_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Export Positive Slope Units
pos_export_cols = ['Sample_ID', 'su_id', 'basin_id', 'Longitude', 'Latitude', 'Year', 'Label', 'Label_Cls', 'Cluster', 'Basin_cluster']
pos_export_df = pos_su_agg[pos_export_cols].copy()
pos_export_df.to_csv(OUTPUT_POSITIVE_CSV, index=False)
print(f'Wrote positive slope units: {OUTPUT_POSITIVE_CSV}')

# Export Negative Slope Units
pa_export_cols = ['PA_ID', 'Sample_ID', 'su_id', 'basin_id', 'Longitude', 'Latitude', 'Year', 'Label', 'Label_Cls', 'Cluster', 'Basin_cluster', 'Strategy']
pa_export_df = pa_df[pa_export_cols].copy()
pa_export_df.to_csv(OUTPUT_CSV, index=False)
print(f'Wrote pseudo absence slope units: {OUTPUT_CSV}')

# ## 5. Distribution Plotting

if SAVE_PNG:
    try:
        import seaborn as sns
        from matplotlib.lines import Line2D
        from matplotlib.ticker import FuncFormatter, MultipleLocator

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

        fig, ax = plt.subplots(figsize=(12.2, 7.3), dpi=200)
        
        # Plot study area boundary if available
        boundary_plotted = False
        if BOUNDARY_PATH.exists():
            try:
                import geopandas as gpd
                boundary_gdf = gpd.read_file(BOUNDARY_PATH)
                boundary_wgs84 = boundary_gdf.to_crs(OUTPUT_CRS_WKT)
                boundary_wgs84.plot(
                    ax=ax,
                    facecolor='#fcfbf6',
                    edgecolor='#9a9a9a',
                    linewidth=0.85,
                    zorder=1,
                )
                boundary_plotted = True
            except Exception:
                try:
                    import shapefile
                    from matplotlib.patches import Polygon as MplPolygon
                    from matplotlib.collections import PatchCollection
                    sf = shapefile.Reader(str(BOUNDARY_PATH))
                    patches = []
                    for shape in sf.shapes():
                        pts = np.array(shape.points)
                        lons, lats = inv_transformer.transform(pts[:, 0], pts[:, 1])
                        polygon = MplPolygon(np.column_stack([lons, lats]), closed=True)
                        patches.append(polygon)
                    p_coll = PatchCollection(patches, facecolor='#fcfbf6', edgecolor='#9a9a9a', linewidth=0.85, zorder=1)
                    ax.add_collection(p_coll)
                    boundary_plotted = True
                    print("Plotted QTP boundary using shapefile.")
                except Exception as e2:
                    print(f"Boundary plotting skipped: {e2}")

        # Plot Background Negatives
        bg_subset = pa_export_df.loc[pa_export_df['Strategy'].eq('spatial_balanced')]
        if not bg_subset.empty:
            ax.scatter(
                bg_subset['Longitude'],
                bg_subset['Latitude'],
                s=8,
                color='#1f77b4',
                alpha=0.65,
                linewidths=0,
                rasterized=True,
                zorder=2,
                label=f'PA Balanced BG ({len(bg_subset)})',
            )

        # Plot Hard Negatives
        hard_subset = pa_export_df.loc[pa_export_df['Strategy'].eq('hard_negative')]
        if not hard_subset.empty:
            ax.scatter(
                hard_subset['Longitude'],
                hard_subset['Latitude'],
                s=8,
                color='#ff7f0e',
                alpha=0.75,
                linewidths=0,
                rasterized=True,
                zorder=3,
                label=f'PA Hard Negatives ({len(hard_subset)})',
            )

        # Plot Presence
        ax.scatter(
            pos_export_df['Longitude'],
            pos_export_df['Latitude'],
            s=9,
            color='#d62728',
            alpha=0.6,
            linewidths=0,
            rasterized=True,
            zorder=4,
            label=f'Presence SU ({len(pos_export_df)})',
        )

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Permafrost Slope Unit PA Sampling: Hard Negatives vs. Spatially Balanced BG', fontsize=13, fontweight='bold', pad=15)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0f}°E'))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:.0f}°N'))
        ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)

        fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f'Wrote distribution plot: {OUTPUT_PNG}')
    except Exception as e:
        print(f"PNG generation skipped due to error: {e}")

print("Step 01 Slope Unit pa_sampling workflow completed successfully.")
