from __future__ import annotations

DEFAULT_BATCH_SIZE = 256
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
SUPPORTED_ROUTES = ("dbscan", "no_cluster", "basin")
DEFAULT_GROUP_COL = "Sample_ID"
DEFAULT_YEAR_COL = "Feature_Year"
DEFAULT_LABEL_COL = "Label_Cls"
DEFAULT_STRATIFY_MODE = "year_plus_label"
DEFAULT_RANK_COL = "Label_Rank"
DEFAULT_BINARY_THRESHOLD = 0.5
RANK_LOSS_MODE_DISABLED = "disabled"
RANK_LOSS_MODE_NDCG_AWARE_LAMBDARANK = "ndcg_aware_lambdarank"
DEFAULT_LAMBDA_SWEEP_GRID = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

FEATURE_COLS = [
    "Slope",
    "Aspect",
    "TWI",
    "Local_Relief",
    "Soil",
    "Vegetation_type",
    "GIC",
    "TDD",
    "SMDP",
    "Precipitation",
    "WSTmax",
    "WSRH",
    "NDVI",
    "Summer_SoilMoisture",
]
