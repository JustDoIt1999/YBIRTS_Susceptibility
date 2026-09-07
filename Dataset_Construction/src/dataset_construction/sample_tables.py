#!/usr/bin/env python3
"""Step_01 sample-table construction workflow."""

# # RTS Total Sample

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas as pd
try:
    from IPython.display import display
except ImportError:
    display = print

from dataset_construction.common import python_output_root, resolve_dataset_construction_root

# ## Plan

DATA_ROOT = resolve_dataset_construction_root(Path.cwd())
PYTHON_OUTPUT_ROOT = python_output_root(DATA_ROOT)
POSITIVE_SU_PATH = PYTHON_OUTPUT_ROOT / "01_pa_sampling" / "positive_su.csv"
POSITIVE_PATH = POSITIVE_SU_PATH if POSITIVE_SU_PATH.exists() else DATA_ROOT / "inputs" / "positive_samples.csv"
PA_PATH = PYTHON_OUTPUT_ROOT / "01_pa_sampling" / "pseudo_absence_random.csv"
OUTPUT_PATH = PYTHON_OUTPUT_ROOT / "02_sample_tables" / "RTS_Total_sample.csv"

POSITIVE_LABEL = 1
NEGATIVE_LABEL = 0

print({
    "data_root": str(DATA_ROOT),
    "positive_path": str(POSITIVE_PATH),
    "pseudo_absence_path": str(PA_PATH),
    "output_path": str(OUTPUT_PATH),
})

# ## Standardize The Two Inputs

positive_df = pd.read_csv(POSITIVE_PATH)
pa_df = pd.read_csv(PA_PATH)

if "Sample_ID" not in positive_df.columns and "RTS_ID" in positive_df.columns:
    positive_df = positive_df.rename(columns={"RTS_ID": "Sample_ID"})

positive_required = {"Sample_ID", "Longitude", "Latitude", "Cluster", "Basin_cluster", "Year", "Label_Cls"}
pa_required = {"Sample_ID", "Longitude", "Latitude", "Year"} if "Sample_ID" in pa_df.columns else {"PA_ID", "Longitude", "Latitude", "Year"}

positive_missing = positive_required.difference(positive_df.columns)
pa_missing = pa_required.difference(pa_df.columns)
if positive_missing:
    raise KeyError(f"Missing columns in {POSITIVE_PATH.name}: {sorted(positive_missing)}")
if pa_missing:
    raise KeyError(f"Missing columns in {PA_PATH.name}: {sorted(pa_missing)}")

positive_binary = (
    positive_df.loc[positive_df["Label_Cls"].eq(POSITIVE_LABEL), ["Sample_ID", "Longitude", "Latitude", "Year", "Cluster", "Basin_cluster"]]
    .copy()
)
positive_binary["Label"] = POSITIVE_LABEL
positive_binary["Year"] = pd.to_numeric(positive_binary["Year"], errors="raise").astype(int)
positive_binary["Cluster"] = pd.to_numeric(positive_binary["Cluster"], errors="raise").astype("Int64")
positive_binary["Basin_cluster"] = pd.to_numeric(positive_binary["Basin_cluster"], errors="raise").astype("Int64")

pa_id_col = "Sample_ID" if "Sample_ID" in pa_df.columns else "PA_ID"
pa_binary = pa_df.loc[:, [pa_id_col, "Longitude", "Latitude", "Year"]].copy().rename(columns={pa_id_col: "Sample_ID"})
pa_binary["Year"] = pd.to_numeric(pa_binary["Year"], errors="raise").astype(int)
pa_binary["Label"] = NEGATIVE_LABEL
pa_binary["Cluster"] = 0
pa_binary["Basin_cluster"] = 0

ordered_cols = [
    "Sample_ID",
    "Longitude",
    "Latitude",
    "Year",
    "Label",
    "Cluster",
    "Basin_cluster",
]

positive_binary = positive_binary[ordered_cols].copy()
pa_binary = pa_binary[ordered_cols].copy()

display(positive_binary.head())
display(pa_binary.head())

# ## Merge And QC

total_sample_df = pd.concat([positive_binary, pa_binary], ignore_index=True)
total_sample_df["Year"] = pd.to_numeric(total_sample_df["Year"], errors="raise").astype(int)
total_sample_df["Label"] = pd.to_numeric(total_sample_df["Label"], errors="raise").astype(int)
total_sample_df["Cluster"] = pd.to_numeric(total_sample_df["Cluster"], errors="raise").astype("Int64")
total_sample_df["Basin_cluster"] = pd.to_numeric(total_sample_df["Basin_cluster"], errors="raise").astype("Int64")
total_sample_df = total_sample_df.sort_values(["Year", "Label", "Sample_ID"], ascending=[True, False, True]).reset_index(drop=True)

negative_mask = total_sample_df["Label"].eq(NEGATIVE_LABEL)

year_summary = pd.concat(
    [
        positive_binary.groupby("Year").size().rename("presence_n"),
        pa_binary.groupby("Year").size().rename("pseudo_absence_n"),
        total_sample_df.groupby("Year").size().rename("merged_n"),
    ],
    axis=1,
).fillna(0).astype(int)
year_summary["balanced"] = year_summary["presence_n"].eq(year_summary["pseudo_absence_n"])

qc_summary = {
    "total_rows": int(len(total_sample_df)),
    "presence_rows": int((total_sample_df["Label"] == 1).sum()),
    "pseudo_absence_rows": int((total_sample_df["Label"] == 0).sum()),
    "pseudo_absence_cluster_zero_rows": int(total_sample_df.loc[negative_mask, "Cluster"].eq(0).sum()),
    "pseudo_absence_basin_cluster_zero_rows": int(total_sample_df.loc[negative_mask, "Basin_cluster"].eq(0).sum()),
    "pseudo_absence_cluster_missing_rows": int(total_sample_df.loc[negative_mask, "Cluster"].isna().sum()),
    "pseudo_absence_basin_cluster_missing_rows": int(total_sample_df.loc[negative_mask, "Basin_cluster"].isna().sum()),
    "pseudo_absence_cluster_nonzero_rows": int((total_sample_df.loc[negative_mask, "Cluster"].notna() & ~total_sample_df.loc[negative_mask, "Cluster"].eq(0)).sum()),
    "pseudo_absence_basin_cluster_nonzero_rows": int((total_sample_df.loc[negative_mask, "Basin_cluster"].notna() & ~total_sample_df.loc[negative_mask, "Basin_cluster"].eq(0)).sum()),
    "duplicate_sample_id_rows": int(total_sample_df["Sample_ID"].duplicated().sum()),
    "missing_core_values": int(total_sample_df[["Sample_ID", "Longitude", "Latitude", "Year", "Label"]].isna().any(axis=1).sum()),
}

display(year_summary)
display(pd.Series(qc_summary, name="value"))

if not year_summary["balanced"].all():
    raise AssertionError("Presence and pseudo-absence counts are not balanced for every year.")
if qc_summary["duplicate_sample_id_rows"] != 0:
    raise AssertionError("Duplicate Sample_ID values detected in merged table.")
if qc_summary["missing_core_values"] != 0:
    raise AssertionError("Missing core values detected in merged table.")
if qc_summary["pseudo_absence_cluster_missing_rows"] != 0 or qc_summary["pseudo_absence_basin_cluster_missing_rows"] != 0:
    raise AssertionError("Pseudo-absence rows still contain missing `Cluster` or `Basin_cluster` values.")
if qc_summary["pseudo_absence_cluster_nonzero_rows"] != 0 or qc_summary["pseudo_absence_basin_cluster_nonzero_rows"] != 0:
    raise AssertionError("Pseudo-absence rows must use the legacy `Cluster=0` and `Basin_cluster=0` sentinel values.")
if qc_summary["pseudo_absence_cluster_zero_rows"] != qc_summary["pseudo_absence_rows"]:
    raise AssertionError("Not every pseudo-absence row has `Cluster=0`.")
if qc_summary["pseudo_absence_basin_cluster_zero_rows"] != qc_summary["pseudo_absence_rows"]:
    raise AssertionError("Not every pseudo-absence row has `Basin_cluster=0`.")

total_sample_df.head()

# ## Export

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
total_sample_df.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote: {OUTPUT_PATH}")
print(total_sample_df.head(10).to_string(index=False))

# ## Rank Route

RANK_OUTPUT_PATH = PYTHON_OUTPUT_ROOT / "02_sample_tables" / "RTS_Total_sample_rank.csv"
RANK_LOOKBACK_YEARS = 4
CLIP_FEATURE_YEAR_TO_OBSERVED_RANGE = True
FEATURE_YEAR_MIN = int(total_sample_df["Year"].min())
FEATURE_YEAR_MAX = int(total_sample_df["Year"].max())

print({
    "rank_output_path": str(RANK_OUTPUT_PATH),
    "lookback_years": RANK_LOOKBACK_YEARS,
    "clip_feature_year_to_observed_range": CLIP_FEATURE_YEAR_TO_OBSERVED_RANGE,
    "feature_year_min": FEATURE_YEAR_MIN,
    "feature_year_max": FEATURE_YEAR_MAX,
})

def assign_labels(row: pd.Series) -> tuple[int, int]:
    if row["Label"] == NEGATIVE_LABEL:
        return 0, 0

    ttf = int(row["Occurrence_Year"] - row["Feature_Year"])
    if ttf == 0:
        return 1, 5
    if 1 <= ttf <= RANK_LOOKBACK_YEARS:
        return -1, 5 - ttf
    return 0, 0


rank_input_df = total_sample_df.rename(columns={"Year": "Occurrence_Year"}).copy()
expanded_rows = []

for row in rank_input_df.to_dict(orient="records"):
    if int(row["Label"]) == POSITIVE_LABEL:
        for offset in range(RANK_LOOKBACK_YEARS + 1):
            new_row = row.copy()
            new_row["Feature_Year"] = int(row["Occurrence_Year"]) - offset
            expanded_rows.append(new_row)
    else:
        new_row = row.copy()
        new_row["Feature_Year"] = int(row["Occurrence_Year"])
        expanded_rows.append(new_row)

rank_sample_df = pd.DataFrame(expanded_rows)

if CLIP_FEATURE_YEAR_TO_OBSERVED_RANGE:
    rank_sample_df = rank_sample_df.loc[
        rank_sample_df["Feature_Year"].between(FEATURE_YEAR_MIN, FEATURE_YEAR_MAX)
    ].copy()

rank_sample_df["Label_Cls"], rank_sample_df["Label_Rank"] = zip(*rank_sample_df.apply(assign_labels, axis=1))
rank_sample_df["Occurrence_Year"] = rank_sample_df["Occurrence_Year"].astype(int)
rank_sample_df["Feature_Year"] = rank_sample_df["Feature_Year"].astype(int)
rank_sample_df["Label"] = rank_sample_df["Label"].astype(int)
rank_sample_df["Label_Cls"] = rank_sample_df["Label_Cls"].astype(int)
rank_sample_df["Label_Rank"] = rank_sample_df["Label_Rank"].astype(int)
rank_sample_df["Cluster"] = pd.to_numeric(rank_sample_df["Cluster"], errors="raise").astype("Int64")
rank_sample_df["Basin_cluster"] = pd.to_numeric(rank_sample_df["Basin_cluster"], errors="raise").astype("Int64")

rank_ordered_cols = [
    "Sample_ID",
    "Longitude",
    "Latitude",
    "Occurrence_Year",
    "Feature_Year",
    "Label",
    "Label_Cls",
    "Label_Rank",
    "Cluster",
    "Basin_cluster",
]

rank_sample_df = rank_sample_df[rank_ordered_cols].sort_values(
    ["Feature_Year", "Label_Rank", "Occurrence_Year", "Sample_ID"],
    ascending=[True, False, True, True],
).reset_index(drop=True)

rank_sample_df.head(10)

# ## Rank QC And Export

rank_background_mask = rank_sample_df["Label_Cls"].eq(NEGATIVE_LABEL)

rank_qc_summary = {
    "total_rows": int(len(rank_sample_df)),
    "cls_positive_rows": int((rank_sample_df["Label_Cls"] == 1).sum()),
    "cls_mask_rows": int((rank_sample_df["Label_Cls"] == -1).sum()),
    "cls_background_rows": int((rank_sample_df["Label_Cls"] == 0).sum()),
    "background_cluster_zero_rows": int(rank_sample_df.loc[rank_background_mask, "Cluster"].eq(0).sum()),
    "background_basin_cluster_zero_rows": int(rank_sample_df.loc[rank_background_mask, "Basin_cluster"].eq(0).sum()),
    "background_cluster_missing_rows": int(rank_sample_df.loc[rank_background_mask, "Cluster"].isna().sum()),
    "background_basin_cluster_missing_rows": int(rank_sample_df.loc[rank_background_mask, "Basin_cluster"].isna().sum()),
    "background_cluster_nonzero_rows": int((rank_sample_df.loc[rank_background_mask, "Cluster"].notna() & ~rank_sample_df.loc[rank_background_mask, "Cluster"].eq(0)).sum()),
    "background_basin_cluster_nonzero_rows": int((rank_sample_df.loc[rank_background_mask, "Basin_cluster"].notna() & ~rank_sample_df.loc[rank_background_mask, "Basin_cluster"].eq(0)).sum()),
    "duplicate_sample_feature_year_rows": int(rank_sample_df[["Sample_ID", "Feature_Year"]].duplicated().sum()),
    "missing_core_values": int(rank_sample_df[["Sample_ID", "Longitude", "Latitude", "Occurrence_Year", "Feature_Year", "Label_Cls", "Label_Rank"]].isna().any(axis=1).sum()),
}

rank_label_summary = pd.crosstab(rank_sample_df["Label_Cls"], rank_sample_df["Label_Rank"])
rank_feature_year_summary = pd.crosstab(rank_sample_df["Feature_Year"], rank_sample_df["Label_Cls"])

display(pd.Series(rank_qc_summary, name="value"))
display(rank_label_summary)
display(rank_feature_year_summary)

if rank_qc_summary["duplicate_sample_feature_year_rows"] != 0:
    raise AssertionError("Duplicate (Sample_ID, Feature_Year) pairs detected in rank sample table.")
if rank_qc_summary["missing_core_values"] != 0:
    raise AssertionError("Missing core values detected in rank sample table.")
if rank_qc_summary["background_cluster_missing_rows"] != 0 or rank_qc_summary["background_basin_cluster_missing_rows"] != 0:
    raise AssertionError("Background rank rows still contain missing `Cluster` or `Basin_cluster` values.")
if rank_qc_summary["background_cluster_nonzero_rows"] != 0 or rank_qc_summary["background_basin_cluster_nonzero_rows"] != 0:
    raise AssertionError("Background rank rows must use the legacy `Cluster=0` and `Basin_cluster=0` sentinel values.")
if rank_qc_summary["background_cluster_zero_rows"] != rank_qc_summary["cls_background_rows"]:
    raise AssertionError("Not every rank background row has `Cluster=0`.")
if rank_qc_summary["background_basin_cluster_zero_rows"] != rank_qc_summary["cls_background_rows"]:
    raise AssertionError("Not every rank background row has `Basin_cluster=0`.")

RANK_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
rank_sample_df.to_csv(RANK_OUTPUT_PATH, index=False)
print(f"Wrote: {RANK_OUTPUT_PATH}")
print(rank_sample_df.head(10).to_string(index=False))
