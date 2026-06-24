"""
Prepare pbc2 longitudinal data for L-VAE training.

Creates:
  - pbc2_data.csv:  N x D feature matrix (11 longitudinal variables, min-max normalized)
  - pbc2_label.csv: N x Q covariate matrix [visit_time, patient_id, drug, sex]
  - pbc2_mask.csv:  N x D binary mask (1=observed, 0=missing)
  - pbc2_stats.json: normalization statistics for later de-normalization

The data is sorted by (patient_id, visit_time) so that all visits of a patient
are contiguous — required by VaryingLengthSubjectSampler.
"""

import pandas as pd
import numpy as np
import json
import os

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 1. Load raw data ──────────────────────────────────────────────────────────
long_df = pd.read_csv(os.path.join(DATA_DIR, "longitudinal.csv"))
id_df   = pd.read_csv(os.path.join(DATA_DIR, "pbc2_id.csv"))

print(f"Longitudinal records: {len(long_df)}")
print(f"Unique patients:      {long_df['patient_id'].nunique()}")

# ── 2. Map original patient IDs → contiguous 0-based integers ─────────────────
# patient_id in longitudinal.csv is 0-based but may have gaps; re-index
unique_ids = sorted(long_df["patient_id"].unique())
id_map = {old: new for new, old in enumerate(unique_ids)}
long_df["patient_id"] = long_df["patient_id"].map(id_map)

# ── 3. Merge baseline covariates (drug, sex) from pbc2_id.csv ─────────────────
# pbc2_id has 1-based "id"; map to 0-based to match
id_df["patient_id"] = id_df["id"].map(lambda x: id_map.get(x - 1, -1))
id_df = id_df[id_df["patient_id"] >= 0]

# Encode drug: D-penicillamine=0, placebo=1
drug_map = {"D-penicil": 0, "D-penicillamine": 0, "placebo": 1}
id_df["drug_enc"] = id_df["drug"].map(drug_map).fillna(0).astype(int)

# Encode sex: female=0, male=1
sex_map = {"female": 0, "male": 1}
id_df["sex_enc"] = id_df["sex"].map(sex_map).fillna(0).astype(int)

baseline = id_df[["patient_id", "drug_enc", "sex_enc"]].drop_duplicates("patient_id")
long_df = long_df.merge(baseline, on="patient_id", how="left")

# ── 4. Sort by (patient_id, visit_time) — required by L-VAE samplers ──────────
long_df = long_df.sort_values(["patient_id", "visit_time"]).reset_index(drop=True)

# ── 5. Define longitudinal feature columns ─────────────────────────────────────
feature_cols = [
    "ascites", "hepatomegaly", "spiders", "edema",
    "serBilir", "albumin", "alkaline", "SGOT",
    "platelets", "prothrombin", "histologic",
]

# ── 6. Build mask (1=observed, 0=missing) ──────────────────────────────────────
mask_df = long_df[feature_cols].notna().astype(int)

# ── 7. Fill NaN with column median for feature matrix, then min-max normalise ──
features = long_df[feature_cols].copy()
medians = features.median()
features = features.fillna(medians)

feat_min = features.min()
feat_max = features.max()
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0  # avoid div-by-zero for constant cols

features_norm = (features - feat_min) / feat_range

# ── 8. Build label / covariate matrix ──────────────────────────────────────────
# Layout: [visit_time, patient_id, drug, sex]
#   → id_covariate = 1  (column index of patient_id)
#   → sqexp_kernel = [0]  (visit_time → RBF kernel)
#   → bin_kernel = [2, 3]  (drug, sex → binary kernel)
labels = long_df[["visit_time", "patient_id", "drug_enc", "sex_enc"]].copy()
labels.columns = ["visit_time", "patient_id", "drug", "sex"]

# ── 9. Save everything ────────────────────────────────────────────────────────
features_norm.to_csv(os.path.join(DATA_DIR, "pbc2_data.csv"), index=False, header=False)
labels.to_csv(os.path.join(DATA_DIR, "pbc2_label.csv"), index=False, header=False)
mask_df.to_csv(os.path.join(DATA_DIR, "pbc2_mask.csv"), index=False, header=False)

# Save normalization stats for later visualization / de-normalization
stats = {
    "feature_cols": feature_cols,
    "min": feat_min.to_dict(),
    "max": feat_max.to_dict(),
    "median": medians.to_dict(),
    "n_patients": int(long_df["patient_id"].nunique()),
    "n_records": len(long_df),
    "n_features": len(feature_cols),
    "label_cols": ["visit_time", "patient_id", "drug", "sex"],
}
with open(os.path.join(DATA_DIR, "pbc2_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nSaved:")
print(f"  pbc2_data.csv   ({features_norm.shape[0]} x {features_norm.shape[1]})")
print(f"  pbc2_label.csv  ({labels.shape[0]} x {labels.shape[1]})")
print(f"  pbc2_mask.csv   ({mask_df.shape[0]} x {mask_df.shape[1]})")
print(f"  pbc2_stats.json")
print(f"\nP = {stats['n_patients']} patients, D = {stats['n_features']} features")
print(f"Label layout: {stats['label_cols']}  →  id_covariate=1")
