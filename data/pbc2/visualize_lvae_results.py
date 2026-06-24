"""
Visualize L-VAE results on pbc2 data.

Produces:
  1. Training loss curves (total, reconstruction, GP)
  2. Latent space (2D UMAP of 4-dim latent) colored by visit_time and patient
  3. Per-variable reconstruction vs actual for selected patients over time
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── paths ──────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(BASE)
RESULTS = os.path.join(BASE, "results")
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

sys.path.insert(0, PROJ)

# ── 1. Training loss curves ───────────────────────────────────────────────────
print("1. Plotting training loss curves ...")
with open(os.path.join(RESULTS, "diagnostics.pkl"), "rb") as f:
    penalty_arr, net_loss_arr, nll_arr, recon_arr, gp_arr = pickle.load(f)

epochs = np.arange(1, len(net_loss_arr) + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, net_loss_arr, "b-", lw=1.2)
axes[0].set_title("Total Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_yscale("log")

axes[1].plot(epochs, recon_arr, "r-", lw=1.2, label="Recon (MSE)")
axes[1].set_title("Reconstruction Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")

axes[2].plot(epochs, gp_arr, "g-", lw=1.2, label="GP (KLD)")
axes[2].set_title("GP Prior Loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig1_training_loss.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved fig1_training_loss.png")

# ── 2. Latent space visualisation ─────────────────────────────────────────────
print("2. Plotting latent space ...")

# Load normalization stats early (needed for reconstruction too)
with open(os.path.join(BASE, "pbc2_stats.json"), "r") as f:
    stats = json.load(f)
feature_cols = stats["feature_cols"]
feat_min = np.array([stats["min"][c] for c in feature_cols])
feat_max = np.array([stats["max"][c] for c in feature_cols])
feat_range = feat_max - feat_min
feat_range[feat_range == 0] = 1.0

# Load model and encode ALL data to get consistent latent embeddings
from VAE import SimpleVAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_dim = len(feature_cols)

# Infer latent_dim from saved model
state_dict = torch.load(os.path.join(RESULTS, "final-vae_model.pth"), map_location=device)
latent_dim = state_dict["fc211.weight"].shape[0]

nnet_model = SimpleVAE(latent_dim, num_dim, vy_init=1.0, vy_fixed=False).to(device)
nnet_model.load_state_dict(state_dict)
nnet_model.eval()
nnet_model = nnet_model.double()

# Load original data
data_df = pd.read_csv(os.path.join(BASE, "pbc2_data.csv"), header=None).values
label_df = pd.read_csv(os.path.join(BASE, "pbc2_label.csv"), header=None).values
mask_df = pd.read_csv(os.path.join(BASE, "pbc2_mask.csv"), header=None).values

# Encode + reconstruct all data
data_tensor = torch.tensor(data_df, dtype=torch.double).to(device)
with torch.no_grad():
    recon, mu_all, logvar_all = nnet_model(data_tensor)
    mu_np = mu_all.cpu().numpy()
    recon_np = recon.cpu().numpy()

visit_time = label_df[:, 0]
patient_id = label_df[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sc0 = axes[0].scatter(mu_np[:, 0], mu_np[:, 1], c=visit_time, cmap="viridis",
                       s=8, alpha=0.6, edgecolors="none")
axes[0].set_xlabel("Latent dim 1")
axes[0].set_ylabel("Latent dim 2")
axes[0].set_title("Latent Space (colored by visit time)")
plt.colorbar(sc0, ax=axes[0], label="Visit time (years)")

# Color by patient (use modular color for visual variety)
patient_color = patient_id % 20
sc1 = axes[1].scatter(mu_np[:, 0], mu_np[:, 1], c=patient_color, cmap="tab20",
                       s=8, alpha=0.6, edgecolors="none")
axes[1].set_xlabel("Latent dim 1")
axes[1].set_ylabel("Latent dim 2")
axes[1].set_title("Latent Space (colored by patient ID mod 20)")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig2_latent_space.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved fig2_latent_space.png")

# ── 3. Per-patient longitudinal reconstruction ────────────────────────────────
print("3. Plotting per-patient reconstructions ...")

# De-normalise
actual_denorm = data_df * feat_range + feat_min
recon_denorm = recon_np * feat_range + feat_min

# Select a few patients with many visits for clear visualisation
unique_pids = np.unique(label_df[:, 1])
visit_counts = {pid: np.sum(label_df[:, 1] == pid) for pid in unique_pids}
# Pick top 6 patients by number of visits
top_pids = sorted(visit_counts, key=visit_counts.get, reverse=True)[:6]

# Select continuous longitudinal variables for plotting
cont_vars = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]
cont_idx = [feature_cols.index(v) for v in cont_vars]

fig, axes = plt.subplots(len(top_pids), len(cont_vars), figsize=(24, 4 * len(top_pids)),
                          sharex=False)

for i, pid in enumerate(top_pids):
    pid_mask = label_df[:, 1] == pid
    t = label_df[pid_mask, 0]  # visit_time
    sort_idx = np.argsort(t)
    t_sorted = t[sort_idx]

    for j, (var_name, vi) in enumerate(zip(cont_vars, cont_idx)):
        ax = axes[i, j]
        actual_vals = actual_denorm[pid_mask, vi][sort_idx]
        recon_vals = recon_denorm[pid_mask, vi][sort_idx]
        obs_mask = mask_df[pid_mask, vi][sort_idx]

        ax.plot(t_sorted, actual_vals, "ko-", ms=4, lw=1.2, label="Actual")
        ax.plot(t_sorted, recon_vals, "rs--", ms=4, lw=1.0, label="Reconstructed")

        # Mark missing values
        missing = obs_mask == 0
        if missing.any():
            ax.scatter(t_sorted[missing], actual_vals[missing], c="gray", marker="x",
                       s=30, zorder=5, label="Missing (imputed)")

        if i == 0:
            ax.set_title(var_name, fontsize=11, fontweight="bold")
        if j == 0:
            ax.set_ylabel(f"Patient {int(pid)}\n({int(visit_counts[pid])} visits)", fontsize=9)
        if i == len(top_pids) - 1:
            ax.set_xlabel("Visit time (years)")
        if i == 0 and j == 0:
            ax.legend(fontsize=7, loc="upper right")

fig.suptitle("L-VAE Longitudinal Reconstruction: Actual vs Predicted (6 continuous variables)",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig3_reconstruction.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved fig3_reconstruction.png")

# ── 4. Latent trajectories for selected patients ─────────────────────────────
print("4. Plotting latent trajectories ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot trajectories in latent dim 1 vs dim 2
for pid in top_pids:
    pid_mask = label_df[:, 1] == pid
    t = label_df[pid_mask, 0]
    sort_idx = np.argsort(t)
    mu_pid = mu_np[pid_mask][sort_idx]

    axes[0].plot(mu_pid[:, 0], mu_pid[:, 1], "o-", ms=4, lw=1.0, alpha=0.8,
                 label=f"Patient {int(pid)}")
    # Mark start
    axes[0].scatter(mu_pid[0, 0], mu_pid[0, 1], marker="^", s=60, zorder=5, edgecolors="k")

axes[0].set_xlabel("Latent dim 1")
axes[0].set_ylabel("Latent dim 2")
axes[0].set_title("Patient Latent Trajectories (dims 1-2)")
axes[0].legend(fontsize=7, loc="best")

# Plot latent dim 1 over time
for pid in top_pids:
    pid_mask = label_df[:, 1] == pid
    t = label_df[pid_mask, 0]
    sort_idx = np.argsort(t)
    mu_pid = mu_np[pid_mask][sort_idx]
    t_sorted = t[sort_idx]

    axes[1].plot(t_sorted, mu_pid[:, 0], "o-", ms=4, lw=1.0, alpha=0.8,
                 label=f"Patient {int(pid)}")

axes[1].set_xlabel("Visit time (years)")
axes[1].set_ylabel("Latent dim 1")
axes[1].set_title("Latent Dimension 1 Over Time")
axes[1].legend(fontsize=7, loc="best")

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig4_latent_trajectories.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"   Saved fig4_latent_trajectories.png")

print(f"\nAll figures saved to {FIG_DIR}/")
