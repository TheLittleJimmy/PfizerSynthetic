from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pdc2.data import LongitudinalPanel


def plot_median_trajectories(panel: LongitudinalPanel, synthetic_raw: np.ndarray, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = panel.times.detach().cpu().numpy()
    saved: list[Path] = []
    for idx in panel.continuous_indices:
        spec = panel.specs[idx]
        x_vals, real_med, syn_med = [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx]) & np.isfinite(synthetic_raw[:, visit, idx])
            if not obs.any():
                continue
            x_vals.append(float(np.median(times[:, visit][obs])))
            real_med.append(float(np.median(real[:, visit, idx][obs])))
            syn_med.append(float(np.median(synthetic_raw[:, visit, idx][obs])))
        if not x_vals:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x_vals, real_med, marker="o", label="Real", color="#2f6f9f")
        ax.plot(x_vals, syn_med, marker="s", label="Synthetic", color="#c44e52")
        ax.set_title(f"{spec.name} median trajectory")
        ax.set_xlabel("Normalized visit time")
        ax.set_ylabel(spec.name)
        ax.grid(alpha=0.25)
        ax.legend()
        path = output / f"{spec.name}_median_trajectory.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_categorical_frequencies(panel: LongitudinalPanel, synthetic_raw: np.ndarray, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    saved: list[Path] = []
    for idx in panel.categorical_indices:
        spec = panel.specs[idx]
        obs = mask[:, :, idx] & np.isfinite(real[:, :, idx]) & np.isfinite(synthetic_raw[:, :, idx])
        if not obs.any():
            continue
        nclass = int(spec.nclass or 2)
        real_counts = np.bincount(real[:, :, idx][obs].astype(int), minlength=nclass) / obs.sum()
        syn_counts = np.bincount(np.rint(synthetic_raw[:, :, idx][obs]).astype(int).clip(0, nclass - 1), minlength=nclass) / obs.sum()
        x = np.arange(nclass)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(x - 0.18, real_counts, width=0.36, label="Real", color="#2f6f9f")
        ax.bar(x + 0.18, syn_counts, width=0.36, label="Synthetic", color="#c44e52")
        ax.set_title(f"{spec.name} visit frequencies")
        ax.set_xlabel("Category")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x)
        ax.legend()
        path = output / f"{spec.name}_category_frequency.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_observed_vs_reconstructed(panel: LongitudinalPanel, synthetic_raw: np.ndarray, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    saved: list[Path] = []
    for idx in panel.continuous_indices:
        spec = panel.specs[idx]
        obs = mask[:, :, idx] & np.isfinite(real[:, :, idx]) & np.isfinite(synthetic_raw[:, :, idx])
        if not obs.any():
            continue
        x = real[:, :, idx][obs]
        y = synthetic_raw[:, :, idx][obs]
        lo = float(min(np.min(x), np.min(y)))
        hi = float(max(np.max(x), np.max(y)))
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.scatter(x, y, s=14, alpha=0.65, color="#3b6ea8", edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#111111", linewidth=1.0, linestyle="--")
        ax.set_title(f"{spec.name} observed vs reconstructed")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Reconstructed")
        ax.grid(alpha=0.2)
        path = output / f"{spec.name}_observed_vs_reconstructed.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(path)
    return saved
