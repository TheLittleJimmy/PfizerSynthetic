from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .data import LongitudinalPanel, load_pdc2_bundle, select_overfit_indices, subset_bundle
from .models import longitudinal_observed_rows

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is optional for plot annotations.
    stats = None


COLS = [
    "time",
    "censor",
    "drug",
    "sex",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "histologic",
    "serBilir",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
]
STATIC_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "age"]
LONG_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]
CAT_COLS = ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
ALL_COVARIATES = CAT_COLS + STATIC_CONT_COLS
CAT_LABELS = {
    "drug": {0: "D-penicil", 1: "placebo"},
    "sex": {0: "female", 1: "male"},
    "ascites": {0: "No", 1: "Yes"},
    "hepatomegaly": {0: "No", 1: "Yes"},
    "spiders": {0: "No", 1: "Yes"},
    "edema": {0: "None", 1: "no diuret", 2: "diuretics"},
    "histologic": {0: "1", 1: "2", 2: "3", 3: "4"},
}
REAL_COLOR = "#2f6f9f"
SYN_COLOR = "#c44e52"
BASELINE_STYLE_DIRS = [
    "covariate",
    "survival",
    "trajectories",
    "single_replicate_subject_95ci",
    "visit_correlation",
    "variable_correlation",
]


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.loc[:, [c for c in cols if c in df.columns]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in CAT_COLS:
        if col in out:
            out[col] = np.rint(out[col]).astype("Int64")
    return out


def _empty_corr(cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols, dtype=float)


def _corr(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame()
    if len(df) < 2:
        return _empty_corr(cols)
    mat = df[cols].corr().reindex(index=cols, columns=cols)
    values = mat.to_numpy(dtype=float)
    values[~np.isfinite(values)] = 0.0
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=cols, columns=cols)


def _heatmap(ax: plt.Axes, mat: pd.DataFrame, title: str, cmap: str, vmin: float = -1.0, vmax: float = 1.0) -> None:
    values = mat.to_numpy(dtype=float)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color="white" if abs(val) > 0.65 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0.5
    ok = np.isfinite(times)
    times = times[ok]
    events = events[ok]
    if times.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t in np.unique(times):
        at_risk = np.sum(times >= t)
        n_events = np.sum((times == t) & events)
        if at_risk > 0:
            surv *= 1.0 - n_events / at_risk
        xs.append(float(t))
        ys.append(float(surv))
    return np.asarray(xs), np.asarray(ys)


def plot_training_loss(curves: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    x = curves["epoch"] if "epoch" in curves else np.arange(1, len(curves) + 1)
    for col, label, color in [
        ("loss", "total", "#315f9d"),
        ("hivae_loss", "HI-VAE", "#7f7f7f"),
        ("longitudinal_loss", "longitudinal", "#c44e52"),
    ]:
        if col in curves and np.isfinite(curves[col]).any() and not np.allclose(curves[col].fillna(0), 0):
            ax.plot(x, curves[col], label=label, linewidth=1.4, color=color)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    _savefig(fig, output_dir / "training_loss.png")
    if "loss" in curves:
        np.save(output_dir / "train_loss.npy", curves["loss"].to_numpy(dtype=float))
        np.save(output_dir / "val_loss.npy", np.full(len(curves), np.nan, dtype=float))


def plot_continuous_distributions(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    cont_cols = [c for c in STATIC_CONT_COLS if c in real.columns and c in syn.columns]
    if not cont_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cont_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cont_cols):
        ax = axes[i]
        r = real[feat].dropna().to_numpy(dtype=float)
        s = syn[feat].dropna().to_numpy(dtype=float)
        if r.size == 0 or s.size == 0:
            ax.set_visible(False)
            continue
        lo, hi = float(min(r.min(), s.min())), float(max(r.max(), s.max()))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 30)
        ax.hist(r, bins=bins, density=True, alpha=0.45, color=REAL_COLOR, label="Real")
        ax.hist(s, bins=bins, density=True, alpha=0.45, color=SYN_COLOR, label="Synthetic")
        if stats is not None and r.size > 2 and s.size > 2:
            try:
                xs = np.linspace(lo, hi, 200)
                ax.plot(xs, stats.gaussian_kde(r)(xs), color=REAL_COLOR, linewidth=1.2)
                ax.plot(xs, stats.gaussian_kde(s)(xs), color=SYN_COLOR, linewidth=1.2)
                ks, p = stats.ks_2samp(r, s)
                ax.text(0.98, 0.95, f"KS={ks:.3f}\np={p:.1e}", transform=ax.transAxes, ha="right", va="top", fontsize=7)
            except Exception:
                pass
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(cont_cols):]:
        ax.set_visible(False)
    fig.suptitle("Continuous Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, output_dir / "covariate" / "continuous_distributions.png")


def plot_categorical_distributions(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    cat_cols = [c for c in CAT_COLS if c in real.columns and c in syn.columns]
    if not cat_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cat_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cat_cols):
        ax = axes[i]
        rc = real[feat].dropna().astype(int).value_counts(normalize=True).sort_index()
        sc = syn[feat].dropna().astype(int).value_counts(normalize=True).sort_index()
        cats = sorted(set(rc.index) | set(sc.index))
        x = np.arange(len(cats))
        labels = CAT_LABELS.get(feat, {})
        ax.bar(x - 0.18, [rc.get(c, 0.0) for c in cats], 0.36, color=REAL_COLOR, alpha=0.75, label="Real")
        ax.bar(x + 0.18, [sc.get(c, 0.0) for c in cats], 0.36, color=SYN_COLOR, alpha=0.75, label="Synthetic")
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(int(c), str(int(c))) for c in cats], fontsize=8)
        ax.set_ylabel("Proportion")
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(cat_cols):]:
        ax.set_visible(False)
    fig.suptitle("Categorical Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, output_dir / "covariate" / "categorical_distributions.png")


def plot_correlation_matrices(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    corr_cols = [c for c in ALL_COVARIATES if c in real.columns and c in syn.columns]
    if not corr_cols:
        return
    corr_r = _corr(real, corr_cols)
    corr_s = _corr(syn, corr_cols)
    diff = corr_r - corr_s
    fig, axes = plt.subplots(1, 3, figsize=(23, 7))
    _heatmap(axes[0], corr_r, "Real", "RdBu_r")
    _heatmap(axes[1], corr_s, "Synthetic", "RdBu_r")
    _heatmap(axes[2], diff, "Diff (Real - Synthetic)", "PiYG")
    fig.suptitle("Correlation Matrices", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, output_dir / "covariate" / "correlation_matrices.png")


def plot_qq(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    cont_cols = [c for c in STATIC_CONT_COLS if c in real.columns and c in syn.columns]
    if not cont_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cont_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cont_cols):
        ax = axes[i]
        r = real[feat].dropna().to_numpy(dtype=float)
        s = syn[feat].dropna().to_numpy(dtype=float)
        if r.size == 0 or s.size == 0:
            ax.set_visible(False)
            continue
        q = np.linspace(0, 1, min(r.size, s.size, 500))
        rq = np.quantile(np.sort(r), q)
        sq = np.quantile(np.sort(s), q)
        ax.scatter(rq, sq, s=12, alpha=0.6, color="#6a4c93")
        lo, hi = float(min(rq.min(), sq.min())), float(max(rq.max(), sq.max()))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9)
        ax.set_xlabel("Real quantiles")
        ax.set_ylabel("Synthetic quantiles")
        ax.set_title(feat, fontweight="bold")
    for ax in axes[len(cont_cols):]:
        ax.set_visible(False)
    fig.suptitle("Q-Q Plots", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, output_dir / "covariate" / "qq_plots.png")


def plot_summary_statistics(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for feat in [c for c in COLS if c in real.columns and c in syn.columns]:
        r = real[feat].dropna()
        s = syn[feat].dropna()
        rows.append([
            feat,
            f"{r.mean():.3f}",
            f"{s.mean():.3f}",
            f"{r.std():.3f}",
            f"{s.std():.3f}",
            f"{r.median():.3f}",
            f"{s.median():.3f}",
            f"{r.min():.3f}",
            f"{s.min():.3f}",
            f"{r.max():.3f}",
            f"{s.max():.3f}",
        ])
    labels = ["Feature", "Mean(R)", "Mean(S)", "Std(R)", "Std(S)", "Med(R)", "Med(S)", "Min(R)", "Min(S)", "Max(R)", "Max(S)"]
    fig, ax = plt.subplots(figsize=(18, 0.42 * len(rows) + 2))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    for j in range(len(labels)):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(len(labels)):
            table[i, j].set_facecolor(bg)
    ax.set_title("Summary Statistics", fontweight="bold", fontsize=13, pad=12)
    _savefig(fig, output_dir / "covariate" / "summary_statistics.png")


def plot_survival(real: pd.DataFrame, syn: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rx, ry = _km_curve(real["time"].to_numpy(dtype=float), real["censor"].to_numpy(dtype=float))
    sx, sy = _km_curve(syn["time"].to_numpy(dtype=float), syn["censor"].to_numpy(dtype=float))
    axes[0].step(rx, ry, where="post", color="black", linewidth=2.2, label="Real")
    axes[0].step(sx, sy, where="post", color=SYN_COLOR, linewidth=1.8, label="Synthetic")
    axes[0].set_title("Kaplan-Meier: Event-Free Survival", fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Survival Probability")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    grid = np.linspace(0, max(real["time"].max(), syn["time"].max()), 200)
    r_interp = np.interp(grid, rx, ry, left=1.0, right=ry[-1])
    s_interp = np.interp(grid, sx, sy, left=1.0, right=sy[-1])
    axes[1].plot(grid, r_interp, color=REAL_COLOR, linewidth=2.0, label="Real")
    axes[1].plot(grid, s_interp, color=SYN_COLOR, linewidth=2.0, label="Synthetic")
    axes[1].fill_between(grid, np.minimum(r_interp, s_interp), np.maximum(r_interp, s_interp), color="#aaaaaa", alpha=0.18, label="Gap")
    axes[1].set_title("KM: Real vs Synthetic", fontweight="bold")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Survival Probability")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    _savefig(fig, output_dir / "survival" / "km_curves.png")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(real["time"], bins=20, alpha=0.5, color=REAL_COLOR, density=True, label="Real")
    axes[0].hist(syn["time"], bins=20, alpha=0.5, color=SYN_COLOR, density=True, label="Synthetic")
    axes[0].set_title("Survival Time Distribution", fontweight="bold")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    rates = [real["censor"].mean(), syn["censor"].mean()]
    axes[1].bar(["Real", "Synthetic"], rates, color=[REAL_COLOR, SYN_COLOR], edgecolor="black", linewidth=0.5)
    axes[1].axhline(rates[0], color="black", linestyle="--", linewidth=1.2)
    axes[1].set_ylim(0, max(1.0, max(rates) * 1.2))
    axes[1].set_ylabel("Event Rate")
    axes[1].set_title("Event Rate", fontweight="bold")
    r = np.sort(real["time"].dropna().to_numpy(dtype=float))
    s = np.sort(syn["time"].dropna().to_numpy(dtype=float))
    q = np.linspace(0, 1, min(len(r), len(s)))
    rq = np.quantile(r, q)
    sq = np.quantile(s, q)
    lim = float(max(rq.max(), sq.max()) * 1.05) if rq.size else 1.0
    axes[2].scatter(rq, sq, s=18, alpha=0.7, color=SYN_COLOR)
    axes[2].plot([0, lim], [0, lim], "k--", linewidth=1.0)
    axes[2].set_xlabel("Real Quantiles")
    axes[2].set_ylabel("Synthetic Quantiles")
    axes[2].set_title("Q-Q Plot: Survival Time", fontweight="bold")
    _savefig(fig, output_dir / "survival" / "survival_time_dist.png")

    corr_cols = [c for c in ["time"] + ALL_COVARIATES if c in real.columns and c in syn.columns]
    corr_r = _corr(real, corr_cols)
    corr_s = _corr(syn, corr_cols)
    diff = corr_r - corr_s
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    _heatmap(axes[0], corr_r, "Real Data Correlation", "RdBu_r")
    _heatmap(axes[1], corr_s, "Synthetic Data Correlation", "RdBu_r")
    _heatmap(axes[2], diff, "Difference (Real - Synthetic)", "RdBu_r", vmin=-0.5, vmax=0.5)
    fig.suptitle("Pairwise Correlation: Time + All Covariates", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, output_dir / "survival" / "correlation_heatmap.png")


def _longitudinal_times(panel: LongitudinalPanel) -> np.ndarray:
    times = panel.times.detach().cpu().numpy()
    return times * (panel.time_max - panel.time_min) + panel.time_min


def _longitudinal_from_csv(path: Path, panel: LongitudinalPanel) -> np.ndarray:
    out = np.full(panel.raw_values.shape, np.nan, dtype=float)
    syn = pd.read_csv(path)
    subject_to_i = {int(subject_id): i for i, subject_id in enumerate(panel.subject_ids)}
    spec_names = [s.name for s in panel.specs]
    for _, row in syn.iterrows():
        subject = int(row["patient_id"])
        if subject not in subject_to_i:
            continue
        i = subject_to_i[subject]
        visit = int(row["visit_index"])
        if visit < 0 or visit >= out.shape[1]:
            continue
        for j, name in enumerate(spec_names):
            if name in row and pd.notna(row[name]):
                out[i, visit, j] = float(row[name])
    return out


def _longitudinal_index(panel: LongitudinalPanel, name: str) -> int | None:
    for i, spec in enumerate(panel.specs):
        if spec.name == name:
            return i
    return None


def _ci(values: np.ndarray) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, 0.0
    se = float(np.nanstd(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0
    return float(np.nanmean(values)), 1.96 * se


def plot_trajectories(panel: LongitudinalPanel, synthetic_long: np.ndarray, output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    observed_rows = longitudinal_observed_rows(panel.masks).detach().cpu().numpy().astype(bool)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        obs = mask[:, :, idx] & np.isfinite(real[:, :, idx])
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
        axes[0].scatter(times[obs], real[:, :, idx][obs], s=18, alpha=0.2, color=REAL_COLOR, label="All observations")
        for subj in range(real.shape[0]):
            sub_obs = obs[subj]
            if sub_obs.sum() >= 2:
                axes[0].plot(times[subj, sub_obs], real[subj, sub_obs, idx], color=REAL_COLOR, alpha=0.14, linewidth=1.0)
        syn_obs = obs & observed_rows & np.isfinite(synthetic_long[:, :, idx])
        for subj in range(synthetic_long.shape[0]):
            sub_obs = syn_obs[subj]
            if sub_obs.sum() >= 2:
                axes[1].plot(times[subj, sub_obs], synthetic_long[subj, sub_obs, idx], alpha=0.65, linewidth=1.1)
            elif sub_obs.any():
                axes[1].scatter(times[subj, sub_obs], synthetic_long[subj, sub_obs, idx], s=18, alpha=0.65)
        axes[0].set_title(f"Real: {name}", fontweight="bold")
        axes[1].set_title(f"Synthetic: {name}", fontweight="bold")
        for ax in axes:
            ax.set_xlabel("Time (years)")
            ax.set_ylabel(name)
            ax.grid(alpha=0.28)
        axes[0].legend(loc="best")
        fig.suptitle(f"Trajectory Comparison: {name}", fontsize=15, fontweight="bold", y=1.02)
        _savefig(fig, output_dir / "trajectories" / f"traj_{name}.png")


def plot_mean_ci(panel: LongitudinalPanel, synthetic_long: np.ndarray, output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        xs, r_mean, r_ci, s_mean, s_ci = [], [], [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            rm, rc = _ci(real[:, visit, idx][obs])
            sm, sc = _ci(synthetic_long[:, visit, idx][obs])
            r_mean.append(rm)
            r_ci.append(rc)
            s_mean.append(sm)
            s_ci.append(sc)
        if not xs:
            continue
        x = np.asarray(xs)
        r_mean_arr = np.asarray(r_mean)
        r_ci_arr = np.asarray(r_ci)
        s_mean_arr = np.asarray(s_mean)
        s_ci_arr = np.asarray(s_ci)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, r_mean_arr, marker="o", color=REAL_COLOR, linewidth=2.0, label="Real mean")
        ax.fill_between(x, r_mean_arr - r_ci_arr, r_mean_arr + r_ci_arr, color=REAL_COLOR, alpha=0.18, label="Real 95% CI")
        ax.plot(x, s_mean_arr, marker="s", color=SYN_COLOR, linewidth=2.0, label="Synthetic mean")
        ax.fill_between(x, s_mean_arr - s_ci_arr, s_mean_arr + s_ci_arr, color=SYN_COLOR, alpha=0.15, label="Synthetic 95% CI")
        ax.set_title(f"Single Synthetic Replicate Subject-Level 95% CI: {name}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / "single_replicate_subject_95ci" / f"{name}_single_replicate_subject_95ci.png")


def _array_corr(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(values, columns=names)
    return _corr(df, names)


def plot_visit_correlations(panel: LongitudinalPanel, synthetic_long: np.ndarray, output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    visit_count = real.shape[1]
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        real_wide = np.full((real.shape[0], visit_count), np.nan, dtype=float)
        syn_wide = np.full((real.shape[0], visit_count), np.nan, dtype=float)
        for visit in range(visit_count):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            real_wide[obs, visit] = real[:, visit, idx][obs]
            syn_wide[obs, visit] = synthetic_long[:, visit, idx][obs]
        labels = [f"v{v + 1}" for v in range(visit_count)]
        corr_r = _array_corr(real_wide, labels)
        corr_s = _array_corr(syn_wide, labels)
        diff = corr_r - corr_s
        fig, axes = plt.subplots(1, 3, figsize=(22, 6))
        _heatmap(axes[0], corr_r, "Real", "RdBu_r")
        _heatmap(axes[1], corr_s, "Synthetic", "RdBu_r")
        _heatmap(axes[2], diff, "Diff", "PiYG")
        fig.suptitle(f"Visit Correlation: {name}", fontsize=14, fontweight="bold", y=1.02)
        _savefig(fig, output_dir / "visit_correlation" / f"visit_corr_{name}.png")


def plot_variable_corr_per_visit(panel: LongitudinalPanel, synthetic_long: np.ndarray, output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    var_indices = [_longitudinal_index(panel, name) for name in LONG_CONT_COLS]
    if any(idx is None for idx in var_indices):
        return
    var_indices = [int(idx) for idx in var_indices]
    candidates = [0, 2, 4, 6, 8, 10]
    visits = [v for v in candidates if v < real.shape[1]]
    if not visits:
        visits = list(range(min(4, real.shape[1])))
    visits = visits[:4]
    fig, axes = plt.subplots(len(visits), 3, figsize=(22, 6 * len(visits)))
    axes = np.atleast_2d(axes)
    for row, visit in enumerate(visits):
        real_vals = np.full((real.shape[0], len(var_indices)), np.nan, dtype=float)
        syn_vals = np.full((real.shape[0], len(var_indices)), np.nan, dtype=float)
        for col_idx, var_idx in enumerate(var_indices):
            obs = mask[:, visit, var_idx] & np.isfinite(real[:, visit, var_idx])
            real_vals[obs, col_idx] = real[:, visit, var_idx][obs]
            syn_vals[obs, col_idx] = synthetic_long[:, visit, var_idx][obs]
        corr_r = _array_corr(real_vals, LONG_CONT_COLS)
        corr_s = _array_corr(syn_vals, LONG_CONT_COLS)
        diff = corr_r - corr_s
        _heatmap(axes[row, 0], corr_r, f"Real (visit {visit + 1})", "RdBu_r")
        _heatmap(axes[row, 1], corr_s, f"Synthetic (visit {visit + 1})", "RdBu_r")
        _heatmap(axes[row, 2], diff, f"Diff (visit {visit + 1})", "PiYG")
    fig.suptitle("Between-Variable Correlation per Visit", fontsize=15, fontweight="bold", y=1.01)
    _savefig(fig, output_dir / "variable_correlation" / "variable_corr_per_visit.png")


def write_support_artifacts(result_dir: Path, output_dir: Path, panel: LongitudinalPanel, synthetic_long: np.ndarray, synthetic_df: pd.DataFrame) -> None:
    synthetic_df.to_csv(output_dir / "synthetic_sample.csv", index=False)
    if (result_dir / "metrics.json").exists():
        shutil.copy2(result_dir / "metrics.json", output_dir / "longitudinal_metrics.json")
    params = {
        spec.name: {"type": spec.type, "mean": spec.mean, "std": spec.std, "nclass": spec.nclass}
        for spec in panel.specs
    }
    with open(output_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    np.savez_compressed(
        output_dir / "longitudinal_generated.npz",
        real=panel.raw_values,
        synthetic=synthetic_long,
        mask=panel.masks.detach().cpu().numpy(),
        times=_longitudinal_times(panel),
        variables=np.asarray([spec.name for spec in panel.specs]),
    )


def plot_result_dir(result_dir: Path, output_dir: Path | None = None) -> int:
    config_path = result_dir / "config.yaml"
    synthetic_path = result_dir / "synthetic_samples.csv"
    synthetic_long_path = result_dir / "synthetic_longitudinal_samples.csv"
    curves_path = result_dir / "train_curves.csv"
    if not config_path.exists() or not synthetic_path.exists() or not synthetic_long_path.exists():
        return 0

    cfg = load_config(config_path)
    full_bundle = load_pdc2_bundle(cfg)
    subset_size = int(cfg.get("overfit", {}).get("subset_size", cfg.get("training", {}).get("subset_size", 32)))
    seed = int(cfg.get("overfit", {}).get("seed", cfg.get("training", {}).get("seed", 1)))
    indices = select_overfit_indices(full_bundle, subset_size=subset_size, seed=seed)
    bundle = subset_bundle(full_bundle, indices)

    output = output_dir or result_dir
    for subdir in BASELINE_STYLE_DIRS:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    real_df = _numeric(bundle.raw_df, COLS)
    synthetic_df = _numeric(pd.read_csv(synthetic_path), COLS)
    synthetic_long = _longitudinal_from_csv(synthetic_long_path, bundle.longitudinal)

    if curves_path.exists():
        plot_training_loss(pd.read_csv(curves_path), output)
    plot_continuous_distributions(real_df, synthetic_df, output)
    plot_categorical_distributions(real_df, synthetic_df, output)
    plot_correlation_matrices(real_df, synthetic_df, output)
    plot_qq(real_df, synthetic_df, output)
    plot_summary_statistics(real_df, synthetic_df, output)
    plot_survival(real_df, synthetic_df, output)
    plot_trajectories(bundle.longitudinal, synthetic_long, output)
    plot_mean_ci(bundle.longitudinal, synthetic_long, output)
    plot_visit_correlations(bundle.longitudinal, synthetic_long, output)
    plot_variable_corr_per_visit(bundle.longitudinal, synthetic_long, output)
    write_support_artifacts(result_dir, output, bundle.longitudinal, synthetic_long, synthetic_df)
    generated_png = [output / "training_loss.png"]
    for subdir in BASELINE_STYLE_DIRS:
        generated_png.extend((output / subdir).glob("*.png"))
    return sum(1 for path in generated_png if path.exists())


def discover_overfit_dirs(root: Path) -> list[Path]:
    dirs = []
    for metrics in sorted(root.glob("model*/overfit/*/metrics.json")):
        result_dir = metrics.parent
        if (result_dir / "config.yaml").exists():
            dirs.append(result_dir)
    return dirs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create diagnostic figures for saved PDC2 overfit results without retraining.")
    parser.add_argument("--root", default="outputs/pdc2", help="PDC2 output root containing model*/overfit/* directories.")
    parser.add_argument("--result-dir", action="append", default=None, help="Specific overfit result directory. Can be repeated.")
    parser.add_argument("--output-subdir", default=None, help="Optional subdirectory under each result; default writes into the result directory root.")
    parser.add_argument("--summary", default="outputs/pdc2/reports/overfit_baseline_style_figures_summary.md")
    args = parser.parse_args(argv)

    root = Path(args.root)
    result_dirs = [Path(p) for p in args.result_dir] if args.result_dir else discover_overfit_dirs(root)
    rows = []
    for result_dir in result_dirs:
        output = result_dir / args.output_subdir if args.output_subdir else result_dir
        n_png = plot_result_dir(result_dir, output_dir=output)
        rows.append((result_dir, output, n_png))
        print(f"{result_dir}: wrote {n_png} png files to {output}")

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("# PDC2 Overfit Baseline-Style Figures\n\n")
        f.write("Generated from saved overfit CSV artifacts only; no model retraining or checkpoint evaluation was run.\n\n")
        f.write("| Result | Figure root | PNG files |\n")
        f.write("|---|---|---:|\n")
        for result_dir, output, n_png in rows:
            f.write(f"| `{result_dir}` | `{output}` | {n_png} |\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
