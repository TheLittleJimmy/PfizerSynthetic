from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pdc2.data import LongitudinalPanel


STATIC_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "age"]
CAT_COLS = ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
LONG_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _km_curve_with_ci(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0.5
    ok = np.isfinite(times)
    times = times[ok]
    events = events[ok]
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    greenwood = 0.0
    xs = [0.0]
    ys = [1.0]
    lo = [1.0]
    hi = [1.0]
    for t in np.unique(times):
        at_risk = int(np.sum(times >= t))
        n_events = int(np.sum((times == t) & events))
        if at_risk > 0 and n_events > 0:
            surv *= 1.0 - n_events / at_risk
            if at_risk > n_events:
                greenwood += n_events / (at_risk * (at_risk - n_events))
        se = math.sqrt(max(surv * surv * greenwood, 0.0))
        xs.append(float(t))
        ys.append(float(surv))
        lo.append(float(max(0.0, surv - 1.96 * se)))
        hi.append(float(min(1.0, surv + 1.96 * se)))
    return np.asarray(xs), np.asarray(ys), np.asarray(lo), np.asarray(hi)


def _step_at_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(xs, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ys) - 1)
    return ys[idx]


def _longitudinal_times(panel: LongitudinalPanel) -> np.ndarray:
    times = panel.times.detach().cpu().numpy()
    return times * (panel.time_max - panel.time_min) + panel.time_min


def _longitudinal_index(panel: LongitudinalPanel, name: str) -> int | None:
    for i, spec in enumerate(panel.specs):
        if spec.name == name:
            return i
    return None


def plot_km_curves_replicates(real_df: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rx, ry, rlo, rhi = _km_curve_with_ci(real_df["time"].to_numpy(), real_df["censor"].to_numpy())

    ax = axes[0]
    ax.step(rx, ry, where="post", color="black", linewidth=2.5, label="Real")
    ax.fill_between(rx, rlo, rhi, step="post", color="gray", alpha=0.28)
    for i, syn in enumerate(synth_list[:10]):
        sx, sy, _, _ = _km_curve_with_ci(syn["time"].to_numpy(), syn["censor"].to_numpy())
        ax.step(sx, sy, where="post", alpha=0.45, linewidth=1.0, label=f"Syn {i + 1}")
    ax.set_title("Kaplan-Meier: Event-Free Survival", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.step(rx, ry, where="post", color="black", linewidth=2.5, label="Real")
    ax.fill_between(rx, rlo, rhi, step="post", color="gray", alpha=0.28)
    grid = np.linspace(0.0, float(real_df["time"].max()), 240)
    curves = []
    for syn in synth_list:
        sx, sy, _, _ = _km_curve_with_ci(syn["time"].to_numpy(), syn["censor"].to_numpy())
        curves.append(_step_at_grid(sx, sy, grid))
    arr = np.asarray(curves)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)
    ax.plot(grid, mean, color="#d55e00", linestyle="--", linewidth=2.2, label="Synthetic mean")
    ax.fill_between(grid, mean - sd, mean + sd, color="#d55e00", alpha=0.22)
    ax.set_title("KM: Real vs Synthetic Mean +/- SD", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=10)
    _savefig(fig, path)


def plot_survival_time_replicates(real_df: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(0.0, max([real_df["time"].max(), *[s["time"].max() for s in synth_list]]), 32)
    ax = axes[0]
    ax.hist(real_df["time"], bins=bins, alpha=0.5, density=True, color="#2f6f9f", label="Real")
    for syn in synth_list:
        ax.hist(syn["time"], bins=bins, alpha=0.08, density=True, color="#c44e52")
    ax.hist(synth_list[0]["time"], bins=bins, alpha=0.32, density=True, color="#c44e52", label="Synthetic reps")
    ax.set_title("Survival Time Distribution", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[1]
    real_rate = float(real_df["censor"].mean())
    rates = np.asarray([float(s["censor"].mean()) for s in synth_list])
    ax.bar(["Real", "Synthetic mean"], [real_rate, rates.mean()], color=["#2f6f9f", "#c44e52"], edgecolor="black")
    ax.errorbar([1], [rates.mean()], yerr=[rates.std()], fmt="none", color="black", capsize=5)
    ax.axhline(real_rate, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, max(1.0, real_rate, rates.max()) * 1.15)
    ax.set_ylabel("Event Rate")
    ax.set_title("Event Rate", fontweight="bold")

    ax = axes[2]
    q = np.linspace(0.0, 1.0, len(real_df))
    real_q = np.quantile(real_df["time"].to_numpy(dtype=float), q)
    for syn in synth_list:
        syn_q = np.quantile(syn["time"].to_numpy(dtype=float), q)
        ax.scatter(real_q, syn_q, s=8, alpha=0.14, color="#c44e52")
    lim = float(max(real_q.max(), max(s["time"].max() for s in synth_list)) * 1.05)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0)
    ax.set_xlabel("Real Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_title("Q-Q Plot: Survival Time", fontweight="bold")
    _savefig(fig, path)


def plot_continuous_replicates(real: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    cont_cols = [c for c in STATIC_CONT_COLS if c in real.columns and all(c in s.columns for s in synth_list)]
    if not cont_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cont_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cont_cols):
        ax = axes[i]
        r = real[feat].dropna().to_numpy(dtype=float)
        syn_vals = [s[feat].dropna().to_numpy(dtype=float) for s in synth_list]
        lo = float(min([r.min(), *[v.min() for v in syn_vals]]))
        hi = float(max([r.max(), *[v.max() for v in syn_vals]]))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 35)
        ax.hist(r, bins=bins, density=True, alpha=0.42, color="#2f6f9f", label="Real")
        for vals in syn_vals:
            ax.hist(vals, bins=bins, density=True, alpha=0.06, color="#c44e52")
        ax.hist(syn_vals[0], bins=bins, density=True, alpha=0.28, color="#c44e52", label="Synthetic reps")
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(cont_cols):]:
        ax.set_visible(False)
    fig.suptitle("Continuous Feature Distributions: Replicates", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, path)


def plot_categorical_replicates(real: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    cat_cols = [c for c in CAT_COLS if c in real.columns and all(c in s.columns for s in synth_list)]
    if not cat_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cat_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cat_cols):
        ax = axes[i]
        rc = real[feat].dropna().value_counts(normalize=True).sort_index()
        cats = sorted(set(rc.index) | set().union(*[set(s[feat].dropna().unique()) for s in synth_list]))
        syn_props = []
        for syn in synth_list:
            sc = syn[feat].dropna().value_counts(normalize=True).sort_index()
            syn_props.append([float(sc.get(c, 0.0)) for c in cats])
        syn_arr = np.asarray(syn_props)
        x = np.arange(len(cats))
        ax.bar(x - 0.18, [float(rc.get(c, 0.0)) for c in cats], 0.36, color="#2f6f9f", label="Real")
        ax.bar(x + 0.18, syn_arr.mean(axis=0), 0.36, yerr=syn_arr.std(axis=0), color="#c44e52", alpha=0.78, label="Synthetic mean")
        ax.set_title(feat, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in cats], fontsize=8)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=8)
    for ax in axes[len(cat_cols):]:
        ax.set_visible(False)
    fig.suptitle("Categorical Feature Distributions: Mean +/- SD", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, path)


def plot_longitudinal_replicate_means(panel: LongitudinalPanel, long_reps: list[np.ndarray], output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        xs, real_means, real_ci, syn_mean, syn_ci = [], [], [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            vals = real[:, visit, idx][obs]
            real_means.append(float(np.nanmean(vals)))
            real_ci.append(float(1.96 * np.nanstd(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
            rep_means = [float(np.nanmean(rep[:, visit, idx][obs])) for rep in long_reps]
            syn_mean.append(float(np.mean(rep_means)))
            syn_ci.append(float(1.96 * np.std(rep_means, ddof=1) / math.sqrt(len(rep_means))) if len(rep_means) > 1 else 0.0)
        if not xs:
            continue
        x = np.asarray(xs)
        r = np.asarray(real_means)
        rci = np.asarray(real_ci)
        sm = np.asarray(syn_mean)
        sci = np.asarray(syn_ci)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, r, color="#2f6f9f", marker="o", linewidth=2.0, label="Real mean")
        ax.fill_between(x, r - rci, r + rci, color="#2f6f9f", alpha=0.18, label="Real 95% CI")
        ax.plot(x, sm, color="#c44e52", marker="s", linewidth=2.0, label="Synthetic mean")
        ax.fill_between(x, sm - sci, sm + sci, color="#c44e52", alpha=0.2, label="Synthetic 95% CI")
        ax.set_title(f"Longitudinal Mean + 95% CI Across Replicates: {name}", fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / f"{name}_replicate_mean_95ci.png")


def plot_longitudinal_replicate_medians(panel: LongitudinalPanel, long_reps: list[np.ndarray], output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        xs, real_medians, rep_medians = [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            real_medians.append(float(np.nanmedian(real[:, visit, idx][obs])))
            rep_medians.append([float(np.nanmedian(rep[:, visit, idx][obs])) for rep in long_reps])
        if not xs:
            continue
        x = np.asarray(xs)
        reps = np.asarray(rep_medians)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, real_medians, color="black", linewidth=2.5, label="Real")
        for r in range(reps.shape[1]):
            ax.plot(x, reps[:, r], linewidth=1.0, alpha=0.28)
        ax.plot(x, reps.mean(axis=1), color="#c44e52", linestyle="--", linewidth=2.0, label="Synthetic mean")
        ax.set_title(f"Median Trajectory Replicates: {name}", fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / f"median_replicates_{name}.png")
