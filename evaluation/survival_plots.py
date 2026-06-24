from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def event_rate_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float]:
    real_rate = float(np.mean(real_df["censor"].to_numpy(dtype=float) > 0.5))
    syn_rate = float(np.mean(synthetic_df["censor"].to_numpy(dtype=float) > 0.5))
    real_time = real_df["time"].to_numpy(dtype=float)
    syn_time = synthetic_df["time"].to_numpy(dtype=float)
    rx, ry = _km_curve(real_time, real_df["censor"].to_numpy(dtype=float) > 0.5)
    sx, sy = _km_curve(syn_time, synthetic_df["censor"].to_numpy(dtype=float) > 0.5)
    upper = float(np.nanmax([np.nanmax(real_time), np.nanmax(syn_time)])) if len(real_time) and len(syn_time) else 1.0
    grid = np.linspace(0.0, max(upper, 1e-6), 256)
    r_interp = np.interp(grid, rx, ry, left=1.0, right=ry[-1])
    s_interp = np.interp(grid, sx, sy, left=1.0, right=sy[-1])
    km_abs = np.abs(r_interp - s_interp)
    km_area = np.sum(0.5 * (km_abs[1:] + km_abs[:-1]) * np.diff(grid))
    return {
        "real_event_rate": real_rate,
        "synthetic_event_rate": syn_rate,
        "event_rate_diff": abs(real_rate - syn_rate),
        "survival_time_median_diff": abs(float(np.nanmedian(real_time)) - float(np.nanmedian(syn_time))),
        "survival_time_mean_diff": abs(float(np.nanmean(real_time)) - float(np.nanmean(syn_time))),
        "survival_km_integrated_abs_error": float(km_area / max(grid[-1] - grid[0], 1e-8)),
    }


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(times)
    times = times[order]
    events = events[order].astype(bool)
    unique_times = np.unique(times)
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t in unique_times:
        at_risk = np.sum(times >= t)
        n_events = np.sum((times == t) & events)
        if at_risk > 0:
            surv *= 1.0 - n_events / at_risk
        xs.append(float(t))
        ys.append(float(surv))
    return np.asarray(xs), np.asarray(ys)


def plot_survival_curves(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for df, label, color in [(real_df, "Real", "#2f6f9f"), (synthetic_df, "Synthetic", "#c44e52")]:
        x, y = _km_curve(df["time"].to_numpy(dtype=float), df["censor"].to_numpy(dtype=float) > 0.5)
        ax.step(x, y, where="post", label=label, color=color)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Survival curve")
    ax.grid(alpha=0.25)
    ax.legend()
    path = output / "survival_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
