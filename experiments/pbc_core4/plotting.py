from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .load_pbc import LONGITUDINAL_NAMES, TREATMENT_NAME
from .metrics import km_curve


def savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_km_by_method(static_by_method: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    max_time = max((pd.to_numeric(df["time"], errors="coerce").max() for df in static_by_method.values() if not df.empty), default=1.0)
    grid = np.linspace(0, float(max_time), 128)
    for method, df in static_by_method.items():
        if df.empty:
            continue
        surv = km_curve(df["time"], df["event"], grid)
        ax.step(grid, surv, where="post", label=method)
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    savefig(fig, path)


def plot_longitudinal_by_method(long_by_method: dict[str, pd.DataFrame], path: Path, variable: str = "bili") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, df in long_by_method.items():
        if df.empty or variable not in df:
            continue
        g = df.groupby("visit_time")[variable].mean(numeric_only=True).dropna()
        if not g.empty:
            ax.plot(g.index, g.to_numpy(dtype=float), marker="o", label=method)
    ax.set_xlabel("Years")
    ax.set_ylabel(f"Mean {variable}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)
    savefig(fig, path)


def plot_metric_bar(df: pd.DataFrame, metric: str, path: Path, title: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not df.empty and metric in df:
        summary = df.groupby("method")[metric].mean(numeric_only=True).sort_values()
        ax.bar(summary.index.astype(str), summary.to_numpy(dtype=float), color="#4c78a8")
        ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel(metric)
    if title:
        ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, path)


def plot_line(df: pd.DataFrame, x: str, y: str, group: str | None, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not df.empty and x in df and y in df:
        if group and group in df:
            for key, g in df.groupby(group):
                ax.plot(g[x], g[y], marker="o", label=str(key))
            ax.legend(fontsize=7)
        else:
            ax.plot(df[x], df[y], marker="o")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.25)
    savefig(fig, path)


def plot_smd(df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not df.empty and {"comparison", "mean_abs_smd"}.issubset(df.columns):
        ax.bar(df["comparison"].astype(str), pd.to_numeric(df["mean_abs_smd"], errors="coerce"), color="#59a14f")
        ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Mean absolute SMD")
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, path)


def placeholder_pdf(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=14)
    ax.text(0.5, 0.45, message, ha="center", va="center", fontsize=9, wrap=True)
    savefig(fig, path)


def plot_survival_calibration(real_static: pd.DataFrame, pred_static: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not pred_static.empty and "survival_risk_score" in pred_static:
        merged = real_static[["subject_id", "event"]].merge(
            pred_static[["subject_id", "survival_risk_score"]], on="subject_id", how="inner"
        )
        if not merged.empty:
            merged["bin"] = pd.qcut(merged["survival_risk_score"].rank(method="first"), q=min(5, len(merged)), duplicates="drop")
            g = merged.groupby("bin", observed=False).agg(pred=("survival_risk_score", "mean"), obs=("event", "mean")).dropna()
            ax.plot(g["pred"], g["obs"], marker="o", label="risk bins")
            lo, hi = 0.0, max(1.0, float(g[["pred", "obs"]].max().max()))
            ax.plot([lo, hi], [lo, hi], linestyle="--", color="black", label="ideal")
            ax.legend()
    ax.set_xlabel("Predicted event risk")
    ax.set_ylabel("Observed event rate")
    ax.grid(alpha=0.25)
    savefig(fig, path)


def plot_responder_km(real_static: pd.DataFrame, real_long: pd.DataFrame, path: Path, variable: str = "bili") -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    summaries = []
    for sid, g in real_long.sort_values("visit_time").groupby("subject_id"):
        y = pd.to_numeric(g.get(variable, pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
        if len(y) >= 2:
            summaries.append({"subject_id": int(sid), "early_change": float(y[1] - y[0])})
    s = pd.DataFrame(summaries)
    if not s.empty:
        merged = real_static.merge(s, on="subject_id", how="inner")
        threshold = merged["early_change"].median()
        grid = np.linspace(0, pd.to_numeric(merged["time"], errors="coerce").max(), 128)
        for label, sub in [("early_responder", merged[merged["early_change"] <= threshold]), ("non_responder", merged[merged["early_change"] > threshold])]:
            if not sub.empty:
                ax.step(grid, km_curve(sub["time"], sub["event"], grid), where="post", label=label)
        ax.legend()
    ax.set_xlabel("Years")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.25)
    savefig(fig, path)
