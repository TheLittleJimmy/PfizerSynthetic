from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_figures(cfg: dict[str, Any], output_dir: str | Path) -> list[str]:
    output = Path(output_dir)
    figures = output / "figures"
    paths: list[str] = []
    method_order = [m for m in ["PhaseSyn", "LMM-AFT", "JM-RE", "TVAE", "CTGAN"] if m in cfg.get("methods", {}).get("active", [])]
    palette = {
        "PhaseSyn": "#D55E00",
        "LMM-AFT": "#0072B2",
        "JM-RE": "#009E73",
        "TVAE": "#CC79A7",
        "CTGAN": "#56B4E9",
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    steps = [
        "Nonlinear high-dimensional DGM",
        "Pseudo Phase II data, n=200",
        "Train PhaseSyn and 20260617 benchmarks",
        "Generate virtual Phase III trials, n=300 or 600",
        "Run planned log-rank/Cox survival analysis",
        "Predicted PoS",
        f"Compare with {int(cfg['m_oracle'])}-trial oracle true PoS",
    ]
    y = np.linspace(0.92, 0.08, len(steps))
    for i, (label, yy) in enumerate(zip(steps, y)):
        ax.text(0.5, yy, label, ha="center", va="center", fontsize=11, bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec="#444444", lw=0.8))
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y[i + 1] + 0.045), xytext=(0.5, yy - 0.045), arrowprops=dict(arrowstyle="->", lw=1.2))
    path = figures / "fig1_simulation_schematic.png"
    _save(fig, path)
    paths.append(str(path))

    merged = pd.read_csv(output / "tables" / "pos_estimates_with_oracle.csv")
    acc = pd.read_csv(output / "tables" / "pos_bias_rmse_mae.csv")
    event_censor = pd.read_csv(output / "tables" / "event_censoring_rate_errors.csv")

    cal = merged.groupby(["method", "scenario", "n_phase3"], dropna=False).agg(pos_hat=("pos_hat", "mean"), true_pos=("true_pos", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, g in cal.groupby("method"):
        ax.scatter(g["true_pos"], g["pos_hat"], label=method, s=36, alpha=0.85)
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--")
    ax.set_xlabel("Oracle true PoS")
    ax.set_ylabel("Estimated PoS")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, ncol=2)
    path = figures / "fig2_pos_calibration.png"
    _save(fig, path)
    paths.append(str(path))

    fig, axes = plt.subplots(1, max(1, len(cfg["effect_scenarios"])), figsize=(5 * max(1, len(cfg["effect_scenarios"])), 4), squeeze=False)
    for ax, scenario in zip(axes.ravel(), cfg["effect_scenarios"].keys()):
        g = cal[cal["scenario"].eq(scenario)]
        oracle = g.drop_duplicates(["n_phase3"]).sort_values("n_phase3")
        ax.plot(oracle["n_phase3"], oracle["true_pos"], color="black", marker="o", label="Oracle")
        for method, gm in g.groupby("method"):
            gm = gm.sort_values("n_phase3")
            ax.plot(gm["n_phase3"], gm["pos_hat"], marker="o", alpha=0.8, label=method)
        ax.set_title(scenario)
        ax.set_xlabel("Phase III sample size")
        ax.set_ylabel("PoS")
        ax.set_ylim(-0.03, 1.03)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    path = figures / "fig3_power_curves.png"
    _save(fig, path)
    paths.append(str(path))

    rmse = acc.groupby("method", dropna=False).agg(pos_rmse=("pos_rmse", "mean")).reset_index().sort_values("pos_rmse")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rmse["method"], rmse["pos_rmse"], color="#4C78A8")
    ax.set_ylabel("Mean PoS RMSE")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=25)
    path = figures / "fig4_pos_rmse.png"
    _save(fig, path)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(event_censor))
    width = 0.36
    ax.bar(x - width / 2, event_censor["event_rate_error"], width, label="Event rate")
    ax.bar(x + width / 2, event_censor["censoring_rate_error"], width, label="Censoring rate")
    ax.set_xticks(x)
    ax.set_xticklabels(event_censor["method"], rotation=25, ha="right")
    ax.set_ylabel("Absolute error")
    ax.legend()
    path = figures / "fig5_event_censoring_error.png"
    _save(fig, path)
    paths.append(str(path))

    merged["abs_pos_error"] = merged["pos_error"].abs()
    ordered_methods = [m for m in method_order if m in set(merged["method"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box_data = [merged.loc[merged["method"].eq(method), "abs_pos_error"].to_numpy(dtype=float) for method in ordered_methods]
    bp = ax.boxplot(box_data, labels=ordered_methods, patch_artist=True, showfliers=False)
    for patch, method in zip(bp["boxes"], ordered_methods):
        patch.set_facecolor(palette.get(method, "#999999"))
        patch.set_alpha(0.45 if method != "PhaseSyn" else 0.70)
        patch.set_edgecolor("#333333")
    rng = np.random.default_rng(20260618)
    for i, vals in enumerate(box_data, start=1):
        vals = vals[np.isfinite(vals)]
        if vals.size:
            sample = vals if vals.size <= 220 else rng.choice(vals, size=220, replace=False)
            ax.scatter(rng.normal(i, 0.045, size=sample.size), sample, s=8, color="#222222", alpha=0.22, linewidths=0)
    ax.set_ylabel("Absolute PoS error")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(bottom=0.0)
    path = figures / "fig6_pos_error_distribution.png"
    _save(fig, path)
    paths.append(str(path))

    phase = merged[merged["method"].eq("PhaseSyn")].copy()
    phase["design"] = phase["scenario"].astype(str) + "\nn=" + phase["n_phase3"].astype(str)
    design_order = [
        f"{scenario}\nn={int(n_phase3)}"
        for scenario in cfg["effect_scenarios"].keys()
        for n_phase3 in cfg["n_phase3_grid"]
    ]
    design_order = [design for design in design_order if design in set(phase["design"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    phase_data = [phase.loc[phase["design"].eq(design), "pos_hat"].to_numpy(dtype=float) for design in design_order]
    bp = ax.boxplot(phase_data, labels=design_order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(palette["PhaseSyn"])
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
    for i, design in enumerate(design_order, start=1):
        vals = phase.loc[phase["design"].eq(design), "pos_hat"].to_numpy(dtype=float)
        if vals.size:
            ax.scatter(rng.normal(i, 0.04, size=vals.size), vals, s=12, color=palette["PhaseSyn"], alpha=0.35, linewidths=0)
            truth = float(phase.loc[phase["design"].eq(design), "true_pos"].iloc[0])
            ax.scatter(i, truth, marker="D", s=44, color="black", zorder=5, label="Oracle true PoS" if i == 1 else None)
    ax.set_ylabel("PoS")
    ax.set_xlabel("Scenario and Phase III sample size")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    path = figures / "fig7_phasesyn_design_calibration.png"
    _save(fig, path)
    paths.append(str(path))

    bias_rows = []
    for (scenario, n_phase3), g in phase.groupby(["scenario", "n_phase3"], dropna=False):
        errors = g["pos_error"].to_numpy(dtype=float)
        se = float(np.nanstd(errors, ddof=1) / np.sqrt(len(errors))) if len(errors) > 1 else 0.0
        bias_rows.append({
            "scenario": scenario,
            "n_phase3": int(n_phase3),
            "design": f"{scenario}\nn={int(n_phase3)}",
            "mean_error": float(np.nanmean(errors)),
            "ci95": 1.96 * se,
        })
    bias = pd.DataFrame(bias_rows)
    bias["design"] = pd.Categorical(bias["design"], categories=design_order, ordered=True)
    bias = bias.sort_values("design")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(bias))
    colors = [palette["PhaseSyn"] if val >= 0 else "#0072B2" for val in bias["mean_error"]]
    ax.bar(x, bias["mean_error"], yerr=bias["ci95"], capsize=4, color=colors, alpha=0.75, edgecolor="#333333", linewidth=0.6)
    ax.axhline(0.0, color="black", lw=1.0, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(bias["design"], rotation=0)
    ax.set_ylabel("PhaseSyn PoS bias\n(mean estimated - oracle)")
    ax.set_xlabel("Scenario and Phase III sample size")
    path = figures / "fig8_phasesyn_bias_ci.png"
    _save(fig, path)
    paths.append(str(path))

    decision = pd.read_csv(output / "go_no_go_decision_metrics.csv")
    ranking = pd.read_csv(output / "design_ranking_accuracy.csv")
    efficacy = decision[["method", "decision_accuracy"]].merge(ranking, on="method", how="left")
    efficacy = efficacy.set_index("method").reindex(ordered_methods).reset_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(efficacy))
    width = 0.36
    method_colors = [palette.get(method, "#999999") for method in efficacy["method"]]
    ax.bar(x - width / 2, efficacy["decision_accuracy"], width, color=method_colors, alpha=0.75, label="Go/no-go accuracy")
    ax.bar(x + width / 2, efficacy["ranking_accuracy"], width, color=method_colors, alpha=0.35, hatch="//", label="Design-ranking accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(efficacy["method"], rotation=25)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8, frameon=False)
    path = figures / "fig9_decision_ranking_accuracy.png"
    _save(fig, path)
    paths.append(str(path))

    phase_design = (
        phase.groupby(["scenario", "n_phase3"], dropna=False)
        .agg(
            event_rate_hat=("event_rate_hat", "mean"),
            censoring_rate_hat=("censoring_rate_hat", "mean"),
            true_event_rate=("true_event_rate", "first"),
            true_censoring_rate=("true_censoring_rate", "first"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(phase_design["true_event_rate"], phase_design["event_rate_hat"], s=52, color="#0072B2", label="Event rate")
    ax.scatter(phase_design["true_censoring_rate"], phase_design["censoring_rate_hat"], s=52, marker="s", color="#D55E00", label="Censoring rate")
    for _, row in phase_design.iterrows():
        label = f"{row['scenario']}, n={int(row['n_phase3'])}"
        ax.annotate(label, (row["true_event_rate"], row["event_rate_hat"]), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#333333")
    lo = min(phase_design["true_event_rate"].min(), phase_design["true_censoring_rate"].min(), phase_design["event_rate_hat"].min(), phase_design["censoring_rate_hat"].min()) - 0.03
    hi = max(phase_design["true_event_rate"].max(), phase_design["true_censoring_rate"].max(), phase_design["event_rate_hat"].max(), phase_design["censoring_rate_hat"].max()) + 0.03
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Oracle rate")
    ax.set_ylabel("PhaseSyn estimated rate")
    ax.legend(fontsize=8, frameon=False)
    path = figures / "fig10_phasesyn_event_censoring_calibration.png"
    _save(fig, path)
    paths.append(str(path))
    return paths
