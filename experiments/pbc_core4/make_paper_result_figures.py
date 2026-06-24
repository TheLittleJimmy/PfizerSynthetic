from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "PhaseSyn": "PhaseSyn",
    "empirical_subject_bootstrap": "Bootstrap",
    "LMM-AFT": "LMM-AFT",
    "JM-RE": "JM-RE",
    "classical_lmm_cox_aft_simulator": "LMM-AFT",
    "joint_longitudinal_survival_baseline": "JM-RE",
    "TVAE": "TVAE",
    "CTGAN": "CTGAN",
    "modular_deep_generator": "TVAE (old name)",
}

COLORS = {
    "PhaseSyn": "#D55E00",
    "Bootstrap": "#4D4D4D",
    "LMM-AFT": "#0072B2",
    "JM-RE": "#009E73",
    "TVAE": "#CC79A7",
    "TVAE (old name)": "#CC79A7",
    "CTGAN": "#E69F00",
    "Reference": "#999999",
}


def _setup() -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _read(tables: Path, name: str) -> pd.DataFrame:
    path = tables / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _method_order(df: pd.DataFrame) -> list[str]:
    preferred = [
        "PhaseSyn",
        "empirical_subject_bootstrap",
        "TVAE",
        "CTGAN",
        "JM-RE",
        "LMM-AFT",
        "joint_longitudinal_survival_baseline",
        "classical_lmm_cox_aft_simulator",
        "modular_deep_generator",
    ]
    return [m for m in preferred if m in set(df["method"])]


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def figure_control_arm(tables: Path, out: Path) -> None:
    primary = _read(tables, "exp1_table_primary_fidelity_utility_privacy.csv")
    wins = _read(tables, "exp1_table_pairwise_phaseSyn_wins.csv")
    order = _method_order(primary)
    x = np.arange(len(order))
    labels = [_label(m) for m in order]
    colors = [COLORS.get(_label(m), "#666666") for m in order]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.2))
    ax = axes[0, 0]
    vals = primary.set_index("method").loc[order, "longitudinal_trajectory_error"].astype(float)
    ax.bar(x, vals, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Trajectory error (log)")
    ax.set_title("A. Longitudinal fidelity")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[0, 1]
    vals = primary.set_index("method").loc[order, "km_iae"].astype(float)
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("KM integrated error")
    ax.set_title("B. Survival curve fidelity")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    vals = primary.set_index("method").loc[order, "cox_hr_mean"].astype(float)
    ax.scatter(x, vals, s=45, color=colors, zorder=3)
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Mean Cox HR")
    ax.set_title("C. Trial estimand preservation")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    wins = wins.copy()
    wins["label"] = wins["baseline_method"].map(_label)
    wins = wins.sort_values("phaseSyn_win_rate", ascending=True)
    y = np.arange(len(wins))
    ax.barh(y, wins["phaseSyn_win_rate"], color="#D55E00")
    ax.axvline(0.5, color="#777777", linestyle="--", linewidth=1)
    ax.set_yticks(y, wins["label"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("PhaseSyn metric win rate")
    ax.set_title("D. Pairwise metric wins")
    ax.grid(axis="x", alpha=0.25)

    _save(fig, out / "pbc_fig1_control_arm.pdf")


def _parse_r(comparison: str) -> float | None:
    if "phasesyn_matched_R" not in comparison:
        return None
    try:
        return float(comparison.rsplit("R", 1)[1])
    except Exception:
        return None


def figure_matched_controls(tables: Path, out: Path) -> None:
    align = _read(tables, "exp2_table_baseline_alignment.csv")
    variance = _read(tables, "exp2_table_variance_reduction_by_R.csv")
    null = _read(tables, "exp2_table_null_calibration.csv")
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55))

    ax = axes[0]
    keep = {
        "treated_vs_original_randomized_controls": "Randomized controls",
        "treated_vs_empirical_subject_bootstrap_approximation": "Bootstrap approx.",
        "treated_vs_phasesyn_matched_R50": "PhaseSyn matched",
    }
    sub = align[align["comparison"].isin(keep)].copy()
    sub["label"] = sub["comparison"].map(keep)
    colors = ["#999999" if "PhaseSyn" not in x else "#D55E00" for x in sub["label"]]
    ax.bar(np.arange(len(sub)), sub["mean_abs_smd"], color=colors)
    ax.axhline(0.1, color="#777777", linestyle="--", linewidth=1)
    ax.set_xticks(np.arange(len(sub)), sub["label"], rotation=25, ha="right")
    ax.set_ylabel("Mean absolute SMD")
    ax.set_title("A. Baseline balance")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    phase = align[align["comparison"].str.contains("phasesyn_matched", na=False)].copy()
    phase["R"] = phase["comparison"].map(_parse_r)
    phase = phase.sort_values("R")
    ax.plot(phase["R"], phase["propensity_auc"], marker="o", color="#D55E00", label="PhaseSyn")
    ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1, label="chance")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50], ["1", "5", "10", "20", "50"])
    ax.set_ylim(0.48, 0.54)
    ax.set_xlabel("Futures per treated baseline (R)")
    ax.set_ylabel("Propensity AUC")
    ax.set_title("B. Matching stays at chance")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    ax.plot(variance["R"], variance["confidence_interval_width"], marker="o", color="#D55E00")
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 10, 20, 50], ["1", "5", "10", "20", "50"])
    ax.set_xlabel("Futures per treated baseline (R)")
    ax.set_ylabel("95% CI width")
    ax.set_title("C. Precision improves")
    ax.grid(axis="y", alpha=0.25)
    for _, row in null.iterrows():
        label = str(row["test"]).replace("synthetic_", "").replace("_vs_", " vs ")
        ax.text(0.03, 0.09 if "A0" in label else 0.02, f"Type-I {label}: {row['empirical_type1_error']:.3f}",
                transform=ax.transAxes, fontsize=6.5)

    _save(fig, out / "pbc_fig2_baseline_aligned_estimand_controls.pdf")


def figure_digital_twin(tables: Path, out: Path) -> None:
    coverage = _read(tables, "exp3_table_interval_coverage.csv")
    survival = _read(tables, "exp3_table_survival_prediction.csv").iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55))

    ax = axes[0]
    metrics = pd.Series({
        "C-index": survival["c_index"],
        "Risk-event corr.": survival["survival_risk_event_correlation"],
        "1 - IBS proxy": 1.0 - survival["integrated_brier_score_proxy"],
    })
    ax.bar(np.arange(len(metrics)), metrics.values, color=["#D55E00", "#E69F00", "#56B4E9"])
    ax.set_xticks(np.arange(len(metrics)), metrics.index, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("A. Survival risk signal")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    horizons = [2, 5, 10]
    briers = [survival["brier_score_2y"], survival["brier_score_5y"], survival["brier_score_10y"]]
    ax.plot(horizons, briers, marker="o", color="#D55E00")
    ax.set_xlabel("Horizon (years)")
    ax.set_ylabel("Brier score proxy")
    ax.set_title("B. Survival prediction error")
    ax.grid(alpha=0.25)

    ax = axes[2]
    mean_cov = coverage.groupby("interval", as_index=False)["coverage"].mean()
    nominal = mean_cov["interval"].astype(float) / 100.0
    ax.plot(nominal, mean_cov["coverage"], marker="o", color="#D55E00", label="observed")
    ax.plot([0.45, 1.0], [0.45, 1.0], linestyle="--", color="#777777", label="ideal")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("Nominal coverage")
    ax.set_ylabel("Mean empirical coverage")
    ax.set_title("C. Intervals need calibration")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)

    _save(fig, out / "pbc_fig3_factual_risk_calibration.pdf")


def figure_virtual_trials(tables: Path, out: Path) -> None:
    semi = _read(tables, "exp4_table_semisynthetic_type1_power.csv")
    real = _read(tables, "exp4_table_realdata_virtual_trial_power.csv")
    event = _read(tables, "exp4_table_event_accrual.csv")
    cal = _read(tables, "exp4_table_calibration_filtering_before_after.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))

    ax = axes[0, 0]
    p = semi[semi["gamma_A"] > 0].copy()
    ax.plot(p["gamma_A"], p["power"], marker="o", color="#D55E00", label="power")
    type1 = float(semi.loc[semi["gamma_A"].eq(0), "type1_error"].iloc[0])
    ax.scatter([0], [type1], color="#4D4D4D", zorder=3, label="type-I")
    ax.axhline(0.05, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Known treatment effect $\gamma_A$")
    ax.set_ylabel("Rejection rate")
    ax.set_title("A. Known-effect calibration")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for ratio, g in real.groupby("allocation_ratio"):
        g = g.sort_values("n")
        ax.plot(g["n"], g["power"], marker="o", label=ratio)
    ax.set_xlabel("Trial size")
    ax.set_ylabel("Power")
    ax.set_title("B. PBC-like trials remain low power")
    ax.set_ylim(0, max(0.15, real["power"].max() * 1.2))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Allocation", frameon=False)

    ax = axes[1, 0]
    mean_event = event.groupby("n", as_index=False).agg(events=("expected_number_events", "mean"), censoring=("censoring_rate", "mean"))
    ax.plot(mean_event["n"], mean_event["events"], marker="o", color="#D55E00")
    ax.set_xlabel("Trial size")
    ax.set_ylabel("Expected events")
    ax.set_title("C. Event accrual scales predictably")
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(mean_event["n"], mean_event["censoring"], marker="s", color="#0072B2")
    ax2.set_ylabel("Censoring rate", color="#0072B2")
    ax2.tick_params(axis="y", labelcolor="#0072B2")
    ax2.set_ylim(0.35, 0.55)

    ax = axes[1, 1]
    mean_cal = cal.groupby("n", as_index=False).agg(pass_rate=("pass_rate", "mean"), before=("before_filter_power", "mean"), after=("after_filter_power", "mean"))
    ax.plot(mean_cal["n"], mean_cal["pass_rate"], marker="o", color="#009E73", label="accepted")
    ax.set_xlabel("Trial size")
    ax.set_ylabel("Calibration pass rate")
    ax.set_title("D. Filtering is selective")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(mean_cal["n"], mean_cal["before"], marker="s", color="#999999", linestyle="--", label="before")
    ax2.plot(mean_cal["n"], mean_cal["after"], marker="^", color="#D55E00", linestyle="--", label="after")
    ax2.set_ylabel("Mean power")
    ax2.set_ylim(0, 0.15)

    _save(fig, out / "pbc_fig4_virtual_trials.pdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/pbc_experiments/experiment_20260604_core4_revision"))
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    _setup()
    tables = args.results_dir / "tables"
    figure_control_arm(tables, args.out_dir)
    figure_matched_controls(tables, args.out_dir)
    figure_digital_twin(tables, args.out_dir)
    figure_virtual_trials(tables, args.out_dir)


if __name__ == "__main__":
    main()
