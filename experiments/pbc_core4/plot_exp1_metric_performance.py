from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "PhaseSyn": "PhaseSyn",
    "LMM-AFT": "LMM-AFT",
    "JM-RE": "JM-RE",
    "classical_lmm_cox_aft_simulator": "LMM-AFT",
    "joint_longitudinal_survival_baseline": "JM-RE",
    "TVAE": "TVAE",
    "CTGAN": "CTGAN",
    "modular_deep_generator": "TVAE (old name)",
}

METHOD_ORDER = [
    "PhaseSyn",
    "TVAE",
    "CTGAN",
    "JM-RE",
    "LMM-AFT",
    "joint_longitudinal_survival_baseline",
    "classical_lmm_cox_aft_simulator",
    "modular_deep_generator",
]

COLORS = {
    "PhaseSyn": "#D55E00",
    "LMM-AFT": "#0072B2",
    "JM-RE": "#009E73",
    "classical_lmm_cox_aft_simulator": "#0072B2",
    "joint_longitudinal_survival_baseline": "#009E73",
    "TVAE": "#CC79A7",
    "CTGAN": "#E69F00",
    "modular_deep_generator": "#CC79A7",
}

TASK_LABELS = {
    "benchmark1_prior_generation": "Prior",
    "benchmark2_baseline_conditioned": "Baseline conditioned",
}

EXCLUDE_COLUMNS = {
    "replicate",
    "nu",
    "benchmark_task",
    "control_train_subjects",
    "generator_training_runtime_seconds",
    "exp1_target_baseline_used",
    "target_baseline_used",
    "bootstrap_removed",
    "baseline_generated_from_prior",
    "loaded_without_retraining",
}

METRIC_GROUPS = {
    "baseline": re.compile(r"^(baseline_|.*_smd$|.*_ks$|.*_js$)"),
    "longitudinal": re.compile(r"^longitudinal_"),
    "survival": re.compile(r"^survival_"),
    "estimand": re.compile(r"^(cox_|logrank_|rmst_|mmrm_|responder_)"),
    "privacy": re.compile(r"^privacy_"),
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


def _label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return text or "metric"


def _metric_group(metric: str) -> str:
    for group, pattern in METRIC_GROUPS.items():
        if pattern.search(metric):
            return group
    return "other"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _numeric_metrics(df: pd.DataFrame) -> list[str]:
    metrics: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_COLUMNS or col in {"method", "setting", "status", "fit_status", "cox_status", "privacy_detection_classifier_status"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() > 0:
            metrics.append(col)
    return metrics


def _ordered_methods(df: pd.DataFrame) -> list[str]:
    observed = [m for m in METHOD_ORDER if m in set(df["method"].dropna().astype(str))]
    extras = sorted(set(df["method"].dropna().astype(str)) - set(observed))
    return observed + extras


def _summary(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if "benchmark_task" in df:
        work = df[["method", "benchmark_task", metric]].copy()
        task_order = {name: i for i, name in enumerate(TASK_LABELS)}
        work["x"] = work["benchmark_task"].map(task_order)
        work["x_label"] = work["benchmark_task"].map(TASK_LABELS).fillna(work["benchmark_task"].astype(str))
        group_cols = ["method", "benchmark_task", "x", "x_label"]
    elif "nu" in df:
        work = df[["method", "nu", metric]].copy()
        work["nu"] = pd.to_numeric(work["nu"], errors="coerce")
        work["x"] = work["nu"]
        work["x_label"] = work["nu"].map(lambda v: {1 / 3: "1/3", 2 / 3: "2/3", 1.0: "1"}.get(float(v), f"{float(v):g}") if pd.notna(v) else "")
        group_cols = ["method", "nu", "x", "x_label"]
    else:
        work = df[["method", metric]].copy()
        work["x"] = 0.0
        work["x_label"] = "Overall"
        group_cols = ["method", "x", "x_label"]
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.dropna(subset=[metric])
    if work.empty:
        return pd.DataFrame()
    grouped = (
        work.groupby(group_cols, dropna=False)[metric]
        .agg(mean="mean", std="std", n="count", median="median")
        .reset_index()
    )
    grouped["se"] = grouped["std"].fillna(0.0) / np.sqrt(grouped["n"].clip(lower=1))
    grouped["ci95"] = 1.96 * grouped["se"]
    return grouped


def _plot_metric(df: pd.DataFrame, metric: str, source: str, out_dir: Path) -> dict[str, str] | None:
    summary = _summary(df, metric)
    if summary.empty:
        return None

    group = _metric_group(metric)
    group_dir = out_dir / f"metric_performance_{group}"
    group_dir.mkdir(parents=True, exist_ok=True)
    path = group_dir / f"{_slug(source)}__{_slug(metric)}.pdf"

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in _ordered_methods(df):
        sub = summary[summary["method"].astype(str).eq(method)].sort_values("x")
        if sub.empty:
            continue
        x = sub["x"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        ci = sub["ci95"].fillna(0.0).to_numpy(dtype=float)
        color = COLORS.get(method, "#666666")
        ax.plot(x, y, marker="o", linewidth=1.6, color=color, label=_label(method))
        if len(sub) > 1:
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.12, linewidth=0)
        else:
            ax.errorbar(x, y, yerr=ci, color=color, capsize=2, linewidth=0)

    if "benchmark_task" in summary:
        ticks = summary[["x", "x_label"]].drop_duplicates().sort_values("x")
        ax.set_xticks(ticks["x"].to_numpy(dtype=float), ticks["x_label"].astype(str).to_list())
        ax.set_xlabel("Experiment 1 benchmark task")
    elif "nu" in summary:
        ax.set_xticks([1 / 3, 2 / 3, 1.0], ["1/3", "2/3", "1"])
        ax.set_xlabel("Control training fraction (nu)")
    else:
        ax.set_xticks([0.0], ["Overall"])
        ax.set_xlabel("Summary")
    ax.set_ylabel(metric)
    ax.set_title(metric.replace("_", " "))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        "source_table": source,
        "metric": metric,
        "metric_group": group,
        "figure_path": str(path),
        "rows_used": str(int(summary["n"].sum())),
        "methods": ";".join(_ordered_methods(df)),
    }


def _plot_metric_index(manifest: pd.DataFrame, out_dir: Path) -> None:
    if manifest.empty:
        return
    counts = manifest.groupby("metric_group", as_index=False).size().sort_values("metric_group")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(counts["metric_group"], counts["size"], color="#4D4D4D")
    ax.set_xlabel("Metric group")
    ax.set_ylabel("Number of per-metric figures")
    ax.set_title("Experiment 1 per-metric performance figure index")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "exp1_metric_performance_index.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(tables_dir: Path, figures_dir: Path) -> pd.DataFrame:
    _setup()
    sources = {
        "exp1_metrics_all_methods": _read_csv(tables_dir / "exp1_metrics_all_methods.csv"),
        "exp1_estimands_all_methods": _read_csv(tables_dir / "exp1_estimands_all_methods.csv"),
        "exp1_privacy_all_methods": _read_csv(tables_dir / "exp1_privacy_all_methods.csv"),
    }
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for source, df in sources.items():
        if df.empty or "method" not in df:
            continue
        for metric in _numeric_metrics(df):
            item = _plot_metric(df, metric, source, figures_dir)
            if item:
                rows.append(item)
    manifest = pd.DataFrame(rows)
    manifest_path = figures_dir / "exp1_metric_performance_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    _plot_metric_index(manifest, figures_dir)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one Experiment 1 performance PDF per numeric metric.")
    parser.add_argument(
        "--exp1-dir",
        type=Path,
        default=Path("outputs/pbc_experiments/experiment_20260604_core4/exp1_control_arm"),
    )
    parser.add_argument("--tables-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    args = parser.parse_args()

    tables_dir = args.tables_dir or args.exp1_dir / "tables"
    figures_dir = args.figures_dir or args.exp1_dir / "figures"
    manifest = generate_figures(tables_dir, figures_dir)
    print(f"wrote {len(manifest)} per-metric figures")
    print(f"figures_dir={figures_dir}")
    print(f"manifest={figures_dir / 'exp1_metric_performance_manifest.csv'}")


if __name__ == "__main__":
    main()
