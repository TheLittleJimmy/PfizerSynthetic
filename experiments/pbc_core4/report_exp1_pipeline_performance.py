from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .load_pbc import project_path


KEY_METRICS = [
    "baseline_mean_abs_smd",
    "baseline_categorical_prevalence_abs_error",
    "longitudinal_mean_trajectory_error",
    "longitudinal_change_from_baseline_error",
    "survival_km_integrated_abs_distance",
    "abs_survival_event_rate_error",
    "abs_survival_rmst_difference",
    "privacy_exact_duplicate_rate",
    "privacy_detection_auc_distance_from_0p5",
    "cox_hr_median",
    "logrank_sig_rate",
]


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        value = float(value)
    except Exception:
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    if abs(value) >= 10:
        return f"{value:.3f}"
    if abs(value) >= 0.01:
        return f"{value:.4f}"
    return f"{value:.3g}"


def _markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "No rows available."
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_numeric_dtype(view[col]):
            view[col] = view[col].map(_fmt)
    return view.to_markdown(index=False)


def _completed(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df:
        return df.copy()
    return df[df["status"].fillna("completed").astype(str).eq("completed")].copy()


def _dataset_summary(config: dict[str, Any]) -> list[str]:
    processed = project_path(config.get("processed_data_dir", "data/processed/pbc_core4"))
    subjects = _read_csv(processed / "pbc_subjects.csv")
    long_df = _read_csv(processed / "pbc_longitudinal.csv")
    survival = _read_csv(processed / "pbc_survival.csv")
    splits_path = processed / f"pbc_splits_seed{config.get('seed', 20260604)}.json"
    split_text = "split file not found"
    if splits_path.exists():
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
        split_text = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    lines = []
    if not subjects.empty:
        n_control = int(pd.to_numeric(subjects.get("treatment"), errors="coerce").eq(0).sum())
        n_treated = int(pd.to_numeric(subjects.get("treatment"), errors="coerce").eq(1).sum())
        event_rate = pd.to_numeric(survival.get("event_composite") if not survival.empty else None, errors="coerce").mean()
        lines.append(f"- subjects={len(subjects)}, controls={n_control}, treated={n_treated}, composite event rate={_fmt(event_rate)}")
    if not long_df.empty:
        lines.append(f"- longitudinal visits={len(long_df)}, unique longitudinal subjects={long_df['subject_id'].nunique() if 'subject_id' in long_df else 'NA'}")
    if not survival.empty:
        lines.append(f"- survival rows={len(survival)}")
    lines.append(f"- subject split: {split_text}")
    return lines


def _method_summary(metrics: pd.DataFrame, estimands: pd.DataFrame, privacy: pd.DataFrame) -> pd.DataFrame:
    metrics = _completed(metrics)
    if metrics.empty:
        return pd.DataFrame()
    work = metrics.copy()
    for col in ["survival_event_rate_error", "survival_rmst_difference"]:
        if col in work:
            work[f"abs_{col}"] = pd.to_numeric(work[col], errors="coerce").abs()
    metric_cols = [
        "baseline_mean_abs_smd",
        "baseline_categorical_prevalence_abs_error",
        "longitudinal_mean_trajectory_error",
        "longitudinal_change_from_baseline_error",
        "survival_km_integrated_abs_distance",
        "abs_survival_event_rate_error",
        "abs_survival_rmst_difference",
    ]
    out = work.groupby(["benchmark_task", "method"], dropna=False)[[c for c in metric_cols if c in work]].mean().reset_index()
    out.insert(2, "metric_rows", work.groupby(["benchmark_task", "method"], dropna=False).size().to_numpy())

    if not privacy.empty:
        pwork = privacy.copy()
        if "privacy_detection_classifier_auc" in pwork:
            pwork["privacy_detection_auc_distance_from_0p5"] = (pd.to_numeric(pwork["privacy_detection_classifier_auc"], errors="coerce") - 0.5).abs()
        pcols = [c for c in ["privacy_exact_duplicate_rate", "privacy_detection_auc_distance_from_0p5"] if c in pwork]
        if pcols:
            out = out.merge(pwork.groupby(["benchmark_task", "method"], dropna=False)[pcols].mean().reset_index(), on=["benchmark_task", "method"], how="left")

    if not estimands.empty:
        ework = estimands.copy()
        if "cox_hr" in ework:
            ework["cox_hr"] = pd.to_numeric(ework["cox_hr"], errors="coerce")
        if "logrank_p" in ework:
            ework["logrank_sig"] = pd.to_numeric(ework["logrank_p"], errors="coerce") < 0.05
        erows = []
        for keys, sub in ework.groupby(["benchmark_task", "method"], dropna=False):
            row = {"benchmark_task": keys[0], "method": keys[1]}
            if "cox_hr" in sub:
                row["cox_hr_median"] = sub["cox_hr"].median()
            if "logrank_sig" in sub:
                row["logrank_sig_rate"] = sub["logrank_sig"].mean()
            erows.append(row)
        out = out.merge(pd.DataFrame(erows), on=["benchmark_task", "method"], how="left")
    keep = [c for c in ["benchmark_task", "method", "metric_rows", *KEY_METRICS] if c in out]
    return out[keep].sort_values(["benchmark_task", "method"]).reset_index(drop=True)


def _audit(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for task, sub in metrics.groupby("benchmark_task", dropna=False):
        methods = sorted(sub["method"].dropna().astype(str).unique())
        rows.append({
            "benchmark_task": task,
            "methods": ";".join(methods),
            "bootstrap_present": "empirical_subject_bootstrap" in methods,
            "target_baseline_used_values": ";".join(sorted(sub.get("target_baseline_used", pd.Series(dtype=str)).dropna().astype(str).unique())),
            "generation_modes": ";".join(sorted(sub.get("generation_mode", pd.Series(dtype=str)).dropna().astype(str).unique())),
            "train_split_values": ";".join(sorted(sub.get("train_split", pd.Series(dtype=str)).dropna().astype(str).unique())),
            "eval_split_values": ";".join(sorted(sub.get("eval_split", pd.Series(dtype=str)).dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def _figure_summary(figures_dir: Path) -> pd.DataFrame:
    manifest = _read_csv(figures_dir / "exp1_metric_performance_manifest.csv")
    if manifest.empty or "metric_group" not in manifest:
        return pd.DataFrame()
    return manifest.groupby("metric_group", as_index=False).size().rename(columns={"size": "figure_count"})


def generate_report(config_path: Path, exp1_dir: Path | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    exp1_dir = exp1_dir or (project_path(config["output_dir"]) / "exp1_control_arm")
    tables_dir = exp1_dir / "tables"
    figures_dir = exp1_dir / "figures"
    reports_dir = exp1_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_csv(tables_dir / "exp1_metrics_all_methods.csv")
    estimands = _read_csv(tables_dir / "exp1_estimands_all_methods.csv")
    privacy = _read_csv(tables_dir / "exp1_privacy_all_methods.csv")
    summary = _method_summary(metrics, estimands, privacy)
    audit = _audit(metrics)
    figures = _figure_summary(figures_dir)

    summary.to_csv(tables_dir / "exp1_performance_summary_by_method.csv", index=False)
    audit.to_csv(tables_dir / "exp1_benchmark_fairness_audit.csv", index=False)

    lines = [
        "# Experiment 1 Redesigned Benchmark Report",
        "",
        "## Dataset and endpoint",
        "",
        *(_dataset_summary(config)),
        "- treatment coding: placebo/control=0, D-penicillamine/treated=1",
        "- primary endpoint: death-or-transplant composite (`event = status != 0`)",
        "",
        "## Redesign summary",
        "",
        "Experiment 1 now contains two train/test held-out benchmarks rather than the old control-fraction synthetic-control benchmark.",
        "All methods are trained on the existing `train` split and evaluated on the existing `test` split.",
        "`empirical_subject_bootstrap` is intentionally removed from Experiment 1.",
        "",
        "1. **Benchmark 1: prior generation ability.** Generate a held-out-sized cohort without using test baselines. Generation is arm-stratified to match the test treatment counts.",
        "2. **Benchmark 2: baseline-conditioned generation ability.** Use known test baselines and randomized treatment assignments, then generate post-baseline longitudinal and survival outcomes.",
        "",
        "## PhaseSyn generation modes",
        "",
        "Benchmark 1 uses PhaseSyn's learned prior cohort generator:",
        "",
        "```math",
        "\\widetilde B \\sim p_\\theta(B\\mid z,s),\\quad s\\sim p(s),\\quad z\\sim p_\\alpha(z\\mid s),\\quad",
        "(\\widetilde L_{>0},\\widetilde U,\\widetilde\\delta)\\sim p_\\Theta(L_{>0},U,\\delta\\mid \\widetilde B,A).",
        "```",
        "",
        "Benchmark 2 uses PhaseSyn's baseline-conditioned posterior generator:",
        "",
        "```math",
        "(\\widetilde s,\\widetilde z)\\sim q_\\phi(s,z\\mid B_{test}),\\quad",
        "(\\widetilde L_{>0},\\widetilde U,\\widetilde\\delta)\\sim p_\\Theta(L_{>0},U,\\delta\\mid B_{test},A_{test}).",
        "```",
        "",
        "Benchmark 2 therefore evaluates factual baseline-conditioned generation, not individual counterfactual truth.",
        "",
        "## Fairness audit",
        "",
        _markdown_table(audit),
        "",
        "## Method-level performance summary",
        "",
        "Lower is better for fidelity errors, event-rate error, RMST absolute difference, duplicate rate, and detection-AUC distance from 0.5.",
        "",
        _markdown_table(summary),
        "",
        "## Figures generated",
        "",
        f"Figure directory: `{figures_dir}`",
        "",
        "- `benchmark1_prior_generation_km.pdf`",
        "- `benchmark1_prior_generation_longitudinal_bili.pdf`",
        "- `benchmark2_baseline_conditioned_km.pdf`",
        "- `benchmark2_baseline_conditioned_longitudinal_bili.pdf`",
        "- `exp1_survival_km_distance_by_task.pdf`",
        "- `exp1_metric_performance_index.pdf`",
        "",
        "Per-metric figure counts:",
        "",
        _markdown_table(figures),
        "",
        "## Reproducibility",
        "",
        "```bash",
        "conda run -n env_2502 python -m experiments.pbc_core4.run_all --config experiments/pbc_core4/config_pbc_core4.yaml --only exp1",
        "conda run -n env_2502 python -m experiments.pbc_core4.plot_exp1_metric_performance --exp1-dir outputs/pbc_experiments/experiment_20260604_core4/exp1_control_arm",
        "conda run -n env_2502 python -m experiments.pbc_core4.report_exp1_pipeline_performance --config experiments/pbc_core4/config_pbc_core4.yaml",
        "```",
        "",
    ]
    out_path = reports_dir / "exp1_pipeline_performance_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Write redesigned Experiment 1 benchmark report.")
    parser.add_argument("--config", type=Path, default=Path("experiments/pbc_core4/config_pbc_core4.yaml"))
    parser.add_argument("--exp1-dir", type=Path, default=None)
    args = parser.parse_args()
    report = generate_report(args.config, args.exp1_dir)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
