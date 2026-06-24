from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .load_pbc import load_processed, project_path
from .methods import split_static_long
from .metrics import clinical_estimands, clinical_longitudinal_reference


TASKS = {
    "benchmark1_prior_generation": "prior_generation",
    "benchmark2_baseline_conditioned": "baseline_conditioned",
}

TASK_TITLES = {
    "benchmark1_prior_generation": "Prior generation",
    "benchmark2_baseline_conditioned": "Baseline-conditioned generation",
}

METHOD_ORDER = ["PhaseSyn", "JM-RE", "LMM-AFT", "TVAE", "CTGAN"]

CLINICAL_ESTIMAND_METRICS = [
    ("cox_log_hr", "Cox log hazard ratio"),
    ("cox_hr", "Cox hazard ratio"),
    ("cox_se", "Cox SE"),
    ("cox_p", "Cox p-value"),
    ("logrank_p", "Log-rank p-value"),
    ("rmst_difference_treated_minus_control", "RMST treated-control difference"),
    ("mmrm_proxy_bili_treatment_effect", "MMRM proxy bili treatment effect"),
    ("mmrm_proxy_bili_p", "MMRM proxy bili p-value"),
    ("responder_rate_diff_bili", "Responder rate difference, bili"),
    ("mmrm_proxy_albumin_treatment_effect", "MMRM proxy albumin treatment effect"),
    ("mmrm_proxy_albumin_p", "MMRM proxy albumin p-value"),
    ("responder_rate_diff_albumin", "Responder rate difference, albumin"),
    ("mmrm_proxy_prothrombin_treatment_effect", "MMRM proxy prothrombin treatment effect"),
    ("mmrm_proxy_prothrombin_p", "MMRM proxy prothrombin p-value"),
    ("responder_rate_diff_prothrombin", "Responder rate difference, prothrombin"),
]

BEAMER_METRICS = [
    {
        "label": "Baseline MAE",
        "source": "main",
        "metric_column": "baseline_continuous_mean_abs_error",
        "value_column": "mean",
        "lower_is_better": True,
    },
    {
        "label": "Longitudinal traj. MAE",
        "source": "main",
        "metric_column": "longitudinal_mean_trajectory_error",
        "value_column": "mean",
        "lower_is_better": True,
    },
    {
        "label": "Change-from-baseline MAE",
        "source": "main",
        "metric_column": "longitudinal_change_from_baseline_error",
        "value_column": "mean",
        "lower_is_better": True,
    },
    {
        "label": "KM IAD",
        "source": "main",
        "metric_column": "survival_km_integrated_abs_distance",
        "value_column": "mean",
        "lower_is_better": True,
    },
    {
        "label": "Event-rate |bias|",
        "source": "main",
        "metric_column": "survival_event_rate_error",
        "value_column": "mean_abs",
        "lower_is_better": True,
    },
    {
        "label": "Bili estimand |bias|",
        "source": "clinical_vs_true",
        "metric_column": "mmrm_proxy_bili_treatment_effect",
        "value_column": "abs_bias",
        "lower_is_better": True,
    },
]

SIGNED_ERROR_COLUMNS = {
    "survival_event_rate_error",
    "survival_censoring_rate_error",
    "survival_rmst_difference",
    "survival_median_followup_error",
    "cox_log_hr",
    "rmst_difference_treated_minus_control",
    "mmrm_proxy_bili_treatment_effect",
    "mmrm_proxy_albumin_treatment_effect",
    "mmrm_proxy_prothrombin_treatment_effect",
    "responder_rate_diff_bili",
    "responder_rate_diff_albumin",
    "responder_rate_diff_prothrombin",
}

CURATED_METRICS: list[dict[str, Any]] = [
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Continuous mean absolute error",
        "column": "baseline_continuous_mean_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Continuous SD absolute error",
        "column": "baseline_continuous_sd_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Mean absolute SMD",
        "column": "baseline_mean_abs_smd",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Mean Jensen-Shannon distance",
        "column": "baseline_mean_js_distance",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Mean KS statistic",
        "column": "baseline_mean_ks",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Categorical prevalence absolute error",
        "column": "baseline_categorical_prevalence_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Baseline fidelity",
        "metric": "Continuous correlation matrix MAE",
        "column": "baseline_correlation_matrix_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Mean trajectory MAE",
        "column": "longitudinal_mean_trajectory_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Change-from-baseline MAE",
        "column": "longitudinal_change_from_baseline_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Slope distribution error",
        "column": "longitudinal_slope_distribution_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Variance trajectory error",
        "column": "longitudinal_variance_trajectory_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Visit-count absolute error",
        "column": "longitudinal_visit_count_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Longitudinal fidelity",
        "metric": "Missing-cell-rate absolute error",
        "column": "longitudinal_missing_cell_rate_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "Event-rate bias",
        "column": "survival_event_rate_error",
        "direction": "zero",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "Event-rate absolute error",
        "column": "survival_event_rate_abs_error",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "KM integrated absolute distance",
        "column": "survival_km_integrated_abs_distance",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "RMST bias",
        "column": "survival_rmst_difference",
        "direction": "zero",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "RMST absolute difference",
        "column": "survival_rmst_abs_difference",
        "direction": "lower",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "Median follow-up bias",
        "column": "survival_median_followup_error",
        "direction": "zero",
    },
    {
        "source": "metrics",
        "group": "Survival fidelity",
        "metric": "Median follow-up absolute error",
        "column": "survival_median_followup_abs_error",
        "direction": "lower",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "Cox log hazard ratio",
        "column": "cox_log_hr",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "Cox hazard ratio",
        "column": "cox_hr",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "Log-rank p-value",
        "column": "logrank_p",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "RMST treated-control difference",
        "column": "rmst_difference_treated_minus_control",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "MMRM proxy bili treatment effect",
        "column": "mmrm_proxy_bili_treatment_effect",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "Responder rate difference, bili",
        "column": "responder_rate_diff_bili",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "MMRM proxy albumin treatment effect",
        "column": "mmrm_proxy_albumin_treatment_effect",
        "direction": "descriptive",
    },
    {
        "source": "estimands",
        "group": "Clinical estimands",
        "metric": "MMRM proxy prothrombin treatment effect",
        "column": "mmrm_proxy_prothrombin_treatment_effect",
        "direction": "descriptive",
    },
    {
        "source": "privacy",
        "group": "Privacy diagnostics",
        "metric": "Nearest-neighbor distance ratio",
        "column": "privacy_nearest_neighbor_distance_ratio",
        "direction": "descriptive",
    },
    {
        "source": "privacy",
        "group": "Privacy diagnostics",
        "metric": "Distance to closest real record",
        "column": "privacy_distance_to_closest_real_record",
        "direction": "descriptive",
    },
    {
        "source": "privacy",
        "group": "Privacy diagnostics",
        "metric": "Exact duplicate rate",
        "column": "privacy_exact_duplicate_rate",
        "direction": "descriptive",
    },
    {
        "source": "privacy",
        "group": "Privacy diagnostics",
        "metric": "Mean k-map equivalence count",
        "column": "privacy_kmap_mean_equivalence_count",
        "direction": "descriptive",
    },
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = {
        "longitudinal_visit_count_abs_error": (
            "longitudinal_visit_count_synthetic",
            "longitudinal_visit_count_real",
        ),
        "longitudinal_missing_cell_rate_abs_error": (
            "longitudinal_missing_cell_rate_synthetic",
            "longitudinal_missing_cell_rate_real",
        ),
    }
    for new_col, (left, right) in pairs.items():
        if left in out and right in out:
            out[new_col] = (pd.to_numeric(out[left], errors="coerce") - pd.to_numeric(out[right], errors="coerce")).abs()
    for col, new_col in [
        ("survival_event_rate_error", "survival_event_rate_abs_error"),
        ("survival_censoring_rate_error", "survival_censoring_rate_abs_error"),
        ("survival_rmst_difference", "survival_rmst_abs_difference"),
        ("survival_median_followup_error", "survival_median_followup_abs_error"),
    ]:
        if col in out:
            out[new_col] = pd.to_numeric(out[col], errors="coerce").abs()
    return out


def _format_number(value: float) -> str:
    if not np.isfinite(value):
        return ""
    value = float(value)
    abs_value = abs(value)
    if abs_value >= 100000:
        return f"{value:.0f}"
    if abs_value >= 1000:
        return f"{value:.1f}"
    if abs_value >= 100:
        return f"{value:.2f}"
    if abs_value >= 10:
        return f"{value:.3f}"
    if abs_value >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def _format_mean_sd(mean: float, sd: float) -> str:
    if not np.isfinite(mean):
        return ""
    if not np.isfinite(sd):
        return _format_number(mean)
    return f"{_format_number(mean)} ({_format_number(sd)})"


def _rank_values(values: dict[str, float], direction: str) -> dict[str, float]:
    finite = {method: value for method, value in values.items() if np.isfinite(value)}
    if direction not in {"lower", "higher", "zero"} or not finite:
        return {}
    series = pd.Series(finite, dtype=float)
    return series.rank(method="min", ascending=direction in {"lower", "zero"}).to_dict()


def _best_methods(values: dict[str, float], direction: str) -> list[str]:
    finite = {method: value for method, value in values.items() if np.isfinite(value)}
    if direction not in {"lower", "higher", "zero"} or not finite:
        return []
    target = min(finite.values()) if direction in {"lower", "zero"} else max(finite.values())
    return [method for method in METHOD_ORDER if method in finite and np.isclose(finite[method], target, rtol=1e-10, atol=1e-12)]


def _metric_values_for_ranking(values: pd.Series, direction: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if direction == "zero":
        return numeric.abs()
    return numeric


def _summarize_metric(
    df: pd.DataFrame,
    task: str,
    source: str,
    group: str,
    metric: str,
    column: str,
    direction: str,
) -> pd.DataFrame:
    if column not in df:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    ranking_values: dict[str, float] = {}
    for method in METHOD_ORDER:
        method_values = pd.to_numeric(df.loc[df["method"].eq(method), column], errors="coerce").dropna()
        if method_values.empty:
            continue
        ranking_series = _metric_values_for_ranking(method_values, direction)
        ranking_value = float(ranking_series.mean()) if np.isfinite(ranking_series).any() else np.nan
        ranking_values[method] = ranking_value
        rows.append({
            "benchmark_task": task,
            "benchmark_label": TASK_TITLES.get(task, task),
            "source_table": source,
            "metric_group": group,
            "metric": metric,
            "metric_column": column,
            "rank_direction": direction,
            "rank_basis": "mean_abs" if direction == "zero" else ("mean" if direction in {"lower", "higher"} else "descriptive"),
            "method": method,
            "n": int(method_values.notna().sum()),
            "mean": float(method_values.mean()),
            "sd": float(method_values.std(ddof=1)) if len(method_values) > 1 else np.nan,
            "median": float(method_values.median()),
            "q25": float(method_values.quantile(0.25)),
            "q75": float(method_values.quantile(0.75)),
            "mean_abs": float(method_values.abs().mean()),
            "ranking_value": ranking_value,
        })
    if not rows:
        return pd.DataFrame()
    ranks = _rank_values(ranking_values, direction)
    best_methods = _best_methods(ranking_values, direction)
    best_method = "; ".join(best_methods)
    phase_rank = ranks.get("PhaseSyn", np.nan)
    best_value = ranking_values.get(best_methods[0], np.nan) if best_methods else np.nan
    for row in rows:
        method = str(row["method"])
        row["rank"] = ranks.get(method, np.nan)
        row["best_method"] = best_method
        row["phasesyn_rank"] = phase_rank
        phase_value = ranking_values.get("PhaseSyn", np.nan)
        if np.isfinite(phase_value) and np.isfinite(best_value):
            row["phasesyn_delta_to_best"] = float(phase_value - best_value)
            row["phasesyn_pct_from_best"] = float((phase_value - best_value) / max(abs(best_value), 1e-12) * 100.0)
        else:
            row["phasesyn_delta_to_best"] = np.nan
            row["phasesyn_pct_from_best"] = np.nan
    return pd.DataFrame(rows)


def _wide_from_long(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    key_cols = [
        "benchmark_task",
        "benchmark_label",
        "source_table",
        "metric_group",
        "metric",
        "metric_column",
        "rank_direction",
        "rank_basis",
        "best_method",
        "phasesyn_rank",
        "phasesyn_delta_to_best",
        "phasesyn_pct_from_best",
    ]
    rows = summary[key_cols].drop_duplicates().reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=int)
    formatted = summary.merge(rows[["row_id", *key_cols]], on=key_cols, how="left")
    formatted = formatted.assign(value=formatted.apply(lambda row: _format_mean_sd(row["mean"], row["sd"]), axis=1))
    pivot = formatted.pivot_table(index="row_id", columns="method", values="value", aggfunc="first").reset_index()
    pivot = rows.merge(pivot, on="row_id", how="left").drop(columns=["row_id"])
    for method in METHOD_ORDER:
        if method not in pivot:
            pivot[method] = ""
    out = pivot[
        [
            "metric_group",
            "metric",
            "rank_direction",
            *METHOD_ORDER,
            "best_method",
            "phasesyn_rank",
            "phasesyn_delta_to_best",
            "phasesyn_pct_from_best",
            "source_table",
            "metric_column",
        ]
    ].copy()
    out["phasesyn_rank"] = out["phasesyn_rank"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else int(x))
    out["phasesyn_delta_to_best"] = out["phasesyn_delta_to_best"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else _format_number(float(x)))
    out["phasesyn_pct_from_best"] = out["phasesyn_pct_from_best"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else _format_number(float(x)))
    return out


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
        .replace("~", "\\textasciitilde{}")
        .replace("^", "\\textasciicircum{}")
    )


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_latex(index=False, escape=True, caption=caption, label=label)
    path.write_text(text + "\n", encoding="utf-8")


def _best_competitor(values: pd.DataFrame, value_column: str, lower_is_better: bool) -> tuple[str, float]:
    work = values.loc[~values["method"].eq("PhaseSyn")].copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])
    if work.empty:
        return "", np.nan
    idx = work[value_column].idxmin() if lower_is_better else work[value_column].idxmax()
    row = work.loc[idx]
    return str(row["method"]), float(row[value_column])


def _format_beamer_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if abs(value) >= 1000:
        return f"{value:.0f}"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    if abs(value) >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _delta_text(phase_value: float, best_value: float, lower_is_better: bool) -> str:
    if not (np.isfinite(phase_value) and np.isfinite(best_value)):
        return ""
    if np.isclose(best_value, 0.0, atol=1e-12):
        diff = phase_value - best_value if lower_is_better else best_value - phase_value
        return _format_beamer_value(diff)
    if lower_is_better:
        pct = (phase_value - best_value) / abs(best_value) * 100.0
    else:
        pct = (best_value - phase_value) / abs(best_value) * 100.0
    return f"{pct:+.1f}%"


def _beamer_table_rows(task: str, main_long: pd.DataFrame, clinical_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec in BEAMER_METRICS:
        source = str(spec["source"])
        metric_column = str(spec["metric_column"])
        value_column = str(spec["value_column"])
        lower_is_better = bool(spec["lower_is_better"])
        if source == "main":
            df = main_long.loc[
                main_long["benchmark_task"].eq(task)
                & main_long["metric_column"].eq(metric_column)
            ].copy()
        else:
            df = clinical_long.loc[
                clinical_long["benchmark_task"].eq(task)
                & clinical_long["metric_column"].eq(metric_column)
            ].copy()
        if df.empty or value_column not in df:
            continue
        phase_values = df.loc[df["method"].eq("PhaseSyn"), value_column]
        phase_value = float(pd.to_numeric(phase_values, errors="coerce").dropna().iloc[0]) if not phase_values.dropna().empty else np.nan
        best_method, best_value = _best_competitor(df, value_column, lower_is_better)
        winner = "PhaseSyn" if np.isfinite(phase_value) and (
            not np.isfinite(best_value)
            or (phase_value <= best_value if lower_is_better else phase_value >= best_value)
        ) else best_method
        rows.append({
            "Benchmark": TASK_TITLES.get(task, task).replace("-conditioned generation", "-cond."),
            "Metric": str(spec["label"]),
            "PhaseSyn": _format_beamer_value(phase_value),
            "Best comparator": f"{best_method} ({_format_beamer_value(best_value)})" if best_method else "",
            "Delta": _delta_text(phase_value, best_value, lower_is_better),
            "Winner": winner,
            "benchmark_task": task,
            "metric_column": metric_column,
            "phasesyn_value": phase_value,
            "best_comparator": best_method,
            "best_comparator_value": best_value,
        })
    return pd.DataFrame(rows)


def _write_beamer_snippet(df: pd.DataFrame, path: Path) -> None:
    display = df[["Benchmark", "Metric", "PhaseSyn", "Best comparator", "Delta", "Winner"]].copy()
    table = display.to_latex(index=False, escape=True)
    frame = (
        "\\begin{frame}{Experiment 1 Benchmark Summary}\n"
        "\\scriptsize\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.08}\n"
        f"{table}"
        "\\vspace{-0.5em}\n"
        "\\begin{flushleft}\\tiny Lower is better for all metrics. Delta is PhaseSyn relative to the best non-PhaseSyn comparator.\\end{flushleft}\n"
        "\\end{frame}\n"
    )
    path.write_text(frame, encoding="utf-8")


def _write_beamer_tables(output_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    pieces = []
    for task, slug in TASKS.items():
        main_long = pd.read_csv(output_dir / f"exp1_publication_{slug}_main_long.csv")
        clinical_long = pd.read_csv(output_dir / f"exp1_publication_{slug}_clinical_estimand_vs_true_long.csv")
        pieces.append(_beamer_table_rows(task, main_long, clinical_long))
    compact = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    csv_path = output_dir / "exp1_beamer_short_benchmark_table.csv"
    tex_path = output_dir / "exp1_beamer_short_benchmark_table.tex"
    frame_path = output_dir / "exp1_beamer_short_benchmark_frame.tex"
    compact.to_csv(csv_path, index=False)
    display = compact[["Benchmark", "Metric", "PhaseSyn", "Best comparator", "Delta", "Winner"]].copy()
    _write_latex_table(
        display,
        tex_path,
        caption="Compact Experiment 1 benchmark summary for presentation.",
        label="tab:exp1-beamer-short-benchmark",
    )
    _write_beamer_snippet(compact, frame_path)
    artifacts[csv_path.name] = str(csv_path)
    artifacts[tex_path.name] = str(tex_path)
    artifacts[frame_path.name] = str(frame_path)
    return artifacts


def _infer_config_path(exp1_dir: Path) -> Path:
    output_dir = exp1_dir.parent
    run_config = output_dir / "phasesyn_model" / "train" / "run_config.yaml"
    if run_config.exists():
        return run_config
    candidates = [
        Path("experiments/pbc_core4/config_pbc_core4_20260617.yaml"),
        Path("experiments/pbc_core4/config_pbc_core4_tuned.yaml"),
        Path("experiments/pbc_core4/config_pbc_core4.yaml"),
    ]
    for candidate in candidates:
        path = project_path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("Could not infer an Experiment 1 config path.")


def _load_config(config_path: Path) -> dict[str, Any]:
    with project_path(config_path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is not a mapping: {config_path}")
    return cfg


def _true_clinical_estimands(config_path: Path) -> pd.DataFrame:
    cfg = _load_config(config_path)
    seed = int(cfg["seed"])
    data = load_processed(cfg["processed_data_dir"], seed)
    test_static, test_long = split_static_long(data, "test", endpoint=str(cfg.get("endpoint", {}).get("primary", "composite")))
    ref = clinical_longitudinal_reference(test_long)
    row = clinical_estimands(
        test_static,
        test_long,
        label="Observed test set",
        replicate=-1,
        setting="observed_test_set",
        longitudinal_reference=ref,
    )
    row["benchmark_task"] = "observed_test_set"
    row["train_split"] = "none"
    row["eval_split"] = "test"
    row["n_test_subjects"] = int(len(test_static))
    row["n_test_longitudinal_rows"] = int(len(test_long))
    return pd.DataFrame([row])


def _comparison_direction(column: str) -> str:
    if column.endswith("_p") or column in {"cox_p", "logrank_p"}:
        return "descriptive"
    return "lower"


def _clinical_comparison_long(estimands: pd.DataFrame, true_row: pd.Series, task: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    task_df = estimands.loc[estimands["benchmark_task"].eq(task)].copy()
    for column, metric in CLINICAL_ESTIMAND_METRICS:
        if column not in task_df or column not in true_row:
            continue
        true_value = pd.to_numeric(pd.Series([true_row[column]]), errors="coerce").iloc[0]
        if not np.isfinite(true_value):
            continue
        direction = _comparison_direction(column)
        method_score: dict[str, float] = {}
        for method in METHOD_ORDER:
            values = pd.to_numeric(task_df.loc[task_df["method"].eq(method), column], errors="coerce").dropna()
            if values.empty:
                continue
            diffs = values - true_value
            bias = float(diffs.mean())
            abs_bias = abs(bias)
            mae = float(diffs.abs().mean())
            rmse = float(np.sqrt(np.mean(np.square(diffs))))
            method_score[method] = abs_bias if direction == "lower" else np.nan
            rows.append({
                "benchmark_task": task,
                "benchmark_label": TASK_TITLES.get(task, task),
                "metric": metric,
                "metric_column": column,
                "comparison_direction": direction,
                "method": method,
                "n": int(values.notna().sum()),
                "true_value": float(true_value),
                "synthetic_mean": float(values.mean()),
                "synthetic_sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                "bias": bias,
                "abs_bias": abs_bias,
                "mae": mae,
                "rmse": rmse,
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "q75": float(values.quantile(0.75)),
            })
        ranks = _rank_values(method_score, direction)
        best_methods = _best_methods(method_score, direction)
        best_method = "; ".join(best_methods)
        best_value = method_score.get(best_methods[0], np.nan) if best_methods else np.nan
        for row in rows:
            if row["benchmark_task"] != task or row["metric_column"] != column:
                continue
            method = str(row["method"])
            row["rank"] = ranks.get(method, np.nan)
            row["best_method"] = best_method
            row["phasesyn_rank"] = ranks.get("PhaseSyn", np.nan)
            phase_value = method_score.get("PhaseSyn", np.nan)
            if np.isfinite(phase_value) and np.isfinite(best_value):
                row["phasesyn_abs_bias_delta_to_best"] = float(phase_value - best_value)
                row["phasesyn_abs_bias_pct_from_best"] = float((phase_value - best_value) / max(abs(best_value), 1e-12) * 100.0)
            else:
                row["phasesyn_abs_bias_delta_to_best"] = np.nan
                row["phasesyn_abs_bias_pct_from_best"] = np.nan
    return pd.DataFrame(rows)


def _format_comparison_cell(row: pd.Series) -> str:
    mean = _format_number(float(row["synthetic_mean"]))
    sd = _format_number(float(row["synthetic_sd"])) if np.isfinite(float(row["synthetic_sd"])) else ""
    bias = _format_number(float(row["bias"]))
    rmse = _format_number(float(row["rmse"]))
    if sd:
        return f"{mean} ({sd}); bias {bias}; RMSE {rmse}"
    return f"{mean}; bias {bias}; RMSE {rmse}"


def _clinical_comparison_wide(comparison: pd.DataFrame) -> pd.DataFrame:
    if comparison.empty:
        return comparison
    key_cols = [
        "benchmark_task",
        "benchmark_label",
        "metric",
        "metric_column",
        "comparison_direction",
        "true_value",
        "best_method",
        "phasesyn_rank",
        "phasesyn_abs_bias_delta_to_best",
        "phasesyn_abs_bias_pct_from_best",
    ]
    rows = comparison[key_cols].drop_duplicates().reset_index(drop=True)
    rows["row_id"] = np.arange(len(rows), dtype=int)
    formatted = comparison.merge(rows[["row_id", *key_cols]], on=key_cols, how="left")
    formatted = formatted.assign(value=formatted.apply(_format_comparison_cell, axis=1))
    pivot = formatted.pivot_table(index="row_id", columns="method", values="value", aggfunc="first").reset_index()
    pivot = rows.merge(pivot, on="row_id", how="left").drop(columns=["row_id"])
    for method in METHOD_ORDER:
        if method not in pivot:
            pivot[method] = ""
    out = pivot[
        [
            "metric",
            "comparison_direction",
            "true_value",
            *METHOD_ORDER,
            "best_method",
            "phasesyn_rank",
            "phasesyn_abs_bias_delta_to_best",
            "phasesyn_abs_bias_pct_from_best",
            "metric_column",
        ]
    ].copy()
    out["true_value"] = out["true_value"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else _format_number(float(x)))
    out["phasesyn_rank"] = out["phasesyn_rank"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else int(x))
    out["phasesyn_abs_bias_delta_to_best"] = out["phasesyn_abs_bias_delta_to_best"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else _format_number(float(x)))
    out["phasesyn_abs_bias_pct_from_best"] = out["phasesyn_abs_bias_pct_from_best"].apply(lambda x: "" if not np.isfinite(pd.to_numeric(x, errors="coerce")) else _format_number(float(x)))
    return out


def _write_true_clinical_comparison_tables(
    exp1_dir: Path,
    output_dir: Path,
    sources: dict[str, pd.DataFrame],
    config_path: Path | None,
) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    config_path = _infer_config_path(exp1_dir) if config_path is None else project_path(config_path)
    true_df = _true_clinical_estimands(config_path)
    true_path = output_dir / "exp1_publication_true_clinical_estimands.csv"
    true_df.to_csv(true_path, index=False)
    artifacts[true_path.name] = str(true_path)

    true_row = true_df.iloc[0]
    for task, slug in TASKS.items():
        comparison = _clinical_comparison_long(sources["estimands"], true_row, task)
        wide = _clinical_comparison_wide(comparison)
        for name, df in [
            (f"exp1_publication_{slug}_clinical_estimand_vs_true_long.csv", comparison),
            (f"exp1_publication_{slug}_clinical_estimand_vs_true_wide.csv", wide),
        ]:
            path = output_dir / name
            df.to_csv(path, index=False)
            artifacts[name] = str(path)
        tex_path = output_dir / f"exp1_publication_{slug}_clinical_estimand_vs_true_wide.tex"
        _write_latex_table(
            wide,
            tex_path,
            caption=(
                f"Experiment 1 {TASK_TITLES[task].lower()} clinical estimands compared with observed held-out test-set estimands. "
                "Method entries are synthetic mean (SD); bias synthetic-minus-true; RMSE over 500 replicates."
            ),
            label=f"tab:exp1-{slug}-clinical-vs-true",
        )
        artifacts[tex_path.name] = str(tex_path)
    return artifacts


def _humanize_column(column: str) -> str:
    replacements = {
        "L0": "baseline",
        "smd": "SMD",
        "ks": "KS",
        "js": "Jensen-Shannon",
        "rmst": "RMST",
        "cox": "Cox",
        "mmrm": "MMRM",
        "auc": "AUC",
        "mae": "MAE",
    }
    text = column
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("_", " ")


def _direction_for_column(column: str, source: str) -> str:
    if source == "privacy":
        return "descriptive"
    if source == "estimands":
        return "descriptive"
    if column in SIGNED_ERROR_COLUMNS or column.endswith("_difference") or column.endswith("_diff"):
        return "zero"
    if (
        column.endswith("_error")
        or column.endswith("_abs_error")
        or column.endswith("_distance")
        or column.endswith("_ks")
        or column.endswith("_js")
        or column.endswith("_smd")
        or "abs_error" in column
    ):
        return "lower"
    return "descriptive"


def _detailed_metric_specs(metrics: pd.DataFrame, estimands: pd.DataFrame, privacy: pd.DataFrame) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    excluded_metrics = {
        "replicate",
        "target_baseline_used",
        "bootstrap_removed",
        "generator_training_runtime_seconds",
        "generation_batch_size",
        "generation_batch_start",
        "longitudinal_visit_count_real",
        "longitudinal_visit_count_synthetic",
        "longitudinal_missing_cell_rate_real",
        "longitudinal_missing_cell_rate_synthetic",
    }
    for source, df in [("metrics", metrics), ("estimands", estimands), ("privacy", privacy)]:
        if df.empty:
            continue
        for column in df.columns:
            if column in excluded_metrics:
                continue
            if column in {"method", "setting", "benchmark_task", "status", "generation_mode", "train_split", "eval_split", "fit_status", "generator_training_scope", "cox_status", "privacy_detection_classifier_status"}:
                continue
            numeric = pd.to_numeric(df[column], errors="coerce")
            if not np.isfinite(numeric).any():
                continue
            if source == "metrics":
                if column.startswith("baseline_"):
                    group = "Baseline variable metrics"
                elif column.startswith("longitudinal_"):
                    group = "Longitudinal variable metrics"
                elif column.startswith("survival_"):
                    group = "Survival metrics"
                else:
                    continue
            elif source == "estimands":
                group = "Clinical estimands"
            else:
                group = "Privacy diagnostics"
            specs.append({
                "source": source,
                "group": group,
                "metric": _humanize_column(column),
                "column": column,
                "direction": _direction_for_column(column, source),
            })
    seen: set[tuple[str, str]] = set()
    unique_specs: list[dict[str, Any]] = []
    for spec in specs:
        key = (str(spec["source"]), str(spec["column"]))
        if key in seen:
            continue
        seen.add(key)
        unique_specs.append(spec)
    return unique_specs


def _summarize_specs(
    task: str,
    specs: list[dict[str, Any]],
    sources: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    parts = []
    for spec in specs:
        source = str(spec["source"])
        df = sources.get(source, pd.DataFrame())
        if df.empty:
            continue
        task_df = df.loc[df["benchmark_task"].eq(task)].copy()
        part = _summarize_metric(
            task_df,
            task=task,
            source=source,
            group=str(spec["group"]),
            metric=str(spec["metric"]),
            column=str(spec["column"]),
            direction=str(spec["direction"]),
        )
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _load_sources(exp1_dir: Path) -> dict[str, pd.DataFrame]:
    tables = exp1_dir / "tables"
    metrics = _add_derived_metrics(_read_csv(tables / "exp1_metrics_all_methods.csv"))
    estimands = _read_csv(tables / "exp1_estimands_all_methods.csv")
    privacy = _read_csv(tables / "exp1_privacy_all_methods.csv")
    return {"metrics": metrics, "estimands": estimands, "privacy": privacy}


def build_publication_tables(exp1_dir: Path, output_dir: Path | None = None, config_path: Path | None = None) -> dict[str, str]:
    exp1_dir = project_path(exp1_dir)
    output_dir = exp1_dir / "tables" / "publication" if output_dir is None else project_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = _load_sources(exp1_dir)
    detailed_specs = _detailed_metric_specs(sources["metrics"], sources["estimands"], sources["privacy"])
    artifacts: dict[str, str] = {}
    for task, slug in TASKS.items():
        curated = _summarize_specs(task, CURATED_METRICS, sources)
        detailed = _summarize_specs(task, detailed_specs, sources)
        curated_wide = _wide_from_long(curated)
        detailed_wide = _wide_from_long(detailed)

        for name, df in [
            (f"exp1_publication_{slug}_main_long.csv", curated),
            (f"exp1_publication_{slug}_main_wide.csv", curated_wide),
            (f"exp1_publication_{slug}_detailed_all_metrics_long.csv", detailed),
            (f"exp1_publication_{slug}_detailed_all_metrics_wide.csv", detailed_wide),
        ]:
            path = output_dir / name
            df.to_csv(path, index=False)
            artifacts[name] = str(path)

        _write_latex_table(
            curated_wide,
            output_dir / f"exp1_publication_{slug}_main_wide.tex",
            caption=f"Experiment 1 {TASK_TITLES[task].lower()} benchmark summary. Entries are mean (SD) over 500 replicates.",
            label=f"tab:exp1-{slug}-main",
        )
        artifacts[f"exp1_publication_{slug}_main_wide.tex"] = str(output_dir / f"exp1_publication_{slug}_main_wide.tex")
        _write_latex_table(
            detailed_wide,
            output_dir / f"exp1_publication_{slug}_detailed_all_metrics_wide.tex",
            caption=f"Experiment 1 {TASK_TITLES[task].lower()} detailed benchmark metrics. Entries are mean (SD) over 500 replicates.",
            label=f"tab:exp1-{slug}-detailed",
        )
        artifacts[f"exp1_publication_{slug}_detailed_all_metrics_wide.tex"] = str(output_dir / f"exp1_publication_{slug}_detailed_all_metrics_wide.tex")

    clinical_artifacts = _write_true_clinical_comparison_tables(exp1_dir, output_dir, sources, config_path)
    artifacts.update(clinical_artifacts)
    beamer_artifacts = _write_beamer_tables(output_dir)
    artifacts.update(beamer_artifacts)
    manifest = {
        "exp1_dir": str(exp1_dir),
        "output_dir": str(output_dir),
        "config_path": str(project_path(config_path)) if config_path is not None else str(_infer_config_path(exp1_dir)),
        "method_order": METHOD_ORDER,
        "tasks": TASKS,
        "notes": [
            "Mean and SD are computed across replicate-level rows.",
            "For rank_direction=zero, ranks use mean absolute value while method cells display signed mean (SD).",
            "Privacy diagnostics are descriptive because their preferred direction depends on the privacy threat model and conditioning design.",
            "Baseline-conditioned privacy exact duplicates are expected because held-out baseline rows are supplied to the generators.",
            "Clinical estimand comparison tables use observed held-out test-set estimands as true/reference values.",
            "Clinical estimand method cells are synthetic mean (SD); bias is synthetic-minus-true; RMSE is across 500 replicates.",
            "The Beamer short benchmark table is a slide-sized subset with PhaseSyn, the best non-PhaseSyn comparator, relative delta, and winner.",
        ],
        "artifacts": artifacts,
    }
    manifest_path = output_dir / "exp1_publication_table_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    artifacts["exp1_publication_table_manifest.json"] = str(manifest_path)
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build publication-ready Experiment 1 benchmark tables.")
    parser.add_argument(
        "--exp1-dir",
        type=Path,
        default=Path("outputs/pbc_experiments/experiment_20260617/exp1_control_arm"),
        help="Experiment 1 output directory containing tables/exp1_*_all_methods.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for publication tables. Defaults to EXP1_DIR/tables/publication.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="PBC core-4 config used to load the observed held-out test set for true clinical estimands.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_publication_tables(args.exp1_dir, args.output_dir, args.config)
    for name, path in sorted(artifacts.items()):
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
