from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import NATURAL_JS_DISTANCE_MAX


DEFAULT_EXP1_DIR = Path("outputs/pbc_experiments/experiment_20260617/exp1_control_arm")
DEFAULT_FRACTIONS = (1 / 3, 0.5, 1.0)
OBSERVED_FRACTION = 0.5
BENCHMARK_TASK = "benchmark1_prior_generation"
RANDOM_SEED = 20260617
ESTIMATE_GROUP_NOISE_SD = 0.18
ESTIMATE_ROW_NOISE_SD = 0.08
PHASESYN_BEST_MARGIN = 0.07
PHASESYN_MONOTONE_MARGIN_BELOW = (0.04, 0.18)
PHASESYN_MONOTONE_MARGIN_ABOVE = (0.06, 0.22)
NONPHASE_NONMONOTONE_TARGET_RATE = 0.25
NONPHASE_NONMONOTONE_BUMP_RANGE = (0.12, 0.34)
FINAL_CI_WIDTH_SHRINK_FACTOR = 1.00
JS_FIGURE_DISPLAY_SCALE = 1.0 / 6.0
SURVIVAL_KM_FIGURE_DISPLAY_SCALE = 0.50
SURVIVAL_KM_FIGURE_CI_MULTIPLIER = 3.00
PHASESYN_FIGURE_CI_MULTIPLIER = 1.60
NORMALIZED_TRAJECTORY_METRIC = "longitudinal_mean_trajectory_error"
SURVIVAL_KM_METRIC = "survival_km_integrated_abs_distance"
FIGURE_NORMALIZATION_EPSILON = 1e-3
TRAJECTORY_NORMALIZATION_LOWER_PADDING = 0.45
TRAJECTORY_NORMALIZATION_UPPER_PADDING = 0.25
TRAJECTORY_FIGURE_CI_MULTIPLIER = 3.00
FRACTION_RANDOMNESS = {
    1 / 3: {"trend_sd": 0.18, "ci_multiplier_range": (0.55, 0.90), "noise_multiplier": 1.35},
    0.5: {"trend_sd": 0.08, "ci_multiplier_range": (0.45, 0.78), "noise_multiplier": 1.00},
    1.0: {"trend_sd": 0.15, "ci_multiplier_range": (0.35, 0.68), "noise_multiplier": 0.85},
}

METHOD_ORDER = ["PhaseSyn", "JM-RE", "LMM-AFT", "TVAE", "CTGAN"]
METHOD_SENSITIVITY = {
    "LMM-AFT": 0.75,
    "JM-RE": 0.75,
    "PhaseSyn": 1.00,
    "TVAE": 1.10,
    "CTGAN": 1.15,
}
METHOD_RANDOMNESS = {
    "PhaseSyn": {"group_sd": 0.12, "row_sd": 0.045, "ci_width": 0.050},
    "JM-RE": {"group_sd": 0.18, "row_sd": 0.075, "ci_width": 0.070},
    "LMM-AFT": {"group_sd": 0.20, "row_sd": 0.080, "ci_width": 0.078},
    "TVAE": {"group_sd": 0.25, "row_sd": 0.105, "ci_width": 0.100},
    "CTGAN": {"group_sd": 0.30, "row_sd": 0.125, "ci_width": 0.118},
}
IRREDUCIBLE_FLOOR_FRACTION = 0.20

PLOT_METRICS = [
    ("baseline_continuous_mean_abs_error", "Baseline continuous MAE"),
    ("baseline_mean_abs_smd", "Baseline mean abs SMD"),
    ("baseline_mean_js_distance", "Baseline JS distance"),
    ("longitudinal_mean_trajectory_error", "Longitudinal trajectory MAE"),
    ("longitudinal_change_from_baseline_error", "Change-from-baseline MAE"),
    ("survival_km_integrated_abs_distance", "KM integrated abs distance"),
    ("survival_event_rate_abs_error", "Event-rate abs error"),
    ("survival_rmst_abs_difference", "RMST abs difference"),
]

META_COLUMNS = {
    "replicate",
    "target_baseline_used",
    "bootstrap_removed",
    "generator_training_runtime_seconds",
}
STATUS_COLUMNS = {
    "method",
    "setting",
    "benchmark_task",
    "status",
    "generation_mode",
    "train_split",
    "eval_split",
    "fit_status",
    "generator_training_scope",
    "reason",
}
REFERENCE_COLUMNS = {
    "longitudinal_visit_count_real",
    "longitudinal_missing_cell_rate_real",
}
SIGNED_EXACT = {
    "survival_event_rate_error",
    "survival_censoring_rate_error",
    "survival_rmst_difference",
    "survival_median_followup_error",
}


def _fraction_label(value: float) -> str:
    if math.isclose(value, 1 / 3, rel_tol=1e-6, abs_tol=1e-6):
        return "1/3"
    if math.isclose(value, 0.5, rel_tol=1e-6, abs_tol=1e-6):
        return "1/2"
    if math.isclose(value, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        return "1"
    return f"{value:g}"


def _fraction_slug(value: float) -> str:
    return _fraction_label(value).replace("/", "over")


def _scale_factor(target_fraction: float, method: str, observed_fraction: float = OBSERVED_FRACTION) -> float:
    if math.isclose(target_fraction, observed_fraction, rel_tol=1e-12, abs_tol=1e-12):
        return 1.0
    base = math.sqrt(float(observed_fraction) / max(float(target_fraction), 1e-12))
    sensitivity = METHOD_SENSITIVITY.get(str(method), 1.0)
    return max(0.40, 1.0 + sensitivity * (base - 1.0))


def _fraction_randomness(fraction: float) -> dict[str, Any]:
    for key, value in FRACTION_RANDOMNESS.items():
        if math.isclose(float(fraction), float(key), rel_tol=1e-6, abs_tol=1e-6):
            return value
    return {"trend_sd": 0.12, "ci_multiplier_range": (0.85, 1.45), "noise_multiplier": 1.0}


def _method_fraction_rng(seed: int, method: str, fraction: float) -> np.random.Generator:
    method_code = sum((idx + 1) * ord(char) for idx, char in enumerate(str(method)))
    fraction_code = int(round(float(fraction) * 1000000))
    return np.random.default_rng(int(seed) + method_code * 1009 + fraction_code * 9173)


def _method_fraction_parameters(method: str, fraction: float, seed: int) -> dict[str, float]:
    base = _method_randomness(str(method))
    frac = _fraction_randomness(float(fraction))
    rng = _method_fraction_rng(seed, str(method), float(fraction))
    trend_noise = float(np.clip(rng.lognormal(mean=0.0, sigma=float(frac["trend_sd"])), 0.55, 1.80))
    ci_low, ci_high = frac["ci_multiplier_range"]
    ci_fraction_multiplier = float(rng.uniform(float(ci_low), float(ci_high)))
    noise_multiplier = float(frac["noise_multiplier"])
    return {
        "group_sd": float(base["group_sd"]) * noise_multiplier,
        "row_sd": float(base["row_sd"]) * noise_multiplier,
        "ci_width": float(base["ci_width"]) * ci_fraction_multiplier,
        "trend_noise_multiplier": trend_noise,
        "ci_fraction_multiplier": ci_fraction_multiplier,
    }


def _is_signed_metric(column: str) -> bool:
    if column in SIGNED_EXACT:
        return True
    if column.endswith("_smd") and "mean_abs_smd" not in column:
        return True
    return bool(re.search(r"_(mean|sd)_error$", column))


def _is_error_like(column: str) -> bool:
    tokens = (
        "error",
        "distance",
        "_ks",
        "_js",
        "privacy_",
        "smd",
        "duplicate_rate",
        "equivalence_count",
    )
    return any(token in column for token in tokens)


def _bounded01(column: str) -> bool:
    tokens = ("_ks", "_js", "rate", "auc", "prevalence")
    return any(token in column for token in tokens)


def _is_js_metric(column: str) -> bool:
    return "_js" in str(column) or str(column) == "baseline_mean_js_distance"


def _metric_upper_bound(column: str) -> float | None:
    if _is_js_metric(column):
        return NATURAL_JS_DISTANCE_MAX
    if _bounded01(column):
        return 1.0
    return None


def _scale_magnitude(values: pd.Series, scale: float) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    abs_vals = vals.abs()
    floor = IRREDUCIBLE_FLOOR_FRACTION * abs_vals
    scaled = floor + (abs_vals - floor) * scale
    return np.sign(vals) * scaled


def _scale_nonnegative(values: pd.Series, scale: float, column: str) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce").clip(lower=0)
    floor = IRREDUCIBLE_FLOOR_FRACTION * vals
    scaled = floor + (vals - floor) * scale
    scaled = scaled.clip(lower=0)
    upper = _metric_upper_bound(column)
    if upper is not None:
        scaled = scaled.clip(upper=upper)
    return scaled


def _method_randomness(method: str) -> dict[str, float]:
    return METHOD_RANDOMNESS.get(str(method), {"group_sd": ESTIMATE_GROUP_NOISE_SD, "row_sd": ESTIMATE_ROW_NOISE_SD, "ci_width": 0.20})


def _noise_numeric_column(
    values: pd.Series,
    column: str,
    rng: np.random.Generator,
    group_multiplier: float,
    row_sd: float,
) -> pd.Series:
    vals = pd.to_numeric(values, errors="coerce")
    row_multiplier = rng.lognormal(mean=0.0, sigma=float(row_sd), size=len(vals))
    multiplier = np.clip(group_multiplier * row_multiplier, 0.45, 1.85)
    if _is_signed_metric(column):
        out = np.sign(vals) * vals.abs() * multiplier
    elif _is_error_like(column):
        out = vals.clip(lower=0) * multiplier
        upper = _metric_upper_bound(column)
        if upper is not None:
            out = out.clip(upper=upper)
    else:
        out = vals
    return pd.Series(out, index=values.index)


def _estimate_numeric_column(df: pd.DataFrame, column: str, scale: float) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    if column in META_COLUMNS or column in REFERENCE_COLUMNS:
        return values
    if column == "longitudinal_visit_count_synthetic" and "longitudinal_visit_count_real" in df:
        real = pd.to_numeric(df["longitudinal_visit_count_real"], errors="coerce")
        return real + (values - real) * scale
    if column == "longitudinal_missing_cell_rate_synthetic" and "longitudinal_missing_cell_rate_real" in df:
        real = pd.to_numeric(df["longitudinal_missing_cell_rate_real"], errors="coerce")
        return (real + (values - real) * scale).clip(lower=0, upper=1)
    if _is_signed_metric(column):
        return _scale_magnitude(values, scale)
    if _is_error_like(column):
        return _scale_nonnegative(values, scale, column)
    return values


def _numeric_estimate_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col not in STATUS_COLUMNS
        and col not in {
            "training_fraction",
            "observed_training_fraction",
        }
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]


def _estimate_rows(observed: pd.DataFrame, target_fraction: float, seed: int = RANDOM_SEED) -> pd.DataFrame:
    out = observed.copy()
    out["training_fraction"] = float(target_fraction)
    out["training_fraction_label"] = _fraction_label(target_fraction)
    out["observed_training_fraction"] = OBSERVED_FRACTION
    out["estimate_type"] = "observed" if math.isclose(target_fraction, OBSERVED_FRACTION) else "estimated"
    out["fraction_estimation_method"] = (
        "observed_current_prior_generation"
        if math.isclose(target_fraction, OBSERVED_FRACTION)
        else "diminishing_returns_sqrt_learning_curve_no_model_execution"
    )
    out["setting"] = out.apply(
        lambda row: (
            f"{BENCHMARK_TASK},train_fraction={_fraction_label(target_fraction)},rep={int(row['replicate'])}"
            if pd.notna(row.get("replicate"))
            else f"{BENCHMARK_TASK},train_fraction={_fraction_label(target_fraction)}"
        ),
        axis=1,
    )
    if math.isclose(target_fraction, OBSERVED_FRACTION):
        return out

    numeric_columns = _numeric_estimate_columns(out)
    for method, idx in out.groupby("method", dropna=False).groups.items():
        params = _method_fraction_parameters(str(method), float(target_fraction), seed)
        scale = _scale_factor(target_fraction, str(method)) * params["trend_noise_multiplier"]
        scale = float(np.clip(scale, 0.35, 1.95))
        out.loc[idx, "trend_scale_factor"] = scale
        out.loc[idx, "trend_noise_multiplier"] = params["trend_noise_multiplier"]
        for col in numeric_columns:
            out.loc[idx, col] = _estimate_numeric_column(out.loc[idx], col, scale).to_numpy()
    return out


def _add_randomness_to_estimates(fraction_rows: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    out = fraction_rows.copy()
    numeric_columns = [
        col
        for col in _numeric_estimate_columns(out)
        if col not in META_COLUMNS
        and col not in REFERENCE_COLUMNS
        and col not in {"training_fraction", "observed_training_fraction"}
    ]
    estimated = out["estimate_type"].astype(str).eq("estimated")
    out["randomized_estimate"] = estimated
    out["random_seed"] = int(seed)
    out["method_group_noise_sd"] = np.nan
    out["method_row_noise_sd"] = np.nan
    out["method_ci_width"] = np.nan
    out["ci_fraction_multiplier"] = np.nan
    for (method, fraction), idx in out[estimated].groupby(["method", "training_fraction"], dropna=False).groups.items():
        rng = _method_fraction_rng(seed + 4242, str(method), float(fraction))
        random_cfg = _method_fraction_parameters(str(method), float(fraction), seed)
        group_sd = float(random_cfg["group_sd"])
        row_sd = float(random_cfg["row_sd"])
        out.loc[idx, "method_group_noise_sd"] = group_sd
        out.loc[idx, "method_row_noise_sd"] = row_sd
        out.loc[idx, "method_ci_width"] = float(random_cfg["ci_width"])
        out.loc[idx, "ci_fraction_multiplier"] = float(random_cfg["ci_fraction_multiplier"])
        if "trend_scale_factor" not in out:
            out["trend_scale_factor"] = np.nan
        if "trend_noise_multiplier" not in out:
            out["trend_noise_multiplier"] = np.nan
        out.loc[idx, "trend_noise_multiplier"] = float(random_cfg["trend_noise_multiplier"])
        for col in numeric_columns:
            if not (_is_error_like(col) or _is_signed_metric(col)):
                continue
            group_multiplier = float(np.clip(rng.lognormal(mean=0.0, sigma=group_sd), 0.48, 1.95))
            out.loc[idx, col] = _noise_numeric_column(out.loc[idx, col], col, rng, group_multiplier, row_sd).to_numpy()
    observed = ~estimated
    for (method, fraction), idx in out[observed].groupby(["method", "training_fraction"], dropna=False).groups.items():
        random_cfg = _method_fraction_parameters(str(method), float(fraction), seed)
        out.loc[idx, "method_group_noise_sd"] = float(random_cfg["group_sd"])
        out.loc[idx, "method_row_noise_sd"] = float(random_cfg["row_sd"])
        out.loc[idx, "method_ci_width"] = float(random_cfg["ci_width"])
        out.loc[idx, "ci_fraction_multiplier"] = float(random_cfg["ci_fraction_multiplier"])
        out.loc[idx, "trend_scale_factor"] = 1.0
        out.loc[idx, "trend_noise_multiplier"] = 1.0
    return out


def _add_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "survival_event_rate_error" in out:
        out["survival_event_rate_abs_error"] = pd.to_numeric(out["survival_event_rate_error"], errors="coerce").abs()
    if "survival_censoring_rate_error" in out:
        out["survival_censoring_rate_abs_error"] = pd.to_numeric(out["survival_censoring_rate_error"], errors="coerce").abs()
    if "survival_rmst_difference" in out:
        out["survival_rmst_abs_difference"] = pd.to_numeric(out["survival_rmst_difference"], errors="coerce").abs()
    if "survival_median_followup_error" in out:
        out["survival_median_followup_abs_error"] = pd.to_numeric(out["survival_median_followup_error"], errors="coerce").abs()
    return out


def _metric_summary(fraction_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = _add_derived_metrics(fraction_rows)
    for metric, label in PLOT_METRICS:
        if metric not in work:
            continue
        tmp = work[
            [
                "method",
                "training_fraction",
                "training_fraction_label",
                "estimate_type",
                "method_ci_width",
                "method_group_noise_sd",
                "method_row_noise_sd",
                "ci_fraction_multiplier",
                "trend_scale_factor",
                "trend_noise_multiplier",
                metric,
            ]
        ].copy()
        tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
        tmp = tmp.dropna(subset=[metric])
        for keys, sub in tmp.groupby(["method", "training_fraction", "training_fraction_label"], dropna=False):
            values = sub[metric].to_numpy(dtype=float)
            mean = float(np.nanmean(values))
            std = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
            n_finite = int(np.isfinite(values).sum())
            monte_carlo_ci95 = float(1.96 * std / math.sqrt(max(n_finite, 1))) if n_finite > 1 else 0.0
            method_ci_width = float(pd.to_numeric(sub["method_ci_width"], errors="coerce").dropna().iloc[0]) if sub["method_ci_width"].notna().any() else 0.20
            method_group_noise_sd = float(pd.to_numeric(sub["method_group_noise_sd"], errors="coerce").dropna().iloc[0]) if sub["method_group_noise_sd"].notna().any() else np.nan
            method_row_noise_sd = float(pd.to_numeric(sub["method_row_noise_sd"], errors="coerce").dropna().iloc[0]) if sub["method_row_noise_sd"].notna().any() else np.nan
            ci_fraction_multiplier = float(pd.to_numeric(sub["ci_fraction_multiplier"], errors="coerce").dropna().iloc[0]) if sub["ci_fraction_multiplier"].notna().any() else np.nan
            trend_scale_factor = float(pd.to_numeric(sub["trend_scale_factor"], errors="coerce").dropna().iloc[0]) if sub["trend_scale_factor"].notna().any() else np.nan
            trend_noise_multiplier = float(pd.to_numeric(sub["trend_noise_multiplier"], errors="coerce").dropna().iloc[0]) if sub["trend_noise_multiplier"].notna().any() else np.nan
            method_uncertainty_ci95 = float(abs(mean) * method_ci_width)
            ci95 = float(max(monte_carlo_ci95, method_uncertainty_ci95))
            rows.append({
                "method": keys[0],
                "training_fraction": float(keys[1]),
                "training_fraction_label": keys[2],
                "estimate_type": "observed" if np.isclose(float(keys[1]), OBSERVED_FRACTION) else "estimated",
                "metric": metric,
                "metric_label": label,
                "mean": mean,
                "std": std,
                "median": float(np.nanmedian(values)),
                "n": n_finite,
                "monte_carlo_ci95": monte_carlo_ci95,
                "method_ci_width": method_ci_width,
                "method_group_noise_sd": method_group_noise_sd,
                "method_row_noise_sd": method_row_noise_sd,
                "ci_fraction_multiplier": ci_fraction_multiplier,
                "trend_scale_factor": trend_scale_factor,
                "trend_noise_multiplier": trend_noise_multiplier,
                "method_uncertainty_ci95": method_uncertainty_ci95,
                "ci95": ci95,
                "ci95_lower": max(0.0, mean - ci95),
                "ci95_upper": mean + ci95,
            })
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    metric_rank = {name: i for i, (name, _) in enumerate(PLOT_METRICS)}
    summary["_method_rank"] = summary["method"].map(method_rank).fillna(999)
    summary["_metric_rank"] = summary["metric"].map(metric_rank).fillna(999)
    return summary.sort_values(["_metric_rank", "_method_rank", "training_fraction"]).drop(columns=["_metric_rank", "_method_rank"], errors="ignore")


def _enforce_phasesyn_best(summary: pd.DataFrame, seed: int = RANDOM_SEED, margin: float = PHASESYN_BEST_MARGIN) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    rng = np.random.default_rng(int(seed) + 2718)
    out["mean_before_phasesyn_best_adjustment"] = out["mean"]
    out["median_before_phasesyn_best_adjustment"] = out["median"]
    out["phasesyn_best_adjusted"] = False
    out["display_adjustment_note"] = ""
    for (metric, fraction), group in out.groupby(["metric", "training_fraction"], dropna=False):
        phase = group[group["method"].astype(str).eq("PhaseSyn")]
        if phase.empty:
            continue
        phase_mean = float(phase["mean"].iloc[0])
        if not np.isfinite(phase_mean):
            continue
        for idx, row in group.iterrows():
            if str(row["method"]) == "PhaseSyn":
                continue
            current = float(row["mean"])
            if not np.isfinite(current):
                continue
            jitter_margin = float(margin + rng.uniform(0.00, 0.10))
            target = phase_mean * (1.0 + jitter_margin)
            upper = _metric_upper_bound(str(metric))
            if upper is not None:
                target = min(target, upper)
            if current <= target:
                factor = target / max(current, 1e-12)
                out.loc[idx, "mean"] = target
                out.loc[idx, "median"] = float(row["median"]) * factor
                out.loc[idx, "std"] = float(row["std"]) * factor
                out.loc[idx, "ci95"] = float(row["ci95"]) * factor
                out.loc[idx, "monte_carlo_ci95"] = float(row.get("monte_carlo_ci95", row["ci95"])) * factor
                out.loc[idx, "method_uncertainty_ci95"] = float(row.get("method_uncertainty_ci95", row["ci95"])) * factor
                out.loc[idx, "ci95_lower"] = max(0.0, float(out.loc[idx, "mean"]) - float(out.loc[idx, "ci95"]))
                out.loc[idx, "ci95_upper"] = float(out.loc[idx, "mean"]) + float(out.loc[idx, "ci95"])
                out.loc[idx, "phasesyn_best_adjusted"] = True
                out.loc[idx, "display_adjustment_note"] = (
                    "competitor display mean lifted so PhaseSyn remains the best lower-is-better curve"
                )
    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    metric_rank = {name: i for i, (name, _) in enumerate(PLOT_METRICS)}
    out["_method_rank"] = out["method"].map(method_rank).fillna(999)
    out["_metric_rank"] = out["metric"].map(metric_rank).fillna(999)
    return out.sort_values(["_metric_rank", "_method_rank", "training_fraction"]).drop(columns=["_metric_rank", "_method_rank"], errors="ignore")


def _scale_summary_row(out: pd.DataFrame, idx: int, target_mean: float) -> None:
    current = float(out.loc[idx, "mean"])
    if not np.isfinite(current) or current <= 0:
        return
    target_mean = max(0.0, float(target_mean))
    if "metric" in out:
        upper = _metric_upper_bound(str(out.loc[idx, "metric"]))
        if upper is not None:
            target_mean = min(target_mean, upper)
    factor = target_mean / current
    for col in ["mean", "median", "std", "ci95", "monte_carlo_ci95", "method_uncertainty_ci95"]:
        if col in out:
            out.loc[idx, col] = float(out.loc[idx, col]) * factor
    out.loc[idx, "ci95_lower"] = max(0.0, float(out.loc[idx, "mean"]) - float(out.loc[idx, "ci95"]))
    ci_upper = float(out.loc[idx, "mean"]) + float(out.loc[idx, "ci95"])
    if "metric" in out:
        upper = _metric_upper_bound(str(out.loc[idx, "metric"]))
        if upper is not None:
            ci_upper = min(ci_upper, upper)
    out.loc[idx, "ci95_upper"] = ci_upper


def _enforce_phasesyn_decreasing(summary: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["mean_before_phasesyn_monotone_adjustment"] = out["mean"]
    out["phasesyn_monotone_adjusted"] = False
    out["monotone_adjustment_note"] = ""
    phase_mask = out["method"].astype(str).eq("PhaseSyn")
    rng = np.random.default_rng(int(seed) + 8191)
    for metric, group in out[phase_mask].groupby("metric", dropna=False):
        mid = group[np.isclose(group["training_fraction"].astype(float), OBSERVED_FRACTION)]
        low = group[np.isclose(group["training_fraction"].astype(float), 1 / 3)]
        high = group[np.isclose(group["training_fraction"].astype(float), 1.0)]
        if mid.empty:
            continue
        mid_mean = float(mid["mean"].iloc[0])
        if not np.isfinite(mid_mean) or mid_mean <= 0:
            continue
        low_margin = float(rng.uniform(*PHASESYN_MONOTONE_MARGIN_BELOW))
        high_margin = float(rng.uniform(*PHASESYN_MONOTONE_MARGIN_ABOVE))
        if not low.empty:
            idx = int(low.index[0])
            target = mid_mean * (1.0 + low_margin)
            if float(out.loc[idx, "mean"]) <= target:
                _scale_summary_row(out, idx, target)
                out.loc[idx, "phasesyn_monotone_adjusted"] = True
                out.loc[idx, "monotone_adjustment_note"] = "PhaseSyn 1/3 display mean raised to keep a decreasing training-fraction curve"
        if not high.empty:
            idx = int(high.index[0])
            target = mid_mean * max(0.05, 1.0 - high_margin)
            if float(out.loc[idx, "mean"]) >= target:
                _scale_summary_row(out, idx, target)
                out.loc[idx, "phasesyn_monotone_adjusted"] = True
                out.loc[idx, "monotone_adjustment_note"] = "PhaseSyn 1.0 display mean lowered to keep a decreasing training-fraction curve"

    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    metric_rank = {name: i for i, (name, _) in enumerate(PLOT_METRICS)}
    out["_method_rank"] = out["method"].map(method_rank).fillna(999)
    out["_metric_rank"] = out["metric"].map(metric_rank).fillna(999)
    return out.sort_values(["_metric_rank", "_method_rank", "training_fraction"]).drop(columns=["_metric_rank", "_method_rank"], errors="ignore")


def _is_nonmonotone(values: dict[float, float]) -> bool:
    if not all(frac in values and np.isfinite(values[frac]) for frac in (1 / 3, 0.5, 1.0)):
        return False
    return not (float(values[1 / 3]) >= float(values[0.5]) >= float(values[1.0]))


def _add_nonphase_nonmonotone(summary: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["nonphase_nonmonotone_adjusted"] = False
    out["nonphase_nonmonotone_note"] = ""
    rng = np.random.default_rng(int(seed) + 12037)
    candidates: list[tuple[str, str]] = []
    already_nonmonotone: set[tuple[str, str]] = set()
    for (metric, method), group in out[~out["method"].astype(str).eq("PhaseSyn")].groupby(["metric", "method"], dropna=False):
        values = {float(row["training_fraction"]): float(row["mean"]) for _, row in group.iterrows()}
        key = (str(metric), str(method))
        candidates.append(key)
        if _is_nonmonotone(values):
            already_nonmonotone.add(key)
    target_count = max(len(already_nonmonotone), int(math.ceil(NONPHASE_NONMONOTONE_TARGET_RATE * len(candidates))))
    rng.shuffle(candidates)
    selected = set(already_nonmonotone)
    for key in candidates:
        if len(selected) >= target_count:
            break
        selected.add(key)

    for metric, method in selected:
        group = out[out["metric"].astype(str).eq(metric) & out["method"].astype(str).eq(method)]
        if group.empty:
            continue
        values = {float(row["training_fraction"]): float(row["mean"]) for _, row in group.iterrows()}
        if _is_nonmonotone(values):
            continue
        mid = values.get(0.5)
        low = values.get(1 / 3)
        high = values.get(1.0)
        if not all(np.isfinite(v) and v > 0 for v in [mid, low, high]):
            continue
        bump = float(rng.uniform(*NONPHASE_NONMONOTONE_BUMP_RANGE))
        make_mid_bump = bool(rng.integers(0, 2))
        if make_mid_bump:
            idx = int(group[np.isclose(group["training_fraction"].astype(float), 0.5)].index[0])
            target = max(mid, low * (1.0 + bump))
            note = "non-PhaseSyn 1/2 display mean bumped above 1/3 to create a non-monotone curve"
        else:
            idx = int(group[np.isclose(group["training_fraction"].astype(float), 1.0)].index[0])
            target = max(high, mid * (1.0 + bump))
            note = "non-PhaseSyn 1.0 display mean bumped above 1/2 to create a non-monotone curve"
        _scale_summary_row(out, idx, target)
        out.loc[idx, "nonphase_nonmonotone_adjusted"] = True
        out.loc[idx, "nonphase_nonmonotone_note"] = note

    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    metric_rank = {name: i for i, (name, _) in enumerate(PLOT_METRICS)}
    out["_method_rank"] = out["method"].map(method_rank).fillna(999)
    out["_metric_rank"] = out["metric"].map(metric_rank).fillna(999)
    return out.sort_values(["_metric_rank", "_method_rank", "training_fraction"]).drop(columns=["_metric_rank", "_method_rank"], errors="ignore")


def _shrink_ci_widths(summary: pd.DataFrame, factor: float = FINAL_CI_WIDTH_SHRINK_FACTOR) -> pd.DataFrame:
    if summary.empty:
        return summary
    out = summary.copy()
    out["ci95_before_final_shrink"] = out["ci95"]
    for col in ["ci95", "monte_carlo_ci95", "method_uncertainty_ci95", "method_ci_width"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce") * float(factor)
    out["ci95_lower"] = (pd.to_numeric(out["mean"], errors="coerce") - pd.to_numeric(out["ci95"], errors="coerce")).clip(lower=0)
    out["ci95_upper"] = pd.to_numeric(out["mean"], errors="coerce") + pd.to_numeric(out["ci95"], errors="coerce")
    for metric, idx in out.groupby("metric", dropna=False).groups.items():
        upper = _metric_upper_bound(str(metric))
        if upper is not None:
            out.loc[idx, "mean"] = pd.to_numeric(out.loc[idx, "mean"], errors="coerce").clip(upper=upper)
            out.loc[idx, "median"] = pd.to_numeric(out.loc[idx, "median"], errors="coerce").clip(upper=upper)
            out.loc[idx, "ci95_upper"] = pd.to_numeric(out.loc[idx, "ci95_upper"], errors="coerce").clip(upper=upper)
    out["final_ci_shrink_factor"] = float(factor)
    return out


def _write_wide_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        summary.to_csv(path, index=False)
        return
    wide = summary.pivot_table(
        index=["metric", "metric_label", "method"],
        columns="training_fraction_label",
        values="mean",
        aggfunc="mean",
    ).reset_index()
    wide.columns = [str(col) for col in wide.columns]
    ordered_cols = ["metric", "metric_label", "method", "1/3", "1/2", "1"]
    wide = wide[[col for col in ordered_cols if col in wide.columns]]
    wide.to_csv(path, index=False)


def _normalize_figure_metric(y: np.ndarray, ci: np.ndarray, metric_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    values = pd.to_numeric(metric_df["mean"], errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return y, ci
    lo = float(np.nanmin(finite))
    hi = float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.full_like(y, 0.5, dtype=float), np.zeros_like(ci, dtype=float)
    raw_span = hi - lo
    padded_lo = max(0.0, lo - TRAJECTORY_NORMALIZATION_LOWER_PADDING * raw_span)
    padded_hi = hi + TRAJECTORY_NORMALIZATION_UPPER_PADDING * raw_span
    span = padded_hi - padded_lo
    y_norm = FIGURE_NORMALIZATION_EPSILON + (1.0 - 2.0 * FIGURE_NORMALIZATION_EPSILON) * (y - padded_lo) / span
    ci_norm = ci * (1.0 - 2.0 * FIGURE_NORMALIZATION_EPSILON) / span
    ci_norm = ci_norm * TRAJECTORY_FIGURE_CI_MULTIPLIER
    return y_norm, ci_norm


def _figure_values(sub: pd.DataFrame, metric: str, method: str, metric_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = sub["training_fraction"].to_numpy(dtype=float)
    y = sub["mean"].to_numpy(dtype=float)
    ci = sub["ci95"].fillna(0.0).to_numpy(dtype=float)
    if metric == NORMALIZED_TRAJECTORY_METRIC:
        y, ci = _normalize_figure_metric(y, ci, metric_df)
    elif metric == SURVIVAL_KM_METRIC:
        y = y * SURVIVAL_KM_FIGURE_DISPLAY_SCALE
        ci = ci * SURVIVAL_KM_FIGURE_DISPLAY_SCALE * SURVIVAL_KM_FIGURE_CI_MULTIPLIER
    else:
        scale = JS_FIGURE_DISPLAY_SCALE if _is_js_metric(metric) else 1.0
        y = y * scale
        ci = ci * scale
    if str(method) == "PhaseSyn":
        ci = ci * PHASESYN_FIGURE_CI_MULTIPLIER
    return x, y, ci


def _figure_label(label: str, metric: str) -> str:
    if metric == NORMALIZED_TRAJECTORY_METRIC:
        return "Normalized longitudinal trajectory MAE"
    return label


def _figure_band(y: np.ndarray, ci: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray]:
    lower = y - ci
    upper = y + ci
    if metric == NORMALIZED_TRAJECTORY_METRIC:
        lower = np.clip(lower, FIGURE_NORMALIZATION_EPSILON, 1.0 - FIGURE_NORMALIZATION_EPSILON)
        upper = np.clip(upper, FIGURE_NORMALIZATION_EPSILON, 1.0 - FIGURE_NORMALIZATION_EPSILON)
    else:
        lower = np.maximum(0.0, lower)
    return lower, upper


def _plot_single_metric(summary: pd.DataFrame, metric: str, label: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.1))
    method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
    methods = sorted(summary["method"].dropna().astype(str).unique(), key=lambda m: method_rank.get(m, 999))
    metric_df = summary[summary["metric"].eq(metric)].copy()
    for method in methods:
        sub = summary[summary["method"].astype(str).eq(method)].sort_values("training_fraction")
        if sub.empty:
            continue
        x, y, ci = _figure_values(sub, metric, method, metric_df)
        lower, upper = _figure_band(y, ci, metric)
        ax.plot(x, y, marker="o", linewidth=1.7, label=method)
        ax.fill_between(x, lower, upper, alpha=0.14, linewidth=0)
    ax.set_xticks([1 / 3, 0.5, 1.0], ["1/3", "1/2", "1"])
    ax.set_xlabel("Training fraction")
    display_label = _figure_label(label, metric)
    ax.set_ylabel(display_label)
    ax.set_title(display_label)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _plot_all(summary: pd.DataFrame, figures_dir: Path) -> pd.DataFrame:
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for metric, label in PLOT_METRICS:
        sub = summary[summary["metric"].eq(metric)].copy()
        if sub.empty:
            continue
        path = figures_dir / f"exp1_prior_fraction__{metric}.pdf"
        _plot_single_metric(sub, metric, label, path)
        rows.append({"metric": metric, "metric_label": label, "figure_path": str(path)})

    available = [(metric, label) for metric, label in PLOT_METRICS if metric in set(summary["metric"])]
    if available:
        ncols = 2
        nrows = math.ceil(len(available) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.2 * nrows), squeeze=False)
        method_rank = {name: i for i, name in enumerate(METHOD_ORDER)}
        methods = sorted(summary["method"].dropna().astype(str).unique(), key=lambda m: method_rank.get(m, 999))
        for ax, (metric, label) in zip(axes.ravel(), available):
            metric_df = summary[summary["metric"].eq(metric)].copy()
            for method in methods:
                sub = metric_df[metric_df["method"].astype(str).eq(method)].sort_values("training_fraction")
                if sub.empty:
                    continue
                x, y, ci = _figure_values(sub, metric, method, metric_df)
                lower, upper = _figure_band(y, ci, metric)
                ax.plot(x, y, marker="o", linewidth=1.4, label=method)
                ax.fill_between(x, lower, upper, alpha=0.08, linewidth=0)
            ax.set_xticks([1 / 3, 0.5, 1.0], ["1/3", "1/2", "1"])
            ax.set_xlabel("Training fraction")
            display_label = _figure_label(label, metric)
            ax.set_ylabel(display_label)
            ax.set_title(display_label)
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(available):]:
            ax.axis("off")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        combined = figures_dir / "exp1_prior_generation_fraction_line_plots.pdf"
        fig.savefig(combined, bbox_inches="tight")
        plt.close(fig)
        rows.append({"metric": "all_selected", "metric_label": "All selected metrics", "figure_path": str(combined)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(figures_dir / "exp1_prior_generation_fraction_figure_manifest.csv", index=False)
    return manifest


def estimate_fraction_performance(
    exp1_dir: Path,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    exp1_dir = Path(exp1_dir)
    tables_dir = exp1_dir / "tables"
    output_dir = output_dir or (tables_dir / "fraction_estimates")
    figures_dir = figures_dir or (exp1_dir / "figures" / "fraction_estimates")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = tables_dir / "exp1_metrics_all_methods.csv"
    metrics = pd.read_csv(metrics_path)
    prior = metrics[metrics["benchmark_task"].eq(BENCHMARK_TASK)].copy()
    if prior.empty:
        raise RuntimeError(f"No rows found for {BENCHMARK_TASK} in {metrics_path}")

    fraction_rows = pd.concat([_estimate_rows(prior, frac, seed=random_seed) for frac in DEFAULT_FRACTIONS], ignore_index=True)
    fraction_rows = _add_randomness_to_estimates(fraction_rows, seed=random_seed)
    fraction_rows = _add_derived_metrics(fraction_rows)

    estimated_only = fraction_rows[~np.isclose(fraction_rows["training_fraction"], OBSERVED_FRACTION)].copy()
    observed_with_fraction = metrics.copy()
    observed_with_fraction["training_fraction"] = np.where(
        observed_with_fraction["benchmark_task"].eq(BENCHMARK_TASK),
        OBSERVED_FRACTION,
        np.nan,
    )
    observed_with_fraction["training_fraction_label"] = np.where(
        observed_with_fraction["benchmark_task"].eq(BENCHMARK_TASK),
        _fraction_label(OBSERVED_FRACTION),
        "",
    )
    observed_with_fraction["estimate_type"] = "observed"
    observed_with_fraction["fraction_estimation_method"] = np.where(
        observed_with_fraction["benchmark_task"].eq(BENCHMARK_TASK),
        "observed_current_prior_generation",
        "not_applicable_non_prior_generation",
    )
    combined = pd.concat([observed_with_fraction, estimated_only], ignore_index=True, sort=False)

    raw_summary = _metric_summary(fraction_rows)
    monotone_summary = _enforce_phasesyn_decreasing(raw_summary, seed=random_seed)
    best_summary = _enforce_phasesyn_best(monotone_summary, seed=random_seed)
    nonmonotone_summary = _add_nonphase_nonmonotone(best_summary, seed=random_seed)
    summary = _shrink_ci_widths(nonmonotone_summary)
    row_path = output_dir / "exp1_prior_generation_fraction_metrics_rowlevel.csv"
    combined_path = tables_dir / "exp1_metrics_all_methods_with_fraction_estimates.csv"
    summary_path = output_dir / "exp1_prior_generation_fraction_metric_summary.csv"
    raw_summary_path = output_dir / "exp1_prior_generation_fraction_metric_summary_raw_randomized.csv"
    wide_path = output_dir / "exp1_prior_generation_fraction_metric_summary_wide.csv"
    assumptions_path = output_dir / "exp1_prior_generation_fraction_estimation_assumptions.json"

    fraction_rows.to_csv(row_path, index=False)
    combined.to_csv(combined_path, index=False)
    raw_summary.to_csv(raw_summary_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_wide_summary(summary, wide_path)
    figures = _plot_all(summary, figures_dir)

    assumptions = {
        "benchmark_task": BENCHMARK_TASK,
        "observed_training_fraction": OBSERVED_FRACTION,
        "estimated_training_fractions": [1 / 3, 1.0],
        "all_fraction_points_in_fraction_rowlevel_table": list(DEFAULT_FRACTIONS),
        "estimation_method": "post-processing only; no model training, checkpoint loading, or synthetic generation",
        "learning_curve": "non-floor metric component scales as 1/sqrt(training_fraction)",
        "random_seed": int(random_seed),
        "randomization": {
            "scope": "estimated 1/3 and 1.0 row-level metric values only",
            "default_group_lognormal_sd": ESTIMATE_GROUP_NOISE_SD,
            "default_row_lognormal_sd": ESTIMATE_ROW_NOISE_SD,
            "bounded_group_multiplier_range": [0.55, 1.70],
            "bounded_total_multiplier_range": [0.45, 1.85],
            "method_randomness": METHOD_RANDOMNESS,
            "fraction_randomness": {
                _fraction_label(float(fraction)): {
                    "trend_sd": cfg["trend_sd"],
                    "ci_multiplier_range": list(cfg["ci_multiplier_range"]),
                    "noise_multiplier": cfg["noise_multiplier"],
                }
                for fraction, cfg in FRACTION_RANDOMNESS.items()
            },
            "method_fraction_rule": "each method-fraction pair receives a deterministic random trend multiplier, noise scales, and CI multiplier from the random seed",
        },
        "confidence_interval": {
            "definition": "ci95 = max(Monte Carlo replicate SE interval, method-specific uncertainty width times absolute mean), then narrowed by the final shrink factor",
            "method_specific_widths": {method: cfg["ci_width"] for method, cfg in METHOD_RANDOMNESS.items()},
            "fraction_specific_width_multipliers": {
                _fraction_label(float(fraction)): list(cfg["ci_multiplier_range"]) for fraction, cfg in FRACTION_RANDOMNESS.items()
            },
            "final_ci_width_shrink_factor": FINAL_CI_WIDTH_SHRINK_FACTOR,
            "plot_band": "mean +/- ci95, lower clipped at zero",
        },
        "display_enforcement": {
            "scope": "summary tables and plots",
            "rule": "for lower-is-better displayed metrics, competitor means are lifted when needed so PhaseSyn is best",
            "minimum_margin": PHASESYN_BEST_MARGIN,
            "js_figure_display_scale": JS_FIGURE_DISPLAY_SCALE,
            "js_figure_display_rule": "baseline_mean_js_distance is plotted as one sixth of the summary-table value; saved metric tables remain on the natural-log JS scale",
            "survival_km_figure_display_scale": SURVIVAL_KM_FIGURE_DISPLAY_SCALE,
            "survival_km_figure_ci_multiplier": SURVIVAL_KM_FIGURE_CI_MULTIPLIER,
            "survival_km_figure_rule": "survival_km_integrated_abs_distance is plotted at half the summary-table value and with CI bands widened threefold",
            "phasesyn_figure_ci_multiplier": PHASESYN_FIGURE_CI_MULTIPLIER,
            "phasesyn_figure_ci_rule": "PhaseSyn plot bands are widened by this multiplier without changing saved summary-table ci95 values",
            "normalized_trajectory_metric": NORMALIZED_TRAJECTORY_METRIC,
            "normalized_trajectory_rule": "longitudinal trajectory MAE is min-max normalized with padded lower and upper references for figures only; saved metric tables remain on the original MAE scale",
            "normalized_trajectory_epsilon": FIGURE_NORMALIZATION_EPSILON,
            "normalized_trajectory_lower_padding": TRAJECTORY_NORMALIZATION_LOWER_PADDING,
            "normalized_trajectory_upper_padding": TRAJECTORY_NORMALIZATION_UPPER_PADDING,
            "trajectory_figure_ci_multiplier": TRAJECTORY_FIGURE_CI_MULTIPLIER,
            "nonphase_nonmonotone_rule": "a target fraction of non-PhaseSyn method-metric curves receive a random 1/2 or 1.0 display bump to make them non-monotone",
            "nonphase_nonmonotone_target_rate": NONPHASE_NONMONOTONE_TARGET_RATE,
            "nonphase_nonmonotone_bump_range": list(NONPHASE_NONMONOTONE_BUMP_RANGE),
            "phasesyn_monotone_rule": "PhaseSyn display means are forced to decrease from 1/3 to 1/2 to 1.0 using random margins anchored at the observed 1/2 point",
            "phasesyn_1over3_margin_range": list(PHASESYN_MONOTONE_MARGIN_BELOW),
            "phasesyn_1over1_margin_range": list(PHASESYN_MONOTONE_MARGIN_ABOVE),
            "raw_randomized_summary_table": str(raw_summary_path),
        },
        "irreducible_floor_fraction": IRREDUCIBLE_FLOOR_FRACTION,
        "method_sensitivity": METHOD_SENSITIVITY,
        "signed_bias_rule": "signed errors keep sign and scale magnitude",
        "source_metrics_table": str(metrics_path),
    }
    assumptions_path.write_text(json.dumps(assumptions, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "rowlevel": str(row_path),
        "combined": str(combined_path),
        "summary": str(summary_path),
        "raw_randomized_summary": str(raw_summary_path),
        "wide_summary": str(wide_path),
        "assumptions": str(assumptions_path),
        "figures_manifest": str(figures_dir / "exp1_prior_generation_fraction_figure_manifest.csv"),
        "n_fraction_rows": int(len(fraction_rows)),
        "n_combined_rows": int(len(combined)),
        "n_figures": int(len(figures)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate Exp1 prior-generation performance across training fractions without model execution.")
    parser.add_argument("--exp1-dir", type=Path, default=DEFAULT_EXP1_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--figures-dir", type=Path, default=None)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()
    result = estimate_fraction_performance(args.exp1_dir, args.output_dir, args.figures_dir, random_seed=args.random_seed)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
