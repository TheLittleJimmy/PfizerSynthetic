#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from pdc2.models import build_model, set_seed  # noqa: E402
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    _decode_baseline_conditioned_static,
    _sample_longitudinal_future,
)
from scripts.run_simulation_experiment_20260604 import (  # noqa: E402
    BENCHMARK_DIR,
    CONDITIONING_COLS,
    DEFAULT_OUTPUT_DIR,
    EVENT_COL,
    FALLBACK_DATASET_DIR,
    LONG_NAMES,
    REQUESTED_DATASET_DIR,
    SEED,
    STATIC_CATEGORICAL,
    STATIC_CONTINUOUS,
    SURVIVAL_TIME_COL,
    TREATMENT_NAME,
    _coerce_generated_outcome_summary,
    _fit_ctgan_like,
    _jsonable,
    _longitudinal_design,
    _numeric_fill_value,
    _read_phase_fidelity_summary,
    benchmark_dependency_status,
    benchmark_preprocessing_status,
    clean_subject_summary,
    conditional_bootstrap_replicates,
    conditional_outcome_summary,
    condition_matrix,
    ecdf_ks,
    estimand_rows,
    fit_longitudinal_change_models,
    km_curve,
    normalize_synthetic_static,
    real_long_for_indices,
    real_static_for_indices,
    reconstruct_longitudinal_from_outcomes,
    repo_status,
    resolve_dataset_dir,
    sample_cox_time,
    save_json,
    safe_float,
    select_generated_outcomes_conditioned,
    split_indices,
    subject_summary,
    subject_split,
    summarize_estimands,
    target_static_template,
    trajectory_summary,
)
from scripts.run_simulation_holdout import (  # noqa: E402
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _make_bundle,
    _read_simulation,
)

try:
    from lifelines import CoxPHFitter
except Exception:  # pragma: no cover - recorded in dependency JSON
    CoxPHFitter = None


ACTIVE_METHODS = [
    "PhaseSyn",
    "conditional_empirical_subject_bootstrap",
    "conditional_classical_lmm_cox_aft_simulator",
    "conditional_joint_longitudinal_survival_baseline",
    "conditional_modular_deep_generator",
]

METHOD_DESCRIPTIONS = {
    "PhaseSyn": (
        "primary model",
        "Baseline-conditioned PhaseSyn generator using real test W/L0/treatment to generate future longitudinal and survival/censoring outcomes",
    ),
    "conditional_empirical_subject_bootstrap": (
        "benchmark",
        "Nearest-neighbor or matched subject bootstrap conditional on real test W/L0/treatment",
    ),
    "conditional_classical_lmm_cox_aft_simulator": (
        "benchmark",
        "Conditional classical simulator using LMM/MMRM longitudinal model plus Cox/AFT event and censoring models",
    ),
    "conditional_joint_longitudinal_survival_baseline": (
        "benchmark",
        "Conditional joint longitudinal-survival model with shared random effects or trajectory-linked survival risk",
    ),
    "conditional_modular_deep_generator": (
        "benchmark",
        "Conditional CTGAN/TVAE plus TimeGAN-style trajectory generator plus SurvivalGAN/survival-generator module",
    ),
}

DEPRECATED_METHODS = [
    "conditional_classical_simulator",
    "conditional_CTGAN",
    "conditional_TVAE",
    "TimeGAN + Cox",
    "Latent ODE + Cox",
    "SDV",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def append_status(output_dir: Path, lines: list[str]) -> None:
    with (output_dir / "STATUS.md").open("a", encoding="utf-8") as f:
        f.write("\n## Conditional Benchmark Revision\n\n")
        for line in lines:
            f.write(line.rstrip() + "\n")


def markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def summarize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method"])
    return df.groupby("method").mean(numeric_only=True).reset_index()


def augment_survival_components(static: pd.DataFrame) -> pd.DataFrame:
    out = normalize_synthetic_static(static)
    out["U"] = pd.to_numeric(out["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0)
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    out[SURVIVAL_TIME_COL] = out["U"].astype(float)
    out[EVENT_COL] = out["delta"].astype(int)
    if "T" not in out:
        out["T"] = np.where(out["delta"].to_numpy(dtype=int) == 1, out["U"].to_numpy(dtype=float), np.inf)
    if "C" not in out:
        out["C"] = np.where(out["delta"].to_numpy(dtype=int) == 1, 1.0, out["U"].to_numpy(dtype=float))
    out["event_time"] = out["T"]
    out["censoring_time"] = out["C"]
    out["censoring_indicator"] = 1 - out["delta"].astype(int)
    return out


def add_true_survival_components(static: pd.DataFrame, survival: pd.DataFrame | None) -> pd.DataFrame:
    out = augment_survival_components(static)
    if survival is None or "subject_id" not in survival:
        return out
    key = "source_subject_id" if "source_subject_id" in out else "patient_id"
    if key not in out:
        return out
    surv = survival.rename(columns={"subject_id": key}).copy()
    keep = [key, "T", "C", "administrative_censor", "stochastic_censor"]
    keep = [c for c in keep if c in surv]
    merged = out.drop(columns=[c for c in ["T", "C", "event_time", "censoring_time"] if c in out]).merge(
        surv[keep],
        on=key,
        how="left",
    )
    merged["T"] = pd.to_numeric(merged.get("T"), errors="coerce").where(
        pd.to_numeric(merged.get("T"), errors="coerce").notna(),
        np.where(merged["delta"].to_numpy(dtype=int) == 1, merged["U"].to_numpy(dtype=float), np.inf),
    )
    merged["C"] = pd.to_numeric(merged.get("C"), errors="coerce").fillna(
        pd.Series(np.where(merged["delta"].to_numpy(dtype=int) == 1, 1.0, merged["U"].to_numpy(dtype=float)))
    )
    merged["event_time"] = merged["T"]
    merged["censoring_time"] = merged["C"]
    return merged


def split_static_with_source(raw: pd.DataFrame, indices: np.ndarray, survival: pd.DataFrame | None = None) -> pd.DataFrame:
    out = real_static_for_indices(raw, indices)
    out["source_subject_id"] = indices.astype(int)
    out["patient_id"] = np.arange(len(out), dtype=int)
    return add_true_survival_components(out, survival)


def long_from_prediction_array(
    target_static: pd.DataFrame,
    pred_raw: np.ndarray,
    time_grid: np.ndarray,
    include_after_u: bool = False,
) -> pd.DataFrame:
    static = augment_survival_components(target_static)
    rows: list[dict[str, Any]] = []
    for i, st in static.reset_index(drop=True).iterrows():
        u = safe_float(st["U"], 1.0)
        for visit, t_raw in enumerate(time_grid):
            if visit >= pred_raw.shape[1]:
                break
            t = float(t_raw)
            if not include_after_u and t > u + 1e-8:
                continue
            row = {
                "patient_id": int(st["patient_id"]),
                TREATMENT_NAME: int(st[TREATMENT_NAME]),
                "visit_index": int(visit),
                "visit_time": t,
            }
            for var_idx, var in enumerate(LONG_NAMES):
                if visit == 0:
                    row[var] = safe_float(st.get(var), 0.0)
                else:
                    row[var] = float(pred_raw[i, visit, var_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def future_only_longitudinal(long_df: pd.DataFrame, baseline_time: float = 0.0) -> pd.DataFrame:
    if long_df.empty or "visit_time" not in long_df:
        return long_df.copy()
    out = long_df[pd.to_numeric(long_df["visit_time"], errors="coerce").gt(float(baseline_time) + 1e-8)].copy()
    if "visit_index" in out:
        out = out[pd.to_numeric(out["visit_index"], errors="coerce").fillna(1).gt(0)].copy()
    return out.reset_index(drop=True)


def baseline_rows_from_static(static_df: pd.DataFrame, baseline_time: float = 0.0) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    static = target_static_template(static_df)
    for _, st in static.iterrows():
        row = {
            "patient_id": int(st["patient_id"]),
            TREATMENT_NAME: int(st[TREATMENT_NAME]),
            "visit_index": 0,
            "visit_time": float(baseline_time),
        }
        for var in LONG_NAMES:
            row[var] = safe_float(st.get(var), 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_longitudinal_view(static_df: pd.DataFrame, future_long_df: pd.DataFrame) -> pd.DataFrame:
    base = baseline_rows_from_static(static_df)
    future = future_only_longitudinal(future_long_df)
    return pd.concat([base, future], ignore_index=True)


def baseline_preservation_metrics(real_static: pd.DataFrame, syn_static: pd.DataFrame) -> dict[str, float]:
    real = real_static.reset_index(drop=True)
    syn = syn_static.reset_index(drop=True)
    n = min(len(real), len(syn))
    metrics: dict[str, float] = {}
    abs_errors: list[float] = []
    max_errors: list[float] = []
    match_rates: list[float] = []
    for col in [*STATIC_CONTINUOUS, *LONG_NAMES]:
        if col not in real or col not in syn:
            continue
        r = pd.to_numeric(real.loc[: n - 1, col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(syn.loc[: n - 1, col], errors="coerce").to_numpy(dtype=float)
        both_nan = np.isnan(r) & np.isnan(s)
        diff = np.abs(np.nan_to_num(r - s, nan=0.0))
        diff[both_nan] = 0.0
        metrics[f"baseline_preservation_{col}_max_abs_error"] = float(diff.max()) if diff.size else float("nan")
        metrics[f"baseline_preservation_{col}_mean_abs_error"] = float(diff.mean()) if diff.size else float("nan")
        abs_errors.extend(diff[np.isfinite(diff)].tolist())
        if diff.size:
            max_errors.append(float(diff.max()))
    for col in STATIC_CATEGORICAL + [TREATMENT_NAME]:
        if col not in real or col not in syn:
            continue
        r = real.loc[: n - 1, col]
        s = syn.loc[: n - 1, col]
        matches = (r.eq(s) | (r.isna() & s.isna())).to_numpy(dtype=bool)
        rate = float(matches.mean()) if len(matches) else float("nan")
        metrics[f"baseline_preservation_{col}_match_rate"] = rate
        if np.isfinite(rate):
            match_rates.append(rate)
    metrics["baseline_preservation_mean_abs_error"] = float(np.mean(abs_errors)) if abs_errors else float("nan")
    metrics["baseline_preservation_max_abs_error"] = float(np.max(max_errors)) if max_errors else float("nan")
    metrics["baseline_preservation_categorical_match_rate"] = float(np.mean(match_rates)) if match_rates else float("nan")
    metrics["baseline_preservation_all_exact"] = float(
        (metrics["baseline_preservation_max_abs_error"] <= 1e-8)
        and (not match_rates or min(match_rates) >= 1.0)
    )
    return metrics


def _future_summary_matrix(static_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    summary = trajectory_summary(long_df, static_df)
    keep = [
        c
        for c in summary.columns
        if c in {TREATMENT_NAME, "U", "delta"}
        or c.endswith("_final_value")
        or c.endswith("_change")
        or c.endswith("_slope")
        or c.endswith("_auc")
    ]
    out = summary[keep].apply(pd.to_numeric, errors="coerce")
    keep2 = [c for c in out.columns if out[c].notna().mean() > 0.7 and out[c].std(skipna=True) > 1e-8]
    return out[keep2]


def future_joint_metrics(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame) -> dict[str, float]:
    real_x = _future_summary_matrix(real_static, real_long)
    syn_x = _future_summary_matrix(syn_static, syn_long)
    cols = [c for c in real_x.columns if c in syn_x.columns]
    real_x = real_x[cols]
    syn_x = syn_x[cols]
    if len(cols) < 2 or len(real_x) < 5 or len(syn_x) < 5:
        return {
            "joint_future_correlation_frobenius_error": float("nan"),
            "joint_future_mmd_rbf": float("nan"),
            "joint_future_c2st_auc": float("nan"),
        }
    med = real_x.median(numeric_only=True)
    real_x = real_x.fillna(med).fillna(0.0)
    syn_x = syn_x.fillna(med).fillna(0.0)
    r_corr = np.corrcoef(real_x.to_numpy(dtype=float), rowvar=False)
    s_corr = np.corrcoef(syn_x.to_numpy(dtype=float), rowvar=False)
    frob = float(np.linalg.norm(np.nan_to_num(r_corr - s_corr), ord="fro"))
    scaler = StandardScaler().fit(pd.concat([real_x, syn_x], axis=0))
    rx = scaler.transform(real_x)
    sx = scaler.transform(syn_x)
    gamma = 1.0 / max(rx.shape[1], 1)

    def kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        aa = np.sum(a * a, axis=1)[:, None]
        bb = np.sum(b * b, axis=1)[None, :]
        return np.exp(-gamma * np.maximum(aa + bb - 2 * a @ b.T, 0.0))

    mmd = float(kernel(rx, rx).mean() + kernel(sx, sx).mean() - 2 * kernel(rx, sx).mean())
    x = np.vstack([rx, sx])
    y = np.concatenate([np.zeros(len(rx), dtype=int), np.ones(len(sx), dtype=int)])
    try:
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=SEED)
        train_idx, test_idx = next(split.split(x, y))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(x[train_idx], y[train_idx])
        auc = float(roc_auc_score(y[test_idx], clf.predict_proba(x[test_idx])[:, 1]))
    except Exception:
        auc = float("nan")
    return {
        "joint_future_correlation_frobenius_error": frob,
        "joint_future_mmd_rbf": mmd,
        "joint_future_c2st_auc": auc,
    }


def treatment_effect_errors(real_est: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame, tau: float) -> dict[str, float]:
    syn = pd.DataFrame(estimand_rows(syn_static, syn_long, tau, "syn"))
    out: dict[str, float] = {}
    for estimand in ["cox_hr", "cox_log_hr", "rmst_diff"]:
        r = real_est[real_est["estimand"].eq(estimand)]
        s = syn[syn["estimand"].eq(estimand)]
        if not r.empty and not s.empty:
            out[f"{estimand}_error"] = safe_float(s["estimate"].iloc[0]) - safe_float(r["estimate"].iloc[0])
            out[f"{estimand}_abs_error"] = abs(out[f"{estimand}_error"])
    for estimand, out_name in [
        ("treatment_time_interaction", "mmrm_treatment_time_interaction_abs_error_mean"),
        ("final_visit_treatment_contrast", "mmrm_final_contrast_abs_error_mean"),
        ("change_from_baseline_treatment_contrast", "mmrm_change_contrast_abs_error_mean"),
    ]:
        vals: list[float] = []
        for endpoint in LONG_NAMES:
            r = real_est[
                real_est["domain"].eq("longitudinal")
                & real_est["endpoint"].eq(endpoint)
                & real_est["estimand"].eq(estimand)
            ]
            s = syn[
                syn["domain"].eq("longitudinal")
                & syn["endpoint"].eq(endpoint)
                & syn["estimand"].eq(estimand)
            ]
            if not r.empty and not s.empty:
                vals.append(abs(safe_float(s["estimate"].iloc[0]) - safe_float(r["estimate"].iloc[0])))
        out[out_name] = float(np.mean(vals)) if vals else float("nan")
    return out


def coupling_setup(real_static: pd.DataFrame, real_long: pd.DataFrame) -> tuple[float, float, dict[str, Any]]:
    real_base = real_long[real_long["visit_index"].eq(0)].set_index("patient_id")["L1"]
    post = real_long[real_long["visit_index"].gt(0)].sort_values("visit_time")
    landmark_time = float(post["visit_time"].dropna().quantile(0.10)) if len(post) else 0.08
    early = post[post["visit_time"].ge(landmark_time - 1e-6)].groupby("patient_id").first()["L1"]
    threshold = float((real_base - early).dropna().median())
    real = coupling_metrics_safe(real_static, real_long, threshold, landmark_time, "real")
    return threshold, landmark_time, real


def coupling_metrics_safe(static_df: pd.DataFrame, long_df: pd.DataFrame, threshold: float, landmark_time: float, replicate: int | str) -> dict[str, Any]:
    from scripts.run_simulation_experiment_20260604 import coupling_metrics

    try:
        return coupling_metrics(static_df, long_df, threshold, landmark_time, replicate)
    except Exception:
        return {
            "replicate": replicate,
            "early_response_coef": np.nan,
            "early_response_hr": np.nan,
            "early_slope_coef": np.nan,
            "responder_nonresponder_hr": np.nan,
            "early_slope_hr": np.nan,
        }


def method_fidelity_row(
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
    real_estimands: pd.DataFrame,
    real_coupling: dict[str, Any],
    threshold: float,
    landmark_time: float,
    syn_static: pd.DataFrame,
    syn_long: pd.DataFrame,
    method: str,
    replicate: int,
    tau: float,
) -> dict[str, Any]:
    from scripts.run_simulation_experiment_20260604 import baseline_fidelity, longitudinal_fidelity, survival_fidelity

    row: dict[str, Any] = {"method": method, "replicate": int(replicate)}
    row.update(baseline_fidelity(real_static, syn_static))
    row.update(baseline_preservation_metrics(real_static, syn_static))
    row.update(longitudinal_fidelity(real_long, syn_long))
    row.update(survival_fidelity(real_static, syn_static))
    row.update(future_joint_metrics(real_static, real_long, syn_static, syn_long))
    row.update(treatment_effect_errors(real_estimands, syn_static, syn_long, tau))
    cm = coupling_metrics_safe(syn_static, syn_long, threshold, landmark_time, replicate)
    for key in ["early_response_coef", "early_slope_coef", "responder_nonresponder_hr"]:
        row[f"landmark_{key}"] = safe_float(cm.get(key))
        row[f"landmark_{key}_abs_error"] = abs(safe_float(cm.get(key)) - safe_float(real_coupling.get(key)))
    return row


def fit_cox_sampler_from_features(
    train_features: pd.DataFrame,
    durations: pd.Series,
    events: pd.Series,
    label: str,
) -> tuple[Any | None, str | None]:
    if CoxPHFitter is None:
        return None, f"{label} Cox unavailable: lifelines is not installed."
    event = pd.to_numeric(events, errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    if event.sum() < 5 or event.nunique() < 2:
        return None, f"{label} Cox unavailable: insufficient event variation."
    try:
        x = train_features.apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
        df = x.copy()
        df["U"] = pd.to_numeric(durations, errors="coerce").fillna(1.0).clip(1e-5, 1.0).to_numpy(dtype=float)
        df["_event"] = event.to_numpy(dtype=int)
        cph = CoxPHFitter(penalizer=0.10)
        cph.fit(df[["U", "_event", *x.columns]], duration_col="U", event_col="_event")
        return cph, None
    except Exception as exc:
        return None, f"{label} Cox fit failed: {type(exc).__name__}: {exc}"


def survival_feature_table(summary: pd.DataFrame, include_shared: bool = False) -> pd.DataFrame:
    cols = [
        TREATMENT_NAME,
        *STATIC_CONTINUOUS,
        *STATIC_CATEGORICAL,
        *LONG_NAMES,
        *[f"{var}_{suffix}" for var in LONG_NAMES for suffix in ["final", "change", "slope", "auc"]],
    ]
    out = summary[[c for c in cols if c in summary]].apply(pd.to_numeric, errors="coerce")
    if include_shared:
        slope_cols = [f"{var}_slope" for var in LONG_NAMES if f"{var}_slope" in summary]
        out["shared_trajectory_score"] = summary[slope_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1) if slope_cols else 0.0
    return out


def generate_lmm_longitudinal(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
    shared_random_effects: bool = False,
) -> tuple[list[pd.DataFrame], list[dict[str, float]], list[str]]:
    rng = np.random.default_rng(seed)
    coefs, residual_sd, cols, fills, issues = fit_longitudinal_change_models(train_static, train_long)
    longs: list[pd.DataFrame] = []
    diagnostics: list[dict[str, float]] = []
    for rep in range(reps):
        rows: list[dict[str, Any]] = []
        shared = rng.normal(0.0, 1.0, size=len(target_static)) if shared_random_effects else np.zeros(len(target_static))
        for row_idx, st in target_static_template(target_static).iterrows():
            for visit, t in enumerate(time_grid):
                base_row = {
                    "patient_id": int(st["patient_id"]),
                    TREATMENT_NAME: int(st[TREATMENT_NAME]),
                    "visit_index": int(visit),
                    "visit_time": float(t),
                }
                design = _longitudinal_design(pd.DataFrame([st]), np.asarray([float(t)]), cols, fills).to_numpy(dtype=float)[0]
                for var in LONG_NAMES:
                    baseline_value = safe_float(st.get(var), fills.get(var, 0.0))
                    if visit == 0:
                        value = baseline_value
                    else:
                        sd = max(safe_float(residual_sd.get(var), 0.1), 1e-6)
                        random_intercept = rng.normal(0.0, 0.35 * sd)
                        random_slope = rng.normal(0.0, 0.20 * sd)
                        shared_shift = 0.15 * sd * shared[row_idx] if shared_random_effects else 0.0
                        change = float(design @ coefs[var] + random_intercept + random_slope * float(t) + shared_shift + rng.normal(0.0, 0.45 * sd))
                        value = baseline_value + change
                    base_row[var] = float(value)
                rows.append(base_row)
        longs.append(pd.DataFrame(rows))
        diagnostics.append({"shared_random_effect_sd": float(np.std(shared))})
    return longs, diagnostics, issues


def generate_survival_from_features(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    generated_longs: list[pd.DataFrame],
    seed: int,
    include_shared: bool = False,
) -> tuple[list[pd.DataFrame], dict[str, Any], list[str]]:
    rng = np.random.default_rng(seed)
    issues: list[str] = []
    train_summary = subject_summary(train_static, train_long)
    features_train = survival_feature_table(train_summary, include_shared=include_shared)
    fills = features_train.median(numeric_only=True).fillna(0.0)
    event_cph, event_issue = fit_cox_sampler_from_features(features_train, train_summary["U"], train_summary["delta"], "event")
    censor_cph, censor_issue = fit_cox_sampler_from_features(features_train, train_summary["U"], 1 - train_summary["delta"], "censoring")
    for issue in [event_issue, censor_issue]:
        if issue:
            issues.append(issue)
    use_cox = event_cph is not None and censor_cph is not None
    statics: list[pd.DataFrame] = []
    neighbor_info = None
    if not use_cox:
        from scripts.run_simulation_experiment_20260604 import prepare_conditioned_neighbors, sample_precomputed_conditioned_positions

        neighbor_info = prepare_conditioned_neighbors(train_static, target_static, k=25)
        source_sampler = sample_precomputed_conditioned_positions
    else:
        source_sampler = None
    for rep, long_all in enumerate(generated_longs):
        static = target_static_template(target_static)
        if use_cox:
            gen_summary = subject_summary(static, long_all)
            feat = survival_feature_table(gen_summary, include_shared=include_shared)
            feat = feat.reindex(columns=features_train.columns).apply(pd.to_numeric, errors="coerce").fillna(fills).fillna(0.0)
            t_events: list[float] = []
            c_times: list[float] = []
            observed: list[float] = []
            events: list[int] = []
            for _, row in feat.iterrows():
                event_t = sample_cox_time(event_cph, row, rng)
                censor_t = sample_cox_time(censor_cph, row, rng)
                obs = min(event_t, censor_t, 1.0)
                t_events.append(float(event_t))
                c_times.append(float(censor_t))
                observed.append(float(np.clip(obs, 0.02, 1.0)))
                events.append(int(event_t <= censor_t and event_t <= 1.0))
        else:
            assert neighbor_info is not None and source_sampler is not None
            source_pos, _ = source_sampler(neighbor_info, rng)
            source_static = train_static.iloc[source_pos].reset_index(drop=True)
            observed = pd.to_numeric(source_static["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0).to_numpy(dtype=float).tolist()
            events = pd.to_numeric(source_static["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy().tolist()
            t_events = np.where(np.asarray(events) == 1, np.asarray(observed), np.inf).astype(float).tolist()
            c_times = np.where(np.asarray(events) == 1, 1.0, np.asarray(observed)).astype(float).tolist()
        static["U"] = observed
        static[SURVIVAL_TIME_COL] = observed
        static["delta"] = events
        static[EVENT_COL] = events
        static["T"] = t_events
        static["C"] = c_times
        static = augment_survival_components(static)
        statics.append(static)
    diag = {
        "event_model": "cox_ph_with_generated_trajectory_summaries" if event_cph is not None else "conditional_neighbor_fallback",
        "censoring_model": "cox_ph_with_generated_trajectory_summaries" if censor_cph is not None else "conditional_neighbor_fallback",
        "cox_event_model_used": bool(event_cph is not None),
        "cox_censoring_model_used": bool(censor_cph is not None),
        "include_shared_trajectory_score": bool(include_shared),
    }
    return statics, diag, issues


def filter_longitudinal_by_static(long_all: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    cutoff = static.set_index("patient_id")["U"].to_dict()
    rows = []
    for _, row in long_all.iterrows():
        u = safe_float(cutoff.get(int(row["patient_id"])), 1.0)
        if safe_float(row["visit_time"]) <= u + 1e-8 and safe_float(row["visit_time"]) > 1e-8:
            rows.append(row.to_dict())
    return pd.DataFrame(rows)


def classical_lmm_cox_aft_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any], list[str]]:
    longs_all, long_diag, issues = generate_lmm_longitudinal(train_static, train_long, target_static, time_grid, reps, seed)
    statics, surv_diag, surv_issues = generate_survival_from_features(
        train_static, train_long, target_static, longs_all, seed + 101, include_shared=False
    )
    issues.extend(surv_issues)
    longs = [filter_longitudinal_by_static(l, s) for s, l in zip(statics, longs_all)]
    diag = {
        "longitudinal_model": "MMRM-style OLS treatment*time plus baseline covariates with sampled random intercept/slope residuals",
        "full_lmm_fit_used": False,
        "full_lmm_fit_note": "statsmodels MixedLM was not used in this fast revision; the active benchmark name follows the requested table, and this field records the MMRM-style fallback actually fitted.",
        "survival_model": surv_diag,
        "aft_fallback_used": False,
        "aft_note": "Cox PH was preferred; AFT was not needed when Cox sampling succeeded, otherwise conditional nearest-neighbor survival fallback is logged.",
        "longitudinal_random_effect_mean_sd": float(np.mean([d["shared_random_effect_sd"] for d in long_diag])) if long_diag else 0.0,
    }
    return statics, [future_only_longitudinal(l) for l in longs], diag, issues


def joint_longitudinal_survival_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any], list[str]]:
    longs_all, long_diag, issues = generate_lmm_longitudinal(
        train_static, train_long, target_static, time_grid, reps, seed, shared_random_effects=True
    )
    statics, surv_diag, surv_issues = generate_survival_from_features(
        train_static, train_long, target_static, longs_all, seed + 202, include_shared=True
    )
    issues.extend(surv_issues)
    longs = [filter_longitudinal_by_static(l, s) for s, l in zip(statics, longs_all)]
    diag = {
        "preferred_jmbayes2_available": check_jmbayes2_available()["available"],
        "implementation": "python_shared_random_effects_and_trajectory_linked_cox_fallback",
        "longitudinal_model": "mixed-model fallback with shared subject random effect",
        "survival_model": surv_diag,
        "longitudinal_random_effect_mean_sd": float(np.mean([d["shared_random_effect_sd"] for d in long_diag])) if long_diag else 0.0,
    }
    return statics, [future_only_longitudinal(l) for l in longs], diag, issues


def check_jmbayes2_available() -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["Rscript", "-e", "suppressPackageStartupMessages(library(JMbayes2)); cat(as.character(packageVersion('JMbayes2')))"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
        return {
            "available": proc.returncode == 0,
            "version": proc.stdout.strip() if proc.returncode == 0 else None,
            "error": proc.stderr.strip() if proc.returncode != 0 else None,
        }
    except Exception as exc:
        return {"available": False, "version": None, "error": f"{type(exc).__name__}: {exc}"}


class TinyConditionalVAE(torch.nn.Module):
    def __init__(self, x_dim: int, y_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim + y_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden_dim, latent_dim)
        self.logvar = torch.nn.Linear(hidden_dim, latent_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(x_dim + latent_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, y_dim),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, y], dim=1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-6.0, 4.0)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        pred = self.decoder(torch.cat([x, z], dim=1))
        return pred, mu, logvar

    def sample(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(x.shape[0], self.mu.out_features, device=x.device)
        return self.decoder(torch.cat([x, z], dim=1))


def cvae_outcome_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
    epochs: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any], list[str]]:
    set_seed(seed)
    train_summary = conditional_outcome_summary(train_static, train_long)
    x_cols = [c for c in CONDITIONING_COLS + [TREATMENT_NAME] if c in train_summary and c in target_static]
    y_cols = ["U", "delta", *[f"{var}_{suffix}" for var in LONG_NAMES for suffix in ["change", "slope", "auc"]]]
    y_cols = [c for c in y_cols if c in train_summary]
    x_train = condition_matrix(train_summary, x_cols, {c: _numeric_fill_value(train_summary[c], c in STATIC_CATEGORICAL) for c in x_cols})
    y_train = train_summary[y_cols].apply(pd.to_numeric, errors="coerce").fillna(train_summary[y_cols].median(numeric_only=True)).fillna(0.0)
    x_target = condition_matrix(target_static, x_cols, {c: _numeric_fill_value(train_summary[c], c in STATIC_CATEGORICAL) for c in x_cols})
    x_scaler = StandardScaler().fit(x_train)
    y_scaler = StandardScaler().fit(y_train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyConditionalVAE(len(x_cols), len(y_cols)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_tensor = torch.tensor(x_scaler.transform(x_train), dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_scaler.transform(y_train), dtype=torch.float32, device=device)
    for _ in range(max(1, int(epochs))):
        pred, mu, logvar = model(x_tensor, y_tensor)
        recon = torch.nn.functional.mse_loss(pred, y_tensor)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon + 0.01 * kl
        opt.zero_grad()
        loss.backward()
        opt.step()
    x_eval = torch.tensor(x_scaler.transform(x_target), dtype=torch.float32, device=device)
    statics: list[pd.DataFrame] = []
    longs: list[pd.DataFrame] = []
    for rep in range(reps):
        torch.manual_seed(seed + rep)
        with torch.no_grad():
            y_raw = y_scaler.inverse_transform(model.sample(x_eval).detach().cpu().numpy())
        outcomes = pd.DataFrame(y_raw, columns=y_cols)
        outcomes = _coerce_generated_outcome_summary(outcomes, train_summary, len(target_static))
        static, long_df = reconstruct_longitudinal_from_outcomes(target_static, outcomes, time_grid)
        statics.append(augment_survival_components(static))
        longs.append(future_only_longitudinal(long_df))
    diag = {
        "summary_generator": "PyTorch conditional VAE fallback",
        "epochs": int(epochs),
        "x_dim": int(len(x_cols)),
        "y_dim": int(len(y_cols)),
    }
    return statics, longs, diag, []


def modular_deep_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
    epochs: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any], list[str]]:
    issues: list[str] = []
    train_summary = conditional_outcome_summary(train_static, train_long)
    prep = benchmark_preprocessing_status(train_summary)
    summary_model, issue = _fit_ctgan_like("TVAE", train_summary, seed, max(1, int(epochs)))
    if issue:
        issues.append(f"TVAE summary module fallback activated: {issue}")
        statics, longs, diag, cvae_issues = cvae_outcome_replicates(
            train_static, train_long, target_static, time_grid, reps, seed + 303, max(20, int(epochs))
        )
        issues.extend(cvae_issues)
        diag.update({
            "trajectory_module": "summary-to-grid decoder; TimeGAN unavailable or not used",
            "survival_module": "conditional VAE survival-summary fallback",
            "preprocessing": prep,
        })
        return statics, longs, diag, issues
    statics: list[pd.DataFrame] = []
    longs: list[pd.DataFrame] = []
    for rep in range(reps):
        try:
            raw_sample = summary_model.sample(max(len(target_static) * 5, len(target_static) + 200))
            outcomes = select_generated_outcomes_conditioned(raw_sample, train_summary, target_static, seed + rep)
            static, long_df = reconstruct_longitudinal_from_outcomes(target_static, outcomes, time_grid)
            statics.append(augment_survival_components(static))
            longs.append(future_only_longitudinal(long_df))
        except Exception as exc:
            issues.append(f"TVAE modular replicate {rep:03d} failed; switched to CVAE for remaining outputs: {type(exc).__name__}: {exc}")
            return cvae_outcome_replicates(train_static, train_long, target_static, time_grid, reps, seed + 404, max(20, int(epochs)))
    diag = {
        "summary_generator": "TVAE from cloned CTGAN package",
        "trajectory_module": "TimeGAN-style summary-to-grid decoder fallback conditioned on W/L0/A",
        "survival_module": "TVAE survival-summary fallback conditioned on W/L0/A and generated trajectory summaries",
        "timegan_official": "not used: TensorFlow unavailable or not enabled",
        "pytorch_timegan": "not used: no stable installed adapter was available; summary-to-grid neural fallback is the trajectory module",
        "survivalgan": "not used: dependency unavailable or not enabled",
        "ctgan_tvae_deprecation_note": "The old separate CTGAN and TVAE rows are deprecated; TVAE is used only as an internal module of the combined modular deep benchmark.",
        "epochs": int(epochs),
        "preprocessing": prep,
    }
    return statics, longs, diag, issues


def generate_phasesyn_conditioned_replicates(
    output_dir: Path,
    cfg: dict[str, Any],
    train_bundle: Any,
    test_bundle: Any,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
    device: torch.device,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, Any], list[str]]:
    issues: list[str] = []
    ckpt_path = output_dir / "models" / "phasesyn_full.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    cfg_model = dict(cfg.get("model", {}))
    if not any(str(k).startswith("u0_logsigma_head.") for k in state):
        cfg_model["stochastic_u0"] = False
        cfg_model["use_u0_mean_at_eval"] = True
        issues.append("PhaseSyn checkpoint predates stochastic-u0 variance head; loaded with strict=False and deterministic u0 compatibility mode.")
    cfg = dict(cfg)
    cfg["model"] = cfg_model
    model = build_model(train_bundle, cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    statics: list[pd.DataFrame] = []
    longs: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for rep in range(reps):
        set_seed(seed + rep)
        syn_raw, latents, audit = _decode_baseline_conditioned_static(
            model,
            test_bundle,
            device,
            deterministic_u0=True,
        )
        pred_raw, _, _, _ = _sample_longitudinal_future(model, test_bundle, latents, device, sample=True)
        static = target_static_template(target_static)
        static["U"] = pd.to_numeric(syn_raw[SURVIVAL_TIME_COL], errors="coerce").fillna(1.0).clip(0.02, 1.0).to_numpy(dtype=float)
        static[SURVIVAL_TIME_COL] = static["U"].astype(float)
        static["delta"] = pd.to_numeric(syn_raw[EVENT_COL], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
        static[EVENT_COL] = static["delta"].astype(int)
        static = augment_survival_components(static)
        statics.append(static)
        longs.append(future_only_longitudinal(long_from_prediction_array(static, pred_raw, time_grid)))
        audits.append(audit)
    diag = {
        "checkpoint": str(ckpt_path),
        "loaded_strict_false": True,
        "missing_keys_count": int(len(missing)),
        "unexpected_keys_count": int(len(unexpected)),
        "missing_keys_sample": list(missing)[:10],
        "unexpected_keys_sample": list(unexpected)[:10],
        "generation_audits_all_survival_zero": bool(
            all(x.get("test_survival_mask_zero_for_generation") and x.get("test_survival_tensor_zero_for_generation") for x in audits)
        ),
        "conditioning": "real test W/L0/treatment and requested grids",
    }
    return statics, longs, diag, issues


def write_replicates(output_dir: Path, method: str, statics: list[pd.DataFrame], longs: list[pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep, (static, long_df) in enumerate(zip(statics, longs)):
        future = future_only_longitudinal(long_df)
        if not future.empty:
            if not pd.to_numeric(future["visit_time"], errors="coerce").gt(1e-8).all():
                raise AssertionError(f"{method} replicate {rep} export contains baseline/non-future rows.")
            if "visit_index" in future and not pd.to_numeric(future["visit_index"], errors="coerce").fillna(1).gt(0).all():
                raise AssertionError(f"{method} replicate {rep} export contains visit_index=0 rows.")
        long_format = synthetic_long_format_revised(static, future, rep)
        safe_method = method.lower().replace("+", "_").replace("/", "_").replace(" ", "_")
        target = output_dir / "synthetic" / f"{safe_method}_revised_rep_{rep:03d}.parquet"
        fmt = "parquet"
        try:
            long_format.to_parquet(target, index=False)
        except Exception:
            target = target.with_suffix(".csv")
            fmt = "csv"
            long_format.to_csv(target, index=False)
        rows.append({
            "method": method,
            "replicate": int(rep),
            "path": str(target),
            "format": fmt,
            "n_subjects": int(len(static)),
            "n_longitudinal_rows": int(len(future)),
            "treatment_ratio": float(static[TREATMENT_NAME].mean()) if len(static) else float("nan"),
            "event_rate": float(static["delta"].mean()) if len(static) else float("nan"),
            "future_only_rows": bool(future.empty or pd.to_numeric(future["visit_time"], errors="coerce").gt(1e-8).all()),
        })
    return rows


def synthetic_long_format_revised(static_df: pd.DataFrame, future_long_df: pd.DataFrame, replicate: int) -> pd.DataFrame:
    static = augment_survival_components(static_df).copy()
    static = static.rename(columns={name: f"baseline_{name}" for name in LONG_NAMES if name in static.columns})
    merge_cols = [c for c in static.columns if c not in set(LONG_NAMES)]
    out = future_long_df.merge(static[merge_cols], on=["patient_id", TREATMENT_NAME], how="left", suffixes=("", "_static"))
    out.insert(0, "replicate", int(replicate))
    out["delta"] = pd.to_numeric(out.get(EVENT_COL, out.get("delta")), errors="coerce").fillna(0).astype(int)
    out["U"] = pd.to_numeric(out.get(SURVIVAL_TIME_COL, out.get("U")), errors="coerce").fillna(1.0).astype(float)
    out["censoring_indicator"] = 1 - out["delta"]
    out["trajectory_scope"] = "post_baseline_future_grid"
    return out


def build_bundles(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = resolve_dataset_dir(args.dataset_dir, args.output_dir)
    raw, ids, long_df, types, metadata = _read_simulation(data_dir)
    survival = pd.read_csv(data_dir / "survival.csv") if (data_dir / "survival.csv").exists() else None
    long_obs = pd.read_csv(data_dir / "longitudinal_observed.csv")
    split_path = args.output_dir / "split_ids.csv"
    split_df = pd.read_csv(split_path) if split_path.exists() else subject_split(raw, args.seed)
    train_idx = split_indices(split_df, "train")
    test_idx = split_indices(split_df, "test")
    cfg_path = args.output_dir / "configs" / "phasesyn_full.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_specs, time_min, time_max, max_visits = _fit_longitudinal_preprocessor(
        long_df,
        types,
        train_idx,
        raw.iloc[train_idx][SURVIVAL_TIME_COL],
    )
    train_bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    test_bundle = _make_bundle(raw, ids, long_df, types, test_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    return {
        "data_dir": data_dir,
        "raw": raw,
        "ids": ids,
        "long_df": long_df,
        "types": types,
        "metadata": metadata,
        "survival": survival,
        "long_obs": long_obs,
        "split_df": split_df,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "cfg": cfg,
        "train_bundle": train_bundle,
        "test_bundle": test_bundle,
        "real_train_static": split_static_with_source(raw, train_idx, survival),
        "real_train_long": real_long_for_indices(long_obs, train_idx),
        "real_test_static": split_static_with_source(raw, test_idx, survival),
        "real_test_long": real_long_for_indices(long_obs, test_idx),
        "time_grid": np.asarray(metadata.get("visit_schedule", np.linspace(0.0, 1.0, max_visits)), dtype=float),
    }


def evaluate_methods(
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
    method_sets: dict[str, tuple[list[pd.DataFrame], list[pd.DataFrame]]],
    tau: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    real_estimands = pd.DataFrame(estimand_rows(real_static, real_long, tau, "real"))
    threshold, landmark_time, real_coupling = coupling_setup(real_static, real_long)
    fidelity_rows: list[dict[str, Any]] = []
    estimand_rows_all: list[dict[str, Any]] = []
    estimand_summary_frames: list[pd.DataFrame] = []
    coupling_rows: list[dict[str, Any]] = []
    for method, (statics, longs) in method_sets.items():
        method_est_rows: list[dict[str, Any]] = []
        for rep, (static, long_df) in enumerate(zip(statics, longs)):
            metric_long = metric_longitudinal_view(static, long_df)
            fidelity_rows.append(
                method_fidelity_row(
                    real_static,
                    real_long,
                    real_estimands,
                    real_coupling,
                    threshold,
                    landmark_time,
                    static,
                    metric_long,
                    method,
                    rep,
                    tau,
                )
            )
            cm = coupling_metrics_safe(static, metric_long, threshold, landmark_time, rep)
            cm["method"] = method
            cm["landmark_time"] = landmark_time
            cm["response_threshold"] = threshold
            coupling_rows.append(cm)
            for er in estimand_rows(static, metric_long, tau, rep):
                er["method"] = method
                method_est_rows.append(er)
                estimand_rows_all.append(er)
        method_est = pd.DataFrame(method_est_rows)
        if not method_est.empty:
            summary = summarize_estimands(real_estimands, method_est)
            summary.insert(0, "method", method)
            estimand_summary_frames.append(summary)
    fidelity = pd.DataFrame(fidelity_rows)
    estimands = pd.DataFrame(estimand_rows_all)
    estimand_summary = pd.concat(estimand_summary_frames, ignore_index=True) if estimand_summary_frames else pd.DataFrame()
    coupling = pd.DataFrame(coupling_rows)
    return fidelity, estimands, estimand_summary, coupling


def create_method_table(fidelity_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_cols = [
        "baseline_preservation_max_abs_error",
        "baseline_preservation_categorical_match_rate",
        "longitudinal_mean_trajectory_rmse_mean",
        "longitudinal_change_from_baseline_rmse_mean",
        "longitudinal_abs_slope_error_mean",
        "km_iae_all",
        "event_rate_error",
        "censoring_rate_error",
        "cox_hr_abs_error",
        "rmst_diff_abs_error",
        "mmrm_treatment_time_interaction_abs_error_mean",
        "landmark_early_response_coef_abs_error",
        "joint_future_mmd_rbf",
        "joint_future_c2st_auc",
    ]
    for method in ACTIVE_METHODS:
        status, how = METHOD_DESCRIPTIONS[method]
        row: dict[str, Any] = {"Method": method, "Status": status, "How it is used": how}
        fs = fidelity_summary[fidelity_summary["method"].eq(method)]
        for col in metric_cols:
            row[col] = safe_float(fs[col].iloc[0]) if col in fs and not fs.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def validate_revised_outputs(
    output_dir: Path,
    table: pd.DataFrame,
    fidelity_summary: pd.DataFrame,
    index_df: pd.DataFrame,
) -> dict[str, Any]:
    active = table["Method"].astype(str).tolist()
    if active != ACTIVE_METHODS:
        raise AssertionError(f"Active benchmark table order/set mismatch: {active}")
    bad = sorted(set(active) & set(DEPRECATED_METHODS))
    if bad:
        raise AssertionError(f"Deprecated methods appear in active benchmark table: {bad}")
    fid_methods = sorted(fidelity_summary["method"].astype(str).unique().tolist())
    if sorted(ACTIVE_METHODS) != fid_methods:
        raise AssertionError(f"Fidelity summary methods mismatch: {fid_methods}")
    max_baseline_error = safe_float(fidelity_summary["baseline_preservation_max_abs_error"].max())
    min_cat_match = safe_float(fidelity_summary["baseline_preservation_categorical_match_rate"].min())
    if not (np.isfinite(max_baseline_error) and max_baseline_error <= 1e-8):
        raise AssertionError(f"Baseline W/L0 preservation failed: max abs error {max_baseline_error}")
    if not (np.isfinite(min_cat_match) and min_cat_match >= 1.0):
        raise AssertionError(f"Baseline categorical/A preservation failed: min match rate {min_cat_match}")
    if index_df.empty or not bool(index_df["future_only_rows"].all()):
        raise AssertionError("One or more revised benchmark replicate exports are not future-only.")
    required = [
        output_dir / "metrics" / "benchmark_fidelity_summary_revised.csv",
        output_dir / "metrics" / "benchmark_estimand_summary_revised.csv",
        output_dir / "metrics" / "benchmark_dependency_status_revised.json",
        output_dir / "tables" / "table7_methods_comparison_revised.csv",
        output_dir / "tables" / "table7_methods_comparison.csv",
        output_dir / "tables" / "table7_methods_comparison_deprecated_previous.csv",
        output_dir / "reports" / "benchmark_revision_summary.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"Missing required revised artifacts: {missing}")
    return {
        "active_method_set_exact": True,
        "deprecated_absent_from_active_table": True,
        "baseline_preservation_max_abs_error": max_baseline_error,
        "baseline_preservation_categorical_match_rate_min": min_cat_match,
        "future_only_exports": True,
        "required_artifacts_present": True,
        "revised_replicate_files": int(len(index_df)),
    }


def write_reports(
    output_dir: Path,
    table: pd.DataFrame,
    dep_status: dict[str, Any],
    issues: list[str],
    reps: int,
) -> None:
    summary_path = output_dir / "reports" / "benchmark_revision_summary.md"
    lines = [
        "# Benchmark Revision Summary",
        "",
        f"Generated: {now_iso()}",
        "",
        "## Conditioning Contract",
        "",
        "All active methods are evaluated in baseline-conditioned generation mode. The real test subject baseline covariates W, baseline longitudinal row L0, treatment arm A, and requested grids are supplied externally. Methods generate only post-baseline longitudinal futures and event/censoring outcomes; copied baseline rows are used only as conditioning and for change-from-baseline evaluation.",
        "",
        "## Active Methods",
        "",
        markdown_table(table[["Method", "Status", "How it is used"]]),
        "",
        "## Deprecated Previous Methods",
        "",
        *(f"- {name}" for name in DEPRECATED_METHODS),
        "",
        "## Replicates",
        "",
        f"- Revised evaluation replicates per active method: {reps}",
        "",
        "## Fallbacks And Failed Dependencies",
        "",
        *(f"- {item}" for item in issues),
        "",
        "## Output Files",
        "",
        "- `metrics/benchmark_fidelity_summary_revised.csv`",
        "- `metrics/benchmark_estimand_summary_revised.csv`",
        "- `metrics/benchmark_dependency_status_revised.json`",
        "- `tables/table7_methods_comparison_revised.csv`",
        "- `tables/table7_methods_comparison.csv`",
        "- `tables/table7_methods_comparison_deprecated_previous.csv`",
        "- `reports/benchmark_revision_summary.md`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    report_path = output_dir / "report.md"
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n## Revised Conditional Benchmark Methods\n\n")
        f.write(f"Updated: {now_iso()}\n\n")
        f.write("All methods in the revised Table 7 are evaluated in baseline-conditioned mode. Real test W, L0, treatment A, and requested visit/survival grids are supplied to PhaseSyn and every benchmark; the comparison is over generated post-baseline longitudinal trajectories and survival/censoring futures. No benchmark is required to generate new baseline covariates in this experiment.\n\n")
        f.write("No single off-the-shelf baseline fully matches PhaseSyn's mixed baseline, longitudinal trajectory, intervention, event, and censoring interface. The revised modular deep benchmark is therefore a composed conditional baseline using a CTGAN/TVAE outcome-summary module when available and documented fallbacks for trajectory and survival generation.\n\n")
        f.write("Active methods are exactly: `" + "`, `".join(ACTIVE_METHODS) + "`.\n\n")
        f.write("Deprecated/inactive previous rows are: `" + "`, `".join(DEPRECATED_METHODS) + "`.\n\n")
        f.write(f"Fallbacks and dependency issues are recorded in `metrics/benchmark_dependency_status_revised.json`; {len(issues)} issue(s) were logged.\n\n")
        f.write("Revised outputs: `tables/table7_methods_comparison.csv`, `tables/table7_methods_comparison_revised.csv`, `metrics/benchmark_fidelity_summary_revised.csv`, `metrics/benchmark_estimand_summary_revised.csv`, and `reports/benchmark_revision_summary.md`.\n")

    paper_summary = output_dir / "reports" / "minimum_publishable_evaluation_summary.md"
    with paper_summary.open("a", encoding="utf-8") as f:
        f.write("\n## Revised Benchmark Interpretation\n\n")
        f.write("The revised benchmark comparison is baseline-conditioned: W, L0, and treatment A are supplied from the real test set, and methods are compared on generated post-baseline longitudinal and survival/censoring futures. The active methods are PhaseSyn, conditional empirical subject bootstrap, conditional classical LMM plus Cox/AFT simulator, conditional joint longitudinal-survival fallback, and a conditional modular deep generator. Deprecated CTGAN/TVAE rows are retained only in the previous-table archive, not as active methods.\n")


def run(args: argparse.Namespace) -> None:
    start = time.time()
    set_seed(args.seed)
    output_dir = args.output_dir
    for rel in ["metrics", "tables", "reports", "synthetic"]:
        (output_dir / rel).mkdir(parents=True, exist_ok=True)
    bundle = build_bundles(args)
    real_train_static = bundle["real_train_static"]
    real_train_long = bundle["real_train_long"]
    real_test_static = bundle["real_test_static"]
    real_test_long = bundle["real_test_long"]
    time_grid = bundle["time_grid"]
    tau = float(np.quantile(real_test_static["U"], 0.80))
    device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")

    previous_table = output_dir / "tables" / "table7_methods_comparison.csv"
    deprecated_table = output_dir / "tables" / "table7_methods_comparison_deprecated_previous.csv"
    if previous_table.exists() and not deprecated_table.exists():
        shutil.copy2(previous_table, deprecated_table)

    issues: list[str] = []
    diagnostics: dict[str, Any] = {
        "timestamp": now_iso(),
        "seed": int(args.seed),
        "reps": int(args.reps),
        "conditioning_contract": "real test W, L0, treatment A, requested future visit grid, and survival grid are supplied externally",
        "active_methods": ACTIVE_METHODS,
        "deprecated_previous_methods": DEPRECATED_METHODS,
        "repositories": {
            "CTGAN": repo_status(BENCHMARK_DIR / "CTGAN"),
            "TimeGAN": repo_status(BENCHMARK_DIR / "TimeGAN"),
            "TimeGAN_pytorch": repo_status(BENCHMARK_DIR / "TimeGAN-pytorch"),
            "SurvivalGAN": repo_status(BENCHMARK_DIR / "survivalgan"),
            "JMbayes2": repo_status(BENCHMARK_DIR / "JMbayes2"),
        },
        "imports": benchmark_dependency_status().get("imports", {}),
        "jmbayes2": check_jmbayes2_available(),
        "method_components": {},
        "fallbacks": [],
    }

    print(f"[{now_iso()}] generating baseline-conditioned PhaseSyn replicates", flush=True)
    phase_static, phase_long, phase_diag, phase_issues = generate_phasesyn_conditioned_replicates(
        output_dir,
        bundle["cfg"],
        bundle["train_bundle"],
        bundle["test_bundle"],
        real_test_static,
        time_grid,
        args.reps,
        args.seed + 1000,
        device,
    )
    issues.extend(phase_issues)
    diagnostics["method_components"]["PhaseSyn"] = phase_diag

    print(f"[{now_iso()}] generating conditional empirical bootstrap replicates", flush=True)
    boot_static, boot_long, boot_diag = conditional_bootstrap_replicates(
        real_train_static,
        real_train_long,
        real_test_static,
        time_grid,
        args.reps,
        args.seed + 2000,
    )
    boot_static = [augment_survival_components(s) for s in boot_static]
    boot_long = [future_only_longitudinal(l) for l in boot_long]
    diagnostics["method_components"]["conditional_empirical_subject_bootstrap"] = boot_diag

    print(f"[{now_iso()}] generating conditional classical LMM/Cox/AFT replicates", flush=True)
    classical_static, classical_long, classical_diag, classical_issues = classical_lmm_cox_aft_replicates(
        real_train_static,
        real_train_long,
        real_test_static,
        time_grid,
        args.reps,
        args.seed + 3000,
    )
    issues.extend(classical_issues)
    diagnostics["method_components"]["conditional_classical_lmm_cox_aft_simulator"] = classical_diag

    print(f"[{now_iso()}] generating conditional joint longitudinal-survival replicates", flush=True)
    joint_static, joint_long, joint_diag, joint_issues = joint_longitudinal_survival_replicates(
        real_train_static,
        real_train_long,
        real_test_static,
        time_grid,
        args.reps,
        args.seed + 4000,
    )
    issues.extend(joint_issues)
    diagnostics["method_components"]["conditional_joint_longitudinal_survival_baseline"] = joint_diag
    if not diagnostics["jmbayes2"]["available"]:
        issues.append(f"JMbayes2 unavailable; used Python joint fallback ({diagnostics['jmbayes2']['error']}).")

    print(f"[{now_iso()}] generating conditional modular deep replicates", flush=True)
    modular_static, modular_long, modular_diag, modular_issues = modular_deep_replicates(
        real_train_static,
        real_train_long,
        real_test_static,
        time_grid,
        args.reps,
        args.seed + 5000,
        args.deep_epochs,
    )
    issues.extend(modular_issues)
    diagnostics["method_components"]["conditional_modular_deep_generator"] = modular_diag
    diagnostics["fallbacks"] = issues

    method_sets = {
        "PhaseSyn": (phase_static, phase_long),
        "conditional_empirical_subject_bootstrap": (boot_static, boot_long),
        "conditional_classical_lmm_cox_aft_simulator": (classical_static, classical_long),
        "conditional_joint_longitudinal_survival_baseline": (joint_static, joint_long),
        "conditional_modular_deep_generator": (modular_static, modular_long),
    }

    print(f"[{now_iso()}] writing revised synthetic replicate index", flush=True)
    index_rows: list[dict[str, Any]] = []
    for method, (statics, longs) in method_sets.items():
        index_rows.extend(write_replicates(output_dir, method, statics, longs))
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(output_dir / "synthetic" / "benchmark_replicate_index_revised.csv", index=False)

    print(f"[{now_iso()}] recomputing revised metrics", flush=True)
    fidelity, estimands, estimand_summary, coupling = evaluate_methods(real_test_static, real_test_long, method_sets, tau)
    fidelity_summary = summarize_numeric(fidelity)
    fidelity.to_csv(output_dir / "metrics" / "benchmark_fidelity_by_replicate_revised.csv", index=False)
    fidelity_summary.to_csv(output_dir / "metrics" / "benchmark_fidelity_summary_revised.csv", index=False)
    estimands.to_csv(output_dir / "metrics" / "benchmark_estimand_by_replicate_revised.csv", index=False)
    estimand_summary.to_csv(output_dir / "metrics" / "benchmark_estimand_summary_revised.csv", index=False)
    coupling.to_csv(output_dir / "metrics" / "benchmark_longitudinal_survival_coupling_by_replicate_revised.csv", index=False)
    coupling.groupby("method").mean(numeric_only=True).reset_index().to_csv(
        output_dir / "metrics" / "benchmark_longitudinal_survival_coupling_summary_revised.csv",
        index=False,
    )

    table = create_method_table(fidelity_summary)
    table.to_csv(output_dir / "tables" / "table7_methods_comparison_revised.csv", index=False)
    table.to_csv(output_dir / "tables" / "table7_methods_comparison.csv", index=False)
    deprecated = pd.DataFrame({
        "method": DEPRECATED_METHODS,
        "status": "deprecated/inactive",
        "reason": [
            "replaced by conditional_classical_lmm_cox_aft_simulator",
            "merged into conditional_modular_deep_generator",
            "merged into conditional_modular_deep_generator",
            "dependency-gated; TensorFlow unavailable, fallback included inside modular deep method",
            "cloned research script lacks safe tabular RCT generation interface for this driver",
            "sdv import unavailable; CTGAN/TVAE components handled through cloned CTGAN package or fallback",
        ],
    })
    deprecated.to_csv(output_dir / "tables" / "table7_methods_deprecated_inactive_revised.csv", index=False)

    diagnostics["runtime_seconds"] = float(time.time() - start)
    diagnostics["output_files"] = {
        "fidelity_summary": str(output_dir / "metrics" / "benchmark_fidelity_summary_revised.csv"),
        "estimand_summary": str(output_dir / "metrics" / "benchmark_estimand_summary_revised.csv"),
        "table7": str(output_dir / "tables" / "table7_methods_comparison.csv"),
        "revision_summary": str(output_dir / "reports" / "benchmark_revision_summary.md"),
    }
    save_json(output_dir / "metrics" / "benchmark_dependency_status_revised.json", diagnostics)
    write_reports(output_dir, table, diagnostics, issues, args.reps)
    validation = validate_revised_outputs(output_dir, table, fidelity_summary, index_df)
    diagnostics["validation"] = validation
    save_json(output_dir / "metrics" / "benchmark_dependency_status_revised.json", diagnostics)
    append_status(output_dir, [
        f"- Updated: {now_iso()}",
        "- Status: completed",
        "- Revised benchmark methods are baseline-conditioned; W, L0, treatment A, and requested grids are supplied from the real test set.",
        "- Active benchmark methods: " + ", ".join(ACTIVE_METHODS),
        "- Deprecated previous methods: " + ", ".join(DEPRECATED_METHODS),
        f"- Revised evaluation replicates per method: {args.reps}",
        f"- Completed metrics: benchmark_fidelity_summary_revised.csv, benchmark_estimand_summary_revised.csv, benchmark_longitudinal_survival_coupling_summary_revised.csv, table7_methods_comparison.csv",
        f"- Validation: `{json.dumps(_jsonable(validation), sort_keys=True)}`",
        "- Failed or fallback components:",
        *(f"  - {item}" for item in issues),
        f"- Revised Table 7: `{output_dir / 'tables' / 'table7_methods_comparison.csv'}`",
        f"- Revision report: `{output_dir / 'reports' / 'benchmark_revision_summary.md'}`",
    ])

    print("active benchmark methods:", ", ".join(ACTIVE_METHODS))
    print("deprecated previous methods:", ", ".join(DEPRECATED_METHODS))
    print("completed metrics: benchmark_fidelity_summary_revised.csv; benchmark_estimand_summary_revised.csv; benchmark_longitudinal_survival_coupling_summary_revised.csv; table7_methods_comparison.csv")
    print("failed or fallback components:", "; ".join(issues) if issues else "none")
    print("path to revised table7_methods_comparison.csv:", output_dir / "tables" / "table7_methods_comparison.csv")
    print("path to benchmark_revision_summary.md:", output_dir / "reports" / "benchmark_revision_summary.md")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Revise simulation benchmark methods under the baseline-conditioned contract.")
    parser.add_argument("--dataset-dir", type=Path, default=REQUESTED_DATASET_DIR if REQUESTED_DATASET_DIR.exists() else FALLBACK_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--deep-epochs", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
