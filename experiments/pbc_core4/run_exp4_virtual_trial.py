from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .load_pbc import TREATMENT_NAME, project_path
from .methods import analysis_static
from .metrics import baseline_fidelity, clinical_estimands, longitudinal_fidelity, survival_fidelity
from .plotting import plot_line, placeholder_pdf
from .report import write_markdown_table


def _semisynthetic(static: pd.DataFrame, gamma: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = static.copy().reset_index(drop=True)
    age = pd.to_numeric(out["age"], errors="coerce").fillna(out["age"].median())
    trt = pd.to_numeric(out[TREATMENT_NAME], errors="coerce").fillna(0)
    lin = 0.02 * (age - age.mean()) + gamma * trt
    base_rate = 0.08
    event_time = rng.exponential(1.0 / np.maximum(base_rate * np.exp(lin), 1e-5))
    censor_time = rng.exponential(8.0, size=len(out))
    out["time"] = np.minimum(event_time, censor_time)
    out["event"] = (event_time <= censor_time).astype(int)
    return out


def _split_trial_replicates(static: pd.DataFrame, long_df: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    out = []
    if "replicate" not in static:
        return [(0, static.reset_index(drop=True), long_df.reset_index(drop=True))]
    for rep, s in static.groupby("replicate", sort=True):
        ids = set(pd.to_numeric(s["subject_id"], errors="coerce").astype(int))
        if "replicate" in long_df:
            l = long_df[long_df["replicate"].eq(rep)].copy()
        else:
            l = long_df[pd.to_numeric(long_df["subject_id"], errors="coerce").astype(int).isin(ids)].copy()
        out.append((int(rep), s.drop(columns=["replicate"], errors="ignore").reset_index(drop=True), l.drop(columns=["replicate"], errors="ignore").reset_index(drop=True)))
    return out


def _calibration_components(real_static: pd.DataFrame, real_long: pd.DataFrame, synth_static: pd.DataFrame, synth_long: pd.DataFrame) -> dict[str, float]:
    b = baseline_fidelity(real_static, synth_static, "calibration", 0)
    l = longitudinal_fidelity(real_long, synth_long, "calibration", 0)
    s = survival_fidelity(real_static, synth_static, "calibration", 0)
    return {
        "baseline_distance": float(b.get("baseline_mean_abs_smd", np.nan)),
        "trajectory_distance": float(l.get("longitudinal_mean_trajectory_error", np.nan)),
        "km_distance": float(s.get("survival_km_integrated_abs_distance", np.nan)),
        "event_rate_error_abs": float(abs(s.get("survival_event_rate_error", np.nan))),
        "censoring_rate_error_abs": float(abs(s.get("survival_censoring_rate_error", np.nan))),
    }


def _calibration_score(components: dict[str, float], scale: dict[str, float] | None = None) -> float:
    weights = {
        "km_distance": 0.35,
        "baseline_distance": 0.20,
        "trajectory_distance": 0.25,
        "event_rate_error_abs": 0.10,
        "censoring_rate_error_abs": 0.10,
    }
    total = 0.0
    for key, weight in weights.items():
        value = components.get(key, np.nan)
        denom = 1.0 if scale is None else max(float(scale.get(key, 1.0)), 1e-8)
        total += weight * (float(value) / denom if np.isfinite(value) else 1e6)
    return float(total)


def _validation_threshold(static: pd.DataFrame, long_df: pd.DataFrame, seed: int, quantile: float = 0.90, reps: int = 60) -> tuple[float, dict[str, float]]:
    control = static[static[TREATMENT_NAME].eq(0)].reset_index(drop=True)
    control_long = long_df[long_df["subject_id"].isin(control["subject_id"])].reset_index(drop=True)
    if control.empty or control_long.empty:
        return 1.0, {"km_distance": 1.0, "baseline_distance": 1.0, "trajectory_distance": 1.0, "event_rate_error_abs": 1.0, "censoring_rate_error_abs": 1.0}
    rng = np.random.default_rng(seed + 404)
    comps = []
    scores_raw = []
    for r in range(int(reps)):
        sample = control.sample(len(control), replace=True, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
        sample_ids = sample["subject_id"].astype(int).tolist()
        sample_long = control_long[control_long["subject_id"].isin(sample_ids)].copy()
        comp = _calibration_components(control, control_long, sample, sample_long)
        comps.append(comp)
    scale = {
        key: float(np.nanpercentile([c.get(key, np.nan) for c in comps], 90)) if comps else 1.0
        for key in ["km_distance", "baseline_distance", "trajectory_distance", "event_rate_error_abs", "censoring_rate_error_abs"]
    }
    scale = {k: (v if np.isfinite(v) and v > 1e-8 else 1.0) for k, v in scale.items()}
    for comp in comps:
        scores_raw.append(_calibration_score(comp, scale))
    threshold = float(np.nanquantile(scores_raw, quantile)) if scores_raw else 1.0
    return threshold, scale


def run_exp4(cfg: dict[str, Any], data: Any, phasesyn: Any | None, smoke: bool = False) -> dict[str, pd.DataFrame]:
    out = project_path(cfg["output_dir"]) / "exp4_virtual_trial_simulation"
    tables, figures, reports = out / "tables", out / "figures", out / "reports"
    for p in [tables, figures, reports]:
        p.mkdir(parents=True, exist_ok=True)
    reps = int(cfg["smoke"]["exp4_replicates"] if smoke else cfg["generation"]["exp4_replicates"])
    static = analysis_static(data)
    long_df = data.longitudinal.copy()
    semi_rows, recover_rows = [], []
    for gamma in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        pvals, hrs = [], []
        for r in range(reps):
            sim = _semisynthetic(static, gamma, int(cfg["seed"]) + r + int(gamma * 1000))
            est = clinical_estimands(sim, long_df, "semi_synthetic", r, f"gamma={gamma}")
            pvals.append(est.get("logrank_p", np.nan))
            hrs.append(est.get("cox_hr", np.nan))
        semi_rows.append({
            "gamma_A": gamma,
            "type1_error" if gamma == 0 else "power": float(np.nanmean(np.asarray(pvals) < 0.05)),
            "alpha": 0.05,
            "replicates": reps,
        })
        recover_rows.append({
            "gamma_A": gamma,
            "cox_hr_mean": float(np.nanmean(hrs)),
            "target_hr": float(np.exp(gamma)),
            "rmst_recovery_status": "computed_in_estimand_rows",
            "alpha_recovery_status": "not_identified_from_real_pbc",
        })
    real_rows, accrual_rows, cal_rows = [], [], []
    if phasesyn is not None:
        gen_cfg = cfg.get("generation", {})
        sample_sizes = cfg["smoke"].get("exp4_sample_sizes", [24, 32]) if smoke else gen_cfg.get("exp4_sample_sizes", [80, 120, 160, 240, 312, 500])
        ratios = cfg["smoke"].get("exp4_allocation_ratios", ["1:1"]) if smoke else gen_cfg.get("exp4_allocation_ratios", ["1:1", "2:1", "1:2"])
        threshold_q = float(gen_cfg.get("exp4_calibration_threshold_quantile", 0.90))
        cal_threshold, cal_scale = _validation_threshold(static, long_df, int(cfg["seed"]), threshold_q, reps=10 if smoke else 60)
        progress_rows = []
        progress_path = tables / "exp4_progress.csv"
        total_designs = len(sample_sizes) * len(ratios)
        design_index = 0
        for n in sample_sizes:
            for ratio in ratios:
                design_index += 1
                design_start = time.time()
                trt_prob = {"1:1": 0.5, "2:1": 2 / 3, "1:2": 1 / 3}[ratio]
                pvals, events, censoring, hrs, pass_flags, cal_scores = [], [], [], [], [], []
                train_age_mean = float(pd.to_numeric(static["age"], errors="coerce").mean())
                train_event_rate = float(static["event"].mean())
                batch_size = int(cfg.get("generation", {}).get("exp4_trial_batch_size", 25 if not smoke else reps))
                for start in range(0, reps, max(1, batch_size)):
                    targets = []
                    for r in range(start, min(reps, start + max(1, batch_size))):
                        target = static.sample(n, replace=True, random_state=int(cfg["seed"]) + n + r).reset_index(drop=True)
                        target[TREATMENT_NAME] = np.random.default_rng(int(cfg["seed"]) + r).binomial(1, trt_prob, size=n)
                        target["replicate"] = int(r)
                        target["subject_id"] = np.arange(len(target), dtype=int) + r * (n + 1)
                        targets.append(target)
                    target_all = pd.concat(targets, ignore_index=True)
                    static_batch, long_batch, _ = phasesyn.generate(len(target_all), treatment=None, target_baseline=target_all)
                    for r, s, l in _split_trial_replicates(static_batch, long_batch):
                        s = s.drop(columns=["sample"], errors="ignore")
                        l = l.drop(columns=["sample"], errors="ignore")
                        est = clinical_estimands(s, l, "PhaseSyn", r, f"n={n},ratio={ratio}")
                        pvals.append(est.get("logrank_p", np.nan))
                        hrs.append(est.get("cox_hr", np.nan))
                        events.append(float(s["event"].sum()))
                        censoring.append(float(1.0 - s["event"].mean()))
                        comp = _calibration_components(static, long_df, s, l)
                        score = _calibration_score(comp, cal_scale)
                        cal_scores.append(score)
                        pass_flags.append(bool(score <= cal_threshold))
                    progress_rows.append({
                        "n": n,
                        "allocation_ratio": ratio,
                        "design_index": design_index,
                        "total_designs": total_designs,
                        "completed_replicates": min(reps, start + max(1, batch_size)),
                        "target_replicates": reps,
                        "status": "running",
                        "elapsed_seconds_design": float(time.time() - design_start),
                    })
                    pd.DataFrame(progress_rows).to_csv(progress_path, index=False)
                pvals_arr = np.asarray(pvals, dtype=float)
                pass_arr = np.asarray(pass_flags, dtype=bool)
                score_arr = np.asarray(cal_scores, dtype=float)
                if len(pass_arr) and not pass_arr.any() and np.isfinite(score_arr).any():
                    pass_arr[int(np.nanargmin(score_arr))] = True
                    forced_best = True
                else:
                    forced_best = False
                real_rows.append({
                    "n": n,
                    "allocation_ratio": ratio,
                    "design_grid": "PBC-sized virtual trials",
                    "power": float(np.nanmean(pvals_arr < 0.05)),
                    "hr_mean": float(np.nanmean(hrs)),
                    "hr_sd": float(np.nanstd(hrs)),
                    "replicates": reps,
                })
                accrual_rows.append({
                    "n": n,
                    "allocation_ratio": ratio,
                    "design_grid": "PBC-sized virtual trials",
                    "expected_number_events": float(np.nanmean(events)),
                    "censoring_rate": float(np.nanmean(censoring)),
                })
                cal_rows.append({
                    "n": n,
                    "allocation_ratio": ratio,
                    "design_grid": "PBC-sized virtual trials",
                    "before_filter_power": real_rows[-1]["power"],
                    "after_filter_power": float(np.nanmean(pvals_arr[pass_arr] < 0.05)) if pass_arr.any() else np.nan,
                    "pass_rate": float(pass_arr.mean()) if len(pass_arr) else np.nan,
                    "forced_best": bool(forced_best),
                    "calibration_score_threshold": cal_threshold,
                    "mean_calibration_score": float(np.nanmean(score_arr)) if len(score_arr) else np.nan,
                    "filter_rule": f"weighted validation-bootstrap calibration score <= {cal_threshold:.4f} at q={threshold_q}",
                    "ordinary_fidelity_sufficient_for_calibration": False,
                })
                progress_rows.append({
                    "n": n,
                    "allocation_ratio": ratio,
                    "design_index": design_index,
                    "total_designs": total_designs,
                    "completed_replicates": reps,
                    "target_replicates": reps,
                    "status": "completed",
                    "elapsed_seconds_design": float(time.time() - design_start),
                })
                pd.DataFrame(progress_rows).to_csv(progress_path, index=False)
    semi_df = pd.DataFrame(semi_rows)
    recover_df = pd.DataFrame(recover_rows)
    real_df = pd.DataFrame(real_rows)
    accrual_df = pd.DataFrame(accrual_rows)
    cal_df = pd.DataFrame(cal_rows)
    semi_df.to_csv(tables / "exp4_semisynthetic_type1_power.csv", index=False)
    recover_df.to_csv(tables / "exp4_semisynthetic_ground_truth_recovery.csv", index=False)
    real_df.to_csv(tables / "exp4_real_virtual_trial_power.csv", index=False)
    accrual_df.to_csv(tables / "exp4_event_accrual.csv", index=False)
    cal_df.to_csv(tables / "exp4_calibration_filtering.csv", index=False)
    plot_line(semi_df.fillna(0), "gamma_A", "power" if "power" in semi_df else "type1_error", None, figures / "type1_power_curves.pdf")
    plot_line(real_df, "n", "power", "allocation_ratio", figures / "power_vs_sample_size.pdf")
    plot_line(accrual_df, "n", "expected_number_events", "allocation_ratio", figures / "event_accrual_curves.pdf")
    plot_line(cal_df, "n", "after_filter_power", "allocation_ratio", figures / "calibration_before_after.pdf")
    write_markdown_table(reports / "exp4_summary.md", "Experiment 4 Summary", real_df, "Fidelity metrics are not assumed sufficient for type-I/power calibration.")
    return {"semi": semi_df, "recovery": recover_df, "real": real_df, "accrual": accrual_df, "calibration": cal_df}
