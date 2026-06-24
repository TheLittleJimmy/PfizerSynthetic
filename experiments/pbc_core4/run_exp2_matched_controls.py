from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .load_pbc import TREATMENT_NAME, project_path
from .methods import analysis_static
from .metrics import BASELINE_COLUMNS, baseline_fidelity, clinical_estimands, smd
from .plotting import plot_line, plot_metric_bar, plot_smd
from .report import write_markdown_table


def _alignment(treated: pd.DataFrame, synth: pd.DataFrame, comparison: str) -> dict[str, Any]:
    smds = [abs(smd(treated[c], synth[c])) for c in BASELINE_COLUMNS if c in treated and c in synth]
    auc = np.nan
    try:
        cols = [c for c in BASELINE_COLUMNS if c in treated and c in synth]
        x = pd.concat([treated[cols], synth[cols]], ignore_index=True).apply(pd.to_numeric, errors="coerce")
        x = x.fillna(x.median(numeric_only=True).fillna(0.0))
        y = np.concatenate([np.ones(len(treated)), np.zeros(len(synth))])
        if len(np.unique(y)) == 2:
            xs = StandardScaler().fit_transform(x)
            clf = LogisticRegression(max_iter=500).fit(xs, y)
            auc = float(roc_auc_score(y, clf.predict_proba(xs)[:, 1]))
    except Exception:
        auc = np.nan
    return {"comparison": comparison, "mean_abs_smd": float(np.nanmean(smds)) if smds else np.nan, "propensity_auc": auc}


def _same_treatment_null(
    cfg: dict[str, Any],
    static: pd.DataFrame,
    phasesyn: Any,
    treatment: int,
    replicates: int,
    smoke: bool,
) -> dict[str, Any]:
    reps = min(replicates, int(cfg.get("smoke", {}).get("exp4_replicates", 3))) if smoke else int(replicates)
    rng = np.random.default_rng(int(cfg["seed"]) + 9100 + int(treatment))
    pvals: list[float] = []
    hr_vals: list[float] = []
    n_per_arm = min(80, max(12, len(static) // 2))
    batch_size = int(cfg.get("generation", {}).get("exp2_null_batch_size", 25 if not smoke else reps))
    batch_size = max(1, min(batch_size, reps))
    for start in range(0, reps, batch_size):
        batch_reps = list(range(start, min(reps, start + batch_size)))
        left_targets = []
        right_targets = []
        for r in batch_reps:
            target = static.sample(2 * n_per_arm, replace=True, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
            left = target.iloc[:n_per_arm].copy()
            right = target.iloc[n_per_arm:].copy()
            left[TREATMENT_NAME] = int(treatment)
            right[TREATMENT_NAME] = int(treatment)
            left["null_replicate"] = int(r)
            right["null_replicate"] = int(r)
            left["subject_id"] = np.arange(len(left), dtype=int) + r * (2 * n_per_arm + 1)
            right["subject_id"] = np.arange(len(right), dtype=int) + r * (2 * n_per_arm + 1) + n_per_arm
            left_targets.append(left)
            right_targets.append(right)
        left_all = pd.concat(left_targets, ignore_index=True)
        right_all = pd.concat(right_targets, ignore_index=True)
        s0_all, l0_all, _ = phasesyn.generate(len(left_all), treatment=int(treatment), target_baseline=left_all)
        s1_all, l1_all, _ = phasesyn.generate(len(right_all), treatment=int(treatment), target_baseline=right_all)
        for r in batch_reps:
            s0 = s0_all[s0_all["null_replicate"].eq(r)].drop(columns=["null_replicate"], errors="ignore").reset_index(drop=True)
            s1 = s1_all[s1_all["null_replicate"].eq(r)].drop(columns=["null_replicate"], errors="ignore").reset_index(drop=True)
            ids0 = set(pd.to_numeric(s0["subject_id"], errors="coerce").dropna().astype(int))
            ids1 = set(pd.to_numeric(s1["subject_id"], errors="coerce").dropna().astype(int))
            if "null_replicate" in l0_all:
                l0 = l0_all[l0_all["null_replicate"].eq(r)].drop(columns=["null_replicate"], errors="ignore").reset_index(drop=True)
            else:
                l0 = l0_all[pd.to_numeric(l0_all["subject_id"], errors="coerce").astype("Int64").isin(ids0)].reset_index(drop=True)
            if "null_replicate" in l1_all:
                l1 = l1_all[l1_all["null_replicate"].eq(r)].drop(columns=["null_replicate"], errors="ignore").reset_index(drop=True)
            else:
                l1 = l1_all[pd.to_numeric(l1_all["subject_id"], errors="coerce").astype("Int64").isin(ids1)].reset_index(drop=True)
            s0[TREATMENT_NAME] = 0
            s1[TREATMENT_NAME] = 1
            l0[TREATMENT_NAME] = 0
            l1[TREATMENT_NAME] = 1
            est = clinical_estimands(pd.concat([s0, s1], ignore_index=True), pd.concat([l0, l1], ignore_index=True), "PhaseSyn_null", r, f"A{treatment}_vs_A{treatment}")
            pvals.append(est.get("logrank_p", np.nan))
            hr_vals.append(est.get("cox_hr", np.nan))
    p = np.asarray(pvals, dtype=float)
    return {
        "R": np.nan,
        "test": f"synthetic_A{treatment}_vs_A{treatment}",
        "empirical_type1_error": float(np.nanmean(p < 0.05)) if np.isfinite(p).any() else np.nan,
        "replicates": int(reps),
        "mean_hr_under_null": float(np.nanmean(hr_vals)) if np.isfinite(hr_vals).any() else np.nan,
        "status": "proper_same_treatment_synthetic_null",
    }


def run_exp2(cfg: dict[str, Any], data: Any, methods: dict[str, Any], phasesyn: Any | None, smoke: bool = False) -> dict[str, pd.DataFrame]:
    out = project_path(cfg["output_dir"]) / "exp2_matched_counterfactual_controls"
    tables, figures, reports = out / "tables", out / "figures", out / "reports"
    for p in [tables, figures, reports]:
        p.mkdir(parents=True, exist_ok=True)
    static = analysis_static(data)
    long_df = data.longitudinal.copy()
    test_ids = set(data.splits["test"])
    treated = static[static["subject_id"].isin(test_ids) & static[TREATMENT_NAME].eq(1)].reset_index(drop=True)
    original_control = static[static[TREATMENT_NAME].eq(0)].reset_index(drop=True)
    if smoke:
        cap = int(cfg["smoke"].get("max_eval_subjects", 24))
        treated = treated.head(cap).reset_index(drop=True)
        original_control = original_control.head(cap).reset_index(drop=True)
    alignment_rows = [_alignment(treated, original_control, "treated_vs_original_randomized_controls")]
    effects_rows = []
    variance_rows = []
    null_rows = []
    if phasesyn is not None and not treated.empty:
        for R in cfg["generation"]["exp2_replicates_by_R"]:
            reps = min(int(R), 3) if smoke else int(R)
            statics, longs = [], []
            for r in range(reps):
                s, l, _ = phasesyn.generate(len(treated), treatment=0, target_baseline=treated)
                s["counterfactual_draw"] = r
                l["counterfactual_draw"] = r
                statics.append(s)
                longs.append(l)
            synth = pd.concat(statics, ignore_index=True)
            synth_long = pd.concat(longs, ignore_index=True)
            alignment_rows.append(_alignment(treated, synth, f"treated_vs_phasesyn_matched_R{R}"))
            trial_static = pd.concat([treated, synth.drop(columns=["counterfactual_draw"], errors="ignore")], ignore_index=True)
            trial_long = pd.concat([long_df[long_df["subject_id"].isin(treated["subject_id"])], synth_long.drop(columns=["counterfactual_draw"], errors="ignore")], ignore_index=True)
            est = clinical_estimands(trial_static, trial_long, "PhaseSyn", int(R), f"matched_R={R}")
            est["R"] = int(R)
            effects_rows.append(est)
            variance_rows.append({
                "R": int(R),
                "treatment_effect_standard_error": est.get("cox_se", np.nan),
                "confidence_interval_width": 3.92 * est.get("cox_se", np.nan) if np.isfinite(est.get("cox_se", np.nan)) else np.nan,
            })
        null_reps = int(cfg.get("generation", {}).get("exp2_null_replicates", 500))
        null_rows.append(_same_treatment_null(cfg, static, phasesyn, 0, null_reps, smoke))
        null_rows.append(_same_treatment_null(cfg, static, phasesyn, 1, null_reps, smoke))
    for name, gen in methods.items():
        if treated.empty:
            continue
        try:
            s, l, _ = gen.generate(len(treated), treatment=0, target_baseline=treated)
            alignment_rows.append(_alignment(treated, s, f"treated_vs_{name}_approximation"))
            effects_rows.append(clinical_estimands(pd.concat([treated, s], ignore_index=True), pd.concat([long_df[long_df["subject_id"].isin(treated["subject_id"])], l], ignore_index=True), name, 0, "benchmark_approximation"))
        except Exception as exc:
            effects_rows.append({"method": name, "status": f"failed: {type(exc).__name__}: {exc}"})
    alignment_df = pd.DataFrame(alignment_rows)
    effects_df = pd.DataFrame(effects_rows)
    variance_df = pd.DataFrame(variance_rows)
    null_df = pd.DataFrame(null_rows)
    alignment_df.to_csv(tables / "exp2_baseline_alignment.csv", index=False)
    effects_df.to_csv(tables / "exp2_treatment_effects.csv", index=False)
    variance_df.to_csv(tables / "exp2_variance_vs_R.csv", index=False)
    null_df.to_csv(tables / "exp2_null_calibration.csv", index=False)
    plot_smd(alignment_df, figures / "smd_treated_vs_controls.pdf")
    plot_metric_bar(effects_df.fillna(np.nan), "cox_hr", figures / "treatment_effect_comparison.pdf")
    plot_line(variance_df, "R", "confidence_interval_width", None, figures / "variance_vs_R.pdf")
    write_markdown_table(reports / "exp2_summary.md", "Experiment 2 Summary", effects_df, "Do not interpret matched real-data draws as individual counterfactual truth.")
    return {"alignment": alignment_df, "effects": effects_df, "variance": variance_df, "null": null_df}
