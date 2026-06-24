from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .load_pbc import TREATMENT_NAME, project_path
from .methods import analysis_static
from .metrics import digital_twin_metrics
from .plotting import plot_metric_bar, plot_responder_km, plot_survival_calibration
from .report import write_markdown_table


def _posterior_predictive(phasesyn: Any, test_static: pd.DataFrame, samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    targets = []
    for r in range(samples):
        target = test_static.copy()
        target["sample"] = int(r)
        target["_original_subject_id"] = target["subject_id"].astype(int)
        target["subject_id"] = np.arange(len(target), dtype=int) + r * (len(target) + 1)
        targets.append(target)
    target_all = pd.concat(targets, ignore_index=True)
    all_s, all_l, _ = phasesyn.generate(len(target_all), treatment=None, target_baseline=target_all)
    id_map = target_all.set_index("subject_id")["_original_subject_id"].to_dict()
    sample_map = target_all.set_index("subject_id")["sample"].to_dict()
    all_s["sample"] = all_s["subject_id"].map(sample_map).astype(int)
    all_s["subject_id"] = all_s["subject_id"].map(id_map).astype(int)
    all_l["sample"] = all_l["subject_id"].map(sample_map).astype(int)
    all_l["subject_id"] = all_l["subject_id"].map(id_map).astype(int)
    pred_s = all_s.groupby("subject_id").agg(time=("time", "mean"), event=("event", "mean")).reset_index()
    pred_s["survival_risk_score"] = pred_s["event"]
    pred_l_rows = []
    for (sid, t), g in all_l.groupby(["subject_id", "visit_time"]):
        row = {"subject_id": sid, "visit_time": t}
        for col in [c for c in g.columns if c not in {"subject_id", "visit_index", "visit_time", TREATMENT_NAME, "sample"}]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna()
            row[col] = float(vals.mean()) if not vals.empty else np.nan
            for level, qlo, qhi in [(50, 0.25, 0.75), (80, 0.10, 0.90), (95, 0.025, 0.975)]:
                row[f"{col}_lo{level}"] = float(vals.quantile(qlo)) if not vals.empty else np.nan
                row[f"{col}_hi{level}"] = float(vals.quantile(qhi)) if not vals.empty else np.nan
        pred_l_rows.append(row)
    return pred_s, pd.DataFrame(pred_l_rows), all_s.reset_index(drop=True), all_l.reset_index(drop=True)


def run_exp3(cfg: dict[str, Any], data: Any, phasesyn: Any | None, smoke: bool = False) -> dict[str, pd.DataFrame]:
    out = project_path(cfg["output_dir"]) / "exp3_digital_twin_validation"
    tables, figures, reports, samples_dir = out / "tables", out / "figures", out / "reports", out / "predictive_samples"
    for p in [tables, figures, reports, samples_dir]:
        p.mkdir(parents=True, exist_ok=True)
    static = analysis_static(data)
    test_static = static[static["subject_id"].isin(data.splits["test"])].reset_index(drop=True)
    real_long = data.longitudinal[data.longitudinal["subject_id"].isin(data.splits["test"])].reset_index(drop=True)
    if smoke:
        cap = int(cfg["smoke"].get("max_eval_subjects", 24))
        test_static = test_static.head(cap).reset_index(drop=True)
        real_long = real_long[real_long["subject_id"].isin(test_static["subject_id"])].reset_index(drop=True)
    if phasesyn is None or test_static.empty:
        long_pred = pd.DataFrame([{"method": "PhaseSyn", "status": "not_run"}])
        surv_pred = pd.DataFrame([{"method": "PhaseSyn", "status": "not_run"}])
        coverage = pd.DataFrame()
        landmark = pd.DataFrame()
    else:
        samples = int(cfg["smoke"]["exp3_posterior_samples"] if smoke else cfg["generation"]["exp3_posterior_samples"])
        pred_static, pred_long, sample_static, sample_long = _posterior_predictive(phasesyn, test_static, samples)
        sample_static.to_csv(samples_dir / "exp3_static_predictive_samples.csv", index=False)
        sample_long.to_csv(samples_dir / "exp3_longitudinal_predictive_samples.csv", index=False)
        long_pred, surv_pred, coverage, landmark = digital_twin_metrics(real_long, pred_long, test_static, pred_static)
        if not coverage.empty:
            width_rows = []
            for var in [c for c in pred_long.columns if not c.startswith("_")]:
                if var in {"subject_id", "visit_time"} or f"{var}_lo50" not in pred_long:
                    continue
                for level in [50, 80, 95]:
                    lo = pd.to_numeric(pred_long[f"{var}_lo{level}"], errors="coerce")
                    hi = pd.to_numeric(pred_long[f"{var}_hi{level}"], errors="coerce")
                    width_rows.append({"method": "PhaseSyn", "variable": var, "interval": level, "mean_interval_width": float((hi - lo).mean())})
            width_df = pd.DataFrame(width_rows)
            if not width_df.empty:
                coverage = coverage.merge(width_df, on=["method", "variable", "interval"], how="left")
    long_pred.to_csv(tables / "exp3_longitudinal_prediction.csv", index=False)
    surv_pred.to_csv(tables / "exp3_survival_prediction.csv", index=False)
    coverage.to_csv(tables / "exp3_prediction_interval_coverage.csv", index=False)
    landmark.to_csv(tables / "exp3_landmark_coupling.csv", index=False)
    coverage_plot = coverage.copy()
    if not coverage_plot.empty:
        coverage_plot["method"] = coverage_plot["variable"].astype(str) + "_" + coverage_plot["interval"].astype(str)
    plot_metric_bar(coverage_plot, "coverage", figures / "prediction_interval_coverage_by_biomarker.pdf")
    plot_survival_calibration(test_static, pred_static if "pred_static" in locals() else pd.DataFrame(), figures / "survival_calibration.pdf")
    plot_responder_km(test_static, real_long, figures / "responder_stratified_km.pdf")
    write_markdown_table(reports / "exp3_summary.md", "Experiment 3 Summary", long_pred, "Factual validation only; treatment remains randomized A_i.")
    return {"longitudinal": long_pred, "survival": surv_pred, "coverage": coverage, "landmark": landmark}
