from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def loss_decrease(curves: pd.DataFrame) -> float:
    clean = curves["loss"].replace([float("inf"), float("-inf")], pd.NA).dropna()
    if len(clean) < 2:
        return 0.0
    start = float(clean.iloc[0])
    end = float(clean.iloc[-1])
    return (start - end) / max(abs(start), 1e-8)


def nan_epochs(curves: pd.DataFrame) -> list[int]:
    return curves.loc[curves.get("nan_epoch", False).astype(bool), "epoch"].astype(int).tolist()


def overfit_gate(
    metrics: dict[str, Any],
    curves: pd.DataFrame,
    cfg: dict[str, Any],
    baseline_event_rate_diff: float | None = None,
) -> dict[str, Any]:
    overfit_cfg = cfg.get("overfit", {})
    loss_drop = loss_decrease(curves)
    nans = nan_epochs(curves)
    rmse_ratio = float(metrics.get("continuous_rmse_ratio", 0.0))
    trend_ratio = float(metrics.get("median_trend_rmse_ratio", 0.0))
    event_diff = float(metrics.get("event_rate_diff", 0.0))
    km_error = float(metrics.get("survival_km_integrated_abs_error", 0.0))
    survival_time_ratio = float(metrics.get("survival_time_rmse_ratio", 0.0))
    survival_event_acc = float(metrics.get("survival_event_accuracy", 0.0))
    static_paired_ratio = float(metrics.get("static_paired_continuous_rmse_ratio", 0.0))
    static_paired_acc = float(metrics.get("static_paired_categorical_accuracy", 0.0))
    static_ks = float(metrics.get("static_continuous_mean_ks", 0.0))
    static_tv = float(metrics.get("static_categorical_mean_tv", 0.0))
    raw_static_ratio = float(metrics.get("raw_model_static_paired_continuous_rmse_ratio", static_paired_ratio))
    raw_static_acc = float(metrics.get("raw_model_static_paired_categorical_accuracy", static_paired_acc))
    raw_survival_ratio = float(metrics.get("raw_model_survival_time_rmse_ratio", survival_time_ratio))
    raw_survival_acc = float(metrics.get("raw_model_survival_event_accuracy", survival_event_acc))
    baseline = event_diff if baseline_event_rate_diff is None else float(baseline_event_rate_diff)
    static_paired_ok = (
        static_paired_ratio <= float(overfit_cfg.get("static_paired_rmse_ratio_threshold", 0.35))
        and static_paired_acc >= float(overfit_cfg.get("static_paired_categorical_accuracy_threshold", 0.90))
    )
    static_marginal_ok = (
        static_ks <= float(overfit_cfg.get("static_continuous_ks_threshold", 0.10))
        and static_tv <= float(overfit_cfg.get("static_categorical_tv_threshold", 0.10))
    )

    checks = {
        "no_nan_epochs": len(nans) == 0,
        "loss_decrease": loss_drop >= float(overfit_cfg.get("min_loss_decrease", 0.02)),
        "rmse_below_mean_imputation": rmse_ratio <= float(overfit_cfg.get("rmse_ratio_threshold", 1.05)),
        "median_trajectory_match": trend_ratio <= float(overfit_cfg.get("median_trend_ratio_threshold", 1.05)),
        "event_rate_non_worse": event_diff <= baseline + float(overfit_cfg.get("event_rate_tolerance", 0.02)),
        "km_shape_match": km_error <= float(overfit_cfg.get("km_error_threshold", 0.05)),
        "survival_subject_time_match": survival_time_ratio <= float(overfit_cfg.get("survival_time_rmse_ratio_threshold", 0.35)),
        "survival_subject_event_match": survival_event_acc >= float(overfit_cfg.get("survival_event_accuracy_threshold", 0.90)),
        "static_subject_continuous_match": static_paired_ratio <= float(overfit_cfg.get("static_paired_rmse_ratio_threshold", 0.35)),
        "static_subject_categorical_match": static_paired_acc >= float(overfit_cfg.get("static_paired_categorical_accuracy_threshold", 0.90)),
        "static_marginal_match": static_marginal_ok or static_paired_ok,
        "raw_model_static_not_collapsed": (
            raw_static_ratio <= float(overfit_cfg.get("raw_model_static_paired_rmse_ratio_threshold", 1.25))
            and raw_static_acc >= float(overfit_cfg.get("raw_model_static_paired_categorical_accuracy_threshold", 0.50))
        ),
        "raw_model_survival_not_collapsed": (
            raw_survival_ratio <= float(overfit_cfg.get("raw_model_survival_time_rmse_ratio_threshold", 1.50))
            and raw_survival_acc >= float(overfit_cfg.get("raw_model_survival_event_accuracy_threshold", 0.50))
        ),
        "valid_inverse_outputs": bool(metrics.get("valid_inverse_outputs", False)),
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "loss_decrease_ratio": loss_drop,
        "nan_epochs": nans,
        "baseline_event_rate_diff": baseline,
    }


def write_diagnostics(path: str | Path, metrics: dict[str, Any], gate: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": metrics, "gate": gate}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
