from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import write_csv


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def evaluate_pos(cfg: dict[str, Any], output_dir: str | Path) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    oracle = pd.read_csv(output / "oracle_true_pos.csv", keep_default_na=False)
    estimates = pd.read_csv(output / "method_pos_estimates.csv", keep_default_na=False)
    merged = estimates.merge(oracle, on=["scenario", "n_phase3"], how="left")
    merged["pos_error"] = merged["pos_hat"] - merged["true_pos"]
    merged["event_rate_error_abs"] = (merged["event_rate_hat"] - merged["true_event_rate"]).abs()
    merged["censoring_rate_error_abs"] = (merged["censoring_rate_hat"] - merged["true_censoring_rate"]).abs()
    write_csv(output / "tables" / "pos_estimates_with_oracle.csv", merged)

    acc = (
        merged.groupby(["method", "scenario", "n_phase3"], dropna=False)
        .agg(
            pos_bias=("pos_error", "mean"),
            pos_rmse=("pos_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            pos_mae=("pos_error", lambda x: float(np.mean(np.abs(x)))),
            event_rate_error=("event_rate_error_abs", "mean"),
            censoring_rate_error=("censoring_rate_error_abs", "mean"),
            mean_pos_hat=("pos_hat", "mean"),
            true_pos=("true_pos", "first"),
        )
        .reset_index()
    )
    write_csv(output / "pos_accuracy_table.csv", acc)
    write_csv(output / "tables" / "pos_bias_rmse_mae.csv", acc)

    null = merged[merged["scenario"].eq("null")].copy()
    null_metrics = (
        null.groupby("method", dropna=False)
        .agg(
            null_bias=("pos_error", "mean"),
            null_rmse=("pos_error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
            false_positive_pos_rate=("pos_hat", lambda x: float(np.mean(np.asarray(x) > 0.20))),
        )
        .reset_index()
        if len(null)
        else pd.DataFrame(columns=["method", "null_bias", "null_rmse", "false_positive_pos_rate"])
    )
    write_csv(output / "null_calibration_metrics.csv", null_metrics)
    write_csv(output / "tables" / "null_calibration_metrics.csv", null_metrics)

    decision = merged.copy()
    threshold = float(cfg.get("go_threshold", 0.8))
    decision["model_go"] = decision["pos_hat"] >= threshold
    decision["oracle_go"] = decision["true_pos"] >= threshold
    decision["correct"] = decision["model_go"].eq(decision["oracle_go"])
    decision_metrics = (
        decision.groupby("method", dropna=False)
        .agg(
            decision_accuracy=("correct", "mean"),
            false_go_rate=("model_go", lambda x: float(np.mean(np.asarray(x) & ~decision.loc[x.index, "oracle_go"].to_numpy()))),
            false_stop_rate=("model_go", lambda x: float(np.mean((~np.asarray(x)) & decision.loc[x.index, "oracle_go"].to_numpy()))),
        )
        .reset_index()
    )
    write_csv(output / "go_no_go_decision_metrics.csv", decision_metrics)
    write_csv(output / "tables" / "go_no_go_decision_metrics.csv", decision_metrics)

    rank_rows = []
    for (method, scenario, replicate), g in merged.groupby(["method", "scenario", "replicate"], dropna=False):
        if g["n_phase3"].nunique() < 2:
            continue
        pred_order = g.sort_values("n_phase3")["pos_hat"].to_numpy()
        oracle_order = g.sort_values("n_phase3")["true_pos"].to_numpy()
        oracle_prefers_larger = bool(oracle_order[-1] >= oracle_order[0])
        model_prefers_larger = bool(pred_order[-1] >= pred_order[0])
        rank_rows.append({
            "method": method,
            "scenario": scenario,
            "replicate": replicate,
            "ranking_correct": model_prefers_larger == oracle_prefers_larger,
        })
    ranking = pd.DataFrame(rank_rows)
    ranking_metrics = (
        ranking.groupby("method", dropna=False)
        .agg(ranking_accuracy=("ranking_correct", "mean"))
        .reset_index()
        if len(ranking)
        else pd.DataFrame(columns=["method", "ranking_accuracy"])
    )
    write_csv(output / "design_ranking_accuracy.csv", ranking_metrics)
    write_csv(output / "tables" / "design_ranking_accuracy.csv", ranking_metrics)

    event_censor = (
        merged.groupby("method", dropna=False)
        .agg(
            event_rate_error=("event_rate_error_abs", "mean"),
            censoring_rate_error=("censoring_rate_error_abs", "mean"),
        )
        .reset_index()
    )
    write_csv(output / "event_censoring_error_table.csv", event_censor)
    write_csv(output / "tables" / "event_censoring_rate_errors.csv", event_censor)

    utility = (
        acc.groupby("method", dropna=False)
        .agg(overall_pos_rmse=("pos_rmse", "mean"))
        .reset_index()
        .merge(null_metrics[["method", "null_bias"]], on="method", how="left")
        .merge(decision_metrics[["method", "false_go_rate", "false_stop_rate"]], on="method", how="left")
        .merge(ranking_metrics, on="method", how="left")
    )
    write_csv(output / "decision_utility_table.csv", utility)
    write_csv(output / "tables" / "decision_utility.csv", utility)
    return {
        "merged": merged,
        "accuracy": acc,
        "null": null_metrics,
        "decision": decision_metrics,
        "ranking": ranking_metrics,
        "event_censor": event_censor,
        "utility": utility,
    }
