from __future__ import annotations

import numpy as np

from pdc2.data import LongitudinalPanel


def _safe_mean(values: np.ndarray, default: float = 0.0) -> float:
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else default


def continuous_longitudinal_metrics(panel: LongitudinalPanel, synthetic_raw: np.ndarray) -> dict[str, float]:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    metrics: dict[str, float] = {}
    rmses = []
    maes = []
    baseline_rmses = []
    trend_rmses = []
    trend_baselines = []

    for idx in panel.continuous_indices:
        spec = panel.specs[idx]
        obs = mask[:, :, idx] & np.isfinite(real[:, :, idx])
        if not obs.any():
            continue
        pred = synthetic_raw[:, :, idx]
        finite_obs = obs & np.isfinite(pred)
        if not finite_obs.any():
            continue
        diff = pred[finite_obs] - real[:, :, idx][finite_obs]
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        mae = float(np.mean(np.abs(diff)))
        mean_value = _safe_mean(real[:, :, idx][finite_obs])
        baseline = float(np.sqrt(np.mean((mean_value - real[:, :, idx][finite_obs]) ** 2)))

        real_medians = []
        pred_medians = []
        for visit in range(real.shape[1]):
            visit_obs = obs[:, visit] & np.isfinite(pred[:, visit])
            if visit_obs.any():
                real_medians.append(float(np.median(real[:, visit, idx][visit_obs])))
                pred_medians.append(float(np.median(pred[:, visit][visit_obs])))
        trend_rmse = float(np.sqrt(np.mean((np.asarray(real_medians) - np.asarray(pred_medians)) ** 2))) if real_medians else 0.0
        if real_medians:
            real_medians_arr = np.asarray(real_medians)
            trend_baseline = float(np.sqrt(np.mean((real_medians_arr.mean() - real_medians_arr) ** 2)))
        else:
            trend_baseline = 0.0

        metrics[f"{spec.name}_rmse"] = rmse
        metrics[f"{spec.name}_mae"] = mae
        metrics[f"{spec.name}_mean_impute_rmse"] = baseline
        metrics[f"{spec.name}_trend_rmse"] = trend_rmse
        metrics[f"{spec.name}_trend_rmse_ratio"] = trend_rmse / max(trend_baseline, 1e-8) if trend_baseline > 0 else 0.0
        rmses.append(rmse)
        maes.append(mae)
        baseline_rmses.append(baseline)
        trend_rmses.append(trend_rmse)
        trend_baselines.append(trend_baseline)

    metrics["continuous_rmse"] = float(np.mean(rmses)) if rmses else 0.0
    metrics["continuous_mae"] = float(np.mean(maes)) if maes else 0.0
    metrics["continuous_mean_impute_rmse"] = float(np.mean(baseline_rmses)) if baseline_rmses else 0.0
    metrics["continuous_rmse_ratio"] = (
        metrics["continuous_rmse"] / max(metrics["continuous_mean_impute_rmse"], 1e-8)
        if baseline_rmses else 0.0
    )
    metrics["median_trend_rmse"] = float(np.mean(trend_rmses)) if trend_rmses else 0.0
    metrics["median_trend_baseline_rmse"] = float(np.mean(trend_baselines)) if trend_baselines else 0.0
    metrics["median_trend_rmse_ratio"] = (
        metrics["median_trend_rmse"] / max(metrics["median_trend_baseline_rmse"], 1e-8)
        if trend_baselines else 0.0
    )
    return metrics


def categorical_longitudinal_metrics(panel: LongitudinalPanel, synthetic_raw: np.ndarray) -> dict[str, float]:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    accs = []
    out: dict[str, float] = {}
    for idx in panel.categorical_indices:
        spec = panel.specs[idx]
        obs = mask[:, :, idx] & np.isfinite(real[:, :, idx])
        if not obs.any():
            continue
        obs = obs & np.isfinite(synthetic_raw[:, :, idx])
        if not obs.any():
            continue
        pred = np.rint(synthetic_raw[:, :, idx]).astype(int)
        acc = float(np.mean(pred[obs] == real[:, :, idx][obs].astype(int)))
        out[f"{spec.name}_accuracy"] = acc
        accs.append(acc)
    out["categorical_accuracy"] = float(np.mean(accs)) if accs else 0.0
    return out


def longitudinal_metrics(panel: LongitudinalPanel, synthetic_raw: np.ndarray) -> dict[str, float]:
    out = continuous_longitudinal_metrics(panel, synthetic_raw)
    out.update(categorical_longitudinal_metrics(panel, synthetic_raw))
    return out


def valid_inverse_outputs(panel: LongitudinalPanel, synthetic_raw: np.ndarray) -> bool:
    if np.isinf(synthetic_raw).any():
        return False
    for idx, spec in enumerate(panel.specs):
        values = synthetic_raw[:, :, idx]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        if spec.type == "pos" and np.min(finite) < -1e-6:
            return False
        if spec.type in {"cat", "ordinal"}:
            rounded = np.rint(finite)
            if np.min(rounded) < 0 or np.max(rounded) >= int(spec.nclass or 1):
                return False
    return True
