from __future__ import annotations

import math
from typing import Any

import numpy as np


def _normal_two_sided_p(z: float) -> float:
    return float(math.erfc(abs(float(z)) / math.sqrt(2.0)))


def _chi_square_1df_p(x: float) -> float:
    return float(math.erfc(math.sqrt(max(float(x), 0.0) / 2.0)))


def _event_time_summaries(T_obs: np.ndarray, delta: np.ndarray, A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(T_obs, dtype=float)
    e = np.asarray(delta, dtype=int)
    a = np.asarray(A, dtype=float)
    event_mask = e == 1
    if not event_mask.any():
        empty = np.asarray([], dtype=float)
        return empty, empty, empty, empty

    event_times, inverse = np.unique(t[event_mask], return_inverse=True)
    d_total = np.bincount(inverse).astype(float)
    d_treat = np.bincount(inverse, weights=a[event_mask]).astype(float)

    order = np.argsort(t, kind="mergesort")
    sorted_t = t[order]
    sorted_a = a[order]
    starts = np.searchsorted(sorted_t, event_times - 1e-12, side="left")
    cum_treat = np.concatenate(([0.0], np.cumsum(sorted_a)))
    n_risk = float(len(t)) - starts.astype(float)
    n_treat = float(sorted_a.sum()) - cum_treat[starts]
    return n_risk, n_treat, d_total, d_treat


def logrank_p_value(T_obs: np.ndarray, delta: np.ndarray, A: np.ndarray) -> float:
    n_risk, n_treat, d_total, d_treat = _event_time_summaries(T_obs, delta, A)
    valid = (n_risk > 1.0) & (d_total > 0.0)
    if not valid.any():
        return 1.0
    n_risk = n_risk[valid]
    n_treat = n_treat[valid]
    d_total = d_total[valid]
    d_treat = d_treat[valid]
    observed = float(d_treat.sum())
    expected = float((d_total * n_treat / n_risk).sum())
    variance_terms = (
        n_treat
        * (n_risk - n_treat)
        * d_total
        * (n_risk - d_total)
        / (n_risk**2 * np.maximum(n_risk - 1.0, 1.0))
    )
    variance = float(variance_terms.sum())
    if variance <= 1e-12:
        return 1.0
    stat = (observed - expected) ** 2 / variance
    return _chi_square_1df_p(stat)


def cox_log_hr_one_binary_covariate(T_obs: np.ndarray, delta: np.ndarray, A: np.ndarray) -> tuple[float, float]:
    e = np.asarray(delta, dtype=int)
    a = np.asarray(A, dtype=float)
    if e.sum() < 2 or len(np.unique(a)) < 2:
        return 0.0, 1.0
    n_risk, n_treat, d_total, d_treat = _event_time_summaries(T_obs, delta, A)
    valid = (n_risk > 0.0) & (d_total > 0.0)
    n_risk = n_risk[valid]
    n_treat = n_treat[valid]
    d_total = d_total[valid]
    d_treat = d_treat[valid]
    n_control = n_risk - n_treat
    beta = 0.0
    info = 0.0
    for _ in range(50):
        exp_beta = math.exp(float(np.clip(beta, -40.0, 40.0)))
        denom = n_control + exp_beta * n_treat
        good = denom > 0.0
        if not good.any():
            break
        mean = np.zeros_like(denom)
        mean[good] = exp_beta * n_treat[good] / denom[good]
        var = mean * (1.0 - mean)
        score = float((d_treat - d_total * mean).sum())
        info = float((d_total * var).sum())
        if info <= 1e-10:
            break
        step = np.clip(score / info, -1.0, 1.0)
        beta += step
        if abs(step) < 1e-7:
            break
    if not np.isfinite(beta):
        beta = 0.0
    se = math.sqrt(1.0 / max(info, 1e-10))
    p = _normal_two_sided_p(beta / se)
    return float(beta), float(p)


def analyze_trial(T_obs: np.ndarray, delta: np.ndarray, A: np.ndarray, admin_end: float = 1.0) -> dict[str, Any]:
    t = np.asarray(T_obs, dtype=float)
    d = np.asarray(delta, dtype=int)
    a = np.asarray(A, dtype=int)
    log_hr, cox_p = cox_log_hr_one_binary_covariate(t, d, a)
    p_logrank = logrank_p_value(t, d, a)
    hr = float(np.exp(np.clip(log_hr, -20.0, 20.0)))
    administrative_end = float(admin_end)
    censoring_rate = float(np.mean((d == 0) & (t < administrative_end - 1e-8))) if t.size else float("nan")
    return {
        "hr": hr,
        "log_hr": float(log_hr),
        "cox_p_value": float(cox_p),
        "p_value": float(p_logrank),
        "success": bool((p_logrank < 0.05) and (hr < 1.0)),
        "event_rate": float(np.mean(d)) if d.size else float("nan"),
        "censoring_rate": censoring_rate,
    }


def summarize_trial_analyses(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "pos": float("nan"),
            "event_rate": float("nan"),
            "censoring_rate": float("nan"),
            "mean_hr": float("nan"),
            "sd_log_hr": float("nan"),
        }
    log_hr = np.asarray([r["log_hr"] for r in rows], dtype=float)
    return {
        "pos": float(np.mean([bool(r["success"]) for r in rows])),
        "event_rate": float(np.mean([r["event_rate"] for r in rows])),
        "censoring_rate": float(np.mean([r["censoring_rate"] for r in rows])),
        "mean_hr": float(np.mean([r["hr"] for r in rows])),
        "sd_log_hr": float(np.nanstd(log_hr, ddof=1)) if len(log_hr) > 1 else 0.0,
    }
