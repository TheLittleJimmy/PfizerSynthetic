from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .load_pbc import LONGITUDINAL_NAMES, STATIC_BASELINE_NAMES, TREATMENT_NAME

try:
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import logrank_test
except Exception:  # pragma: no cover
    CoxPHFitter = None
    concordance_index = None
    logrank_test = None


BASELINE_COLUMNS = STATIC_BASELINE_NAMES + [f"L0_{name}" for name in LONGITUDINAL_NAMES]
CONTINUOUS_BASELINE = ["age", *[f"L0_{name}" for name in LONGITUDINAL_NAMES if name not in {"ascites", "hepatomegaly", "spiders", "edema", "stage"}]]
CATEGORICAL_BASELINE = ["sex", "treatment", "L0_ascites", "L0_hepatomegaly", "L0_spiders", "L0_edema", "L0_stage"]
NATURAL_JS_DISTANCE_MAX = math.sqrt(math.log(2.0))


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def km_curve(time: pd.Series | np.ndarray, event: pd.Series | np.ndarray, grid: np.ndarray) -> np.ndarray:
    t = pd.to_numeric(pd.Series(time), errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(pd.Series(event), errors="coerce").fillna(0).to_numpy(dtype=int)
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    if t.size == 0:
        return np.ones_like(grid, dtype=float)
    surv = []
    s = 1.0
    event_times = np.sort(np.unique(t[e == 1]))
    idx = 0
    for g in grid:
        while idx < len(event_times) and event_times[idx] <= g:
            et = event_times[idx]
            at_risk = np.sum(t >= et)
            d = np.sum((t == et) & (e == 1))
            if at_risk > 0:
                s *= max(0.0, 1.0 - d / at_risk)
            idx += 1
        surv.append(s)
    return np.asarray(surv, dtype=float)


def rmst(time: pd.Series | np.ndarray, event: pd.Series | np.ndarray, tau: float | None = None) -> float:
    t = pd.to_numeric(pd.Series(time), errors="coerce").dropna().to_numpy(dtype=float)
    if t.size == 0:
        return np.nan
    tau = float(np.nanquantile(t, 0.8)) if tau is None else float(tau)
    grid = np.linspace(0.0, max(tau, 1e-6), 128)
    s = km_curve(time, event, grid)
    return float(np.trapz(s, grid))


def ks_stat(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(stats.ks_2samp(a, b).statistic)


def js_distance(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, bins: int = 20) -> float:
    """Histogram Jensen-Shannon distance using the natural logarithm."""
    a = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(y), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    lo, hi = np.nanmin(np.concatenate([a, b])), np.nanmax(np.concatenate([a, b]))
    if not np.isfinite(lo) or hi <= lo:
        return 0.0
    pa, _ = np.histogram(a, bins=bins, range=(lo, hi), density=False)
    pb, _ = np.histogram(b, bins=bins, range=(lo, hi), density=False)
    pa = pa.astype(float) + 1e-12
    pb = pb.astype(float) + 1e-12
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    return float(math.sqrt(0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))))


def smd(x0: pd.Series | np.ndarray, x1: pd.Series | np.ndarray) -> float:
    a = pd.to_numeric(pd.Series(x0), errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(pd.Series(x1), errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(b) - np.mean(a)) / max(pooled, 1e-8))


def baseline_fidelity(real: pd.DataFrame, synthetic: pd.DataFrame, method: str, replicate: int, setting: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "replicate": int(replicate), "setting": setting}
    mean_abs, sd_abs, smd_abs, js_vals, ks_vals, prev_abs = [], [], [], [], [], []
    for col in BASELINE_COLUMNS + [TREATMENT_NAME]:
        if col not in real or col not in synthetic:
            continue
        if col in CATEGORICAL_BASELINE:
            r = pd.to_numeric(real[col], errors="coerce")
            s = pd.to_numeric(synthetic[col], errors="coerce")
            cats = sorted(set(r.dropna().unique()) | set(s.dropna().unique()))
            for cat in cats:
                diff = abs(float((r == cat).mean()) - float((s == cat).mean()))
                row[f"baseline_{col}_prev_error_{cat}"] = diff
                prev_abs.append(diff)
        else:
            r = pd.to_numeric(real[col], errors="coerce")
            s = pd.to_numeric(synthetic[col], errors="coerce")
            mean_err = safe_float(s.mean() - r.mean())
            sd_err = safe_float(s.std(ddof=1) - r.std(ddof=1))
            row[f"baseline_{col}_mean_error"] = mean_err
            row[f"baseline_{col}_sd_error"] = sd_err
            row[f"baseline_{col}_smd"] = smd(r, s)
            row[f"baseline_{col}_ks"] = ks_stat(r, s)
            row[f"baseline_{col}_js"] = js_distance(r, s)
            mean_abs.append(abs(mean_err) if np.isfinite(mean_err) else np.nan)
            sd_abs.append(abs(sd_err) if np.isfinite(sd_err) else np.nan)
            smd_abs.append(abs(row[f"baseline_{col}_smd"]) if np.isfinite(row[f"baseline_{col}_smd"]) else np.nan)
            ks_vals.append(row[f"baseline_{col}_ks"])
            js_vals.append(row[f"baseline_{col}_js"])
    common = [c for c in CONTINUOUS_BASELINE if c in real and c in synthetic]
    if len(common) >= 2:
        rc = real[common].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0).to_numpy()
        sc = synthetic[common].apply(pd.to_numeric, errors="coerce").corr().fillna(0.0).to_numpy()
        row["baseline_correlation_matrix_error"] = float(np.mean(np.abs(rc - sc)))
    else:
        row["baseline_correlation_matrix_error"] = np.nan
    row.update({
        "baseline_continuous_mean_abs_error": float(np.nanmean(mean_abs)) if mean_abs else np.nan,
        "baseline_continuous_sd_abs_error": float(np.nanmean(sd_abs)) if sd_abs else np.nan,
        "baseline_mean_abs_smd": float(np.nanmean(smd_abs)) if smd_abs else np.nan,
        "baseline_mean_js_distance": float(np.nanmean(js_vals)) if js_vals else np.nan,
        "baseline_mean_ks": float(np.nanmean(ks_vals)) if ks_vals else np.nan,
        "baseline_categorical_prevalence_abs_error": float(np.nanmean(prev_abs)) if prev_abs else np.nan,
    })
    return row


def longitudinal_reference(real_long: pd.DataFrame) -> dict[str, Any]:
    ref: dict[str, Any] = {
        "mean": {},
        "var": {},
        "slope_mean": {},
        "change_mean": {},
        "arms": [],
        "visit_count_mean": float(real_long.groupby("subject_id").size().mean()) if len(real_long) else np.nan,
        "missing_cell_rate": float(real_long[LONGITUDINAL_NAMES].isna().mean().mean()) if len(real_long) else np.nan,
    }
    if real_long.empty:
        return ref
    arms = sorted(pd.to_numeric(real_long[TREATMENT_NAME], errors="coerce").dropna().unique())
    ref["arms"] = arms
    for var in LONGITUDINAL_NAMES:
        if var not in real_long:
            continue
        for arm in arms:
            r = real_long[real_long[TREATMENT_NAME].eq(arm)]
            ref["mean"][(var, arm)] = r.groupby("visit_time")[var].mean(numeric_only=True).dropna()
            ref["var"][(var, arm)] = r.groupby("visit_time")[var].var(numeric_only=True).dropna()
        r_slope = _subject_slope(real_long, var)
        r_change = _subject_change(real_long, var)
        ref["slope_mean"][var] = float(np.nanmean(r_slope)) if len(r_slope) else np.nan
        ref["change_mean"][var] = float(np.nanmean(r_change)) if len(r_change) else np.nan
    return ref


def longitudinal_fidelity(
    real_long: pd.DataFrame,
    synthetic_long: pd.DataFrame,
    method: str,
    replicate: int,
    setting: str = "",
    real_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": method, "replicate": int(replicate), "setting": setting}
    real_reference = real_reference or longitudinal_reference(real_long)
    mean_errors, change_errors, slope_errors, var_errors = [], [], [], []
    for var in LONGITUDINAL_NAMES:
        if var not in real_long or var not in synthetic_long:
            continue
        syn_arms = pd.to_numeric(synthetic_long[TREATMENT_NAME], errors="coerce").dropna().unique() if TREATMENT_NAME in synthetic_long else []
        for arm in sorted(set(real_reference.get("arms", [])) | set(syn_arms)):
            s = synthetic_long[synthetic_long[TREATMENT_NAME].eq(arm)]
            rm = real_reference.get("mean", {}).get((var, arm), pd.Series(dtype=float))
            sm = s.groupby("visit_time")[var].mean(numeric_only=True).dropna()
            common = sorted(set(rm.index) & set(sm.index))
            if common:
                err = float(np.mean(np.abs(sm.loc[common].to_numpy(dtype=float) - rm.loc[common].to_numpy(dtype=float))))
                row[f"longitudinal_{var}_arm{arm}_mean_traj_error"] = err
                mean_errors.append(err)
            else:
                err = _interpolated_curve_error(rm, sm)
                if np.isfinite(err):
                    row[f"longitudinal_{var}_arm{arm}_mean_traj_error"] = err
                    mean_errors.append(err)
            rv = real_reference.get("var", {}).get((var, arm), pd.Series(dtype=float))
            sv = s.groupby("visit_time")[var].var(numeric_only=True).dropna()
            common_v = sorted(set(rv.index) & set(sv.index))
            if common_v:
                verr = float(np.mean(np.abs(sv.loc[common_v].to_numpy(dtype=float) - rv.loc[common_v].to_numpy(dtype=float))))
                var_errors.append(verr)
            else:
                verr = _interpolated_curve_error(rv, sv)
                if np.isfinite(verr):
                    var_errors.append(verr)
        s_slope = _subject_slope(synthetic_long, var)
        r_slope_mean = real_reference.get("slope_mean", {}).get(var, np.nan)
        if np.isfinite(r_slope_mean) and len(s_slope):
            slope_errors.append(abs(float(np.nanmean(s_slope) - r_slope_mean)))
        s_change = _subject_change(synthetic_long, var)
        r_change_mean = real_reference.get("change_mean", {}).get(var, np.nan)
        if np.isfinite(r_change_mean) and len(s_change):
            change_errors.append(abs(float(np.nanmean(s_change) - r_change_mean)))
    row.update({
        "longitudinal_mean_trajectory_error": float(np.nanmean(mean_errors)) if mean_errors else np.nan,
        "longitudinal_change_from_baseline_error": float(np.nanmean(change_errors)) if change_errors else np.nan,
        "longitudinal_slope_distribution_error": float(np.nanmean(slope_errors)) if slope_errors else np.nan,
        "longitudinal_variance_trajectory_error": float(np.nanmean(var_errors)) if var_errors else np.nan,
        "longitudinal_visit_count_real": real_reference.get("visit_count_mean", np.nan),
        "longitudinal_visit_count_synthetic": float(synthetic_long.groupby("subject_id").size().mean()) if len(synthetic_long) else np.nan,
        "longitudinal_missing_cell_rate_real": real_reference.get("missing_cell_rate", np.nan),
        "longitudinal_missing_cell_rate_synthetic": float(synthetic_long[LONGITUDINAL_NAMES].isna().mean().mean()) if len(synthetic_long) else np.nan,
    })
    return row


def _interpolated_curve_error(real_curve: pd.Series, synthetic_curve: pd.Series, points: int = 64) -> float:
    if real_curve is None or synthetic_curve is None or len(real_curve) < 2 or len(synthetic_curve) < 2:
        return np.nan
    r = real_curve.sort_index().dropna()
    s = synthetic_curve.sort_index().dropna()
    if len(r) < 2 or len(s) < 2:
        return np.nan
    rx = pd.to_numeric(pd.Series(r.index), errors="coerce").to_numpy(dtype=float)
    sx = pd.to_numeric(pd.Series(s.index), errors="coerce").to_numpy(dtype=float)
    ry = pd.to_numeric(pd.Series(r.to_numpy()), errors="coerce").to_numpy(dtype=float)
    sy = pd.to_numeric(pd.Series(s.to_numpy()), errors="coerce").to_numpy(dtype=float)
    rok = np.isfinite(rx) & np.isfinite(ry)
    sok = np.isfinite(sx) & np.isfinite(sy)
    rx, ry = rx[rok], ry[rok]
    sx, sy = sx[sok], sy[sok]
    if len(rx) < 2 or len(sx) < 2:
        return np.nan
    lo = max(float(np.min(rx)), float(np.min(sx)))
    hi = min(float(np.max(rx)), float(np.max(sx)))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    grid = np.linspace(lo, hi, int(points))
    return float(np.mean(np.abs(np.interp(grid, sx, sy) - np.interp(grid, rx, ry))))


def _subject_slope(df: pd.DataFrame, var: str) -> np.ndarray:
    if df.empty or var not in df:
        return np.asarray([], dtype=float)
    tmp = df[["subject_id", "visit_time", var]].copy()
    tmp["visit_time"] = pd.to_numeric(tmp["visit_time"], errors="coerce")
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp = tmp.dropna(subset=["subject_id", "visit_time", var])
    if tmp.empty:
        return np.asarray([], dtype=float)
    g = tmp.groupby("subject_id", sort=False)
    n = g[var].count()
    sum_t = g["visit_time"].sum()
    sum_y = g[var].sum()
    tmp["_ty"] = tmp["visit_time"] * tmp[var]
    tmp["_tt"] = tmp["visit_time"] * tmp["visit_time"]
    sum_ty = tmp.groupby("subject_id", sort=False)["_ty"].sum()
    sum_tt = tmp.groupby("subject_id", sort=False)["_tt"].sum()
    denom = n * sum_tt - sum_t * sum_t
    slopes = (n * sum_ty - sum_t * sum_y) / denom.replace(0, np.nan)
    return slopes.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def _subject_change(df: pd.DataFrame, var: str) -> np.ndarray:
    if df.empty or var not in df:
        return np.asarray([], dtype=float)
    tmp = df[["subject_id", "visit_time", var]].copy()
    tmp["visit_time"] = pd.to_numeric(tmp["visit_time"], errors="coerce")
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp = tmp.dropna(subset=["subject_id", "visit_time", var]).sort_values(["subject_id", "visit_time"])
    if tmp.empty:
        return np.asarray([], dtype=float)
    g = tmp.groupby("subject_id", sort=False)[var]
    counts = g.count()
    change = g.last() - g.first()
    change = change[counts >= 2]
    return change.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def survival_reference(real_surv: pd.DataFrame) -> dict[str, Any]:
    grid_max = float(np.nanmax(pd.to_numeric(real_surv["time"], errors="coerce"))) if len(real_surv) else 1.0
    grid = np.linspace(0.0, max(grid_max, 1e-6), 128)
    r_event = pd.to_numeric(real_surv["event"], errors="coerce").fillna(0)
    return {
        "grid": grid,
        "event_mean": float(r_event.mean()) if len(r_event) else np.nan,
        "censor_mean": float(1.0 - r_event.mean()) if len(r_event) else np.nan,
        "km": km_curve(real_surv["time"], r_event, grid),
        "rmst": rmst(real_surv["time"], r_event),
        "median_followup": safe_float(pd.to_numeric(real_surv["time"], errors="coerce").median()),
    }


def survival_fidelity(
    real_surv: pd.DataFrame,
    synthetic_surv: pd.DataFrame,
    method: str,
    replicate: int,
    setting: str = "",
    real_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    real_reference = real_reference or survival_reference(real_surv)
    grid = real_reference["grid"]
    s_event = pd.to_numeric(synthetic_surv["event"], errors="coerce").fillna(0)
    row = {
        "method": method,
        "replicate": int(replicate),
        "setting": setting,
        "survival_event_rate_error": float(s_event.mean() - real_reference["event_mean"]) if len(s_event) and np.isfinite(real_reference["event_mean"]) else np.nan,
        "survival_censoring_rate_error": float((1.0 - s_event.mean()) - real_reference["censor_mean"]) if len(s_event) and np.isfinite(real_reference["censor_mean"]) else np.nan,
        "survival_km_integrated_abs_distance": float(np.mean(np.abs(real_reference["km"] - km_curve(synthetic_surv["time"], s_event, grid)))),
        "survival_rmst_difference": safe_float(rmst(synthetic_surv["time"], s_event) - real_reference["rmst"]),
        "survival_median_followup_error": safe_float(pd.to_numeric(synthetic_surv["time"], errors="coerce").median() - real_reference["median_followup"]),
    }
    return row


def _fast_univariate_cox(time: pd.Series, event: pd.Series, treatment: pd.Series, penalizer: float = 0.05) -> dict[str, Any]:
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(event, errors="coerce").fillna(0).round().clip(0, 1).to_numpy(dtype=int)
    x = pd.to_numeric(treatment, errors="coerce").fillna(0).round().clip(0, 1).to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(x)
    t, e, x = t[ok], e[ok], x[ok]
    if len(t) < 4 or len(np.unique(x)) < 2 or e.sum() < 2:
        return {"cox_status": "not_estimable"}
    beta = 0.0
    event_times = np.sort(np.unique(t[e == 1]))
    for _ in range(30):
        score = -penalizer * beta
        info = penalizer
        for et in event_times:
            event_mask = (t == et) & (e == 1)
            d = int(event_mask.sum())
            if d == 0:
                continue
            risk = t >= et
            xb = np.clip(beta * x[risk], -40.0, 40.0)
            w = np.exp(xb)
            sw = float(w.sum())
            if sw <= 0:
                continue
            wx = float(np.sum(w * x[risk]) / sw)
            wx2 = float(np.sum(w * x[risk] * x[risk]) / sw)
            score += float(np.sum(x[event_mask]) - d * wx)
            info += float(d * max(wx2 - wx * wx, 0.0))
        if info <= 1e-10:
            break
        step = float(np.clip(score / info, -2.0, 2.0))
        beta += step
        if abs(step) < 1e-7:
            break
    se = math.sqrt(1.0 / max(info, 1e-10))
    z = beta / se if se > 0 else np.nan
    return {
        "cox_log_hr": float(beta),
        "cox_hr": float(np.exp(np.clip(beta, -20.0, 20.0))),
        "cox_se": float(se),
        "cox_p": float(2.0 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
        "cox_status": "fast_breslow_univariate",
    }


def _fast_logrank_p(time: pd.Series, event: pd.Series, treatment: pd.Series) -> float:
    t = pd.to_numeric(time, errors="coerce").to_numpy(dtype=float)
    e = pd.to_numeric(event, errors="coerce").fillna(0).round().clip(0, 1).to_numpy(dtype=int)
    x = pd.to_numeric(treatment, errors="coerce").fillna(0).round().clip(0, 1).to_numpy(dtype=int)
    ok = np.isfinite(t)
    t, e, x = t[ok], e[ok], x[ok]
    if len(t) < 4 or len(np.unique(x)) < 2 or e.sum() < 1:
        return np.nan
    observed = 0.0
    expected = 0.0
    variance = 0.0
    for et in np.sort(np.unique(t[e == 1])):
        risk = t >= et
        events = (t == et) & (e == 1)
        n = int(risk.sum())
        d = int(events.sum())
        if n <= 1 or d == 0:
            continue
        n1 = int((risk & (x == 1)).sum())
        d1 = int((events & (x == 1)).sum())
        frac = n1 / n
        observed += d1
        expected += d * frac
        variance += d * frac * (1.0 - frac) * (n - d) / max(n - 1, 1)
    if variance <= 1e-12:
        return np.nan
    z2 = (observed - expected) ** 2 / variance
    return float(stats.chi2.sf(z2, df=1))


def clinical_longitudinal_reference(long_df: pd.DataFrame) -> dict[str, Any]:
    ref: dict[str, Any] = {}
    for var in ["bili", "albumin", "prothrombin"]:
        if var not in long_df:
            continue
        changes = _change_by_subject_with_arm(long_df, var)
        if changes.empty:
            continue
        ref[var] = {
            "changes": changes,
            "threshold": float(changes["baseline"].median()) if "baseline" in changes else np.nan,
        }
    return ref


def clinical_estimands(
    static: pd.DataFrame,
    long_df: pd.DataFrame,
    label: str,
    replicate: int,
    setting: str = "",
    longitudinal_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"method": label, "replicate": int(replicate), "setting": setting}
    df = static.copy()
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0.0).clip(lower=1e-6)
    df["event"] = pd.to_numeric(df["event"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    df[TREATMENT_NAME] = pd.to_numeric(df[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int)
    if df[TREATMENT_NAME].nunique() == 2:
        row.update(_fast_univariate_cox(df["time"], df["event"], df[TREATMENT_NAME]))
        row["logrank_p"] = _fast_logrank_p(df["time"], df["event"], df[TREATMENT_NAME])
    control = df[df[TREATMENT_NAME].eq(0)]
    treated = df[df[TREATMENT_NAME].eq(1)]
    row["rmst_difference_treated_minus_control"] = safe_float(rmst(treated["time"], treated["event"]) - rmst(control["time"], control["event"]))
    for var in ["bili", "albumin", "prothrombin"]:
        if var not in long_df:
            continue
        if longitudinal_reference is not None and var in longitudinal_reference:
            changes = longitudinal_reference[var]["changes"]
            threshold = longitudinal_reference[var]["threshold"]
        else:
            changes = _change_by_subject_with_arm(long_df, var)
            threshold = changes["baseline"].median() if not changes.empty and "baseline" in changes else np.nan
        if not changes.empty and changes[TREATMENT_NAME].nunique() == 2:
            c = changes[changes[TREATMENT_NAME].eq(0)]["change"].dropna()
            t = changes[changes[TREATMENT_NAME].eq(1)]["change"].dropna()
            row[f"mmrm_proxy_{var}_treatment_effect"] = safe_float(t.mean() - c.mean())
            if len(c) > 1 and len(t) > 1:
                row[f"mmrm_proxy_{var}_p"] = safe_float(stats.ttest_ind(t, c, equal_var=False, nan_policy="omit").pvalue)
            row[f"responder_rate_diff_{var}"] = safe_float((t < threshold).mean() - (c < threshold).mean()) if np.isfinite(threshold) else np.nan
    return row


def _change_by_subject_with_arm(long_df: pd.DataFrame, var: str) -> pd.DataFrame:
    if long_df.empty or var not in long_df:
        return pd.DataFrame(columns=["subject_id", TREATMENT_NAME, "baseline", "change"])
    tmp = long_df[["subject_id", "visit_time", TREATMENT_NAME, var]].copy()
    tmp["visit_time"] = pd.to_numeric(tmp["visit_time"], errors="coerce")
    tmp[var] = pd.to_numeric(tmp[var], errors="coerce")
    tmp[TREATMENT_NAME] = pd.to_numeric(tmp[TREATMENT_NAME], errors="coerce")
    tmp = tmp.dropna(subset=["subject_id", "visit_time", var]).sort_values(["subject_id", "visit_time"])
    if tmp.empty:
        return pd.DataFrame(columns=["subject_id", TREATMENT_NAME, "baseline", "change"])
    g = tmp.groupby("subject_id", sort=False)
    out = pd.DataFrame({
        "subject_id": g["subject_id"].first().astype(int),
        TREATMENT_NAME: g[TREATMENT_NAME].first().fillna(0).round().astype(int),
        "baseline": g[var].first().astype(float),
        "change": (g[var].last() - g[var].first()).astype(float),
    }).reset_index(drop=True)
    return out


def privacy_metrics(real_subjects: pd.DataFrame, synthetic_subjects: pd.DataFrame, method: str, replicate: int, setting: str = "", fast: bool = False) -> dict[str, Any]:
    cols = [c for c in BASELINE_COLUMNS if c in real_subjects and c in synthetic_subjects]
    if fast:
        cols = [c for c in ["age", "sex", TREATMENT_NAME, "L0_bili", "L0_albumin", "L0_prothrombin"] if c in real_subjects and c in synthetic_subjects]
    if not cols or synthetic_subjects.empty or real_subjects.empty:
        return {"method": method, "replicate": int(replicate), "setting": setting, "privacy_status": "missing_baseline_columns"}
    combined = pd.concat([real_subjects[cols], synthetic_subjects[cols]], ignore_index=True)
    fills = combined.apply(pd.to_numeric, errors="coerce").median(numeric_only=True).fillna(0.0)
    r = real_subjects[cols].apply(pd.to_numeric, errors="coerce").fillna(fills)
    s = synthetic_subjects[cols].apply(pd.to_numeric, errors="coerce").fillna(fills)
    scaler = StandardScaler().fit(r)
    rx = scaler.transform(r)
    sx = scaler.transform(s)
    d = np.linalg.norm(sx[:, None, :] - rx[None, :, :], axis=2)
    closest = np.min(d, axis=1)
    sorted_d = np.sort(d, axis=1)
    second = sorted_d[:, 1] if sorted_d.shape[1] > 1 else np.full_like(closest, np.nan)
    exact = float(np.mean(closest < 1e-10))
    auc = np.nan
    if not fast:
        try:
            x = np.vstack([rx, sx])
            y = np.concatenate([np.zeros(len(rx)), np.ones(len(sx))])
            if len(np.unique(y)) == 2 and len(x) > 10:
                xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.35, random_state=17, stratify=y)
                clf = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear", random_state=17)
                clf.fit(xtr, ytr)
                auc = float(roc_auc_score(yte, clf.predict_proba(xte)[:, 1]))
        except Exception:
            auc = np.nan
    qi = [c for c in ["age", "sex", TREATMENT_NAME] if c in real_subjects and c in synthetic_subjects]
    kmap = np.nan
    if qi:
        keys = real_subjects[qi].round(2).astype(str).agg("|".join, axis=1).value_counts()
        syn_keys = synthetic_subjects[qi].round(2).astype(str).agg("|".join, axis=1)
        kmap = float(syn_keys.map(keys).fillna(0).mean())
    return {
        "method": method,
        "replicate": int(replicate),
        "setting": setting,
        "privacy_nearest_neighbor_distance_ratio": float(np.nanmean(closest / np.maximum(second, 1e-8))),
        "privacy_distance_to_closest_real_record": float(np.nanmean(closest)),
        "privacy_exact_duplicate_rate": exact,
        "privacy_kmap_mean_equivalence_count": kmap,
        "privacy_detection_classifier_auc": auc,
        "privacy_detection_classifier_status": "not_run_fast_replicate_loop" if fast else "completed_or_not_estimable",
    }


def digital_twin_metrics(real_long: pd.DataFrame, pred_long: pd.DataFrame, real_static: pd.DataFrame, pred_static: pd.DataFrame, method: str = "PhaseSyn") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows = []
    cover_rows = []
    for var in LONGITUDINAL_NAMES:
        merged = real_long[["subject_id", "visit_time", var]].merge(
            pred_long[["subject_id", "visit_time", var, f"{var}_lo50", f"{var}_hi50", f"{var}_lo80", f"{var}_hi80", f"{var}_lo95", f"{var}_hi95"]],
            on=["subject_id", "visit_time"],
            suffixes=("_real", "_pred"),
            how="inner",
        ) if f"{var}_lo50" in pred_long else pd.DataFrame()
        if merged.empty:
            continue
        y = pd.to_numeric(merged[f"{var}_real"], errors="coerce")
        p = pd.to_numeric(merged[f"{var}_pred"], errors="coerce")
        ok = y.notna() & p.notna()
        if ok.any():
            long_rows.append({
                "method": method,
                "variable": var,
                "rmse": float(np.sqrt(np.mean((p[ok] - y[ok]) ** 2))),
                "mae": float(np.mean(np.abs(p[ok] - y[ok]))),
                "crps_proxy_mae": float(np.mean(np.abs(p[ok] - y[ok]))),
            })
        for level in [50, 80, 95]:
            lo = pd.to_numeric(merged[f"{var}_lo{level}"], errors="coerce")
            hi = pd.to_numeric(merged[f"{var}_hi{level}"], errors="coerce")
            valid = ok & lo.notna() & hi.notna()
            cover_rows.append({
                "method": method,
                "variable": var,
                "interval": level,
                "coverage": float(((y[valid] >= lo[valid]) & (y[valid] <= hi[valid])).mean()) if valid.any() else np.nan,
            })
    surv_rows = []
    if not pred_static.empty and not real_static.empty:
        merged_s = real_static[["subject_id", "time", "event"]].merge(
            pred_static[["subject_id", "time", "event", "survival_risk_score"]].drop_duplicates("subject_id"),
            on="subject_id",
            suffixes=("_real", "_pred"),
            how="inner",
        ) if "survival_risk_score" in pred_static else pd.DataFrame()
        if not merged_s.empty:
            risk = pd.to_numeric(merged_s["survival_risk_score"], errors="coerce").fillna(0.0)
            evt = pd.to_numeric(merged_s["event_real"], errors="coerce")
            time = pd.to_numeric(merged_s["time_real"], errors="coerce").fillna(pd.to_numeric(merged_s["time_real"], errors="coerce").median())
            brier_rows = {}
            for horizon in [2.0, 5.0, 10.0]:
                observed_by_h = ((time <= horizon) & (evt > 0.5)).astype(float)
                brier_rows[f"brier_score_{horizon:g}y"] = float(np.mean((risk - observed_by_h) ** 2))
            c_index = np.nan
            if concordance_index is not None and len(merged_s) > 2:
                try:
                    c_index = float(concordance_index(time, -risk, evt))
                except Exception:
                    c_index = np.nan
            surv_rows.append({
                "method": method,
                "c_index": c_index,
                "integrated_brier_score_proxy": float(np.nanmean(list(brier_rows.values()))) if brier_rows else np.nan,
                "time_dependent_auc_status": "not_implemented_without_sksurv",
                "survival_calibration_event_rate_error": float(pd.to_numeric(merged_s["event_pred"], errors="coerce").mean() - evt.mean()),
                "survival_risk_event_correlation": float(np.corrcoef(risk.fillna(risk.mean()), evt.fillna(0))[0, 1]) if len(merged_s) > 2 else np.nan,
                **brier_rows,
            })
    landmark = pd.DataFrame([{
        "method": method,
        "landmark_status": "computed_proxy",
        "early_response_definition": "first-to-second visit change summaries; responder-stratified KM figure generated when data permit",
    }])
    return pd.DataFrame(long_rows), pd.DataFrame(surv_rows), pd.DataFrame(cover_rows), landmark


def summarize_status(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=by)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    return df.groupby(by, dropna=False)[numeric].mean().reset_index()
