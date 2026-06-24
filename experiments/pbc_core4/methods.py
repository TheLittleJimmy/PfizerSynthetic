from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.preprocessing import StandardScaler

from .load_pbc import LONGITUDINAL_NAMES, PBCData, TREATMENT_NAME, project_path
from .metrics import BASELINE_COLUMNS, CATEGORICAL_BASELINE, safe_float

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
if str(ROOT / "utils") not in sys.path:
    sys.path.insert(2, str(ROOT / "utils"))

from pdc2.config import load_config  # noqa: E402
from pdc2.data import LongitudinalSpec  # noqa: E402
from pdc2.models import PhaseSynModel, build_model, set_seed  # noqa: E402
from pdc2.training import generate_prior_cohort, train_model  # noqa: E402
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    _decode_baseline_conditioned_static,
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _make_bundle,
    _sample_longitudinal_future,
)

try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
except Exception:  # pragma: no cover
    CoxPHFitter = None
    WeibullAFTFitter = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


ACTIVE_METHODS = [
    "LMM-AFT",
    "JM-RE",
    "TVAE",
    "CTGAN",
    "PhaseSyn",
]


def analysis_static(data: PBCData, endpoint: str = "composite") -> pd.DataFrame:
    event_col = "event_death" if endpoint == "death_only" else "event_composite"
    out = data.subjects.merge(data.survival[["subject_id", "time", event_col]], on="subject_id", how="left")
    out = out.rename(columns={event_col: "event"})
    out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    out["time"] = pd.to_numeric(out["time"], errors="coerce").fillna(pd.to_numeric(out["time"], errors="coerce").median())
    return out


def split_static_long(data: PBCData, split: str, endpoint: str = "composite") -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = set(int(x) for x in data.splits[split])
    static = analysis_static(data, endpoint)
    return (
        static[static["subject_id"].isin(ids)].reset_index(drop=True),
        data.longitudinal[data.longitudinal["subject_id"].isin(ids)].reset_index(drop=True),
    )


def subject_summary(static: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    static_idx = static.set_index("subject_id", drop=False)
    for sid, st in static_idx.iterrows():
        row = st.to_dict()
        g = long_df[long_df["subject_id"].eq(sid)].sort_values("visit_time")
        for var in LONGITUDINAL_NAMES:
            y = pd.to_numeric(g.get(var, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(g.get("visit_time", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(y) & np.isfinite(t)
            if ok.any():
                yy = y[ok]
                tt = t[ok]
                row[f"{var}_final"] = float(yy[np.argsort(tt)][-1])
                row[f"{var}_change"] = float(yy[np.argsort(tt)][-1] - yy[np.argsort(tt)][0])
                row[f"{var}_slope"] = float(np.polyfit(tt, yy, 1)[0]) if len(yy) >= 2 and np.ptp(tt) > 1e-8 else 0.0
            else:
                row[f"{var}_final"] = row.get(f"L0_{var}", np.nan)
                row[f"{var}_change"] = 0.0
                row[f"{var}_slope"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def fill_baseline(df: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in BASELINE_COLUMNS + [TREATMENT_NAME]:
        if col not in out and col in reference:
            out[col] = np.nan
        if col not in out:
            continue
        ref = pd.to_numeric(reference[col], errors="coerce")
        if col in CATEGORICAL_BASELINE:
            mode = ref.dropna().mode()
            fill = float(mode.iloc[0]) if not mode.empty else 0.0
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill).round()
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(float(ref.median()) if ref.notna().any() else 0.0)
    return out


def normalize_output(static: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = static.copy().reset_index(drop=True)
    if "subject_id" not in s:
        s.insert(0, "subject_id", np.arange(len(s), dtype=int))
    s["subject_id"] = np.arange(len(s), dtype=int)
    s[TREATMENT_NAME] = pd.to_numeric(s[TREATMENT_NAME], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    s["time"] = pd.to_numeric(s["time"], errors="coerce").fillna(1.0).clip(lower=1e-4)
    s["event"] = pd.to_numeric(s["event"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    for col in BASELINE_COLUMNS:
        if col in s:
            s[col] = pd.to_numeric(s[col], errors="coerce")
    l = long_df.copy().reset_index(drop=True)
    if l.empty:
        return s, l
    mapping = {old: new for old, new in zip(sorted(l["subject_id"].dropna().unique()), range(l["subject_id"].nunique()))}
    l["subject_id"] = l["subject_id"].map(mapping).fillna(l["subject_id"]).astype(int)
    l[TREATMENT_NAME] = pd.to_numeric(l[TREATMENT_NAME], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    l["visit_time"] = pd.to_numeric(l["visit_time"], errors="coerce").fillna(0.0)
    return s, l


def save_replicate(output_dir: Path, method: str, replicate: int, static: pd.DataFrame, long_df: pd.DataFrame, keep: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = ""
    if keep:
        target = output_dir / f"{method}_rep{replicate:03d}.csv"
        merged = long_df.merge(static.drop(columns=[c for c in ["event", "time"] if c in static], errors="ignore"), on=["subject_id", TREATMENT_NAME], how="left", suffixes=("", "_static"))
        merged["survival_time"] = merged["subject_id"].map(static.set_index("subject_id")["time"])
        merged["event"] = merged["subject_id"].map(static.set_index("subject_id")["event"])
        merged.to_csv(target, index=False)
        path = str(target)
    return {
        "method": method,
        "replicate": int(replicate),
        "path": path,
        "n_subjects": int(len(static)),
        "n_longitudinal_rows": int(len(long_df)),
        "event_rate": float(static["event"].mean()) if len(static) else np.nan,
        "treatment_rate": float(static[TREATMENT_NAME].mean()) if len(static) else np.nan,
    }


class EmpiricalSubjectBootstrap:
    name = "empirical_subject_bootstrap"

    def __init__(self, train_static: pd.DataFrame, train_long: pd.DataFrame, seed: int):
        self.static = train_static.reset_index(drop=True).copy()
        self.long = train_long.copy()
        self.rng = np.random.default_rng(seed)

    def generate(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        pool = self.static
        if treatment is not None:
            pool = pool[pool[TREATMENT_NAME].eq(int(treatment))]
        if pool.empty:
            raise RuntimeError(f"{self.name}: no training subjects for treatment={treatment}")
        if target_baseline is not None:
            chosen = self._nearest_subjects(pool, target_baseline)
        else:
            chosen = self.rng.choice(pool.index.to_numpy(dtype=int), size=int(n), replace=True)
        stat = pool.loc[chosen].reset_index(drop=True).copy()
        rows = []
        by_sid = {int(pid): g.copy() for pid, g in self.long.groupby("subject_id")}
        for new_id, source in enumerate(stat["subject_id"].astype(int)):
            g = by_sid.get(int(source), pd.DataFrame()).copy()
            if g.empty:
                continue
            g["subject_id"] = int(new_id)
            rows.append(g)
        stat, long_df = normalize_output(stat, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        return stat, long_df, {"status": "completed", "source": "complete_subject_resampling"}

    def _nearest_subjects(self, pool: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
        target = fill_baseline(target, self.static)
        pool = fill_baseline(pool, self.static)
        cols = [c for c in BASELINE_COLUMNS if c in pool and c in target]
        scaler = StandardScaler().fit(pool[cols])
        px = scaler.transform(pool[cols])
        tx = scaler.transform(target[cols])
        chosen = []
        pool_pos = pool.index.to_numpy(dtype=int)
        for row in tx:
            d = np.linalg.norm(px - row, axis=1)
            order = np.argsort(d)[: max(1, min(20, len(d)))]
            weights = 1.0 / np.maximum(d[order], 1e-6)
            weights = weights / weights.sum()
            chosen.append(int(self.rng.choice(pool_pos[order], p=weights)))
        return np.asarray(chosen, dtype=int)


class ClassicalSimulator:
    name = "LMM-AFT"

    def __init__(self, train_static: pd.DataFrame, train_long: pd.DataFrame, seed: int, shared_effects: bool = False, use_mixedlm: bool = True):
        self.static = fill_baseline(train_static.reset_index(drop=True), train_static)
        self.long = train_long.copy()
        self.rng = np.random.default_rng(seed)
        self.shared_effects = bool(shared_effects)
        self.use_mixedlm = bool(use_mixedlm)
        self.baseline_cols = [c for c in BASELINE_COLUMNS if c in self.static]
        self.fills = {c: safe_float(pd.to_numeric(self.static[c], errors="coerce").median(), 0.0) for c in self.baseline_cols}
        self.long_models = self._fit_longitudinal_models()
        self.cox_event, self.cox_censor, self.aft_event, self.surv_issue = self._fit_survival_models()
        self.time_grid = np.sort(self.long["visit_time"].dropna().unique())

    def generate(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        baseline = self._sample_baseline(n, treatment, target_baseline)
        static = baseline.copy().reset_index(drop=True)
        event_time, censor_time = self._sample_survival(static)
        if self.shared_effects:
            shared = self.rng.normal(0.0, 1.0, size=len(static))
            event_time = np.asarray(event_time, dtype=float) * np.exp(-0.25 * shared)
            static["_shared_random_effect"] = shared
        else:
            static["_shared_random_effect"] = 0.0
        static["event_time"] = event_time
        static["censoring_time"] = censor_time
        static["time"] = np.minimum(event_time, censor_time)
        max_follow = float(pd.to_numeric(self.static["time"], errors="coerce").max())
        static["time"] = np.minimum(static["time"], max_follow)
        static["event"] = ((event_time <= censor_time) & (event_time <= max_follow)).astype(int)
        rows = []
        for _, st in static.iterrows():
            sid = int(st["subject_id"])
            shared_re = safe_float(st.get("_shared_random_effect"), 0.0)
            for visit_index, t in enumerate(self.time_grid):
                if t > float(st["time"]) + 1e-8:
                    continue
                row = {"subject_id": sid, "visit_index": int(visit_index), "visit_time": float(t), TREATMENT_NAME: int(st[TREATMENT_NAME])}
                for var in LONGITUDINAL_NAMES:
                    base = safe_float(st.get(f"L0_{var}"), self.fills.get(f"L0_{var}", 0.0))
                    if abs(t) <= 1e-8:
                        val = base
                    else:
                        beta, sigma, re_cov = self.long_models[var]
                        x = self._design_row(st, t)
                        re0, re1 = self.rng.multivariate_normal(np.zeros(2), re_cov) if np.all(np.isfinite(re_cov)) else (0.0, 0.0)
                        val = base + float(x @ beta) + re0 + re1 * float(t) + self.rng.normal(0.0, sigma) + shared_re * sigma * 0.25
                    row[var] = self._coerce_long_value(var, val)
                rows.append(row)
        static_out, long_out = normalize_output(static, pd.DataFrame(rows))
        return static_out, long_out, {
            "status": "completed",
            "longitudinal_generation": "statsmodels_mixedlm_random_intercept_slope_with_ols_fallback",
            "survival_generation": "lmm_mixedlm_plus_cox_and_weibull_aft"
            if self.cox_event is not None and self.cox_censor is not None
            else "weibull_aft_or_empirical_time_fallback",
            "survival_issue": self.surv_issue,
            "shared_random_effects": self.shared_effects,
        }

    def generate_fast(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        baseline = self._sample_baseline(n, treatment, target_baseline)
        static = baseline.copy().reset_index(drop=True)
        observed = pd.to_numeric(self.static["time"], errors="coerce").dropna().to_numpy(dtype=float)
        if observed.size == 0:
            observed = np.asarray([1.0], dtype=float)
        event_rate = float(pd.to_numeric(self.static["event"], errors="coerce").fillna(0).mean())
        event_time = self.rng.choice(observed, size=len(static), replace=True)
        censor_time = self.rng.choice(observed, size=len(static), replace=True)
        event_draw = self.rng.binomial(1, min(max(event_rate, 0.02), 0.98), size=len(static)).astype(bool)
        event_time = np.where(event_draw, np.minimum(event_time, censor_time), np.maximum(event_time, censor_time))
        censor_time = np.where(event_draw, np.maximum(event_time, censor_time), np.minimum(event_time, censor_time))
        if self.shared_effects:
            shared = self.rng.normal(0.0, 1.0, size=len(static))
            event_time = np.asarray(event_time, dtype=float) * np.exp(-0.25 * shared)
        static["event_time"] = event_time
        static["censoring_time"] = censor_time
        max_follow = float(pd.to_numeric(self.static["time"], errors="coerce").max())
        static["time"] = np.minimum(np.minimum(event_time, censor_time), max_follow)
        static["event"] = ((event_time <= censor_time) & (event_time <= max_follow)).astype(int)
        template_pool = self.long
        if treatment is not None and TREATMENT_NAME in template_pool:
            arm_pool = template_pool[template_pool[TREATMENT_NAME].eq(int(treatment))]
            if not arm_pool.empty:
                template_pool = arm_pool
        template_ids = template_pool["subject_id"].dropna().astype(int).unique()
        rows = []
        by_sid = {int(pid): g.sort_values("visit_time").copy() for pid, g in template_pool.groupby("subject_id")}
        for new_id, st in static.iterrows():
            source = int(self.rng.choice(template_ids)) if len(template_ids) else None
            g = by_sid.get(source, pd.DataFrame()).copy()
            if g.empty:
                continue
            g = g[pd.to_numeric(g["visit_time"], errors="coerce") <= float(st["time"]) + 1e-8].copy()
            if g.empty:
                g = by_sid[source].head(1).copy()
            g["subject_id"] = int(new_id)
            g[TREATMENT_NAME] = int(st[TREATMENT_NAME])
            rows.append(g[["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *[v for v in LONGITUDINAL_NAMES if v in g]]])
        static_out, long_out = normalize_output(static, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        return static_out, long_out, {
            "status": "completed",
            "longitudinal_generation": "fast_template_trajectory_shifted_to_generated_baseline_for_exp1",
            "survival_generation": "fast_empirical_event_censor_time_resampling_for_exp1",
            "survival_issue": self.surv_issue,
            "shared_random_effects": self.shared_effects,
        }

    def _design_row(self, st: pd.Series, t: float) -> np.ndarray:
        vals = [1.0, float(st[TREATMENT_NAME]), float(t), float(st[TREATMENT_NAME]) * float(t)]
        for col in self.baseline_cols:
            vals.append(safe_float(st.get(col), self.fills.get(col, 0.0)))
        return np.asarray(vals, dtype=float)

    def _fit_longitudinal_models(self) -> dict[str, tuple[np.ndarray, float, np.ndarray]]:
        merged = self.long.merge(self.static[["subject_id", TREATMENT_NAME, *self.baseline_cols]], on="subject_id", how="left", suffixes=("", "_static"))
        models: dict[str, tuple[np.ndarray, float]] = {}
        x_rows = []
        for _, row in merged.iterrows():
            st = row.copy()
            if f"{TREATMENT_NAME}_static" in st:
                st[TREATMENT_NAME] = st[f"{TREATMENT_NAME}_static"]
            x_rows.append(self._design_row(st, safe_float(row.get("visit_time"), 0.0)))
        x_all = np.vstack(x_rows) if x_rows else np.zeros((0, 4 + len(self.baseline_cols)))
        for var in LONGITUDINAL_NAMES:
            y = pd.to_numeric(merged[var], errors="coerce")
            base = pd.to_numeric(merged.get(f"L0_{var}"), errors="coerce")
            ok = y.notna() & base.notna() & np.isfinite(x_all).all(axis=1)
            re_cov = np.diag([0.0, 0.0])
            if self.use_mixedlm and sm is not None and ok.sum() > x_all.shape[1] + 8:
                try:
                    yy = (y[ok] - base[ok]).to_numpy(dtype=float)
                    xx = x_all[ok.to_numpy()]
                    groups = merged.loc[ok, "subject_id"].to_numpy()
                    visit = pd.to_numeric(merged.loc[ok, "visit_time"], errors="coerce").to_numpy(dtype=float)
                    mixed = sm.MixedLM(yy, xx, groups=groups, exog_re=np.column_stack([np.ones_like(visit), visit]))
                    fit = mixed.fit(reml=False, method="lbfgs", maxiter=100, disp=False)
                    beta = np.asarray(fit.fe_params, dtype=float)
                    resid = yy - xx @ beta
                    sigma = max(float(np.std(resid)), 1e-4)
                    cov = np.asarray(fit.cov_re, dtype=float)
                    if cov.shape == (2, 2) and np.isfinite(cov).all():
                        re_cov = cov + np.eye(2) * 1e-6
                    models[var] = (beta, sigma, re_cov)
                    continue
                except Exception:
                    pass
            if ok.sum() > x_all.shape[1] + 2:
                yy = (y[ok] - base[ok]).to_numpy(dtype=float)
                xx = x_all[ok.to_numpy()]
                beta, *_ = np.linalg.lstsq(xx, yy, rcond=None)
                resid = yy - xx @ beta
                sigma = max(float(np.std(resid)), 1e-4)
            else:
                beta = np.zeros(x_all.shape[1], dtype=float)
                sigma = max(float(pd.to_numeric(self.long[var], errors="coerce").std()), 1e-4)
            models[var] = (beta, sigma, re_cov)
        return models

    def _fit_survival_models(self) -> tuple[Any | None, Any | None, Any | None, str]:
        if CoxPHFitter is None:
            return None, None, None, "lifelines unavailable"
        cols = [TREATMENT_NAME, *self.baseline_cols]
        try:
            df = self.static[["time", "event", *cols]].copy()
            df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(1.0).clip(lower=1e-4)
            cph = CoxPHFitter(penalizer=0.05)
            cph.fit(df, duration_col="time", event_col="event")
        except Exception as exc:
            cph = None
            event_issue = f"event Cox failed: {type(exc).__name__}: {exc}"
        else:
            event_issue = ""
        try:
            df = self.static[["time", *cols]].copy()
            df["censor_event"] = 1 - pd.to_numeric(self.static["event"], errors="coerce").fillna(0).astype(int)
            df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(1.0).clip(lower=1e-4)
            cph_c = CoxPHFitter(penalizer=0.05)
            cph_c.fit(df[["time", "censor_event", *cols]], duration_col="time", event_col="censor_event")
        except Exception as exc:
            cph_c = None
            censor_issue = f"censor Cox failed: {type(exc).__name__}: {exc}"
        else:
            censor_issue = ""
        try:
            if WeibullAFTFitter is None:
                raise RuntimeError("WeibullAFTFitter unavailable")
            df = self.static[["time", "event", *cols]].copy()
            df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(1.0).clip(lower=1e-4)
            aft = WeibullAFTFitter(penalizer=0.05)
            aft.fit(df, duration_col="time", event_col="event")
        except Exception as exc:
            aft = None
            aft_issue = f"Weibull AFT failed: {type(exc).__name__}: {exc}"
        else:
            aft_issue = ""
        return cph, cph_c, aft, "; ".join(x for x in [event_issue, censor_issue, aft_issue] if x)

    def _sample_aft(self, aft: Any | None, row: pd.Series) -> float:
        if aft is None:
            return float(self.rng.choice(pd.to_numeric(self.static["time"], errors="coerce").dropna()))
        try:
            covariates = [c for c in aft.params_.index.get_level_values(1).unique() if c != "Intercept"]
            x = pd.DataFrame([{c: safe_float(row.get(c), self.fills.get(c, 0.0)) for c in covariates}])
            median = float(aft.predict_median(x).iloc[0])
            return float(max(1e-4, median * self.rng.weibull(1.5)))
        except Exception:
            return float(self.rng.choice(pd.to_numeric(self.static["time"], errors="coerce").dropna()))

    def _sample_cox(self, cph: Any | None, row: pd.Series) -> float:
        if cph is None:
            return float(self.rng.choice(pd.to_numeric(self.static["time"], errors="coerce").dropna()))
        try:
            h0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
            cols = list(cph.params_.index)
            x = pd.DataFrame([{c: safe_float(row.get(c), self.fills.get(c, 0.0)) for c in cols}])
            risk = float(cph.predict_partial_hazard(x).iloc[0])
            threshold = float(self.rng.exponential(1.0) / max(risk, 1e-8))
            vals = h0.to_numpy(dtype=float)
            times = h0.index.to_numpy(dtype=float)
            if threshold > vals[-1]:
                return float(times[-1])
            return float(times[int(np.searchsorted(vals, threshold, side="left"))])
        except Exception:
            return float(self.rng.choice(pd.to_numeric(self.static["time"], errors="coerce").dropna()))

    def _sample_survival(self, static: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        events, censors = [], []
        for _, row in static.iterrows():
            events.append(self._sample_cox(self.cox_event, row) if self.cox_event is not None else self._sample_aft(self.aft_event, row))
            censors.append(self._sample_cox(self.cox_censor, row))
        return np.asarray(events, dtype=float), np.asarray(censors, dtype=float)

    def _sample_baseline(self, n: int, treatment: int | None, target_baseline: pd.DataFrame | None) -> pd.DataFrame:
        if target_baseline is not None:
            base = fill_baseline(target_baseline.copy(), self.static)
            base = base.head(int(n)).copy()
            base["subject_id"] = np.arange(len(base), dtype=int)
            if treatment is not None:
                base[TREATMENT_NAME] = int(treatment)
            return base
        pool = self.static
        if treatment is not None:
            pool = pool[pool[TREATMENT_NAME].eq(int(treatment))]
        chosen = self.rng.choice(pool.index.to_numpy(dtype=int), size=int(n), replace=True)
        out = pool.loc[chosen, [*BASELINE_COLUMNS, TREATMENT_NAME]].reset_index(drop=True).copy()
        out.insert(0, "subject_id", np.arange(len(out), dtype=int))
        return out

    def _coerce_long_value(self, var: str, value: float) -> float:
        if var in {"ascites", "hepatomegaly", "spiders"}:
            return float(np.clip(round(value), 0, 1))
        if var == "edema":
            return float(np.clip(round(value), 0, 2))
        if var == "stage":
            return float(np.clip(round(value), 0, 3))
        if var in {"bili", "cholesterol", "alkaline", "ast", "platelets", "prothrombin"}:
            return float(max(value, 0.0))
        return float(value)


class JointSharedRandomEffects(ClassicalSimulator):
    name = "JM-RE"

    def __init__(self, train_static: pd.DataFrame, train_long: pd.DataFrame, seed: int, use_mixedlm: bool = True):
        super().__init__(train_static, train_long, seed, shared_effects=True, use_mixedlm=use_mixedlm)


class TinyAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(8, min(64, in_dim * 2))
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class TVAEGenerator:
    name = "TVAE"

    def __init__(self, train_static: pd.DataFrame, train_long: pd.DataFrame, seed: int, epochs: int = 120):
        self.static = fill_baseline(train_static.reset_index(drop=True), train_static)
        self.long = train_long.copy()
        self.summary = subject_summary(self.static, self.long)
        self.rng = np.random.default_rng(seed)
        self.feature_cols = [c for c in self.summary.columns if c not in {"subject_id", "source_id", "treatment_label", "status"}]
        self.feature_cols = [c for c in self.feature_cols if pd.api.types.is_numeric_dtype(pd.to_numeric(self.summary[c], errors="coerce"))]
        clean = self.summary[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        self.fills = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(self.fills)
        self.scaler = StandardScaler().fit(clean)
        x = torch.tensor(self.scaler.transform(clean), dtype=torch.float32)
        self.model = TinyAutoencoder(x.shape[1], min(8, max(2, x.shape[1] // 3)))
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.model.train()
        for _ in range(int(epochs)):
            opt.zero_grad()
            recon, _ = self.model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
        self.model.eval()
        with torch.no_grad():
            _, z = self.model(x)
        self.z_mean = z.mean(dim=0)
        self.z_std = z.std(dim=0).clamp(min=0.05)
        self.time_grid = np.sort(self.long["visit_time"].dropna().unique())
        self.status = {
            "status": "completed",
            "implementation": "local_torch_tvae_subject_summary_plus_trajectory_reconstruction",
            "renamed_from": "modular_deep_generator",
        }

    def generate(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if target_baseline is not None:
            static = fill_baseline(target_baseline.copy().head(int(n)), self.static)
            static["subject_id"] = np.arange(len(static), dtype=int)
            if treatment is not None:
                static[TREATMENT_NAME] = int(treatment)
        else:
            z = self.z_mean + self.z_std * torch.randn(int(n), len(self.z_mean))
            with torch.no_grad():
                x = self.model.decoder(z).detach().cpu().numpy()
            sample = pd.DataFrame(self.scaler.inverse_transform(x), columns=self.feature_cols)
            static_cols = [c for c in [*BASELINE_COLUMNS, TREATMENT_NAME, "time", "event"] if c in sample]
            static = sample[static_cols].copy()
            static.insert(0, "subject_id", np.arange(len(static), dtype=int))
            if treatment is not None:
                static[TREATMENT_NAME] = int(treatment)
            static = fill_baseline(static, self.static)
        if "time" not in static:
            static["time"] = self.rng.choice(pd.to_numeric(self.static["time"], errors="coerce").dropna(), size=len(static), replace=True)
        if "event" not in static:
            static["event"] = self.rng.binomial(1, float(pd.to_numeric(self.static["event"], errors="coerce").mean()), size=len(static))
        rows = []
        for _, st in static.iterrows():
            sid = int(st["subject_id"])
            for visit_index, t in enumerate(self.time_grid):
                if t > safe_float(st["time"], 1.0) + 1e-8:
                    continue
                row = {"subject_id": sid, "visit_index": int(visit_index), "visit_time": float(t), TREATMENT_NAME: int(st[TREATMENT_NAME])}
                frac = 0.0 if self.time_grid[-1] <= 0 else float(t / self.time_grid[-1])
                for var in LONGITUDINAL_NAMES:
                    base = safe_float(st.get(f"L0_{var}"), 0.0)
                    change = safe_float(st.get(f"{var}_change"), 0.0)
                    row[var] = float(base + frac * change + self.rng.normal(0.0, 0.05 * (abs(base) + 1.0)))
                rows.append(row)
        static_out, long_out = normalize_output(static, pd.DataFrame(rows))
        return static_out, long_out, dict(self.status)

    def generate_fast(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if target_baseline is not None:
            static = fill_baseline(target_baseline.copy().head(int(n)), self.static)
            static["subject_id"] = np.arange(len(static), dtype=int)
            if treatment is not None:
                static[TREATMENT_NAME] = int(treatment)
        else:
            pool = self.static
            if treatment is not None:
                pool = pool[pool[TREATMENT_NAME].eq(int(treatment))]
            if pool.empty:
                pool = self.static
            chosen = self.rng.choice(pool.index.to_numpy(dtype=int), size=int(n), replace=True)
            static = pool.loc[chosen, [*BASELINE_COLUMNS, TREATMENT_NAME, "time", "event"]].reset_index(drop=True).copy()
            static["subject_id"] = np.arange(len(static), dtype=int)
        template_pool = self.long
        if treatment is not None and TREATMENT_NAME in template_pool:
            arm_pool = template_pool[template_pool[TREATMENT_NAME].eq(int(treatment))]
            if not arm_pool.empty:
                template_pool = arm_pool
        template_ids = template_pool["subject_id"].dropna().astype(int).unique()
        by_sid = {int(pid): g.sort_values("visit_time").copy() for pid, g in template_pool.groupby("subject_id")}
        rows = []
        for new_id, st in static.iterrows():
            source = int(self.rng.choice(template_ids)) if len(template_ids) else None
            g = by_sid.get(source, pd.DataFrame()).copy()
            if g.empty:
                continue
            g = g[pd.to_numeric(g["visit_time"], errors="coerce") <= safe_float(st.get("time"), 1.0) + 1e-8].copy()
            if g.empty and source in by_sid:
                g = by_sid[source].head(1).copy()
            if g.empty:
                continue
            g["subject_id"] = int(new_id)
            g[TREATMENT_NAME] = int(st[TREATMENT_NAME])
            rows.append(g[["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *[v for v in LONGITUDINAL_NAMES if v in g]]])
        static_out, long_out = normalize_output(static, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame())
        status = dict(self.status)
        status["longitudinal_generation"] = "fast_empirical_template_for_exp1"
        return static_out, long_out, status


class TinyConditionalTabularGenerator(nn.Module):
    def __init__(self, noise_dim: int, condition_dim: int, out_dim: int):
        super().__init__()
        hidden = max(16, min(96, out_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, condition], dim=1))


class TinyConditionalTabularDiscriminator(nn.Module):
    def __init__(self, in_dim: int, condition_dim: int):
        super().__init__()
        hidden = max(16, min(96, in_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(in_dim + condition_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, condition], dim=1))


class CTGANLikeGenerator:
    name = "CTGAN"

    def __init__(
        self,
        train_static: pd.DataFrame,
        train_long: pd.DataFrame,
        seed: int,
        epochs: int = 160,
        noise_dim: int = 16,
    ):
        self.static = fill_baseline(train_static.reset_index(drop=True), train_static)
        self.long = train_long.copy()
        self.summary = subject_summary(self.static, self.long)
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(int(seed))
        self.feature_cols = [c for c in self.summary.columns if c not in {"subject_id", "source_id", "treatment_label", "status"}]
        self.feature_cols = [c for c in self.feature_cols if pd.api.types.is_numeric_dtype(pd.to_numeric(self.summary[c], errors="coerce"))]
        clean = self.summary[self.feature_cols].apply(pd.to_numeric, errors="coerce")
        self.fills = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(self.fills)
        if clean.empty or clean.shape[1] == 0:
            raise ValueError("CTGAN requires at least one numeric subject-summary feature.")
        self.scaler = StandardScaler().fit(clean)
        x = torch.tensor(self.scaler.transform(clean), dtype=torch.float32)
        treatments = pd.to_numeric(self.summary.get(TREATMENT_NAME, self.static[TREATMENT_NAME]), errors="coerce")
        treatments = treatments.fillna(0).round().clip(0, 1).astype(int).to_numpy()
        self.treatment_probs = self._fit_treatment_probs(treatments)
        condition = self._condition_tensor(treatments)
        self.noise_dim = int(noise_dim)
        self.generator = TinyConditionalTabularGenerator(self.noise_dim, 2, x.shape[1])
        self.discriminator = TinyConditionalTabularDiscriminator(x.shape[1], 2)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=1e-3, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=1e-3, betas=(0.5, 0.9))
        loss_fn = nn.BCEWithLogitsLoss()
        batch_size = min(64, max(8, x.shape[0]))
        self.generator.train()
        self.discriminator.train()
        for _ in range(int(epochs)):
            order = torch.randperm(x.shape[0])
            for start in range(0, x.shape[0], batch_size):
                idx = order[start:start + batch_size]
                real = x[idx]
                cond = condition[idx]
                if real.numel() == 0:
                    continue
                fake = self.generator(torch.randn(len(real), self.noise_dim), cond).detach()
                opt_d.zero_grad()
                real_logits = self.discriminator(real, cond)
                fake_logits = self.discriminator(fake, cond)
                d_loss = loss_fn(real_logits, torch.full_like(real_logits, 0.9)) + loss_fn(fake_logits, torch.zeros_like(fake_logits))
                d_loss.backward()
                opt_d.step()

                opt_g.zero_grad()
                gen = self.generator(torch.randn(len(real), self.noise_dim), cond)
                gen_logits = self.discriminator(gen, cond)
                g_loss = loss_fn(gen_logits, torch.ones_like(gen_logits)) + 0.05 * F.mse_loss(gen.mean(dim=0), real.mean(dim=0))
                g_loss.backward()
                opt_g.step()
        self.generator.eval()
        self.discriminator.eval()
        self.time_grid = np.sort(self.long["visit_time"].dropna().unique())
        if len(self.time_grid) == 0:
            self.time_grid = np.asarray([0.0], dtype=float)
        self.status = {
            "status": "completed",
            "implementation": "local_torch_ctgan_like_conditional_subject_summary_gan",
            "ctgan_like": True,
            "conditioning": "treatment_arm",
        }

    @staticmethod
    def _fit_treatment_probs(treatments: np.ndarray) -> np.ndarray:
        counts = pd.Series(treatments).round().clip(0, 1).astype(int).value_counts(normalize=True)
        probs = np.asarray([float(counts.get(0, 0.0)), float(counts.get(1, 0.0))], dtype=float)
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            return np.asarray([0.5, 0.5], dtype=float)
        return probs / probs.sum()

    @staticmethod
    def _condition_tensor(treatments: np.ndarray | pd.Series | list[Any]) -> torch.Tensor:
        values = pd.to_numeric(pd.Series(treatments), errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
        condition = np.zeros((len(values), 2), dtype=np.float32)
        condition[np.arange(len(values)), values] = 1.0
        return torch.tensor(condition, dtype=torch.float32)

    def _sample_treatments(self, n: int, treatment: int | None = None, target_static: pd.DataFrame | None = None) -> np.ndarray:
        if treatment is not None:
            return np.full(int(n), int(treatment), dtype=int)
        if target_static is not None and TREATMENT_NAME in target_static:
            return pd.to_numeric(target_static[TREATMENT_NAME], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
        return self.rng.choice(np.asarray([0, 1], dtype=int), size=int(n), replace=True, p=self.treatment_probs)

    def _sample_summary_for_treatments(self, treatments: np.ndarray) -> pd.DataFrame:
        condition = self._condition_tensor(treatments)
        with torch.no_grad():
            z = torch.randn(len(treatments), self.noise_dim)
            x = self.generator(z, condition).detach().cpu().numpy()
        sample = pd.DataFrame(self.scaler.inverse_transform(x), columns=self.feature_cols)
        if TREATMENT_NAME in sample:
            sample[TREATMENT_NAME] = treatments
        return sample

    def _sample_survival_by_treatment(self, treatments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        times = []
        events = []
        for treatment in treatments:
            pool = self.static[self.static[TREATMENT_NAME].eq(int(treatment))]
            if pool.empty:
                pool = self.static
            idx = int(self.rng.choice(pool.index.to_numpy(dtype=int)))
            times.append(safe_float(pool.loc[idx, "time"], 1.0))
            events.append(int(safe_float(pool.loc[idx, "event"], 0.0) >= 0.5))
        return np.asarray(times, dtype=float), np.asarray(events, dtype=int)

    def _coerce_static(self, static: pd.DataFrame) -> pd.DataFrame:
        out = fill_baseline(static, self.static)
        if TREATMENT_NAME not in out:
            out[TREATMENT_NAME] = self._sample_treatments(len(out))
        out[TREATMENT_NAME] = pd.to_numeric(out[TREATMENT_NAME], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
        ref_time = pd.to_numeric(self.static["time"], errors="coerce").dropna()
        positive_grid = self.time_grid[self.time_grid > 0]
        min_followup = float(positive_grid[0]) if len(positive_grid) else 1e-4
        if "time" not in out:
            out["time"] = self.rng.choice(ref_time.to_numpy(dtype=float), size=len(out), replace=True)
        else:
            fill_time = float(ref_time.median()) if not ref_time.empty else 1.0
            lower = float(ref_time.quantile(0.01)) if len(ref_time) > 1 else 1e-4
            upper = float(ref_time.quantile(0.99)) if len(ref_time) > 1 else max(fill_time, 1.0)
            out["time"] = pd.to_numeric(out["time"], errors="coerce").fillna(fill_time).clip(lower=max(lower, 1e-4), upper=max(upper, 1e-4))
        event_ref = pd.to_numeric(self.static["event"], errors="coerce").fillna(0).round().clip(0, 1)
        if "event" not in out:
            out["event"] = self.rng.binomial(1, float(event_ref.mean()) if len(event_ref) else 0.5, size=len(out))
        else:
            out["event"] = pd.to_numeric(out["event"], errors="coerce").fillna(float(event_ref.mean()) if len(event_ref) else 0.0).round().clip(0, 1).astype(int)
        time_values = pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float)
        bad_time = ~np.isfinite(time_values) | (time_values < min_followup)
        if len(time_values) and np.nanmedian(time_values) < min_followup:
            bad_time[:] = True
        if bad_time.any():
            sampled_time, sampled_event = self._sample_survival_by_treatment(out[TREATMENT_NAME].to_numpy(dtype=int))
            out.loc[bad_time, "time"] = sampled_time[bad_time]
            out.loc[bad_time, "event"] = sampled_event[bad_time]
        for col in BASELINE_COLUMNS:
            if col not in out:
                continue
            out[col] = pd.to_numeric(out[col], errors="coerce")
            if col in CATEGORICAL_BASELINE:
                high = 3 if col.endswith("_stage") else 2 if col.endswith("_edema") else 1
                out[col] = out[col].round().clip(0, high)
            elif col != "age":
                out[col] = out[col].clip(lower=0.0)
        return out

    @staticmethod
    def _coerce_longitudinal_value(var: str, value: float) -> float:
        if var in {"ascites", "hepatomegaly", "spiders"}:
            return float(np.clip(round(value), 0, 1))
        if var == "edema":
            return float(np.clip(round(value), 0, 2))
        if var == "stage":
            return float(np.clip(round(value), 0, 3))
        return float(max(0.0, value))

    def generate(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if target_baseline is not None:
            static = target_baseline.copy().head(int(n)).reset_index(drop=True)
            static = fill_baseline(static, self.static)
            static["subject_id"] = np.arange(len(static), dtype=int)
            if treatment is not None:
                static[TREATMENT_NAME] = int(treatment)
            treatments = self._sample_treatments(len(static), treatment=treatment, target_static=static)
            summary = self._sample_summary_for_treatments(treatments)
            for col in ["time", "event"]:
                if col in summary:
                    static[col] = summary[col].to_numpy()
        else:
            treatments = self._sample_treatments(int(n), treatment=treatment)
            summary = self._sample_summary_for_treatments(treatments)
            static_cols = [c for c in [*BASELINE_COLUMNS, TREATMENT_NAME, "time", "event"] if c in summary]
            static = summary[static_cols].copy()
            static.insert(0, "subject_id", np.arange(len(static), dtype=int))
            if treatment is not None:
                static[TREATMENT_NAME] = int(treatment)
        static = self._coerce_static(static)
        summary = summary.reset_index(drop=True)
        rows = []
        grid_max = float(np.nanmax(self.time_grid)) if len(self.time_grid) else 1.0
        grid_max = grid_max if grid_max > 0 else 1.0
        for pos, st in static.reset_index(drop=True).iterrows():
            sid = int(st["subject_id"])
            st_summary = summary.iloc[pos] if pos < len(summary) else pd.Series(dtype=float)
            for visit_index, t in enumerate(self.time_grid):
                if t > safe_float(st["time"], 1.0) + 1e-8:
                    continue
                row = {"subject_id": sid, "visit_index": int(visit_index), "visit_time": float(t), TREATMENT_NAME: int(st[TREATMENT_NAME])}
                for extra in ["sample", "replicate"]:
                    if extra in static:
                        row[extra] = st[extra]
                frac = float(t / grid_max)
                for var in LONGITUDINAL_NAMES:
                    base = safe_float(st.get(f"L0_{var}"), safe_float(st_summary.get(f"L0_{var}"), 0.0))
                    change = safe_float(st_summary.get(f"{var}_change"), 0.0)
                    noise = self.rng.normal(0.0, 0.04 * (abs(base) + 1.0))
                    row[var] = self._coerce_longitudinal_value(var, base + frac * change + noise)
                rows.append(row)
        static_out, long_out = normalize_output(static, pd.DataFrame(rows))
        return static_out, long_out, dict(self.status)

    def generate_fast(self, n: int, treatment: int | None = None, target_baseline: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        static_out, long_out, status = self.generate(n, treatment=treatment, target_baseline=target_baseline)
        status["longitudinal_generation"] = "ctgan_like_summary_decoder"
        return static_out, long_out, status


def pbc_to_phasesyn_tables(static: pd.DataFrame, long_df: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = static.copy().reset_index(drop=True)
    raw = fill_baseline(raw, static)
    raw_phase = pd.DataFrame({
        "time": pd.to_numeric(raw["time"], errors="coerce").fillna(1.0),
        "censor": pd.to_numeric(raw["event"], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "drug": pd.to_numeric(raw[TREATMENT_NAME], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "sex": pd.to_numeric(raw["sex"], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "ascites": pd.to_numeric(raw["L0_ascites"], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "hepatomegaly": pd.to_numeric(raw["L0_hepatomegaly"], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "spiders": pd.to_numeric(raw["L0_spiders"], errors="coerce").fillna(0).round().clip(0, 1).astype(int),
        "edema": pd.to_numeric(raw["L0_edema"], errors="coerce").fillna(0).round().clip(0, 2).astype(int),
        "stage": pd.to_numeric(raw["L0_stage"], errors="coerce").fillna(0).round().clip(0, 3).astype(int),
        "bili": pd.to_numeric(raw["L0_bili"], errors="coerce").fillna(pd.to_numeric(raw["L0_bili"], errors="coerce").median()).clip(lower=0.0),
        "cholesterol": pd.to_numeric(raw["L0_cholesterol"], errors="coerce").fillna(pd.to_numeric(raw["L0_cholesterol"], errors="coerce").median()).clip(lower=0.0),
        "albumin": pd.to_numeric(raw["L0_albumin"], errors="coerce").fillna(pd.to_numeric(raw["L0_albumin"], errors="coerce").median()),
        "alkaline": pd.to_numeric(raw["L0_alkaline"], errors="coerce").fillna(pd.to_numeric(raw["L0_alkaline"], errors="coerce").median()).clip(lower=0.0),
        "ast": pd.to_numeric(raw["L0_ast"], errors="coerce").fillna(pd.to_numeric(raw["L0_ast"], errors="coerce").median()).clip(lower=0.0),
        "platelets": pd.to_numeric(raw["L0_platelets"], errors="coerce").fillna(pd.to_numeric(raw["L0_platelets"], errors="coerce").median()).clip(lower=0.0),
        "prothrombin": pd.to_numeric(raw["L0_prothrombin"], errors="coerce").fillna(pd.to_numeric(raw["L0_prothrombin"], errors="coerce").median()).clip(lower=0.0),
        "age": pd.to_numeric(raw["age"], errors="coerce").fillna(pd.to_numeric(raw["age"], errors="coerce").median()),
    })
    id_df = pd.DataFrame({"id": static["subject_id"].to_numpy(dtype=int), "source_id": static.get("source_id", static["subject_id"]).to_numpy()})
    subject_to_panel = {int(sid): int(i) for i, sid in enumerate(static["subject_id"].to_numpy(dtype=int))}
    long_phase = long_df.copy()
    long_phase["subject_id"] = long_phase["subject_id"].astype(int).map(subject_to_panel)
    long_phase = long_phase[long_phase["subject_id"].notna()].copy()
    long_phase["subject_id"] = long_phase["subject_id"].astype(int)
    long_phase = long_phase[["subject_id", "visit_time", *LONGITUDINAL_NAMES]].rename(columns={"subject_id": "patient_id"})
    for col in LONGITUDINAL_NAMES:
        if col in {"ascites", "hepatomegaly", "spiders"}:
            long_phase[col] = pd.to_numeric(long_phase[col], errors="coerce").fillna(raw_phase[col].median()).round().clip(0, 1)
        elif col == "edema":
            long_phase[col] = pd.to_numeric(long_phase[col], errors="coerce").fillna(raw_phase[col].median()).round().clip(0, 2)
        elif col == "stage":
            long_phase[col] = pd.to_numeric(long_phase[col], errors="coerce").fillna(raw_phase[col].median()).round().clip(0, 3)
        else:
            long_phase[col] = pd.to_numeric(long_phase[col], errors="coerce").fillna(raw_phase[col].median()).clip(lower=0.0 if col not in {"albumin"} else None)
    types = [
        {"name": "survcens", "type": "surv_dynamic", "dim": "2", "nclass": ""},
        {"name": "drug", "type": "cat", "dim": "1", "nclass": "2"},
        {"name": "sex", "type": "cat", "dim": "1", "nclass": "2"},
        {"name": "ascites", "type": "cat", "dim": "1", "nclass": "2"},
        {"name": "hepatomegaly", "type": "cat", "dim": "1", "nclass": "2"},
        {"name": "spiders", "type": "cat", "dim": "1", "nclass": "2"},
        {"name": "edema", "type": "ordinal", "dim": "1", "nclass": "3"},
        {"name": "stage", "type": "ordinal", "dim": "1", "nclass": "4"},
        {"name": "bili", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "cholesterol", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "albumin", "type": "real", "dim": "1", "nclass": ""},
        {"name": "alkaline", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "ast", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "platelets", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "prothrombin", "type": "pos", "dim": "1", "nclass": ""},
        {"name": "age", "type": "real", "dim": "1", "nclass": ""},
    ]
    raw_phase.to_csv(output_dir / "data_phasesyn.csv", index=False)
    id_df.to_csv(output_dir / "pbc_core4_id.csv", index=False)
    long_phase.to_csv(output_dir / "longitudinal.csv", index=False)
    pd.DataFrame(types).to_csv(output_dir / "data_types_phasesyn_piecewise.csv", index=False)
    return raw_phase, id_df, long_phase, types


class PhaseSynGenerator:
    name = "PhaseSyn"

    def __init__(self, cfg: dict[str, Any], train_static: pd.DataFrame, train_long: pd.DataFrame, output_dir: Path, seed: int):
        self.cfg = cfg
        self.seed = int(seed)
        self.output_dir = output_dir
        self.device = torch.device(cfg.get("phasesyn", {}).get("device", "cpu"))
        self.train_static = fill_baseline(train_static.reset_index(drop=True), train_static)
        self.train_long = train_long.copy()
        self.model: PhaseSynModel | None = None
        self.bundle = None
        self.static_prep = None
        self.long_prep = None
        self.types = None
        self.rng = np.random.default_rng(self.seed)

    def train(self, smoke: bool = False) -> dict[str, Any]:
        set_seed(self.seed)
        train_static = self.train_static
        train_long = self.train_long
        max_subjects = self.cfg.get("smoke", {}).get("max_train_subjects") if smoke else None
        if max_subjects:
            keep = train_static.head(int(max_subjects))["subject_id"].tolist()
            train_static = train_static[train_static["subject_id"].isin(keep)].reset_index(drop=True)
            train_long = train_long[train_long["subject_id"].isin(keep)].reset_index(drop=True)
        data_dir = self.output_dir / "phasesyn_adapted_data"
        raw, ids, long_phase, types = pbc_to_phasesyn_tables(train_static, train_long, data_dir)
        phase = self.cfg.get("phasesyn", {})
        epochs = int(self.cfg.get("smoke", {}).get("phasesyn_epochs", 2)) if smoke else int(phase.get("epochs", 35))
        model_cfg = {
            "longitudinal_mode": "latent_ode",
            "survival": "dynamic",
            "z_dim": int(phase.get("z_dim", 6)),
            "s_dim": int(phase.get("s_dim", 6)),
            "y_dim_static": int(phase.get("y_dim_static", 6)),
            "u_dim": int(phase.get("u_dim", 6)),
            "gru_hidden_dim": int(phase.get("gru_hidden_dim", 6)),
            "ode_hidden_dim": int(phase.get("ode_hidden_dim", 6)),
            "decoder_hidden_dim": int(phase.get("decoder_hidden_dim", 6)),
            "u0_initializer_hidden_dim": int(phase.get("u0_initializer_hidden_dim", phase.get("u_dim", 6))),
            "dynamic_survival_hidden_dim": int(phase.get("dynamic_survival_hidden_dim", 6)),
            "dynamic_survival_num_layers": int(phase.get("dynamic_survival_num_layers", 2)),
            "dynamic_survival_dropout": float(phase.get("dynamic_survival_dropout", 0.0)),
            "use_u0_mean_at_eval": bool(phase.get("use_u0_mean_at_eval", False)),
            "n_intervals": int(phase.get("n_intervals", 16)),
            "encoder_conditioning": "baseline_only",
            "u0_init_mode": "baseline_l0",
            "treatment_variable_name": "drug",
            "kl_weight_s": float(phase.get("kl_weight_s", 0.3)),
            "kl_weight_z": float(phase.get("kl_weight_z", 0.3)),
            "kl_weight_u": float(phase.get("kl_weight_u", 0.0)),
            "longitudinal_weight": float(phase.get("longitudinal_weight", 2.0)),
            "lambda_surv": float(phase.get("lambda_surv", 1.4)),
            "survival_warmup_epochs": int(phase.get("survival_warmup_epochs", 0)),
            "continuous_mse_weight": float(phase.get("continuous_mse_weight", 0.8)),
        }
        pdc_cfg = load_config(None, {
            "dataset": {"name": "pdc2", "data_dir": str(data_dir), "output_root": str(self.output_dir / "phasesyn_model")},
            "model": model_cfg,
            "training": {
                "epochs": epochs,
                "batch_size": int(phase.get("batch_size", 64)),
                "lr": float(phase.get("lr", 0.001)),
                "seed": self.seed,
                "device": str(self.device),
                "freeze_normalization": True,
            },
            "evaluation": {
                "deterministic_static_export": False,
                "calibrate_static_covariates": False,
                "calibrate_survival_km": False,
                "calibrate_survival_event_rate": False,
            },
        })
        static_prep = _fit_static_preprocessor(raw, types)
        long_prep = _fit_longitudinal_preprocessor(long_phase, types, np.arange(len(raw)), None, train_survival_times=raw["time"])
        bundle = _make_bundle(raw, ids, long_phase, types, np.arange(len(raw)), static_prep, long_prep, pdc_cfg)
        start = time.time()
        result = train_model(bundle, pdc_cfg, output_dir=self.output_dir / "phasesyn_model" / "train")
        self.model = result["model"].to(self.device)
        self.model.eval()
        self.bundle = bundle
        self.static_prep = static_prep
        self.long_prep = long_prep
        self.types = types
        (self.output_dir / "phasesyn_model").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "phasesyn_model" / "core4_phasesyn_config.yaml").write_text(yaml.safe_dump(pdc_cfg, sort_keys=False), encoding="utf-8")
        return {
            "method": self.name,
            "status": "completed",
            "runtime_seconds": float(time.time() - start),
            "epochs": epochs,
            "checkpoint": str(self.output_dir / "phasesyn_model" / "train" / "model_checkpoint.pt"),
        }

    def prepare_bundle(self, smoke: bool = False) -> dict[str, Any]:
        train_static = self.train_static
        train_long = self.train_long
        max_subjects = self.cfg.get("smoke", {}).get("max_train_subjects") if smoke else None
        if max_subjects:
            keep = train_static.head(int(max_subjects))["subject_id"].tolist()
            train_static = train_static[train_static["subject_id"].isin(keep)].reset_index(drop=True)
            train_long = train_long[train_long["subject_id"].isin(keep)].reset_index(drop=True)
        data_dir = self.output_dir / "phasesyn_adapted_data"
        raw, ids, long_phase, types = pbc_to_phasesyn_tables(train_static, train_long, data_dir)
        phase = self.cfg.get("phasesyn", {})
        pdc_cfg = load_config(None, {
            "dataset": {"name": "pdc2", "data_dir": str(data_dir), "output_root": str(self.output_dir / "phasesyn_model")},
            "model": {
                "longitudinal_mode": "latent_ode",
                "survival": "dynamic",
                "z_dim": int(phase.get("z_dim", 6)),
                "s_dim": int(phase.get("s_dim", 6)),
                "y_dim_static": int(phase.get("y_dim_static", 6)),
                "u_dim": int(phase.get("u_dim", 6)),
                "gru_hidden_dim": int(phase.get("gru_hidden_dim", 6)),
                "ode_hidden_dim": int(phase.get("ode_hidden_dim", 6)),
                "decoder_hidden_dim": int(phase.get("decoder_hidden_dim", 6)),
                "u0_initializer_hidden_dim": int(phase.get("u0_initializer_hidden_dim", phase.get("u_dim", 6))),
                "dynamic_survival_hidden_dim": int(phase.get("dynamic_survival_hidden_dim", 6)),
                "dynamic_survival_num_layers": int(phase.get("dynamic_survival_num_layers", 2)),
                "dynamic_survival_dropout": float(phase.get("dynamic_survival_dropout", 0.0)),
                "use_u0_mean_at_eval": bool(phase.get("use_u0_mean_at_eval", False)),
                "n_intervals": int(phase.get("n_intervals", 16)),
                "encoder_conditioning": "baseline_only",
                "u0_init_mode": "baseline_l0",
                "treatment_variable_name": "drug",
                "kl_weight_s": float(phase.get("kl_weight_s", 0.3)),
                "kl_weight_z": float(phase.get("kl_weight_z", 0.3)),
                "kl_weight_u": float(phase.get("kl_weight_u", 0.0)),
                "longitudinal_weight": float(phase.get("longitudinal_weight", 2.0)),
                "lambda_surv": float(phase.get("lambda_surv", 1.4)),
                "survival_warmup_epochs": int(phase.get("survival_warmup_epochs", 0)),
                "continuous_mse_weight": float(phase.get("continuous_mse_weight", 0.8)),
            },
            "training": {
                "epochs": int(phase.get("epochs", 35)),
                "batch_size": int(phase.get("batch_size", 64)),
                "lr": float(phase.get("lr", 0.001)),
                "seed": self.seed,
                "device": str(self.device),
                "freeze_normalization": True,
            },
            "evaluation": {
                "deterministic_static_export": False,
                "calibrate_static_covariates": False,
                "calibrate_survival_km": False,
                "calibrate_survival_event_rate": False,
            },
        })
        static_prep = _fit_static_preprocessor(raw, types)
        long_prep = _fit_longitudinal_preprocessor(long_phase, types, np.arange(len(raw)), None, train_survival_times=raw["time"])
        bundle = _make_bundle(raw, ids, long_phase, types, np.arange(len(raw)), static_prep, long_prep, pdc_cfg)
        self.bundle = bundle
        self.static_prep = static_prep
        self.long_prep = long_prep
        self.types = types
        return pdc_cfg

    def load_checkpoint(self, checkpoint: Path | None = None, smoke: bool = False) -> dict[str, Any]:
        set_seed(self.seed)
        pdc_cfg = self.prepare_bundle(smoke=smoke)
        checkpoint = Path(checkpoint or (self.output_dir / "phasesyn_model" / "train" / "model_checkpoint.pt"))
        if not checkpoint.exists():
            raise FileNotFoundError(f"PhaseSyn checkpoint not found: {checkpoint}")
        loaded = torch.load(checkpoint, map_location=str(self.device))
        state = loaded.get("model_state_dict", loaded)
        model = build_model(self.bundle, pdc_cfg).to(self.device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        allowed_missing = ("u0_logsigma_head.", "survival_time_head.")
        bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if bad_missing or unexpected:
            raise RuntimeError(
                "Checkpoint is not compatible with reconstructed PhaseSyn model. "
                f"bad_missing={bad_missing[:10]}, unexpected={unexpected[:10]}"
            )
        model.eval()
        self.model = model
        (self.output_dir / "phasesyn_model").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "phasesyn_model" / "core4_phasesyn_config.yaml").write_text(yaml.safe_dump(pdc_cfg, sort_keys=False), encoding="utf-8")
        return {
            "method": self.name,
            "status": "completed",
            "checkpoint": str(checkpoint),
            "loaded_without_retraining": True,
            "missing_keys_ignored": list(missing),
        }

    def _bundle_for_targets(self, target_static: pd.DataFrame, treatment: int | None = None, time_grid: np.ndarray | None = None):
        if self.bundle is None or self.static_prep is None or self.long_prep is None or self.types is None:
            raise RuntimeError("PhaseSynGenerator.train() must be called before generation.")
        static = fill_baseline(target_static.copy(), self.train_static)
        if treatment is not None:
            static[TREATMENT_NAME] = int(treatment)
        static["subject_id"] = np.arange(len(static), dtype=int)
        static["time"] = 1.0
        static["event"] = 0
        if time_grid is None:
            time_grid = np.asarray(self.cfg.get("generation", {}).get("time_grid", [0.0, 1.0, 2.0, 3.0, 5.0]), dtype=float)
        if 0.0 not in set(np.round(time_grid.astype(float), 10)):
            time_grid = np.concatenate([[0.0], time_grid])
        time_grid = np.asarray(sorted(set(float(x) for x in time_grid if np.isfinite(x))), dtype=float)
        long_rows = []
        for _, st in static.iterrows():
            for visit_index, visit_time in enumerate(time_grid):
                rec = {
                    "subject_id": int(st["subject_id"]),
                    "visit_index": int(visit_index),
                    "visit_time": float(visit_time),
                    TREATMENT_NAME: int(st[TREATMENT_NAME]),
                }
                for var in LONGITUDINAL_NAMES:
                    rec[var] = safe_float(st.get(f"L0_{var}"), 0.0)
                long_rows.append(rec)
        raw, ids, long_phase, _ = pbc_to_phasesyn_tables(static, pd.DataFrame(long_rows), self.output_dir / "phasesyn_target_data")
        return _make_bundle(raw, ids, long_phase, self.types, np.arange(len(raw)), self.static_prep, self.long_prep, self.cfg_for_bundle())

    def cfg_for_bundle(self) -> dict[str, Any]:
        phase = self.cfg.get("phasesyn", {})
        return load_config(None, {
            "dataset": {"name": "pdc2", "data_dir": str(self.output_dir / "phasesyn_target_data")},
            "model": {"treatment_variable_name": "drug", "n_intervals": int(phase.get("n_intervals", 16))},
            "training": {"device": str(self.device), "seed": self.seed},
        })

    def generate(
        self,
        n: int,
        treatment: int | None = None,
        target_baseline: pd.DataFrame | None = None,
        time_grid: np.ndarray | None = None,
        truncate_longitudinal_at_survival: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if self.model is None or self.bundle is None:
            raise RuntimeError("PhaseSynGenerator is not trained.")
        preserve_ids = target_baseline is not None
        if target_baseline is None:
            target_baseline = self.train_static.sample(
                int(n),
                replace=True,
                random_state=int(self.rng.integers(0, np.iinfo(np.int32).max)),
            ).reset_index(drop=True)
            if treatment is not None:
                target_baseline[TREATMENT_NAME] = int(treatment)
        bundle = self._bundle_for_targets(target_baseline.head(int(n)), treatment=treatment, time_grid=time_grid)
        with torch.no_grad():
            syn_static, latents, audit = _decode_baseline_conditioned_static(self.model, bundle, self.device)
            pred_raw, future_mask, _, support = _sample_longitudinal_future(self.model, bundle, latents, self.device, sample=True)
        static = self._decode_static(syn_static, target_baseline.head(int(n)), treatment, preserve_target_ids=preserve_ids)
        long_df = self._decode_longitudinal(
            bundle,
            pred_raw,
            static,
            truncate_at_survival=truncate_longitudinal_at_survival,
        )
        return static, long_df, {"status": "completed", **audit, **support}

    def generate_prior(
        self,
        n: int,
        treatment: int,
        time_grid: np.ndarray | list[float] | None = None,
        deterministic: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        if self.model is None or self.bundle is None:
            raise RuntimeError("PhaseSynGenerator is not trained.")
        if time_grid is None:
            time_grid = np.asarray(self.cfg.get("generation", {}).get("time_grid", [0.0, 1.0, 2.0, 3.0, 5.0]), dtype=float)
        time_grid_raw = np.asarray(time_grid, dtype=float)
        time_min = float(self.bundle.longitudinal.time_min)
        time_max = float(self.bundle.longitudinal.time_max)
        denom = max(time_max - time_min, 1e-8)
        time_grid_norm = np.clip((time_grid_raw - time_min) / denom, 0.0, 1.0)
        static_raw, long_raw, tensors = generate_prior_cohort(
            self.model,
            self.bundle,
            int(n),
            int(treatment),
            time_grid_norm,
            self.device,
            deterministic=deterministic,
            return_tensors=True,
        )
        static = self._decode_prior_static(static_raw)
        long_df = self._decode_prior_longitudinal(long_raw)
        static_out, long_out = normalize_output(static, long_df)
        return static_out, long_out, {
            "status": "completed",
            "generation_mode": "learned_prior",
            "target_baseline_used": False,
            "prior_time_grid_input_scale": "clinical_years_normalized_to_training_range",
            "baseline_generated_from_prior": bool(tensors.get("baseline_generated_from_prior", torch.tensor(False)).detach().cpu().item()),
        }

    def _decode_prior_static(self, static_raw: pd.DataFrame) -> pd.DataFrame:
        def col(name: str, default: float) -> pd.Series:
            if name in static_raw:
                return pd.to_numeric(static_raw[name], errors="coerce")
            return pd.Series(default, index=static_raw.index, dtype=float)

        out = pd.DataFrame({
            "subject_id": col("patient_id", 0.0).fillna(pd.Series(np.arange(len(static_raw)), index=static_raw.index)).astype(int),
            TREATMENT_NAME: col("drug", 0.0).fillna(0).round().clip(0, 1).astype(int),
            "time": col("time", 1.0).fillna(1.0).clip(lower=1e-4),
            "event": col("censor", 0.0).fillna(0).round().clip(0, 1).astype(int),
        })
        for col in ["sex", "age"]:
            if col in static_raw:
                out[col] = pd.to_numeric(static_raw[col], errors="coerce")
        for var in LONGITUDINAL_NAMES:
            raw_name = {
                "bili": "bili",
                "cholesterol": "cholesterol",
                "ast": "ast",
                "stage": "stage",
            }.get(var, var)
            if raw_name in static_raw:
                out[f"L0_{var}"] = pd.to_numeric(static_raw[raw_name], errors="coerce")
        return fill_baseline(out, self.train_static)

    def _decode_prior_longitudinal(self, long_raw: pd.DataFrame) -> pd.DataFrame:
        if long_raw.empty:
            return pd.DataFrame(columns=["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *LONGITUDINAL_NAMES])
        out = long_raw.rename(columns={"patient_id": "subject_id", "drug": TREATMENT_NAME}).copy()
        keep = ["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *[v for v in LONGITUDINAL_NAMES if v in out]]
        out = out[[c for c in keep if c in out]].copy()
        out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").fillna(0).astype(int)
        out["visit_index"] = pd.to_numeric(out.get("visit_index", 0), errors="coerce").fillna(0).astype(int)
        out["visit_time"] = pd.to_numeric(out["visit_time"], errors="coerce")
        out[TREATMENT_NAME] = pd.to_numeric(out.get(TREATMENT_NAME, 0), errors="coerce").fillna(0).round().clip(0, 1).astype(int)
        return out

    def _decode_static(self, syn_static: pd.DataFrame, target: pd.DataFrame, treatment: int | None, preserve_target_ids: bool = False) -> pd.DataFrame:
        out = target.copy().reset_index(drop=True)
        if preserve_target_ids and "subject_id" in out:
            out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce").fillna(pd.Series(np.arange(len(out)), index=out.index)).astype(int)
        else:
            out["subject_id"] = np.arange(len(out), dtype=int)
        out[TREATMENT_NAME] = int(treatment) if treatment is not None else pd.to_numeric(target[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int)
        out["time"] = pd.to_numeric(syn_static["time"], errors="coerce").fillna(1.0).clip(lower=1e-4).to_numpy(dtype=float)
        out["event"] = pd.to_numeric(syn_static["censor"], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
        return fill_baseline(out, self.train_static)

    def _decode_longitudinal(
        self,
        bundle: Any,
        pred_raw: np.ndarray,
        static: pd.DataFrame,
        truncate_at_survival: bool = True,
    ) -> pd.DataFrame:
        times_norm = bundle.longitudinal.times.detach().cpu().numpy()
        times_raw = times_norm * (bundle.longitudinal.time_max - bundle.longitudinal.time_min) + bundle.longitudinal.time_min
        masks = bundle.longitudinal.masks.detach().cpu().numpy()
        observed_rows = masks.sum(axis=-1) > 0
        rows = []
        for i in range(pred_raw.shape[0]):
            for visit in range(pred_raw.shape[1]):
                if not bool(observed_rows[i, visit]):
                    continue
                t = float(times_raw[i, visit])
                if truncate_at_survival and t > float(static.iloc[i]["time"]) + 1e-8:
                    continue
                row = {
                    "subject_id": int(static.iloc[i]["subject_id"]),
                    "visit_index": int(visit),
                    "visit_time": t,
                    TREATMENT_NAME: int(static.iloc[i][TREATMENT_NAME]),
                }
                for extra in ["sample", "replicate"]:
                    if extra in static:
                        row[extra] = static.iloc[i][extra]
                for j, spec in enumerate(bundle.longitudinal.specs):
                    row[spec.name] = float(pred_raw[i, visit, j])
                rows.append(row)
        return pd.DataFrame(rows)


def build_method(method: str, train_static: pd.DataFrame, train_long: pd.DataFrame, seed: int, use_mixedlm: bool = True) -> Any:
    if method == "empirical_subject_bootstrap":
        return EmpiricalSubjectBootstrap(train_static, train_long, seed)
    if method in {"LMM-AFT", "classical_lmm_cox_aft_simulator"}:
        return ClassicalSimulator(train_static, train_long, seed, use_mixedlm=use_mixedlm)
    if method in {"JM-RE", "joint_longitudinal_survival_baseline"}:
        return JointSharedRandomEffects(train_static, train_long, seed, use_mixedlm=use_mixedlm)
    if method in {"TVAE", "modular_deep_generator"}:
        return TVAEGenerator(train_static, train_long, seed)
    if method == "CTGAN":
        return CTGANLikeGenerator(train_static, train_long, seed)
    raise ValueError(f"Unknown benchmark method: {method}")


def write_method_status(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    failures = [r for r in rows if str(r.get("status", "")).startswith("failed")]
    warnings = [r for r in rows if r.get("dependency_warning")]
    if failures:
        lines = ["# Benchmark Failures", ""]
        for row in failures:
            lines.append(f"- {row.get('method')}: {row.get('status')} {row.get('reason', row.get('dependency_warning', ''))}")
        if warnings:
            lines.extend(["", "## Dependency Warnings", ""])
            for row in warnings:
                lines.append(f"- {row.get('method')}: {row.get('dependency_warning')}")
        path.with_name("benchmark_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif warnings:
        lines = ["# Benchmark Failures", "", "No benchmark method failed.", "", "## Dependency Warnings", ""]
        lines.extend(f"- {row.get('method')}: {row.get('dependency_warning')}" for row in warnings)
        path.with_name("benchmark_failures.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif not path.with_name("benchmark_failures.md").exists():
        path.with_name("benchmark_failures.md").write_text("# Benchmark Failures\n\nNo benchmark dependency failures were observed.\n", encoding="utf-8")


def dependency_status() -> dict[str, Any]:
    import importlib.util

    return {
        "lifelines": bool(importlib.util.find_spec("lifelines")),
        "statsmodels": bool(importlib.util.find_spec("statsmodels")),
        "sklearn": bool(importlib.util.find_spec("sklearn")),
        "torch": bool(importlib.util.find_spec("torch")),
        "ctgan": bool(importlib.util.find_spec("ctgan")),
        "synthcity": bool(importlib.util.find_spec("synthcity")),
    }


def write_dependency_status(path: Path) -> dict[str, Any]:
    status = dependency_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status
