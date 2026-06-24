#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from pdc2.models import PhaseSynModel, set_seed  # noqa: E402
from pdc2.training import generate_prior_cohort, train_model  # noqa: E402
from scripts.run_simulation_holdout import (  # noqa: E402
    TREATMENT_NAME,
    _candidate_config,
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _make_bundle,
    _model_audit,
    _read_simulation,
)
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    _future_generation_perturbation_audit,
    _jsonable,
    _leakage_diagnostics,
    _survival_generation_perturbation_audit,
)

try:
    from lifelines import CoxPHFitter
    from lifelines.statistics import logrank_test
except Exception:  # pragma: no cover - recorded at runtime
    CoxPHFitter = None
    logrank_test = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - recorded at runtime
    sm = None


REQUESTED_DATASET_DIR = Path(
    "/project/Stat/s1155202253/myproject/pfizer_projects/data/simulation data/simple_linear_rct_n1200 data"
)
FALLBACK_DATASET_DIR = Path(
    "/project/Stat/s1155202253/myproject/pfizer_projects/data/simulation data/simple_linear_rct_n1200"
)
DEFAULT_OUTPUT_DIR = Path(
    "/project/Stat/s1155202253/myproject/pfizer_projects/PhaseSyn/outputs/simulation/experiment_20260604"
)
BENCHMARK_DIR = ROOT / "benchmark_methods"

SEED = 20260604
LONG_NAMES = [f"L{i}" for i in range(1, 7)]
STATIC_CONTINUOUS = ["W_cont_1", "W_cont_2", "W_count_1", "W_pos_1"]
STATIC_CATEGORICAL = ["W_bin_1", "W_bin_2", "W_cat_1", "W_ord_1"]
BASELINE_COLS = STATIC_CONTINUOUS + STATIC_CATEGORICAL + LONG_NAMES
SURVIVAL_TIME_COL = "time"
EVENT_COL = "censor"


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "configs": output_dir / "configs",
        "logs": output_dir / "logs",
        "models": output_dir / "models",
        "synthetic": output_dir / "synthetic",
        "metrics": output_dir / "metrics",
        "figures": output_dir / "figures",
        "tables": output_dir / "tables",
        "reports": output_dir / "reports",
    }
    for path in [output_dir, *paths.values()]:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_status(output_dir: Path, lines: list[str]) -> None:
    text = ["# STATUS", "", f"Updated: {now_iso()}", "", *lines]
    (output_dir / "STATUS.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def append_status(output_dir: Path, line: str) -> None:
    status = output_dir / "STATUS.md"
    if status.exists():
        with status.open("a", encoding="utf-8") as f:
            f.write(line.rstrip() + "\n")
    else:
        write_status(output_dir, [line])


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if np.isfinite(out) else default


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def save_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_jsonable(obj), f, sort_keys=False)


def savefig(fig: plt.Figure, png_path: Path, also_pdf: bool = True) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    if also_pdf:
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def package_versions() -> dict[str, str]:
    packages = [
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "lifelines",
        "statsmodels",
        "matplotlib",
        "torch",
        "PyYAML",
        "pyarrow",
        "ctgan",
        "sdv",
    ]
    out = {}
    for pkg in packages:
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except Exception:
            out[pkg] = "unavailable"
    return out


def git_hash(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def repo_status(path: Path) -> dict[str, Any]:
    status: dict[str, Any] = {"path": str(path), "exists": path.exists(), "git_commit": None}
    if not path.exists():
        return status
    status["git_commit"] = git_hash(path)
    return status


def benchmark_repo_status() -> dict[str, dict[str, Any]]:
    return {
        "CTGAN": repo_status(BENCHMARK_DIR / "CTGAN"),
        "SDV": repo_status(BENCHMARK_DIR / "SDV"),
        "TimeGAN": repo_status(BENCHMARK_DIR / "TimeGAN"),
        "latent_ode": repo_status(BENCHMARK_DIR / "latent_ode"),
        "lifelines": repo_status(BENCHMARK_DIR / "lifelines"),
    }


@contextlib.contextmanager
def temporary_sys_path(*paths: Path):
    old_path = list(sys.path)
    for path in reversed([str(p) for p in paths if p.exists()]):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = old_path


def import_status(module_name: str, *paths: Path) -> dict[str, Any]:
    with temporary_sys_path(*paths):
        try:
            module = __import__(module_name)
            return {
                "module": module_name,
                "available": True,
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
                "error": None,
            }
        except Exception as exc:
            return {
                "module": module_name,
                "available": False,
                "version": None,
                "file": None,
                "error": f"{type(exc).__name__}: {exc}",
            }


def benchmark_dependency_status() -> dict[str, Any]:
    return {
        "repositories": benchmark_repo_status(),
        "imports": {
            "ctgan_from_clone": import_status("ctgan", BENCHMARK_DIR / "CTGAN"),
            "sdv_from_clone": import_status("sdv", BENCHMARK_DIR / "SDV", BENCHMARK_DIR / "CTGAN"),
            "tensorflow": import_status("tensorflow"),
            "latent_ode_lib": import_status("lib", BENCHMARK_DIR / "latent_ode"),
            "torchdiffeq": import_status("torchdiffeq"),
            "lifelines_installed": import_status("lifelines"),
        },
    }


def resolve_dataset_dir(requested: Path, output_dir: Path) -> Path:
    if requested.exists():
        return requested
    if FALLBACK_DATASET_DIR.exists():
        append_status(
            output_dir,
            f"- Dataset path correction: requested `{requested}` was not found; using `{FALLBACK_DATASET_DIR}`.",
        )
        return FALLBACK_DATASET_DIR
    raise FileNotFoundError(f"Neither requested dataset path nor fallback exists: {requested}")


def infer_schema(data_dir: Path, raw: pd.DataFrame, long_df: pd.DataFrame) -> dict[str, Any]:
    observed_path = data_dir / "longitudinal_observed.csv"
    survival_path = data_dir / "survival.csv"
    baseline_path = data_dir / "baseline.csv"
    observed = pd.read_csv(observed_path) if observed_path.exists() else pd.DataFrame()
    survival = pd.read_csv(survival_path) if survival_path.exists() else pd.DataFrame()
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    subject_id = "subject_id" if "subject_id" in baseline.columns else "patient_id"
    visit_col = "visit_time"
    baseline_rows = observed[observed.get("visit_name", pd.Series(dtype=str)).astype(str).eq("baseline")]
    return {
        "dataset_dir": str(data_dir),
        "phase_syn_static_file": "data_phasesyn.csv",
        "subject_id_column": subject_id,
        "phase_syn_panel_subject_id_column": "patient_id",
        "treatment_arm_column": TREATMENT_NAME,
        "baseline_covariates": [c for c in baseline.columns if c.startswith("W_") and not c.startswith("obs_")],
        "baseline_missingness_indicators": [c for c in baseline.columns if c.startswith("obs_W_")],
        "baseline_longitudinal_row": {
            "source_file": "baseline.csv and longitudinal_observed.csv",
            "visit_name": "baseline",
            "visit_time_value": 0.0,
            "complete_l0_columns": LONG_NAMES,
            "n_baseline_rows": int(len(baseline_rows)) if not baseline_rows.empty else int(len(baseline)),
        },
        "post_baseline_longitudinal_variables": LONG_NAMES,
        "visit_time_column": visit_col,
        "visit_index_column": "visit_index" if "visit_index" in observed.columns else None,
        "observed_event_or_censoring_time": {
            "phase_syn_column": SURVIVAL_TIME_COL,
            "survival_file_column": "U" if "U" in survival.columns else SURVIVAL_TIME_COL,
        },
        "event_indicator": {
            "phase_syn_column": EVENT_COL,
            "survival_file_column": "delta" if "delta" in survival.columns else EVENT_COL,
            "meaning": "1=event, 0=censored",
        },
        "censoring_indicator_if_present": {
            "administrative_censor": "administrative_censor" if "administrative_censor" in survival.columns else None,
            "stochastic_censor": "stochastic_censor" if "stochastic_censor" in survival.columns else None,
            "derived_any_censoring": "1 - event_indicator",
        },
        "longitudinal_observation_mask_columns": [c for c in observed.columns if c.startswith("obs_L")],
        "n_subjects": int(raw.shape[0]),
        "n_longitudinal_rows": int(long_df.shape[0]),
        "note": "PhaseSyn-compatible training uses longitudinal.csv; observed masks and dropout are documented from longitudinal_observed.csv.",
    }


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    use_df = df if max_rows is None else df.head(max_rows)
    cols = [str(c) for c in use_df.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in use_df.iterrows():
        vals = []
        for c in use_df.columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.5g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def dataset_summary_table(raw: pd.DataFrame, baseline: pd.DataFrame, survival: pd.DataFrame, long_obs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"section": "cohort", "variable": "subjects", "level": "all", "value": len(raw)})
    rows.append({"section": "cohort", "variable": "rows_longitudinal_observed", "level": "all", "value": len(long_obs)})
    for arm, n in raw[TREATMENT_NAME].value_counts().sort_index().items():
        rows.append({"section": "treatment", "variable": TREATMENT_NAME, "level": int(arm), "value": int(n)})
    rows.append({"section": "survival", "variable": "event_rate", "level": "all", "value": float(raw[EVENT_COL].mean())})
    rows.append({"section": "survival", "variable": "censoring_rate", "level": "all", "value": float(1.0 - raw[EVENT_COL].mean())})
    rows.append({"section": "survival", "variable": "median_followup", "level": "all", "value": float(raw[SURVIVAL_TIME_COL].median())})
    for col in STATIC_CONTINUOUS + LONG_NAMES:
        series = pd.to_numeric(baseline[col], errors="coerce") if col in baseline else pd.to_numeric(raw[col], errors="coerce")
        rows.append({"section": "baseline", "variable": col, "level": "mean", "value": float(series.mean())})
        rows.append({"section": "baseline", "variable": col, "level": "sd", "value": float(series.std(ddof=1))})
        rows.append({"section": "baseline", "variable": col, "level": "missing_rate", "value": float(series.isna().mean())})
    for col in STATIC_CATEGORICAL:
        series = baseline[col] if col in baseline else raw[col]
        rows.append({"section": "baseline", "variable": col, "level": "missing_rate", "value": float(series.isna().mean())})
        for level, prop in series.dropna().value_counts(normalize=True).sort_index().items():
            rows.append({"section": "baseline", "variable": col, "level": level, "value": float(prop)})
    for col in LONG_NAMES:
        mask_col = f"obs_{col}"
        if mask_col in long_obs:
            rows.append({"section": "longitudinal", "variable": col, "level": "observed_cell_rate", "value": float(long_obs[mask_col].mean())})
    if {"administrative_censor", "stochastic_censor"}.issubset(survival.columns):
        rows.append({"section": "survival", "variable": "administrative_censor_rate", "level": "all", "value": float(survival["administrative_censor"].mean())})
        rows.append({"section": "survival", "variable": "stochastic_censor_rate", "level": "all", "value": float(survival["stochastic_censor"].mean())})
    return pd.DataFrame(rows)


def subject_split(raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    labels = raw[TREATMENT_NAME].astype(int).astype(str) + "_" + raw[EVENT_COL].astype(int).astype(str)
    idx = np.arange(len(raw))
    train_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=seed, stratify=labels)
    temp_labels = labels.iloc[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 1, stratify=temp_labels)
    rows = []
    for split, arr in [("train", train_idx), ("validation", val_idx), ("test", test_idx)]:
        for i in sorted(map(int, arr)):
            rows.append({
                "subject_id": int(i),
                "row_index": int(i),
                "split": split,
                TREATMENT_NAME: int(raw.iloc[i][TREATMENT_NAME]),
                SURVIVAL_TIME_COL: float(raw.iloc[i][SURVIVAL_TIME_COL]),
                EVENT_COL: int(raw.iloc[i][EVENT_COL]),
            })
    return pd.DataFrame(rows)


def split_indices(split_df: pd.DataFrame, split: str) -> np.ndarray:
    return split_df.loc[split_df["split"].eq(split), "row_index"].to_numpy(dtype=int)


def real_static_for_indices(raw: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    out = raw.iloc[indices].reset_index(drop=True).copy()
    out.insert(0, "patient_id", indices.astype(int))
    out["delta"] = out[EVENT_COL].astype(int)
    out["U"] = out[SURVIVAL_TIME_COL].astype(float)
    return out


def real_long_for_indices(long_obs: pd.DataFrame, indices: np.ndarray) -> pd.DataFrame:
    id_set = set(map(int, indices))
    df = long_obs[long_obs["subject_id"].astype(int).isin(id_set)].copy()
    df = df.rename(columns={"subject_id": "patient_id"})
    return df.reset_index(drop=True)


def normalize_synthetic_static(static_df: pd.DataFrame) -> pd.DataFrame:
    out = static_df.copy()
    if "patient_id" not in out and "subject_id" in out:
        out = out.rename(columns={"subject_id": "patient_id"})
    out["delta"] = pd.to_numeric(out.get(EVENT_COL, out.get("delta")), errors="coerce").fillna(0).astype(int)
    out["U"] = pd.to_numeric(out.get(SURVIVAL_TIME_COL, out.get("U")), errors="coerce").fillna(1.0).astype(float)
    return out


def generate_prior_replicate(
    model: PhaseSynModel,
    train_bundle: Any,
    n: int,
    treatment_ratio: float,
    rep_seed: int,
    time_grid: np.ndarray,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    rng = np.random.default_rng(rep_seed)
    n_treated = int(round(float(treatment_ratio) * n))
    treatment = np.concatenate([np.ones(n_treated, dtype=int), np.zeros(n - n_treated, dtype=int)])
    rng.shuffle(treatment)
    torch.manual_seed(rep_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rep_seed)
    static_df, long_df, tensors = generate_prior_cohort(
        model,
        train_bundle,
        n=n,
        treatment=torch.tensor(treatment, dtype=torch.long, device=device),
        time_grid=time_grid,
        device=device,
        deterministic=deterministic,
        return_tensors=True,
    )
    static_df = normalize_synthetic_static(static_df)
    static_df[TREATMENT_NAME] = treatment.astype(int)
    long_df[TREATMENT_NAME] = long_df[TREATMENT_NAME].astype(int)
    return static_df, long_df, tensors


def synthetic_long_format(static_df: pd.DataFrame, long_df: pd.DataFrame, replicate: int) -> pd.DataFrame:
    static = static_df.copy()
    static = static.rename(columns={name: f"baseline_{name}" for name in LONG_NAMES if name in static.columns})
    merge_cols = [c for c in static.columns if c not in set(LONG_NAMES)]
    out = long_df.merge(static[merge_cols], on=["patient_id", TREATMENT_NAME], how="left", suffixes=("", "_static"))
    out.insert(0, "replicate", int(replicate))
    out["delta"] = pd.to_numeric(out.get(EVENT_COL, out.get("delta")), errors="coerce").fillna(0).astype(int)
    out["U"] = pd.to_numeric(out.get(SURVIVAL_TIME_COL, out.get("U")), errors="coerce").fillna(1.0).astype(float)
    out["censoring_indicator"] = 1 - out["delta"]
    return out


def write_synthetic_replicates(
    model: PhaseSynModel,
    train_bundle: Any,
    output_dir: Path,
    n_replicates: int,
    n_subjects: int,
    treatment_ratio: float,
    time_grid: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], pd.DataFrame, list[str]]:
    static_reps: list[pd.DataFrame] = []
    long_reps: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    issues: list[str] = []
    for rep in range(n_replicates):
        static_df, long_df, _ = generate_prior_replicate(
            model, train_bundle, n_subjects, treatment_ratio, seed + 10000 + rep, time_grid, device
        )
        long_format = synthetic_long_format(static_df, long_df, rep)
        target = output_dir / "synthetic" / f"phasesyn_rep_{rep:03d}.parquet"
        fmt = "parquet"
        try:
            long_format.to_parquet(target, index=False)
        except Exception as exc:
            fmt = "csv"
            target = target.with_suffix(".csv")
            long_format.to_csv(target, index=False)
            issues.append(f"Replicate {rep:03d}: parquet unavailable, wrote CSV because {type(exc).__name__}: {exc}")
        static_reps.append(static_df)
        long_reps.append(long_df)
        index_rows.append({
            "replicate": rep,
            "path": str(target),
            "format": fmt,
            "n_subjects": int(len(static_df)),
            "n_longitudinal_rows": int(len(long_df)),
            "treatment_ratio": float(static_df[TREATMENT_NAME].mean()),
            "event_rate": float(static_df["delta"].mean()),
            "median_followup": float(static_df["U"].median()),
        })
    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(output_dir / "synthetic" / "synthetic_replicate_index.csv", index=False)
    return static_reps, long_reps, index_df, issues


def ecdf_ks(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float)[np.isfinite(x)])
    y = np.sort(np.asarray(y, dtype=float)[np.isfinite(y)])
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    grid = np.unique(np.concatenate([x, y]))
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x) - np.searchsorted(y, grid, side="right") / len(y))))


def standardized_mean_difference(x0: pd.Series, x1: pd.Series) -> float:
    a = pd.to_numeric(x0, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(x1, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(b) - np.mean(a)) / max(pooled, 1e-8))


def total_variation(real: pd.Series, syn: pd.Series) -> float:
    r = real.dropna().value_counts(normalize=True)
    s = syn.dropna().value_counts(normalize=True)
    cats = sorted(set(r.index) | set(s.index))
    return float(0.5 * sum(abs(float(r.get(c, 0.0)) - float(s.get(c, 0.0))) for c in cats))


def baseline_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mean_abs_errors = []
    smds = []
    ks_values = []
    prev_errors = []
    tv_values = []
    for col in STATIC_CONTINUOUS + LONG_NAMES:
        if col not in real or col not in syn:
            continue
        r = pd.to_numeric(real[col], errors="coerce")
        s = pd.to_numeric(syn[col], errors="coerce")
        mean_error = float(s.mean() - r.mean())
        sd_error = float(s.std(ddof=1) - r.std(ddof=1))
        ks = ecdf_ks(r.to_numpy(dtype=float), s.to_numpy(dtype=float))
        smd = mean_error / max(float(r.std(ddof=1)), 1e-8)
        metrics[f"baseline_{col}_mean_error"] = mean_error
        metrics[f"baseline_{col}_sd_error"] = sd_error
        metrics[f"baseline_{col}_smd"] = smd
        metrics[f"baseline_{col}_ks"] = ks
        mean_abs_errors.append(abs(mean_error))
        smds.append(abs(smd))
        if np.isfinite(ks):
            ks_values.append(ks)
    for col in STATIC_CATEGORICAL:
        if col not in real or col not in syn:
            continue
        tv = total_variation(real[col], syn[col])
        metrics[f"baseline_{col}_tv"] = tv
        tv_values.append(tv)
        cats = sorted(set(real[col].dropna().unique()) | set(syn[col].dropna().unique()))
        errs = []
        for cat in cats:
            err = float((syn[col] == cat).mean() - (real[col] == cat).mean())
            metrics[f"baseline_{col}_prevalence_error_{cat}"] = err
            errs.append(abs(err))
        if errs:
            prev_errors.extend(errs)
    metrics["baseline_continuous_mean_abs_mean_error"] = float(np.mean(mean_abs_errors)) if mean_abs_errors else float("nan")
    metrics["baseline_continuous_mean_abs_smd"] = float(np.mean(smds)) if smds else float("nan")
    metrics["baseline_continuous_mean_ks"] = float(np.mean(ks_values)) if ks_values else float("nan")
    metrics["baseline_categorical_mean_abs_prevalence_error"] = float(np.mean(prev_errors)) if prev_errors else float("nan")
    metrics["baseline_categorical_mean_tv"] = float(np.mean(tv_values)) if tv_values else float("nan")
    return metrics


def km_curve(times: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    ok = np.isfinite(times) & np.isfinite(events)
    times, events = times[ok], events[ok]
    if len(times) == 0:
        return np.ones_like(grid, dtype=float)
    surv = 1.0
    out = np.ones_like(grid, dtype=float)
    for t in np.sort(np.unique(times)):
        at_risk = np.sum(times >= t)
        n_events = np.sum((times == t) & (events > 0))
        if at_risk > 0:
            surv *= 1.0 - n_events / at_risk
        out[grid >= t] = surv
    return out


def km_iae(real: pd.DataFrame, syn: pd.DataFrame, arm: int | None = None, grid: np.ndarray | None = None) -> float:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 101)
    r = real if arm is None else real[real[TREATMENT_NAME].astype(int).eq(arm)]
    s = syn if arm is None else syn[syn[TREATMENT_NAME].astype(int).eq(arm)]
    if len(r) == 0 or len(s) == 0:
        return float("nan")
    rk = km_curve(r["U"].to_numpy(dtype=float), r["delta"].to_numpy(dtype=int), grid)
    sk = km_curve(s["U"].to_numpy(dtype=float), s["delta"].to_numpy(dtype=int), grid)
    return float(np.trapz(np.abs(rk - sk), grid) / max(grid[-1] - grid[0], 1e-8))


def survival_at_times(df: pd.DataFrame, times: list[float]) -> dict[str, float]:
    grid = np.asarray(times, dtype=float)
    surv = km_curve(df["U"].to_numpy(dtype=float), df["delta"].to_numpy(dtype=int), grid)
    return {f"survival_probability_t{t:g}": float(v) for t, v in zip(times, surv)}


def survival_fidelity(real: pd.DataFrame, syn: pd.DataFrame) -> dict[str, float]:
    metrics = {
        "event_rate_error": float(syn["delta"].mean() - real["delta"].mean()),
        "censoring_rate_error": float((1.0 - syn["delta"].mean()) - (1.0 - real["delta"].mean())),
        "median_followup_error": float(syn["U"].median() - real["U"].median()),
        "km_iae_all": km_iae(real, syn),
    }
    for arm in sorted(real[TREATMENT_NAME].dropna().astype(int).unique()):
        metrics[f"km_iae_arm_{arm}"] = km_iae(real, syn, arm=arm)
        r_arm = real[real[TREATMENT_NAME].astype(int).eq(arm)]
        s_arm = syn[syn[TREATMENT_NAME].astype(int).eq(arm)]
        for t in [0.25, 0.50, 0.75, 1.00]:
            rv = survival_at_times(r_arm, [t])[f"survival_probability_t{t:g}"]
            sv = survival_at_times(s_arm, [t])[f"survival_probability_t{t:g}"]
            metrics[f"survival_probability_error_arm_{arm}_t{t:g}"] = float(sv - rv)
    return metrics


def trajectory_summary(long_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    static = static_df.set_index("patient_id")
    for pid, g in long_df.groupby("patient_id"):
        row: dict[str, Any] = {"patient_id": int(pid)}
        if pid in static.index:
            st = static.loc[pid]
            row[TREATMENT_NAME] = int(st[TREATMENT_NAME])
            row["U"] = float(st["U"])
            row["delta"] = int(st["delta"])
            for c in STATIC_CONTINUOUS + STATIC_CATEGORICAL + LONG_NAMES:
                if c in st:
                    row[f"baseline_{c}"] = safe_float(st[c])
        times = pd.to_numeric(g["visit_time"], errors="coerce").to_numpy(dtype=float)
        for var in LONG_NAMES:
            y = pd.to_numeric(g[var], errors="coerce").to_numpy(dtype=float) if var in g else np.asarray([])
            ok = np.isfinite(times) & np.isfinite(y)
            if not ok.any():
                row[f"{var}_baseline_value"] = float("nan")
                row[f"{var}_final_value"] = float("nan")
                row[f"{var}_change"] = float("nan")
                row[f"{var}_slope"] = float("nan")
                row[f"{var}_auc"] = float("nan")
                continue
            order = np.argsort(times[ok])
            tx = times[ok][order]
            yy = y[ok][order]
            row[f"{var}_baseline_value"] = float(yy[0])
            row[f"{var}_final_value"] = float(yy[-1])
            row[f"{var}_change"] = float(yy[-1] - yy[0])
            row[f"{var}_slope"] = float(np.polyfit(tx, yy, 1)[0]) if len(tx) >= 2 and np.ptp(tx) > 1e-8 else 0.0
            row[f"{var}_auc"] = float(np.trapz(yy, tx)) if len(tx) >= 2 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def longitudinal_fidelity(real_long: pd.DataFrame, syn_long: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    mean_rmses: list[float] = []
    change_rmses: list[float] = []
    slope_errors: list[float] = []
    var_errors: list[float] = []
    for var in LONG_NAMES:
        if var not in real_long or var not in syn_long:
            continue
        for arm in sorted(real_long[TREATMENT_NAME].dropna().astype(int).unique()):
            r = real_long[real_long[TREATMENT_NAME].astype(int).eq(arm)]
            s = syn_long[syn_long[TREATMENT_NAME].astype(int).eq(arm)]
            r_mean = r.groupby("visit_index")[var].mean()
            s_mean = s.groupby("visit_index")[var].mean()
            visits = sorted(set(r_mean.index) & set(s_mean.index))
            if not visits:
                continue
            rmse = float(np.sqrt(np.nanmean([(s_mean.loc[v] - r_mean.loc[v]) ** 2 for v in visits])))
            metrics[f"longitudinal_{var}_arm_{arm}_mean_trajectory_rmse"] = rmse
            mean_rmses.append(rmse)
            r_base = r[r["visit_index"].eq(0)].set_index("patient_id")[var]
            s_base = s[s["visit_index"].eq(0)].set_index("patient_id")[var]
            r_tmp = r.join(r_base.rename("_base"), on="patient_id")
            s_tmp = s.join(s_base.rename("_base"), on="patient_id")
            r_ch = r_tmp.groupby("visit_index").apply(lambda x: (x[var] - x["_base"]).mean())
            s_ch = s_tmp.groupby("visit_index").apply(lambda x: (x[var] - x["_base"]).mean())
            visits = sorted(set(r_ch.index) & set(s_ch.index))
            ch_rmse = float(np.sqrt(np.nanmean([(s_ch.loc[v] - r_ch.loc[v]) ** 2 for v in visits]))) if visits else float("nan")
            metrics[f"longitudinal_{var}_arm_{arm}_change_from_baseline_rmse"] = ch_rmse
            if np.isfinite(ch_rmse):
                change_rmses.append(ch_rmse)
            r_var = r.groupby("visit_index")[var].var()
            s_var = s.groupby("visit_index")[var].var()
            visits = sorted(set(r_var.index) & set(s_var.index))
            ve = float(np.sqrt(np.nanmean([(s_var.loc[v] - r_var.loc[v]) ** 2 for v in visits]))) if visits else float("nan")
            metrics[f"longitudinal_{var}_arm_{arm}_variance_trajectory_error"] = ve
            if np.isfinite(ve):
                var_errors.append(ve)
            def slopes(df: pd.DataFrame) -> pd.Series:
                vals = []
                for _, g in df.groupby("patient_id"):
                    t = pd.to_numeric(g["visit_time"], errors="coerce").to_numpy(dtype=float)
                    y = pd.to_numeric(g[var], errors="coerce").to_numpy(dtype=float)
                    ok = np.isfinite(t) & np.isfinite(y)
                    vals.append(float(np.polyfit(t[ok], y[ok], 1)[0]) if ok.sum() >= 2 and np.ptp(t[ok]) > 1e-8 else float("nan"))
                return pd.Series(vals)
            slope_error = float(slopes(s).mean() - slopes(r).mean())
            metrics[f"longitudinal_{var}_arm_{arm}_slope_error"] = slope_error
            if np.isfinite(slope_error):
                slope_errors.append(abs(slope_error))
    metrics["longitudinal_mean_trajectory_rmse_mean"] = float(np.nanmean(mean_rmses)) if mean_rmses else float("nan")
    metrics["longitudinal_change_from_baseline_rmse_mean"] = float(np.nanmean(change_rmses)) if change_rmses else float("nan")
    metrics["longitudinal_abs_slope_error_mean"] = float(np.nanmean(slope_errors)) if slope_errors else float("nan")
    metrics["longitudinal_variance_trajectory_error_mean"] = float(np.nanmean(var_errors)) if var_errors else float("nan")
    return metrics


def joint_fidelity(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame) -> dict[str, float]:
    real_sum = trajectory_summary(real_long, real_static)
    syn_sum = trajectory_summary(syn_long, syn_static)
    feature_cols = [c for c in real_sum.columns if c not in {"patient_id"} and c in syn_sum.columns]
    real_x = real_sum[feature_cols].apply(pd.to_numeric, errors="coerce")
    syn_x = syn_sum[feature_cols].apply(pd.to_numeric, errors="coerce")
    keep = [c for c in feature_cols if real_x[c].notna().mean() > 0.7 and syn_x[c].notna().mean() > 0.7 and real_x[c].std(skipna=True) > 1e-8]
    real_x = real_x[keep].fillna(real_x[keep].median(numeric_only=True))
    syn_x = syn_x[keep].fillna(syn_x[keep].median(numeric_only=True))
    if len(keep) < 2 or len(real_x) < 5 or len(syn_x) < 5:
        return {"joint_correlation_frobenius_error": float("nan"), "joint_mmd_rbf": float("nan"), "joint_c2st_auc": float("nan")}
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
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=SEED)
        train_idx, test_idx = next(sss.split(x, y))
        clf = LogisticRegression(max_iter=1000, solver="lbfgs").fit(x[train_idx], y[train_idx])
        auc = roc_auc_score(y[test_idx], clf.predict_proba(x[test_idx])[:, 1])
    except Exception:
        try:
            clf = RandomForestClassifier(n_estimators=100, random_state=SEED).fit(x, y)
            auc = roc_auc_score(y, clf.predict_proba(x)[:, 1])
        except Exception:
            auc = float("nan")
    return {"joint_correlation_frobenius_error": frob, "joint_mmd_rbf": mmd, "joint_c2st_auc": float(auc)}


def fidelity_metrics(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame) -> dict[str, float]:
    syn_static = normalize_synthetic_static(syn_static)
    real_static = normalize_synthetic_static(real_static)
    row: dict[str, float] = {}
    row.update(baseline_fidelity(real_static, syn_static))
    row.update(longitudinal_fidelity(real_long, syn_long))
    row.update(survival_fidelity(real_static, syn_static))
    row.update(joint_fidelity(real_static, real_long, syn_static, syn_long))
    return row


def benchmark_fidelity_metrics(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame) -> dict[str, float]:
    syn_static = normalize_synthetic_static(syn_static)
    real_static = normalize_synthetic_static(real_static)
    row: dict[str, float] = {}
    row.update(baseline_fidelity(real_static, syn_static))
    row.update(longitudinal_fidelity(real_long, syn_long))
    row.update(survival_fidelity(real_static, syn_static))
    row["joint_correlation_frobenius_error"] = float("nan")
    row["joint_mmd_rbf"] = float("nan")
    row["joint_c2st_auc"] = float("nan")
    return row


def summarize_replicate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col == "replicate" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        x = df[col].dropna()
        if x.empty:
            continue
        rows.append({
            "metric": col,
            "mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "median": float(x.median()),
            "q025": float(x.quantile(0.025)),
            "q975": float(x.quantile(0.975)),
            "min": float(x.min()),
            "max": float(x.max()),
            "n": int(len(x)),
        })
    return pd.DataFrame(rows)


def fit_cox_treatment(static_df: pd.DataFrame) -> dict[str, float]:
    df = normalize_synthetic_static(static_df)
    if CoxPHFitter is None or len(df[TREATMENT_NAME].unique()) < 2 or df["delta"].sum() < 3:
        return {"cox_log_hr": float("nan"), "cox_hr": float("nan"), "cox_p": float("nan")}
    try:
        cdf = df[["U", "delta", TREATMENT_NAME]].dropna().copy()
        cdf["U"] = cdf["U"].clip(lower=1e-5)
        cph = CoxPHFitter()
        cph.fit(cdf, duration_col="U", event_col="delta", formula=TREATMENT_NAME)
        log_hr = float(cph.params_[TREATMENT_NAME])
        p = float(cph.summary.loc[TREATMENT_NAME, "p"])
        return {"cox_log_hr": log_hr, "cox_hr": float(np.exp(log_hr)), "cox_p": p}
    except Exception:
        return {"cox_log_hr": float("nan"), "cox_hr": float("nan"), "cox_p": float("nan")}


def logrank_pvalue(static_df: pd.DataFrame) -> float:
    df = normalize_synthetic_static(static_df)
    if logrank_test is None or len(df[TREATMENT_NAME].unique()) < 2:
        return float("nan")
    try:
        a0 = df[df[TREATMENT_NAME].astype(int).eq(0)]
        a1 = df[df[TREATMENT_NAME].astype(int).eq(1)]
        return float(logrank_test(a0["U"], a1["U"], a0["delta"], a1["delta"]).p_value)
    except Exception:
        return float("nan")


def rmst(df: pd.DataFrame, tau: float) -> float:
    grid = np.linspace(0.0, tau, 256)
    return float(np.trapz(km_curve(df["U"].to_numpy(dtype=float), df["delta"].to_numpy(dtype=int), grid), grid))


def survival_estimands(static_df: pd.DataFrame, tau: float) -> dict[str, float]:
    df = normalize_synthetic_static(static_df)
    out = fit_cox_treatment(df)
    out["logrank_p"] = logrank_pvalue(df)
    if len(df[TREATMENT_NAME].unique()) >= 2:
        a0 = df[df[TREATMENT_NAME].astype(int).eq(0)]
        a1 = df[df[TREATMENT_NAME].astype(int).eq(1)]
        out["rmst_diff"] = rmst(a1, tau) - rmst(a0, tau)
        out["event_rate_difference"] = float(a1["delta"].mean() - a0["delta"].mean())
    else:
        out["rmst_diff"] = float("nan")
        out["event_rate_difference"] = float("nan")
    out["tau"] = float(tau)
    return out


def longitudinal_estimands(static_df: pd.DataFrame, long_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    static = normalize_synthetic_static(static_df)[["patient_id", TREATMENT_NAME] + [c for c in LONG_NAMES if c in static_df]]
    df0 = long_df.merge(static, on=["patient_id", TREATMENT_NAME], how="left", suffixes=("", "_base_static"))
    for var in LONG_NAMES:
        if var not in df0:
            continue
        df = df0[["patient_id", TREATMENT_NAME, "visit_time", "visit_index", var]].copy()
        base = df[df["visit_index"].eq(0)].set_index("patient_id")[var]
        df = df.join(base.rename("baseline_value"), on="patient_id")
        df = df.dropna(subset=[var, "visit_time", TREATMENT_NAME, "baseline_value"])
        if df.empty:
            continue
        method = "OLS_cluster_subject"
        coefs = {"treatment_main": float("nan"), "treatment_time_interaction": float("nan")}
        pvals = {"treatment_main_p": float("nan"), "treatment_time_interaction_p": float("nan")}
        try:
            if sm is None:
                raise RuntimeError("statsmodels unavailable")
            x = pd.DataFrame({
                "const": 1.0,
                "A": df[TREATMENT_NAME].astype(float),
                "time": df["visit_time"].astype(float),
                "A_time": df[TREATMENT_NAME].astype(float) * df["visit_time"].astype(float),
                "baseline_value": df["baseline_value"].astype(float),
            })
            model = sm.OLS(df[var].astype(float), x).fit(cov_type="cluster", cov_kwds={"groups": df["patient_id"].astype(int)})
            coefs["treatment_main"] = float(model.params.get("A", np.nan))
            coefs["treatment_time_interaction"] = float(model.params.get("A_time", np.nan))
            pvals["treatment_main_p"] = float(model.pvalues.get("A", np.nan))
            pvals["treatment_time_interaction_p"] = float(model.pvalues.get("A_time", np.nan))
        except Exception:
            method = "mean_contrast_fallback"
        final_visit = int(df["visit_index"].max())
        final = df[df["visit_index"].eq(final_visit)].copy()
        final_contrast = float(
            final[final[TREATMENT_NAME].astype(int).eq(1)][var].mean()
            - final[final[TREATMENT_NAME].astype(int).eq(0)][var].mean()
        ) if len(final[TREATMENT_NAME].unique()) >= 2 else float("nan")
        ch = final.assign(change=final[var] - final["baseline_value"])
        change_contrast = float(
            ch[ch[TREATMENT_NAME].astype(int).eq(1)]["change"].mean()
            - ch[ch[TREATMENT_NAME].astype(int).eq(0)]["change"].mean()
        ) if len(ch[TREATMENT_NAME].unique()) >= 2 else float("nan")
        rows.append({
            "endpoint": var,
            "method": method,
            **coefs,
            **pvals,
            "final_visit_treatment_contrast": final_contrast,
            "change_from_baseline_treatment_contrast": change_contrast,
        })
    return rows


def estimand_rows(static_df: pd.DataFrame, long_df: pd.DataFrame, tau: float, replicate: int | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    surv = survival_estimands(static_df, tau)
    for k, v in surv.items():
        if k == "tau":
            continue
        rows.append({"replicate": replicate, "domain": "survival", "endpoint": "survival", "estimand": k, "estimate": v, "p_value": surv.get("cox_p") if k in {"cox_log_hr", "cox_hr"} else (surv.get("logrank_p") if k == "logrank_p" else np.nan), "tau": tau})
    for item in longitudinal_estimands(static_df, long_df):
        endpoint = item.pop("endpoint")
        method = item.pop("method")
        for k, v in item.items():
            if k.endswith("_p"):
                continue
            p = item.get(f"{k}_p", np.nan)
            rows.append({"replicate": replicate, "domain": "longitudinal", "endpoint": endpoint, "estimand": k, "estimate": v, "p_value": p, "method": method, "tau": np.nan})
    return rows


def summarize_estimands(real_rows: pd.DataFrame, syn_rows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["domain", "endpoint", "estimand"]
    for key, real_group in real_rows.groupby(keys):
        real_est = float(real_group["estimate"].iloc[0])
        sg = syn_rows
        for col, val in zip(keys, key):
            sg = sg[sg[col].eq(val)]
        estimates = pd.to_numeric(sg["estimate"], errors="coerce").dropna()
        if estimates.empty:
            continue
        sign_real = np.sign(real_est)
        real_sig = bool((pd.to_numeric(real_group["p_value"], errors="coerce").iloc[0] < 0.05)) if np.isfinite(pd.to_numeric(real_group["p_value"], errors="coerce").iloc[0]) else False
        pvals = pd.to_numeric(sg["p_value"], errors="coerce")
        rows.append({
            "domain": key[0],
            "endpoint": key[1],
            "estimand": key[2],
            "real_estimate": real_est,
            "synthetic_median": float(estimates.median()),
            "synthetic_q025": float(estimates.quantile(0.025)),
            "synthetic_q975": float(estimates.quantile(0.975)),
            "bias_vs_real": float(estimates.mean() - real_est),
            "rmse_vs_real": float(np.sqrt(np.mean((estimates - real_est) ** 2))),
            "sign_agreement": float(np.mean(np.sign(estimates) == sign_real)) if sign_real != 0 else float("nan"),
            "significance_agreement": float(np.mean((pvals < 0.05).fillna(False) == real_sig)) if pvals.notna().any() else float("nan"),
            "n": int(len(estimates)),
        })
    return pd.DataFrame(rows)


def baseline_smd_by_arm(static_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    df = normalize_synthetic_static(static_df)
    for col in STATIC_CONTINUOUS + LONG_NAMES:
        rows.append({"variable": col, "smd": standardized_mean_difference(df[df[TREATMENT_NAME].astype(int).eq(0)][col], df[df[TREATMENT_NAME].astype(int).eq(1)][col])})
    for col in STATIC_CATEGORICAL:
        rows.append({"variable": col, "smd": standardized_mean_difference(df[df[TREATMENT_NAME].astype(int).eq(0)][col], df[df[TREATMENT_NAME].astype(int).eq(1)][col])})
    return pd.DataFrame(rows)


def treatment_propensity_auc(static_df: pd.DataFrame, latent: np.ndarray | None = None) -> tuple[float, float]:
    df = normalize_synthetic_static(static_df)
    x = df[[c for c in BASELINE_COLS if c in df]].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(numeric_only=True))
    y = df[TREATMENT_NAME].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        base_auc = float("nan")
    else:
        try:
            clf = LogisticRegression(max_iter=1000).fit(x, y)
            base_auc = float(roc_auc_score(y, clf.predict_proba(x)[:, 1]))
        except Exception:
            base_auc = float("nan")
    latent_auc = float("nan")
    if latent is not None and len(np.unique(y)) >= 2:
        try:
            clf = LogisticRegression(max_iter=1000).fit(latent, y)
            latent_auc = float(roc_auc_score(y, clf.predict_proba(latent)[:, 1]))
        except Exception:
            latent_auc = float("nan")
    return base_auc, latent_auc


def intervention_tests(
    model: PhaseSynModel,
    train_bundle: Any,
    output_dir: Path,
    n_subjects: int,
    original_ratio: float,
    time_grid: np.ndarray,
    device: torch.device,
    tau: float,
    seed: int,
) -> pd.DataFrame:
    scenarios = {
        "original_ratio": original_ratio,
        "one_to_one": 0.5,
        "two_to_one_treatment_control": 2.0 / 3.0,
        "control_only": 0.0,
        "treatment_only": 1.0,
    }
    rows: list[dict[str, Any]] = []
    figure_long: dict[str, pd.DataFrame] = {}
    figure_static: dict[str, pd.DataFrame] = {}
    for i, (name, ratio) in enumerate(scenarios.items()):
        static_df, long_df, tensors = generate_prior_replicate(model, train_bundle, n_subjects, ratio, seed + 30000 + i, time_grid, device)
        figure_long[name] = long_df
        figure_static[name] = static_df
        latent = tensors.get("z")
        latent_np = latent.detach().cpu().numpy() if isinstance(latent, torch.Tensor) else None
        auc, latent_auc = treatment_propensity_auc(static_df, latent_np)
        smd_df = baseline_smd_by_arm(static_df)
        surv = survival_estimands(static_df, tau)
        row = {
            "scenario": name,
            "target_treatment_ratio": float(ratio),
            "observed_treatment_ratio": float(static_df[TREATMENT_NAME].mean()),
            "baseline_smd_mean_abs": float(smd_df["smd"].abs().mean()) if smd_df["smd"].notna().any() else float("nan"),
            "baseline_smd_max_abs": float(smd_df["smd"].abs().max()) if smd_df["smd"].notna().any() else float("nan"),
            "treatment_propensity_auc_baseline": auc,
            "treatment_propensity_auc_latent_z": latent_auc,
            "cox_hr": surv.get("cox_hr"),
            "cox_log_hr": surv.get("cox_log_hr"),
            "rmst_diff": surv.get("rmst_diff"),
            "event_rate_difference": surv.get("event_rate_difference"),
        }
        rows.append(row)
        smd_df.insert(0, "scenario", name)
        smd_df.to_csv(output_dir / "metrics" / f"randomization_smd_{name}.csv", index=False)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "metrics" / "randomization_intervention_test.csv", index=False)
    out.to_csv(output_dir / "tables" / "table4_randomization_test.csv", index=False)
    plot_intervention_figures(figure_static, figure_long, output_dir / "figures")
    return out


def coupling_metrics(static_df: pd.DataFrame, long_df: pd.DataFrame, threshold: float, landmark_time: float, replicate: int | str) -> dict[str, Any]:
    static = normalize_synthetic_static(static_df)
    base = long_df[long_df["visit_index"].eq(0)].set_index("patient_id")["L1"]
    early = long_df[long_df["visit_time"].ge(landmark_time - 1e-6)].sort_values("visit_time").groupby("patient_id").first()["L1"]
    df = static.set_index("patient_id").join(base.rename("baseline_L1")).join(early.rename("early_L1"))
    df["early_improvement"] = df["baseline_L1"] - df["early_L1"]
    df["early_response"] = (df["early_improvement"] > threshold).astype(int)
    slope_rows = []
    for pid, g in long_df.groupby("patient_id"):
        t = pd.to_numeric(g["visit_time"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(g["L1"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(t) & np.isfinite(y) & (t <= max(landmark_time, np.nanmax(t)))
        slope_rows.append({"patient_id": pid, "early_slope": float(np.polyfit(t[ok], y[ok], 1)[0]) if ok.sum() >= 2 and np.ptp(t[ok]) > 1e-8 else np.nan})
    df = df.join(pd.DataFrame(slope_rows).set_index("patient_id"))
    out = {"replicate": replicate, "early_response_coef": np.nan, "early_response_hr": np.nan, "early_slope_coef": np.nan, "responder_nonresponder_hr": np.nan, "early_slope_hr": np.nan}
    if CoxPHFitter is not None:
        try:
            lm = df[df["U"].gt(landmark_time)].dropna(subset=["baseline_L1", "early_response", "U", "delta", TREATMENT_NAME]).copy()
            if len(lm) > 10 and lm["early_response"].nunique() == 2:
                lm["duration_after_landmark"] = (lm["U"] - landmark_time).clip(lower=1e-5)
                cph = CoxPHFitter()
                cph.fit(lm[["duration_after_landmark", "delta", TREATMENT_NAME, "baseline_L1", "early_response"]], duration_col="duration_after_landmark", event_col="delta")
                coef = float(cph.params_["early_response"])
                out["early_response_coef"] = coef
                out["early_response_hr"] = float(np.exp(coef))
                out["responder_nonresponder_hr"] = float(np.exp(coef))
        except Exception:
            pass
        try:
            sl = df.dropna(subset=["baseline_L1", "early_slope", "U", "delta", TREATMENT_NAME]).copy()
            if len(sl) > 10:
                sl["U"] = sl["U"].clip(lower=1e-5)
                cph = CoxPHFitter()
                cph.fit(sl[["U", "delta", TREATMENT_NAME, "baseline_L1", "early_slope"]], duration_col="U", event_col="delta")
                coef = float(cph.params_["early_slope"])
                out["early_slope_coef"] = coef
                out["early_slope_hr"] = float(np.exp(coef))
        except Exception:
            pass
    return out


def coupling_summary(real_static: pd.DataFrame, real_long: pd.DataFrame, static_reps: list[pd.DataFrame], long_reps: list[pd.DataFrame], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_base = real_long[real_long["visit_index"].eq(0)].set_index("patient_id")["L1"]
    post = real_long[real_long["visit_index"].gt(0)].sort_values("visit_time")
    landmark_time = float(post["visit_time"].dropna().quantile(0.10)) if len(post) else 0.08
    early = post[post["visit_time"].ge(landmark_time - 1e-6)].groupby("patient_id").first()["L1"]
    threshold = float((real_base - early).dropna().median())
    real = pd.DataFrame([coupling_metrics(real_static, real_long, threshold, landmark_time, "real")])
    rows = [coupling_metrics(s, l, threshold, landmark_time, i) for i, (s, l) in enumerate(zip(static_reps, long_reps))]
    by_rep = pd.DataFrame(rows)
    by_rep.to_csv(output_dir / "metrics" / "longitudinal_survival_coupling_by_replicate.csv", index=False)
    summary_rows = []
    for metric in ["early_response_coef", "early_response_hr", "early_slope_coef", "early_slope_hr", "responder_nonresponder_hr"]:
        vals = pd.to_numeric(by_rep[metric], errors="coerce").dropna()
        real_val = float(real[metric].iloc[0])
        if vals.empty:
            continue
        summary_rows.append({
            "metric": metric,
            "real_estimate": real_val,
            "synthetic_median": float(vals.median()),
            "synthetic_q025": float(vals.quantile(0.025)),
            "synthetic_q975": float(vals.quantile(0.975)),
            "bias_vs_real": float(vals.mean() - real_val),
            "rmse_vs_real": float(np.sqrt(np.mean((vals - real_val) ** 2))),
            "landmark_time": landmark_time,
            "response_threshold": threshold,
            "n": int(len(vals)),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "metrics" / "longitudinal_survival_coupling_summary.csv", index=False)
    summary.to_csv(output_dir / "tables" / "table5_longitudinal_survival_coupling.csv", index=False)
    return by_rep, summary


def trial_design_simulation(
    model: PhaseSynModel,
    train_bundle: Any,
    output_dir: Path,
    sample_sizes: list[int],
    reps: int,
    time_grid: np.ndarray,
    device: torch.device,
    seed: int,
    alpha: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for n in sample_sizes:
        for r in range(reps):
            static_df, long_df, _ = generate_prior_replicate(model, train_bundle, n, 0.5, seed + 50000 + n * 10 + r, time_grid, device)
            surv = survival_estimands(static_df, tau=0.80)
            long_est = longitudinal_estimands(static_df, long_df)
            primary_long = next((x for x in long_est if x["endpoint"] == "L1"), long_est[0] if long_est else {})
            p_long = safe_float(primary_long.get("treatment_time_interaction_p", np.nan))
            p_surv = safe_float(surv.get("logrank_p", np.nan))
            rows.append({
                "n": int(n),
                "replicate": int(r),
                "survival_p": p_surv,
                "survival_success": bool(np.isfinite(p_surv) and p_surv < alpha),
                "longitudinal_p": p_long,
                "longitudinal_success": bool(np.isfinite(p_long) and p_long < alpha),
                "trial_success": bool((np.isfinite(p_surv) and p_surv < alpha) or (np.isfinite(p_long) and p_long < alpha)),
                "events": int(static_df["delta"].sum()),
                "cox_hr": surv.get("cox_hr"),
                "cox_log_hr": surv.get("cox_log_hr"),
                "censoring_rate": float(1.0 - static_df["delta"].mean()),
                "longitudinal_treatment_time_interaction": safe_float(primary_long.get("treatment_time_interaction", np.nan)),
            })
    by_rep = pd.DataFrame(rows)
    by_rep.to_csv(output_dir / "metrics" / "trial_design_power_by_replicate.csv", index=False)
    summary_rows = []
    for n, g in by_rep.groupby("n"):
        summary_rows.append({
            "n": int(n),
            "survival_power": float(g["survival_success"].mean()),
            "longitudinal_power": float(g["longitudinal_success"].mean()),
            "probability_trial_success": float(g["trial_success"].mean()),
            "expected_events": float(g["events"].mean()),
            "median_hr_estimate": float(pd.to_numeric(g["cox_hr"], errors="coerce").median()),
            "hr_q025": float(pd.to_numeric(g["cox_hr"], errors="coerce").quantile(0.025)),
            "hr_q975": float(pd.to_numeric(g["cox_hr"], errors="coerce").quantile(0.975)),
            "expected_censoring_rate": float(g["censoring_rate"].mean()),
            "reps": int(len(g)),
        })
    summary = pd.DataFrame(summary_rows)
    eligible = summary[summary["survival_power"].ge(0.80)]
    required = int(eligible["n"].min()) if not eligible.empty else None
    summary["required_sample_size_for_80pct_survival_power"] = required if required is not None else np.nan
    summary.to_csv(output_dir / "metrics" / "trial_design_power_summary.csv", index=False)
    summary.to_csv(output_dir / "tables" / "table6_trial_design_utility.csv", index=False)
    return by_rep, summary


CONDITIONING_COLS = STATIC_CONTINUOUS + STATIC_CATEGORICAL + LONG_NAMES


def _numeric_fill_value(series: pd.Series, categorical: bool = False) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0
    if categorical:
        return float(values.mode().iloc[0])
    return float(values.median())


def fit_conditioning_space(reference: pd.DataFrame) -> dict[str, Any]:
    cols = [c for c in CONDITIONING_COLS if c in reference]
    fills = {
        c: _numeric_fill_value(reference[c], categorical=c in STATIC_CATEGORICAL)
        for c in cols
    }
    x = condition_matrix(reference, cols, fills)
    scaler = StandardScaler().fit(x)
    return {"columns": cols, "fills": fills, "scaler": scaler, "reference_scaled": scaler.transform(x)}


def condition_matrix(df: pd.DataFrame, cols: list[str], fills: dict[str, float]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        out[col] = pd.to_numeric(df.get(col, pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(fills.get(col, 0.0))
    return out


def transform_conditioning(df: pd.DataFrame, space: dict[str, Any]) -> np.ndarray:
    x = condition_matrix(df, space["columns"], space["fills"])
    return space["scaler"].transform(x)


def sample_conditioned_positions(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    rng: np.random.Generator,
    k: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    space = fit_conditioning_space(reference)
    ref_x = space["reference_scaled"]
    target_x = transform_conditioning(target, space)
    ref_arm = pd.to_numeric(reference[TREATMENT_NAME], errors="coerce").fillna(0).astype(int).to_numpy()
    target_arm = pd.to_numeric(target[TREATMENT_NAME], errors="coerce").fillna(0).astype(int).to_numpy()
    chosen: list[int] = []
    distances: list[float] = []
    all_pos = np.arange(len(reference), dtype=int)
    for i, arm in enumerate(target_arm):
        candidates = all_pos[ref_arm == arm]
        if len(candidates) == 0:
            candidates = all_pos
        d = np.linalg.norm(ref_x[candidates] - target_x[i], axis=1)
        order = np.argsort(d)[: max(1, min(k, len(d)))]
        top = candidates[order]
        top_d = d[order]
        weights = 1.0 / np.maximum(top_d, 1e-6)
        weights = weights / weights.sum() if np.isfinite(weights).all() and weights.sum() > 0 else None
        pick = int(rng.choice(top, p=weights))
        chosen.append(pick)
        distances.append(float(d[np.where(candidates == pick)[0][0]]))
    return np.asarray(chosen, dtype=int), np.asarray(distances, dtype=float)


def prepare_conditioned_neighbors(reference: pd.DataFrame, target: pd.DataFrame, k: int = 25) -> dict[str, Any]:
    space = fit_conditioning_space(reference)
    ref_x = space["reference_scaled"]
    target_x = transform_conditioning(target, space)
    ref_arm = pd.to_numeric(reference[TREATMENT_NAME], errors="coerce").fillna(0).astype(int).to_numpy()
    target_arm = pd.to_numeric(target[TREATMENT_NAME], errors="coerce").fillna(0).astype(int).to_numpy()
    all_pos = np.arange(len(reference), dtype=int)
    top_positions: list[np.ndarray] = []
    top_weights: list[np.ndarray | None] = []
    top_distances: list[np.ndarray] = []
    for i, arm in enumerate(target_arm):
        candidates = all_pos[ref_arm == arm]
        if len(candidates) == 0:
            candidates = all_pos
        d = np.linalg.norm(ref_x[candidates] - target_x[i], axis=1)
        order = np.argsort(d)[: max(1, min(k, len(d)))]
        top = candidates[order]
        top_d = d[order]
        weights = 1.0 / np.maximum(top_d, 1e-6)
        weights = weights / weights.sum() if np.isfinite(weights).all() and weights.sum() > 0 else None
        top_positions.append(top)
        top_weights.append(weights)
        top_distances.append(top_d)
    return {"positions": top_positions, "weights": top_weights, "distances": top_distances, "neighbors": int(k)}


def sample_precomputed_conditioned_positions(neighbor_info: dict[str, Any], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    chosen: list[int] = []
    distances: list[float] = []
    for top, weights, top_d in zip(neighbor_info["positions"], neighbor_info["weights"], neighbor_info["distances"]):
        local_idx = int(rng.choice(np.arange(len(top)), p=weights))
        chosen.append(int(top[local_idx]))
        distances.append(float(top_d[local_idx]))
    return np.asarray(chosen, dtype=int), np.asarray(distances, dtype=float)


def target_static_template(target_static: pd.DataFrame) -> pd.DataFrame:
    static = target_static.reset_index(drop=True).copy()
    static["patient_id"] = np.arange(len(static), dtype=int)
    static[TREATMENT_NAME] = pd.to_numeric(static[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int)
    return static


def _sorted_source_long(g: pd.DataFrame, var: str) -> tuple[np.ndarray, np.ndarray]:
    if g.empty or var not in g:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    t = pd.to_numeric(g["visit_time"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(g[var], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if not ok.any():
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    order = np.argsort(t[ok])
    return t[ok][order], y[ok][order]


def conditional_bootstrap_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    time_grid: np.ndarray,
    reps: int,
    seed: int,
    neighbors: int = 25,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], dict[str, float]]:
    rng = np.random.default_rng(seed)
    statics, longs = [], []
    train_by_pid = {int(pid): g.copy() for pid, g in train_long.groupby("patient_id")}
    all_distances: list[float] = []
    neighbor_info = prepare_conditioned_neighbors(train_static, target_static, neighbors)
    for r in range(reps):
        source_pos, distances = sample_precomputed_conditioned_positions(neighbor_info, rng)
        all_distances.extend(distances.tolist())
        source_static = train_static.iloc[source_pos].reset_index(drop=True)
        static = target_static_template(target_static)
        static["U"] = pd.to_numeric(source_static["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0).to_numpy(dtype=float)
        static[SURVIVAL_TIME_COL] = static["U"].astype(float)
        static["delta"] = pd.to_numeric(source_static["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
        static[EVENT_COL] = static["delta"].astype(int)
        long_rows = []
        for new_id, (target_row, source_row) in enumerate(zip(static.to_dict("records"), source_static.to_dict("records"))):
            source_pid = int(source_row["patient_id"])
            source_long = train_by_pid.get(source_pid, pd.DataFrame())
            for visit, t in enumerate(time_grid):
                if float(t) > float(target_row["U"]) + 1e-8:
                    continue
                row = {
                    "patient_id": int(new_id),
                    TREATMENT_NAME: int(target_row[TREATMENT_NAME]),
                    "visit_index": int(visit),
                    "visit_time": float(t),
                }
                for var in LONG_NAMES:
                    target_base = safe_float(target_row.get(var), 0.0)
                    source_base = safe_float(source_row.get(var), target_base)
                    tx, yy = _sorted_source_long(source_long, var)
                    if len(tx) > 0:
                        source_value = float(np.interp(float(t), tx, yy))
                        change = source_value - source_base
                    else:
                        change = 0.0
                    row[var] = float(target_base + change)
                long_rows.append(row)
        statics.append(static.reset_index(drop=True))
        longs.append(pd.DataFrame(long_rows))
    diagnostics = {
        "median_conditioning_distance": float(np.median(all_distances)) if all_distances else float("nan"),
        "mean_conditioning_distance": float(np.mean(all_distances)) if all_distances else float("nan"),
        "neighbors": int(neighbors),
    }
    return statics, longs, diagnostics


def _longitudinal_design(static_part: pd.DataFrame, visit_time: pd.Series | np.ndarray, cols: list[str], fills: dict[str, float]) -> pd.DataFrame:
    n = len(static_part)
    vt = pd.Series(visit_time, index=static_part.index, dtype=float)
    treatment = pd.to_numeric(static_part[TREATMENT_NAME], errors="coerce").fillna(0).astype(float)
    x = pd.DataFrame({
        "const": np.ones(n, dtype=float),
        TREATMENT_NAME: treatment.to_numpy(dtype=float),
        "visit_time": vt.to_numpy(dtype=float),
        "A_time": treatment.to_numpy(dtype=float) * vt.to_numpy(dtype=float),
    }, index=static_part.index)
    base = condition_matrix(static_part, cols, fills)
    for col in cols:
        x[f"base_{col}"] = base[col].to_numpy(dtype=float)
    return x


def fit_longitudinal_change_models(train_static: pd.DataFrame, train_long: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, float], list[str], dict[str, float], list[str]]:
    issues: list[str] = []
    cols = [c for c in CONDITIONING_COLS if c in train_static]
    fills = {c: _numeric_fill_value(train_static[c], categorical=c in STATIC_CATEGORICAL) for c in cols}
    static_base = train_static[["patient_id", TREATMENT_NAME, *cols]].copy()
    rename = {c: f"base_{c}" for c in cols}
    merged = train_long.merge(static_base.rename(columns=rename), on="patient_id", how="left", suffixes=("", "_static"))
    if f"{TREATMENT_NAME}_static" in merged:
        merged[TREATMENT_NAME] = pd.to_numeric(merged[TREATMENT_NAME], errors="coerce").fillna(merged[f"{TREATMENT_NAME}_static"])
    static_for_design = pd.DataFrame(index=merged.index)
    static_for_design[TREATMENT_NAME] = pd.to_numeric(merged[TREATMENT_NAME], errors="coerce").fillna(0).astype(float)
    for col in cols:
        static_for_design[col] = pd.to_numeric(merged.get(f"base_{col}"), errors="coerce").fillna(fills[col])
    x_all = _longitudinal_design(static_for_design, pd.to_numeric(merged["visit_time"], errors="coerce").fillna(0.0), cols, fills)
    coefs: dict[str, np.ndarray] = {}
    residual_sd: dict[str, float] = {}
    for var in LONG_NAMES:
        try:
            y_raw = pd.to_numeric(merged[var], errors="coerce")
            base = pd.to_numeric(merged.get(f"base_{var}"), errors="coerce").fillna(fills.get(var, 0.0))
            y = y_raw - base
            ok = y.notna() & np.isfinite(pd.to_numeric(merged["visit_time"], errors="coerce"))
            x = x_all.loc[ok].to_numpy(dtype=float)
            yy = y.loc[ok].to_numpy(dtype=float)
            beta, *_ = np.linalg.lstsq(x, yy, rcond=None)
            resid = yy - x @ beta
            coefs[var] = beta
            residual_sd[var] = float(np.std(resid, ddof=min(x.shape[1], max(len(yy) - 1, 1)))) if len(yy) > x.shape[1] else float(np.std(resid))
        except Exception as exc:
            issues.append(f"Conditional classical longitudinal fit failed for {var}: {type(exc).__name__}: {exc}")
            coefs[var] = np.zeros(x_all.shape[1], dtype=float)
            residual_sd[var] = float(pd.to_numeric(train_long.get(var), errors="coerce").std(skipna=True))
    return coefs, residual_sd, cols, fills, issues


def _survival_feature_frame(static: pd.DataFrame, cols: list[str], fills: dict[str, float]) -> pd.DataFrame:
    out = condition_matrix(static, cols, fills)
    out.insert(0, TREATMENT_NAME, pd.to_numeric(static[TREATMENT_NAME], errors="coerce").fillna(0).astype(float).to_numpy())
    return out


def fit_cox_time_sampler(train_static: pd.DataFrame, cols: list[str], fills: dict[str, float], event: pd.Series, label: str) -> tuple[Any | None, str | None]:
    if CoxPHFitter is None:
        return None, f"Conditional classical {label} Cox unavailable: lifelines is not installed."
    event = pd.to_numeric(event, errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    if event.sum() < 5 or event.nunique() < 2:
        return None, f"Conditional classical {label} Cox unavailable: insufficient events."
    try:
        x = _survival_feature_frame(train_static, cols, fills)
        df = x.copy()
        df["U"] = pd.to_numeric(train_static["U"], errors="coerce").fillna(1.0).clip(1e-5, 1.0)
        df["_event"] = event.to_numpy(dtype=int)
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(df[["U", "_event", *x.columns]], duration_col="U", event_col="_event")
        return cph, None
    except Exception as exc:
        return None, f"Conditional classical {label} Cox fit failed: {type(exc).__name__}: {exc}"


def sample_cox_time(cph: Any | None, feature_row: pd.Series, rng: np.random.Generator) -> float:
    if cph is None:
        return float("inf")
    try:
        h0 = cph.baseline_cumulative_hazard_.iloc[:, 0]
        h0 = h0[np.isfinite(h0.to_numpy(dtype=float))]
        if h0.empty or float(h0.iloc[-1]) <= 0:
            return float("inf")
        row = pd.DataFrame([feature_row.reindex(cph.params_.index).astype(float).to_dict()])
        risk = float(cph.predict_partial_hazard(row).iloc[0])
        if not np.isfinite(risk) or risk <= 0:
            return float("inf")
        threshold = float(rng.exponential(1.0) / risk)
        hazards = h0.to_numpy(dtype=float)
        times = h0.index.to_numpy(dtype=float)
        if threshold > hazards[-1]:
            return float("inf")
        idx = int(np.searchsorted(hazards, threshold, side="left"))
        if idx <= 0:
            return float(times[0])
        h_lo, h_hi = hazards[idx - 1], hazards[idx]
        t_lo, t_hi = times[idx - 1], times[idx]
        if h_hi <= h_lo:
            return float(t_hi)
        return float(t_lo + (threshold - h_lo) * (t_hi - t_lo) / (h_hi - h_lo))
    except Exception:
        return float("inf")


def classical_replicates(
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    reps: int,
    seed: int,
    time_grid: np.ndarray,
    neighbors: int = 25,
    use_cox_survival_model: bool = False,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[str], dict[str, float]]:
    rng = np.random.default_rng(seed)
    issues: list[str] = []
    statics, longs = [], []
    coefs, residual_sd, cols, fills, long_issues = fit_longitudinal_change_models(train_static, train_long)
    issues.extend(long_issues)
    event_cph = censor_cph = None
    if use_cox_survival_model:
        event_cph, event_issue = fit_cox_time_sampler(train_static, cols, fills, train_static["delta"], "event")
        censor_cph, censor_issue = fit_cox_time_sampler(train_static, cols, fills, 1 - pd.to_numeric(train_static["delta"], errors="coerce").fillna(0), "censoring")
        for issue in [event_issue, censor_issue]:
            if issue:
                issues.append(issue)
    use_cox_survival = bool(use_cox_survival_model and event_cph is not None and censor_cph is not None)
    fallback_distances: list[float] = []
    fallback_neighbors = None if use_cox_survival else prepare_conditioned_neighbors(train_static, target_static, neighbors)
    for r in range(reps):
        static = target_static_template(target_static)
        if use_cox_survival:
            features = _survival_feature_frame(static, cols, fills)
            times, events = [], []
            for _, feat in features.iterrows():
                event_time = sample_cox_time(event_cph, feat, rng)
                censor_time = sample_cox_time(censor_cph, feat, rng)
                observed = min(event_time, censor_time, 1.0)
                times.append(float(np.clip(observed, 0.02, 1.0)))
                events.append(int(event_time <= censor_time and event_time <= 1.0))
        else:
            source_pos, distances = sample_precomputed_conditioned_positions(fallback_neighbors, rng)
            fallback_distances.extend(distances.tolist())
            source_static = train_static.iloc[source_pos].reset_index(drop=True)
            times = pd.to_numeric(source_static["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0).to_numpy(dtype=float).tolist()
            events = pd.to_numeric(source_static["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy().tolist()
        static["U"] = times
        static[SURVIVAL_TIME_COL] = times
        static["delta"] = events
        static[EVENT_COL] = events
        long_rows = []
        for _, st in static.iterrows():
            for visit, t in enumerate(time_grid):
                if t > st["U"]:
                    continue
                row = {"patient_id": int(st["patient_id"]), TREATMENT_NAME: int(st[TREATMENT_NAME]), "visit_index": int(visit), "visit_time": float(t)}
                design = _longitudinal_design(pd.DataFrame([st]), np.asarray([float(t)]), cols, fills).to_numpy(dtype=float)[0]
                for var in LONG_NAMES:
                    baseline_value = safe_float(st.get(var), fills.get(var, 0.0))
                    if abs(float(t)) <= 1e-8:
                        row[var] = float(baseline_value)
                    else:
                        change = float(design @ coefs[var] + rng.normal(0.0, max(residual_sd[var], 1e-6)))
                        row[var] = float(baseline_value + change)
                long_rows.append(row)
        statics.append(static)
        longs.append(pd.DataFrame(long_rows))
    diagnostics = {
        "cox_event_model_used": bool(event_cph is not None),
        "cox_censoring_model_used": bool(censor_cph is not None),
        "survival_generation": "cox_event_and_censoring_models" if use_cox_survival else "conditioned_nearest_neighbor_survival_resampling",
        "survival_fallback_median_conditioning_distance": float(np.median(fallback_distances)) if fallback_distances else float("nan"),
        "neighbors": int(neighbors),
    }
    return statics, longs, issues, diagnostics


SUMMARY_STATIC_COLS = STATIC_CONTINUOUS + STATIC_CATEGORICAL + LONG_NAMES + [TREATMENT_NAME, "U", "delta"]
SUMMARY_LONG_FEATURES = [f"{var}_{suffix}" for var in LONG_NAMES for suffix in ["final", "change", "slope", "auc"]]


def clean_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for col in [*SUMMARY_STATIC_COLS, *SUMMARY_LONG_FEATURES]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [*STATIC_CONTINUOUS, *LONG_NAMES, *SUMMARY_LONG_FEATURES, "U"]:
        if col in out:
            fill = pd.to_numeric(out[col], errors="coerce").median()
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(safe_float(fill, 0.0))
    for col in [*STATIC_CATEGORICAL, TREATMENT_NAME, "delta"]:
        if col in out:
            valid = pd.to_numeric(out[col], errors="coerce").dropna()
            fill = int(valid.mode().iloc[0]) if not valid.empty else 0
            lo = int(valid.min()) if not valid.empty else 0
            hi = int(valid.max()) if not valid.empty else max(fill, 1)
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill).round().clip(lo, hi).astype(int)
    if "U" in out:
        out["U"] = pd.to_numeric(out["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0)
    if "delta" in out:
        out["delta"] = pd.to_numeric(out["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    return out


def benchmark_preprocessing_status(summary: pd.DataFrame) -> dict[str, Any]:
    missing = summary.isna().sum()
    return {
        "conditional_outcome_summary_shape": [int(summary.shape[0]), int(summary.shape[1])],
        "missing_cells_before_ctgan_tvae_imputation": int(missing.sum()),
        "columns_with_missing_before_imputation": {str(k): int(v) for k, v in missing[missing.gt(0)].sort_values(ascending=False).items()},
        "imputation_rule": "continuous outcome-summary columns use train median; event indicator uses train mode and valid-range clipping",
        "conditional_benchmark_note": "CTGAN/TVAE fit baseline covariates, L0 values, treatment, and outcome summaries; real test baseline covariates, L0 values, and treatment are supplied externally at generation after nearest-neighbor conditional selection.",
    }


def subject_summary(static_df: pd.DataFrame, long_df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    static = normalize_synthetic_static(static_df).copy()
    rows: list[dict[str, Any]] = []
    for _, st in static.iterrows():
        pid = int(st["patient_id"])
        row: dict[str, Any] = {"patient_id": pid}
        for col in SUMMARY_STATIC_COLS:
            if col in st:
                row[col] = st[col]
        g = long_df[long_df["patient_id"].astype(int).eq(pid)].copy()
        times = pd.to_numeric(g.get("visit_time", pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
        for var in LONG_NAMES:
            y = pd.to_numeric(g.get(var, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(times) & np.isfinite(y)
            if ok.any():
                order = np.argsort(times[ok])
                tx = times[ok][order]
                yy = y[ok][order]
                baseline = float(row.get(var, yy[0]))
                row[f"{var}_final"] = float(yy[-1])
                row[f"{var}_change"] = float(yy[-1] - baseline)
                row[f"{var}_slope"] = float(np.polyfit(tx, yy, 1)[0]) if len(tx) >= 2 and np.ptp(tx) > 1e-8 else 0.0
                row[f"{var}_auc"] = float(np.trapz(yy, tx)) if len(tx) >= 2 else 0.0
            else:
                for suffix in ["final", "change", "slope", "auc"]:
                    row[f"{var}_{suffix}"] = float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    return clean_subject_summary(out) if clean else out


def outcome_summary_from_subject_summary(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ["U", "delta", *SUMMARY_LONG_FEATURES]
    return summary[[c for c in cols if c in summary]].copy()


def conditional_outcome_summary(train_static: pd.DataFrame, train_long: pd.DataFrame) -> pd.DataFrame:
    summary = subject_summary(train_static, train_long).reset_index(drop=True)
    out = pd.DataFrame(index=summary.index)
    for col in CONDITIONING_COLS:
        if col in summary:
            out[col] = summary[col]
    out[TREATMENT_NAME] = pd.to_numeric(summary[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int)
    for var in LONG_NAMES:
        out[f"{var}_change"] = pd.to_numeric(summary.get(f"{var}_change"), errors="coerce").fillna(0.0)
        out[f"{var}_slope"] = pd.to_numeric(summary.get(f"{var}_slope"), errors="coerce").fillna(0.0)
        out[f"{var}_auc"] = pd.to_numeric(summary.get(f"{var}_auc"), errors="coerce").fillna(0.0)
    out["U"] = pd.to_numeric(summary["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0)
    out["delta"] = pd.to_numeric(summary["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    return out


def _coerce_generated_outcome_summary(sample: pd.DataFrame, train_outcomes: pd.DataFrame, n: int) -> pd.DataFrame:
    out = sample.reset_index(drop=True).copy()
    for col in train_outcomes.columns:
        if col not in out:
            out[col] = train_outcomes[col].sample(len(out), replace=True, random_state=SEED).reset_index(drop=True) if len(out) else []
    if len(out) < n:
        extra = train_outcomes.sample(n - len(out), replace=True, random_state=SEED).reset_index(drop=True)
        out = pd.concat([out, extra], ignore_index=True)
    out = out.head(n).copy()
    for col in CONDITIONING_COLS + SUMMARY_LONG_FEATURES + ["U"]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            fill = pd.to_numeric(train_outcomes[col], errors="coerce").median() if col in train_outcomes else 0.0
            out[col] = out[col].fillna(fill)
    for col in [TREATMENT_NAME, *STATIC_CATEGORICAL]:
        if col in out:
            valid = pd.to_numeric(train_outcomes[col], errors="coerce").dropna() if col in train_outcomes else pd.Series(dtype=float)
            fill = int(valid.mode().iloc[0]) if not valid.empty else 0
            lo = int(valid.min()) if not valid.empty else 0
            hi = int(valid.max()) if not valid.empty else max(fill, 1)
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill).round().clip(lo, hi).astype(int)
    if "U" not in out:
        out["U"] = 1.0
    out["U"] = pd.to_numeric(out["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0)
    if "delta" not in out:
        out["delta"] = 0
    out["delta"] = pd.to_numeric(out["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int)
    return out


def select_generated_outcomes_conditioned(
    sample: pd.DataFrame,
    train_outcomes: pd.DataFrame,
    target_static: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    candidates = _coerce_generated_outcome_summary(sample, train_outcomes, len(sample))
    if candidates.empty:
        return _coerce_generated_outcome_summary(train_outcomes.sample(len(target_static), replace=True, random_state=seed), train_outcomes, len(target_static))
    cols = [c for c in CONDITIONING_COLS if c in candidates and c in target_static]
    fills = {c: _numeric_fill_value(train_outcomes[c], categorical=c in STATIC_CATEGORICAL) for c in cols}
    cand_x = condition_matrix(candidates, cols, fills)
    target_x = condition_matrix(target_static, cols, fills)
    scaler = StandardScaler().fit(pd.concat([condition_matrix(train_outcomes, cols, fills), target_x], ignore_index=True))
    cx = scaler.transform(cand_x)
    tx = scaler.transform(target_x)
    cand_arm = pd.to_numeric(candidates.get(TREATMENT_NAME, pd.Series(0, index=candidates.index)), errors="coerce").fillna(0).round().astype(int).to_numpy()
    target_arm = pd.to_numeric(target_static[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int).to_numpy()
    chosen: list[int] = []
    available = np.arange(len(candidates), dtype=int)
    for i, arm in enumerate(target_arm):
        pool = available[cand_arm[available] == arm]
        if len(pool) == 0:
            pool = available
        d = np.linalg.norm(cx[pool] - tx[i], axis=1)
        order = np.argsort(d)[: max(1, min(25, len(d)))]
        top = pool[order]
        top_d = d[order]
        weights = 1.0 / np.maximum(top_d, 1e-6)
        weights = weights / weights.sum() if np.isfinite(weights).all() and weights.sum() > 0 else None
        chosen.append(int(rng.choice(top, p=weights)))
    return candidates.iloc[chosen].reset_index(drop=True)


def reconstruct_longitudinal_from_outcomes(target_static: pd.DataFrame, outcomes: pd.DataFrame, time_grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    static = target_static_template(target_static)
    static["U"] = pd.to_numeric(outcomes["U"], errors="coerce").fillna(1.0).clip(0.02, 1.0).to_numpy(dtype=float)
    static["delta"] = pd.to_numeric(outcomes["delta"], errors="coerce").fillna(0).round().clip(0, 1).astype(int).to_numpy()
    static[SURVIVAL_TIME_COL] = static["U"].astype(float)
    static[EVENT_COL] = static["delta"].astype(int)
    rows: list[dict[str, Any]] = []
    for _, st in static.iterrows():
        pid = int(st["patient_id"])
        u = float(st["U"])
        for visit, t in enumerate(time_grid):
            if float(t) > u + 1e-8:
                continue
            row = {"patient_id": pid, TREATMENT_NAME: int(st[TREATMENT_NAME]), "visit_index": int(visit), "visit_time": float(t)}
            for var in LONG_NAMES:
                base = safe_float(st.get(var), 0.0)
                change = safe_float(outcomes.iloc[pid].get(f"{var}_change"), 0.0) if f"{var}_change" in outcomes else 0.0
                slope = safe_float(outcomes.iloc[pid].get(f"{var}_slope"), 0.0) if f"{var}_slope" in outcomes else 0.0
                final = base + change if np.isfinite(change) else base + slope
                if len(time_grid) > 1:
                    value = base + (final - base) * min(max(float(t) / max(float(time_grid[-1]), 1e-8), 0.0), 1.0)
                else:
                    value = base
                row[var] = float(value)
            rows.append(row)
    return static, pd.DataFrame(rows)


def write_benchmark_replicate(output_dir: Path, method: str, rep: int, static_df: pd.DataFrame, long_df: pd.DataFrame) -> dict[str, Any]:
    long_format = synthetic_long_format(static_df, long_df, rep)
    safe_method = method.lower().replace("+", "_").replace("/", "_").replace(" ", "_")
    target = output_dir / "synthetic" / f"{safe_method}_rep_{rep:03d}.parquet"
    fmt = "parquet"
    try:
        long_format.to_parquet(target, index=False)
    except Exception:
        target = target.with_suffix(".csv")
        fmt = "csv"
        long_format.to_csv(target, index=False)
    return {
        "method": method,
        "replicate": rep,
        "path": str(target),
        "format": fmt,
        "n_subjects": int(len(static_df)),
        "n_longitudinal_rows": int(len(long_df)),
        "treatment_ratio": float(static_df[TREATMENT_NAME].mean()) if len(static_df) else float("nan"),
        "event_rate": float(static_df["delta"].mean()) if len(static_df) else float("nan"),
    }


def _fit_ctgan_like(method: str, train_summary: pd.DataFrame, seed: int, epochs: int) -> tuple[Any | None, str | None]:
    with temporary_sys_path(BENCHMARK_DIR / "CTGAN"):
        try:
            from ctgan import CTGAN, TVAE
        except Exception as exc:
            return None, f"{method} unavailable: cloned CTGAN import failed ({type(exc).__name__}: {exc})."
        try:
            cls = CTGAN if method == "CTGAN" else TVAE
            kwargs: dict[str, Any] = {"epochs": int(epochs)}
            if method == "CTGAN":
                kwargs.update({"batch_size": 100, "verbose": False, "cuda": torch.cuda.is_available()})
            else:
                kwargs.update({"batch_size": 64, "cuda": torch.cuda.is_available()})
            model = cls(**kwargs)
            discrete = [c for c in [TREATMENT_NAME, "delta", *STATIC_CATEGORICAL] if c in train_summary]
            model.fit(train_summary.drop(columns=["patient_id"], errors="ignore"), discrete_columns=discrete)
            return model, None
        except Exception as exc:
            return None, f"{method} fit failed: {type(exc).__name__}: {exc}"


def ctgan_tvae_replicates(
    method: str,
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    target_static: pd.DataFrame,
    output_dir: Path,
    reps: int,
    seed: int,
    time_grid: np.ndarray,
    epochs: int,
) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    train_outcomes = conditional_outcome_summary(train_static, train_long)
    model, issue = _fit_ctgan_like(method, train_outcomes, seed, epochs)
    if issue:
        return [], [], [], [issue]
    n = len(target_static)
    static_reps: list[pd.DataFrame] = []
    long_reps: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    label = f"conditional_{method}"
    for rep in range(reps):
        try:
            raw_sample = model.sample(max(n * 5, n + 200))
            outcomes = select_generated_outcomes_conditioned(raw_sample, train_outcomes, target_static, seed + rep)
            static_df, long_df = reconstruct_longitudinal_from_outcomes(target_static, outcomes, time_grid)
            static_reps.append(static_df)
            long_reps.append(long_df)
            index_rows.append(write_benchmark_replicate(output_dir, label, rep, static_df, long_df))
        except Exception as exc:
            issues.append(f"{method} replicate {rep:03d} failed: {type(exc).__name__}: {exc}")
    return static_reps, long_reps, index_rows, issues


def dependency_gated_unavailable(method: str, reason: str) -> list[str]:
    return [f"{method} unavailable: {reason}"]


def run_external_benchmarks(
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    output_dir: Path,
    reps: int,
    seed: int,
    time_grid: np.ndarray,
    tau: float,
    ctgan_epochs: int,
    compute_estimands: bool = True,
    eval_replicates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    dep_status = benchmark_dependency_status()
    save_json(output_dir / "metrics" / "benchmark_dependency_status.json", dep_status)
    raw_train_summary = conditional_outcome_summary(train_static, train_long)
    prep_status = benchmark_preprocessing_status(raw_train_summary)
    save_json(output_dir / "metrics" / "benchmark_preprocessing_status.json", prep_status)
    issues: list[str] = []
    issues.append("CTGAN/TVAE conditional benchmarks fit baseline/L0/treatment plus outcome summaries and use nearest-neighbor conditional selection before receiving exact real test baseline covariates, baseline L1-L6, and treatment externally.")
    if prep_status["missing_cells_before_ctgan_tvae_imputation"] > 0:
        issues.append(
            "CTGAN/TVAE conditional outcome-summary benchmark used deterministic train-median/mode imputation because cloned CTGAN rejects null continuous training cells."
        )
    all_index: list[dict[str, Any]] = []
    fidelity_rows: list[dict[str, Any]] = []
    estimand_rows_all: list[dict[str, Any]] = []

    benchmark_sets: list[tuple[str, list[pd.DataFrame], list[pd.DataFrame]]] = []
    for method in ["CTGAN", "TVAE"]:
        statics, longs, index_rows, method_issues = ctgan_tvae_replicates(
            method, train_static, train_long, real_static, output_dir, reps, seed + (70000 if method == "CTGAN" else 80000),
            time_grid, ctgan_epochs
        )
        issues.extend(method_issues)
        all_index.extend(index_rows)
        if statics and longs:
            benchmark_sets.append((f"conditional_{method}", statics, longs))

    if not dep_status["imports"]["tensorflow"]["available"]:
        issues.extend(dependency_gated_unavailable("TimeGAN + Cox", dep_status["imports"]["tensorflow"]["error"] or "tensorflow import unavailable"))
    else:
        issues.append("TimeGAN + Cox not run: adapter requires TensorFlow 1.x API compatibility and was not enabled for this reproducible driver.")

    if not dep_status["imports"]["torchdiffeq"]["available"]:
        issues.extend(dependency_gated_unavailable("Latent ODE + Cox", dep_status["imports"]["torchdiffeq"]["error"] or "torchdiffeq import unavailable"))
    else:
        issues.append("Latent ODE + Cox not run: cloned latent_ode repo is research-script oriented and lacks a safe tabular RCT generation interface for this driver.")

    for method, statics, longs in benchmark_sets:
        eval_n = len(statics) if eval_replicates is None else min(len(statics), int(eval_replicates))
        for i, (s, l) in enumerate(zip(statics[:eval_n], longs[:eval_n])):
            f = benchmark_fidelity_metrics(real_static, real_long, s, l)
            f.update({"method": method, "replicate": i})
            fidelity_rows.append(f)
            if compute_estimands:
                for er in estimand_rows(s, l, tau, i):
                    er["method"] = method
                    estimand_rows_all.append(er)

    fidelity = pd.DataFrame(fidelity_rows)
    estimands = pd.DataFrame(estimand_rows_all)
    index_columns = ["method", "replicate", "path", "format", "n_subjects", "n_longitudinal_rows", "treatment_ratio", "event_rate"]
    index_df = pd.DataFrame(all_index, columns=index_columns)
    if not fidelity.empty:
        fidelity.to_csv(output_dir / "metrics" / "benchmark_fidelity_by_replicate.csv", index=False)
        fidelity.groupby("method").mean(numeric_only=True).reset_index().to_csv(output_dir / "metrics" / "benchmark_fidelity_summary.csv", index=False)
    else:
        pd.DataFrame(columns=["method", "replicate"]).to_csv(output_dir / "metrics" / "benchmark_fidelity_by_replicate.csv", index=False)
        pd.DataFrame(columns=["method"]).to_csv(output_dir / "metrics" / "benchmark_fidelity_summary.csv", index=False)
    if not estimands.empty:
        estimands.to_csv(output_dir / "metrics" / "benchmark_estimand_by_replicate.csv", index=False)
        estimands.groupby(["method", "domain", "endpoint", "estimand"])["estimate"].agg(["median", "mean", "std"]).reset_index().to_csv(
            output_dir / "metrics" / "benchmark_estimand_summary.csv", index=False
        )
    else:
        pd.DataFrame(columns=["method", "replicate", "domain", "endpoint", "estimand", "estimate"]).to_csv(
            output_dir / "metrics" / "benchmark_estimand_by_replicate.csv", index=False
        )
        pd.DataFrame(columns=["method", "domain", "endpoint", "estimand", "median", "mean", "std"]).to_csv(
            output_dir / "metrics" / "benchmark_estimand_summary.csv", index=False
        )
    index_df.to_csv(output_dir / "synthetic" / "benchmark_replicate_index.csv", index=False)
    return fidelity, estimands, index_df, issues


def _summary_metric(summary: pd.DataFrame, metric: str, field: str = "mean") -> float:
    row = summary[summary["metric"].eq(metric)]
    return float(row[field].iloc[0]) if not row.empty else float("nan")


def compare_baselines(
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
    train_static: pd.DataFrame,
    train_long: pd.DataFrame,
    phase_fidelity_summary: pd.DataFrame,
    output_dir: Path,
    reps: int,
    seed: int,
    time_grid: np.ndarray,
    tau: float,
    compute_estimands: bool = True,
    eval_replicates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    issues = []
    boot_static, boot_long, boot_diag = conditional_bootstrap_replicates(train_static, train_long, real_static, time_grid, reps, seed)
    class_static, class_long, class_issues, class_diag = classical_replicates(train_static, train_long, real_static, reps, seed + 1, time_grid)
    issues.extend(class_issues)
    save_json(output_dir / "metrics" / "conditional_baseline_generation_status.json", {
        "conditioning": "real test baseline covariates, baseline L1-L6, and treatment are supplied externally; baselines generate post-baseline longitudinal paths and survival/censoring outcomes.",
        "empirical_subject_bootstrap": boot_diag,
        "classical_simulator": class_diag,
    })
    rows = []
    est_rows = []
    for method, statics, longs in [("conditional_empirical_subject_bootstrap", boot_static, boot_long), ("conditional_classical_simulator", class_static, class_long)]:
        eval_n = len(statics) if eval_replicates is None else min(len(statics), int(eval_replicates))
        for i, (s, l) in enumerate(zip(statics[:eval_n], longs[:eval_n])):
            f = benchmark_fidelity_metrics(real_static, real_long, s, l)
            f.update({"method": method, "replicate": i})
            rows.append(f)
            if compute_estimands:
                for er in estimand_rows(s, l, tau, i):
                    er["method"] = method
                    est_rows.append(er)
    fidelity = pd.DataFrame(rows)
    fidelity.to_csv(output_dir / "metrics" / "baseline_fidelity_by_replicate.csv", index=False)
    fid_sum = fidelity.groupby("method").mean(numeric_only=True).reset_index()
    fid_sum.to_csv(output_dir / "metrics" / "baseline_fidelity_summary.csv", index=False)
    estimands = pd.DataFrame(est_rows)
    estimands.to_csv(output_dir / "metrics" / "baseline_estimand_by_replicate.csv", index=False)
    if estimands.empty:
        est_sum = pd.DataFrame(columns=["method", "domain", "endpoint", "estimand", "median", "mean", "std"])
    else:
        est_sum = estimands.groupby(["method", "domain", "endpoint", "estimand"])["estimate"].agg(["median", "mean", "std"]).reset_index()
    est_sum.to_csv(output_dir / "metrics" / "baseline_estimand_summary.csv", index=False)
    methods = fid_sum[["method", "baseline_continuous_mean_ks", "longitudinal_mean_trajectory_rmse_mean", "km_iae_all", "joint_c2st_auc"]].copy()
    phase_row = pd.DataFrame([{
        "method": "PhaseSyn",
        "baseline_continuous_mean_ks": _summary_metric(phase_fidelity_summary, "baseline_continuous_mean_ks"),
        "longitudinal_mean_trajectory_rmse_mean": _summary_metric(phase_fidelity_summary, "longitudinal_mean_trajectory_rmse_mean"),
        "km_iae_all": _summary_metric(phase_fidelity_summary, "km_iae_all"),
        "joint_c2st_auc": _summary_metric(phase_fidelity_summary, "joint_c2st_auc", "median"),
    }])
    methods = pd.concat([phase_row, methods], ignore_index=True)
    methods.to_csv(output_dir / "tables" / "table7_methods_comparison.csv", index=False)
    return fid_sum, est_sum, issues


def _read_phase_fidelity_summary(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "metrics" / "fidelity_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"Cannot refresh benchmarks without PhaseSyn fidelity summary: {path}")
    return pd.read_csv(path)


def refresh_conditional_benchmarks(args: argparse.Namespace) -> None:
    start = time.time()
    set_seed(args.seed)
    output_dir = args.output_dir
    dirs = ensure_dirs(output_dir)
    append_status(output_dir, f"- Conditional benchmark refresh started at {now_iso()}.")
    data_dir = resolve_dataset_dir(args.dataset_dir, output_dir)
    raw, ids, long_df, types, metadata = _read_simulation(data_dir)
    if (output_dir / "split_ids.csv").exists():
        split_df = pd.read_csv(output_dir / "split_ids.csv")
    else:
        split_df = subject_split(raw, args.seed)
        split_df.to_csv(output_dir / "split_ids.csv", index=False)
    train_idx = split_indices(split_df, "train")
    test_idx = split_indices(split_df, "test")
    real_train_static = real_static_for_indices(raw, train_idx)
    real_train_long = real_long_for_indices(pd.read_csv(data_dir / "longitudinal_observed.csv"), train_idx)
    real_test_static = real_static_for_indices(raw, test_idx)
    real_test_long = real_long_for_indices(pd.read_csv(data_dir / "longitudinal_observed.csv"), test_idx)
    time_grid = np.asarray(metadata.get("visit_schedule", np.linspace(0, 1, 9)), dtype=np.float32)
    tau = float(np.quantile(real_test_static["U"], 0.80))
    phase_fidelity_summary = _read_phase_fidelity_summary(output_dir)

    print(f"[{now_iso()}] refreshing conditional bootstrap/classical baselines", flush=True)
    baseline_generation_reps = args.baseline_replicates
    if args.benchmark_eval_replicates is not None:
        baseline_generation_reps = min(args.baseline_replicates, int(args.benchmark_eval_replicates))
    baseline_fid, baseline_est, baseline_issues = compare_baselines(
        real_test_static,
        real_test_long,
        real_train_static,
        real_train_long,
        phase_fidelity_summary,
        output_dir,
        baseline_generation_reps,
        args.seed + 60000,
        time_grid,
        tau,
        compute_estimands=not args.skip_benchmark_estimands,
        eval_replicates=args.benchmark_eval_replicates,
    )
    print(f"[{now_iso()}] refreshing conditional CTGAN/TVAE benchmarks", flush=True)
    bench_fid, bench_est, bench_index, bench_issues = run_external_benchmarks(
        real_test_static,
        real_test_long,
        real_train_static,
        real_train_long,
        output_dir,
        args.benchmark_replicates,
        args.seed,
        time_grid,
        tau,
        args.ctgan_epochs,
        compute_estimands=not args.skip_benchmark_estimands,
        eval_replicates=args.benchmark_eval_replicates,
    )
    print(f"[{now_iso()}] aggregating conditional benchmark comparison", flush=True)
    if not bench_fid.empty:
        methods_path = dirs["tables"] / "table7_methods_comparison.csv"
        methods = pd.read_csv(methods_path)
        bench_methods = bench_fid.groupby("method").mean(numeric_only=True).reset_index()
        keep_cols = ["method", "baseline_continuous_mean_ks", "longitudinal_mean_trajectory_rmse_mean", "km_iae_all", "joint_c2st_auc"]
        bench_methods = bench_methods.reindex(columns=keep_cols)
        methods = pd.concat([methods, bench_methods], ignore_index=True)
        methods.to_csv(methods_path, index=False)

    key_metrics = {
        "conditional_benchmarks": {
            "baseline_methods": sorted(baseline_fid["method"].dropna().unique().tolist()) if not baseline_fid.empty else [],
            "external_methods": sorted(bench_fid["method"].dropna().unique().tolist()) if not bench_fid.empty else [],
            "external_index_rows": int(len(bench_index)),
            "issue_count": int(len(baseline_issues) + len(bench_issues)),
        }
    }
    save_json(output_dir / "metrics" / "conditional_benchmark_refresh_summary.json", {
        "timestamp": now_iso(),
        "runtime_seconds": float(time.time() - start),
        "replaced_outputs": [
            "metrics/baseline_fidelity_by_replicate.csv",
            "metrics/baseline_fidelity_summary.csv",
            "metrics/baseline_estimand_by_replicate.csv",
            "metrics/baseline_estimand_summary.csv",
            "metrics/benchmark_fidelity_by_replicate.csv",
            "metrics/benchmark_fidelity_summary.csv",
            "metrics/benchmark_estimand_by_replicate.csv",
            "metrics/benchmark_estimand_summary.csv",
            "synthetic/benchmark_replicate_index.csv",
            "tables/table7_methods_comparison.csv",
        ],
        "issues": [*baseline_issues, *bench_issues],
        "benchmark_estimands_computed": bool(not args.skip_benchmark_estimands),
        "benchmark_eval_replicates": args.benchmark_eval_replicates,
        "baseline_generation_replicates": baseline_generation_reps,
        "key_metrics": key_metrics,
    })
    existing_status = output_dir / "STATUS.md"
    old_status = existing_status.read_text(encoding="utf-8") if existing_status.exists() else ""
    write_status(output_dir, [
        "- Status: completed",
        f"- Conditional benchmark refresh completed at {now_iso()}",
        f"- Runtime seconds: {time.time() - start:.1f}",
        f"- Output directory: `{output_dir}`",
        "- Replaced old benchmark results with baseline/L0-conditioned benchmark results.",
        "- Conditioning contract: real test baseline covariates, baseline L1-L6, and treatment are supplied externally; benchmark methods generate post-baseline longitudinal paths plus event/censoring outcomes.",
        f"- Conditional baseline generation replicates per method: {baseline_generation_reps}",
        f"- Conditional benchmark evaluation replicates per method: {args.benchmark_eval_replicates if args.benchmark_eval_replicates is not None else 'all'}",
        f"- Conditional benchmark estimands recomputed: {not args.skip_benchmark_estimands}",
        "- Completed conditional benchmark methods: conditional_empirical_subject_bootstrap, conditional_classical_simulator, conditional_CTGAN, conditional_TVAE",
        "- Skipped or limited experiments:",
        *(f"  - {item}" for item in [*baseline_issues, *bench_issues]),
        f"- Key metric summary: `{json.dumps(_jsonable(key_metrics), sort_keys=True)}`",
        f"- Report: `{output_dir / 'report.md'}`",
        "",
        "## Previous Full-Run Status Snapshot",
        "",
        old_status,
    ])
    if (output_dir / "report.md").exists():
        with (output_dir / "report.md").open("a", encoding="utf-8") as f:
            f.write("\n## Conditional Benchmark Refresh\n\n")
            f.write(f"Updated: {now_iso()}\n\n")
            f.write("The old benchmark comparison rows were replaced with baseline/L0-conditioned benchmark results. For conditional empirical bootstrap, conditional classical simulator, conditional_CTGAN, and conditional_TVAE, the real test baseline covariates, baseline L1-L6 values, and treatment assignments are supplied externally; each method generates post-baseline longitudinal trajectories and survival/censoring outcomes only.\n\n")
            f.write(f"Conditional benchmark estimands recomputed in this refresh: {not args.skip_benchmark_estimands}.\n\n")
            f.write(f"Conditional benchmark evaluation replicates per method: {args.benchmark_eval_replicates if args.benchmark_eval_replicates is not None else 'all'}.\n\n")
            f.write("Replaced files: `metrics/baseline_fidelity_by_replicate.csv`, `metrics/baseline_fidelity_summary.csv`, `metrics/baseline_estimand_by_replicate.csv`, `metrics/baseline_estimand_summary.csv`, `metrics/benchmark_fidelity_by_replicate.csv`, `metrics/benchmark_fidelity_summary.csv`, `metrics/benchmark_estimand_by_replicate.csv`, `metrics/benchmark_estimand_summary.csv`, `synthetic/benchmark_replicate_index.csv`, and `tables/table7_methods_comparison.csv`.\n")
    print("output directory:", output_dir)
    print("completed experiments: conditional benchmark refresh")
    print("skipped experiments:", "; ".join([*baseline_issues, *bench_issues]) if [*baseline_issues, *bench_issues] else "none")
    print("key metric summary:", json.dumps(_jsonable(key_metrics), sort_keys=True))
    print("path to report.md:", output_dir / "report.md")


def ablation_table(fidelity_summary: pd.DataFrame, estimand_summary: pd.DataFrame, coupling: pd.DataFrame, randomization: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    def metric(name: str) -> float:
        row = fidelity_summary[fidelity_summary["metric"].eq(name)]
        return float(row["mean"].iloc[0]) if not row.empty else float("nan")
    rows = [{
        "ablation": "Full PhaseSyn",
        "status": "completed",
        "longitudinal_trajectory_rmse": metric("longitudinal_mean_trajectory_rmse_mean"),
        "km_integrated_absolute_error": metric("km_iae_all"),
        "event_rate_error": metric("event_rate_error"),
        "censoring_rate_error": metric("censoring_rate_error"),
        "treatment_propensity_auc_from_baseline": float(randomization.loc[randomization["scenario"].eq("one_to_one"), "treatment_propensity_auc_baseline"].iloc[0]) if not randomization.empty else np.nan,
    }]
    skipped = [
        ("No dynamic survival", "not run: existing public PhaseSyn training script does not expose a safe static-hazard switch for this simulation driver"),
        ("No censoring model", "not run: existing dynamic survival head jointly returns event and censoring hazards without a no-censoring config path"),
        ("No randomization balance", "not separately run: randomization_loss_weight was 0.0 in the full config, so this run did not evaluate a nonzero randomization-balance penalty"),
        ("Treatment-as-covariate variant", "not run: no existing implementation that reconstructs treatment as an ordinary HI-VAE covariate without code changes"),
    ]
    for name, status in skipped:
        row = dict(rows[0])
        row["ablation"] = name
        row["status"] = status
        for k in list(row):
            if k not in {"ablation", "status"}:
                row[k] = np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "metrics" / "ablation_summary.csv", index=False)
    out.to_csv(output_dir / "metrics" / "ablation_by_replicate.csv", index=False)
    out.to_csv(output_dir / "tables" / "table8_ablation.csv", index=False)
    return out


def plot_baseline_overlay(real: pd.DataFrame, synth: pd.DataFrame, figures: Path) -> None:
    cols = ["W_cont_1", "W_cont_2", "W_count_1", "W_pos_1", "L1", "L2"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.ravel(), cols):
        ax.hist(pd.to_numeric(real[col], errors="coerce").dropna(), bins=30, alpha=0.45, density=True, label="real", color="#1f77b4")
        ax.hist(pd.to_numeric(synth[col], errors="coerce").dropna(), bins=30, alpha=0.45, density=True, label="synthetic", color="#d62728")
        ax.set_title(col)
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend()
    savefig(fig, figures / "baseline_distribution_overlay.png")


def plot_longitudinal_trajectories(real_long: pd.DataFrame, synth_long: pd.DataFrame, figures: Path, filename: str = "longitudinal_mean_trajectories_by_arm.png") -> None:
    cols = LONG_NAMES[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, var in zip(axes.ravel(), cols):
        for arm, color in [(0, "#1f77b4"), (1, "#d62728")]:
            r = real_long[real_long[TREATMENT_NAME].astype(int).eq(arm)].groupby("visit_index").agg(time=("visit_time", "mean"), y=(var, "mean")).dropna()
            s = synth_long[synth_long[TREATMENT_NAME].astype(int).eq(arm)].groupby("visit_index").agg(time=("visit_time", "mean"), y=(var, "mean")).dropna()
            ax.plot(r["time"], r["y"], marker="o", color=color, linestyle="-", label=f"real A={arm}")
            ax.plot(s["time"], s["y"], marker="s", color=color, linestyle="--", label=f"syn A={arm}")
        ax.set_title(var)
        ax.grid(alpha=0.2)
    axes.ravel()[0].legend(fontsize=8)
    savefig(fig, figures / filename)


def plot_km(real: pd.DataFrame, synth: pd.DataFrame, figures: Path, filename: str = "km_curves_by_arm.png") -> None:
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(8, 6))
    for arm, color in [(0, "#1f77b4"), (1, "#d62728")]:
        r = real[real[TREATMENT_NAME].astype(int).eq(arm)]
        s = synth[synth[TREATMENT_NAME].astype(int).eq(arm)]
        ax.step(grid, km_curve(r["U"], r["delta"], grid), where="post", color=color, linestyle="-", label=f"real A={arm}")
        ax.step(grid, km_curve(s["U"], s["delta"], grid), where="post", color=color, linestyle="--", label=f"synthetic A={arm}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.2)
    ax.legend()
    savefig(fig, figures / filename)


def plot_corr_heatmap(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame, figures: Path) -> None:
    r = trajectory_summary(real_long, real_static)
    s = trajectory_summary(syn_long, syn_static)
    cols = [c for c in r.columns if c in s.columns and c not in {"patient_id"}][:30]
    rc = r[cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
    sc = s[cols].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, mat, title in zip(axes, [rc, sc], ["Real", "Synthetic"]):
        im = ax.imshow(np.nan_to_num(mat), vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    savefig(fig, figures / "correlation_heatmap_real_vs_synthetic.png")


def plot_distribution(values: pd.Series, real_value: float, figures: Path, filename: str, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pd.to_numeric(values, errors="coerce").dropna(), bins=30, alpha=0.7, color="#4c78a8")
    if np.isfinite(real_value):
        ax.axvline(real_value, color="#d62728", linewidth=2.0, label="real")
        ax.legend()
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.2)
    savefig(fig, figures / filename)


def plot_randomization(randomization: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(randomization["scenario"], randomization["treatment_propensity_auc_baseline"], color="#4c78a8")
    ax.axhline(0.5, color="black", linestyle="--")
    ax.set_ylim(0.0, 1.0)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("AUC")
    savefig(fig, figures / "treatment_propensity_auc.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(randomization["scenario"], randomization["baseline_smd_mean_abs"], color="#59a14f")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("Mean absolute baseline SMD")
    savefig(fig, figures / "baseline_smd_by_arm.png")


def plot_intervention_figures(statics: dict[str, pd.DataFrame], longs: dict[str, pd.DataFrame], figures: Path) -> None:
    grid = np.linspace(0, 1, 101)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, color in [("original_ratio", "#1f77b4"), ("one_to_one", "#d62728"), ("two_to_one_treatment_control", "#59a14f")]:
        df = normalize_synthetic_static(statics[name])
        for arm, ls in [(0, "-"), (1, "--")]:
            sub = df[df[TREATMENT_NAME].astype(int).eq(arm)]
            if sub.empty:
                continue
            ax.step(grid, km_curve(sub["U"], sub["delta"], grid), where="post", color=color, linestyle=ls, label=f"{name} A={arm}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    savefig(fig, figures / "intervention_km_curves.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, name in zip(axes, ["control_only", "treatment_only"]):
        g = longs[name].groupby("visit_index").agg(time=("visit_time", "mean"), y=("L1", "mean")).dropna()
        ax.plot(g["time"], g["y"], marker="o", color="#4c78a8")
        ax.set_title(name)
        ax.set_xlabel("Time")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Mean L1")
    savefig(fig, figures / "intervention_longitudinal_trajectories.png")


def plot_trial_design(summary: pd.DataFrame, by_rep: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n"], summary["survival_power"], marker="o")
    ax.axhline(0.8, color="black", linestyle="--")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Estimated power")
    ax.grid(alpha=0.2)
    savefig(fig, figures / "power_vs_sample_size_survival.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n"], summary["longitudinal_power"], marker="o", color="#59a14f")
    ax.axhline(0.8, color="black", linestyle="--")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Estimated power")
    ax.grid(alpha=0.2)
    savefig(fig, figures / "power_vs_sample_size_longitudinal.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(summary["n"], summary["expected_events"], marker="o", color="#f28e2b")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Expected events")
    ax.grid(alpha=0.2)
    savefig(fig, figures / "expected_events_vs_sample_size.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    data = [pd.to_numeric(g["cox_hr"], errors="coerce").dropna().to_numpy() for _, g in by_rep.groupby("n")]
    labels = [str(n) for n in sorted(by_rep["n"].unique())]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_xlabel("Sample size")
    ax.set_ylabel("HR estimate")
    ax.grid(alpha=0.2)
    savefig(fig, figures / "hr_distribution_by_sample_size.png")


def write_report(output_dir: Path, completed: list[str], skipped: list[str], key_metrics: dict[str, Any]) -> None:
    files = sorted(str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file())
    report = [
        "# PhaseSyn Minimum Publishable Simulation Evaluation",
        "",
        f"Generated: {now_iso()}",
        f"Output directory: `{output_dir}`",
        "",
        "## Dataset Summary",
        "",
        "Schema and Table 1 are saved in `data_schema.json`, `tables/table1_dataset_summary.csv`, and `tables/table1_dataset_summary.md`.",
        "",
        "## Model Training Summary",
        "",
        f"Training config: `configs/phasesyn_full.yaml`. Checkpoint: `models/phasesyn_full.pt`. Log: `logs/phasesyn_full_train.log`.",
        f"Key training metrics: `{json.dumps(_jsonable(key_metrics.get('training', {})), sort_keys=True)}`.",
        "",
        "## Main Fidelity Results",
        "",
        f"Fidelity summary: `metrics/fidelity_summary.csv`. Selected metrics: `{json.dumps(_jsonable(key_metrics.get('fidelity', {})), sort_keys=True)}`.",
        "High joint C2ST or categorical baseline errors should be treated as limitations rather than cosmetic deviations.",
        "",
        "## Clinical Estimand Preservation Results",
        "",
        "Clinical estimand outputs are in `metrics/estimand_by_replicate.csv`, `metrics/estimand_summary.csv`, and `tables/table3_estimand_preservation.csv`.",
        "",
        "## Randomization/Intervention Test Results",
        "",
        "Treatment intervention tests are in `metrics/randomization_intervention_test.csv` and `tables/table4_randomization_test.csv`.",
        "These results should be interpreted as an intervention sanity check showing externally controlled treatment ratios and downstream arm-specific effects, not as evidence for a nonzero randomization-balance regularizer unless `randomization_loss_weight` is positive.",
        "",
        "## Longitudinal-Survival Coupling Results",
        "",
        "Coupling outputs are in `metrics/longitudinal_survival_coupling_by_replicate.csv` and `tables/table5_longitudinal_survival_coupling.csv`.",
        "",
        "## Virtual Trial Design Results",
        "",
        "Trial design outputs are in `metrics/trial_design_power_by_replicate.csv`, `metrics/trial_design_power_summary.csv`, and `tables/table6_trial_design_utility.csv`.",
        "",
        "## Baseline Comparison",
        "",
        "PhaseSyn and conditional benchmark method comparisons are summarized in `tables/table7_methods_comparison.csv`.",
        "External benchmark outputs, when available, are summarized in `metrics/benchmark_fidelity_summary.csv`, `metrics/benchmark_estimand_summary.csv`, and `synthetic/benchmark_replicate_index.csv`.",
        "Conditional bootstrap, conditional classical simulator, conditional_CTGAN, and conditional_TVAE receive real test baseline covariates, baseline L1-L6 values, and treatment externally; they generate post-baseline longitudinal and survival/censoring outcomes.",
        "",
        "## Ablation Study",
        "",
        "Ablation feasibility/status and completed full-model metrics are in `tables/table8_ablation.csv`; component-necessity claims require retraining actual variants.",
        "",
        "## Failed Or Skipped Experiments",
        "",
        *(f"- {item}" for item in skipped),
        "",
        "## Completed Experiments",
        "",
        *(f"- {item}" for item in completed),
        "",
        "## Output Files",
        "",
        *(f"- `{f}`" for f in files),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    summary = [
        "# Minimum Publishable Evaluation Summary",
        "",
        "## Main Claims",
        "",
        "- PhaseSyn trained on the simple linear RCT simulation with baseline-only encoder conditioning, treatment-explicit generation, dynamic survival, and separate censoring hazard outputs.",
        "- Synthetic RCT replicates were generated under the learned prior with externally controlled treatment ratios.",
        "- Fidelity, clinical estimands, intervention checks, longitudinal-survival coupling, virtual trial design, and conditional benchmark baselines were evaluated.",
        "",
        "## Strongest Quantitative Results",
        "",
        *(f"- {k}: {v}" for k, v in _jsonable(key_metrics).items()),
        "",
        "## Figures And Tables For Manuscript",
        "",
        "- Table 1: `tables/table1_dataset_summary.csv`",
        "- Table 2: `tables/table2_fidelity_metrics.csv`",
        "- Table 3: `tables/table3_estimand_preservation.csv`",
        "- Table 4: `tables/table4_randomization_test.csv`",
        "- Table 5: `tables/table5_longitudinal_survival_coupling.csv`",
        "- Table 6: `tables/table6_trial_design_utility.csv`",
        "- Table 7: `tables/table7_methods_comparison.csv`",
        "- Table 8: `tables/table8_ablation.csv`",
        "- Key figures are in `figures/` with PNG and PDF versions where generated.",
        "",
        "## Limitations",
        "",
        "- External benchmark availability is dependency-gated and recorded in `metrics/benchmark_dependency_status.json`.",
        "- CTGAN/TVAE are conditional outcome-summary benchmarks, not native longitudinal or missingness-aware competitors.",
        "- Conditional benchmark refreshes may cap expensive metric evaluation while still exporting the requested synthetic replicate files; see `metrics/conditional_benchmark_refresh_summary.json`.",
        "- The existing PhaseSyn training entry point is single-device; all GPUs were detected but the run used the supported `cuda` device path.",
        "- The ablation table is a feasibility/status table only; component-necessity claims require retraining actual variants.",
    ]
    (output_dir / "reports" / "minimum_publishable_evaluation_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")


def build_config_args(args: argparse.Namespace, data_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(ROOT / "configs" / "pdc2.yaml"),
        data_dir=data_dir,
        reference_pdc2=str(ROOT / "outputs" / "pdc2" / "experiments_20260602"),
        output_root=output_dir,
        device=args.device,
        seed=args.seed,
        split_seed=args.seed,
        test_fraction=0.2,
        n_replicates=args.n_replicates,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        n_intervals=16,
        lambda_surv=args.lambda_surv,
        kl_weight_s=0.3,
        kl_weight_z=0.3,
        longitudinal_weight=2.0,
        continuous_mse_weight=0.8,
        deterministic_longitudinal=False,
        randomization_loss_weight=args.randomization_loss_weight,
        prior_n=100,
        prior_treatment=0,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the 20260604 PhaseSyn simple linear RCT evaluation.")
    parser.add_argument("--dataset-dir", type=Path, default=REQUESTED_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lambda-surv", type=float, default=1.4)
    parser.add_argument("--randomization-loss-weight", type=float, default=0.0)
    parser.add_argument("--n-replicates", type=int, default=200)
    parser.add_argument("--validation-replicates", type=int, default=20)
    parser.add_argument("--baseline-replicates", type=int, default=200)
    parser.add_argument("--benchmark-replicates", type=int, default=200)
    parser.add_argument("--ctgan-epochs", type=int, default=80)
    parser.add_argument("--trial-reps", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--benchmarks-only", action="store_true", help="Replace baseline/external benchmark outputs using baseline/L0-conditioned generation without retraining PhaseSyn.")
    parser.add_argument("--skip-benchmark-estimands", action="store_true", help="When refreshing benchmarks only, replace fidelity/Table 7 outputs without recomputing per-replicate benchmark estimands.")
    parser.add_argument("--benchmark-eval-replicates", type=int, default=None, help="Limit expensive conditional benchmark metric evaluation to the first K generated replicates per method; generation still uses requested replicate counts.")
    args = parser.parse_args(argv)

    if args.benchmarks_only:
        refresh_conditional_benchmarks(args)
        return

    start = time.time()
    set_seed(args.seed)
    output_dir = args.output_dir
    dirs = ensure_dirs(output_dir)
    write_status(output_dir, [
        "- Status: running",
        f"- Started foreground run at {now_iso()}",
        f"- Requested dataset path: `{args.dataset_dir}`",
        f"- Requested synthetic replicates: {args.n_replicates}",
        f"- Trial design replicates requested by driver: {args.trial_reps}",
    ])
    skipped: list[str] = []
    completed: list[str] = []

    data_dir = resolve_dataset_dir(args.dataset_dir, output_dir)
    raw, ids, long_df, types, metadata = _read_simulation(data_dir)
    baseline = pd.read_csv(data_dir / "baseline.csv")
    survival = pd.read_csv(data_dir / "survival.csv")
    long_obs = pd.read_csv(data_dir / "longitudinal_observed.csv")

    schema = infer_schema(data_dir, raw, long_df)
    save_json(output_dir / "data_schema.json", schema)
    table1 = dataset_summary_table(raw, baseline, survival, long_obs)
    table1.to_csv(dirs["tables"] / "table1_dataset_summary.csv", index=False)
    (dirs["tables"] / "table1_dataset_summary.md").write_text(markdown_table(table1), encoding="utf-8")
    completed.append("Step 0 dataset schema and Table 1")

    split_df = subject_split(raw, args.seed)
    split_df.to_csv(output_dir / "split_ids.csv", index=False)
    train_idx = split_indices(split_df, "train")
    val_idx = split_indices(split_df, "validation")
    test_idx = split_indices(split_df, "test")

    cfg_args = build_config_args(args, data_dir, output_dir)
    cfg = _candidate_config(cfg_args)
    cfg["dataset"]["data_dir"] = str(data_dir)
    cfg["dataset"]["output_root"] = str(output_dir)
    cfg["training"]["seed"] = int(args.seed)
    cfg["training"]["device"] = str(args.device)
    cfg["evaluation"]["n_replicates"] = int(args.n_replicates)
    cfg["simulation_experiment"]["split"] = {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx), "seed": args.seed}
    cfg["simulation_experiment"]["generation_mode"] = "prior_with_external_treatment"
    save_yaml(dirs["configs"] / "phasesyn_full.yaml", cfg)
    completed.append("Step 1 output structure and config")

    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_specs, time_min, time_max, max_visits = _fit_longitudinal_preprocessor(long_df, types, train_idx, raw.iloc[train_idx][SURVIVAL_TIME_COL])
    train_bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    val_bundle = _make_bundle(raw, ids, long_df, types, val_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    test_bundle = _make_bundle(raw, ids, long_df, types, test_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    completed.append("Step 2 subject-level split")

    manifest = {
        "timestamp": now_iso(),
        "git_commit": git_hash(ROOT),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": package_versions(),
        "dataset_path_requested": str(args.dataset_dir),
        "dataset_path_used": str(data_dir),
        "output_path": str(output_dir),
        "random_seeds": {"main": args.seed, "split": args.seed, "replicate_base": args.seed + 10000},
        "train_validation_test_split_sizes": {"train": len(train_idx), "validation": len(val_idx), "test": len(test_idx)},
        "model_config": cfg["model"],
        "number_of_synthetic_replicates": args.n_replicates,
        "number_of_validation_replicates": args.validation_replicates,
        "number_of_external_benchmark_replicates": args.benchmark_replicates,
        "number_of_trial_design_replicates": args.trial_reps,
        "benchmark_repositories": benchmark_repo_status(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "torch_cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "multi_gpu_note": "Existing PhaseSyn train_model accepts one device string; this run used the supported cuda single-device path.",
    }
    save_json(output_dir / "run_manifest.json", manifest)

    train_log = dirs["logs"] / "phasesyn_full_train.log"
    with train_log.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        print(f"PhaseSyn full training started at {now_iso()}")
        print(json.dumps(_jsonable(manifest), indent=2))
        result = train_model(train_bundle, cfg, output_dir=dirs["models"], overfit_name=None)
        print(f"PhaseSyn full training finished at {now_iso()}")
    model = result["model"].to(torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"))
    if not isinstance(model, PhaseSynModel):
        raise TypeError("Expected PhaseSynModel after training.")
    torch.save({"model_state_dict": model.state_dict(), "config": cfg, "manifest": manifest}, dirs["models"] / "phasesyn_full.pt")
    if (dirs["models"] / "train_curves.csv").exists():
        shutil.copy2(dirs["models"] / "train_curves.csv", dirs["logs"] / "phasesyn_full_train_curves.csv")
    training_metrics = {
        "final_loss": float(result["curves"]["loss"].dropna().iloc[-1]),
        "initial_loss": float(result["curves"]["loss"].dropna().iloc[0]),
        "loss_decrease": float((result["curves"]["loss"].dropna().iloc[0] - result["curves"]["loss"].dropna().iloc[-1]) / max(abs(result["curves"]["loss"].dropna().iloc[0]), 1e-8)),
        "nan_epoch_count": int(result["curves"]["nan_epoch"].astype(bool).sum()) if "nan_epoch" in result["curves"] else 0,
    }
    device = next(model.parameters()).device
    audit = _model_audit(cfg, model)
    audit.update(_leakage_diagnostics(model, test_bundle, device))
    audit.update(_survival_generation_perturbation_audit(model, test_bundle, device, args.seed + 997))
    audit.update(_future_generation_perturbation_audit(model, test_bundle, device, args.seed + 1997))
    save_json(dirs["metrics"] / "phasesyn_training_audit.json", audit)
    completed.append("Step 3 trained full PhaseSyn")

    real_train_static = real_static_for_indices(raw, train_idx)
    real_train_long = real_long_for_indices(long_obs, train_idx)
    real_val_static = real_static_for_indices(raw, val_idx)
    real_val_long = real_long_for_indices(long_obs, val_idx)
    real_test_static = real_static_for_indices(raw, test_idx)
    real_test_long = real_long_for_indices(long_obs, test_idx)
    time_grid = np.asarray(metadata.get("visit_schedule", np.linspace(0, 1, max_visits)), dtype=np.float32)
    test_ratio = float(real_test_static[TREATMENT_NAME].mean())
    tau = float(np.quantile(real_test_static["U"], 0.80))

    val_static_reps: list[pd.DataFrame] = []
    val_long_reps: list[pd.DataFrame] = []
    for rep in range(args.validation_replicates):
        s, l, _ = generate_prior_replicate(model, train_bundle, len(val_idx), float(real_val_static[TREATMENT_NAME].mean()), args.seed + 20000 + rep, time_grid, device)
        val_static_reps.append(s)
        val_long_reps.append(l)
    val_fid = pd.DataFrame([
        {"replicate": i, **fidelity_metrics(real_val_static, real_val_long, s, l)}
        for i, (s, l) in enumerate(zip(val_static_reps, val_long_reps))
    ])
    val_metrics = summarize_replicate_metrics(val_fid).set_index("metric").to_dict(orient="index")
    save_json(dirs["metrics"] / "phasesyn_validation_metrics.json", val_metrics)

    static_reps, long_reps, syn_index, syn_issues = write_synthetic_replicates(
        model, train_bundle, output_dir, args.n_replicates, len(test_idx), test_ratio, time_grid, device, args.seed
    )
    skipped.extend(syn_issues)
    completed.append("Step 4 synthetic RCT replicates")

    fidelity_rows = []
    for i, (s, l) in enumerate(zip(static_reps, long_reps)):
        fidelity_rows.append({"replicate": i, **fidelity_metrics(real_test_static, real_test_long, s, l)})
    fidelity_df = pd.DataFrame(fidelity_rows)
    fidelity_df.to_csv(dirs["metrics"] / "fidelity_by_replicate.csv", index=False)
    fidelity_summary = summarize_replicate_metrics(fidelity_df)
    fidelity_summary.to_csv(dirs["metrics"] / "fidelity_summary.csv", index=False)
    fidelity_summary.to_csv(dirs["tables"] / "table2_fidelity_metrics.csv", index=False)
    completed.append("Step 5 fidelity evaluation")

    real_estimands = pd.DataFrame(estimand_rows(real_test_static, real_test_long, tau, "real"))
    syn_estimand_rows: list[dict[str, Any]] = []
    for i, (s, l) in enumerate(zip(static_reps, long_reps)):
        syn_estimand_rows.extend(estimand_rows(s, l, tau, i))
    syn_estimands = pd.DataFrame(syn_estimand_rows)
    all_estimands = pd.concat([real_estimands, syn_estimands], ignore_index=True)
    all_estimands.to_csv(dirs["metrics"] / "estimand_by_replicate.csv", index=False)
    estimand_summary = summarize_estimands(real_estimands, syn_estimands)
    estimand_summary.to_csv(dirs["metrics"] / "estimand_summary.csv", index=False)
    estimand_summary.to_csv(dirs["tables"] / "table3_estimand_preservation.csv", index=False)
    completed.append("Step 6 clinical estimand preservation")

    randomization = intervention_tests(model, train_bundle, output_dir, len(test_idx), test_ratio, time_grid, device, tau, args.seed)
    completed.append("Step 7 randomization and intervention test")

    coupling_by_rep, coupling_sum = coupling_summary(real_test_static, real_test_long, static_reps, long_reps, output_dir)
    completed.append("Step 8 longitudinal-survival coupling")

    if args.trial_reps < 500:
        skipped.append(f"Virtual trial design used R={args.trial_reps} instead of R=500 to keep the foreground run tractable; recorded per user instruction.")
    trial_by_rep, trial_summary = trial_design_simulation(
        model, train_bundle, output_dir, [100, 200, 300, 500, 800, 1200], args.trial_reps, time_grid, device, args.seed
    )
    completed.append("Step 9 virtual trial design utility")

    baseline_fid, baseline_est, baseline_issues = compare_baselines(
        real_test_static,
        real_test_long,
        real_train_static,
        real_train_long,
        fidelity_summary,
        output_dir,
        args.baseline_replicates,
        args.seed + 60000,
        time_grid,
        tau,
    )
    skipped.extend(baseline_issues)
    completed.append("Step 10 empirical bootstrap and classical simulator baselines")

    bench_fid, bench_est, bench_index, bench_issues = run_external_benchmarks(
        real_test_static,
        real_test_long,
        real_train_static,
        real_train_long,
        output_dir,
        args.benchmark_replicates,
        args.seed,
        time_grid,
        tau,
        args.ctgan_epochs,
    )
    skipped.extend(bench_issues)
    completed.append(f"External benchmark repository/dependency audit ({len(bench_issues)} recorded limitations)")
    if not bench_fid.empty:
        methods_path = dirs["tables"] / "table7_methods_comparison.csv"
        methods = pd.read_csv(methods_path)
        bench_methods = bench_fid.groupby("method").mean(numeric_only=True).reset_index()
        keep_cols = ["method", "baseline_continuous_mean_ks", "longitudinal_mean_trajectory_rmse_mean", "km_iae_all", "joint_c2st_auc"]
        bench_methods = bench_methods.reindex(columns=keep_cols)
        methods = pd.concat([methods, bench_methods], ignore_index=True)
        methods.to_csv(methods_path, index=False)
        completed.append("External CTGAN/TVAE benchmark evaluation")

    ablations = ablation_table(fidelity_summary, estimand_summary, coupling_sum, randomization, output_dir)
    skipped.extend([f"Ablation `{r.ablation}`: {r.status}" for r in ablations.itertuples() if r.status != "completed"])
    completed.append("Step 11 ablation status table")

    first_syn_static = static_reps[0]
    first_syn_long = long_reps[0]
    plot_baseline_overlay(real_test_static, first_syn_static, dirs["figures"])
    plot_longitudinal_trajectories(real_test_long, first_syn_long, dirs["figures"])
    plot_km(real_test_static, first_syn_static, dirs["figures"])
    plot_corr_heatmap(real_test_static, real_test_long, first_syn_static, first_syn_long, dirs["figures"])
    plot_distribution(fidelity_df["joint_c2st_auc"], float(fidelity_df["joint_c2st_auc"].median()), dirs["figures"], "c2st_auc_distribution.png", "C2ST AUC")
    real_hr = float(real_estimands[(real_estimands["estimand"].eq("cox_hr"))]["estimate"].iloc[0])
    plot_distribution(syn_estimands[syn_estimands["estimand"].eq("cox_hr")]["estimate"], real_hr, dirs["figures"], "cox_hr_distribution_vs_real.png", "Cox HR")
    real_rmst = float(real_estimands[(real_estimands["estimand"].eq("rmst_diff"))]["estimate"].iloc[0])
    plot_distribution(syn_estimands[syn_estimands["estimand"].eq("rmst_diff")]["estimate"], real_rmst, dirs["figures"], "rmst_distribution_vs_real.png", "RMST difference")
    mmrm = syn_estimands[(syn_estimands["domain"].eq("longitudinal")) & (syn_estimands["estimand"].eq("treatment_time_interaction"))]
    real_mmrm = real_estimands[(real_estimands["domain"].eq("longitudinal")) & (real_estimands["estimand"].eq("treatment_time_interaction"))]["estimate"].median()
    plot_distribution(mmrm["estimate"], float(real_mmrm), dirs["figures"], "mmrm_treatment_effect_distribution_vs_real.png", "Treatment by time")
    plot_randomization(randomization, dirs["figures"])
    plot_distribution(coupling_by_rep["early_response_coef"], float(coupling_sum[coupling_sum["metric"].eq("early_response_coef")]["real_estimate"].iloc[0]) if not coupling_sum.empty else np.nan, dirs["figures"], "landmark_cox_coefficient_distribution.png", "Early response Cox coefficient")
    plot_distribution(coupling_by_rep["early_slope_coef"], float(coupling_sum[coupling_sum["metric"].eq("early_slope_coef")]["real_estimate"].iloc[0]) if not coupling_sum.empty else np.nan, dirs["figures"], "slope_survival_association_distribution.png", "Early slope Cox coefficient")
    plot_km(real_test_static, first_syn_static, dirs["figures"], "responder_nonresponder_km_real_vs_synthetic.png")
    plot_trial_design(trial_summary, trial_by_rep, dirs["figures"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(ablations["ablation"], ablations["km_integrated_absolute_error"].fillna(0.0), color="#4c78a8")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylabel("KM IAE")
    savefig(fig, dirs["figures"] / "ablation_metric_comparison.png")
    completed.append("Figures")

    key_metrics = {
        "training": training_metrics,
        "fidelity": {
            "km_iae_all_mean": float(fidelity_summary.loc[fidelity_summary["metric"].eq("km_iae_all"), "mean"].iloc[0]),
            "event_rate_error_mean": float(fidelity_summary.loc[fidelity_summary["metric"].eq("event_rate_error"), "mean"].iloc[0]),
            "longitudinal_rmse_mean": float(fidelity_summary.loc[fidelity_summary["metric"].eq("longitudinal_mean_trajectory_rmse_mean"), "mean"].iloc[0]),
            "c2st_auc_median": float(fidelity_summary.loc[fidelity_summary["metric"].eq("joint_c2st_auc"), "median"].iloc[0]),
        },
        "estimands": {
            "real_cox_hr": real_hr,
            "synthetic_cox_hr_median": float(syn_estimands[syn_estimands["estimand"].eq("cox_hr")]["estimate"].median()),
        },
        "trial_design": {
            "max_survival_power": float(trial_summary["survival_power"].max()),
            "max_longitudinal_power": float(trial_summary["longitudinal_power"].max()),
        },
        "external_benchmarks": {
            "available_methods": sorted(bench_fid["method"].dropna().unique().tolist()) if not bench_fid.empty else [],
            "issue_count": len(bench_issues),
        },
    }
    write_report(output_dir, completed, skipped, key_metrics)
    completed.append("Step 12 final reports")

    write_status(output_dir, [
        "- Status: completed",
        f"- Completed at {now_iso()}",
        f"- Runtime seconds: {time.time() - start:.1f}",
        f"- Output directory: `{output_dir}`",
        f"- Completed experiments: {', '.join(completed)}",
        "- Skipped or limited experiments:",
        *(f"  - {item}" for item in skipped),
        f"- Key metric summary: `{json.dumps(_jsonable(key_metrics), sort_keys=True)}`",
        f"- Report: `{output_dir / 'report.md'}`",
    ])
    print("output directory:", output_dir)
    print("completed experiments:", "; ".join(completed))
    print("skipped experiments:", "; ".join(skipped) if skipped else "none")
    print("key metric summary:", json.dumps(_jsonable(key_metrics), sort_keys=True))
    print("path to report.md:", output_dir / "report.md")


if __name__ == "__main__":
    main()
