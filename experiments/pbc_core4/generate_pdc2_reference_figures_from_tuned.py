from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from scripts.pdc2.plot_holdout_by_treatment import generate_treatment_figures
from scripts.pdc2.plot_holdout_test_figures import generate_figures

from .load_pbc import LONGITUDINAL_NAMES, TREATMENT_NAME, load_processed, project_path
from .methods import PhaseSynGenerator, analysis_static, fill_baseline, split_static_long


PDC2_LONG_NAME_MAP = {
    "ascites": "ascites",
    "hepatomegaly": "hepatomegaly",
    "spiders": "spiders",
    "edema": "edema",
    "bili": "serBilir",
    "albumin": "albumin",
    "alkaline": "alkaline",
    "ast": "SGOT",
    "platelets": "platelets",
    "prothrombin": "prothrombin",
    "stage": "histologic",
}
PDC2_LONG_COLUMNS = list(PDC2_LONG_NAME_MAP.values())
PDC2_CONTINUOUS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]
PDC2_CATEGORICAL = ["ascites", "hepatomegaly", "spiders", "edema", "histologic"]
PDC2_STATIC_COLUMNS = [
    "time",
    "censor",
    "drug",
    "sex",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "histologic",
    "serBilir",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
]


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _to_num(series: pd.Series, default: float = 0.0) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.notna().any():
        return vals.fillna(float(vals.median()))
    return vals.fillna(float(default))


def _pbc_static_to_pdc2(static: pd.DataFrame) -> pd.DataFrame:
    src = fill_baseline(static.copy(), static)
    out = pd.DataFrame(
        {
            "time": _to_num(src["time"], 1.0).clip(lower=1e-4),
            "censor": _to_num(src["event"], 0.0).round().clip(0, 1).astype(int),
            "drug": _to_num(src[TREATMENT_NAME], 0.0).round().clip(0, 1).astype(int),
            "sex": _to_num(src["sex"], 0.0).round().clip(0, 1).astype(int),
            "ascites": _to_num(src["L0_ascites"], 0.0).round().clip(0, 1).astype(int),
            "hepatomegaly": _to_num(src["L0_hepatomegaly"], 0.0).round().clip(0, 1).astype(int),
            "spiders": _to_num(src["L0_spiders"], 0.0).round().clip(0, 1).astype(int),
            "edema": _to_num(src["L0_edema"], 0.0).round().clip(0, 2).astype(int),
            "histologic": _to_num(src["L0_stage"], 0.0).round().clip(0, 3).astype(int),
            "serBilir": _to_num(src["L0_bili"], 1.0).clip(lower=0.0),
            "albumin": _to_num(src["L0_albumin"], 3.0),
            "alkaline": _to_num(src["L0_alkaline"], 1.0).clip(lower=0.0),
            "SGOT": _to_num(src["L0_ast"], 1.0).clip(lower=0.0),
            "platelets": _to_num(src["L0_platelets"], 1.0).clip(lower=0.0),
            "prothrombin": _to_num(src["L0_prothrombin"], 1.0).clip(lower=0.0),
            "age": _to_num(src["age"], 50.0),
        }
    )
    return out[PDC2_STATIC_COLUMNS]


def _pbc_long_to_pdc2(long_df: pd.DataFrame, subject_to_row: dict[int, int], full_static_pdc2: pd.DataFrame) -> pd.DataFrame:
    out = long_df.copy()
    out["patient_id"] = out["subject_id"].astype(int).map(subject_to_row)
    out = out[out["patient_id"].notna()].copy()
    out["patient_id"] = out["patient_id"].astype(int)
    out["visit_time"] = pd.to_numeric(out["visit_time"], errors="coerce").fillna(0.0)
    keep = ["patient_id", "visit_time", *[col for col in LONGITUDINAL_NAMES if col in out]]
    out = out[keep].copy()
    for pbc_name, pdc2_name in PDC2_LONG_NAME_MAP.items():
        if pbc_name not in out:
            out[pdc2_name] = np.nan
        else:
            out[pdc2_name] = pd.to_numeric(out[pbc_name], errors="coerce")
    out = out[["patient_id", "visit_time", *PDC2_LONG_COLUMNS]]
    for col in PDC2_CATEGORICAL:
        nmax = 3 if col == "histologic" else 2 if col == "edema" else 1
        fill = float(pd.to_numeric(full_static_pdc2[col], errors="coerce").mode().iloc[0])
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill).round().clip(0, nmax)
    for col in PDC2_CONTINUOUS:
        fill = float(pd.to_numeric(full_static_pdc2[col], errors="coerce").median())
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill)
        if col != "albumin":
            out[col] = out[col].clip(lower=0.0)
    out = out.sort_values(["patient_id", "visit_time"]).reset_index(drop=True)
    out["visit_index"] = out.groupby("patient_id").cumcount()
    return out[["patient_id", "visit_time", *PDC2_LONG_COLUMNS]]


def _types_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"name": "survcens", "type": "surv_dynamic", "dim": "2", "nclass": ""},
            {"name": "drug", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "sex", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "ascites", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "hepatomegaly", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "spiders", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "edema", "type": "ordinal", "dim": "1", "nclass": "3"},
            {"name": "histologic", "type": "ordinal", "dim": "1", "nclass": "4"},
            {"name": "serBilir", "type": "pos", "dim": "1", "nclass": ""},
            {"name": "albumin", "type": "real", "dim": "1", "nclass": ""},
            {"name": "alkaline", "type": "pos", "dim": "1", "nclass": ""},
            {"name": "SGOT", "type": "pos", "dim": "1", "nclass": ""},
            {"name": "platelets", "type": "pos", "dim": "1", "nclass": ""},
            {"name": "prothrombin", "type": "pos", "dim": "1", "nclass": ""},
            {"name": "age", "type": "real", "dim": "1", "nclass": ""},
        ]
    )


def _stage_pdc2_compatible_data(cfg: dict[str, Any], output_root: Path) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[int, int]]:
    data = load_processed(cfg["processed_data_dir"], int(cfg["seed"]))
    full_static = analysis_static(data, cfg.get("endpoint", {}).get("primary", "composite"))
    full_static = full_static.sort_values("subject_id").reset_index(drop=True)
    subject_to_row = {int(sid): int(i) for i, sid in enumerate(full_static["subject_id"].astype(int))}
    full_static_pdc2 = _pbc_static_to_pdc2(full_static)
    full_long_pdc2 = _pbc_long_to_pdc2(data.longitudinal, subject_to_row, full_static_pdc2)

    data_dir = output_root / "pdc2_compatible_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    full_static_pdc2.to_csv(data_dir / "data_phasesyn.csv", index=False, header=False)
    pd.DataFrame(
        {
            "id": np.arange(len(full_static), dtype=int),
            "source_id": full_static["subject_id"].astype(int).to_numpy(),
        }
    ).to_csv(data_dir / "pbc2_id.csv", index=False)
    full_long_pdc2.to_csv(data_dir / "longitudinal.csv", index=False)
    _types_table().to_csv(data_dir / "data_types_phasesyn_piecewise.csv", index=False)

    split_rows: list[dict[str, Any]] = []
    for split, subject_ids in data.splits.items():
        for sid in subject_ids:
            row_index = subject_to_row[int(sid)]
            raw = full_static_pdc2.iloc[row_index]
            split_rows.append(
                {
                    "row_index": int(row_index),
                    "panel_patient_id": int(row_index),
                    "original_subject_id": int(sid),
                    "split": str(split),
                    "time": float(raw["time"]),
                    "censor": float(raw["censor"]),
                }
            )
    split_df = pd.DataFrame(split_rows).sort_values(["split", "row_index"]).reset_index(drop=True)
    split_df.to_csv(output_root / "subject_splits.csv", index=False)

    run_cfg = {
        "dataset": {
            "name": "pdc2",
            "data_dir": str(data_dir),
            "output_root": str(output_root),
            "max_visits": None,
        },
        "model": {
            "survival": "dynamic",
            "treatment_variable_name": "drug",
            "baseline_time_eps": 1e-6,
        },
        "training": {
            "seed": int(cfg["seed"]),
            "device": str(cfg.get("phasesyn", {}).get("device", "cpu")),
        },
    }
    (output_root / "run_config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    return data, full_static, full_static_pdc2, full_long_pdc2, subject_to_row


def _time_grid_from_visit_index(long_df: pd.DataFrame) -> np.ndarray:
    grid = (
        long_df[["visit_index", "visit_time"]]
        .dropna()
        .groupby("visit_index", sort=True)["visit_time"]
        .median()
        .sort_index()
        .to_numpy(dtype=float)
    )
    if grid.size == 0:
        grid = np.asarray([0.0, 1.0, 2.0, 3.0, 5.0], dtype=float)
    grid[0] = 0.0
    for idx in range(1, len(grid)):
        if grid[idx] <= grid[idx - 1]:
            grid[idx] = grid[idx - 1] + 1e-3
    return grid


def _target_static_with_row_ids(static: pd.DataFrame, subject_to_row: dict[int, int]) -> pd.DataFrame:
    target = static.copy().reset_index(drop=True)
    target["source_id"] = target["subject_id"].astype(int)
    target["subject_id"] = target["source_id"].map(subject_to_row).astype(int)
    return target.sort_values("subject_id").reset_index(drop=True)


def _generated_static_to_pdc2(static: pd.DataFrame, replicate: int) -> pd.DataFrame:
    out = _pbc_static_to_pdc2(static)
    out.insert(0, "patient_id", static["subject_id"].astype(int).to_numpy())
    out.insert(0, "replicate", int(replicate))
    return out


def _generated_long_to_pdc2(long_df: pd.DataFrame, replicate: int, time_grid: np.ndarray, future_only: bool = True) -> pd.DataFrame:
    rows = long_df.copy()
    if rows.empty:
        columns = ["replicate", "patient_id", "visit_index", "visit_time", "visit_time_norm", "trajectory_scope"]
        for col in PDC2_LONG_COLUMNS:
            columns.extend([col, f"{col}_observed"])
        return pd.DataFrame(columns=columns)
    rows = rows.rename(columns={"subject_id": "patient_id"}).copy()
    rows["visit_index"] = pd.to_numeric(rows["visit_index"], errors="coerce").fillna(0).astype(int)
    if future_only:
        rows = rows[rows["visit_index"] > 0].copy()
    rows["visit_time"] = pd.to_numeric(rows["visit_time"], errors="coerce").fillna(
        rows["visit_index"].map({idx: float(val) for idx, val in enumerate(time_grid)})
    )
    t_min = float(np.nanmin(time_grid)) if len(time_grid) else 0.0
    t_max = float(np.nanmax(time_grid)) if len(time_grid) else 1.0
    denom = max(t_max - t_min, 1e-8)
    out = pd.DataFrame(
        {
            "replicate": int(replicate),
            "patient_id": rows["patient_id"].astype(int),
            "visit_index": rows["visit_index"].astype(int),
            "visit_time": rows["visit_time"].astype(float),
            "visit_time_norm": ((rows["visit_time"].astype(float) - t_min) / denom).clip(0.0, 1.0),
            "trajectory_scope": "generated_future_t_gt_0" if future_only else "generated_t0_plus_future",
        }
    )
    for pbc_name, pdc2_name in PDC2_LONG_NAME_MAP.items():
        value = pd.to_numeric(rows[pbc_name], errors="coerce") if pbc_name in rows else pd.Series(np.nan, index=rows.index)
        if pdc2_name in PDC2_CATEGORICAL:
            nmax = 3 if pdc2_name == "histologic" else 2 if pdc2_name == "edema" else 1
            value = value.round().clip(0, nmax)
        elif pdc2_name != "albumin":
            value = value.clip(lower=0.0)
        out[pdc2_name] = value.to_numpy(dtype=float)
        out[f"{pdc2_name}_observed"] = out[pdc2_name].notna()
    return out


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0.5
    ok = np.isfinite(times)
    times = times[ok]
    events = events[ok]
    if times.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    xs = [0.0]
    ys = [1.0]
    surv = 1.0
    for t in np.unique(times[events]):
        at_risk = np.sum(times >= t)
        if at_risk <= 0:
            continue
        deaths = np.sum((times == t) & events)
        xs.extend([float(t), float(t)])
        ys.extend([surv, surv * (1.0 - deaths / at_risk)])
        surv = ys[-1]
    xs.append(float(np.nanmax(times)))
    ys.append(surv)
    return np.asarray(xs), np.asarray(ys)


def _km_iad(real: pd.DataFrame, syn: pd.DataFrame) -> float:
    rx, ry = _km_curve(real["time"].to_numpy(float), real["censor"].to_numpy(float))
    sx, sy = _km_curve(syn["time"].to_numpy(float), syn["censor"].to_numpy(float))
    horizon = max(float(np.nanmax(rx)), float(np.nanmax(sx)), 1e-8)
    grid = np.linspace(0.0, horizon, 256)
    r_idx = np.clip(np.searchsorted(rx, grid, side="right") - 1, 0, len(ry) - 1)
    s_idx = np.clip(np.searchsorted(sx, grid, side="right") - 1, 0, len(sy) - 1)
    return float(np.trapz(np.abs(ry[r_idx] - sy[s_idx]), grid) / horizon)


def _ks_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.sort(np.asarray(a, dtype=float)[np.isfinite(a)])
    b = np.sort(np.asarray(b, dtype=float)[np.isfinite(b)])
    if a.size == 0 or b.size == 0:
        return float("nan")
    vals = np.unique(np.concatenate([a, b]))
    ca = np.searchsorted(a, vals, side="right") / a.size
    cb = np.searchsorted(b, vals, side="right") / b.size
    return float(np.max(np.abs(ca - cb)))


def _tv_distance(a: np.ndarray, b: np.ndarray, categories: list[int]) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    pa = np.asarray([np.mean(np.rint(a).astype(int) == c) for c in categories])
    pb = np.asarray([np.mean(np.rint(b).astype(int) == c) for c in categories])
    return float(0.5 * np.abs(pa - pb).sum())


def _replicate_metrics(real_static: pd.DataFrame, real_long: pd.DataFrame, syn_static: pd.DataFrame, syn_long: pd.DataFrame, replicate: int) -> dict[str, float]:
    real = real_static.copy()
    real["patient_id"] = real.index.astype(int)
    syn = syn_static.copy()
    merged_static = real[["patient_id", "time", "censor"]].merge(
        syn[["patient_id", "time", "censor"]], on="patient_id", suffixes=("_real", "_syn")
    )
    time_diff = merged_static["time_syn"].to_numpy(float) - merged_static["time_real"].to_numpy(float)
    baseline = merged_static["time_real"].to_numpy(float) - float(merged_static["time_real"].median())
    event_acc = np.mean(
        np.rint(merged_static["censor_syn"].to_numpy(float)).astype(int)
        == np.rint(merged_static["censor_real"].to_numpy(float)).astype(int)
    )

    real_future = real_long[real_long["visit_time"] > 1e-8].copy()
    syn_future = syn_long.copy()
    if "trajectory_scope" in syn_future:
        syn_future = syn_future.drop(columns=["trajectory_scope"])
    joined = real_future.merge(
        syn_future,
        on=["patient_id", "visit_index"],
        suffixes=("_real", "_syn"),
    )
    cont_sq: list[float] = []
    cont_abs: list[float] = []
    cont_base_sq: list[float] = []
    ks_vals: list[float] = []
    baseline_long = real_long.sort_values("visit_time").groupby("patient_id").head(1).set_index("patient_id")
    for col in PDC2_CONTINUOUS:
        r = pd.to_numeric(joined.get(f"{col}_real"), errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(joined.get(f"{col}_syn"), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(r) & np.isfinite(s)
        if not ok.any():
            continue
        diff = s[ok] - r[ok]
        cont_sq.extend((diff**2).tolist())
        cont_abs.extend(np.abs(diff).tolist())
        pid = joined.loc[ok, "patient_id"].astype(int)
        l0 = pd.to_numeric(baseline_long.loc[pid, col].reset_index(drop=True), errors="coerce").to_numpy(dtype=float)
        base_ok = np.isfinite(l0)
        if base_ok.any():
            cont_base_sq.extend(((l0[base_ok] - r[ok][base_ok]) ** 2).tolist())
        ks_vals.append(_ks_distance(r[ok], s[ok]))

    cat_accs: list[float] = []
    tv_vals: list[float] = []
    for col in PDC2_CATEGORICAL:
        r = pd.to_numeric(joined.get(f"{col}_real"), errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(joined.get(f"{col}_syn"), errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(r) & np.isfinite(s)
        if not ok.any():
            continue
        rr = np.rint(r[ok]).astype(int)
        ss = np.rint(s[ok]).astype(int)
        cat_accs.append(float(np.mean(rr == ss)))
        cats = [0, 1, 2, 3] if col == "histologic" else [0, 1, 2] if col == "edema" else [0, 1]
        tv_vals.append(_tv_distance(rr, ss, cats))

    survival_rmse = float(np.sqrt(np.mean(time_diff**2))) if len(time_diff) else float("nan")
    baseline_rmse = float(np.sqrt(np.mean(baseline**2))) if len(baseline) else float("nan")
    cont_rmse = float(np.sqrt(np.mean(cont_sq))) if cont_sq else float("nan")
    cont_base_rmse = float(np.sqrt(np.mean(cont_base_sq))) if cont_base_sq else float("nan")
    return {
        "replicate": float(replicate),
        "real_event_rate": float(real_static["censor"].mean()),
        "synthetic_event_rate": float(syn_static["censor"].mean()),
        "event_rate_diff": float(syn_static["censor"].mean() - real_static["censor"].mean()),
        "survival_time_median_diff": float(syn_static["time"].median() - real_static["time"].median()),
        "survival_time_mean_diff": float(syn_static["time"].mean() - real_static["time"].mean()),
        "survival_km_integrated_abs_error": _km_iad(real_static, syn_static),
        "survival_time_rmse": survival_rmse,
        "survival_time_baseline_rmse": baseline_rmse,
        "survival_time_rmse_ratio": float(survival_rmse / max(baseline_rmse, 1e-8)),
        "survival_time_corr": float(merged_static[["time_real", "time_syn"]].corr().iloc[0, 1])
        if len(merged_static) > 1
        else float("nan"),
        "survival_event_accuracy": float(event_acc),
        "future_continuous_rmse": cont_rmse,
        "future_continuous_mae": float(np.mean(cont_abs)) if cont_abs else float("nan"),
        "future_continuous_l0_carryforward_rmse": cont_base_rmse,
        "future_continuous_rmse_ratio_vs_l0_carryforward": float(cont_rmse / max(cont_base_rmse, 1e-8))
        if np.isfinite(cont_rmse) and np.isfinite(cont_base_rmse)
        else float("nan"),
        "future_continuous_ks_mean": float(np.nanmean(ks_vals)) if ks_vals else float("nan"),
        "future_continuous_ks_max": float(np.nanmax(ks_vals)) if ks_vals else float("nan"),
        "future_categorical_accuracy": float(np.nanmean(cat_accs)) if cat_accs else float("nan"),
        "future_categorical_l0_carryforward_accuracy": float("nan"),
        "future_categorical_tv_mean": float(np.nanmean(tv_vals)) if tv_vals else float("nan"),
        "future_categorical_tv_max": float(np.nanmax(tv_vals)) if tv_vals else float("nan"),
        "valid_inverse_outputs": 1.0,
    }


def _prepare_real_tables_for_split(
    split_df: pd.DataFrame,
    split: str,
    full_static_pdc2: pd.DataFrame,
    full_long_pdc2: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    row_idx = split_df.loc[split_df["split"].eq(split), "row_index"].astype(int).sort_values().to_numpy()
    real_static = full_static_pdc2.iloc[row_idx].copy().reset_index(drop=True)
    real_static.index = row_idx
    real_long = full_long_pdc2[full_long_pdc2["patient_id"].isin(row_idx)].copy()
    real_long["visit_index"] = real_long.groupby("patient_id").cumcount()
    return real_static, real_long, row_idx


def _generate_split_artifacts(
    generator: PhaseSynGenerator,
    split_name: str,
    target_static: pd.DataFrame,
    full_static_pdc2: pd.DataFrame,
    full_long_pdc2: pd.DataFrame,
    split_df: pd.DataFrame,
    output_root: Path,
    n_replicates: int,
    time_grid: np.ndarray,
    seed: int,
) -> pd.DataFrame:
    split_dir = output_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    static_parts: list[pd.DataFrame] = []
    long_parts: list[pd.DataFrame] = []
    real_static, real_long, _ = _prepare_real_tables_for_split(split_df, split_name, full_static_pdc2, full_long_pdc2)
    metrics_rows: list[dict[str, float]] = []
    for rep in range(1, int(n_replicates) + 1):
        _set_seed(int(seed) + 1000 * (1 if split_name == "test" else 0) + rep)
        syn_static, syn_long, _ = generator.generate(
            len(target_static),
            target_baseline=target_static,
            treatment=None,
            time_grid=time_grid,
            truncate_longitudinal_at_survival=False,
        )
        pdc2_static = _generated_static_to_pdc2(syn_static, rep)
        pdc2_long = _generated_long_to_pdc2(syn_long, rep, time_grid, future_only=True)
        static_parts.append(pdc2_static)
        long_parts.append(pdc2_long)
        metrics_rows.append(_replicate_metrics(real_static, real_long, pdc2_static, pdc2_long, rep))
    static_all = pd.concat(static_parts, ignore_index=True)
    long_all = pd.concat(long_parts, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    if split_name == "test":
        static_all.to_csv(split_dir / "holdout_synthetic_static_all.csv", index=False)
        long_all.to_csv(split_dir / "holdout_synthetic_longitudinal_future_all.csv", index=False)
        metrics.to_csv(split_dir / "holdout_replicate_metrics.csv", index=False)
    else:
        static_all.to_csv(split_dir / "synthetic_samples.csv", index=False)
        long_all.to_csv(split_dir / "synthetic_longitudinal_samples.csv", index=False)
        metric_mean = metrics.drop(columns=["replicate"], errors="ignore").mean(numeric_only=True).to_dict()
        metric_mean["replicate"] = 1
        (split_dir / "metrics.json").write_text(json.dumps(metric_mean, indent=2), encoding="utf-8")
        metrics.to_csv(split_dir / "train_replicate_metrics.csv", index=False)
    return metrics


def _remove_generated_markdown(output_root: Path) -> int:
    count = 0
    for path in output_root.rglob("*.md"):
        path.unlink()
        count += 1
    return count


def _png_count(path: Path) -> int:
    return sum(1 for _ in path.rglob("*.png"))


def _infer_reference_replicates(reference_root: Path | None, default: int) -> int:
    if reference_root is None:
        return int(default)
    metrics = reference_root / "test" / "holdout_replicate_metrics.csv"
    if metrics.exists():
        try:
            df = pd.read_csv(metrics)
            if "replicate" in df and df["replicate"].nunique() > 0:
                return int(df["replicate"].nunique())
        except Exception:
            pass
    return int(default)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg_path = project_path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    tuned_root = project_path(args.tuned_root)
    checkpoint = project_path(args.checkpoint) if args.checkpoint else tuned_root / "phasesyn_model" / "train" / "model_checkpoint.pt"
    output_root = project_path(args.output_root) if args.output_root else tuned_root / "pdc2_reference_figures_from_tuned_model"
    output_root.mkdir(parents=True, exist_ok=True)
    reference_root = project_path(args.reference_root) if args.reference_root else None
    n_replicates = int(args.n_replicates or _infer_reference_replicates(reference_root, 20))

    data, full_static, full_static_pdc2, full_long_pdc2, subject_to_row = _stage_pdc2_compatible_data(cfg, output_root)
    split_df = pd.read_csv(output_root / "subject_splits.csv")
    train_static, train_long = split_static_long(data, "train", cfg.get("endpoint", {}).get("primary", "composite"))
    generator = PhaseSynGenerator(cfg, train_static, train_long, output_root / "model_adapter", int(cfg["seed"]))
    load_status = generator.load_checkpoint(checkpoint)
    time_grid = _time_grid_from_visit_index(data.longitudinal)

    target_by_split = {
        "train": _target_static_with_row_ids(train_static, subject_to_row),
        "test": _target_static_with_row_ids(split_static_long(data, "test", cfg.get("endpoint", {}).get("primary", "composite"))[0], subject_to_row),
    }
    metrics_by_split = {}
    for split_name in ["train", "test"]:
        metrics_by_split[split_name] = _generate_split_artifacts(
            generator,
            split_name,
            target_by_split[split_name],
            full_static_pdc2,
            full_long_pdc2,
            split_df,
            output_root,
            n_replicates,
            time_grid,
            int(cfg["seed"]),
        )

    figure_summaries: dict[str, Any] = {}
    for split_name in ["train", "test"]:
        figure_summaries[f"{split_name}_overall"] = generate_figures(output_root, output_root / split_name / "figures", split=split_name)
        figure_summaries[f"{split_name}_by_treatment"] = generate_treatment_figures(
            output_root,
            output_root / split_name / "figures_by_treatment",
            split=split_name,
        )
    removed_md = 0 if args.keep_markdown else _remove_generated_markdown(output_root)
    manifest = {
        "source_tuned_root": str(tuned_root),
        "checkpoint": str(checkpoint),
        "output_root": str(output_root),
        "reference_root": str(reference_root) if reference_root else "",
        "n_replicates": int(n_replicates),
        "seed": int(cfg["seed"]),
        "load_status": load_status,
        "time_grid": [float(x) for x in time_grid],
        "figure_summaries": figure_summaries,
        "png_counts": {
            "train_figures": _png_count(output_root / "train" / "figures"),
            "test_figures": _png_count(output_root / "test" / "figures"),
            "train_figures_by_treatment": _png_count(output_root / "train" / "figures_by_treatment"),
            "test_figures_by_treatment": _png_count(output_root / "test" / "figures_by_treatment"),
            "all": _png_count(output_root),
        },
        "markdown_removed": int(removed_md),
        "schema_note": "PBC tuned variables were mapped to the PDC2 plotting schema: bili->serBilir, ast->SGOT, stage->histologic; cholesterol is omitted because the reference PDC2 figure schema does not plot it.",
        "metrics_rows": {split: int(len(df)) for split, df in metrics_by_split.items()},
    }
    (output_root / "pdc2_reference_figure_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDC2 reference-style train/test figures from the tuned PBC PhaseSyn checkpoint."
    )
    parser.add_argument("--config", default="experiments/pbc_core4/config_pbc_core4_tuned.yaml")
    parser.add_argument("--tuned-root", default="outputs/pbc_experiments/experiment_20260604_core4_tuned")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--reference-root", default="outputs/pdc2/experiments_20260602")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--keep-markdown", action="store_true")
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
