from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

from .load_pbc import LONGITUDINAL_NAMES, TREATMENT_NAME, load_processed, project_path
from .methods import analysis_static
from .metrics import privacy_metrics
from .run_revision import run_revision


def _nu_label(nu: float) -> str:
    return f"{nu:.3f}".replace(".", "p")


def _saved_static(saved: pd.DataFrame, real_static: pd.DataFrame) -> pd.DataFrame:
    static_cols = [c for c in real_static.columns if c in saved.columns]
    out = saved[static_cols].drop_duplicates("subject_id").reset_index(drop=True)
    if "time" not in out and "survival_time" in saved:
        out["time"] = saved.drop_duplicates("subject_id")["survival_time"].to_numpy()
    if "event" not in out and "event" in saved:
        out["event"] = saved.drop_duplicates("subject_id")["event"].to_numpy()
    return out


def _trajectory_summary(long_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in long_df.groupby("subject_id", sort=False):
        row: dict[str, Any] = {"subject_id": int(sid)}
        row["visit_count"] = int(len(g))
        row["max_visit_time"] = float(pd.to_numeric(g["visit_time"], errors="coerce").max()) if "visit_time" in g else np.nan
        for var in LONGITUDINAL_NAMES:
            if var not in g:
                continue
            vals = pd.to_numeric(g[var], errors="coerce").dropna()
            if vals.empty:
                row[f"{var}_mean"] = np.nan
                row[f"{var}_final"] = np.nan
                row[f"{var}_slope"] = np.nan
                continue
            tt = pd.to_numeric(g.loc[vals.index, "visit_time"], errors="coerce") if "visit_time" in g else pd.Series(np.arange(len(vals)))
            row[f"{var}_mean"] = float(vals.mean())
            row[f"{var}_final"] = float(vals.iloc[-1])
            row[f"{var}_slope"] = float(np.polyfit(tt.to_numpy(dtype=float), vals.to_numpy(dtype=float), 1)[0]) if len(vals) > 1 and np.nanstd(tt) > 1e-8 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _trajectory_privacy(real_long: pd.DataFrame, synth_long: pd.DataFrame, method: str, replicate: int, setting: str) -> dict[str, Any]:
    real = _trajectory_summary(real_long)
    synth = _trajectory_summary(synth_long)
    cols = [c for c in real.columns if c in synth.columns and c != "subject_id"]
    if real.empty or synth.empty or not cols:
        return {
            "method": method,
            "replicate": int(replicate),
            "setting": setting,
            "privacy_level": "full_trajectory",
            "trajectory_privacy_status": "missing_trajectory_columns",
        }
    combined = pd.concat([real[cols], synth[cols]], ignore_index=True)
    fills = combined.apply(pd.to_numeric, errors="coerce").median(numeric_only=True).fillna(0.0)
    rx = real[cols].apply(pd.to_numeric, errors="coerce").fillna(fills)
    sx = synth[cols].apply(pd.to_numeric, errors="coerce").fillna(fills)
    scaler = StandardScaler().fit(rx)
    rmat = scaler.transform(rx)
    smat = scaler.transform(sx)
    dist = np.linalg.norm(smat[:, None, :] - rmat[None, :, :], axis=2)
    closest = np.min(dist, axis=1)
    sorted_d = np.sort(dist, axis=1)
    second = sorted_d[:, 1] if sorted_d.shape[1] > 1 else np.full_like(closest, np.nan)
    return {
        "method": method,
        "replicate": int(replicate),
        "setting": setting,
        "privacy_level": "full_trajectory",
        "privacy_trajectory_distance_to_closest_real_record": float(np.nanmean(closest)),
        "privacy_trajectory_nearest_neighbor_distance_ratio": float(np.nanmean(closest / np.maximum(second, 1e-8))),
        "privacy_trajectory_exact_duplicate_rate": float(np.mean(closest < 1e-10)),
        "trajectory_privacy_status": "completed",
    }


def refresh_privacy(config_path: Path, max_replicates_per_method_nu: int = 3) -> dict[str, Any]:
    cfg = yaml.safe_load(project_path(config_path).read_text(encoding="utf-8"))
    output = project_path(cfg["output_dir"])
    data = load_processed(cfg["processed_data_dir"], int(cfg["seed"]))
    static = analysis_static(data)
    real_control = static[static[TREATMENT_NAME].eq(0)].reset_index(drop=True)
    real_control_long = data.longitudinal[data.longitudinal[TREATMENT_NAME].eq(0)].reset_index(drop=True)
    synth_root = output / "exp1_control_arm" / "synthetic"
    rows = []
    for nu in [1 / 3, 2 / 3, 1.0]:
        setting = f"nu={nu:g}"
        folder = synth_root / f"nu_{_nu_label(nu)}"
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*_rep*.csv")):
            stem = path.stem
            try:
                method, rep_text = stem.rsplit("_rep", 1)
                rep = int(rep_text)
            except ValueError:
                continue
            if rep >= int(max_replicates_per_method_nu):
                continue
            saved = pd.read_csv(path)
            synth_static = _saved_static(saved, real_control)
            subject = privacy_metrics(real_control, synth_static, method, rep, setting, fast=False)
            subject["privacy_level"] = "subject_baseline"
            subject["nu"] = float(nu)
            rows.append(subject)
            long_cols = [c for c in ["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *LONGITUDINAL_NAMES] if c in saved.columns]
            traj = _trajectory_privacy(real_control_long, saved[long_cols].copy(), method, rep, setting)
            traj["nu"] = float(nu)
            rows.append(traj)
    privacy = pd.DataFrame(rows)
    tables = output / "exp1_control_arm" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    privacy.to_csv(tables / "exp1_privacy_subject_trajectory_metrics.csv", index=False)
    subject = privacy[privacy["privacy_level"].eq("subject_baseline")].copy()
    if not subject.empty:
        # Keep the legacy file name used by the reporting layer, but now with non-fast metrics.
        subject.to_csv(tables / "exp1_privacy_all_methods.csv", index=False)
    run_summary = run_revision(config_path, preserve_existing_experiments=True)
    key = {
        "output_dir": str(output),
        "privacy_rows": int(len(privacy)),
        "max_replicates_per_method_nu": int(max_replicates_per_method_nu),
        "revision_summary": run_summary,
    }
    (output / "run_privacy_refresh.json").write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(key, indent=2, sort_keys=True))
    return key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh subject-level and trajectory-level Exp1 privacy metrics.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4_revision.yaml"))
    parser.add_argument("--max-replicates-per-method-nu", type=int, default=3)
    args = parser.parse_args(argv)
    refresh_privacy(args.config, max_replicates_per_method_nu=args.max_replicates_per_method_nu)


if __name__ == "__main__":
    main()
