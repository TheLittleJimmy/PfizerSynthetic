from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .load_pbc import project_path
from .run_all import run_all


KEY_EXP1_METRICS = [
    "longitudinal_mean_trajectory_error",
    "longitudinal_change_from_baseline_error",
    "longitudinal_slope_distribution_error",
    "survival_event_rate_error",
    "survival_km_integrated_abs_distance",
    "survival_rmst_difference",
    "survival_median_followup_error",
]


def _set_nested(cfg: dict[str, Any], path: str, value: Any) -> None:
    cur = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


def build_tuning_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    base_path = project_path(args.config)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    cfg["output_dir"] = str(args.output_dir)
    cfg["generation"]["exp1_replicates"] = int(args.exp1_replicates)
    cfg.setdefault("exp1", {})["train_phasesyn_control_fraction_models"] = not args.global_exp1
    if args.epochs is not None:
        cfg["phasesyn"]["epochs"] = int(args.epochs)
    if args.lr is not None:
        cfg["phasesyn"]["lr"] = float(args.lr)
    if args.lambda_surv is not None:
        cfg["phasesyn"]["lambda_surv"] = float(args.lambda_surv)
    if args.n_intervals is not None:
        cfg["phasesyn"]["n_intervals"] = int(args.n_intervals)
    if args.kl_weight_s is not None:
        cfg["phasesyn"]["kl_weight_s"] = float(args.kl_weight_s)
    if args.kl_weight_z is not None:
        cfg["phasesyn"]["kl_weight_z"] = float(args.kl_weight_z)
    if args.longitudinal_weight is not None:
        cfg["phasesyn"]["longitudinal_weight"] = float(args.longitudinal_weight)
    if args.continuous_mse_weight is not None:
        cfg["phasesyn"]["continuous_mse_weight"] = float(args.continuous_mse_weight)
    for override in args.override:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}")
        key, raw = override.split("=", 1)
        try:
            value = yaml.safe_load(raw)
        except Exception:
            value = raw
        _set_nested(cfg, key, value)
    out_dir = project_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config_tuning.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg, config_path


def summarize_exp1(output_dir: Path) -> pd.DataFrame:
    metrics_path = output_dir / "exp1_control_arm" / "tables" / "exp1_metrics_all_methods.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    rows = []
    for method, group in metrics.groupby("method", dropna=False):
        row: dict[str, Any] = {"method": method, "rows": int(len(group))}
        for col in KEY_EXP1_METRICS:
            if col not in group:
                continue
            values = pd.to_numeric(group[col], errors="coerce").to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else np.nan
            if col.endswith("_error") or col in {"survival_rmst_difference", "survival_median_followup_error"}:
                row[f"{col}_mean_abs"] = float(np.nanmean(np.abs(values))) if np.isfinite(values).any() else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    tables = output_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tables / "tuning_exp1_method_summary.csv", index=False)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a reduced PBC Exp1 diagnostic for PhaseSyn tuning.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exp1-replicates", type=int, default=20)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lambda-surv", type=float)
    parser.add_argument("--n-intervals", type=int)
    parser.add_argument("--kl-weight-s", type=float)
    parser.add_argument("--kl-weight-z", type=float)
    parser.add_argument("--longitudinal-weight", type=float)
    parser.add_argument("--continuous-mse-weight", type=float)
    parser.add_argument("--global-exp1", action="store_true", help="Use the global PhaseSyn fit for Exp1 diagnostic generation.")
    parser.add_argument("--override", action="append", default=[], help="Additional YAML-style dotted override, e.g. phasesyn.epochs=120.")
    args = parser.parse_args(argv)

    cfg, config_path = build_tuning_config(args)
    print(f"tuning config: {config_path}")
    run_all(config_path, smoke=False, only="exp1")
    summary = summarize_exp1(project_path(cfg["output_dir"]))
    if summary.empty:
        print("tuning summary: no Exp1 metrics found")
    else:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
