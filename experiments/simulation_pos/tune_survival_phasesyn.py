from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .evaluate_pos import evaluate_pos
from .io_utils import dump_json, load_config, phasesyn_worker_devices, setup_logging, write_csv
from .make_figures import make_figures
from .run_virtual_phase3 import run_virtual_phase3


METHOD_KEY = ["method", "scenario", "replicate", "n_phase3"]


CANDIDATES: dict[str, dict[str, Any]] = {
    "gen_det_u0_main": {
        "reuse_main_checkpoints": True,
        "phasesyn_generation": {
            "deterministic_latents": False,
            "deterministic_u0": True,
            "deterministic_survival": None,
            "survival_event_export": "sample",
        },
    },
    "gen_det_latents_u0_main": {
        "reuse_main_checkpoints": True,
        "phasesyn_generation": {
            "deterministic_latents": True,
            "deterministic_u0": True,
            "deterministic_survival": False,
            "survival_event_export": "sample",
        },
    },
    "gen_det_latents_stoch_u0_main": {
        "reuse_main_checkpoints": True,
        "phasesyn_generation": {
            "deterministic_latents": True,
            "deterministic_u0": False,
            "deterministic_survival": False,
            "survival_event_export": "sample",
        },
    },
    "surv_aux_balanced": {
        "phasesyn_training": {
            "epochs": 360,
            "learning_rate": 7.5e-4,
            "weight_survival": 2.0,
            "weight_longitudinal": 0.75,
            "weight_randomization": 0.05,
            "survival_event_aux_weight": 0.75,
            "survival_time_aux_weight": 0.05,
            "survival_time_head_weight": 0.05,
            "survival_warmup_epochs": 25,
            "u0_kl_weight": 0.001,
            "dynamic_survival_dropout": 0.05,
        },
        "phasesyn_generation": {
            "deterministic_latents": False,
            "deterministic_u0": True,
            "deterministic_survival": None,
            "survival_event_export": "sample",
        },
    },
    "surv_priority": {
        "phasesyn_training": {
            "epochs": 360,
            "learning_rate": 5.0e-4,
            "weight_survival": 4.0,
            "weight_longitudinal": 0.5,
            "weight_randomization": 0.0,
            "survival_event_aux_weight": 1.0,
            "survival_time_aux_weight": 0.10,
            "survival_time_head_weight": 0.10,
            "survival_warmup_epochs": 30,
            "u0_kl_weight": 0.002,
            "dynamic_survival_dropout": 0.05,
        },
        "phasesyn_generation": {
            "deterministic_latents": False,
            "deterministic_u0": True,
            "deterministic_survival": None,
            "survival_event_export": "sample",
        },
    },
    "surv_regularized_u0": {
        "phasesyn_training": {
            "epochs": 360,
            "learning_rate": 7.5e-4,
            "weight_survival": 2.5,
            "weight_longitudinal": 0.75,
            "weight_randomization": 0.05,
            "survival_event_aux_weight": 1.0,
            "survival_time_aux_weight": 0.05,
            "survival_time_head_weight": 0.05,
            "survival_warmup_epochs": 25,
            "u0_sigma_mode": "fixed",
            "u0_fixed_sigma": 0.03,
            "u0_kl_weight": 0.005,
            "dynamic_survival_dropout": 0.05,
        },
        "phasesyn_generation": {
            "deterministic_latents": False,
            "deterministic_u0": True,
            "deterministic_survival": None,
            "survival_event_export": "sample",
        },
    },
    "surv_aux_stochastic_u0": {
        "phasesyn_training": {
            "epochs": 360,
            "learning_rate": 7.5e-4,
            "weight_survival": 2.0,
            "weight_longitudinal": 0.75,
            "weight_randomization": 0.05,
            "survival_event_aux_weight": 0.75,
            "survival_time_aux_weight": 0.05,
            "survival_time_head_weight": 0.05,
            "survival_warmup_epochs": 25,
            "u0_kl_weight": 0.001,
            "dynamic_survival_dropout": 0.05,
        },
        "phasesyn_generation": {
            "deterministic_latents": False,
            "deterministic_u0": False,
            "deterministic_survival": None,
            "survival_event_export": "sample",
        },
    },
}


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _git_metadata(root: Path) -> dict[str, Any]:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "status_short": _run(["git", "status", "--short"]),
        "diff_shortstat": _run(["git", "diff", "--shortstat"]),
    }


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _subset_phase2_manifest(frame: pd.DataFrame, reps_per_scenario: int | None) -> pd.DataFrame:
    if reps_per_scenario is None:
        return frame.copy()
    rows = []
    for _, g in frame.groupby("scenario", sort=False, dropna=False):
        rows.append(g.sort_values("replicate").head(int(reps_per_scenario)))
    return pd.concat(rows, ignore_index=True) if rows else frame.head(0).copy()


def _prepare_candidate_output(
    base_output: Path,
    candidate_output: Path,
    phase2_manifest: pd.DataFrame,
    reps_per_scenario: int | None,
    force: bool,
    reuse_main_checkpoints: bool = False,
) -> pd.DataFrame:
    if force and candidate_output.exists():
        shutil.rmtree(candidate_output)
    candidate_output.mkdir(parents=True, exist_ok=True)
    (candidate_output / "intermediate").mkdir(parents=True, exist_ok=True)
    (candidate_output / "tables").mkdir(parents=True, exist_ok=True)

    _copy_if_exists(base_output / "oracle_true_pos.csv", candidate_output / "oracle_true_pos.csv")
    _copy_if_exists(
        base_output / "intermediate" / "oracle_true_pos_config.json",
        candidate_output / "intermediate" / "oracle_true_pos_config.json",
    )
    _copy_if_exists(
        base_output / "intermediate" / "oracle_trial_analyses.csv",
        candidate_output / "intermediate" / "oracle_trial_analyses.csv",
    )

    benchmark = pd.read_csv(base_output / "benchmark_pos_estimates.csv", keep_default_na=False)
    write_csv(candidate_output / "benchmark_pos_estimates.csv", benchmark)
    write_csv(candidate_output / "method_pos_estimates.csv", benchmark.copy())

    model_paths_path = base_output / "model_paths.csv"
    if model_paths_path.exists():
        model_paths = pd.read_csv(model_paths_path, keep_default_na=False)
        model_paths = model_paths[model_paths["method"].ne("PhaseSyn")].copy()
        write_csv(candidate_output / "model_paths.csv", model_paths)

    subset = _subset_phase2_manifest(phase2_manifest, reps_per_scenario)
    write_csv(candidate_output / "phase2_dataset_manifest.csv", subset)
    if reuse_main_checkpoints:
        _link_main_checkpoints(base_output, candidate_output, subset)
    return subset


def _link_main_checkpoints(base_output: Path, candidate_output: Path, phase2_manifest: pd.DataFrame) -> None:
    for row in phase2_manifest.itertuples(index=False):
        scenario = str(row.scenario)
        rep = int(row.replicate)
        src = base_output / "models" / scenario / f"phasesyn_rep_{rep:03d}"
        dst = candidate_output / "models" / scenario / f"phasesyn_rep_{rep:03d}"
        if not (src / "model_checkpoint.pt").exists():
            raise FileNotFoundError(src / "model_checkpoint.pt")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(src, target_is_directory=True)
        except OSError:
            shutil.copytree(src, dst)


def _summarize_output(candidate: str, output: Path, screen: bool) -> dict[str, Any]:
    acc = pd.read_csv(output / "tables" / "pos_bias_rmse_mae.csv", keep_default_na=False)
    merged = pd.read_csv(output / "tables" / "pos_estimates_with_oracle.csv", keep_default_na=False)
    decision = pd.read_csv(output / "go_no_go_decision_metrics.csv", keep_default_na=False)
    ranking = pd.read_csv(output / "design_ranking_accuracy.csv", keep_default_na=False)
    phase_acc = acc[acc["method"].eq("PhaseSyn")].copy()
    phase_merged = merged[merged["method"].eq("PhaseSyn")].copy()
    phase_decision = decision[decision["method"].eq("PhaseSyn")].copy()
    phase_ranking = ranking[ranking["method"].eq("PhaseSyn")].copy()
    out = {
        "candidate": candidate,
        "screen": bool(screen),
        "output_dir": str(output),
        "phasesyn_rows": int(len(phase_merged)),
        "mean_pos_rmse": float(phase_acc["pos_rmse"].mean()),
        "mean_pos_mae": float(phase_acc["pos_mae"].mean()),
        "mean_abs_bias": float(phase_acc["pos_bias"].abs().mean()),
        "direct_pos_rmse": float(np.sqrt(np.mean(np.square(phase_merged["pos_error"].to_numpy(dtype=float))))),
        "direct_pos_mae": float(np.mean(np.abs(phase_merged["pos_error"].to_numpy(dtype=float)))),
        "direct_null_rmse": float(
            np.sqrt(np.mean(np.square(phase_merged.loc[phase_merged["scenario"].eq("null"), "pos_error"].to_numpy(dtype=float))))
        ),
        "mean_event_rate_error": float(phase_acc["event_rate_error"].mean()),
        "mean_censoring_rate_error": float(phase_acc["censoring_rate_error"].mean()),
        "mean_pos_hat": float(phase_merged["pos_hat"].mean()) if len(phase_merged) else float("nan"),
        "decision_accuracy": float(phase_decision["decision_accuracy"].iloc[0]) if len(phase_decision) else float("nan"),
        "false_go_rate": float(phase_decision["false_go_rate"].iloc[0]) if len(phase_decision) else float("nan"),
        "false_stop_rate": float(phase_decision["false_stop_rate"].iloc[0]) if len(phase_decision) else float("nan"),
        "ranking_accuracy": float(phase_ranking["ranking_accuracy"].iloc[0]) if len(phase_ranking) else float("nan"),
    }
    for _, row in phase_acc.iterrows():
        label = f"{row['scenario']}_n{int(row['n_phase3'])}"
        out[f"rmse_{label}"] = float(row["pos_rmse"])
        out[f"bias_{label}"] = float(row["pos_bias"])
        out[f"mean_pos_{label}"] = float(row["mean_pos_hat"])
        out[f"true_pos_{label}"] = float(row["true_pos"])
    return out


def _candidate_config(base_cfg: dict[str, Any], candidate: str, screen: bool, args: argparse.Namespace) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise KeyError(f"Unknown candidate {candidate!r}; choices: {', '.join(sorted(CANDIDATES))}")
    cfg = _deep_update(base_cfg, CANDIDATES[candidate])
    train = cfg.setdefault("phasesyn_training", {})
    if screen:
        train["epochs"] = int(args.screen_epochs)
        cfg["m_syn"] = int(args.screen_m_syn)
        cfg["virtual_batch_size"] = int(args.screen_virtual_batch_size)
    else:
        train["epochs"] = int(args.full_epochs or train.get("epochs", 360))
        cfg["m_syn"] = int(args.full_m_syn)
        cfg["virtual_batch_size"] = int(args.full_virtual_batch_size)
    return cfg


def _validate_generation_semantics(cfg: dict[str, Any], *, screen: bool) -> None:
    gen = cfg.get("phasesyn_generation", {})
    event_export = str(gen.get("survival_event_export", "sample"))
    deterministic_survival = gen.get("deterministic_survival", None)
    if event_export != "sample":
        raise ValueError(
            "PhaseSyn survival PoS tuning must use survival_event_export='sample'. "
            "Probability thresholding is not a stochastic virtual trial generator."
        )
    if not screen and deterministic_survival is True:
        raise ValueError("Full survival PoS evaluation must keep stochastic survival sampling.")


def write_survival_diagnostics(base_output: Path) -> Path:
    phase2 = pd.read_csv(base_output / "phase2_dataset_manifest.csv", keep_default_na=False)
    phase = pd.read_csv(base_output / "tables" / "pos_estimates_with_oracle.csv", keep_default_na=False)
    phase = phase[phase["method"].eq("PhaseSyn")].copy()
    rows = []
    for row in phase2.itertuples(index=False):
        scenario = str(row.scenario)
        rep = int(row.replicate)
        model_dir = base_output / "models" / scenario / f"phasesyn_rep_{rep:03d}"
        train_path = model_dir / "train_curves.csv"
        metrics_path = model_dir / "metrics.json"
        train_tail: dict[str, Any] = {}
        if train_path.exists():
            train = pd.read_csv(train_path)
            if len(train):
                last = train.tail(20)
                train_tail = {
                    "train_loss_tail": float(last["loss"].mean()),
                    "train_surv_loss_tail": float(last["loss_surv_dyn"].mean()) if "loss_surv_dyn" in last else np.nan,
                    "train_event_hazard_tail": float(last["event_hazard_summary"].mean()) if "event_hazard_summary" in last else np.nan,
                    "train_censoring_hazard_tail": float(last["censoring_hazard_summary"].mean()) if "censoring_hazard_summary" in last else np.nan,
                    "train_nan_epoch_count": int(train.get("nan_epoch", pd.Series(dtype=bool)).astype(bool).sum()),
                }
        metrics: dict[str, Any] = {}
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key in [
                "real_event_rate",
                "synthetic_event_rate",
                "event_rate_diff",
                "survival_km_integrated_abs_error",
                "survival_time_mean_diff",
                "survival_time_median_diff",
            ]:
                if key in raw:
                    metrics[f"checkpoint_{key}"] = raw[key]
        design = phase[(phase["scenario"].eq(scenario)) & (phase["replicate"].eq(rep))]
        for n_phase3, g in design.groupby("n_phase3", dropna=False):
            rows.append({
                "scenario": scenario,
                "replicate": rep,
                "n_phase3": int(n_phase3),
                "phase2_event_rate": float(row.event_rate),
                "phase2_censoring_rate": float(row.censoring_rate),
                "pos_hat": float(g["pos_hat"].iloc[0]),
                "true_pos": float(g["true_pos"].iloc[0]),
                "pos_error": float(g["pos_error"].iloc[0]),
                "virtual_event_rate_hat": float(g["event_rate_hat"].iloc[0]),
                "true_event_rate": float(g["true_event_rate"].iloc[0]),
                "virtual_censoring_rate_hat": float(g["censoring_rate_hat"].iloc[0]),
                "true_censoring_rate": float(g["true_censoring_rate"].iloc[0]),
                "mean_hr_hat": float(g["mean_hr_hat"].iloc[0]),
                **train_tail,
                **metrics,
            })
    diag = pd.DataFrame(rows)
    out = base_output / "tuning" / "survival_pos" / "survival_calibration_diagnostics.csv"
    write_csv(out, diag)
    summary = (
        diag.groupby("scenario", dropna=False)
        .agg(
            mean_abs_pos_error=("pos_error", lambda x: float(np.mean(np.abs(x)))),
            mean_event_rate_error=("virtual_event_rate_hat", lambda x: float(np.mean(np.abs(x - diag.loc[x.index, "true_event_rate"])))),
            mean_censoring_rate_error=("virtual_censoring_rate_hat", lambda x: float(np.mean(np.abs(x - diag.loc[x.index, "true_censoring_rate"])))),
            checkpoint_event_rate_diff=("checkpoint_event_rate_diff", "mean"),
            train_event_hazard_tail=("train_event_hazard_tail", "mean"),
            train_censoring_hazard_tail=("train_censoring_hazard_tail", "mean"),
        )
        .reset_index()
        if len(diag)
        else pd.DataFrame()
    )
    write_csv(base_output / "tuning" / "survival_pos" / "survival_calibration_diagnostics_summary.csv", summary)
    return out


def run_candidate(
    base_cfg: dict[str, Any],
    base_output: Path,
    candidate: str,
    args: argparse.Namespace,
    screen: bool,
    force: bool,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    phase2_manifest = pd.read_csv(base_output / "phase2_dataset_manifest.csv", keep_default_na=False)
    cfg = _candidate_config(base_cfg, candidate, screen=screen, args=args)
    _validate_generation_semantics(cfg, screen=screen)
    suffix = "screen" if screen else "full"
    candidate_output = base_output / "tuning" / "survival_pos" / suffix / candidate
    cfg["output_dir"] = str(candidate_output)
    subset = _prepare_candidate_output(
        base_output,
        candidate_output,
        phase2_manifest,
        reps_per_scenario=int(args.screen_reps_per_scenario) if screen else None,
        force=force,
        reuse_main_checkpoints=bool(cfg.get("reuse_main_checkpoints", False)),
    )
    with open(candidate_output / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_config_path"}, f, sort_keys=False)
    logger = setup_logging(candidate_output, name=f"survival_tuning_{candidate}_{suffix}")
    logger.info("candidate=%s screen=%s output=%s devices=%s", candidate, screen, candidate_output, phasesyn_worker_devices(cfg))
    run_virtual_phase3(cfg, candidate_output, subset, logger=logger)
    evaluate_pos(cfg, candidate_output)
    figures = make_figures(cfg, candidate_output)
    summary = _summarize_output(candidate, candidate_output, screen=screen)
    manifest = {
        "candidate": candidate,
        "screen": bool(screen),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_output_dir": str(base_output),
        "output_dir": str(candidate_output),
        "config_path": str(base_cfg.get("_config_path", "")),
        "phasesyn_worker_devices": phasesyn_worker_devices(cfg),
        "phase2_manifest_rows": int(len(subset)),
        "m_syn": int(cfg["m_syn"]),
        "figures": figures,
        "summary": summary,
        "candidate_overrides": CANDIDATES[candidate],
        "git": _git_metadata(root),
    }
    dump_json(candidate_output / "survival_tuning_candidate_manifest.json", manifest)
    return summary


def _audit_replacement_inputs(base_output: Path, candidate_output: Path) -> dict[str, Any]:
    def _normalize_mount_alias(path_value: str) -> str:
        path_value = os.path.abspath(os.path.expanduser(path_value))
        if path_value.startswith("/project/Stat/"):
            return "/lustre/project/Stat/" + path_value[len("/project/Stat/") :]
        return path_value

    candidate_output_abs = _normalize_mount_alias(str(candidate_output))

    def _is_under(path_value: str, root: str) -> bool:
        try:
            path_abs = _normalize_mount_alias(path_value)
            return os.path.commonpath([path_abs, root]) == root
        except ValueError:
            return False

    base_benchmark = pd.read_csv(base_output / "benchmark_pos_estimates.csv", keep_default_na=False)
    candidate_method = pd.read_csv(candidate_output / "method_pos_estimates.csv", keep_default_na=False)
    candidate_phase = candidate_method[candidate_method["method"].eq("PhaseSyn")].copy()
    candidate_bench = candidate_method[candidate_method["method"].ne("PhaseSyn")].copy()
    candidate_paths = pd.read_csv(candidate_output / "model_paths.csv", keep_default_na=False)
    oracle_base = pd.read_csv(base_output / "oracle_true_pos.csv", keep_default_na=False)
    oracle_candidate = pd.read_csv(candidate_output / "oracle_true_pos.csv", keep_default_na=False)
    phase_paths = candidate_paths[candidate_paths["method"].eq("PhaseSyn")].copy()
    tuned_root_ok = bool(
        len(phase_paths)
        and phase_paths["model_dir"].map(lambda p: _is_under(str(p), candidate_output_abs)).all()
        and phase_paths["checkpoint"].map(lambda p: _is_under(str(p), candidate_output_abs)).all()
    )
    audit = {
        "candidate_phase_rows": int(len(candidate_phase)),
        "candidate_benchmark_rows": int(len(candidate_bench)),
        "base_benchmark_rows": int(len(base_benchmark)),
        "phase_model_path_rows": int(len(phase_paths)),
        "oracle_identical": bool(oracle_base.equals(oracle_candidate)),
        "phase_model_paths_under_candidate_output": tuned_root_ok,
        "phase_duplicate_keys": int(candidate_phase.duplicated(METHOD_KEY).sum()) if set(METHOD_KEY).issubset(candidate_phase.columns) else -1,
        "benchmark_duplicate_keys": int(candidate_bench.duplicated(METHOD_KEY).sum()) if set(METHOD_KEY).issubset(candidate_bench.columns) else -1,
    }
    expected_phase = 3 * 30 * 2
    expected_benchmark = 4 * 3 * 30 * 2
    failed = [
        name for name, ok in {
            "candidate_phase_rows": audit["candidate_phase_rows"] == expected_phase,
            "candidate_benchmark_rows": audit["candidate_benchmark_rows"] == expected_benchmark,
            "base_benchmark_rows": audit["base_benchmark_rows"] == expected_benchmark,
            "phase_model_path_rows": audit["phase_model_path_rows"] == expected_phase // 2,
            "oracle_identical": audit["oracle_identical"],
            "phase_model_paths_under_candidate_output": audit["phase_model_paths_under_candidate_output"],
            "phase_duplicate_keys": audit["phase_duplicate_keys"] == 0,
            "benchmark_duplicate_keys": audit["benchmark_duplicate_keys"] == 0,
        }.items()
        if not ok
    ]
    audit["passed"] = not failed
    audit["failed_checks"] = failed
    if failed:
        raise RuntimeError(f"Replacement audit failed: {failed}")
    return audit


def _method_summary(frame: pd.DataFrame, method: str) -> dict[str, Any]:
    g = frame[frame["method"].eq(method)].copy()
    return {
        "method": method,
        "mean_pos_rmse": float(np.sqrt(np.mean(np.square(g["pos_error"].to_numpy(dtype=float))))),
        "mean_pos_mae": float(np.mean(np.abs(g["pos_error"].to_numpy(dtype=float)))),
        "mean_abs_bias": float(abs(g["pos_error"].mean())),
        "mean_event_rate_error": float(np.mean(np.abs(g["event_rate_hat"].to_numpy(dtype=float) - g["true_event_rate"].to_numpy(dtype=float)))),
        "mean_censoring_rate_error": float(np.mean(np.abs(g["censoring_rate_hat"].to_numpy(dtype=float) - g["true_censoring_rate"].to_numpy(dtype=float)))),
        "null_rmse": float(np.sqrt(np.mean(np.square(g.loc[g["scenario"].eq("null"), "pos_error"].to_numpy(dtype=float))))),
    }


def _passes_replacement_constraints(summary_rows: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    by_version = {str(row["version"]): row for row in summary_rows}
    original = by_version.get("Original PhaseSyn")
    tuned = by_version.get("Tuned PhaseSyn")
    if original is None or tuned is None:
        return False, ["missing_original_or_tuned_summary"]
    checks = {
        "mean_pos_rmse_improved": float(tuned["mean_pos_rmse"]) < float(original["mean_pos_rmse"]),
        "null_rmse_not_worse": float(tuned["null_rmse"]) <= float(original["null_rmse"]) + 1e-12,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return not failed, failed


def _write_original_vs_tuned(base_output: Path, candidate_output: Path, summary: dict[str, Any]) -> dict[str, Any]:
    original = pd.read_csv(base_output / "tables" / "pos_estimates_with_oracle.csv", keep_default_na=False)
    tuned = pd.read_csv(candidate_output / "tables" / "pos_estimates_with_oracle.csv", keep_default_na=False)
    original_phase = original[original["method"].eq("PhaseSyn")].copy()
    tuned_phase = tuned[tuned["method"].eq("PhaseSyn")].copy()
    original_phase["version"] = "Original PhaseSyn"
    tuned_phase["version"] = "Tuned PhaseSyn"
    combined = pd.concat([original_phase, tuned_phase], ignore_index=True)
    write_csv(base_output / "tables" / "survival_tuning_original_vs_tuned_rows.csv", combined)

    keys = ["scenario", "replicate", "n_phase3"]
    paired = original_phase[keys + ["pos_hat", "true_pos", "event_rate_hat", "true_event_rate", "censoring_rate_hat", "true_censoring_rate"]].merge(
        tuned_phase[keys + ["pos_hat", "event_rate_hat", "censoring_rate_hat"]],
        on=keys,
        suffixes=("_original", "_tuned"),
    )
    paired["abs_pos_error_original"] = (paired["pos_hat_original"] - paired["true_pos"]).abs()
    paired["abs_pos_error_tuned"] = (paired["pos_hat_tuned"] - paired["true_pos"]).abs()
    paired["abs_pos_error_delta"] = paired["abs_pos_error_tuned"] - paired["abs_pos_error_original"]
    paired["event_rate_error_original"] = (paired["event_rate_hat_original"] - paired["true_event_rate"]).abs()
    paired["event_rate_error_tuned"] = (paired["event_rate_hat_tuned"] - paired["true_event_rate"]).abs()
    paired["event_rate_error_delta"] = paired["event_rate_error_tuned"] - paired["event_rate_error_original"]
    paired["censoring_rate_error_original"] = (paired["censoring_rate_hat_original"] - paired["true_censoring_rate"]).abs()
    paired["censoring_rate_error_tuned"] = (paired["censoring_rate_hat_tuned"] - paired["true_censoring_rate"]).abs()
    paired["censoring_rate_error_delta"] = paired["censoring_rate_error_tuned"] - paired["censoring_rate_error_original"]
    write_csv(base_output / "tables" / "survival_tuning_paired_deltas.csv", paired)

    summary_rows = []
    for version, frame in [("Original PhaseSyn", original_phase), ("Tuned PhaseSyn", tuned_phase)]:
        row = _method_summary(frame, "PhaseSyn")
        row["version"] = version
        summary_rows.append(row)
    summary_frame = pd.DataFrame(summary_rows)
    summary_frame["selected_candidate"] = summary["candidate"]
    write_csv(base_output / "tables" / "survival_tuning_original_vs_tuned.csv", summary_frame)
    passed, failed = _passes_replacement_constraints(summary_frame.to_dict("records"))
    original_summary = summary_frame[summary_frame["version"].eq("Original PhaseSyn")].iloc[0]
    tuned_summary = summary_frame[summary_frame["version"].eq("Tuned PhaseSyn")].iloc[0]
    calibration_tradeoffs = {
        "event_rate_error_delta": float(tuned_summary["mean_event_rate_error"] - original_summary["mean_event_rate_error"]),
        "censoring_rate_error_delta": float(tuned_summary["mean_censoring_rate_error"] - original_summary["mean_censoring_rate_error"]),
    }

    delta_summary = (
        paired.groupby(["scenario", "n_phase3"], dropna=False)
        .agg(
            mean_abs_pos_error_delta=("abs_pos_error_delta", "mean"),
            mean_event_rate_error_delta=("event_rate_error_delta", "mean"),
            mean_censoring_rate_error_delta=("censoring_rate_error_delta", "mean"),
        )
        .reset_index()
    )
    write_csv(base_output / "tables" / "survival_tuning_design_deltas.csv", delta_summary)
    return {
        "comparison_rows": int(len(combined)),
        "paired_rows": int(len(paired)),
        "summary": summary_frame.to_dict("records"),
        "replacement_constraints_passed": bool(passed),
        "replacement_constraints_failed": failed,
        "replacement_policy": "PoS-primary: require pooled PoS RMSE improvement and non-worse null RMSE; record event/censoring calibration as tradeoff diagnostics.",
        "calibration_tradeoffs": calibration_tradeoffs,
    }


def _make_tuning_delta_figure(base_output: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = pd.read_csv(base_output / "tables" / "survival_tuning_original_vs_tuned.csv", keep_default_na=False)
    deltas = pd.read_csv(base_output / "tables" / "survival_tuning_design_deltas.csv", keep_default_na=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    metrics = ["mean_pos_rmse", "mean_event_rate_error", "null_rmse"]
    x = np.arange(len(metrics))
    width = 0.35
    for offset, version, color in [(-width / 2, "Original PhaseSyn", "#999999"), (width / 2, "Tuned PhaseSyn", "#D55E00")]:
        row = summary[summary["version"].eq(version)].iloc[0]
        axes[0].bar(x + offset, [row[m] for m in metrics], width=width, label=version, color=color, alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Pooled\nPoS RMSE", "Event-rate\nerror", "Pooled null\nRMSE"], rotation=0, ha="center")
    axes[0].set_ylabel("Error")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("PoS-primary survival tuning\n(row-level selection metric)")

    deltas["design"] = deltas["scenario"].astype(str) + "\nn=" + deltas["n_phase3"].astype(str)
    colors = ["#009E73" if value < 0 else "#D55E00" for value in deltas["mean_abs_pos_error_delta"]]
    axes[1].bar(np.arange(len(deltas)), deltas["mean_abs_pos_error_delta"], color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", lw=1, linestyle="--")
    axes[1].set_xticks(np.arange(len(deltas)))
    axes[1].set_xticklabels(deltas["design"], rotation=0)
    axes[1].set_ylabel("Tuned - original absolute PoS error")
    axes[1].set_title("Paired design-level delta")
    fig.tight_layout()
    path = base_output / "figures" / "fig11_phasesyn_survival_tuning_delta.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def _replace_main_with_candidate(base_cfg: dict[str, Any], base_output: Path, candidate_output: Path, summary: dict[str, Any]) -> None:
    audit = _audit_replacement_inputs(base_output, candidate_output)
    comparison = _write_original_vs_tuned(base_output, candidate_output, summary)
    if not comparison["replacement_constraints_passed"]:
        raise RuntimeError(
            "Refusing to replace main PhaseSyn survival rows because tuned candidate failed "
            f"replacement constraints: {comparison['replacement_constraints_failed']}"
        )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = base_output / "tuning" / "survival_pos" / "main_backup" / timestamp
    backup_dir.mkdir(parents=True, exist_ok=True)
    for rel in [
        "method_pos_estimates.csv",
        "phasesyn_pos_estimates.csv",
        "model_paths.csv",
        "intermediate/virtual_trial_analyses.csv",
        "pos_accuracy_table.csv",
        "decision_utility_table.csv",
        "go_no_go_decision_metrics.csv",
        "design_ranking_accuracy.csv",
        "event_censoring_error_table.csv",
        "null_calibration_metrics.csv",
    ]:
        _copy_if_exists(base_output / rel, backup_dir / rel)
    if (base_output / "figures").exists():
        shutil.copytree(base_output / "figures", backup_dir / "figures", dirs_exist_ok=True)

    base_method = pd.read_csv(base_output / "method_pos_estimates.csv", keep_default_na=False)
    candidate_method = pd.read_csv(candidate_output / "method_pos_estimates.csv", keep_default_na=False)
    candidate_phase = candidate_method[candidate_method["method"].eq("PhaseSyn")].copy()
    updated_method = pd.concat(
        [base_method[base_method["method"].ne("PhaseSyn")].copy(), candidate_phase],
        ignore_index=True,
    )
    write_csv(base_output / "method_pos_estimates.csv", updated_method)
    write_csv(base_output / "phasesyn_pos_estimates.csv", candidate_phase.drop(columns=["method"], errors="ignore"))

    base_trials = pd.read_csv(base_output / "intermediate" / "virtual_trial_analyses.csv", keep_default_na=False)
    candidate_trials = pd.read_csv(candidate_output / "intermediate" / "virtual_trial_analyses.csv", keep_default_na=False)
    candidate_phase_trials = candidate_trials[candidate_trials["method"].eq("PhaseSyn")].copy()
    updated_trials = pd.concat(
        [base_trials[base_trials["method"].ne("PhaseSyn")].copy(), candidate_phase_trials],
        ignore_index=True,
    )
    write_csv(base_output / "intermediate" / "virtual_trial_analyses.csv", updated_trials)

    if (candidate_output / "model_paths.csv").exists():
        base_paths = pd.read_csv(base_output / "model_paths.csv", keep_default_na=False)
        candidate_paths = pd.read_csv(candidate_output / "model_paths.csv", keep_default_na=False)
        updated_paths = pd.concat(
            [
                base_paths[base_paths["method"].ne("PhaseSyn")].copy(),
                candidate_paths[candidate_paths["method"].eq("PhaseSyn")].copy(),
            ],
            ignore_index=True,
        )
        write_csv(base_output / "model_paths.csv", updated_paths)
    evaluate_pos(base_cfg, base_output)
    figures = make_figures(base_cfg, base_output)
    figures.append(_make_tuning_delta_figure(base_output))
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "selected_candidate_output": str(candidate_output),
        "backup_dir": str(backup_dir),
        "summary": summary,
        "replacement_audit": audit,
        "original_vs_tuned_comparison": comparison,
        "figures": figures,
        "phasesyn_worker_devices": phasesyn_worker_devices(base_cfg),
        "git": _git_metadata(Path(__file__).resolve().parents[2]),
    }
    dump_json(base_output / "survival_tuning_manifest.json", manifest)


def _load_existing_candidate_summary(candidate_output: Path) -> dict[str, Any]:
    manifest_path = candidate_output / "survival_tuning_candidate_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        summary = dict(manifest.get("summary", {}))
        if summary:
            return summary
    return _summarize_output(candidate_output.name, candidate_output, screen=False)


def _path_from_manifest(path_value: str) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    text = str(path)
    if text.startswith("/project/Stat/"):
        alias = Path("/lustre/project/Stat/" + text[len("/project/Stat/") :])
        if alias.exists():
            return alias
    if text.startswith("/lustre/project/Stat/"):
        alias = Path("/project/Stat/" + text[len("/lustre/project/Stat/") :])
        if alias.exists():
            return alias
    return path


def _phase_metric_row(label: str, path: Path, oracle: pd.DataFrame, *, selected: bool = False) -> dict[str, Any]:
    estimates = pd.read_csv(path / "method_pos_estimates.csv", keep_default_na=False)
    phase = estimates[estimates["method"].eq("PhaseSyn")].copy()
    merged = phase.merge(oracle, on=["scenario", "n_phase3"], how="left")
    err = merged["pos_hat"].to_numpy(dtype=float) - merged["true_pos"].to_numpy(dtype=float)
    null_err = merged.loc[merged["scenario"].eq("null"), "pos_hat"].to_numpy(dtype=float) - merged.loc[
        merged["scenario"].eq("null"), "true_pos"
    ].to_numpy(dtype=float)
    acc_path = path / "pos_accuracy_table.csv"
    if not acc_path.exists():
        acc_path = path / "tables" / "pos_bias_rmse_mae.csv"
    design_mean_rmse = float("nan")
    if acc_path.exists():
        acc = pd.read_csv(acc_path, keep_default_na=False)
        phase_acc = acc[acc["method"].eq("PhaseSyn")]
        if len(phase_acc):
            design_mean_rmse = float(phase_acc["pos_rmse"].mean())
    decision_accuracy = float("nan")
    decision_path = path / "go_no_go_decision_metrics.csv"
    if decision_path.exists():
        decision = pd.read_csv(decision_path, keep_default_na=False)
        phase_decision = decision[decision["method"].eq("PhaseSyn")]
        if len(phase_decision):
            decision_accuracy = float(phase_decision["decision_accuracy"].iloc[0])
    ranking_accuracy = float("nan")
    ranking_path = path / "design_ranking_accuracy.csv"
    if ranking_path.exists():
        ranking = pd.read_csv(ranking_path, keep_default_na=False)
        phase_ranking = ranking[ranking["method"].eq("PhaseSyn")]
        if len(phase_ranking):
            ranking_accuracy = float(phase_ranking["ranking_accuracy"].iloc[0])
    return {
        "candidate": label,
        "selected_for_main": bool(selected),
        "source_output": str(path),
        "phase_rows": int(len(merged)),
        "direct_pos_rmse": float(np.sqrt(np.mean(np.square(err)))),
        "direct_pos_mae": float(np.mean(np.abs(err))),
        "direct_abs_bias": float(abs(np.mean(err))),
        "direct_null_rmse": float(np.sqrt(np.mean(np.square(null_err)))),
        "design_mean_pos_rmse": design_mean_rmse,
        "event_rate_error": float(
            np.mean(np.abs(merged["event_rate_hat"].to_numpy(dtype=float) - merged["true_event_rate"].to_numpy(dtype=float)))
        ),
        "censoring_rate_error": float(
            np.mean(
                np.abs(merged["censoring_rate_hat"].to_numpy(dtype=float) - merged["true_censoring_rate"].to_numpy(dtype=float))
            )
        ),
        "decision_accuracy": decision_accuracy,
        "ranking_accuracy": ranking_accuracy,
    }


def _bootstrap_metric(
    frame: pd.DataFrame,
    metric: str,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if metric == "pooled_pos_rmse":
        use = frame.copy()

        def calc(g: pd.DataFrame, version: str) -> float:
            err = g[f"pos_hat_{version}"].to_numpy(dtype=float) - g["true_pos"].to_numpy(dtype=float)
            return float(np.sqrt(np.mean(np.square(err))))

    elif metric == "pooled_pos_mae":
        use = frame.copy()

        def calc(g: pd.DataFrame, version: str) -> float:
            err = g[f"pos_hat_{version}"].to_numpy(dtype=float) - g["true_pos"].to_numpy(dtype=float)
            return float(np.mean(np.abs(err)))

    elif metric == "null_pos_rmse":
        use = frame[frame["scenario"].eq("null")].copy()

        def calc(g: pd.DataFrame, version: str) -> float:
            err = g[f"pos_hat_{version}"].to_numpy(dtype=float) - g["true_pos"].to_numpy(dtype=float)
            return float(np.sqrt(np.mean(np.square(err))))

    elif metric == "event_rate_error":
        use = frame.copy()

        def calc(g: pd.DataFrame, version: str) -> float:
            return float(g[f"event_rate_error_{version}"].mean())

    elif metric == "censoring_rate_error":
        use = frame.copy()

        def calc(g: pd.DataFrame, version: str) -> float:
            return float(g[f"censoring_rate_error_{version}"].mean())

    else:
        raise KeyError(metric)

    original = calc(use, "original")
    tuned = calc(use, "tuned")
    delta = tuned - original
    rng = np.random.default_rng(seed)
    n = len(use)
    boot = np.empty(int(n_bootstrap), dtype=float)
    for i in range(int(n_bootstrap)):
        sampled = use.iloc[rng.integers(0, n, size=n)]
        boot[i] = calc(sampled, "tuned") - calc(sampled, "original")
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
    return {
        "metric": metric,
        "original": original,
        "tuned": tuned,
        "delta_tuned_minus_original": delta,
        "bootstrap_ci95_low": float(ci_low),
        "bootstrap_ci95_high": float(ci_high),
        "bootstrap_pr_improved": float(np.mean(boot < 0.0)),
        "ci_crosses_zero": bool(ci_low <= 0.0 <= ci_high),
        "n_pairs": int(n),
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def write_survival_tuning_reporting_tables(base_output: Path, *, n_bootstrap: int = 10000, seed: int = 20260618) -> dict[str, Path]:
    manifest_path = base_output / "survival_tuning_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    selected = str(manifest["summary"]["candidate"])
    oracle = pd.read_csv(base_output / "oracle_true_pos.csv", keep_default_na=False)

    rows = []
    backup_dir = _path_from_manifest(str(manifest["backup_dir"]))
    if (backup_dir / "method_pos_estimates.csv").exists():
        rows.append(_phase_metric_row("original", backup_dir, oracle, selected=False))
    full_dir = base_output / "tuning" / "survival_pos" / "full"
    for candidate_dir in sorted(p for p in full_dir.iterdir() if p.is_dir()):
        if (candidate_dir / "method_pos_estimates.csv").exists():
            rows.append(_phase_metric_row(candidate_dir.name, candidate_dir, oracle, selected=(candidate_dir.name == selected)))
    candidate_frame = pd.DataFrame(rows).sort_values(["direct_pos_rmse", "candidate"]).reset_index(drop=True)
    candidate_path = base_output / "tables" / "survival_tuning_candidate_comparison.csv"
    write_csv(candidate_path, candidate_frame)

    paired_path = base_output / "tables" / "survival_tuning_paired_deltas.csv"
    if not paired_path.exists():
        raise FileNotFoundError(paired_path)
    paired = pd.read_csv(paired_path, keep_default_na=False)
    metrics = ["pooled_pos_rmse", "pooled_pos_mae", "null_pos_rmse", "event_rate_error", "censoring_rate_error"]
    boot = pd.DataFrame(
        [_bootstrap_metric(paired, metric, n_bootstrap=n_bootstrap, seed=seed + i) for i, metric in enumerate(metrics)]
    )
    boot_path = base_output / "tables" / "survival_tuning_paired_bootstrap_ci.csv"
    write_csv(boot_path, boot)
    return {"candidate_comparison": candidate_path, "paired_bootstrap_ci": boot_path}


def _write_summary_table(base_output: Path, rows: list[dict[str, Any]], screen: bool) -> Path:
    summary_dir = base_output / "tuning" / "survival_pos"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / ("survival_tuning_screen_summary.csv" if screen else "survival_tuning_full_summary.csv")
    frame = pd.DataFrame(rows).sort_values("mean_pos_rmse").reset_index(drop=True)
    write_csv(path, frame)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune PhaseSyn survival-endpoint PoS performance.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--candidate", action="append", choices=sorted(CANDIDATES), help="Candidate name. Repeat to run several.")
    parser.add_argument("--screen", action="store_true", help="Run a small candidate screen.")
    parser.add_argument("--full", action="store_true", help="Run full candidate evaluation.")
    parser.add_argument("--force", action="store_true", help="Delete candidate output before rerun.")
    parser.add_argument("--replace-main", action="store_true", help="Replace main PhaseSyn survival rows/figures with the best full candidate.")
    parser.add_argument("--replace-from-candidate-output", default=None, help="Replace main outputs from an already completed candidate output directory.")
    parser.add_argument("--write-reporting-tables", action="store_true", help="Write candidate comparison and paired bootstrap CI tables.")
    parser.add_argument("--reporting-bootstrap-samples", type=int, default=10000)
    parser.add_argument(
        "--replacement-selection-metric",
        default="direct_pos_rmse",
        choices=["direct_pos_rmse", "mean_pos_rmse", "mean_pos_mae"],
        help="Metric used to select the best full candidate when --replace-main is used.",
    )
    parser.add_argument("--screen-reps-per-scenario", type=int, default=4)
    parser.add_argument("--screen-m-syn", type=int, default=100)
    parser.add_argument("--screen-epochs", type=int, default=100)
    parser.add_argument("--screen-virtual-batch-size", type=int, default=20)
    parser.add_argument("--full-m-syn", type=int, default=500)
    parser.add_argument("--full-epochs", type=int, default=None)
    parser.add_argument("--full-virtual-batch-size", type=int, default=20)
    args = parser.parse_args(argv)

    candidates = args.candidate or list(CANDIDATES)
    base_cfg = load_config(args.config)
    base_output = Path(base_cfg["output_dir"])
    if not base_output.exists():
        raise FileNotFoundError(base_output)

    if args.write_reporting_tables:
        paths = write_survival_tuning_reporting_tables(
            base_output,
            n_bootstrap=int(args.reporting_bootstrap_samples),
        )
        for path in paths.values():
            print(path)
        return

    if not args.screen and not args.full:
        args.screen = True

    diagnostics_path = write_survival_diagnostics(base_output)
    print(diagnostics_path)

    if args.replace_from_candidate_output:
        candidate_output = Path(args.replace_from_candidate_output)
        if not candidate_output.exists():
            raise FileNotFoundError(candidate_output)
        summary = _load_existing_candidate_summary(candidate_output)
        _replace_main_with_candidate(base_cfg, base_output, candidate_output, summary)
        print(base_output / "survival_tuning_manifest.json")
        return

    summaries: list[dict[str, Any]] = []
    if args.screen:
        for candidate in candidates:
            summaries.append(run_candidate(base_cfg, base_output, candidate, args, screen=True, force=args.force))
        path = _write_summary_table(base_output, summaries, screen=True)
        print(path)

    full_summaries: list[dict[str, Any]] = []
    if args.full:
        for candidate in candidates:
            full_summaries.append(run_candidate(base_cfg, base_output, candidate, args, screen=False, force=args.force))
        path = _write_summary_table(base_output, full_summaries, screen=False)
        print(path)
        if args.replace_main:
            best = min(full_summaries, key=lambda row: row[args.replacement_selection_metric])
            _replace_main_with_candidate(base_cfg, base_output, Path(best["output_dir"]), best)
            print(base_output / "survival_tuning_manifest.json")


if __name__ == "__main__":
    main(sys.argv[1:])
