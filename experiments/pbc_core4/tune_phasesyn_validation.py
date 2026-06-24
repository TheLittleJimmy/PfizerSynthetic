from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .load_pbc import TREATMENT_NAME, load_processed, project_path
from .methods import PhaseSynGenerator, split_static_long
from .metrics import (
    BASELINE_COLUMNS,
    longitudinal_fidelity,
    longitudinal_reference,
    survival_fidelity,
    survival_reference,
)


DEFAULT_CANDIDATES: list[dict[str, Any]] = [
    {
        "candidate_id": "c00_current_small_regularized",
        "description": "Current small model with the existing regularization balance.",
        "phasesyn": {
            "lr": 0.0011,
            "z_dim": 4,
            "s_dim": 4,
            "u_dim": 4,
            "y_dim_static": 4,
            "gru_hidden_dim": 6,
            "ode_hidden_dim": 6,
            "decoder_hidden_dim": 6,
            "u0_initializer_hidden_dim": 6,
            "dynamic_survival_hidden_dim": 6,
            "lambda_surv": 1.6,
            "kl_weight_s": 0.15,
            "kl_weight_z": 0.15,
            "longitudinal_weight": 1.2,
            "continuous_mse_weight": 0.5,
            "n_intervals": 10,
        },
    },
    {
        "candidate_id": "c01_longitudinal_low_kl",
        "description": "More capacity, lower KL pressure, and stronger continuous longitudinal reconstruction.",
        "phasesyn": {
            "lr": 0.0009,
            "z_dim": 6,
            "s_dim": 6,
            "u_dim": 6,
            "y_dim_static": 6,
            "gru_hidden_dim": 10,
            "ode_hidden_dim": 10,
            "decoder_hidden_dim": 10,
            "u0_initializer_hidden_dim": 10,
            "dynamic_survival_hidden_dim": 10,
            "lambda_surv": 1.2,
            "kl_weight_s": 0.05,
            "kl_weight_z": 0.05,
            "longitudinal_weight": 2.0,
            "continuous_mse_weight": 1.2,
            "n_intervals": 12,
        },
    },
    {
        "candidate_id": "c02_balanced_capacity",
        "description": "Moderate capacity with balanced longitudinal and survival terms.",
        "phasesyn": {
            "lr": 0.0008,
            "z_dim": 8,
            "s_dim": 6,
            "u_dim": 8,
            "y_dim_static": 6,
            "gru_hidden_dim": 12,
            "ode_hidden_dim": 12,
            "decoder_hidden_dim": 12,
            "u0_initializer_hidden_dim": 12,
            "dynamic_survival_hidden_dim": 12,
            "lambda_surv": 1.6,
            "kl_weight_s": 0.04,
            "kl_weight_z": 0.04,
            "longitudinal_weight": 1.8,
            "continuous_mse_weight": 1.0,
            "n_intervals": 12,
        },
    },
    {
        "candidate_id": "c03_survival_emphasis",
        "description": "Larger survival head weight for event-time and KM calibration.",
        "phasesyn": {
            "lr": 0.0008,
            "z_dim": 6,
            "s_dim": 6,
            "u_dim": 6,
            "y_dim_static": 6,
            "gru_hidden_dim": 10,
            "ode_hidden_dim": 10,
            "decoder_hidden_dim": 10,
            "u0_initializer_hidden_dim": 10,
            "dynamic_survival_hidden_dim": 12,
            "lambda_surv": 2.4,
            "kl_weight_s": 0.05,
            "kl_weight_z": 0.05,
            "longitudinal_weight": 1.4,
            "continuous_mse_weight": 0.8,
            "n_intervals": 16,
        },
    },
    {
        "candidate_id": "c04_conservative_low_kl",
        "description": "Small-to-moderate capacity with low KL and strong longitudinal scale matching.",
        "phasesyn": {
            "lr": 0.0007,
            "z_dim": 6,
            "s_dim": 4,
            "u_dim": 6,
            "y_dim_static": 4,
            "gru_hidden_dim": 8,
            "ode_hidden_dim": 8,
            "decoder_hidden_dim": 8,
            "u0_initializer_hidden_dim": 8,
            "dynamic_survival_hidden_dim": 8,
            "lambda_surv": 1.4,
            "kl_weight_s": 0.02,
            "kl_weight_z": 0.02,
            "longitudinal_weight": 2.4,
            "continuous_mse_weight": 1.5,
            "n_intervals": 12,
        },
    },
]


SCORE_WEIGHTS = {
    "longitudinal_mean_trajectory_error": 2.0,
    "longitudinal_change_from_baseline_error": 1.5,
    "longitudinal_slope_distribution_error": 1.0,
    "survival_km_integrated_abs_distance": 2.0,
    "abs_survival_event_rate_error": 1.0,
    "abs_survival_rmst_difference": 1.0,
    "abs_survival_median_followup_error": 0.5,
}


def _baseline_target(static: pd.DataFrame) -> pd.DataFrame:
    cols = ["subject_id", TREATMENT_NAME, *[c for c in BASELINE_COLUMNS if c in static]]
    return static[[c for c in cols if c in static]].copy().reset_index(drop=True)


def _observed_time_grid(long_df: pd.DataFrame) -> np.ndarray:
    if long_df.empty or "visit_time" not in long_df:
        return np.asarray([0.0, 1.0, 2.0, 3.0, 5.0], dtype=float)
    times = pd.to_numeric(long_df["visit_time"], errors="coerce").dropna().to_numpy(dtype=float)
    times = np.asarray(sorted(set(float(x) for x in times if np.isfinite(x))), dtype=float)
    if times.size == 0:
        return np.asarray([0.0, 1.0, 2.0, 3.0, 5.0], dtype=float)
    if not np.any(np.isclose(times, 0.0)):
        times = np.concatenate([[0.0], times])
    return times


def _post_baseline_longitudinal(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty or "visit_time" not in long_df:
        return long_df.copy()
    out = long_df[pd.to_numeric(long_df["visit_time"], errors="coerce") > 1e-8].copy()
    return out.reset_index(drop=True)


def _split_by_replicate(static: pd.DataFrame, long_df: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    if "replicate" not in static:
        return [(0, static.reset_index(drop=True), long_df.reset_index(drop=True))]
    rows: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    for rep, s in static.groupby("replicate", sort=True):
        if "replicate" in long_df:
            l = long_df[long_df["replicate"].eq(rep)].copy()
        elif long_df.empty or "subject_id" not in long_df:
            l = long_df.copy()
        else:
            ids = set(pd.to_numeric(s["subject_id"], errors="coerce").dropna().astype(int))
            l = long_df[pd.to_numeric(long_df["subject_id"], errors="coerce").astype("Int64").isin(ids)].copy()
        rows.append((
            int(rep),
            s.drop(columns=["replicate"], errors="ignore").reset_index(drop=True),
            l.drop(columns=["replicate"], errors="ignore").reset_index(drop=True),
        ))
    return rows


def _generate_validation_replicates(
    generator: PhaseSynGenerator,
    val_static: pd.DataFrame,
    val_time_grid: np.ndarray,
    replicates: int,
    batch_size: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    base = _baseline_target(val_static)
    out: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    stride = len(base) + 1
    for start in range(0, int(replicates), int(batch_size)):
        batch_reps = list(range(start, min(int(replicates), start + int(batch_size))))
        targets = []
        for rep in batch_reps:
            target = base.copy()
            target["replicate"] = int(rep)
            target["subject_id"] = np.arange(len(target), dtype=int) + int(rep) * stride
            targets.append(target)
        target_all = pd.concat(targets, ignore_index=True)
        static, long_df, _ = generator.generate(
            len(target_all),
            treatment=None,
            target_baseline=target_all,
            time_grid=val_time_grid,
        )
        out.extend(_split_by_replicate(static, long_df))
    return out


def _evaluate_candidate(
    candidate: dict[str, Any],
    base_cfg: dict[str, Any],
    data: Any,
    output_root: Path,
    epochs: int,
    validation_replicates: int,
    validation_batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidate_id = str(candidate["candidate_id"])
    cfg = copy.deepcopy(base_cfg)
    cfg["output_dir"] = str(output_root / "candidates" / candidate_id)
    cfg["seed"] = int(base_cfg["seed"]) + 1009 * (1 + int(candidate.get("candidate_index", 0)))
    cfg.setdefault("phasesyn", {}).update(candidate["phasesyn"])
    cfg["phasesyn"]["epochs"] = int(epochs)

    candidate_dir = project_path(cfg["output_dir"])
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "candidate_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    train_static, train_long = split_static_long(data, "train")
    val_static, val_long = split_static_long(data, "validation")
    generator = PhaseSynGenerator(cfg, train_static, train_long, candidate_dir, int(cfg["seed"]))

    started = time.time()
    status = generator.train(smoke=False)
    train_seconds = time.time() - started
    status = dict(status)
    status.update({
        "candidate_id": candidate_id,
        "candidate_description": candidate.get("description", ""),
        "train_seconds": float(train_seconds),
        "epochs": int(epochs),
    })
    if status.get("status") != "completed":
        return [], status, cfg

    val_long_eval = _post_baseline_longitudinal(val_long)
    val_time_grid = _observed_time_grid(val_long)
    long_ref = longitudinal_reference(val_long_eval)
    surv_ref = survival_reference(val_static)
    rows: list[dict[str, Any]] = []
    for rep, static, long_df in _generate_validation_replicates(generator, val_static, val_time_grid, validation_replicates, validation_batch_size):
        long_eval = _post_baseline_longitudinal(long_df)
        row = {
            "candidate_id": candidate_id,
            "candidate_description": candidate.get("description", ""),
            "replicate": int(rep),
            "validation_split": "validation",
            "generation_mode": "baseline_conditioned_validation",
            "target_baseline_used": True,
            "longitudinal_scoring_scope": "post_baseline_only",
            "validation_time_grid_size": int(len(val_time_grid)),
            "epochs": int(epochs),
        }
        row.update(longitudinal_fidelity(val_long_eval, long_eval, "PhaseSyn", int(rep), f"{candidate_id},rep={rep}", real_reference=long_ref))
        row.update(survival_fidelity(val_static, static, "PhaseSyn", int(rep), f"{candidate_id},rep={rep}", real_reference=surv_ref))
        rows.append(row)
    return rows, status, cfg


def _score_candidates(metrics: pd.DataFrame, statuses: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return statuses.copy()
    work = metrics.copy()
    for col in ["survival_event_rate_error", "survival_rmst_difference", "survival_median_followup_error"]:
        if col in work:
            work[f"abs_{col}"] = pd.to_numeric(work[col], errors="coerce").abs()
    score_cols = [c for c in SCORE_WEIGHTS if c in work]
    summary = work.groupby("candidate_id", as_index=False)[score_cols].mean(numeric_only=True)
    for col in score_cols:
        vals = pd.to_numeric(summary[col], errors="coerce").abs()
        scale = float(np.nanmedian(vals.to_numpy(dtype=float)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = float(np.nanstd(vals.to_numpy(dtype=float)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = 1.0
        summary[f"scaled_{col}"] = vals / scale
    denom = sum(SCORE_WEIGHTS[c] for c in score_cols)
    if denom <= 0:
        summary["validation_score"] = np.nan
    else:
        weighted_terms = []
        weight_terms = []
        for col in score_cols:
            scaled = pd.to_numeric(summary[f"scaled_{col}"], errors="coerce")
            finite = scaled.notna()
            weighted_terms.append(scaled.fillna(0.0) * SCORE_WEIGHTS[col])
            weight_terms.append(finite.astype(float) * SCORE_WEIGHTS[col])
        numerator = sum(weighted_terms)
        denominator = sum(weight_terms)
        summary["validation_score"] = numerator / denominator.replace(0.0, np.nan)
    if not statuses.empty:
        keep = [c for c in ["candidate_id", "candidate_description", "status", "train_seconds", "epochs", "checkpoint"] if c in statuses]
        summary = summary.merge(statuses[keep], on="candidate_id", how="left")
    return summary.sort_values("validation_score", na_position="last").reset_index(drop=True)


def _plot_summary(summary: pd.DataFrame, output_root: Path) -> None:
    if summary.empty or "validation_score" not in summary:
        return
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    view = summary.sort_values("validation_score", ascending=True)
    labels = view["candidate_id"].astype(str).to_list()
    scores = pd.to_numeric(view["validation_score"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(labels, scores, color="#4C78A8")
    ax.set_ylabel("Validation score, lower is better")
    ax.set_xlabel("PhaseSyn hyperparameter candidate")
    ax.set_title("PhaseSyn validation tuning score")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "phasesyn_tuning_validation_score.pdf", bbox_inches="tight")
    plt.close(fig)


def _write_selected_config(
    base_cfg: dict[str, Any],
    best_row: pd.Series,
    candidates: list[dict[str, Any]],
    output_root: Path,
) -> Path:
    best_id = str(best_row["candidate_id"])
    candidate = next(c for c in candidates if c["candidate_id"] == best_id)
    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("phasesyn", {}).update(candidate["phasesyn"])
    cfg["phasesyn"]["epochs"] = int(best_row.get("epochs", cfg["phasesyn"].get("epochs", 220)))
    cfg["output_dir"] = "outputs/pbc_experiments/experiment_20260604_core4_tuned"
    cfg.setdefault("tuning", {})["selected_from"] = str(output_root)
    cfg["tuning"]["selected_candidate_id"] = best_id
    cfg["tuning"]["selection_metric"] = "baseline_conditioned_validation_score"
    out_path = Path("experiments/pbc_core4/config_pbc_core4_tuned.yaml")
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (output_root / "config_pbc_core4_tuned.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def tune(args: argparse.Namespace) -> dict[str, Any]:
    base_cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    output_root = project_path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "tables").mkdir(parents=True, exist_ok=True)
    data = load_processed(base_cfg["processed_data_dir"], int(base_cfg["seed"]))
    candidates = copy.deepcopy(DEFAULT_CANDIDATES[: int(args.max_candidates)])
    for idx, candidate in enumerate(candidates):
        candidate["candidate_index"] = idx

    metrics_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    candidate_cfgs: dict[str, Any] = {}
    for candidate in candidates:
        print(f"candidate {candidate['candidate_id']}: {candidate.get('description', '')}", flush=True)
        try:
            rows, status, cfg = _evaluate_candidate(
                candidate,
                base_cfg,
                data,
                output_root,
                int(args.epochs),
                int(args.validation_replicates),
                max(1, int(args.validation_batch_size)),
            )
            metrics_rows.extend(rows)
            status_rows.append(status)
            candidate_cfgs[str(candidate["candidate_id"])] = cfg
        except Exception as exc:
            status_rows.append({
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_description": candidate.get("description", ""),
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
                "epochs": int(args.epochs),
            })

    metrics = pd.DataFrame(metrics_rows)
    statuses = pd.DataFrame(status_rows)
    summary = _score_candidates(metrics, statuses)
    metrics.to_csv(output_root / "tables" / "phasesyn_tuning_validation_metrics.csv", index=False)
    statuses.to_csv(output_root / "tables" / "phasesyn_tuning_status.csv", index=False)
    summary.to_csv(output_root / "tables" / "phasesyn_tuning_summary.csv", index=False)
    _plot_summary(summary, output_root)

    selected_config = ""
    selected_checkpoint = ""
    if not summary.empty and summary["validation_score"].notna().any():
        best = summary.iloc[0]
        selected_config = str(_write_selected_config(base_cfg, best, candidates, output_root))
        checkpoint = project_path(candidate_cfgs[str(best["candidate_id"])]["output_dir"]) / "phasesyn_model" / "train" / "model_checkpoint.pt"
        selected_checkpoint = str(checkpoint)
        selected_model_dir = output_root / "selected_phasesyn_model"
        if checkpoint.exists():
            selected_model_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint, selected_model_dir / "model_checkpoint.pt")
    manifest = {
        "output_root": str(output_root),
        "n_candidates": len(candidates),
        "epochs": int(args.epochs),
        "validation_replicates": int(args.validation_replicates),
        "selected_config": selected_config,
        "selected_checkpoint": selected_checkpoint,
        "status": "completed" if selected_config else "no_successful_candidate",
    }
    (output_root / "tuning_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tune PhaseSyn hyperparameters on the PBC validation split.")
    parser.add_argument("--config", type=Path, default=Path("experiments/pbc_core4/config_pbc_core4.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pbc_experiments/experiment_20260604_core4_tuning"))
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--validation-replicates", type=int, default=5)
    parser.add_argument("--validation-batch-size", type=int, default=5)
    args = parser.parse_args(argv)
    manifest = tune(args)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
