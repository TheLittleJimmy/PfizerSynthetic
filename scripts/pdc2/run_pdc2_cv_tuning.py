#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from evaluation.survival_plots import event_rate_metrics  # noqa: E402
from pdc2.config import load_config  # noqa: E402
from pdc2.models import PhaseSynModel, set_seed  # noqa: E402
from pdc2.training import paired_survival_metrics, train_model  # noqa: E402
from scripts.pdc2.run_compact_posterior_search import _count_parameters  # noqa: E402
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    _decode_baseline_conditioned_static,
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _future_generation_perturbation_audit,
    _future_longitudinal_metrics,
    _jsonable,
    _leakage_diagnostics,
    _make_bundle,
    _model_audit,
    _plot_longitudinal,
    _plot_metric_summary,
    _plot_survival,
    _read_raw_tables,
    _sample_longitudinal_future,
    _save_future_longitudinal,
    _summarize_metric_rows,
    _survival_generation_perturbation_audit,
)


CANDIDATES: dict[str, dict[str, Any]] = {
    "nano2_balanced": {
        "epochs": 180,
        "lr": 0.0012,
        "batch_size": 64,
        "z_dim": 2,
        "s_dim": 2,
        "y_dim_static": 4,
        "u_dim": 4,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 6,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.4,
        "continuous_mse_weight": 0.8,
    },
    "tiny4_balanced": {
        "epochs": 200,
        "lr": 0.0011,
        "batch_size": 64,
        "z_dim": 4,
        "s_dim": 4,
        "y_dim_static": 4,
        "u_dim": 4,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.4,
        "continuous_mse_weight": 0.8,
    },
    "tiny4_survival": {
        "epochs": 220,
        "lr": 0.0011,
        "batch_size": 64,
        "z_dim": 4,
        "s_dim": 4,
        "y_dim_static": 4,
        "u_dim": 4,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.15,
        "kl_weight_z": 0.15,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.5,
    },
    "mid6_regularized": {
        "epochs": 240,
        "lr": 0.0010,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.8,
    },
    "small8_reference": {
        "epochs": 260,
        "lr": 0.0010,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.4,
        "continuous_mse_weight": 0.8,
    },
}

PERFORMANCE_KEYS = [
    "survival_km_integrated_abs_error_mean",
    "event_rate_diff_mean_abs",
    "survival_time_rmse_ratio_mean",
    "survival_event_error_mean",
    "future_continuous_rmse_ratio_vs_l0_carryforward_mean",
    "future_continuous_ks_mean_mean",
    "future_categorical_error_mean",
    "future_categorical_tv_mean_mean",
]


def stratified_kfold_indices(raw: pd.DataFrame, folds: int, seed: int, treatment_col: str = "drug") -> list[np.ndarray]:
    if folds < 2:
        raise ValueError("folds must be at least 2.")
    labels = (
        pd.to_numeric(raw[treatment_col], errors="coerce").fillna(0).astype(int).astype(str)
        + "_"
        + (pd.to_numeric(raw["censor"], errors="coerce").fillna(0.0) > 0.5).astype(int).astype(str)
    )
    rng = np.random.default_rng(seed)
    fold_parts: list[list[np.ndarray]] = [[] for _ in range(folds)]
    for _, group in labels.groupby(labels, sort=True):
        idx = group.index.to_numpy(dtype=int)
        rng.shuffle(idx)
        for fold, part in enumerate(np.array_split(idx, folds)):
            fold_parts[fold].append(part.astype(int))
    out = []
    for parts in fold_parts:
        idx = np.concatenate(parts).astype(int) if parts else np.asarray([], dtype=int)
        rng.shuffle(idx)
        out.append(idx)
    return out


def write_fold_file(raw: pd.DataFrame, fold_indices: list[np.ndarray], path: Path) -> None:
    rows = []
    for fold_id, idx in enumerate(fold_indices):
        for row_index in idx:
            rows.append({
                "fold": int(fold_id),
                "row_index": int(row_index),
                "drug": int(pd.to_numeric(raw.iloc[int(row_index)]["drug"], errors="coerce")),
                "censor": float(pd.to_numeric(raw.iloc[int(row_index)]["censor"], errors="coerce")),
                "time": float(pd.to_numeric(raw.iloc[int(row_index)]["time"], errors="coerce")),
            })
    pd.DataFrame(rows).sort_values(["fold", "row_index"]).to_csv(path, index=False)


def candidate_config(candidate: str, args: argparse.Namespace, seed: int | None = None) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate!r}; expected one of {sorted(CANDIDATES)}.")
    spec = dict(CANDIDATES[candidate])
    if getattr(args, "epochs_override", None) is not None:
        spec["epochs"] = int(args.epochs_override)
    cfg = load_config(args.config, {
        "dataset": {"name": args.dataset},
        "model": {
            "longitudinal_mode": "latent_ode",
            "survival": "dynamic",
            "z_dim": spec["z_dim"],
            "s_dim": spec["s_dim"],
            "y_dim_static": spec["y_dim_static"],
            "u_dim": spec["u_dim"],
            "gru_hidden_dim": int(spec.get("gru_hidden_dim", spec["decoder_hidden_dim"])),
            "ode_hidden_dim": spec["ode_hidden_dim"],
            "decoder_hidden_dim": spec["decoder_hidden_dim"],
            "n_intervals": spec["n_intervals"],
            "use_randomization_loss": True,
            "randomization_loss_weight": 0.05,
            "randomization_loss_warmup_epochs": 0,
            "randomization_loss_ramp_epochs": 1,
            "randomization_mmd_bandwidths": "0.5,1.0,2.0,4.0",
            "randomization_loss_on": "z_mean",
            "u0_init_mode": "baseline_l0",
            "encoder_conditioning": "baseline_only",
            "detach_l0_for_u0_init": False,
            "baseline_time_eps": 1e-6,
            "lambda_l0_hivae": 1.0,
            "baseline_long_weight": 1.0,
            "condition_ode_on_baseline": True,
            "condition_longitudinal_decoder_on_baseline": True,
            "generation_baseline_mode": "sampled",
            "deterministic_u": True,
            "longitudinal_only_loss": False,
            "kl_weight_s": spec["kl_weight_s"],
            "kl_weight_z": spec["kl_weight_z"],
            "kl_weight_u": 0.0,
            "static_weight": spec["static_weight"],
            "longitudinal_weight": spec["longitudinal_weight"],
            "lambda_surv": spec["lambda_surv"],
            "continuous_mse_weight": spec["continuous_mse_weight"],
        },
        "training": {
            "epochs": spec["epochs"],
            "batch_size": spec["batch_size"],
            "lr": spec["lr"],
            "seed": int(args.seed if seed is None else seed),
            "device": args.device,
            "n_generated_dataset": 1,
            "early_stopping": False,
            "subset_size": None,
            "freeze_normalization": True,
        },
        "evaluation": {
            "deterministic_static_export": False,
            "copy_static_overfit_reference": False,
            "calibrate_static_covariates": False,
            "copy_survival_overfit_reference": False,
            "calibrate_survival_km": False,
            "calibrate_survival_event_rate": False,
            "calibrate_longitudinal_observed": False,
            "posterior_generation": True,
            "n_replicates": int(args.n_replicates),
        },
    })
    cfg["cv_candidate"] = candidate
    cfg["cv_candidate_spec"] = spec
    return cfg


def _train_loss_summary(curves: pd.DataFrame) -> dict[str, float]:
    loss = pd.to_numeric(curves.get("loss", pd.Series(dtype=float)), errors="coerce").dropna()
    out = {
        "train_final_loss": float(loss.iloc[-1]) if not loss.empty else math.nan,
        "train_loss_decrease": 0.0,
        "nan_epoch_count": int(curves["nan_epoch"].astype(bool).sum()) if "nan_epoch" in curves else 0,
    }
    if len(loss) >= 2:
        out["train_loss_decrease"] = float((loss.iloc[0] - loss.iloc[-1]) / max(abs(loss.iloc[0]), 1e-8))
    return out


def _run_validation_replicates(
    model: PhaseSynModel,
    val_bundle,
    device: torch.device,
    output_dir: Path,
    n_replicates: int,
    seed: int,
    deterministic_longitudinal: bool,
    save_artifacts: bool,
) -> tuple[dict[str, Any], pd.DataFrame, list[pd.DataFrame], list[np.ndarray]]:
    rep_static = []
    rep_long_csv = []
    rep_metric_rows = []
    rep_pred_arrays = []
    generation_audits = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for rep in range(1, n_replicates + 1):
        rep_seed = seed + 1000 + rep
        torch.manual_seed(rep_seed)
        np.random.seed(rep_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rep_seed)
        syn_static, latents, gen_audit = _decode_baseline_conditioned_static(model, val_bundle, device)
        syn_static.insert(0, "replicate", rep)
        expected_ids = val_bundle.longitudinal.subject_ids.astype(int)
        if not np.array_equal(syn_static["patient_id"].to_numpy(dtype=int), expected_ids):
            raise RuntimeError("Generated static patient_id order does not match validation bundle order.")
        pred_raw, future_mask, baseline_idx, support = _sample_longitudinal_future(
            model,
            val_bundle,
            latents,
            device,
            sample=not deterministic_longitudinal,
        )
        long_df_rep = _save_future_longitudinal(
            val_bundle,
            pred_raw,
            future_mask,
            output_dir / f"synthetic_longitudinal_future_val_rep{rep:02d}.csv",
            rep,
        )
        long_metrics, _, _ = _future_longitudinal_metrics(val_bundle, pred_raw, future_mask, baseline_idx)
        survival_metrics = event_rate_metrics(val_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"]))
        survival_metrics.update(paired_survival_metrics(val_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"])))
        row: dict[str, float] = {"replicate": float(rep)}
        row.update({k: float(v) for k, v in survival_metrics.items()})
        row.update({k: float(v) for k, v in long_metrics.items()})
        row.update({k: float(v) for k, v in support.items()})
        rep_metric_rows.append(row)
        rep_static.append(syn_static)
        rep_long_csv.append(long_df_rep)
        rep_pred_arrays.append(pred_raw)
        generation_audits.append(gen_audit)
        if save_artifacts:
            syn_static.to_csv(output_dir / f"synthetic_static_val_rep{rep:02d}.csv", index=False)

    metrics_df = pd.DataFrame(rep_metric_rows)
    metrics_df.to_csv(output_dir / "validation_replicate_metrics.csv", index=False)
    summary = _summarize_metric_rows(rep_metric_rows, len(val_bundle.raw_df))
    summary["generation_audits_all_survival_zero"] = bool(
        all(x["test_survival_mask_zero_for_generation"] and x["test_survival_tensor_zero_for_generation"] for x in generation_audits)
    )
    if save_artifacts:
        pd.concat(rep_static, ignore_index=True).to_csv(output_dir / "validation_synthetic_static_all.csv", index=False)
        pd.concat(rep_long_csv, ignore_index=True).to_csv(output_dir / "validation_synthetic_longitudinal_future_all.csv", index=False)
    return summary, metrics_df, rep_static, rep_pred_arrays


def run_fold(
    candidate: str,
    fold_id: int,
    fold_indices: list[np.ndarray],
    raw: pd.DataFrame,
    ids: pd.DataFrame,
    long_df: pd.DataFrame,
    types: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_root) / candidate / f"fold_{fold_id:02d}"
    summary_path = output_dir / "fold_summary.json"
    if bool(args.resume) and summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = candidate_config(candidate, args, seed=int(args.seed) + fold_id)
    val_idx = np.asarray(fold_indices[fold_id], dtype=int)
    train_idx = np.setdiff1d(np.arange(len(raw), dtype=int), val_idx, assume_unique=False)
    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_prep = _fit_longitudinal_preprocessor(long_df, types, train_idx, cfg["dataset"].get("max_visits"))
    train_bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_prep, cfg)
    val_bundle = _make_bundle(raw, ids, long_df, types, val_idx, static_prep, long_prep, cfg)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable({
            "static_preprocessor": static_prep,
            "longitudinal_preprocessor": long_prep,
            "train_subject_count": int(len(train_idx)),
            "validation_subject_count": int(len(val_idx)),
            "preprocessing_fit_on_training_fold_only": True,
        }), f, indent=2)
    result = train_model(train_bundle, cfg, output_dir=output_dir / "train", overfit_name=None)
    model = result["model"].to(torch.device(cfg["training"].get("device", "cpu")))
    if not isinstance(model, PhaseSynModel):
        raise TypeError("PDC2 CV evaluation requires PhaseSynModel.")
    device = torch.device(cfg["training"].get("device", "cpu"))
    audit = _model_audit(cfg, model)
    audit.update(_leakage_diagnostics(model, val_bundle, device))
    audit.update(_survival_generation_perturbation_audit(model, val_bundle, device, int(args.seed) + 997 + fold_id))
    audit.update(_future_generation_perturbation_audit(model, val_bundle, device, int(args.seed) + 1997 + fold_id))
    audit.update({
        "fold": int(fold_id),
        "candidate": candidate,
        "train_subject_count": int(len(train_idx)),
        "validation_subject_count": int(len(val_idx)),
        "generation_batch_keys": ["B", "mask_B", "L0", "A", "future_times"],
        "forbidden_generation_inputs": ["validation_survival_time", "validation_censor", "validation_future_longitudinal_values"],
        "forbidden_generation_inputs_present": [],
        "hivae_uses_frozen_train_normalization": model.hivae._global_norm_params is not None,
    })
    audit["passes_audit"] = bool(
        audit["passes_audit"]
        and audit["survival_generation_invariant_to_test_survival"]
        and audit["future_generation_invariant_to_test_future_values"]
        and audit["hivae_uses_frozen_train_normalization"]
    )
    with open(output_dir / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(audit), f, indent=2)
    if not audit["passes_audit"]:
        raise RuntimeError(f"CV fold audit failed for {candidate} fold {fold_id}: {audit}")

    val_summary, _, rep_static, rep_pred_arrays = _run_validation_replicates(
        model,
        val_bundle,
        device,
        output_dir / "validation",
        int(args.n_replicates),
        int(args.seed) + 10_000 * (fold_id + 1),
        bool(args.deterministic_longitudinal),
        bool(args.save_replicate_artifacts),
    )
    if not args.skip_plots:
        _plot_survival(val_bundle.raw_df, [df.drop(columns=["replicate", "patient_id"]) for df in rep_static], output_dir / "figures")
        split = model.split_longitudinal_batch(
            val_bundle.longitudinal.times,
            val_bundle.longitudinal.values,
            val_bundle.longitudinal.masks,
        )
        future_mask = split["future_masks"].detach().cpu().numpy().astype(bool)
        _plot_longitudinal(val_bundle, rep_pred_arrays, future_mask, output_dir / "figures")
        _plot_metric_summary(val_summary, output_dir / "figures")

    summary: dict[str, Any] = {
        "candidate": candidate,
        "fold": int(fold_id),
        "train_subject_count": int(len(train_idx)),
        "validation_subject_count": int(len(val_idx)),
        "n_replicates": int(args.n_replicates),
        "passes_audit": bool(audit["passes_audit"]),
    }
    summary.update(_count_parameters(model))
    summary.update(_train_loss_summary(result["curves"]))
    summary.update(val_summary)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    return summary


def performance_score(row: pd.Series | dict[str, Any]) -> float:
    get = row.get if isinstance(row, dict) else row.get
    def metric(name: str, default: float = 0.0) -> float:
        value = get(name, None)
        if value is None:
            value = get(f"{name}_cv_mean", default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    survival_event_error = 1.0 - metric("survival_event_accuracy_mean")
    categorical_error = 1.0 - metric("future_categorical_accuracy_mean")
    event_rate_diff_abs = abs(metric("event_rate_diff_mean"))
    values = [
        metric("survival_km_integrated_abs_error_mean"),
        event_rate_diff_abs,
        metric("survival_time_rmse_ratio_mean"),
        survival_event_error,
        metric("future_continuous_rmse_ratio_vs_l0_carryforward_mean"),
        metric("future_continuous_ks_mean_mean"),
        categorical_error,
        metric("future_categorical_tv_mean_mean"),
    ]
    if any(not np.isfinite(v) for v in values):
        return math.inf
    return float(sum(values))


def summarize_cv(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows = []
    for path in sorted(output_root.glob("*/fold_*/fold_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        raise FileNotFoundError(f"No fold summaries found under {output_root}.")
    long_df = pd.DataFrame(rows)
    long_df.to_csv(output_root / "cv_results_long.csv", index=False)
    group_rows = []
    numeric = long_df.select_dtypes(include=[np.number]).columns.tolist()
    for candidate, group in long_df.groupby("candidate", sort=True):
        row: dict[str, Any] = {"candidate": candidate, "n_folds": int(len(group))}
        for col in numeric:
            if col == "fold":
                continue
            row[f"{col}_cv_mean"] = float(group[col].mean())
            row[f"{col}_cv_sd"] = float(group[col].std(ddof=0))
        row["all_folds_pass_audit"] = bool(group["passes_audit"].astype(bool).all()) if "passes_audit" in group else False
        row["performance_score"] = performance_score(row)
        group_rows.append(row)
    summary_df = pd.DataFrame(group_rows)
    if summary_df.empty:
        raise RuntimeError("No CV candidate summaries were produced.")
    min_params = float(summary_df["parameter_count_cv_mean"].min())
    max_params = float(summary_df["parameter_count_cv_mean"].max())
    if max_params > min_params:
        denom = math.log(max_params / min_params)
        summary_df["size_score"] = summary_df["parameter_count_cv_mean"].apply(lambda x: math.log(float(x) / min_params) / denom)
    else:
        summary_df["size_score"] = 0.0
    summary_df.loc[~summary_df["all_folds_pass_audit"], "performance_score"] = math.inf
    summary_df["compact_score"] = summary_df["performance_score"] + 0.15 * summary_df["size_score"]
    summary_df["latent_dim_sum"] = summary_df["candidate"].map(
        lambda name: int(CANDIDATES[name]["z_dim"] + CANDIDATES[name]["s_dim"] + CANDIDATES[name]["y_dim_static"] + CANDIDATES[name]["u_dim"])
    )
    summary_df = summary_df.sort_values(["compact_score", "parameter_count_cv_mean", "latent_dim_sum", "candidate"]).reset_index(drop=True)
    summary_df.to_csv(output_root / "cv_results_summary.csv", index=False)
    best = summary_df.iloc[0].to_dict()
    with open(output_root / "best_candidate.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(best), f, indent=2)
    with open(output_root / "cv_results_summary.md", "w", encoding="utf-8") as f:
        f.write("# PhaseSyn PDC2 Compact CV Tuning\n\n")
        f.write("Lower `compact_score` is better. Randomization loss is fixed at weight 0.05 for all candidates.\n\n")
        cols = [
            "candidate",
            "compact_score",
            "performance_score",
            "size_score",
            "parameter_count_cv_mean",
            "future_continuous_rmse_ratio_vs_l0_carryforward_mean_cv_mean",
            "survival_km_integrated_abs_error_mean_cv_mean",
            "survival_time_rmse_ratio_mean_cv_mean",
            "future_categorical_accuracy_mean_cv_mean",
            "all_folds_pass_audit",
        ]
        f.write(summary_df[[c for c in cols if c in summary_df.columns]].to_markdown(index=False))
        f.write(f"\n\nBest candidate: `{best['candidate']}`\n")
    return long_df, summary_df, best


def refit_best(best_candidate: str, raw: pd.DataFrame, ids: pd.DataFrame, long_df: pd.DataFrame, types: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_root) / "refit_best" / best_candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = candidate_config(best_candidate, args, seed=int(args.seed) + 99_000)
    idx = np.arange(len(raw), dtype=int)
    static_prep = _fit_static_preprocessor(raw.iloc[idx], types)
    long_prep = _fit_longitudinal_preprocessor(long_df, types, idx, cfg["dataset"].get("max_visits"))
    bundle = _make_bundle(raw, ids, long_df, types, idx, static_prep, long_prep, cfg)
    with open(output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    result = train_model(bundle, cfg, output_dir=output_dir / "train", overfit_name=None)
    model = result["model"].to(torch.device(cfg["training"].get("device", "cpu")))
    summary: dict[str, Any] = {
        "candidate": best_candidate,
        "refit_on_all_subjects": True,
        "n_subjects": int(len(raw)),
    }
    summary.update(_count_parameters(model))
    summary.update(_train_loss_summary(result["curves"]))
    with open(output_dir / "refit_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config(args.config, {"dataset": {"name": args.dataset}})
    raw, ids, long_df, types = _read_raw_tables(base_cfg)
    fold_indices = stratified_kfold_indices(raw, int(args.folds), int(args.seed), treatment_col="drug")
    write_fold_file(raw, fold_indices, output_root / "folds.csv")
    candidates = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    for candidate in candidates:
        for fold_id in range(int(args.folds)):
            run_fold(candidate, fold_id, fold_indices, raw, ids, long_df, types, args)
        summarize_cv(output_root)
    _, summary_df, best = summarize_cv(output_root)
    refit_summary = None
    if args.refit_best:
        refit_summary = refit_best(str(best["candidate"]), raw, ids, long_df, types, args)
    result = {
        "output_root": str(output_root),
        "n_candidates": int(len(candidates)),
        "folds": int(args.folds),
        "best_candidate": str(best["candidate"]),
        "best_compact_score": float(best["compact_score"]),
        "best_parameter_count_cv_mean": float(best["parameter_count_cv_mean"]),
        "refit_summary": refit_summary,
    }
    with open(output_root / "cv_tuning_run_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(result), f, indent=2)
    print(json.dumps(_jsonable(result), indent=2))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-validated compact hyperparameter tuning for the current PhaseSyn PDC2 model.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pdc2.yaml"))
    parser.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    parser.add_argument("--candidate", choices=[*CANDIDATES.keys(), "all"], default="all")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260526)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-replicates", type=int, default=10)
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--output-root", default="outputs/pdc2/experiments_20260526/cv_tuning_rand_0p05_small")
    parser.add_argument("--deterministic-longitudinal", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--save-replicate-artifacts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--refit-best", action="store_true")
    args = parser.parse_args(argv)
    if args.summarize_only:
        _, _, best = summarize_cv(Path(args.output_root))
        print(json.dumps(_jsonable(best), indent=2))
        return
    run(args)


if __name__ == "__main__":
    main()
