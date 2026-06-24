#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from pdc2.config import load_config  # noqa: E402
from pdc2.data import LongitudinalPanel, LongitudinalSpec, PDC2Bundle, validate_complete_l0, y_dim_partition_for_types  # noqa: E402
from pdc2.models import PhaseSynModel, set_seed  # noqa: E402
from pdc2.training import _apply_longitudinal_support, generate_prior_cohort, paired_survival_metrics, train_model  # noqa: E402
from evaluation.survival_plots import event_rate_metrics  # noqa: E402
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    StaticPreprocessor,
    _decode_baseline_conditioned_static,
    _feature_lists,
    _fit_static_preprocessor,
    _future_generation_perturbation_audit,
    _future_longitudinal_metrics,
    _jsonable,
    _leakage_diagnostics,
    _save_future_longitudinal,
    _stratified_split,
    _survival_generation_perturbation_audit,
    _transform_static,
)


SIM_DATA_DIR = ROOT / "data" / "simulation" / "simple_linear_rct_n1200"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "simulation"
TREATMENT_NAME = "A"
TREATMENT_N_CLASSES = 2


def _read_simulation(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    required = ["data_phasesyn.csv", "data_types_phasesyn_piecewise.csv", "longitudinal.csv", "metadata.json"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Simulation data directory {data_dir} is missing files: {missing}")
    raw = pd.read_csv(data_dir / "data_phasesyn.csv")
    types = pd.read_csv(data_dir / "data_types_phasesyn_piecewise.csv").fillna("").to_dict("records")
    for item in types:
        item["dim"] = str(int(float(item["dim"])))
        item["nclass"] = "" if item.get("nclass", "") == "" else str(int(float(item["nclass"])))
        if item["name"] == "survcens":
            item["type"] = "surv_dynamic"
    long_df = pd.read_csv(data_dir / "longitudinal.csv")
    if "patient_id" not in long_df.columns:
        raise ValueError("Simulation longitudinal.csv must contain patient_id.")
    if "simulation_id.csv" in {p.name for p in data_dir.iterdir()}:
        ids = pd.read_csv(data_dir / "simulation_id.csv")
        if "subject_id" not in ids.columns:
            ids.insert(0, "subject_id", np.arange(len(raw), dtype=int))
    else:
        ids = pd.DataFrame({"subject_id": np.arange(len(raw), dtype=int), "simulation_id": np.arange(len(raw), dtype=int)})
    metadata = json.loads((data_dir / "metadata.json").read_text())
    return raw, ids, long_df, types, metadata


def _longitudinal_names(types: list[dict[str, Any]]) -> list[str]:
    names = []
    for item in types:
        name = str(item["name"])
        if name not in {"survcens", TREATMENT_NAME} and name.startswith("L"):
            names.append(name)
    if not names:
        raise ValueError("No longitudinal L* features found in simulation type table.")
    return names


def _fit_longitudinal_preprocessor(
    long_df: pd.DataFrame,
    types: list[dict[str, Any]],
    train_subject_ids: np.ndarray,
    train_survival_times: pd.Series | np.ndarray,
) -> tuple[list[LongitudinalSpec], float, float, int]:
    train_set = set(int(x) for x in train_subject_ids)
    train_long = long_df[long_df["patient_id"].isin(train_set)].copy()
    type_map = {str(item["name"]): item for item in types}
    specs: list[LongitudinalSpec] = []
    for name in _longitudinal_names(types):
        obs = pd.to_numeric(train_long[name], errors="coerce").dropna().to_numpy(dtype=float)
        spec_type = str(type_map[name].get("type", "real"))
        nclass = type_map[name].get("nclass", "")
        specs.append(
            LongitudinalSpec(
                name=name,
                type=spec_type,
                nclass=int(nclass) if nclass != "" else None,
                mean=float(obs.mean()) if obs.size else 0.0,
                std=max(float(obs.std()) if obs.size else 1.0, 1e-6),
            )
        )
    visit_times = pd.to_numeric(train_long["visit_time"], errors="coerce").dropna().to_numpy(dtype=float)
    survival_times = pd.to_numeric(pd.Series(train_survival_times), errors="coerce").dropna().to_numpy(dtype=float)
    combined = np.concatenate([visit_times, survival_times]) if survival_times.size else visit_times
    time_min = float(np.min(combined)) if combined.size else 0.0
    time_max = float(np.max(combined)) if combined.size else 1.0
    max_visits = int(train_long.groupby("patient_id").size().max())
    return specs, time_min, max(time_max, time_min + 1e-6), max_visits


def _transform_longitudinal_panel(
    long_df: pd.DataFrame,
    subject_ids: np.ndarray,
    specs: list[LongitudinalSpec],
    time_min: float,
    time_max: float,
    max_visits: int,
) -> LongitudinalPanel:
    n_subjects = len(subject_ids)
    n_vars = len(specs)
    raw_values = np.full((n_subjects, max_visits, n_vars), np.nan, dtype=np.float32)
    masks = np.zeros((n_subjects, max_visits, n_vars), dtype=np.float32)
    times = np.zeros((n_subjects, max_visits), dtype=np.float32)
    id_to_row = {int(pid): idx for idx, pid in enumerate(subject_ids)}
    grouped = long_df.sort_values(["patient_id", "visit_time"]).groupby("patient_id", sort=True)
    for pid, rows in grouped:
        row_idx = id_to_row.get(int(pid))
        if row_idx is None:
            continue
        for visit_idx, (_, record) in enumerate(rows.head(max_visits).iterrows()):
            times[row_idx, visit_idx] = float(record["visit_time"])
            for var_idx, spec in enumerate(specs):
                value = record[spec.name]
                if pd.isna(value):
                    continue
                raw_values[row_idx, visit_idx, var_idx] = float(value)
                masks[row_idx, visit_idx, var_idx] = 1.0
    observed_rows = (masks.sum(axis=-1) > 0).astype(np.float32)
    time_rng = max(time_max - time_min, 1e-6)
    times_norm = ((times - time_min) / time_rng) * observed_rows
    values = np.nan_to_num(raw_values, nan=0.0).astype(np.float32)
    for idx, spec in enumerate(specs):
        values[:, :, idx] = ((values[:, :, idx] - spec.mean) / max(spec.std, 1e-6)) * masks[:, :, idx]
    return LongitudinalPanel(
        subject_ids=np.asarray(subject_ids, dtype=int),
        times=torch.tensor(times_norm, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        masks=torch.tensor(masks, dtype=torch.float32),
        raw_values=raw_values,
        specs=specs,
        time_min=time_min,
        time_max=time_max,
    )


def _make_bundle(
    raw_all: pd.DataFrame,
    ids_all: pd.DataFrame,
    long_df: pd.DataFrame,
    types: list[dict[str, Any]],
    subject_indices: np.ndarray,
    static_prep: StaticPreprocessor,
    long_specs: list[LongitudinalSpec],
    time_min: float,
    time_max: float,
    max_visits: int,
    cfg: dict[str, Any],
) -> PDC2Bundle:
    raw = raw_all.iloc[subject_indices].reset_index(drop=True).copy()
    ids = ids_all.iloc[subject_indices].reset_index(drop=True).copy()
    treatment = F.one_hot(
        torch.tensor(np.clip(raw[TREATMENT_NAME].fillna(0).to_numpy(dtype=int), 0, TREATMENT_N_CLASSES - 1), dtype=torch.long),
        num_classes=TREATMENT_N_CLASSES,
    ).float()
    hivae_types = [dict(t) for t in types if t["name"] != TREATMENT_NAME]
    encoded, miss, true_miss = _transform_static(raw, hivae_types, static_prep)
    panel = _transform_longitudinal_panel(long_df, subject_indices, long_specs, time_min, time_max, max_visits)
    validate_complete_l0(panel, float(cfg.get("model", {}).get("baseline_time_eps", 1e-6)))
    l0_names = {spec.name for spec in panel.specs}
    for idx, feature in enumerate(hivae_types):
        if feature["name"] in l0_names:
            miss[:, idx] = 1.0
            true_miss[:, idx] = 1.0
    return PDC2Bundle(
        raw_df=raw,
        encoded_df=encoded,
        types=hivae_types,
        miss_mask=miss,
        true_miss_mask=true_miss,
        longitudinal=panel,
        ids_df=ids,
        y_dim_partition=y_dim_partition_for_types(hivae_types, int(cfg["model"].get("y_dim_static", 6))),
        static_feature_count=len(hivae_types),
        treatment=treatment,
        treatment_name=TREATMENT_NAME,
        treatment_n_classes=TREATMENT_N_CLASSES,
    )


def _write_adapted_tables(output_root: Path, data_dir: Path, raw: pd.DataFrame, ids: pd.DataFrame, long_df: pd.DataFrame, types: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    adapted = output_root / "adapted_data"
    adapted.mkdir(parents=True, exist_ok=True)
    raw.to_csv(adapted / "data_phasesyn.csv", index=False)
    ids.to_csv(adapted / "simulation_id.csv", index=False)
    long_df.to_csv(adapted / "longitudinal.csv", index=False)
    pd.DataFrame(types).to_csv(adapted / "data_types_phasesyn_piecewise.csv", index=False)
    out_meta = {
        "source": str(data_dir),
        "treatment": TREATMENT_NAME,
        "survival_time": "time",
        "event": "censor",
        "simulator": metadata.get("name", "unknown"),
        "source_n": metadata.get("n"),
    }
    (adapted / "adaptation_metadata.json").write_text(json.dumps(_jsonable(out_meta), indent=2) + "\n")


def _write_split_file(raw: pd.DataFrame, ids: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray, path: Path) -> None:
    id_col = "simulation_id" if "simulation_id" in ids.columns else ids.columns[-1]
    rows = []
    for split, indices in [("train", train_idx), ("test", test_idx)]:
        for idx in indices:
            rows.append({
                "row_index": int(idx),
                "panel_patient_id": int(idx),
                "original_subject_id": int(ids.iloc[int(idx)][id_col]),
                "split": split,
                "time": float(raw.iloc[int(idx)]["time"]),
                "censor": float(raw.iloc[int(idx)]["censor"]),
                TREATMENT_NAME: int(raw.iloc[int(idx)][TREATMENT_NAME]),
            })
    pd.DataFrame(rows).sort_values("row_index").to_csv(path, index=False)


def _sample_longitudinal_future(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    latents: dict[str, torch.Tensor],
    device: torch.device,
    sample: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    panel = bundle.longitudinal
    times = panel.times.to(device)
    split = model.split_longitudinal_batch(times, panel.values.to(device), panel.masks.to(device))
    z = latents["z"].to(device)
    s = latents["s"].to(device)
    a = latents.get("a", bundle.treatment).to(device)
    with torch.no_grad():
        if latents.get("u0") is not None:
            u0 = latents["u0"].to(device)
            u_path = model.integrate_path(u0, times, z, s, a)
        else:
            _, u_path = model.sample_u_path_from_l0(z, s, split["L0"], times, a)
        features = model.decoder._path_features(u_path, times, z, s, a)
        outs: list[torch.Tensor] = []
        for idx, spec in enumerate(panel.specs):
            params = model.decoder.heads[idx](features)
            mu = params[:, :, 0]
            if sample:
                var = F.softplus(params[:, :, 1].clamp(-8.0, 8.0)).clamp(min=1e-4, max=1e4)
                value = torch.normal(mu, torch.sqrt(var))
            else:
                value = mu
            outs.append(value.unsqueeze(-1))
        pred_norm = torch.cat(outs, dim=-1).detach().cpu().numpy()
    split_cpu = model.split_longitudinal_batch(panel.times, panel.values, panel.masks)
    baseline_idx = split_cpu["baseline_index"].detach().cpu().numpy()
    l0 = split_cpu["L0"].detach().cpu().numpy()
    m0 = split_cpu["M0"].detach().cpu().numpy().astype(bool)
    for i, visit in enumerate(baseline_idx):
        pred_norm[i, visit, m0[i]] = l0[i, m0[i]]
    pred_raw = pred_norm.copy()
    for idx, spec in enumerate(panel.specs):
        pred_raw[:, :, idx] = pred_raw[:, :, idx] * spec.std + spec.mean
    pred_raw, support = _apply_longitudinal_support(bundle, pred_raw)
    future_mask = split_cpu["future_masks"].detach().cpu().numpy().astype(bool)
    support["future_rows_only"] = 1.0
    support["future_observed_cell_count"] = float(future_mask.sum())
    support["future_observed_visit_count"] = float(split_cpu["future_visit_mask"].sum().item())
    return pred_raw, future_mask, baseline_idx, support


def _summarize_metric_rows(rows: list[dict[str, float]], n_subjects: int) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    summary: dict[str, Any] = {"n_replicates": int(len(df)), "test_subject_count": int(n_subjects)}
    for col in df.columns:
        if col == "replicate" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_sd"] = float(df[col].std(ddof=0))
    return summary


def _km_curve(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0.5
    ok = np.isfinite(times)
    times = times[ok]
    events = events[ok]
    if times.size == 0:
        return np.asarray([0.0]), np.asarray([1.0])
    xs = [0.0]
    ys = [1.0]
    surv = 1.0
    for t in np.unique(np.sort(times)):
        at_risk = np.sum(times >= t)
        n_events = np.sum((times == t) & events)
        if at_risk > 0:
            surv *= 1.0 - n_events / at_risk
        xs.append(float(t))
        ys.append(float(surv))
    return np.asarray(xs), np.asarray(ys)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_survival(real_df: pd.DataFrame, synth_reps: list[pd.DataFrame], out_dir: Path, label: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rx, ry = _km_curve(real_df["time"].to_numpy(), real_df["censor"].to_numpy())
    axes[0].step(rx, ry, where="post", color="black", linewidth=2.2, label=f"Observed {label}")
    grid = np.linspace(0.0, max(float(real_df["time"].max()), 1e-6), 256)
    curves = []
    for syn in synth_reps:
        sx, sy = _km_curve(syn["time"].to_numpy(), syn["censor"].to_numpy())
        idx = np.searchsorted(sx, grid, side="right") - 1
        idx = np.clip(idx, 0, len(sy) - 1)
        curves.append(sy[idx])
        axes[0].step(sx, sy, where="post", alpha=0.18, linewidth=0.8)
    if curves:
        arr = np.asarray(curves)
        axes[0].plot(grid, arr.mean(axis=0), color="#c44e52", linestyle="--", linewidth=2.0, label="Generated mean")
        axes[0].fill_between(grid, arr.mean(axis=0) - arr.std(axis=0), arr.mean(axis=0) + arr.std(axis=0), color="#c44e52", alpha=0.18)
    axes[0].set_title(f"{label.title()} Kaplan-Meier")
    axes[0].set_xlabel("Normalized time")
    axes[0].set_ylabel("Survival probability")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    upper = max(float(real_df["time"].max()), *(float(s["time"].max()) for s in synth_reps))
    bins = np.linspace(0.0, max(upper, 1e-6), 28)
    axes[1].hist(real_df["time"], bins=bins, density=True, alpha=0.45, color="#2f6f9f", label="Observed")
    for syn in synth_reps:
        axes[1].hist(syn["time"], bins=bins, density=True, alpha=0.04, color="#c44e52")
    axes[1].set_title("Observed-Time Distribution")
    axes[1].set_xlabel("Normalized time")
    axes[1].legend(fontsize=8)

    real_rate = float((real_df["censor"] > 0.5).mean())
    rates = np.asarray([float((syn["censor"] > 0.5).mean()) for syn in synth_reps])
    axes[2].bar([0, 1], [real_rate, rates.mean()], yerr=[0.0, rates.std()], color=["#2f6f9f", "#c44e52"], tick_label=["Observed", "Generated"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Event rate")
    axes[2].set_title("Event Rate")
    _savefig(fig, out_dir / "survival" / f"{label}_survival_summary.png")


def _plot_longitudinal(bundle: PDC2Bundle, pred_reps: list[np.ndarray], future_mask: np.ndarray, out_dir: Path, label: str) -> None:
    panel = bundle.longitudinal
    real = panel.raw_values
    times = panel.times.detach().cpu().numpy() * (panel.time_max - panel.time_min) + panel.time_min
    for idx, spec in enumerate(panel.specs):
        xs, real_mean, gen_mean, gen_ci = [], [], [], []
        for visit in range(real.shape[1]):
            obs = future_mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            real_mean.append(float(np.nanmean(real[:, visit, idx][obs])))
            rep_means = [float(np.nanmean(pred[:, visit, idx][obs])) for pred in pred_reps]
            gen_mean.append(float(np.nanmean(rep_means)))
            gen_ci.append(float(1.96 * np.nanstd(rep_means, ddof=1) / math.sqrt(len(rep_means))) if len(rep_means) > 1 else 0.0)
        if not xs:
            continue
        x = np.asarray(xs)
        gm = np.asarray(gen_mean)
        gci = np.asarray(gen_ci)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x, real_mean, marker="o", color="black", linewidth=2.1, label=f"Observed {label} future")
        ax.plot(x, gm, marker="s", color="#c44e52", linewidth=2.0, label="Generated replicate mean")
        ax.fill_between(x, gm - gci, gm + gci, color="#c44e52", alpha=0.2, label="Generated 95% CI")
        ax.set_title(f"{label.title()} Future Replicate-Mean 95% CI: {spec.name}")
        ax.set_xlabel("Normalized time")
        ax.set_ylabel(spec.name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out_dir / "replicate_mean_95ci" / f"{spec.name}_replicate_mean_95ci.png")


def _plot_metric_summary(summary: dict[str, Any], out_dir: Path) -> None:
    keys = [
        "survival_km_integrated_abs_error_mean",
        "event_rate_diff_mean",
        "survival_time_rmse_ratio_mean",
        "survival_event_accuracy_mean",
        "future_continuous_rmse_ratio_vs_l0_carryforward_mean",
        "future_continuous_ks_mean_mean",
    ]
    values = [float(summary[k]) for k in keys if k in summary]
    labels = [k.replace("_mean", "").replace("_", "\n") for k in keys if k in summary]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(np.arange(len(values)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Simulation Holdout Metric Summary")
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, out_dir / "holdout_metric_summary.png")


def _plot_by_treatment(real_df: pd.DataFrame, static_all: pd.DataFrame, out_dir: Path) -> None:
    arms = sorted(int(x) for x in real_df[TREATMENT_NAME].dropna().unique())
    fig, axes = plt.subplots(1, len(arms), figsize=(5.5 * len(arms), 4.5), squeeze=False)
    for ax, arm in zip(axes.ravel(), arms):
        real = real_df[real_df[TREATMENT_NAME].astype(int) == arm]
        rx, ry = _km_curve(real["time"].to_numpy(), real["censor"].to_numpy())
        ax.step(rx, ry, where="post", color="black", linewidth=2.0, label=f"Observed A={arm}")
        for rep in sorted(static_all["replicate"].astype(int).unique()):
            syn = static_all[(static_all["replicate"].astype(int) == rep) & (static_all[TREATMENT_NAME].astype(int) == arm)]
            sx, sy = _km_curve(syn["time"].to_numpy(), syn["censor"].to_numpy())
            ax.step(sx, sy, where="post", color="#c44e52", alpha=0.10, linewidth=0.8)
        ax.set_title(f"A={arm} KM")
        ax.set_xlabel("Normalized time")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    _savefig(fig, out_dir / "figures_by_treatment" / "km_curves_by_treatment.png")


def _model_audit(cfg: dict[str, Any], model: PhaseSynModel) -> dict[str, Any]:
    return {
        "dataset": "simulation",
        "encoder_conditioning": model.encoder_conditioning,
        "u0_init_mode": model.u0_init_mode,
        "treatment_name": model.treatment_name,
        "treatment_n_classes": model.treatment_dim,
        "dynamic_survival_head": hasattr(model, "dynamic_survival_head"),
        "uses_baseline_inclusive_longitudinal_loss": float(cfg["model"].get("baseline_long_weight", 1.0)) > 0.0,
        "passes_audit": (
            model.encoder_conditioning == "baseline_only"
            and model.u0_init_mode == "baseline_l0"
            and float(cfg["model"].get("baseline_long_weight", 1.0)) > 0.0
            and hasattr(model, "dynamic_survival_head")
            and model.treatment_dim == TREATMENT_N_CLASSES
        ),
    }


def _candidate_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_config(args.config, {
        "dataset": {
            "name": "pdc2",
            "data_dir": str(args.data_dir),
            "output_root": str(args.output_root),
            "max_visits": None,
        },
        "model": {
            "longitudinal_mode": "latent_ode",
            "survival": "dynamic",
            "z_dim": 6,
            "s_dim": 6,
            "y_dim_static": 6,
            "u_dim": 6,
            "gru_hidden_dim": 6,
            "ode_hidden_dim": 6,
            "decoder_hidden_dim": 6,
            "u0_initializer_hidden_dim": 6,
            "dynamic_survival_hidden_dim": 6,
            "dynamic_survival_num_layers": 2,
            "dynamic_survival_dropout": 0.0,
            "n_intervals": int(args.n_intervals),
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
            "kl_weight_s": float(args.kl_weight_s),
            "kl_weight_z": float(args.kl_weight_z),
            "kl_weight_u": 0.0,
            "static_weight": 1.0,
            "longitudinal_weight": float(args.longitudinal_weight),
            "lambda_surv": float(args.lambda_surv),
            "continuous_mse_weight": float(args.continuous_mse_weight),
            "use_randomization_loss": True,
            "randomization_loss_weight": float(args.randomization_loss_weight),
            "randomization_loss_on": "z_mean",
            "treatment_variable_name": TREATMENT_NAME,
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "device": str(args.device),
            "n_generated_dataset": 1,
            "early_stopping": False,
            "subset_size": None,
            "freeze_normalization": True,
        },
        "generation": {
            "prior_n": int(args.prior_n),
            "prior_treatment": int(args.prior_treatment),
            "time_grid": [float(x) for x in np.linspace(0.0, 1.0, 9)],
            "deterministic": False,
        },
        "evaluation": {
            "deterministic_static_export": False,
            "posterior_generation": True,
            "n_replicates": int(args.n_replicates),
        },
    })
    cfg["simulation_experiment"] = {
        "source_data": str(args.data_dir),
        "test_fraction": float(args.test_fraction),
        "split_seed": int(args.split_seed),
        "treatment": TREATMENT_NAME,
        "mirrors_pdc2_experiment": str(args.reference_pdc2),
    }
    return cfg


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(int(args.seed))
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    output_root.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    raw, ids, long_df, types, source_metadata = _read_simulation(Path(args.data_dir))
    cfg = _candidate_config(args)
    _write_adapted_tables(output_root, Path(args.data_dir), raw, ids, long_df, types, source_metadata)

    train_idx, test_idx = _stratified_split(raw, float(args.test_fraction), int(args.split_seed))
    _write_split_file(raw, ids, train_idx, test_idx, output_root / "subject_splits.csv")

    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_specs, time_min, time_max, max_visits = _fit_longitudinal_preprocessor(long_df, types, train_idx, raw.iloc[train_idx]["time"])
    train_bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)
    test_bundle = _make_bundle(raw, ids, long_df, types, test_idx, static_prep, long_specs, time_min, time_max, max_visits, cfg)

    with open(output_root / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    preprocessing = {
        "static_preprocessor": _jsonable(static_prep),
        "longitudinal_specs": [dataclasses.asdict(spec) for spec in long_specs],
        "time_min": time_min,
        "time_max": time_max,
        "max_visits": max_visits,
        "train_subject_count": int(len(train_idx)),
        "test_subject_count": int(len(test_idx)),
        "preprocessing_fit_on_train_only": True,
    }
    (output_root / "preprocessing_metadata.json").write_text(json.dumps(_jsonable(preprocessing), indent=2) + "\n")

    result = train_model(train_bundle, cfg, output_dir=train_dir, overfit_name=None)
    model = result["model"].to(torch.device(cfg["training"].get("device", "cpu")))
    if not isinstance(model, PhaseSynModel):
        raise TypeError("Simulation experiment requires PhaseSynModel.")
    device = torch.device(cfg["training"].get("device", "cpu"))
    audit = _model_audit(cfg, model)
    audit.update(_leakage_diagnostics(model, test_bundle, device))
    audit.update(_survival_generation_perturbation_audit(model, test_bundle, device, int(args.seed) + 997))
    audit.update(_future_generation_perturbation_audit(model, test_bundle, device, int(args.seed) + 1997))
    audit["hivae_uses_frozen_train_normalization"] = model.hivae._global_norm_params is not None
    audit["passes_audit"] = bool(
        audit["passes_audit"]
        and audit["survival_generation_invariant_to_test_survival"]
        and audit["future_generation_invariant_to_test_future_values"]
        and audit["hivae_uses_frozen_train_normalization"]
    )
    if not audit["passes_audit"]:
        raise RuntimeError(f"Simulation audit failed: {audit}")
    (output_root / "leakage_audit.json").write_text(json.dumps(_jsonable(audit), indent=2) + "\n")

    rep_static: list[pd.DataFrame] = []
    rep_long_csv: list[pd.DataFrame] = []
    rep_metric_rows: list[dict[str, Any]] = []
    rep_pred_arrays: list[np.ndarray] = []
    per_visit_frames: list[pd.DataFrame] = []
    per_var_frames: list[pd.DataFrame] = []
    generation_audits: list[dict[str, Any]] = []
    future_mask = None
    for rep in range(1, int(args.n_replicates) + 1):
        seed = int(args.seed) + 1000 + rep
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        syn_static, latents, gen_audit = _decode_baseline_conditioned_static(model, test_bundle, device)
        syn_static.insert(0, "replicate", rep)
        syn_static.to_csv(test_dir / f"synthetic_static_test_rep{rep:02d}.csv", index=False)
        pred_raw, future_mask, baseline_idx, support = _sample_longitudinal_future(model, test_bundle, latents, device, sample=not args.deterministic_longitudinal)
        long_df_rep = _save_future_longitudinal(test_bundle, pred_raw, future_mask, test_dir / f"synthetic_longitudinal_future_test_rep{rep:02d}.csv", rep)
        long_metrics, per_visit, per_var = _future_longitudinal_metrics(test_bundle, pred_raw, future_mask, baseline_idx)
        survival_metrics = event_rate_metrics(test_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"]))
        survival_metrics.update(paired_survival_metrics(test_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"])))
        row: dict[str, Any] = {"replicate": float(rep)}
        row.update({k: float(v) for k, v in survival_metrics.items()})
        row.update({k: float(v) for k, v in long_metrics.items()})
        row.update({k: float(v) for k, v in support.items()})
        rep_metric_rows.append(row)
        per_visit.insert(0, "replicate", rep)
        per_var.insert(0, "replicate", rep)
        per_visit_frames.append(per_visit)
        per_var_frames.append(per_var)
        rep_static.append(syn_static)
        rep_long_csv.append(long_df_rep)
        rep_pred_arrays.append(pred_raw)
        generation_audits.append(gen_audit)

    static_all = pd.concat(rep_static, ignore_index=True)
    long_all = pd.concat(rep_long_csv, ignore_index=True)
    metrics_df = pd.DataFrame(rep_metric_rows)
    static_all.to_csv(test_dir / "holdout_synthetic_static_all.csv", index=False)
    long_all.to_csv(test_dir / "holdout_synthetic_longitudinal_future_all.csv", index=False)
    metrics_df.to_csv(test_dir / "holdout_replicate_metrics.csv", index=False)
    pd.concat(per_visit_frames, ignore_index=True).to_csv(test_dir / "longitudinal_future_per_visit_metrics.csv", index=False)
    pd.concat(per_var_frames, ignore_index=True).to_csv(test_dir / "longitudinal_future_variable_metrics.csv", index=False)

    summary = _summarize_metric_rows(rep_metric_rows, len(test_idx))
    summary.update(audit)
    summary["source_simulation_n"] = int(source_metadata.get("n", len(raw)))
    summary["source_overall_event_rate"] = float(raw["censor"].mean())
    summary["train_final_loss"] = float(result["curves"]["loss"].dropna().iloc[-1])
    summary["train_loss_decrease"] = float(
        (result["curves"]["loss"].dropna().iloc[0] - result["curves"]["loss"].dropna().iloc[-1])
        / max(abs(result["curves"]["loss"].dropna().iloc[0]), 1e-8)
    )
    summary["nan_epoch_count"] = int(result["curves"]["nan_epoch"].astype(bool).sum()) if "nan_epoch" in result["curves"] else 0
    summary["generation_audits_all_survival_zero"] = bool(all(x["test_survival_mask_zero_for_generation"] and x["test_survival_tensor_zero_for_generation"] for x in generation_audits))
    (test_dir / "holdout_summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    metrics_df.describe().to_csv(test_dir / "holdout_metric_describe.csv")

    if future_mask is not None:
        _plot_survival(test_bundle.raw_df, [df.drop(columns=["replicate", "patient_id"]) for df in rep_static], test_dir / "figures", "test")
        _plot_longitudinal(test_bundle, rep_pred_arrays, future_mask, test_dir / "figures", "test")
        _plot_metric_summary(summary, test_dir / "figures")
        _plot_by_treatment(test_bundle.raw_df, static_all, test_dir)

    prior_dir = output_root / "prior_generation"
    prior_static, prior_long, prior_tensors = generate_prior_cohort(
        model,
        train_bundle,
        n=int(args.prior_n),
        treatment=int(args.prior_treatment),
        time_grid=np.asarray(np.linspace(0.0, 1.0, max_visits), dtype=np.float32),
        device=device,
        deterministic=False,
        return_tensors=True,
    )
    prior_dir.mkdir(parents=True, exist_ok=True)
    prior_static.to_csv(prior_dir / "prior_synthetic_static.csv", index=False)
    prior_long.to_csv(prior_dir / "prior_synthetic_longitudinal.csv", index=False)
    prior_meta = {
        "mode": "prior",
        "n": int(args.prior_n),
        "treatment": int(args.prior_treatment),
        "baseline_generated_from_prior": bool(prior_tensors["baseline_generated_from_prior"].item()),
        "uses_observed_future_outcomes": bool(prior_tensors["uses_observed_future_outcomes"].item()),
        "checkpoint": str(train_dir / "model_checkpoint.pt"),
    }
    (prior_dir / "prior_generation_metadata.json").write_text(json.dumps(_jsonable(prior_meta), indent=2) + "\n")

    print(json.dumps(_jsonable(summary), indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the current PhaseSyn dynamic-survival holdout experiment on simple simulation data.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pdc2.yaml"))
    parser.add_argument("--data-dir", type=Path, default=SIM_DATA_DIR)
    parser.add_argument("--reference-pdc2", default=str(ROOT / "outputs" / "pdc2" / "experiments_20260602"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260602)
    parser.add_argument("--split-seed", type=int, default=20260521)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-intervals", type=int, default=16)
    parser.add_argument("--lambda-surv", type=float, default=1.4)
    parser.add_argument("--kl-weight-s", type=float, default=0.3)
    parser.add_argument("--kl-weight-z", type=float, default=0.3)
    parser.add_argument("--longitudinal-weight", type=float, default=2.0)
    parser.add_argument("--continuous-mse-weight", type=float, default=0.8)
    parser.add_argument("--deterministic-longitudinal", action="store_true")
    parser.add_argument("--randomization-loss-weight", type=float, default=0.0)
    parser.add_argument("--prior-n", type=int, default=100)
    parser.add_argument("--prior-treatment", type=int, default=0)
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
