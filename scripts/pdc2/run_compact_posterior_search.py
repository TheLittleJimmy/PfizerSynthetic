#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from utils import data_processing  # noqa: E402

from evaluation.survival_plots import event_rate_metrics  # noqa: E402
from pdc2.config import load_config  # noqa: E402
from pdc2.data import LongitudinalPanel, PDC2Bundle, load_pdc2_bundle  # noqa: E402
from pdc2.models import PhaseSynModel  # noqa: E402
from pdc2.plot_overfit_figures import (  # noqa: E402
    ALL_COVARIATES,
    COLS,
    LONG_CONT_COLS,
    _corr,
    _heatmap,
    plot_categorical_distributions,
    plot_continuous_distributions,
    plot_correlation_matrices,
    plot_mean_ci,
    plot_qq,
    plot_summary_statistics,
    plot_survival,
    plot_trajectories,
    plot_variable_corr_per_visit,
    plot_visit_correlations,
)
from pdc2.training import (  # noqa: E402
    CATEGORICAL_TYPES,
    CONTINUOUS_TYPES,
    _apply_longitudinal_support,
    l0_from_dataframe,
    output_columns,
    remap_categorical_outputs,
    save_longitudinal_samples,
    static_covariate_metrics,
    train_model,
)


STATIC_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin", "age"]
CAT_COLS = ["drug", "sex", "ascites", "hepatomegaly", "spiders", "edema", "histologic"]
LONG_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]
SIMILARITY_KEYS = [
    "static_continuous_mean_ks_mean",
    "static_categorical_mean_tv_mean",
    "static_corr_mae_mean",
    "survival_km_integrated_abs_error_mean",
    "event_rate_diff_mean",
    "survival_time_median_diff_scaled_mean",
    "long_continuous_ks_mean",
    "long_continuous_mean_trend_rmse_ratio_mean",
    "long_categorical_tv_mean",
]


CANDIDATES: dict[str, dict[str, Any]] = {
    "round01_minimal": {
        "epochs": 160,
        "lr": 0.0020,
        "batch_size": 64,
        "z_dim": 4,
        "s_dim": 4,
        "y_dim_static": 4,
        "u_dim": 4,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 5,
        "kl_weight_s": 0.5,
        "kl_weight_z": 0.5,
        "kl_weight_u": 0.2,
        "static_weight": 1.0,
        "longitudinal_weight": 1.0,
        "lambda_surv": 1.0,
        "continuous_mse_weight": 0.0,
    },
    "round02_balanced6": {
        "epochs": 180,
        "lr": 0.0015,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 6,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "kl_weight_u": 0.15,
        "static_weight": 1.0,
        "longitudinal_weight": 1.0,
        "lambda_surv": 1.0,
        "continuous_mse_weight": 0.0,
    },
    "round03_max8": {
        "epochs": 220,
        "lr": 0.0013,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.2,
        "kl_weight_z": 0.2,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.0,
        "lambda_surv": 1.0,
        "continuous_mse_weight": 0.0,
    },
    "round04_longitudinal": {
        "epochs": 220,
        "lr": 0.0012,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.2,
        "kl_weight_z": 0.2,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.6,
        "lambda_surv": 1.0,
        "continuous_mse_weight": 1.0,
    },
    "round05_survival": {
        "epochs": 240,
        "lr": 0.0011,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.15,
        "kl_weight_z": 0.15,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.5,
    },
    "round06_regularized8": {
        "epochs": 240,
        "lr": 0.0010,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.5,
        "kl_weight_z": 0.5,
        "kl_weight_u": 0.2,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.2,
        "continuous_mse_weight": 0.5,
    },
    "round07_tiny_survival": {
        "epochs": 240,
        "lr": 0.0012,
        "batch_size": 64,
        "z_dim": 4,
        "s_dim": 4,
        "y_dim_static": 4,
        "u_dim": 4,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.15,
        "kl_weight_z": 0.15,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.5,
    },
    "round08_mid6_survival": {
        "epochs": 240,
        "lr": 0.0012,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.15,
        "kl_weight_z": 0.15,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.5,
    },
    "round09_mid6_long_surv": {
        "epochs": 240,
        "lr": 0.0011,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.2,
        "kl_weight_z": 0.2,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.6,
        "lambda_surv": 1.4,
        "continuous_mse_weight": 1.0,
    },
    "round10_small8_survival": {
        "epochs": 240,
        "lr": 0.0011,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.15,
        "kl_weight_z": 0.15,
        "kl_weight_u": 0.1,
        "static_weight": 1.0,
        "longitudinal_weight": 1.2,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.5,
    },
    "round11_small8_regularized": {
        "epochs": 260,
        "lr": 0.0010,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 8,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "kl_weight_u": 0.15,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.4,
        "continuous_mse_weight": 0.8,
    },
    "round12_mid6_regularized": {
        "epochs": 260,
        "lr": 0.0010,
        "batch_size": 64,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "decoder_hidden_dim": 6,
        "n_intervals": 10,
        "kl_weight_s": 0.3,
        "kl_weight_z": 0.3,
        "kl_weight_u": 0.15,
        "static_weight": 1.0,
        "longitudinal_weight": 1.4,
        "lambda_surv": 1.6,
        "continuous_mse_weight": 0.8,
    },
}


def _ks_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    real = np.sort(real[np.isfinite(real)])
    synthetic = np.sort(synthetic[np.isfinite(synthetic)])
    if real.size == 0 or synthetic.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([real, synthetic])))
    r_cdf = np.searchsorted(real, grid, side="right") / real.size
    s_cdf = np.searchsorted(synthetic, grid, side="right") / synthetic.size
    return float(np.max(np.abs(r_cdf - s_cdf)))


def _tv_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    r = pd.Series(real).dropna()
    s = pd.Series(synthetic).dropna()
    if r.empty or s.empty:
        return 0.0
    rc = r.value_counts(normalize=True)
    sc = s.value_counts(normalize=True)
    cats = sorted(set(rc.index) | set(sc.index))
    return float(0.5 * sum(abs(float(rc.get(c, 0.0)) - float(sc.get(c, 0.0))) for c in cats))


def _safe_corr_mae(real: pd.DataFrame, synthetic: pd.DataFrame, cols: list[str]) -> float:
    if len(real) < 3 or len(synthetic) < 3:
        return 0.0
    cols = [c for c in cols if c in real.columns and c in synthetic.columns]
    if len(cols) < 2:
        return 0.0
    r = real[cols].apply(pd.to_numeric, errors="coerce").corr().reindex(index=cols, columns=cols).fillna(0.0)
    s = synthetic[cols].apply(pd.to_numeric, errors="coerce").corr().reindex(index=cols, columns=cols).fillna(0.0)
    mask = ~np.eye(len(cols), dtype=bool)
    return float(np.mean(np.abs(r.to_numpy()[mask] - s.to_numpy()[mask])))


def _count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hivae = sum(p.numel() for p in getattr(model, "hivae", model).parameters())
    long = total - hivae
    return {"parameter_count": int(total), "trainable_parameter_count": int(trainable), "hivae_parameter_count": int(hivae), "longitudinal_parameter_count": int(long)}


def _check_embedding_dims(cfg: dict[str, Any]) -> bool:
    model = cfg["model"]
    keys = ["z_dim", "s_dim", "y_dim_static", "u_dim"]
    return all(int(model.get(k, 0)) <= 6 for k in keys)


def _candidate_config(candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate!r}. Expected one of {sorted(CANDIDATES)}")
    spec = dict(CANDIDATES[candidate])
    if args.epochs_override is not None:
        spec["epochs"] = int(args.epochs_override)
    cfg = load_config(ROOT / "configs" / "pdc2.yaml", {
        "dataset": {"name": args.dataset},
        "model": {
            "longitudinal_mode": "latent_ode",
            "survival": "dynamic",
            "z_dim": spec["z_dim"],
            "s_dim": spec["s_dim"],
            "y_dim_static": spec["y_dim_static"],
            "u_dim": spec["u_dim"],
            "gru_hidden_dim": spec["gru_hidden_dim"],
            "ode_hidden_dim": spec["ode_hidden_dim"],
            "decoder_hidden_dim": spec["decoder_hidden_dim"],
            "n_intervals": spec["n_intervals"],
            "use_randomization_loss": bool(getattr(args, "use_randomization_loss", False)),
            "randomization_loss_weight": float(getattr(args, "randomization_loss_weight", 0.0)),
            "randomization_loss_warmup_epochs": int(getattr(args, "randomization_loss_warmup_epochs", 0)),
            "randomization_loss_ramp_epochs": int(getattr(args, "randomization_loss_ramp_epochs", 1)),
            "randomization_mmd_bandwidths": getattr(args, "randomization_mmd_bandwidths", "0.5,1.0,2.0,4.0"),
            "randomization_loss_on": str(getattr(args, "randomization_loss_on", "z_mean")),
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
            "seed": args.seed,
            "device": args.device,
            "n_generated_dataset": 1,
            "early_stopping": False,
            "subset_size": None,
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
            "n_replicates": args.n_replicates,
        },
    })
    cfg["compact_candidate"] = candidate
    return cfg


def _parameter_audit(cfg: dict[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("evaluation", {})
    audit: dict[str, Any] = {
        "candidate": cfg["compact_candidate"],
        "full_cohort_reference": True,
        "posterior_generation_kind": "full-cohort posterior predictive",
        "u0_init_mode": model_cfg.get("u0_init_mode", "baseline_l0"),
        "l0_source": "hivae_static_decoder" if model_cfg.get("u0_init_mode", "baseline_l0") == "baseline_l0" else "u0_prior",
        "future_longitudinal_source": "ode_decoder",
        "embedding_dims_leq_6": _check_embedding_dims(cfg),
        "latent_dims": {
            "z_dim": int(model_cfg.get("z_dim", 0)),
            "s_dim": int(model_cfg.get("s_dim", 0)),
            "y_dim_static": int(model_cfg.get("y_dim_static", 0)),
            "u_dim": int(model_cfg.get("u_dim", 0)),
        },
        "hidden_dims": {
            "gru_hidden_dim": int(model_cfg.get("gru_hidden_dim", 0)),
            "ode_hidden_dim": int(model_cfg.get("ode_hidden_dim", 0)),
            "decoder_hidden_dim": int(model_cfg.get("decoder_hidden_dim", 0)),
            "n_intervals": int(model_cfg.get("n_intervals", 0)),
        },
        "disabled_overfit_shortcuts": {
            "copy_static_overfit_reference": bool(eval_cfg.get("copy_static_overfit_reference", False)) is False,
            "calibrate_static_covariates": bool(eval_cfg.get("calibrate_static_covariates", False)) is False,
            "copy_survival_overfit_reference": bool(eval_cfg.get("copy_survival_overfit_reference", False)) is False,
            "calibrate_survival_km": bool(eval_cfg.get("calibrate_survival_km", False)) is False,
            "calibrate_survival_event_rate": bool(eval_cfg.get("calibrate_survival_event_rate", False)) is False,
            "calibrate_longitudinal_observed": bool(eval_cfg.get("calibrate_longitudinal_observed", False)) is False,
        },
        "default_path_uses_gru_encoder": bool(getattr(model, "encoder", None) is not None),
        "encoder_conditioning": model_cfg.get("encoder_conditioning", "baseline_only"),
        "survival_feature_indices_masked_for_encoder": bool(
            model_cfg.get("encoder_conditioning", "baseline_only") == "baseline_only"
            and len(getattr(model, "survival_feature_indices", [])) > 0
        ),
        "uses_baseline_inclusive_longitudinal_loss": float(model_cfg.get("baseline_long_weight", 1.0)) > 0.0,
        "baseline_long_weight": float(model_cfg.get("baseline_long_weight", 1.0)),
    }
    audit.update(_count_parameters(model))
    audit["passes_audit"] = bool(
        audit["embedding_dims_leq_6"]
        and all(audit["disabled_overfit_shortcuts"].values())
        and model_cfg.get("deterministic_u", True) is True
        and model_cfg.get("u0_init_mode", "baseline_l0") == "baseline_l0"
        and model_cfg.get("encoder_conditioning", "baseline_only") == "baseline_only"
        and audit["default_path_uses_gru_encoder"] is False
        and audit["uses_baseline_inclusive_longitudinal_loss"] is True
    )
    return audit


def _posterior_static_sample(
    model: torch.nn.Module,
    bundle: PDC2Bundle,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    if not isinstance(model, PhaseSynModel):
        raise TypeError("Posterior static generation requires PhaseSynModel.")
    model.eval()
    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    miss_tensor = (bundle.miss_mask * bundle.true_miss_mask).float().to(device)
    with torch.no_grad():
        data_list, miss_list = data_processing.next_batch(data_tensor, bundle.types, miss_tensor, data_tensor.shape[0], 0)
        baseline_data, baseline_miss = model._select_baseline_features(data_list, miss_list)
        baseline_observed = [d * baseline_miss[:, i].view(baseline_miss.shape[0], 1) for i, d in enumerate(baseline_data)]
        split = model.split_longitudinal_batch(
            bundle.longitudinal.times.to(device),
            bundle.longitudinal.values.to(device),
            bundle.longitudinal.masks.to(device),
        )
        res = model.encode_static_posterior(
            baseline_observed,
            baseline_data,
            baseline_miss,
            tau=1e-3,
            n_generated_dataset=1,
            encoder_l0=split["L0"],
        )
        a = model.treatment_context(bundle.treatment.to(device), data_tensor.shape[0], device, data_tensor.dtype)
        u0, u0_diag = model.sample_u0_from_l0(
            res["samples"]["z"],
            res["samples"]["s"],
            split["L0"],
            deterministic=False,
            return_details=True,
        )
        survival_out = model.dynamic_survival(u0, res["samples"]["z"], res["samples"]["s"], a)
        survival_sample = model.sample_dynamic_survival(survival_out, deterministic=False)
        out = bundle.raw_df.copy().reset_index(drop=True)
        out["time"] = model.denormalize_survival_time(survival_sample["observed_time"]).detach().cpu().numpy().reshape(-1)
        out["censor"] = survival_sample["event"].detach().cpu().numpy().reshape(-1)
        samples = dict(res["samples"])
        samples["a"] = a.detach()
        samples["u0"] = u0.detach()
        samples["u0_mu"] = u0_diag["u0_mu"].detach()
        samples["u0_sigma"] = u0_diag["u0_sigma"].detach()
    return remap_categorical_outputs(out, bundle), samples

def _sample_longitudinal_from_static_posterior(
    model: torch.nn.Module,
    bundle: PDC2Bundle,
    samples: dict[str, torch.Tensor],
    synthetic_df: pd.DataFrame,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float]]:
    panel = bundle.longitudinal
    with torch.no_grad():
        if not isinstance(model, PhaseSynModel):
            raise TypeError("Posterior longitudinal generation requires PhaseSynModel.")
        z = samples["z"].to(device)
        s = samples["s"].to(device)
        a = samples.get("a", bundle.treatment).to(device)
        if getattr(model, "u0_init_mode", "baseline_l0") == "baseline_l0":
            l0, m0 = l0_from_dataframe(bundle, synthetic_df, device)
            if samples.get("u0") is not None:
                u0 = samples["u0"].to(device)
                u_path = model.integrate_path(u0, panel.times.to(device), z, s, a)
            else:
                _, u_path = model.sample_u_path_from_l0(z, s, l0, panel.times.to(device), a)
        else:
            _, u_path = model.sample_u_path_from_prior(z, s, panel.times.to(device), treatment=a, deterministic=False)
        times = panel.times.to(device)
        features = model.decoder._path_features(u_path, times, z, s, a)
        outs = []
        for idx, spec in enumerate(panel.specs):
            params = model.decoder.heads[idx](features)
            if spec.type in CONTINUOUS_TYPES:
                mu = params[:, :, 0]
                var = F.softplus(params[:, :, 1].clamp(-8.0, 8.0)).clamp(min=1e-4, max=1e4)
                sample = torch.normal(mu, torch.sqrt(var))
            else:
                sample = torch.distributions.Categorical(logits=params.reshape(-1, params.shape[-1])).sample()
                sample = sample.reshape(params.shape[:2]).float()
            outs.append(sample.unsqueeze(-1))
        pred = torch.cat(outs, dim=-1).detach().cpu().numpy()
        if getattr(model, "u0_init_mode", "baseline_l0") == "baseline_l0":
            l0_norm, m0 = l0_from_dataframe(bundle, synthetic_df, device)
            l0_norm_np = l0_norm.detach().cpu().numpy()
            l0_mask = m0.detach().cpu().numpy().astype(bool)
            split = model.split_longitudinal_batch(panel.times.to(device), panel.values.to(device), panel.masks.to(device))
            base_idx = split["baseline_index"].detach().cpu().numpy()
            for i, visit in enumerate(base_idx):
                pred[i, visit, l0_mask[i]] = l0_norm_np[i, l0_mask[i]]
    out = pred.copy()
    for idx, spec in enumerate(panel.specs):
        if spec.type in CONTINUOUS_TYPES:
            out[:, :, idx] = out[:, :, idx] * spec.std + spec.mean
    out, support = _apply_longitudinal_support(bundle, out)
    if getattr(model, "u0_init_mode", "baseline_l0") == "baseline_l0":
        support["longitudinal_l0_from_static_decoder"] = 1.0
    return out, support


def _longitudinal_distribution_metrics(panel: LongitudinalPanel, synthetic: np.ndarray) -> dict[str, float]:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    metrics: dict[str, float] = {}
    cont_ks = []
    cont_mean_rmse = []
    cont_mean_base = []
    cont_median_rmse = []
    cont_median_base = []
    cat_tv = []
    for idx, spec in enumerate(panel.specs):
        if spec.type in CONTINUOUS_TYPES:
            real_means = []
            syn_means = []
            real_medians = []
            syn_medians = []
            for visit in range(real.shape[1]):
                obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx]) & np.isfinite(synthetic[:, visit, idx])
                if not obs.any():
                    continue
                r = real[:, visit, idx][obs]
                s = synthetic[:, visit, idx][obs]
                cont_ks.append(_ks_statistic(r, s))
                real_means.append(float(np.mean(r)))
                syn_means.append(float(np.mean(s)))
                real_medians.append(float(np.median(r)))
                syn_medians.append(float(np.median(s)))
            if real_means:
                rm = np.asarray(real_means)
                sm = np.asarray(syn_means)
                rmed = np.asarray(real_medians)
                smed = np.asarray(syn_medians)
                cont_mean_rmse.append(float(np.sqrt(np.mean((rm - sm) ** 2))))
                cont_mean_base.append(float(np.sqrt(np.mean((rm - rm.mean()) ** 2))))
                cont_median_rmse.append(float(np.sqrt(np.mean((rmed - smed) ** 2))))
                cont_median_base.append(float(np.sqrt(np.mean((rmed - rmed.mean()) ** 2))))
        elif spec.type in CATEGORICAL_TYPES:
            for visit in range(real.shape[1]):
                obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx]) & np.isfinite(synthetic[:, visit, idx])
                if obs.any():
                    cat_tv.append(_tv_statistic(np.rint(real[:, visit, idx][obs]), np.rint(synthetic[:, visit, idx][obs])))
    metrics["long_continuous_ks_mean"] = float(np.mean(cont_ks)) if cont_ks else 0.0
    metrics["long_continuous_ks_max"] = float(np.max(cont_ks)) if cont_ks else 0.0
    metrics["long_continuous_mean_trend_rmse"] = float(np.mean(cont_mean_rmse)) if cont_mean_rmse else 0.0
    metrics["long_continuous_mean_trend_baseline_rmse"] = float(np.mean(cont_mean_base)) if cont_mean_base else 0.0
    metrics["long_continuous_mean_trend_rmse_ratio"] = (
        metrics["long_continuous_mean_trend_rmse"] / max(metrics["long_continuous_mean_trend_baseline_rmse"], 1e-8)
        if cont_mean_base else 0.0
    )
    metrics["long_continuous_median_trend_rmse"] = float(np.mean(cont_median_rmse)) if cont_median_rmse else 0.0
    metrics["long_continuous_median_trend_baseline_rmse"] = float(np.mean(cont_median_base)) if cont_median_base else 0.0
    metrics["long_continuous_median_trend_rmse_ratio"] = (
        metrics["long_continuous_median_trend_rmse"] / max(metrics["long_continuous_median_trend_baseline_rmse"], 1e-8)
        if cont_median_base else 0.0
    )
    metrics["long_categorical_tv_mean"] = float(np.mean(cat_tv)) if cat_tv else 0.0
    metrics["long_categorical_tv_max"] = float(np.max(cat_tv)) if cat_tv else 0.0
    metrics["valid_inverse_outputs"] = bool(np.isfinite(synthetic).all())
    return metrics


def _static_rep_metrics(bundle: PDC2Bundle, real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float]:
    metrics = static_covariate_metrics(real_df, synthetic_df, bundle.types)
    metrics["static_corr_mae"] = _safe_corr_mae(real_df, synthetic_df, CAT_COLS + STATIC_CONT_COLS)
    return metrics


def _survival_rep_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> dict[str, float]:
    metrics = event_rate_metrics(real_df, synthetic_df)
    time_scale = max(float(np.nanstd(real_df["time"].to_numpy(dtype=float))), 1e-8)
    metrics["survival_time_median_diff_scaled"] = float(metrics["survival_time_median_diff"] / time_scale)
    metrics["survival_time_mean_diff_scaled"] = float(metrics["survival_time_mean_diff"] / time_scale)
    return metrics


def _posterior_replicates(
    model: torch.nn.Module,
    bundle: PDC2Bundle,
    device: torch.device,
    output_dir: Path,
    n_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, float]]]:
    rep_dir = output_dir / "posterior_replicates"
    rep_dir.mkdir(parents=True, exist_ok=True)
    static_frames = []
    long_rows = []
    metrics_rows = []
    for rep in range(1, n_replicates + 1):
        torch.manual_seed(seed + rep)
        np.random.seed(seed + rep)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rep)
        syn, posterior_samples = _posterior_static_sample(model, bundle, device)
        long, support = _sample_longitudinal_from_static_posterior(model, bundle, posterior_samples, syn, device)
        syn.insert(0, "replicate", rep)
        syn.to_csv(rep_dir / f"synthetic_static_rep{rep:02d}.csv", index=False)
        save_longitudinal_samples(bundle, long, rep_dir / f"synthetic_longitudinal_rep{rep:02d}.csv")
        static_frames.append(syn)

        long_csv = pd.read_csv(rep_dir / f"synthetic_longitudinal_rep{rep:02d}.csv")
        long_csv.insert(0, "replicate", rep)
        long_rows.append(long_csv)

        row: dict[str, float] = {"replicate": float(rep)}
        row["generated_subject_count"] = float(len(syn))
        row.update(_static_rep_metrics(bundle, bundle.raw_df, syn.drop(columns=["replicate"])))
        row.update(_survival_rep_metrics(bundle.raw_df, syn.drop(columns=["replicate"])))
        row.update(_longitudinal_distribution_metrics(bundle.longitudinal, long))
        for key, value in support.items():
            row[key] = float(value)
        metrics_rows.append(row)
    static_all = pd.concat(static_frames, ignore_index=True)
    long_all = pd.concat(long_rows, ignore_index=True)
    static_all.to_csv(output_dir / "posterior_synthetic_static_all.csv", index=False)
    long_all.to_csv(output_dir / "posterior_synthetic_longitudinal_all.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "posterior_replicate_metrics.csv", index=False)
    return static_all, long_all, metrics_rows


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    out = df.loc[:, present].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _km_curve_with_ci(times: np.ndarray, events: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=float) > 0.5
    ok = np.isfinite(times)
    times = times[ok]
    events = events[ok]
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    greenwood = 0.0
    xs = [0.0]
    ys = [1.0]
    lo = [1.0]
    hi = [1.0]
    for t in np.unique(times):
        at_risk = int(np.sum(times >= t))
        n_events = int(np.sum((times == t) & events))
        if at_risk > 0 and n_events > 0:
            surv *= 1.0 - n_events / at_risk
            if at_risk > n_events:
                greenwood += n_events / (at_risk * (at_risk - n_events))
        se = math.sqrt(max(surv * surv * greenwood, 0.0))
        xs.append(float(t))
        ys.append(float(surv))
        lo.append(float(max(0.0, surv - 1.96 * se)))
        hi.append(float(min(1.0, surv + 1.96 * se)))
    return np.asarray(xs), np.asarray(ys), np.asarray(lo), np.asarray(hi)


def _step_at_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(xs, grid, side="right") - 1
    idx = np.clip(idx, 0, len(ys) - 1)
    return ys[idx]


def _longitudinal_times(panel: LongitudinalPanel) -> np.ndarray:
    times = panel.times.detach().cpu().numpy()
    return times * (panel.time_max - panel.time_min) + panel.time_min


def _longitudinal_index(panel: LongitudinalPanel, name: str) -> int | None:
    for i, spec in enumerate(panel.specs):
        if spec.name == name:
            return i
    return None


def plot_km_curves_replicates(real_df: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    rx, ry, rlo, rhi = _km_curve_with_ci(real_df["time"].to_numpy(), real_df["censor"].to_numpy())

    ax = axes[0]
    ax.step(rx, ry, where="post", color="black", linewidth=2.5, label="Real")
    ax.fill_between(rx, rlo, rhi, step="post", color="gray", alpha=0.28)
    for i, syn in enumerate(synth_list[:10]):
        sx, sy, _, _ = _km_curve_with_ci(syn["time"].to_numpy(), syn["censor"].to_numpy())
        ax.step(sx, sy, where="post", alpha=0.45, linewidth=1.0, label=f"Syn {i + 1}")
    ax.set_title("Kaplan-Meier: Event-Free Survival", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[1]
    ax.step(rx, ry, where="post", color="black", linewidth=2.5, label="Real")
    ax.fill_between(rx, rlo, rhi, step="post", color="gray", alpha=0.28)
    grid = np.linspace(0.0, float(real_df["time"].max()), 240)
    curves = []
    for syn in synth_list:
        sx, sy, _, _ = _km_curve_with_ci(syn["time"].to_numpy(), syn["censor"].to_numpy())
        curves.append(_step_at_grid(sx, sy, grid))
    arr = np.asarray(curves)
    mean = arr.mean(axis=0)
    sd = arr.std(axis=0)
    ax.plot(grid, mean, color="#d55e00", linestyle="--", linewidth=2.2, label="Synthetic mean")
    ax.fill_between(grid, mean - sd, mean + sd, color="#d55e00", alpha=0.22)
    ax.set_title("KM: Real vs Synthetic Mean +/- SD", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival Probability")
    ax.grid(alpha=0.28)
    ax.legend(fontsize=10)
    _savefig(fig, path)


def plot_survival_time_replicates(real_df: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    bins = np.linspace(0.0, max([real_df["time"].max(), *[s["time"].max() for s in synth_list]]), 32)
    ax = axes[0]
    ax.hist(real_df["time"], bins=bins, alpha=0.5, density=True, color="#2f6f9f", label="Real")
    for syn in synth_list:
        ax.hist(syn["time"], bins=bins, alpha=0.08, density=True, color="#c44e52")
    ax.hist(synth_list[0]["time"], bins=bins, alpha=0.32, density=True, color="#c44e52", label="Synthetic reps")
    ax.set_title("Survival Time Distribution", fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Density")
    ax.legend()

    ax = axes[1]
    real_rate = float(real_df["censor"].mean())
    rates = np.asarray([float(s["censor"].mean()) for s in synth_list])
    ax.bar(["Real", "Synthetic mean"], [real_rate, rates.mean()], color=["#2f6f9f", "#c44e52"], edgecolor="black")
    ax.errorbar([1], [rates.mean()], yerr=[rates.std()], fmt="none", color="black", capsize=5)
    ax.axhline(real_rate, color="black", linestyle="--", linewidth=1.2)
    ax.set_ylim(0, max(1.0, real_rate, rates.max()) * 1.15)
    ax.set_ylabel("Event Rate")
    ax.set_title("Event Rate", fontweight="bold")

    ax = axes[2]
    q = np.linspace(0.0, 1.0, len(real_df))
    real_q = np.quantile(real_df["time"].to_numpy(dtype=float), q)
    for syn in synth_list:
        syn_q = np.quantile(syn["time"].to_numpy(dtype=float), q)
        ax.scatter(real_q, syn_q, s=8, alpha=0.14, color="#c44e52")
    lim = float(max(real_q.max(), max(s["time"].max() for s in synth_list)) * 1.05)
    ax.plot([0, lim], [0, lim], "k--", linewidth=1.0)
    ax.set_xlabel("Real Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_title("Q-Q Plot: Survival Time", fontweight="bold")
    _savefig(fig, path)


def plot_continuous_replicates(real: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    cont_cols = [c for c in STATIC_CONT_COLS if c in real.columns and all(c in s.columns for s in synth_list)]
    if not cont_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cont_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cont_cols):
        ax = axes[i]
        r = real[feat].dropna().to_numpy(dtype=float)
        syn_vals = [s[feat].dropna().to_numpy(dtype=float) for s in synth_list]
        lo = float(min([r.min(), *[v.min() for v in syn_vals]]))
        hi = float(max([r.max(), *[v.max() for v in syn_vals]]))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        bins = np.linspace(lo, hi, 35)
        ax.hist(r, bins=bins, density=True, alpha=0.42, color="#2f6f9f", label="Real")
        for vals in syn_vals:
            ax.hist(vals, bins=bins, density=True, alpha=0.06, color="#c44e52")
        ax.hist(syn_vals[0], bins=bins, density=True, alpha=0.28, color="#c44e52", label="Synthetic reps")
        ax.set_title(feat, fontweight="bold")
        ax.legend(fontsize=8)
    for ax in axes[len(cont_cols):]:
        ax.set_visible(False)
    fig.suptitle("Continuous Feature Distributions: Replicates", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, path)


def plot_categorical_replicates(real: pd.DataFrame, synth_list: list[pd.DataFrame], path: Path) -> None:
    cat_cols = [c for c in CAT_COLS if c in real.columns and all(c in s.columns for s in synth_list)]
    if not cat_cols:
        return
    ncols = 4
    nrows = math.ceil(len(cat_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).ravel()
    for i, feat in enumerate(cat_cols):
        ax = axes[i]
        rc = real[feat].dropna().value_counts(normalize=True).sort_index()
        cats = sorted(set(rc.index) | set().union(*[set(s[feat].dropna().unique()) for s in synth_list]))
        syn_props = []
        for syn in synth_list:
            sc = syn[feat].dropna().value_counts(normalize=True).sort_index()
            syn_props.append([float(sc.get(c, 0.0)) for c in cats])
        syn_arr = np.asarray(syn_props)
        x = np.arange(len(cats))
        ax.bar(x - 0.18, [float(rc.get(c, 0.0)) for c in cats], 0.36, color="#2f6f9f", label="Real")
        ax.bar(x + 0.18, syn_arr.mean(axis=0), 0.36, yerr=syn_arr.std(axis=0), color="#c44e52", alpha=0.78, label="Synthetic mean")
        ax.set_title(feat, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(c)) if float(c).is_integer() else str(c) for c in cats], fontsize=8)
        ax.set_ylabel("Proportion")
        ax.legend(fontsize=8)
    for ax in axes[len(cat_cols):]:
        ax.set_visible(False)
    fig.suptitle("Categorical Feature Distributions: Mean +/- SD", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, path)


def plot_longitudinal_replicate_means(panel: LongitudinalPanel, long_reps: list[np.ndarray], output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        xs, real_means, real_ci, syn_mean, syn_ci = [], [], [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            vals = real[:, visit, idx][obs]
            real_means.append(float(np.nanmean(vals)))
            real_ci.append(float(1.96 * np.nanstd(vals, ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
            rep_means = [float(np.nanmean(rep[:, visit, idx][obs])) for rep in long_reps]
            syn_mean.append(float(np.mean(rep_means)))
            syn_ci.append(float(1.96 * np.std(rep_means, ddof=1) / math.sqrt(len(rep_means))) if len(rep_means) > 1 else 0.0)
        if not xs:
            continue
        x = np.asarray(xs)
        r = np.asarray(real_means)
        rci = np.asarray(real_ci)
        sm = np.asarray(syn_mean)
        sci = np.asarray(syn_ci)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, r, color="#2f6f9f", marker="o", linewidth=2.0, label="Real mean")
        ax.fill_between(x, r - rci, r + rci, color="#2f6f9f", alpha=0.18, label="Real 95% CI")
        ax.plot(x, sm, color="#c44e52", marker="s", linewidth=2.0, label="Synthetic mean")
        ax.fill_between(x, sm - sci, sm + sci, color="#c44e52", alpha=0.2, label="Synthetic 95% CI")
        ax.set_title(f"Longitudinal Mean + 95% CI Across Replicates: {name}", fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / f"{name}_replicate_mean_95ci.png")


def plot_longitudinal_replicate_medians(panel: LongitudinalPanel, long_reps: list[np.ndarray], output_dir: Path) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = _longitudinal_times(panel)
    for name in LONG_CONT_COLS:
        idx = _longitudinal_index(panel, name)
        if idx is None:
            continue
        xs, real_medians, rep_medians = [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            real_medians.append(float(np.nanmedian(real[:, visit, idx][obs])))
            rep_medians.append([float(np.nanmedian(rep[:, visit, idx][obs])) for rep in long_reps])
        if not xs:
            continue
        x = np.asarray(xs)
        reps = np.asarray(rep_medians)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, real_medians, color="black", linewidth=2.5, label="Real")
        for r in range(reps.shape[1]):
            ax.plot(x, reps[:, r], linewidth=1.0, alpha=0.28)
        ax.plot(x, reps.mean(axis=1), color="#c44e52", linestyle="--", linewidth=2.0, label="Synthetic mean")
        ax.set_title(f"Median Trajectory Replicates: {name}", fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / f"median_replicates_{name}.png")


def _plot_full_cohort_protocol(bundle: PDC2Bundle, static_all: pd.DataFrame, long_all: pd.DataFrame, output_dir: Path) -> None:
    real_df = _numeric(bundle.raw_df, COLS)
    synth_list = []
    long_reps = []
    for rep in sorted(static_all["replicate"].unique()):
        syn = static_all[static_all["replicate"] == rep].drop(columns=["replicate"]).reset_index(drop=True)
        synth_list.append(_numeric(syn, COLS))
        rep_csv = long_all[long_all["replicate"] == rep].drop(columns=["replicate"])
        values = np.full_like(bundle.longitudinal.raw_values, np.nan, dtype=np.float32)
        id_to_i = {int(pid): i for i, pid in enumerate(bundle.longitudinal.subject_ids)}
        for row in rep_csv.itertuples(index=False):
            i = id_to_i.get(int(getattr(row, "patient_id")))
            visit = int(getattr(row, "visit_index"))
            if i is None or visit >= values.shape[1]:
                continue
            for j, spec in enumerate(bundle.longitudinal.specs):
                values[i, visit, j] = float(getattr(row, spec.name))
        long_reps.append(values)
    if not synth_list:
        return
    primary = synth_list[0]
    common_static_cols = [c for c in real_df.columns if c in primary.columns]
    real_df = real_df[common_static_cols]
    synth_list = [s[common_static_cols] for s in synth_list]
    primary = synth_list[0]
    figure_dir = output_dir / "figures"
    plot_continuous_distributions(real_df, primary, figure_dir)
    plot_categorical_distributions(real_df, primary, figure_dir)
    plot_correlation_matrices(real_df, primary, figure_dir)
    plot_qq(real_df, primary, figure_dir)
    plot_summary_statistics(real_df, primary, figure_dir)
    plot_continuous_replicates(real_df, synth_list, figure_dir / "covariate" / "continuous_distributions_replicates.png")
    plot_categorical_replicates(real_df, synth_list, figure_dir / "covariate" / "categorical_distributions_replicates.png")

    plot_survival(real_df, primary, figure_dir)
    plot_km_curves_replicates(real_df, synth_list, figure_dir / "survival" / "km_curves_replicates.png")
    plot_survival_time_replicates(real_df, synth_list, figure_dir / "survival" / "survival_time_dist_replicates.png")
    corr_cols = [c for c in ["time"] + ALL_COVARIATES if c in real_df.columns and c in primary.columns]
    corr_r = _corr(real_df, corr_cols)
    corr_s = _corr(primary, corr_cols)
    diff = corr_r - corr_s
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    _heatmap(axes[0], corr_r, "Real Data Correlation", "RdBu_r")
    _heatmap(axes[1], corr_s, "Synthetic Rep 1 Correlation", "RdBu_r")
    _heatmap(axes[2], diff, "Difference (Real - Synthetic)", "RdBu_r", vmin=-0.5, vmax=0.5)
    fig.suptitle("Pairwise Correlation: Time + All Covariates", fontsize=14, fontweight="bold", y=1.02)
    _savefig(fig, figure_dir / "survival" / "correlation_heatmap_replicates.png")

    primary_long = long_reps[0]
    plot_trajectories(bundle.longitudinal, primary_long, figure_dir)
    plot_mean_ci(bundle.longitudinal, primary_long, figure_dir)
    plot_visit_correlations(bundle.longitudinal, primary_long, figure_dir)
    plot_variable_corr_per_visit(bundle.longitudinal, primary_long, figure_dir)
    plot_longitudinal_replicate_means(bundle.longitudinal, long_reps, figure_dir / "replicate_mean_95ci")
    plot_longitudinal_replicate_medians(bundle.longitudinal, long_reps, figure_dir / "trajectories_replicates")


def _summarize_replicates(
    metrics_rows: list[dict[str, float]],
    model: torch.nn.Module,
    cfg: dict[str, Any],
    output_dir: Path,
    n_subjects: int,
) -> dict[str, Any]:
    df = pd.DataFrame(metrics_rows)
    summary: dict[str, Any] = {
        "candidate": cfg["compact_candidate"],
        "n_replicates": int(len(df)),
        "n_subjects": int(n_subjects),
        "generated_total_subject_rows": int(len(df) * n_subjects),
        "full_cohort_reference": True,
        "posterior_generation": True,
        "posterior_generation_kind": "full-cohort posterior predictive",
        "embedding_dims_leq_6": _check_embedding_dims(cfg),
    }
    summary.update(_count_parameters(model))
    for col in df.columns:
        if col == "replicate" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_sd"] = float(df[col].std(ddof=0))
    param_penalty = summary["parameter_count"] / 1_000_000.0
    summary["generation_similarity_score"] = float(sum(float(summary.get(key, 0.0)) for key in SIMILARITY_KEYS) + 0.05 * param_penalty)
    with open(output_dir / "posterior_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _train_and_evaluate(candidate: str, args: argparse.Namespace) -> dict[str, Any]:
    cfg = _candidate_config(candidate, args)
    output_dir = Path(args.output_root) / candidate
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_pdc2_bundle(cfg)
    result = train_model(bundle, cfg, output_dir=output_dir, overfit_name=None)
    model = result["model"].to(torch.device(cfg["training"].get("device", "cpu")))
    audit = _parameter_audit(cfg, model)
    with open(output_dir / "parameter_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    if not audit["passes_audit"]:
        raise RuntimeError(f"Parameter audit failed for {candidate}: {audit}")
    with open(output_dir / "compact_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    static_all, long_all, metrics_rows = _posterior_replicates(
        model,
        bundle,
        torch.device(cfg["training"].get("device", "cpu")),
        output_dir,
        int(args.n_replicates),
        int(args.seed) + 1000,
    )
    summary = _summarize_replicates(metrics_rows, model, cfg, output_dir, len(bundle.raw_df))
    summary.update(audit)
    summary["train_final_loss"] = float(result["curves"]["loss"].dropna().iloc[-1]) if not result["curves"]["loss"].dropna().empty else math.nan
    summary["train_loss_decrease"] = float((result["curves"]["loss"].dropna().iloc[0] - result["curves"]["loss"].dropna().iloc[-1]) / max(abs(result["curves"]["loss"].dropna().iloc[0]), 1e-8)) if len(result["curves"]["loss"].dropna()) >= 2 else 0.0
    summary["nan_epoch_count"] = int(result["curves"]["nan_epoch"].astype(bool).sum()) if "nan_epoch" in result["curves"] else 0
    if not args.skip_plots:
        _plot_full_cohort_protocol(bundle, static_all, long_all, output_dir)
    del static_all, long_all
    with open(output_dir / "posterior_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def _write_search_summary(output_root: Path) -> None:
    rows = []
    for path in sorted(output_root.glob("round*/posterior_generation_summary.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        return
    df = pd.DataFrame(rows).sort_values("generation_similarity_score")
    df.to_csv(output_root / "compact_search_summary.csv", index=False)
    best = df.iloc[0].to_dict()
    with open(output_root / "best_candidate.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)
    md_cols = [
        "candidate",
        "generation_similarity_score",
        "parameter_count",
        "generated_total_subject_rows",
        "static_continuous_mean_ks_mean",
        "static_categorical_mean_tv_mean",
        "static_corr_mae_mean",
        "survival_km_integrated_abs_error_mean",
        "event_rate_diff_mean",
        "survival_time_median_diff_scaled_mean",
        "long_continuous_ks_mean",
        "long_continuous_mean_trend_rmse_ratio_mean",
        "long_categorical_tv_mean",
        "train_loss_decrease",
        "nan_epoch_count",
    ]
    with open(output_root / "compact_search_summary.md", "w", encoding="utf-8") as f:
        f.write("# PhaseSyn Compact Posterior Search\n\n")
        f.write("Lower `generation_similarity_score` is better. All candidates use full-cohort posterior generation, no overfit copy/calibration shortcuts, and latent/embedding dimensions <= 6.\n\n")
        f.write(df[[c for c in md_cols if c in df.columns]].to_markdown(index=False))
        f.write("\n\n")
        f.write(f"Best candidate: `{best['candidate']}`\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train compact non-overfit PhaseSyn models and evaluate posterior generation similarity.")
    parser.add_argument("--candidate", choices=[*CANDIDATES.keys(), "all"], default="all")
    parser.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260502)
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/pdc2/model/compact_posterior_search")
    parser.add_argument("--epochs-override", type=int, default=None)
    parser.add_argument("--use-randomization-loss", action="store_true")
    parser.add_argument("--randomization-loss-weight", type=float, default=0.0)
    parser.add_argument("--randomization-loss-warmup-epochs", type=int, default=0)
    parser.add_argument("--randomization-loss-ramp-epochs", type=int, default=1)
    parser.add_argument("--randomization-mmd-bandwidths", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--randomization-loss-on", choices=["z_mean", "z_sample"], default="z_mean")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    if args.list:
        for name, spec in CANDIDATES.items():
            print(name, json.dumps(spec, sort_keys=True))
        return
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        _write_search_summary(output_root)
        return
    names = list(CANDIDATES) if args.candidate == "all" else [args.candidate]
    for name in names:
        _train_and_evaluate(name, args)
    _write_search_summary(output_root)


if __name__ == "__main__":
    main()
