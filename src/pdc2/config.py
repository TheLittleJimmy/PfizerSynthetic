from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT / "data" / "pbc2"
MAX_HIDDEN_DIM = 6


def _hidden_dim(value: Any, default: int = MAX_HIDDEN_DIM, max_dim: int | None = None) -> int:
    cap = MAX_HIDDEN_DIM if max_dim is None else max(1, int(max_dim))
    return max(1, min(int(default if value is None else value), cap))


DEFAULT_CONFIG: dict[str, Any] = {
    "dataset": {
        "name": "pdc2",
        "data_dir": str(DEFAULT_DATA_DIR),
        "output_root": str(ROOT / "outputs" / "pdc2"),
        "max_visits": None,
    },
    "model": {
        "longitudinal_mode": "latent_ode",
        "survival": "dynamic",
        "max_hidden_dim": MAX_HIDDEN_DIM,
        "z_dim": 6,
        "s_dim": 6,
        "y_dim_static": 6,
        "u_dim": 6,
        "gru_hidden_dim": 6,
        "ode_hidden_dim": 6,
        "n_intervals": 16,
        "u0_init_mode": "baseline_l0",
        "stochastic_u0": True,
        "u0_sigma_mode": "learned",
        "u0_fixed_sigma": 0.05,
        "u0_sigma_min": 0.03,
        "u0_kl_weight": 0.0,
        "use_u0_mean_at_eval": False,
        "encoder_conditioning": "baseline_only",
        "detach_l0_for_u0_init": False,
        "baseline_time_eps": 1e-6,
        "lambda_l0_hivae": 1.0,
        "baseline_long_weight": 1.0,
        "lambda_surv": 1.0,
        "survival_event_weight": 1.0,
        "survival_event_aux_weight": 0.0,
        "survival_time_aux_weight": 0.0,
        "survival_time_head_weight": 0.0,
        "survival_warmup_epochs": 0,
        "admin_censoring_mode": "event_and_censor_survival",
        "admin_end_threshold": 1.0 - 1e-6,
        "decoder_hidden_dim": 6,
        "u0_initializer_hidden_dim": 6,
        "dynamic_survival_hidden_dim": 6,
        "dynamic_survival_num_layers": 2,
        "dynamic_survival_dropout": 0.0,
        "survival_history_pooling": "boundary",
        "condition_ode_on_baseline": True,
        "condition_longitudinal_decoder_on_baseline": True,
        "generation_baseline_mode": "sampled",
        "kl_weight_s": 1.0,
        "kl_weight_z": 1.0,
        "kl_weight_u": 1.0,
        "static_weight": 1.0,
        "longitudinal_weight": 1.0,
        "use_randomization_loss": False,
        "randomization_loss_weight": 0.0,
        "randomization_loss_warmup_epochs": 0,
        "randomization_loss_ramp_epochs": 1,
        "randomization_mmd_bandwidths": [0.5, 1.0, 2.0, 4.0],
        "randomization_loss_on": "z_mean",
        "treatment_variable_name": "drug",
    },
    "training": {
        "epochs": 30,
        "batch_size": 64,
        "lr": 1e-3,
        "seed": 1,
        "early_stopping": False,
        "subset_size": None,
        "device": "cpu",
        "n_generated_dataset": 1,
    },
    "generation": {
        "prior_n": 100,
        "prior_treatment": 0,
        "time_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "deterministic": False,
    },
    "overfit": {
        "settings": ["small", "medium", "large"],
        "subset_size": 32,
        "seed": 1,
        "disable_early_stopping": True,
        "min_loss_decrease": 0.02,
        "rmse_ratio_threshold": 1.05,
        "event_rate_tolerance": 0.02,
        "km_error_threshold": 0.05,
        "survival_time_rmse_ratio_threshold": 0.35,
        "survival_event_accuracy_threshold": 0.90,
        "static_paired_rmse_ratio_threshold": 0.35,
        "static_paired_categorical_accuracy_threshold": 0.90,
        "static_continuous_ks_threshold": 0.10,
        "static_categorical_tv_threshold": 0.10,
        "raw_model_static_paired_rmse_ratio_threshold": 1.25,
        "raw_model_static_paired_categorical_accuracy_threshold": 0.50,
        "raw_model_survival_time_rmse_ratio_threshold": 1.50,
        "raw_model_survival_event_accuracy_threshold": 0.95,
    },
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        path_obj = Path(path)
        if not path_obj.is_absolute() and not path_obj.exists():
            candidate = ROOT / path_obj
            if candidate.exists():
                path_obj = candidate
        with open(path_obj, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        cfg = deep_update(cfg, loaded)
    if overrides:
        cfg = deep_update(cfg, overrides)
    normalise_config(cfg)
    return cfg


def normalise_config(cfg: dict[str, Any]) -> None:
    dataset = cfg.setdefault("dataset", {})
    name = dataset.get("name", "pdc2")
    if name not in {"pdc2", "pbc2"}:
        raise ValueError(f"Unsupported dataset {name!r}; expected 'pdc2' or 'pbc2'.")
    dataset["name"] = "pdc2" if name == "pdc2" else "pbc2"

    data_dir = Path(dataset.get("data_dir", DEFAULT_DATA_DIR)).expanduser()
    output_root = Path(dataset.get("output_root", ROOT / "outputs" / "pdc2")).expanduser()
    if not data_dir.is_absolute():
        data_dir = (ROOT / data_dir).resolve()
    if not output_root.is_absolute():
        output_root = (ROOT / output_root).resolve()
    dataset["data_dir"] = str(data_dir)
    dataset["output_root"] = str(output_root)

    model = cfg.setdefault("model", {})
    if "stage" in model:
        raise ValueError("model.stage is no longer supported; PhaseSyn uses a single model.")
    model["max_hidden_dim"] = int(model.get("max_hidden_dim", MAX_HIDDEN_DIM))
    if model["max_hidden_dim"] <= 0:
        raise ValueError("model.max_hidden_dim must be positive.")
    if model.get("longitudinal_mode", "latent_ode") != "latent_ode":
        raise ValueError("Only model.longitudinal_mode = latent_ode is supported.")
    model["longitudinal_mode"] = "latent_ode"
    if model.get("survival", "dynamic") != "dynamic":
        raise ValueError("model.survival must be dynamic.")
    model["survival"] = "dynamic"
    if model.get("u0_init_mode", "baseline_l0") != "baseline_l0":
        raise ValueError("model.u0_init_mode must be baseline_l0 for dynamic survival.")
    model["u0_init_mode"] = model.get("u0_init_mode", "baseline_l0")
    model["stochastic_u0"] = bool(model.get("stochastic_u0", True))
    model["u0_sigma_mode"] = str(model.get("u0_sigma_mode", "learned"))
    if model["u0_sigma_mode"] not in {"learned", "fixed"}:
        raise ValueError("model.u0_sigma_mode must be learned or fixed.")
    model["u0_fixed_sigma"] = float(model.get("u0_fixed_sigma", 0.05))
    if model["u0_fixed_sigma"] <= 0:
        raise ValueError("model.u0_fixed_sigma must be positive.")
    model["u0_sigma_min"] = float(model.get("u0_sigma_min", 0.03))
    if model["u0_sigma_min"] < 0:
        raise ValueError("model.u0_sigma_min must be nonnegative.")
    model["u0_kl_weight"] = float(model.get("u0_kl_weight", 0.0))
    if model["u0_kl_weight"] < 0:
        raise ValueError("model.u0_kl_weight must be nonnegative.")
    model["use_u0_mean_at_eval"] = bool(model.get("use_u0_mean_at_eval", False))
    if model.get("encoder_conditioning", "baseline_only") != "baseline_only":
        raise ValueError("model.encoder_conditioning must be baseline_only for dynamic survival.")
    model["encoder_conditioning"] = "baseline_only"
    model["baseline_long_weight"] = float(model.get("baseline_long_weight", 1.0))
    model["lambda_surv"] = float(model.get("lambda_surv", 1.0))
    model["survival_event_weight"] = float(model.get("survival_event_weight", 1.0))
    if model["survival_event_weight"] <= 0:
        raise ValueError("model.survival_event_weight must be positive.")
    model["survival_event_aux_weight"] = float(model.get("survival_event_aux_weight", 0.0))
    model["survival_time_aux_weight"] = float(model.get("survival_time_aux_weight", 0.0))
    model["survival_time_head_weight"] = float(model.get("survival_time_head_weight", 0.0))
    if model["survival_event_aux_weight"] < 0 or model["survival_time_aux_weight"] < 0 or model["survival_time_head_weight"] < 0:
        raise ValueError("survival auxiliary weights must be nonnegative.")
    model["survival_warmup_epochs"] = int(model.get("survival_warmup_epochs", 0))
    model.pop("detach_survival_longitudinal_repr", None)
    model["admin_censoring_mode"] = str(model.get("admin_censoring_mode", "event_and_censor_survival"))
    if model["admin_censoring_mode"] != "event_and_censor_survival":
        raise ValueError("model.admin_censoring_mode must be event_and_censor_survival.")
    model["admin_end_threshold"] = float(model.get("admin_end_threshold", 1.0 - 1e-6))
    max_hidden_dim = int(model["max_hidden_dim"])
    for latent_key in ("z_dim", "s_dim", "y_dim_static", "u_dim"):
        if latent_key in model:
            model[latent_key] = _hidden_dim(model[latent_key], max_dim=max_hidden_dim)
    for key in [
        "gru_hidden_dim",
        "ode_hidden_dim",
        "decoder_hidden_dim",
        "u0_initializer_hidden_dim",
        "l0_embedding_dim",
        "dynamic_survival_hidden_dim",
    ]:
        if key in model:
            model[key] = _hidden_dim(model[key], max_dim=max_hidden_dim)
    model["dynamic_survival_hidden_dim"] = _hidden_dim(
        model.get("dynamic_survival_hidden_dim", MAX_HIDDEN_DIM),
        max_dim=max_hidden_dim,
    )
    model["dynamic_survival_num_layers"] = int(model.get("dynamic_survival_num_layers", 2))
    model["dynamic_survival_dropout"] = float(model.get("dynamic_survival_dropout", 0.0))
    if model.get("survival_history_pooling", "boundary") != "boundary":
        raise ValueError("model.survival_history_pooling must be boundary.")
    model["survival_history_pooling"] = "boundary"
    model["use_randomization_loss"] = bool(model.get("use_randomization_loss", False))
    model["randomization_loss_weight"] = float(model.get("randomization_loss_weight", 0.0))
    model["randomization_loss_warmup_epochs"] = int(model.get("randomization_loss_warmup_epochs", 0))
    model["randomization_loss_ramp_epochs"] = int(model.get("randomization_loss_ramp_epochs", 1))
    if isinstance(model.get("randomization_mmd_bandwidths"), str):
        model["randomization_mmd_bandwidths"] = [
            float(x) for x in model["randomization_mmd_bandwidths"].replace(",", " ").split() if x
        ]
    else:
        model["randomization_mmd_bandwidths"] = [float(x) for x in model.get("randomization_mmd_bandwidths", [0.5, 1.0, 2.0, 4.0])]
    if not model["randomization_mmd_bandwidths"]:
        raise ValueError("model.randomization_mmd_bandwidths must contain at least one positive bandwidth.")
    if any(x <= 0 for x in model["randomization_mmd_bandwidths"]):
        raise ValueError("model.randomization_mmd_bandwidths must be positive.")
    if model.get("randomization_loss_on", "z_mean") not in {"z_mean", "z_sample"}:
        raise ValueError("model.randomization_loss_on must be z_mean or z_sample.")
    model["randomization_loss_on"] = model.get("randomization_loss_on", "z_mean")

    training = cfg.setdefault("training", {})
    training["epochs"] = int(training.get("epochs", 30))
    training["batch_size"] = int(training.get("batch_size", 64))
    training["seed"] = int(training.get("seed", 1))
    if training.get("subset_size") is not None:
        training["subset_size"] = int(training["subset_size"])

    generation = cfg.setdefault("generation", {})
    generation["prior_n"] = int(generation.get("prior_n", 100))
    if generation["prior_n"] <= 0:
        raise ValueError("generation.prior_n must be positive.")
    generation["prior_treatment"] = int(generation.get("prior_treatment", 0))
    generation["deterministic"] = bool(generation.get("deterministic", False))
    if isinstance(generation.get("time_grid"), str):
        generation["time_grid"] = [
            float(x) for x in generation["time_grid"].replace(",", " ").split() if x
        ]
    else:
        generation["time_grid"] = [float(x) for x in generation.get("time_grid", [0.0, 0.25, 0.5, 0.75, 1.0])]
    if not generation["time_grid"]:
        raise ValueError("generation.time_grid must contain at least one time point.")


def model_output_dir(cfg: dict[str, Any], overfit_name: str | None = None) -> Path:
    root = Path(cfg["dataset"]["output_root"])
    base = root / "model"
    if overfit_name:
        return base / "overfit" / overfit_name
    return base


def config_for_overfit(cfg: dict[str, Any], setting_name: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    settings = out.get("overfit_settings", {})
    if setting_name in settings:
        out = deep_update(out, settings[setting_name])
    out["training"]["subset_size"] = int(out.get("overfit", {}).get("subset_size", 32))
    out["training"]["early_stopping"] = False
    model = out.setdefault("model", {})
    if "stage" in model:
        raise ValueError("model.stage is no longer supported; PhaseSyn uses a single model.")
    model["longitudinal_mode"] = "latent_ode"
    training = out.setdefault("training", {})
    overfit = out.setdefault("overfit", {})
    evaluation = out.setdefault("evaluation", {})
    evaluation["deterministic_static_export"] = True
    evaluation["copy_static_overfit_reference"] = False
    evaluation["calibrate_static_covariates"] = True
    evaluation["calibrate_survival_km"] = False
    evaluation["copy_survival_overfit_reference"] = False
    evaluation["calibrate_longitudinal_observed"] = True
    training["batch_size"] = int(out["training"]["subset_size"])
    overfit["rmse_ratio_threshold"] = min(float(overfit.get("rmse_ratio_threshold", 1.05)), 0.35)
    overfit["median_trend_ratio_threshold"] = min(float(overfit.get("median_trend_ratio_threshold", 1.05)), 0.35)
    overfit["km_error_threshold"] = min(float(overfit.get("km_error_threshold", 0.05)), 0.02)
    overfit["survival_time_rmse_ratio_threshold"] = min(float(overfit.get("survival_time_rmse_ratio_threshold", 0.35)), 0.05)
    overfit["survival_event_accuracy_threshold"] = max(float(overfit.get("survival_event_accuracy_threshold", 0.90)), 0.95)
    overfit["static_paired_rmse_ratio_threshold"] = min(float(overfit.get("static_paired_rmse_ratio_threshold", 0.35)), 0.10)
    overfit["static_paired_categorical_accuracy_threshold"] = max(float(overfit.get("static_paired_categorical_accuracy_threshold", 0.90)), 0.95)
    overfit["static_continuous_ks_threshold"] = min(float(overfit.get("static_continuous_ks_threshold", 0.10)), 0.05)
    overfit["static_categorical_tv_threshold"] = min(float(overfit.get("static_categorical_tv_threshold", 0.10)), 0.05)
    overfit["raw_model_survival_event_accuracy_threshold"] = max(float(overfit.get("raw_model_survival_event_accuracy_threshold", 0.95)), 0.95)
    model["max_hidden_dim"] = int(model.get("max_hidden_dim", MAX_HIDDEN_DIM))
    if model["max_hidden_dim"] <= 0:
        raise ValueError("model.max_hidden_dim must be positive.")
    max_hidden_dim = int(model["max_hidden_dim"])
    model["gru_hidden_dim"] = _hidden_dim(model.get("gru_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
    model["decoder_hidden_dim"] = _hidden_dim(model.get("decoder_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
    model["ode_hidden_dim"] = _hidden_dim(model.get("ode_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
    model["dynamic_survival_hidden_dim"] = _hidden_dim(
        model.get("dynamic_survival_hidden_dim", MAX_HIDDEN_DIM),
        max_dim=max_hidden_dim,
    )
    for latent_key in ("z_dim", "s_dim", "y_dim_static", "u_dim"):
        model[latent_key] = _hidden_dim(model.get(latent_key, MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
    model["continuous_mse_weight"] = max(float(model.get("continuous_mse_weight", 0.0)), 25.0)
    model["deterministic_u"] = True
    model["longitudinal_only_loss"] = False
    model["survival"] = "dynamic"
    model["kl_weight_s"] = 0.0
    model["kl_weight_z"] = 0.0
    model["kl_weight_u"] = 0.0
    training["epochs"] = max(int(training.get("epochs", 0)), 500)
    return out
