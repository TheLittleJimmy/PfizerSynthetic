from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))
if str(ROOT / "utils") not in sys.path:
    sys.path.insert(2, str(ROOT / "utils"))

from utils import data_processing  # noqa: E402
from pdc2.config import load_config as load_pdc_config  # noqa: E402
from pdc2.data import LongitudinalPanel, LongitudinalSpec, PDC2Bundle, validate_complete_l0, y_dim_partition_for_types  # noqa: E402
from pdc2.models import build_model, set_seed  # noqa: E402
from pdc2.training import train_model  # noqa: E402

from .dgm import TrialData


def baseline_columns(n_baseline: int) -> list[str]:
    return [f"X{k + 1:02d}" for k in range(int(n_baseline))]


def biomarker_columns(n_biomarkers: int) -> list[str]:
    return [f"L{k + 1:02d}" for k in range(int(n_biomarkers))]


def trial_static_frame(trial: TrialData) -> pd.DataFrame:
    n = trial.X.shape[0]
    frame = pd.DataFrame(trial.X, columns=baseline_columns(trial.X.shape[1]))
    frame.insert(0, "subject_id", np.arange(n, dtype=int))
    frame["A"] = trial.A.astype(int)
    for j, col in enumerate(biomarker_columns(trial.L.shape[2])):
        frame[col] = trial.L[:, 0, j]
    frame["time"] = trial.T_obs
    frame["censor"] = trial.delta.astype(int)
    frame["event"] = trial.delta.astype(int)
    return frame


def trial_longitudinal_frame(trial: TrialData, masked: bool = True) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    names = biomarker_columns(trial.L.shape[2])
    for i in range(trial.L.shape[0]):
        for visit, t in enumerate(trial.time_grid):
            row: dict[str, Any] = {
                "patient_id": int(i),
                "subject_id": int(i),
                "visit_index": int(visit),
                "visit_time": float(t),
                "A": int(trial.A[i]),
            }
            for ell, name in enumerate(names):
                value = float(trial.L[i, visit, ell])
                if masked and trial.R[i, visit, ell] < 0.5:
                    value = np.nan
                row[name] = value
            rows.append(row)
    return pd.DataFrame(rows)


def _baseline_type_for_index(index_1based: int) -> dict[str, str]:
    name = f"X{index_1based:02d}"
    if 7 <= index_1based <= 10 or 21 <= index_1based <= 30:
        return {"name": name, "type": "cat", "dim": "1", "nclass": "2"}
    return {"name": name, "type": "real", "dim": "1", "nclass": ""}


def types_for_trial(n_baseline: int, n_biomarkers: int) -> list[dict[str, str]]:
    types: list[dict[str, str]] = [{"name": "survcens", "type": "surv_dynamic", "dim": "2", "nclass": ""}]
    types.extend(_baseline_type_for_index(k + 1) for k in range(int(n_baseline)))
    types.extend({"name": col, "type": "real", "dim": "1", "nclass": ""} for col in biomarker_columns(n_biomarkers))
    types.append({"name": "A", "type": "cat", "dim": "1", "nclass": "2"})
    return types


def _encoded_static(raw: pd.DataFrame, types: list[dict[str, str]]) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    encoded_parts: list[np.ndarray] = []
    encoded_names: list[str] = []
    miss = torch.ones((len(raw), len(types)), dtype=torch.float32)
    for idx, feature in enumerate(types):
        name = "time" if feature["type"].startswith("surv") else feature["name"]
        ftype = feature["type"]
        if ftype.startswith("surv"):
            encoded_parts.append(raw[["time", "censor"]].to_numpy(dtype=np.float32))
            encoded_names.extend(["time", "censor"])
            continue
        series = pd.to_numeric(raw[name], errors="coerce")
        missing = series.isna().to_numpy()
        if missing.any():
            miss[missing, idx] = 0.0
        if ftype in {"cat", "ordinal"}:
            nclass = int(feature["nclass"])
            vals = series.fillna(0).round().clip(0, nclass - 1).astype(int).to_numpy()
            one_hot = np.zeros((len(raw), nclass), dtype=np.float32)
            one_hot[np.arange(len(raw)), vals] = 1.0
            encoded_parts.append(one_hot)
            encoded_names.extend([f"{feature['name']}_{j}" for j in range(nclass)])
        else:
            encoded_parts.append(series.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1))
            encoded_names.append(feature["name"])
    return pd.DataFrame(np.concatenate(encoded_parts, axis=1), columns=encoded_names), miss, miss.clone()


def _longitudinal_panel(trial: TrialData, masked: bool = True) -> LongitudinalPanel:
    l = trial.L.astype(np.float32).copy()
    masks = trial.R.astype(np.float32).copy() if masked else np.ones_like(l, dtype=np.float32)
    masks[:, 0, :] = 1.0
    raw_values = l.copy()
    values = np.nan_to_num(l, nan=0.0).astype(np.float32)
    specs: list[LongitudinalSpec] = []
    for j, name in enumerate(biomarker_columns(l.shape[2])):
        obs = l[:, :, j][masks[:, :, j].astype(bool)]
        mean = float(np.mean(obs)) if obs.size else 0.0
        std = max(float(np.std(obs)) if obs.size else 1.0, 1e-6)
        values[:, :, j] = ((values[:, :, j] - mean) / std) * masks[:, :, j]
        specs.append(LongitudinalSpec(name=name, type="real", nclass=None, mean=mean, std=std))
    times = np.tile(trial.time_grid.reshape(1, -1), (trial.L.shape[0], 1)).astype(np.float32)
    panel = LongitudinalPanel(
        subject_ids=np.arange(trial.L.shape[0], dtype=int),
        times=torch.tensor(times, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        masks=torch.tensor(masks, dtype=torch.float32),
        raw_values=raw_values,
        specs=specs,
        time_min=float(trial.time_grid[0]),
        time_max=float(trial.time_grid[-1]),
    )
    validate_complete_l0(panel, 1e-6)
    return panel


def trial_to_bundle(trial: TrialData, masked: bool = True) -> PDC2Bundle:
    raw = trial_static_frame(trial)
    raw_for_model = raw[["time", "censor", *baseline_columns(trial.X.shape[1]), *biomarker_columns(trial.L.shape[2]), "A"]].copy()
    types_full = types_for_trial(trial.X.shape[1], trial.L.shape[2])
    hivae_types = [dict(t) for t in types_full if t["name"] != "A"]
    encoded, miss, true_miss = _encoded_static(raw_for_model, hivae_types)
    l0_names = set(biomarker_columns(trial.L.shape[2]))
    for idx, feature in enumerate(hivae_types):
        if feature["name"] in l0_names:
            miss[:, idx] = 1.0
            true_miss[:, idx] = 1.0
    treatment = torch.nn.functional.one_hot(torch.tensor(trial.A.astype(int)), num_classes=2).float()
    ids = pd.DataFrame({"id": np.arange(len(raw_for_model), dtype=int)})
    return PDC2Bundle(
        raw_df=raw_for_model,
        encoded_df=encoded,
        types=hivae_types,
        miss_mask=miss,
        true_miss_mask=true_miss,
        longitudinal=_longitudinal_panel(trial, masked=masked),
        ids_df=ids,
        y_dim_partition=y_dim_partition_for_types(hivae_types, 8),
        static_feature_count=len(hivae_types),
        treatment=treatment,
        treatment_name="A",
        treatment_n_classes=2,
        category_values={
            **{f"X{k:02d}": [0.0, 1.0] for k in [*range(7, 11), *range(21, 31)]},
            "A": [0.0, 1.0],
        },
    )


def build_phasesyn_config(cfg: dict[str, Any], seed: int, output_root: Path) -> dict[str, Any]:
    train = cfg.get("phasesyn_training", {})
    max_hidden = int(train.get("hidden_dim", 64))
    pdc = load_pdc_config(
        None,
        {
            "dataset": {
                "name": "pdc2",
                "data_dir": str(output_root / "phasesyn_bundle"),
                "output_root": str(output_root),
                "max_visits": int(cfg["n_timepoints"]),
            },
            "model": {
                "longitudinal_mode": "latent_ode",
                "survival": "dynamic",
                "max_hidden_dim": max_hidden,
                "z_dim": int(train.get("latent_z_dim", 8)),
                "s_dim": int(train.get("latent_z_dim", 8)),
                "y_dim_static": int(train.get("latent_z_dim", 8)),
                "u_dim": int(train.get("latent_u_dim", 8)),
                "gru_hidden_dim": max_hidden,
                "ode_hidden_dim": int(train.get("ode_hidden_dim", 64)),
                "decoder_hidden_dim": max_hidden,
                "u0_initializer_hidden_dim": max_hidden,
                "dynamic_survival_hidden_dim": max_hidden,
                "dynamic_survival_num_layers": int(train.get("dynamic_survival_num_layers", 2)),
                "dynamic_survival_dropout": float(train.get("dynamic_survival_dropout", 0.0)),
                "n_intervals": int(train.get("survival_intervals", cfg["n_timepoints"] - 1)),
                "baseline_time_eps": 1e-6,
                "u0_init_mode": "baseline_l0",
                "u0_sigma_mode": str(train.get("u0_sigma_mode", "learned")),
                "u0_fixed_sigma": float(train.get("u0_fixed_sigma", 0.05)),
                "u0_sigma_min": float(train.get("u0_sigma_min", 0.03)),
                "u0_kl_weight": float(train.get("u0_kl_weight", 0.0)),
                "use_u0_mean_at_eval": bool(train.get("use_u0_mean_at_eval", False)),
                "encoder_conditioning": "baseline_only",
                "condition_ode_on_baseline": True,
                "condition_longitudinal_decoder_on_baseline": True,
                "static_weight": float(train.get("weight_baseline", 1.0)),
                "longitudinal_weight": float(train.get("weight_longitudinal", 1.0)),
                "lambda_surv": float(train.get("weight_survival", 1.0)),
                "survival_event_weight": float(train.get("survival_event_weight", 1.0)),
                "survival_event_aux_weight": float(train.get("survival_event_aux_weight", 0.0)),
                "survival_time_aux_weight": float(train.get("survival_time_aux_weight", 0.0)),
                "survival_time_head_weight": float(train.get("survival_time_head_weight", 0.0)),
                "survival_warmup_epochs": int(train.get("survival_warmup_epochs", 0)),
                "use_randomization_loss": float(train.get("weight_randomization", 0.0)) > 0.0,
                "randomization_loss_weight": float(train.get("weight_randomization", 0.0)),
                "randomization_loss_warmup_epochs": int(train.get("randomization_loss_warmup_epochs", 0)),
                "randomization_loss_ramp_epochs": int(train.get("randomization_loss_ramp_epochs", 1)),
                "randomization_loss_on": str(train.get("randomization_loss_on", "z_mean")),
                "treatment_variable_name": "A",
            },
            "training": {
                "epochs": int(train.get("epochs", 300)),
                "batch_size": int(train.get("batch_size", 64)),
                "lr": float(train.get("learning_rate", 1e-3)),
                "seed": int(seed),
                "device": str(train.get("device", "cpu")),
                "early_stopping": False,
            },
            "generation": {
                "time_grid": list(cfg["time_grid"]),
            },
        },
    )
    return pdc


def train_phasesyn_model(
    trial: TrialData,
    cfg: dict[str, Any],
    seed: int,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_seed(int(seed))
    bundle = trial_to_bundle(trial, masked=True)
    pdc_cfg = build_phasesyn_config(cfg, seed, output)
    result = train_model(bundle, pdc_cfg, output_dir=output)
    return {
        "model": result["model"],
        "bundle": bundle,
        "pdc_config": pdc_cfg,
        "checkpoint": str(output / "model_checkpoint.pt"),
        "output_dir": str(output),
        "metrics": result.get("metrics", {}),
    }


def _feature_list_for_frame(bundle: PDC2Bundle, frame: pd.DataFrame, device: torch.device) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    hivae_types = bundle.types
    raw = frame[["time", "censor", *[f["name"] for f in hivae_types if not f["type"].startswith("surv")]]].copy()
    encoded, miss, _ = _encoded_static(raw, hivae_types)
    data_tensor = torch.tensor(encoded.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    miss = miss.to(device)
    data_list, miss_list = data_processing.next_batch(data_tensor, hivae_types, miss, len(raw), 0)
    return [d.to(device) for d in data_list], miss_list.to(device), data_tensor


def generate_phasesyn_trial(
    model: torch.nn.Module,
    bundle: PDC2Bundle,
    target_trial: TrialData,
    device: str | torch.device = "cpu",
    deterministic_latents: bool = False,
    deterministic_u0: bool | None = None,
    deterministic_survival: bool | None = None,
    survival_event_export: str = "sample",
    mask_longitudinal_by_observed_time: bool = True,
    sample_longitudinal_values: bool = False,
) -> dict[str, np.ndarray]:
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    target_frame = trial_static_frame(target_trial)
    raw = target_frame[["time", "censor", *baseline_columns(target_trial.X.shape[1]), *biomarker_columns(target_trial.L.shape[2]), "A"]].copy()
    raw["time"] = 1.0
    raw["censor"] = 0.0
    data_list, miss_list, _ = _feature_list_for_frame(bundle, raw, dev)
    with torch.no_grad():
        l0 = np.zeros((len(raw), len(bundle.longitudinal.specs)), dtype=np.float32)
        for j, spec in enumerate(bundle.longitudinal.specs):
            value = raw[spec.name].to_numpy(dtype=np.float32)
            l0[:, j] = (value - spec.mean) / max(spec.std, 1e-6)
        l0_tensor = torch.tensor(l0, dtype=torch.float32, device=dev)
        treatment = torch.nn.functional.one_hot(
            torch.tensor(target_trial.A.astype(int), dtype=torch.long, device=dev),
            num_classes=2,
        ).float()
        out = model.generate_observed_baseline(
            data_list,
            miss_list,
            l0_tensor,
            torch.tensor(target_trial.time_grid, dtype=torch.float32, device=dev),
            treatment,
            deterministic_latents=deterministic_latents,
            deterministic_u0=deterministic_u0,
        )
        if deterministic_survival is not None or survival_event_export != "sample":
            survival_out = out["dynamic_survival"]
            survival_sample = model.sample_dynamic_survival(
                survival_out,
                deterministic=bool(deterministic_survival),
            )
            if survival_event_export == "probability":
                survival_summary = model.dynamic_survival_distribution_summary(
                    survival_out["event_hazard"],
                    survival_out["censoring_hazard"],
                    survival_out["boundary_times"],
                )
                event_probability = survival_summary["event_probability"].view(-1, 1)
                survival_sample["event"] = (event_probability >= 0.5).to(event_probability.dtype)
            elif survival_event_export != "sample":
                raise ValueError("survival_event_export must be 'sample' or 'probability'.")
            out["event_time"] = model.denormalize_survival_time(survival_sample["event_time"])
            out["censoring_time"] = model.denormalize_survival_time(survival_sample["censoring_time"])
            out["observed_time"] = model.denormalize_survival_time(survival_sample["observed_time"])
            out["event"] = survival_sample["event"]
            out["event_time_normalized"] = survival_sample["event_time"]
            out["censoring_time_normalized"] = survival_sample["censoring_time"]
            out["observed_time_normalized"] = survival_sample["observed_time"]
    if mask_longitudinal_by_observed_time:
        longitudinal_tensor = out["longitudinal_mean"]
    else:
        if sample_longitudinal_values:
            times = torch.tensor(target_trial.time_grid, dtype=torch.float32, device=dev)
            longitudinal_tensor = model.decoder.sample_from_path_conditioned(
                out["u_path"],
                times.unsqueeze(0).expand(len(raw), -1).clone(),
                out["z"],
                out["s"],
                treatment,
                deterministic=False,
            )
        else:
            longitudinal_tensor = out.get("ode_longitudinal_mean", out["longitudinal_mean"]).clone()
        times = torch.tensor(target_trial.time_grid, dtype=torch.float32, device=dev)
        t0_rows = times.abs() <= float(getattr(model, "baseline_time_eps", 1e-6))
        if t0_rows.any():
            longitudinal_tensor[:, t0_rows, :] = l0_tensor.unsqueeze(1)
    longitudinal = longitudinal_tensor.detach().cpu().numpy()
    for j, spec in enumerate(bundle.longitudinal.specs):
        longitudinal[:, :, j] = longitudinal[:, :, j] * spec.std + spec.mean
    observed_time = out["observed_time"].detach().cpu().numpy().reshape(-1)
    event = out["event"].detach().cpu().numpy().reshape(-1).astype(int)
    return {
        "X": target_trial.X,
        "A": target_trial.A,
        "L": longitudinal,
        "T_obs": observed_time,
        "delta": event,
    }


def load_trained_phasesyn(checkpoint: str | Path, bundle: PDC2Bundle, pdc_cfg: dict[str, Any], device: str = "cpu") -> torch.nn.Module:
    model = build_model(bundle, pdc_cfg)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.to(torch.device(device))
    model.eval()
    return model


def clone_config_for_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.pop("_config_path", None)
    return out
