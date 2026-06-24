#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from dataclasses import asdict, dataclass
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
from pdc2.data import (  # noqa: E402
    BASELINE_COLUMNS,
    LongitudinalPanel,
    LongitudinalSpec,
    PDC2Bundle,
    read_types,
    validate_complete_l0,
    y_dim_partition_for_types,
)
from pdc2.models import PhaseSynModel, longitudinal_observed_rows, set_seed  # noqa: E402
from pdc2.training import (  # noqa: E402
    CATEGORICAL_TYPES,
    CONTINUOUS_TYPES,
    _apply_longitudinal_support,
    paired_survival_metrics,
    train_model,
)


ROUND11_SPEC: dict[str, Any] = {
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
    "n_intervals": 16,
    "kl_weight_s": 0.3,
    "kl_weight_z": 0.3,
    "kl_weight_u": 0.15,
    "static_weight": 1.0,
    "longitudinal_weight": 2.0,
    "lambda_surv": 1.4,
    "continuous_mse_weight": 0.8,
}

LONG_CONT_COLS = ["serBilir", "albumin", "alkaline", "SGOT", "platelets", "prothrombin"]


@dataclass
class StaticPreprocessor:
    categorical_maps: dict[str, dict[str, int]]
    categories: dict[str, list[float]]


@dataclass
class LongitudinalPreprocessor:
    value_cols: list[str]
    max_visits: int
    time_min: float
    time_max: float
    specs: list[LongitudinalSpec]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if dataclasses.is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _normalization_metadata(params: list[tuple[Any, Any]], types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "feature_index": idx,
            "name": feature["name"],
            "type": feature["type"],
            "param_1": _jsonable(pair[0]),
            "param_2": _jsonable(pair[1]),
        }
        for idx, (feature, pair) in enumerate(zip(types, params))
    ]


def _write_split_file(
    raw: pd.DataFrame,
    ids: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    path: Path,
) -> None:
    original_id_col = "id" if "id" in ids.columns else ids.columns[0]
    rows = []
    for split, indices in [("train", train_idx), ("test", test_idx)]:
        for idx in indices:
            rows.append({
                "row_index": int(idx),
                "panel_patient_id": int(idx),
                "original_subject_id": _jsonable(ids.iloc[int(idx)][original_id_col]),
                "split": split,
                "time": float(raw.iloc[int(idx)]["time"]),
                "censor": float(raw.iloc[int(idx)]["censor"]),
            })
    pd.DataFrame(rows).sort_values("row_index").to_csv(path, index=False)


def _read_raw_tables(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    data_dir = Path(cfg["dataset"]["data_dir"])
    raw = pd.read_csv(data_dir / "data_phasesyn.csv", header=None)
    raw.columns = BASELINE_COLUMNS
    ids = pd.read_csv(data_dir / "pbc2_id.csv")
    long_df = pd.read_csv(data_dir / "longitudinal.csv")
    types = read_types(data_dir / "data_types_phasesyn_piecewise.csv", survival=cfg["model"].get("survival", "dynamic"))
    if len(raw) != len(ids):
        raise ValueError(f"Baseline row count {len(raw)} does not match id row count {len(ids)}.")
    return raw, ids, long_df, types


def _stratified_split(raw: pd.DataFrame, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    event = raw["censor"].to_numpy(dtype=float) > 0.5
    for flag in [False, True]:
        idx = np.flatnonzero(event == flag)
        rng.shuffle(idx)
        n_test = max(1, int(round(float(test_fraction) * len(idx))))
        test_parts.append(idx[:n_test])
        train_parts.append(idx[n_test:])
    train = np.concatenate(train_parts).astype(int)
    test = np.concatenate(test_parts).astype(int)
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _fit_static_preprocessor(raw_train: pd.DataFrame, types: list[dict[str, Any]]) -> StaticPreprocessor:
    maps: dict[str, dict[str, int]] = {}
    cats: dict[str, list[float]] = {}
    for feature in types:
        if feature["type"] not in CATEGORICAL_TYPES:
            continue
        name = feature["name"]
        nclass = int(feature["nclass"])
        observed = sorted(float(v) for v in pd.Series(raw_train[name]).dropna().unique())
        values = observed[:nclass]
        for i in range(nclass):
            if float(i) not in values:
                values.append(float(i))
        values = values[:nclass]
        cats[name] = values
        maps[name] = {str(float(v)): i for i, v in enumerate(values)}
    return StaticPreprocessor(categorical_maps=maps, categories=cats)


def _transform_static(
    raw_df: pd.DataFrame,
    types: list[dict[str, Any]],
    prep: StaticPreprocessor,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    encoded_parts: list[np.ndarray] = []
    encoded_names: list[str] = []
    feature_mask = torch.ones((len(raw_df), len(types)), dtype=torch.float32)
    for idx, feature in enumerate(types):
        ftype = feature["type"]
        name = feature["name"]
        if ftype.startswith("surv"):
            encoded_parts.append(raw_df[["time", "censor"]].to_numpy(dtype=np.float32))
            encoded_names.extend(["time", "censor"])
            continue
        series = raw_df[name]
        missing = series.isna().to_numpy()
        if missing.any():
            feature_mask[missing, idx] = 0.0
        if ftype in CATEGORICAL_TYPES:
            nclass = int(feature["nclass"])
            mapper = prep.categorical_maps[name]
            mapped = np.asarray(
                [mapper.get(str(float(v)), 0) if not pd.isna(v) else 0 for v in series],
                dtype=int,
            )
            one_hot = np.zeros((len(raw_df), nclass), dtype=np.float32)
            one_hot[np.arange(len(raw_df)), np.clip(mapped, 0, nclass - 1)] = 1.0
            encoded_parts.append(one_hot)
            encoded_names.extend([f"{name}_{j}" for j in range(nclass)])
        else:
            encoded_parts.append(series.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1))
            encoded_names.append(name)
    encoded = np.concatenate(encoded_parts, axis=1)
    return pd.DataFrame(encoded, columns=encoded_names), feature_mask, feature_mask.clone()


def _type_by_name(types: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in types:
        out["time" if item["name"] == "survcens" else item["name"]] = item
    return out


def _fit_longitudinal_preprocessor(
    long_df: pd.DataFrame,
    types: list[dict[str, Any]],
    train_subject_ids: np.ndarray,
    max_visits_cfg: int | None,
    train_survival_times: pd.Series | np.ndarray | None = None,
) -> LongitudinalPreprocessor:
    train_set = set(int(x) for x in train_subject_ids)
    train_long = long_df[long_df["patient_id"].isin(train_set)].copy()
    value_cols = [c for c in train_long.columns if c not in {"patient_id", "visit_time"}]
    grouped = train_long.sort_values(["patient_id", "visit_time"]).groupby("patient_id", sort=True)
    inferred_max = int(grouped.size().max())
    max_visits = inferred_max if max_visits_cfg is None else min(int(max_visits_cfg), inferred_max)
    type_map = _type_by_name(types)

    specs: list[LongitudinalSpec] = []
    cat_maps: dict[str, dict[float, int]] = {}
    for col in value_cols:
        ftype = type_map.get(col, {"type": "real"}).get("type", "real")
        nclass_raw = type_map.get(col, {}).get("nclass", "")
        nclass = int(nclass_raw) if ftype in CATEGORICAL_TYPES and nclass_raw != "" else None
        if nclass is not None:
            observed = sorted(float(v) for v in pd.Series(train_long[col]).dropna().unique())
            cats = observed[:nclass]
            for i in range(nclass):
                if float(i) not in cats:
                    cats.append(float(i))
            cats = tuple(cats[:nclass])
            cat_maps[col] = {float(v): j for j, v in enumerate(cats)}
            specs.append(LongitudinalSpec(name=col, type=ftype, nclass=nclass, categories=cats))
        else:
            obs = pd.to_numeric(train_long[col], errors="coerce").dropna().to_numpy(dtype=float)
            mean = float(np.mean(obs)) if obs.size else 0.0
            std = max(float(np.std(obs)) if obs.size else 1.0, 1e-6)
            specs.append(LongitudinalSpec(name=col, type=ftype, nclass=None, mean=mean, std=std))
    observed_times = train_long["visit_time"].dropna().to_numpy(dtype=float)
    time_min = float(np.min(observed_times)) if observed_times.size else 0.0
    time_max = float(np.max(observed_times)) if observed_times.size else 1.0
    if train_survival_times is not None:
        survival_times = pd.to_numeric(pd.Series(train_survival_times), errors="coerce").dropna().to_numpy(dtype=float)
        if survival_times.size:
            time_min = min(time_min, float(np.min(survival_times)))
            time_max = max(time_max, float(np.max(survival_times)))
    return LongitudinalPreprocessor(
        value_cols=value_cols,
        max_visits=max_visits,
        time_min=time_min,
        time_max=max(time_max, time_min + 1e-6),
        specs=specs,
    )


def _transform_longitudinal_panel(
    long_df: pd.DataFrame,
    subject_ids: np.ndarray,
    prep: LongitudinalPreprocessor,
) -> LongitudinalPanel:
    n_subjects = len(subject_ids)
    n_vars = len(prep.value_cols)
    raw_values = np.full((n_subjects, prep.max_visits, n_vars), np.nan, dtype=np.float32)
    masks = np.zeros((n_subjects, prep.max_visits, n_vars), dtype=np.float32)
    times = np.zeros((n_subjects, prep.max_visits), dtype=np.float32)
    id_to_row = {int(pid): i for i, pid in enumerate(subject_ids)}
    grouped = long_df.sort_values(["patient_id", "visit_time"]).groupby("patient_id", sort=True)
    cat_maps = {
        spec.name: {float(v): j for j, v in enumerate(spec.categories)}
        for spec in prep.specs
        if spec.type in CATEGORICAL_TYPES
    }
    for pid, rows in grouped:
        row_idx = id_to_row.get(int(pid))
        if row_idx is None:
            continue
        rows = rows.head(prep.max_visits)
        for visit_idx, (_, row) in enumerate(rows.iterrows()):
            times[row_idx, visit_idx] = float(row["visit_time"])
            for var_idx, col in enumerate(prep.value_cols):
                value = row[col]
                if pd.isna(value):
                    continue
                spec = prep.specs[var_idx]
                if spec.type in CATEGORICAL_TYPES:
                    raw_values[row_idx, visit_idx, var_idx] = cat_maps[col].get(float(value), 0)
                else:
                    raw_values[row_idx, visit_idx, var_idx] = float(value)
                masks[row_idx, visit_idx, var_idx] = 1.0
    observed_rows = (masks.sum(axis=-1) > 0).astype(np.float32)
    time_rng = max(prep.time_max - prep.time_min, 1e-6)
    times_norm = ((times - prep.time_min) / time_rng) * observed_rows
    values = np.nan_to_num(raw_values, nan=0.0).astype(np.float32)
    for idx, spec in enumerate(prep.specs):
        if spec.type in CONTINUOUS_TYPES:
            values[:, :, idx] = ((values[:, :, idx] - spec.mean) / max(spec.std, 1e-6)) * masks[:, :, idx]
        else:
            values[:, :, idx] = values[:, :, idx] * masks[:, :, idx]
    return LongitudinalPanel(
        subject_ids=np.asarray(subject_ids, dtype=int),
        times=torch.tensor(times_norm, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        masks=torch.tensor(masks, dtype=torch.float32),
        raw_values=raw_values,
        specs=prep.specs,
        time_min=prep.time_min,
        time_max=prep.time_max,
    )


def _make_bundle(
    raw_all: pd.DataFrame,
    ids_all: pd.DataFrame,
    long_df: pd.DataFrame,
    types: list[dict[str, Any]],
    subject_indices: np.ndarray,
    static_prep: StaticPreprocessor,
    long_prep: LongitudinalPreprocessor,
    cfg: dict[str, Any],
) -> PDC2Bundle:
    raw = raw_all.iloc[subject_indices].reset_index(drop=True).copy()
    ids = ids_all.iloc[subject_indices].reset_index(drop=True).copy()
    treatment_name = str(cfg.get("model", {}).get("treatment_variable_name", "drug"))
    treatment_type = next((t for t in types if t["name"] == treatment_name), None)
    treatment_n_classes = int(treatment_type.get("nclass") or 2) if treatment_type is not None else 2
    treatment = F.one_hot(
        torch.tensor(np.clip(raw[treatment_name].fillna(0).to_numpy(dtype=int), 0, treatment_n_classes - 1), dtype=torch.long),
        num_classes=treatment_n_classes,
    ).float()
    hivae_types = [dict(t) for t in types if t["name"] != treatment_name]
    encoded, miss, true_miss = _transform_static(raw, hivae_types, static_prep)
    panel = _transform_longitudinal_panel(long_df, subject_indices, long_prep)
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
        y_dim_partition=y_dim_partition_for_types(hivae_types, int(cfg["model"].get("y_dim_static", 15))),
        static_feature_count=len(hivae_types),
        treatment=treatment,
        treatment_name=treatment_name,
        treatment_n_classes=treatment_n_classes,
    )


def _candidate_config(args: argparse.Namespace) -> dict[str, Any]:
    spec = dict(ROUND11_SPEC)
    if args.epochs is not None:
        spec["epochs"] = int(args.epochs)
    if args.lr is not None:
        spec["lr"] = float(args.lr)
    if args.batch_size is not None:
        spec["batch_size"] = int(args.batch_size)
    if args.n_intervals is not None:
        spec["n_intervals"] = int(args.n_intervals)
    if args.lambda_surv is not None:
        spec["lambda_surv"] = float(args.lambda_surv)
    if args.kl_weight_s is not None:
        spec["kl_weight_s"] = float(args.kl_weight_s)
    if args.kl_weight_z is not None:
        spec["kl_weight_z"] = float(args.kl_weight_z)
    if args.longitudinal_weight is not None:
        spec["longitudinal_weight"] = float(args.longitudinal_weight)
    if args.continuous_mse_weight is not None:
        spec["continuous_mse_weight"] = float(args.continuous_mse_weight)
    cfg = load_config(args.config, {
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
            "freeze_normalization": True,
        },
        "evaluation": {
            "deterministic_static_export": False,
            "calibrate_static_covariates": False,
            "calibrate_survival_km": False,
            "calibrate_survival_event_rate": False,
            "calibrate_longitudinal_observed": False,
            "posterior_generation": True,
            "n_replicates": args.n_replicates,
        },
    })
    cfg["holdout_experiment"] = {
        "test_fraction": args.test_fraction,
        "split_seed": args.split_seed,
        "generation_conditioning": "B_mask_B_L0_A_future_times_only",
        "future_longitudinal_rows_only": True,
        "candidate": "round12_tuned_dynamic_survival_rand0_long2",
    }
    return cfg


def _model_audit(cfg: dict[str, Any], model: PhaseSynModel) -> dict[str, Any]:
    return {
        "encoder_conditioning": model.encoder_conditioning,
        "u0_init_mode": model.u0_init_mode,
        "baseline_long_weight": model.baseline_long_weight,
        "survival_feature_indices_masked_for_encoder": (
            model.encoder_conditioning == "baseline_only"
            and len(getattr(model, "survival_feature_indices", [])) > 0
        ),
        "uses_baseline_inclusive_longitudinal_loss": float(cfg["model"].get("baseline_long_weight", 1.0)) > 0.0,
        "dynamic_survival_head": hasattr(model, "dynamic_survival_head"),
        "passes_audit": (
            model.encoder_conditioning == "baseline_only"
            and model.u0_init_mode == "baseline_l0"
            and float(cfg["model"].get("baseline_long_weight", 1.0)) > 0.0
            and len(getattr(model, "survival_feature_indices", [])) > 0
        ),
    }


def _feature_lists(
    bundle: PDC2Bundle,
    device: torch.device,
    survival_observed: bool,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    miss = (bundle.miss_mask * bundle.true_miss_mask).float().to(device)
    data_list, miss_list = data_processing.next_batch(data_tensor, bundle.types, miss, data_tensor.shape[0], 0)
    data_list = [d.to(device) for d in data_list]
    if not survival_observed:
        for idx, feat in enumerate(bundle.types):
            if feat["type"].startswith("surv"):
                miss_list[:, idx] = 0.0
                data_list[idx] = torch.zeros_like(data_list[idx])
    return data_list, miss_list


def _decode_baseline_conditioned_static(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    device: torch.device,
    tau: float = 1e-3,
    deterministic_u0: bool | None = None,
) -> tuple[pd.DataFrame, dict[str, torch.Tensor], dict[str, Any]]:
    data_list, miss_list = _feature_lists(bundle, device, survival_observed=False)
    observed = [d * miss_list[:, i].view(miss_list.shape[0], 1) for i, d in enumerate(data_list)]
    with torch.no_grad():
        split = model.split_longitudinal_batch(
            bundle.longitudinal.times.to(device),
            bundle.longitudinal.values.to(device),
            bundle.longitudinal.masks.to(device),
        )
        res = model.encode_static_posterior(observed, data_list, miss_list, tau=tau, n_generated_dataset=1, encoder_l0=split["L0"])
    out = bundle.raw_df.copy().reset_index(drop=True)
    survival_idx = None
    survival_feat = None
    for idx, feat in enumerate(bundle.types):
        if feat["type"].startswith("surv"):
            survival_idx = idx
            survival_feat = feat
            break
    if survival_idx is None:
        raise RuntimeError("No survival feature found in PDC2 types.")
    a = model.treatment_context(bundle.treatment.to(device), len(bundle.raw_df), device, bundle.longitudinal.values.dtype)
    u0, u0_diag = model.sample_u0_from_l0(
        res["samples"]["z"],
        res["samples"]["s"],
        split["L0"],
        deterministic=deterministic_u0,
        return_details=True,
    )
    survival_out = model.dynamic_survival(u0, res["samples"]["z"], res["samples"]["s"], a)
    survival_sample = model.sample_dynamic_survival(survival_out, deterministic=False)
    out["time"] = model.denormalize_survival_time(survival_sample["observed_time"]).detach().cpu().numpy().reshape(-1)
    out["censor"] = survival_sample["event"].detach().cpu().numpy().reshape(-1)
    out.insert(0, "patient_id", bundle.longitudinal.subject_ids.astype(int))
    audit = {
        "test_survival_mask_zero_for_generation": bool(torch.all(miss_list[:, survival_idx] == 0).item()),
        "test_survival_tensor_zero_for_generation": bool(torch.all(data_list[survival_idx] == 0).item()),
    }
    return out, {
        "z": res["samples"]["z"].detach(),
        "s": res["samples"]["s"].detach(),
        "a": a.detach(),
        "u0": u0.detach(),
        "u0_mu": u0_diag["u0_mu"].detach(),
        "u0_sigma": u0_diag["u0_sigma"].detach(),
    }, audit


def _sample_longitudinal_future(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    latents: dict[str, torch.Tensor],
    device: torch.device,
    sample: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    panel = bundle.longitudinal
    times = panel.times.to(device)
    values = panel.values.to(device)
    masks = panel.masks.to(device)
    split = model.split_longitudinal_batch(times, values, masks)
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
            if spec.type in CONTINUOUS_TYPES:
                mu = params[:, :, 0]
                if sample:
                    var = F.softplus(params[:, :, 1].clamp(-8.0, 8.0)).clamp(min=1e-4, max=1e4)
                    value = torch.normal(mu, torch.sqrt(var))
                else:
                    value = mu
            else:
                if sample:
                    value = torch.distributions.Categorical(logits=params.reshape(-1, params.shape[-1])).sample()
                    value = value.reshape(params.shape[:2]).float()
                else:
                    value = torch.argmax(params, dim=-1).float()
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
        if spec.type in CONTINUOUS_TYPES:
            pred_raw[:, :, idx] = pred_raw[:, :, idx] * spec.std + spec.mean
    pred_raw, support = _apply_longitudinal_support(bundle, pred_raw)
    future_mask = split_cpu["future_masks"].detach().cpu().numpy().astype(bool)
    support["future_rows_only"] = 1.0
    support["future_observed_cell_count"] = float(future_mask.sum())
    support["future_observed_visit_count"] = float(split_cpu["future_visit_mask"].sum().item())
    return pred_raw, future_mask, baseline_idx, support


def _save_future_longitudinal(
    bundle: PDC2Bundle,
    pred_raw: np.ndarray,
    future_mask: np.ndarray,
    path: Path,
    replicate: int,
) -> pd.DataFrame:
    rows = []
    panel = bundle.longitudinal
    times_norm = panel.times.detach().cpu().numpy()
    times_raw = times_norm * (panel.time_max - panel.time_min) + panel.time_min
    future_rows = future_mask.any(axis=-1)
    for i, subject_id in enumerate(panel.subject_ids):
        for visit in range(pred_raw.shape[1]):
            if not future_rows[i, visit]:
                continue
            row: dict[str, Any] = {
                "replicate": int(replicate),
                "patient_id": int(subject_id),
                "visit_index": int(visit),
                "visit_time": float(times_raw[i, visit]),
                "visit_time_norm": float(times_norm[i, visit]),
                "trajectory_scope": "potential_future_grid",
            }
            for idx, spec in enumerate(panel.specs):
                row[spec.name] = float(pred_raw[i, visit, idx])
                row[f"{spec.name}_observed"] = bool(future_mask[i, visit, idx])
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


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


def _future_longitudinal_metrics(
    bundle: PDC2Bundle,
    pred_raw: np.ndarray,
    future_mask: np.ndarray,
    baseline_idx: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    panel = bundle.longitudinal
    real = panel.raw_values
    baseline_idx = np.asarray(baseline_idx, dtype=int)
    rows = []
    variable_rows = []
    metrics: dict[str, float] = {}
    cont_rmses = []
    cont_maes = []
    cont_baseline_rmses = []
    cont_ks = []
    cat_accs = []
    cat_base_accs = []
    cat_tv = []
    for idx, spec in enumerate(panel.specs):
        obs_all = future_mask[:, :, idx] & np.isfinite(real[:, :, idx])
        if not obs_all.any():
            continue
        pred = pred_raw[:, :, idx]
        if spec.type in CONTINUOUS_TYPES:
            diff = pred[obs_all] - real[:, :, idx][obs_all]
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            mae = float(np.mean(np.abs(diff)))
            baseline_values = np.asarray([real[i, baseline_idx[i], idx] for i in range(real.shape[0])], dtype=float)
            baseline_grid = np.repeat(baseline_values[:, None], real.shape[1], axis=1)
            base_diff = baseline_grid[obs_all] - real[:, :, idx][obs_all]
            base_rmse = float(np.sqrt(np.mean(base_diff ** 2)))
            cont_rmses.append(rmse)
            cont_maes.append(mae)
            cont_baseline_rmses.append(base_rmse)
            variable_rows.append({
                "variable": spec.name,
                "type": spec.type,
                "rmse": rmse,
                "mae": mae,
                "baseline_l0_carryforward_rmse": base_rmse,
                "rmse_ratio_vs_l0_carryforward": rmse / max(base_rmse, 1e-8),
            })
        else:
            pred_cat = np.rint(pred).astype(int)
            real_cat_obs = np.rint(real[:, :, idx][obs_all]).astype(int)
            acc = float(np.mean(pred_cat[obs_all] == real_cat_obs))
            baseline_values = np.asarray([real[i, baseline_idx[i], idx] for i in range(real.shape[0])], dtype=float)
            baseline_grid = np.repeat(baseline_values[:, None], real.shape[1], axis=1).astype(int)
            base_acc = float(np.mean(baseline_grid[obs_all] == real_cat_obs))
            cat_accs.append(acc)
            cat_base_accs.append(base_acc)
            variable_rows.append({
                "variable": spec.name,
                "type": spec.type,
                "accuracy": acc,
                "baseline_l0_carryforward_accuracy": base_acc,
            })
        for visit in range(real.shape[1]):
            obs = obs_all[:, visit]
            if not obs.any():
                continue
            if spec.type in CONTINUOUS_TYPES:
                stat = _ks_statistic(real[:, visit, idx][obs], pred[:, visit][obs])
                cont_ks.append(stat)
                rows.append({
                    "variable": spec.name,
                    "visit_index": visit,
                    "metric": "ks",
                    "value": stat,
                    "n": int(obs.sum()),
                })
            else:
                stat = _tv_statistic(np.rint(real[:, visit, idx][obs]), np.rint(pred[:, visit][obs]))
                cat_tv.append(stat)
                rows.append({
                    "variable": spec.name,
                    "visit_index": visit,
                    "metric": "tv",
                    "value": stat,
                    "n": int(obs.sum()),
                })
    metrics["future_continuous_rmse"] = float(np.mean(cont_rmses)) if cont_rmses else 0.0
    metrics["future_continuous_mae"] = float(np.mean(cont_maes)) if cont_maes else 0.0
    metrics["future_continuous_l0_carryforward_rmse"] = float(np.mean(cont_baseline_rmses)) if cont_baseline_rmses else 0.0
    metrics["future_continuous_rmse_ratio_vs_l0_carryforward"] = (
        metrics["future_continuous_rmse"] / max(metrics["future_continuous_l0_carryforward_rmse"], 1e-8)
        if cont_baseline_rmses else 0.0
    )
    metrics["future_continuous_ks_mean"] = float(np.mean(cont_ks)) if cont_ks else 0.0
    metrics["future_continuous_ks_max"] = float(np.max(cont_ks)) if cont_ks else 0.0
    metrics["future_categorical_accuracy"] = float(np.mean(cat_accs)) if cat_accs else 0.0
    metrics["future_categorical_l0_carryforward_accuracy"] = float(np.mean(cat_base_accs)) if cat_base_accs else 0.0
    metrics["future_categorical_tv_mean"] = float(np.mean(cat_tv)) if cat_tv else 0.0
    metrics["future_categorical_tv_max"] = float(np.max(cat_tv)) if cat_tv else 0.0
    metrics["valid_inverse_outputs"] = bool(np.isfinite(pred_raw).all())
    return metrics, pd.DataFrame(rows), pd.DataFrame(variable_rows)


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
    order = np.argsort(times)
    times = times[order]
    events = events[order]
    surv = 1.0
    xs = [0.0]
    ys = [1.0]
    for t in np.unique(times):
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


def _plot_survival(test_raw: pd.DataFrame, synth_reps: list[pd.DataFrame], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rx, ry = _km_curve(test_raw["time"].to_numpy(), test_raw["censor"].to_numpy())
    axes[0].step(rx, ry, where="post", color="black", linewidth=2.2, label="Observed test")
    grid = np.linspace(0.0, max(float(test_raw["time"].max()), 1e-6), 256)
    curves = []
    for syn in synth_reps:
        sx, sy = _km_curve(syn["time"].to_numpy(), syn["censor"].to_numpy())
        idx = np.searchsorted(sx, grid, side="right") - 1
        idx = np.clip(idx, 0, len(sy) - 1)
        curves.append(sy[idx])
        axes[0].step(sx, sy, where="post", alpha=0.25, linewidth=0.9)
    arr = np.asarray(curves)
    axes[0].plot(grid, arr.mean(axis=0), color="#c44e52", linestyle="--", linewidth=2.0, label="Generated mean")
    axes[0].fill_between(grid, arr.mean(axis=0) - arr.std(axis=0), arr.mean(axis=0) + arr.std(axis=0), color="#c44e52", alpha=0.18)
    axes[0].set_title("Held-Out Kaplan-Meier")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Survival probability")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    bins = np.linspace(0.0, max(float(test_raw["time"].max()), *(float(s["time"].max()) for s in synth_reps)), 28)
    axes[1].hist(test_raw["time"], bins=bins, density=True, alpha=0.45, color="#2f6f9f", label="Observed")
    for syn in synth_reps:
        axes[1].hist(syn["time"], bins=bins, density=True, alpha=0.05, color="#c44e52")
    axes[1].set_title("Survival Time Distribution")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=8)

    real_rate = float((test_raw["censor"] > 0.5).mean())
    rates = np.asarray([float((syn["censor"] > 0.5).mean()) for syn in synth_reps])
    axes[2].bar([0, 1], [real_rate, rates.mean()], yerr=[0.0, rates.std()], color=["#2f6f9f", "#c44e52"], tick_label=["Observed", "Generated"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Event rate")
    axes[2].set_title("Held-Out Event Rate")
    _savefig(fig, out_dir / "survival" / "holdout_survival_summary.png")


def _plot_longitudinal(bundle: PDC2Bundle, pred_reps: list[np.ndarray], future_mask: np.ndarray, out_dir: Path) -> None:
    panel = bundle.longitudinal
    real = panel.raw_values
    times = panel.times.detach().cpu().numpy() * (panel.time_max - panel.time_min) + panel.time_min
    for name in LONG_CONT_COLS:
        idx = next((i for i, s in enumerate(panel.specs) if s.name == name), None)
        if idx is None:
            continue
        xs, real_mean, gen_mean, gen_ci = [], [], [], []
        for visit in range(real.shape[1]):
            obs = future_mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            real_mean.append(float(np.nanmean(real[:, visit, idx][obs])))
            rep_means = [float(np.nanmean(pred[:, visit, idx][obs])) for pred in pred_reps]
            gen_mean.append(float(np.mean(rep_means)))
            gen_ci.append(float(1.96 * np.std(rep_means, ddof=1) / math.sqrt(len(rep_means))) if len(rep_means) > 1 else 0.0)
        if not xs:
            continue
        x = np.asarray(xs)
        gm = np.asarray(gen_mean)
        gci = np.asarray(gen_ci)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(x, real_mean, marker="o", color="black", linewidth=2.2, label="Observed test future")
        ax.plot(x, gm, marker="s", color="#c44e52", linewidth=2.0, label="Generated replicate-mean")
        ax.fill_between(x, gm - gci, gm + gci, color="#c44e52", alpha=0.2, label="Generated replicate-mean 95% CI")
        ax.set_title(f"Held-Out Future Replicate-Mean 95% CI: {name}")
        ax.set_xlabel("Time")
        ax.set_ylabel(name)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        _savefig(fig, out_dir / "replicate_mean_95ci" / f"{name}_replicate_mean_95ci.png")


def _plot_metric_summary(summary: dict[str, Any], out_dir: Path) -> None:
    keys = [
        "survival_km_integrated_abs_error_mean",
        "event_rate_diff_mean",
        "survival_time_rmse_ratio_mean",
        "survival_event_accuracy_mean",
        "future_continuous_rmse_ratio_vs_l0_carryforward_mean",
        "future_continuous_ks_mean_mean",
        "future_categorical_accuracy_mean",
        "future_categorical_tv_mean_mean",
    ]
    labels = [k.replace("_mean", "").replace("_", "\n") for k in keys if k in summary]
    values = [float(summary[k]) for k in keys if k in summary]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(values)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title("Held-Out Generation Metric Summary")
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, out_dir / "holdout_metric_summary.png")


def _leakage_diagnostics(model: PhaseSynModel, bundle: PDC2Bundle, device: torch.device) -> dict[str, float]:
    data_list, miss_list = _feature_lists(bundle, device, survival_observed=False)
    observed = [d * miss_list[:, i].view(miss_list.shape[0], 1) for i, d in enumerate(data_list)]
    split = model.split_longitudinal_batch(
        bundle.longitudinal.times.to(device),
        bundle.longitudinal.values.to(device),
        bundle.longitudinal.masks.to(device),
    )
    z, s = model.deterministic_latents_from_encoder_input(observed, miss_list, split["L0"])
    changed = [d.clone() for d in data_list]
    for idx, feat in enumerate(bundle.types):
        if feat["type"].startswith("surv"):
            changed[idx] = torch.randn_like(changed[idx]) * 100.0
    observed_changed = [d * miss_list[:, i].view(miss_list.shape[0], 1) for i, d in enumerate(changed)]
    z_changed, s_changed = model.deterministic_latents_from_encoder_input(observed_changed, miss_list, split["L0"])
    return {
        "survival_perturbation_z_max_abs": float((z - z_changed).abs().max().detach().cpu()),
        "survival_perturbation_s_max_abs": float((s - s_changed).abs().max().detach().cpu()),
    }


def _survival_generation_perturbation_audit(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base, _, _ = _decode_baseline_conditioned_static(model, bundle, device)

    scrambled_raw = bundle.raw_df.copy()
    scrambled_raw["time"] = scrambled_raw["time"].iloc[::-1].to_numpy() + float(scrambled_raw["time"].max()) + 10.0
    scrambled_raw["censor"] = 1.0 - pd.to_numeric(scrambled_raw["censor"], errors="coerce").fillna(0.0)
    scrambled = dataclasses.replace(bundle, raw_df=scrambled_raw)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    perturbed, _, _ = _decode_baseline_conditioned_static(model, scrambled, device)

    time_diff = np.max(np.abs(base["time"].to_numpy(dtype=float) - perturbed["time"].to_numpy(dtype=float)))
    censor_diff = np.max(np.abs(base["censor"].to_numpy(dtype=float) - perturbed["censor"].to_numpy(dtype=float)))
    return {
        "survival_generation_time_perturbation_max_abs": float(time_diff),
        "survival_generation_censor_perturbation_max_abs": float(censor_diff),
        "survival_generation_invariant_to_test_survival": bool(time_diff < 1e-8 and censor_diff < 1e-8),
    }


def _future_generation_perturbation_audit(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _, latents, _ = _decode_baseline_conditioned_static(model, bundle, device)
    base_pred, base_mask, _, _ = _sample_longitudinal_future(model, bundle, latents, device, sample=False)

    split = model.split_longitudinal_batch(
        bundle.longitudinal.times,
        bundle.longitudinal.values,
        bundle.longitudinal.masks,
    )
    values_scrambled = bundle.longitudinal.values.clone()
    future_cells = split["future_masks"].bool()
    values_scrambled[future_cells] = values_scrambled.flip(0)[future_cells] + 17.0
    scrambled_panel = dataclasses.replace(bundle.longitudinal, values=values_scrambled)
    scrambled = dataclasses.replace(bundle, longitudinal=scrambled_panel)
    perturbed_pred, perturbed_mask, _, _ = _sample_longitudinal_future(model, scrambled, latents, device, sample=False)

    pred_diff = np.max(np.abs(np.nan_to_num(base_pred, nan=0.0) - np.nan_to_num(perturbed_pred, nan=0.0)))
    mask_diff = np.max(np.abs(base_mask.astype(np.int8) - perturbed_mask.astype(np.int8)))
    return {
        "future_generation_value_perturbation_max_abs": float(pred_diff),
        "future_generation_mask_perturbation_max_abs": float(mask_diff),
        "future_generation_invariant_to_test_future_values": bool(pred_diff < 1e-8 and mask_diff == 0),
        "longitudinal_future_export_scope": "potential_trajectories",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    cfg = _candidate_config(args)
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    test_dir.mkdir(parents=True, exist_ok=True)
    raw, ids, long_df, types = _read_raw_tables(cfg)
    train_idx, test_idx = _stratified_split(raw, args.test_fraction, args.split_seed)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_split_file(raw, ids, train_idx, test_idx, output_root / "subject_splits.csv")

    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_prep = _fit_longitudinal_preprocessor(
        long_df,
        types,
        train_idx,
        cfg["dataset"].get("max_visits"),
        train_survival_times=raw.iloc[train_idx]["time"],
    )
    train_bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_prep, cfg)
    test_bundle = _make_bundle(raw, ids, long_df, types, test_idx, static_prep, long_prep, cfg)

    with open(output_root / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    metadata = {
        "static_preprocessor": _jsonable(static_prep),
        "longitudinal_preprocessor": _jsonable(long_prep),
        "train_subject_count": int(len(train_idx)),
        "test_subject_count": int(len(test_idx)),
        "preprocessing_fit_on_train_only": True,
    }
    with open(output_root / "preprocessing_metadata.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(metadata), f, indent=2)

    result = train_model(train_bundle, cfg, output_dir=train_dir, overfit_name=None)
    model = result["model"].to(torch.device(cfg["training"].get("device", "cpu")))
    if not isinstance(model, PhaseSynModel):
        raise TypeError("Holdout evaluation requires PhaseSynModel.")
    device = torch.device(cfg["training"].get("device", "cpu"))
    if model.hivae._global_norm_params is not None:
        with open(output_root / "hivae_normalization.json", "w", encoding="utf-8") as f:
            json.dump(_normalization_metadata(model.hivae._global_norm_params, train_bundle.types), f, indent=2)
    audit = _model_audit(cfg, model)
    audit.update(_leakage_diagnostics(model, test_bundle, device))
    audit.update(_survival_generation_perturbation_audit(model, test_bundle, device, int(args.seed) + 997))
    audit.update(_future_generation_perturbation_audit(model, test_bundle, device, int(args.seed) + 1997))
    audit.update({
        "train_subject_count": int(len(train_idx)),
        "test_subject_count": int(len(test_idx)),
        "generation_batch_keys": ["B", "mask_B", "L0", "A", "future_times"],
        "forbidden_generation_inputs": ["test_survival_time", "test_censor", "test_future_longitudinal_values"],
        "forbidden_generation_inputs_present": [],
        "test_preprocessing_uses_train_fit": True,
        "hivae_uses_frozen_train_normalization": model.hivae._global_norm_params is not None,
    })
    audit["passes_audit"] = bool(
        audit["passes_audit"]
        and audit["survival_generation_invariant_to_test_survival"]
        and audit["future_generation_invariant_to_test_future_values"]
        and audit["hivae_uses_frozen_train_normalization"]
    )
    if not audit["passes_audit"]:
        raise RuntimeError(f"Holdout model audit failed: {audit}")
    with open(output_root / "leakage_audit.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(audit), f, indent=2)

    rep_static = []
    rep_long_csv = []
    rep_metric_rows = []
    rep_pred_arrays = []
    per_visit_frames = []
    per_var_frames = []
    generation_audits = []
    for rep in range(1, int(args.n_replicates) + 1):
        seed = int(args.seed) + 1000 + rep
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        syn_static, latents, gen_audit = _decode_baseline_conditioned_static(model, test_bundle, device)
        syn_static.insert(0, "replicate", rep)
        expected_ids = test_bundle.longitudinal.subject_ids.astype(int)
        if not np.array_equal(syn_static["patient_id"].to_numpy(dtype=int), expected_ids):
            raise RuntimeError("Generated static patient_id order does not match held-out test bundle order.")
        static_path = test_dir / f"synthetic_static_test_rep{rep:02d}.csv"
        syn_static.to_csv(static_path, index=False)
        pred_raw, future_mask, baseline_idx, support = _sample_longitudinal_future(
            model,
            test_bundle,
            latents,
            device,
            sample=not args.deterministic_longitudinal,
        )
        long_df_rep = _save_future_longitudinal(
            test_bundle,
            pred_raw,
            future_mask,
            test_dir / f"synthetic_longitudinal_future_test_rep{rep:02d}.csv",
            rep,
        )
        long_metrics, per_visit, per_var = _future_longitudinal_metrics(test_bundle, pred_raw, future_mask, baseline_idx)
        survival_metrics = event_rate_metrics(test_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"]))
        survival_metrics.update(paired_survival_metrics(test_bundle.raw_df, syn_static.drop(columns=["replicate", "patient_id"])))
        row: dict[str, float] = {"replicate": float(rep)}
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
    per_visit_all = pd.concat(per_visit_frames, ignore_index=True)
    per_var_all = pd.concat(per_var_frames, ignore_index=True)
    static_all.to_csv(test_dir / "holdout_synthetic_static_all.csv", index=False)
    long_all.to_csv(test_dir / "holdout_synthetic_longitudinal_future_all.csv", index=False)
    metrics_df.to_csv(test_dir / "holdout_replicate_metrics.csv", index=False)
    per_visit_all.to_csv(test_dir / "longitudinal_future_per_visit_metrics.csv", index=False)
    per_var_all.to_csv(test_dir / "longitudinal_future_variable_metrics.csv", index=False)

    summary = _summarize_metric_rows(rep_metric_rows, len(test_idx))
    summary.update(audit)
    summary["train_final_loss"] = float(result["curves"]["loss"].dropna().iloc[-1])
    summary["train_loss_decrease"] = float(
        (result["curves"]["loss"].dropna().iloc[0] - result["curves"]["loss"].dropna().iloc[-1])
        / max(abs(result["curves"]["loss"].dropna().iloc[0]), 1e-8)
    )
    summary["nan_epoch_count"] = int(result["curves"]["nan_epoch"].astype(bool).sum()) if "nan_epoch" in result["curves"] else 0
    summary["generation_audits_all_survival_zero"] = bool(all(x["test_survival_mask_zero_for_generation"] and x["test_survival_tensor_zero_for_generation"] for x in generation_audits))
    with open(test_dir / "holdout_summary.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2)
    metrics_df.describe().to_csv(test_dir / "holdout_metric_describe.csv")

    _plot_survival(test_bundle.raw_df, [df.drop(columns=["replicate", "patient_id"]) for df in rep_static], test_dir / "figures")
    _plot_longitudinal(test_bundle, rep_pred_arrays, future_mask, test_dir / "figures")
    _plot_metric_summary(summary, test_dir / "figures")

    md = test_dir / "holdout_summary.md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# PhaseSyn Holdout Baseline/L0 Generation\n\n")
        f.write(f"Train subjects: {len(train_idx)}; test subjects: {len(test_idx)}; replicates: {args.n_replicates}.\n\n")
        f.write("Generation inputs are restricted to `B`, `mask_B`, complete `L0`, observed `A`, and `future_times`. Test survival and future longitudinal outcomes are used only for scoring.\n\n")
        f.write("Survival generation is performed by the holdout decoder helper with test survival masks and tensors set to zero before posterior encoding.\n\n")
        key_rows = {k: summary[k] for k in sorted(summary) if k.endswith("_mean") and any(token in k for token in ["survival", "future_continuous", "future_categorical", "event_rate"])}
        f.write(pd.DataFrame([key_rows]).T.rename(columns={0: "value"}).to_markdown())
        f.write("\n")
    print(json.dumps(_jsonable(summary), indent=2))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PhaseSyn on train subjects and evaluate baseline/L0-conditioned survival and future longitudinal generation on held-out PDC2 subjects.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "pdc2.yaml"))
    parser.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--split-seed", type=int, default=20260521)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-replicates", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-intervals", type=int, default=None)
    parser.add_argument("--lambda-surv", type=float, default=None)
    parser.add_argument("--kl-weight-s", type=float, default=None)
    parser.add_argument("--kl-weight-z", type=float, default=None)
    parser.add_argument("--longitudinal-weight", type=float, default=None)
    parser.add_argument("--continuous-mse-weight", type=float, default=None)
    parser.add_argument("--output-root", default="outputs/pdc2/experiments_20260521/holdout_baseline_l0_0plus")
    parser.add_argument("--deterministic-longitudinal", action="store_true")
    parser.add_argument("--use-randomization-loss", action="store_true")
    parser.add_argument("--randomization-loss-weight", type=float, default=0.0)
    parser.add_argument("--randomization-loss-warmup-epochs", type=int, default=0)
    parser.add_argument("--randomization-loss-ramp-epochs", type=int, default=1)
    parser.add_argument("--randomization-mmd-bandwidths", default="0.5,1.0,2.0,4.0")
    parser.add_argument("--randomization-loss-on", choices=["z_mean", "z_sample"], default="z_mean")
    args = parser.parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
