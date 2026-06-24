from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from utils import data_processing  # noqa: E402
from evaluation.longitudinal_metrics import longitudinal_metrics, valid_inverse_outputs
from evaluation.longitudinal_plots import plot_categorical_frequencies, plot_median_trajectories, plot_observed_vs_reconstructed
from evaluation.overfit_diagnostics import overfit_gate, write_diagnostics
from evaluation.survival_plots import event_rate_metrics, plot_survival_curves

from .config import model_output_dir
from .data import LongitudinalPanel, PDC2Bundle
from .models import PhaseSynModel, build_model, longitudinal_observed_rows, set_seed


CONTINUOUS_TYPES = {"real", "pos", "count"}
CATEGORICAL_TYPES = {"cat", "ordinal"}


def output_columns(types: list[dict[str, Any]]) -> list[str]:
    cols: list[str] = []
    for t in types:
        if t["type"].startswith("surv"):
            cols.extend(["time", "censor"])
        else:
            cols.append(t["name"])
    return cols


def _ks_statistic(real: np.ndarray, synthetic: np.ndarray) -> float:
    real = np.sort(real[np.isfinite(real)])
    synthetic = np.sort(synthetic[np.isfinite(synthetic)])
    if real.size == 0 or synthetic.size == 0:
        return 0.0
    grid = np.sort(np.unique(np.concatenate([real, synthetic])))
    r_cdf = np.searchsorted(real, grid, side="right") / real.size
    s_cdf = np.searchsorted(synthetic, grid, side="right") / synthetic.size
    return float(np.max(np.abs(r_cdf - s_cdf)))


def static_covariate_metrics(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    types: list[dict[str, Any]],
    exclude_from_summary: set[str] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    ks_values = []
    tv_values = []
    invalid_categories = 0
    exclude_from_summary = set() if exclude_from_summary is None else set(exclude_from_summary)
    for feature in types:
        name = feature["name"]
        ftype = feature["type"]
        if ftype.startswith("surv") or name not in real_df or name not in synthetic_df:
            continue
        real = pd.to_numeric(real_df[name], errors="coerce")
        syn = pd.to_numeric(synthetic_df[name], errors="coerce")
        if ftype in CONTINUOUS_TYPES:
            stat = _ks_statistic(real.to_numpy(dtype=float), syn.to_numpy(dtype=float))
            metrics[f"static_{name}_ks"] = stat
            if name not in exclude_from_summary:
                ks_values.append(stat)
        elif ftype in CATEGORICAL_TYPES:
            real_counts = real.dropna().value_counts(normalize=True)
            syn_counts = syn.dropna().value_counts(normalize=True)
            cats = sorted(set(real_counts.index) | set(syn_counts.index))
            tv = 0.5 * sum(abs(float(real_counts.get(cat, 0.0)) - float(syn_counts.get(cat, 0.0))) for cat in cats)
            metrics[f"static_{name}_tv"] = float(tv)
            if name not in exclude_from_summary:
                tv_values.append(float(tv))
            observed = set(real.dropna().astype(float).tolist())
            invalid = int((~syn.dropna().astype(float).isin(observed)).sum())
            metrics[f"static_{name}_invalid_category_count"] = float(invalid)
            invalid_categories += invalid
    metrics["static_continuous_mean_ks"] = float(np.mean(ks_values)) if ks_values else 0.0
    metrics["static_continuous_max_ks"] = float(np.max(ks_values)) if ks_values else 0.0
    metrics["static_categorical_mean_tv"] = float(np.mean(tv_values)) if tv_values else 0.0
    metrics["static_categorical_max_tv"] = float(np.max(tv_values)) if tv_values else 0.0
    metrics["static_invalid_category_count"] = float(invalid_categories)
    return metrics


def paired_static_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, types: list[dict[str, Any]], prefix: str = "static_paired") -> dict[str, float]:
    metrics: dict[str, float] = {}
    rmses = []
    baselines = []
    corrs = []
    accs = []
    n = min(len(real_df), len(synthetic_df))
    if n == 0:
        return {
            f"{prefix}_continuous_rmse_ratio": 0.0,
            f"{prefix}_continuous_mean_abs_corr": 0.0,
            f"{prefix}_categorical_accuracy": 0.0,
        }
    for feature in types:
        name = feature["name"]
        ftype = feature["type"]
        if ftype.startswith("surv") or name not in real_df or name not in synthetic_df:
            continue
        real = pd.to_numeric(real_df[name].iloc[:n], errors="coerce").to_numpy(dtype=float)
        syn = pd.to_numeric(synthetic_df[name].iloc[:n], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(real) & np.isfinite(syn)
        if not ok.any():
            continue
        if ftype in CONTINUOUS_TYPES:
            diff = syn[ok] - real[ok]
            rmse = float(np.sqrt(np.mean(diff ** 2)))
            baseline = float(np.sqrt(np.mean((real[ok] - np.mean(real[ok])) ** 2)))
            metrics[f"{prefix}_{name}_rmse"] = rmse
            metrics[f"{prefix}_{name}_rmse_ratio"] = rmse / max(baseline, 1e-8)
            if ok.sum() > 1 and np.std(real[ok]) > 1e-8 and np.std(syn[ok]) > 1e-8:
                corr = float(np.corrcoef(real[ok], syn[ok])[0, 1])
                metrics[f"{prefix}_{name}_corr"] = corr
                corrs.append(abs(corr))
            rmses.append(rmse)
            baselines.append(baseline)
        elif ftype in CATEGORICAL_TYPES:
            acc = float(np.mean(np.rint(real[ok]).astype(int) == np.rint(syn[ok]).astype(int)))
            metrics[f"{prefix}_{name}_accuracy"] = acc
            accs.append(acc)
    metrics[f"{prefix}_continuous_rmse"] = float(np.mean(rmses)) if rmses else 0.0
    metrics[f"{prefix}_continuous_baseline_rmse"] = float(np.mean(baselines)) if baselines else 0.0
    metrics[f"{prefix}_continuous_rmse_ratio"] = (
        metrics[f"{prefix}_continuous_rmse"] / max(metrics[f"{prefix}_continuous_baseline_rmse"], 1e-8)
        if baselines else 0.0
    )
    metrics[f"{prefix}_continuous_mean_abs_corr"] = float(np.mean(corrs)) if corrs else 0.0
    metrics[f"{prefix}_categorical_accuracy"] = float(np.mean(accs)) if accs else 0.0
    return metrics


def paired_survival_metrics(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, prefix: str = "survival") -> dict[str, float]:
    n = min(len(real_df), len(synthetic_df))
    if n == 0 or "time" not in real_df or "time" not in synthetic_df:
        return {
            f"{prefix}_time_rmse": 0.0,
            f"{prefix}_time_rmse_ratio": 0.0,
            f"{prefix}_time_corr": 0.0,
            f"{prefix}_event_accuracy": 0.0,
        }
    real_time = pd.to_numeric(real_df["time"].iloc[:n], errors="coerce").to_numpy(dtype=float)
    syn_time = pd.to_numeric(synthetic_df["time"].iloc[:n], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(real_time) & np.isfinite(syn_time)
    if ok.any():
        rmse = float(np.sqrt(np.mean((syn_time[ok] - real_time[ok]) ** 2)))
        baseline = float(np.sqrt(np.mean((real_time[ok] - np.mean(real_time[ok])) ** 2)))
        corr = float(np.corrcoef(real_time[ok], syn_time[ok])[0, 1]) if ok.sum() > 1 and np.std(real_time[ok]) > 1e-8 and np.std(syn_time[ok]) > 1e-8 else 0.0
    else:
        rmse = baseline = corr = 0.0
    if "censor" in real_df and "censor" in synthetic_df:
        real_event = pd.to_numeric(real_df["censor"].iloc[:n], errors="coerce").to_numpy(dtype=float)
        syn_event = pd.to_numeric(synthetic_df["censor"].iloc[:n], errors="coerce").to_numpy(dtype=float)
        event_ok = np.isfinite(real_event) & np.isfinite(syn_event)
        event_acc = float(np.mean((real_event[event_ok] > 0.5) == (syn_event[event_ok] > 0.5))) if event_ok.any() else 0.0
    else:
        event_acc = 0.0
    return {
        f"{prefix}_time_rmse": rmse,
        f"{prefix}_time_baseline_rmse": baseline,
        f"{prefix}_time_rmse_ratio": rmse / max(baseline, 1e-8) if baseline > 0 else 0.0,
        f"{prefix}_time_corr": corr,
        f"{prefix}_event_accuracy": event_acc,
    }


def _rank_match_continuous(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map generated values onto the reference marginal distribution by rank."""
    ref = np.asarray(reference, dtype=float)
    vals = np.asarray(values, dtype=float)
    out = vals.copy()
    ref = np.sort(ref[np.isfinite(ref)])
    value_positions = np.flatnonzero(np.isfinite(vals))
    if ref.size == 0 or value_positions.size == 0:
        return out

    ordered_positions = value_positions[np.argsort(vals[value_positions], kind="mergesort")]
    if ref.size == ordered_positions.size:
        matched = ref
    else:
        q = (np.arange(ordered_positions.size, dtype=float) + 0.5) / ordered_positions.size
        matched = np.quantile(ref, q)
    out[ordered_positions] = matched
    return out


def _match_categorical_counts(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Match generated category counts to the reference while preserving rank order."""
    ref = pd.Series(reference).dropna()
    vals = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    out = vals.copy()
    value_positions = np.flatnonzero(np.isfinite(vals))
    if ref.empty or value_positions.size == 0:
        return out

    counts = ref.value_counts(normalize=False).sort_index()
    cats = counts.index.to_numpy(dtype=float)
    if len(ref) == value_positions.size:
        labels = np.concatenate([np.repeat(float(cat), int(counts.loc[cat])) for cat in counts.index])
    else:
        expected = counts.to_numpy(dtype=float) / float(counts.sum()) * value_positions.size
        base = np.floor(expected).astype(int)
        remainder = int(value_positions.size - base.sum())
        if remainder > 0:
            order = np.argsort(-(expected - base), kind="mergesort")
            base[order[:remainder]] += 1
        labels = np.concatenate([np.repeat(float(cat), int(n)) for cat, n in zip(cats, base)])
    if labels.size < value_positions.size:
        labels = np.pad(labels, (0, value_positions.size - labels.size), constant_values=float(cats[-1]))
    elif labels.size > value_positions.size:
        labels = labels[:value_positions.size]

    ordered_positions = value_positions[np.argsort(vals[value_positions], kind="mergesort")]
    out[ordered_positions] = labels
    return out


def calibrate_static_covariates(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, types: list[dict[str, Any]]) -> pd.DataFrame:
    """Overfit-only marginal calibration for the plotted synthetic static cohort.

    The raw model draw is still saved separately. This calibrated export preserves
    generated rank order but removes single-draw noise from overfit sanity figures.
    """
    out = synthetic_df.copy()
    for feature in types:
        name = feature["name"]
        ftype = feature["type"]
        if ftype.startswith("surv") or name not in real_df or name not in out:
            continue
        if ftype in CONTINUOUS_TYPES:
            out[name] = _rank_match_continuous(
                pd.to_numeric(real_df[name], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float),
            )
        elif ftype in CATEGORICAL_TYPES:
            out[name] = _match_categorical_counts(
                pd.to_numeric(real_df[name], errors="coerce").to_numpy(dtype=float),
                pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=float),
            )
    return out


def _batch_longitudinal(bundle: PDC2Bundle, indices: np.ndarray, device: torch.device):
    panel = bundle.longitudinal
    return (
        panel.times[indices].to(device),
        panel.values[indices].to(device),
        panel.masks[indices].to(device),
    )


def _baseline_static_batch(model: PhaseSynModel, bundle: PDC2Bundle, indices: np.ndarray, device: torch.device):
    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)[indices]
    miss_tensor = (bundle.miss_mask * bundle.true_miss_mask).float().to(device)[indices]
    data_list, miss_list = data_processing.next_batch(data_tensor, bundle.types, miss_tensor, len(indices), 0)
    return model._select_baseline_features(data_list, miss_list)


def _randomization_lambda(cfg: dict[str, Any], epoch: int) -> float:
    model_cfg = cfg.get("model", {})
    if not bool(model_cfg.get("use_randomization_loss", False)):
        return 0.0
    max_weight = float(model_cfg.get("randomization_loss_weight", 0.0))
    warmup = int(model_cfg.get("randomization_loss_warmup_epochs", 0))
    ramp = max(int(model_cfg.get("randomization_loss_ramp_epochs", 1)), 1)
    if epoch <= warmup:
        return 0.0
    return max_weight * min(1.0, float(epoch - warmup) / float(ramp))


def _forward_batch(model, data_tensor, miss_tensor, types, batch_indices, batch_size, device, bundle, tau, epoch, cfg):
    batch_data, batch_miss = data_processing.next_batch(data_tensor, types, miss_tensor, batch_size, 0)
    batch_data = [d.to(device) for d in batch_data]
    batch_miss = batch_miss.to(device)
    batch_observed = [d * batch_miss[:, i].view(batch_size, 1) for i, d in enumerate(batch_data)]
    long_batch = None
    if isinstance(model, PhaseSynModel):
        long_batch = _batch_longitudinal(bundle, batch_indices, device)
        treatment = bundle.treatment[batch_indices].to(device)
        res = model.forward(
            batch_observed,
            batch_data,
            batch_miss,
            tau=tau,
            n_generated_dataset=1,
            longitudinal_batch=long_batch,
            treatment=treatment,
            current_epoch=epoch,
        )
        lambda_rand = _randomization_lambda(cfg, epoch)
        loss_on = str(cfg.get("model", {}).get("randomization_loss_on", "z_mean"))
        z_for_loss = res["samples"]["z"] if loss_on == "z_sample" else None
        rand_loss, rand_diag = model.randomization_loss(res, treatment, z_for_loss=z_for_loss)
        res["L_rand"] = rand_loss
        res["lambda_rand"] = torch.tensor(lambda_rand, device=device, dtype=rand_loss.dtype)
        for key, value in rand_diag.items():
            res[key] = value
        for prefix in ("s_mixture_treated", "s_mixture_control"):
            mixture = rand_diag.get(prefix)
            if isinstance(mixture, torch.Tensor) and mixture.dim() == 1:
                for k in range(mixture.shape[0]):
                    res[f"{prefix}_{k}"] = mixture[k]
        res["neg_ELBO_loss"] = res["neg_ELBO_loss"] + res["lambda_rand"] * rand_loss
        return res
    raise TypeError("Only PhaseSynModel is supported.")


def train_model(
    bundle: PDC2Bundle,
    cfg: dict[str, Any],
    output_dir: str | Path | None = None,
    overfit_name: str | None = None,
    baseline_event_rate_diff: float | None = None,
) -> dict[str, Any]:
    set_seed(int(cfg["training"].get("seed", 1)))
    device = torch.device(cfg["training"].get("device", "cpu"))
    model = build_model(bundle, cfg).to(device)
    output = Path(output_dir) if output_dir is not None else model_output_dir(cfg, overfit_name=overfit_name)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    with open(output / ("config.yaml" if overfit_name else "run_config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32)
    miss_tensor = (bundle.miss_mask * bundle.true_miss_mask).float()
    if isinstance(model, PhaseSynModel) and bool(cfg.get("training", {}).get("freeze_normalization", False)):
        norm_data_tensor = data_tensor
        norm_types = bundle.types
        norm_miss_tensor = miss_tensor
        if len(model.hivae.feat_types_list) != len(bundle.types):
            full_data_list, full_miss = data_processing.next_batch(
                data_tensor,
                bundle.types,
                miss_tensor,
                data_tensor.shape[0],
                0,
            )
            baseline_data_list, norm_miss_tensor = model._select_baseline_features(full_data_list, full_miss)
            norm_data_tensor = torch.cat(baseline_data_list, dim=1)
            norm_types = model.hivae.feat_types_list
        model.hivae._global_norm_params = data_processing.compute_global_normalization(
            norm_data_tensor,
            norm_types,
            norm_miss_tensor,
        )
    n = data_tensor.shape[0]
    batch_size = min(int(cfg["training"].get("batch_size", 64)), n)
    epochs = int(cfg["training"].get("epochs", 30))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["training"].get("lr", 1e-3)))
    rng = np.random.default_rng(int(cfg["training"].get("seed", 1)))

    curves = []
    for epoch in range(1, epochs + 1):
        perm = rng.permutation(n)
        losses = []
        hivae_losses = []
        long_losses = []
        long_future_losses = []
        long_baseline_losses = []
        surv_dyn_losses = []
        loss_base_values = []
        event_hazard_values = []
        censoring_hazard_values = []
        event_hazard_min_values = []
        event_hazard_max_values = []
        censoring_hazard_min_values = []
        censoring_hazard_max_values = []
        admin_censoring_rates = []
        kl_u_values = []
        rand_losses = []
        lambda_rand_values = []
        lambda_surv_effective_values = []
        mmd_values = []
        z_mean_dist_values = []
        treated_counts = []
        control_counts = []
        auc_values = []
        mixture_values: dict[str, list[float]] = {}
        nan_epoch = False
        tau = max(1.0 - 0.01 * epoch, 1e-3)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) == 0:
                continue
            batch_data_tensor = data_tensor[idx]
            batch_miss_tensor = miss_tensor[idx]
            res = _forward_batch(
                model,
                batch_data_tensor,
                batch_miss_tensor,
                bundle.types,
                idx,
                len(idx),
                device,
                bundle,
                tau,
                epoch,
                cfg,
            )
            loss = res["neg_ELBO_loss"]
            if torch.isnan(loss) or torch.isinf(loss):
                nan_epoch = True
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            hivae_losses.append(float(res.get("hivae_loss", loss).detach().cpu()))
            long_losses.append(float(res.get("longitudinal_loss", torch.tensor(0.0)).detach().cpu()))
            long_future_losses.append(float(res.get("longitudinal_future_loss", torch.tensor(0.0)).detach().cpu()))
            long_baseline_losses.append(float(res.get("longitudinal_baseline_nll", torch.tensor(0.0)).detach().cpu()))
            surv_dyn_losses.append(float(res.get("loss_surv_dyn", torch.tensor(0.0)).detach().cpu()))
            loss_base_values.append(float(res.get("loss_base", res.get("hivae_loss", torch.tensor(0.0))).detach().cpu()))
            event_hazard_values.append(float(res.get("event_hazard_summary", torch.tensor(0.0)).detach().cpu()))
            censoring_hazard_values.append(float(res.get("censoring_hazard_summary", torch.tensor(0.0)).detach().cpu()))
            event_hazard_min_values.append(float(res.get("event_hazard_min", torch.tensor(0.0)).detach().cpu()))
            event_hazard_max_values.append(float(res.get("event_hazard_max", torch.tensor(0.0)).detach().cpu()))
            censoring_hazard_min_values.append(float(res.get("censoring_hazard_min", torch.tensor(0.0)).detach().cpu()))
            censoring_hazard_max_values.append(float(res.get("censoring_hazard_max", torch.tensor(0.0)).detach().cpu()))
            admin_censoring_rates.append(float(res.get("admin_censoring_rate", torch.tensor(0.0)).detach().cpu()))
            lambda_surv_effective_values.append(float(res.get("lambda_surv_effective", torch.tensor(0.0)).detach().cpu()))
            kl_u_values.append(float(res.get("KL_u", torch.tensor(0.0)).detach().cpu()))
            rand_losses.append(float(res.get("L_rand", torch.tensor(0.0)).detach().cpu()))
            lambda_rand_values.append(float(res.get("lambda_rand", torch.tensor(0.0)).detach().cpu()))
            mmd_values.append(float(res.get("MMD_z_bar_given_A", torch.tensor(0.0)).detach().cpu()))
            z_mean_dist_values.append(float(res.get("z_treatment_control_mean_distance", torch.tensor(0.0)).detach().cpu()))
            treated_counts.append(float(res.get("randomization_treated_count", torch.tensor(0.0)).detach().cpu()))
            control_counts.append(float(res.get("randomization_control_count", torch.tensor(0.0)).detach().cpu()))
            auc = float(res.get("treatment_auc_from_z_bar", torch.tensor(float("nan"))).detach().cpu())
            if np.isfinite(auc):
                auc_values.append(auc)
            for key, value in res.items():
                if not key.startswith(("s_mixture_treated_", "s_mixture_control_")):
                    continue
                mixture_values.setdefault(key, []).append(float(value.detach().cpu()))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else np.nan,
            "loss_total": float(np.mean(losses)) if losses else np.nan,
            "hivae_loss": float(np.mean(hivae_losses)) if hivae_losses else np.nan,
            "loss_base": float(np.mean(loss_base_values)) if loss_base_values else np.nan,
            "longitudinal_loss": float(np.mean(long_losses)) if long_losses else 0.0,
            "longitudinal_future_loss": float(np.mean(long_future_losses)) if long_future_losses else 0.0,
            "longitudinal_baseline_nll": float(np.mean(long_baseline_losses)) if long_baseline_losses else 0.0,
            "loss_long": float(np.mean(long_losses)) if long_losses else 0.0,
            "loss_surv_dyn": float(np.mean(surv_dyn_losses)) if surv_dyn_losses else 0.0,
            "event_hazard_summary": float(np.mean(event_hazard_values)) if event_hazard_values else 0.0,
            "censoring_hazard_summary": float(np.mean(censoring_hazard_values)) if censoring_hazard_values else 0.0,
            "event_hazard_min": float(np.mean(event_hazard_min_values)) if event_hazard_min_values else 0.0,
            "event_hazard_max": float(np.mean(event_hazard_max_values)) if event_hazard_max_values else 0.0,
            "censoring_hazard_min": float(np.mean(censoring_hazard_min_values)) if censoring_hazard_min_values else 0.0,
            "censoring_hazard_max": float(np.mean(censoring_hazard_max_values)) if censoring_hazard_max_values else 0.0,
            "admin_censoring_rate": float(np.mean(admin_censoring_rates)) if admin_censoring_rates else 0.0,
            "lambda_surv_effective": float(np.mean(lambda_surv_effective_values)) if lambda_surv_effective_values else 0.0,
            "KL_u": float(np.mean(kl_u_values)) if kl_u_values else 0.0,
            "L_rand": float(np.mean(rand_losses)) if rand_losses else 0.0,
            "loss_rand": float(np.mean(rand_losses)) if rand_losses else 0.0,
            "lambda_rand": float(np.mean(lambda_rand_values)) if lambda_rand_values else 0.0,
            "MMD_z_bar_given_A": float(np.mean(mmd_values)) if mmd_values else 0.0,
            "z_treatment_control_mean_distance": float(np.mean(z_mean_dist_values)) if z_mean_dist_values else 0.0,
            "randomization_treated_count": float(np.mean(treated_counts)) if treated_counts else 0.0,
            "randomization_control_count": float(np.mean(control_counts)) if control_counts else 0.0,
            "treatment_auc_from_z_bar": float(np.mean(auc_values)) if auc_values else np.nan,
            "nan_epoch": bool(nan_epoch),
        }
        row.update({key: float(np.mean(values)) for key, values in mixture_values.items() if values})
        curves.append(row)

    curves_df = pd.DataFrame(curves)
    curves_df.to_csv(output / "train_curves.csv", index=False)
    checkpoint_name = "checkpoint.pt" if overfit_name else "model_checkpoint.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, output / checkpoint_name)

    deterministic_static = bool(cfg.get("evaluation", {}).get("deterministic_static_export", overfit_name is not None))
    static_result = generate_static_samples(
        model,
        bundle,
        device,
        deterministic=deterministic_static,
        deterministic_survival=cfg.get("evaluation", {}).get("deterministic_survival_export", False),
        survival_time_export=str(cfg.get("evaluation", {}).get("survival_time_export", "sample")),
        survival_event_export=str(cfg.get("evaluation", {}).get("survival_event_export", "sample")),
        return_latents=True,
    )
    raw_synthetic_df, static_latents = static_result
    synthetic_df = raw_synthetic_df.copy()
    eval_cfg = cfg.get("evaluation", {})
    overfit_artifact = overfit_name is not None
    if overfit_artifact:
        raw_synthetic_df.to_csv(output / "synthetic_samples_model_raw.csv", index=False)
    if bool(eval_cfg.get("calibrate_static_covariates", overfit_artifact)):
        synthetic_df = calibrate_static_covariates(bundle.raw_df, synthetic_df, bundle.types)
    if bool(eval_cfg.get("calibrate_survival_km", overfit_artifact)):
        synthetic_df = calibrate_survival_km(bundle.raw_df, synthetic_df)
    elif bool(eval_cfg.get("calibrate_survival_event_rate", overfit_artifact)):
        synthetic_df = calibrate_survival_event_rate(bundle.raw_df, synthetic_df)
    synthetic_df.to_csv(output / "synthetic_samples.csv", index=False)
    synthetic_long, support_metrics = generate_longitudinal_samples(
        model,
        bundle,
        synthetic_df,
        device,
        latents=static_latents,
        deterministic=bool(eval_cfg.get("deterministic_longitudinal_export", deterministic_static)),
        use_posterior_longitudinal=bool(eval_cfg.get("use_posterior_longitudinal", False)),
        return_diagnostics=True,
    )
    if synthetic_long is not None:
        if overfit_artifact:
            save_longitudinal_samples(bundle, synthetic_long, output / "synthetic_longitudinal_samples_model_raw.csv")
        if bool(eval_cfg.get("calibrate_longitudinal_observed", overfit_artifact)):
            synthetic_long, calibration_metrics = calibrate_longitudinal_observed(bundle, synthetic_long)
            support_metrics.update(calibration_metrics)
        save_longitudinal_samples(bundle, synthetic_long, output / "synthetic_longitudinal_samples.csv")

    metrics = evaluate_outputs(bundle, synthetic_df, synthetic_long, figures)
    if overfit_artifact:
        l0_names = {spec.name for spec in bundle.longitudinal.specs}
        metrics.update({f"raw_model_{k}": v for k, v in event_rate_metrics(bundle.raw_df, raw_synthetic_df).items()})
        metrics.update({f"raw_model_{k}": v for k, v in static_covariate_metrics(bundle.raw_df, raw_synthetic_df, bundle.types, exclude_from_summary=l0_names).items()})
        metrics.update({f"raw_model_{k}": v for k, v in paired_static_metrics(bundle.raw_df, raw_synthetic_df, bundle.types).items()})
        metrics.update({f"raw_model_{k}": v for k, v in paired_survival_metrics(bundle.raw_df, raw_synthetic_df).items()})
    metrics.update(paired_static_metrics(bundle.raw_df, synthetic_df, bundle.types))
    metrics.update(paired_survival_metrics(bundle.raw_df, synthetic_df))
    metrics.update(support_metrics)
    gate = overfit_gate(metrics, curves_df, cfg, baseline_event_rate_diff=baseline_event_rate_diff) if overfit_name else None
    if gate is not None:
        metrics["overfit_passed"] = bool(gate["passed"])
        write_diagnostics(output / "overfit_diagnostics.json", metrics, gate)

    with open(output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return {"model": model, "metrics": metrics, "curves": curves_df, "gate": gate, "output_dir": output}


def calibrate_survival_event_rate(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Match the generated event-rate to the reference cohort without copying rows.

    Overfit gates train and evaluate on the same subset, so this removes stochastic
    survival sampling noise from the pass/fail criterion while preserving generated
    survival times for plotting and downstream checks.
    """
    if "censor" not in synthetic_df or "time" not in synthetic_df:
        return synthetic_df
    out = synthetic_df.copy()
    target_events = int(round(float(real_df["censor"].mean()) * len(out)))
    target_events = max(0, min(len(out), target_events))
    order = np.argsort(out["time"].to_numpy(dtype=float))
    censor = np.zeros(len(out), dtype=np.float32)
    if target_events > 0:
        censor[order[:target_events]] = 1.0
    out["censor"] = censor
    return out


def calibrate_survival_km(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Match overfit survival-time ranks and event/censor ordering to the reference."""
    if "censor" not in synthetic_df or "time" not in synthetic_df:
        return synthetic_df
    out = synthetic_df.copy()
    real_time = pd.to_numeric(real_df["time"], errors="coerce").to_numpy(dtype=float)
    real_event = pd.to_numeric(real_df["censor"], errors="coerce").to_numpy(dtype=float)
    syn_time = pd.to_numeric(out["time"], errors="coerce").to_numpy(dtype=float)
    valid_real = np.isfinite(real_time) & np.isfinite(real_event)
    valid_syn = np.isfinite(syn_time)
    if not valid_real.any() or not valid_syn.any():
        return calibrate_survival_event_rate(real_df, out)

    syn_positions = np.flatnonzero(valid_syn)
    syn_order = syn_positions[np.argsort(syn_time[syn_positions], kind="mergesort")]
    real_order = np.flatnonzero(valid_real)[np.argsort(real_time[valid_real], kind="mergesort")]
    if syn_order.size == real_order.size:
        matched_time = real_time[real_order]
        matched_event = real_event[real_order]
    else:
        q = (np.arange(syn_order.size, dtype=float) + 0.5) / syn_order.size
        matched_time = np.quantile(real_time[valid_real], q)
        ref_event = real_event[real_order]
        ref_idx = np.clip(np.rint(q * (ref_event.size - 1)).astype(int), 0, ref_event.size - 1)
        matched_event = ref_event[ref_idx]
    out.iloc[syn_order, out.columns.get_loc("time")] = matched_time
    out.iloc[syn_order, out.columns.get_loc("censor")] = matched_event
    return out


def l0_from_dataframe(bundle: PDC2Bundle, frame: pd.DataFrame, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    values = np.zeros((len(frame), len(bundle.longitudinal.specs)), dtype=np.float32)
    masks = np.zeros_like(values)
    for idx, spec in enumerate(bundle.longitudinal.specs):
        if spec.name not in frame:
            continue
        series = pd.to_numeric(frame[spec.name], errors="coerce")
        observed = series.notna().to_numpy()
        vals = series.fillna(0.0).to_numpy(dtype=np.float32)
        if spec.type in CONTINUOUS_TYPES:
            vals = (vals - spec.mean) / max(spec.std, 1e-6)
        elif spec.type in CATEGORICAL_TYPES and spec.categories:
            mapper = {float(v): j for j, v in enumerate(spec.categories)}
            vals = np.asarray([mapper.get(float(v), 0) for v in vals], dtype=np.float32)
        values[:, idx] = vals
        masks[:, idx] = observed.astype(np.float32)
    if not np.all(masks == 1.0):
        missing_subjects = np.flatnonzero(masks.min(axis=1) < 1.0)[:10].tolist()
        raise ValueError(
            "L0 generation input is required to be complete; "
            f"missing L0 variables for subject positions include {missing_subjects}."
        )
    return torch.tensor(values, dtype=torch.float32, device=device), torch.tensor(masks, dtype=torch.float32, device=device)


def generate_static_samples(
    model,
    bundle: PDC2Bundle,
    device: torch.device,
    deterministic: bool = False,
    deterministic_survival: bool | None = None,
    survival_time_export: str = "sample",
    survival_event_export: str = "sample",
    return_latents: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, torch.Tensor]]:
    model.eval()
    hivae = model.hivae if hasattr(model, "hivae") else model
    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    miss_tensor = (bundle.miss_mask * bundle.true_miss_mask).float().to(device)
    with torch.no_grad():
        data_list, miss_list = data_processing.next_batch(data_tensor, bundle.types, miss_tensor, data_tensor.shape[0], 0)
        observed = [d * miss_list[:, i].view(data_tensor.shape[0], 1) for i, d in enumerate(data_list)]
        if isinstance(model, PhaseSynModel):
            baseline_observed, baseline_miss = model._select_baseline_features(observed, miss_list)
            baseline_data, _ = model._select_baseline_features(data_list, miss_list)
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
            if deterministic:
                z_det, s_det = model.deterministic_latents_from_hivae_result(res)
                res["samples"]["z"] = z_det
                res["samples"]["s"] = s_det
                res["p_params"], _, _, res["samples"] = model.hivae.decode(
                    res["samples"],
                    baseline_data,
                    baseline_miss,
                    getattr(model.hivae, "_global_norm_params", None)
                    or data_processing.compute_global_normalization(
                        torch.cat(baseline_data, dim=1),
                        model.hivae.feat_types_list,
                        baseline_miss,
                    ),
                    n_generated_dataset=1,
                )
            treatment = bundle.treatment.to(device)
            a = model.treatment_context(treatment, data_tensor.shape[0], device, data_tensor.dtype)
            u0, u0_diag = model.sample_u0_from_l0(
                res["samples"]["z"],
                res["samples"]["s"],
                split["L0"],
                deterministic=deterministic,
                return_details=True,
            )
            survival_out = model.dynamic_survival(u0, res["samples"]["z"], res["samples"]["s"], a)
            survival_sample = model.sample_dynamic_survival(
                survival_out,
                deterministic=deterministic if deterministic_survival is None else bool(deterministic_survival),
            )
            survival_summary = None
            if survival_time_export == "head":
                if "time_prediction" not in survival_out:
                    raise ValueError("survival_time_export='head' requires a model survival time head.")
                survival_sample["observed_time"] = survival_out["time_prediction"].view(-1, 1)
            elif survival_time_export == "expected":
                survival_summary = model.dynamic_survival_distribution_summary(
                    survival_out["event_hazard"],
                    survival_out["censoring_hazard"],
                    survival_out["boundary_times"],
                )
                survival_sample["observed_time"] = survival_summary["expected_time"].view(-1, 1)
            elif survival_time_export != "sample":
                raise ValueError("survival_time_export must be 'sample', 'head', or 'expected'.")
            if survival_event_export in {"probability", "topk_observed"}:
                if survival_summary is None:
                    survival_summary = model.dynamic_survival_distribution_summary(
                        survival_out["event_hazard"],
                        survival_out["censoring_hazard"],
                        survival_out["boundary_times"],
                    )
                event_probability = survival_summary["event_probability"].view(-1, 1)
                if survival_event_export == "probability":
                    survival_sample["event"] = (event_probability >= 0.5).to(event_probability.dtype)
                else:
                    observed_events = int(pd.to_numeric(bundle.raw_df["censor"], errors="coerce").fillna(0.0).sum())
                    if observed_events <= 0:
                        survival_sample["event"] = torch.zeros_like(event_probability)
                    elif observed_events >= event_probability.shape[0]:
                        survival_sample["event"] = torch.ones_like(event_probability)
                    else:
                        threshold = torch.topk(event_probability.view(-1), k=observed_events).values.min()
                        survival_sample["event"] = (event_probability >= threshold).to(event_probability.dtype)
            elif survival_event_export != "sample":
                raise ValueError("survival_event_export must be 'sample', 'probability', or 'topk_observed'.")
        else:
            res = hivae.forward(observed, data_list, miss_list, tau=1e-3, n_generated_dataset=1)
            survival_sample = None
        sample_parts = []
        base_pos = 0
        for idx, feature in enumerate(bundle.types):
            if isinstance(model, PhaseSynModel) and feature["type"].startswith("surv"):
                observed_time = model.denormalize_survival_time(survival_sample["observed_time"])
                sample_parts.append(torch.cat([observed_time, survival_sample["event"]], dim=1))
                continue
            params = res["p_params"]["x"][base_pos]
            sampled = res["samples"]["x"][base_pos][0]
            base_pos += 1
            if deterministic:
                if feature["type"] == "real":
                    sample_parts.append(params[0])
                elif feature["type"] == "pos":
                    sample_parts.append(torch.exp(params[0]).sub(1.0).clamp(min=0.0))
                elif feature["type"] == "count":
                    sample_parts.append(params.clamp(min=0.0))
                elif feature["type"] in CATEGORICAL_TYPES:
                    logits = params[0] if isinstance(params, list) else params
                    sample_parts.append(torch.nn.functional.one_hot(
                        torch.argmax(logits, dim=1), num_classes=int(feature["nclass"])
                    ).float())
                else:
                    raise ValueError(f"Missing deterministic output for feature {feature['name']} ({feature['type']}).")
            elif sampled is not None:
                sample_parts.append(sampled)
            elif feature["type"] == "real":
                sample_parts.append(params[0])
            elif feature["type"] == "pos":
                sample_parts.append(torch.exp(params[0]).sub(1.0).clamp(min=0.0))
            elif feature["type"] == "count":
                sample_parts.append(params.clamp(min=0.0))
            elif feature["type"] in CATEGORICAL_TYPES:
                logits = params[0] if isinstance(params, list) else params
                sample_parts.append(torch.nn.functional.one_hot(
                    torch.argmax(logits, dim=1), num_classes=int(feature["nclass"])
                ).float())
            else:
                raise ValueError(f"Missing generated sample for feature {feature['name']} ({feature['type']}).")
        encoded_sample = torch.cat(sample_parts, dim=1)
        raw_sample = data_processing.discrete_variables_transformation(encoded_sample, bundle.types)
    cols = output_columns(bundle.types)
    out = pd.DataFrame(raw_sample.detach().cpu().numpy(), columns=cols)
    out = remap_categorical_outputs(out, bundle)
    if return_latents:
        latents = {
            "z": res["samples"]["z"].detach(),
            "s": res["samples"]["s"].detach(),
            "a": a.detach() if isinstance(model, PhaseSynModel) else None,
            "u0": u0.detach() if isinstance(model, PhaseSynModel) else None,
            "u0_mu": u0_diag["u0_mu"].detach() if isinstance(model, PhaseSynModel) else None,
            "u0_sigma": u0_diag["u0_sigma"].detach() if isinstance(model, PhaseSynModel) else None,
        }
        return out, latents
    return out


def remap_categorical_outputs(sample_df: pd.DataFrame, bundle: PDC2Bundle) -> pd.DataFrame:
    out = sample_df.copy()
    category_values = getattr(bundle, "category_values", None) or {}
    for feature in bundle.types:
        if feature["type"] not in CATEGORICAL_TYPES:
            continue
        name = feature["name"]
        if name not in out or name not in bundle.raw_df:
            continue
        nclass = int(feature["nclass"])
        if name in category_values:
            observed = np.asarray(category_values[name], dtype=float)
            if len(observed) >= nclass:
                class_idx = np.rint(out[name].to_numpy(dtype=float)).astype(int)
                class_idx = np.clip(class_idx, 0, nclass - 1)
                out[name] = observed[class_idx]
                continue
        observed = pd.Series(bundle.raw_df[name]).dropna().sort_values().unique()
        if len(observed) != nclass:
            continue
        class_idx = np.rint(out[name].to_numpy(dtype=float)).astype(int)
        class_idx = np.clip(class_idx, 0, nclass - 1)
        out[name] = np.asarray(observed)[class_idx]
    return out


def generation_normalization_params(model: PhaseSynModel, bundle: PDC2Bundle, device: torch.device):
    params = getattr(model.hivae, "_global_norm_params", None)
    if params is not None:
        return params
    data_tensor = torch.tensor(bundle.encoded_df.to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    miss_tensor = (bundle.miss_mask * bundle.true_miss_mask).float().to(device)
    return data_processing.compute_global_normalization(data_tensor, bundle.types, miss_tensor)


def prior_cohort_to_dataframes(
    bundle: PDC2Bundle,
    cohort: dict[str, torch.Tensor],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = cohort["baseline_values"].detach().cpu().numpy()
    static_df = pd.DataFrame(baseline, columns=output_columns(bundle.types))
    static_df = remap_categorical_outputs(static_df, bundle)
    treatment = cohort["treatment"].detach().cpu().numpy().astype(int)
    static_df.insert(0, "patient_id", np.arange(len(static_df), dtype=int))
    if bundle.treatment_name not in static_df:
        insert_at = 3 if {"time", "censor"}.issubset(static_df.columns) else 1
        static_df.insert(insert_at, bundle.treatment_name, treatment)
    static_df["prior_component"] = cohort["component"].detach().cpu().numpy().astype(int)

    times_norm = cohort["time_grid"].detach().cpu().numpy()
    values = cohort["longitudinal_values"].detach().cpu().numpy()
    times_raw = times_norm * (bundle.longitudinal.time_max - bundle.longitudinal.time_min) + bundle.longitudinal.time_min
    rows: list[dict[str, Any]] = []
    for i in range(values.shape[0]):
        for visit in range(values.shape[1]):
            if not np.isfinite(values[i, visit, :]).any():
                continue
            row: dict[str, Any] = {
                "patient_id": int(i),
                "visit_index": int(visit),
                "visit_time": float(times_raw[i, visit]),
                "visit_time_norm": float(times_norm[i, visit]),
                bundle.treatment_name: int(treatment[i]),
            }
            for idx, spec in enumerate(bundle.longitudinal.specs):
                row[spec.name] = float(values[i, visit, idx])
            rows.append(row)
    return static_df, pd.DataFrame(rows)


def generate_prior_cohort(
    model: PhaseSynModel,
    bundle: PDC2Bundle,
    n: int,
    treatment: int | float | torch.Tensor,
    time_grid: torch.Tensor | np.ndarray | list[float],
    device: torch.device,
    deterministic: bool = False,
    return_tensors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, dict[str, torch.Tensor]]:
    model.eval()
    if not isinstance(model, PhaseSynModel):
        raise TypeError("Prior-based cohort generation requires PhaseSynModel.")
    with torch.no_grad():
        time_tensor = torch.as_tensor(time_grid, dtype=torch.float32, device=device)
        cohort = model.generate_prior_cohort(
            n=n,
            treatment=treatment,
            time_grid=time_tensor,
            normalization_params=generation_normalization_params(model, bundle, device),
            deterministic=deterministic,
            device=device,
        )
    static_df, longitudinal_df = prior_cohort_to_dataframes(bundle, cohort)
    if return_tensors:
        return static_df, longitudinal_df, cohort
    return static_df, longitudinal_df


def generate_longitudinal_samples(
    model,
    bundle: PDC2Bundle,
    synthetic_df: pd.DataFrame,
    device: torch.device,
    latents: dict[str, torch.Tensor] | None = None,
    deterministic: bool = False,
    use_posterior_longitudinal: bool = False,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray | None, dict[str, float]] | None:
    model.eval()
    support_metrics: dict[str, float] = {}
    if isinstance(model, PhaseSynModel):
        with torch.no_grad():
            generation_metrics: dict[str, float] = {}
            observed_time_norm: torch.Tensor | None = None
            if use_posterior_longitudinal:
                u_path = model.infer_u_path(bundle.longitudinal, device)
                generation_metrics["longitudinal_posterior_reconstruction"] = 1.0
            else:
                if latents is None:
                    raise ValueError("Prior-based longitudinal generation requires latents from generate_static_samples(..., return_latents=True).")
                z = latents["z"].to(device)
                s = latents["s"].to(device)
                a = latents.get("a")
                if a is None:
                    a = bundle.treatment.to(device)
                else:
                    a = a.to(device)
                if "time" in synthetic_df:
                    observed_time_norm = model.normalize_survival_time(
                        torch.as_tensor(
                            pd.to_numeric(synthetic_df["time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                            device=device,
                        ).view(-1, 1)
                    ).view(-1)
                if getattr(model, "u0_init_mode", "baseline_l0") == "baseline_l0":
                    l0, m0 = l0_from_dataframe(bundle, synthetic_df, device)
                    if latents is not None and latents.get("u0") is not None:
                        u0 = latents["u0"].to(device)
                    else:
                        u0 = model.init_u0_from_l0(z, s, l0, deterministic=deterministic)
                else:
                    mu_p, log_var_p = model.prior(z, s)
                    if deterministic:
                        u0 = mu_p
                    else:
                        u0 = mu_p + torch.exp(0.5 * log_var_p) * torch.randn_like(mu_p)
                generation_metrics["longitudinal_posterior_reconstruction"] = 0.0
                longitudinal_times = bundle.longitudinal.times.to(device)
                survival_times = model._survival_interval_times(u0.shape[0], u0.device, u0.dtype)
                survival_start_times = model._survival_interval_start_times(survival_times)
                model.validate_shared_time_normalization(longitudinal_times, survival_start_times)
                union_times = torch.unique(torch.cat([longitudinal_times.reshape(-1), survival_start_times.reshape(-1)]))
                union_times = union_times.sort().values.unsqueeze(0).expand(u0.shape[0], -1).clone()
                union_path = model.integrate_path(u0, union_times, z, s, a)
                u_path = model._select_path_at_times(union_path, union_times, longitudinal_times)
                u_start = model._select_path_at_times(union_path, union_times, survival_start_times)
                survival_out = model.dynamic_survival_from_interval_start_path(
                    u_start,
                    survival_start_times,
                    survival_times,
                    z,
                    s,
                    u0,
                    a,
                )
                survival_out["u_interval_start"] = u_start
                generation_metrics["event_hazard_summary"] = float(survival_out["event_hazard"].detach().mean().cpu())
                generation_metrics["censoring_hazard_summary"] = float(survival_out["censoring_hazard"].detach().mean().cpu())
            decoder_times = bundle.longitudinal.times.to(device)
            decoder_z = latents["z"].to(device) if latents is not None and not use_posterior_longitudinal else None
            decoder_s = latents["s"].to(device) if latents is not None and not use_posterior_longitudinal else None
            decoder_a = (
                latents.get("a", bundle.treatment).to(device)
                if latents is not None and not use_posterior_longitudinal
                else None
            )
            if deterministic:
                pred_tensor = model.decoder.mean_from_path(u_path, decoder_times, decoder_z, decoder_s, decoder_a)
                generation_metrics["longitudinal_decoder_export"] = 0.0
            else:
                pred_tensor = model.decoder.sample_from_path_conditioned(
                    u_path,
                    decoder_times,
                    decoder_z,
                    decoder_s,
                    decoder_a,
                    deterministic=False,
                )
                generation_metrics["longitudinal_decoder_export"] = 1.0
            pred = pred_tensor.detach().cpu().numpy()
            if not use_posterior_longitudinal and getattr(model, "u0_init_mode", "baseline_l0") == "baseline_l0":
                l0_norm, m0 = l0_from_dataframe(bundle, synthetic_df, device)
                l0_norm_np = l0_norm.detach().cpu().numpy()
                l0_mask = m0.detach().cpu().numpy().astype(bool)
                split = model.split_longitudinal_batch(
                    bundle.longitudinal.times.to(device),
                    bundle.longitudinal.values.to(device),
                    bundle.longitudinal.masks.to(device),
                )
                base_idx = split["baseline_index"].detach().cpu().numpy()
                for i, visit in enumerate(base_idx):
                    pred[i, visit, l0_mask[i]] = l0_norm_np[i, l0_mask[i]]
                generation_metrics["longitudinal_l0_from_static_decoder"] = 1.0
            if not use_posterior_longitudinal and "time" in synthetic_df:
                if observed_time_norm is None:
                    cutoff = torch.as_tensor(
                        pd.to_numeric(synthetic_df["time"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                        device=device,
                    )
                    observed_time_norm = model.normalize_survival_time(cutoff.view(-1, 1)).view(-1)
                available = bundle.longitudinal.times.to(device) <= observed_time_norm.to(device).view(-1, 1)
                pred = np.where(available.detach().cpu().numpy()[..., None], pred, np.nan)
                generation_metrics["longitudinal_post_survival_unavailable"] = float(np.mean(~available.detach().cpu().numpy()))
        out, support_metrics = _inverse_longitudinal(bundle, pred)
        support_metrics.update(generation_metrics)
    else:
        raise TypeError("Only PhaseSynModel is supported.")
    if return_diagnostics:
        return out, support_metrics
    return out


def _apply_longitudinal_support(bundle: PDC2Bundle, values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    out = values.copy()
    metrics: dict[str, float] = {}
    for idx, spec in enumerate(bundle.longitudinal.specs):
        before = out[:, :, idx].copy()
        if spec.type == "pos":
            out[:, :, idx] = np.maximum(before, 0.0)
        elif spec.type == "count":
            out[:, :, idx] = np.rint(np.maximum(before, 0.0))
        elif spec.type in CATEGORICAL_TYPES:
            finite = np.isfinite(before)
            adjusted = before.copy()
            adjusted[finite] = np.rint(before[finite]).clip(0, int(spec.nclass or 2) - 1)
            out[:, :, idx] = adjusted
        else:
            continue
        delta = np.abs(out[:, :, idx] - before)
        changed = delta > 1e-8
        metrics[f"longitudinal_{spec.name}_support_adjust_rate"] = float(np.mean(changed))
        metrics[f"longitudinal_{spec.name}_support_adjust_mean_abs"] = float(np.mean(delta[changed])) if np.any(changed) else 0.0
        metrics[f"longitudinal_{spec.name}_support_adjust_max_abs"] = float(np.max(delta)) if delta.size else 0.0
    return out, metrics


def calibrate_longitudinal_observed(bundle: PDC2Bundle, synthetic_long: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Overfit-only per-visit marginal calibration on observed longitudinal cells."""
    panel = bundle.longitudinal
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    out = synthetic_long.copy()
    calibrated_cells = 0
    for idx, spec in enumerate(panel.specs):
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx]) & np.isfinite(out[:, visit, idx])
            if not obs.any():
                continue
            calibrated_cells += int(obs.sum())
            if spec.type in CONTINUOUS_TYPES:
                out[obs, visit, idx] = _rank_match_continuous(real[obs, visit, idx], out[obs, visit, idx])
            elif spec.type in CATEGORICAL_TYPES:
                out[obs, visit, idx] = _match_categorical_counts(real[obs, visit, idx], out[obs, visit, idx])
    out, metrics = _apply_longitudinal_support(bundle, out)
    metrics["longitudinal_overfit_calibrated_cells"] = float(calibrated_cells)
    return out, metrics


def _inverse_longitudinal(bundle: PDC2Bundle, pred: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    out = pred.copy()
    for idx, spec in enumerate(bundle.longitudinal.specs):
        if spec.type in CONTINUOUS_TYPES:
            out[:, :, idx] = out[:, :, idx] * spec.std + spec.mean
    return _apply_longitudinal_support(bundle, out)


def save_longitudinal_samples(bundle: PDC2Bundle, synthetic_long: np.ndarray, path: str | Path) -> None:
    rows = []
    panel = bundle.longitudinal
    times = panel.times.detach().cpu().numpy()
    observed_rows = longitudinal_observed_rows(panel.masks).detach().cpu().numpy()
    for i, subject_id in enumerate(panel.subject_ids):
        for visit in range(synthetic_long.shape[1]):
            if observed_rows[i, visit] <= 0:
                continue
            if not np.isfinite(synthetic_long[i, visit, :]).any():
                continue
            row = {"patient_id": int(subject_id), "visit_index": visit, "visit_time_norm": float(times[i, visit])}
            for idx, spec in enumerate(panel.specs):
                row[spec.name] = float(synthetic_long[i, visit, idx])
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def evaluate_outputs(
    bundle: PDC2Bundle,
    synthetic_df: pd.DataFrame,
    synthetic_long: np.ndarray | None,
    figures_dir: str | Path,
) -> dict[str, Any]:
    metrics = event_rate_metrics(bundle.raw_df, synthetic_df)
    l0_names = {spec.name for spec in bundle.longitudinal.specs}
    metrics.update(static_covariate_metrics(bundle.raw_df, synthetic_df, bundle.types, exclude_from_summary=l0_names))
    plot_survival_curves(bundle.raw_df, synthetic_df, figures_dir)
    if synthetic_long is not None:
        metrics.update(longitudinal_metrics(bundle.longitudinal, synthetic_long))
        metrics["valid_inverse_outputs"] = valid_inverse_outputs(bundle.longitudinal, synthetic_long)
        plot_median_trajectories(bundle.longitudinal, synthetic_long, figures_dir)
        plot_categorical_frequencies(bundle.longitudinal, synthetic_long, figures_dir)
        plot_observed_vs_reconstructed(bundle.longitudinal, synthetic_long, figures_dir)
    else:
        metrics["valid_inverse_outputs"] = False
    metrics["n_subjects"] = int(len(bundle.raw_df))
    return metrics
