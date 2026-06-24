#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from evaluation.longitudinal_plots import (  # noqa: E402
    plot_categorical_frequencies,
    plot_median_trajectories,
    plot_observed_vs_reconstructed,
)
from evaluation.survival_plots import plot_survival_curves  # noqa: E402
from pdc2.data import LongitudinalPanel  # noqa: E402
from pdc2.models import longitudinal_observed_rows  # noqa: E402
from pdc2.plot_overfit_figures import (  # noqa: E402
    ALL_COVARIATES,
    BASELINE_STYLE_DIRS,
    COLS,
    LONG_CONT_COLS,
    REAL_COLOR,
    SYN_COLOR,
    _corr,
    _heatmap,
    _ci,
    _km_curve,
    _numeric,
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
from scripts.pdc2.run_compact_posterior_search import (  # noqa: E402
    plot_categorical_replicates,
    plot_continuous_replicates,
    plot_km_curves_replicates,
    plot_longitudinal_replicate_means,
    plot_longitudinal_replicate_medians,
    plot_survival_time_replicates,
)
from scripts.pdc2.run_holdout_evaluation import (  # noqa: E402
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _make_bundle,
    _read_raw_tables,
)


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _split_order(split_df: pd.DataFrame, split: str, static_all: pd.DataFrame | None = None) -> np.ndarray:
    split_ids = set(split_df.loc[split_df["split"] == split, "row_index"].astype(int).tolist())
    if static_all is not None and "replicate" in static_all and "patient_id" in static_all:
        first_rep = int(static_all["replicate"].min())
        ordered = static_all.loc[static_all["replicate"] == first_rep, "patient_id"].astype(int).to_numpy()
        if set(ordered.tolist()) == split_ids:
            return ordered
    return split_df.loc[split_df["split"] == split, "row_index"].astype(int).to_numpy()


def _test_order(split_df: pd.DataFrame, static_all: pd.DataFrame) -> np.ndarray:
    return _split_order(split_df, "test", static_all)


def _future_mask(panel: LongitudinalPanel, baseline_time_eps: float = 1e-6) -> np.ndarray:
    masks = panel.masks.detach().cpu()
    times = panel.times.detach().cpu()
    observed_rows = longitudinal_observed_rows(masks).bool()
    baseline_candidates = (times.abs() <= baseline_time_eps) & observed_rows
    if baseline_candidates.any(dim=1).all():
        baseline_idx = baseline_candidates.float().argmax(dim=1)
    else:
        baseline_idx = observed_rows.float().argmax(dim=1)
    visit_index = torch.arange(times.shape[1]).view(1, -1)
    future_visit_mask = observed_rows & (visit_index != baseline_idx.view(-1, 1))
    return (masks.bool() & future_visit_mask.unsqueeze(-1)).numpy()


def _baseline_mask(panel: LongitudinalPanel, baseline_time_eps: float = 1e-6) -> np.ndarray:
    masks = panel.masks.detach().cpu()
    times = panel.times.detach().cpu()
    observed_rows = longitudinal_observed_rows(masks).bool()
    baseline_candidates = (times.abs() <= baseline_time_eps) & observed_rows
    if baseline_candidates.any(dim=1).all():
        baseline_idx = baseline_candidates.float().argmax(dim=1)
    else:
        baseline_idx = observed_rows.float().argmax(dim=1)
    visit_index = torch.arange(times.shape[1]).view(1, -1)
    baseline_visit_mask = observed_rows & (visit_index == baseline_idx.view(-1, 1))
    return (masks.bool() & baseline_visit_mask.unsqueeze(-1)).numpy()


def _masked_panel(panel: LongitudinalPanel, mask: np.ndarray) -> LongitudinalPanel:
    return LongitudinalPanel(
        subject_ids=panel.subject_ids.copy(),
        times=panel.times.clone(),
        values=panel.values.clone(),
        masks=torch.tensor(mask.astype(np.float32), dtype=torch.float32),
        raw_values=panel.raw_values.copy(),
        specs=copy.deepcopy(panel.specs),
        time_min=panel.time_min,
        time_max=panel.time_max,
    )


def _longitudinal_replicates(
    long_all: pd.DataFrame,
    panel: LongitudinalPanel,
    future_mask: np.ndarray,
    plot_mask: np.ndarray,
    baseline_mask: np.ndarray,
) -> list[np.ndarray]:
    subject_to_i = {int(pid): i for i, pid in enumerate(panel.subject_ids)}
    spec_names = [spec.name for spec in panel.specs]
    reps: list[np.ndarray] = []
    for rep in sorted(long_all["replicate"].dropna().astype(int).unique()):
        values = np.full(panel.raw_values.shape, np.nan, dtype=np.float32)
        rep_df = long_all[long_all["replicate"].astype(int) == rep]
        for row in rep_df.itertuples(index=False):
            subject = int(getattr(row, "patient_id"))
            visit = int(getattr(row, "visit_index"))
            i = subject_to_i.get(subject)
            if i is None or visit < 0 or visit >= values.shape[1]:
                continue
            for j, name in enumerate(spec_names):
                observed_col = f"{name}_observed"
                if hasattr(row, observed_col) and not bool(getattr(row, observed_col)):
                    continue
                if hasattr(row, name):
                    value = getattr(row, name)
                    if pd.notna(value):
                        values[i, visit, j] = float(value)
        values[baseline_mask] = panel.raw_values[baseline_mask]
        values[~plot_mask] = np.nan
        reps.append(values)
    return reps


def _longitudinal_replicates_from_train_csv(
    long_all: pd.DataFrame,
    panel: LongitudinalPanel,
    plot_mask: np.ndarray,
    baseline_mask: np.ndarray,
) -> list[np.ndarray]:
    subject_to_i = {int(pid): i for i, pid in enumerate(panel.subject_ids)}
    spec_names = [spec.name for spec in panel.specs]
    replicate_values = [1]
    if "replicate" in long_all:
        replicate_values = sorted(long_all["replicate"].dropna().astype(int).unique().tolist())
    reps: list[np.ndarray] = []
    for rep in replicate_values:
        values = np.full(panel.raw_values.shape, np.nan, dtype=np.float32)
        rep_df = long_all
        if "replicate" in long_all:
            rep_df = long_all[long_all["replicate"].astype(int) == rep]
        for row in rep_df.itertuples(index=False):
            subject = int(getattr(row, "patient_id"))
            visit = int(getattr(row, "visit_index"))
            i = subject_to_i.get(subject)
            if i is None or visit < 0 or visit >= values.shape[1]:
                continue
            for j, name in enumerate(spec_names):
                if hasattr(row, name):
                    value = getattr(row, name)
                    if pd.notna(value):
                        values[i, visit, j] = float(value)
        values[baseline_mask] = panel.raw_values[baseline_mask]
        values[~plot_mask] = np.nan
        reps.append(values)
    return reps


def _plot_longitudinal_mean_bands(
    panel: LongitudinalPanel,
    long_reps: list[np.ndarray],
    output_dir: Path,
    split_label: str = "Held-Out Test",
) -> None:
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = panel.times.detach().cpu().numpy() * (panel.time_max - panel.time_min) + panel.time_min
    for name in LONG_CONT_COLS:
        idx = next((i for i, spec in enumerate(panel.specs) if spec.name == name), None)
        if idx is None:
            continue
        xs, real_mean, real_ci, gen_mean, gen_ci = [], [], [], [], []
        for visit in range(real.shape[1]):
            obs = mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
            if not obs.any():
                continue
            rm, rc = _ci(real[:, visit, idx][obs])
            rep_means = []
            for rep in long_reps:
                vals = rep[:, visit, idx][obs]
                if np.isfinite(vals).any():
                    rep_means.append(float(np.nanmean(vals)))
            gm, gc = _ci(np.asarray(rep_means, dtype=float))
            if not (np.isfinite(rm) and np.isfinite(gm)):
                continue
            xs.append(float(np.nanmedian(times[:, visit][obs])))
            real_mean.append(rm)
            real_ci.append(rc)
            gen_mean.append(gm)
            gen_ci.append(gc)
        if not xs:
            continue
        x = np.asarray(xs)
        rm = np.asarray(real_mean)
        rci = np.asarray(real_ci)
        gm = np.asarray(gen_mean)
        gci = np.asarray(gen_ci)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x, rm, marker="o", color=REAL_COLOR, linewidth=2.0, label="Real mean")
        ax.fill_between(x, rm - rci, rm + rci, color=REAL_COLOR, alpha=0.18, label="Real 95% CI")
        ax.plot(x, gm, marker="s", color=SYN_COLOR, linewidth=2.0, label="Synthetic mean")
        ax.fill_between(x, gm - gci, gm + gci, color=SYN_COLOR, alpha=0.15, label="Synthetic 95% CI")
        ax.set_title(f"Replicate Mean 95% CI: {name}", fontsize=15, fontweight="bold")
        ax.set_xlabel("Time (years)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.28)
        ax.legend()
        _savefig(fig, output_dir / "replicate_mean_95ci" / f"{name}_replicate_mean_95ci.png")


def _plot_holdout_metric_panels(
    metrics: pd.DataFrame,
    out_dir: Path,
    split_label: str = "Held-Out Test",
    file_prefix: str = "test",
) -> list[Path]:
    saved: list[Path] = []
    metric_cols = [
        "survival_km_integrated_abs_error",
        "event_rate_diff",
        "survival_time_rmse_ratio",
        "survival_event_accuracy",
        "future_continuous_rmse_ratio_vs_l0_carryforward",
        "future_continuous_ks_mean",
        "future_categorical_accuracy",
        "future_categorical_tv_mean",
    ]
    present = [c for c in metric_cols if c in metrics]
    if present:
        vals = metrics[present].mean(numeric_only=True)
        err = metrics[present].std(ddof=0, numeric_only=True)
        fig, ax = plt.subplots(figsize=(13, 5))
        x = np.arange(len(present))
        ax.bar(x, vals.to_numpy(dtype=float), yerr=err.to_numpy(dtype=float), color="#4c78a8", capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", "\n") for c in present], fontsize=8)
        ax.set_title(f"{split_label} Performance Metrics", fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        path = out_dir / f"{file_prefix}_metric_summary.png"
        _savefig(fig, path)
        saved.append(path)

    if present:
        fig, axes = plt.subplots(math.ceil(len(present) / 3), 3, figsize=(15, 4 * math.ceil(len(present) / 3)))
        axes = np.asarray(axes).ravel()
        for ax, col in zip(axes, present):
            ax.plot(metrics["replicate"], metrics[col], marker="o", linewidth=1.0, color="#4c78a8")
            ax.axhline(float(metrics[col].mean()), color="#c44e52", linestyle="--", linewidth=1.2)
            ax.set_title(col, fontsize=9, fontweight="bold")
            ax.set_xlabel("Replicate")
            ax.grid(alpha=0.25)
        for ax in axes[len(present):]:
            ax.set_visible(False)
        path = out_dir / f"{file_prefix}_metric_replicates.png"
        _savefig(fig, path)
        saved.append(path)
    return saved


def _plot_generation_metric_summary(metrics: pd.DataFrame, out_dir: Path, split_label: str) -> Path | None:
    keys = [
        "survival_km_integrated_abs_error",
        "event_rate_diff",
        "survival_time_rmse_ratio",
        "survival_event_accuracy",
        "future_continuous_rmse_ratio_vs_l0_carryforward",
        "future_continuous_ks_mean",
        "future_categorical_accuracy",
        "future_categorical_tv_mean",
    ]
    present = [key for key in keys if key in metrics]
    if not present:
        return None
    labels = [key.replace("_", "\n") for key in present]
    values = [float(pd.to_numeric(metrics[key], errors="coerce").mean()) for key in present]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(values)), values, color="#4c78a8")
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_title(f"{split_label} Generation Metric Summary")
    ax.grid(axis="y", alpha=0.25)
    path = out_dir / "holdout_metric_summary.png"
    _savefig(fig, path)
    return path


def _plot_survival_replicate_summary(
    real_df: pd.DataFrame,
    synth_list: list[pd.DataFrame],
    out_dir: Path,
    split_label: str,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    rx, ry = _km_curve(real_df["time"].to_numpy(dtype=float), real_df["censor"].to_numpy(dtype=float))
    axes[0].step(rx, ry, where="post", color="black", linewidth=2.2, label=f"Observed {split_label.lower()}")
    grid = np.linspace(0.0, max(float(real_df["time"].max()), 1e-6), 256)
    curves = []
    for syn in synth_list:
        sx, sy = _km_curve(syn["time"].to_numpy(dtype=float), syn["censor"].to_numpy(dtype=float))
        idx = np.searchsorted(sx, grid, side="right") - 1
        idx = np.clip(idx, 0, len(sy) - 1)
        curves.append(sy[idx])
        axes[0].step(sx, sy, where="post", alpha=0.25, linewidth=0.9)
    arr = np.asarray(curves)
    axes[0].plot(grid, arr.mean(axis=0), color="#c44e52", linestyle="--", linewidth=2.0, label="Generated mean")
    axes[0].fill_between(grid, arr.mean(axis=0) - arr.std(axis=0), arr.mean(axis=0) + arr.std(axis=0), color="#c44e52", alpha=0.18)
    axes[0].set_title(f"{split_label} Kaplan-Meier")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Survival probability")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    bins = np.linspace(0.0, max(float(real_df["time"].max()), *(float(s["time"].max()) for s in synth_list)), 28)
    axes[1].hist(real_df["time"], bins=bins, density=True, alpha=0.45, color="#2f6f9f", label="Observed")
    for syn in synth_list:
        axes[1].hist(syn["time"], bins=bins, density=True, alpha=0.05, color="#c44e52")
    axes[1].set_title("Survival Time Distribution")
    axes[1].set_xlabel("Time")
    axes[1].legend(fontsize=8)

    real_rate = float((real_df["censor"] > 0.5).mean())
    rates = np.asarray([float((syn["censor"] > 0.5).mean()) for syn in synth_list])
    axes[2].bar([0, 1], [real_rate, rates.mean()], yerr=[0.0, rates.std()], color=["#2f6f9f", "#c44e52"], tick_label=["Observed", "Generated"])
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Event rate")
    axes[2].set_title(f"{split_label} Event Rate")
    path = out_dir / "survival" / "holdout_survival_summary.png"
    _savefig(fig, path)
    return path


def _plot_replicate_survival_correlation(
    real_df: pd.DataFrame,
    primary: pd.DataFrame,
    output_dir: Path,
    split_label: str = "Held-Out Test",
) -> Path:
    corr_cols = [c for c in ["time"] + ALL_COVARIATES if c in real_df.columns and c in primary.columns]
    corr_r = _corr(real_df, corr_cols)
    corr_s = _corr(primary, corr_cols)
    diff = corr_r - corr_s
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    _heatmap(axes[0], corr_r, f"{split_label} Correlation", "RdBu_r")
    _heatmap(axes[1], corr_s, "Synthetic Rep 1 Correlation", "RdBu_r")
    _heatmap(axes[2], diff, f"Difference ({split_label} - Synthetic)", "RdBu_r", vmin=-0.5, vmax=0.5)
    fig.suptitle("Pairwise Correlation: Time + Conditioning Covariates", fontsize=14, fontweight="bold", y=1.02)
    path = output_dir / "survival" / "correlation_heatmap_replicates.png"
    _savefig(fig, path)
    return path


def _longitudinal_figure_scope(relative_path: str, split: str = "test") -> str:
    source = "heldout_test" if split == "test" else "train"
    if (
        relative_path.endswith("_median_trajectory.png")
        or relative_path.endswith("_observed_vs_reconstructed.png")
        or relative_path.endswith("_category_frequency.png")
        or relative_path.startswith("replicate_mean_95ci/")
        or relative_path.startswith("single_replicate_subject_95ci/")
        or relative_path.startswith("trajectories/")
        or relative_path.startswith("trajectories_replicates/")
        or relative_path.startswith("visit_correlation/")
        or relative_path.startswith("variable_correlation/")
    ):
        return f"{source}_longitudinal_observed_t0_plus_generated_future_t_gt_0"
    if relative_path.startswith("survival/") or relative_path == "survival_curve.png":
        return f"{source}_survival_targets_for_evaluation"
    if relative_path.startswith("covariate/"):
        return f"{source}_baseline_covariate_conditioning_inputs"
    return f"{source}_summary_metrics"


def _figure_dataset_audit(
    split_df: pd.DataFrame,
    static_all: pd.DataFrame,
    long_all: pd.DataFrame,
    metrics: pd.DataFrame,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
    future_mask: np.ndarray,
    baseline_mask: np.ndarray,
    plot_mask: np.ndarray,
    pngs: list[Path],
    output: Path,
    split: str = "test",
) -> dict[str, Any]:
    source = "heldout_test" if split == "test" else "train"
    train_set = set(int(x) for x in train_idx)
    target_set = set(int(x) for x in target_idx)
    split_train = set(split_df.loc[split_df["split"] == "train", "row_index"].astype(int).tolist())
    split_target = set(split_df.loc[split_df["split"] == split, "row_index"].astype(int).tolist())
    reps = sorted(static_all["replicate"].dropna().astype(int).unique().tolist())
    static_rep_audits: dict[str, Any] = {}
    for rep in reps:
        ids = set(static_all.loc[static_all["replicate"].astype(int) == rep, "patient_id"].astype(int).tolist())
        static_rep_audits[str(rep)] = {
            "n_subjects": int(len(ids)),
            "matches_target_subject_ids": ids == target_set,
            "train_overlap": sorted(ids & train_set),
            "missing_target_ids": sorted(target_set - ids),
            "extra_ids_not_in_target": sorted(ids - target_set),
        }
    long_ids = set(long_all["patient_id"].dropna().astype(int).tolist())
    plot_subject_ids = set(int(pid) for pid in target_idx[np.any(plot_mask, axis=(1, 2))])
    png_rows = []
    for path in pngs:
        rel = str(path.relative_to(output))
        png_rows.append({
            "figure": rel,
            "bytes": int(path.stat().st_size),
            "source_dataset": source,
            "source_scope": _longitudinal_figure_scope(rel, split),
        })
    pd.DataFrame(png_rows).to_csv(output / f"{split}_figure_manifest.csv", index=False)
    return {
        "source_dataset": source,
        "all_figures_use_requested_split_reference": True,
        "train_subject_count": int(len(train_set)),
        "target_split": split,
        "target_subject_count": int(len(target_set)),
        "train_target_overlap": sorted(train_set & target_set),
        "split_train_matches_runner": split_train == train_set,
        "split_target_matches_figure_inputs": split_target == target_set,
        "static_replicates_cover_exact_target_subjects": bool(all(item["matches_target_subject_ids"] for item in static_rep_audits.values())),
        "static_replicate_audit": static_rep_audits,
        "longitudinal_subject_ids_subset_of_target": bool(long_ids <= target_set),
        "longitudinal_train_overlap": sorted(long_ids & train_set),
        "longitudinal_missing_target_ids_with_rows": sorted(target_set - long_ids),
        "longitudinal_generated_csv_rows_are_future_only_t_gt_0": bool((long_all["visit_index"].astype(int) > 0).all()),
        "longitudinal_plots_include_time0": bool(baseline_mask.sum() > 0),
        "longitudinal_plot_time0_cell_count": int(baseline_mask.sum()),
        "longitudinal_plot_observed_cells": int(plot_mask.sum()),
        "longitudinal_plot_subject_ids_cover_target": plot_subject_ids == target_set,
        "longitudinal_plot_missing_target_ids": sorted(target_set - plot_subject_ids),
        "longitudinal_future_observed_cells": int(future_mask.sum()),
        "metrics_replicates": int(metrics["replicate"].nunique()) if "replicate" in metrics else int(len(metrics)),
        "png_count": int(len(pngs)),
        "png_zero_byte_count": int(sum(path.stat().st_size <= 0 for path in pngs)),
        "manifest_path": str(output / f"{split}_figure_manifest.csv"),
    }


def generate_figures(holdout_root: Path, output_dir: Path | None = None, split: str = "test") -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    split_dir = holdout_root / split
    output = output_dir or (split_dir / "figures")
    output.mkdir(parents=True, exist_ok=True)
    for subdir in BASELINE_STYLE_DIRS:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    cfg_path = holdout_root / "run_config.yaml"
    split_path = holdout_root / "subject_splits.csv"
    if split == "test":
        static_path = split_dir / "holdout_synthetic_static_all.csv"
        long_path = split_dir / "holdout_synthetic_longitudinal_future_all.csv"
        metrics_path = split_dir / "holdout_replicate_metrics.csv"
    else:
        static_path = split_dir / "synthetic_samples.csv"
        long_path = split_dir / "synthetic_longitudinal_samples.csv"
        metrics_path = split_dir / "metrics.json"
    required = [cfg_path, split_path, static_path, long_path, metrics_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {split} artifacts: {missing}")

    cfg = yaml.safe_load(cfg_path.read_text())
    raw, ids, long_df, types = _read_raw_tables(cfg)
    split_df = pd.read_csv(split_path)
    static_all = pd.read_csv(static_path)
    long_all = pd.read_csv(long_path)
    if split == "test":
        metrics = pd.read_csv(metrics_path)
    else:
        train_metrics = json.loads(metrics_path.read_text())
        train_metrics.setdefault("replicate", 1)
        metrics = pd.DataFrame([train_metrics])
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].astype(int).to_numpy()
    target_idx = _split_order(split_df, split, static_all if split == "test" else None)
    if split == "train":
        if "patient_id" not in static_all:
            if len(static_all) != len(target_idx):
                raise ValueError(f"Train static rows ({len(static_all)}) do not match train subjects ({len(target_idx)}).")
            static_all.insert(0, "patient_id", target_idx)
        if "replicate" not in static_all:
            static_all.insert(0, "replicate", 1)

    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_prep = _fit_longitudinal_preprocessor(long_df, types, train_idx, cfg["dataset"].get("max_visits"))
    bundle = _make_bundle(raw, ids, long_df, types, target_idx, static_prep, long_prep, cfg)
    baseline_time_eps = float(cfg.get("model", {}).get("baseline_time_eps", 1e-6))
    future_mask = _future_mask(bundle.longitudinal, baseline_time_eps=baseline_time_eps)
    baseline_mask = _baseline_mask(bundle.longitudinal, baseline_time_eps=baseline_time_eps)
    plot_mask = future_mask | baseline_mask
    plot_panel = _masked_panel(bundle.longitudinal, plot_mask)

    real_df = _numeric(bundle.raw_df, COLS)
    synth_list: list[pd.DataFrame] = []
    for rep in sorted(static_all["replicate"].dropna().astype(int).unique()):
        syn = static_all[static_all["replicate"].astype(int) == rep].drop(columns=["replicate", "patient_id"], errors="ignore").reset_index(drop=True)
        synth_list.append(_numeric(syn, COLS))
    if not synth_list:
        raise ValueError("No synthetic static replicates found.")
    primary = synth_list[0]
    common_static_cols = [c for c in real_df.columns if c in primary.columns]
    real_df = real_df[common_static_cols]
    synth_list = [s[common_static_cols] for s in synth_list]
    primary = synth_list[0]

    if split == "test":
        long_reps = _longitudinal_replicates(long_all, bundle.longitudinal, future_mask, plot_mask, baseline_mask)
    else:
        long_reps = _longitudinal_replicates_from_train_csv(long_all, bundle.longitudinal, plot_mask, baseline_mask)
    if not long_reps:
        raise ValueError("No synthetic longitudinal replicates found.")
    primary_long = long_reps[0]

    split_label = "Held-Out Test" if split == "test" else "Train"
    saved: list[Path] = []
    saved.extend(_plot_holdout_metric_panels(metrics, output, split_label=split_label, file_prefix=split))
    metric_summary_path = _plot_generation_metric_summary(metrics, output, split_label=split_label)
    if metric_summary_path is not None:
        saved.append(metric_summary_path)
    saved.extend(plot_median_trajectories(plot_panel, primary_long, output))
    saved.extend(plot_categorical_frequencies(plot_panel, primary_long, output))
    saved.extend(plot_observed_vs_reconstructed(plot_panel, primary_long, output))
    saved.append(plot_survival_curves(real_df, primary, output))

    plot_continuous_distributions(real_df, primary, output)
    plot_categorical_distributions(real_df, primary, output)
    plot_correlation_matrices(real_df, primary, output)
    plot_qq(real_df, primary, output)
    plot_summary_statistics(real_df, primary, output)
    plot_continuous_replicates(real_df, synth_list, output / "covariate" / "continuous_distributions_replicates.png")
    plot_categorical_replicates(real_df, synth_list, output / "covariate" / "categorical_distributions_replicates.png")

    plot_survival(real_df, primary, output)
    saved.append(_plot_survival_replicate_summary(real_df, synth_list, output, split_label=split_label))
    plot_km_curves_replicates(real_df, synth_list, output / "survival" / "km_curves_replicates.png")
    plot_survival_time_replicates(real_df, synth_list, output / "survival" / "survival_time_dist_replicates.png")
    _plot_replicate_survival_correlation(real_df, primary, output, split_label=split_label)

    plot_trajectories(plot_panel, primary_long, output)
    plot_mean_ci(plot_panel, primary_long, output)
    plot_visit_correlations(plot_panel, primary_long, output)
    plot_variable_corr_per_visit(plot_panel, primary_long, output)
    plot_longitudinal_replicate_means(plot_panel, long_reps, output / "replicate_mean_95ci")
    plot_longitudinal_replicate_medians(plot_panel, long_reps, output / "trajectories_replicates")
    _plot_longitudinal_mean_bands(plot_panel, long_reps, output, split_label=split_label)

    pngs = sorted(output.rglob("*.png"))
    audit = _figure_dataset_audit(
        split_df,
        static_all,
        long_all,
        metrics,
        train_idx,
        target_idx,
        future_mask,
        baseline_mask,
        plot_mask,
        pngs,
        output,
        split=split,
    )
    summary = {
        "holdout_root": str(holdout_root),
        "split": split,
        "figure_root": str(output),
        "n_png": int(len(pngs)),
        "n_subjects": int(len(target_idx)),
        "n_replicates": int(len(synth_list)),
        "future_observed_cells": int(future_mask.sum()),
        "future_only_longitudinal_figures": False,
        "longitudinal_figures_include_time0": audit["longitudinal_plots_include_time0"],
        "longitudinal_plot_time0_cell_count": audit["longitudinal_plot_time0_cell_count"],
        "longitudinal_generated_csv_rows_are_future_only_t_gt_0": audit["longitudinal_generated_csv_rows_are_future_only_t_gt_0"],
        "baseline_covariates_are_conditioning_inputs": True,
        "dataset_audit_path": str(output / f"{split}_figure_dataset_audit.json"),
        "all_figures_use_requested_split_reference": audit["all_figures_use_requested_split_reference"],
        "train_target_overlap_count": len(audit["train_target_overlap"]),
        "static_replicates_cover_exact_target_subjects": audit["static_replicates_cover_exact_target_subjects"],
        "longitudinal_plot_subject_ids_cover_target": audit["longitudinal_plot_subject_ids_cover_target"],
    }
    with open(output / f"{split}_figure_dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    with open(output / f"{split}_figure_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(output / f"{split}_figure_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {split_label} Performance Figures\n\n")
        f.write(f"Figure root: `{output}`\n\n")
        f.write(f"PNG files: {len(pngs)}; subjects: {len(target_idx)}; replicates: {len(synth_list)}.\n\n")
        f.write("Longitudinal plots include the observed baseline row (`t = 0`) and generated future visits (`t > 0`). Baseline/static covariates are conditioning inputs; survival and future longitudinal variables are generated outputs.\n")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate baseline-style performance figures for PDC2 train/test generation artifacts.")
    parser.add_argument("--holdout-root", default="outputs/pdc2/experiments_20260521/holdout_baseline_l0_0plus")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    args = parser.parse_args(argv)
    summary = generate_figures(Path(args.holdout_root), Path(args.output_dir) if args.output_dir else None, split=args.split)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
