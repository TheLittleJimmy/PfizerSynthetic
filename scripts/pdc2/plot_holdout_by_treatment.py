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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))
sys.path.insert(2, str(ROOT / "utils"))

from pdc2.plot_overfit_figures import LONG_CONT_COLS  # noqa: E402
from scripts.pdc2.plot_holdout_test_figures import (  # noqa: E402
    _baseline_mask,
    _fit_longitudinal_preprocessor,
    _fit_static_preprocessor,
    _future_mask,
    _longitudinal_replicates,
    _make_bundle,
    _masked_panel,
    _read_raw_tables,
    _savefig,
    _test_order,
)


CONTINUOUS_TYPES = {"real", "pos", "count"}
CATEGORICAL_TYPES = {"cat", "ordinal"}


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
    unique_event_times = np.unique(times[events])
    xs = [0.0]
    ys = [1.0]
    surv = 1.0
    for t in unique_event_times:
        at_risk = np.sum(times >= t)
        if at_risk <= 0:
            continue
        d = np.sum((times == t) & events)
        xs.extend([float(t), float(t)])
        ys.extend([surv, surv * (1.0 - d / at_risk)])
        surv = ys[-1]
    xs.append(float(np.nanmax(times)))
    ys.append(surv)
    return np.asarray(xs), np.asarray(ys)


def _patient_treatment_map(static_all: pd.DataFrame, treatment_col: str) -> dict[int, int]:
    first_rep = int(static_all["replicate"].min())
    first = static_all[static_all["replicate"].astype(int) == first_rep]
    return {
        int(row.patient_id): int(getattr(row, treatment_col))
        for row in first.itertuples(index=False)
        if pd.notna(getattr(row, treatment_col))
    }


def _plot_group_counts(group_summary: pd.DataFrame, output: Path, treatment_col: str, split_label: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{treatment_col}={int(v)}" for v in group_summary["treatment"]]
    ax.bar(labels, group_summary["n_subjects"], color=["#4c78a8", "#c44e52", "#54a24b", "#f58518"][: len(labels)])
    ax.set_ylabel("Subjects")
    ax.set_title(f"{split_label.title()} Subjects by Treatment")
    ax.grid(axis="y", alpha=0.25)
    path = output / "summary" / "treatment_group_counts.png"
    _savefig(fig, path)
    return path


def _plot_survival_by_treatment(real_df: pd.DataFrame, static_all: pd.DataFrame, output: Path, treatment_col: str) -> list[Path]:
    saved: list[Path] = []
    treatments = sorted(int(x) for x in real_df[treatment_col].dropna().unique())
    n = len(treatments)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, trt in zip(axes.ravel(), treatments):
        real_g = real_df[real_df[treatment_col].astype(int) == trt]
        x, y = _km_curve(real_g["time"].to_numpy(float), real_g["censor"].to_numpy(float))
        ax.step(x, y, where="post", color="black", linewidth=2.3, label=f"Observed {treatment_col}={trt}")
        rep_curves: list[tuple[np.ndarray, np.ndarray]] = []
        for rep in sorted(static_all["replicate"].dropna().astype(int).unique()):
            syn = static_all[(static_all["replicate"].astype(int) == rep) & (static_all[treatment_col].astype(int) == trt)]
            sx, sy = _km_curve(syn["time"].to_numpy(float), syn["censor"].to_numpy(float))
            rep_curves.append((sx, sy))
            ax.step(sx, sy, where="post", color="#c44e52", alpha=0.13, linewidth=0.9)
        ax.set_title(f"Event-Free Survival: {treatment_col}={trt} (n={len(real_g)})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Survival probability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
        if rep_curves:
            ax.plot([], [], color="#c44e52", alpha=0.6, label="Generated replicates")
        ax.legend(fontsize=8)
    path = output / "survival" / "km_curves_by_treatment.png"
    _savefig(fig, path)
    saved.append(path)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    for ax, trt in zip(axes.ravel(), treatments):
        real_g = real_df[real_df[treatment_col].astype(int) == trt]
        syn_g = static_all[static_all[treatment_col].astype(int) == trt]
        bins = np.linspace(
            min(real_g["time"].min(), syn_g["time"].min()),
            max(real_g["time"].max(), syn_g["time"].max()),
            18,
        )
        ax.hist(real_g["time"], bins=bins, density=True, color="black", alpha=0.35, label="Observed")
        ax.hist(syn_g["time"], bins=bins, density=True, color="#c44e52", alpha=0.35, label="Generated all reps")
        ax.set_title(f"Survival Time Distribution: {treatment_col}={trt}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Density")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    path = output / "survival" / "survival_time_dist_by_treatment.png"
    _savefig(fig, path)
    saved.append(path)
    return saved


def _plot_longitudinal_by_treatment(
    panel,
    long_reps: list[np.ndarray],
    output: Path,
    treatment_by_subject: np.ndarray,
    treatment_col: str,
) -> list[Path]:
    saved: list[Path] = []
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    times = panel.times.detach().cpu().numpy() * (panel.time_max - panel.time_min) + panel.time_min
    treatments = sorted(int(x) for x in np.unique(treatment_by_subject))
    for name in LONG_CONT_COLS:
        idx = next((i for i, spec in enumerate(panel.specs) if spec.name == name), None)
        if idx is None:
            continue
        fig, axes = plt.subplots(1, len(treatments), figsize=(7 * len(treatments), 5), squeeze=False)
        any_panel = False
        for ax, trt in zip(axes.ravel(), treatments):
            subj = treatment_by_subject == trt
            xs, real_mean, gen_mean, gen_ci = [], [], [], []
            for visit in range(real.shape[1]):
                obs = subj & mask[:, visit, idx] & np.isfinite(real[:, visit, idx])
                if not obs.any():
                    continue
                xs.append(float(np.nanmedian(times[:, visit][obs])))
                real_mean.append(float(np.nanmean(real[:, visit, idx][obs])))
                rep_means = [float(np.nanmean(rep[:, visit, idx][obs])) for rep in long_reps]
                gen_mean.append(float(np.nanmean(rep_means)))
                gen_ci.append(float(1.96 * np.nanstd(rep_means, ddof=1) / math.sqrt(len(rep_means))) if len(rep_means) > 1 else 0.0)
            if xs:
                any_panel = True
                x = np.asarray(xs)
                gm = np.asarray(gen_mean)
                gci = np.asarray(gen_ci)
                ax.plot(x, real_mean, marker="o", color="black", linewidth=2.0, label="Observed")
                ax.plot(x, gm, marker="s", color="#c44e52", linewidth=1.8, label="Generated mean")
                ax.fill_between(x, gm - gci, gm + gci, color="#c44e52", alpha=0.2, label="Generated 95% CI")
            ax.axvline(0.0, color="#666666", linestyle=":", linewidth=1.0)
            ax.set_title(f"{name}: {treatment_col}={trt} (n={int(subj.sum())})")
            ax.set_xlabel("Time")
            ax.set_ylabel(name)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        if any_panel:
            path = output / "replicate_mean_95ci_by_treatment" / f"{name}_replicate_mean_95ci_by_treatment.png"
            _savefig(fig, path)
            saved.append(path)
        else:
            plt.close(fig)
    return saved


def _plot_categorical_by_treatment(
    panel,
    long_reps: list[np.ndarray],
    output: Path,
    treatment_by_subject: np.ndarray,
    treatment_col: str,
) -> list[Path]:
    saved: list[Path] = []
    real = panel.raw_values
    mask = panel.masks.detach().cpu().numpy().astype(bool)
    treatments = sorted(int(x) for x in np.unique(treatment_by_subject))
    cat_indices = [i for i, spec in enumerate(panel.specs) if spec.type in CATEGORICAL_TYPES]
    for idx in cat_indices:
        spec = panel.specs[idx]
        cats = list(range(int(spec.nclass or np.nanmax(real[:, :, idx]) + 1)))
        fig, axes = plt.subplots(1, len(treatments), figsize=(7 * len(treatments), 4.5), squeeze=False)
        for ax, trt in zip(axes.ravel(), treatments):
            subj = treatment_by_subject == trt
            obs = subj[:, None] & mask[:, :, idx] & np.isfinite(real[:, :, idx])
            real_vals = np.rint(real[:, :, idx][obs]).astype(int) if obs.any() else np.asarray([], dtype=int)
            real_freq = np.asarray([np.mean(real_vals == c) if real_vals.size else 0.0 for c in cats])
            rep_freqs = []
            for rep in long_reps:
                gen_vals = np.rint(rep[:, :, idx][obs]).astype(int) if obs.any() else np.asarray([], dtype=int)
                rep_freqs.append([np.mean(gen_vals == c) if gen_vals.size else 0.0 for c in cats])
            gen_freq = np.mean(rep_freqs, axis=0) if rep_freqs else np.zeros(len(cats))
            gen_sd = np.std(rep_freqs, axis=0) if rep_freqs else np.zeros(len(cats))
            x = np.arange(len(cats))
            ax.bar(x - 0.18, real_freq, width=0.36, color="black", alpha=0.55, label="Observed")
            ax.bar(x + 0.18, gen_freq, width=0.36, yerr=gen_sd, color="#c44e52", alpha=0.65, label="Generated")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in cats])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"{spec.name}: {treatment_col}={trt} (n={int(subj.sum())})")
            ax.set_xlabel("Category")
            ax.set_ylabel("Frequency")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=8)
        path = output / "longitudinal_categorical" / f"{spec.name}_category_frequency_by_treatment.png"
        _savefig(fig, path)
        saved.append(path)
    return saved


def _group_summary(
    real_df: pd.DataFrame,
    static_all: pd.DataFrame,
    panel,
    long_reps: list[np.ndarray],
    future_mask: np.ndarray,
    treatment_by_subject: np.ndarray,
    treatment_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cont_idx = [i for i, spec in enumerate(panel.specs) if spec.type in CONTINUOUS_TYPES]
    cat_idx = [i for i, spec in enumerate(panel.specs) if spec.type in CATEGORICAL_TYPES]
    for trt in sorted(int(x) for x in np.unique(treatment_by_subject)):
        subj = treatment_by_subject == trt
        real_g = real_df[real_df[treatment_col].astype(int) == trt]
        syn_rates = []
        syn_medians = []
        cont_rmses = []
        cat_accs = []
        for rep_id, rep_arr in zip(sorted(static_all["replicate"].dropna().astype(int).unique()), long_reps):
            syn = static_all[(static_all["replicate"].astype(int) == rep_id) & (static_all[treatment_col].astype(int) == trt)]
            syn_rates.append(float((syn["censor"].astype(float) > 0.5).mean()))
            syn_medians.append(float(syn["time"].median()))
            if cont_idx:
                obs = subj[:, None, None] & future_mask
                obs_cont = obs[:, :, cont_idx] & np.isfinite(panel.raw_values[:, :, cont_idx])
                if obs_cont.any():
                    diff = rep_arr[:, :, cont_idx][obs_cont] - panel.raw_values[:, :, cont_idx][obs_cont]
                    cont_rmses.append(float(np.sqrt(np.mean(diff ** 2))))
            if cat_idx:
                obs = subj[:, None, None] & future_mask
                obs_cat = obs[:, :, cat_idx] & np.isfinite(panel.raw_values[:, :, cat_idx])
                if obs_cat.any():
                    real_cat = np.rint(panel.raw_values[:, :, cat_idx][obs_cat]).astype(int)
                    gen_cat = np.rint(rep_arr[:, :, cat_idx][obs_cat]).astype(int)
                    cat_accs.append(float(np.mean(real_cat == gen_cat)))
        rows.append({
            "treatment": trt,
            "n_subjects": int(subj.sum()),
            "real_event_rate": float((real_g["censor"].astype(float) > 0.5).mean()),
            "synthetic_event_rate_mean": float(np.mean(syn_rates)) if syn_rates else np.nan,
            "synthetic_event_rate_sd": float(np.std(syn_rates)) if syn_rates else np.nan,
            "real_median_time": float(real_g["time"].median()),
            "synthetic_median_time_mean": float(np.mean(syn_medians)) if syn_medians else np.nan,
            "synthetic_median_time_sd": float(np.std(syn_medians)) if syn_medians else np.nan,
            "future_continuous_rmse_mean": float(np.mean(cont_rmses)) if cont_rmses else np.nan,
            "future_continuous_rmse_sd": float(np.std(cont_rmses)) if cont_rmses else np.nan,
            "future_categorical_accuracy_mean": float(np.mean(cat_accs)) if cat_accs else np.nan,
            "future_categorical_accuracy_sd": float(np.std(cat_accs)) if cat_accs else np.nan,
        })
    return pd.DataFrame(rows)


def _longitudinal_replicates_from_train_csv(long_csv: pd.DataFrame, panel) -> list[np.ndarray]:
    subject_to_i = {int(pid): i for i, pid in enumerate(panel.subject_ids)}
    spec_names = [spec.name for spec in panel.specs]
    values = np.full(panel.raw_values.shape, np.nan, dtype=np.float32)
    for row in long_csv.itertuples(index=False):
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
    return [values]


def _add_train_treatment(static_df: pd.DataFrame, bundle, treatment_col: str) -> pd.DataFrame:
    out = static_df.copy()
    if treatment_col not in out:
        out[treatment_col] = bundle.raw_df[treatment_col].to_numpy(dtype=int)
    if "patient_id" not in out:
        out.insert(0, "patient_id", bundle.longitudinal.subject_ids.astype(int))
    if "replicate" not in out:
        out.insert(0, "replicate", 1)
    return out


def _write_treatment_outputs(
    holdout_root: Path,
    output: Path,
    split_label: str,
    treatment_col: str,
    real_df: pd.DataFrame,
    static_all: pd.DataFrame,
    panel,
    long_reps: list[np.ndarray],
    future_mask: np.ndarray,
    treatment_by_subject: np.ndarray,
) -> dict[str, Any]:
    for subdir in ["summary", "survival", "replicate_mean_95ci_by_treatment", "longitudinal_categorical"]:
        (output / subdir).mkdir(parents=True, exist_ok=True)

    summary = _group_summary(real_df, static_all, panel, long_reps, future_mask, treatment_by_subject, treatment_col)
    summary.to_csv(output / "summary" / "treatment_group_summary.csv", index=False)

    _plot_group_counts(summary, output, treatment_col, split_label)
    _plot_survival_by_treatment(real_df, static_all, output, treatment_col)
    _plot_longitudinal_by_treatment(panel, long_reps, output, treatment_by_subject, treatment_col)
    _plot_categorical_by_treatment(panel, long_reps, output, treatment_by_subject, treatment_col)

    pngs = sorted(output.rglob("*.png"))
    report = {
        "holdout_root": str(holdout_root),
        "split": split_label,
        "figure_root": str(output),
        "treatment_column": treatment_col,
        "treatment_groups": summary.to_dict(orient="records"),
        "n_png": int(len(pngs)),
        "n_subjects": int(len(treatment_by_subject)),
        "n_replicates": int(static_all["replicate"].nunique()),
        "uses_observed_treatment_assignment": True,
        "longitudinal_scope": "observed_baseline_t0_plus_generated_visits_by_treatment",
        "survival_scope": f"{split_label}_observed_vs_generated_survival_by_treatment",
    }
    with open(output / "summary" / "treatment_group_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(output / "summary" / "treatment_group_summary.md", "w", encoding="utf-8") as f:
        f.write(f"# {split_label.title()} Results by Treatment Group\n\n")
        f.write(f"Holdout root: `{holdout_root}`\n\n")
        f.write(f"Figure root: `{output}`\n\n")
        f.write(f"Treatment column: `{treatment_col}`; PNG files: {len(pngs)}; subjects: {len(treatment_by_subject)}.\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")
    return report


def _prepare_bundles(holdout_root: Path):
    cfg_path = holdout_root / "run_config.yaml"
    split_path = holdout_root / "subject_splits.csv"
    if not cfg_path.exists() or not split_path.exists():
        raise FileNotFoundError(f"Missing holdout config or split file under {holdout_root}.")
    cfg = yaml.safe_load(cfg_path.read_text())
    raw, ids, long_df, types = _read_raw_tables(cfg)
    split_df = pd.read_csv(split_path)
    train_idx = split_df.loc[split_df["split"] == "train", "row_index"].astype(int).to_numpy()
    static_prep = _fit_static_preprocessor(raw.iloc[train_idx], types)
    long_prep = _fit_longitudinal_preprocessor(long_df, types, train_idx, cfg["dataset"].get("max_visits"))
    return cfg, raw, ids, long_df, types, split_df, train_idx, static_prep, long_prep


def generate_treatment_figures(holdout_root: Path, output_dir: Path | None = None, split: str = "test") -> dict[str, Any]:
    cfg, raw, ids, long_df, types, split_df, train_idx, static_prep, long_prep = _prepare_bundles(holdout_root)
    treatment_col = str(cfg.get("model", {}).get("treatment_variable_name", "drug"))
    baseline_time_eps = float(cfg.get("model", {}).get("baseline_time_eps", 1e-6))

    if split == "both":
        train_output = output_dir / "train" if output_dir is not None else None
        test_output = output_dir / "test" if output_dir is not None else None
        reports = {
            "train": generate_treatment_figures(holdout_root, train_output, split="train"),
            "test": generate_treatment_figures(holdout_root, test_output, split="test"),
        }
        return {
            "holdout_root": str(holdout_root),
            "split": "both",
            "reports": reports,
            "n_png": int(sum(item["n_png"] for item in reports.values())),
        }

    if split == "test":
        test_dir = holdout_root / "test"
        output = output_dir or (test_dir / "figures_by_treatment")
        static_path = test_dir / "holdout_synthetic_static_all.csv"
        long_path = test_dir / "holdout_synthetic_longitudinal_future_all.csv"
        missing = [str(p) for p in [static_path, long_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing holdout test artifacts: {missing}")
        static_all = pd.read_csv(static_path)
        long_all = pd.read_csv(long_path)
        test_idx = _test_order(split_df, static_all)
        bundle = _make_bundle(raw, ids, long_df, types, test_idx, static_prep, long_prep, cfg)
        future_mask = _future_mask(bundle.longitudinal, baseline_time_eps=baseline_time_eps)
        baseline_mask = _baseline_mask(bundle.longitudinal, baseline_time_eps=baseline_time_eps)
        plot_mask = future_mask | baseline_mask
        plot_panel = _masked_panel(bundle.longitudinal, plot_mask)
        long_reps = _longitudinal_replicates(long_all, bundle.longitudinal, future_mask, plot_mask, baseline_mask)
        patient_to_treatment = _patient_treatment_map(static_all, treatment_col)
        treatment_by_subject = np.asarray([patient_to_treatment[int(pid)] for pid in bundle.longitudinal.subject_ids], dtype=int)
        real_df = bundle.raw_df.copy()
        real_df["patient_id"] = test_idx
        real_df[treatment_col] = treatment_by_subject
        return _write_treatment_outputs(
            holdout_root,
            output,
            "test",
            treatment_col,
            real_df,
            static_all,
            plot_panel,
            long_reps,
            future_mask,
            treatment_by_subject,
        )

    if split == "train":
        train_dir = holdout_root / "train"
        output = output_dir or (train_dir / "figures_by_treatment")
        static_path = train_dir / "synthetic_samples.csv"
        long_path = train_dir / "synthetic_longitudinal_samples.csv"
        missing = [str(p) for p in [static_path, long_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing holdout train artifacts: {missing}")
        bundle = _make_bundle(raw, ids, long_df, types, train_idx, static_prep, long_prep, cfg)
        static_all = _add_train_treatment(pd.read_csv(static_path), bundle, treatment_col)
        long_reps = _longitudinal_replicates_from_train_csv(pd.read_csv(long_path), bundle.longitudinal)
        plot_mask = bundle.longitudinal.masks.detach().cpu().numpy().astype(bool)
        plot_panel = _masked_panel(bundle.longitudinal, plot_mask)
        future_mask = _future_mask(bundle.longitudinal, baseline_time_eps=baseline_time_eps)
        treatment_by_subject = bundle.raw_df[treatment_col].to_numpy(dtype=int)
        real_df = bundle.raw_df.copy()
        real_df["patient_id"] = bundle.longitudinal.subject_ids.astype(int)
        return _write_treatment_outputs(
            holdout_root,
            output,
            "train",
            treatment_col,
            real_df,
            static_all,
            plot_panel,
            long_reps,
            future_mask,
            treatment_by_subject,
        )

    raise ValueError(f"Unsupported split: {split}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate held-out PDC2 plots stratified by randomized treatment group.")
    parser.add_argument("--holdout-root", default="outputs/pdc2/experiments_20260525/holdout_baseline_l0_treatment_explicit")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["train", "test", "both"], default="test")
    args = parser.parse_args(argv)
    summary = generate_treatment_figures(
        Path(args.holdout_root),
        Path(args.output_dir) if args.output_dir else None,
        split=args.split,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
