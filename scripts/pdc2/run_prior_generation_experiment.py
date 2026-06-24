#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
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

from evaluation.survival_plots import _km_curve  # noqa: E402
from pdc2.config import load_config  # noqa: E402
from pdc2.data import PDC2Bundle, load_pdc2_bundle  # noqa: E402
from pdc2.models import build_model  # noqa: E402
from pdc2.training import generate_prior_cohort  # noqa: E402
from scripts.pdc2.run_compact_posterior_search import (  # noqa: E402
    CAT_COLS,
    SIMILARITY_KEYS,
    STATIC_CONT_COLS,
    _check_embedding_dims,
    _count_parameters,
    _longitudinal_distribution_metrics,
    _plot_full_cohort_protocol,
    _safe_corr_mae,
    _static_rep_metrics,
    _survival_rep_metrics,
)


DEFAULT_REFERENCE_DIR = Path(
    "outputs/pdc2/experiments_20260525/phase_syn_treatment_explicit_final/"
    "rand_0p01/round11_small8_regularized"
)
DEFAULT_OUTPUT_ROOT = Path("outputs/pdc2/experiments_20260527")


def _parse_treatments(value: str) -> list[int]:
    treatments = [int(x) for x in value.replace(",", " ").split() if x]
    if not treatments:
        raise ValueError("--treatments must contain at least one integer arm.")
    return treatments


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    loaded = torch.load(checkpoint, map_location="cpu")
    return loaded.get("model_state_dict", loaded)


def _configure_checkpoint_compatibility(cfg: dict[str, Any], state: dict[str, torch.Tensor]) -> dict[str, Any]:
    info: dict[str, Any] = {"legacy_l0_encoder_checkpoint": False}
    if any(key.startswith("l0_encoder.") for key in state):
        cfg["model"]["l0_initializer_mode"] = "encoded"
        if "l0_encoder.net.2.weight" in state:
            cfg["model"]["l0_embedding_dim"] = int(state["l0_encoder.net.2.weight"].shape[0])
        info["legacy_l0_encoder_checkpoint"] = True
        info["l0_initializer_mode"] = "encoded"
        info["l0_embedding_dim"] = int(cfg["model"].get("l0_embedding_dim", cfg["model"].get("u_dim", 0)))
    return info


def _load_model(bundle: PDC2Bundle, cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    state = _load_state_dict(checkpoint)
    compat = _configure_checkpoint_compatibility(cfg, state)
    model = build_model(bundle, cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing_prefixes = ("longitudinal_baseline_decoder.", "u0_logsigma_head.")
    bad_missing = [key for key in missing if not key.startswith(allowed_missing_prefixes)]
    if bad_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with the current model. "
            f"bad_missing={bad_missing[:10]}, unexpected={unexpected[:10]}"
        )
    model.eval()
    compat["checkpoint"] = str(checkpoint)
    compat["missing_keys_ignored"] = list(missing)
    compat["unexpected_keys"] = list(unexpected)
    return model, compat


def _metric_static_frame(static_df: pd.DataFrame, treatment_name: str) -> pd.DataFrame:
    return static_df.drop(
        columns=["replicate", "patient_id", "prior_component", "treatment_arm", treatment_name],
        errors="ignore",
    )


def _plot_treatment_counts(bundle: PDC2Bundle, static_all: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    real_counts = bundle.raw_df[bundle.treatment_name].value_counts().sort_index()
    synth_counts = (
        static_all.groupby(["treatment_arm", "replicate"], sort=True)
        .size()
        .groupby(level=0)
        .mean()
        .sort_index()
    )
    labels = [f"Real A={int(a)}" for a in real_counts.index] + [f"Prior A={int(a)}" for a in synth_counts.index]
    values = [float(v) for v in real_counts.to_list()] + [float(v) for v in synth_counts.to_list()]
    colors = ["#4c78a8"] * len(real_counts) + ["#c44e52"] * len(synth_counts)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(np.arange(len(values)), values, color=colors, alpha=0.88)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Subjects")
    ax.set_title("Observed Treatment Counts and Fixed Prior Cohorts", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "prior_fixed_treatment_counts.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_km_by_treatment(bundle: PDC2Bundle, static_all: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {0: "#4c78a8", 1: "#c44e52"}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for arm in sorted(static_all["treatment_arm"].unique()):
        arm_int = int(arm)
        real = bundle.raw_df[pd.to_numeric(bundle.raw_df[bundle.treatment_name], errors="coerce") == arm_int]
        if not real.empty:
            x, y = _km_curve(real["time"].to_numpy(dtype=float), real["censor"].to_numpy(dtype=float) > 0.5)
            ax.step(x, y, where="post", color=colors.get(arm_int, "black"), linewidth=2.5, label=f"Real A={arm_int}")
        for rep in sorted(static_all["replicate"].unique()):
            syn = static_all[(static_all["treatment_arm"] == arm) & (static_all["replicate"] == rep)]
            x, y = _km_curve(syn["time"].to_numpy(dtype=float), syn["censor"].to_numpy(dtype=float) > 0.5)
            ax.step(x, y, where="post", color=colors.get(arm_int, "black"), alpha=0.16, linewidth=0.9)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.set_title("Prior Survival Curves Under Predefined Treatment", fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prior_km_by_predefined_treatment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_metric_summary(metrics: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_names = [
        "static_continuous_mean_ks",
        "static_categorical_mean_tv",
        "static_corr_mae",
        "survival_km_integrated_abs_error",
        "event_rate_diff",
        "long_continuous_ks_mean",
        "long_categorical_tv_mean",
    ]
    metric_names = [name for name in metric_names if name in metrics.columns]
    if not metric_names:
        return
    arms = sorted(metrics["treatment_arm"].unique())
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(arms), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for pos, arm in enumerate(arms):
        vals = [float(metrics.loc[metrics["treatment_arm"] == arm, name].mean()) for name in metric_names]
        offset = (pos - (len(arms) - 1) / 2.0) * width
        ax.bar(x + offset, vals, width=width, label=f"A={int(arm)}", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=25, ha="right")
    ax.set_ylabel("Metric value")
    ax.set_title("Prior Generation Replicate Metrics", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "prior_metric_summary_by_treatment.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _summarize(
    metrics_rows: list[dict[str, Any]],
    model: torch.nn.Module,
    cfg: dict[str, Any],
    output_dir: Path,
    n_subjects: int,
    treatments: list[int],
    compat: dict[str, Any],
) -> dict[str, Any]:
    df = pd.DataFrame(metrics_rows)
    summary: dict[str, Any] = {
        "candidate": str(cfg.get("compact_candidate", output_dir.name)),
        "mode": "prior_based_generation",
        "prior_generation": True,
        "prior_generation_kind": "prior synthetic cohort under predefined treatment",
        "baseline_generated_from_prior": bool(df["baseline_generated_from_prior"].all()) if "baseline_generated_from_prior" in df else True,
        "uses_observed_future_outcomes": bool(df["uses_observed_future_outcomes"].any()) if "uses_observed_future_outcomes" in df else False,
        "n_replicates_per_treatment": int(df.groupby("treatment_arm")["replicate"].nunique().min()),
        "treatment_arms": [int(x) for x in treatments],
        "n_subjects_per_replicate": int(n_subjects),
        "generated_total_subject_rows": int(len(df) * n_subjects),
        "full_cohort_reference": True,
        "embedding_dims_leq_6": _check_embedding_dims(cfg),
        "checkpoint_compatibility": compat,
    }
    summary.update(_count_parameters(model))
    for col in df.columns:
        if col in {"replicate", "treatment_arm"} or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_sd"] = float(df[col].std(ddof=0))
    by_arm: dict[str, Any] = {}
    for arm, arm_df in df.groupby("treatment_arm", sort=True):
        arm_summary: dict[str, float] = {}
        for col in arm_df.columns:
            if col in {"replicate", "treatment_arm"} or not pd.api.types.is_numeric_dtype(arm_df[col]):
                continue
            arm_summary[f"{col}_mean"] = float(arm_df[col].mean())
            arm_summary[f"{col}_sd"] = float(arm_df[col].std(ddof=0))
        by_arm[str(int(arm))] = arm_summary
    summary["by_treatment"] = by_arm
    param_penalty = float(summary["parameter_count"]) / 1_000_000.0
    summary["generation_similarity_score"] = float(sum(float(summary.get(key, 0.0)) for key in SIMILARITY_KEYS) + 0.05 * param_penalty)
    with open(output_dir / "prior_generation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def _write_markdown(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Prior-Based PhaseSyn Generation",
        "",
        f"Source checkpoint: `{summary['checkpoint_compatibility']['checkpoint']}`",
        f"Treatment arms: {summary['treatment_arms']}",
        f"Replicates per arm: {summary['n_replicates_per_treatment']}",
        f"Subjects per replicate: {summary['n_subjects_per_replicate']}",
        f"Baseline generated from prior: {summary['baseline_generated_from_prior']}",
        f"Uses observed future outcomes: {summary['uses_observed_future_outcomes']}",
        "",
        "Key outputs:",
        "",
        "- `prior_synthetic_static_all.csv`",
        "- `prior_synthetic_longitudinal_all.csv`",
        "- `prior_replicate_metrics.csv`",
        "- `figures/`",
        "- `treatment_<arm>/figures/`",
        "",
    ]
    with open(output_dir / "README_prior_generation.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run(args: argparse.Namespace) -> Path:
    reference_dir = Path(args.reference_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else reference_dir / "model_checkpoint.pt"
    cfg = load_config(reference_dir / "compact_config.yaml")
    state = _load_state_dict(checkpoint)
    compat = _configure_checkpoint_compatibility(cfg, state)
    device = _device(args.device)
    cfg["training"]["device"] = str(device)
    treatments = _parse_treatments(args.treatments)
    n_replicates = int(args.n_replicates or cfg.get("evaluation", {}).get("n_replicates", 20))

    bundle = load_pdc2_bundle(cfg)
    n_subjects = int(args.n_subjects or len(bundle.raw_df))
    if n_subjects != len(bundle.raw_df):
        raise ValueError("This experiment script expects full-cohort generation so standard plots remain aligned.")
    model, load_info = _load_model(bundle, cfg, checkpoint, device)
    compat.update(load_info)

    output_dir = Path(args.output_root) / "prior_based_generation" / str(cfg.get("compact_candidate", "round11_small8_regularized"))
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(reference_dir / "compact_config.yaml", output_dir / "source_compact_config.yaml")
    with open(output_dir / "prior_run_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    all_static: list[pd.DataFrame] = []
    all_long: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, Any]] = []
    time_grid = bundle.longitudinal.times.to(device)
    for treatment in treatments:
        arm_dir = output_dir / f"treatment_{treatment}"
        rep_dir = arm_dir / "prior_replicates"
        rep_dir.mkdir(parents=True, exist_ok=True)
        arm_static: list[pd.DataFrame] = []
        arm_long: list[pd.DataFrame] = []
        for rep in range(1, n_replicates + 1):
            seed = int(args.seed) + treatment * 1000 + rep
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            static_df, long_df, tensors = generate_prior_cohort(
                model,
                bundle,
                n=n_subjects,
                treatment=treatment,
                time_grid=time_grid,
                device=device,
                deterministic=bool(args.deterministic),
                return_tensors=True,
            )
            static_df.insert(0, "replicate", rep)
            static_df.insert(1, "treatment_arm", treatment)
            long_df.insert(0, "replicate", rep)
            long_df.insert(1, "treatment_arm", treatment)
            static_df.to_csv(rep_dir / f"prior_synthetic_static_rep{rep:02d}.csv", index=False)
            long_df.to_csv(rep_dir / f"prior_synthetic_longitudinal_rep{rep:02d}.csv", index=False)
            arm_static.append(static_df)
            arm_long.append(long_df)
            all_static.append(static_df)
            all_long.append(long_df)

            metric_static = _metric_static_frame(static_df, bundle.treatment_name)
            long_values = tensors["longitudinal_values"].detach().cpu().numpy()
            row: dict[str, Any] = {
                "replicate": int(rep),
                "treatment_arm": int(treatment),
                "generated_subject_count": int(len(static_df)),
                "fixed_treatment_unique_count": int(static_df[bundle.treatment_name].nunique()),
                "fixed_treatment_matches": bool((static_df[bundle.treatment_name].astype(int) == int(treatment)).all()),
                "baseline_generated_from_prior": bool(tensors["baseline_generated_from_prior"].item()),
                "uses_observed_future_outcomes": bool(tensors["uses_observed_future_outcomes"].item()),
                "prior_component_n_unique": int(static_df["prior_component"].nunique()),
            }
            row.update(_static_rep_metrics(bundle, bundle.raw_df, metric_static))
            row["static_corr_mae_without_treatment"] = _safe_corr_mae(bundle.raw_df, metric_static, CAT_COLS + STATIC_CONT_COLS)
            row.update(_survival_rep_metrics(bundle.raw_df, metric_static))
            row.update(_longitudinal_distribution_metrics(bundle.longitudinal, long_values))
            metrics_rows.append(row)

        arm_static_all = pd.concat(arm_static, ignore_index=True)
        arm_long_all = pd.concat(arm_long, ignore_index=True)
        arm_static_all.to_csv(arm_dir / "prior_synthetic_static_all.csv", index=False)
        arm_long_all.to_csv(arm_dir / "prior_synthetic_longitudinal_all.csv", index=False)
        pd.DataFrame([row for row in metrics_rows if row["treatment_arm"] == treatment]).to_csv(
            arm_dir / "prior_replicate_metrics.csv", index=False
        )
        if not args.skip_plots:
            plot_static = arm_static_all.drop(columns=["patient_id", "prior_component", "treatment_arm"], errors="ignore")
            _plot_full_cohort_protocol(bundle, plot_static, arm_long_all.drop(columns=["treatment_arm"], errors="ignore"), arm_dir)

    static_all = pd.concat(all_static, ignore_index=True)
    long_all = pd.concat(all_long, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows)
    static_all.to_csv(output_dir / "prior_synthetic_static_all.csv", index=False)
    long_all.to_csv(output_dir / "prior_synthetic_longitudinal_all.csv", index=False)
    metrics_df.to_csv(output_dir / "prior_replicate_metrics.csv", index=False)

    if not args.skip_plots:
        figure_dir = output_dir / "figures"
        _plot_treatment_counts(bundle, static_all, figure_dir)
        _plot_km_by_treatment(bundle, static_all, figure_dir)
        _plot_metric_summary(metrics_df, figure_dir)

    summary = _summarize(metrics_rows, model, cfg, output_dir, n_subjects, treatments, compat)
    _write_markdown(output_dir, summary)
    print(json.dumps({k: summary[k] for k in [
        "mode",
        "candidate",
        "treatment_arms",
        "n_replicates_per_treatment",
        "n_subjects_per_replicate",
        "baseline_generated_from_prior",
        "uses_observed_future_outcomes",
        "generation_similarity_score",
    ]}, indent=2))
    print(f"wrote {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prior-based PhaseSyn cohort generation and full-cohort figures.")
    parser.add_argument("--reference-dir", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-replicates", type=int, default=None)
    parser.add_argument("--n-subjects", type=int, default=None)
    parser.add_argument("--treatments", default="0,1")
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
