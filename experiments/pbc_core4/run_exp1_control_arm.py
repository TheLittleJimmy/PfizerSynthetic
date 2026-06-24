from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .load_pbc import LONGITUDINAL_NAMES, TREATMENT_NAME, project_path
from .methods import ACTIVE_METHODS, PhaseSynGenerator, analysis_static, build_method, save_replicate, split_static_long
from .metrics import (
    BASELINE_COLUMNS,
    baseline_fidelity,
    clinical_estimands,
    clinical_longitudinal_reference,
    longitudinal_fidelity,
    longitudinal_reference,
    privacy_metrics,
    survival_fidelity,
    survival_reference,
)
from .plotting import plot_km_by_method, plot_longitudinal_by_method, plot_metric_bar
from .report import write_markdown_table


EXP1_BENCHMARK_METHODS = [m for m in ACTIVE_METHODS if m != "PhaseSyn"]
EXP1_METHODS = [*EXP1_BENCHMARK_METHODS, "PhaseSyn"]
CATEGORY_METRIC_PATTERNS = {
    "baseline": (
        "baseline_continuous_mean_abs_error",
        "baseline_continuous_sd_abs_error",
        "baseline_mean_abs_smd",
        "baseline_mean_js_distance",
        "baseline_mean_ks",
        "baseline_categorical_prevalence_abs_error",
        "baseline_correlation_matrix_error",
    ),
    "longitudinal": (
        "longitudinal_mean_trajectory_error",
        "longitudinal_change_from_baseline_error",
        "longitudinal_slope_distribution_error",
        "longitudinal_variance_trajectory_error",
    ),
    "survival": (
        "survival_event_rate_error",
        "survival_censoring_rate_error",
        "survival_km_integrated_abs_distance",
        "survival_rmst_difference",
        "survival_median_followup_error",
    ),
    "clinical_estimand": (
        "cox_log_hr",
        "cox_hr",
        "cox_se",
        "cox_p",
        "logrank_p",
        "rmst_difference_treated_minus_control",
        "mmrm_proxy_bili_treatment_effect",
        "mmrm_proxy_albumin_treatment_effect",
        "mmrm_proxy_prothrombin_treatment_effect",
        "responder_rate_diff_bili",
        "responder_rate_diff_albumin",
        "responder_rate_diff_prothrombin",
    ),
    "privacy": (
        "privacy_nearest_neighbor_distance_ratio",
        "privacy_distance_to_closest_real_record",
        "privacy_exact_duplicate_rate",
        "privacy_kmap_mean_equivalence_count",
        "privacy_detection_classifier_auc",
    ),
}


def _clean_exp1_dir(out: Path) -> None:
    if not out.exists():
        return
    for child in out.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _ensure_dirs(out: Path) -> dict[str, Path]:
    paths = {
        "tables": out / "tables",
        "figures": out / "figures",
        "reports": out / "reports",
        "synthetic": out / "synthetic",
        "models": out / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for panel in ["benchmark1_prior_generation", "benchmark2_baseline_conditioned"]:
        for sub in ["tables", "figures", "reports", "synthetic"]:
            (out / panel / sub).mkdir(parents=True, exist_ok=True)
    return paths


def _copy_loaded_phasesyn(cfg: dict[str, Any], data: Any, phasesyn: Any | None, out: Path, smoke: bool) -> PhaseSynGenerator:
    if phasesyn is not None:
        return phasesyn
    train_static, train_long = split_static_long(data, "train")
    generator = PhaseSynGenerator(cfg, train_static, train_long, project_path(cfg["output_dir"]), int(cfg["seed"]))
    checkpoint = project_path(cfg["output_dir"]) / "phasesyn_model" / "train" / "model_checkpoint.pt"
    if checkpoint.exists():
        generator.load_checkpoint(checkpoint, smoke=smoke)
        return generator
    status = generator.train(smoke=smoke)
    if status.get("status") != "completed":
        raise RuntimeError(f"PhaseSyn training failed for Exp1: {status}")
    return generator


def _fit_train_generator(
    cfg: dict[str, Any],
    data: Any,
    method_name: str,
    methods: dict[str, Any],
    phasesyn: Any | None,
    out: Path,
    smoke: bool,
) -> tuple[Any, dict[str, Any]]:
    if method_name == "PhaseSyn":
        generator = _copy_loaded_phasesyn(cfg, data, phasesyn, out, smoke)
        return generator, {
            "fit_status": "completed",
            "generator_training_scope": "train_split_phasesyn",
            "generator_training_runtime_seconds": np.nan,
        }
    if method_name in methods:
        generator = methods[method_name]
        _set_exp1_time_grid(generator, cfg)
        return generator, {
            "fit_status": "completed",
            "generator_training_scope": "train_split_prefit",
            "generator_training_runtime_seconds": np.nan,
        }
    train_static, train_long = split_static_long(data, "train")
    generator = build_method(method_name, train_static, train_long, int(cfg["seed"]) + 7000 + EXP1_METHODS.index(method_name))
    _set_exp1_time_grid(generator, cfg)
    return generator, {
        "fit_status": "completed",
        "generator_training_scope": "train_split_fit_in_exp1",
        "generator_training_runtime_seconds": np.nan,
    }


def _set_exp1_time_grid(generator: Any, cfg: dict[str, Any]) -> None:
    grid = np.asarray(cfg.get("generation", {}).get("time_grid", []), dtype=float)
    grid = np.asarray(sorted(set(float(x) for x in grid if np.isfinite(x))), dtype=float)
    if len(grid) and hasattr(generator, "time_grid"):
        generator.time_grid = grid


def _offset_subject_ids(static: pd.DataFrame, long_df: pd.DataFrame, offset: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = static.copy()
    l = long_df.copy()
    ids = pd.to_numeric(s["subject_id"], errors="coerce").fillna(0).astype(int)
    mapping = {old: int(offset + i) for i, old in enumerate(ids.to_list())}
    s["subject_id"] = ids.map(mapping).to_numpy(dtype=int)
    if not l.empty and "subject_id" in l:
        l["subject_id"] = pd.to_numeric(l["subject_id"], errors="coerce").fillna(-1).astype(int).map(mapping)
        l = l[l["subject_id"].notna()].copy()
        l["subject_id"] = l["subject_id"].astype(int)
    return s, l


def _split_generated_replicates(static: pd.DataFrame, long_df: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    if "replicate" not in static:
        return [(0, static.reset_index(drop=True), long_df.reset_index(drop=True))]
    out = []
    for rep, s in static.groupby("replicate", sort=True):
        if "replicate" in long_df:
            l = long_df[long_df["replicate"].eq(rep)].copy()
        elif long_df.empty or "subject_id" not in long_df:
            l = long_df.copy()
        else:
            ids = set(pd.to_numeric(s["subject_id"], errors="coerce").fillna(-1).astype(int))
            l = long_df[pd.to_numeric(long_df["subject_id"], errors="coerce").fillna(-1).astype(int).isin(ids)].copy()
        out.append((
            int(rep),
            s.drop(columns=["replicate"], errors="ignore").reset_index(drop=True),
            l.drop(columns=["replicate"], errors="ignore").reset_index(drop=True),
        ))
    return out


def _tag_batch_replicates(static: pd.DataFrame, long_df: pd.DataFrame, reps: int, target_size: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = static.copy()
    l = long_df.copy()
    s["replicate"] = np.repeat(np.arange(reps, dtype=int), target_size)[:len(s)]
    if not l.empty and "subject_id" in l:
        rep_map = s.set_index("subject_id")["replicate"]
        l["replicate"] = pd.to_numeric(l["subject_id"], errors="coerce").map(rep_map)
        l = l[l["replicate"].notna()].copy()
        l["replicate"] = l["replicate"].astype(int)
    return s, l


def _combine_by_arm(
    pieces: list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]],
    offset_base: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    static_parts, long_parts = [], []
    diag: dict[str, Any] = {"status": "completed"}
    offset = int(offset_base)
    for static, long_df, part_diag in pieces:
        s, l = _offset_subject_ids(static, long_df, offset)
        offset += len(s) + 1000
        static_parts.append(s)
        long_parts.append(l)
        diag.update({k: v for k, v in part_diag.items() if k not in {"status"}})
        if part_diag.get("status", "completed") != "completed":
            diag["status"] = part_diag.get("status")
    static_out = pd.concat(static_parts, ignore_index=True) if static_parts else pd.DataFrame()
    long_out = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame()
    return static_out, long_out, diag


def _generate_prior_once(
    generator: Any,
    method_name: str,
    n_by_arm: dict[int, int],
    seed_offset: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pieces = []
    for arm, n in sorted(n_by_arm.items()):
        if n <= 0:
            continue
        if method_name == "PhaseSyn":
            static, long_df, diag = generator.generate_prior(int(n), treatment=int(arm))
        else:
            static, long_df, diag = generator.generate(int(n), treatment=int(arm))
            diag = dict(diag)
            diag["generation_mode"] = "train_fit_marginal_prior"
            diag["target_baseline_used"] = False
        pieces.append((static, long_df, diag))
    static_out, long_out, diag = _combine_by_arm(pieces, 1000000 + int(seed_offset) * 100000)
    diag["target_baseline_used"] = False
    return static_out, long_out, diag


def _generate_baseline_conditioned_once(
    generator: Any,
    method_name: str,
    target_static: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    extra_cols = [c for c in ["replicate", "sample"] if c in target_static]
    baseline_cols = ["subject_id", *extra_cols, TREATMENT_NAME, *[c for c in BASELINE_COLUMNS if c in target_static]]
    target_baseline = target_static[[c for c in baseline_cols if c in target_static]].copy()
    if method_name == "PhaseSyn":
        static, long_df, diag = generator.generate(len(target_baseline), treatment=None, target_baseline=target_baseline)
        diag = dict(diag)
        diag["generation_mode"] = "posterior_baseline_conditioned"
        diag["target_baseline_used"] = True
        return static, long_df, diag
    static, long_df, diag = generator.generate(len(target_baseline), treatment=None, target_baseline=target_baseline)
    diag = dict(diag)
    diag["generation_mode"] = "approximate_baseline_conditioned"
    diag["target_baseline_used"] = True
    return static, long_df, diag


def _split_sequential_chunks(
    static: pd.DataFrame,
    long_df: pd.DataFrame,
    n_per_rep: int,
    batch_reps: list[int],
) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    chunks: list[tuple[int, pd.DataFrame, pd.DataFrame]] = []
    s_all = static.reset_index(drop=True).copy()
    l_all = long_df.copy()
    for pos, rep in enumerate(batch_reps):
        start = pos * int(n_per_rep)
        end = start + int(n_per_rep)
        s = s_all.iloc[start:end].copy().reset_index(drop=True)
        if s.empty:
            chunks.append((int(rep), s, pd.DataFrame()))
            continue
        ids = set(pd.to_numeric(s["subject_id"], errors="coerce").dropna().astype(int))
        if not l_all.empty and "subject_id" in l_all:
            l = l_all[pd.to_numeric(l_all["subject_id"], errors="coerce").astype("Int64").isin(ids)].copy().reset_index(drop=True)
        else:
            l = pd.DataFrame()
        chunks.append((int(rep), s, l))
    return chunks


def _generate_prior_batch(
    generator: Any,
    method_name: str,
    n_by_arm: dict[int, int],
    batch_reps: list[int],
    seed_offset: int,
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    rep_pieces: dict[int, list[tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]] = {int(r): [] for r in batch_reps}
    for arm, n in sorted(n_by_arm.items()):
        if n <= 0:
            continue
        total = int(n) * len(batch_reps)
        if method_name == "PhaseSyn":
            static, long_df, diag = generator.generate_prior(total, treatment=int(arm))
        else:
            static, long_df, diag = generator.generate(total, treatment=int(arm))
            diag = dict(diag)
            diag["generation_mode"] = "train_fit_marginal_prior"
            diag["target_baseline_used"] = False
        for rep, s, l in _split_sequential_chunks(static, long_df, int(n), batch_reps):
            rep_pieces[int(rep)].append((s, l, diag))
    out: list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]] = []
    for rep in batch_reps:
        static_out, long_out, diag = _combine_by_arm(rep_pieces[int(rep)], 1000000 + int(seed_offset) * 100000 + int(rep) * 1000)
        diag["target_baseline_used"] = False
        out.append((int(rep), static_out, long_out, diag))
    return out


def _generate_baseline_conditioned_batch(
    generator: Any,
    method_name: str,
    target_static: pd.DataFrame,
    batch_reps: list[int],
) -> list[tuple[int, pd.DataFrame, pd.DataFrame, dict[str, Any]]]:
    baseline_cols = ["subject_id", TREATMENT_NAME, *[c for c in BASELINE_COLUMNS if c in target_static]]
    base = target_static[[c for c in baseline_cols if c in target_static]].copy().reset_index(drop=True)
    targets = []
    stride = len(base) + 1
    for rep in batch_reps:
        target = base.copy()
        target["replicate"] = int(rep)
        target["subject_id"] = np.arange(len(target), dtype=int) + int(rep) * stride
        targets.append(target)
    target_all = pd.concat(targets, ignore_index=True) if targets else base.iloc[0:0].copy()
    static, long_df, diag = _generate_baseline_conditioned_once(generator, method_name, target_all)
    return [(rep, s, l, diag) for rep, s, l in _split_generated_replicates(static, long_df)]


def _metric_row(
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
    synth_static: pd.DataFrame,
    synth_long: pd.DataFrame,
    method_name: str,
    replicate: int,
    setting: str,
    benchmark_task: str,
    diag: dict[str, Any],
    fit_diag: dict[str, Any],
    long_ref: dict[str, Any],
    surv_ref: dict[str, Any],
) -> dict[str, Any]:
    row = baseline_fidelity(real_static, synth_static, method_name, replicate, setting)
    row.update(longitudinal_fidelity(real_long, synth_long, method_name, replicate, setting, real_reference=long_ref))
    row.update(survival_fidelity(real_static, synth_static, method_name, replicate, setting, real_reference=surv_ref))
    row.update({
        "benchmark_task": benchmark_task,
        "status": diag.get("status", "completed"),
        "generation_mode": diag.get("generation_mode", np.nan),
        "target_baseline_used": diag.get("target_baseline_used", np.nan),
        "train_split": "train",
        "eval_split": "test",
        "bootstrap_removed": True,
        **fit_diag,
    })
    return row


def _estimand_row(
    test_static: pd.DataFrame,
    test_long: pd.DataFrame,
    synth_static: pd.DataFrame,
    synth_long: pd.DataFrame,
    method_name: str,
    replicate: int,
    setting: str,
    benchmark_task: str,
) -> dict[str, Any]:
    ref = clinical_longitudinal_reference(pd.concat([test_long, synth_long], ignore_index=True))
    row = clinical_estimands(synth_static, synth_long, method_name, replicate, setting, longitudinal_reference=ref)
    row["benchmark_task"] = benchmark_task
    row["train_split"] = "train"
    row["eval_split"] = "test"
    row["bootstrap_removed"] = True
    return row


def _privacy_row(
    real_static: pd.DataFrame,
    synth_static: pd.DataFrame,
    method_name: str,
    replicate: int,
    setting: str,
    benchmark_task: str,
    smoke: bool,
) -> dict[str, Any]:
    row = privacy_metrics(real_static, synth_static, method_name, replicate, setting, fast=not smoke)
    row["benchmark_task"] = benchmark_task
    row["train_split"] = "train"
    row["eval_split"] = "test"
    row["bootstrap_removed"] = True
    return row


def _write_task_outputs(
    out: Path,
    task: str,
    metrics: pd.DataFrame,
    estimands: pd.DataFrame,
    privacy: pd.DataFrame,
    real_static: pd.DataFrame,
    real_long: pd.DataFrame,
) -> None:
    task_dir = out / task
    tables = task_dir / "tables"
    figures = task_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(tables / f"{task}_metrics_all_methods.csv", index=False)
    estimands.to_csv(tables / f"{task}_estimands_all_methods.csv", index=False)
    privacy.to_csv(tables / f"{task}_privacy_all_methods.csv", index=False)
    plot_metric_bar(metrics, "survival_km_integrated_abs_distance", figures / f"{task}_survival_km_distance.pdf")


def _category_mean_table(metrics: pd.DataFrame, estimands: pd.DataFrame, privacy: pd.DataFrame) -> pd.DataFrame:
    sources = {
        "metrics": metrics,
        "estimands": estimands,
        "privacy": privacy,
    }
    rows: list[dict[str, Any]] = []
    for source_name, df in sources.items():
        if df.empty or "method" not in df or "benchmark_task" not in df:
            continue
        for category, cols in CATEGORY_METRIC_PATTERNS.items():
            available = [col for col in cols if col in df]
            if not available:
                continue
            work = df[["method", "benchmark_task", *available]].copy()
            for col in available:
                values = pd.to_numeric(work[col], errors="coerce")
                if col.endswith("_error") or col.endswith("_difference") or col.endswith("_diff") or col.endswith("_log_hr"):
                    values = values.abs()
                work[col] = values
            for (task, method), group in work.groupby(["benchmark_task", "method"], dropna=False):
                values = group[available].to_numpy(dtype=float).ravel()
                rows.append({
                    "benchmark_task": task,
                    "method": method,
                    "metric_category": category,
                    "source_table": source_name,
                    "category_mean_metric": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                    "category_median_metric": float(np.nanmedian(values)) if np.isfinite(values).any() else np.nan,
                    "n_metric_values": int(np.isfinite(values).sum()),
                    "included_metrics": ";".join(available),
                })
    return pd.DataFrame(rows)


def _plot_category_mean_benchmark(summary: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    categories = [cat for cat in ["baseline", "longitudinal", "survival", "clinical_estimand", "privacy"] if cat in set(summary.get("metric_category", []))]
    tasks = [task for task in ["benchmark1_prior_generation", "benchmark2_baseline_conditioned"] if task in set(summary.get("benchmark_task", []))]
    methods = [method for method in EXP1_METHODS if method in set(summary.get("method", []))]
    if not categories or not tasks or not methods:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(6.0, 4.2 * len(tasks)), 4.6), sharey=False)
    if len(tasks) == 1:
        axes = [axes]
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(categories), dtype=float)
    colors = {
        "PhaseSyn": "#D55E00",
        "LMM-AFT": "#0072B2",
        "JM-RE": "#009E73",
        "TVAE": "#CC79A7",
        "CTGAN": "#E69F00",
    }
    for ax, task in zip(axes, tasks):
        task_df = summary[summary["benchmark_task"].eq(task)]
        for i, method in enumerate(methods):
            vals = []
            for category in categories:
                sub = task_df[task_df["method"].eq(method) & task_df["metric_category"].eq(category)]
                vals.append(float(pd.to_numeric(sub["category_mean_metric"], errors="coerce").mean()) if not sub.empty else np.nan)
            ax.bar(x + (i - (len(methods) - 1) / 2) * width, vals, width=width, label=method, color=colors.get(method, "#666666"))
        ax.set_title("Prior generation" if task == "benchmark1_prior_generation" else "Baseline conditioned")
        ax.set_xticks(x, [cat.replace("_", "\n") for cat in categories])
        ax.set_ylabel("Mean metric within category")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(frameon=False, fontsize=7, ncol=1, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _plot_inputs_from_saved(synth_dir: Path, methods: list[str], real_static: pd.DataFrame, real_long: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    plot_static: dict[str, pd.DataFrame] = {"real_test": real_static}
    plot_long: dict[str, pd.DataFrame] = {"real_test": real_long}
    for method in methods:
        saved = synth_dir / f"{method}_rep000.csv"
        if not saved.exists():
            continue
        merged = pd.read_csv(saved)
        static_cols = [c for c in real_static.columns if c in merged.columns]
        if static_cols and "subject_id" in merged:
            subject = merged[static_cols].drop_duplicates("subject_id").reset_index(drop=True)
            if "time" not in subject and "survival_time" in merged:
                time_map = merged.drop_duplicates("subject_id").set_index("subject_id")["survival_time"]
                subject["time"] = pd.to_numeric(subject["subject_id"], errors="coerce").map(time_map)
            plot_static[method] = subject
        long_cols = [c for c in ["subject_id", "visit_index", "visit_time", TREATMENT_NAME, *LONGITUDINAL_NAMES] if c in merged.columns]
        if {"subject_id", "visit_time"}.issubset(merged.columns) and long_cols:
            plot_long[method] = merged[long_cols].copy()
    return plot_static, plot_long


def run_exp1(
    cfg: dict[str, Any],
    data: Any,
    methods: dict[str, Any],
    phasesyn: Any | None,
    smoke: bool = False,
    method_names: list[str] | None = None,
    append_existing: bool = False,
) -> dict[str, pd.DataFrame]:
    out = project_path(cfg["output_dir"]) / "exp1_control_arm"
    if not smoke and not append_existing:
        _clean_exp1_dir(out)
    paths = _ensure_dirs(out)
    tables, figures, reports, synth = paths["tables"], paths["figures"], paths["reports"], paths["synthetic"]

    requested_methods = method_names or list(EXP1_METHODS)
    requested_methods = [m for m in requested_methods if m in EXP1_METHODS]
    reps = int(cfg["smoke"]["exp1_replicates"] if smoke else cfg["generation"]["exp1_replicates"])

    train_static, train_long = split_static_long(data, "train")
    test_static, test_long = split_static_long(data, "test")
    if smoke:
        cap = int(cfg["smoke"].get("max_eval_subjects", 24))
        keep = set(test_static.head(cap)["subject_id"].astype(int))
        test_static = test_static[test_static["subject_id"].astype(int).isin(keep)].reset_index(drop=True)
        test_long = test_long[test_long["subject_id"].astype(int).isin(keep)].reset_index(drop=True)
    n_by_arm = {
        int(arm): int(count)
        for arm, count in pd.to_numeric(test_static[TREATMENT_NAME], errors="coerce").fillna(0).round().astype(int).value_counts().items()
    }

    long_ref = longitudinal_reference(test_long)
    surv_ref = survival_reference(test_static)
    existing_metrics = pd.read_csv(tables / "exp1_metrics_all_methods.csv") if append_existing and (tables / "exp1_metrics_all_methods.csv").exists() else pd.DataFrame()
    existing_estimands = pd.read_csv(tables / "exp1_estimands_all_methods.csv") if append_existing and (tables / "exp1_estimands_all_methods.csv").exists() else pd.DataFrame()
    existing_privacy = pd.read_csv(tables / "exp1_privacy_all_methods.csv") if append_existing and (tables / "exp1_privacy_all_methods.csv").exists() else pd.DataFrame()
    existing_status = pd.read_csv(tables / "exp1_method_status.csv") if append_existing and (tables / "exp1_method_status.csv").exists() else pd.DataFrame()
    metrics_rows: list[dict[str, Any]] = existing_metrics.to_dict("records") if not existing_metrics.empty else []
    estimand_rows: list[dict[str, Any]] = existing_estimands.to_dict("records") if not existing_estimands.empty else []
    privacy_rows: list[dict[str, Any]] = existing_privacy.to_dict("records") if not existing_privacy.empty else []
    status_rows: list[dict[str, Any]] = existing_status.to_dict("records") if not existing_status.empty else []

    for method_index, method_name in enumerate(requested_methods):
        try:
            generator, fit_diag = _fit_train_generator(cfg, data, method_name, methods, phasesyn, out, smoke)
        except Exception as exc:
            for task in ["benchmark1_prior_generation", "benchmark2_baseline_conditioned"]:
                metrics_rows.append({
                    "method": method_name,
                    "replicate": -1,
                    "setting": task,
                    "benchmark_task": task,
                    "status": "failed_fit",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "train_split": "train",
                    "eval_split": "test",
                    "bootstrap_removed": True,
                })
            status_rows.append({"method": method_name, "status": "failed_fit", "reason": f"{type(exc).__name__}: {exc}"})
            continue
        status_rows.append({"method": method_name, "status": "completed", **fit_diag})

        for task in ["benchmark1_prior_generation", "benchmark2_baseline_conditioned"]:
            task_synth = out / task / "synthetic"
            task_synth.mkdir(parents=True, exist_ok=True)
            batch_size = int(cfg.get("generation", {}).get("exp1_batch_size", 25 if not smoke else reps))
            batch_size = max(1, min(batch_size, reps))
            for start in range(0, reps, batch_size):
                batch_reps = list(range(start, min(reps, start + batch_size)))
                try:
                    if task == "benchmark1_prior_generation":
                        generated = _generate_prior_batch(generator, method_name, n_by_arm, batch_reps, method_index * 1000 + start)
                    else:
                        generated = _generate_baseline_conditioned_batch(generator, method_name, test_static, batch_reps)
                    generated_by_rep = {int(rep): (s, l, d) for rep, s, l, d in generated}
                    for rep in batch_reps:
                        setting = f"{task},rep={rep}"
                        static, long_df, diag = generated_by_rep[int(rep)]
                        diag = dict(diag)
                        diag["generation_batch_size"] = int(len(batch_reps))
                        diag["generation_batch_start"] = int(start)
                        if rep < 3:
                            save_replicate(task_synth, method_name, rep, static, long_df, keep=True)
                            save_replicate(synth / task, method_name, rep, static, long_df, keep=True)
                        metrics_rows.append(_metric_row(test_static, test_long, static, long_df, method_name, rep, setting, task, diag, fit_diag, long_ref, surv_ref))
                        estimand_rows.append(_estimand_row(test_static, test_long, static, long_df, method_name, rep, setting, task))
                        privacy_rows.append(_privacy_row(test_static, static, method_name, rep, setting, task, smoke))
                except Exception as exc:
                    for rep in batch_reps:
                        setting = f"{task},rep={rep}"
                        metrics_rows.append({
                            "method": method_name,
                            "replicate": rep,
                            "setting": setting,
                            "benchmark_task": task,
                            "status": "failed_runtime",
                            "reason": f"{type(exc).__name__}: {exc}",
                            "train_split": "train",
                            "eval_split": "test",
                            "bootstrap_removed": True,
                            **fit_diag,
                        })
                    continue
                if start % max(batch_size * 5, 1) == 0:
                    partial = pd.DataFrame(metrics_rows)
                    if not partial.empty:
                        partial.to_csv(tables / "exp1_metrics_partial_current_run.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    estimands_df = pd.DataFrame(estimand_rows)
    privacy_df = pd.DataFrame(privacy_rows)
    status_df = pd.DataFrame(status_rows)

    metrics_df.to_csv(tables / "exp1_metrics_all_methods.csv", index=False)
    estimands_df.to_csv(tables / "exp1_estimands_all_methods.csv", index=False)
    privacy_df.to_csv(tables / "exp1_privacy_all_methods.csv", index=False)
    status_df.to_csv(tables / "exp1_method_status.csv", index=False)

    for task in ["benchmark1_prior_generation", "benchmark2_baseline_conditioned"]:
        tm = metrics_df[metrics_df["benchmark_task"].eq(task)].reset_index(drop=True)
        te = estimands_df[estimands_df["benchmark_task"].eq(task)].reset_index(drop=True) if not estimands_df.empty else pd.DataFrame()
        tp = privacy_df[privacy_df["benchmark_task"].eq(task)].reset_index(drop=True) if not privacy_df.empty else pd.DataFrame()
        _write_task_outputs(out, task, tm, te, tp, test_static, test_long)

    methods_done = [m for m in EXP1_METHODS if "method" in metrics_df and m in set(metrics_df["method"].dropna())]
    pstatic, plong = _plot_inputs_from_saved(out / "benchmark1_prior_generation" / "synthetic", methods_done, test_static, test_long)
    plot_km_by_method(pstatic, figures / "benchmark1_prior_generation_km.pdf")
    plot_longitudinal_by_method(plong, figures / "benchmark1_prior_generation_longitudinal_bili.pdf", variable="bili")
    pstatic, plong = _plot_inputs_from_saved(out / "benchmark2_baseline_conditioned" / "synthetic", methods_done, test_static, test_long)
    plot_km_by_method(pstatic, figures / "benchmark2_baseline_conditioned_km.pdf")
    plot_longitudinal_by_method(plong, figures / "benchmark2_baseline_conditioned_longitudinal_bili.pdf", variable="bili")
    plot_metric_bar(metrics_df, "survival_km_integrated_abs_distance", figures / "exp1_survival_km_distance_by_task.pdf")
    category_summary = _category_mean_table(metrics_df, estimands_df, privacy_df)
    category_summary.to_csv(tables / "exp1_category_mean_metrics_all_methods.csv", index=False)
    _plot_category_mean_benchmark(category_summary, figures / "exp1_benchmark_category_mean_metrics.pdf")

    summary = metrics_df.groupby(["benchmark_task", "method"], dropna=False).mean(numeric_only=True).reset_index() if not metrics_df.empty else pd.DataFrame()
    write_markdown_table(
        reports / "exp1_summary.md",
        "Experiment 1 Redesigned Summary",
        summary,
        note=(
            "Experiment 1 now has two held-out test benchmarks: prior generation and baseline-conditioned generation. "
            "The empirical_subject_bootstrap method is removed from this benchmark."
        ),
    )
    return {"metrics": metrics_df, "estimands": estimands_df, "privacy": privacy_df}
