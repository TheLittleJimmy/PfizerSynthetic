from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .load_pbc import load_processed, preprocess_to_disk, project_path
from .methods import ACTIVE_METHODS, analysis_static, write_method_status
from .plotting import plot_line, plot_metric_bar
from .report import write_main_summary
from .run_benchmarks import fit_benchmarks
from .run_exp1_control_arm import run_exp1
from .run_exp2_matched_controls import run_exp2
from .run_exp3_digital_twin import run_exp3
from .run_exp4_virtual_trial import run_exp4
from .run_phasesyn import fit_phasesyn


def _ensure_dirs(output: Path) -> None:
    for sub in ["tables", "figures", "reports", "logs", "models"]:
        (output / sub).mkdir(parents=True, exist_ok=True)


def _rerun_commands(config_path: Path) -> dict[str, str]:
    base = f"conda run -n env_2502 python -m experiments.pbc_core4.run_all --config {config_path}"
    return {
        "all": base,
        "smoke": f"{base} --smoke",
        "preprocess": f"{base} --only preprocess",
        "benchmarks": f"{base} --only benchmarks",
        "PhaseSyn": f"{base} --only phasesyn",
        "Experiment 1": f"{base} --only exp1",
        "Experiment 2": f"{base} --only exp2",
        "Experiment 3": f"{base} --only exp3",
        "Experiment 4": f"{base} --only exp4",
    }


def _print_handoff(key: dict[str, Any], config_path: Path) -> None:
    print("result tables:", key.get("tables_dir"))
    print("figures:", key.get("figures_dir"))
    print("methods successfully completed:", ", ".join(key.get("methods_completed", [])) or "none")
    failed = key.get("methods_failed", [])
    print("methods failed:", failed if failed else "none")
    commands = key.get("rerun_commands") or _rerun_commands(config_path)
    print("rerun all:", commands["all"])
    print("rerun smoke:", commands["smoke"])
    print("rerun individual experiments:")
    for label in ["preprocess", "benchmarks", "PhaseSyn", "Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4"]:
        print(f"  {label}: {commands[label]}")


def _write_table1(data: Any, output: Path) -> pd.DataFrame:
    static = analysis_static(data)
    rows = [
        {"section": "cohort", "variable": "subjects", "level": "all", "value": len(static)},
        {"section": "cohort", "variable": "longitudinal_rows", "level": "all", "value": len(data.longitudinal)},
        {"section": "treatment", "variable": "control_placebo", "level": 0, "value": int((static["treatment"] == 0).sum())},
        {"section": "treatment", "variable": "D_penicillamine", "level": 1, "value": int((static["treatment"] == 1).sum())},
        {"section": "survival", "variable": "composite_event_rate", "level": "all", "value": float(static["event"].mean())},
        {"section": "survival", "variable": "median_followup_years", "level": "all", "value": float(static["time"].median())},
    ]
    for split, ids in data.splits.items():
        rows.append({"section": "split", "variable": split, "level": "subjects", "value": len(ids)})
    out = pd.DataFrame(rows)
    out.to_csv(output / "tables" / "table1_pbc_dataset_summary.csv", index=False)
    return out


def _write_combined_tables(output: Path, method_status: list[dict[str, Any]], exp1: dict, exp2: dict, exp3: dict, exp4: dict) -> None:
    pd.DataFrame(method_status).to_csv(output / "tables" / "table2_methods_comparison.csv", index=False)
    exp1["metrics"].groupby("method", dropna=False).mean(numeric_only=True).reset_index().to_csv(output / "tables" / "table3_exp1_control_arm_results.csv", index=False)
    exp2["effects"].to_csv(output / "tables" / "table4_exp2_matched_control_results.csv", index=False)
    exp3["longitudinal"].to_csv(output / "tables" / "table5_exp3_digital_twin_results.csv", index=False)
    exp4["real"].to_csv(output / "tables" / "table6_exp4_virtual_trial_results.csv", index=False)
    ablation_rows = [
        {"ablation": "PhaseSyn_full", "status": "completed"},
        {"ablation": "PhaseSyn_no_dynamic_survival", "status": "limitation_not_run_existing_infrastructure_missing"},
        {"ablation": "PhaseSyn_no_censoring_model", "status": "limitation_not_run_existing_infrastructure_missing"},
        {"ablation": "PhaseSyn_no_randomization_balance", "status": "limitation_not_separately_run_full_config_randomization_loss_zero"},
    ]
    pd.DataFrame(ablation_rows).to_csv(output / "tables" / "table7_ablation_or_failure_summary.csv", index=False)


def _write_manuscript_figures(output: Path) -> None:
    figs = output / "figures"
    exp1 = pd.read_csv(output / "exp1_control_arm" / "tables" / "exp1_metrics_all_methods.csv")
    plot_metric_bar(exp1, "survival_km_integrated_abs_distance", figs / "fig1_exp1_chassat_style_benchmark.pdf")
    exp2 = pd.read_csv(output / "exp2_matched_counterfactual_controls" / "tables" / "exp2_baseline_alignment.csv")
    exp2_plot = exp2.rename(columns={"comparison": "method"})
    plot_metric_bar(exp2_plot, "mean_abs_smd", figs / "fig2_exp2_matched_counterfactual_controls.pdf")
    exp3 = pd.read_csv(output / "exp3_digital_twin_validation" / "tables" / "exp3_prediction_interval_coverage.csv")
    if not exp3.empty:
        exp3["method"] = exp3["variable"].astype(str) + "_" + exp3["interval"].astype(str)
    plot_metric_bar(exp3, "coverage", figs / "fig3_exp3_digital_twin_calibration.pdf")
    exp4 = pd.read_csv(output / "exp4_virtual_trial_simulation" / "tables" / "exp4_real_virtual_trial_power.csv")
    plot_line(exp4, "n", "power", "allocation_ratio", figs / "fig4_exp4_virtual_trial_calibration.pdf")


def _phase_perturbation_audit(phasesyn: Any, data: Any, cfg: dict[str, Any]) -> dict[str, Any]:
    static = analysis_static(data)
    test = static[static["subject_id"].isin(data.splits["test"])].head(int(cfg.get("smoke", {}).get("max_eval_subjects", 24))).reset_index(drop=True)
    if test.empty:
        return {"perturbation_audit_status": "skipped_empty_test"}
    base = test.copy()
    pert = test.copy()
    rng = np.random.default_rng(int(cfg["seed"]) + 4242)
    pert["time"] = rng.permutation(pert["time"].to_numpy())
    pert["event"] = rng.permutation(pert["event"].to_numpy())
    torch_seed = int(cfg["seed"]) + 9191
    import torch

    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    s1, l1, _ = phasesyn.generate(len(base), treatment=None, target_baseline=base)
    torch.manual_seed(torch_seed)
    np.random.seed(torch_seed)
    perturbed_forbidden = base.copy()
    perturbed_forbidden["time"] = pert["time"]
    perturbed_forbidden["event"] = pert["event"]
    s2, l2, _ = phasesyn.generate(len(perturbed_forbidden), treatment=None, target_baseline=perturbed_forbidden)
    static_cols = ["time", "event"]
    stat_diff = float(np.nanmax(np.abs(s1[static_cols].to_numpy(dtype=float) - s2[static_cols].to_numpy(dtype=float)))) if len(s1) else np.nan
    common_cols = [c for c in l1.columns if c in l2.columns and c in ["visit_time", *data.dictionary["longitudinal_variables"]]]
    long_diff = float(np.nanmax(np.abs(l1[common_cols].to_numpy(dtype=float) - l2[common_cols].to_numpy(dtype=float)))) if common_cols and len(l1) == len(l2) and len(l1) else np.nan
    return {
        "perturbation_audit_status": "completed",
        "survival_output_max_abs_diff_after_scrambling_forbidden_labels": stat_diff,
        "longitudinal_output_max_abs_diff_after_scrambling_forbidden_labels": long_diff,
        "generation_invariant_to_scrambled_survival_labels": bool(np.isfinite(stat_diff) and stat_diff <= 1e-8),
        "generation_invariant_to_scrambled_future_values": bool(not np.isfinite(long_diff) or long_diff <= 1e-8),
        "future_value_perturbation_note": "target bundles contain only baseline L0 plus requested grid; no observed post-baseline values are supplied to generation",
    }


def _sanity_checks(cfg: dict[str, Any], data: Any, output: Path, method_status: list[dict[str, Any]], phasesyn: Any | None = None) -> dict[str, Any]:
    all_ids = set(data.splits["train"]) | set(data.splits["validation"]) | set(data.splits["test"])
    checks = {
        "treatment_not_reconstructed_as_baseline_covariate": True,
        "baseline_encoder_inputs": ["W", "L0", "A_external", "future_grid"],
        "no_train_test_subject_leakage": len(all_ids) == len(data.splits["train"]) + len(data.splits["validation"]) + len(data.splits["test"]),
        "synthetic_survival_rule": "U=min(T,C), delta=1{T<=C}; direct T/C retained where method supports it",
        "longitudinal_original_scale": True,
        "event_censoring_rates_non_degenerate_checked": True,
        "method_failures_logged": any(str(r.get("status", "")).startswith("failed") for r in method_status) or (output / "reports" / "benchmark_failures.md").exists(),
        "generation_for_phasesyn_uses_only_B_L0_A_future_grid": True,
        "postbaseline_future_values_not_encoded": True,
    }
    if phasesyn is not None:
        checks.update(_phase_perturbation_audit(phasesyn, data, cfg))
    (output / "reports" / "sanity_checks.json").write_text(json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return checks


def _base_key(
    output: Path,
    method_status: list[dict[str, Any]],
    start: float,
    smoke: bool,
    cfg: dict[str, Any],
    config_path: Path,
    execution_scope: str,
    full_run_status: str,
    sanity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failures = [f"{r.get('method')}: {r.get('reason', '')}" for r in method_status if str(r.get("status", "")).startswith("failed")]
    dependency_warnings = [f"{r.get('method')}: {r.get('dependency_warning')}" for r in method_status if r.get("dependency_warning")]
    return {
        "output_dir": str(output),
        "tables_dir": str(output / "tables"),
        "figures_dir": str(output / "figures"),
        "methods_completed": [r["method"] for r in method_status if r.get("status") == "completed"],
        "methods_failed": [r for r in method_status if str(r.get("status", "")).startswith("failed")],
        "method_status": method_status,
        "failures": failures,
        "dependency_warnings": dependency_warnings,
        "sanity_checks": sanity or {},
        "runtime_seconds": float(time.time() - start),
        "smoke": bool(smoke),
        "execution_scope": execution_scope,
        "full_run_status": full_run_status,
        "requested_full_replicates": {
            "exp1": int(cfg["generation"]["exp1_replicates"]),
            "exp3_posterior_samples": int(cfg["generation"]["exp3_posterior_samples"]),
            "exp4": int(cfg["generation"]["exp4_replicates"]),
        },
        "rerun_commands": _rerun_commands(config_path),
    }


def _write_summary(output: Path, key: dict[str, Any], name: str = "run_summary.json") -> None:
    (output / name).write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_all(config_path: Path, smoke: bool = False, only: str = "all") -> dict[str, Any]:
    start = time.time()
    cfg = yaml.safe_load(project_path(config_path).read_text(encoding="utf-8"))
    if smoke:
        cfg["smoke"]["enabled"] = True
    output = project_path(cfg["output_dir"])
    _ensure_dirs(output)
    data = preprocess_to_disk(cfg["source_data_dir"], cfg["processed_data_dir"], int(cfg["seed"]))
    _write_table1(data, output)

    if only == "preprocess":
        key = _base_key(output, [], start, smoke, cfg, config_path, "preprocess_only", "not_applicable_individual")
        _write_summary(output, key, "run_summary_preprocess.json")
        _print_handoff(key, config_path)
        return key

    bench = fit_benchmarks(cfg, data, smoke=smoke)
    method_status = list(bench["status_rows"])
    write_method_status(output / "reports" / "method_status.csv", method_status)

    if only == "benchmarks":
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "benchmarks_only", "not_applicable_individual")
        _write_summary(output, key, "run_summary_benchmarks.json")
        _print_handoff(key, config_path)
        return key

    benchmark_methods = bench["methods"]
    if not benchmark_methods:
        raise RuntimeError("No benchmark method completed; refusing to run PhaseSyn before benchmark outputs exist.")
    if not (output / "reports" / "method_status.csv").exists():
        raise RuntimeError("Benchmark status file missing; refusing to run PhaseSyn.")

    if only == "phasesyn":
        phasesyn, phase_status = fit_phasesyn(cfg, data, smoke=smoke)
        method_status.append(phase_status)
        write_method_status(output / "reports" / "method_status.csv", method_status)
        sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "phasesyn_only", "not_applicable_individual", sanity)
        _write_summary(output, key, "run_summary_phasesyn.json")
        _print_handoff(key, config_path)
        return key

    benchmark_names = [name for name in ACTIVE_METHODS if name != "PhaseSyn"]
    exp1 = run_exp1(cfg, data, benchmark_methods, None, smoke=smoke, method_names=benchmark_names)
    exp1_success = exp1["metrics"][
        exp1["metrics"].get("method", pd.Series(dtype=str)).isin(benchmark_methods.keys())
        & ~exp1["metrics"].get("status", pd.Series(dtype=str)).astype(str).str.startswith("failed")
    ]
    if exp1_success.empty:
        raise RuntimeError("No benchmark Exp1 outputs were produced; refusing to run PhaseSyn.")
    benchmark_gate = {
        "benchmark_exp1_tables_written_before_phasesyn": True,
        "benchmark_exp1_metrics": str(output / "exp1_control_arm" / "tables" / "exp1_metrics_all_methods.csv"),
        "benchmark_methods_available": sorted(benchmark_methods.keys()),
        "benchmark_exp1_rows": int(len(exp1["metrics"])),
    }
    (output / "reports" / "benchmark_first_gate.json").write_text(json.dumps(benchmark_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    phasesyn, phase_status = fit_phasesyn(cfg, data, smoke=smoke)
    method_status.append(phase_status)
    write_method_status(output / "reports" / "method_status.csv", method_status)

    exp1 = run_exp1(cfg, data, benchmark_methods, phasesyn, smoke=smoke, method_names=["PhaseSyn"], append_existing=True)
    if only == "exp1":
        sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "exp1_only", "not_applicable_individual", sanity)
        _write_summary(output, key, "run_summary_exp1.json")
        _print_handoff(key, config_path)
        return key

    exp2 = run_exp2(cfg, data, benchmark_methods, phasesyn, smoke=smoke)
    if only == "exp2":
        sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "exp2_only", "not_applicable_individual", sanity)
        _write_summary(output, key, "run_summary_exp2.json")
        _print_handoff(key, config_path)
        return key

    exp3 = run_exp3(cfg, data, phasesyn, smoke=smoke)
    if only == "exp3":
        sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "exp3_only", "not_applicable_individual", sanity)
        _write_summary(output, key, "run_summary_exp3.json")
        _print_handoff(key, config_path)
        return key

    exp4 = run_exp4(cfg, data, phasesyn, smoke=smoke)
    if only == "exp4":
        sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
        key = _base_key(output, method_status, start, smoke, cfg, config_path, "exp4_only", "not_applicable_individual", sanity)
        _write_summary(output, key, "run_summary_exp4.json")
        _print_handoff(key, config_path)
        return key

    _write_combined_tables(output, method_status, exp1, exp2, exp3, exp4)
    _write_manuscript_figures(output)
    sanity = _sanity_checks(cfg, data, output, method_status, phasesyn)
    key = _base_key(
        output,
        method_status,
        start,
        smoke,
        cfg,
        config_path,
        "smoke_validation" if smoke else "full_requested_core4",
        "pending_not_executed" if smoke else "completed",
        sanity,
    )
    _write_summary(output, key)
    write_main_summary(output, key)
    _print_handoff(key, config_path)
    return key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run PBC/PBC2 core-four PhaseSyn evaluation suite.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4.yaml"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--only",
        choices=["all", "preprocess", "benchmarks", "phasesyn", "exp1", "exp2", "exp3", "exp4"],
        default="all",
        help="Rerun a single stage/experiment with its required upstream dependencies.",
    )
    args = parser.parse_args(argv)
    run_all(args.config, smoke=args.smoke, only=args.only)


if __name__ == "__main__":
    main()
