from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .dgm import TrialData, generate_dgm_parameters, load_trial_npz, simulate_trial
from .io_utils import (
    DEFAULT_CONFIG_PATH,
    apply_smoke_overrides,
    config_with_training_device,
    dump_json,
    load_config,
    phasesyn_worker_devices,
    scenario_seed,
    setup_logging,
    write_csv,
)
from .phasesyn_adapter import generate_phasesyn_trial
from .run_virtual_phase3 import (
    _completed_keys,
    _dedupe_pos,
    _phase2_records,
    _read_existing_csv,
    _split_generated_batch,
    _stack_trial_batch,
    _target_trial_batch,
)
from .train_benchmarks import fit_all_benchmarks
from .train_phasesyn import load_or_train_phasesyn


warnings.filterwarnings("ignore", message="Deterministic behavior was enabled.*")
warnings.filterwarnings("ignore", message="cumsum_cuda_kernel.*")


DEFAULT_ENDPOINT = {
    "biomarker": "L26",
    "biomarker_index": 25,
    "visit_index": -1,
    "favorable_direction": "lower",
    "alpha": 0.05,
    "go_threshold": 0.80,
}


def _endpoint_from_config(
    cfg: dict[str, Any],
    biomarker: str | None = None,
    visit_index: int | None = None,
    favorable_direction: str | None = None,
) -> dict[str, Any]:
    endpoint = dict(DEFAULT_ENDPOINT)
    endpoint.update(cfg.get("longitudinal_endpoint", {}) or {})
    if biomarker is not None:
        endpoint["biomarker"] = str(biomarker)
        endpoint["biomarker_index"] = int(str(biomarker).upper().replace("L", "")) - 1
    if visit_index is not None:
        endpoint["visit_index"] = int(visit_index)
    if favorable_direction is not None:
        endpoint["favorable_direction"] = str(favorable_direction).lower()

    if "biomarker_index" not in endpoint or endpoint["biomarker_index"] is None:
        endpoint["biomarker_index"] = int(str(endpoint["biomarker"]).upper().replace("L", "")) - 1
    endpoint["biomarker_index"] = int(endpoint["biomarker_index"])
    endpoint["biomarker"] = f"L{endpoint['biomarker_index'] + 1:02d}"
    if endpoint["biomarker_index"] < 0 or endpoint["biomarker_index"] >= int(cfg["n_longitudinal_biomarkers"]):
        raise ValueError(f"Endpoint biomarker index out of range: {endpoint['biomarker_index']}")

    visit = int(endpoint["visit_index"])
    if visit < 0:
        visit = int(cfg["n_timepoints"]) + visit
    if visit < 0 or visit >= int(cfg["n_timepoints"]):
        raise ValueError(f"Endpoint visit_index out of range: {endpoint['visit_index']}")
    endpoint["visit_index_resolved"] = visit
    endpoint["visit_time"] = float(cfg["time_grid"][visit])
    endpoint["favorable_direction"] = str(endpoint["favorable_direction"]).lower()
    if endpoint["favorable_direction"] not in {"lower", "higher"}:
        raise ValueError("favorable_direction must be 'lower' or 'higher'")
    endpoint["alpha"] = float(endpoint.get("alpha", 0.05))
    endpoint["go_threshold"] = float(endpoint.get("go_threshold", cfg.get("go_threshold", 0.80)))
    endpoint.setdefault("output_subdir", f"longitudinal_endpoint_{endpoint['biomarker']}_final_visit")
    return endpoint


def analyze_longitudinal_endpoint(data: dict[str, np.ndarray] | TrialData, endpoint: dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, TrialData):
        l_values = data.L
        treatment = data.A
    else:
        l_values = data["L"]
        treatment = data["A"]
    visit = int(endpoint["visit_index_resolved"])
    biomarker = int(endpoint["biomarker_index"])
    y = np.asarray(l_values[:, visit, biomarker], dtype=float)
    a = np.asarray(treatment, dtype=int)
    treated = y[a == 1]
    control = y[a == 0]
    n_treatment = int(treated.size)
    n_control = int(control.size)
    mean_treatment = float(np.mean(treated)) if n_treatment else float("nan")
    mean_control = float(np.mean(control)) if n_control else float("nan")
    mean_diff = mean_treatment - mean_control
    var_treatment = float(np.var(treated, ddof=1)) if n_treatment > 1 else 0.0
    var_control = float(np.var(control, ddof=1)) if n_control > 1 else 0.0
    se_diff = math.sqrt(var_treatment / max(n_treatment, 1) + var_control / max(n_control, 1))
    if se_diff > 0.0 and np.isfinite(mean_diff):
        z_stat = mean_diff / se_diff
        p_value = math.erfc(abs(z_stat) / math.sqrt(2.0))
    else:
        z_stat = 0.0
        p_value = 1.0
    favorable = mean_diff < 0.0 if endpoint["favorable_direction"] == "lower" else mean_diff > 0.0
    success = bool(favorable and p_value < float(endpoint["alpha"]))
    return {
        "success": int(success),
        "endpoint_mean": float(np.mean(y)),
        "endpoint_sd": float(np.std(y, ddof=1)) if y.size > 1 else 0.0,
        "control_mean": mean_control,
        "treatment_mean": mean_treatment,
        "mean_diff": float(mean_diff),
        "se_diff": float(se_diff),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "n_control": n_control,
        "n_treatment": n_treatment,
    }


def summarize_endpoint_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(analyses)
    return {
        "pos": float(frame["success"].mean()),
        "endpoint_mean": float(frame["endpoint_mean"].mean()),
        "endpoint_sd": float(frame["endpoint_sd"].mean()),
        "control_mean": float(frame["control_mean"].mean()),
        "treatment_mean": float(frame["treatment_mean"].mean()),
        "mean_diff": float(frame["mean_diff"].mean()),
        "sd_diff": float(frame["mean_diff"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "mean_p_value": float(frame["p_value"].mean()),
        "success_count": int(frame["success"].sum()),
        "n_trials": int(len(frame)),
    }


def _nan_rmse(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.sqrt(np.mean(np.square(vals)))) if vals.size else float("nan")


def _nan_mae(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    return float(np.mean(np.abs(vals))) if vals.size else float("nan")


def _endpoint_run_metadata(cfg: dict[str, Any], endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        "m_syn": int(cfg["m_syn"]),
        "n_phase3_grid": [int(n) for n in cfg["n_phase3_grid"]],
        "effect_scenarios": list(cfg["effect_scenarios"].keys()),
        "random_seed": int(cfg["random_seed"]),
        "n_baseline_covariates": int(cfg["n_baseline_covariates"]),
        "n_longitudinal_biomarkers": int(cfg["n_longitudinal_biomarkers"]),
        "n_timepoints": int(cfg["n_timepoints"]),
        "endpoint": endpoint,
        "methods": list(cfg.get("methods", {}).get("active", [])),
        "benchmark_methods": list(cfg.get("methods", {}).get("benchmarks", [])),
        "phase2_replicates": int(cfg["n_phase2_replicates"]),
        "phasesyn_longitudinal_values": "sampled_decoder_distribution_unmasked_scheduled_visit",
    }


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _existing_virtual_outputs_compatible(output: Path, expected_meta: dict[str, Any]) -> bool:
    meta = _load_json_if_exists(output / "intermediate" / "longitudinal_virtual_pos_config.json")
    if meta is None:
        existing = _read_existing_csv(output / "longitudinal_method_pos_estimates.csv")
        if existing.empty:
            return True
        expected_rows = (
            len(expected_meta["methods"])
            * len(expected_meta["effect_scenarios"])
            * int(expected_meta["phase2_replicates"])
            * len(expected_meta["n_phase3_grid"])
        )
        required = {"method", "scenario", "replicate", "n_phase3", "pos_hat", "mean_diff_hat"}
        if len(existing) != expected_rows or not required.issubset(existing.columns):
            return False
        existing_methods = set(existing["method"].astype(str))
        existing_scenarios = set(existing["scenario"].astype(str))
        existing_n = {int(n) for n in existing["n_phase3"]}
        if (
            existing_methods == set(expected_meta["methods"])
            and existing_scenarios == set(expected_meta["effect_scenarios"])
            and existing_n == set(expected_meta["n_phase3_grid"])
        ):
            dump_json(output / "intermediate" / "longitudinal_virtual_pos_config.json", expected_meta)
            return True
        return False
    return meta == expected_meta


def run_oracle_longitudinal_endpoint_pos(
    cfg: dict[str, Any],
    output_dir: str | Path,
    endpoint: dict[str, Any],
    logger=None,
) -> pd.DataFrame:
    output = Path(output_dir)
    oracle_path = output / "longitudinal_oracle_true_pos.csv"
    meta_path = output / "intermediate" / "longitudinal_oracle_true_pos_config.json"
    expected_meta = {
        "m_oracle": int(cfg["m_oracle"]),
        "n_phase3_grid": [int(n) for n in cfg["n_phase3_grid"]],
        "effect_scenarios": list(cfg["effect_scenarios"].keys()),
        "random_seed": int(cfg["random_seed"]),
        "n_baseline_covariates": int(cfg["n_baseline_covariates"]),
        "n_longitudinal_biomarkers": int(cfg["n_longitudinal_biomarkers"]),
        "n_timepoints": int(cfg["n_timepoints"]),
        "endpoint": endpoint,
    }
    expected_rows = len(cfg["effect_scenarios"]) * len(cfg["n_phase3_grid"])
    if oracle_path.exists() and meta_path.exists():
        import json

        existing = pd.read_csv(oracle_path, keep_default_na=False)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_ok = json.load(f) == expected_meta
        if len(existing) == expected_rows and meta_ok:
            if logger:
                logger.info("reuse existing longitudinal oracle PoS table %s", oracle_path)
            return existing

    params = generate_dgm_parameters(
        int(cfg["random_seed"]),
        n_baseline=int(cfg["n_baseline_covariates"]),
        n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
    )
    rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    for scenario_name, scenario in cfg["effect_scenarios"].items():
        for n_phase3 in cfg["n_phase3_grid"]:
            analyses: list[dict[str, Any]] = []
            if logger:
                logger.info(
                    "longitudinal oracle endpoint=%s visit=%s scenario=%s n=%s m=%s",
                    endpoint["biomarker"],
                    endpoint["visit_index_resolved"],
                    scenario_name,
                    n_phase3,
                    cfg["m_oracle"],
                )
            for m in range(int(cfg["m_oracle"])):
                trial = simulate_trial(
                    int(n_phase3),
                    scenario,
                    params,
                    seed=scenario_seed(cfg, scenario_name, int(n_phase3), m, 1701),
                    n_timepoints=int(cfg["n_timepoints"]),
                    n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
                    time_grid=cfg["time_grid"],
                    missing_rate_target=float(cfg["missing_rate_target"]),
                )
                analysis = analyze_longitudinal_endpoint(trial, endpoint)
                analyses.append(analysis)
                trial_rows.append({
                    "source": "oracle",
                    "scenario": scenario_name,
                    "n_phase3": int(n_phase3),
                    "trial": int(m),
                    **analysis,
                })
            summary = summarize_endpoint_analyses(analyses)
            rows.append({
                "scenario": scenario_name,
                "n_phase3": int(n_phase3),
                "true_pos": summary["pos"],
                "true_endpoint_mean": summary["endpoint_mean"],
                "true_endpoint_sd": summary["endpoint_sd"],
                "true_control_mean": summary["control_mean"],
                "true_treatment_mean": summary["treatment_mean"],
                "true_mean_diff": summary["mean_diff"],
                "true_sd_diff": summary["sd_diff"],
                "true_mean_p_value": summary["mean_p_value"],
                "true_success_count": summary["success_count"],
                "oracle_trials": summary["n_trials"],
            })

    oracle = pd.DataFrame(rows)
    write_csv(oracle_path, oracle)
    write_csv(output / "intermediate" / "longitudinal_oracle_trial_analyses.csv", pd.DataFrame(trial_rows))
    dump_json(meta_path, expected_meta)
    return oracle


def _summarize_method_endpoint_trials(
    method: str,
    scenario_name: str,
    replicate: int,
    n_phase3: int,
    analyses: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_endpoint_analyses(analyses)
    return {
        "method": method,
        "scenario": scenario_name,
        "replicate": int(replicate),
        "n_phase3": int(n_phase3),
        "pos_hat": summary["pos"],
        "endpoint_mean_hat": summary["endpoint_mean"],
        "endpoint_sd_hat": summary["endpoint_sd"],
        "control_mean_hat": summary["control_mean"],
        "treatment_mean_hat": summary["treatment_mean"],
        "mean_diff_hat": summary["mean_diff"],
        "sd_diff_hat": summary["sd_diff"],
        "mean_p_value_hat": summary["mean_p_value"],
        "success_count": summary["success_count"],
        "n_virtual_trials": summary["n_trials"],
    }


def _run_endpoint_design_methods(
    method_generators: dict[str, Any],
    cfg: dict[str, Any],
    endpoint: dict[str, Any],
    scenario_name: str,
    replicate: int,
    n_phase3: int,
    dgm_params,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    analyses_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in method_generators}
    trials_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in method_generators}
    m_syn = int(cfg["m_syn"])
    batch_size = max(1, int(cfg.get("virtual_batch_size", 10)))

    for batch_start in range(0, m_syn, batch_size):
        batch_end = min(m_syn, batch_start + batch_size)
        trial_ids = list(range(batch_start, batch_end))
        targets = _target_trial_batch(cfg, scenario_name, replicate, int(n_phase3), dgm_params, trial_ids)
        target_batch = _stack_trial_batch(targets)
        split_sizes = [len(target.A) for target in targets]

        for method, generator in method_generators.items():
            if method == "PhaseSyn":
                seed = scenario_seed(cfg, scenario_name, replicate, int(n_phase3), batch_start, 2301)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            generated_batch = generator(target_batch)
            generated_trials = _split_generated_batch(generated_batch, split_sizes)
            for m, generated in zip(trial_ids, generated_trials):
                analysis = analyze_longitudinal_endpoint(generated, endpoint)
                analyses_by_method[method].append(analysis)
                trials_by_method[method].append({
                    "method": method,
                    "scenario": scenario_name,
                    "replicate": int(replicate),
                    "n_phase3": int(n_phase3),
                    "trial": int(m),
                    **analysis,
                })

    pos_rows = [
        _summarize_method_endpoint_trials(method, scenario_name, replicate, int(n_phase3), analyses_by_method[method])
        for method in method_generators
    ]
    trial_rows = [row for method in method_generators for row in trials_by_method[method]]
    return pos_rows, trial_rows


def _run_endpoint_records_worker(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    endpoint: dict[str, Any],
    endpoint_output_dir: str,
    model_output_dir: str,
    completed_keys: set[tuple[str, str, int, int]],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    warnings.filterwarnings("ignore", message="Deterministic behavior was enabled.*")
    warnings.filterwarnings("ignore", message="cumsum_cuda_kernel.*")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    endpoint_output = Path(endpoint_output_dir)
    model_output = Path(model_output_dir)
    worker_cfg = config_with_training_device(cfg, device)
    dgm_params = generate_dgm_parameters(
        int(worker_cfg["random_seed"]),
        n_baseline=int(worker_cfg["n_baseline_covariates"]),
        n_biomarkers=int(worker_cfg["n_longitudinal_biomarkers"]),
    )
    completed = set(completed_keys)
    pos_rows: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    benchmark_rows: list[dict[str, Any]] = []
    phasesyn_rows: list[dict[str, Any]] = []
    model_path_rows: list[dict[str, Any]] = []
    benchmark_methods = list(worker_cfg.get("methods", {}).get("benchmarks", []))
    active_methods = ["PhaseSyn", *benchmark_methods]
    n_values = [int(n) for n in worker_cfg["n_phase3_grid"]]

    for phase2 in records:
        scenario = str(phase2["scenario"])
        rep = int(phase2["replicate"])
        rep_expected = {
            (method, scenario, rep, n_phase3)
            for method in active_methods
            for n_phase3 in n_values
        }
        if rep_expected.issubset(completed):
            continue

        train_trial = load_trial_npz(phase2["path"])
        phase_needed = any(("PhaseSyn", scenario, rep, n_phase3) not in completed for n_phase3 in n_values)
        phase_result = None
        if phase_needed:
            phase_result = load_or_train_phasesyn(worker_cfg, model_output, scenario, rep, train_trial, logger=None)
            model_path_rows.append({
                "method": "PhaseSyn",
                "scenario": scenario,
                "replicate": rep,
                "model_dir": phase_result["output_dir"],
                "checkpoint": phase_result["checkpoint"],
                "model_artifact": phase_result["checkpoint"],
                "status": "completed",
            })

        needed_benchmarks = [
            method for method in benchmark_methods
            if any((method, scenario, rep, n_phase3) not in completed for n_phase3 in n_values)
        ]
        benchmark_dir = model_output / "models" / scenario / f"benchmarks_rep_{rep:03d}"
        benchmark_models = {}
        if needed_benchmarks:
            bench_cfg = dict(worker_cfg)
            bench_cfg["methods"] = dict(worker_cfg.get("methods", {}))
            bench_cfg["methods"]["benchmarks"] = needed_benchmarks
            benchmark_models = fit_all_benchmarks(
                train_trial,
                seed=scenario_seed(worker_cfg, scenario, rep, 809),
                cfg=bench_cfg,
                output_dir=benchmark_dir,
            )
            for method, model in benchmark_models.items():
                model_path_rows.append({
                    "method": method,
                    "scenario": scenario,
                    "replicate": rep,
                    "model_dir": str(benchmark_dir),
                    "checkpoint": getattr(model, "model_artifact", ""),
                    "model_artifact": getattr(model, "model_artifact", ""),
                    "status": "completed",
                })

        for n_phase3 in n_values:
            generators: dict[str, Any] = {}
            phase_key = ("PhaseSyn", scenario, rep, int(n_phase3))
            if phase_key not in completed:
                if phase_result is None:
                    phase_result = load_or_train_phasesyn(worker_cfg, model_output, scenario, rep, train_trial, logger=None)
                generators["PhaseSyn"] = lambda target, result=phase_result, worker_device=device: generate_phasesyn_trial(
                    result["model"],
                    result["bundle"],
                    target,
                    device=worker_device,
                    deterministic_latents=False,
                    mask_longitudinal_by_observed_time=False,
                    sample_longitudinal_values=True,
                )

            for method, model in benchmark_models.items():
                method_key = (method, scenario, rep, int(n_phase3))
                if method_key in completed:
                    continue
                generators[method] = model.generate_trial

            if generators:
                design_pos_rows, design_trial_rows = _run_endpoint_design_methods(
                    generators,
                    worker_cfg,
                    endpoint,
                    scenario,
                    rep,
                    int(n_phase3),
                    dgm_params,
                )
                for row in design_pos_rows:
                    method = str(row["method"])
                    method_key = (method, scenario, rep, int(n_phase3))
                    if method == "PhaseSyn":
                        row["model_path"] = phase_result["checkpoint"]
                        phasesyn_rows.append({k: v for k, v in row.items() if k != "method"})
                    else:
                        benchmark_rows.append(row)
                    pos_rows.append(row)
                    completed.add(method_key)
                trial_rows.extend(design_trial_rows)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"longitudinal endpoint worker device={device} output={endpoint_output.name} "
            f"completed scenario={scenario} replicate={rep} pos_rows={len(pos_rows)} trial_rows={len(trial_rows)}",
            flush=True,
        )

    return {
        "pos": pos_rows,
        "trials": trial_rows,
        "benchmark": benchmark_rows,
        "phasesyn": phasesyn_rows,
        "model_paths": model_path_rows,
    }


def _run_endpoint_records_parallel(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    endpoint: dict[str, Any],
    endpoint_output: Path,
    model_output: Path,
    completed: set[tuple[str, str, int, int]],
    logger=None,
) -> dict[str, list[dict[str, Any]]]:
    empty = {"pos": [], "trials": [], "benchmark": [], "phasesyn": [], "model_paths": []}
    if not records:
        return empty
    devices = phasesyn_worker_devices(cfg)
    if len(devices) <= 1 or len(records) <= 1:
        device = devices[0]
        if logger:
            logger.info("run longitudinal endpoint work sequentially on device=%s tasks=%s", device, len(records))
        return _run_endpoint_records_worker(records, cfg, endpoint, str(endpoint_output), str(model_output), completed, device)

    n_workers = min(len(devices), len(records))
    shards = [[] for _ in range(n_workers)]
    for idx, record in enumerate(records):
        shards[idx % n_workers].append(record)
    jobs = [
        (shard, cfg, endpoint, str(endpoint_output), str(model_output), completed, devices[i])
        for i, shard in enumerate(shards)
        if shard
    ]
    if logger:
        logger.info(
            "run longitudinal endpoint work with %s GPU workers devices=%s tasks=%s",
            len(jobs),
            [job[6] for job in jobs],
            len(records),
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        chunks = pool.starmap(_run_endpoint_records_worker, jobs)
    merged = {key: [] for key in empty}
    for chunk in chunks:
        for key, values in chunk.items():
            merged[key].extend(values)
    return merged


def run_virtual_longitudinal_endpoint_pos(
    cfg: dict[str, Any],
    endpoint_output_dir: str | Path,
    model_output_dir: str | Path,
    phase2_manifest: pd.DataFrame,
    endpoint: dict[str, Any],
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    endpoint_output = Path(endpoint_output_dir)
    model_output = Path(model_output_dir)
    expected_meta = _endpoint_run_metadata(cfg, endpoint)
    if not _existing_virtual_outputs_compatible(endpoint_output, expected_meta):
        raise RuntimeError(
            "Existing longitudinal endpoint virtual outputs do not match the requested endpoint/config. "
            "Use a different --output-subdir or clear the existing endpoint virtual outputs before rerunning."
        )
    existing_pos = _read_existing_csv(endpoint_output / "longitudinal_method_pos_estimates.csv")
    existing_benchmark = _read_existing_csv(endpoint_output / "longitudinal_benchmark_pos_estimates.csv")
    existing_phasesyn = _read_existing_csv(endpoint_output / "longitudinal_phasesyn_pos_estimates.csv")
    existing_trials = _read_existing_csv(endpoint_output / "intermediate" / "longitudinal_virtual_trial_analyses.csv")
    existing_model_paths = _read_existing_csv(endpoint_output / "longitudinal_model_paths.csv")
    completed = _completed_keys(existing_pos)
    pos_rows = existing_pos.to_dict("records") if not existing_pos.empty else []
    trial_rows = existing_trials.to_dict("records") if not existing_trials.empty else []
    benchmark_rows = existing_benchmark.to_dict("records") if not existing_benchmark.empty else []
    phasesyn_rows = existing_phasesyn.to_dict("records") if not existing_phasesyn.empty else []
    model_path_rows = existing_model_paths.to_dict("records") if not existing_model_paths.empty else []
    benchmark_methods = list(cfg.get("methods", {}).get("benchmarks", []))
    active_methods = ["PhaseSyn", *benchmark_methods]
    n_values = [int(n) for n in cfg["n_phase3_grid"]]
    records: list[dict[str, Any]] = []
    for phase2 in _phase2_records(phase2_manifest):
        scenario = str(phase2["scenario"])
        rep = int(phase2["replicate"])
        rep_expected = {
            (method, scenario, rep, n_phase3)
            for method in active_methods
            for n_phase3 in n_values
        }
        if rep_expected.issubset(completed):
            if logger:
                logger.info("skip completed longitudinal endpoint replicate scenario=%s replicate=%s", scenario, rep)
            continue
        if logger:
            logger.info("queue longitudinal endpoint replicate scenario=%s replicate=%s", scenario, rep)
        records.append(phase2)

    generated = _run_endpoint_records_parallel(records, cfg, endpoint, endpoint_output, model_output, completed, logger=logger)
    pos_rows.extend(generated["pos"])
    trial_rows.extend(generated["trials"])
    benchmark_rows.extend(generated["benchmark"])
    phasesyn_rows.extend(generated["phasesyn"])
    model_path_rows.extend(generated["model_paths"])
    all_pos = _dedupe_pos(pd.DataFrame(pos_rows))
    benchmark_pos = _dedupe_pos(pd.DataFrame(benchmark_rows))
    phasesyn_pos = (
        pd.DataFrame(phasesyn_rows)
        .sort_values(["scenario", "replicate", "n_phase3"])
        .drop_duplicates(["scenario", "replicate", "n_phase3"], keep="last")
        .reset_index(drop=True)
        if phasesyn_rows else pd.DataFrame()
    )
    model_paths = (
        pd.DataFrame(model_path_rows)
        .sort_values(["method", "scenario", "replicate"])
        .drop_duplicates(["method", "scenario", "replicate"], keep="last")
        .reset_index(drop=True)
        if model_path_rows else pd.DataFrame()
    )
    write_csv(endpoint_output / "longitudinal_method_pos_estimates.csv", all_pos)
    write_csv(endpoint_output / "longitudinal_benchmark_pos_estimates.csv", benchmark_pos)
    write_csv(endpoint_output / "longitudinal_phasesyn_pos_estimates.csv", phasesyn_pos)
    write_csv(endpoint_output / "longitudinal_model_paths.csv", model_paths)
    write_csv(endpoint_output / "intermediate" / "longitudinal_virtual_trial_analyses.csv", pd.DataFrame(trial_rows))
    dump_json(endpoint_output / "intermediate" / "longitudinal_virtual_pos_config.json", expected_meta)
    return phasesyn_pos, benchmark_pos


def evaluate_longitudinal_endpoint_pos(
    cfg: dict[str, Any],
    output_dir: str | Path,
    endpoint: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    output = Path(output_dir)
    oracle = pd.read_csv(output / "longitudinal_oracle_true_pos.csv", keep_default_na=False)
    estimates = pd.read_csv(output / "longitudinal_method_pos_estimates.csv", keep_default_na=False)
    merged = estimates.merge(oracle, on=["scenario", "n_phase3"], how="left")
    numeric_columns = [
        "replicate",
        "n_phase3",
        "pos_hat",
        "endpoint_mean_hat",
        "endpoint_sd_hat",
        "control_mean_hat",
        "treatment_mean_hat",
        "mean_diff_hat",
        "sd_diff_hat",
        "mean_p_value_hat",
        "success_count",
        "n_virtual_trials",
        "true_pos",
        "true_endpoint_mean",
        "true_endpoint_sd",
        "true_control_mean",
        "true_treatment_mean",
        "true_mean_diff",
        "true_sd_diff",
        "true_mean_p_value",
        "true_success_count",
        "oracle_trials",
    ]
    for column in numeric_columns:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    merged["pos_error"] = merged["pos_hat"] - merged["true_pos"]
    merged["endpoint_mean_error_abs"] = (merged["endpoint_mean_hat"] - merged["true_endpoint_mean"]).abs()
    merged["control_mean_error_abs"] = (merged["control_mean_hat"] - merged["true_control_mean"]).abs()
    merged["treatment_mean_error_abs"] = (merged["treatment_mean_hat"] - merged["true_treatment_mean"]).abs()
    merged["mean_diff_error"] = merged["mean_diff_hat"] - merged["true_mean_diff"]
    merged["mean_diff_error_abs"] = merged["mean_diff_error"].abs()
    write_csv(output / "tables" / "longitudinal_pos_estimates_with_oracle.csv", merged)

    acc = (
        merged.groupby(["method", "scenario", "n_phase3"], dropna=False)
        .agg(
            pos_bias=("pos_error", "mean"),
            pos_rmse=("pos_error", _nan_rmse),
            pos_mae=("pos_error", _nan_mae),
            endpoint_mean_error=("endpoint_mean_error_abs", "mean"),
            control_mean_error=("control_mean_error_abs", "mean"),
            treatment_mean_error=("treatment_mean_error_abs", "mean"),
            mean_diff_bias=("mean_diff_error", "mean"),
            mean_diff_rmse=("mean_diff_error", _nan_rmse),
            mean_diff_mae=("mean_diff_error_abs", "mean"),
            mean_pos_hat=("pos_hat", "mean"),
            true_pos=("true_pos", "first"),
            true_mean_diff=("true_mean_diff", "first"),
        )
        .reset_index()
    )
    write_csv(output / "longitudinal_pos_accuracy_table.csv", acc)
    write_csv(output / "tables" / "longitudinal_pos_bias_rmse_mae.csv", acc)

    null = merged[merged["scenario"].eq("null")].copy()
    null_metrics = (
        null.groupby("method", dropna=False)
        .agg(
            null_bias=("pos_error", "mean"),
            null_rmse=("pos_error", _nan_rmse),
            false_positive_pos_rate=("pos_hat", lambda x: float(np.mean(np.asarray(x) > 0.20))),
            null_mean_diff_bias=("mean_diff_error", "mean"),
        )
        .reset_index()
        if len(null)
        else pd.DataFrame(columns=["method", "null_bias", "null_rmse", "false_positive_pos_rate", "null_mean_diff_bias"])
    )
    write_csv(output / "longitudinal_null_calibration_metrics.csv", null_metrics)
    write_csv(output / "tables" / "longitudinal_null_calibration_metrics.csv", null_metrics)

    decision = merged.copy()
    threshold = float(endpoint.get("go_threshold", cfg.get("go_threshold", 0.8)))
    decision["model_go"] = decision["pos_hat"] >= threshold
    decision["oracle_go"] = decision["true_pos"] >= threshold
    decision["correct"] = decision["model_go"].eq(decision["oracle_go"])
    decision_metrics = (
        decision.groupby("method", dropna=False)
        .agg(
            decision_accuracy=("correct", "mean"),
            false_go_rate=("model_go", lambda x: float(np.mean(np.asarray(x) & ~decision.loc[x.index, "oracle_go"].to_numpy()))),
            false_stop_rate=("model_go", lambda x: float(np.mean((~np.asarray(x)) & decision.loc[x.index, "oracle_go"].to_numpy()))),
        )
        .reset_index()
    )
    write_csv(output / "longitudinal_go_no_go_decision_metrics.csv", decision_metrics)
    write_csv(output / "tables" / "longitudinal_go_no_go_decision_metrics.csv", decision_metrics)

    rank_rows = []
    for (method, scenario, replicate), g in merged.groupby(["method", "scenario", "replicate"], dropna=False):
        if g["n_phase3"].nunique() < 2:
            continue
        pred_order = g.sort_values("n_phase3")["pos_hat"].to_numpy()
        oracle_order = g.sort_values("n_phase3")["true_pos"].to_numpy()
        oracle_prefers_larger = bool(oracle_order[-1] >= oracle_order[0])
        model_prefers_larger = bool(pred_order[-1] >= pred_order[0])
        rank_rows.append({
            "method": method,
            "scenario": scenario,
            "replicate": replicate,
            "ranking_correct": model_prefers_larger == oracle_prefers_larger,
        })
    ranking = pd.DataFrame(rank_rows)
    ranking_metrics = (
        ranking.groupby("method", dropna=False)
        .agg(ranking_accuracy=("ranking_correct", "mean"))
        .reset_index()
        if len(ranking)
        else pd.DataFrame(columns=["method", "ranking_accuracy"])
    )
    write_csv(output / "longitudinal_design_ranking_accuracy.csv", ranking_metrics)
    write_csv(output / "tables" / "longitudinal_design_ranking_accuracy.csv", ranking_metrics)

    effect = (
        merged.groupby("method", dropna=False)
        .agg(
            endpoint_mean_error=("endpoint_mean_error_abs", "mean"),
            treatment_mean_error=("treatment_mean_error_abs", "mean"),
            control_mean_error=("control_mean_error_abs", "mean"),
            mean_diff_mae=("mean_diff_error_abs", "mean"),
        )
        .reset_index()
    )
    write_csv(output / "longitudinal_endpoint_mean_error_table.csv", effect)
    write_csv(output / "tables" / "longitudinal_endpoint_mean_errors.csv", effect)

    utility = (
        acc.groupby("method", dropna=False)
        .agg(overall_pos_rmse=("pos_rmse", "mean"), overall_mean_diff_rmse=("mean_diff_rmse", "mean"))
        .reset_index()
        .merge(null_metrics[["method", "null_bias", "null_mean_diff_bias"]], on="method", how="left")
        .merge(decision_metrics[["method", "false_go_rate", "false_stop_rate"]], on="method", how="left")
        .merge(ranking_metrics, on="method", how="left")
    )
    write_csv(output / "longitudinal_decision_utility_table.csv", utility)
    write_csv(output / "tables" / "longitudinal_decision_utility.csv", utility)
    return {
        "merged": merged,
        "accuracy": acc,
        "null": null_metrics,
        "decision": decision_metrics,
        "ranking": ranking_metrics,
        "effect": effect,
        "utility": utility,
    }


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _git_metadata(root: Path, output: Path | None = None) -> dict[str, Any]:
    def run_git(args: list[str]) -> str:
        try:
            return subprocess.check_output(["git", *args], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    commit = run_git(["rev-parse", "HEAD"])
    dirty_files = run_git(["status", "--short"])
    diff_shortstat = run_git(["diff", "--shortstat"])
    patch_path = ""
    if output is not None:
        diff_text = run_git(["diff", "--binary"])
        if diff_text:
            patch_file = output / "longitudinal_endpoint_git_diff.patch"
            patch_file.write_text(diff_text, encoding="utf-8")
            patch_path = str(patch_file)
    return {
        "commit": commit,
        "dirty": bool(dirty_files),
        "dirty_file_count": len([line for line in dirty_files.splitlines() if line.strip()]),
        "status_short": dirty_files,
        "diff_shortstat": diff_shortstat,
        "diff_patch_path": patch_path,
    }


def _validation_summary(output: Path) -> dict[str, Any]:
    paths = {
        "oracle": output / "longitudinal_oracle_true_pos.csv",
        "method": output / "longitudinal_method_pos_estimates.csv",
        "benchmark": output / "longitudinal_benchmark_pos_estimates.csv",
        "phasesyn": output / "longitudinal_phasesyn_pos_estimates.csv",
        "virtual_trials": output / "intermediate" / "longitudinal_virtual_trial_analyses.csv",
        "merged": output / "tables" / "longitudinal_pos_estimates_with_oracle.csv",
        "accuracy": output / "tables" / "longitudinal_pos_bias_rmse_mae.csv",
    }
    summary: dict[str, Any] = {}
    for name, path in paths.items():
        if not path.exists():
            summary[name] = {"exists": False}
            continue
        frame = pd.read_csv(path, keep_default_na=False)
        numeric = frame.apply(pd.to_numeric, errors="ignore").select_dtypes(include="number")
        summary[name] = {
            "exists": True,
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "numeric_missing": int(numeric.isna().sum().sum()) if not numeric.empty else 0,
        }
    return summary


def make_longitudinal_endpoint_figures(
    cfg: dict[str, Any],
    output_dir: str | Path,
    endpoint: dict[str, Any],
) -> list[str]:
    output = Path(output_dir)
    figures = output / "figures"
    paths: list[str] = []
    merged = pd.read_csv(output / "tables" / "longitudinal_pos_estimates_with_oracle.csv", keep_default_na=False)
    acc = pd.read_csv(output / "tables" / "longitudinal_pos_bias_rmse_mae.csv", keep_default_na=False)
    merged_numeric = [
        "replicate",
        "n_phase3",
        "pos_hat",
        "endpoint_mean_hat",
        "control_mean_hat",
        "treatment_mean_hat",
        "mean_diff_hat",
        "true_pos",
        "true_endpoint_mean",
        "true_control_mean",
        "true_treatment_mean",
        "true_mean_diff",
        "pos_error",
        "mean_diff_error",
    ]
    acc_numeric = ["n_phase3", "pos_rmse", "mean_diff_rmse", "true_pos", "true_mean_diff"]
    for column in merged_numeric:
        if column in merged.columns:
            merged[column] = pd.to_numeric(merged[column], errors="coerce")
    for column in acc_numeric:
        if column in acc.columns:
            acc[column] = pd.to_numeric(acc[column], errors="coerce")
    method_order = [m for m in ["PhaseSyn", "LMM-AFT", "JM-RE", "TVAE", "CTGAN"] if m in set(merged["method"])]
    palette = {
        "PhaseSyn": "#D55E00",
        "LMM-AFT": "#0072B2",
        "JM-RE": "#009E73",
        "TVAE": "#CC79A7",
        "CTGAN": "#56B4E9",
    }
    endpoint_label = f"{endpoint['biomarker']} at visit {endpoint['visit_index_resolved']} (time={endpoint['visit_time']:.2f})"

    cal = (
        merged.groupby(["method", "scenario", "n_phase3"], dropna=False)
        .agg(
            pos_hat=("pos_hat", "mean"),
            true_pos=("true_pos", "first"),
            mean_diff_hat=("mean_diff_hat", "mean"),
            true_mean_diff=("true_mean_diff", "first"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    for method, g in cal.groupby("method"):
        ax.scatter(g["true_pos"], g["pos_hat"], label=method, s=38, alpha=0.85, color=palette.get(method))
    ax.plot([0, 1], [0, 1], color="black", lw=1.0, linestyle="--")
    ax.set_xlabel("Oracle endpoint PoS")
    ax.set_ylabel("Estimated endpoint PoS")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(endpoint_label)
    ax.legend(fontsize=8, ncol=2)
    path = figures / "long_fig1_endpoint_pos_calibration.png"
    _save(fig, path)
    paths.append(str(path))

    fig, axes = plt.subplots(1, max(1, len(cfg["effect_scenarios"])), figsize=(5 * max(1, len(cfg["effect_scenarios"])), 4), squeeze=False)
    for ax, scenario in zip(axes.ravel(), cfg["effect_scenarios"].keys()):
        g = cal[cal["scenario"].eq(scenario)]
        oracle = g.drop_duplicates(["n_phase3"]).sort_values("n_phase3")
        ax.plot(oracle["n_phase3"], oracle["true_pos"], color="black", marker="o", label="Oracle")
        for method, gm in g.groupby("method"):
            gm = gm.sort_values("n_phase3")
            ax.plot(gm["n_phase3"], gm["pos_hat"], marker="o", alpha=0.85, label=method, color=palette.get(method))
        ax.set_title(scenario)
        ax.set_xlabel("Phase III sample size")
        ax.set_ylabel("Endpoint PoS")
        ax.set_ylim(-0.03, 1.03)
    axes.ravel()[0].legend(fontsize=8, ncol=2)
    path = figures / "long_fig2_endpoint_power_curves.png"
    _save(fig, path)
    paths.append(str(path))

    rmse = acc.groupby("method", dropna=False).agg(pos_rmse=("pos_rmse", "mean")).reset_index()
    rmse["method"] = pd.Categorical(rmse["method"], categories=method_order, ordered=True)
    rmse = rmse.sort_values("method")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(rmse["method"].astype(str), rmse["pos_rmse"], color=[palette.get(m, "#999999") for m in rmse["method"].astype(str)])
    ax.set_ylabel("Mean endpoint PoS RMSE")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=25)
    path = figures / "long_fig3_endpoint_pos_rmse.png"
    _save(fig, path)
    paths.append(str(path))

    phase = merged[merged["method"].eq("PhaseSyn")].copy()
    phase["design"] = phase["scenario"].astype(str) + "\nn=" + phase["n_phase3"].astype(str)
    design_order = [
        f"{scenario}\nn={int(n_phase3)}"
        for scenario in cfg["effect_scenarios"].keys()
        for n_phase3 in cfg["n_phase3_grid"]
    ]
    design_order = [design for design in design_order if design in set(phase["design"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    phase_data = [phase.loc[phase["design"].eq(design), "pos_hat"].to_numpy(dtype=float) for design in design_order]
    bp = ax.boxplot(phase_data, labels=design_order, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor(palette["PhaseSyn"])
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
    rng = np.random.default_rng(20260619)
    for i, design in enumerate(design_order, start=1):
        vals = phase.loc[phase["design"].eq(design), "pos_hat"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.scatter(rng.normal(i, 0.04, size=vals.size), vals, s=12, color=palette["PhaseSyn"], alpha=0.35, linewidths=0)
        truth_vals = phase.loc[phase["design"].eq(design), "true_pos"].to_numpy(dtype=float)
        truth_vals = truth_vals[np.isfinite(truth_vals)]
        if truth_vals.size:
            ax.scatter(i, float(truth_vals[0]), marker="D", s=44, color="black", zorder=5, label="Oracle endpoint PoS" if i == 1 else None)
    ax.set_ylabel("Endpoint PoS")
    ax.set_xlabel("Scenario and Phase III sample size")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8, frameon=False)
    path = figures / "long_fig4_phasesyn_endpoint_design_calibration.png"
    _save(fig, path)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 5))
    effect_cal = cal[np.isfinite(cal["true_mean_diff"]) & np.isfinite(cal["mean_diff_hat"])].copy()
    for method, g in effect_cal.groupby("method"):
        ax.scatter(g["true_mean_diff"], g["mean_diff_hat"], label=method, s=38, alpha=0.85, color=palette.get(method))
    if len(effect_cal):
        lo = min(effect_cal["true_mean_diff"].min(), effect_cal["mean_diff_hat"].min()) - 0.08
        hi = max(effect_cal["true_mean_diff"].max(), effect_cal["mean_diff_hat"].max()) + 0.08
    else:
        lo, hi = -1.0, 1.0
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, linestyle="--")
    ax.axvline(0.0, color="#666666", lw=0.8, linestyle=":")
    ax.axhline(0.0, color="#666666", lw=0.8, linestyle=":")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Oracle treatment-control mean difference")
    ax.set_ylabel("Estimated treatment-control mean difference")
    ax.set_title(f"{endpoint_label}; lower is favorable")
    ax.legend(fontsize=8, ncol=2)
    path = figures / "long_fig5_endpoint_effect_calibration.png"
    _save(fig, path)
    paths.append(str(path))

    merged["abs_pos_error"] = merged["pos_error"].abs()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    box_data = [merged.loc[merged["method"].eq(method), "abs_pos_error"].to_numpy(dtype=float) for method in method_order]
    bp = ax.boxplot(box_data, labels=method_order, patch_artist=True, showfliers=False)
    for patch, method in zip(bp["boxes"], method_order):
        patch.set_facecolor(palette.get(method, "#999999"))
        patch.set_alpha(0.45 if method != "PhaseSyn" else 0.70)
        patch.set_edgecolor("#333333")
    for i, vals in enumerate(box_data, start=1):
        vals = vals[np.isfinite(vals)]
        if vals.size:
            sample = vals if vals.size <= 220 else rng.choice(vals, size=220, replace=False)
            ax.scatter(rng.normal(i, 0.045, size=sample.size), sample, s=8, color="#222222", alpha=0.22, linewidths=0)
    ax.set_ylabel("Absolute endpoint PoS error")
    ax.set_xlabel("Method")
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(bottom=0.0)
    path = figures / "long_fig6_endpoint_pos_error_distribution.png"
    _save(fig, path)
    paths.append(str(path))
    return paths


def run_longitudinal_endpoint_pipeline(
    config_path: str | Path | None = None,
    smoke: bool = False,
    stages: list[str] | None = None,
    biomarker: str | None = None,
    visit_index: int | None = None,
    favorable_direction: str | None = None,
    output_subdir: str | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    endpoint = _endpoint_from_config(cfg, biomarker=biomarker, visit_index=visit_index, favorable_direction=favorable_direction)
    model_output = Path(cfg["output_dir"])
    if output_subdir:
        endpoint["output_subdir"] = output_subdir
    output_candidate = Path(str(endpoint["output_subdir"]))
    if output_candidate.is_absolute():
        output = output_candidate
    elif output_candidate.parts and output_candidate.parts[0] == model_output.name:
        output = model_output.parent / output_candidate
    else:
        output = model_output / output_candidate
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output, name="longitudinal_endpoint_pos")
    logger.info("starting longitudinal endpoint PoS pipeline output=%s smoke=%s endpoint=%s", output, smoke, endpoint)
    with open(output / "longitudinal_endpoint_config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {
                **{k: v for k, v in cfg.items() if k != "_config_path"},
                "longitudinal_endpoint": endpoint,
                "model_output_dir": str(model_output),
            },
            f,
            sort_keys=False,
        )

    requested = set(stages or ["oracle", "virtual", "evaluate", "figures"])
    if "oracle" in requested:
        run_oracle_longitudinal_endpoint_pos(cfg, output, endpoint, logger=logger)
    if "virtual" in requested:
        phase2_manifest_path = model_output / "phase2_dataset_manifest.csv"
        if not phase2_manifest_path.exists():
            raise FileNotFoundError(f"Missing Phase II manifest: {phase2_manifest_path}")
        phase2_manifest = pd.read_csv(phase2_manifest_path, keep_default_na=False)
        run_virtual_longitudinal_endpoint_pos(cfg, output, model_output, phase2_manifest, endpoint, logger=logger)
    if "evaluate" in requested:
        evaluate_longitudinal_endpoint_pos(cfg, output, endpoint)
    figure_paths: list[str] = []
    if "figures" in requested:
        figure_paths = make_longitudinal_endpoint_figures(cfg, output, endpoint)
    manifest = {
        "output_dir": str(output),
        "model_output_dir": str(model_output),
        "smoke": bool(smoke),
        "stages": sorted(requested),
        "command": " ".join([sys.executable, "-m", "experiments.simulation_pos.run_longitudinal_endpoint_pos", *sys.argv[1:]]),
        "config_path": str(config_path or DEFAULT_CONFIG_PATH),
        "environment": "env_2502",
        "git": _git_metadata(Path(__file__).resolve().parents[2], output),
        "endpoint": endpoint,
        "oracle_true_pos": str(output / "longitudinal_oracle_true_pos.csv"),
        "method_pos_estimates": str(output / "longitudinal_method_pos_estimates.csv"),
        "phasesyn_pos_estimates": str(output / "longitudinal_phasesyn_pos_estimates.csv"),
        "benchmark_pos_estimates": str(output / "longitudinal_benchmark_pos_estimates.csv"),
        "figures": figure_paths,
        "methods": cfg.get("methods", {}).get("active", []),
        "n_baseline_covariates": int(cfg["n_baseline_covariates"]),
        "phasesyn_worker_devices": phasesyn_worker_devices(cfg),
        "validation_summary": _validation_summary(output),
    }
    dump_json(output / "longitudinal_endpoint_run_manifest.json", manifest)
    logger.info("finished longitudinal endpoint PoS pipeline")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run endpoint-based PoS analysis for one longitudinal biomarker.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--biomarker", default=None, help="Primary longitudinal endpoint, for example L26.")
    parser.add_argument("--visit-index", type=int, default=None, help="Endpoint visit index. Negative indices count from the end.")
    parser.add_argument("--favorable-direction", choices=["lower", "higher"], default=None)
    parser.add_argument("--output-subdir", default=None)
    parser.add_argument(
        "--stage",
        action="append",
        choices=["oracle", "virtual", "evaluate", "figures"],
        help="Run selected stage. Repeat to run multiple stages. Default runs all stages.",
    )
    args = parser.parse_args(argv)
    manifest = run_longitudinal_endpoint_pipeline(
        args.config,
        smoke=args.smoke,
        stages=args.stage,
        biomarker=args.biomarker,
        visit_index=args.visit_index,
        favorable_direction=args.favorable_direction,
        output_subdir=args.output_subdir,
    )
    print(manifest["output_dir"])


if __name__ == "__main__":
    main()
