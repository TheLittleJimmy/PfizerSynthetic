from __future__ import annotations

import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .dgm import TrialData, generate_dgm_parameters, load_trial_npz, simulate_trial
from .io_utils import config_with_training_device, phasesyn_worker_devices, scenario_seed, write_csv
from .phasesyn_adapter import generate_phasesyn_trial
from .survival_analysis import analyze_trial, summarize_trial_analyses
from .train_benchmarks import fit_all_benchmarks
from .train_phasesyn import load_or_train_phasesyn


def _analyze_generated(data: dict[str, np.ndarray], admin_end: float = 1.0) -> dict[str, Any]:
    return analyze_trial(data["T_obs"], data["delta"], data["A"], admin_end=admin_end)


def _stack_trial_batch(trials: list[TrialData]) -> TrialData:
    if len(trials) == 1:
        return trials[0]
    first = trials[0]
    return TrialData(
        X=np.concatenate([trial.X for trial in trials], axis=0),
        A=np.concatenate([trial.A for trial in trials], axis=0),
        L=np.concatenate([trial.L for trial in trials], axis=0),
        R=np.concatenate([trial.R for trial in trials], axis=0),
        time_grid=first.time_grid,
        T_obs=np.concatenate([trial.T_obs for trial in trials], axis=0),
        delta=np.concatenate([trial.delta for trial in trials], axis=0),
        event_time=np.concatenate([trial.event_time for trial in trials], axis=0),
        censoring_time=np.concatenate([trial.censoring_time for trial in trials], axis=0),
        G=np.concatenate([trial.G for trial in trials], axis=0),
        Z=np.concatenate([trial.Z for trial in trials], axis=0),
        H1=np.concatenate([trial.H1 for trial in trials], axis=0),
        H2=np.concatenate([trial.H2 for trial in trials], axis=0),
    )


def _split_generated_batch(data: dict[str, np.ndarray], sizes: list[int]) -> list[dict[str, np.ndarray]]:
    chunks = []
    start = 0
    for size in sizes:
        end = start + int(size)
        chunks.append({
            "X": data["X"][start:end],
            "A": data["A"][start:end],
            "L": data["L"][start:end],
            "T_obs": data["T_obs"][start:end],
            "delta": data["delta"][start:end],
        })
        start = end
    return chunks


def _summarize_method_trials(
    method: str,
    scenario_name: str,
    replicate: int,
    n_phase3: int,
    analyses: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = summarize_trial_analyses(analyses)
    return {
        "method": method,
        "scenario": scenario_name,
        "replicate": int(replicate),
        "n_phase3": int(n_phase3),
        "pos_hat": summary["pos"],
        "event_rate_hat": summary["event_rate"],
        "censoring_rate_hat": summary["censoring_rate"],
        "mean_hr_hat": summary["mean_hr"],
        "sd_log_hr_hat": summary["sd_log_hr"],
    }


def _target_trial_batch(
    cfg: dict[str, Any],
    scenario_name: str,
    replicate: int,
    n_phase3: int,
    dgm_params,
    trial_ids: list[int],
) -> list[TrialData]:
    scenario = cfg["effect_scenarios"][scenario_name]
    return [
        simulate_trial(
            int(n_phase3),
            scenario,
            dgm_params,
            seed=scenario_seed(cfg, scenario_name, replicate, int(n_phase3), m, 701),
            n_timepoints=int(cfg["n_timepoints"]),
            n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
            time_grid=cfg["time_grid"],
            missing_rate_target=float(cfg["missing_rate_target"]),
        )
        for m in trial_ids
    ]


def _run_design_methods(
    method_generators: dict[str, Any],
    cfg: dict[str, Any],
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
                seed = scenario_seed(cfg, scenario_name, replicate, int(n_phase3), batch_start, 1301)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            generated_batch = generator(target_batch)
            generated_trials = _split_generated_batch(generated_batch, split_sizes)
            for m, target, generated in zip(trial_ids, targets, generated_trials):
                analysis = _analyze_generated(generated, admin_end=float(target.time_grid[-1]))
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
        _summarize_method_trials(method, scenario_name, replicate, int(n_phase3), analyses_by_method[method], trials_by_method[method])
        for method in method_generators
    ]
    trial_rows = [row for method in method_generators for row in trials_by_method[method]]
    return pos_rows, trial_rows


def _read_existing_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, keep_default_na=False)


def _completed_keys(frame: pd.DataFrame) -> set[tuple[str, str, int, int]]:
    if frame.empty:
        return set()
    keys = set()
    for row in frame.itertuples(index=False):
        keys.add((str(row.method), str(row.scenario), int(row.replicate), int(row.n_phase3)))
    return keys


def _dedupe_pos(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return (
        frame.sort_values(["method", "scenario", "replicate", "n_phase3"])
        .drop_duplicates(["method", "scenario", "replicate", "n_phase3"], keep="last")
        .reset_index(drop=True)
    )


def _phase2_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "scenario": str(row.scenario),
            "replicate": int(row.replicate),
            "path": str(row.path),
        }
        for row in frame.itertuples(index=False)
    ]


def _run_virtual_records_worker(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: str,
    completed_keys: set[tuple[str, str, int, int]],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(output_dir)
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
            phase_result = load_or_train_phasesyn(worker_cfg, output, scenario, rep, train_trial, logger=None)
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
        benchmark_dir = output / "models" / scenario / f"benchmarks_rep_{rep:03d}"
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
                    phase_result = load_or_train_phasesyn(worker_cfg, output, scenario, rep, train_trial, logger=None)
                gen_cfg = dict(worker_cfg.get("phasesyn_generation", {}))
                deterministic_u0_value = gen_cfg.get("deterministic_u0", None)
                deterministic_survival_value = gen_cfg.get("deterministic_survival", None)
                generators["PhaseSyn"] = lambda target, result=phase_result, worker_device=device: generate_phasesyn_trial(
                        result["model"],
                        result["bundle"],
                        target,
                        device=worker_device,
                        deterministic_latents=bool(gen_cfg.get("deterministic_latents", False)),
                        deterministic_u0=None if deterministic_u0_value is None else bool(deterministic_u0_value),
                        deterministic_survival=None if deterministic_survival_value is None else bool(deterministic_survival_value),
                        survival_event_export=str(gen_cfg.get("survival_event_export", "sample")),
                    )

            for method, model in benchmark_models.items():
                method_key = (method, scenario, rep, int(n_phase3))
                if method_key in completed:
                    continue
                generators[method] = model.generate_trial

            if generators:
                design_pos_rows, design_trial_rows = _run_design_methods(
                    generators,
                    worker_cfg,
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
            f"virtual worker device={device} completed scenario={scenario} replicate={rep} "
            f"pos_rows={len(pos_rows)} trial_rows={len(trial_rows)}",
            flush=True,
        )

    return {
        "pos": pos_rows,
        "trials": trial_rows,
        "benchmark": benchmark_rows,
        "phasesyn": phasesyn_rows,
        "model_paths": model_path_rows,
    }


def _run_virtual_records_parallel(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    output: Path,
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
            logger.info("run virtual replicate work sequentially on device=%s tasks=%s", device, len(records))
        return _run_virtual_records_worker(records, cfg, str(output), completed, device)

    n_workers = min(len(devices), len(records))
    shards = [[] for _ in range(n_workers)]
    for idx, record in enumerate(records):
        shards[idx % n_workers].append(record)
    jobs = [(shard, cfg, str(output), completed, devices[i]) for i, shard in enumerate(shards) if shard]
    if logger:
        logger.info(
            "run virtual replicate work with %s GPU workers devices=%s tasks=%s",
            len(jobs),
            [job[4] for job in jobs],
            len(records),
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        chunks = pool.starmap(_run_virtual_records_worker, jobs)
    merged = {key: [] for key in empty}
    for chunk in chunks:
        for key, values in chunk.items():
            merged[key].extend(values)
    return merged


def run_virtual_phase3(
    cfg: dict[str, Any],
    output_dir: str | Path,
    phase2_manifest: pd.DataFrame,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = Path(output_dir)
    dgm_params = generate_dgm_parameters(
        int(cfg["random_seed"]),
        n_baseline=int(cfg["n_baseline_covariates"]),
        n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
    )
    existing_pos = _read_existing_csv(output / "method_pos_estimates.csv")
    existing_benchmark = _read_existing_csv(output / "benchmark_pos_estimates.csv")
    existing_phasesyn = _read_existing_csv(output / "phasesyn_pos_estimates.csv")
    existing_trials = _read_existing_csv(output / "intermediate" / "virtual_trial_analyses.csv")
    existing_model_paths = _read_existing_csv(output / "model_paths.csv")
    completed = _completed_keys(existing_pos)
    pos_rows = existing_pos.to_dict("records") if not existing_pos.empty else []
    trial_rows = existing_trials.to_dict("records") if not existing_trials.empty else []
    benchmark_rows = existing_benchmark.to_dict("records") if not existing_benchmark.empty else []
    phasesyn_rows = existing_phasesyn.to_dict("records") if not existing_phasesyn.empty else []
    model_path_rows = existing_model_paths.to_dict("records") if not existing_model_paths.empty else []
    benchmark_methods = list(cfg.get("methods", {}).get("benchmarks", []))
    active_methods = ["PhaseSyn", *benchmark_methods]
    records = []
    for phase2 in _phase2_records(phase2_manifest):
        scenario = str(phase2["scenario"])
        rep = int(phase2["replicate"])
        n_values = [int(n) for n in cfg["n_phase3_grid"]]
        rep_expected = {
            (method, scenario, rep, n_phase3)
            for method in active_methods
            for n_phase3 in n_values
        }
        if rep_expected.issubset(completed):
            if logger:
                logger.info("skip completed virtual replicate scenario=%s replicate=%s", scenario, rep)
            continue
        if logger:
            logger.info("queue virtual replicate scenario=%s replicate=%s", scenario, rep)
        records.append(phase2)

    generated = _run_virtual_records_parallel(records, cfg, output, completed, logger=logger)
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
    write_csv(output / "method_pos_estimates.csv", all_pos)
    write_csv(output / "benchmark_pos_estimates.csv", benchmark_pos)
    write_csv(output / "phasesyn_pos_estimates.csv", phasesyn_pos)
    write_csv(output / "model_paths.csv", model_paths)
    write_csv(output / "intermediate" / "virtual_trial_analyses.csv", pd.DataFrame(trial_rows))
    return phasesyn_pos, benchmark_pos
