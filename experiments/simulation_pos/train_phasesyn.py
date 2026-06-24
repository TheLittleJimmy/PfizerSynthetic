from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
from typing import Any

import pandas as pd

from .dgm import load_trial_npz
from .io_utils import config_with_training_device, phasesyn_worker_devices, scenario_seed, write_csv
from .phasesyn_adapter import clone_config_for_manifest, train_phasesyn_model


def _phase2_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "scenario": str(row.scenario),
            "replicate": int(row.replicate),
            "path": str(row.path),
        }
        for row in frame.itertuples(index=False)
    ]


def _manifest_row(output: Path, scenario: str, rep: int) -> dict[str, Any]:
    model_dir = output / "models" / scenario / f"phasesyn_rep_{rep:03d}"
    checkpoint = model_dir / "model_checkpoint.pt"
    return {
        "method": "PhaseSyn",
        "scenario": scenario,
        "replicate": rep,
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint),
        "model_artifact": str(checkpoint),
        "status": "completed" if checkpoint.exists() else "missing",
    }


def _train_worker(records: list[dict[str, Any]], cfg: dict[str, Any], output_dir: str, device: str) -> list[dict[str, Any]]:
    output = Path(output_dir)
    worker_cfg = config_with_training_device(cfg, device)
    rows = []
    for rec in records:
        scenario = str(rec["scenario"])
        rep = int(rec["replicate"])
        model_dir = output / "models" / scenario / f"phasesyn_rep_{rep:03d}"
        checkpoint = model_dir / "model_checkpoint.pt"
        if checkpoint.exists():
            rows.append(_manifest_row(output, scenario, rep))
            continue
        trial = load_trial_npz(rec["path"])
        train_phasesyn_model(
            trial,
            worker_cfg,
            seed=scenario_seed(worker_cfg, scenario, rep, 311),
            output_dir=model_dir,
        )
        rows.append(_manifest_row(output, scenario, rep))
    return rows


def _train_records_parallel(
    records: list[dict[str, Any]],
    cfg: dict[str, Any],
    output: Path,
    logger=None,
) -> list[dict[str, Any]]:
    if not records:
        return []
    devices = phasesyn_worker_devices(cfg)
    if len(devices) <= 1 or len(records) <= 1:
        device = devices[0]
        if logger:
            logger.info("train PhaseSyn sequentially on device=%s tasks=%s", device, len(records))
        return _train_worker(records, cfg, str(output), device)

    n_workers = min(len(devices), len(records))
    shards = [[] for _ in range(n_workers)]
    for idx, rec in enumerate(records):
        shards[idx % n_workers].append(rec)
    jobs = [(shard, cfg, str(output), devices[i]) for i, shard in enumerate(shards) if shard]
    if logger:
        logger.info(
            "train PhaseSyn with %s GPU workers devices=%s tasks=%s",
            len(jobs),
            [job[3] for job in jobs],
            len(records),
        )
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(jobs)) as pool:
        nested = pool.starmap(_train_worker, jobs)
    return [row for rows in nested for row in rows]


def train_all_phasesyn(cfg: dict[str, Any], output_dir: str | Path, phase2_manifest: pd.DataFrame, logger=None) -> pd.DataFrame:
    output = Path(output_dir)
    existing_rows = []
    missing_records = []
    for rec in _phase2_records(phase2_manifest):
        scenario = str(rec["scenario"])
        rep = int(rec["replicate"])
        checkpoint = output / "models" / scenario / f"phasesyn_rep_{rep:03d}" / "model_checkpoint.pt"
        if checkpoint.exists():
            if logger:
                logger.info("reuse PhaseSyn checkpoint scenario=%s replicate=%s", scenario, rep)
            existing_rows.append(_manifest_row(output, scenario, rep))
        else:
            if logger:
                logger.info("queue PhaseSyn training scenario=%s replicate=%s", scenario, rep)
            missing_records.append(rec)
    rows = existing_rows + _train_records_parallel(missing_records, cfg, output, logger=logger)
    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["scenario", "replicate"]).reset_index(drop=True)
    write_csv(output / "model_paths.csv", manifest)
    write_csv(output / "phasesyn_model_paths.csv", manifest)
    pd.DataFrame([clone_config_for_manifest(cfg)]).to_json(output / "run_config_snapshot.json", orient="records", indent=2)
    return manifest


def load_or_train_phasesyn(
    cfg: dict[str, Any],
    output_dir: str | Path,
    scenario: str,
    replicate: int,
    trial,
    logger=None,
) -> dict[str, Any]:
    output = Path(output_dir)
    model_dir = output / "models" / scenario / f"phasesyn_rep_{replicate:03d}"
    checkpoint = model_dir / "model_checkpoint.pt"
    if checkpoint.exists():
        if logger:
            logger.info("reuse PhaseSyn checkpoint scenario=%s replicate=%s", scenario, replicate)
        from .phasesyn_adapter import build_phasesyn_config, load_trained_phasesyn, trial_to_bundle

        bundle = trial_to_bundle(trial, masked=True)
        pdc_cfg = build_phasesyn_config(cfg, scenario_seed(cfg, scenario, replicate, 311), model_dir)
        model = load_trained_phasesyn(checkpoint, bundle, pdc_cfg, device=cfg.get("phasesyn_training", {}).get("device", "cpu"))
        return {
            "model": model,
            "bundle": bundle,
            "pdc_config": pdc_cfg,
            "checkpoint": str(checkpoint),
            "output_dir": str(model_dir),
            "metrics": {},
        }
    if logger:
        logger.info("train PhaseSyn checkpoint scenario=%s replicate=%s", scenario, replicate)
    return train_phasesyn_model(
        trial,
        cfg,
        seed=scenario_seed(cfg, scenario, replicate, 311),
        output_dir=model_dir,
    )
