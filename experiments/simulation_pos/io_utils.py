from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PHASESYN_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PHASESYN_ROOT
    / "outputs"
    / "pbc_experiments"
    / "experiment_20260618"
    / "simulation_pos"
)
DEFAULT_CONFIG_PATH = DEFAULT_OUTPUT_DIR / "config.yaml"


def resolve_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = (PHASESYN_ROOT / p).resolve()
    return p


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = DEFAULT_CONFIG_PATH if path is None else resolve_path(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["_config_path"] = str(cfg_path)
    cfg["output_dir"] = str(resolve_path(cfg.get("output_dir", DEFAULT_OUTPUT_DIR)))
    if "effect_scenarios" in cfg:
        cfg["effect_scenarios"] = {
            ("null" if key is None else str(key)): value
            for key, value in cfg["effect_scenarios"].items()
        }
    smoke = cfg.get("smoke")
    if isinstance(smoke, dict) and "effect_scenarios" in smoke:
        smoke["effect_scenarios"] = [
            "null" if item is None else str(item)
            for item in smoke["effect_scenarios"]
        ]
    _resolve_training_device(cfg)
    return cfg


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(make_jsonable(payload), f, indent=2, sort_keys=True)


def make_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    return value


def write_csv(path: str | Path, frame: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def setup_logging(output_dir: str | Path, name: str = "simulation_pos") -> logging.Logger:
    output = Path(output_dir)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(output / "logs" / f"{name}.log", mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def apply_smoke_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    smoke = cfg.get("smoke", {})
    if not smoke:
        return cfg
    out = dict(cfg)
    out["output_dir"] = str(resolve_path(smoke.get("output_dir", Path(cfg["output_dir"]) / "smoke_test")))
    for key in [
        "n_phase2",
        "n_phase2_replicates",
        "m_oracle",
        "m_syn",
    ]:
        if key in smoke:
            out[key] = smoke[key]
    if "n_phase3_grid" in smoke:
        out["n_phase3_grid"] = list(smoke["n_phase3_grid"])
    if "effect_scenarios" in smoke:
        allowed = set(smoke["effect_scenarios"])
        out["effect_scenarios"] = {
            k: v for k, v in cfg["effect_scenarios"].items() if k in allowed
        }
    if "phasesyn_training" in smoke:
        train = dict(cfg.get("phasesyn_training", {}))
        train.update(smoke["phasesyn_training"])
        out["phasesyn_training"] = train
    if "benchmark_training" in smoke:
        bench = dict(cfg.get("benchmark_training", {}))
        bench.update(smoke["benchmark_training"])
        out["benchmark_training"] = bench
    _resolve_training_device(out)
    return out


def _resolve_training_device(cfg: dict[str, Any]) -> None:
    train = cfg.setdefault("phasesyn_training", {})
    requested = str(train.get("device", "cpu")).lower()
    if requested != "auto":
        return
    try:
        import torch

        train["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        train["device"] = "cpu"


def available_cuda_devices() -> list[str]:
    try:
        import torch

        count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        count = 0
    return [f"cuda:{idx}" for idx in range(count)]


def phasesyn_worker_devices(cfg: dict[str, Any]) -> list[str]:
    train = cfg.get("phasesyn_training", {})
    device = str(train.get("device", "cpu")).lower()
    if not device.startswith("cuda"):
        return [str(train.get("device", "cpu"))]

    available = available_cuda_devices()
    if not available:
        return ["cpu"]

    requested = train.get("gpu_ids", "all")
    if isinstance(requested, str) and requested.lower() in {"all", "auto"}:
        selected = available
    else:
        if isinstance(requested, str):
            parts = [part.strip() for part in requested.split(",") if part.strip()]
        else:
            parts = [str(part) for part in requested]
        selected = []
        for part in parts:
            dev = part if part.startswith("cuda:") else f"cuda:{part}"
            if dev in available:
                selected.append(dev)
        if not selected:
            selected = available

    workers = train.get("parallel_gpu_workers", "auto")
    if isinstance(workers, str) and workers.lower() == "auto":
        max_workers = len(selected)
    else:
        max_workers = max(1, min(int(workers), len(selected)))
    return selected[:max_workers]


def config_with_training_device(cfg: dict[str, Any], device: str) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    out.setdefault("phasesyn_training", {})["device"] = str(device)
    return out


def scenario_seed(cfg: dict[str, Any], scenario: str, *parts: int) -> int:
    scenario_names = list(cfg["effect_scenarios"].keys())
    idx = scenario_names.index(scenario) if scenario in scenario_names else abs(hash(scenario)) % 1000
    seed = int(cfg["random_seed"]) + 100000 * idx
    for j, part in enumerate(parts):
        seed += int(part) * (1009 ** (j + 1))
    return int(seed % (2**31 - 1))
