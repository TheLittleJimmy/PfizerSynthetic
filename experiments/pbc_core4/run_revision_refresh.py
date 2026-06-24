from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

from .load_pbc import load_processed, project_path
from .methods import PhaseSynGenerator, split_static_long
from .run_benchmarks import fit_benchmarks
from .run_exp2_matched_controls import run_exp2
from .run_exp3_digital_twin import run_exp3
from .run_exp4_virtual_trial import run_exp4
from .run_revision import run_revision


def _load_phasesyn_from_checkpoint(cfg: dict[str, Any], data: Any, checkpoint: Path | None, smoke: bool) -> tuple[PhaseSynGenerator, dict[str, Any]]:
    train_static, train_long = split_static_long(data, "train")
    generator = PhaseSynGenerator(cfg, train_static, train_long, project_path(cfg["output_dir"]), int(cfg["seed"]))
    status = generator.load_checkpoint(checkpoint=checkpoint, smoke=smoke)
    return generator, status


def run_revision_refresh(config_path: Path, stages: list[str], checkpoint: Path | None = None, smoke: bool = False) -> dict[str, Any]:
    start = time.time()
    cfg = yaml.safe_load(project_path(config_path).read_text(encoding="utf-8"))
    if smoke:
        cfg.setdefault("smoke", {})["enabled"] = True
    output = project_path(cfg["output_dir"])
    data = load_processed(cfg["processed_data_dir"], int(cfg["seed"]))
    phasesyn, phase_status = _load_phasesyn_from_checkpoint(cfg, data, project_path(checkpoint) if checkpoint else None, smoke)
    refreshed: dict[str, str] = {}
    if "exp2" in stages:
        bench = fit_benchmarks(cfg, data, smoke=smoke)
        result = run_exp2(cfg, data, bench["methods"], phasesyn, smoke=smoke)
        refreshed["exp2_rows"] = json.dumps({key: len(value) for key, value in result.items()}, sort_keys=True)
    if "exp3" in stages:
        result = run_exp3(cfg, data, phasesyn, smoke=smoke)
        refreshed["exp3_rows"] = json.dumps({key: len(value) for key, value in result.items()}, sort_keys=True)
    if "exp4" in stages:
        result = run_exp4(cfg, data, phasesyn, smoke=smoke)
        refreshed["exp4_rows"] = json.dumps({key: len(value) for key, value in result.items()}, sort_keys=True)
    revision_summary = run_revision(config_path, preserve_existing_experiments=True)
    key = {
        "output_dir": str(output),
        "checkpoint_status": phase_status,
        "refreshed": refreshed,
        "revision_summary": revision_summary,
        "runtime_seconds": float(time.time() - start),
        "smoke": bool(smoke),
    }
    name = "run_revision_refresh_smoke.json" if smoke else "run_revision_refresh.json"
    (output / name).write_text(json.dumps(key, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(key, indent=2, sort_keys=True))
    return key


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Refresh selected PBC revision experiments from a trained PhaseSyn checkpoint.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4_revision.yaml"))
    parser.add_argument("--stage", action="append", choices=["exp2", "exp3", "exp4"], required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    run_revision_refresh(args.config, args.stage, checkpoint=args.checkpoint, smoke=args.smoke)


if __name__ == "__main__":
    main()
