from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .evaluate_pos import evaluate_pos
from .generate_phase2 import generate_phase2_datasets
from .io_utils import DEFAULT_CONFIG_PATH, apply_smoke_overrides, dump_json, load_config, phasesyn_worker_devices, setup_logging
from .make_figures import make_figures
from .run_oracle_pos import run_oracle_pos
from .run_virtual_phase3 import run_virtual_phase3
from .train_phasesyn import train_all_phasesyn


def run_pipeline(config_path: str | Path | None = None, smoke: bool = False, stages: list[str] | None = None) -> dict:
    cfg = load_config(config_path)
    if smoke:
        cfg = apply_smoke_overrides(cfg)
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(output)
    logger.info("starting simulation_pos pipeline output=%s smoke=%s", output, smoke)
    with open(output / "config.resolved.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: v for k, v in cfg.items() if k != "_config_path"}, f, sort_keys=False)

    requested = set(stages or ["oracle", "phase2", "train", "virtual", "evaluate", "figures"])
    phase2_manifest = None
    if "oracle" in requested:
        run_oracle_pos(cfg, output, logger=logger)
    if "phase2" in requested:
        phase2_manifest = generate_phase2_datasets(cfg, output, logger=logger)
    if "train" in requested:
        if phase2_manifest is None:
            import pandas as pd

            phase2_manifest = pd.read_csv(output / "phase2_dataset_manifest.csv", keep_default_na=False)
        train_all_phasesyn(cfg, output, phase2_manifest, logger=logger)
    if "virtual" in requested:
        if phase2_manifest is None:
            import pandas as pd

            phase2_manifest = pd.read_csv(output / "phase2_dataset_manifest.csv", keep_default_na=False)
        run_virtual_phase3(cfg, output, phase2_manifest, logger=logger)
    if "evaluate" in requested:
        evaluate_pos(cfg, output)
    figure_paths = []
    if "figures" in requested:
        figure_paths = make_figures(cfg, output)
    manifest = {
        "output_dir": str(output),
        "smoke": bool(smoke),
        "stages": sorted(requested),
        "oracle_true_pos": str(output / "oracle_true_pos.csv"),
        "method_pos_estimates": str(output / "method_pos_estimates.csv"),
        "phasesyn_pos_estimates": str(output / "phasesyn_pos_estimates.csv"),
        "benchmark_pos_estimates": str(output / "benchmark_pos_estimates.csv"),
        "figures": figure_paths,
        "methods": cfg.get("methods", {}).get("active", []),
        "n_baseline_covariates": int(cfg["n_baseline_covariates"]),
        "phasesyn_worker_devices": phasesyn_worker_devices(cfg),
    }
    dump_json(output / "run_manifest.json", manifest)
    logger.info("finished simulation_pos pipeline")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the PhaseSyn simulation PoS experiment.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        action="append",
        choices=["oracle", "phase2", "train", "virtual", "evaluate", "figures"],
        help="Run selected stage. Repeat to run multiple stages. Default runs all stages.",
    )
    args = parser.parse_args(argv)
    manifest = run_pipeline(args.config, smoke=args.smoke, stages=args.stage)
    print(manifest["output_dir"])


if __name__ == "__main__":
    main()
