from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .load_pbc import load_processed, project_path
from .methods import PhaseSynGenerator, split_static_long


def fit_phasesyn(cfg: dict[str, Any], data=None, smoke: bool = False) -> tuple[PhaseSynGenerator, dict[str, Any]]:
    data = data or load_processed(cfg["processed_data_dir"], int(cfg["seed"]))
    train_static, train_long = split_static_long(data, "train")
    generator = PhaseSynGenerator(cfg, train_static, train_long, project_path(cfg["output_dir"]), int(cfg["seed"]))
    status = generator.train(smoke=smoke)
    return generator, status


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train PhaseSyn for PBC core-4.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4.yaml"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    _, status = fit_phasesyn(cfg, smoke=args.smoke)
    print(status)


if __name__ == "__main__":
    main()

