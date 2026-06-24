from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .load_pbc import preprocess_to_disk, project_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preprocess local PBC2 data for the core-four suite.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4.yaml"))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    data = preprocess_to_disk(cfg["source_data_dir"], cfg["processed_data_dir"], int(cfg["seed"]))
    print(f"processed subjects: {len(data.subjects)}")
    print(f"processed longitudinal rows: {len(data.longitudinal)}")
    print(f"processed directory: {project_path(cfg['processed_data_dir'])}")


if __name__ == "__main__":
    main()
