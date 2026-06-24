from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .load_pbc import load_processed, project_path
from .methods import ACTIVE_METHODS, build_method, split_static_long, write_dependency_status


def fit_benchmarks(cfg: dict, data=None, smoke: bool = False) -> dict:
    data = data or load_processed(cfg["processed_data_dir"], int(cfg["seed"]))
    train_static, train_long = split_static_long(data, "train")
    if smoke:
        cap = int(cfg.get("smoke", {}).get("max_train_subjects", 32))
        keep = train_static.head(cap)["subject_id"].tolist()
        train_static = train_static[train_static["subject_id"].isin(keep)].reset_index(drop=True)
        train_long = train_long[train_long["subject_id"].isin(keep)].reset_index(drop=True)
    output = project_path(cfg["output_dir"])
    deps = write_dependency_status(output / "reports" / "benchmark_dependency_status.json")
    methods = {}
    status_rows = []
    use_mixedlm = bool(cfg.get("generation", {}).get("exp1_use_mixedlm", True))
    for i, name in enumerate(cfg["methods"]["active"]):
        if name == "PhaseSyn":
            continue
        if name not in ACTIVE_METHODS:
            status_rows.append({"method": name, "status": "skipped_unknown", "reason": "not in ACTIVE_METHODS"})
            continue
        try:
            model = build_method(name, train_static, train_long, int(cfg["seed"]) + i * 101, use_mixedlm=use_mixedlm)
            methods[name] = model
            warning = ""
            if name == "CTGAN" and not deps.get("ctgan", False):
                warning = "external ctgan package unavailable; used local Torch CTGAN-like conditional GAN"
            if name == "TVAE" and not deps.get("sdv", False):
                warning = "external SDV TVAE unavailable; used local Torch TVAE-style autoencoder"
            status_rows.append({"method": name, "status": "completed", "dependency_warning": warning})
        except Exception as exc:
            status_rows.append({"method": name, "status": "failed_dependency", "reason": f"{type(exc).__name__}: {exc}"})
    return {"methods": methods, "status_rows": status_rows}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fit PBC core-4 benchmark generators.")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config_pbc_core4.yaml"))
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    result = fit_benchmarks(cfg)
    print("benchmark methods:", ", ".join(result["methods"].keys()))


if __name__ == "__main__":
    main()
