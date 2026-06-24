from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .config import load_config, model_output_dir
from .data import load_pdc2_bundle
from .overfit import run_overfit_suite
from .training import train_model


def run_dynamic_migration(dataset: str = "pdc2", subset_size: int = 32, seed: int = 1) -> Path:
    cfg = load_config("configs/pdc2.yaml", {"dataset": {"name": dataset}})
    report_path = Path(cfg["dataset"]["output_root"]) / "reports" / "migration_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "# PDC2 PhaseSyn Migration Report",
        "",
        f"Started: {datetime.now().isoformat(timespec='seconds')}",
        f"Dataset: `{dataset}`",
        "Survival: `dynamic`",
        "",
        "## Ultimate Model",
    ]
    suite = run_overfit_suite(dataset=dataset, subset_size=subset_size, seed=seed, survival="dynamic")
    report_lines.append(f"Overfit summary: `{suite['suite_dir'] / 'summary.md'}`")
    report_lines.append(f"Overfit gate: {'passed' if suite['passed'] else 'failed'}")
    if suite["passed"]:
        bundle = load_pdc2_bundle(cfg)
        result = train_model(bundle, cfg, output_dir=model_output_dir(cfg))
        report_lines.append(f"Full output: `{result['output_dir']}`")
        report_lines.append(f"Metrics: `{result['output_dir'] / 'metrics.json'}`")
    else:
        report_lines.append("Full training skipped because the overfit gate failed.")
    report_lines.append("")
    report_lines.append(f"Finished: {datetime.now().isoformat(timespec='seconds')}")
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report_path


def main(argv: list[str] | None = None) -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Run canonical dynamic-survival PDC2 migration")
    parser.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    parser.add_argument("--subset-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    report = run_dynamic_migration(args.dataset, args.subset_size, args.seed)
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
