#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(1, str(ROOT))

from pdc2.config import config_for_overfit, load_config  # noqa: E402
from pdc2.data import load_pdc2_bundle, subset_bundle  # noqa: E402
from pdc2.overfit import baseline_event_rate_diff  # noqa: E402
from pdc2.training import train_model  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the PhaseSyn large overfit model on the full PDC2 cohort.")
    parser.add_argument("--dataset", default="pdc2", choices=["pdc2", "pbc2"])
    parser.add_argument("--survival", default="dynamic", choices=["dynamic"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)

    suffix = ""
    cfg = load_config(ROOT / "configs" / "pdc2_overfit_large.yaml", {
        "dataset": {"name": args.dataset},
        "model": {"survival": args.survival},
        "training": {"device": args.device, "seed": args.seed},
    })
    full_bundle = load_pdc2_bundle(cfg)
    indices = np.arange(len(full_bundle.raw_df), dtype=int)
    cfg["overfit"]["subset_size"] = len(indices)
    cfg["overfit"]["seed"] = args.seed
    cfg = config_for_overfit(cfg, "large")

    bundle = subset_bundle(full_bundle, indices)
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs/pdc2/model/overfit_full_cohort/large")
    result = train_model(
        bundle,
        cfg,
        output_dir=output_dir,
        overfit_name="large",
        baseline_event_rate_diff=baseline_event_rate_diff(cfg),
    )

    ids = [int(x) for x in bundle.ids_df.iloc[:, 0].tolist()] if not bundle.ids_df.empty else [int(x) for x in indices.tolist()]
    with open(output_dir / "overfit_indices.json", "w", encoding="utf-8") as f:
        json.dump({"indices": [int(x) for x in indices.tolist()], "patient_ids": ids, "full_cohort": True}, f, indent=2)

    gate = result.get("gate") or {}
    metrics = result["metrics"]
    summary = pd.DataFrame([{
        "setting": "large",
        "passed": bool(gate.get("passed", False)),
        "loss_decrease_ratio": gate.get("loss_decrease_ratio", 0.0),
        "continuous_rmse_ratio": metrics.get("continuous_rmse_ratio", 0.0),
        "static_paired_continuous_rmse_ratio": metrics.get("static_paired_continuous_rmse_ratio", 0.0),
        "static_paired_categorical_accuracy": metrics.get("static_paired_categorical_accuracy", 0.0),
        "survival_time_rmse_ratio": metrics.get("survival_time_rmse_ratio", 0.0),
        "survival_event_accuracy": metrics.get("survival_event_accuracy", 0.0),
        "survival_km_integrated_abs_error": metrics.get("survival_km_integrated_abs_error", 0.0),
        "event_rate_diff": metrics.get("event_rate_diff", 0.0),
        "output_dir": str(output_dir),
    }])
    summary_path = output_dir.parent / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    with open(output_dir.parent / "summary.md", "w", encoding="utf-8") as f:
        f.write("# PDC2 PhaseSyn Full-Cohort Overfit Summary\n\n")
        f.write(f"Dataset: `{args.dataset}`  \n")
        f.write(f"Survival: `{args.survival}`  \n")
        f.write(f"Subset size: `{len(indices)}` seed `{args.seed}`  \n\n")
        try:
            f.write(summary.to_markdown(index=False))
        except Exception:
            f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write("Gate: **passed**\n" if bool(summary["passed"].all()) else "Gate: **failed**\n")

    print(f"wrote {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
