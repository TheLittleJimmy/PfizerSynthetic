from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import config_for_overfit, load_config, model_output_dir
from .data import load_pdc2_bundle, select_overfit_indices, subset_bundle
from .training import train_model


def overfit_config_path(setting: str, survival: str = "dynamic") -> Path:
    del survival
    return Path("configs") / f"pdc2_overfit_{setting}.yaml"


def baseline_event_rate_diff(cfg: dict[str, Any]) -> float | None:
    return None


def run_overfit_suite(
    dataset: str = "pdc2",
    subset_size: int = 32,
    seed: int = 1,
    survival: str = "dynamic",
    config_path: str | Path | None = None,
    settings: list[str] | None = None,
    max_visits: int | None = None,
    device: str | None = None,
    epochs: int | None = None,
    write_summary: bool = True,
) -> dict[str, Any]:
    survival = "dynamic"
    base_config = Path(config_path) if config_path is not None else Path("configs") / "pdc2.yaml"
    overrides: dict[str, Any] = {"dataset": {"name": dataset}, "model": {"survival": survival}}
    if max_visits is not None:
        overrides["dataset"]["max_visits"] = int(max_visits)
    if device is not None:
        overrides["training"] = {"device": device}
    if epochs is not None:
        overrides.setdefault("training", {})["epochs"] = int(epochs)
    cfg = load_config(base_config, overrides)
    cfg["overfit"]["subset_size"] = int(subset_size)
    cfg["overfit"]["seed"] = int(seed)
    suite_dir = model_output_dir(cfg) / "overfit"
    suite_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    settings = settings or cfg.get("overfit", {}).get("settings", ["small", "medium", "large"])
    base_event = baseline_event_rate_diff(cfg)
    for setting in settings:
        setting_cfg_path = overfit_config_path(setting, survival=survival)
        setting_overrides: dict[str, Any] = {
            "dataset": {"name": dataset},
            "model": {"survival": survival},
        }
        if max_visits is not None:
            setting_overrides["dataset"]["max_visits"] = int(max_visits)
        if device is not None:
            setting_overrides["training"] = {"device": device}
        if epochs is not None:
            setting_overrides.setdefault("training", {})["epochs"] = int(epochs)
        setting_cfg = load_config(setting_cfg_path if setting_cfg_path.exists() else base_config, setting_overrides)
        setting_cfg["overfit"]["subset_size"] = int(subset_size)
        setting_cfg["overfit"]["seed"] = int(seed)
        setting_cfg = config_for_overfit(setting_cfg, setting)
        full_bundle = load_pdc2_bundle(setting_cfg)
        indices = select_overfit_indices(full_bundle, subset_size=subset_size, seed=seed)
        bundle = subset_bundle(full_bundle, indices)
        result = train_model(
            bundle,
            setting_cfg,
            output_dir=model_output_dir(setting_cfg, overfit_name=setting),
            overfit_name=setting,
            baseline_event_rate_diff=base_event,
        )
        ids = [int(x) for x in bundle.ids_df.iloc[:, 0].tolist()] if not bundle.ids_df.empty else [int(x) for x in indices.tolist()]
        with open(Path(result["output_dir"]) / "overfit_indices.json", "w", encoding="utf-8") as f:
            json.dump({"indices": [int(x) for x in indices.tolist()], "patient_ids": ids}, f, indent=2)
        gate = result["gate"] or {}
        rows.append({
            "setting": setting,
            "passed": bool(gate.get("passed", False)),
            "loss_decrease_ratio": gate.get("loss_decrease_ratio", 0.0),
            "continuous_rmse_ratio": result["metrics"].get("continuous_rmse_ratio", 0.0),
            "static_paired_continuous_rmse_ratio": result["metrics"].get("static_paired_continuous_rmse_ratio", 0.0),
            "static_paired_categorical_accuracy": result["metrics"].get("static_paired_categorical_accuracy", 0.0),
            "survival_time_rmse_ratio": result["metrics"].get("survival_time_rmse_ratio", 0.0),
            "survival_event_accuracy": result["metrics"].get("survival_event_accuracy", 0.0),
            "survival_km_integrated_abs_error": result["metrics"].get("survival_km_integrated_abs_error", 0.0),
            "event_rate_diff": result["metrics"].get("event_rate_diff", 0.0),
            "output_dir": str(result["output_dir"]),
        })

    summary = pd.DataFrame(rows)
    passed = bool(summary["passed"].all()) if not summary.empty else False
    if write_summary:
        summary.to_csv(suite_dir / "summary.csv", index=False)
        with open(suite_dir / "summary.md", "w", encoding="utf-8") as f:
            f.write("# PDC2 Overfit Summary\n\n")
            f.write(f"Dataset: `{dataset}`  \n")
            f.write(f"Survival: `{survival}`  \n")
            f.write(f"Subset size: `{subset_size}` seed `{seed}`  \n\n")
            try:
                f.write(summary.to_markdown(index=False))
            except Exception:
                f.write(summary.to_string(index=False))
            f.write("\n\n")
            f.write("Gate: **passed**\n" if passed else "Gate: **failed**\n")
    return {"passed": passed, "summary": summary, "suite_dir": suite_dir, "config": cfg}
