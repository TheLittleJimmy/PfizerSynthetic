from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .dgm import generate_dgm_parameters, save_trial_npz, simulate_trial, trial_summary
from .io_utils import scenario_seed, write_csv
from .phasesyn_adapter import trial_longitudinal_frame, trial_static_frame


def generate_phase2_datasets(cfg: dict[str, Any], output_dir: str | Path, logger=None) -> pd.DataFrame:
    output = Path(output_dir)
    manifest_path = output / "phase2_dataset_manifest.csv"
    if manifest_path.exists():
        existing = pd.read_csv(manifest_path, keep_default_na=False)
        expected = int(cfg["n_phase2_replicates"]) * len(cfg["effect_scenarios"])
        if len(existing) == expected and existing["path"].map(lambda p: Path(p).exists()).all():
            if logger:
                logger.info("reuse existing Phase II dataset manifest %s", manifest_path)
            return existing
    params = generate_dgm_parameters(
        int(cfg["random_seed"]),
        n_baseline=int(cfg["n_baseline_covariates"]),
        n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
    )
    rows = []
    for scenario_name, scenario in cfg["effect_scenarios"].items():
        for rep in range(int(cfg["n_phase2_replicates"])):
            if logger:
                logger.info("simulate phase II scenario=%s replicate=%s", scenario_name, rep)
            trial = simulate_trial(
                int(cfg["n_phase2"]),
                scenario,
                params,
                seed=scenario_seed(cfg, scenario_name, rep, 17),
                n_timepoints=int(cfg["n_timepoints"]),
                n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
                time_grid=cfg["time_grid"],
                missing_rate_target=float(cfg["missing_rate_target"]),
            )
            path = output / "data" / "simulation_pos" / scenario_name / f"phase2_rep_{rep:03d}.npz"
            save_trial_npz(path, trial)
            bundle_dir = output / "intermediate" / "phase2_csv" / scenario_name / f"rep_{rep:03d}"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            trial_static_frame(trial).to_csv(bundle_dir / "subjects.csv", index=False)
            trial_longitudinal_frame(trial, masked=True).to_csv(bundle_dir / "longitudinal.csv", index=False)
            rows.append({
                "scenario": scenario_name,
                "replicate": int(rep),
                "path": str(path),
                "subjects_csv": str(bundle_dir / "subjects.csv"),
                "longitudinal_csv": str(bundle_dir / "longitudinal.csv"),
                **trial_summary(trial),
            })
    manifest = pd.DataFrame(rows)
    write_csv(manifest_path, manifest)
    return manifest
