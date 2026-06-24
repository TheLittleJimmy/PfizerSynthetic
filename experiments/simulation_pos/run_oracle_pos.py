from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .dgm import generate_dgm_parameters, simulate_trial
from .io_utils import dump_json, scenario_seed, write_csv
from .survival_analysis import analyze_trial, summarize_trial_analyses


def run_oracle_pos(cfg: dict[str, Any], output_dir: str | Path, logger=None) -> pd.DataFrame:
    output = Path(output_dir)
    oracle_path = output / "oracle_true_pos.csv"
    meta_path = output / "intermediate" / "oracle_true_pos_config.json"
    expected_meta = {
        "m_oracle": int(cfg["m_oracle"]),
        "n_phase3_grid": [int(n) for n in cfg["n_phase3_grid"]],
        "effect_scenarios": list(cfg["effect_scenarios"].keys()),
        "random_seed": int(cfg["random_seed"]),
        "n_baseline_covariates": int(cfg["n_baseline_covariates"]),
        "n_longitudinal_biomarkers": int(cfg["n_longitudinal_biomarkers"]),
        "n_timepoints": int(cfg["n_timepoints"]),
    }
    if oracle_path.exists():
        existing = pd.read_csv(oracle_path, keep_default_na=False)
        expected = len(cfg["effect_scenarios"]) * len(cfg["n_phase3_grid"])
        meta_ok = False
        if meta_path.exists():
            import json

            with open(meta_path, "r", encoding="utf-8") as f:
                meta_ok = json.load(f) == expected_meta
        if len(existing) == expected and meta_ok:
            if logger:
                logger.info("reuse existing oracle PoS table %s", oracle_path)
            return existing
    params = generate_dgm_parameters(
        int(cfg["random_seed"]),
        n_baseline=int(cfg["n_baseline_covariates"]),
        n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
    )
    rows = []
    trial_rows = []
    for scenario_name, scenario in cfg["effect_scenarios"].items():
        for n_phase3 in cfg["n_phase3_grid"]:
            analyses = []
            if logger:
                logger.info("oracle PoS scenario=%s n=%s m=%s", scenario_name, n_phase3, cfg["m_oracle"])
            for m in range(int(cfg["m_oracle"])):
                trial = simulate_trial(
                    int(n_phase3),
                    scenario,
                    params,
                    seed=scenario_seed(cfg, scenario_name, int(n_phase3), m),
                    n_timepoints=int(cfg["n_timepoints"]),
                    n_biomarkers=int(cfg["n_longitudinal_biomarkers"]),
                    time_grid=cfg["time_grid"],
                    missing_rate_target=float(cfg["missing_rate_target"]),
                )
                analysis = analyze_trial(trial.T_obs, trial.delta, trial.A, admin_end=float(trial.time_grid[-1]))
                analyses.append(analysis)
                trial_rows.append({
                    "source": "oracle",
                    "scenario": scenario_name,
                    "n_phase3": int(n_phase3),
                    "trial": int(m),
                    **analysis,
                })
            summary = summarize_trial_analyses(analyses)
            rows.append({
                "scenario": scenario_name,
                "n_phase3": int(n_phase3),
                "true_pos": summary["pos"],
                "true_event_rate": summary["event_rate"],
                "true_censoring_rate": summary["censoring_rate"],
                "true_mean_hr": summary["mean_hr"],
                "true_sd_log_hr": summary["sd_log_hr"],
            })
    oracle = pd.DataFrame(rows)
    write_csv(oracle_path, oracle)
    write_csv(output / "intermediate" / "oracle_trial_analyses.csv", pd.DataFrame(trial_rows))
    dump_json(meta_path, expected_meta)
    return oracle
