from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_markdown_table(path: Path, title: str, df: pd.DataFrame, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    if df.empty:
        lines.append("No rows were generated.")
    else:
        lines.append(df.head(80).to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_main_summary(output_dir: Path, key: dict[str, Any]) -> None:
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PBC Core-4 Main Results Summary",
        "",
        "## Dataset",
        "",
        "The suite uses the local PBC2/PBCSeq-style table with 312 randomized subjects and 1,945 longitudinal visits.",
        "Treatment is coded as placebo = 0 and D-penicillamine = 1.",
        "The main endpoint is death-or-transplant composite; death-only sensitivity is implemented in preprocessing.",
        "",
        "## Method Status",
        "",
    ]
    for row in key.get("method_status", []):
        lines.append(f"- {row.get('method')}: {row.get('status')}")
    lines.extend([
        "",
        "## Experiment Results",
        "",
        f"- Experiment 1 metrics: `{output_dir / 'exp1_control_arm/tables/exp1_metrics_all_methods.csv'}`",
        f"- Experiment 2 treatment effects: `{output_dir / 'exp2_matched_counterfactual_controls/tables/exp2_treatment_effects.csv'}`",
        f"- Experiment 3 prediction metrics: `{output_dir / 'exp3_digital_twin_validation/tables/exp3_longitudinal_prediction.csv'}`",
        f"- Experiment 4 trial simulation: `{output_dir / 'exp4_virtual_trial_simulation/tables/exp4_real_virtual_trial_power.csv'}`",
        "",
        "## Interpretation",
        "",
        "Experiment 2 is interpreted as matched synthetic control construction and estimand-level evaluation, not individual counterfactual truth on real data.",
        "Experiment 3 is factual digital-twin validation from baseline and randomized treatment only.",
        "Experiment 4 separates semi-synthetic ground-truth calibration from real-data virtual trial operating characteristics.",
        "Ordinary fidelity metrics are reported alongside type-I error/power calibration because survival synthetic-data utility is not guaranteed by fidelity alone.",
        "When `execution_scope` is `smoke_validation`, reported survival prediction and calibration-filtering values are implementation checks and proxy metrics, not final manuscript results.",
        "Exp3 survival prediction uses available lifelines/proxy metrics unless optional time-dependent survival tooling is installed.",
        "Exp4 calibration filtering uses conservative validation-proxy thresholds until full-run replicate distributions are available.",
        "",
        "## Known Failures Or Dependencies",
        "",
    ])
    failures = key.get("failures", [])
    if failures:
        lines.extend(f"- {item}" for item in failures)
    else:
        lines.append("- No method-level hard failures were logged.")
    warnings = key.get("dependency_warnings", [])
    if warnings:
        lines.extend(["", "Dependency warnings:", ""])
        lines.extend(f"- {item}" for item in warnings)
    lines.extend([
        "",
        "## Reproducibility",
        "",
        "Top-level command:",
        "",
        "```bash",
        "conda run -n env_2502 python -m experiments.pbc_core4.run_all --config experiments/pbc_core4/config_pbc_core4.yaml",
        "```",
        "",
        "Individual rerun commands:",
        "",
        "```bash",
    ])
    commands = key.get("rerun_commands", {})
    for label in ["preprocess", "benchmarks", "PhaseSyn", "Experiment 1", "Experiment 2", "Experiment 3", "Experiment 4", "smoke"]:
        command = commands.get(label)
        if command:
            lines.append(f"# {label}")
            lines.append(command)
    lines.extend([
        "```",
        "",
        "Machine-readable run summary:",
        "",
        "```json",
        json.dumps(key, indent=2, sort_keys=True),
        "```",
    ])
    (reports / "main_results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
