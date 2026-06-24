# PfizerSynthetic Tutorial

This tutorial walks through a local reproduction workflow for the packaged PhaseSyn code.

## 1. Install

From the repository root:

```bash
conda activate env_2502
python -m pip install -e .
```

Check the command entry points:

```bash
pfizer-synthetic --help
pfizer-pbc-core4 --help
pfizer-simulate-rct --help
pfizer-simulation-holdout --help
```

## 2. PBC/PBC2 Core4 Workflow

The repository includes the small local PBC/PBC2 source tables in `data/pbc2`. The main PBC config is repository-relative:

```bash
experiments/pbc_core4/config_pbc_core4_tuned.yaml
```

Preprocess only:

```bash
pfizer-pbc-core4 \
  --config experiments/pbc_core4/config_pbc_core4_tuned.yaml \
  --only preprocess
```

Smoke run:

```bash
pfizer-pbc-core4 \
  --config experiments/pbc_core4/config_pbc_core4_tuned.yaml \
  --smoke
```

Full run:

```bash
pfizer-pbc-core4 \
  --config experiments/pbc_core4/config_pbc_core4_tuned.yaml
```

Important output locations:

```text
data/processed/pbc_core4/
outputs/pbc_experiments/experiment_20260604_core4_tuned/
outputs/pbc_experiments/experiment_20260604_core4_tuned/tables/
outputs/pbc_experiments/experiment_20260604_core4_tuned/figures/
```

## 3. PBC Reference Figures

After the tuned PBC model exists, generate PDC2-style train/test and treatment-split figures:

```bash
pfizer-pbc-reference-figures \
  --config experiments/pbc_core4/config_pbc_core4_tuned.yaml \
  --tuned-root outputs/pbc_experiments/experiment_20260604_core4_tuned
```

The generated manifest is:

```text
outputs/pbc_experiments/experiment_20260604_core4_tuned/pdc2_reference_figures_from_tuned_model/pdc2_reference_figure_generation_manifest.json
```

Expected figure groups include:

```text
train/figures/
test/figures/
train/figures_by_treatment/
test/figures_by_treatment/
```

## 4. Simple Simulation Holdout Workflow

Generate a PhaseSyn-compatible synthetic randomized-trial dataset:

```bash
pfizer-simulate-rct \
  --out data/simulation/simple_linear_rct_n1200 \
  --n 1200 \
  --seed 20260602
```

The simulator writes:

```text
data/simulation/simple_linear_rct_n1200/data_phasesyn.csv
data/simulation/simple_linear_rct_n1200/data_types_phasesyn_piecewise.csv
data/simulation/simple_linear_rct_n1200/longitudinal.csv
data/simulation/simple_linear_rct_n1200/simulation_id.csv
data/simulation/simple_linear_rct_n1200/metadata.json
```

Run a quick CPU smoke holdout:

```bash
pfizer-simulation-holdout \
  --data-dir data/simulation/simple_linear_rct_n1200 \
  --output-root outputs/simulation_smoke \
  --device cpu \
  --epochs 2 \
  --n-replicates 2 \
  --prior-n 10
```

Run a GPU experiment:

```bash
pfizer-simulation-holdout \
  --data-dir data/simulation/simple_linear_rct_n1200 \
  --output-root outputs/simulation \
  --device cuda \
  --epochs 260 \
  --n-replicates 20 \
  --prior-n 100
```

Important outputs:

```text
outputs/simulation/train/model_checkpoint.pt
outputs/simulation/test/holdout_summary.json
outputs/simulation/test/holdout_replicate_metrics.csv
outputs/simulation/prior_generation/prior_generation_metadata.json
outputs/simulation/leakage_audit.json
```

## 5. Probability-of-Success Simulation Pipeline

The PoS simulation code lives in `experiments/simulation_pos`.

Run a configured smoke test:

```bash
pfizer-simulation-pos --config examples/simulation_pos_smoke.yaml --smoke
```

Run selected stages:

```bash
pfizer-simulation-pos --config examples/simulation_pos_smoke.yaml --stage oracle --stage phase2
pfizer-simulation-pos --config examples/simulation_pos_smoke.yaml --stage train --stage virtual --stage evaluate --stage figures
```

## 6. Core PDC2 CLI

Train on the base PDC2/PBC2-style config:

```bash
pfizer-synthetic train --config configs/pdc2.yaml --device cpu --epochs 2
```

Generate from a trained checkpoint:

```bash
pfizer-synthetic generate-prior \
  --config configs/pdc2.yaml \
  --checkpoint outputs/pdc2/pdc2_dynamic/model_checkpoint.pt \
  --n 100 \
  --treatment 0
```

## 7. Practical Notes

- Use `--device cuda` only when CUDA is available.
- Generated outputs are ignored by Git.
- Full PBC and simulation runs are long-running experiments; start with the smoke commands.
- Benchmark methods may require optional packages such as SynthCity or SDV.
- If a run writes markdown summaries, those summaries are generated outputs and should generally stay under `outputs/`.
