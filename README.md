# PfizerSynthetic

PfizerSynthetic packages the PhaseSyn synthetic clinical-trial trajectory model and the experiment scripts used for PBC/PBC2 and simulation studies.

PhaseSyn is a treatment-aware generative model for randomized-trial data with:

- mixed-type baseline covariates,
- complete baseline longitudinal measurements,
- treatment-conditioned latent ODE longitudinal dynamics,
- dynamic discrete-time event and censoring hazards,
- posterior generation for observed trial participants, and
- prior generation for synthetic trial cohorts under specified treatment arms.

The repository is organized as an installable Python package while preserving the original research experiment entry points.

## Repository Layout

```text
src/pdc2/                  Core PhaseSyn/PDC2 model, training, generation, plotting CLI
evaluation/                Longitudinal and survival metrics/plots
utils/                     Legacy HI-VAE helper modules used by the core model
experiments/pbc_core4/     PBC/PBC2 Core4 experiment suite and tuned reference-figure script
experiments/simulation_pos/ Probability-of-success simulation pipeline
scripts/pdc2/              Holdout training/evaluation and figure-generation scripts
scripts/                   Simulation data generator and simulation holdout drivers
configs/                   Base model configs
data/pbc2/                 Small local PBC/PBC2 source tables used by the tutorial
tests/                     Smoke and chronology tests
docs/tutorial.md           Step-by-step reproduction tutorial
```

Generated artifacts are intentionally ignored under `outputs/`, `data/processed/`, and `data/simulation/`.

## Installation

Use the project environment if available:

```bash
conda activate env_2502
python -m pip install -e .
```

For a fresh environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional benchmark integrations such as SynthCity/SDV can be installed with:

```bash
python -m pip install -e ".[benchmarks]"
```

## Quick Checks

```bash
python -m pytest tests/test_prior_cohort_generation.py -q
pfizer-synthetic --help
pfizer-pbc-core4 --help
pfizer-simulation-holdout --help
```

## PBC/PBC2 Experiment

The PBC Core4 suite uses `data/pbc2/pbc2.csv` and writes processed data and outputs inside the repository.

Smoke run:

```bash
pfizer-pbc-core4 --config experiments/pbc_core4/config_pbc_core4_tuned.yaml --smoke
```

Full run:

```bash
pfizer-pbc-core4 --config experiments/pbc_core4/config_pbc_core4_tuned.yaml
```

Generate train/test/reference-style figures from a tuned checkpoint:

```bash
pfizer-pbc-reference-figures \
  --config experiments/pbc_core4/config_pbc_core4_tuned.yaml \
  --tuned-root outputs/pbc_experiments/experiment_20260604_core4_tuned
```

The figure generator produces train, test, and split-by-treatment figure directories when the tuned model checkpoint exists.

## Simulation Data Experiment

Create a PhaseSyn-compatible simulation dataset:

```bash
pfizer-simulate-rct \
  --out data/simulation/simple_linear_rct_n1200 \
  --n 1200 \
  --seed 20260602
```

Run the simulation holdout experiment:

```bash
pfizer-simulation-holdout \
  --data-dir data/simulation/simple_linear_rct_n1200 \
  --output-root outputs/simulation \
  --device cuda \
  --n-replicates 20
```

For a CPU smoke run:

```bash
pfizer-simulation-holdout \
  --data-dir data/simulation/simple_linear_rct_n1200 \
  --output-root outputs/simulation_smoke \
  --device cpu \
  --epochs 2 \
  --n-replicates 2 \
  --prior-n 10
```

## Probability-of-Success Simulation

The `experiments/simulation_pos` package contains the larger Phase II to Phase III probability-of-success simulation pipeline.

```bash
pfizer-simulation-pos --config examples/simulation_pos_smoke.yaml --smoke
```

Use selected stages when resuming:

```bash
pfizer-simulation-pos --config examples/simulation_pos_smoke.yaml --stage phase2 --stage train
```

## Core Model CLI

The `pfizer-synthetic` command wraps the core `pdc2` package:

```bash
pfizer-synthetic train --config configs/pdc2.yaml --device cpu --epochs 2
pfizer-synthetic generate-prior --config configs/pdc2.yaml --checkpoint outputs/pdc2/model_checkpoint.pt --n 100
```

## Reproducibility Notes

- PBC configs are repository-relative and default to `data/pbc2`.
- Simulation data are generated locally; generated files are not committed.
- Long-running experiment outputs are written under `outputs/`.
- GPU use is controlled by `--device` arguments or the corresponding YAML config fields.
- The code preserves the original research modules and helper names to keep existing experiment scripts reproducible.

## Tutorial

See [docs/tutorial.md](docs/tutorial.md) for a full walkthrough covering installation, PBC preprocessing/smoke execution, reference figure generation, simulation data generation, and simulation holdout runs.
