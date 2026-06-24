#!/usr/bin/env bash
set -euo pipefail

# CPU-only runner for the current optimal PhaseSyn PDC2 holdout pipeline.
# It trains the dynamic-survival model, evaluates held-out baseline-conditioned
# generation, regenerates train/test figures, and writes a small prior-generation
# sample from the trained checkpoint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  # AGENTS.md for this workspace specifies env_2502.
  eval "$(conda shell.bash hook)"
  conda activate env_2502
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PROJECT_ROOT}/utils:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-10}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-10}"

OUTPUT_ROOT="${1:-outputs/pdc2/experiments_20260602_cpu}"
SEED="${SEED:-20260602}"
SPLIT_SEED="${SPLIT_SEED:-20260521}"
EPOCHS="${EPOCHS:-260}"
N_REPLICATES="${N_REPLICATES:-20}"
BATCH_SIZE="${BATCH_SIZE:-64}"

echo "Running PhaseSyn PDC2 optimal pipeline on CPU"
echo "Project root: ${PROJECT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Epochs: ${EPOCHS}; replicates: ${N_REPLICATES}; batch size: ${BATCH_SIZE}"

python scripts/pdc2/run_holdout_evaluation.py \
  --config configs/pdc2.yaml \
  --dataset pdc2 \
  --device cpu \
  --seed "${SEED}" \
  --split-seed "${SPLIT_SEED}" \
  --test-fraction 0.2 \
  --n-replicates "${N_REPLICATES}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --n-intervals 16 \
  --lambda-surv 1.4 \
  --kl-weight-s 0.3 \
  --kl-weight-z 0.3 \
  --longitudinal-weight 2.0 \
  --continuous-mse-weight 0.8 \
  --use-randomization-loss \
  --randomization-loss-weight 0.0 \
  --randomization-loss-on z_mean \
  --output-root "${OUTPUT_ROOT}"

echo "Regenerating train/test aggregate figures"
python scripts/pdc2/plot_holdout_test_figures.py \
  --holdout-root "${OUTPUT_ROOT}" \
  --split train
python scripts/pdc2/plot_holdout_test_figures.py \
  --holdout-root "${OUTPUT_ROOT}" \
  --split test

echo "Regenerating treatment-stratified figures"
python scripts/pdc2/plot_holdout_by_treatment.py \
  --holdout-root "${OUTPUT_ROOT}" \
  --split both

echo "Generating a small prior-based cohort from the CPU-trained checkpoint"
python - <<'PY' "${OUTPUT_ROOT}"
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from pdc2.config import normalise_config
from pdc2.data import load_pdc2_bundle
from pdc2.models import build_model
from pdc2.training import generate_prior_cohort

output_root = Path(sys.argv[1])
cfg = yaml.safe_load((output_root / "run_config.yaml").read_text())
cfg["training"]["device"] = "cpu"
normalise_config(cfg)

bundle = load_pdc2_bundle(cfg)
model = build_model(bundle, cfg).to(torch.device("cpu"))
checkpoint = output_root / "train" / "model_checkpoint.pt"
state = torch.load(checkpoint, map_location="cpu")
missing, unexpected = model.load_state_dict(state["model_state_dict"], strict=False)
bad_missing = [key for key in missing if not key.startswith("u0_logsigma_head.")]
if bad_missing or unexpected:
    raise RuntimeError(
        "Checkpoint is not compatible with the current model. "
        f"bad_missing={bad_missing[:10]}, unexpected={unexpected[:10]}"
    )
model.eval()

time_grid = np.asarray(cfg.get("generation", {}).get("time_grid", [0.0, 0.25, 0.5, 0.75, 1.0]), dtype=np.float32)
n = int(cfg.get("generation", {}).get("prior_n", 100))
treatment = int(cfg.get("generation", {}).get("prior_treatment", 0))
deterministic = bool(cfg.get("generation", {}).get("deterministic", False))

static_df, long_df, tensors = generate_prior_cohort(
    model,
    bundle,
    n=n,
    treatment=treatment,
    time_grid=time_grid,
    device=torch.device("cpu"),
    deterministic=deterministic,
    return_tensors=True,
)

prior_dir = output_root / "prior_generation"
prior_dir.mkdir(parents=True, exist_ok=True)
static_df.to_csv(prior_dir / "prior_synthetic_static.csv", index=False)
long_df.to_csv(prior_dir / "prior_synthetic_longitudinal.csv", index=False)
metadata = {
    "mode": "prior",
    "n": n,
    "treatment": treatment,
    "time_grid": time_grid.tolist(),
    "baseline_generated_from_prior": bool(tensors["baseline_generated_from_prior"].item()),
    "uses_observed_future_outcomes": bool(tensors["uses_observed_future_outcomes"].item()),
    "checkpoint": str(checkpoint),
    "device": "cpu",
}
(prior_dir / "prior_generation_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
PY

echo "CPU pipeline completed: ${OUTPUT_ROOT}"
