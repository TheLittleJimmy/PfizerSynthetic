from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from pdc2.config import normalise_config
from scripts.pdc2.run_pdc2_cv_tuning import (
    CANDIDATES,
    candidate_config,
    performance_score,
    stratified_kfold_indices,
    summarize_cv,
)


def _args(**overrides):
    values = {
        "config": "configs/pdc2.yaml",
        "dataset": "pdc2",
        "seed": 20260526,
        "device": "cpu",
        "n_replicates": 1,
        "epochs_override": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_stratified_kfold_covers_each_subject_once() -> None:
    raw = pd.DataFrame({
        "drug": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        "censor": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
        "time": np.arange(12, dtype=float),
    })

    folds = stratified_kfold_indices(raw, folds=3, seed=7)
    seen = np.concatenate(folds)

    assert sorted(seen.tolist()) == list(range(len(raw)))
    assert len(set(seen.tolist())) == len(raw)
    assert all(len(fold) > 0 for fold in folds)


def test_candidate_config_fixes_randomization_loss_and_compact_dimensions() -> None:
    cfg = candidate_config("nano2_balanced", _args())
    model = cfg["model"]

    assert model["use_randomization_loss"] is True
    assert model["randomization_loss_weight"] == 0.05
    assert model["randomization_loss_on"] == "z_mean"
    assert model["encoder_conditioning"] == "baseline_only"
    assert model["u0_init_mode"] == "baseline_l0"
    assert model["z_dim"] <= 6
    assert model["s_dim"] <= 6
    assert model["y_dim_static"] <= 6
    assert model["u_dim"] <= 6


def test_config_rejects_gru_u0_init_for_dynamic_survival() -> None:
    cfg = {
        "dataset": {"name": "pdc2"},
        "model": {
            "survival": "dynamic",
            "longitudinal_mode": "latent_ode",
            "u0_init_mode": "gru",
            "encoder_conditioning": "baseline_only",
        },
    }

    with pytest.raises(ValueError, match="u0_init_mode must be baseline_l0"):
        normalise_config(cfg)


def test_performance_score_penalizes_errors() -> None:
    good = {
        "survival_km_integrated_abs_error_mean_cv_mean": 0.05,
        "event_rate_diff_mean_cv_mean": 0.01,
        "survival_time_rmse_ratio_mean_cv_mean": 0.8,
        "survival_event_accuracy_mean_cv_mean": 0.8,
        "future_continuous_rmse_ratio_vs_l0_carryforward_mean_cv_mean": 0.4,
        "future_continuous_ks_mean_mean_cv_mean": 0.2,
        "future_categorical_accuracy_mean_cv_mean": 0.7,
        "future_categorical_tv_mean_mean_cv_mean": 0.2,
    }
    worse = dict(good)
    worse["future_categorical_accuracy_mean_cv_mean"] = 0.4
    worse["event_rate_diff_mean_cv_mean"] = 0.2

    assert performance_score(good) < performance_score(worse)


def test_summarize_cv_prefers_smaller_near_tie(tmp_path) -> None:
    for candidate, params, metric in [
        ("nano2_balanced", 1000, 0.1),
        ("tiny4_balanced", 2000, 0.1),
    ]:
        for fold in range(3):
            out = tmp_path / candidate / f"fold_{fold:02d}"
            out.mkdir(parents=True)
            row = {
                "candidate": candidate,
                "fold": fold,
                "passes_audit": True,
                "parameter_count": params,
                "survival_km_integrated_abs_error_mean": metric,
                "event_rate_diff_mean": 0.01,
                "survival_time_rmse_ratio_mean": 0.8,
                "survival_event_accuracy_mean": 0.8,
                "future_continuous_rmse_ratio_vs_l0_carryforward_mean": 0.4,
                "future_continuous_ks_mean_mean": 0.2,
                "future_categorical_accuracy_mean": 0.7,
                "future_categorical_tv_mean_mean": 0.2,
            }
            (out / "fold_summary.json").write_text(json.dumps(row), encoding="utf-8")

    _, summary, best = summarize_cv(tmp_path)

    assert set(summary["candidate"]) == {"nano2_balanced", "tiny4_balanced"}
    assert best["candidate"] == "nano2_balanced"
    assert set(CANDIDATES) >= {"nano2_balanced", "tiny4_balanced"}
