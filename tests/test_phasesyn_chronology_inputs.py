from __future__ import annotations

import unittest
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from pdc2.data import LongitudinalPanel, LongitudinalSpec, PDC2Bundle
from pdc2.models import PhaseSynModel, build_hivae


def _toy_model() -> PhaseSynModel:
    import numpy as np
    import pandas as pd

    panel = LongitudinalPanel(
        subject_ids=np.arange(3),
        times=torch.tensor([[0.0, 0.5], [0.0, 0.5], [0.0, 0.5]], dtype=torch.float32),
        values=torch.zeros(3, 2, 2),
        masks=torch.ones(3, 2, 2),
        raw_values=np.zeros((3, 2, 2), dtype=np.float32),
        specs=[
            LongitudinalSpec(name="bili", type="pos", mean=1.0, std=0.5),
            LongitudinalSpec(name="albumin", type="real", mean=3.2, std=0.3),
        ],
        time_min=0.0,
        time_max=1.0,
    )
    raw_df = pd.DataFrame(
        {
            "time": [1.0, 1.0, 1.0],
            "censor": [0.0, 1.0, 0.0],
            "sex": [0.0, 1.0, 0.0],
            "age": [40.0, 50.0, 60.0],
        }
    )
    bundle = PDC2Bundle(
        raw_df=raw_df,
        encoded_df=raw_df,
        types=[
            {"name": "survcens", "type": "surv_dynamic", "dim": "2", "nclass": ""},
            {"name": "sex", "type": "cat", "dim": "1", "nclass": "2"},
            {"name": "age", "type": "real", "dim": "1", "nclass": ""},
        ],
        miss_mask=torch.ones(3, 3),
        true_miss_mask=torch.ones(3, 3),
        longitudinal=panel,
        ids_df=pd.DataFrame({"id": [0, 1, 2]}),
        y_dim_partition=[2, 2, 2],
        static_feature_count=3,
        treatment=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        treatment_name="drug",
        treatment_n_classes=2,
    )
    cfg = {
        "model": {
            "z_dim": 2,
            "s_dim": 2,
            "y_dim_static": 2,
            "u_dim": 2,
            "gru_hidden_dim": 4,
            "ode_hidden_dim": 4,
            "decoder_hidden_dim": 4,
            "dynamic_survival_hidden_dim": 4,
            "dynamic_survival_num_layers": 1,
            "n_intervals": 4,
            "encoder_conditioning": "baseline_only",
            "u0_init_mode": "baseline_l0",
            "treatment_n_classes": 2,
        },
        "_bundle_meta": {"treatment_name": "drug", "treatment_n_classes": 2},
    }
    hivae = build_hivae(bundle, cfg)
    return PhaseSynModel(hivae, bundle.longitudinal, cfg)


class PhaseSynChronologyInputTests(unittest.TestCase):
    def test_baseline_encoder_receives_only_w_and_l0(self):
        model = _toy_model()
        model.eval()
        captured: list[torch.Tensor] = []
        original = model._encoder_input_tensor

        def capture(batch_data_observed, batch_miss, encoder_l0):
            out = original(batch_data_observed, batch_miss, encoder_l0)
            captured.append(out.detach().clone())
            return out

        model._encoder_input_tensor = capture  # type: ignore[method-assign]

        batch = 3
        w = [
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
            torch.tensor([[40.0], [50.0], [60.0]]),
        ]
        mask_w = torch.ones(batch, 2)
        l0 = torch.tensor([[1.0, 3.4], [2.0, 3.2], [1.5, 3.1]])
        times = torch.tensor([0.0, 0.5, 1.0])
        a0 = torch.tensor([0, 1, 0])

        with torch.no_grad():
            model.generate_observed_baseline(w, mask_w, l0, times, a0, deterministic_latents=True, deterministic_u0=True)
            first = captured[-1].clone()
            model.generate_observed_baseline(w, mask_w, l0, torch.tensor([0.0, 0.25, 0.75]), 1 - a0, deterministic_latents=True, deterministic_u0=True)
            second = captured[-1].clone()

        self.assertTrue(torch.equal(first, second))

        w_changed = [w[0].clone(), w[1].clone()]
        w_changed[0] = 1.0 - w_changed[0]
        with torch.no_grad():
            model.generate_observed_baseline(w_changed, mask_w, l0, times, a0, deterministic_latents=True, deterministic_u0=True)
            changed_w = captured[-1].clone()
        self.assertFalse(torch.equal(first, changed_w))

        l0_changed = l0 + torch.tensor([[0.2, 0.0], [0.0, 0.1], [0.3, -0.2]])
        with torch.no_grad():
            model.generate_observed_baseline(w, mask_w, l0_changed, times, a0, deterministic_latents=True, deterministic_u0=True)
            changed_l0 = captured[-1].clone()
        self.assertFalse(torch.equal(first, changed_l0))


if __name__ == "__main__":
    unittest.main()
