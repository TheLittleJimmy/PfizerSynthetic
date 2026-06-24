from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.pbc_core4.load_pbc import preprocess_to_disk
from experiments.pbc_core4.methods import PhaseSynGenerator, analysis_static, build_method, split_static_long
from experiments.pbc_core4.metrics import baseline_fidelity, longitudinal_fidelity, survival_fidelity


ROOT = Path(__file__).resolve().parents[1]
PBC_DATA_DIR = ROOT / "data" / "pbc2"


class PBCCore4SmokeTests(unittest.TestCase):
    def test_pbc_preprocess_no_subject_leakage(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data = preprocess_to_disk(PBC_DATA_DIR, Path(tmp) / "processed", 20260604)
        self.assertEqual(len(data.subjects), 312)
        self.assertEqual(set(data.subjects["treatment"].unique()), {0, 1})
        train = set(data.splits["train"])
        val = set(data.splits["validation"])
        test = set(data.splits["test"])
        self.assertTrue(train.isdisjoint(val))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(val.isdisjoint(test))
        self.assertEqual(len(train | val | test), 312)

    def test_benchmark_smoke_outputs_finite_metrics(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            data = preprocess_to_disk(PBC_DATA_DIR, Path(tmp) / "processed", 20260604)
        train_static, train_long = split_static_long(data, "train")
        real_static = analysis_static(data)
        gen = build_method("classical_lmm_cox_aft_simulator", train_static, train_long, 20260604, use_mixedlm=False)
        syn_static, syn_long, diag = gen.generate(12, treatment=0)
        self.assertEqual(diag["status"], "completed")
        self.assertEqual(len(syn_static), 12)
        self.assertFalse(syn_long.empty)
        m = baseline_fidelity(real_static[real_static["treatment"].eq(0)], syn_static, gen.name, 0)
        l = longitudinal_fidelity(data.longitudinal[data.longitudinal["treatment"].eq(0)], syn_long, gen.name, 0)
        s = survival_fidelity(real_static[real_static["treatment"].eq(0)], syn_static, gen.name, 0)
        values = [v for row in [m, l, s] for v in row.values() if isinstance(v, float)]
        self.assertTrue(any(np.isfinite(values)))

    def test_phasesyn_decode_longitudinal_skips_padded_visits(self):
        specs = [SimpleNamespace(name="bili"), SimpleNamespace(name="albumin")]
        times = torch.tensor(
            [
                [0.0, 0.5, 1.0, 0.0, 0.0],
                [0.0, 0.4, 0.8, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        masks = torch.tensor(
            [
                [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
                [[1, 1], [1, 1], [1, 1], [0, 0], [0, 0]],
            ],
            dtype=torch.float32,
        )
        bundle = SimpleNamespace(
            longitudinal=SimpleNamespace(
                times=times,
                masks=masks,
                time_min=0.0,
                time_max=1.0,
                specs=specs,
            )
        )
        pred_raw = np.arange(2 * 5 * 2, dtype=float).reshape(2, 5, 2)
        static = pd.DataFrame(
            {
                "subject_id": [101, 202],
                "treatment": [0, 1],
                "time": [1.0, 0.5],
                "event": [0, 1],
            }
        )

        decoded = PhaseSynGenerator._decode_longitudinal(None, bundle, pred_raw, static)

        self.assertEqual(decoded["subject_id"].tolist(), [101, 101, 101, 202, 202])
        zero_time_counts = decoded[decoded["visit_time"].eq(0.0)].groupby("subject_id").size().to_dict()
        self.assertEqual(zero_time_counts, {101: 1, 202: 1})
        self.assertEqual(decoded.groupby("subject_id")["visit_index"].max().to_dict(), {101: 2, 202: 1})


if __name__ == "__main__":
    unittest.main()
