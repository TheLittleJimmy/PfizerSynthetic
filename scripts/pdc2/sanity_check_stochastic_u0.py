from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(1, str(ROOT))

from pdc2.models import set_seed
from tests.test_prior_cohort_generation import _model


def main() -> None:
    set_seed(20260608)
    model = _model()
    model.eval()
    model.use_u0_mean_at_eval = False
    z = torch.randn(4, model.hivae.z_dim)
    s = torch.softmax(torch.randn(4, model.hivae.s_dim), dim=-1)
    l0 = torch.randn(4, 1)

    u0, diag = model.sample_u0_from_l0(z, s, l0, deterministic=False, return_details=True)
    assert u0.shape == (4, model.u_dim)
    assert diag["u0_mu"].shape == (4, model.u_dim)
    assert diag["u0_sigma"].shape == (4, model.u_dim)
    assert torch.all(diag["u0_sigma"] >= model.u0_sigma_min)

    torch.manual_seed(1)
    first = model.init_u0_from_l0(z, s, l0, deterministic=False)
    torch.manual_seed(2)
    second = model.init_u0_from_l0(z, s, l0, deterministic=False)
    assert not torch.allclose(first, second)

    det_first = model.init_u0_from_l0(z, s, l0, deterministic=True)
    det_second = model.init_u0_from_l0(z, s, l0, deterministic=True)
    assert torch.allclose(det_first, det_second)
    assert torch.allclose(det_first, model.u0_params_from_l0(z, s, l0)["u0_mu"])
    print("stochastic_u0 sanity checks passed")


if __name__ == "__main__":
    main()
