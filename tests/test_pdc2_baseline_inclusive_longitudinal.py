from __future__ import annotations

import torch

from pdc2.data import LongitudinalSpec
from pdc2.models import BaselineODEInitializer, ODELongitudinalDecoder, PhaseSynModel
from tests.test_prior_cohort_generation import _model


def _decoder() -> ODELongitudinalDecoder:
    return ODELongitudinalDecoder(
        [LongitudinalSpec(name="albumin", type="real")],
        u_dim=2,
        z_dim=1,
        s_dim=1,
        treatment_dim=2,
        condition_on_baseline=True,
        hidden_dim=6,
    )


def test_post_baseline_decoder_receives_real_treatment_context() -> None:
    decoder = _decoder()
    captured: dict[str, torch.Tensor] = {}
    original = decoder._path_features

    def wrapped(u_path, times, z=None, s=None, a=None):
        captured["a"] = a.detach().clone()
        return original(u_path, times, z, s, a)

    decoder._path_features = wrapped  # type: ignore[method-assign]
    u_path = torch.zeros(2, 2, 2)
    times = torch.tensor([[0.0, 0.5], [0.0, 0.5]])
    values = torch.zeros(2, 2, 1)
    masks = torch.ones(2, 2, 1)
    z = torch.zeros(2, 1)
    s = torch.zeros(2, 1)
    a = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )

    loss, _ = decoder.loss_sum_from_path_conditioned(u_path, times, values, masks, z, s, a)

    assert torch.isfinite(loss)
    assert torch.equal(captured["a"], a)


def test_longitudinal_loss_0plus_includes_baseline_and_future_terms() -> None:
    model = object.__new__(PhaseSynModel)
    torch.nn.Module.__init__(model)
    model.treatment_dim = 2
    model.baseline_long_weight = 1.0
    model.decoder = _decoder()
    model.longitudinal_baseline_decoder = ODELongitudinalDecoder(
        [LongitudinalSpec(name="albumin", type="real")],
        u_dim=2,
        z_dim=1,
        s_dim=1,
        treatment_dim=0,
        condition_on_baseline=True,
        hidden_dim=6,
    )
    captured: dict[str, torch.Tensor | None] = {}
    original_baseline = model.longitudinal_baseline_decoder._path_features
    original_future = model.decoder._path_features

    def wrapped_baseline(u_path, times, z=None, s=None, a=None):
        captured["baseline_a"] = a
        return original_baseline(u_path, times, z, s, a)

    def wrapped_future(u_path, times, z=None, s=None, a=None):
        captured["future_a"] = a.detach().clone()
        return original_future(u_path, times, z, s, a)

    model.longitudinal_baseline_decoder._path_features = wrapped_baseline  # type: ignore[method-assign]
    model.decoder._path_features = wrapped_future  # type: ignore[method-assign]

    u0 = torch.zeros(2, 2)
    u_path = torch.zeros(2, 2, 2)
    future_times = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    future_values = torch.zeros(2, 2, 1)
    future_masks = torch.tensor([[[0.0], [1.0]], [[0.0], [1.0]]])
    l0 = torch.zeros(2, 1)
    m0 = torch.ones_like(l0)
    z = torch.zeros(2, 1)
    s = torch.zeros(2, 1)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    loss, aux = model.longitudinal_loss_0plus(u0, u_path, future_times, future_values, future_masks, l0, m0, z, s, a)

    assert torch.isfinite(loss)
    assert torch.isfinite(aux["longitudinal_baseline_nll"])
    assert torch.isfinite(aux["longitudinal_future_nll"])
    assert aux["baseline_long_weight"].item() == 1.0
    assert captured["baseline_a"] is None
    assert torch.equal(captured["future_a"], a)


def test_u0_initializer_uses_direct_l0_without_treatment() -> None:
    initializer = BaselineODEInitializer(z_dim=2, s_dim=3, l0_dim=4, u_dim=5, hidden_dim=6)
    captured: dict[str, torch.Tensor] = {}
    original_forward = initializer.net[0].forward

    def wrapped_first_layer(x):
        captured["input"] = x.detach().clone()
        return original_forward(x)

    initializer.net[0].forward = wrapped_first_layer  # type: ignore[method-assign]
    z = torch.randn(2, 2)
    s = torch.randn(2, 3)
    l0 = torch.randn(2, 4)

    out_a0 = initializer(z, s, l0)
    out_a1 = initializer(z, s, l0)

    assert out_a0.shape == (2, 5)
    assert captured["input"].shape[-1] == z.shape[-1] + s.shape[-1] + l0.shape[-1]
    assert torch.equal(captured["input"], torch.cat([z, s, l0], dim=-1))
    assert torch.equal(out_a0, out_a1)


def test_stochastic_u0_shapes_and_deterministic_switch() -> None:
    torch.manual_seed(21)
    model = _model()
    model.train()
    z = torch.randn(3, model.hivae.z_dim)
    s = torch.softmax(torch.randn(3, model.hivae.s_dim), dim=-1)
    l0 = torch.randn(3, 1)

    u0, diag = model.sample_u0_from_l0(z, s, l0, return_details=True)
    u0_det, diag_det = model.sample_u0_from_l0(z, s, l0, deterministic=True, return_details=True)

    assert u0.shape == (3, model.u_dim)
    assert diag["u0_mu"].shape == (3, model.u_dim)
    assert diag["u0_sigma"].shape == (3, model.u_dim)
    assert diag["u0_sample"].shape == (3, model.u_dim)
    assert torch.all(diag["u0_sigma"] >= model.u0_sigma_min)
    assert torch.allclose(u0_det, diag_det["u0_mu"])


def test_stochastic_u0_replicates_vary_and_deterministic_matches_mean() -> None:
    torch.manual_seed(22)
    model = _model()
    model.eval()
    model.use_u0_mean_at_eval = False
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    l0 = torch.randn(2, 1)

    torch.manual_seed(1)
    sampled_a = model.init_u0_from_l0(z, s, l0, deterministic=False)
    torch.manual_seed(2)
    sampled_b = model.init_u0_from_l0(z, s, l0, deterministic=False)
    det_a = model.init_u0_from_l0(z, s, l0, deterministic=True)
    det_b = model.init_u0_from_l0(z, s, l0, deterministic=True)
    params = model.u0_params_from_l0(z, s, l0)

    assert not torch.allclose(sampled_a, sampled_b)
    assert torch.allclose(det_a, det_b)
    assert torch.allclose(det_a, params["u0_mu"])


def test_deterministic_u0_backward_compatibility_flag() -> None:
    model = _model()
    model.stochastic_u0 = False
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    l0 = torch.randn(2, 1)

    torch.manual_seed(3)
    u0_a = model.init_u0_from_l0(z, s, l0, deterministic=False)
    torch.manual_seed(4)
    u0_b = model.init_u0_from_l0(z, s, l0, deterministic=False)

    assert torch.allclose(u0_a, u0_b)
    assert torch.allclose(u0_a, model.u0_params_from_l0(z, s, l0)["u0_mu"])


def test_u0_sampler_uses_only_z_s_and_l0() -> None:
    model = _model()
    captured: dict[str, torch.Tensor] = {}
    original_forward = model.u0_initializer.net[0].forward

    def wrapped_first_layer(x):
        captured["input"] = x.detach().clone()
        return original_forward(x)

    model.u0_initializer.net[0].forward = wrapped_first_layer  # type: ignore[method-assign]
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    l0 = torch.randn(2, 1)

    model.init_u0_from_l0(z, s, l0, deterministic=True)

    expected = torch.cat([z, s, l0], dim=-1)
    assert captured["input"].shape == expected.shape
    assert torch.equal(captured["input"], expected)


def test_generation_output_keeps_observed_l0_for_t0_rows() -> None:
    model = object.__new__(PhaseSynModel)
    model.baseline_time_eps = 1e-6
    times = torch.tensor([[0.0, 0.25], [0.25, 0.0]])
    l0 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    ode_mean = torch.zeros(2, 2, 2)

    official_mean = ode_mean.clone()
    t0_rows = times.abs() <= model.baseline_time_eps
    l0_expand = l0.unsqueeze(1).expand_as(official_mean)
    replace = t0_rows.unsqueeze(-1).expand_as(official_mean)
    official_mean = torch.where(replace, l0_expand, official_mean)

    assert torch.equal(official_mean[0, 0], l0[0])
    assert torch.equal(official_mean[1, 1], l0[1])
    assert torch.equal(official_mean[0, 1], torch.zeros(2))
    assert torch.equal(official_mean[1, 0], torch.zeros(2))
