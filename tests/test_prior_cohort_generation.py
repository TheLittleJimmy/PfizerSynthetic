from __future__ import annotations

import types

import pandas as pd
import torch

from pdc2.data import LongitudinalPanel, LongitudinalSpec, PDC2Bundle
from pdc2.models import PhaseSynModel, build_hivae
from pdc2.training import generate_longitudinal_samples, prior_cohort_to_dataframes


class GuardedEncoder(torch.nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise AssertionError("prior cohort generation must not call the baseline encoder")


def _bundle() -> PDC2Bundle:
    import pandas as pd
    import numpy as np

    specs = [LongitudinalSpec(name="l0", type="real", mean=10.0, std=2.0)]
    panel = LongitudinalPanel(
        subject_ids=np.arange(2),
        times=torch.tensor([[0.0, 0.5], [0.0, 0.5]], dtype=torch.float32),
        values=torch.zeros(2, 2, 1),
        masks=torch.ones(2, 2, 1),
        raw_values=np.zeros((2, 2, 1), dtype=np.float32),
        specs=specs,
        time_min=0.0,
        time_max=1.0,
    )
    raw_df = pd.DataFrame({"time": [1.0, 2.0], "censor": [1.0, 0.0], "x": [0.0, 1.0], "l0": [10.0, 12.0]})
    return PDC2Bundle(
        raw_df=raw_df,
        encoded_df=raw_df[["time", "censor", "x", "l0"]],
        types=[
            {"name": "survcens", "type": "surv_dynamic", "dim": "2", "nclass": ""},
            {"name": "x", "type": "real", "dim": "1", "nclass": ""},
            {"name": "l0", "type": "real", "dim": "1", "nclass": ""},
        ],
        miss_mask=torch.ones(2, 3),
        true_miss_mask=torch.ones(2, 3),
        longitudinal=panel,
        ids_df=pd.DataFrame({"id": [0, 1]}),
        y_dim_partition=[2, 2, 2],
        static_feature_count=3,
        treatment=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        treatment_name="drug",
        treatment_n_classes=2,
    )


def _model(guard_encoder: bool = True) -> PhaseSynModel:
    bundle = _bundle()
    cfg = {
        "model": {
            "z_dim": 2,
            "s_dim": 3,
            "y_dim_static": 2,
            "u_dim": 2,
            "ode_hidden_dim": 4,
            "decoder_hidden_dim": 4,
            "n_intervals": 3,
            "survival": "dynamic",
            "lambda_surv": 1.0,
            "dynamic_survival_hidden_dim": 4,
            "dynamic_survival_num_layers": 1,
            "dynamic_survival_dropout": 0.0,
            "survival_history_pooling": "boundary",
            "u0_init_mode": "baseline_l0",
            "encoder_conditioning": "baseline_only",
            "condition_ode_on_baseline": True,
            "condition_longitudinal_decoder_on_baseline": True,
        },
        "_bundle_meta": {"treatment_name": "drug", "treatment_n_classes": 2},
    }
    hivae = build_hivae(bundle, cfg)
    model = PhaseSynModel(hivae, bundle.longitudinal, cfg)
    if guard_encoder:
        model.hivae.s_layer = GuardedEncoder(model.hivae.s_dim)
        model.hivae.z_layer = GuardedEncoder(model.hivae.z_dim * 2)
    return model


def test_prior_cohort_shapes_and_fixed_treatment() -> None:
    torch.manual_seed(7)
    model = _model()
    out = model.generate_prior_cohort(
        n=5,
        treatment=1,
        time_grid=torch.tensor([0.0, 0.25, 0.5]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
        deterministic=True,
    )

    assert out["baseline_values"].shape == (5, 4)
    assert out["longitudinal_values"].shape == (5, 3, 1)
    assert out["time_grid"].shape == (5, 3)
    assert out["observed_time"].shape == (5, 1)
    assert out["event"].shape == (5, 1)
    assert out["event_hazard"].shape == (5, model.survival_interval_grid.numel())
    assert out["censoring_hazard"].shape == (5, model.survival_interval_grid.numel())
    assert torch.equal(out["treatment"], torch.ones(5, dtype=torch.long))
    assert torch.equal(out["treatment_context"], torch.tensor([[0.0, 1.0]]).expand(5, -1))


def test_prior_cohort_samples_baseline_from_prior_and_copies_generated_l0_at_t0() -> None:
    torch.manual_seed(11)
    model = _model()
    out = model.generate_prior_cohort(
        n=6,
        treatment=torch.zeros(6, dtype=torch.long),
        time_grid=torch.tensor([0.0, 0.5]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
    )

    assert bool(out["baseline_generated_from_prior"].item()) is True
    assert torch.equal(
        torch.isnan(out["longitudinal_values"]),
        ~out["longitudinal_available"].unsqueeze(-1).expand_as(out["longitudinal_values"]),
    )
    assert out["component"].shape == (6,)
    assert out["s"].shape == (6, model.hivae.s_dim)
    assert torch.allclose(out["s"].sum(dim=1), torch.ones(6))
    assert out["z"].shape == (6, model.hivae.z_dim)
    assert torch.equal(out["longitudinal_values_normalized"][:, 0, :], out["L0"])


def test_prior_generation_has_no_observed_future_inputs() -> None:
    model = _model()
    out = model.generate_prior_cohort(
        n=3,
        treatment=0,
        time_grid=torch.tensor([0.2, 0.4]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
        deterministic=True,
    )

    assert bool(out["uses_observed_future_outcomes"].item()) is False


def test_train_longitudinal_export_samples_decoder_when_not_deterministic() -> None:
    model = _model()
    bundle = _bundle()
    calls = {"sample": 0}

    def fake_mean(self, u_path, times, z=None, s=None, a=None):
        return torch.full((u_path.shape[0], u_path.shape[1], 1), -4.0, device=u_path.device, dtype=u_path.dtype)

    def fake_sample(self, u_path, times, z=None, s=None, a=None, deterministic=False):
        calls["sample"] += 1
        return torch.full((u_path.shape[0], u_path.shape[1], 1), 3.0, device=u_path.device, dtype=u_path.dtype)

    model.decoder.mean_from_path = types.MethodType(fake_mean, model.decoder)  # type: ignore[method-assign]
    model.decoder.sample_from_path_conditioned = types.MethodType(fake_sample, model.decoder)  # type: ignore[method-assign]
    synthetic_df = pd.DataFrame({"l0": [10.0, 12.0]})
    latents = {
        "z": torch.zeros(2, model.hivae.z_dim),
        "s": torch.zeros(2, model.hivae.s_dim),
        "a": bundle.treatment,
    }

    out, metrics = generate_longitudinal_samples(
        model,
        bundle,
        synthetic_df,
        torch.device("cpu"),
        latents=latents,
        deterministic=False,
        return_diagnostics=True,
    )

    assert calls["sample"] == 1
    assert metrics["longitudinal_decoder_export"] == 1.0
    assert torch.allclose(torch.as_tensor(out[:, 1, 0]), torch.full((2,), 16.0))


def test_model_omits_removed_survival_coupler_and_baseline_survival_decoder() -> None:
    model = _model()

    assert not hasattr(model, "survival_coupler")
    assert not hasattr(model, "survival_decoder")
    assert not hasattr(model, "event_time_decoder")
    assert not hasattr(model, "censoring_time_decoder")
    assert not hasattr(model, "w_U")
    assert all(not feat["type"].startswith("surv") for feat in model.hivae.feat_types_list)
    assert any(feat["type"].startswith("surv") for feat in model.full_feat_types_list)


def test_removed_direct_survival_losses_are_not_present() -> None:
    model = _model()

    for name, _ in model.named_modules():
        assert "direct_survival" not in name
        assert "auxiliary_survival" not in name
    for name, _ in model.named_parameters():
        assert "w_U" not in name
        assert "r_T" not in name
        assert "r_C" not in name


def test_shared_time_normalization_check_rejects_out_of_range_times() -> None:
    model = _model()

    model.validate_shared_time_normalization(torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0, 0.5]]))
    try:
        model.validate_shared_time_normalization(torch.tensor([[0.0, 1.1]]), torch.tensor([[0.0, 0.5]]))
    except ValueError as exc:
        assert "Longitudinal times" in str(exc)
    else:
        raise AssertionError("Expected shared time normalization check to reject out-of-range longitudinal time.")


def test_raw_model_constructor_caps_u_dim() -> None:
    bundle = _bundle()
    cfg = {
        "model": {
            "z_dim": 6,
            "s_dim": 6,
            "y_dim_static": 6,
            "u_dim": 99,
            "gru_hidden_dim": 6,
            "ode_hidden_dim": 6,
            "decoder_hidden_dim": 6,
            "n_intervals": 3,
            "survival": "dynamic",
            "lambda_surv": 1.0,
            "dynamic_survival_hidden_dim": 6,
            "dynamic_survival_num_layers": 1,
            "dynamic_survival_dropout": 0.0,
            "survival_history_pooling": "boundary",
            "u0_init_mode": "baseline_l0",
            "encoder_conditioning": "baseline_only",
            "condition_ode_on_baseline": True,
            "treatment_dim": 2,
        }
    }
    hivae = build_hivae(bundle, cfg)

    model = PhaseSynModel(hivae, bundle.longitudinal, cfg)

    assert model.u_dim == 6


def test_dynamic_survival_loss_uses_dynamic_hazards_only() -> None:
    event_hazard = torch.tensor([[0.2, 0.3, 0.4], [0.1, 0.2, 0.3]])
    censoring_hazard = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.3, 0.2]])
    times = torch.tensor([[0.5, 0.75, 1.0], [0.5, 0.75, 1.0]])
    observed_time = torch.tensor([0.5, 0.25])
    event = torch.tensor([1.0, 0.0])

    loss, aux = PhaseSynModel.dynamic_survival_nll(event_hazard, censoring_hazard, observed_time, event, times)

    expected_first = torch.log(torch.tensor(0.2))
    expected_second = torch.log(torch.tensor(0.4))
    expected = -torch.stack([expected_first, expected_second]).mean()
    assert torch.allclose(loss, expected, atol=1e-6)
    assert torch.equal(aux["survival_interval_index"], torch.tensor([0, 0]))


def test_interval_assignment_uses_right_closed_boundaries() -> None:
    boundaries = torch.tensor([[0.25, 0.5, 0.75, 1.0]]).expand(7, -1)
    tau = torch.tensor([-0.1, 0.0, 0.25, 0.25001, 0.5, 0.999, 1.2])

    idx = PhaseSynModel.survival_interval_indices(tau, boundaries)

    assert torch.equal(idx, torch.tensor([0, 0, 0, 1, 1, 3, 3]))


def test_admin_censoring_uses_event_and_censor_survival_to_end() -> None:
    event_hazard = torch.tensor([[0.2, 0.3, 0.4]])
    censoring_hazard = torch.tensor([[0.1, 0.2, 0.3]])
    boundaries = torch.tensor([[0.25, 0.5, 1.0]])
    observed_time = torch.tensor([1.0])
    event = torch.tensor([0.0])

    loss, aux = PhaseSynModel.dynamic_survival_nll(
        event_hazard,
        censoring_hazard,
        observed_time,
        event,
        boundaries,
        admin_end_threshold=0.999999,
    )

    expected_loglik = torch.log1p(-event_hazard).sum() + torch.log1p(-censoring_hazard).sum()
    assert torch.allclose(loss, -expected_loglik, atol=1e-6)
    assert bool(aux["admin_censoring_mask"].item()) is True


def test_non_admin_censoring_uses_stochastic_censoring_hazard() -> None:
    event_hazard = torch.tensor([[0.2, 0.3, 0.4]])
    censoring_hazard = torch.tensor([[0.1, 0.2, 0.3]])
    boundaries = torch.tensor([[0.25, 0.5, 1.0]])
    observed_time = torch.tensor([0.5])
    event = torch.tensor([0.0])

    loss, aux = PhaseSynModel.dynamic_survival_nll(
        event_hazard,
        censoring_hazard,
        observed_time,
        event,
        boundaries,
    )

    expected_loglik = torch.log1p(-event_hazard[0, 0]) + torch.log1p(-censoring_hazard[0, 0]) + torch.log(censoring_hazard[0, 1])
    assert torch.allclose(loss, -expected_loglik, atol=1e-6)
    assert bool(aux["admin_censoring_mask"].item()) is False


def test_dynamic_survival_event_uses_ode_state_and_censoring_does_not() -> None:
    torch.manual_seed(3)
    model = _model()
    u0 = torch.randn(2, model.u_dim)
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    boundaries = torch.tensor([0.25, 0.5, 1.0])

    out = model.dynamic_survival(u0, z, s, a, boundaries)

    expected_starts = torch.tensor([[0.0, 0.25, 0.5], [0.0, 0.25, 0.5]])
    assert torch.allclose(out["interval_start_times"], expected_starts)
    assert torch.allclose(out["history_summary"][..., : model.u_dim], out["u_interval_start"])
    assert torch.allclose(out["history_summary"][..., -1], expected_starts)
    assert torch.allclose(out["boundary_times"], boundaries.expand(2, -1))
    expected_event_dim = model.u_dim + model.hivae.z_dim + model.hivae.s_dim + model.u_dim + model.treatment_dim + 1
    expected_censor_dim = model.hivae.z_dim + model.hivae.s_dim + model.u_dim + model.treatment_dim + 1
    assert out["event_head_input"].shape[-1] == expected_event_dim
    assert out["censoring_head_input"].shape[-1] == expected_censor_dim
    assert torch.allclose(out["event_head_input"][..., : model.u_dim], out["u_interval_start"])
    assert torch.allclose(out["censoring_head_input"][..., -1], expected_starts)
    assert not torch.equal(out["event_head_input"][..., :expected_censor_dim], out["censoring_head_input"])
    assert model.treatment_dim == 2


def test_censoring_logits_are_invariant_to_interval_start_trajectory() -> None:
    torch.manual_seed(4)
    model = _model()
    u0 = torch.randn(2, model.u_dim)
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    boundaries = torch.tensor([[0.25, 0.5, 1.0], [0.25, 0.5, 1.0]])
    starts = model._survival_interval_start_times(boundaries)
    u_start = torch.randn(2, 3, model.u_dim)
    perturbed_u_start = u_start + 10.0

    base = model.dynamic_survival_from_interval_start_path(u_start, starts, boundaries, z, s, u0, a)
    perturbed = model.dynamic_survival_from_interval_start_path(perturbed_u_start, starts, boundaries, z, s, u0, a)

    assert not torch.allclose(base["event_hazard_logits"], perturbed["event_hazard_logits"])
    assert torch.allclose(base["censoring_hazard_logits"], perturbed["censoring_hazard_logits"], atol=1e-7)


def test_dynamic_survival_interval_intercepts_shift_logits() -> None:
    torch.manual_seed(5)
    model = _model()
    u0 = torch.randn(2, model.u_dim)
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    boundaries = torch.tensor([0.25, 0.5, 1.0])

    base = model.dynamic_survival(u0, z, s, a, boundaries)
    with torch.no_grad():
        model.dynamic_survival_head.alpha_T.copy_(torch.tensor([0.1, -0.2, 0.3]))
        model.dynamic_survival_head.alpha_C.copy_(torch.tensor([-0.4, 0.5, -0.6]))
    shifted = model.dynamic_survival(u0, z, s, a, boundaries)

    assert torch.allclose(
        shifted["event_hazard_logits"] - base["event_hazard_logits"],
        model.dynamic_survival_head.alpha_T.view(1, -1).expand_as(base["event_hazard_logits"]),
        atol=1e-6,
    )
    assert torch.allclose(
        shifted["censoring_hazard_logits"] - base["censoring_hazard_logits"],
        model.dynamic_survival_head.alpha_C.view(1, -1).expand_as(base["censoring_hazard_logits"]),
        atol=1e-6,
    )


def test_dynamic_survival_rejects_invalid_interval_grids() -> None:
    model = _model()
    u0 = torch.randn(2, model.u_dim)
    z = torch.randn(2, model.hivae.z_dim)
    s = torch.softmax(torch.randn(2, model.hivae.s_dim), dim=-1)
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    invalid_grids = [
        torch.tensor([0.25, 0.5]),
        torch.tensor([0.25, 0.25, 1.0]),
        torch.tensor([0.25, 0.5, 0.9]),
        torch.tensor([-0.1, 0.5, 1.0]),
    ]
    for grid in invalid_grids:
        try:
            model.dynamic_survival(u0, z, s, a, grid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid survival interval grid to be rejected: {grid}.")


def test_dynamic_survival_rejects_gru_u0_initialization() -> None:
    bundle = _bundle()
    cfg = {
        "model": {
            "z_dim": 2,
            "s_dim": 3,
            "y_dim_static": 2,
            "u_dim": 2,
            "ode_hidden_dim": 4,
            "decoder_hidden_dim": 4,
            "n_intervals": 3,
            "survival": "dynamic",
            "lambda_surv": 1.0,
            "dynamic_survival_hidden_dim": 4,
            "dynamic_survival_num_layers": 1,
            "dynamic_survival_dropout": 0.0,
            "survival_history_pooling": "boundary",
            "u0_init_mode": "gru",
            "encoder_conditioning": "baseline_only",
            "condition_ode_on_baseline": True,
            "condition_longitudinal_decoder_on_baseline": True,
        },
        "_bundle_meta": {"treatment_name": "drug", "treatment_n_classes": 2},
    }
    hivae = build_hivae(bundle, cfg)

    try:
        PhaseSynModel(hivae, bundle.longitudinal, cfg)
    except ValueError as exc:
        assert "u0_init_mode='baseline_l0'" in str(exc)
    else:
        raise AssertionError("Expected dynamic survival to reject GRU u0 initialization.")


def test_prior_generated_survival_schema_uses_observed_time_and_event() -> None:
    model = _model()
    out = model.generate_prior_cohort(
        n=5,
        treatment=1,
        time_grid=torch.tensor([0.0, 0.5, 1.0]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
        deterministic=True,
    )

    assert torch.allclose(out["baseline_encoded"][:, :1], out["observed_time"])
    assert torch.allclose(out["baseline_encoded"][:, 1:2], out["event"])
    assert torch.isfinite(out["baseline_encoded"][:, :2]).all()


def test_survival_warmup_weight() -> None:
    model = _model()
    model.lambda_surv = 1.4
    model.survival_warmup_epochs = 10

    assert model.survival_weight_for_epoch(None) == 1.4
    assert model.survival_weight_for_epoch(0) == 0.0
    assert abs(model.survival_weight_for_epoch(5) - 0.7) < 1e-8
    assert model.survival_weight_for_epoch(20) == 1.4


def test_dynamic_survival_sampling_jitters_within_intervals_and_avoids_forced_ties() -> None:
    torch.manual_seed(123)
    model = _model()
    hazards = torch.full((512, 3), 0.35)
    survival_out = {
        "event_hazard": hazards,
        "censoring_hazard": hazards.clone(),
        "boundary_times": torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 1.0]]).expand(512, -1),
    }

    out = model.sample_dynamic_survival(survival_out, deterministic=False)

    assert torch.isfinite(out["observed_time"]).all()
    assert ((out["observed_time"] >= 0.0) & (out["observed_time"] <= 1.0)).all()
    grid = survival_out["boundary_times"][0, :-1]
    on_boundary = torch.isclose(out["observed_time"], grid.view(1, -1)).any(dim=1)
    assert float(on_boundary.float().mean()) < 0.05
    assert 0.35 < float(out["event"].mean()) < 0.65
    assert torch.any(out["event_interval_index"] == out["censoring_interval_index"])


def test_dynamic_survival_sampling_labels_tail_tail_as_censored() -> None:
    model = _model()
    survival_out = {
        "event_hazard": torch.zeros(4, 3),
        "censoring_hazard": torch.zeros(4, 3),
        "boundary_times": torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 1.0]]).expand(4, -1),
    }

    out = model.sample_dynamic_survival(survival_out, deterministic=True)

    assert torch.equal(out["event_tail"], torch.ones(4, 1, dtype=torch.bool))
    assert torch.equal(out["censoring_tail"], torch.ones(4, 1, dtype=torch.bool))
    assert torch.isinf(out["event_time"]).all()
    assert torch.equal(out["censoring_time"], torch.ones(4, 1))
    assert torch.equal(out["event"], torch.zeros(4, 1))
    assert torch.equal(out["observed_time"], torch.ones(4, 1))


def test_dynamic_survival_sampling_sets_event_tail_to_infinity_and_censor_tail_to_endpoint() -> None:
    model = _model()
    survival_out = {
        "event_hazard": torch.zeros(1, 3),
        "censoring_hazard": torch.tensor([[1.0, 0.0, 0.0]]),
        "boundary_times": torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 1.0]]),
    }

    out = model.sample_dynamic_survival(survival_out, deterministic=True)

    assert torch.isinf(out["event_time"]).all()
    assert torch.allclose(out["censoring_time"], torch.tensor([[1.0 / 6.0]]))
    assert torch.equal(out["event"], torch.zeros(1, 1))
    assert torch.allclose(out["observed_time"], out["censoring_time"])


def test_dynamic_survival_sampling_preserves_same_interval_tie_convention() -> None:
    model = _model()
    survival_out = {
        "event_hazard": torch.tensor([[1.0, 0.0, 0.0]]),
        "censoring_hazard": torch.tensor([[1.0, 0.0, 0.0]]),
        "boundary_times": torch.tensor([[1.0 / 3.0, 2.0 / 3.0, 1.0]]),
    }

    out = model.sample_dynamic_survival(survival_out, deterministic=True)

    assert torch.equal(out["event_interval_index"], torch.tensor([[0]]))
    assert torch.equal(out["censoring_interval_index"], torch.tensor([[0]]))
    assert torch.equal(out["event"], torch.ones(1, 1))
    assert torch.allclose(out["event_time"], torch.tensor([[1.0 / 6.0]]))


def test_generated_longitudinal_records_after_u_are_unavailable() -> None:
    model = _model()
    out = model.generate_prior_cohort(
        n=4,
        treatment=1,
        time_grid=torch.tensor([0.0, 0.4, 0.8, 1.0]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
        deterministic=True,
    )

    available = out["longitudinal_available"].unsqueeze(-1).expand_as(out["longitudinal_values"])
    assert torch.equal(torch.isnan(out["longitudinal_values"]), ~available)


def test_prior_dataframe_export_drops_post_u_unavailable_rows() -> None:
    bundle = _bundle()
    model = _model()
    cohort = model.generate_prior_cohort(
        n=4,
        treatment=1,
        time_grid=torch.tensor([0.0, 0.4, 0.8, 1.0]),
        normalization_params=[(0.0, 2.0), (3.0, 4.0), (10.0, 4.0)],
        deterministic=True,
    )

    _, long_df = prior_cohort_to_dataframes(bundle, cohort)

    assert len(long_df) == int(cohort["longitudinal_available"].sum().item())
    for row in long_df.itertuples(index=False):
        assert bool(cohort["longitudinal_available"][int(row.patient_id), int(row.visit_index)].item())


def test_baseline_conditioned_generation_samples_survival_after_ode_and_truncates() -> None:
    model = _model(guard_encoder=False)
    calls: list[str] = []
    original_integrate = model.integrate_path
    original_sample = model.sample_dynamic_survival

    def wrapped_integrate(*args, **kwargs):
        calls.append("integrate")
        return original_integrate(*args, **kwargs)

    def wrapped_sample(survival_out, deterministic=False):
        calls.append("sample_survival")
        return {
            "event_time": torch.full((2, 1), 0.25),
            "censoring_time": torch.ones(2, 1),
            "observed_time": torch.full((2, 1), 0.25),
            "event": torch.ones(2, 1),
            "event_interval_index": torch.zeros(2, 1, dtype=torch.long),
            "censoring_interval_index": torch.full((2, 1), 3, dtype=torch.long),
            "event_tail": torch.zeros(2, 1, dtype=torch.bool),
            "censoring_tail": torch.ones(2, 1, dtype=torch.bool),
        }

    model.integrate_path = wrapped_integrate  # type: ignore[method-assign]
    model.sample_dynamic_survival = wrapped_sample  # type: ignore[method-assign]
    bundle = _bundle()
    batch_data = [
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([[10.0], [12.0]]),
    ]
    mask = torch.ones(2, 2)
    L0 = torch.tensor([[0.0], [1.0]])
    out = model.generate_observed_baseline(
        batch_data,
        mask,
        L0,
        torch.tensor([0.0, 0.5, 1.0]),
        bundle.treatment,
        deterministic_latents=True,
    )

    assert "integrate" in calls
    assert calls.index("sample_survival") > max(i for i, name in enumerate(calls) if name == "integrate")
    assert torch.equal(out["longitudinal_available"], torch.tensor([[True, False, False], [True, False, False]]))
    assert torch.isnan(out["longitudinal_mean"][:, 1:, :]).all()
