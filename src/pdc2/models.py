from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
UTILS = ROOT / "utils"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))

from utils import data_processing  # noqa: E402
from utils import src as hivae_src  # noqa: E402
from utils import theta_estimation  # noqa: E402

from .data import LongitudinalPanel, LongitudinalSpec, PDC2Bundle


MAX_HIDDEN_DIM = 6


def cap_hidden_dim(value: int | float | None, default: int = MAX_HIDDEN_DIM, max_dim: int | None = None) -> int:
    cap = MAX_HIDDEN_DIM if max_dim is None else max(1, int(max_dim))
    return max(1, min(int(default if value is None else value), cap))


def set_seed(seed: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def feature_group_weights(types: list[dict[str, Any]], cfg: dict[str, Any], l0_names: set[str] | None = None) -> torch.Tensor:
    model_cfg = cfg["model"]
    static_w = float(model_cfg.get("static_weight", 1.0))
    l0_w = float(model_cfg.get("lambda_l0_hivae", 1.0))
    # Features whose names overlap the longitudinal panel are the HI-VAE L0
    # reconstruction block, not a second ordinary baseline block.
    l0_names = set() if l0_names is None else set(l0_names)
    weights = []
    for t in types:
        if t["name"] in l0_names:
            weights.append(static_w * l0_w)
        else:
            weights.append(static_w)
    return torch.tensor(weights, dtype=torch.float32)


def longitudinal_observed_rows(masks: torch.Tensor) -> torch.Tensor:
    return (masks.sum(dim=-1) > 0).to(dtype=masks.dtype)


def split_baseline_future_longitudinal(
    times: torch.Tensor,
    values: torch.Tensor,
    masks: torch.Tensor,
    baseline_time_eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    observed_rows = longitudinal_observed_rows(masks).bool()
    baseline_candidates = (times.abs() <= float(baseline_time_eps)) & observed_rows
    baseline_counts = baseline_candidates.sum(dim=1)
    if not torch.all(baseline_counts == 1):
        bad = torch.nonzero(baseline_counts != 1, as_tuple=False).flatten()[:10].detach().cpu().tolist()
        raise ValueError(
            "Every subject must have exactly one observed t=0 longitudinal baseline row; "
            f"bad subject positions include {bad}."
        )
    baseline_idx = baseline_candidates.float().argmax(dim=1)
    batch_idx = torch.arange(times.shape[0], device=times.device)
    l0 = values[batch_idx, baseline_idx]
    m0 = masks[batch_idx, baseline_idx]
    if bool((m0 < 1.0).any().item()):
        bad = torch.nonzero((m0 < 1.0).any(dim=1), as_tuple=False).flatten()[:10].detach().cpu().tolist()
        raise ValueError(
            "L0 is assumed fully observed, but at least one baseline longitudinal variable is missing; "
            f"bad subject positions include {bad}."
        )
    t0 = times[batch_idx, baseline_idx]
    future_rows = observed_rows & ~baseline_candidates
    future_masks = masks * future_rows.unsqueeze(-1).to(dtype=masks.dtype)
    return {
        "L0": l0,
        "M0": m0,
        "t0": t0,
        "future_values": values,
        "future_masks": future_masks,
        "future_times": times,
        "future_visit_mask": future_rows.to(dtype=values.dtype),
        "baseline_index": baseline_idx,
    }


def build_hivae(bundle: PDC2Bundle, cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    max_hidden_dim = int(model_cfg.get("max_hidden_dim", MAX_HIDDEN_DIM))
    baseline_types = [feat for feat in bundle.types if not feat["type"].startswith("surv")]
    baseline_y_dim_partition = [
        cap_hidden_dim(dim, max_dim=max_hidden_dim)
        for dim, feat in zip(bundle.y_dim_partition, bundle.types)
        if not feat["type"].startswith("surv")
    ]
    static_input_dim = sum(int(feat["nclass"]) if feat["type"] in {"cat", "ordinal"} else int(feat["dim"]) for feat in baseline_types)
    encoder_input_dim = static_input_dim + len(baseline_types) + len(bundle.longitudinal.specs)
    hivae = hivae_src.HIVAE_inputDropout(
        input_dim=encoder_input_dim,
        z_dim=cap_hidden_dim(model_cfg.get("z_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
        s_dim=cap_hidden_dim(model_cfg.get("s_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
        y_dim=sum(baseline_y_dim_partition),
        y_dim_partition=baseline_y_dim_partition,
        feat_types_dict=baseline_types,
        intervals_surv_piecewise=None,
        n_layers_surv_piecewise=None,
        model_version=None,
        n_long_outcomes=len(bundle.longitudinal.specs),
    )
    hivae.kl_weight_s = float(model_cfg.get("kl_weight_s", 1.0))
    hivae.kl_weight_z = float(model_cfg.get("kl_weight_z", 1.0))
    hivae.l0_feature_names = {spec.name for spec in bundle.longitudinal.specs}
    hivae.feature_group_weights = feature_group_weights(baseline_types, cfg, hivae.l0_feature_names)
    hivae.static_input_dim = static_input_dim
    hivae.encoder_mask_dim = len(baseline_types)
    hivae.encoder_l0_dim = len(bundle.longitudinal.specs)
    hivae.full_feat_types_list = [dict(feat) for feat in bundle.types]
    hivae.full_y_dim_partition = [cap_hidden_dim(dim, max_dim=max_hidden_dim) for dim in bundle.y_dim_partition]
    hivae.full_to_baseline_indices = [
        idx for idx, feat in enumerate(bundle.types) if not feat["type"].startswith("surv")
    ]
    hivae.full_survival_indices = [
        idx for idx, feat in enumerate(bundle.types) if feat["type"].startswith("surv")
    ]
    return hivae


class LongitudinalImputer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.fill = nn.Parameter(torch.zeros(n_features))

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return values * masks + self.fill.view(1, 1, -1) * (1.0 - masks)


class GRULongitudinalEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, u_dim: int):
        super().__init__()
        hidden_dim = max(1, int(hidden_dim))
        self.gru = nn.GRU(1 + 2 * n_features, hidden_dim, batch_first=True)
        self.to_params = nn.Linear(hidden_dim, 2 * u_dim)

    def forward(self, times: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([times.unsqueeze(-1), values, masks], dim=-1)
        observed_rows = longitudinal_observed_rows(masks)
        x = x * observed_rows.unsqueeze(-1)
        _, h = self.gru(x)
        mu, log_var = self.to_params(h[-1]).chunk(2, dim=-1)
        return mu, log_var.clamp(-8.0, 8.0)


class UPrior(nn.Module):
    def __init__(self, z_dim: int, s_dim: int, u_dim: int):
        super().__init__()
        hidden = cap_hidden_dim(None)
        self.net = nn.Sequential(
            nn.Linear(z_dim + s_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * u_dim),
        )

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_var = self.net(torch.cat([z, s], dim=-1)).chunk(2, dim=-1)
        return mu, log_var.clamp(-8.0, 8.0)


class BaselineL0Encoder(nn.Module):
    def __init__(self, n_features: int, embedding_dim: int):
        super().__init__()
        hidden = cap_hidden_dim(None)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, l0: torch.Tensor) -> torch.Tensor:
        return self.net(l0)


class BaselineODEInitializer(nn.Module):
    def __init__(self, z_dim: int, s_dim: int, l0_dim: int, u_dim: int, hidden_dim: int):
        super().__init__()
        hidden = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(z_dim + s_dim + l0_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, u_dim),
        )

    def forward(self, z: torch.Tensor, s: torch.Tensor, l0: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, s, l0], dim=-1))


def _softplus_inverse(value: float) -> float:
    value = max(float(value), 1e-6)
    return math.log(math.expm1(value))


class BaselineU0SigmaHead(nn.Module):
    """Lightweight variance head for p_eta(u0 | z, s, L0)."""

    def __init__(
        self,
        z_dim: int,
        s_dim: int,
        l0_dim: int,
        u_dim: int,
        hidden_dim: int,
        initial_sigma: float,
        sigma_min: float,
    ):
        super().__init__()
        hidden = max(1, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(z_dim + s_dim + l0_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, u_dim),
        )
        target_softplus = max(float(initial_sigma) - float(sigma_min), 1e-4)
        with torch.no_grad():
            final = self.net[-1]
            if isinstance(final, nn.Linear):
                final.weight.zero_()
                final.bias.fill_(_softplus_inverse(target_softplus))

    def forward(self, z: torch.Tensor, s: torch.Tensor, l0: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, s, l0], dim=-1))


class TypeAwareLongitudinalDecoder(nn.Module):
    def __init__(
        self,
        specs: list[LongitudinalSpec],
        u_dim: int,
        time_dim: int = 8,
        hidden_dim: int = 64,
        continuous_mse_weight: float = 0.0,
    ):
        super().__init__()
        self.specs = specs
        self.continuous_mse_weight = float(continuous_mse_weight)
        time_dim = max(1, int(time_dim))
        hidden_dim = max(1, int(hidden_dim))
        self.time_embed = nn.Sequential(nn.Linear(1, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))
        self.heads = nn.ModuleList()
        for spec in specs:
            out_dim = 2 if spec.type in {"real", "pos", "count"} else int(spec.nclass or 2)
            self.heads.append(nn.Sequential(
                nn.Linear(u_dim + time_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            ))

    def _features(self, u: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(times.unsqueeze(-1))
        u_exp = u.unsqueeze(1).expand(-1, times.shape[1], -1)
        return torch.cat([u_exp, t_emb], dim=-1)

    def loss(self, u: torch.Tensor, times: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self._features(u, times)
        losses = []
        cont_sq = []
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            mask = masks[:, :, idx]
            if spec.type in {"real", "pos", "count"}:
                mu = out[:, :, 0]
                log_var = out[:, :, 1].clamp(-8.0, 8.0)
                var = F.softplus(log_var).clamp(min=1e-4, max=1e4)
                target = values[:, :, idx]
                sq = (target - mu).pow(2)
                nll = 0.5 * (torch.log(var) + sq / var + math.log(2.0 * math.pi))
                losses.append(((nll + self.continuous_mse_weight * sq) * mask).sum() / mask.sum().clamp(min=1.0))
                cont_sq.append((sq * mask).sum() / mask.sum().clamp(min=1.0))
            else:
                target = values[:, :, idx].long().clamp(min=0, max=int(spec.nclass or 2) - 1)
                ce = F.cross_entropy(out.reshape(-1, out.shape[-1]), target.reshape(-1), reduction="none").reshape_as(mask)
                losses.append((ce * mask).sum() / mask.sum().clamp(min=1.0))
        total = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=values.device)
        aux = {"long_rmse_norm": torch.sqrt(torch.stack(cont_sq).mean()) if cont_sq else torch.tensor(0.0, device=values.device)}
        return total, aux

    def mean(self, u: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        features = self._features(u, times)
        outs = []
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            if spec.type in {"real", "pos", "count"}:
                outs.append(out[:, :, 0].unsqueeze(-1))
            else:
                outs.append(torch.argmax(out, dim=-1).float().unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def inverse_mean_numpy(self, u: torch.Tensor, panel: LongitudinalPanel) -> np.ndarray:
        pred = self.mean(u, panel.times.to(u.device)).detach().cpu().numpy()
        for idx, spec in enumerate(panel.specs):
            if spec.type in {"real", "pos", "count"}:
                pred[:, :, idx] = pred[:, :, idx] * spec.std + spec.mean
        return pred


def kl_normal(mu_q: torch.Tensor, log_var_q: torch.Tensor, mu_p: torch.Tensor, log_var_p: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        log_var_p - log_var_q + (torch.exp(log_var_q) + (mu_q - mu_p).pow(2)) / torch.exp(log_var_p) - 1.0
    ).sum(dim=-1).mean()


def multi_rbf_mmd2(
    x: torch.Tensor,
    y: torch.Tensor,
    bandwidths: list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Biased nonnegative multi-bandwidth RBF MMD^2 estimator."""
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    d_xx = torch.cdist(x, x).pow(2)
    d_yy = torch.cdist(y, y).pow(2)
    d_xy = torch.cdist(x, y).pow(2)
    k_xx = torch.zeros_like(d_xx)
    k_yy = torch.zeros_like(d_yy)
    k_xy = torch.zeros_like(d_xy)
    for bw in bandwidths:
        denom = 2.0 * float(bw) * float(bw)
        k_xx = k_xx + torch.exp(-d_xx / denom)
        k_yy = k_yy + torch.exp(-d_yy / denom)
        k_xy = k_xy + torch.exp(-d_xy / denom)
    scale = 1.0 / max(len(bandwidths), 1)
    k_xx = k_xx * scale
    k_yy = k_yy * scale
    k_xy = k_xy * scale
    mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    return mmd.clamp_min(0.0)


class LatentODEFunc(nn.Module):
    def __init__(
        self,
        u_dim: int,
        hidden_dim: int,
        z_dim: int = 0,
        s_dim: int = 0,
        treatment_dim: int = 0,
        condition_on_baseline: bool = False,
    ):
        super().__init__()
        self.condition_on_baseline = bool(condition_on_baseline)
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.treatment_dim = int(treatment_dim)
        in_dim = u_dim + 1 + (self.z_dim + self.s_dim + self.treatment_dim if self.condition_on_baseline else 0)
        hidden_dim = max(1, int(hidden_dim))
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, u_dim))
        self._context: torch.Tensor | None = None

    def set_context(self, z: torch.Tensor | None, s: torch.Tensor | None, a: torch.Tensor | None = None) -> None:
        if self.condition_on_baseline and z is not None and s is not None:
            pieces = [z, s]
            if self.treatment_dim > 0:
                if a is None:
                    a = torch.zeros(z.shape[0], self.treatment_dim, device=z.device, dtype=z.dtype)
                pieces.append(a)
            self._context = torch.cat(pieces, dim=-1)
        else:
            self._context = None

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(u.shape[0], 1)
        elif t.dim() == 1:
            t = t.view(-1, 1)
        pieces = [u, t.to(u.device, u.dtype)]
        if self.condition_on_baseline:
            if self._context is None:
                context = torch.zeros(u.shape[0], self.z_dim + self.s_dim + self.treatment_dim, device=u.device, dtype=u.dtype)
            else:
                context = self._context.to(u.device, u.dtype)
            pieces.append(context)
        return self.net(torch.cat(pieces, dim=-1))


def rk4_integrate(func: nn.Module, u0: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
    states = [u0]
    current = u0
    prev_t = torch.zeros(u0.shape[0], 1, device=u0.device, dtype=u0.dtype)
    for j in range(times.shape[1]):
        t = times[:, j:j + 1]
        dt = (t - prev_t).clamp(min=0.0)
        k1 = func(prev_t, current)
        k2 = func(prev_t + dt / 2, current + dt * k1 / 2)
        k3 = func(prev_t + dt / 2, current + dt * k2 / 2)
        k4 = func(t, current + dt * k3)
        current = current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        states.append(current)
        prev_t = t
    return torch.stack(states[1:], dim=1)


class ODELongitudinalDecoder(TypeAwareLongitudinalDecoder):
    def __init__(
        self,
        specs: list[LongitudinalSpec],
        u_dim: int,
        z_dim: int = 0,
        s_dim: int = 0,
        treatment_dim: int = 0,
        condition_on_baseline: bool = False,
        time_dim: int = 8,
        hidden_dim: int = 64,
        continuous_mse_weight: float = 0.0,
    ):
        super().__init__(
            specs,
            u_dim + (int(z_dim) + int(s_dim) + int(treatment_dim) if condition_on_baseline else 0),
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            continuous_mse_weight=continuous_mse_weight,
        )
        self.base_u_dim = u_dim
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.treatment_dim = int(treatment_dim)
        self.condition_on_baseline = bool(condition_on_baseline)

    def _path_features(
        self,
        u_path: torch.Tensor,
        times: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pieces = [u_path]
        if self.condition_on_baseline:
            if z is None or s is None:
                context = torch.zeros(
                    u_path.shape[0],
                    u_path.shape[1],
                    self.z_dim + self.s_dim + self.treatment_dim,
                    device=u_path.device,
                    dtype=u_path.dtype,
                )
            else:
                context_pieces = [z, s]
                if self.treatment_dim > 0:
                    if a is None:
                        a_context = torch.zeros(
                            z.shape[0],
                            u_path.shape[1],
                            self.treatment_dim,
                            device=z.device,
                            dtype=z.dtype,
                        )
                    else:
                        a = a.to(device=z.device, dtype=z.dtype)
                        if a.dim() == 2:
                            a_context = a.unsqueeze(1).expand(-1, u_path.shape[1], -1)
                        elif a.dim() == 3:
                            if a.shape[:2] != u_path.shape[:2] or a.shape[2] != self.treatment_dim:
                                raise ValueError(
                                    "Per-time treatment context must have shape "
                                    f"{(u_path.shape[0], u_path.shape[1], self.treatment_dim)}, got {tuple(a.shape)}."
                                )
                            a_context = a
                        else:
                            raise ValueError(f"Treatment context must be 2D or 3D, got shape {tuple(a.shape)}.")
                    context_pieces.append(a_context)
                expanded = []
                for item in context_pieces:
                    if item.dim() == 2:
                        expanded.append(item.unsqueeze(1).expand(-1, u_path.shape[1], -1))
                    elif item.dim() == 3:
                        expanded.append(item)
                    else:
                        raise ValueError(f"Context tensor must be 2D or 3D, got shape {tuple(item.shape)}.")
                context = torch.cat(expanded, dim=-1)
            pieces.append(context)
        pieces.append(self.time_embed(times.unsqueeze(-1)))
        return torch.cat(pieces, dim=-1)

    def loss_from_path(self, u_path: torch.Tensor, times: torch.Tensor, values: torch.Tensor, masks: torch.Tensor):
        return self.loss_from_path_conditioned(u_path, times, values, masks)

    def loss_from_path_conditioned(
        self,
        u_path: torch.Tensor,
        times: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ):
        losses = []
        cont_sq = []
        features = self._path_features(u_path, times, z, s, a)
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            mask = masks[:, :, idx]
            if spec.type in {"real", "pos", "count"}:
                mu = out[:, :, 0]
                log_var = out[:, :, 1].clamp(-8.0, 8.0)
                var = F.softplus(log_var).clamp(min=1e-4, max=1e4)
                target = values[:, :, idx]
                sq = (target - mu).pow(2)
                nll = 0.5 * (torch.log(var) + sq / var + math.log(2.0 * math.pi))
                losses.append(((nll + self.continuous_mse_weight * sq) * mask).sum() / mask.sum().clamp(min=1.0))
                cont_sq.append((sq * mask).sum() / mask.sum().clamp(min=1.0))
            else:
                target = values[:, :, idx].long().clamp(min=0, max=int(spec.nclass or 2) - 1)
                ce = F.cross_entropy(out.reshape(-1, out.shape[-1]), target.reshape(-1), reduction="none").reshape_as(mask)
                losses.append((ce * mask).sum() / mask.sum().clamp(min=1.0))
        total = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=u_path.device)
        aux = {"long_rmse_norm": torch.sqrt(torch.stack(cont_sq).mean()) if cont_sq else torch.tensor(0.0, device=u_path.device)}
        return total, aux

    def loss_sum_from_path_conditioned(
        self,
        u_path: torch.Tensor,
        times: torch.Tensor,
        values: torch.Tensor,
        masks: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        features = self._path_features(u_path, times, z, s, a)
        total = torch.zeros((), device=u_path.device, dtype=u_path.dtype)
        cont_sq_sum = torch.zeros((), device=u_path.device, dtype=u_path.dtype)
        cont_count = torch.zeros((), device=u_path.device, dtype=u_path.dtype)
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            mask = masks[:, :, idx].to(device=u_path.device, dtype=u_path.dtype)
            if spec.type in {"real", "pos", "count"}:
                mu = out[:, :, 0]
                log_var = out[:, :, 1].clamp(-8.0, 8.0)
                var = F.softplus(log_var).clamp(min=1e-4, max=1e4)
                target = values[:, :, idx].to(device=u_path.device, dtype=u_path.dtype)
                sq = (target - mu).pow(2)
                nll = 0.5 * (torch.log(var) + sq / var + math.log(2.0 * math.pi))
                total = total + ((nll + self.continuous_mse_weight * sq) * mask).sum()
                cont_sq_sum = cont_sq_sum + (sq * mask).sum()
                cont_count = cont_count + mask.sum()
            else:
                target = values[:, :, idx].to(device=u_path.device).long().clamp(min=0, max=int(spec.nclass or 2) - 1)
                ce = F.cross_entropy(out.reshape(-1, out.shape[-1]), target.reshape(-1), reduction="none").reshape_as(mask)
                total = total + (ce * mask).sum()
        return total, {"long_sq_sum": cont_sq_sum, "long_cont_count": cont_count}

    def mean_from_path(
        self,
        u_path: torch.Tensor,
        times: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = self._path_features(u_path, times, z, s, a)
        outs = []
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            if spec.type in {"real", "pos", "count"}:
                outs.append(out[:, :, 0].unsqueeze(-1))
            else:
                outs.append(torch.argmax(out, dim=-1).float().unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def sample_from_path_conditioned(
        self,
        u_path: torch.Tensor,
        times: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        features = self._path_features(u_path, times, z, s, a)
        outs = []
        for idx, spec in enumerate(self.specs):
            out = self.heads[idx](features)
            if spec.type in {"real", "pos", "count"}:
                mu = out[:, :, 0]
                if deterministic:
                    value = mu
                else:
                    log_var = out[:, :, 1].clamp(-8.0, 8.0)
                    var = F.softplus(log_var).clamp(min=1e-4, max=1e4)
                    value = torch.normal(mu, torch.sqrt(var))
            else:
                if deterministic:
                    value = torch.argmax(out, dim=-1).float()
                else:
                    value = torch.distributions.Categorical(
                        logits=out.reshape(-1, out.shape[-1])
                    ).sample().reshape(out.shape[:2]).float()
            outs.append(value.unsqueeze(-1))
        return torch.cat(outs, dim=-1)


class DynamicSurvivalHead(nn.Module):
    def __init__(
        self,
        u_dim: int,
        z_dim: int,
        s_dim: int,
        treatment_dim: int,
        n_intervals: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.u_dim = int(u_dim)
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.treatment_dim = int(treatment_dim)
        self.n_intervals = int(n_intervals)
        if self.n_intervals <= 0:
            raise ValueError("DynamicSurvivalHead requires n_intervals > 0.")
        # Event input uses the solved interval-start latent trajectory. Censoring
        # stays baseline-, treatment-, and time-dependent.
        self.event_input_dim = self.u_dim + self.z_dim + self.s_dim + self.u_dim + self.treatment_dim + 1
        self.censoring_input_dim = self.z_dim + self.s_dim + self.u_dim + self.treatment_dim + 1
        self.event_net = self._make_branch(self.event_input_dim, hidden_dim, num_layers, dropout)
        self.censoring_net = self._make_branch(self.censoring_input_dim, hidden_dim, num_layers, dropout)
        self.alpha_T = nn.Parameter(torch.zeros(self.n_intervals))
        self.alpha_C = nn.Parameter(torch.zeros(self.n_intervals))

    @staticmethod
    def _make_branch(in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        depth = max(int(num_layers), 1)
        current = int(in_dim)
        hidden = max(1, int(hidden_dim))
        for _ in range(depth):
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.ReLU())
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            current = hidden
        layers.append(nn.Linear(current, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        u_start: torch.Tensor,
        tau_start: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        u0: torch.Tensor,
        treatment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if tau_start.shape != u_start.shape[:2]:
            raise ValueError(
                "Dynamic survival inputs must agree on batch and interval dimensions; "
                f"got u={tuple(u_start.shape)}, tau={tuple(tau_start.shape)}."
            )
        batch_size, n_intervals = tau_start.shape
        if n_intervals != self.n_intervals:
            raise ValueError(
                "DynamicSurvivalHead interval count must match configured n_intervals; "
                f"got {n_intervals}, expected {self.n_intervals}."
            )
        z_exp = z.unsqueeze(1).expand(batch_size, n_intervals, -1)
        s_exp = s.unsqueeze(1).expand(batch_size, n_intervals, -1)
        u0_exp = u0.unsqueeze(1).expand(batch_size, n_intervals, -1)
        a_exp = treatment.unsqueeze(1).expand(batch_size, n_intervals, -1)
        tau_feature = tau_start.unsqueeze(-1)
        event_features = torch.cat([u_start, z_exp, s_exp, u0_exp, a_exp, tau_feature], dim=-1)
        censoring_features = torch.cat([z_exp, s_exp, u0_exp, a_exp, tau_feature], dim=-1)
        alpha_t = self.alpha_T.to(device=event_features.device, dtype=event_features.dtype).view(1, -1)
        alpha_c = self.alpha_C.to(device=censoring_features.device, dtype=censoring_features.dtype).view(1, -1)
        event_logits = self.event_net(event_features).squeeze(-1) + alpha_t
        censoring_logits = self.censoring_net(censoring_features).squeeze(-1) + alpha_c
        return {
            "event_hazard_logits": event_logits,
            "censoring_hazard_logits": censoring_logits,
            "event_hazard": torch.sigmoid(event_logits),
            "censoring_hazard": torch.sigmoid(censoring_logits),
            "history_summary": event_features,
            "event_head_input": event_features,
            "censoring_head_input": censoring_features,
            "alpha_T": self.alpha_T,
            "alpha_C": self.alpha_C,
        }


class PhaseSynModel(nn.Module):
    def __init__(self, hivae: nn.Module, panel: LongitudinalPanel, cfg: dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        max_hidden_dim = int(model_cfg.get("max_hidden_dim", MAX_HIDDEN_DIM))
        self.hivae = hivae
        n_features = len(panel.specs)
        u_dim = cap_hidden_dim(model_cfg.get("u_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
        self.u_dim = u_dim
        self.full_feat_types_list = [dict(feat) for feat in getattr(hivae, "full_feat_types_list", hivae.feat_types_list)]
        self.full_to_baseline_indices = list(getattr(hivae, "full_to_baseline_indices", range(len(hivae.feat_types_list))))
        self.full_survival_indices = list(getattr(hivae, "full_survival_indices", []))
        self.u0_init_mode = str(model_cfg.get("u0_init_mode", "baseline_l0"))
        if self.u0_init_mode != "baseline_l0":
            raise ValueError("Dynamic survival requires model.u0_init_mode='baseline_l0' to avoid future longitudinal leakage.")
        self.encoder_conditioning = str(model_cfg.get("encoder_conditioning", "baseline_only"))
        self.baseline_time_eps = float(model_cfg.get("baseline_time_eps", 1e-6))
        self.lambda_l0_hivae = float(model_cfg.get("lambda_l0_hivae", 1.0))
        self.baseline_long_weight = float(model_cfg.get("baseline_long_weight", 1.0))
        self.lambda_surv = float(model_cfg.get("lambda_surv", 1.0))
        self.survival_event_weight = float(model_cfg.get("survival_event_weight", 1.0))
        self.survival_event_aux_weight = float(model_cfg.get("survival_event_aux_weight", 0.0))
        self.survival_time_aux_weight = float(model_cfg.get("survival_time_aux_weight", 0.0))
        self.survival_time_head_weight = float(model_cfg.get("survival_time_head_weight", 0.0))
        self.survival_warmup_epochs = int(model_cfg.get("survival_warmup_epochs", 0))
        self.admin_censoring_mode = str(model_cfg.get("admin_censoring_mode", "event_and_censor_survival"))
        if self.admin_censoring_mode != "event_and_censor_survival":
            raise ValueError("Only admin_censoring_mode='event_and_censor_survival' is currently supported.")
        self.admin_end_threshold = float(model_cfg.get("admin_end_threshold", 1.0 - 1e-6))
        self.detach_l0_for_u0_init = bool(model_cfg.get("detach_l0_for_u0_init", False))
        self.treatment_dim = int(
            getattr(cfg.get("_bundle_meta", {}), "get", lambda *_: None)("treatment_n_classes", None)
            or model_cfg.get("treatment_dim", 0)
        )
        self.treatment_name = str(getattr(cfg.get("_bundle_meta", {}), "get", lambda *_: "drug")("treatment_name", "drug"))
        self.survival_feature_indices = [
            idx for idx, feat in enumerate(self.full_feat_types_list) if feat["type"].startswith("surv")
        ]
        meta = cfg.get("_bundle_meta", {})
        self.survival_time_min = float(meta.get("survival_time_min", 0.0))
        self.survival_time_max = float(meta.get("survival_time_max", 1.0))
        self.non_survival_feature_indices = [
            idx for idx, feat in enumerate(self.full_feat_types_list) if not feat["type"].startswith("surv")
        ]
        self._full_to_hivae_pos = {full_idx: pos for pos, full_idx in enumerate(self.full_to_baseline_indices)}
        self.imputer = LongitudinalImputer(n_features)
        self.encoder = (
            GRULongitudinalEncoder(
                n_features,
                cap_hidden_dim(model_cfg.get("gru_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
                u_dim,
            )
            if self.u0_init_mode == "gru"
            else None
        )
        self.prior = UPrior(hivae.z_dim, hivae.s_dim, u_dim)
        self.l0_initializer_mode = str(model_cfg.get("l0_initializer_mode", "direct"))
        if self.l0_initializer_mode not in {"direct", "encoded"}:
            raise ValueError("model.l0_initializer_mode must be 'direct' or 'encoded'.")
        l0_initializer_dim = n_features
        self.l0_encoder: BaselineL0Encoder | None = None
        if self.l0_initializer_mode == "encoded":
            l0_initializer_dim = cap_hidden_dim(model_cfg.get("l0_embedding_dim", u_dim), max_dim=max_hidden_dim)
            self.l0_encoder = BaselineL0Encoder(n_features, l0_initializer_dim)
        self.u0_initializer = BaselineODEInitializer(
            hivae.z_dim,
            hivae.s_dim,
            l0_initializer_dim,
            u_dim,
            cap_hidden_dim(model_cfg.get("u0_initializer_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
        )
        self.stochastic_u0 = bool(model_cfg.get("stochastic_u0", True))
        self.u0_sigma_mode = str(model_cfg.get("u0_sigma_mode", "learned"))
        if self.u0_sigma_mode not in {"learned", "fixed"}:
            raise ValueError("model.u0_sigma_mode must be 'learned' or 'fixed'.")
        self.u0_fixed_sigma = float(model_cfg.get("u0_fixed_sigma", 0.05))
        if self.u0_fixed_sigma <= 0:
            raise ValueError("model.u0_fixed_sigma must be positive.")
        self.u0_sigma_min = float(model_cfg.get("u0_sigma_min", 0.03))
        if self.u0_sigma_min < 0:
            raise ValueError("model.u0_sigma_min must be nonnegative.")
        self.u0_kl_weight = float(model_cfg.get("u0_kl_weight", 0.0))
        if self.u0_kl_weight < 0:
            raise ValueError("model.u0_kl_weight must be nonnegative.")
        self.use_u0_mean_at_eval = bool(model_cfg.get("use_u0_mean_at_eval", False))
        self.u0_logsigma_head: BaselineU0SigmaHead | None = None
        if self.u0_sigma_mode == "learned":
            self.u0_logsigma_head = BaselineU0SigmaHead(
                hivae.z_dim,
                hivae.s_dim,
                l0_initializer_dim,
                u_dim,
                cap_hidden_dim(model_cfg.get("u0_initializer_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
                initial_sigma=self.u0_fixed_sigma,
                sigma_min=self.u0_sigma_min,
            )
        self.ode_func = LatentODEFunc(
            u_dim,
            cap_hidden_dim(model_cfg.get("ode_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
            z_dim=hivae.z_dim,
            s_dim=hivae.s_dim,
            treatment_dim=self.treatment_dim,
            condition_on_baseline=bool(model_cfg.get("condition_ode_on_baseline", True)),
        )
        self.decoder = ODELongitudinalDecoder(
            panel.specs,
            u_dim,
            z_dim=hivae.z_dim,
            s_dim=hivae.s_dim,
            treatment_dim=self.treatment_dim,
            condition_on_baseline=bool(model_cfg.get("condition_longitudinal_decoder_on_baseline", True)),
            time_dim=cap_hidden_dim(
                model_cfg.get("decoder_time_dim", model_cfg.get("time_embedding_dim", model_cfg.get("decoder_hidden_dim", MAX_HIDDEN_DIM))),
                max_dim=max_hidden_dim,
            ),
            hidden_dim=cap_hidden_dim(model_cfg.get("decoder_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
            continuous_mse_weight=float(model_cfg.get("continuous_mse_weight", 0.0)),
        )
        self.longitudinal_baseline_decoder = ODELongitudinalDecoder(
            panel.specs,
            u_dim,
            z_dim=hivae.z_dim,
            s_dim=hivae.s_dim,
            treatment_dim=0,
            condition_on_baseline=bool(model_cfg.get("condition_longitudinal_decoder_on_baseline", True)),
            time_dim=cap_hidden_dim(
                model_cfg.get("decoder_time_dim", model_cfg.get("time_embedding_dim", model_cfg.get("decoder_hidden_dim", MAX_HIDDEN_DIM))),
                max_dim=max_hidden_dim,
            ),
            hidden_dim=cap_hidden_dim(model_cfg.get("decoder_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
            continuous_mse_weight=float(model_cfg.get("continuous_mse_weight", 0.0)),
        )
        self.kl_weight_u = float(model_cfg.get("kl_weight_u", 1.0)) if self.u0_init_mode == "gru" else 0.0
        self.longitudinal_weight = float(model_cfg.get("longitudinal_weight", 1.0))
        self.deterministic_u = bool(model_cfg.get("deterministic_u", False)) if self.u0_init_mode == "gru" else True
        self.longitudinal_only_loss = bool(model_cfg.get("longitudinal_only_loss", False))
        self.n_survival_intervals = int(model_cfg.get("n_intervals", 10))
        self.survival_interval_grid = self._default_survival_interval_grid(
            self.n_survival_intervals,
        )
        self.dynamic_survival_head = DynamicSurvivalHead(
            u_dim=u_dim,
            z_dim=hivae.z_dim,
            s_dim=hivae.s_dim,
            treatment_dim=self.treatment_dim,
            n_intervals=self.n_survival_intervals,
            hidden_dim=cap_hidden_dim(model_cfg.get("dynamic_survival_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim),
            num_layers=int(model_cfg.get("dynamic_survival_num_layers", 2)),
            dropout=float(model_cfg.get("dynamic_survival_dropout", 0.0)),
        )
        surv_hidden = cap_hidden_dim(model_cfg.get("dynamic_survival_hidden_dim", MAX_HIDDEN_DIM), max_dim=max_hidden_dim)
        self.survival_time_head = nn.Sequential(
            nn.Linear(u_dim + hivae.z_dim + hivae.s_dim + self.treatment_dim, surv_hidden),
            nn.ReLU(),
            nn.Linear(surv_hidden, 1),
        )
        self.survival_history_pooling = str(model_cfg.get("survival_history_pooling", "boundary"))
        if self.survival_history_pooling != "boundary":
            raise ValueError("Only survival_history_pooling='boundary' is currently supported.")
        self.randomization_mmd_bandwidths = [float(x) for x in model_cfg.get("randomization_mmd_bandwidths", [0.5, 1.0, 2.0, 4.0])]
        self.randomization_loss_on = str(model_cfg.get("randomization_loss_on", "z_mean"))

    @staticmethod
    def _default_survival_interval_grid(n_intervals: int) -> torch.Tensor:
        if int(n_intervals) <= 0:
            raise ValueError("model.n_intervals must be positive for dynamic survival.")
        # Values are interval upper boundaries. Interval b covers
        # (tau_{b-1}, tau_b], with tau_0=0.
        return torch.linspace(0.0, 1.0, int(n_intervals) + 1, dtype=torch.float32)[1:]

    def normalize_survival_time(self, time: torch.Tensor) -> torch.Tensor:
        span = max(self.survival_time_max - self.survival_time_min, 1e-6)
        return ((time - self.survival_time_min) / span).clamp(min=0.0, max=1.0)

    def denormalize_survival_time(self, tau: torch.Tensor) -> torch.Tensor:
        span = max(self.survival_time_max - self.survival_time_min, 1e-6)
        return tau * span + self.survival_time_min

    def validate_shared_time_normalization(self, longitudinal_times: torch.Tensor, survival_times: torch.Tensor) -> None:
        if bool((longitudinal_times < -1e-6).any().item() or (longitudinal_times > 1.0 + 1e-6).any().item()):
            raise ValueError("Longitudinal times must be normalized to [0,1] before ODE/survival use.")
        if bool((survival_times < -1e-6).any().item() or (survival_times > 1.0 + 1e-6).any().item()):
            raise ValueError("Survival interval times must be normalized to [0,1] before ODE/survival use.")

    def _select_baseline_features(self, full_data: list[torch.Tensor], full_miss: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        if len(full_data) == len(self.hivae.feat_types_list) and full_miss.shape[1] == len(self.hivae.feat_types_list):
            return full_data, full_miss
        if len(full_data) != len(self.full_feat_types_list) or full_miss.shape[1] != len(self.full_feat_types_list):
            raise ValueError(
                "Expected either baseline-only features or the full feature list with survival columns; "
                f"got {len(full_data)} tensors and mask shape {tuple(full_miss.shape)}."
            )
        data = [full_data[idx] for idx in self.full_to_baseline_indices]
        miss = full_miss[:, self.full_to_baseline_indices]
        return data, miss

    def _full_survival_data(self, full_data: list[torch.Tensor], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.survival_feature_indices) != 1:
            raise ValueError(f"Dynamic survival expects exactly one survival feature, found {len(self.survival_feature_indices)}.")
        data = full_data[self.survival_feature_indices[0]].to(device=device, dtype=dtype)
        if data.shape[1] < 2:
            raise ValueError("Survival feature tensor must contain observed time and event/censor indicator columns.")
        return self.normalize_survival_time(data[:, 0]), data[:, 1]

    def _survival_interval_times(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        survival_interval_grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = self.survival_interval_grid if survival_interval_grid is None else survival_interval_grid
        times = torch.as_tensor(grid, device=device, dtype=dtype)
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1).clone()
        elif times.dim() == 2 and times.shape[0] == 1 and batch_size > 1:
            times = times.expand(batch_size, -1).clone()
        elif times.dim() != 2 or times.shape[0] != batch_size:
            raise ValueError(
                "survival_interval_grid must have shape (B,), (1,B), or (batch,B); "
                f"got {tuple(times.shape)}."
            )
        if times.shape[1] == 0:
            raise ValueError("survival_interval_grid must contain at least one interval boundary.")
        if times.shape[1] != self.n_survival_intervals:
            raise ValueError(
                "survival_interval_grid length must match model.n_intervals; "
                f"got {times.shape[1]}, expected {self.n_survival_intervals}."
            )
        if not bool(torch.isfinite(times).all().item()):
            raise ValueError("survival interval upper boundaries must be finite.")
        tol = 1e-6
        if bool(((times <= 0.0) | (times > 1.0 + tol)).any().item()):
            raise ValueError("survival interval upper boundaries must lie in (0, 1].")
        if bool((times[:, 1:] <= times[:, :-1] + tol).any().item()):
            raise ValueError("survival interval upper boundaries must be strictly increasing.")
        if not torch.allclose(times[:, -1], torch.ones_like(times[:, -1]), atol=tol, rtol=0.0):
            raise ValueError("survival interval grid must end at normalized time 1.0.")
        return times

    @staticmethod
    def _survival_interval_start_times(boundary_times: torch.Tensor) -> torch.Tensor:
        if boundary_times.dim() != 2:
            raise ValueError(f"boundary_times must be 2D, got {tuple(boundary_times.shape)}.")
        zero = torch.zeros(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype)
        return torch.cat([zero, boundary_times[:, :-1]], dim=1)

    @staticmethod
    def survival_interval_indices(observed_time: torch.Tensor, boundary_times: torch.Tensor) -> torch.Tensor:
        tau = observed_time.to(boundary_times.device, boundary_times.dtype).view(-1).clamp(min=0.0, max=1.0)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(tau.shape[0], -1)
        if boundary_times.dim() != 2 or boundary_times.shape[0] != tau.shape[0]:
            raise ValueError(
                "boundary_times must have shape (B,), (1,B), or (batch,B) for interval assignment; "
                f"got observed={tuple(tau.shape)}, boundaries={tuple(boundary_times.shape)}."
            )
        return torch.sum(tau.unsqueeze(1) > boundary_times, dim=1).clamp(min=0, max=boundary_times.shape[1] - 1)

    @staticmethod
    def _select_path_at_times(
        path: torch.Tensor,
        source_times: torch.Tensor,
        target_times: torch.Tensor,
    ) -> torch.Tensor:
        source = source_times.to(device=path.device, dtype=path.dtype)
        target = target_times.to(device=path.device, dtype=path.dtype)
        if source.dim() == 2:
            source = source[0]
        idx = torch.argmin(torch.abs(target.unsqueeze(-1) - source.view(1, 1, -1)), dim=-1)
        batch_idx = torch.arange(path.shape[0], device=path.device).unsqueeze(1)
        return path[batch_idx, idx]

    def dynamic_survival_from_interval_start_path(
        self,
        u_interval_start: torch.Tensor,
        interval_start_times: torch.Tensor,
        boundary_times: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        u0: torch.Tensor,
        treatment: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        boundary_times = boundary_times.to(device=u_interval_start.device, dtype=u_interval_start.dtype)
        start_times = interval_start_times.to(device=u_interval_start.device, dtype=u_interval_start.dtype)
        if start_times.shape != boundary_times.shape or u_interval_start.shape[:2] != boundary_times.shape:
            raise ValueError(
                "Dynamic survival requires interval-start states and matching start/end grids; "
                f"got u_start={tuple(u_interval_start.shape)}, start={tuple(start_times.shape)}, end={tuple(boundary_times.shape)}."
            )
        expected_start = self._survival_interval_start_times(boundary_times)
        if not torch.allclose(start_times, expected_start, atol=1e-6, rtol=1e-6):
            raise ValueError("interval_start_times must equal [0, tau_1, ..., tau_{B-1}] for the supplied boundaries.")
        return self.dynamic_survival_head(
            u_interval_start,
            start_times,
            z,
            s,
            u0,
            treatment,
        ) | {
            "boundary_times": boundary_times,
            "interval_start_times": start_times,
        }

    def dynamic_survival(
        self,
        u0: torch.Tensor,
        z: torch.Tensor,
        s: torch.Tensor,
        treatment: torch.Tensor,
        survival_interval_grid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        boundary_times = self._survival_interval_times(u0.shape[0], u0.device, u0.dtype, survival_interval_grid)
        start_times = self._survival_interval_start_times(boundary_times)
        self.validate_shared_time_normalization(start_times, boundary_times)
        u_start = self.integrate_path(u0, start_times, z, s, treatment)
        out = self.dynamic_survival_from_interval_start_path(u_start, start_times, boundary_times, z, s, u0, treatment)
        out["u_interval_start"] = u_start
        out["time_prediction"] = torch.sigmoid(self.survival_time_head(torch.cat([u0, z, s, treatment], dim=-1))).view(-1)
        return out

    @staticmethod
    def dynamic_survival_nll(
        event_hazard: torch.Tensor,
        censoring_hazard: torch.Tensor,
        observed_time: torch.Tensor,
        event: torch.Tensor,
        boundary_times: torch.Tensor,
        eps: float = 1e-7,
        admin_end_threshold: float = 1.0 - 1e-6,
        admin_censoring_mode: str = "event_and_censor_survival",
        event_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if admin_censoring_mode != "event_and_censor_survival":
            raise ValueError("Only admin_censoring_mode='event_and_censor_survival' is currently supported.")
        event_hazard = event_hazard.clamp(min=eps, max=1.0 - eps)
        censoring_hazard = censoring_hazard.clamp(min=eps, max=1.0 - eps)
        observed_time = observed_time.to(event_hazard.device, event_hazard.dtype).view(-1).clamp(min=0.0, max=1.0)
        event = event.to(event_hazard.device, event_hazard.dtype).view(-1)
        boundary_times = boundary_times.to(event_hazard.device, event_hazard.dtype)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(event_hazard.shape[0], -1)
        if event_hazard.shape != censoring_hazard.shape or event_hazard.shape != boundary_times.shape:
            raise ValueError(
                "Hazards and boundary_times must have matching shape; "
                f"got event={tuple(event_hazard.shape)}, censor={tuple(censoring_hazard.shape)}, "
                f"times={tuple(boundary_times.shape)}."
            )
        b_idx = PhaseSynModel.survival_interval_indices(observed_time, boundary_times)
        log_surv_t_terms = torch.log1p(-event_hazard)
        log_surv_c_terms = torch.log1p(-censoring_hazard)
        zero = torch.zeros(event_hazard.shape[0], 1, device=event_hazard.device, dtype=event_hazard.dtype)
        log_s_t_prefix = torch.cat([zero, torch.cumsum(log_surv_t_terms, dim=1)[:, :-1]], dim=1)
        log_s_c_prefix = torch.cat([zero, torch.cumsum(log_surv_c_terms, dim=1)[:, :-1]], dim=1)
        row = torch.arange(event_hazard.shape[0], device=event_hazard.device)
        log_pi_t = log_s_t_prefix[row, b_idx] + torch.log(event_hazard[row, b_idx])
        log_pi_c = log_s_c_prefix[row, b_idx] + torch.log(censoring_hazard[row, b_idx])
        log_s_t = log_s_t_prefix[row, b_idx]
        log_s_c = log_s_c_prefix[row, b_idx]
        log_s_t_end = torch.cumsum(log_surv_t_terms, dim=1)[:, -1]
        log_s_c_end = torch.cumsum(log_surv_c_terms, dim=1)[:, -1]
        admin_threshold = torch.as_tensor(admin_end_threshold, device=event_hazard.device, dtype=event_hazard.dtype)
        admin_censored = (event <= 0.5) & (observed_time >= admin_threshold)
        loglik = torch.where(
            event > 0.5,
            log_pi_t + log_s_c,
            torch.where(admin_censored, log_s_t_end + log_s_c_end, log_s_t + log_pi_c),
        )
        weights = torch.where(event > 0.5, torch.as_tensor(float(event_weight), device=event.device, dtype=event.dtype), torch.ones_like(event))
        loss = -(weights * loglik).sum() / weights.sum().clamp(min=eps)
        return loss, {
            "dynamic_survival_loglik": loglik.detach(),
            "survival_interval_index": b_idx.detach(),
            "admin_censoring_mask": admin_censored.detach(),
            "admin_censoring_rate": admin_censored.to(event_hazard.dtype).detach().mean(),
            "event_hazard_mean": event_hazard.detach().mean(),
            "censoring_hazard_mean": censoring_hazard.detach().mean(),
            "event_hazard_min": event_hazard.detach().min(),
            "event_hazard_max": event_hazard.detach().max(),
            "censoring_hazard_min": censoring_hazard.detach().min(),
            "censoring_hazard_max": censoring_hazard.detach().max(),
            "survival_event_weight": torch.as_tensor(float(event_weight), device=event_hazard.device, dtype=event_hazard.dtype),
        }

    @staticmethod
    def dynamic_survival_auxiliary_loss(
        event_hazard: torch.Tensor,
        censoring_hazard: torch.Tensor,
        observed_time: torch.Tensor,
        event: torch.Tensor,
        boundary_times: torch.Tensor,
        event_weight: float = 1.0,
        eps: float = 1e-7,
    ) -> dict[str, torch.Tensor]:
        event_hazard = event_hazard.clamp(min=eps, max=1.0 - eps)
        censoring_hazard = censoring_hazard.clamp(min=eps, max=1.0 - eps)
        observed_time = observed_time.to(event_hazard.device, event_hazard.dtype).view(-1).clamp(min=0.0, max=1.0)
        event = event.to(event_hazard.device, event_hazard.dtype).view(-1)
        boundary_times = boundary_times.to(event_hazard.device, event_hazard.dtype)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(event_hazard.shape[0], -1)
        if event_hazard.shape != censoring_hazard.shape or event_hazard.shape != boundary_times.shape:
            raise ValueError(
                "Hazards and boundary_times must have matching shape; "
                f"got event={tuple(event_hazard.shape)}, censor={tuple(censoring_hazard.shape)}, "
                f"times={tuple(boundary_times.shape)}."
            )

        log_surv_t_terms = torch.log1p(-event_hazard)
        log_surv_c_terms = torch.log1p(-censoring_hazard)
        zero = torch.zeros(event_hazard.shape[0], 1, device=event_hazard.device, dtype=event_hazard.dtype)
        log_s_t_prefix = torch.cat([zero, torch.cumsum(log_surv_t_terms, dim=1)[:, :-1]], dim=1)
        log_s_c_prefix = torch.cat([zero, torch.cumsum(log_surv_c_terms, dim=1)[:, :-1]], dim=1)
        event_log_probs = log_s_t_prefix + torch.log(event_hazard) + log_s_c_prefix
        censor_log_probs = log_s_t_prefix + log_s_c_prefix + torch.log(censoring_hazard)
        tail_log_prob = (
            torch.cumsum(log_surv_t_terms, dim=1)[:, -1:]
            + torch.cumsum(log_surv_c_terms, dim=1)[:, -1:]
        )
        probs = torch.softmax(torch.cat([event_log_probs, censor_log_probs, tail_log_prob], dim=1), dim=1)
        n_intervals = event_hazard.shape[1]
        event_prob = probs[:, :n_intervals].sum(dim=1).clamp(min=eps, max=1.0 - eps)
        weights = torch.where(event > 0.5, torch.as_tensor(float(event_weight), device=event.device, dtype=event.dtype), torch.ones_like(event))
        event_bce = F.binary_cross_entropy(event_prob, event, weight=weights, reduction="sum") / weights.sum().clamp(min=eps)

        starts = torch.cat(
            [torch.zeros(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype), boundary_times[:, :-1]],
            dim=1,
        )
        midpoints = starts + 0.5 * (boundary_times - starts).clamp(min=eps)
        times = torch.cat([midpoints, midpoints, torch.ones(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype)], dim=1)
        expected_time = (probs * times).sum(dim=1)
        time_mse = F.mse_loss(expected_time, observed_time)
        return {
            "survival_event_aux_bce": event_bce,
            "survival_time_aux_mse": time_mse,
            "survival_event_probability_mean": event_prob.detach().mean(),
            "survival_expected_time_mean": expected_time.detach().mean(),
        }

    @staticmethod
    def dynamic_survival_distribution_summary(
        event_hazard: torch.Tensor,
        censoring_hazard: torch.Tensor,
        boundary_times: torch.Tensor,
        eps: float = 1e-7,
    ) -> dict[str, torch.Tensor]:
        event_hazard = event_hazard.clamp(min=eps, max=1.0 - eps)
        censoring_hazard = censoring_hazard.clamp(min=eps, max=1.0 - eps)
        boundary_times = boundary_times.to(device=event_hazard.device, dtype=event_hazard.dtype)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(event_hazard.shape[0], -1)
        if event_hazard.shape != censoring_hazard.shape or event_hazard.shape != boundary_times.shape:
            raise ValueError(
                "Hazards and boundary_times must have matching shape; "
                f"got event={tuple(event_hazard.shape)}, censor={tuple(censoring_hazard.shape)}, "
                f"times={tuple(boundary_times.shape)}."
            )
        log_surv_t_terms = torch.log1p(-event_hazard)
        log_surv_c_terms = torch.log1p(-censoring_hazard)
        zero = torch.zeros(event_hazard.shape[0], 1, device=event_hazard.device, dtype=event_hazard.dtype)
        log_s_t_prefix = torch.cat([zero, torch.cumsum(log_surv_t_terms, dim=1)[:, :-1]], dim=1)
        log_s_c_prefix = torch.cat([zero, torch.cumsum(log_surv_c_terms, dim=1)[:, :-1]], dim=1)
        event_mass = torch.exp(log_s_t_prefix + torch.log(event_hazard) + log_s_c_prefix)
        censor_mass = torch.exp(log_s_t_prefix + log_s_c_prefix + torch.log(censoring_hazard))
        tail_mass = torch.exp(
            torch.cumsum(log_surv_t_terms, dim=1)[:, -1:]
            + torch.cumsum(log_surv_c_terms, dim=1)[:, -1:]
        )
        starts = torch.cat(
            [torch.zeros(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype), boundary_times[:, :-1]],
            dim=1,
        )
        midpoints = starts + 0.5 * (boundary_times - starts).clamp(min=eps)
        event_probability = event_mass.sum(dim=1)
        total_mass = (event_mass.sum(dim=1, keepdim=True) + censor_mass.sum(dim=1, keepdim=True) + tail_mass).clamp(min=eps)
        expected_time = ((event_mass + censor_mass) * midpoints).sum(dim=1, keepdim=True) + tail_mass
        expected_time = (expected_time / total_mass).view(-1)
        return {
            "event_probability": event_probability,
            "expected_time": expected_time,
        }

    @staticmethod
    def _sample_time_from_hazard(
        hazard: torch.Tensor,
        boundary_times: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eps = 1e-7
        hazard = hazard.clamp(min=eps, max=1.0 - eps)
        boundary_times = boundary_times.to(device=hazard.device, dtype=hazard.dtype)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(hazard.shape[0], -1)
        if boundary_times.shape != hazard.shape:
            raise ValueError(
                "hazard and boundary_times must have matching shape for dynamic survival sampling; "
                f"got hazard={tuple(hazard.shape)}, times={tuple(boundary_times.shape)}."
            )
        log_surv_terms = torch.log1p(-hazard)
        zero = torch.zeros(hazard.shape[0], 1, device=hazard.device, dtype=hazard.dtype)
        log_surv_prefix = torch.cat([zero, torch.cumsum(log_surv_terms, dim=1)[:, :-1]], dim=1)
        probs = torch.exp(log_surv_prefix) * hazard
        tail = torch.exp(torch.cumsum(log_surv_terms, dim=1)[:, -1:])
        probs_ext = torch.cat([probs, tail], dim=1).clamp(min=eps)
        probs_ext = probs_ext / probs_ext.sum(dim=1, keepdim=True).clamp(min=eps)
        if deterministic:
            idx = torch.argmax(probs_ext, dim=1)
        else:
            idx = torch.multinomial(probs_ext, num_samples=1).squeeze(1)
        row = torch.arange(boundary_times.shape[0], device=boundary_times.device)
        n_intervals = boundary_times.shape[1]
        interval_idx = idx.clamp(max=n_intervals - 1)
        starts = torch.cat(
            [torch.zeros(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype), boundary_times[:, :-1]],
            dim=1,
        )[row, interval_idx]
        ends = boundary_times[row, interval_idx]
        width = (ends - starts).clamp(min=eps)
        if deterministic:
            frac = torch.full_like(starts, 0.5)
        else:
            frac = torch.rand_like(starts)
        sampled = starts + frac * width
        tail_mask = idx >= n_intervals
        sampled = torch.where(tail_mask, torch.ones_like(sampled), sampled)
        return sampled.view(-1, 1), idx.view(-1, 1), tail_mask.view(-1, 1)

    @staticmethod
    def _deterministic_dynamic_survival_outcome(
        event_hazard: torch.Tensor,
        censoring_hazard: torch.Tensor,
        boundary_times: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        eps = 1e-7
        event_hazard = event_hazard.clamp(min=eps, max=1.0 - eps)
        censoring_hazard = censoring_hazard.clamp(min=eps, max=1.0 - eps)
        boundary_times = boundary_times.to(device=event_hazard.device, dtype=event_hazard.dtype)
        if boundary_times.dim() == 1:
            boundary_times = boundary_times.unsqueeze(0).expand(event_hazard.shape[0], -1)
        if event_hazard.shape != censoring_hazard.shape or event_hazard.shape != boundary_times.shape:
            raise ValueError(
                "hazards and boundary_times must match for deterministic dynamic survival sampling; "
                f"got event={tuple(event_hazard.shape)}, censor={tuple(censoring_hazard.shape)}, "
                f"times={tuple(boundary_times.shape)}."
            )

        log_surv_t_terms = torch.log1p(-event_hazard)
        log_surv_c_terms = torch.log1p(-censoring_hazard)
        zero = torch.zeros(event_hazard.shape[0], 1, device=event_hazard.device, dtype=event_hazard.dtype)
        log_s_t_prefix = torch.cat([zero, torch.cumsum(log_surv_t_terms, dim=1)[:, :-1]], dim=1)
        log_s_c_prefix = torch.cat([zero, torch.cumsum(log_surv_c_terms, dim=1)[:, :-1]], dim=1)
        event_log_probs = log_s_t_prefix + torch.log(event_hazard) + log_s_c_prefix
        censor_log_probs = log_s_t_prefix + log_s_c_prefix + torch.log(censoring_hazard)
        tail_log_prob = (
            torch.cumsum(log_surv_t_terms, dim=1)[:, -1:]
            + torch.cumsum(log_surv_c_terms, dim=1)[:, -1:]
        )
        choice = torch.argmax(torch.cat([event_log_probs, censor_log_probs, tail_log_prob], dim=1), dim=1)

        n_intervals = boundary_times.shape[1]
        row = torch.arange(boundary_times.shape[0], device=boundary_times.device)
        is_event = choice < n_intervals
        is_censor = (choice >= n_intervals) & (choice < 2 * n_intervals)
        interval_idx = torch.where(is_event, choice, choice - n_intervals).clamp(min=0, max=n_intervals - 1)
        starts = torch.cat(
            [torch.zeros(boundary_times.shape[0], 1, device=boundary_times.device, dtype=boundary_times.dtype), boundary_times[:, :-1]],
            dim=1,
        )[row, interval_idx]
        ends = boundary_times[row, interval_idx]
        midpoint = starts + 0.5 * (ends - starts).clamp(min=eps)
        tail_idx = torch.full_like(interval_idx, n_intervals)

        event_time = torch.where(is_event, midpoint, torch.full_like(midpoint, float("inf")))
        censoring_time = torch.where(is_event | is_censor, midpoint, torch.ones_like(midpoint))
        observed_time = torch.where(is_event | is_censor, midpoint, torch.ones_like(midpoint))
        event_interval_index = torch.where(is_event, interval_idx, tail_idx)
        censoring_interval_index = torch.where(is_event | is_censor, interval_idx, tail_idx)
        return {
            "event_time": event_time.view(-1, 1),
            "censoring_time": censoring_time.view(-1, 1),
            "observed_time": observed_time.view(-1, 1),
            "event": is_event.to(event_hazard.dtype).view(-1, 1),
            "event_interval_index": event_interval_index.view(-1, 1),
            "censoring_interval_index": censoring_interval_index.view(-1, 1),
            "event_tail": (~is_event).view(-1, 1),
            "censoring_tail": (~is_censor).view(-1, 1),
        }

    def sample_dynamic_survival(
        self,
        survival_out: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        boundary_times = survival_out["boundary_times"]
        if deterministic:
            return self._deterministic_dynamic_survival_outcome(
                survival_out["event_hazard"],
                survival_out["censoring_hazard"],
                boundary_times,
            )
        event_time, event_interval_index, event_tail = self._sample_time_from_hazard(
            survival_out["event_hazard"],
            boundary_times,
            deterministic,
        )
        censoring_time, censoring_interval_index, censoring_tail = self._sample_time_from_hazard(
            survival_out["censoring_hazard"],
            boundary_times,
            deterministic,
        )
        event_time = torch.where(event_tail, torch.full_like(event_time, float("inf")), event_time)
        censoring_time = torch.where(censoring_tail, torch.ones_like(censoring_time), censoring_time)
        observed_time = torch.minimum(event_time, censoring_time)
        event = (event_time <= censoring_time).to(event_time.dtype)
        return {
            "event_time": event_time,
            "censoring_time": censoring_time,
            "observed_time": observed_time,
            "event": event,
            "event_interval_index": event_interval_index,
            "censoring_interval_index": censoring_interval_index,
            "event_tail": event_tail,
            "censoring_tail": censoring_tail,
        }

    def treatment_context(self, treatment: torch.Tensor | None, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.treatment_dim <= 0:
            return torch.zeros(batch_size, 0, device=device, dtype=dtype)
        if treatment is None:
            raise ValueError("Treatment assignment A is required for treatment-explicit PhaseSyn generation/training.")
        a = treatment.to(device=device, dtype=dtype)
        if a.dim() == 0:
            a = a.view(1).expand(batch_size)
        elif a.dim() == 1 and a.numel() == 1 and batch_size > 1:
            a = a.expand(batch_size)
        if a.dim() == 1:
            a_long = a.long().clamp(min=0, max=self.treatment_dim - 1)
            a = F.one_hot(a_long, num_classes=self.treatment_dim).to(device=device, dtype=dtype)
        if a.shape != (batch_size, self.treatment_dim):
            raise ValueError(f"Treatment tensor must have shape {(batch_size, self.treatment_dim)}, got {tuple(a.shape)}.")
        return a

    def _generation_time_grid(
        self,
        time_grid: torch.Tensor,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        times = torch.as_tensor(time_grid, device=device, dtype=dtype)
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1).clone()
        elif times.dim() == 2 and times.shape[0] == 1 and batch_size > 1:
            times = times.expand(batch_size, -1).clone()
        elif times.dim() != 2 or times.shape[0] != batch_size:
            raise ValueError(f"time_grid must have shape (T,), (1,T), or (n,T); got {tuple(times.shape)}.")
        if times.shape[1] == 0:
            raise ValueError("time_grid must contain at least one requested generation time.")
        return times

    @staticmethod
    def _normalization_pair(
        normalization_params: list[Any] | None,
        feature_idx: int,
        default: tuple[float, float],
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = default if normalization_params is None else normalization_params[feature_idx]
        first = torch.as_tensor(pair[0], device=device, dtype=dtype)
        second = torch.as_tensor(pair[1], device=device, dtype=dtype)
        return first, second

    def sample_prior_latents(
        self,
        n: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        if n <= 0:
            raise ValueError("n must be positive.")
        param = next(self.parameters())
        device = param.device if device is None else device
        dtype = param.dtype if dtype is None else dtype
        component = torch.randint(self.hivae.s_dim, (int(n),), device=device)
        s = F.one_hot(component, num_classes=self.hivae.s_dim).to(device=device, dtype=dtype)
        z_mean = self.hivae.z_distribution_layer(s)
        z = z_mean + torch.randn_like(z_mean)
        return {"component": component, "s": s, "z": z, "z_prior_mean": z_mean}

    def _prior_decoder_params(self, z: torch.Tensor, s: torch.Tensor) -> tuple[dict[str, torch.Tensor], list[Any]]:
        samples = {"z": z, "s": s, "y": self.hivae.y_layer(z)}
        grouped_y = data_processing.y_partition(samples["y"], self.hivae.feat_types_list, self.hivae.y_dim_partition)
        miss = torch.zeros(z.shape[0], len(self.hivae.feat_types_list), device=z.device, dtype=z.dtype)
        theta_view = self.hivae.get_theta_view()
        if not all(f"feat_{idx}" in theta_view for idx in range(len(self.hivae.feat_types_list))):
            theta_view = {
                f"feat_{idx}": theta_view[f"feat_{full_idx}"]
                for idx, full_idx in enumerate(self.full_to_baseline_indices)
            }
        theta = theta_estimation.theta_estimation_from_ys(
            grouped_y,
            s,
            self.hivae.feat_types_list,
            miss,
            theta_view,
        )
        return samples, theta

    def _sample_baseline_feature(
        self,
        feature: dict[str, Any],
        params,
        normalization_pair: tuple[torch.Tensor, torch.Tensor],
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ftype = feature["type"]
        if ftype == "real":
            data_mean, data_var = normalization_pair
            mean, var_raw = params
            var = data_var * F.softplus(var_raw).clamp(min=1e-3, max=1e20)
            mean = torch.sqrt(data_var) * mean + data_mean
            value = mean if deterministic else torch.normal(mean, torch.sqrt(var))
            return value, value
        if ftype == "pos":
            data_mean_log, data_var_log = normalization_pair
            mean, var_raw = params
            var = data_var_log * F.softplus(var_raw).clamp(min=1e-3, max=1.0)
            mean = torch.sqrt(data_var_log) * mean + data_mean_log
            log_value = mean if deterministic else torch.normal(mean, torch.sqrt(var))
            value = torch.exp(log_value).sub(1.0).clamp(min=0.0)
            return value, value
        if ftype == "count":
            rate = F.softplus(params).clamp(min=1e-6, max=1e20)
            value = rate if deterministic else torch.distributions.Poisson(rate).sample()
            return value, value
        if ftype == "cat":
            nclass = int(feature["nclass"])
            if deterministic:
                cls = torch.argmax(params, dim=1)
            else:
                cls = torch.distributions.Categorical(logits=params).sample()
            one_hot = F.one_hot(cls, num_classes=nclass).to(device=params.device, dtype=params.dtype)
            return one_hot, cls.to(device=params.device, dtype=params.dtype).view(-1, 1)
        if ftype == "ordinal":
            partition_param, mean_param = params
            epsilon = 1e-6
            batch_size = mean_param.shape[0]
            nclass = int(feature["nclass"])
            theta_values = torch.cumsum(F.softplus(partition_param).clamp(min=epsilon, max=1e20), dim=1)
            sigmoid_est_mean = torch.sigmoid(theta_values - mean_param.view(-1, 1))
            probs = torch.cat([sigmoid_est_mean, torch.ones((batch_size, 1), device=mean_param.device)], dim=1) - torch.cat([
                torch.zeros((batch_size, 1), device=mean_param.device),
                sigmoid_est_mean,
            ], dim=1)
            probs = probs.clamp(min=epsilon, max=1.0)
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=epsilon)
            cls = torch.argmax(probs, dim=1) if deterministic else torch.distributions.Categorical(probs=probs).sample()
            encoded = (
                torch.arange(nclass, device=mean_param.device).unsqueeze(0) < (cls + 1).unsqueeze(1)
            ).to(dtype=mean_param.dtype)
            return encoded, cls.to(device=mean_param.device, dtype=mean_param.dtype).view(-1, 1)
        raise ValueError(f"Unsupported baseline feature type for prior generation: {ftype}")

    def _longitudinal_l0_from_generated_features(self, generated_by_name: dict[str, torch.Tensor]) -> torch.Tensor:
        l0_parts = []
        for spec in self.decoder.specs:
            if spec.name not in generated_by_name:
                raise ValueError(
                    f"Cannot initialize prior longitudinal trajectory: generated baseline lacks L0 feature {spec.name!r}."
                )
            value = generated_by_name[spec.name].view(-1, 1)
            if spec.type in {"real", "pos", "count"}:
                value = (value - float(spec.mean)) / max(float(spec.std), 1e-6)
            l0_parts.append(value)
        return torch.cat(l0_parts, dim=1)

    def _inverse_longitudinal_tensor(self, values: torch.Tensor) -> torch.Tensor:
        out = values.clone()
        for idx, spec in enumerate(self.decoder.specs):
            if spec.type in {"real", "pos", "count"}:
                out[:, :, idx] = out[:, :, idx] * float(spec.std) + float(spec.mean)
            if spec.type == "pos":
                out[:, :, idx] = out[:, :, idx].clamp(min=0.0)
            elif spec.type == "count":
                out[:, :, idx] = out[:, :, idx].clamp(min=0.0).round()
            elif spec.type in {"cat", "ordinal"}:
                out[:, :, idx] = out[:, :, idx].round().clamp(min=0.0, max=float((spec.nclass or 2) - 1))
        return out

    def generate_prior_cohort(
        self,
        n: int,
        treatment: torch.Tensor | int | float,
        time_grid: torch.Tensor,
        normalization_params: list[Any] | None = None,
        deterministic: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate a complete synthetic cohort from the learned prior under fixed treatment A.

        This is the prior-based generation mode from docs/method.md. It never encodes an
        observed baseline and it has no argument for observed post-baseline outcomes.
        """
        param = next(self.parameters())
        device = param.device if device is None else device
        dtype = param.dtype if dtype is None else dtype
        n = int(n)
        times = self._generation_time_grid(time_grid, n, device, dtype)
        latents = self.sample_prior_latents(n, device=device, dtype=dtype)
        z, s = latents["z"], latents["s"]
        _, theta = self._prior_decoder_params(z, s)

        encoded_parts: list[torch.Tensor] = []
        scalar_parts: list[torch.Tensor] = []
        encoded_baseline_parts: list[torch.Tensor] = []
        scalar_baseline_parts: list[torch.Tensor] = []
        generated_by_name: dict[str, torch.Tensor] = {}
        for idx, feature in enumerate(self.hivae.feat_types_list):
            norm_idx = self.full_to_baseline_indices[idx] if normalization_params is not None and len(normalization_params) == len(self.full_feat_types_list) else idx
            norm = self._normalization_pair(normalization_params, norm_idx, (0.0, 1.0), device, dtype)
            encoded, scalar = self._sample_baseline_feature(feature, theta[idx], norm, deterministic)
            encoded_parts.append(encoded.to(device=device, dtype=dtype))
            scalar = scalar.to(device=device, dtype=dtype)
            scalar_parts.append(scalar)
            encoded_baseline_parts.append(encoded.to(device=device, dtype=dtype))
            scalar_baseline_parts.append(scalar)
            generated_by_name[str(feature["name"])] = scalar[:, :1]

        baseline_encoded_without_survival = torch.cat(encoded_baseline_parts, dim=1)
        baseline_values_without_survival = torch.cat(scalar_baseline_parts, dim=1)
        l0 = self._longitudinal_l0_from_generated_features(generated_by_name).to(device=device, dtype=dtype)
        a = self.treatment_context(torch.as_tensor(treatment, device=device), n, device, dtype)
        u0, u0_diag = self.sample_u0_from_l0(z, s, l0, deterministic=deterministic, return_details=True)
        survival_times = self._survival_interval_times(n, device, dtype)
        survival_start_times = self._survival_interval_start_times(survival_times)
        self.validate_shared_time_normalization(times, survival_start_times)
        union_times = torch.unique(torch.cat([times.reshape(-1), survival_start_times.reshape(-1)]))
        union_times = union_times.sort().values.unsqueeze(0).expand(n, -1).clone()
        union_path = self.integrate_path(u0, union_times, z, s, a)
        u_path = self._select_path_at_times(union_path, union_times, times)
        u_start = self._select_path_at_times(union_path, union_times, survival_start_times)
        survival_out = self.dynamic_survival_from_interval_start_path(u_start, survival_start_times, survival_times, z, s, u0, a)
        survival_out["u_interval_start"] = u_start
        survival_sample = self.sample_dynamic_survival(survival_out, deterministic=deterministic)
        event_time_normalized = survival_sample["event_time"]
        censoring_time_normalized = survival_sample["censoring_time"]
        observed_time_normalized = survival_sample["observed_time"]
        event_time = self.denormalize_survival_time(event_time_normalized)
        censoring_time = self.denormalize_survival_time(censoring_time_normalized)
        observed_time = self.denormalize_survival_time(observed_time_normalized)
        event = survival_sample["event"]
        observed_event_pair = torch.cat([observed_time, event], dim=1)

        encoded_full_parts: list[torch.Tensor] = []
        values_full_parts: list[torch.Tensor] = []
        base_pos = 0
        for feature in self.full_feat_types_list:
            if feature["type"].startswith("surv"):
                encoded_full_parts.append(observed_event_pair)
                values_full_parts.append(observed_event_pair)
            else:
                encoded_full_parts.append(encoded_parts[base_pos])
                values_full_parts.append(scalar_parts[base_pos])
                base_pos += 1
        baseline_encoded = torch.cat(encoded_full_parts, dim=1)
        baseline_values = torch.cat(values_full_parts, dim=1)

        longitudinal_norm = self.decoder.sample_from_path_conditioned(u_path, times, z, s, a, deterministic=deterministic)
        t0_rows = times.abs() <= self.baseline_time_eps
        if t0_rows.any():
            replace = t0_rows.unsqueeze(-1).expand_as(longitudinal_norm)
            longitudinal_norm = torch.where(replace, l0.unsqueeze(1).expand_as(longitudinal_norm), longitudinal_norm)
        longitudinal_raw = self._inverse_longitudinal_tensor(longitudinal_norm)
        available = times <= observed_time_normalized.to(times.device, times.dtype)
        longitudinal_norm = longitudinal_norm.masked_fill(~available.unsqueeze(-1), float("nan"))
        longitudinal_raw = longitudinal_raw.masked_fill(~available.unsqueeze(-1), float("nan"))

        if a.shape[1] > 0:
            treatment_class = torch.argmax(a, dim=1)
        else:
            treatment_class = torch.zeros(n, device=device, dtype=torch.long)
        return {
            **latents,
            "baseline_encoded": baseline_encoded,
            "baseline_values": baseline_values,
            "baseline_encoded_without_survival": baseline_encoded_without_survival,
            "baseline_values_without_survival": baseline_values_without_survival,
            "L0": l0,
            "treatment": treatment_class,
            "treatment_context": a,
            "event_time": event_time,
            "censoring_time": censoring_time,
            "observed_time": observed_time,
            "event_time_normalized": event_time_normalized,
            "censoring_time_normalized": censoring_time_normalized,
            "observed_time_normalized": observed_time_normalized,
            "event": event,
            "time_grid": times,
            "longitudinal_available": available,
            "u0": u0,
            "u0_mu": u0_diag["u0_mu"],
            "u0_sigma": u0_diag["u0_sigma"],
            "u0_sample": u0_diag["u0_sample"],
            "u_path": u_path,
            "union_time_grid": union_times,
            "union_u_path": union_path,
            "dynamic_survival": survival_out,
            "event_hazard": survival_out["event_hazard"],
            "censoring_hazard": survival_out["censoring_hazard"],
            "longitudinal_values": longitudinal_raw,
            "longitudinal_values_normalized": longitudinal_norm,
            "uses_observed_future_outcomes": torch.zeros((), device=device, dtype=torch.bool),
            "baseline_generated_from_prior": torch.ones((), device=device, dtype=torch.bool),
        }

    def encoder_miss_mask(self, batch_miss: torch.Tensor) -> torch.Tensor:
        if batch_miss.shape[1] == len(self.full_feat_types_list):
            batch_miss = batch_miss[:, self.full_to_baseline_indices]
        if batch_miss.shape[1] != len(self.hivae.feat_types_list):
            raise ValueError(
                "PhaseSyn baseline encoder expects non-survival baseline masks; "
                f"got {tuple(batch_miss.shape)} for {len(self.hivae.feat_types_list)} baseline features."
            )
        if self.encoder_conditioning != "baseline_only":
            raise ValueError(f"Unsupported encoder_conditioning: {self.encoder_conditioning}")
        return batch_miss.clone()

    def split_longitudinal_batch(self, times: torch.Tensor, values: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
        return split_baseline_future_longitudinal(times, values, masks, self.baseline_time_eps)

    def u0_params_from_l0(self, z: torch.Tensor, s: torch.Tensor, l0: torch.Tensor, m0: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return p_eta(u0 | z, s, L0) parameters without using future outcomes.

        The initial ODE state is sampled from a lightweight baseline-conditioned
        Gaussian p_eta(u0 | z, s, L0). This preserves the baseline-only generation
        chronology while allowing subject-level latent heterogeneity in future
        trajectories.
        """
        if self.detach_l0_for_u0_init:
            l0 = l0.detach()
        if m0 is not None and bool((m0 < 1.0).any().item()):
            raise ValueError("The ODE initializer expects fully observed L0 and no mask-based missing fill.")
        l0_input = self.l0_encoder(l0) if self.l0_encoder is not None else l0
        u0_mu = self.u0_initializer(z, s, l0_input)
        if self.u0_sigma_mode == "learned":
            if self.u0_logsigma_head is None:
                raise RuntimeError("u0_sigma_mode='learned' requires u0_logsigma_head.")
            u0_logsigma_raw = self.u0_logsigma_head(z, s, l0_input)
            u0_sigma = self.u0_sigma_min + F.softplus(u0_logsigma_raw)
        else:
            u0_logsigma_raw = torch.zeros_like(u0_mu)
            u0_sigma = torch.full_like(u0_mu, self.u0_fixed_sigma)
        return {
            "u0_mu": u0_mu,
            "u0_logsigma_raw": u0_logsigma_raw,
            "u0_sigma": u0_sigma,
        }

    def _use_u0_mean(self, deterministic: bool | None) -> bool:
        if not self.stochastic_u0:
            return True
        if deterministic is not None:
            return bool(deterministic)
        if not self.training and self.use_u0_mean_at_eval:
            return True
        return False

    def sample_u0_from_l0(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        l0: torch.Tensor,
        m0: torch.Tensor | None = None,
        deterministic: bool | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        params = self.u0_params_from_l0(z, s, l0, m0)
        u0_mu = params["u0_mu"]
        u0_sigma = params["u0_sigma"]
        if self._use_u0_mean(deterministic):
            u0 = u0_mu
        else:
            u0 = u0_mu + u0_sigma * torch.randn_like(u0_mu)
        details = {
            **params,
            "u0_sample": u0,
            "u0_sigma_mean": u0_sigma.detach().mean(),
            "u0_var_regularization": u0_sigma.pow(2).mean(),
        }
        if return_details:
            return u0, details
        return u0

    def init_u0_from_l0(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        l0: torch.Tensor,
        m0: torch.Tensor | None = None,
        deterministic: bool | None = None,
    ) -> torch.Tensor:
        return self.sample_u0_from_l0(z, s, l0, m0=m0, deterministic=deterministic, return_details=False)

    def integrate_path(
        self,
        u0: torch.Tensor,
        times: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.ode_func.set_context(z, s, a)
        try:
            return rk4_integrate(self.ode_func, u0, times)
        finally:
            self.ode_func.set_context(None, None, None)

    def longitudinal_loss_0plus(
        self,
        u0: torch.Tensor,
        u_path: torch.Tensor,
        future_times: torch.Tensor,
        future_values: torch.Tensor,
        future_masks: torch.Tensor,
        l0: torch.Tensor,
        m0: torch.Tensor,
        z: torch.Tensor | None = None,
        s: torch.Tensor | None = None,
        a: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        gamma0 = torch.as_tensor(self.baseline_long_weight, device=u0.device, dtype=u0.dtype)
        baseline_times = torch.zeros(u0.shape[0], 1, device=u0.device, dtype=u0.dtype)
        baseline_path = u0.unsqueeze(1)
        baseline_mask = torch.ones_like(l0, device=u0.device, dtype=u0.dtype).unsqueeze(1)
        baseline_sum, base_aux = self.longitudinal_baseline_decoder.loss_sum_from_path_conditioned(
            baseline_path,
            baseline_times,
            l0.to(u0.device, u0.dtype).unsqueeze(1),
            baseline_mask,
            z,
            s,
        )
        future_sum, future_aux = self.decoder.loss_sum_from_path_conditioned(
            u_path,
            future_times.to(device=u0.device, dtype=u0.dtype),
            future_values.to(device=u0.device, dtype=u0.dtype),
            future_masks.to(device=u0.device, dtype=u0.dtype),
            z,
            s,
            a,
        )
        baseline_count = gamma0 * torch.as_tensor(l0.shape[0] * l0.shape[1], device=u0.device, dtype=u0.dtype)
        future_count = future_masks.to(device=u0.device, dtype=u0.dtype).sum()
        denom = (baseline_count + future_count).clamp(min=1.0)
        loss = (gamma0 * baseline_sum + future_sum) / denom
        cont_sq = gamma0 * base_aux["long_sq_sum"] + future_aux["long_sq_sum"]
        cont_count = (gamma0 * base_aux["long_cont_count"] + future_aux["long_cont_count"]).clamp(min=1.0)
        aux = {
            "long_rmse_norm": torch.sqrt(cont_sq / cont_count),
            "longitudinal_baseline_nll": baseline_sum / torch.as_tensor(l0.shape[0] * l0.shape[1], device=u0.device, dtype=u0.dtype).clamp(min=1.0),
            "longitudinal_future_nll": future_sum / future_count.clamp(min=1.0),
            "baseline_long_weight": gamma0,
        }
        return loss, aux

    def encode_static_posterior(
        self,
        batch_data_observed,
        batch_data,
        batch_miss: torch.Tensor,
        tau: float = 1e-3,
        n_generated_dataset: int = 1,
        encoder_l0: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        original_miss = batch_miss
        batch_data_observed, batch_miss = self._select_baseline_features(batch_data_observed, original_miss)
        batch_data, _ = self._select_baseline_features(batch_data, original_miss)
        return self.hivae.forward(
            batch_data_observed,
            batch_data,
            batch_miss,
            tau=tau,
            n_generated_dataset=n_generated_dataset,
            encoder_batch_miss=self.encoder_miss_mask(batch_miss),
            encoder_l0=encoder_l0,
        )

    def _encoder_input_tensor(self, batch_data_observed, batch_miss: torch.Tensor, encoder_l0: torch.Tensor) -> torch.Tensor:
        batch_data_observed, batch_miss = self._select_baseline_features(batch_data_observed, batch_miss)
        encoder_miss = self.encoder_miss_mask(batch_miss)
        if self.hivae._global_norm_params is not None:
            x_list, _ = data_processing.batch_normalization_frozen(
                batch_data_observed,
                self.hivae.feat_types_list,
                encoder_miss,
                self.hivae._global_norm_params,
            )
        else:
            x_list, _ = data_processing.batch_normalization(
                batch_data_observed,
                self.hivae.feat_types_list,
                encoder_miss,
            )
        x = torch.cat(x_list, dim=1)
        encoder_l0 = encoder_l0.to(device=x.device, dtype=x.dtype)
        return torch.cat([x, encoder_miss.to(device=x.device, dtype=x.dtype), encoder_l0], dim=1)

    def deterministic_latents_from_encoder_input(self, batch_data_observed, batch_miss: torch.Tensor, encoder_l0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._encoder_input_tensor(batch_data_observed, batch_miss, encoder_l0)
        logits_s = self.hivae.s_layer(x)
        s = F.softmax(logits_s, dim=-1)
        mean_qz, _ = torch.chunk(self.hivae.z_layer(torch.cat([x, s], dim=1)), 2, dim=1)
        return mean_qz, s

    def encode_baseline_latent(
        self,
        B,
        mask_B: torch.Tensor,
        L0: torch.Tensor,
        deterministic: bool = True,
        tau: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        """Encode q_phi^B(s,z | B, mask_B, L0); treatment A is intentionally not an argument."""
        x = self._encoder_input_tensor(B, mask_B, L0)
        logits_s = self.hivae.s_layer(x)
        probs_s = F.softmax(logits_s, dim=-1)
        if deterministic:
            z_mean, z_logvar = torch.chunk(self.hivae.z_layer(torch.cat([x, probs_s], dim=1)), 2, dim=1)
            z = z_mean
            s = probs_s
        else:
            q_params, samples = self.hivae.encode(x, tau)
            z_mean, z_logvar = q_params["z"]
            z = samples["z"]
            s = samples["s"]
        means = []
        for k in range(probs_s.shape[1]):
            s_k = torch.zeros_like(probs_s)
            s_k[:, k] = 1.0
            mean_k, _ = torch.chunk(self.hivae.z_layer(torch.cat([x, s_k], dim=1)), 2, dim=1)
            means.append(mean_k)
        zbar = torch.sum(probs_s.unsqueeze(-1) * torch.stack(means, dim=1), dim=1)
        return {
            "encoder_input": x,
            "s_logits": logits_s,
            "s_probs": probs_s,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "z": z,
            "s": s,
            "zbar": zbar,
        }

    def sample_latents_from_encoder_input(
        self,
        batch_data_observed,
        batch_miss: torch.Tensor,
        encoder_l0: torch.Tensor,
        tau: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        x = self._encoder_input_tensor(batch_data_observed, batch_miss, encoder_l0)
        q_params, samples = self.hivae.encode(x, tau)
        return samples["z"], samples["s"], {"q_params": q_params, "samples": samples}

    def deterministic_latents_from_hivae_result(self, hres: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        logits_s = hres["q_params"]["s"]
        s = F.one_hot(torch.argmax(logits_s, dim=-1), num_classes=logits_s.shape[-1]).to(dtype=logits_s.dtype)
        if "encoder_input" not in hres:
            mean_qz, _ = hres["q_params"]["z"]
            return mean_qz, s
        x = hres["encoder_input"]
        mean_qz, _ = torch.chunk(self.hivae.z_layer(torch.cat([x, s], dim=1)), 2, dim=1)
        return mean_qz, s

    def posterior_z_bar(self, hres: dict[str, Any]) -> torch.Tensor:
        """Return E_q[z | B, mask_B, L0] marginalizing over q(s | B, mask_B, L0)."""
        if "encoder_input" not in hres:
            mean_qz, _ = hres["q_params"]["z"]
            return mean_qz
        x = hres["encoder_input"]
        logits_s = hres["q_params"]["s"]
        probs_s = F.softmax(logits_s, dim=-1)
        means = []
        for k in range(probs_s.shape[1]):
            s_k = torch.zeros_like(probs_s)
            s_k[:, k] = 1.0
            mean_k, _ = torch.chunk(self.hivae.z_layer(torch.cat([x, s_k], dim=1)), 2, dim=1)
            means.append(mean_k)
        z_means = torch.stack(means, dim=1)
        return torch.sum(probs_s.unsqueeze(-1) * z_means, dim=1)

    def randomization_loss(
        self,
        hres: dict[str, Any],
        treatment: torch.Tensor,
        z_for_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits_s = hres["q_params"]["s"]
        z_bar = self.posterior_z_bar(hres) if self.randomization_loss_on == "z_mean" or z_for_loss is None else z_for_loss
        a = self.treatment_context(treatment, z_bar.shape[0], z_bar.device, z_bar.dtype)
        if a.shape[1] < 2:
            raise ValueError("Randomization loss currently expects binary treatment with two one-hot columns.")
        # G is the scalar treatment label derived from one-hot A.
        arm = torch.argmax(a.detach(), dim=1)
        treated = arm == 1
        control = arm == 0
        n_treated = treated.sum()
        n_control = control.sum()
        if int(n_treated.item()) == 0 or int(n_control.item()) == 0:
            zero = torch.zeros((), device=z_bar.device, dtype=z_bar.dtype)
            probs = F.softmax(logits_s.detach(), dim=-1)
            return zero, {
                "L_rand": zero,
                "MMD_z_bar_given_A": zero,
                "z_treatment_control_mean_distance": zero,
                "randomization_treated_count": n_treated.to(z_bar.dtype),
                "randomization_control_count": n_control.to(z_bar.dtype),
                "randomization_skipped": torch.ones((), device=z_bar.device, dtype=z_bar.dtype),
                "treatment_auc_from_z_bar": self._detached_treatment_auc(z_bar.detach(), arm.detach()),
                "s_mixture_treated": probs[treated].mean(dim=0) if bool(treated.any().item()) else torch.zeros(probs.shape[1], device=z_bar.device),
                "s_mixture_control": probs[control].mean(dim=0) if bool(control.any().item()) else torch.zeros(probs.shape[1], device=z_bar.device),
            }
        mmd = multi_rbf_mmd2(z_bar[treated], z_bar[control], self.randomization_mmd_bandwidths)
        mean_dist = (z_bar[treated].mean(dim=0) - z_bar[control].mean(dim=0)).norm()
        probs = F.softmax(logits_s.detach(), dim=-1)
        return mmd, {
            "L_rand": mmd,
            "MMD_z_bar_given_A": mmd.detach(),
            "z_treatment_control_mean_distance": mean_dist.detach(),
            "randomization_treated_count": n_treated.to(z_bar.dtype),
            "randomization_control_count": n_control.to(z_bar.dtype),
            "randomization_skipped": torch.zeros((), device=z_bar.device, dtype=z_bar.dtype),
            "treatment_auc_from_z_bar": self._detached_treatment_auc(z_bar.detach(), arm.detach()),
            "s_mixture_treated": probs[treated].mean(dim=0),
            "s_mixture_control": probs[control].mean(dim=0),
        }

    @staticmethod
    def _detached_treatment_auc(z: torch.Tensor, arm: torch.Tensor) -> torch.Tensor:
        pos = z[arm == 1]
        neg = z[arm == 0]
        if pos.shape[0] == 0 or neg.shape[0] == 0:
            return torch.tensor(float("nan"), device=z.device, dtype=z.dtype)
        score = z.mean(dim=1)
        pos_score = score[arm == 1]
        neg_score = score[arm == 0]
        pair = pos_score[:, None] - neg_score[None, :]
        auc = (pair > 0).to(z.dtype).mean() + 0.5 * (pair == 0).to(z.dtype).mean()
        return auc.detach()

    def generate_future_from_baseline(
        self,
        batch_data,
        batch_miss: torch.Tensor,
        l0: torch.Tensor,
        future_times: torch.Tensor,
        treatment: torch.Tensor,
        survival_interval_grid: torch.Tensor | None = None,
        deterministic_latents: bool = True,
        deterministic_u0: bool | None = None,
        tau: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        if len(batch_data) != len(self.hivae.feat_types_list) or batch_miss.shape[1] != len(self.hivae.feat_types_list):
            raise ValueError(
                "generate_future_from_baseline expects baseline-only W/L0 feature tensors; "
                f"got {len(batch_data)} tensors and mask shape {tuple(batch_miss.shape)}."
            )
        observed = [d * batch_miss[:, i].view(batch_miss.shape[0], 1) for i, d in enumerate(batch_data)]
        if deterministic_latents:
            z, s = self.deterministic_latents_from_encoder_input(observed, batch_miss, l0)
            hres: dict[str, Any] = {"samples": {"z": z, "s": s}}
        else:
            z, s, hres = self.sample_latents_from_encoder_input(observed, batch_miss, l0, tau=tau)
        z = z.to(l0.device, l0.dtype)
        s = s.to(l0.device, l0.dtype)
        a = self.treatment_context(treatment, l0.shape[0], l0.device, l0.dtype)
        u0, u0_diag = self.sample_u0_from_l0(z, s, l0, deterministic=deterministic_u0, return_details=True)
        batch_size = l0.shape[0]
        times = future_times.to(device=l0.device, dtype=l0.dtype)
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1).clone()
        elif times.dim() == 2 and times.shape[0] == 1 and batch_size > 1:
            times = times.expand(batch_size, -1).clone()
        elif times.dim() != 2 or times.shape[0] != batch_size:
            raise ValueError(f"future_times must have shape (T,), (1,T), or (batch,T); got {tuple(future_times.shape)}.")
        survival_times = self._survival_interval_times(batch_size, l0.device, l0.dtype, survival_interval_grid)
        survival_start_times = self._survival_interval_start_times(survival_times)
        self.validate_shared_time_normalization(times, survival_start_times)
        union_times = torch.unique(torch.cat([times.reshape(-1), survival_start_times.reshape(-1)]))
        union_times = union_times.sort().values.unsqueeze(0).expand(batch_size, -1).clone()
        union_u_path = self.integrate_path(u0, union_times, z, s, a)
        u_start = self._select_path_at_times(union_u_path, union_times, survival_start_times)
        survival_out = self.dynamic_survival_from_interval_start_path(
            u_start,
            survival_start_times,
            survival_times,
            z,
            s,
            u0,
            a,
        )
        survival_out["u_interval_start"] = u_start
        survival_sample = self.sample_dynamic_survival(survival_out, deterministic=deterministic_latents)
        event_time_normalized = survival_sample["event_time"]
        censoring_time_normalized = survival_sample["censoring_time"]
        observed_time_normalized = survival_sample["observed_time"]
        survival_sample = dict(survival_sample)
        survival_sample["event_time_normalized"] = event_time_normalized
        survival_sample["censoring_time_normalized"] = censoring_time_normalized
        survival_sample["observed_time_normalized"] = observed_time_normalized
        survival_sample["event_time"] = self.denormalize_survival_time(event_time_normalized)
        survival_sample["censoring_time"] = self.denormalize_survival_time(censoring_time_normalized)
        survival_sample["observed_time"] = self.denormalize_survival_time(observed_time_normalized)
        u_path = self._select_path_at_times(union_u_path, union_times, times)
        ode_mean = self.decoder.mean_from_path(u_path, times.to(device=u0.device, dtype=u0.dtype), z, s, a)
        longitudinal_mean = ode_mean.clone()
        t0_rows = times.abs() <= self.baseline_time_eps
        if t0_rows.any():
            l0_expand = l0.to(longitudinal_mean.device, longitudinal_mean.dtype).unsqueeze(1).expand_as(longitudinal_mean)
            replace = t0_rows.to(longitudinal_mean.device).unsqueeze(-1).expand_as(longitudinal_mean)
            longitudinal_mean = torch.where(replace, l0_expand, longitudinal_mean)
        available = times <= observed_time_normalized.to(times.device, times.dtype)
        longitudinal_mean = longitudinal_mean.masked_fill(~available.unsqueeze(-1), float("nan"))
        out: dict[str, torch.Tensor | dict[str, Any]] = {
            "z": z,
            "s": s,
            "a": a,
            "u0": u0,
            "u0_mu": u0_diag["u0_mu"],
            "u0_sigma": u0_diag["u0_sigma"],
            "u0_sample": u0_diag["u0_sample"],
            "u_path": u_path,
            "ode_longitudinal_mean": ode_mean,
            "longitudinal_mean": longitudinal_mean,
            "future_times": times,
            "union_time_grid": union_times,
            "union_u_path": union_u_path,
            "dynamic_survival": survival_out,
            "event_hazard": survival_out["event_hazard"],
            "censoring_hazard": survival_out["censoring_hazard"],
            "longitudinal_available": available,
            "hivae_result": hres,
        }
        out.update(survival_sample)
        return out

    @staticmethod
    def _encoded_feature_dim(feat: dict[str, Any]) -> int:
        if feat["type"] in {"cat", "ordinal"}:
            return int(feat["nclass"])
        return int(feat["dim"])

    def generate_observed_baseline(
        self,
        W: list[torch.Tensor],
        mask_W: torch.Tensor,
        L0: torch.Tensor,
        future_times: torch.Tensor,
        A: torch.Tensor,
        survival_interval_grid: torch.Tensor | None = None,
        deterministic_latents: bool = True,
        deterministic_u0: bool | None = None,
        tau: float = 1e-3,
    ) -> dict[str, torch.Tensor]:
        W, mask_W = self._select_baseline_features(W, mask_W)
        if len(W) != len(self.hivae.feat_types_list):
            raise ValueError(
                "W must contain encoded non-survival baseline features in model feature order; "
                f"expected {len(self.hivae.feat_types_list)}, got {len(W)}."
            )
        batch_size = L0.shape[0]
        if mask_W.shape != (batch_size, len(self.hivae.feat_types_list)):
            raise ValueError(
                "mask_W must have shape (batch, number_of_non_survival_features); "
                f"expected {(batch_size, len(self.hivae.feat_types_list))}, got {tuple(mask_W.shape)}."
            )
        times = future_times.to(device=L0.device, dtype=L0.dtype)
        if times.dim() == 1:
            times = times.unsqueeze(0).expand(batch_size, -1).clone()
        elif times.dim() == 2 and times.shape[0] == 1 and batch_size > 1:
            times = times.expand(batch_size, -1).clone()
        elif times.dim() != 2 or times.shape[0] != batch_size:
            raise ValueError(f"future_times must have shape (T,), (1,T), or (batch,T); got {tuple(future_times.shape)}.")
        batch_data: list[torch.Tensor] = []
        batch_miss = torch.zeros(
            batch_size,
            len(self.hivae.feat_types_list),
            device=L0.device,
            dtype=L0.dtype,
        )
        for feat_idx, feat in enumerate(self.hivae.feat_types_list):
            value = W[feat_idx].to(device=L0.device, dtype=L0.dtype)
            if value.shape[0] != batch_size:
                raise ValueError(f"W[{feat_idx}] batch size {value.shape[0]} does not match L0 batch size {batch_size}.")
            batch_data.append(value)
            batch_miss[:, feat_idx] = mask_W[:, feat_idx].to(device=L0.device, dtype=L0.dtype)

        return self.generate_future_from_baseline(
            batch_data,
            batch_miss,
            L0,
            times,
            A,
            survival_interval_grid=survival_interval_grid,
            deterministic_latents=deterministic_latents,
            deterministic_u0=deterministic_u0,
            tau=tau,
        )

    def survival_weight_for_epoch(self, current_epoch: int | None = None) -> float:
        if self.survival_warmup_epochs <= 0 or current_epoch is None:
            return self.lambda_surv
        return self.lambda_surv * min(1.0, max(float(current_epoch), 0.0) / float(self.survival_warmup_epochs))

    def forward(
        self,
        batch_data_observed,
        batch_data,
        batch_miss,
        tau=1.0,
        n_generated_dataset=1,
        longitudinal_batch=None,
        treatment=None,
        current_epoch: int | None = None,
    ):
        a: torch.Tensor | None = None
        if self.longitudinal_only_loss and longitudinal_batch is not None:
            times, values, masks = longitudinal_batch
            split = self.split_longitudinal_batch(times, values, masks)
            z0 = torch.zeros(times.shape[0], self.hivae.z_dim, device=times.device, dtype=times.dtype)
            s0 = torch.zeros(times.shape[0], self.hivae.s_dim, device=times.device, dtype=times.dtype)
            a = self.treatment_context(treatment, times.shape[0], times.device, times.dtype)
            u0, u0_diag = self.sample_u0_from_l0(z0, s0, split["L0"], deterministic=None, return_details=True)
            u_path = self.integrate_path(u0, times, z0, s0, a)
            long_loss, aux = self.longitudinal_loss_0plus(
                u0,
                u_path,
                split["future_times"],
                split["future_values"],
                split["future_masks"],
                split["L0"],
                split["M0"],
                z0,
                s0,
                a,
            )
            total = self.longitudinal_weight * long_loss
            return {
                "neg_ELBO_loss": total,
                "hivae_loss": torch.tensor(0.0, device=total.device),
                "longitudinal_loss": long_loss,
                "longitudinal_loss_0plus": long_loss,
                "longitudinal_future_loss": aux["longitudinal_future_nll"],
                "KL_u": torch.tensor(0.0, device=total.device),
                "u0": u0,
                "u0_mu": u0_diag["u0_mu"],
                "u0_sigma": u0_diag["u0_sigma"],
                "u0_sample": u0_diag["u0_sample"],
                "u0_var_regularization": u0_diag["u0_var_regularization"],
                "u_path": u_path,
                **aux,
            }
        if longitudinal_batch is None:
            raise ValueError("PhaseSynModel.forward requires longitudinal_batch so q_phi can condition on complete L0.")
        times, values, masks = longitudinal_batch
        split = self.split_longitudinal_batch(times, values, masks)
        baseline_observed, baseline_miss = self._select_baseline_features(batch_data_observed, batch_miss)
        baseline_data, _ = self._select_baseline_features(batch_data, batch_miss)
        hres = self.encode_static_posterior(
            baseline_observed,
            baseline_data,
            baseline_miss,
            tau,
            n_generated_dataset,
            encoder_l0=split["L0"],
        )
        z = hres["samples"]["z"]
        s = hres["samples"]["s"]
        a = self.treatment_context(treatment, times.shape[0], times.device, times.dtype)
        if self.u0_init_mode == "gru":
            if self.encoder is None:
                raise RuntimeError("GRU mode requested but encoder is not initialized.")
            imputed = self.imputer(values, masks)
            mu_q, log_var_q = self.encoder(times, imputed, masks)
            eps = torch.randn_like(mu_q)
            u0 = mu_q if self.deterministic_u else mu_q + torch.exp(0.5 * log_var_q) * eps
            mu_p, log_var_p = self.prior(z, s)
            u_path = self.integrate_path(u0, times, z, s, a)
            long_loss, aux = self.longitudinal_loss_0plus(
                u0,
                u_path,
                split["future_times"],
                split["future_values"],
                split["future_masks"],
                split["L0"],
                split["M0"],
                z,
                s,
                a,
            )
            kl_u = kl_normal(mu_q, log_var_q, mu_p, log_var_p)
            u0_mu_diag = mu_q
            u0_sigma_diag = torch.exp(0.5 * log_var_q)
        else:
            u0, u0_diag = self.sample_u0_from_l0(z, s, split["L0"], deterministic=None, return_details=True)
            u_path = self.integrate_path(u0, times, z, s, a)
            long_loss, aux = self.longitudinal_loss_0plus(
                u0,
                u_path,
                split["future_times"],
                split["future_values"],
                split["future_masks"],
                split["L0"],
                split["M0"],
                z,
                s,
                a,
            )
            kl_u = u0_diag["u0_var_regularization"]
            u0_mu_diag = u0_diag["u0_mu"]
            u0_sigma_diag = u0_diag["u0_sigma"]

        hivae_loss = hres["neg_ELBO_loss"]
        survival_out = self.dynamic_survival(u0, z, s, a)
        observed_time, event = self._full_survival_data(batch_data, times.device, times.dtype)
        surv_loss, surv_aux = self.dynamic_survival_nll(
            survival_out["event_hazard"],
            survival_out["censoring_hazard"],
            observed_time,
            event,
            survival_out["boundary_times"],
            admin_end_threshold=self.admin_end_threshold,
            admin_censoring_mode=self.admin_censoring_mode,
            event_weight=self.survival_event_weight,
        )
        surv_event_aux = torch.tensor(0.0, device=times.device, dtype=times.dtype)
        surv_time_aux = torch.tensor(0.0, device=times.device, dtype=times.dtype)
        surv_time_head_loss = torch.tensor(0.0, device=times.device, dtype=times.dtype)
        surv_aux_extra: dict[str, torch.Tensor] = {}
        if self.survival_event_aux_weight > 0.0 or self.survival_time_aux_weight > 0.0:
            surv_aux_extra = self.dynamic_survival_auxiliary_loss(
                survival_out["event_hazard"],
                survival_out["censoring_hazard"],
                observed_time,
                event,
                survival_out["boundary_times"],
                event_weight=self.survival_event_weight,
            )
            surv_event_aux = surv_aux_extra["survival_event_aux_bce"]
            surv_time_aux = surv_aux_extra["survival_time_aux_mse"]
        if self.survival_time_head_weight > 0.0:
            surv_time_head_loss = F.mse_loss(survival_out["time_prediction"], observed_time)

        kl_u_term = self.kl_weight_u * kl_u if self.u0_init_mode == "gru" else self.u0_kl_weight * kl_u
        lambda_surv_effective = self.survival_weight_for_epoch(current_epoch)
        survival_total = (
            surv_loss
            + self.survival_event_aux_weight * surv_event_aux
            + self.survival_time_aux_weight * surv_time_aux
            + self.survival_time_head_weight * surv_time_head_loss
        )
        total = hivae_loss + self.longitudinal_weight * long_loss + lambda_surv_effective * survival_total + kl_u_term
        hres.update({
            "neg_ELBO_loss": total,
            "hivae_loss": hivae_loss.detach(),
            "loss_base": hivae_loss,
            "longitudinal_loss": long_loss,
            "longitudinal_loss_0plus": long_loss,
            "longitudinal_future_loss": aux["longitudinal_future_nll"],
            "loss_surv_dyn": surv_loss,
            "loss_surv_total": survival_total,
            "survival_event_aux_bce": surv_event_aux,
            "survival_time_aux_mse": surv_time_aux,
            "survival_time_head_mse": surv_time_head_loss,
            "lambda_surv": torch.tensor(self.lambda_surv, device=times.device, dtype=times.dtype),
            "lambda_surv_effective": torch.tensor(lambda_surv_effective, device=times.device, dtype=times.dtype),
            "survival_event_aux_weight": torch.tensor(self.survival_event_aux_weight, device=times.device, dtype=times.dtype),
            "survival_time_aux_weight": torch.tensor(self.survival_time_aux_weight, device=times.device, dtype=times.dtype),
            "survival_time_head_weight": torch.tensor(self.survival_time_head_weight, device=times.device, dtype=times.dtype),
            "KL_u": kl_u,
            "a": a,
            "u0": u0,
            "u0_mu": u0_mu_diag,
            "u0_sigma": u0_sigma_diag,
            "u0_sample": u0,
            "u0_var_regularization": kl_u,
            "u0_kl_weight": torch.tensor(self.u0_kl_weight, device=times.device, dtype=times.dtype),
            "u0_reg_term": kl_u_term,
            "u_path": u_path,
            "dynamic_survival": survival_out,
            "event_hazard_logits": survival_out["event_hazard_logits"],
            "censoring_hazard_logits": survival_out["censoring_hazard_logits"],
            "event_hazard_summary": surv_aux["event_hazard_mean"],
            "censoring_hazard_summary": surv_aux["censoring_hazard_mean"],
            **aux,
            **surv_aux,
            **surv_aux_extra,
        })
        return hres

    def infer_u_path(self, panel: LongitudinalPanel, device: torch.device) -> torch.Tensor:
        u0 = self.infer_u(panel, device)
        with torch.no_grad():
            return self.integrate_path(u0, panel.times.to(device))

    def sample_u_path_from_prior(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        times: torch.Tensor,
        treatment: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p, log_var_p = self.prior(z, s)
        if deterministic:
            u0 = mu_p
        else:
            eps = torch.randn_like(mu_p)
            u0 = mu_p + torch.exp(0.5 * log_var_p) * eps
        a = self.treatment_context(treatment, z.shape[0], z.device, z.dtype)
        u_path = self.integrate_path(u0, times.to(device=u0.device, dtype=u0.dtype), z, s, a)
        return u0, u_path

    def sample_u_path_from_l0(
        self,
        z: torch.Tensor,
        s: torch.Tensor,
        l0: torch.Tensor,
        times: torch.Tensor,
        treatment: torch.Tensor,
        deterministic: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        l0 = l0.to(z.device, z.dtype)
        a = self.treatment_context(treatment, z.shape[0], z.device, z.dtype)
        u0 = self.init_u0_from_l0(z, s, l0, deterministic=deterministic)
        u_path = self.integrate_path(u0, times.to(device=z.device, dtype=z.dtype), z, s, a)
        return u0, u_path

    def infer_u(self, panel: LongitudinalPanel, device: torch.device) -> torch.Tensor:
        values = panel.values.to(device)
        masks = panel.masks.to(device)
        times = panel.times.to(device)
        with torch.no_grad():
            if self.u0_init_mode == "gru":
                if self.encoder is None:
                    raise RuntimeError("GRU mode requested but encoder is not initialized.")
                imputed = self.imputer(values, masks)
                mu_q, _ = self.encoder(times, imputed, masks)
                return mu_q
            split = self.split_longitudinal_batch(times, values, masks)
            z0 = torch.zeros(times.shape[0], self.hivae.z_dim, device=device, dtype=values.dtype)
            s0 = torch.zeros(times.shape[0], self.hivae.s_dim, device=device, dtype=values.dtype)
            return self.init_u0_from_l0(z0, s0, split["L0"], deterministic=True)


def build_model(bundle: PDC2Bundle, cfg: dict[str, Any]) -> nn.Module:
    cfg = dict(cfg)
    cfg["_bundle_meta"] = {
        "treatment_name": bundle.treatment_name,
        "treatment_n_classes": bundle.treatment_n_classes,
        "survival_time_min": float(bundle.longitudinal.time_min),
        "survival_time_max": float(bundle.longitudinal.time_max),
        "shared_time_normalization": True,
    }
    hivae = build_hivae(bundle, cfg)
    return PhaseSynModel(hivae, bundle.longitudinal, cfg)
