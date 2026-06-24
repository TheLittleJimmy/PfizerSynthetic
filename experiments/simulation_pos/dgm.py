from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TrialData:
    X: np.ndarray
    A: np.ndarray
    L: np.ndarray
    R: np.ndarray
    time_grid: np.ndarray
    T_obs: np.ndarray
    delta: np.ndarray
    event_time: np.ndarray
    censoring_time: np.ndarray
    G: np.ndarray
    Z: np.ndarray
    H1: np.ndarray
    H2: np.ndarray


@dataclass
class DGMParameters:
    seed: int
    n_baseline: int
    n_biomarkers: int
    a_x: np.ndarray
    b_x: np.ndarray
    mu_l: np.ndarray
    w_l: np.ndarray
    d_l: np.ndarray
    sigma_l: np.ndarray
    gamma_b: np.ndarray


def expit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def generate_dgm_parameters(
    seed: int,
    n_baseline: int = 30,
    n_biomarkers: int = 30,
) -> DGMParameters:
    if n_baseline < 30:
        raise ValueError("This simulation requires at least 30 baseline covariates.")
    rng = np.random.default_rng(int(seed))
    return DGMParameters(
        seed=int(seed),
        n_baseline=int(n_baseline),
        n_biomarkers=int(n_biomarkers),
        a_x=rng.normal(0.0, 0.5, size=(n_baseline, 3)),
        b_x=rng.normal(0.0, 0.4, size=(n_baseline, 4)),
        mu_l=rng.normal(0.0, 0.5, size=n_biomarkers),
        w_l=rng.normal(0.0, 0.5, size=(n_biomarkers, 4)),
        d_l=rng.normal(0.0, 0.15, size=(n_biomarkers, n_baseline)),
        sigma_l=rng.uniform(0.2, 0.5, size=n_biomarkers),
        gamma_b=rng.normal(0.0, 0.08, size=n_baseline),
    )


def _risk_group_and_severity(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    g = rng.choice(np.asarray([0, 1, 2], dtype=int), size=int(n), p=[0.4, 0.4, 0.2])
    means = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.0, 0.0],
            [1.0, 1.0, 0.5, 0.5],
        ],
        dtype=float,
    )
    z = means[g] + rng.normal(0.0, 1.0, size=(n, 4))
    return g, z


def _baseline_covariates(
    rng: np.random.Generator,
    g: np.ndarray,
    z: np.ndarray,
    params: DGMParameters,
) -> np.ndarray:
    n = len(g)
    x = np.zeros((n, params.n_baseline), dtype=float)
    linear = params.a_x[:, g].T + z @ params.b_x.T
    x[:, :6] = linear[:, :6] + rng.normal(0.0, 0.5, size=(n, 6))
    x[:, 6:10] = rng.binomial(1, expit(linear[:, 6:10]))
    x[:, 10:20] = (
        linear[:, 10:20]
        + 0.2 * x[:, [0]]
        - 0.15 * x[:, [1]]
        + rng.normal(0.0, 0.6, size=(n, 10))
    )
    x[:, 20:30] = rng.binomial(
        1,
        expit(linear[:, 20:30] + 0.25 * x[:, [2]] - 0.20 * x[:, [3]]),
    )
    if params.n_baseline > 30:
        x[:, 30:] = linear[:, 30:] + rng.normal(0.0, 0.6, size=(n, params.n_baseline - 30))
    return x


def _latent_trajectories(
    x: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
    z: np.ndarray,
    time_grid: np.ndarray,
    scenario: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    t = time_grid.reshape(1, -1)
    a_col = a.reshape(-1, 1)
    alpha1 = 0.5 * z[:, 0] + 0.3 * z[:, 1] + 0.4 * (g == 2)
    v1 = 0.2 + 0.3 * z[:, 0] - 0.2 * a + 0.1 * z[:, 2]
    c1 = 0.3 + 0.2 * z[:, 1]
    alpha2 = 0.4 * z[:, 2] + 0.2 * x[:, 0] + 0.3 * (g == 1)
    v2 = 0.1 + 0.2 * z[:, 3] + 0.3 * (g == 2)
    c2 = 0.2 + 0.2 * z[:, 0] - 0.1 * z[:, 3]
    h1 = (
        alpha1.reshape(-1, 1)
        + v1.reshape(-1, 1) * t
        + c1.reshape(-1, 1) * np.sin(2.0 * np.pi * t)
        + a_col * float(scenario["psi_1"]) * (1.0 - np.exp(-3.0 * t))
    )
    h2 = (
        alpha2.reshape(-1, 1)
        + v2.reshape(-1, 1) * t**2
        + c2.reshape(-1, 1) * np.cos(2.0 * np.pi * t)
        + a_col * float(scenario["psi_2"]) * t
    )
    return h1, h2


def calibrate_missingness_intercept(linear_predictor_without_intercept: np.ndarray, target_missing_rate: float) -> float:
    target_observed = 1.0 - float(target_missing_rate)
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        observed = float(expit(mid + linear_predictor_without_intercept).mean())
        if observed < target_observed:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _longitudinal_panel(
    rng: np.random.Generator,
    x: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    params: DGMParameters,
) -> np.ndarray:
    n, n_time = h1.shape
    l = np.zeros((n, n_time, params.n_biomarkers), dtype=float)
    x_part = x @ params.d_l.T
    for ell in range(params.n_biomarkers):
        signal = (
            params.mu_l[ell]
            + params.w_l[ell, 0] * h1
            + params.w_l[ell, 1] * h2
            + params.w_l[ell, 2] * h1**2
            + params.w_l[ell, 3] * np.sin(h2)
            + x_part[:, [ell]]
        )
        l[:, :, ell] = signal + rng.normal(0.0, params.sigma_l[ell], size=(n, n_time))
    return l


def _missingness_mask(
    rng: np.random.Generator,
    x: np.ndarray,
    a: np.ndarray,
    l: np.ndarray,
    time_grid: np.ndarray,
    target_missing_rate: float,
) -> np.ndarray:
    n, n_time, n_biomarkers = l.shape
    t = time_grid.reshape(1, n_time, 1)
    base_eta = 0.5 * t - 0.1 * a.reshape(n, 1, 1) + 0.15 * l + 0.2 * x[:, 0].reshape(n, 1, 1)
    omega0 = calibrate_missingness_intercept(base_eta[:, 1:, :], target_missing_rate)
    r = rng.binomial(1, expit(omega0 + base_eta)).astype(float)
    r[:, 0, :] = 1.0
    return r


def sample_event_and_censoring(
    rng: np.random.Generator,
    x: np.ndarray,
    a: np.ndarray,
    g: np.ndarray,
    l: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    time_grid: np.ndarray,
    scenario: dict[str, float],
    params: DGMParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(a)
    n_intervals = len(time_grid) - 1
    event_time = np.full(n, np.inf, dtype=float)
    censor_time = np.full(n, np.inf, dtype=float)
    for b in range(1, n_intervals + 1):
        prev = b - 1
        delta_l = l[:, prev, 10:20] - l[:, 0, 10:20]
        eta_event = (
            -3.2
            + 0.15 * b
            + float(scenario["gamma_A"]) * a
            + x @ params.gamma_b
            + 0.5 * (g == 2)
            + 0.08 * l[:, prev, :10].sum(axis=1)
            - 0.05 * delta_l.sum(axis=1)
            + 0.25 * h1[:, prev] ** 2
            + 0.20 * a * h2[:, prev]
        )
        eta_censor = -3.5 + 0.08 * b + 0.2 * a + 0.2 * x[:, 0] + 0.3 * (g == 2) + 0.1 * time_grid[prev]
        new_event = (event_time == np.inf) & (rng.random(n) < expit(eta_event))
        new_censor = (censor_time == np.inf) & (rng.random(n) < expit(eta_censor))
        event_time[new_event] = time_grid[b]
        censor_time[new_censor] = time_grid[b]
    observed = np.minimum(np.minimum(event_time, censor_time), time_grid[-1])
    delta = ((event_time <= censor_time) & (event_time <= time_grid[-1])).astype(int)
    return observed, delta, event_time, censor_time


def simulate_trial(
    n: int,
    scenario: dict[str, float],
    dgm_params: DGMParameters,
    seed: int,
    n_timepoints: int = 10,
    n_biomarkers: int = 30,
    time_grid: list[float] | np.ndarray | None = None,
    missing_rate_target: float = 0.20,
) -> TrialData:
    del n_biomarkers
    rng = np.random.default_rng(int(seed))
    grid = np.linspace(0.0, 1.0, int(n_timepoints)) if time_grid is None else np.asarray(time_grid, dtype=float)
    g, z = _risk_group_and_severity(rng, int(n))
    x = _baseline_covariates(rng, g, z, dgm_params)
    a = rng.binomial(1, 0.5, size=int(n)).astype(int)
    h1, h2 = _latent_trajectories(x, a, g, z, grid, scenario)
    l = _longitudinal_panel(rng, x, h1, h2, dgm_params)
    r = _missingness_mask(rng, x, a, l, grid, missing_rate_target)
    t_obs, delta, event_time, censor_time = sample_event_and_censoring(
        rng, x, a, g, l, h1, h2, grid, scenario, dgm_params
    )
    return TrialData(
        X=x,
        A=a,
        L=l,
        R=r,
        time_grid=grid,
        T_obs=t_obs,
        delta=delta,
        event_time=event_time,
        censoring_time=censor_time,
        G=g,
        Z=z,
        H1=h1,
        H2=h2,
    )


def save_trial_npz(path: str | Path, trial: TrialData) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=trial.X,
        A=trial.A,
        L=trial.L,
        R=trial.R,
        time_grid=trial.time_grid,
        T_obs=trial.T_obs,
        delta=trial.delta,
        event_time=trial.event_time,
        censoring_time=trial.censoring_time,
        G=trial.G,
        Z=trial.Z,
        H1=trial.H1,
        H2=trial.H2,
    )


def load_trial_npz(path: str | Path) -> TrialData:
    data = np.load(path)
    return TrialData(
        X=data["X"],
        A=data["A"],
        L=data["L"],
        R=data["R"],
        time_grid=data["time_grid"],
        T_obs=data["T_obs"],
        delta=data["delta"],
        event_time=data["event_time"],
        censoring_time=data["censoring_time"],
        G=data["G"],
        Z=data["Z"],
        H1=data["H1"],
        H2=data["H2"],
    )


def trial_summary(trial: TrialData) -> dict[str, Any]:
    administrative_end = float(trial.time_grid[-1])
    censored_before_end = (trial.delta == 0) & (trial.T_obs < administrative_end - 1e-8)
    return {
        "n": int(len(trial.A)),
        "event_rate": float(np.mean(trial.delta)),
        "censoring_rate": float(np.mean(censored_before_end)),
        "treatment_rate": float(np.mean(trial.A)),
        "missing_rate": float(1.0 - np.mean(trial.R[:, 1:, :])),
        "mean_followup": float(np.mean(trial.T_obs)),
    }

