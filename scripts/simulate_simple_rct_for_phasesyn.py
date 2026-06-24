#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, softmax


VISIT_SCHEDULE = np.asarray([0.00, 0.08, 0.16, 0.25, 0.38, 0.50, 0.67, 0.83, 1.00], dtype=float)
LONG_NAMES = [f"L{k}" for k in range(1, 7)]
STATIC_SPECS = [
    {"name": "W_cont_1", "type": "real", "dim": 1, "nclass": ""},
    {"name": "W_cont_2", "type": "real", "dim": 1, "nclass": ""},
    {"name": "W_bin_1", "type": "cat", "dim": 1, "nclass": 2},
    {"name": "W_bin_2", "type": "cat", "dim": 1, "nclass": 2},
    {"name": "W_cat_1", "type": "cat", "dim": 1, "nclass": 3},
    {"name": "W_ord_1", "type": "ordinal", "dim": 1, "nclass": 4},
    {"name": "W_count_1", "type": "count", "dim": 1, "nclass": ""},
    {"name": "W_pos_1", "type": "pos", "dim": 1, "nclass": ""},
]


def sample_latents(n, rng):
    z = rng.normal(0.0, 1.0, size=(n, 3))
    eps_risk = rng.normal(0.0, 0.3, size=n)
    risk = 0.8 * z[:, 0] - 0.5 * z[:, 1] + 0.3 * z[:, 2] + eps_risk
    return {
        "z": z,
        "R": risk,
        "eps_R": eps_risk,
    }


def _softmax_sample(prob, rng):
    draw = rng.uniform(size=prob.shape[0])
    cdf = np.cumsum(prob, axis=1)
    return (draw[:, None] > cdf[:, :-1]).sum(axis=1)


def generate_baseline(latents, rng):
    z = latents["z"]
    risk = latents["R"]
    n = len(risk)

    w_cont_1 = 1.0 + 0.7 * z[:, 0] - 0.4 * z[:, 1] + 0.2 * z[:, 2] + rng.normal(0.0, 0.6, n)
    w_cont_2 = -0.5 + 0.3 * z[:, 0] + 0.6 * z[:, 1] - 0.2 * z[:, 2] + rng.normal(0.0, 0.7, n)

    p_bin_1 = expit(-0.25 + 0.65 * z[:, 0] - 0.35 * z[:, 1] + 0.15 * z[:, 2])
    p_bin_2 = expit(0.15 - 0.25 * z[:, 0] + 0.35 * z[:, 1] + 0.55 * z[:, 2])
    w_bin_1 = rng.binomial(1, p_bin_1, size=n)
    w_bin_2 = rng.binomial(1, p_bin_2, size=n)

    cat_logits = np.column_stack([
        0.2 + 0.4 * z[:, 0] - 0.2 * z[:, 1],
        -0.1 - 0.3 * z[:, 0] + 0.5 * z[:, 2],
        -0.2 + 0.2 * z[:, 1] - 0.35 * z[:, 2],
    ])
    w_cat_1 = _softmax_sample(softmax(cat_logits, axis=1), rng)

    ord_score = 0.15 + 0.5 * z[:, 0] - 0.2 * z[:, 1] + 0.25 * z[:, 2] + rng.normal(0.0, 0.8, n)
    w_ord_1 = np.digitize(ord_score, [-0.8, 0.1, 0.9]).astype(int)

    count_rate = np.exp(0.75 + 0.25 * z[:, 0] - 0.15 * z[:, 1] + 0.1 * z[:, 2])
    w_count_1 = rng.poisson(count_rate)

    log_pos_mean = 0.4 + 0.35 * z[:, 0] + 0.2 * z[:, 1] - 0.15 * z[:, 2]
    w_pos_1 = np.exp(log_pos_mean + rng.normal(0.0, 0.35, n))

    true = pd.DataFrame({
        "W_cont_1": w_cont_1,
        "W_cont_2": w_cont_2,
        "W_bin_1": w_bin_1,
        "W_bin_2": w_bin_2,
        "W_cat_1": w_cat_1,
        "W_ord_1": w_ord_1,
        "W_count_1": w_count_1,
        "W_pos_1": w_pos_1,
    })

    observed = true.copy()
    masks = pd.DataFrame(index=np.arange(n))
    intercepts = {
        "W_cont_1": 2.15,
        "W_cont_2": 2.05,
        "W_bin_1": 2.30,
        "W_bin_2": 2.20,
        "W_cat_1": 2.10,
        "W_ord_1": 2.05,
        "W_count_1": 2.00,
        "W_pos_1": 2.00,
    }
    for name in true.columns:
        values = true[name].to_numpy(dtype=float)
        scaled = (values - np.mean(values)) / max(float(np.std(values)), 1e-6)
        p_obs = expit(intercepts[name] - 0.18 * np.abs(scaled) - 0.22 * risk)
        obs = rng.binomial(1, p_obs, size=n).astype(int)
        observed.loc[obs == 0, name] = np.nan
        masks[f"obs_{name}"] = obs

    return true, observed, masks


def assign_treatment(n, rng):
    return rng.binomial(1, 0.5, size=n).astype(int)


def _make_visit_times(n, rng):
    times = np.tile(VISIT_SCHEDULE[None, :], (n, 1)).astype(float)
    jitter = rng.uniform(-0.015, 0.015, size=times[:, 1:].shape)
    times[:, 1:] = np.clip(times[:, 1:] + jitter, 0.0, 1.0)
    times[:, 0] = 0.0
    times = np.sort(times, axis=1)
    times[:, 0] = 0.0
    return times


def _disease_trajectory(risk, treatment, b0, b1, times):
    return (
        risk[:, None]
        + b0[:, None]
        + (-0.15 + b1[:, None]) * times
        - 0.65 * treatment[:, None] * times
        - 0.22 * treatment[:, None] * risk[:, None] * times
        + 0.35 * times**2
    )


def _longitudinal_mean(risk, treatment, disease, times):
    c = np.asarray([0.3, -0.4, 0.8, -0.2, 0.1, 0.5])
    alpha_d = np.asarray([1.00, 0.75, -0.55, 1.20, -0.85, 0.50])
    alpha_r = np.asarray([0.20, -0.10, 0.25, 0.15, -0.20, 0.10])
    alpha_a = np.asarray([-0.05, 0.10, 0.00, -0.12, 0.08, -0.04])
    alpha_t = np.asarray([0.20, -0.15, 0.10, 0.25, -0.05, 0.15])
    return (
        c[None, None, :]
        + alpha_d[None, None, :] * disease[:, :, None]
        + alpha_r[None, None, :] * risk[:, None, None]
        + alpha_a[None, None, :] * treatment[:, None, None]
        + alpha_t[None, None, :] * times[:, :, None]
    )


def _sample_longitudinal_noise(n, n_visits, rng):
    sigma = np.asarray([0.35, 0.45, 0.40, 0.50, 0.42, 0.38])
    corr = np.full((len(sigma), len(sigma)), 0.28)
    np.fill_diagonal(corr, 1.0)
    cov = corr * np.outer(sigma, sigma)
    return rng.multivariate_normal(np.zeros(len(sigma)), cov, size=n * n_visits).reshape(n, n_visits, len(sigma))


def generate_longitudinal_trajectory(latents, treatment, visit_times, rng, random_effects=None, noise=None):
    risk = latents["R"]
    n, n_visits = visit_times.shape
    if random_effects is None:
        b0 = rng.normal(0.0, 0.35, size=n)
        b1 = rng.normal(0.0, 0.18, size=n)
    else:
        b0 = random_effects["b0"]
        b1 = random_effects["b1"]
    disease = _disease_trajectory(risk, treatment, b0, b1, visit_times)
    mean = _longitudinal_mean(risk, treatment, disease, visit_times)
    if noise is None:
        noise = _sample_longitudinal_noise(n, n_visits, rng)
    values = mean + noise
    return {
        "times": visit_times,
        "D": disease,
        "mean": mean,
        "values": values,
        "noise": noise,
        "b0": b0,
        "b1": b1,
    }


def generate_longitudinal_tables(trajectory, treatment, survival=None, masks=None):
    n, n_visits, _ = trajectory["values"].shape
    rows_full = []
    rows_observed = []
    for i in range(n):
        for j in range(n_visits):
            base = {
                "subject_id": i,
                "A": int(treatment[i]),
                "visit_index": j,
                "visit_name": "baseline" if j == 0 else f"visit_{j}",
                "planned_time": float(VISIT_SCHEDULE[j]),
                "visit_time": float(trajectory["times"][i, j]),
                "D_true": float(trajectory["D"][i, j]),
            }
            if survival is not None:
                base["U"] = float(survival["U"][i])
                base["delta"] = int(survival["delta"][i])
            full_row = dict(base)
            observed_row = dict(base)
            for k, name in enumerate(LONG_NAMES):
                full_row[name] = float(trajectory["values"][i, j, k])
                full_row[f"{name}_mean"] = float(trajectory["mean"][i, j, k])
                obs = 1 if masks is None else int(masks[i, j, k])
                observed_row[name] = float(trajectory["values"][i, j, k]) if obs else np.nan
                observed_row[f"obs_{name}"] = obs
            rows_full.append(full_row)
            rows_observed.append(observed_row)
    return pd.DataFrame(rows_full), pd.DataFrame(rows_observed)


def compute_event_hazards(latents, treatment, random_effects, n_intervals=16):
    risk = latents["R"]
    tau = np.linspace(0.0, 1.0, n_intervals + 1)
    tau_start = tau[:-1]
    disease_start = _disease_trajectory(
        risk,
        treatment,
        random_effects["b0"],
        random_effects["b1"],
        np.tile(tau_start[None, :], (len(risk), 1)),
    )
    alpha_t = np.linspace(-3.60, -2.10, n_intervals)
    logits = alpha_t[None, :] + 0.35 * risk[:, None] + 0.55 * disease_start - 0.50 * treatment[:, None]
    return expit(logits), alpha_t, disease_start, tau


def compute_censoring_hazards(latents, treatment, n_intervals=16):
    risk = latents["R"]
    tau = np.linspace(0.0, 1.0, n_intervals + 1)
    tau_start = tau[:-1]
    alpha_c = np.linspace(-4.40, -3.10, n_intervals)
    logits = alpha_c[None, :] + 0.20 * risk[:, None] + 0.10 * treatment[:, None] + 0.30 * tau_start[None, :]
    return expit(logits), alpha_c, tau


def sample_survival(event_hazards, censoring_hazards, tau, rng, uniforms=None):
    n, n_intervals = event_hazards.shape
    if uniforms is None:
        uniforms = {
            "event": rng.uniform(size=(n, n_intervals)),
            "censor": rng.uniform(size=(n, n_intervals)),
            "event_within": rng.uniform(size=n),
            "censor_within": rng.uniform(size=n),
        }
    hit_event = uniforms["event"] < event_hazards
    hit_censor = uniforms["censor"] < censoring_hazards
    has_event = hit_event.any(axis=1)
    has_censor = hit_censor.any(axis=1)
    event_interval = np.full(n, -1, dtype=int)
    censor_interval = np.full(n, -1, dtype=int)
    event_interval[has_event] = hit_event[has_event].argmax(axis=1)
    censor_interval[has_censor] = hit_censor[has_censor].argmax(axis=1)

    event_time = np.full(n, np.inf, dtype=float)
    censor_time = np.ones(n, dtype=float)
    if has_event.any():
        idx = event_interval[has_event]
        event_time[has_event] = tau[idx] + (tau[idx + 1] - tau[idx]) * uniforms["event_within"][has_event]
    if has_censor.any():
        idx = censor_interval[has_censor]
        censor_time[has_censor] = tau[idx] + (tau[idx + 1] - tau[idx]) * uniforms["censor_within"][has_censor]

    observed_time = np.minimum(event_time, censor_time)
    delta = (event_time <= censor_time).astype(int)
    stochastic_censor = ((delta == 0) & (censor_time < 1.0 - 1e-12)).astype(int)
    administrative_censor = ((delta == 0) & (censor_time >= 1.0 - 1e-12)).astype(int)
    return {
        "T": event_time,
        "C": censor_time,
        "U": observed_time,
        "delta": delta,
        "event_interval": event_interval,
        "censor_interval": censor_interval,
        "stochastic_censor": stochastic_censor,
        "administrative_censor": administrative_censor,
        "uniforms": uniforms,
    }


def apply_missingness(latents, trajectory, survival, rng):
    risk = latents["R"]
    disease = trajectory["D"]
    times = trajectory["times"]
    n, n_visits, q = trajectory["values"].shape
    masks = np.ones((n, n_visits, q), dtype=int)
    intercepts = np.asarray([2.10, 2.00, 1.95, 1.90, 1.85, 1.95])
    for j in range(1, n_visits):
        logits = intercepts[None, :] - 0.80 * times[:, j, None] - 0.30 * risk[:, None] - 0.20 * disease[:, j, None]
        p_obs = expit(logits)
        masks[:, j, :] = rng.binomial(1, p_obs)
    after_u = times > survival["U"][:, None] + 1e-12
    masks[after_u, :] = 0
    masks[:, 0, :] = 1
    return masks


def make_counterfactual_truth(latents, baseline_true, random_effects, visit_times, noise, survival_uniforms, n_intervals, rng):
    del baseline_true
    frames = []
    for arm in [0, 1]:
        treatment = np.full(len(latents["R"]), arm, dtype=int)
        traj = generate_longitudinal_trajectory(
            latents,
            treatment,
            visit_times,
            rng,
            random_effects=random_effects,
            noise=noise,
        )
        event_h, alpha_t, disease_start, tau = compute_event_hazards(latents, treatment, random_effects, n_intervals=n_intervals)
        censor_h, alpha_c, _ = compute_censoring_hazards(latents, treatment, n_intervals=n_intervals)
        surv = sample_survival(event_h, censor_h, tau, rng, uniforms=survival_uniforms)
        n, n_visits, _ = traj["values"].shape
        for i in range(n):
            for j in range(n_visits):
                row = {
                    "subject_id": i,
                    "intervention_A": arm,
                    "visit_index": j,
                    "visit_name": "baseline" if j == 0 else f"visit_{j}",
                    "planned_time": float(VISIT_SCHEDULE[j]),
                    "visit_time": float(traj["times"][i, j]),
                    "D_true": float(traj["D"][i, j]),
                    "T": float(surv["T"][i]),
                    "C": float(surv["C"][i]),
                    "U": float(surv["U"][i]),
                    "delta": int(surv["delta"][i]),
                    "event_interval": int(surv["event_interval"][i]),
                    "censor_interval": int(surv["censor_interval"][i]),
                    "administrative_censor": int(surv["administrative_censor"][i]),
                    "stochastic_censor": int(surv["stochastic_censor"][i]),
                }
                for k, name in enumerate(LONG_NAMES):
                    row[f"{name}_mean"] = float(traj["mean"][i, j, k])
                    row[name] = float(traj["values"][i, j, k])
                frames.append(row)
    return pd.DataFrame(frames)


def _hazards_table(latents, treatment, event_h, censor_h, alpha_t, alpha_c, disease_start, tau):
    rows = []
    risk = latents["R"]
    for i in range(len(risk)):
        for b in range(len(alpha_t)):
            rows.append({
                "subject_id": i,
                "A": int(treatment[i]),
                "interval": b + 1,
                "tau_start": float(tau[b]),
                "tau_end": float(tau[b + 1]),
                "R": float(risk[i]),
                "D_tau_start": float(disease_start[i, b]),
                "alpha_T": float(alpha_t[b]),
                "alpha_C": float(alpha_c[b]),
                "event_hazard": float(event_h[i, b]),
                "censoring_hazard": float(censor_h[i, b]),
            })
    return pd.DataFrame(rows)


def _phase_syn_compat_tables(out, baseline_df, survival_df, longitudinal_observed_df):
    phase_cols = ["time", "censor", "A"]
    static_cols = [item["name"] for item in STATIC_SPECS]
    l0_cols = LONG_NAMES

    phase_df = pd.DataFrame({
        "time": survival_df["U"],
        "censor": survival_df["delta"],
        "A": baseline_df["A"],
    })
    for name in static_cols:
        phase_df[name] = baseline_df[name]
    base_long = longitudinal_observed_df[longitudinal_observed_df["visit_index"] == 0].sort_values("subject_id")
    for name in l0_cols:
        phase_df[name] = base_long[name].to_numpy()
    phase_df.to_csv(out / "data_phasesyn.csv", index=False)

    type_rows = [{"name": "survcens", "type": "surv_dynamic", "dim": 2, "nclass": ""}]
    type_rows.append({"name": "A", "type": "cat", "dim": 1, "nclass": 2})
    type_rows.extend(STATIC_SPECS)
    for name in LONG_NAMES:
        type_rows.append({"name": name, "type": "real", "dim": 1, "nclass": ""})
    pd.DataFrame(type_rows).to_csv(out / "data_types_phasesyn_piecewise.csv", index=False)

    long_cols = ["subject_id", "visit_name", "visit_time"] + LONG_NAMES
    compat_long = longitudinal_observed_df[long_cols].rename(columns={"subject_id": "patient_id"})
    compat_long.to_csv(out / "longitudinal.csv", index=False)
    pd.DataFrame({"subject_id": baseline_df["subject_id"], "simulation_id": baseline_df["subject_id"]}).to_csv(out / "simulation_id.csv", index=False)


def summarize_dataset(baseline_df, survival_df, longitudinal_observed_df, latent_truth_df):
    treatment_counts = baseline_df["A"].value_counts().sort_index().to_dict()
    event_by_arm = survival_df.groupby("A")["delta"].mean().to_dict()
    median_u_by_arm = survival_df.groupby("A")["U"].median().to_dict()
    static_obs_cols = [c for c in baseline_df.columns if c.startswith("obs_W_")]
    long_obs_cols = [c for c in longitudinal_observed_df.columns if c.startswith("obs_L")]
    post = longitudinal_observed_df["visit_index"] > 0
    static_missing = 1.0 - float(baseline_df[static_obs_cols].to_numpy(dtype=float).mean())
    long_missing = 1.0 - float(longitudinal_observed_df.loc[post, long_obs_cols].to_numpy(dtype=float).mean())

    balance_cols = ["R", "z1", "z2", "z3"] + [item["name"] for item in STATIC_SPECS]
    balance = []
    merged = baseline_df[["subject_id", "A"]].merge(latent_truth_df[["subject_id"] + balance_cols], on="subject_id")
    for col in balance_cols:
        m0 = float(pd.to_numeric(merged.loc[merged["A"] == 0, col], errors="coerce").mean())
        m1 = float(pd.to_numeric(merged.loc[merged["A"] == 1, col], errors="coerce").mean())
        sd = float(pd.to_numeric(merged[col], errors="coerce").std(ddof=0))
        balance.append({
            "variable": col,
            "mean_A0": m0,
            "mean_A1": m1,
            "standardized_difference_A1_minus_A0": (m1 - m0) / max(sd, 1e-8),
        })

    long_means = []
    for (arm, visit), grp in longitudinal_observed_df.groupby(["A", "visit_index"], sort=True):
        row = {
            "A": int(arm),
            "visit_index": int(visit),
            "visit_time_mean": float(grp["visit_time"].mean()),
        }
        for name in LONG_NAMES:
            row[f"{name}_mean"] = float(pd.to_numeric(grp[name], errors="coerce").mean())
        long_means.append(row)

    return {
        "n": int(len(baseline_df)),
        "treatment_counts": {str(k): int(v) for k, v in treatment_counts.items()},
        "event_rate_by_arm": {str(k): float(v) for k, v in event_by_arm.items()},
        "median_observed_time_U_by_arm": {str(k): float(v) for k, v in median_u_by_arm.items()},
        "administrative_censoring_rate": float(survival_df["administrative_censor"].mean()),
        "stochastic_censoring_rate": float(survival_df["stochastic_censor"].mean()),
        "static_missingness_rate": static_missing,
        "post_baseline_longitudinal_missingness_rate": long_missing,
        "baseline_balance_checks": balance,
        "mean_longitudinal_trajectory_by_arm": long_means,
    }


def write_outputs(out, baseline_df, longitudinal_full_df, longitudinal_observed_df, survival_df, latent_truth_df, hazards_df, metadata, counterfactual_df=None):
    out.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(out / "baseline.csv", index=False)
    longitudinal_full_df.to_csv(out / "longitudinal_full.csv", index=False)
    longitudinal_observed_df.to_csv(out / "longitudinal_observed.csv", index=False)
    survival_df.to_csv(out / "survival.csv", index=False)
    latent_truth_df.to_csv(out / "latent_truth.csv", index=False)
    hazards_df.to_csv(out / "hazards_truth.csv", index=False)
    if counterfactual_df is not None:
        counterfactual_df.to_csv(out / "counterfactual_truth.csv", index=False)
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    _phase_syn_compat_tables(out, baseline_df, survival_df, longitudinal_observed_df)


def write_readme(out, metadata):
    readme = f"""# Simple Linear RCT Simulation for PhaseSyn Evaluation

## Purpose

This directory contains a standalone randomized-trial simulation dataset for PhaseSyn evaluation. The files are exported in a PhaseSyn-compatible tabular layout, but the true data-generating process is intentionally simple and transparent.

The simulator is intentionally simpler than PhaseSyn. It is deliberately not a copy of PhaseSyn: it does not use a latent ODE, does not use discrete latent mixture components as the primary simulator, and does not reproduce PhaseSyn's dynamic survival architecture. This separation makes the data useful for evaluation because PhaseSyn must recover structure from a transparent external data-generating process rather than from its own modeling assumptions. Instead, the simulator uses baseline linear risk factors, randomized treatment, linear mixed-effect longitudinal trajectories, and simple interval-discrete event and censoring hazards.

## Simulation Chronology

1. Draw three baseline latent factors $z_i \\sim N(0, I_3)$.
2. Compute the scalar baseline disease-risk score
   $R_i = 0.8z_{{i1}} - 0.5z_{{i2}} + 0.3z_{{i3}} + \\epsilon_i$, where $\\epsilon_i \\sim N(0, 0.3^2)$.
3. Generate mixed-type static baseline variables $W_i$ from simple linear, logistic, softmax, ordinal-threshold, Poisson, and log-normal models in $z_i$.
4. Generate static missingness masks `obs_W_*`, where `1=observed` and `0=unavailable`. Static missingness can depend on the generated baseline variable value and $R_i$, but not on treatment.
5. Randomize treatment independently as $A_i \\sim \\mathrm{{Bernoulli}}(0.5)$.
6. Generate subject-level random intercepts and slopes for the longitudinal disease trajectory.
7. Generate full longitudinal trajectories at planned visits with post-baseline jitter.
8. Generate interval-discrete event and censoring hazards from the true risk and disease trajectory.
9. Sample event time $T_i$ and censoring time $C_i$ independently from their hazards.
10. Define $U_i = \\min(T_i, C_i)$ and $\\delta_i = 1\\{{T_i \\le C_i\\}}$.
11. Generate post-baseline longitudinal missingness using only current-time quantities, then mark all visits after $U_i$ unavailable.
12. If requested, generate oracle counterfactual trajectories under both $A=0$ and $A=1$ using the same $z_i$, $W_i$, $R_i$, random intercept, random slope, longitudinal measurement noise, and survival uniform draws.

## Baseline Latent Factors and Risk

The latent factors are `z1`, `z2`, and `z3`. The disease-risk score `R` is higher for subjects with worse baseline risk. These variables are saved in `latent_truth.csv` and should be treated as oracle truth, not as observed training inputs.

## Baseline Variables

The static baseline variables are:

- `W_cont_1`, `W_cont_2`: continuous linear-Gaussian variables.
- `W_bin_1`, `W_bin_2`: binary variables from logistic linear models.
- `W_cat_1`: three-class categorical variable from softmax linear logits.
- `W_ord_1`: four-level ordinal variable from a latent normal score and fixed thresholds.
- `W_count_1`: count variable from a Poisson model with log-rate linear in $z_i$.
- `W_pos_1`: positive variable from a log-normal model with log-mean linear in $z_i$.

Observed static values in `baseline.csv` are set to missing when `obs_W_* = 0`. Complete true static values are available in `latent_truth.csv`.

## Treatment Randomization

Treatment `A` is generated independently of $z_i$, $W_i$, $R_i$, and $L_i(0)$:

$$A_i \\sim \\mathrm{{Bernoulli}}(0.5).$$

This makes the dataset a transparent randomized trial. Baseline balance checks are saved in `metadata.json` and printed by the script.

## Linear Mixed-Effect Longitudinal Model

The true disease trajectory is

$$D_i(t) = R_i + b_{{0i}} + (\\beta_t + b_{{1i}})t + \\beta_A A_i t + \\beta_{{AR}} A_i R_i t + \\beta_{{t2}}t^2.$$

The simulator uses $\\beta_A < 0$, so treatment improves the disease trajectory. Higher $D_i(t)$ means worse disease.

Six continuous longitudinal outcomes are generated as

$$L_{{i\\ell}}(t) = c_\\ell + \\alpha_{{\\ell D}}D_i(t) + \\alpha_{{\\ell R}}R_i + \\alpha_{{\\ell A}}A_i + \\alpha_{{\\ell t}}t + e_{{i\\ell}}(t),$$

where the six-dimensional measurement noise vector is correlated Gaussian. The baseline row $L_i(0)$ is complete for every subject.

## Visit Schedule and Jitter

Planned visits are:

`{metadata["visit_schedule"]}`

The baseline visit is exactly `0.0`. Post-baseline visits receive independent uniform jitter in `[-0.015, 0.015]` and are clipped to `[0, 1]`.

## Missingness Convention

All observation mask columns use:

- `1 = observed`
- `0 = unavailable`

For post-baseline longitudinal outcomes, missingness is MAR/dropout-like:

$$\\mathrm{{logit}}(p^{{obs}}_{{ij\\ell}}) = c_\\ell - 0.8t_{{ij}} - 0.3R_i - 0.2D_i(t_{{ij}}).$$

Missingness does not depend on future outcomes. After the observed survival time $U_i$, all post-$U_i$ longitudinal masks are set to `0`.

## Discrete-Time Event Model

The survival grid has `{metadata["n_intervals"]}` equal intervals on $[0, 1]$. For interval $I_b=(\\tau_{{b-1}},\\tau_b]$, the event hazard is

$$\\lambda^T_{{ib}} = \\sigma(\\alpha^T_b + \\gamma_R R_i + \\gamma_D D_i(\\tau_{{b-1}}) + \\gamma_A A_i).$$

The baseline event intercepts increase over time. The simulator uses $\\gamma_D > 0$, so worse current disease increases event risk, and $\\gamma_A < 0$, so treatment lowers event risk.

## Censoring Model

Censoring is simpler and mostly independent:

$$\\lambda^C_{{ib}} = \\sigma(\\alpha^C_b + 0.2R_i + 0.1A_i + 0.3\\tau_{{b-1}}).$$

This censoring process is intentionally not the same as the PhaseSyn survival mechanism.

## Survival Definitions

- `T`: event time. If no event occurs before the final endpoint, `T = inf`.
- `C`: censoring time. If no stochastic censoring occurs, `C = 1.0`.
- `U`: observed time, $U_i = \\min(T_i, C_i)$.
- `delta`: event indicator, $\\delta_i = 1\\{{T_i \\le C_i\\}}$.
- `administrative_censor`: `1` when `delta=0` and `C=1.0`.
- `stochastic_censor`: `1` when `delta=0` and `C<1.0`.

## Exported Files

- `baseline.csv`: observed baseline table with `subject_id`, randomized treatment `A`, observed static variables, static masks `obs_W_*`, and complete baseline longitudinal row `L1` to `L6`.
- `longitudinal_full.csv`: complete generated longitudinal values at every visit, including `D_true` and noiseless means `L*_mean`. This is oracle truth.
- `longitudinal_observed.csv`: observed longitudinal table with unavailable cells blanked out and masks `obs_L*`.
- `survival.csv`: factual survival outcomes `T`, `C`, `U`, `delta`, interval indices, and censoring-type indicators.
- `latent_truth.csv`: oracle latent factors, risk score, random effects, and complete true static variables.
- `hazards_truth.csv`: factual interval-level event and censoring hazards.
- `counterfactual_truth.csv`: present only when `--make-counterfactual` is used. Contains oracle trajectories and survival outcomes under interventions `A=0` and `A=1`.
- `metadata.json`: simulation parameters, variable specifications, summary statistics, and compatibility notes.
- `data_phasesyn.csv`: convenience PhaseSyn-style baseline table with survival columns, treatment, observed static variables, and complete `L0`.
- `data_types_phasesyn_piecewise.csv`: type specification for the PhaseSyn-style baseline table.
- `longitudinal.csv`: convenience PhaseSyn-style observed longitudinal table.
- `simulation_id.csv`: subject identifier table.

## Column Naming Conventions

- `W_*`: static baseline variables.
- `obs_W_*`: static baseline observation masks.
- `L1` to `L6`: longitudinal outcomes.
- `L*_mean`: noiseless conditional longitudinal means.
- `obs_L*`: longitudinal observation masks.
- `D_true`: true scalar disease trajectory.
- `tau_start`, `tau_end`: survival interval boundaries.
- `event_hazard`, `censoring_hazard`: true interval hazards.

## Counterfactual Truth

When `--make-counterfactual` is used, the simulator evaluates both treatment arms for every subject while keeping the same baseline latent factors, generated static variables, risk score, random intercept, random slope, correlated longitudinal noise, and survival uniform draws. This creates paired oracle potential outcomes for evaluation. These columns should not be used as PhaseSyn training inputs.

## Recommended PhaseSyn Use

Use `baseline.csv`, `longitudinal_observed.csv`, and `survival.csv`, or the PhaseSyn-style convenience files, for training and synthetic-data evaluation. Use `latent_truth.csv`, `hazards_truth.csv`, and `counterfactual_truth.csv` only for diagnostic evaluation, calibration checks, and causal-recovery experiments.

## Limitations

This simulator is intentionally stylized. It has linear baseline effects, a low-dimensional scalar disease process, Gaussian longitudinal outcomes, interval-discrete survival, and simple MAR/dropout missingness. It is useful for controlled PhaseSyn evaluation, but it should not be interpreted as a realistic clinical-trial simulator for any specific disease area.
"""
    with open(out / "README.md", "w", encoding="utf-8") as f:
        f.write(readme)


def _build_latent_truth(latents, random_effects, baseline_true):
    df = pd.DataFrame({
        "subject_id": np.arange(len(latents["R"]), dtype=int),
        "z1": latents["z"][:, 0],
        "z2": latents["z"][:, 1],
        "z3": latents["z"][:, 2],
        "R": latents["R"],
        "eps_R": latents["eps_R"],
        "b0": random_effects["b0"],
        "b1": random_effects["b1"],
    })
    for col in baseline_true.columns:
        df[col] = baseline_true[col].to_numpy()
    return df


def _build_baseline(subject_id, treatment, baseline_observed, baseline_masks, trajectory):
    df = pd.DataFrame({"subject_id": subject_id, "A": treatment.astype(int)})
    for col in baseline_observed.columns:
        df[col] = baseline_observed[col].to_numpy()
    for col in baseline_masks.columns:
        df[col] = baseline_masks[col].to_numpy(dtype=int)
    for k, name in enumerate(LONG_NAMES):
        df[name] = trajectory["values"][:, 0, k]
    return df


def _build_survival_df(treatment, survival):
    return pd.DataFrame({
        "subject_id": np.arange(len(treatment), dtype=int),
        "A": treatment.astype(int),
        "T": survival["T"],
        "C": survival["C"],
        "U": survival["U"],
        "delta": survival["delta"].astype(int),
        "event_interval": survival["event_interval"].astype(int),
        "censor_interval": survival["censor_interval"].astype(int),
        "administrative_censor": survival["administrative_censor"].astype(int),
        "stochastic_censor": survival["stochastic_censor"].astype(int),
    })


def _metadata(args, summary):
    return {
        "name": "simple_linear_rct",
        "n": int(args.n),
        "seed": int(args.seed),
        "n_intervals": int(args.n_intervals),
        "visit_schedule": [float(x) for x in VISIT_SCHEDULE],
        "post_baseline_visit_jitter": "Uniform(-0.015, 0.015), clipped to [0, 1]",
        "treatment_randomization": "A ~ Bernoulli(0.5), independent of z, W, R, and L0",
        "longitudinal_features": LONG_NAMES,
        "static_specs": STATIC_SPECS,
        "event_hazard": {
            "alpha_T": "linear from -3.60 to -2.10",
            "gamma_R": 0.35,
            "gamma_D": 0.55,
            "gamma_A": -0.50,
        },
        "censoring_hazard": {
            "alpha_C": "linear from -4.40 to -3.10",
            "coef_R": 0.20,
            "coef_A": 0.10,
            "coef_tau_start": 0.30,
        },
        "counterfactual_truth": bool(args.make_counterfactual),
        "missingness_convention": "1=observed, 0=unavailable",
        "phase_syn_compatibility_files": [
            "data_phasesyn.csv",
            "data_types_phasesyn_piecewise.csv",
            "longitudinal.csv",
            "simulation_id.csv",
        ],
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate a simple randomized-trial simulation dataset for PhaseSyn evaluation.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--n", type=int, default=1200, help="Number of subjects.")
    parser.add_argument("--seed", type=int, default=20260602, help="Random seed.")
    parser.add_argument("--n-intervals", type=int, default=16, help="Number of equal survival intervals on [0, 1].")
    parser.add_argument("--make-counterfactual", action="store_true", help="Export paired oracle counterfactual outcomes under A=0 and A=1.")
    args = parser.parse_args()

    if args.n <= 0:
        raise ValueError("--n must be positive.")
    if args.n_intervals <= 0:
        raise ValueError("--n-intervals must be positive.")

    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    latents = sample_latents(args.n, rng)
    baseline_true, baseline_observed, baseline_masks = generate_baseline(latents, rng)
    treatment = assign_treatment(args.n, rng)
    visit_times = _make_visit_times(args.n, rng)

    trajectory = generate_longitudinal_trajectory(latents, treatment, visit_times, rng)
    random_effects = {"b0": trajectory["b0"], "b1": trajectory["b1"]}
    event_h, alpha_t, disease_start, tau = compute_event_hazards(latents, treatment, random_effects, n_intervals=args.n_intervals)
    censor_h, alpha_c, _ = compute_censoring_hazards(latents, treatment, n_intervals=args.n_intervals)
    survival = sample_survival(event_h, censor_h, tau, rng)
    masks = apply_missingness(latents, trajectory, survival, rng)

    longitudinal_full_df, longitudinal_observed_df = generate_longitudinal_tables(trajectory, treatment, survival=survival, masks=masks)
    baseline_df = _build_baseline(np.arange(args.n, dtype=int), treatment, baseline_observed, baseline_masks, trajectory)
    survival_df = _build_survival_df(treatment, survival)
    latent_truth_df = _build_latent_truth(latents, random_effects, baseline_true)
    hazards_df = _hazards_table(latents, treatment, event_h, censor_h, alpha_t, alpha_c, disease_start, tau)
    counterfactual_df = None
    if args.make_counterfactual:
        counterfactual_df = make_counterfactual_truth(
            latents,
            baseline_true,
            random_effects,
            visit_times,
            trajectory["noise"],
            survival["uniforms"],
            args.n_intervals,
            rng,
        )

    summary = summarize_dataset(baseline_df, survival_df, longitudinal_observed_df, latent_truth_df)
    metadata = _metadata(args, summary)
    write_outputs(out, baseline_df, longitudinal_full_df, longitudinal_observed_df, survival_df, latent_truth_df, hazards_df, metadata, counterfactual_df=counterfactual_df)
    write_readme(out, metadata)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
