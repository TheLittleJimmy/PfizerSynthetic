from __future__ import annotations

from dataclasses import dataclass
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .dgm import TrialData


METHODS_20260617 = ["LMM-AFT", "JM-RE", "TVAE", "CTGAN"]


def _subject_summary(trial: TrialData) -> pd.DataFrame:
    x_cols = [f"X{k + 1:02d}" for k in range(trial.X.shape[1])]
    l_cols = [f"L{k + 1:02d}" for k in range(trial.L.shape[2])]
    rows = []
    for i in range(trial.X.shape[0]):
        row: dict[str, Any] = {c: float(v) for c, v in zip(x_cols, trial.X[i])}
        row["A"] = int(trial.A[i])
        row["time"] = float(trial.T_obs[i])
        row["event"] = int(trial.delta[i])
        row["event_time"] = float(trial.event_time[i]) if np.isfinite(trial.event_time[i]) else 1.0
        row["censoring_time"] = float(trial.censoring_time[i]) if np.isfinite(trial.censoring_time[i]) else 1.0
        for j, col in enumerate(l_cols):
            vals = trial.L[i, :, j]
            obs = trial.R[i, :, j] > 0.5
            if obs.any():
                y = vals[obs]
                t = trial.time_grid[obs]
            else:
                y = vals
                t = trial.time_grid
            row[f"{col}_baseline"] = float(vals[0])
            row[f"{col}_final"] = float(y[-1])
            row[f"{col}_change"] = float(y[-1] - vals[0])
            row[f"{col}_slope"] = float(np.polyfit(t, y, 1)[0]) if len(y) >= 2 and np.ptp(t) > 1e-8 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _target_baseline_frame(target_trial: TrialData) -> pd.DataFrame:
    x_cols = [f"X{k + 1:02d}" for k in range(target_trial.X.shape[1])]
    l_cols = [f"L{k + 1:02d}" for k in range(target_trial.L.shape[2])]
    frame = pd.DataFrame(target_trial.X, columns=x_cols)
    frame["A"] = target_trial.A.astype(int)
    for j, col in enumerate(l_cols):
        frame[f"{col}_baseline"] = target_trial.L[:, 0, j]
    return frame


def _sample_empirical_survival(rng: np.random.Generator, train: TrialData, target_a: np.ndarray, shared_score: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    t = np.zeros(len(target_a), dtype=float)
    d = np.zeros(len(target_a), dtype=int)
    for i, arm in enumerate(target_a.astype(int)):
        pool = np.flatnonzero(train.A == arm)
        if len(pool) == 0:
            pool = np.arange(len(train.A))
        idx = int(rng.choice(pool))
        t[i] = float(train.T_obs[idx])
        d[i] = int(train.delta[idx])
    if shared_score is not None:
        scale = np.exp(-0.15 * np.clip(shared_score, -3.0, 3.0))
        t = np.clip(t * scale, 1e-4, 1.0)
    return t, d


class BaseBenchmark:
    def __init__(self, train: TrialData, seed: int, method: str):
        self.train = train
        self.seed = int(seed)
        self.method = method
        self.rng = np.random.default_rng(int(seed))
        self.summary = _subject_summary(train)
        self.time_grid = train.time_grid
        self.n_biomarkers = train.L.shape[2]

    def generate_trial(self, target_trial: TrialData) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return {"method": self.method, "seed": self.seed}


class LMMAFTBenchmark(BaseBenchmark):
    def __init__(self, train: TrialData, seed: int, shared: bool = False):
        super().__init__(train, seed, "JM-RE" if shared else "LMM-AFT")
        self.shared = bool(shared)
        self._fit_longitudinal()

    def _fit_longitudinal(self) -> None:
        x = self.train.X
        a = self.train.A.reshape(-1, 1)
        features = np.column_stack([np.ones(len(a)), a, x[:, : min(10, x.shape[1])]])
        self.coef = np.zeros((self.n_biomarkers, features.shape[1], 3), dtype=float)
        self.sigma = np.ones(self.n_biomarkers, dtype=float)
        t = self.time_grid
        basis = np.column_stack([np.ones_like(t), t, t**2])
        for ell in range(self.n_biomarkers):
            rows = []
            y = []
            for i in range(self.train.L.shape[0]):
                for j, tj in enumerate(t):
                    if self.train.R[i, j, ell] < 0.5:
                        continue
                    rows.append(np.kron(basis[j], features[i]))
                    y.append(self.train.L[i, j, ell])
            design = np.asarray(rows, dtype=float)
            yy = np.asarray(y, dtype=float)
            if design.shape[0] > design.shape[1] + 2:
                beta, *_ = np.linalg.lstsq(design, yy, rcond=None)
                resid = yy - design @ beta
                self.coef[ell] = beta.reshape(3, features.shape[1]).T
                self.sigma[ell] = max(float(np.std(resid)), 1e-4)

    def generate_trial(self, target_trial: TrialData) -> dict[str, np.ndarray]:
        x = target_trial.X
        a = target_trial.A.reshape(-1, 1)
        features = np.column_stack([np.ones(len(a)), a, x[:, : min(10, x.shape[1])]])
        basis = np.column_stack([np.ones_like(target_trial.time_grid), target_trial.time_grid, target_trial.time_grid**2])
        l = np.zeros((len(a), len(target_trial.time_grid), self.n_biomarkers), dtype=float)
        shared = self.rng.normal(0.0, 1.0, size=len(a)) if self.shared else np.zeros(len(a))
        for ell in range(self.n_biomarkers):
            pred = features @ self.coef[ell] @ basis.T
            noise = self.rng.normal(0.0, self.sigma[ell], size=pred.shape)
            l[:, :, ell] = pred + noise + 0.10 * shared.reshape(-1, 1) * self.sigma[ell]
            l[:, 0, ell] = target_trial.L[:, 0, ell]
        t_obs, delta = _sample_empirical_survival(self.rng, self.train, target_trial.A, shared if self.shared else None)
        return {"X": x, "A": target_trial.A, "L": l, "T_obs": t_obs, "delta": delta}


class TinyAutoencoder(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int):
        super().__init__()
        hidden = max(8, min(96, 2 * in_dim))
        self.encoder = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


class TVAEBenchmark(BaseBenchmark):
    def __init__(self, train: TrialData, seed: int, epochs: int = 120):
        super().__init__(train, seed, "TVAE")
        torch.manual_seed(int(seed))
        self.cols = [c for c in self.summary.columns if c not in {"event_time", "censoring_time"}]
        clean = self.summary[self.cols].apply(pd.to_numeric, errors="coerce")
        self.fills = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(self.fills)
        self.scaler = StandardScaler().fit(clean)
        x = torch.tensor(self.scaler.transform(clean), dtype=torch.float32)
        self.model = TinyAutoencoder(x.shape[1], min(12, max(2, x.shape[1] // 4)))
        opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        for _ in range(int(epochs)):
            opt.zero_grad()
            recon, _ = self.model(x)
            loss = F.mse_loss(recon, x)
            loss.backward()
            opt.step()
        self.model.eval()
        with torch.no_grad():
            _, z = self.model(x)
        self.z_mean = z.mean(dim=0)
        self.z_std = z.std(dim=0).clamp(min=0.05)

    def generate_trial(self, target_trial: TrialData) -> dict[str, np.ndarray]:
        target = _target_baseline_frame(target_trial)
        n = len(target)
        z = self.z_mean + self.z_std * torch.randn(n, len(self.z_mean))
        with torch.no_grad():
            decoded = self.model.decoder(z).detach().cpu().numpy()
        sample = pd.DataFrame(self.scaler.inverse_transform(decoded), columns=self.cols)
        l = np.zeros((n, len(target_trial.time_grid), self.n_biomarkers), dtype=float)
        frac = target_trial.time_grid / max(target_trial.time_grid[-1], 1e-8)
        for ell in range(self.n_biomarkers):
            base = target_trial.L[:, 0, ell]
            change = pd.to_numeric(sample.get(f"L{ell + 1:02d}_change", 0.0), errors="coerce").fillna(0.0).to_numpy()
            noise = self.rng.normal(0.0, 0.05 * (np.abs(base).reshape(-1, 1) + 1.0), size=(n, len(frac)))
            l[:, :, ell] = base.reshape(-1, 1) + change.reshape(-1, 1) * frac.reshape(1, -1) + noise
            l[:, 0, ell] = base
        t_obs, delta = _sample_empirical_survival(self.rng, self.train, target_trial.A)
        return {"X": target_trial.X, "A": target_trial.A, "L": l, "T_obs": t_obs, "delta": delta}


class TinyConditionalGenerator(nn.Module):
    def __init__(self, noise_dim: int, out_dim: int):
        super().__init__()
        hidden = max(16, min(128, out_dim * 2))
        self.net = nn.Sequential(nn.Linear(noise_dim + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1))


class CTGANBenchmark(BaseBenchmark):
    def __init__(self, train: TrialData, seed: int, epochs: int = 160, noise_dim: int = 16):
        super().__init__(train, seed, "CTGAN")
        torch.manual_seed(int(seed))
        self.cols = [c for c in self.summary.columns if c not in {"event_time", "censoring_time"}]
        clean = self.summary[self.cols].apply(pd.to_numeric, errors="coerce")
        self.fills = clean.median(numeric_only=True).fillna(0.0)
        clean = clean.fillna(self.fills)
        self.scaler = StandardScaler().fit(clean)
        x = torch.tensor(self.scaler.transform(clean), dtype=torch.float32)
        a = self.train.A.astype(int)
        cond = torch.zeros((len(a), 2), dtype=torch.float32)
        cond[np.arange(len(a)), a] = 1.0
        self.noise_dim = int(noise_dim)
        self.generator = TinyConditionalGenerator(self.noise_dim, x.shape[1])
        opt = torch.optim.Adam(self.generator.parameters(), lr=1e-3)
        for _ in range(int(epochs)):
            idx = torch.randperm(x.shape[0])
            noise = torch.randn(x.shape[0], self.noise_dim)
            pred = self.generator(noise, cond[idx])
            loss = F.mse_loss(pred, x[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.generator.eval()

    def generate_trial(self, target_trial: TrialData) -> dict[str, np.ndarray]:
        n = len(target_trial.A)
        cond = torch.zeros((n, 2), dtype=torch.float32)
        cond[np.arange(n), target_trial.A.astype(int)] = 1.0
        with torch.no_grad():
            decoded = self.generator(torch.randn(n, self.noise_dim), cond).detach().cpu().numpy()
        sample = pd.DataFrame(self.scaler.inverse_transform(decoded), columns=self.cols)
        l = np.zeros((n, len(target_trial.time_grid), self.n_biomarkers), dtype=float)
        frac = target_trial.time_grid / max(target_trial.time_grid[-1], 1e-8)
        for ell in range(self.n_biomarkers):
            base = target_trial.L[:, 0, ell]
            final = pd.to_numeric(sample.get(f"L{ell + 1:02d}_final", base), errors="coerce").fillna(np.mean(base)).to_numpy()
            l[:, :, ell] = base.reshape(-1, 1) + (final - base).reshape(-1, 1) * frac.reshape(1, -1)
            l[:, :, ell] += self.rng.normal(0.0, 0.08 * (np.std(base) + 1.0), size=l[:, :, ell].shape)
            l[:, 0, ell] = base
        t_obs, delta = _sample_empirical_survival(self.rng, self.train, target_trial.A)
        return {"X": target_trial.X, "A": target_trial.A, "L": l, "T_obs": t_obs, "delta": delta}


def fit_benchmark(method: str, train: TrialData, seed: int, cfg: dict[str, Any]) -> BaseBenchmark:
    bench_cfg = cfg.get("benchmark_training", {})
    if method == "LMM-AFT":
        return LMMAFTBenchmark(train, seed, shared=False)
    if method == "JM-RE":
        return LMMAFTBenchmark(train, seed, shared=True)
    if method == "TVAE":
        return TVAEBenchmark(train, seed, epochs=int(bench_cfg.get("torch_epochs_tvae", 120)))
    if method == "CTGAN":
        return CTGANBenchmark(train, seed, epochs=int(bench_cfg.get("torch_epochs_ctgan", 160)))
    raise ValueError(f"Unknown 20260617 benchmark method: {method}")


def fit_all_benchmarks(train: TrialData, seed: int, cfg: dict[str, Any], output_dir: str | Path) -> dict[str, BaseBenchmark]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    models: dict[str, BaseBenchmark] = {}
    for i, method in enumerate(cfg.get("methods", {}).get("benchmarks", METHODS_20260617)):
        artifact = output / f"{method}_model.pkl"
        manifest_path = output / f"{method}_manifest.csv"
        if artifact.exists() and manifest_path.exists():
            with open(artifact, "rb") as f:
                model = pickle.load(f)
            model.model_artifact = str(artifact)
            models[method] = model
            continue

        model = fit_benchmark(method, train, int(seed) + 101 * (i + 1), cfg)
        models[method] = model
        with open(artifact, "wb") as f:
            pickle.dump(model, f)
        model.model_artifact = str(artifact)
        manifest = model.manifest()
        manifest["model_artifact"] = str(artifact)
        pd.DataFrame([manifest]).to_csv(output / f"{method}_manifest.csv", index=False)
    return models
