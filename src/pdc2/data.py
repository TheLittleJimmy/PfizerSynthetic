from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


BASELINE_COLUMNS = [
    "time",
    "censor",
    "drug",
    "sex",
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "histologic",
    "serBilir",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "age",
]


@dataclass
class LongitudinalSpec:
    name: str
    type: str
    nclass: int | None = None
    mean: float = 0.0
    std: float = 1.0
    categories: tuple[float, ...] = ()


@dataclass
class LongitudinalPanel:
    subject_ids: np.ndarray
    times: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor
    raw_values: np.ndarray
    specs: list[LongitudinalSpec]
    time_min: float
    time_max: float

    @property
    def continuous_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.specs) if s.type in {"real", "pos", "count"}]

    @property
    def categorical_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.specs) if s.type in {"cat", "ordinal"}]


@dataclass
class PDC2Bundle:
    raw_df: pd.DataFrame
    encoded_df: pd.DataFrame
    types: list[dict[str, Any]]
    miss_mask: torch.Tensor
    true_miss_mask: torch.Tensor
    longitudinal: LongitudinalPanel
    ids_df: pd.DataFrame
    y_dim_partition: list[int]
    static_feature_count: int
    treatment: torch.Tensor
    treatment_name: str
    treatment_n_classes: int
    category_values: dict[str, list[float]] | None = None


def read_types(path: Path, survival: str = "dynamic") -> list[dict[str, Any]]:
    del survival
    types = pd.read_csv(path).fillna("").to_dict("records")
    for item in types:
        if item["name"] == "survcens":
            item["type"] = "surv_dynamic"
        item["dim"] = str(int(item["dim"]))
        if item.get("nclass", "") != "":
            item["nclass"] = str(int(float(item["nclass"])))
        else:
            item["nclass"] = ""
    return types


def load_baseline(data_dir: Path, survival: str) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    raw = pd.read_csv(data_dir / "data_phasesyn.csv", header=None)
    raw.columns = BASELINE_COLUMNS
    ids = pd.read_csv(data_dir / "pbc2_id.csv")
    if len(raw) != len(ids):
        raise ValueError(f"Baseline row count {len(raw)} does not match pbc2_id row count {len(ids)}.")
    types = read_types(data_dir / "data_types_phasesyn_piecewise.csv", survival=survival)
    return raw, types, ids


def _type_by_name(types: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for t in types:
        name = "time" if t["name"] == "survcens" else t["name"]
        out[name] = t
    return out


def _cat_mapper(values: np.ndarray, nclass: int) -> dict[float, int]:
    observed = sorted(float(v) for v in pd.Series(values).dropna().unique())
    mapper = {v: i for i, v in enumerate(observed[:nclass])}
    for i in range(nclass):
        mapper.setdefault(float(i), i)
    return mapper


def build_longitudinal_panel(
    long_df: pd.DataFrame,
    baseline_types: list[dict[str, Any]],
    n_subjects: int,
    max_visits: int | None = None,
    time_min_override: float | None = None,
    time_max_override: float | None = None,
) -> LongitudinalPanel:
    type_map = _type_by_name(baseline_types)
    value_cols = [c for c in long_df.columns if c not in {"patient_id", "visit_time"}]
    grouped = long_df.sort_values(["patient_id", "visit_time"]).groupby("patient_id", sort=True)
    inferred_max = int(grouped.size().max())
    max_visits = inferred_max if max_visits is None else min(int(max_visits), inferred_max)
    n_vars = len(value_cols)

    raw_values = np.full((n_subjects, max_visits, n_vars), np.nan, dtype=np.float32)
    masks = np.zeros((n_subjects, max_visits, n_vars), dtype=np.float32)
    times = np.zeros((n_subjects, max_visits), dtype=np.float32)

    specs: list[LongitudinalSpec] = []
    cat_maps: dict[str, dict[float, int]] = {}
    for col in value_cols:
        t = type_map.get(col, {"type": "real", "nclass": ""})
        ftype = t["type"]
        nclass = int(t["nclass"]) if ftype in {"cat", "ordinal"} and t.get("nclass") else None
        if nclass is not None:
            cat_maps[col] = _cat_mapper(long_df[col].values, nclass)
            cats = tuple(sorted(cat_maps[col], key=lambda x: cat_maps[col][x])[:nclass])
        else:
            cats = ()
        specs.append(LongitudinalSpec(name=col, type=ftype, nclass=nclass, categories=cats))

    for pid, rows in grouped:
        pid_int = int(pid)
        if pid_int < 0 or pid_int >= n_subjects:
            continue
        rows = rows.head(max_visits)
        for visit_idx, (_, row) in enumerate(rows.iterrows()):
            times[pid_int, visit_idx] = float(row["visit_time"])
            for var_idx, col in enumerate(value_cols):
                value = row[col]
                if pd.isna(value):
                    continue
                spec = specs[var_idx]
                if spec.type in {"cat", "ordinal"}:
                    raw_values[pid_int, visit_idx, var_idx] = cat_maps[col].get(float(value), 0)
                else:
                    raw_values[pid_int, visit_idx, var_idx] = float(value)
                masks[pid_int, visit_idx, var_idx] = 1.0

    observed_visits = (masks.sum(axis=-1) > 0).astype(np.float32)
    obs_times = times[observed_visits.astype(bool)]
    time_min = float(obs_times.min()) if obs_times.size else 0.0
    time_max = float(obs_times.max()) if obs_times.size else 1.0
    if time_min_override is not None:
        time_min = min(time_min, float(time_min_override))
    if time_max_override is not None:
        time_max = max(time_max, float(time_max_override))
    time_rng = max(time_max - time_min, 1e-6)
    times_norm = ((times - time_min) / time_rng) * observed_visits

    values = np.nan_to_num(raw_values, nan=0.0).astype(np.float32)
    for i, spec in enumerate(specs):
        obs = raw_values[:, :, i][masks[:, :, i].astype(bool)]
        if spec.type in {"real", "pos", "count"}:
            mean = float(np.mean(obs)) if obs.size else 0.0
            std = float(np.std(obs)) if obs.size else 1.0
            std = max(std, 1e-6)
            values[:, :, i] = ((values[:, :, i] - mean) / std) * masks[:, :, i]
            spec.mean = mean
            spec.std = std
        else:
            values[:, :, i] = values[:, :, i] * masks[:, :, i]

    return LongitudinalPanel(
        subject_ids=np.arange(n_subjects),
        times=torch.tensor(times_norm, dtype=torch.float32),
        values=torch.tensor(values, dtype=torch.float32),
        masks=torch.tensor(masks, dtype=torch.float32),
        raw_values=raw_values,
        specs=specs,
        time_min=time_min,
        time_max=time_max,
    )


def validate_complete_l0(panel: LongitudinalPanel, baseline_time_eps: float = 1e-6) -> None:
    """Validate the architecture assumption that each subject has one complete t=0 row."""
    observed_rows = (panel.masks.sum(dim=-1) > 0)
    baseline_rows = (panel.times.abs() <= float(baseline_time_eps)) & observed_rows
    counts = baseline_rows.sum(dim=1)
    if not torch.all(counts == 1):
        bad = torch.nonzero(counts != 1, as_tuple=False).flatten()[:10].detach().cpu().tolist()
        raise ValueError(
            "Every subject must have exactly one observed t=0 longitudinal baseline row; "
            f"bad subject positions include {bad}."
        )
    batch_idx = torch.arange(panel.times.shape[0])
    baseline_idx = baseline_rows.float().argmax(dim=1)
    l0_mask = panel.masks[batch_idx, baseline_idx]
    incomplete = l0_mask < 1.0
    if bool(incomplete.any().item()):
        bad_subjects = torch.nonzero(incomplete.any(dim=1), as_tuple=False).flatten()[:10].detach().cpu().tolist()
        raise ValueError(
            "L0 is assumed fully observed, but at least one baseline longitudinal variable is missing; "
            f"bad subject positions include {bad_subjects}."
        )


def encode_raw_dataframe(
    raw_df: pd.DataFrame,
    types: list[dict[str, Any]],
    miss_mask: torch.Tensor | None = None,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor]:
    encoded_parts: list[np.ndarray] = []
    encoded_names: list[str] = []
    feature_mask = torch.ones((len(raw_df), len(types)), dtype=torch.float32) if miss_mask is None else miss_mask.float().clone()

    for idx, feature in enumerate(types):
        ftype = feature["type"]
        name = feature["name"]
        if ftype.startswith("surv"):
            data = raw_df[["time", "censor"]].to_numpy(dtype=np.float32)
            encoded_parts.append(data)
            encoded_names.extend(["time", "censor"])
            continue

        series = raw_df[name]
        missing = series.isna().to_numpy()
        if missing.any():
            feature_mask[missing, idx] = 0.0

        if ftype in {"cat", "ordinal"}:
            nclass = int(feature["nclass"])
            mapper = _cat_mapper(series.to_numpy(), nclass)
            mapped = series.map(lambda x: mapper.get(float(x), 0) if not pd.isna(x) else 0).to_numpy(dtype=int)
            one_hot = np.zeros((len(raw_df), nclass), dtype=np.float32)
            one_hot[np.arange(len(raw_df)), np.clip(mapped, 0, nclass - 1)] = 1.0
            encoded_parts.append(one_hot)
            encoded_names.extend([f"{name}_{j}" for j in range(nclass)])
        else:
            values = series.fillna(0.0).to_numpy(dtype=np.float32).reshape(-1, 1)
            encoded_parts.append(values)
            encoded_names.append(name)

    encoded = np.concatenate(encoded_parts, axis=1)
    true_miss_mask = feature_mask.clone()
    return pd.DataFrame(encoded, columns=encoded_names), feature_mask, true_miss_mask


def y_dim_partition_for_types(types: list[dict[str, Any]], y_static: int) -> list[int]:
    return [int(y_static) for _ in types]


def load_pdc2_bundle(cfg: dict[str, Any]) -> PDC2Bundle:
    data_dir = Path(cfg["dataset"]["data_dir"])
    survival = cfg["model"].get("survival", "dynamic")
    treatment_name = str(cfg.get("model", {}).get("treatment_variable_name", "drug"))
    baseline_df, baseline_types, ids_df = load_baseline(data_dir, survival=survival)
    long_df = pd.read_csv(data_dir / "longitudinal.csv")
    panel = build_longitudinal_panel(
        long_df,
        baseline_types,
        len(baseline_df),
        max_visits=cfg["dataset"].get("max_visits"),
        time_min_override=float(pd.to_numeric(baseline_df["time"], errors="coerce").min()),
        time_max_override=float(pd.to_numeric(baseline_df["time"], errors="coerce").max()),
    )
    validate_complete_l0(panel, float(cfg.get("model", {}).get("baseline_time_eps", 1e-6)))

    if treatment_name not in baseline_df:
        raise ValueError(f"Treatment variable {treatment_name!r} is not present in the PDC2 baseline table.")
    treatment_type = next((t for t in baseline_types if t["name"] == treatment_name), None)
    treatment_n_classes = int(treatment_type.get("nclass") or 2) if treatment_type is not None else 2
    treatment_int = baseline_df[treatment_name].fillna(0).to_numpy(dtype=int)
    treatment = torch.nn.functional.one_hot(
        torch.tensor(np.clip(treatment_int, 0, treatment_n_classes - 1), dtype=torch.long),
        num_classes=treatment_n_classes,
    ).float()

    raw_df = baseline_df.copy()
    types = [dict(t) for t in baseline_types if t["name"] != treatment_name]
    miss_mask = torch.ones((len(raw_df), len(types)), dtype=torch.float32)
    static_count = len(types)

    y_part = y_dim_partition_for_types(
        types,
        int(cfg["model"].get("y_dim_static", 15)),
    )
    encoded_df, miss_mask, true_miss_mask = encode_raw_dataframe(raw_df, types, miss_mask)
    l0_names = {spec.name for spec in panel.specs}
    for idx, feature in enumerate(types):
        if feature["name"] in l0_names:
            miss_mask[:, idx] = 1.0
            true_miss_mask[:, idx] = 1.0
    return PDC2Bundle(
        raw_df=raw_df,
        encoded_df=encoded_df,
        types=types,
        miss_mask=miss_mask,
        true_miss_mask=true_miss_mask,
        longitudinal=panel,
        ids_df=ids_df,
        y_dim_partition=y_part,
        static_feature_count=static_count,
        treatment=treatment,
        treatment_name=treatment_name,
        treatment_n_classes=treatment_n_classes,
    )


def select_overfit_indices(bundle: PDC2Bundle, subset_size: int = 32, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    events = np.flatnonzero(bundle.raw_df["censor"].to_numpy(dtype=float) > 0.5)
    censored = np.flatnonzero(bundle.raw_df["censor"].to_numpy(dtype=float) <= 0.5)
    if len(events) == 0 or len(censored) == 0:
        raise ValueError("Overfit subset requires both event and censored subjects.")
    n_event = max(1, subset_size // 2)
    n_cens = subset_size - n_event
    chosen = np.concatenate([
        rng.choice(events, size=min(n_event, len(events)), replace=False),
        rng.choice(censored, size=min(n_cens, len(censored)), replace=False),
    ])
    if len(chosen) < subset_size:
        remaining = np.setdiff1d(np.arange(len(bundle.raw_df)), chosen)
        extra = rng.choice(remaining, size=subset_size - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    rng.shuffle(chosen)
    return chosen.astype(int)


def subset_bundle(bundle: PDC2Bundle, indices: np.ndarray) -> PDC2Bundle:
    idx = np.asarray(indices, dtype=int)
    panel = bundle.longitudinal
    sub_panel = LongitudinalPanel(
        subject_ids=panel.subject_ids[idx],
        times=panel.times[idx],
        values=panel.values[idx],
        masks=panel.masks[idx],
        raw_values=panel.raw_values[idx],
        specs=panel.specs,
        time_min=panel.time_min,
        time_max=panel.time_max,
    )
    return PDC2Bundle(
        raw_df=bundle.raw_df.iloc[idx].reset_index(drop=True),
        encoded_df=bundle.encoded_df.iloc[idx].reset_index(drop=True),
        types=bundle.types,
        miss_mask=bundle.miss_mask[idx],
        true_miss_mask=bundle.true_miss_mask[idx],
        longitudinal=sub_panel,
        ids_df=bundle.ids_df.iloc[idx].reset_index(drop=True),
        y_dim_partition=bundle.y_dim_partition,
        static_feature_count=bundle.static_feature_count,
        treatment=bundle.treatment[idx],
        treatment_name=bundle.treatment_name,
        treatment_n_classes=bundle.treatment_n_classes,
    )
