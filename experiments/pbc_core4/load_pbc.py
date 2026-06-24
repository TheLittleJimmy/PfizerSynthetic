from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]

RAW_LONGITUDINAL_NAMES = [
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "serBilir",
    "serChol",
    "albumin",
    "alkaline",
    "SGOT",
    "platelets",
    "prothrombin",
    "histologic",
]

LONGITUDINAL_RENAME = {
    "serBilir": "bili",
    "serChol": "cholesterol",
    "SGOT": "ast",
    "histologic": "stage",
}

LONGITUDINAL_NAMES = [
    "ascites",
    "hepatomegaly",
    "spiders",
    "edema",
    "bili",
    "cholesterol",
    "albumin",
    "alkaline",
    "ast",
    "platelets",
    "prothrombin",
    "stage",
]

STATIC_BASELINE_NAMES = ["age", "sex"]
TREATMENT_NAME = "treatment"
CONTROL_VALUE = 0
TREATED_VALUE = 1


@dataclass(frozen=True)
class PBCData:
    subjects: pd.DataFrame
    longitudinal: pd.DataFrame
    survival: pd.DataFrame
    splits: dict[str, list[int]]
    dictionary: dict[str, Any]


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return ROOT / path


def _map_yes_no(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"yes", "y", "1", "true"}:
        return 1.0
    if text in {"no", "n", "0", "false"}:
        return 0.0
    return np.nan


def _map_edema(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "despite" in text:
        return 2.0
    if "no diuretics" in text:
        return 1.0
    if "no edema" in text:
        return 0.0
    try:
        return float(value)
    except Exception:
        return np.nan


def _map_sex(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text.startswith("f"):
        return 0.0
    if text.startswith("m"):
        return 1.0
    return np.nan


def _map_treatment(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if "placebo" in text:
        return float(CONTROL_VALUE)
    if "penicil" in text or "penicillamine" in text:
        return float(TREATED_VALUE)
    try:
        raw = float(value)
    except Exception:
        return np.nan
    if raw in {0.0, 1.0}:
        return raw
    return np.nan


def _status_codes(status: pd.Series) -> tuple[pd.Series, pd.Series]:
    text = status.astype(str).str.strip().str.lower()
    composite = (~text.eq("alive")).astype(int)
    death = text.eq("dead").astype(int)
    return composite, death


def load_local_pbc(source_data_dir: str | Path) -> pd.DataFrame:
    data_dir = project_path(source_data_dir)
    pbc_path = data_dir / "pbc2.csv"
    if pbc_path.exists():
        return pd.read_csv(pbc_path)
    raise FileNotFoundError(
        f"No local PBC/PBC2 data found at {pbc_path}. Add pbc2.csv or install an R/Python package "
        "with survival::pbcseq/JMbayes2::pbc2 and extend experiments/pbc_core4/load_pbc.py."
    )


def normalize_pbc(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = raw.copy()
    required = {"id", "years", "status", "drug", "age", "sex", "year", *RAW_LONGITUDINAL_NAMES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"PBC source table is missing required columns: {missing}")
    df[TREATMENT_NAME] = df["drug"].map(_map_treatment)
    randomized = df[df[TREATMENT_NAME].notna()].copy()
    if randomized.empty:
        raise ValueError("No non-missing randomized treatment rows were found.")

    randomized["subject_id"] = randomized["id"].astype(int)
    randomized["visit_time"] = pd.to_numeric(randomized["year"], errors="coerce").astype(float)
    randomized["sex"] = randomized["sex"].map(_map_sex)
    for col in ["ascites", "hepatomegaly", "spiders"]:
        randomized[col] = randomized[col].map(_map_yes_no)
    randomized["edema"] = randomized["edema"].map(_map_edema)
    randomized["stage"] = pd.to_numeric(randomized["histologic"], errors="coerce") - 1.0
    randomized["bili"] = pd.to_numeric(randomized["serBilir"], errors="coerce")
    randomized["cholesterol"] = pd.to_numeric(randomized["serChol"], errors="coerce")
    randomized["ast"] = pd.to_numeric(randomized["SGOT"], errors="coerce")
    for col in ["albumin", "alkaline", "platelets", "prothrombin", "age"]:
        randomized[col] = pd.to_numeric(randomized[col], errors="coerce")

    subject_rows = []
    long_rows = []
    survival_rows = []
    for sid, group in randomized.sort_values(["subject_id", "visit_time"]).groupby("subject_id", sort=True):
        base = group.sort_values("visit_time").iloc[0].copy()
        composite_event, death_event = _status_codes(pd.Series([base["status"]]))
        subject = {
            "subject_id": int(sid),
            "source_id": int(sid),
            TREATMENT_NAME: int(base[TREATMENT_NAME]),
            "treatment_label": "D-penicillamine" if int(base[TREATMENT_NAME]) == TREATED_VALUE else "placebo",
            "age": float(base["age"]),
            "sex": float(base["sex"]),
        }
        for name in LONGITUDINAL_NAMES:
            subject[f"L0_{name}"] = float(base[name]) if pd.notna(base[name]) else np.nan
        subject_rows.append(subject)
        survival_rows.append({
            "subject_id": int(sid),
            "time": float(pd.to_numeric(base["years"], errors="coerce")),
            "event_composite": int(composite_event.iloc[0]),
            "event_death": int(death_event.iloc[0]),
            "status": str(base["status"]),
            "treatment": int(base[TREATMENT_NAME]),
        })
        for visit_index, (_, row) in enumerate(group.sort_values("visit_time").iterrows()):
            rec: dict[str, Any] = {
                "subject_id": int(sid),
                "visit_index": int(visit_index),
                "visit_time": float(row["visit_time"]),
                "treatment": int(row[TREATMENT_NAME]),
            }
            for name in LONGITUDINAL_NAMES:
                rec[name] = float(row[name]) if pd.notna(row[name]) else np.nan
            long_rows.append(rec)

    subjects = pd.DataFrame(subject_rows).sort_values("subject_id").reset_index(drop=True)
    longitudinal = pd.DataFrame(long_rows).sort_values(["subject_id", "visit_time"]).reset_index(drop=True)
    survival = pd.DataFrame(survival_rows).sort_values("subject_id").reset_index(drop=True)
    dictionary = {
        "source": "local pbc2 data/pbc2.csv",
        "n_subjects": int(len(subjects)),
        "n_longitudinal_rows": int(len(longitudinal)),
        "treatment_mapping": {"placebo": 0, "D-penicillamine": 1, "source_label": "D-penicil"},
        "primary_endpoint": "death or transplant composite, event_composite = status != alive",
        "sensitivity_endpoint": "death only, event_death = status == dead; transplant censored",
        "baseline_variables": STATIC_BASELINE_NAMES + [f"L0_{name}" for name in LONGITUDINAL_NAMES],
        "longitudinal_variables": LONGITUDINAL_NAMES,
        "data_leakage_rule": "Generation may use only W, L0, requested treatment, and future grid.",
    }
    return subjects, longitudinal, survival, dictionary


def make_splits(subjects: pd.DataFrame, survival: pd.DataFrame, seed: int) -> dict[str, list[int]]:
    merged = subjects[["subject_id", TREATMENT_NAME]].merge(
        survival[["subject_id", "event_composite"]], on="subject_id", how="left"
    )
    labels = merged[TREATMENT_NAME].astype(str) + "_" + merged["event_composite"].astype(str)
    idx = np.arange(len(merged))
    train_idx, temp_idx = train_test_split(idx, test_size=0.40, random_state=seed, stratify=labels)
    temp_labels = labels.iloc[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed + 1, stratify=temp_labels)
    out = {
        "train": merged.iloc[train_idx]["subject_id"].astype(int).sort_values().tolist(),
        "validation": merged.iloc[val_idx]["subject_id"].astype(int).sort_values().tolist(),
        "test": merged.iloc[test_idx]["subject_id"].astype(int).sort_values().tolist(),
    }
    overlap = set(out["train"]) & set(out["validation"]) | set(out["train"]) & set(out["test"]) | set(out["validation"]) & set(out["test"])
    if overlap:
        raise RuntimeError(f"Subject split leakage detected: {sorted(overlap)[:10]}")
    return out


def write_data_dictionary(path: Path, dictionary: dict[str, Any]) -> None:
    lines = [
        "# PBC Core-4 Data Dictionary",
        "",
        f"Source: {dictionary['source']}",
        "",
        "## Treatment Coding",
        "",
        "- control = placebo = 0",
        "- treatment = D-penicillamine = 1",
        "",
        "## Endpoints",
        "",
        f"- Primary: {dictionary['primary_endpoint']}",
        f"- Sensitivity: {dictionary['sensitivity_endpoint']}",
        "",
        "## Baseline Variables",
        "",
        *[f"- {name}" for name in dictionary["baseline_variables"]],
        "",
        "## Longitudinal Variables",
        "",
        *[f"- {name}" for name in dictionary["longitudinal_variables"]],
        "",
        "## Leakage Rule",
        "",
        dictionary["data_leakage_rule"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preprocess_to_disk(
    source_data_dir: str | Path,
    processed_dir: str | Path,
    seed: int,
) -> PBCData:
    raw = load_local_pbc(source_data_dir)
    subjects, longitudinal, survival, dictionary = normalize_pbc(raw)
    splits = make_splits(subjects, survival, seed)
    processed = project_path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    subjects.to_csv(processed / "pbc_subjects.csv", index=False)
    longitudinal.to_csv(processed / "pbc_longitudinal.csv", index=False)
    survival.to_csv(processed / "pbc_survival.csv", index=False)
    with (processed / f"pbc_splits_seed{seed}.json").open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)
    write_data_dictionary(processed / "pbc_data_dictionary.md", dictionary)
    return PBCData(subjects, longitudinal, survival, splits, dictionary)


def load_processed(processed_dir: str | Path, seed: int) -> PBCData:
    processed = project_path(processed_dir)
    subjects = pd.read_csv(processed / "pbc_subjects.csv")
    longitudinal = pd.read_csv(processed / "pbc_longitudinal.csv")
    survival = pd.read_csv(processed / "pbc_survival.csv")
    with (processed / f"pbc_splits_seed{seed}.json").open("r", encoding="utf-8") as f:
        splits = json.load(f)
    dictionary = {
        "source": "processed",
        "n_subjects": int(len(subjects)),
        "n_longitudinal_rows": int(len(longitudinal)),
        "treatment_mapping": {"placebo": 0, "D-penicillamine": 1},
        "primary_endpoint": "death or transplant composite, event_composite = status != alive",
        "sensitivity_endpoint": "death only, event_death = status == dead; transplant censored",
        "baseline_variables": STATIC_BASELINE_NAMES + [f"L0_{name}" for name in LONGITUDINAL_NAMES],
        "longitudinal_variables": LONGITUDINAL_NAMES,
        "data_leakage_rule": "Generation may use only W, L0, requested treatment, and future grid.",
    }
    return PBCData(subjects, longitudinal, survival, splits, dictionary)

