# PBC2 Dataset

## Overview

PBC2 is a randomised, double-blind, placebo-controlled trial conducted at the Mayo Clinic studying D-penicillamine for the treatment of primary biliary cholangitis (PBC), a chronic autoimmune liver disease. The study enrolled **312 patients** randomised to D-penicillamine (**n = 158**) or placebo (**n = 154**), with a predominantly female population (88.5%). Over a median follow-up of approximately 5 years (max 14.3 years), 140 patients died, 29 received liver transplants, and 143 were alive at last contact. Each patient had an average of 6.2 longitudinal follow-up visits (range 1–16, total 1,945 visit records) capturing repeated measurements of clinical and laboratory variables.

## Variable Descriptions

### Randomisation Variable

| Variable | Type | Description |
|----------|------|-------------|
| `drug` | Binary (0/1) | Treatment assignment: 0 = D-penicillamine, 1 = placebo |

### Pre-Randomisation (Baseline) Variables

| Variable | Type | Description |
|----------|------|-------------|
| `sex` | Binary (0/1) | Sex: 0 = female, 1 = male |
| `age` | Continuous (26–78) | Age at enrolment (years) |

### Survival (Time-to-Event) Variables

| Variable | Type | Description |
|----------|------|-------------|
| `years` | Continuous (0.1–14.3) | Time to death/transplant or censoring (years) |
| `status` | Categorical | Event status: alive, dead, or transplanted |

### Baseline Clinical Variables (Measured at Enrolment Only)

| Variable | Type | Description |
|----------|------|-------------|
| `serChol` | Continuous | Serum cholesterol (mg/dL); 28 missing values — excluded from modelling |

### Baseline + Longitudinal Clinical Variables

These 11 variables are first measured at enrolment (visit_time = 0) and then **repeatedly** at follow-up visits (avg 6.2 visits per patient, range 1–16, total 1,945 records in `longitudinal.csv`). The baseline snapshot in `pbc2_id.csv` corresponds to the first longitudinal record.

| Variable | Type | Description |
|----------|------|-------------|
| `ascites` | Binary (0/1) | Presence of ascites (abdominal fluid) |
| `hepatomegaly` | Binary (0/1) | Presence of hepatomegaly (enlarged liver) |
| `spiders` | Binary (0/1) | Presence of spider angiomata (vascular lesions) |
| `edema` | Ordinal (0/1/2) | Edema severity: 0 = none, 1 = untreated or resolved with diuretics, 2 = despite diuretic therapy |
| `histologic` | Ordinal (0–3) | Histologic stage of disease (0 = stage I, 3 = stage IV) |
| `serBilir` | Continuous (0.3–28.0) | Serum bilirubin (mg/dL) |
| `albumin` | Continuous (1.96–4.64) | Serum albumin (g/dL) |
| `alkaline` | Continuous (289–13,862) | Alkaline phosphatase (U/L) |
| `SGOT` | Continuous (26–457) | Serum glutamic-oxaloacetic transaminase (U/L) |
| `platelets` | Continuous (62–563) | Platelet count (per µL); 4 missing values filled with median |
| `prothrombin` | Continuous (9.0–15.2) | Prothrombin time (seconds) |
