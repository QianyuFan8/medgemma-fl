# TARGET Childhood Cancer Dataset (RNA-seq & Clinical)

This directory encapsulates the standardized **TARGET** pediatric cancer dataset, including raw manifests, automated preprocessing pipelines, and clean feature/label matrices for **MedGemma** multi-modal LLM and prognostic modeling.

---

## Directory Architecture

```text
target_data/
├── README.md                            # Dataset documentation & preview tables
├── raw/                                 # Raw metadata & manifests
│   ├── gdc_manifest_rnaseq.txt          # GDC download manifest
│   ├── gdc_sample_sheet_rnaseq.tsv      # Sample-to-Case ID mapping file
│   ├── gdc_metadata_rnaseq.json         # Full experimental metadata
│   ├── clinical/                        # 5 raw GDC clinical TSV files
│   └── rnaseq/                          # [Git Ignored] 16.5GB downloaded TSVs
├── processed/                           # Processed gold-standard dataset
│   ├── target_clinical_cleaned.csv      # Clinical survival outcome labels (Tracked in Git)
│   └── target_rnaseq_log2_tpm.parquet   # Gene expression feature matrix (Shared via Release)
└── scripts/                             # Preprocessing pipeline
    └── process_target_data.ipynb        # End-to-end preprocessing notebook
```

---

## Dataset Statistics & Data Preview

### 1. Clinical Phenotype & Survival Outcome Data (`target_clinical_cleaned.csv`)
* **Dimension**: `(3214, 7)` — 3,214 patients across 7 clinical features.
* **Overall Survival Distribution**:
  * `0 (Alive)`: 2,184 patients (67.9%)
  * `1 (Dead)`: 1,030 patients (32.1%)
* **Primary Diagnosis Distribution (Top 5)**:
  1. *Acute myeloid leukemia, NOS*: 1,987 patients
  2. *Acute lymphocytic leukemia*: 573 patients
  3. *Neuroblastoma, NOS*: 130 patients
  4. *Wilms tumor*: 125 patients

#### Clinical Data Preview (First 5 Rows):
| case_id | project_id | primary_diagnosis | gender | age_years | os_event | os_days |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TARGET-20-PARLHW` | `TARGET-AML` | Acute myeloid leukemia, NOS | Unknown | 8.0 | **0** | 3020.0 |
| `TARGET-20-PAVSHX` | `TARGET-AML` | Acute myeloid leukemia, NOS | Unknown | 16.0 | **1** | 623.0 |
| `TARGET-20-PAXBAV` | `TARGET-AML` | Acute myeloid leukemia, NOS | Unknown | 8.0 | **1** | 206.0 |
| `TARGET-10-PAPCUR` | `TARGET-ALL-P2` | Acute lymphocytic leukemia | Unknown | 3.0 | **1** | 369.0 |
| `TARGET-20-PAVBEM` | `TARGET-AML` | Acute myeloid leukemia, NOS | Unknown | 4.0 | **0** | 2090.0 |

---

### 2. RNA-seq Gene Expression Matrix (`target_rnaseq_log2_tpm.parquet`)
* **Dimension**: `(3214, 49436)` — 3,214 patients across 49,436 gene features.
* **Normalization**: $\log_2(\text{TPM} + 1.0)$ normalized, non-zero and quality-filtered.

#### Expression Matrix Preview (First 5 Patients × 8 Genes):
| case_id (Index) | TSPAN6 | TNMD | DPM1 | SCYL3 | C1orf112 | FGR | CFH | FUCA2 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `TARGET-20-PARLHW` | 0.270230 | 0.000000 | 5.322696 | 1.726003 | 0.728922 | 4.941646 | 0.098015 | 3.193172 |
| `TARGET-20-PAVSHX` | 0.116099 | 0.000000 | 5.442632 | 2.325012 | 2.064090 | 6.637896 | 0.110096 | 5.850997 |
| `TARGET-20-PAXBAV` | 0.037031 | 0.000000 | 5.784920 | 3.013837 | 2.077346 | 3.980647 | 0.235605 | 3.435882 |
| `TARGET-10-PAPCUR` | 0.870187 | 0.000000 | 5.281353 | 1.222619 | 1.897628 | 5.795120 | 0.257252 | 3.341730 |
| `TARGET-20-PAVBEM` | 0.069565 | 0.105544 | 5.660777 | 2.532990 | 3.745280 | 7.202201 | 0.039981 | 3.858439 |

---

## Quick Start for Model Training

### 1. Download Processed Parquet Feature Matrix
The 150MB feature matrix can be downloaded directly from GitHub Release:

```bash
mkdir -p target_data/processed
wget https://github.com/QianyuFan8/medgemma-fl/releases/download/v1.0-dataset/target_rnaseq_log2_tpm.parquet -P target_data/processed/
```

### 2. Load Features (X) & Labels (Y) in Python
```python
import pandas as pd

# Load Expression Feature Matrix X
df_expr = pd.read_parquet("target_data/processed/target_rnaseq_log2_tpm.parquet")

# Load Survival Labels Y
df_clinical = pd.read_csv("target_data/processed/target_clinical_cleaned.csv").set_index('case_id')

# Verify patient alignment
assert list(df_expr.index) == list(df_clinical.index)
print("Successfully aligned 3,214 patients for MedGemma training!")
```

---

## Data Sources & Acknowledgments
* Data Provider: [NIH NCI Genomic Data Commons (GDC) Data Portal](https://portal.gdc.cancer.gov/)
* Program: TARGET (Therapeutically Applicable Research To Generate Effective Treatments)
