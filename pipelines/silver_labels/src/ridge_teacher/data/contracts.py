import json
import numpy as np
import pandas as pd
from ..common import digest
from ..cohort.policy import assert_training_policy

def load_inputs(cohort, matrix_dir):
    patients = pd.read_csv(cohort / "patients.csv", keep_default_na=False)
    assert_training_policy(patients)
    if patients.patient_id.duplicated().any():
        raise ValueError("Canonical patients duplicated")
    m = pd.read_csv(matrix_dir / "matrix_samples.csv", keep_default_na=False)
    receipt = json.loads((matrix_dir / "matrix_receipt.json").read_text())
    if receipt["cohort_sha256"] != digest(cohort / "patients.csv"):
        raise ValueError("Matrix was prepared for a different cohort manifest")
    if receipt["sample_qc_failures"]:
        raise ValueError("Sample QC failures require an explicit cohort revision; do not silently replace holdout patients")
    X = np.load(matrix_dir / "beta.npy", mmap_mode="r")
    if len(m) != len(X) or m.patient_id.duplicated().any() or m.sentrix_id.duplicated().any():
        raise ValueError("Matrix rows must be one reviewed array per canonical patient")
    p = patients.set_index("patient_id").loc[m.patient_id].reset_index()
    if not np.array_equal(p.sentrix_id.to_numpy(), m.sentrix_id.to_numpy()):
        raise ValueError("Array identity changed")
    required = patients[patients.teacher_train | patients.split.isin(["validation", "test", "train_gold", "train_silver_candidate"])]
    if set(p.patient_id) != set(required.patient_id):
        raise ValueError("Matrix does not cover the complete planned population")
    teacher = p.teacher_train.to_numpy(dtype=bool)
    if (teacher & p.split.isin(["validation", "test"]).to_numpy()).any():
        raise ValueError("Holdout leakage: teacher training includes a reserved patient")
    if not p.loc[teacher, "gold_label"].isin(["POS", "NEG"]).all():
        raise ValueError("Teacher must train on gold labels only")
    return p, X, receipt


