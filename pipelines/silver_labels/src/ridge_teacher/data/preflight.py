"""Validate a teammate's local raw files against the frozen patient manifest."""
import json
from pathlib import Path
import pandas as pd
from ..common import digest
from ..cohort.policy import assert_training_policy, POLICY_VERSION


def validate_sources(cohort, data_dir, beta, detection=None, allow_missing_detection=False):
    p = pd.read_csv(cohort / "patients.csv", keep_default_na=False)
    assert_training_policy(p)
    if p.patient_id.duplicated().any():
        raise ValueError("Duplicated canonical patients in manifest")
    if (p.teacher_train & p.split.isin(["validation", "test"])).any():
        raise ValueError("Holdout leakage in teacher membership")
    selected = p[p.teacher_train | p.split.isin(["validation", "test", "train_gold", "train_silver_candidate"])]
    if selected.sentrix_id.duplicated().any() or selected.sentrix_id.eq("").any():
        raise ValueError("Representative arrays must be unique and nonempty")
    meta_path = data_dir / "RMSs_meta.txt"
    fusion_path = data_dir / "COMET_RMS_fusion_status.xlsx"
    clinical_path = data_dir / "COMET_RMS.xlsx"
    meta = pd.read_csv(meta_path, sep="\t", dtype=str, keep_default_na=False)
    meta = meta[meta.Sentrix_ID.ne("")]
    if meta.Sentrix_ID.duplicated().any():
        raise ValueError("Duplicate Sentrix IDs in supplied metadata")
    meta = meta.set_index("Sentrix_ID")
    for _, r in selected.iterrows():
        if r.sentrix_id not in meta.index or meta.loc[r.sentrix_id, "UID.Subject"] != r.patient_id:
            raise ValueError("Canonical patient/array mapping changed; review the cohort")
        if meta.loc[r.sentrix_id, "Final.QC"] != "PASS":
            raise ValueError("A selected sample no longer has PASS metadata QC")
        if meta.loc[r.sentrix_id, "Sample.Type"].strip() != "Patient tumor":
            raise ValueError("A selected sample is not a patient tumor")
    fusion = pd.read_excel(fusion_path, keep_default_na=False)
    if fusion["Sample ID"].duplicated().any():
        raise ValueError("Duplicate fusion sample IDs")
    labels = fusion.set_index("Sample ID")["Final Fusion Status"].to_dict()
    for _, r in selected.iterrows():
        observed = labels.get(r.sample_id, "")
        if r.gold_label in {"POS", "NEG"} and observed != r.gold_label:
            raise ValueError("Gold fusion label changed; regenerate and review manifest")
        if r.split == "train_silver_candidate" and observed in {"POS", "NEG"}:
            raise ValueError("A silver candidate now has a real label; do not overwrite it")
    clinical = pd.read_excel(clinical_path, keep_default_na=False)
    unknown = set(p.loc[p.split.eq("train_silver_candidate"), "patient_id"])
    for _, r in clinical.iterrows():
        array = str(r["X850k Tumor File Name"]).strip()
        if array in meta.index and meta.loc[array, "UID.Subject"] in unknown and labels.get(r["Sample ID"]) in {"POS", "NEG"}:
            raise ValueError("A silver patient's other sample has a known label; review patient-level label status")
    header = pd.read_csv(beta, sep="\t", nrows=0).columns
    missing = [s for s in selected.sentrix_id if s + ".Ave_Beta" not in header]
    if missing:
        raise ValueError(f"Raw matrix lacks {len(missing)} selected arrays; this is not the complete agreed cohort")
    dh = pd.read_csv(detection, sep="\t", nrows=0).columns if detection else header
    covered = sum(s+".Detection.Pval" in dh or s in dh for s in selected.sentrix_id)
    if covered < len(selected) and not allow_missing_detection:
        raise ValueError("Detection-p columns missing. Supply --detection-file; --allow-missing-detection-p is an explicit alternate QC protocol.")
    return dict(training_policy=POLICY_VERSION,
                prediction_only=int(p.split.eq("prediction_only").sum()),
                teacher_patients=int(p.teacher_train.sum()), gold_student_train=int(p.split.eq("train_gold").sum()),
                validation=int(p.split.eq("validation").sum()), test=int(p.split.eq("test").sum()),
                silver_candidates=int(p.split.eq("train_silver_candidate").sum()), selected_arrays=len(selected),
                detection_p_covered_arrays=covered, cohort_sha256=digest(cohort/"patients.csv"),
                sources={str(f.resolve()):digest(f) for f in [meta_path,fusion_path,clinical_path]})
