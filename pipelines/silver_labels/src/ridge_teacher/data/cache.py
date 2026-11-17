"""Cache full raw CpG universe, with label-independent measurement/sample QC.

No global variance ranking, imputation, scaling or LIMMA is done here.
All fitted feature filtering belongs inside teacher.py's CV estimator.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..common import digest as sha
from ..cohort.policy import assert_training_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--beta", type=Path, required=True, help="Original RMSs_beta.txt with TargetID and <Sentrix>.Ave_Beta columns")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--detection-file", type=Path, help="Optional separate detection-p TSV: TargetID + Sentrix or Sentrix.Detection.Pval columns")
    p.add_argument("--allow-missing-detection-p", action="store_true", help="Explicit alternate QC protocol, recorded in receipt")
    a = p.parse_args()
    if a.out.exists():
        raise ValueError("Output exists; use a new directory")
    patients = pd.read_csv(a.cohort / "patients.csv", keep_default_na=False)
    assert_training_policy(patients)
    m = patients[patients.teacher_train | patients.split.isin(["validation", "test", "train_gold", "train_silver_candidate"])].copy()
    if m.patient_id.duplicated().any() or m.sentrix_id.duplicated().any() or m.sentrix_id.eq("").any():
        raise ValueError("Require one unique reviewed array per patient")
    header = pd.read_csv(a.beta, sep="\t", nrows=0).columns.tolist()
    cols = [s + ".Ave_Beta" for s in m.sentrix_id]
    missing = set(cols) - set(header)
    if missing:
        raise ValueError(f"Raw beta missing {len(missing)} planned arrays; revise eligibility explicitly")
    dp_header = pd.read_csv(a.detection_file, sep="\t", nrows=0).columns.tolist() if a.detection_file else header
    dp = [s + ".Detection.Pval" if s + ".Detection.Pval" in dp_header else s for s in m.sentrix_id]
    present = [s in dp_header for s in dp]
    if not all(present) and not a.allow_missing_detection_p:
        raise ValueError("Detection-p columns missing; obtain raw detection p-values or explicitly approve alternate QC")
    ids = pd.read_csv(a.beta, sep="\t", usecols=["TargetID"], dtype=str).TargetID.str.strip()
    keep = ids.str.fullmatch(r"cg[0-9]+", na=False)
    cpg = ids[keep].tolist()
    if len(set(cpg)) != len(cpg) or not cpg:
        raise ValueError("Duplicate or missing CpG identifiers")
    if a.detection_file:
        dp_ids = pd.read_csv(a.detection_file, sep="\t", usecols=["TargetID"], dtype=str).TargetID.str.strip()
        if not ids.equals(dp_ids):
            raise ValueError("Detection-p rows must exactly match beta TargetID order; align upstream, never guess")
    a.out.mkdir(parents=True)
    matrix = np.lib.format.open_memmap(a.out / "beta.npy", mode="w+", dtype="float32", shape=(len(m), len(cpg)))
    detected = [s for s, yes in zip(dp, present) if yes]
    use = ["TargetID"] + cols + ([] if a.detection_file else detected)
    det_chunks = iter(pd.read_csv(a.detection_file, sep="\t", usecols=["TargetID"]+detected, chunksize=10000)) if a.detection_file else None
    offset = 0
    for block in pd.read_csv(a.beta, sep="\t", usecols=use, chunksize=10000):
        det_block = next(det_chunks) if det_chunks else block
        block = block[block.TargetID.astype(str).str.strip().str.fullmatch(r"cg[0-9]+", na=False)]
        det_block = det_block.loc[block.index]
        if not len(block):
            continue
        x = block[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float32", copy=True)
        x[(~np.isfinite(x)) | (x < 0) | (x > 1)] = np.nan
        for j, yes in enumerate(present):
            if yes:
                v = pd.to_numeric(det_block[dp[j]], errors="coerce").to_numpy()
                x[(~np.isfinite(v)) | (v < 0) | (v > .01), j] = np.nan
        if block.TargetID.astype(str).str.strip().tolist() != cpg[offset:offset+len(block)]:
            raise ValueError("Raw beta probe order changed while reading")
        matrix[:, offset:offset+len(block)] = x.T
        offset += len(block)
    assert offset == len(cpg)
    matrix.flush()
    # Scan one array at a time to avoid a full-size boolean temporary.
    fractions = np.array([np.mean(~np.isfinite(matrix[i])) for i in range(len(m))])
    qc = m[["patient_id", "sample_id", "sentrix_id", "split"]].copy()
    qc["missing_fraction"], qc["pass_sample_qc"] = fractions, fractions <= .20
    qc.to_csv(a.out / "sample_qc.csv", index=False)
    m[["patient_id", "sample_id", "sentrix_id"]].to_csv(a.out / "matrix_samples.csv", index=False)
    (a.out / "cpg_ids.txt").write_text("\n".join(cpg) + "\n")
    receipt = dict(cohort_sha256=sha(a.cohort / "patients.csv"), raw_beta_path=str(a.beta.resolve()),
                   raw_beta_sha256=sha(a.beta), n_patients=len(m), n_cpgs=len(cpg),
                   detection_p_threshold=.01, max_sample_missing=.20,
                   detection_p_covered_arrays=sum(present), missing_detection_p_allowed=a.allow_missing_detection_p,
                   detection_file=str(a.detection_file.resolve()) if a.detection_file else None,
                   detection_sha256=sha(a.detection_file) if a.detection_file else None,
                   sample_qc_failures=int((fractions > .20).sum()),
                   feature_selection="None; full raw CpG universe retained for fold-specific filtering")
    (a.out / "matrix_receipt.json").write_text(json.dumps(receipt, indent=2))
    if receipt["sample_qc_failures"]:
        raise ValueError("QC failures recorded. Stop and revise eligibility; never silently replace holdout patients")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
