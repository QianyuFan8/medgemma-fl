"""Audited eight-class TARGET + COMET preparation with one frozen shared panel.

Separate audit, preparation and export stages. No model fitting on validation/test.
Feature selection is centralized on pooled training data in this simulation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from audit_mixed_methylation import audit as target_audit
from methylation_utils import make_record, panel_fingerprint
from methylation_validate import validate_clients

CLASSES = ["ALL", "AML", "CCSK", "NBL", "OS", "RT", "WT", "RMS"]
DEFAULT_SITES = {"site-1": ["AML", "CCSK"], "site-2": ["NBL", "OS", "WT"],
                 "site-3": ["ALL", "RT", "RMS"]}
MISSING = {"", "NA", "N/A", "NAN", "NONE"}


def read_tsv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_tsv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def comet_records(matrix, metadata, column_map=None):
    """Match exact array IDs, never infer a diagnosis from an unknown column."""
    meta = read_tsv(metadata)
    required = {"Sample_Name", "UID.Subject", "Disease", "Sample.Type", "Final.QC", "Pass.QC"}
    if not meta or not required <= set(meta[0]):
        raise ValueError(f"COMET metadata requires columns: {sorted(required)}")
    by_sample = {}
    for row in meta:
        sample = row["Sample_Name"].strip()
        if sample.upper() in MISSING or sample in by_sample:
            raise ValueError("Missing/duplicate COMET Sample_Name; resolve metadata first")
        by_sample[sample] = row
    with Path(matrix).open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="\t"))
    if len(header) < 2 or len(set(header)) != len(header):
        raise ValueError("Expected CpG-by-sample TSV with unique column names")
    explicit = {}
    if column_map:
        for row in read_tsv(column_map):
            col, sample = row["beta_column"], row["sample_name"]
            if col not in header[1:] or sample not in by_sample or col in explicit:
                raise ValueError("Invalid/duplicate COMET column map entry")
            explicit[col] = sample
    records, matched = [], set()
    for col in header[1:]:
        if col.endswith(".Detection.Pval"):
            continue
        sample = explicit.get(col, col[:-9] if col.endswith(".Ave_Beta") else col)
        row = by_sample.get(sample)
        patient = row["UID.Subject"].strip() if row else ""
        status = "candidate_pending_label_review"
        if row is None:
            status = "unresolved_comet_column"
        elif row["Disease"].strip().upper() != "RMS":
            status = "unexpected_comet_diagnosis"
        elif row["Sample.Type"].strip().lower() != "patient tumor":
            status = "non_patient_tumor"
        elif any(row[k].strip().upper() != "PASS" for k in ("Final.QC", "Pass.QC")):
            status = "metadata_qc_failed"
        elif patient.upper() in MISSING:
            status = "unresolved_patient_identifier"
        if row:
            if sample in matched:
                raise ValueError(f"Multiple beta columns map to COMET sample {sample}")
            matched.add(sample)
        dp = (col[:-9] if col.endswith(".Ave_Beta") else col) + ".Detection.Pval"
        records.append(dict(sample_id="COMET:" + sample,
                            patient_id="COMET:" + patient if patient.upper() not in MISSING else "",
                            source_patient_id=patient, source="COMET", source_group="COMET_RMS",
                            sample_type="patient_tumor" if row and row["Sample.Type"].strip().lower() == "patient tumor" else "other",
                            platform="COMET_methylation", diagnosis="RMS", status=status,
                            matrix_file=str(Path(matrix).resolve()), beta_column=col,
                            detection_column=dp if dp in header else "",
                            label_source=str(Path(metadata).resolve())))
    return records, {"metadata_rows": len(meta), "metadata_sample_types": dict(Counter(r["Sample.Type"] for r in meta)),
                     "metadata_samples_not_in_matrix": sorted(set(by_sample) - matched),
                     "metadata_sha256": sha256_file(metadata)}


def audit_all(matrix_root, comet_matrix, comet_metadata, output, column_map=None, patient_map=None):
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    target_audit(matrix_root, out / "target")
    records = []
    for r in read_tsv(out / "target/samples_audit.tsv"):
        dp = r["beta_column"][:-9] + ".Detection.Pval"
        records.append(dict(sample_id=r["sample_id"], patient_id=r["patient_id"],
                            source_patient_id=r["patient_id"], source="TARGET",
                            source_group="TARGET_450K" if r["platform"] == "450k" else "TARGET_EPIC",
                            sample_type=r["sample_type"], platform=r["platform"],
                            diagnosis=r["proposed_diagnosis"], status=r["status"],
                            matrix_file=r["matrix_file"], beta_column=r["beta_column"],
                            detection_column=dp, label_source=r["label_source"]))
    rms, rms_report = comet_records(comet_matrix, comet_metadata, column_map)
    records.extend(rms)
    if patient_map:
        mapping = {}
        known = {(r["source"], r["source_patient_id"]) for r in records if r["source_patient_id"]}
        for row in read_tsv(patient_map):
            key = (row["source"], row["source_patient_id"])
            if key in mapping or key not in known or row["patient_id"].upper() in MISSING:
                raise ValueError("Invalid/duplicate patient crosswalk entry")
            mapping[key] = row["patient_id"]
        for r in records:
            r["patient_id"] = mapping.get((r["source"], r["source_patient_id"]), r["patient_id"])
    candidates = defaultdict(list)
    for r in records:
        if r["status"] == "candidate_pending_label_review":
            candidates[r["patient_id"]].append(r)
    for rows in candidates.values():
        if len({r["diagnosis"] for r in rows}) > 1:
            for r in rows:
                r["status"] = "cross_cohort_label_conflict"
            continue
        # One representative sample per patient BEFORE splitting; no beta/label-based choice.
        # COMET metadata does not establish primary/relapse timing: lexicographic array ID only.
        ordered = sorted(rows, key=lambda r: ({"01": 0, "09": 1, "03": 2}.get(r["sample_type"], 3),
                                              r["sample_id"], r["matrix_file"]))
        for r in ordered[1:]:
            r["status"] = "additional_sample_same_patient"
    write_tsv(out / "samples_audit.tsv", records)
    source_files = sorted({r["matrix_file"] for r in records})
    report = {
        "stage": "audit_only_review_before_preparing", "classes": CLASSES,
        "sample_status": dict(Counter(r["status"] for r in records)),
        "candidate_patients_by_diagnosis": dict(Counter(r["diagnosis"] for r in records if r["status"] == "candidate_pending_label_review")),
        "comet": rms_report, "independently_clinically_verified": False,
        "patient_crosswalk": str(Path(patient_map).resolve()) if patient_map else None,
        "identity_caveat": "TARGET and COMET use different patient-ID systems. Absence of matching strings does not prove biological non-overlap; confirm with data owners or provide a crosswalk.",
        "source_files": [{"path": p, "bytes": Path(p).stat().st_size,
                          "mtime_ns": Path(p).stat().st_mtime_ns} for p in source_files],
        "samples_audit_sha256": sha256_file(out / "samples_audit.tsv"),
        "notes": ["Only Patient tumor COMET samples with both QC fields PASS are candidates.",
                  "COMET sample timing is not inferred from SJID suffixes; one array per patient is selected deterministically.",
                  "TARGET labels require the existing user-confirmed cohort-membership assertion.",
                  "Unresolved samples may be excluded explicitly, never guessed."]}
    dump(out / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("source_files", "comet")}, indent=2))
    return report


def prepare(a):
    audit_dir, out = Path(a.audit_dir), Path(a.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((audit_dir / "audit.json").read_text())
    if sha256_file(audit_dir / "samples_audit.tsv") != report["samples_audit_sha256"]:
        raise ValueError("Audit changed: regenerate it with explicit mapping inputs")
    for source in report["source_files"]:
        stat = Path(source["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise ValueError("Matrix changed since audit; regenerate the audit")
    rows = read_tsv(audit_dir / "samples_audit.tsv")
    if not a.confirm_reviewed_labels or not a.confirm_patient_identity:
        raise ValueError("Review audit/labels and confirm patient identities across sources before preparing")
    if any(r["status"] in ("cross_cohort_label_conflict", "unexpected_comet_diagnosis") for r in rows):
        raise ValueError("Resolve conflicting diagnoses before preparation")
    if any(r["status"].startswith("unresolved") for r in rows) and not a.exclude_unresolved:
        raise ValueError("Review unresolved rows or explicitly use --exclude-unresolved")
    included = [{**r, "include": "true"} for r in rows if r["status"] == "candidate_pending_label_review"]
    counts = Counter(r["diagnosis"] for r in included)
    if set(counts) != set(CLASSES) or min(counts.values()) < 5:
        raise ValueError(f"Need all eight diagnoses and >=5 unique patients per class: {dict(counts)}")
    if len({r["patient_id"] for r in included}) != len(included):
        raise ValueError("Duplicate patients remain")
    # R creates output and refuses overwrites. Keep configuration beside it.
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.with_name(out.name + "_manifest.tsv")
    config = out.with_name(out.name + "_config.json")
    if manifest.exists() or config.exists():
        raise FileExistsError("Preparation sidecar files already exist; use a new experiment name")
    write_tsv(manifest, included)
    cfg = dict(metadata=str(manifest), output=str(out), metadata_reviewed=True,
               diagnoses=CLASSES, selection_mode="shared_fraction", top_fraction=a.top_fraction,
               seed=a.seed, max_sample_missing=a.max_sample_missing,
               max_probe_missing=a.max_probe_missing, detection_p=0.01, covariates=[])
    if a.exclude_probes:
        cfg["exclude_probes"] = str(Path(a.exclude_probes).resolve())
    dump(config, cfg)
    env = os.environ.copy()
    env.pop("R_HOME", None)
    if a.r_library:
        env["R_LIBS_USER"] = str(Path(a.r_library).resolve())
    subprocess.run([a.rscript, str(Path(__file__).parent / "methylation/prepare.R"), str(config)], env=env, check=True)
    dump(out / "cohort_provenance.json", {**report, "reviewed_labels_confirmed_by_user": True,
                                          "patient_identity_confirmed_by_user": True,
                                          "exclude_unresolved": a.exclude_unresolved})
    features = read_tsv(out / "features.tsv")
    probes = [r["probe_id"] for r in features]
    summary = json.loads((out / "selection_summary.json").read_text())
    feature_spec = dict(schema_version=1, classes=CLASSES, probes=probes,
                        panel_id=panel_fingerprint(probes), selection=summary,
                        features_sha256=sha256_file(out / "features.tsv"),
                        training_medians=[float(r["training_median"]) for r in features],
                        preprocessing="Mask detection failures when available; frozen pooled-training median imputation; beta values rounded to 6 decimals.")
    dump(out / "features.json", feature_spec)
    (out / "FEATURES.md").write_text(
        "# Frozen eight-class CpG panel\n\n"
        f"Panel ID: `{feature_spec['panel_id']}`\n\n"
        f"Selected {len(probes)} CpGs: ceil({a.top_fraction} × {summary['tested_nonconstant_cpgs']} tested CpGs).\n\n"
        "Use features.tsv in feature_order (1-based) at every site. features.json records the ordered IDs, "
        "training medians and file hash. Do not select or reorder features using validation/test data. "
        "All sites receive the same eight diagnosis options.\n\n"
        "limma uses M-values and a pooled-training omnibus diagnosis test. BH FDR is reported; "
        "top-percent selection does not imply every selected feature has FDR < 0.05. "
        "No 20,000-feature variance cap is used. Feature filtering/selection is centralized, not federated.\n\n"
        "This is processed-beta QC, not full raw-array QC. Missing detection p-values and absent SNP/"
        "cross-reactivity annotations cannot be reconstructed. QC details are in selection_summary.json.\n\n"
        "These artifacts are derived from research data. Share only with authorized collaborators; "
        "do not automatically commit them or patient-level exports to GitHub.\n")
    print(f"Prepared {len(probes)} shared CpGs in {out}; now export IID and non-IID layouts.")


def export_sites(prepared, output, partition, seed=20260929, site_classes=None):
    root, out = Path(prepared), Path(output)
    if out.exists():
        raise FileExistsError(out)
    if partition not in ("iid", "non-iid"):
        raise ValueError("Unknown partition")
    spec = json.loads((root / "features.json").read_text())
    probes = spec["probes"]
    if (spec["panel_id"] != panel_fingerprint(probes) or
            spec["features_sha256"] != sha256_file(root / "features.tsv") or spec['classes'] != CLASSES):
        raise ValueError("Frozen feature artifact was modified")
    features = read_tsv(root / "features.tsv")
    if ([r['probe_id'] for r in features] != probes or
            [float(r['training_median']) for r in features] != spec['training_medians']):
        raise ValueError("Feature TSV and JSON disagree")
    sites = site_classes or DEFAULT_SITES
    flat = [dx for group in sites.values() for dx in group]
    if (set(sites) != {"site-1", "site-2", "site-3"} or not all(sites.values()) or
            sorted(flat) != sorted(CLASSES)):
        raise ValueError("Non-IID site map must assign each diagnosis to exactly one of three nonempty sites")
    owner = {dx: site for site, group in sites.items() for dx in group}
    rng = random.Random(seed)
    assignments, records, seen = [], {}, set()
    counts = defaultdict(lambda: defaultdict(Counter))
    for split in ("train", "validation", "test"):
        with (root / f"{split}.csv").open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["patient_id", "diagnosis"] + probes:
                raise ValueError("CSV feature order differs from frozen panel")
            rows = list(reader)
        if {r["diagnosis"] for r in rows} != set(CLASSES):
            raise ValueError(f"Missing diagnosis in {split}")
        buckets = {site: [] for site in sorted(sites)}
        for dx in CLASSES:
            group = [r for r in rows if r["diagnosis"] == dx]
            rng.shuffle(group)
            names = list(buckets)
            offset = min(range(3), key=lambda i: len(buckets[names[i]]))
            for j, row in enumerate(group):
                site = names[(offset+j) % 3] if partition == "iid" else owner[dx]
                if row["patient_id"] in seen:
                    raise ValueError("Patient leakage in prepared splits")
                seen.add(row["patient_id"])
                record = make_record(row["patient_id"], dx, probes, [float(row[p]) for p in probes],
                                     CLASSES, prompt_style="pediatric_v1")
                record["site_id"] = site
                buckets[site].append(record)
                counts[site][split][dx] += 1
                assignments.append(dict(patient_id=row["patient_id"], diagnosis=dx, split=split, site_id=site))
        records[split] = buckets
    out.mkdir(parents=True)
    for file in ("features.tsv", "features.json", "FEATURES.md", "selected_cpgs.tsv", "selection_summary.json",
                 "preprocessing.rds", "train.csv", "validation.csv", "test.csv", "splits.tsv",
                 "config.json", "cohort_provenance.json", "R_sessionInfo.txt"):
        shutil.copy2(root / file, out / file)
    for site in sorted(sites):
        dest = out / "clients" / site
        dest.mkdir(parents=True)
        for split in ("train", "validation"):
            dump(dest / f"{split}.json", records[split][site])
        for file in ("features.tsv", "features.json", "FEATURES.md"):
            shutil.copy2(root / file, dest / file)
    for split in ("validation", "test"):
        dump(out / f"{split}.json", [r for site in sorted(sites) for r in records[split][site]])
    write_tsv(out / "site_assignments.tsv", assignments)
    dump(out / "site_counts.json", counts)
    dump(out / "task.json", dict(classes=CLASSES, probes=probes, panel_id=spec["panel_id"],
                                 features_sha256=spec["features_sha256"], panel_mode="shared", n_clients=3,
                                 sites=sorted(sites), partition=partition, seed=seed,
                                 prompt_style="pediatric_v1", selection="centralized_train_only_limma_shared_fraction",
                                 selection_summary=spec["selection"],
                                 site_classes=sites if partition == "non-iid" else {s: CLASSES for s in sites},
                                 prepared_split_hashes={s: sha256_file(root / f"{s}.csv") for s in records}))
    validate_clients(out / "clients", 3)
    print(json.dumps(counts, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Read headers/metadata only; write private review artifacts")
    audit.add_argument("--matrix-root", required=True)
    audit.add_argument("--comet-matrix", required=True)
    audit.add_argument("--comet-metadata", required=True)
    audit.add_argument("--comet-column-map", help="Optional TSV: beta_column, sample_name; exact IDs")
    audit.add_argument("--patient-map", help="Optional TSV: source (TARGET/COMET), source_patient_id, patient_id")
    audit.add_argument("--output", required=True)
    prep = sub.add_parser("prepare", help="One shared training-only panel and fixed patient splits")
    prep.add_argument("--audit-dir", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--confirm-reviewed-labels", action="store_true")
    prep.add_argument("--confirm-patient-identity", action="store_true", help="Confirm cross-source non-overlap or reviewed crosswalk")
    prep.add_argument("--exclude-unresolved", action="store_true")
    prep.add_argument("--exclude-probes", help="Optional prespecified one-CpG-per-line exclusion list")
    prep.add_argument("--top-fraction", type=float, default=0.01)
    prep.add_argument("--max-sample-missing", type=float, default=0.2)
    prep.add_argument("--max-probe-missing", type=float, default=0.05)
    prep.add_argument("--seed", type=int, default=20260929)
    prep.add_argument("--rscript", default="Rscript")
    prep.add_argument("--r-library")
    export = sub.add_parser("export", help="Export a layout WITHOUT rerunning selection/splitting")
    export.add_argument("--prepared", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--partition", choices=("iid", "non-iid"), required=True)
    export.add_argument("--site-map", help="Optional JSON diagnosis-disjoint site map")
    export.add_argument("--seed", type=int, default=20260929)
    a = p.parse_args()
    if a.command == "audit":
        audit_all(a.matrix_root, a.comet_matrix, a.comet_metadata, a.output, a.comet_column_map, a.patient_map)
    elif a.command == "prepare":
        if not 0 < a.top_fraction <= 1 or not 0 <= a.max_sample_missing < 1 or not 0 <= a.max_probe_missing < 1:
            p.error("Fractions out of range")
        prepare(a)
    else:
        site_map = json.loads(Path(a.site_map).read_text()) if a.site_map else None
        export_sites(a.prepared, a.output, a.partition, a.seed, site_map)


if __name__ == "__main__":
    main()
