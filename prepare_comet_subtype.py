"""COMET RMS ERMS-vs-ARMS preparation: audit -> prepare (limma panel) -> export (clients).

Labels are histologic subtypes from the advisor's COMET_RMS table (Sample ID, Subtype,
Comp Bio ID, X850k Tumor File Name, Slide Scan File). Only ERMS and ARMS are used.
Feature selection is centralized on training patients; model training may be
centralized (one client) or simulated-federated (several clients).
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

from methylation_utils import panel_fingerprint
from methylation_validate import validate_clients
from subtype_utils import MODALITIES, PROMPT_STYLE, SUBTYPE_CLASSES, make_subtype_record, read_mapping

MISSING = {"", "NA", "N/A", "NAN", "NONE"}
QC_MODES = {"lenient": 0.05, "strict": 0.0}


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


def matrix_header(path):
    with Path(path).open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f, delimiter="\t"))
    if len(header) < 2 or len(set(header)) != len(header):
        raise ValueError("Expected CpG-by-sample TSV with unique column names")
    return header


def list_slides(slide_dir=None, slide_list=None):
    ids = {}
    if slide_dir:
        for p in sorted(Path(slide_dir).glob("*.svs")):
            ids[p.stem] = str(p.resolve())
    if slide_list:
        for line in Path(slide_list).read_text().splitlines():
            name = line.strip()
            if name:
                ids.setdefault(Path(name).stem, name)
    return ids


def audit(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    mapping = read_mapping(a.mapping)
    meta = {r["Sample_Name"].strip(): r for r in read_tsv(a.comet_metadata)}
    header = matrix_header(a.comet_matrix)
    beta_cols = {c[:-9] if c.endswith(".Ave_Beta") else c: c for c in header[1:] if not c.endswith(".Detection.Pval")}
    slides = list_slides(a.slide_dir, a.slide_list)
    slide_users = Counter(r["slide_id"] for r in mapping if r["slide_id"])
    matrix_path = str(Path(a.comet_matrix).resolve())
    records = []
    for r in mapping:
        sentrix, subtype = r["sentrix_id"], r["subtype"].upper()
        m = meta.get(sentrix)
        col = beta_cols.get(sentrix)
        if subtype not in SUBTYPE_CLASSES:
            status = "subtype_not_in_task"
        elif col is None:
            status = "sentrix_not_in_matrix"
        elif m is None:
            status = "sentrix_not_in_metadata"
        elif m["Sample.Type"].strip().lower() != "patient tumor":
            status = "non_patient_tumor"
        elif any(m[k].strip().upper() != "PASS" for k in ("Final.QC", "Pass.QC")):
            status = "metadata_qc_failed"
        elif not r["patient_id"]:
            status = "unresolved_patient_identifier"
        else:
            status = "candidate_pending_label_review"
        slide_id = r["slide_id"]
        if not slide_id:
            slide_status = "no_slide_id"
        elif slide_users[slide_id] > 1:
            slide_status = "slide_shared_by_multiple_samples"
        elif slide_id in slides:
            slide_status = "available"
        else:
            slide_status = "not_available_locally"
        records.append(dict(sample_id="COMET:" + sentrix, patient_id="COMET:" + r["patient_id"] if r["patient_id"] else "",
                            comet_sample_id=r["sample_id"], compbio_id=r["compbio_id"], diagnosis=subtype or "NA",
                            source="COMET", source_group="COMET_RMS", platform="COMET_methylation",
                            sample_type="patient_tumor" if m and m["Sample.Type"].strip().lower() == "patient tumor" else "other",
                            status=status, matrix_file=matrix_path, beta_column=col or "",
                            detection_column=(col[:-9] + ".Detection.Pval") if col and col.endswith(".Ave_Beta") and (col[:-9] + ".Detection.Pval") in header else "",
                            slide_id=slide_id, slide_status=slide_status,
                            slide_file=slides.get(slide_id, "") if slide_status == "available" else "",
                            label_source=str(Path(a.mapping).resolve())))
    by_patient = defaultdict(list)
    for rec in records:
        if rec["status"] == "candidate_pending_label_review":
            by_patient[rec["patient_id"]].append(rec)
    for rows in by_patient.values():
        if len({x["diagnosis"] for x in rows}) > 1:
            for x in rows:
                x["status"] = "patient_subtype_conflict"
            continue
        # One array per patient, chosen by data availability then lexicographic ID; never by beta or label.
        ordered = sorted(rows, key=lambda x: (0 if x["slide_status"] == "available" else 1, x["sample_id"]))
        for x in ordered[1:]:
            x["status"] = "additional_sample_same_patient"
    out.mkdir(parents=True)
    write_tsv(out / "samples_audit.tsv", records)
    candidates = [x for x in records if x["status"] == "candidate_pending_label_review"]
    report = {
        "stage": "audit_only_review_before_preparing", "classes": SUBTYPE_CLASSES,
        "mapping_rows": len(mapping), "mapping_subtypes": dict(Counter(r["subtype"] for r in mapping)),
        "sample_status": dict(Counter(x["status"] for x in records)),
        "candidate_patients_by_subtype": dict(Counter(x["diagnosis"] for x in candidates)),
        "candidate_patients_with_slide_by_subtype": dict(Counter(x["diagnosis"] for x in candidates if x["slide_status"] == "available")),
        "slide_status": dict(Counter(x["slide_status"] for x in records)),
        "slides_without_mapping_row": sorted(set(slides) - {r["slide_id"] for r in mapping}),
        "shared_slides": sorted(s for s, n in slide_users.items() if n > 1),
        "independently_clinically_verified": False,
        "label_basis": "histologic Subtype column of the advisor's COMET_RMS table; ARMS approximates but does not equal FOXO1 fusion status",
        "source_files": [{"path": p, "bytes": Path(p).stat().st_size, "mtime_ns": Path(p).stat().st_mtime_ns}
                         for p in [matrix_path]],
        "samples_audit_sha256": sha256_file(out / "samples_audit.tsv"),
    }
    dump(out / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "source_files"}, indent=2))
    return report


def prepare(a):
    audit_dir, out = Path(a.audit_dir), Path(a.output).resolve()
    if out.exists():
        raise FileExistsError(out)
    report = json.loads((audit_dir / "audit.json").read_text())
    if sha256_file(audit_dir / "samples_audit.tsv") != report["samples_audit_sha256"]:
        raise ValueError("Audit changed; regenerate it")
    for source in report["source_files"]:
        stat = Path(source["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (source["bytes"], source["mtime_ns"]):
            raise ValueError("Matrix changed since audit; regenerate the audit")
    if not a.confirm_reviewed_labels:
        raise ValueError("Review samples_audit.tsv, then pass --confirm-reviewed-labels")
    rows = read_tsv(audit_dir / "samples_audit.tsv")
    included = [{**r, "include": "true"} for r in rows if r["status"] == "candidate_pending_label_review"]
    if a.require_slide:
        included = [r for r in included if r["slide_status"] == "available"]
    counts = Counter(r["diagnosis"] for r in included)
    if set(counts) != set(SUBTYPE_CLASSES) or min(counts.values()) < 5:
        raise ValueError(f"Need ERMS and ARMS with >=5 patients each: {dict(counts)}")
    if len({r["patient_id"] for r in included}) != len(included):
        raise ValueError("Duplicate patients remain")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = out.with_name(out.name + "_manifest.tsv")
    config = out.with_name(out.name + "_config.json")
    if manifest.exists() or config.exists():
        raise FileExistsError("Preparation sidecar files already exist; use a new experiment name")
    write_tsv(manifest, included)
    cfg = dict(metadata=str(manifest), output=str(out), metadata_reviewed=True,
               diagnoses=SUBTYPE_CLASSES, selection_mode="shared_fraction", source_groups=["COMET_RMS"],
               top_fraction=a.top_fraction, seed=a.seed, max_sample_missing=a.max_sample_missing,
               max_probe_missing=QC_MODES[a.qc_mode], qc_mode=a.qc_mode, detection_p=0.01, covariates=[])
    if a.exclude_probes:
        cfg["exclude_probes"] = str(Path(a.exclude_probes).resolve())
    dump(config, cfg)
    env = os.environ.copy()
    env.pop("R_HOME", None)
    if a.r_library:
        env["R_LIBS_USER"] = str(Path(a.r_library).resolve())
    subprocess.run([a.rscript, str(Path(__file__).parent / "methylation/prepare.R"), str(config)], env=env, check=True)
    dump(out / "cohort_provenance.json", {**report, "reviewed_labels_confirmed_by_user": True,
                                          "require_slide": a.require_slide, "qc_mode": a.qc_mode})
    features = read_tsv(out / "features.tsv")
    probes = [r["probe_id"] for r in features]
    summary = json.loads((out / "selection_summary.json").read_text())
    spec = dict(schema_version=1, classes=SUBTYPE_CLASSES, probes=probes, panel_id=panel_fingerprint(probes),
                selection=summary, features_sha256=sha256_file(out / "features.tsv"),
                training_medians=[float(r["training_median"]) for r in features], qc_mode=a.qc_mode,
                preprocessing=(f"{a.qc_mode} probe QC (detection p > 0.01 masked; training-probe missingness "
                               f"<= {QC_MODES[a.qc_mode]:.0%}); training-median imputation; beta rounded to 6 decimals."))
    dump(out / "features.json", spec)
    (out / "FEATURES.md").write_text(
        "# Frozen ERMS-vs-ARMS CpG panel\n\n"
        f"Panel ID: `{spec['panel_id']}`\n\nQC mode: {a.qc_mode}. Selected {len(probes)} CpGs: "
        f"ceil({a.top_fraction} x {summary['tested_nonconstant_cpgs']} tested CpGs), ranked by the training-only "
        "limma moderated test of ERMS vs ARMS on M-values.\n\nUse features.tsv in feature_order at every site. "
        "Do not select or reorder features using validation/test data.\n\n"
        "Labels are histologic subtypes, not fusion status. Derived from research data: do not commit "
        "patient-level exports to GitHub.\n")
    print(f"Prepared {len(probes)} CpGs ({a.qc_mode} QC) in {out}; now export a modality/partition layout.")


def allocate(n, weights):
    """Largest-remainder apportionment of n items over weights."""
    total = sum(weights)
    raw = [n * w / total for w in weights]
    counts = [int(x) for x in raw]
    for i in sorted(range(len(weights)), key=lambda i: raw[i] - counts[i], reverse=True)[: n - sum(counts)]:
        counts[i] += 1
    return counts


def assign_sites(rows, sites, partition, rng, dominant_fraction):
    """Return {patient_id: site}. iid = stratified round robin; skew = per-site dominant class."""
    assignment = {}
    for k, dx in enumerate(SUBTYPE_CLASSES):
        group = [r for r in rows if r["diagnosis"] == dx]
        rng.shuffle(group)
        if partition == "iid":
            offset = rng.randrange(len(sites))
            for j, r in enumerate(group):
                assignment[r["patient_id"]] = sites[(offset + j) % len(sites)]
        else:
            weights = [dominant_fraction if SUBTYPE_CLASSES[i % len(SUBTYPE_CLASSES)] == dx else 1 - dominant_fraction
                       for i in range(len(sites))]
            counts = allocate(len(group), weights)
            cursor = 0
            for site, c in zip(sites, counts):
                for r in group[cursor:cursor + c]:
                    assignment[r["patient_id"]] = site
                cursor += c
    return assignment


def export(a):
    root, out = Path(a.prepared), Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    if a.modality not in MODALITIES:
        raise ValueError("Unknown modality")
    if a.n_clients < 1:
        raise ValueError("--n-clients must be >= 1 (1 = centralized training)")
    spec = json.loads((root / "features.json").read_text())
    probes = spec["probes"]
    if spec["panel_id"] != panel_fingerprint(probes) or spec["features_sha256"] != sha256_file(root / "features.tsv"):
        raise ValueError("Frozen feature artifact was modified")
    use_meth = a.modality in ("methylation", "multimodal")
    use_image = a.modality in ("image", "multimodal")
    splits = {r["patient_id"]: r for r in read_tsv(root / "splits.tsv") if r["qc_pass"].upper() == "TRUE"}
    sites = [f"site-{i}" for i in range(1, a.n_clients + 1)]
    rng = random.Random(a.seed)
    montage_dir = Path(a.montage_dir) if a.montage_dir else None
    if use_image and montage_dir is None:
        raise ValueError("--montage-dir is required for image/multimodal exports")
    records, assignments, dropped = {}, [], Counter()
    counts = defaultdict(lambda: defaultdict(Counter))
    out.mkdir(parents=True)
    (out / "images").mkdir()
    for split in ("train", "validation", "test"):
        with (root / f"{split}.csv").open() as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["patient_id", "diagnosis"] + probes:
                raise ValueError("CSV feature order differs from frozen panel")
            rows = list(reader)
        kept = []
        for row in rows:
            meta = splits[row["patient_id"]]
            image = None
            if use_image:
                if meta["slide_status"] != "available":
                    dropped[f"{split}:{meta['slide_status']}"] += 1
                    continue
                source = montage_dir / f"{meta['slide_id']}.png"
                if not source.is_file():
                    dropped[f"{split}:montage_missing"] += 1
                    continue
                image = f"images/{row['patient_id'].replace(':', '_')}.png"
                shutil.copy2(source, out / image)
            kept.append((row, image))
        if {r["diagnosis"] for r, _ in kept} != set(SUBTYPE_CLASSES):
            raise ValueError(f"Split {split} lost a class after modality filtering")
        # Test patients also receive a site label (for per-site reporting) but never enter client folders.
        site_of = assign_sites([r for r, _ in kept], sites, a.partition, rng, a.dominant_fraction)
        buckets = {s: [] for s in sites}
        for row, image in kept:
            site = site_of[row["patient_id"]]
            record = make_subtype_record(row["patient_id"], row["diagnosis"], probes,
                                         [float(row[p]) for p in probes] if use_meth else [],
                                         SUBTYPE_CLASSES, a.modality, image=image, site_id=site)
            buckets[site].append(record)
            counts[site][split][row["diagnosis"]] += 1
            assignments.append(dict(patient_id=row["patient_id"], diagnosis=row["diagnosis"], split=split,
                                    site_id=site, image=image or ""))
            records.setdefault(split, []).append(record)
        if split != "test":
            records[f"{split}_by_site"] = buckets
    for s in sites:
        if not records["train_by_site"][s]:
            raise ValueError(f"{s} received no training patients; use fewer clients")
    for file in ("features.tsv", "features.json", "FEATURES.md", "selected_cpgs.tsv", "selection_summary.json",
                 "preprocessing.rds", "train.csv", "validation.csv", "test.csv", "splits.tsv", "config.json",
                 "cohort_provenance.json", "R_sessionInfo.txt"):
        shutil.copy2(root / file, out / file)
    for s in sites:
        dest = out / "clients" / s
        dest.mkdir(parents=True)
        dump(dest / "train.json", records["train_by_site"][s])
        if records["validation_by_site"][s]:
            dump(dest / "validation.json", records["validation_by_site"][s])
        else:
            print(f"warning: {s} has no validation patients; the client trains without per-round evaluation")
        for file in ("features.tsv", "features.json", "FEATURES.md"):
            shutil.copy2(root / file, dest / file)
    dump(out / "validation.json", [r for s in sites for r in records["validation_by_site"][s]])
    dump(out / "test.json", records["test"])
    write_tsv(out / "site_assignments.tsv", assignments)
    dump(out / "site_counts.json", counts)
    panel_probes = probes if use_meth else []
    dump(out / "task.json", dict(task="comet_subtype", classes=SUBTYPE_CLASSES, modality=a.modality,
                                 probes=panel_probes, panel_id=panel_fingerprint(panel_probes),
                                 features_sha256=spec["features_sha256"], panel_mode="shared" if use_meth else "image",
                                 n_clients=a.n_clients, sites=sites, partition=a.partition,
                                 dominant_fraction=a.dominant_fraction if a.partition == "skew" else None,
                                 seed=a.seed, prompt_style=PROMPT_STYLE, qc_mode=spec.get("qc_mode"),
                                 selection="centralized_train_only_limma_shared_fraction", selection_summary=spec["selection"],
                                 training_regime="centralized" if a.n_clients == 1 else "simulated_federated",
                                 dropped_for_modality=dict(dropped),
                                 prepared_split_hashes={s: sha256_file(root / f"{s}.csv") for s in ("train", "validation", "test")}))
    validate_clients(out / "clients", a.n_clients)
    print(json.dumps({"site_counts": counts, "dropped_for_modality": dict(dropped), "test": len(records["test"])}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    au = sub.add_parser("audit", help="Join mapping table, metadata, matrix header and slide listing")
    au.add_argument("--mapping", required=True, help="COMET_RMS.xlsx/.csv/.tsv from the advisor")
    au.add_argument("--comet-matrix", required=True)
    au.add_argument("--comet-metadata", required=True)
    au.add_argument("--slide-dir", help="Directory holding downloaded <slide>.svs files")
    au.add_argument("--slide-list", help="Text file with one slide filename per line (e.g. from dx ls)")
    au.add_argument("--output", required=True)
    pr = sub.add_parser("prepare", help="Training-only lenient/strict QC + limma top-fraction panel")
    pr.add_argument("--audit-dir", required=True)
    pr.add_argument("--output", required=True)
    pr.add_argument("--confirm-reviewed-labels", action="store_true")
    pr.add_argument("--qc-mode", choices=sorted(QC_MODES), default="lenient")
    pr.add_argument("--top-fraction", type=float, default=0.001)
    pr.add_argument("--require-slide", action="store_true", help="Restrict the cohort to patients with an available slide")
    pr.add_argument("--exclude-probes")
    pr.add_argument("--max-sample-missing", type=float, default=0.2)
    pr.add_argument("--seed", type=int, default=20261011)
    pr.add_argument("--rscript", default="Rscript")
    pr.add_argument("--r-library")
    ex = sub.add_parser("export", help="Export one modality/partition layout without rerunning selection")
    ex.add_argument("--prepared", required=True)
    ex.add_argument("--output", required=True)
    ex.add_argument("--modality", choices=MODALITIES, required=True)
    ex.add_argument("--partition", choices=("iid", "skew"), default="iid")
    ex.add_argument("--n-clients", type=int, default=3, help="1 = centralized training through the same job")
    ex.add_argument("--dominant-fraction", type=float, default=0.8)
    ex.add_argument("--montage-dir", help="Directory of <slide_id>.png montages from abmil_montage.py")
    ex.add_argument("--seed", type=int, default=20261011)
    a = p.parse_args()
    if a.command == "audit":
        audit(a)
    elif a.command == "prepare":
        if not 0 < a.top_fraction <= 1 or not 0 <= a.max_sample_missing < 1:
            p.error("Fractions out of range")
        prepare(a)
    else:
        if not 0 <= a.dominant_fraction <= 1:
            p.error("--dominant-fraction must be in [0, 1]")
        export(a)


if __name__ == "__main__":
    main()
