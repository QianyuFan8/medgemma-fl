"""Audit canonical patients and reserve gold holdouts BEFORE teacher fitting.

Input is the reviewed COMET sample linkage export, not a preselected CpG matrix.
The manifest remains a candidate manifest until raw beta/detection data pass QC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from .policy import POLICY_VERSION, training_excluded, excluded_subtype, normalized_subtype

GOLD = {"POS", "NEG"}
CONFLICTS = ("identity_conflict", "map_conflict", "slide_mapping_conflict",
             "slide_cross_patient_conflict", "fusion_identity_conflict")


def truth(row, key):
    v = str(row[key]).strip().lower()
    if v not in {"true", "false", "1", "0"}:
        raise ValueError(f"Invalid boolean {key}: {v!r}")
    return v in {"true", "1"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def choose(rows):
    # One observed array per patient; no repeated-patient weighting in Ridge.
    # Prefer paired and block-matched samples, then stable identifiers; no scores.
    return min(rows, key=lambda r: (not truth(r, "live_he"),
                                  not truth(r, "block_match"),
                                  r["Sample ID"], r["sentrix_id"]))


def audit(rows):
    groups = defaultdict(list)
    for r in rows:
        if truth(r, "is_patient_tumor"):
            if not r["patient_id"]:
                raise ValueError("Patient-tumor record lacks canonical identity")
            groups[r["patient_id"]].append(r)
    patients, samples = [], []
    for uid, records in sorted(groups.items()):
        labels = {r["fusion_status"] for r in records if r["fusion_status"] in GOLD}
        conflict = len(labels) > 1
        safe = [r for r in records if truth(r, "has_methylation_metadata")
                and not any(truth(r, c) for c in CONFLICTS)]
        paired = [r for r in safe if truth(r, "live_he")]
        # Image mapping conflicts do not invalidate a separately verified
        # methylation/fusion sample for the image-free teacher.
        gold = [r for r in records if truth(r, "methylation_fusion_eligible")
                and r["fusion_status"] in GOLD and not truth(r, "fusion_identity_conflict")]
        paired_gold = [r for r in gold if r in paired]
        role = ("label_conflict_review" if conflict else
                "paired_gold" if paired_gold else
                "paired_silver_candidate" if paired and not labels else
                "paired_cross_sample_review" if paired and labels else
                "methylation_gold_only" if gold else "not_eligible")
        eligible = not conflict and bool(gold)
        representative = choose(paired_gold or gold or paired or safe or records)
        label = next(iter(labels)) if len(labels) == 1 else "UNKNOWN"
        patients.append(dict(patient_id=uid, category=role, gold_label=label,
                             teacher_eligible=eligible,
                             sample_id=representative["Sample ID"],
                             sentrix_id=representative["sentrix_id"],
                             slide_id=representative["slide_id"],
                             subtype=representative["subtype"],
                             observed_subtypes="|".join(sorted({r["subtype"] for r in records})),
                             training_excluded=any(excluded_subtype(r["subtype"]) for r in records),
                             block_match=truth(representative, "block_match"),
                             raw_values_verified=False))
        for r in records:
            samples.append(dict(patient_id=uid, sample_id=r["Sample ID"],
                                sentrix_id=r["sentrix_id"], slide_id=r["slide_id"],
                                observed_fusion=r["fusion_status"],
                                safe_pair=r in paired, safe_gold=r in gold,
                                representative=r is representative, issues=r["issues"]))
    # A reused array / slide must not cross canonical identities.
    for key in ("sentrix_id", "slide_id"):
        owners = defaultdict(set)
        for r in samples:
            if r["safe_pair"] or (key == "sentrix_id" and r["safe_gold"]):
                if r[key]:
                    owners[r[key]].add(r["patient_id"])
        if any(len(v) > 1 for v in owners.values()):
            raise ValueError(f"Unresolved cross-patient {key}")
    return patients, samples


def allocate(patients, seed=20261116, n_validation=20, n_test=20):
    paired = [p for p in patients if p["category"] == "paired_gold"]
    counts = Counter(p["gold_label"] for p in paired)
    if set(counts) != GOLD or len(paired) <= n_validation + n_test:
        raise ValueError("Insufficient gold paired patients for requested holdouts")
    def quotas(n):
        raw = {k: n * counts[k] / len(paired) for k in sorted(GOLD)}
        q = {k: math.floor(v) for k, v in raw.items()}
        for k in sorted(raw, key=lambda k: (-(raw[k] - q[k]), k))[:n - sum(q.values())]:
            q[k] += 1
        if min(q.values()) < 1:
            raise ValueError("Each gold holdout must have both classes")
        return q
    val, test = quotas(n_validation), quotas(n_test)
    assignment = {}
    for label in sorted(GOLD):
        ids = [p["patient_id"] for p in paired if p["gold_label"] == label]
        ids.sort(key=lambda uid: hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest())
        if len(ids) - val[label] - test[label] < 1:
            raise ValueError("Requested holdouts exhaust a training class")
        for i, uid in enumerate(ids):
            assignment[uid] = "validation" if i < val[label] else "test" if i < val[label] + test[label] else "train_gold"
    # Keep existing holdouts and class counts; move an excluded gold patient
    # out of training by a deterministic same-label validation swap.
    by_id = {p["patient_id"]: p for p in patients}
    for p in sorted(paired, key=lambda p: p["patient_id"]):
        uid = p["patient_id"]
        if assignment[uid] == "train_gold" and training_excluded(p):
            replacements = sorted(other for other, split in assignment.items()
                                  if split == "validation"
                                  and by_id[other]["gold_label"] == p["gold_label"]
                                  and normalized_subtype(by_id[other]["subtype"]) in {"arms", "erms"}
                                  and not training_excluded(by_id[other]))
            if not replacements:
                raise ValueError("No eligible same-label validation patient for subtype swap")
            assignment[uid], assignment[replacements[0]] = "validation", "train_gold"
    for p in patients:
        uid = p["patient_id"]
        p["split"] = assignment.get(uid, "train_silver_candidate" if p["category"] == "paired_silver_candidate" else "excluded")
        # A valid methylation+gold sample can train Ridge even if its patient's
        # image is from a different sample; that image remains excluded.
        if training_excluded(p) and p["split"] not in {"validation", "test"}:
            if p["category"] in {"paired_silver_candidate", "methylation_gold_only", "paired_cross_sample_review"}:
                p["split"] = "prediction_only"
        p["teacher_train"] = bool(p["teacher_eligible"] and not training_excluded(p)
                                  and p["split"] not in {"validation", "test", "prediction_only"})
    return patients


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--inventory", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--validation", type=int, default=20)
    ap.add_argument("--test", type=int, default=20)
    args = ap.parse_args()
    if args.out.exists():
        raise ValueError("Output exists; preserve the existing split, use a new directory for a revision")
    with args.samples.open() as f:
        rows = list(csv.DictReader(f))
    inventory = json.loads(args.inventory.read_text())
    slides = {s.removesuffix(".svs") for s in inventory["filenames"]}
    for r in rows:
        if truth(r, "live_he") != (r["slide_id"] in slides):
            raise ValueError("H&E inventory disagrees with linkage export")
    patients, samples = audit(rows)
    allocate(patients, args.seed, args.validation, args.test)
    split_by_id = {p["patient_id"]: p["split"] for p in patients}
    for r in samples:
        r["split"] = split_by_id[r["patient_id"]]
    summary = dict(seed=args.seed, stage="metadata_verified_raw_values_pending",
                   categories=dict(Counter(p["category"] for p in patients)),
                   paired_gold_labels=dict(Counter(p["gold_label"] for p in patients if p["category"] == "paired_gold")),
                   partitions={}, teacher_train_patients=sum(p["teacher_train"] for p in patients),
                   methylation_gold_patients=sum(p["teacher_eligible"] for p in patients),
                   training_policy=POLICY_VERSION,
                   allowed_training_subtypes=["ERMS", "ARMS"],
                   historical_holdout_exposure="Exploratory re-split; some patients were evaluated in prior experiments.",
                   sources={str(args.samples.resolve()): sha(args.samples), str(args.inventory.resolve()): sha(args.inventory)})
    for split in ("train_gold", "validation", "test", "train_silver_candidate", "prediction_only"):
        selected = [p for p in patients if p["split"] == split]
        summary["partitions"][split] = dict(patients=len(selected), labels=dict(Counter(p["gold_label"] for p in selected)))
    args.out.mkdir(parents=True)
    write_csv(args.out / "patients.csv", patients)
    write_csv(args.out / "samples.csv", samples)
    write_csv(args.out / "teacher_train.csv", [p for p in patients if p["teacher_train"]])
    write_csv(args.out / "prediction_only.csv", [p for p in patients if p["split"] == "prediction_only"], fields=list(patients[0]))
    write_csv(args.out / "paired_predictions_requested.csv", [p for p in patients if p["category"] in {"paired_gold", "paired_silver_candidate"}])
    template = [dict(patient_id=p["patient_id"], sample_id=p["sample_id"], label_source="silver_pending",
                     score_pos="", predicted_label="", included=False, teacher_model_sha256="", reason="Teacher not fitted; not a ground-truth label")
                for p in patients if p["split"] == "train_silver_candidate"]
    write_csv(args.out / "silver_labels_PENDING.csv", template)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    main()
