"""Build an auditable sample sheet using ONLY supplied DNAnexus Sample/Dx metadata."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

COHORTS = {"450k": ["AML", "CCSK", "NBL", "OS", "WT"], "epic": ["ALL-P3", "RT"]}


def patient_id(sample):
    parts = sample.replace(".", "-").split("-")
    if len(parts) < 3 or parts[0] != "TARGET":
        raise ValueError(f"Invalid TARGET identifier: {sample}")
    return "-".join(parts[:3])


def build_manifest(metadata, matrix_root, platform, output):
    metadata, matrix_root, output = Path(metadata), Path(matrix_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    labels = defaultdict(set)
    with metadata.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not {"Sample", "Dx"} <= set(reader.fieldnames or []):
            raise ValueError("Expected DNAnexus TSV with Sample and Dx columns")
        for r in reader:
            dx = r["Dx"].strip().upper()
            if dx not in ("", "NA", "N/A", "NAN"):
                labels[patient_id(r["Sample"])].add(dx)
    records = []
    recognized = {"AML", "ALL", "MPAL", "CCSK", "NBL", "OS", "RT", "WT"}
    for cohort in COHORTS[platform]:
        suffix = "450k" if platform == "450k" else "EPIC"
        file = matrix_root / f"TARGET-{cohort}__{suffix}_beta.txt"
        with file.open() as f:
            header = next(csv.reader(f, delimiter="\t"))
        for col in header:
            if not col.endswith(".Ave_Beta"):
                continue
            sample = col[:-9]; parts = sample.split("-")
            pid = patient_id(sample)
            if len(parts) < 4:
                raise ValueError(f"Missing sample type: {sample}")
            st = parts[3][:2]; dxs = labels[pid]
            status = ("unmatched_diagnosis" if not dxs else "conflicting_diagnosis" if len(dxs)>1
                      else "matched" if next(iter(dxs)) in recognized else "unrecognized_diagnosis")
            dp = sample + ".Detection.Pval"
            records.append(dict(sample_id=sample, patient_id=pid, sample_type=st,
                                diagnosis=next(iter(dxs)) if len(dxs)==1 else "", platform=platform,
                                matrix_file=str(file.resolve()), beta_column=col,
                                detection_column=dp if dp in header else "", source_cohort=cohort,
                                label_status=status, recorded_labels="|".join(sorted(dxs)),
                                label_source=str(metadata.resolve()),
                                exclusion_reason=status if st in ("01", "03", "09") else "not_primary_tumor"))
    if not records:
        raise ValueError("No Ave_Beta columns found")
    records.sort(key=lambda r: (r["patient_id"], {"01":0,"09":1,"03":2}.get(r["sample_type"],99),r["sample_id"],r["matrix_file"]))
    seen = set()
    for r in records:
        if r["exclusion_reason"] == "matched":
            if r["patient_id"] in seen:
                r["exclusion_reason"] = "additional_sample_same_patient"
            else:
                seen.add(r["patient_id"])
                r["exclusion_reason"] = "included"
    counts = Counter(r["diagnosis"] for r in records if r["exclusion_reason"] == "included")
    for r in records:
        if r["exclusion_reason"] == "included" and counts[r["diagnosis"]] < 5:
            r["exclusion_reason"] = "class_below_min_patients"
        r["include"] = "TRUE" if r["exclusion_reason"] == "included" else "FALSE"
    output.mkdir(parents=True)
    with (output / "samples.tsv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), delimiter="\t"); w.writeheader(); w.writerows(records)
    report = {"metadata": str(metadata.resolve()), "platform": platform,
              "included_patients_by_diagnosis": dict(Counter(r["diagnosis"] for r in records if r["include"]=="TRUE")),
              "sample_status": dict(Counter(r["exclusion_reason"] for r in records)),
              "unmatched_by_source": dict(Counter(r["source_cohort"] for r in records if r["label_status"]=="unmatched_diagnosis"))}
    (output / "audit.json").write_text(json.dumps(report, indent=2)+"\n")
    config = {"metadata": str((output / "samples.tsv").resolve()), "metadata_reviewed": False,
              "output": f"data/methylation_{platform}_v1", "top_k":100, "fdr":.05,
              "variance_cap":20000,"seed":20260922,"covariates":[]}
    if platform == "450k":
        config["diagnoses"] = COHORTS[platform]
    (output / "config.json").write_text(json.dumps(config, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    print("Review samples.tsv and audit.json before setting metadata_reviewed=true. Missing diagnoses require more metadata, not filename labels.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", required=True)
    p.add_argument("--matrix-root", required=True)
    p.add_argument("--platform", choices=COHORTS, required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    build_manifest(a.metadata, a.matrix_root, a.platform, a.output)
