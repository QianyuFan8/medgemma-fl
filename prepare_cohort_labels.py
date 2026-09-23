"""450K cohort-membership labels explicitly confirmed by the user, not independent clinical verification."""
import argparse
import csv
import json
from pathlib import Path

from methylation_manifest import build_manifest, COHORTS


def prepare(matrix_root, output, confirmed=False):
    if not confirmed:
        raise ValueError("Explicit --confirm-cohort-labels is required; filenames alone do not establish diagnosis")
    root, out = Path(matrix_root), Path(output)
    evidence = out.parent / (out.name + "_label_evidence.tsv")
    if out.exists() or evidence.exists():
        raise FileExistsError("Use a new output name; do not overwrite label evidence or prior manifests")
    allowed, rows, audit = {}, [], {}
    for cohort in COHORTS["450k"]:
        source = root / (cohort.lower() + ".samples")
        with source.open() as f:
            cells = [cell.strip() for row in csv.reader(f, delimiter="\t") for cell in row]
        samples = [s[:-9] for s in cells if s.endswith(".Ave_Beta")]
        if not samples or len(set(samples)) != len(samples):
            raise ValueError(f"Missing or duplicate Ave_Beta sample identifiers in {source}")
        matrix = root / f"TARGET-{cohort}__450k_beta.txt"
        with matrix.open() as f:
            header = next(csv.reader(f, delimiter="\t"))
        matrix_samples = {s[:-9] for s in header if s.endswith(".Ave_Beta")}
        absent = set(samples) - matrix_samples
        if absent:
            raise ValueError(f"{source} lists {len(absent)} samples absent from its matrix; check versions")
        allowed[cohort] = set(samples)
        rows.extend({"Sample":s,"Dx":cohort,"Source":str(source.resolve())} for s in samples)
        audit[cohort] = {"source":str(source.resolve()),"listed_samples":len(samples),
                         "matrix_samples_not_listed":len(matrix_samples-set(samples))}
    out.parent.mkdir(parents=True, exist_ok=True)
    with evidence.open("w") as f:
        w=csv.DictWriter(f,fieldnames=["Sample","Dx","Source"],delimiter="\t")
        w.writeheader(); w.writerows(rows)
    build_manifest(evidence, root, "450k", out, allowed_samples=allowed)
    provenance = {"label_basis":"user_confirmed_sample_list_cohort_membership",
                  "independently_clinically_verified_by_software":False,"platform":"450k",
                  "sources":audit,"note":"Review exclusions and per-class patient counts before training. EPIC is not handled here."}
    (out/"label_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
    print("Saved explicit cohort-label provenance; config still requires metadata_reviewed=true after audit.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matrix-root",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--confirm-cohort-labels",action="store_true")
    a=p.parse_args()
    prepare(a.matrix_root,a.output,a.confirm_cohort_labels)
