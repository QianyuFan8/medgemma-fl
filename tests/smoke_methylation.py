"""CPU end-to-end synthetic test, including real R limma and ridge; no model download."""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--r-library")
    p.add_argument("--classes", type=int, choices=(2, 3), default=3)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy(); env.pop("R_HOME", None)
    if a.r_library:
        env["R_LIBS_USER"] = str(Path(a.r_library).resolve())
    with tempfile.TemporaryDirectory(prefix="methylation_smoke_") as tmp:
        root = Path(tmp); rng = random.Random(72)
        ids = [f"TARGET-01-P{i:03d}-01A" for i in range(60)]
        matrix = root / "beta.tsv"
        with matrix.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            columns = [name for s in ids for name in (s+".Ave_Beta", s+".Detection.Pval")]
            w.writerow(["TargetID"] + columns)
            for j in range(80):
                row = [f"cg{j:08d}"]
                for i in range(60):
                    # Three classes, not just a binary code path.
                    value = (.15 + .30*(i//20) if j < 20 else .5) + rng.uniform(-.05, .05)
                    row.extend([value, .001])
                w.writerow(row)
        metadata = root / "samples.tsv"
        with metadata.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["sample_id", "patient_id", "diagnosis", "platform", "matrix_file", "beta_column", "include", "sample_type"])
            for i, sample in enumerate(ids):
                w.writerow([sample, "-".join(sample.split("-")[:3]), ["A","B","C"][i//(60//a.classes)], "450k", matrix, sample+".Ave_Beta", "TRUE", "01"])
        output = root / "prepared"
        cfg = {"metadata": str(metadata), "output": str(output), "metadata_reviewed": True,
               "top_k": 10, "variance_cap": 80, "seed": 1,
               "covariates": [], "numeric_covariates": []}
        config = root / "config.json"; config.write_text(json.dumps(cfg))
        subprocess.run([sys.executable, "prepare_methylation.py", "--config", str(config),
                        "--rscript", a.rscript], cwd=repo, env=env, check=True)
        for split in ("validation", "test"):
            subprocess.run([sys.executable, "evaluate_methylation.py", "ridge", "--data-dir", str(output),
                            "--output", str(root / f"ridge_{split}"), "--split", split, "--rscript", a.rscript],
                           cwd=repo, env=env, check=True)
        audit = list(csv.DictReader((output / "splits.tsv").open(), delimiter="\t"))
        heldout = {r["sample_id"] for r in audit if r["split"] != "train"}
        # Perturb EVERY held-out beta: panel must remain identical.
        with matrix.open() as f:
            raw = list(csv.reader(f, delimiter="\t"))
        for j, column in enumerate(raw[0]):
            if column.endswith(".Ave_Beta") and column[:-9] in heldout:
                for row in raw[1:]:
                    row[j] = str(1 - float(row[j]))
        with matrix.open("w") as f:
            csv.writer(f, delimiter="\t").writerows(raw)
        cfg["output"] = str(root / "perturbed"); config.write_text(json.dumps(cfg))
        subprocess.run([a.rscript, "methylation/prepare.R", str(config)], cwd=repo, env=env, check=True)
        assert (output / "selected_cpgs.tsv").read_text() == (root / "perturbed/selected_cpgs.tsv").read_text()
        assert (output / "train.csv").read_text() == (root / "perturbed/train.csv").read_text()
        print(f"PASS: real limma + {a.classes}-class ridge + client export; held-out beta perturbation leaves training panel/data unchanged.")


if __name__ == "__main__":
    main()
