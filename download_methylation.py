"""Download only selected DNAnexus platform matrices; login separately with dx login."""
import argparse
from pathlib import Path
import subprocess
from methylation_manifest import COHORTS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--platform", choices=COHORTS, required=True)
    p.add_argument("--project", default="project-JBXfX9Q0vQQYJ9k4BxGVyYzJ")
    p.add_argument("--output", default="data/raw/methylation")
    p.add_argument("--download", action="store_true", help="Without this flag only print planned transfers")
    p.add_argument("--sample-lists", action="store_true", help="Also download the per-cohort .samples files")
    a = p.parse_args()
    root = Path(a.output)
    transfers = []
    for cohort in COHORTS[a.platform]:
        suffix = "450k" if a.platform == "450k" else "EPIC"
        name = f"TARGET-{cohort}__{suffix}_beta.txt"
        transfers.append((f"idat_sample_sheet_{name}", root / name))
        if a.sample_lists:
            list_name = "all.samples" if cohort == "ALL-P3" else cohort.lower()+".samples"
            transfers.append((list_name, root / list_name))
    for remote_name, dst in transfers:
        if dst.exists():
            print(f"Already local, skipped: {dst}"); continue
        source = f"{a.project}:/methyl/{remote_name}"
        print(f"{source} -> {dst}", flush=True)
        if a.download:
            root.mkdir(parents=True, exist_ok=True)
            part = Path(str(dst)+".part")
            subprocess.run(["dx", "download", source, "-o", str(part)], check=True)
            part.rename(dst)


if __name__ == "__main__":
    main()
