"""Explicit fit / silver / holdout evaluation entry point."""
import argparse
from pathlib import Path
from .data.contracts import load_inputs
from .train.fit import fit
from .inference.predict import predict

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["fit", "silver", "evaluate"])
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--split", choices=["validation", "test"])
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--min-confidence", type=float, default=.90)
    ap.add_argument("--selector", choices=["variance", "limma"], default="variance")
    ap.add_argument("--rscript", default="Rscript")
    args = ap.parse_args()
    p, X, receipt = load_inputs(args.cohort, args.matrix)
    if args.action == "fit":
        fit(args, p, X, receipt)
    else:
        if not args.model:
            ap.error("--model is required for scoring")
        predict(args, p, X, receipt)

if __name__ == "__main__":
    main()
