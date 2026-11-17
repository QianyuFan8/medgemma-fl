"""One-command orchestration with frozen splits and explicit holdout evaluation."""
import argparse
import importlib.metadata
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import joblib
import numpy as np
import pandas as pd
from .common import digest
from .data.preflight import validate_sources
from .data.contracts import load_inputs
from .image_processing.manifest import export_manifest


def run_stage(module, options, destination, log, env):
    marker = destination / "stage_complete.json"
    if marker.exists():
        print(f"Reuse completed stage: {destination.name}", flush=True)
        return
    if destination.exists():
        # Preserve interrupted output rather than merging partial results.
        suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        destination.rename(destination.with_name(destination.name + ".interrupted." + suffix))
    command = [sys.executable, "-m", module, *map(str, options)]
    print(f"Starting {destination.name}", flush=True)
    with log.open("a") as f:
        f.write("\n" + repr(command) + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as process:
            for line in process.stdout:
                print(line, end="", flush=True); f.write(line); f.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Stage {destination.name} failed. See {log}; completed stages can be resumed.")
    marker.write_text(json.dumps({"completed_utc":datetime.now(timezone.utc).isoformat(),"module":module},indent=2))


def signature(path):
    if path is None:
        return None
    stat = path.stat()
    return dict(path=str(path.resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def export_panel(cohort, matrix, model_path, out):
    p, X, receipt = load_inputs(cohort, matrix)
    model = joblib.load(model_path)["model"]
    selected = p.split.isin(["train_gold", "validation", "test", "train_silver_candidate"]).to_numpy()
    beta = model.named_steps["impute"].transform(model.named_steps["cpg"].transform(X[selected]))
    cpgs = (matrix / "cpg_ids.txt").read_text().splitlines()
    names = [cpgs[i] for i in model.named_steps["cpg"].indices_]
    frame = pd.DataFrame(np.round(beta,6), columns=names)
    for position, field in enumerate(["patient_id", "sample_id", "split"]):
        frame.insert(position,field,p.loc[selected,field].to_numpy())
    frame.to_csv(out / "paired_selected_cpg_beta_imputed.csv",index=False)
    (out / "panel_provenance.json").write_text(json.dumps(dict(
        teacher_sha256=digest(model_path),cohort_sha256=digest(cohort/"patients.csv"),
        n_features=len(names),n_patients=len(frame),
        values="Beta, median-filled using teacher training patients only; no StandardScaler transform",
        note="Includes held-out FEATURES for future frozen-model inference; split column must be respected. This is not a training CSV."),indent=2))


def main():
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder with original COMET workbooks, RMSs_meta.txt and RMSs_beta.txt")
    ap.add_argument("--beta", type=Path, help="Override original beta TSV path")
    ap.add_argument("--detection-file", type=Path)
    ap.add_argument("--cohort", type=Path, default=root/"inputs/cohort")
    ap.add_argument("--out", type=Path, default=root/"outputs/teacher_v1")
    ap.add_argument("--selector", choices=["both","limma","variance"], default="both")
    ap.add_argument("--top-k", type=int, default=1000)
    ap.add_argument("--min-confidence", type=float, default=.90)
    ap.add_argument("--rscript", default="Rscript")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--include-test", action="store_true", help="Explicitly evaluate the gold test set; default leaves it unscored")
    ap.add_argument("--allow-missing-detection-p", action="store_true")
    args = ap.parse_args()
    if args.top_k < 1 or not .5 <= args.min_confidence <= 1:
        ap.error("top-k must be positive and min-confidence must be in [0.5,1]")
    args.cohort, args.out, args.data_dir = [p.resolve() for p in [args.cohort,args.out,args.data_dir]]
    beta = (args.beta or args.data_dir/"RMSs_beta.txt").resolve()
    report = validate_sources(args.cohort,args.data_dir,beta,args.detection_file,args.allow_missing_detection_p)
    print(json.dumps(report,indent=2),flush=True)
    if args.preflight_only:
        return
    methods = ["variance","limma"] if args.selector == "both" else [args.selector]
    if "limma" in methods:
        subprocess.run([args.rscript,"--vanilla","-e",'stopifnot(requireNamespace("limma",quietly=TRUE),requireNamespace("matrixStats",quietly=TRUE))'],check=True)
    versions={name:importlib.metadata.version(name) for name in ["numpy","scipy","pandas","scikit-learn","joblib","openpyxl"]}
    versions["python"]=sys.version
    if "limma" in methods:
        versions["R_limma"]=subprocess.run([args.rscript,"--vanilla","-e",'cat(R.version.string,"; limma",as.character(packageVersion("limma")))'],text=True,capture_output=True,check=True).stdout
    config = dict(cohort_sha256=report["cohort_sha256"], beta=signature(beta), detection=signature(args.detection_file),
                  selector=args.selector,top_k=args.top_k,min_confidence=args.min_confidence,seed=args.seed,
                  allow_missing_detection_p=args.allow_missing_detection_p,metadata_sources=report["sources"],
                  environment=versions,
                  code_sha256={str(p.relative_to(root)):digest(p) for p in sorted((root/"src").rglob("*")) if p.suffix in {".py",".R"}})
    if args.out.exists():
        if not args.resume:
            ap.error("Output exists. Use --resume to continue the same configuration, or a new --out.")
        if json.loads((args.out/"run_config.json").read_text()) != config:
            raise ValueError("Configuration/source/code changed. Start a new output directory; do not mix results.")
    else:
        args.out.mkdir(parents=True)
        (args.out/"run_config.json").write_text(json.dumps(config,indent=2))
        (args.out/"preflight.json").write_text(json.dumps(report,indent=2))
    env = os.environ.copy(); env["PYTHONPATH"] = str(root/"src") + os.pathsep + env.get("PYTHONPATH","")
    env.setdefault("OMP_NUM_THREADS","1"); env.setdefault("OPENBLAS_NUM_THREADS","1")
    log = args.out/"pipeline.log"; matrix = args.out/"matrix"
    options = ["--cohort",args.cohort,"--beta",beta,"--out",matrix]
    if args.detection_file: options += ["--detection-file",args.detection_file]
    if args.allow_missing_detection_p: options += ["--allow-missing-detection-p"]
    run_stage("ridge_teacher.data.cache",options,matrix,log,env)
    common = ["--cohort",args.cohort,"--matrix",matrix]
    for method in methods:
        dest=args.out/method/"teacher"
        dest.parent.mkdir(exist_ok=True)
        run_stage("ridge_teacher.cli",["fit",*common,"--out",dest,"--selector",method,"--top-k",args.top_k,"--seed",args.seed,"--rscript",args.rscript],dest,log,env)
    reports={method:json.loads((args.out/method/"teacher/fit_report.json").read_text()) for method in methods}
    # Choose BEFORE external validation scoring. Explicit deterministic tie-break.
    selected=max(methods,key=lambda m:(reports[m]["oof"]["balanced_accuracy"],reports[m]["oof"]["roc_auc"],m=="limma"))
    decision=dict(selected_selector=selected,rule="teacher nested OOF BA; then OOF ROC AUC; exact tie -> LIMMA",
                  holdouts_used_for_selection=False,note="Exploratory selector comparison; selected OOF score is not an unbiased estimate after method selection.")
    (args.out/"teacher_selection.json").write_text(json.dumps(decision,indent=2))
    for method in methods:
        for split in ["validation"] + (["test"] if args.include_test else []):
            dest=args.out/method/split
            run_stage("ridge_teacher.cli",["evaluate",*common,"--split",split,"--model",args.out/method/"teacher/teacher.joblib","--out",dest],dest,log,env)
    model=args.out/selected/"teacher/teacher.joblib"
    silver=args.out/"silver"
    run_stage("ridge_teacher.cli",["silver",*common,"--model",model,"--out",silver,"--min-confidence",args.min_confidence],silver,log,env)
    export_panel(args.cohort,matrix,model,silver)
    export_manifest(args.cohort,silver/"silver_labels.csv",silver/"image_input_manifest.csv")
    rows=[]
    for method in methods:
        rows.append(dict(selector=method,split="teacher_nested_oof",**reports[method]["oof"]))
        for split in ["validation","test"]:
            path=args.out/method/split/"metrics.json"
            if path.exists(): rows.append(dict(selector=method,split=split,**json.loads(path.read_text())))
    pd.DataFrame(rows).to_csv(args.out/"results_summary.csv",index=False)
    # Package compact return artifacts; exclude the full raw matrix and CV cache.
    with zipfile.ZipFile(args.out/"return_results.zip","w",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.out.rglob("*")):
            rel=p.relative_to(args.out)
            if not p.is_file() or p.suffix==".zip" or "cv_cache" in rel.parts or any(".interrupted." in v for v in rel.parts): continue
            if rel.parts[0]=="matrix" and p.name not in {"sample_qc.csv","matrix_receipt.json"}: continue
            z.write(p,str(rel))
    print(f"Completed. Selected teacher: {selected}. Results: {args.out/'results_summary.csv'}",flush=True)
    print(f"Return this PRIVATE results bundle to the project team: {args.out/'return_results.zip'}",flush=True)
    if not args.include_test: print("Gold test set was NOT scored. Use --resume --include-test only after decisions are locked.")


if __name__ == "__main__":
    main()
