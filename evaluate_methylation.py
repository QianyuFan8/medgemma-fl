"""Run ridge or MedGemma and compare predictions on identical held-out patients."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from methylation_utils import classification_metrics, format_methylation_example


def read_predictions(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction patient IDs")
    return rows


def evaluate_rows(rows, reference, classes):
    expected = {r["patient_id"]: r["label_name"] for r in reference}
    if len(expected) != len(reference) or len(rows) != len(expected):
        raise ValueError("Incomplete/duplicate evaluation cohort")
    if {r["patient_id"] for r in rows} != set(expected):
        raise ValueError("Predictions do not cover exactly the requested patients")
    if any(r["truth"] != expected[r["patient_id"]] for r in rows):
        raise ValueError("Prediction ground truths disagree with reference")
    return classification_metrics([r["truth"] for r in rows], [r["prediction"] for r in rows], classes)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("ridge", "medgemma", "compare"))
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--r-library")
    p.add_argument("--model-path", help="MedGemma ID, adapter directory or NVFlare FL_global_model.pt")
    p.add_argument("--base-model", default="google/medgemma-4b-it")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--ridge-predictions")
    p.add_argument("--medgemma-predictions")
    args = p.parse_args()
    root, out = Path(args.data_dir), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    task = json.loads((root / "task.json").read_text())
    reference = json.loads((root / f"{args.split}.json").read_text())
    if args.mode == "ridge":
        env = os.environ.copy(); env.pop("R_HOME", None)
        if args.r_library:
            env["R_LIBS_USER"] = str(Path(args.r_library).resolve())
        subprocess.run([args.rscript, str(Path(__file__).parent / "methylation/ridge.R"),
                        str(root), str(out), args.split], env=env, check=True)
        results = evaluate_rows(read_predictions(out / "predictions.csv"), reference, task["classes"])
    elif args.mode == "medgemma":
        if not args.model_path:
            p.error("--model-path is required for medgemma")
        import torch
        from inference_utils import load_model_and_processor, get_model_device

        model, processor = load_model_and_processor(args.model_path, args.base_model, args.device)
        if not args.device.startswith("cuda"):
            model.to(args.device)
        predictions = []
        for row in reference:
            messages = format_methylation_example(row)["messages"][:1]  # NEVER include assistant truth
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            inputs = processor.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            if inputs["input_ids"].shape[1] + 32 > args.max_seq_length:
                raise ValueError("Prompt too long; no silent truncation is allowed")
            inputs = inputs.to(get_model_device(model))
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            raw = processor.tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            prediction = raw if raw in task["classes"] else "<unparsed>"
            predictions.append({"patient_id": row["patient_id"], "truth": row["label_name"],
                                "prediction": prediction, "raw_response": raw})
        results = evaluate_rows(predictions, reference, task["classes"])
        out.mkdir(parents=True)
        with (out / "predictions.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader(); writer.writerows(predictions)
    else:
        if not args.ridge_predictions or not args.medgemma_predictions:
            p.error("Both prediction files are required for compare")
        results = {"ridge": evaluate_rows(read_predictions(args.ridge_predictions), reference, task["classes"]),
                   "medgemma": evaluate_rows(read_predictions(args.medgemma_predictions), reference, task["classes"])}
        # Comparison requires provenance from these evaluation commands, not arbitrary CSVs.
        for path, expected_mode in ((args.ridge_predictions, "ridge"), (args.medgemma_predictions, "medgemma")):
            provenance = json.loads((Path(path).parent / "evaluation.json").read_text())
            if (provenance["panel_id"] != task["panel_id"] or provenance["split"] != args.split
                    or provenance["mode"] != expected_mode):
                raise ValueError("Model/panel/split mismatch across evaluation runs")
        out.mkdir(parents=True)
    report = {"mode": args.mode, "split": args.split, "panel_id": task["panel_id"],
              "arguments": vars(args), "metrics": results}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
