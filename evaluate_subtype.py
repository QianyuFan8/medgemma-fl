"""Evaluate ERMS vs ARMS: ridge baseline, MedGemma (zero-shot / adapter / FL checkpoint), compare.

MedGemma levels share this entry point:
  Baseline 1 (zero-shot)      --model-path google/medgemma-1.5-4b-it --parse lenient
  Baseline 2 (centralized)    --model-path <FL_global_model.pt from a 1-client job>
  Federated                   --model-path <FL_global_model.pt from a 3-client job>
Class probabilities come from teacher-forced log-likelihoods of each option, so
ROC-AUC / PR-AUC are available for every level; the generated text gives the hard label.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess

from methylation_validate import validate_clients
from subtype_utils import SUBTYPE_CLASSES, format_subtype_example, subtype_metrics

SYNONYMS = {"ERMS": ("erms", "embryonal"), "ARMS": ("arms", "alveolar")}


def read_predictions(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if len({r["patient_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate prediction patient IDs")
    return rows


def positive_score(row, positive, classes):
    if f"score_{positive}" in row and row[f"score_{positive}"] != "":
        return float(row[f"score_{positive}"])
    other = [c for c in classes if c != positive]
    if len(other) == 1 and f"score_{other[0]}" in row:
        return 1.0 - float(row[f"score_{other[0]}"])
    raise ValueError(f"Prediction file lacks a score column for {positive}")


def evaluate_rows(rows, reference, classes, positive):
    expected = {r["patient_id"]: r["label_name"] for r in reference}
    if len(expected) != len(reference) or len(rows) != len(expected):
        raise ValueError("Incomplete/duplicate evaluation cohort")
    if {r["patient_id"] for r in rows} != set(expected):
        raise ValueError("Predictions do not cover exactly the requested patients")
    if any(r["truth"] != expected[r["patient_id"]] for r in rows):
        raise ValueError("Prediction ground truths disagree with reference")
    return subtype_metrics([r["truth"] for r in rows], [r["prediction"] for r in rows],
                           [positive_score(r, positive, classes) for r in rows], classes, positive)


def lenient_parse(raw, classes):
    """Earliest whole-word mention of a class or its histologic synonym; else <unparsed>."""
    best = None
    for cls in classes:
        for word in SYNONYMS.get(cls, (cls.lower(),)):
            m = re.search(rf"\b{re.escape(word)}\b", raw, flags=re.IGNORECASE)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), cls)
    return best[1] if best else "<unparsed>"


def encode(processor, text, images, device):
    if images:
        batch = processor(text=[text], images=images, return_tensors="pt")
    else:
        batch = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt")
    return batch.to(device)


def predict_one(model, processor, row, data_dir, classes, max_seq_length, parse):
    import torch
    from PIL import Image
    from inference_utils import get_model_device

    device = get_model_device(model)
    messages = format_subtype_example(row)["messages"][:1]  # NEVER include the assistant truth
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images = [[Image.open(Path(data_dir) / row["image"]).convert("RGB")]] if row.get("image") else None
    inputs = encode(processor, prompt, images, device)
    n = inputs["input_ids"].shape[1]
    if n + 32 > max_seq_length:
        raise ValueError("Prompt too long; no silent truncation is allowed")
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        raw = processor.tokenizer.decode(generated[0, n:], skip_special_tokens=True).strip()
        logps = []
        for cls in classes:
            full = encode(processor, prompt + cls, images, device)
            ids = full["input_ids"]
            if not torch.equal(ids[0, :n], inputs["input_ids"][0]) or ids.shape[1] <= n:
                raise ValueError("Option continuation does not extend the prompt tokens; cannot score options")
            logits = model(**full).logits[0, n - 1:-1].float()
            logps.append(torch.log_softmax(logits, -1).gather(1, ids[0, n:].unsqueeze(1)).sum())
        probs = torch.stack(logps).softmax(0).tolist()
    prediction = raw if raw in classes else (lenient_parse(raw, classes) if parse == "lenient" else "<unparsed>")
    out = {"patient_id": row["patient_id"], "truth": row["label_name"], "prediction": prediction, "raw_response": raw}
    out.update({f"score_{c}": p for c, p in zip(classes, probs)})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("ridge", "medgemma", "compare"))
    p.add_argument("--data-dir", required=True, help="prepare_comet_subtype.py export directory")
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--r-library")
    p.add_argument("--model-path", help="MedGemma ID (zero-shot), adapter directory or NVFlare FL_global_model.pt")
    p.add_argument("--base-model", default="google/medgemma-1.5-4b-it")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--parse", choices=("strict", "lenient"), default="strict",
                   help="strict = exact option text (fine-tuned models); lenient = first option mentioned (zero-shot)")
    p.add_argument("--positive-class", default=SUBTYPE_CLASSES[-1])
    p.add_argument("--ridge-predictions")
    p.add_argument("--medgemma-predictions")
    args = p.parse_args()
    root, out = Path(args.data_dir), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    task = json.loads((root / "task.json").read_text())
    if task.get("task") != "comet_subtype":
        raise ValueError("Not a comet_subtype export; use evaluate_methylation.py for other tasks")
    classes, positive = task["classes"], args.positive_class
    validate_clients(root / "clients", task["n_clients"])
    reference = json.loads((root / f"{args.split}.json").read_text())
    if args.mode == "ridge":
        if task["modality"] == "image":
            p.error("Ridge needs CpG features; export a methylation or multimodal layout")
        env = os.environ.copy()
        env.pop("R_HOME", None)
        if args.r_library:
            env["R_LIBS_USER"] = str(Path(args.r_library).resolve())
        subprocess.run([args.rscript, str(Path(__file__).parent / "methylation/ridge.R"), str(root), str(out), args.split],
                       env=env, check=True)
        rows = read_predictions(out / "predictions.csv")
        # Ridge ignores images: it scores the same patients from the methylation columns only.
        results = evaluate_rows(rows, reference, classes, positive)
    elif args.mode == "medgemma":
        if not args.model_path:
            p.error("--model-path is required for medgemma")
        from inference_utils import load_model_and_processor

        model, processor = load_model_and_processor(args.model_path, args.base_model, args.device)
        if not args.device.startswith("cuda"):
            model.to(args.device)
        predictions = [predict_one(model, processor, row, root, classes, args.max_seq_length, args.parse) for row in reference]
        results = evaluate_rows(predictions, reference, classes, positive)
        site_by_patient = {r["patient_id"]: r["site_id"] for r in reference}
        results["by_site"] = {}
        for site in task["sites"]:
            subset = [r for r in predictions if site_by_patient[r["patient_id"]] == site]
            if len({r["truth"] for r in subset}) == 2:
                results["by_site"][site] = evaluate_rows(subset, [r for r in reference if r["site_id"] == site], classes, positive)
            else:
                results["by_site"][site] = {"n_patients": len(subset), "note": "one class only at this site; no AUC"}
        out.mkdir(parents=True)
        with (out / "predictions.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
    else:
        if not args.ridge_predictions or not args.medgemma_predictions:
            p.error("Both prediction files are required for compare")
        results = {"ridge": evaluate_rows(read_predictions(args.ridge_predictions), reference, classes, positive),
                   "medgemma": evaluate_rows(read_predictions(args.medgemma_predictions), reference, classes, positive)}
        for path, expected_mode in ((args.ridge_predictions, "ridge"), (args.medgemma_predictions, "medgemma")):
            provenance = json.loads((Path(path).parent / "evaluation.json").read_text())
            if (provenance["panel_id"] != task["panel_id"] or provenance["split"] != args.split
                    or provenance["mode"] != expected_mode):
                raise ValueError("Model/panel/split mismatch across evaluation runs")
        out.mkdir(parents=True)
    report = {"mode": args.mode, "split": args.split, "panel_id": task["panel_id"], "modality": task["modality"],
              "training_regime": task.get("training_regime"), "partition": task.get("partition"),
              "arguments": vars(args), "metrics": results}
    (out / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
