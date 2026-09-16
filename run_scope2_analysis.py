"""Compare a frozen Scope 2 SAE on fixed base/federated MedGemma inputs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import random
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

from data_utils import DEFAULT_MODEL_NAME_OR_PATH, PROMPT, RAW_TISSUE_CODES, collect_image_records, sample_records
from scope2_class_analysis import export_class_report, require_plotting
from scope2_utils import (
    SAE_SOURCE_MODEL, PromptResidualCapture, feature_comparison, file_sha256,
    input_fingerprint, load_scope2_sae, reconstruction_metrics, resolve_decoder_layer,
)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_manifest(dataset_dir: Path, max_samples: int, seed: int, manifest_path=None, samples_per_class=None):
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    if manifest_path and samples_per_class is not None:
        raise ValueError("Choose either an existing manifest or new balanced sampling")
    if manifest_path:
        manifest = json.loads(Path(manifest_path).read_text())
        if manifest.get("schema_version") != 1 or manifest.get("prompt_sha256") != prompt_hash:
            raise ValueError("Manifest schema or classification prompt does not match this code")
        records = manifest["samples"]
    else:
        all_records = collect_image_records(str(dataset_dir))
        if samples_per_class is not None:
            if samples_per_class < 2:
                raise ValueError("samples_per_class must be at least 2 for class comparisons")
            pools = [[r for r in all_records if r["label"] == c] for c in range(9)]
            shortages = {RAW_TISSUE_CODES[c]: len(pool) for c, pool in enumerate(pools) if len(pool) < samples_per_class}
            if shortages:
                raise ValueError(f"Not enough images for {samples_per_class} per class: {shortages}")
            rng = random.Random(seed)
            records = [r for pool in pools for r in rng.sample(pool, samples_per_class)]
            rng.shuffle(records)
        else:
            records = sample_records(all_records, max_samples, seed)
        records = [{"sample_id": f"sample-{i:04d}", "image": r["image"], "label": r["label"],
                    "label_name": r["label_name"], "image_sha256": file_sha256(dataset_dir / r["image"])}
                   for i, r in enumerate(records)]
        manifest = {"schema_version": 1, "seed": seed, "prompt": PROMPT,
                    "sampling": {"method": "balanced_by_true_label" if samples_per_class is not None else "random",
                                 "samples_per_class": samples_per_class,
                                 "class_counts": {RAW_TISSUE_CODES[c]: sum(r["label"] == c for r in records) for c in range(9)}},
                    "prompt_sha256": prompt_hash, "samples": records}
    if not records or len({r["sample_id"] for r in records}) != len(records):
        raise ValueError("Manifest must contain nonempty, unique sample IDs")
    if len({str((dataset_dir / r["image"]).resolve()) for r in records}) != len(records):
        raise ValueError("Manifest contains duplicate image paths")
    for record in records:
        if not isinstance(record["label"], int) or not 0 <= record["label"] < 9:
            raise ValueError("Manifest tissue label must be an integer in [0, 8]")
        if file_sha256(dataset_dir / record["image"]) != record["image_sha256"]:
            raise ValueError(f"Image changed since manifest creation: {record['image']}")
    return manifest


def environment_metadata():
    packages = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "huggingface-hub", "safetensors", "numpy"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    repo = Path(__file__).resolve().parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {"packages": packages, "git_commit": commit, "git_dirty": dirty,
            "cuda_runtime": torch.version.cuda}


def analyze_model(tag, model_path, args, manifest, sae, output_dir, paired_inputs=None):
    # Lazy imports keep --help and CPU diagnostics independent of the training stack.
    from PIL import Image
    from data_utils import parse_prediction_label
    from inference_utils import load_model_and_processor, prepare_inference_inputs

    torch.manual_seed(args.seed)
    model = processor = module = capture = inputs = generated = None
    try:
        model, processor = load_model_and_processor(
            model_path=model_path, base_model=SAE_SOURCE_MODEL if tag == "gemma_control" else args.base_model,
            device=args.device,
        )
        if args.device == "cpu":
            model.to("cpu")
        hook_name, module = resolve_decoder_layer(model, args.sae_layer, sae.d_in)
        print(f"{tag}: observing {hook_name} (raw decoder output)", flush=True)
        rows, vectors, features, reconstructions = [], [], [], []
        for i, record in enumerate(manifest["samples"]):
            with Image.open(Path(args.dataset_dir) / record["image"]) as image:
                inputs = prepare_inference_inputs(model, processor, [image.convert("RGB")])
            fingerprint = input_fingerprint(inputs)
            if paired_inputs is not None and fingerprint != paired_inputs[i]:
                raise ValueError(f"Base/tuned processed inputs differ for {record['sample_id']}; comparison aborted")
            positions = torch.nonzero(inputs["attention_mask"][0], as_tuple=False).flatten()
            if not len(positions):
                raise ValueError("Empty prompt")
            position = int(positions[-1])
            length = inputs["input_ids"].shape[1]
            with PromptResidualCapture(module, length, position, sae.d_in) as capture:
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            response = processor.batch_decode(generated[:, length:], skip_special_tokens=True)[0].strip()
            predicted = parse_prediction_label(response)
            z, reconstructed = sae(capture.activation)
            vectors.append(capture.activation.numpy())
            features.append(z.cpu().numpy())
            reconstructions.append(reconstructed.cpu().numpy())
            rows.append({"sample_id": record["sample_id"], "label": record["label"],
                         "predicted_label": predicted, "correct": predicted == record["label"],
                         "response": response, "input_sha256": fingerprint,
                         "prompt_length": length, "token_position": position,
                         "token_id": int(inputs["input_ids"][0, position]),
                         "token_text": processor.tokenizer.decode([int(inputs["input_ids"][0, position])])})
            print(f"{tag}: {i + 1}/{len(manifest['samples'])}, L0={int((z > 0).sum())}", flush=True)
        features = np.concatenate(features)
        diagnostics, metrics = reconstruction_metrics(np.concatenate(vectors), np.concatenate(reconstructions), features)
        for row, diagnostic in zip(rows, diagnostics):
            row.update(diagnostic)
        metrics.update({"accuracy": sum(row["correct"] for row in rows) / len(rows),
                        "unparsed": sum(row["predicted_label"] < 0 for row in rows), "samples": len(rows)})
        model_info = {"path": model_path, "config_commit": getattr(model.config, "_commit_hash", None),
                      "hook_module": hook_name, "hidden_size": sae.d_in,
                      "quantization": "4bit NF4, bfloat16 compute" if args.device.startswith("cuda") else "bfloat16",
                      "processor_class": type(processor).__name__,
                      "gpu": torch.cuda.get_device_name(torch.device(args.device))
                      if args.device.startswith("cuda") else None}
        write_json(output_dir / f"{tag}.json", {"model": model_info, "metrics": metrics, "samples": rows})
        np.savez_compressed(output_dir / f"{tag}_features.npz", features=features,
                            sample_ids=np.array([row["sample_id"] for row in rows]))
        return {"model": model_info, "metrics": metrics, "samples": rows, "features": features}
    finally:
        # Keep only one language model in GPU memory at a time.
        model = processor = module = capture = inputs = generated = None
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()


def export_comparison(base, tuned, output_dir, top_k, feature_top_k=10):
    base_ids = [row["sample_id"] for row in base["samples"]]
    tuned_ids = [row["sample_id"] for row in tuned["samples"]]
    if base_ids != tuned_ids:
        raise ValueError("Base/tuned sample IDs do not match in order")
    stats = feature_comparison(base["features"], tuned["features"])
    ranking = np.argsort(-stats["mean_abs_delta"], kind="stable")
    write_csv(output_dir / "feature_summary.csv", [
        {"feature_id": int(j), **{key: float(value[j]) for key, value in stats.items()}} for j in ranking
    ])
    changed_ranking = ranking[stats["mean_abs_delta"][ranking] > 0][:feature_top_k]
    top_changes = [{"rank": rank, "feature_id": int(j), **{key: float(value[j]) for key, value in stats.items()}}
                   for rank, j in enumerate(changed_ranking, start=1)]
    with open(output_dir / "top_feature_changes.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["rank", "feature_id"] + list(stats))
        writer.writeheader()
        writer.writerows(top_changes)
    samples, changes = [], []
    groups = {"wrong_to_right": 0, "right_to_wrong": 0, "both_right": 0, "both_wrong": 0}
    for i, (before, after) in enumerate(zip(base["samples"], tuned["samples"])):
        if before["input_sha256"] != after["input_sha256"] or before["label"] != after["label"]:
            raise ValueError("Base/tuned input or reference mismatch")
        group = ("both_right" if after["correct"] else "right_to_wrong") if before["correct"] else (
            "wrong_to_right" if after["correct"] else "both_wrong")
        groups[group] += 1
        samples.append({"sample_id": before["sample_id"], "label": before["label"], "group": group,
                        "base_prediction": before["predicted_label"], "tuned_prediction": after["predicted_label"],
                        "base_response": before["response"], "tuned_response": after["response"],
                        "base_l0": before["l0"], "tuned_l0": after["l0"],
                        "base_relative_squared_error": before["relative_squared_error"],
                        "tuned_relative_squared_error": after["relative_squared_error"]})
        delta = tuned["features"][i] - base["features"][i]
        changed = np.flatnonzero(delta)
        selected = changed[np.argsort(-np.abs(delta[changed]), kind="stable")[:top_k]]
        for j in selected:
            changes.append({"sample_id": before["sample_id"], "group": group, "feature_id": int(j),
                            "base_activation": float(base["features"][i, j]),
                            "tuned_activation": float(tuned["features"][i, j]), "delta": float(delta[j])})
    write_csv(output_dir / "sample_comparison.csv", samples)
    # Always produce an explicit file, even when every feature is unchanged.
    if changes:
        write_csv(output_dir / "sample_feature_deltas.csv", changes)
    else:
        (output_dir / "sample_feature_deltas.csv").write_text(
            "sample_id,group,feature_id,base_activation,tuned_activation,delta\n")
    return {"prediction_groups": groups,
            "accuracy_delta": tuned["metrics"]["accuracy"] - base["metrics"]["accuracy"],
            "changed_feature_count": int(np.count_nonzero(stats["mean_abs_delta"])),
            "mean_absolute_feature_delta": float(stats["mean_abs_delta"].mean()),
            "top_changed_feature_ids": [int(j) for j in changed_ranking]}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuned_model_path", required=True, help="Existing NVFlare .pt or PEFT adapter directory")
    parser.add_argument("--dataset_dir", default="./CRC-VAL-HE-7K")
    parser.add_argument("--base_model", default=DEFAULT_MODEL_NAME_OR_PATH)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max_samples", type=int, default=20, help="Total random samples for the legacy smoke test")
    selection.add_argument("--samples_per_class", type=int, help="Sample this many images independently from EACH of 9 true classes")
    parser.add_argument("--seed", type=int, default=42)
    selection.add_argument("--manifest", help="Reuse samples.json from a prior run; overrides sample count/selection seed")
    parser.add_argument("--sae_layer", type=int, choices=[9, 17, 22, 29], default=17)
    parser.add_argument("--sae_revision", default="main", help="HF revision; resolved commit is recorded in run.json")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--top_k", type=int, default=20, help="Changed features per sample in sample_feature_deltas.csv")
    parser.add_argument("--feature_top_k", type=int, default=10, help="Global changed features and positive associations per class")
    parser.add_argument("--min_class_active", type=int, default=3, help="Minimum active samples in a class for its candidate ranking")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu (slow)")
    parser.add_argument("--output_dir", default="./runs/scope2-smoke", help="Must not already exist")
    parser.add_argument("--with_gemma_control", action="store_true", help="Also run SAE-source Gemma 3 4B IT; requires HF access")
    args = parser.parse_args()
    if min(args.max_samples, args.max_new_tokens, args.top_k) < 1:
        parser.error("max_samples, max_new_tokens and top_k must be positive")
    if min(args.feature_top_k, args.min_class_active) < 1:
        parser.error("feature_top_k and min_class_active must be positive")
    if args.samples_per_class is not None and args.samples_per_class < 2:
        parser.error("samples_per_class must be at least 2")
    if args.device != "cpu" and args.device != "cuda" and not (
            args.device.startswith("cuda:") and args.device[5:].isdigit()):
        parser.error("device must be cpu, cuda, or cuda:N")
    tuned = Path(args.tuned_model_path).expanduser().resolve()
    if not (tuned.is_file() and tuned.suffix == ".pt") and not (tuned / "adapter_config.json").is_file():
        parser.error("tuned_model_path must be an existing NVFlare .pt or local PEFT adapter directory")
    args.tuned_model_path = str(tuned)
    args.dataset_dir = str(Path(args.dataset_dir).expanduser().resolve())
    return args


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on the tested GPU VM")
    manifest = make_manifest(Path(args.dataset_dir), args.max_samples, args.seed, args.manifest, args.samples_per_class)
    counts = [sum(r["label"] == c for r in manifest["samples"]) for c in range(9)]
    if min(counts) >= 2:
        require_plotting()  # Fail before downloading/loading models if a dependency is missing.
    print(f"Selected {len(manifest['samples'])} images; true class counts: {dict(zip(RAW_TISSUE_CODES, counts))}", flush=True)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "samples.json", manifest)
    checkpoint = Path(args.tuned_model_path)
    checkpoint_files = [checkpoint] if checkpoint.is_file() else sorted(
        path for path in checkpoint.rglob("*") if path.is_file())
    run = {"status": "running", "arguments": vars(args), "environment": environment_metadata(),
           "checkpoint_sha256": {str(p.relative_to(checkpoint) if checkpoint.is_dir() else p.name): file_sha256(p)
                                 for p in checkpoint_files},
           "activation_position": "last nonpadding prompt token, first generation forward, raw resid_post",
           "transfer_status": "unvalidated: Gemma 3 SAE applied to MedGemma",
           "limitations": ["Feature IDs have no verified medical meaning or causal interpretation.",
                           "Small fixed sample diagnostics are not clinical or convergence validation.",
                           "4-bit quantization and medical fine-tuning may affect SAE reconstruction."]}
    write_json(output_dir / "run.json", run)
    try:
        sae, run["sae"] = load_scope2_sae(args.sae_layer, args.sae_revision)
        write_json(output_dir / "run.json", run)
        base = analyze_model("base", args.base_model, args, manifest, sae, output_dir)
        tuned = analyze_model("tuned", args.tuned_model_path, args, manifest, sae, output_dir,
                              paired_inputs=[row["input_sha256"] for row in base["samples"]])
        summary = export_comparison(base, tuned, output_dir, args.top_k, args.feature_top_k)
        summary["class_analysis"] = export_class_report(base, tuned, output_dir, args.feature_top_k, args.min_class_active)
        summary.update({"base": base["metrics"], "tuned": tuned["metrics"],
                        "transfer_status": run["transfer_status"], "gemma_control": None})
        if args.with_gemma_control:
            control = analyze_model("gemma_control", SAE_SOURCE_MODEL, args, manifest, sae, output_dir)
            summary["gemma_control"] = control["metrics"]
            summary["control_note"] = "Same images and question, source-model processor; inspect reconstruction, not accuracy alone."
        write_json(output_dir / "summary.json", summary)
        run["status"] = "complete"
        print(json.dumps(summary, indent=2), flush=True)
        print(f"Results: {output_dir}", flush=True)
    except Exception as exc:
        run.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        write_json(output_dir / "run.json", run)


if __name__ == "__main__":
    main()
