"""Measure actual chat-template token lengths without loading model weights.

No feature truncation or automatic panel reduction is permitted.
This checks context capacity, not GPU memory feasibility.
"""
import argparse
import json
from pathlib import Path

from methylation_utils import encode_methylation_example, format_methylation_example
from methylation_validate import validate_clients


def measure_context(root, processor):
    root = Path(root)
    task = json.loads((root / "task.json").read_text())
    validate_clients(root / "clients", task["n_clients"])
    maxima, counts = {}, {}
    for split in ("train", "validation", "test"):
        paths = ([root / "clients" / f"site-{i}" / "train.json" for i in range(1, task["n_clients"]+1)]
                 if split == "train" else [root / f"{split}.json"])
        maximum, count = 0, 0
        for path in paths:
            for row in json.loads(path.read_text()):
                example = format_methylation_example(row)
                user = processor.apply_chat_template(example["messages"][:1], add_generation_prompt=True, tokenize=False)
                needed = len(processor.tokenizer(user, add_special_tokens=False)["input_ids"]) + 32
                if split == "train":
                    ids, _ = encode_methylation_example(processor, example, max_length=10**9)
                    needed = max(needed, len(ids))
                maximum = max(maximum, needed)
                count += 1
        maxima[split], counts[split] = maximum, count
    return {"panel_id": task["panel_id"], "n_features": len(task["probes"]), "sample_counts": counts,
            "max_tokens_by_split_including_32_token_generation_reserve": maxima,
            "required_max_seq_length": max(maxima.values())}


def check_context(root, model_name, max_length):
    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, use_fast=False)
    cfg = AutoConfig.from_pretrained(model_name)
    text_cfg = getattr(cfg, "text_config", cfg)
    capacity = getattr(text_cfg, "max_position_embeddings", None)
    result = measure_context(root, processor)
    result.update(model=model_name, requested_max_seq_length=max_length, model_context_capacity=capacity)
    print(json.dumps(result, indent=2))
    if capacity is None:
        raise ValueError("Cannot verify model context capacity; inspect model config before training")
    if max_length > capacity or result["required_max_seq_length"] > max_length:
        raise ValueError("Context budget does not fit. Set a supported --max_seq_length / --max-seq-length after review; do not silently truncate the CpG panel.")
    print("Context check passed. GPU memory must still be tested with a small smoke run.")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", default="google/medgemma-4b-it")
    p.add_argument("--max-seq-length", type=int, default=4096)
    a = p.parse_args()
    check_context(a.data_dir, a.model, a.max_seq_length)
