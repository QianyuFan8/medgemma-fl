"""Text-only methylation inputs; no patient ID or diagnosis in user prompts."""
from __future__ import annotations

import hashlib
import json
import math


def panel_fingerprint(probes):
    return hashlib.sha256(json.dumps(list(probes), separators=(",", ":")).encode()).hexdigest()


def make_record(patient_id, diagnosis, probes, values, classes):
    if len(probes) != len(values) or len(set(probes)) != len(probes):
        raise ValueError("Inconsistent CpG panel")
    if diagnosis not in classes or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Invalid diagnosis or beta measurement")
    prompt = (
        "Classify this tumor using DNA methylation beta values. "
        "Return exactly one diagnosis code from: " + ", ".join(classes) + ".\n"
        "CpG beta values:\n" + "\n".join(f"{p}: {v:.6f}" for p, v in zip(probes, values))
    )
    return {"patient_id": patient_id, "label_name": diagnosis, "prompt": prompt,
            "panel_id": panel_fingerprint(probes), "classes": list(classes)}


def format_methylation_example(example):
    return {**example, "messages": [
        {"role": "user", "content": [{"type": "text", "text": example["prompt"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": example["label_name"]}]},
    ]}


def encode_methylation_example(processor, example, max_length=4096):
    messages = example["messages"]
    prefix = processor.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False)
    full = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    pre_ids = processor.tokenizer(prefix, add_special_tokens=False)["input_ids"]
    ids = processor.tokenizer(full, add_special_tokens=False)["input_ids"]
    if ids[:len(pre_ids)] != pre_ids or len(ids) <= len(pre_ids):
        raise ValueError("Chat template prefix mismatch: cannot safely mask assistant-only loss")
    if len(ids) > max_length:
        raise ValueError(f"Methylation example has {len(ids)} tokens > {max_length}; reduce panel or increase limit")
    return ids, [-100] * len(pre_ids) + ids[len(pre_ids):]


def build_methylation_collator(processor, max_length=4096):
    """Assistant-only loss, explicit overflow failure, no silent CpG truncation."""
    def collate(examples):
        import torch

        sequences, labels = [], []
        for example in examples:
            ids, supervised = encode_methylation_example(processor, example, max_length)
            sequences.append(ids)
            labels.append(supervised)
        width = max(map(len, sequences))
        pad = processor.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer must define pad_token_id")
        return {
            "input_ids": torch.tensor([s + [pad]*(width-len(s)) for s in sequences]),
            "attention_mask": torch.tensor([[1]*len(s) + [0]*(width-len(s)) for s in sequences]),
            "labels": torch.tensor([s + [-100]*(width-len(s)) for s in labels]),
        }
    return collate


def classification_metrics(truth, prediction, classes):
    if not truth or len(truth) != len(prediction) or not set(truth) <= set(classes):
        raise ValueError("Invalid evaluation labels")
    recalls, f1s = {}, {}
    for label in classes:
        tp = sum(t == label and p == label for t, p in zip(truth, prediction))
        fn = sum(t == label and p != label for t, p in zip(truth, prediction))
        fp = sum(t != label and p == label for t, p in zip(truth, prediction))
        if tp + fn == 0:
            raise ValueError(f"Evaluation split missing class {label}")
        recalls[label] = tp / (tp + fn)
        f1s[label] = 2*tp/(2*tp+fp+fn)
    return {"n_patients": len(truth), "accuracy": sum(t == p for t, p in zip(truth, prediction))/len(truth),
            "balanced_accuracy": sum(recalls.values())/len(classes),
            "macro_F1": sum(f1s.values())/len(classes), "recall": recalls,
            "invalid_predictions": sum(p not in classes for p in prediction)}
