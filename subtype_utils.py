"""ERMS-vs-ARMS records, prompts, collator and metrics for the COMET subtype task.

Records may carry a methylation CpG panel, an H&E montage image, or both. The
true label appears only in the assistant target; prompts never include patient
identifiers, slide numbers or the diagnosis.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from methylation_utils import panel_fingerprint

SUBTYPE_CLASSES = ["ERMS", "ARMS"]
MODALITIES = ("methylation", "image", "multimodal")
PROMPT_STYLE = "comet_subtype_v1"
# MedGemma 4B encodes one image as 256 soft tokens plus begin/end markers.
IMAGE_TOKEN_RESERVE = 280
MAPPING_COLUMNS = {
    "sample_id": ("Sample ID",),
    "subtype": ("Subtype",),
    "compbio_id": ("Comp Bio ID",),
    "sentrix_id": ("X850k Tumor File Name",),
    "slide_id": ("Slide Scan File",),
}


def patient_from_compbio(compbio_id: str) -> str:
    """St. Jude sample IDs are <patient>_<sample>; the prefix identifies the patient."""
    value = (compbio_id or "").strip()
    return value.split("_", 1)[0] if value else ""


def normalize_slide_id(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        value = int(value)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return "" if text.upper() in {"", "NA", "NAN", "NONE"} else text


def read_mapping(path) -> list[dict]:
    """Read the advisor's COMET_RMS table (.xlsx, .csv or .tsv) into normalized rows."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        try:
            import openpyxl
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("Reading .xlsx needs openpyxl; or export the sheet to CSV/TSV first") from exc
        sheet = openpyxl.load_workbook(path, read_only=True, data_only=True).worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = [str(h).strip() if h is not None else "" for h in next(rows)]
        raw = [dict(zip(header, r)) for r in rows if any(v is not None for v in r)]
    else:
        delimiter = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
        with path.open(newline="", encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f, delimiter=delimiter))
    if not raw:
        raise ValueError(f"Empty mapping table: {path}")
    columns = {}
    for key, names in MAPPING_COLUMNS.items():
        found = [n for n in names if n in raw[0]]
        if not found:
            raise ValueError(f"Mapping table lacks column {names[0]!r}")
        columns[key] = found[0]
    out = []
    for r in raw:
        row = {k: ("" if r[c] is None else str(r[c]).strip()) for k, c in columns.items()}
        row["slide_id"] = normalize_slide_id(r[columns["slide_id"]])
        row["patient_id"] = patient_from_compbio(row["compbio_id"])
        out.append(row)
    return out


def make_subtype_record(patient_id, diagnosis, probes, values, classes, modality, image=None, site_id=None):
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality: {modality}")
    if diagnosis not in classes:
        raise ValueError(f"Diagnosis {diagnosis!r} not in {classes}")
    use_meth = modality in ("methylation", "multimodal")
    use_image = modality in ("image", "multimodal")
    if use_meth:
        if len(probes) != len(values) or len(set(probes)) != len(probes) or not probes:
            raise ValueError("Inconsistent CpG panel")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid beta measurement")
    else:
        probes, values = [], []
    if use_image and not image:
        raise ValueError(f"Modality {modality} requires an image for {patient_id}")
    if not use_image:
        image = None
    parts = ["Classify the subtype of this pediatric rhabdomyosarcoma."]
    if use_image:
        parts.append("[H&E]\nThe attached image is a 4x4 montage of 16 high-attention tiles "
                     "spaced across one H&E whole-slide image (224 px at 0.5 um/px).")
    if use_meth:
        parts.append("[DNA methylation]\n" + "\n".join(f"{p}: {v:.6f}" for p, v in zip(probes, values)))
    parts.append("What is the most likely subtype?\nOptions: " + ", ".join(classes) +
                 ".\nReply with only the subtype abbreviation.")
    record = {"patient_id": patient_id, "label_name": diagnosis, "prompt": "\n\n".join(parts),
              "panel_id": panel_fingerprint(probes), "classes": list(classes), "modality": modality,
              "image": image, "prompt_style": PROMPT_STYLE}
    if site_id is not None:
        record["site_id"] = site_id
    return record


def format_subtype_example(example):
    content = []
    if example.get("image"):
        content.append({"type": "image"})
    content.append({"type": "text", "text": example["prompt"]})
    return {**example, "messages": [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": example["label_name"]}]},
    ]}


def build_multimodal_collator(processor, max_length=4096, image_token_id=262144):
    """Image+text batches with assistant-only loss and explicit overflow failure."""
    def collate(examples):
        import torch

        texts, prefixes, images = [], [], []
        for example in examples:
            messages = example["messages"]
            images.append([example["image"].convert("RGB")])
            prefixes.append(processor.apply_chat_template(messages[:1], add_generation_prompt=True, tokenize=False))
            texts.append(processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False))
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        prefix_batch = processor(text=prefixes, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        pad = processor.tokenizer.pad_token_id
        if pad is None:
            raise ValueError("Tokenizer must define pad_token_id")
        for i in range(len(examples)):
            prefix_ids = prefix_batch["input_ids"][i][prefix_batch["attention_mask"][i].bool()]
            n = int(prefix_ids.numel())
            full_len = int(batch["attention_mask"][i].sum())
            if full_len <= n or not torch.equal(batch["input_ids"][i][:n], prefix_ids):
                raise ValueError("Chat template prefix mismatch: cannot safely mask assistant-only loss")
            if full_len > max_length:
                raise ValueError(f"Example has {full_len} tokens > {max_length}; reduce the panel or raise the limit")
            labels[i, :n] = -100
        labels[labels == pad] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        return batch
    return collate


def binary_auroc(truth, scores):
    """Rank-based AUROC for 0/1 truth; ties receive half credit."""
    pos = [s for t, s in zip(truth, scores) if t == 1]
    neg = [s for t, s in zip(truth, scores) if t == 0]
    if not pos or not neg:
        raise ValueError("AUROC needs both classes")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(pos) * len(neg))


def binary_auprc(truth, scores):
    """Area under the precision-recall curve (step-wise, average precision)."""
    order = sorted(range(len(truth)), key=lambda i: -scores[i])
    n_pos = sum(1 for t in truth if t == 1)
    if n_pos == 0 or n_pos == len(truth):
        raise ValueError("AUPRC needs both classes")
    tp, area = 0, 0.0
    for rank, i in enumerate(order, 1):
        if truth[i] == 1:
            tp += 1
            area += tp / rank
    return area / n_pos


def subtype_metrics(truth, prediction, scores, classes, positive=None):
    """Accuracy/balanced accuracy on hard predictions plus AUROC/AUPRC on the positive-class score."""
    from methylation_utils import classification_metrics

    positive = positive or classes[-1]
    metrics = classification_metrics(truth, prediction, classes)
    y = [1 if t == positive else 0 for t in truth]
    metrics["positive_class"] = positive
    metrics["roc_auc"] = binary_auroc(y, scores)
    metrics["pr_auc"] = binary_auprc(y, scores)
    return metrics
