"""Assemble the four COMET subtype settings into one comparison table.

Reads evaluate_subtype.py outputs (evaluation.json + predictions.csv), ABMIL
slide_scores.tsv, and NVFlare client logs (runtime, peak memory, adapter bytes).
Prints Markdown and writes a JSON copy next to it. Validation by default; pass
--split test only after every setting is frozen.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from subtype_utils import SUBTYPE_CLASSES, subtype_metrics

LOG_RE = re.compile(r"site=(?P<site>site-\d+), round=(?P<round>\d+).*?train_runtime_sec=(?P<train>[\d.]+).*?"
                    r"eval_runtime_sec=(?P<eval>[\d.]+).*?cuda_peak_allocated_mb=(?P<peak>[\d.]+)")
SENT_RE = re.compile(r"site=(?P<site>site-\d+), round=(?P<round>\d+), sent updated adapter size: (?P<mb>[\d.]+) MB")
RECV_RE = re.compile(r"site=(?P<site>site-\d+), round=(?P<round>\d+), received adapter size: (?P<mb>[\d.]+) MB")


def tuned_threshold(truth, scores, positive):
    """Threshold maximising balanced accuracy on these (validation) scores."""
    best = (None, -1.0)
    for cut in sorted(set(scores)):
        pred = [positive if s >= cut else [c for c in SUBTYPE_CLASSES if c != positive][0] for s in scores]
        m = subtype_metrics(truth, pred, scores, SUBTYPE_CLASSES, positive)
        if m["balanced_accuracy"] > best[1]:
            best = (cut, m["balanced_accuracy"])
    return best


def from_predictions(path, positive="ARMS"):
    rows = list(csv.DictReader(open(path)))
    truth = [r["truth"] for r in rows]
    if f"score_{positive}" in rows[0]:
        scores = [float(r[f"score_{positive}"]) for r in rows]
    else:
        other = [c for c in SUBTYPE_CLASSES if c != positive][0]
        scores = [1 - float(r[f"score_{other}"]) for r in rows]
    pred = [r["prediction"] for r in rows]
    m = subtype_metrics(truth, [p if p in SUBTYPE_CLASSES else "ERMS" for p in pred], scores, SUBTYPE_CLASSES, positive)
    m["invalid_predictions"] = sum(p not in SUBTYPE_CLASSES for p in pred)
    cut, bacc = tuned_threshold(truth, scores, positive)
    m["tuned_threshold"], m["tuned_balanced_accuracy"] = cut, bacc
    m["mean_score_by_truth"] = {c: sum(s for s, t in zip(scores, truth) if t == c) / max(1, truth.count(c)) for c in SUBTYPE_CLASSES}
    return m


def from_abmil(path, split, positive="ARMS"):
    rows = [r for r in csv.DictReader(open(path), delimiter="\t") if r["split"] == split]
    truth = [r["truth"] for r in rows]
    scores = [float(r["score_ARMS"]) for r in rows]
    pred = ["ARMS" if s >= 0.5 else "ERMS" for s in scores]
    m = subtype_metrics(truth, pred, scores, SUBTYPE_CLASSES, positive)
    m["invalid_predictions"] = 0
    cut, bacc = tuned_threshold(truth, scores, positive)
    m["tuned_threshold"], m["tuned_balanced_accuracy"] = cut, bacc
    m["mean_score_by_truth"] = {c: sum(s for s, t in zip(scores, truth) if t == c) / max(1, truth.count(c)) for c in SUBTYPE_CLASSES}
    return m


def from_log(path):
    """GPU-hours (sum of client train+eval time), peak memory, bytes exchanged."""
    if not path or not Path(path).is_file():
        return {}
    text = Path(path).read_text(errors="ignore")
    train = eval_ = 0.0
    peak = 0.0
    rounds, sites = set(), set()
    for m in LOG_RE.finditer(text):
        train += float(m["train"]); eval_ += float(m["eval"])
        peak = max(peak, float(m["peak"])); rounds.add(int(m["round"])); sites.add(m["site"])
    sent = sum(float(m["mb"]) for m in SENT_RE.finditer(text))
    recv = sum(float(m["mb"]) for m in RECV_RE.finditer(text))
    return {"gpu_hours_train": round(train / 3600, 2), "gpu_hours_eval_in_training": round(eval_ / 3600, 2),
            "peak_cuda_allocated_gb": round(peak / 1024, 1), "rounds": len(rounds), "clients": len(sites),
            "adapter_mb_sent_total": round(sent, 0), "adapter_mb_received_total": round(recv, 0),
            "comm_gb_total": round((sent + recv) / 1024, 1)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--runs", default="runs/comet_subtype")
    p.add_argument("--logs", default="logs")
    p.add_argument("--abmil", default="runs/comet_subtype/abmil_hoptimus1_v1")
    p.add_argument("--tag", default="v1")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    runs, logs, S, T = Path(a.runs), Path(a.logs), a.split, a.tag
    settings = [
        ("01 H-Optimus-1 + ABMIL (image)", "abmil", Path(a.abmil) / "slide_scores.tsv", None),
        ("02 MedGemma 1.5 zero-shot", "pred", runs / f"zero_shot_multimodal_{S}_{T}" / "predictions.csv", None),
        ("03 MedGemma 1.5 + LoRA centralized", "pred", runs / f"central_multimodal_{S}_{T}" / "predictions.csv",
         logs / f"comet_central_multimodal_{T}.log"),
        ("04 FL iid", "pred", runs / f"fl_multimodal_iid_{S}_{T}" / "predictions.csv", logs / f"comet_fl_multimodal_iid_{T}.log"),
        ("04 FL skew 0.8", "pred", runs / f"fl_multimodal_skew08_{S}_{T}" / "predictions.csv", logs / f"comet_fl_multimodal_skew08_{T}.log"),
        ("04 FL skew 1.0", "pred", runs / f"fl_multimodal_skew10_{S}_{T}" / "predictions.csv", logs / f"comet_fl_multimodal_skew10_{T}.log"),
    ]
    table, out = [], {}
    for name, kind, path, log in settings:
        if not path.is_file():
            table.append(f"| {name} | missing: {path} |" + " |" * 8)
            continue
        m = from_abmil(path, S) if kind == "abmil" else from_predictions(path)
        c = from_log(log)
        out[name] = {"metrics": m, "cost": c, "source": str(path)}
        cost = (f"{c['gpu_hours_train']} GPU-h, {c['rounds']}×{c['clients']}, {c['comm_gb_total']} GB"
                if c else ("20 s (+1.03 GPU-h shared embedding)" if kind == "abmil" else "0 (inference only)"))
        table.append(f"| {name} | {m['n_patients']} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['accuracy']:.3f} | "
                     f"{m['balanced_accuracy']:.3f} | {m['recall']['ERMS']:.2f} / {m['recall']['ARMS']:.2f} | "
                     f"{m['tuned_balanced_accuracy']:.3f} @ {m['tuned_threshold']:.3f} | {m['invalid_predictions']} | {cost} |")
    header = (f"| Setting ({S}) | n | ROC-AUC | PR-AUC | Acc | Bal. acc | Recall ERMS / ARMS | "
              "Bal. acc @ tuned thr | Unparsed | Training cost (GPU-h, rounds×clients, adapter traffic) |\n"
              "|---|---|---|---|---|---|---|---|---|---|")
    print(header + "\n" + "\n".join(table))
    dest = Path(a.output or runs / f"summary_{S}_{T}.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nJSON written to {dest}. Tuned thresholds are chosen on {S}; apply them to test only if chosen on validation.")


if __name__ == "__main__":
    main()
