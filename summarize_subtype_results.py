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
    # A reply that is not exactly ERMS or ARMS stays wrong. It is not scored as ERMS.
    m = subtype_metrics(truth, pred, scores, SUBTYPE_CLASSES, positive)
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
    """Per-site training time and peak memory, plus upload/download per round."""
    if not path or not Path(path).is_file():
        return {}
    text = Path(path).read_text(errors="ignore")
    train_by_site, eval_by_site, peak_by_site = {}, {}, {}
    rounds = set()
    for m in LOG_RE.finditer(text):
        site, rnd = m["site"], int(m["round"])
        rounds.add(rnd)
        train_by_site[site] = train_by_site.get(site, 0.0) + float(m["train"])
        eval_by_site[site] = eval_by_site.get(site, 0.0) + float(m["eval"])
        peak_by_site[site] = max(peak_by_site.get(site, 0.0), float(m["peak"]))
    sent_by_round, recv_by_round = {}, {}
    for m in SENT_RE.finditer(text):
        rnd = int(m["round"])
        sent_by_round[rnd] = sent_by_round.get(rnd, 0.0) + float(m["mb"])
    for m in RECV_RE.finditer(text):
        rnd = int(m["round"])
        recv_by_round[rnd] = recv_by_round.get(rnd, 0.0) + float(m["mb"])
    train, eval_ = sum(train_by_site.values()), sum(eval_by_site.values())
    sent, recv = sum(sent_by_round.values()), sum(recv_by_round.values())
    return {
        "train_seconds_total": round(train, 1),
        "gpu_hours_train": round(train / 3600, 2),
        "gpu_hours_eval_in_training": round(eval_ / 3600, 2),
        "peak_cuda_allocated_gb": round(max(peak_by_site.values()) / 1024, 1) if peak_by_site else None,
        "peak_gb_by_site": {s: round(v / 1024, 2) for s, v in sorted(peak_by_site.items())},
        "train_hours_by_site": {s: round(v / 3600, 2) for s, v in sorted(train_by_site.items())},
        "rounds": len(rounds) or len(sent_by_round) or len(recv_by_round),
        "clients": len(train_by_site),
        "upload_mb_by_round": {str(r): round(sent_by_round[r], 1) for r in sorted(sent_by_round)},
        "download_mb_by_round": {str(r): round(recv_by_round[r], 1) for r in sorted(recv_by_round)},
        "adapter_mb_sent_total": round(sent, 1),
        "adapter_mb_received_total": round(recv, 1),
        "comm_gb_total": round((sent + recv) / 1024, 2),
    }


def _by_site(mapping, unit):
    if not mapping:
        return "—"
    return "; ".join(f"{site} {value:.2f} {unit}" for site, value in mapping.items())


def _by_round(mapping):
    if not mapping:
        return "0 MB"
    values = list(mapping.values())
    if len(set(values)) == 1:
        return f"{values[0]:.0f} MB × {len(values)} rounds"
    return "; ".join(f"round {rnd} {mb:.0f} MB" for rnd, mb in mapping.items())


def cost_cells(kind, cost):
    if not cost:
        if kind == "abmil":
            return ["—", "—", "—", "—", "—", "—", "—", "—"]
        return ["0", "0", "—", "—", "0", "0 MB", "0 MB", "0 GB"]
    return [
        f"{cost['train_seconds_total']:.0f} s",
        f"{cost['gpu_hours_train']:.2f}",
        _by_site(cost["peak_gb_by_site"], "GB"),
        _by_site(cost["train_hours_by_site"], "h"),
        str(cost["rounds"]),
        _by_round(cost["upload_mb_by_round"]),
        _by_round(cost["download_mb_by_round"]),
        f"{cost['comm_gb_total']:.2f} GB",
    ]


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
        ("03 MedGemma 1.5 + QLoRA centralized", "pred", runs / f"central_multimodal_{S}_{T}" / "predictions.csv",
         logs / f"comet_central_multimodal_{T}.log"),
        ("04 MedGemma 1.5 + QLoRA federated, IID", "pred", runs / f"fl_multimodal_iid_{S}_{T}" / "predictions.csv",
         logs / f"comet_fl_multimodal_iid_{T}.log"),
        ("04 MedGemma 1.5 + QLoRA federated, skew 0.8", "pred", runs / f"fl_multimodal_skew08_{S}_{T}" / "predictions.csv",
         logs / f"comet_fl_multimodal_skew08_{T}.log"),
        ("04 MedGemma 1.5 + QLoRA federated, skew 1.0", "pred", runs / f"fl_multimodal_skew10_{S}_{T}" / "predictions.csv",
         logs / f"comet_fl_multimodal_skew10_{T}.log"),
    ]
    performance, costs, out = [], [], {}
    for name, kind, path, log in settings:
        if not path.is_file():
            performance.append(f"| {name} | missing: {path} |" + " |" * 7)
            costs.append(f"| {name} | missing log |" + " |" * 7)
            continue
        m = from_abmil(path, S) if kind == "abmil" else from_predictions(path)
        c = from_log(log)
        out[name] = {"metrics": m, "cost": c, "source": str(path)}
        performance.append(
            f"| {name} | {m['n_patients']} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} | {m['accuracy']:.3f} | "
            f"{m['balanced_accuracy']:.3f} | {m['recall']['ERMS']:.2f} | {m['recall']['ARMS']:.2f} | "
            f"{m['invalid_predictions']} |"
        )
        costs.append("| " + " | ".join([name, *cost_cells(kind, c)]) + " |")
    perf_header = (
        f"| Setting ({S}) | n | ROC-AUC | PR-AUC | Accuracy | Balanced accuracy | ERMS recall | ARMS recall | "
        "Replies that are not ERMS or ARMS |\n|---|---|---|---|---|---|---|---|---|"
    )
    cost_header = (
        "| Setting | Train time | GPU hours | Peak GPU memory | Local train time | Rounds | "
        "Upload per round | Download per round | Total traffic |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    print("## Predictive performance\n")
    print(perf_header + "\n" + "\n".join(performance))
    print("\nA reply counts in the last column when it is not exactly ERMS or ARMS. "
          "That case is scored as wrong, not as a prediction of ERMS.\n")
    print("## Computational cost and communication\n")
    print(cost_header + "\n" + "\n".join(costs))
    print("\nTrain time and GPU hours add up every site. Local train time and peak GPU memory are listed per site. "
          "Upload and download add up the sites in that round. Total traffic is upload plus download. "
          "Zero-shot has no training and no adapter exchange.\n")
    dest = Path(a.output or runs / f"summary_{S}_{T}.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"JSON written to {dest}. Tuned thresholds are stored in the JSON and were chosen on {S}.")


if __name__ == "__main__":
    main()
