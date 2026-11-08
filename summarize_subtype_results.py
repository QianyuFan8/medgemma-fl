"""Assemble predictive performance and cost metrics for matched experiments.

The main tables intentionally stay compact: balanced accuracy, ROC-AUC, PR-AUC,
sample counts, computational cost, and federated communication cost. Detailed
metrics (recall, thresholds, invalid replies, and score summaries) remain in
the JSON output. A settings manifest can describe the nine combinations of
three ABMIL feature settings and three MedGemma regimes.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

from subtype_utils import SUBTYPE_CLASSES, subtype_metrics

LOG_RE = re.compile(r"site=(?P<site>site-\d+), round=(?P<round>\d+).*?train_runtime_sec=(?P<train>[\d.]+).*?"
                    r"eval_runtime_sec=(?P<eval>[\d.]+).*?cuda_peak_allocated_mb=(?P<peak>[\d.]+)"
                    r"(?:, cuda_peak_reserved_mb=(?P<reserved>[\d.]+))?")
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


def from_predictions(path, positive="POS"):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty prediction file: {path}")
    n_records = len(rows)
    negative = [c for c in SUBTYPE_CLASSES if c != positive][0]

    def row_score(r):
        """Positive-class score, or None when the row carries no usable score."""
        try:
            if r.get(f"score_{positive}") not in (None, ""):
                return float(r[f"score_{positive}"])
            if r.get(f"score_{negative}") not in (None, ""):
                return 1 - float(r[f"score_{negative}"])
        except ValueError:
            pass
        return None

    # Every patient stays in the evaluation. A reply that is not exactly POS or
    # NEG is scored as wrong (not dropped, and not counted as NEG), so invalid
    # replies lower balanced accuracy instead of silently shrinking the cohort.
    # Repeated rows for one patient are collapsed so no patient is double-weighted.
    groups = {}
    for i, r in enumerate(rows):
        key = r.get("patient_id") or r.get("canonical_uid") or r.get("uid") or f"__row_{i}"
        g = groups.setdefault(key, {"truth": r["truth"], "replies": [], "scores": []})
        if r["truth"] != g["truth"]:
            raise ValueError(f"Conflicting truth labels for patient {key} in {path}")
        g["replies"].append(r.get("prediction"))
        score = row_score(r)
        if score is not None:
            g["scores"].append(score)
    truth, pred, scores = [], [], []
    invalid_patients = missing_score_patients = 0
    for g in groups.values():
        truth.append(g["truth"])
        # AUC needs one score per patient; a patient without any score gets the
        # uninformative 0.5 and is counted so the gap is visible in the JSON.
        if g["scores"]:
            scores.append(sum(g["scores"]) / len(g["scores"]))
        else:
            scores.append(0.5)
            missing_score_patients += 1
        if any(reply not in SUBTYPE_CLASSES for reply in g["replies"]):
            pred.append("INVALID")
            invalid_patients += 1
        elif len(g["replies"]) == 1:
            pred.append(g["replies"][0])  # the model's own reply, as in evaluate_subtype.py
        else:
            # Several valid rows: same patient-level rule as evaluate_subtype.py (mean score >= 0.5).
            pred.append(positive if scores[-1] >= 0.5 else negative)
    m = subtype_metrics(truth, pred, scores, SUBTYPE_CLASSES, positive)
    valid_records = sum(r.get("prediction") in SUBTYPE_CLASSES for r in rows)
    valid_patients = sum(not any(reply not in SUBTYPE_CLASSES for reply in g["replies"])
                         for g in groups.values())
    m.update({"n_records": n_records, "n_patients": len(groups),
              "valid_prediction_count": valid_patients,
              "valid_prediction_records": valid_records,
              "valid_prediction_patients": valid_patients,
              "invalid_predictions": n_records - valid_records,
              "invalid_prediction_records": n_records - valid_records,
              "invalid_prediction_patients": invalid_patients,
              "invalid_patients": invalid_patients,
              "missing_score_patients": missing_score_patients,
              "eligible_patients": len(groups),
              "prediction_coverage": (len(groups) - invalid_patients) / len(groups)})
    cut, bacc = tuned_threshold(truth, scores, positive)
    m["tuned_threshold"], m["tuned_balanced_accuracy"] = cut, bacc
    m["mean_score_by_truth"] = {c: sum(s for s, t in zip(scores, truth) if t == c) / max(1, truth.count(c)) for c in SUBTYPE_CLASSES}
    return m


def from_abmil(path, split, positive="POS"):
    with Path(path).open(newline="") as handle:
        rows = [r for r in csv.DictReader(handle, delimiter="\t") if r["split"] == split]
    truth = [r["truth"] for r in rows]
    scores = [float(r["score_POS"]) for r in rows]
    pred = ["POS" if s >= 0.5 else "NEG" for s in scores]
    m = subtype_metrics(truth, pred, scores, SUBTYPE_CLASSES, positive)
    m.update({"n_records": len(rows), "n_patients": len({r.get('patient_id', r['slide_id']) for r in rows}),
              "valid_prediction_count": len(rows), "valid_prediction_records": len(rows),
              "valid_prediction_patients": len({r.get('patient_id', r['slide_id']) for r in rows}),
              "invalid_predictions": 0, "invalid_prediction_records": 0,
              "invalid_prediction_patients": 0,
              "eligible_patients": len({r.get('patient_id', r['slide_id']) for r in rows}),
              "prediction_coverage": 1.0})
    cut, bacc = tuned_threshold(truth, scores, positive)
    m["tuned_threshold"], m["tuned_balanced_accuracy"] = cut, bacc
    m["mean_score_by_truth"] = {c: sum(s for s, t in zip(scores, truth) if t == c) / max(1, truth.count(c)) for c in SUBTYPE_CLASSES}
    return m


def from_log(path):
    """Per-site training time and peak memory, plus upload/download per round."""
    if not path or not Path(path).is_file():
        return {}
    text = Path(path).read_text(errors="ignore")
    train_by_site, eval_by_site, peak_by_site, reserved_by_site = {}, {}, {}, {}
    rounds = set()
    for m in LOG_RE.finditer(text):
        site, rnd = m["site"], int(m["round"])
        rounds.add(rnd)
        train_by_site[site] = train_by_site.get(site, 0.0) + float(m["train"])
        eval_by_site[site] = eval_by_site.get(site, 0.0) + float(m["eval"])
        peak_by_site[site] = max(peak_by_site.get(site, 0.0), float(m["peak"]))
        if m["reserved"] is not None:
            reserved_by_site[site] = max(reserved_by_site.get(site, 0.0), float(m["reserved"]))
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
        "peak_cuda_reserved_gb": round(max(reserved_by_site.values()) / 1024, 1) if reserved_by_site else None,
        "peak_gb_by_site": {s: round(v / 1024, 2) for s, v in sorted(peak_by_site.items())},
        "peak_reserved_gb_by_site": {s: round(v / 1024, 2) for s, v in sorted(reserved_by_site.items())},
        "train_hours_by_site": {s: round(v / 3600, 2) for s, v in sorted(train_by_site.items())},
        "rounds": len(rounds) or len(sent_by_round) or len(recv_by_round),
        "clients": len(train_by_site),
        "upload_mb_by_round": {str(r): round(sent_by_round[r], 1) for r in sorted(sent_by_round)},
        "download_mb_by_round": {str(r): round(recv_by_round[r], 1) for r in sorted(recv_by_round)},
        "adapter_mb_sent_total": round(sent, 1),
        "adapter_mb_received_total": round(recv, 1),
        "comm_gb_total": round((sent + recv) / 1024, 2),
        "communication_observed": bool(sent_by_round or recv_by_round),
    }


def from_ddp_performance(directory):
    """Cost of a DDP centralized run from train_centralized.py rank-*_performance.json.

    That script records time and memory in JSON, not in NVFlare-style log lines,
    so from_log() finds nothing for it. Each rank is listed like a site; GPU hours
    add up every rank. There are no federated rounds and no adapter exchange.
    """
    if not directory:
        return {}
    directory = Path(directory)
    reports = {}
    for path in sorted(directory.glob("rank-*_performance.json")):
        report = json.loads(path.read_text())
        reports[f"rank-{report.get('rank', path.stem.split('_')[0].split('-')[-1])}"] = report
    if not reports:
        return {}
    train = {rank: float(r["train_runtime_sec"]) for rank, r in reports.items()}
    peak = {rank: float(r["peak_allocated_mb"]) for rank, r in reports.items() if r.get("peak_allocated_mb") is not None}
    reserved = {rank: float(r["peak_reserved_mb"]) for rank, r in reports.items() if r.get("peak_reserved_mb") is not None}
    total = sum(train.values())
    return {
        "source": "ddp_performance_json",
        "completed": (directory / "completed.json").is_file(),
        "world_size": max(int(r.get("world_size", 1)) for r in reports.values()),
        "wall_clock_seconds": round(max(train.values()), 1),
        "train_seconds_total": round(total, 1),
        "gpu_hours_train": round(total / 3600, 2),
        "peak_cuda_allocated_gb": round(max(peak.values()) / 1024, 1) if peak else None,
        "peak_cuda_reserved_gb": round(max(reserved.values()) / 1024, 1) if reserved else None,
        "peak_gb_by_site": {s: round(v / 1024, 2) for s, v in sorted(peak.items())},
        "peak_reserved_gb_by_site": {s: round(v / 1024, 2) for s, v in sorted(reserved.items())},
        "train_hours_by_site": {s: round(v / 3600, 2) for s, v in sorted(train.items())},
        "rounds": 0,
        "clients": len(reports),
        "upload_mb_by_round": {},
        "download_mb_by_round": {},
        "adapter_mb_sent_total": 0.0,
        "adapter_mb_received_total": 0.0,
        "comm_gb_total": 0.0,
        "communication_observed": False,
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


def cost_cells(kind, cost, expected=False):
    """expected=True means the setting trained (it names a log or performance dir),
    so an empty cost is missing data, not the zero cost of zero-shot inference."""
    if not cost:
        if kind == "abmil":
            return ["—", "—", "—", "—", "—", "—", "—", "—"]
        if expected:
            return ["missing cost", "—", "—", "—", "—", "—", "—", "—"]
        return ["0", "0", "—", "—", "0", "0 MB", "0 MB", "0 GB"]
    status_prefix = "" if cost.get("completed", True) else "incomplete; "
    return [
        status_prefix + f"{cost['train_seconds_total']:.0f} s",
        f"{cost['gpu_hours_train']:.2f}",
        _by_site(cost["peak_gb_by_site"], "GB allocated") +
        ("; " + _by_site(cost["peak_reserved_gb_by_site"], "GB reserved")
         if cost.get("peak_reserved_gb_by_site") else ""),
        _by_site(cost["train_hours_by_site"], "h"),
        str(cost["rounds"]),
        _by_round(cost["upload_mb_by_round"]),
        _by_round(cost["download_mb_by_round"]),
        f"{cost['comm_gb_total']:.2f} GB",
    ]


def load_settings(path, runs, split, logs, abmil):
    """Load a comparison manifest while retaining the historical default."""
    if path:
        rows = json.loads(Path(path).read_text())
        if isinstance(rows, dict):
            rows = rows.get("settings", rows.get("experiments", []))
        out = []
        for row in rows:
            item = dict(row)
            item.setdefault("kind", "pred")
            item["path"] = Path(str(item["path"]).replace("{split}", split))
            if item.get("log"):
                item["log"] = Path(str(item["log"]).replace("{split}", split))
            if item.get("performance"):
                item["performance"] = Path(str(item["performance"]).replace("{split}", split))
            out.append(item)
        return out
    return [
        {"name": "01 H-Optimus-1 + ABMIL", "kind": "abmil", "path": Path(abmil), "log": None},
        {"name": "02 MedGemma zero-shot", "kind": "pred", "path": runs / f"zero_shot_multimodal_{split}_v1" / "predictions.csv", "log": None},
        {"name": "03 MedGemma centralized LoRA", "kind": "pred", "path": runs / f"central_multimodal_{split}_v1" / "predictions.csv", "log": logs / f"comet_central_multimodal_{split}.log"},
        {"name": "04 MedGemma federated LoRA IID", "kind": "pred", "path": runs / f"fl_multimodal_iid_{split}_v1" / "predictions.csv", "log": logs / f"comet_fl_multimodal_iid_{split}.log"},
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--runs", default="runs/comet_subtype")
    p.add_argument("--logs", default="logs")
    p.add_argument("--abmil", default="runs/comet_subtype/abmil_hoptimus1_v1")
    p.add_argument("--settings-manifest", help="JSON list of experiment settings; paths may be absolute or relative to the current directory")
    p.add_argument("--abmil-settings-manifest", help="Optional JSON list of the three standalone ABMIL baselines")
    p.add_argument("--tag", default="v1")
    p.add_argument("--output", default=None)
    a = p.parse_args()
    runs, logs, S, T = Path(a.runs), Path(a.logs), a.split, a.tag
    settings = load_settings(a.settings_manifest, runs, S, logs, Path(a.abmil) / "slide_scores.tsv")
    performance, costs, out = [], [], {}
    planned_runs = len(settings)
    completed_runs = 0
    failed_runs = 0
    for setting in settings:
        name, kind, path, log = setting["name"], setting.get("kind", "pred"), Path(setting["path"]), setting.get("log")
        if not path.is_file():
            failed_runs += 1
            performance.append(f"| {name} | missing: {path} |" + " — |" * 5)
            costs.append(f"| {name} | missing |" + " — |" * 7)
            continue
        m = from_abmil(path, S) if kind == "abmil" else from_predictions(path)
        completed_runs += 1
        # DDP centralized runs write rank-*_performance.json; federated runs write NVFlare logs.
        c = from_ddp_performance(setting.get("performance")) or from_log(log)
        out[name] = {"metrics": m, "cost": c, "source": str(path), "status": "completed",
                     "prediction_coverage": m.get("prediction_coverage", 1.0)}
        performance.append(
            f"| {name} | {m.get('n_patients', '—')} | {m.get('n_records', '—')} | "
            f"{m.get('invalid_patients', 0)} | {m['balanced_accuracy']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |"
        )
        expected = bool(log or setting.get("performance"))
        costs.append("| " + " | ".join([name, *cost_cells(kind, c, expected)]) + " |")
    abmil_out = {}
    if a.abmil_settings_manifest:
        baseline_settings = load_settings(a.abmil_settings_manifest, runs, S, logs,
                                          Path(a.abmil) / "slide_scores.tsv")
        for setting in baseline_settings:
            name, path = setting["name"], Path(setting["path"])
            if not path.is_file():
                abmil_out[name] = {"status": "missing", "source": str(path)}
                continue
            metrics = from_abmil(path, S)
            abmil_out[name] = {"status": "completed", "source": str(path), "metrics": metrics}
    perf_header = (
        f"| Setting ({S}) | Patients | Records | Invalid replies (patients) | Balanced accuracy | ROC-AUC | PR-AUC |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    cost_header = (
        "| Setting | Training time | GPU hours | Peak GPU memory (allocated/reserved) | Local computation | Rounds | "
        "Upload per round | Download per round | Total communication |\n"
        "|---|---|---|---|---|---:|---|---|---|"
    )
    print("## Predictive performance\n")
    print(perf_header + "\n" + "\n".join(performance))
    if abmil_out:
        print("\n## Standalone ABMIL baselines\n")
        print(perf_header.replace("Setting", "ABMIL setting"))
        for name, item in abmil_out.items():
            if item["status"] != "completed":
                print(f"| {name} | missing | — | — | — | — | — |")
            else:
                m = item["metrics"]
                print(f"| {name} | {m['n_patients']} | {m['n_records']} | 0 | "
                      f"{m['balanced_accuracy']:.3f} | {m['roc_auc']:.3f} | {m['pr_auc']:.3f} |")
    print("\nA patient whose reply is not exactly POS or NEG stays in the evaluation and is scored as wrong. "
          "Detailed recall, thresholds and score summaries are retained in the JSON output.\n")
    print("## Computational cost and communication\n")
    print(cost_header + "\n" + "\n".join(costs))
    print("\nTrain time and GPU hours add up every site. Local train time and peak GPU memory are listed per site. "
          "Upload and download add up the sites in that round. Total traffic is upload plus download. "
          "Centralized DDP lists each GPU rank in place of a site and has no rounds or adapter exchange. "
          "Zero-shot has no training and no adapter exchange. "
          "\"missing cost\" means the setting trained but its log or performance files were not found.\n")
    # Build the usage block from the experiment entries only, before adding the
    # ABMIL baseline block, which has a different shape (no top-level metrics).
    experiments = dict(out)
    out["__usage__"] = {
        "planned_runs": planned_runs,
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "valid_prediction_count_by_run": {
            name: item["metrics"].get("valid_prediction_count", 0) for name, item in experiments.items()
        },
        "valid_prediction_records_by_run": {
            name: item["metrics"].get("valid_prediction_records", 0) for name, item in experiments.items()
        },
        "valid_prediction_patients_by_run": {
            name: item["metrics"].get("valid_prediction_patients", 0) for name, item in experiments.items()
        },
        "prediction_coverage_by_run": {
            name: item["prediction_coverage"] for name, item in experiments.items()
        },
        "abmil_baselines_completed": sum(item["status"] == "completed" for item in abmil_out.values()),
        "abmil_baselines_missing": sum(item["status"] != "completed" for item in abmil_out.values()),
    }
    out["abmil_baselines"] = abmil_out
    dest = Path(a.output or runs / f"summary_{S}_{T}.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"JSON written to {dest}. Tuned thresholds are stored in the JSON and were chosen on {S}.")


if __name__ == "__main__":
    main()
