"""Cost summary keeps per-site time and per-round traffic separate."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from summarize_subtype_results import cost_cells, from_ddp_performance, from_log, from_predictions

ROOT = Path(__file__).resolve().parents[1]


def write_predictions(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["patient_id", "truth", "prediction", "score_POS"])
        w.writeheader()
        w.writerows(rows)


ROWS = [
    {"patient_id": "p0", "truth": "POS", "prediction": "POS", "score_POS": 0.9},
    {"patient_id": "p1", "truth": "POS", "prediction": "UNKNOWN", "score_POS": 0.2},
    {"patient_id": "p2", "truth": "NEG", "prediction": "NEG", "score_POS": 0.1},
    {"patient_id": "p3", "truth": "NEG", "prediction": "NEG", "score_POS": 0.3},
]


class PredictionScoringTests(unittest.TestCase):
    def test_invalid_reply_stays_in_cohort_and_counts_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            write_predictions(path, ROWS)
            m = from_predictions(path)
        self.assertEqual(m["n_patients"], 4)
        self.assertEqual(m["invalid_patients"], 1)
        self.assertEqual(m["invalid_predictions"], 1)
        self.assertEqual(m["prediction_coverage"], 0.75)
        # POS recall 1/2 (the invalid POS patient is wrong), NEG recall 2/2.
        self.assertAlmostEqual(m["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(m["recall"]["POS"], 0.5)

    def test_single_row_uses_model_reply_not_score(self):
        rows = [dict(r) for r in ROWS if r["patient_id"] != "p1"]
        rows.append({"patient_id": "p4", "truth": "POS", "prediction": "NEG", "score_POS": 0.8})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.csv"
            write_predictions(path, rows)
            m = from_predictions(path)
        self.assertAlmostEqual(m["recall"]["POS"], 0.5)


class DdpCostTests(unittest.TestCase):
    def test_ddp_performance_json_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            for rank, sec in ((0, 3600.0), (1, 3500.0)):
                (Path(tmp) / f"rank-{rank}_performance.json").write_text(json.dumps(
                    {"rank": rank, "world_size": 2, "train_runtime_sec": sec,
                     "peak_allocated_mb": 20480.0, "peak_reserved_mb": 24576.0}))
            (Path(tmp) / "completed.json").write_text("{}")
            cost = from_ddp_performance(tmp)
        self.assertTrue(cost["completed"])
        self.assertEqual(cost["world_size"], 2)
        self.assertEqual(cost["wall_clock_seconds"], 3600.0)
        self.assertEqual(cost["gpu_hours_train"], 1.97)
        self.assertEqual(cost["peak_gb_by_site"], {"rank-0": 20.0, "rank-1": 20.0})
        self.assertEqual(cost["rounds"], 0)
        self.assertNotEqual(cost_cells("pred", cost, True)[0], "0")

    def test_trained_setting_without_cost_is_not_shown_as_zero(self):
        self.assertEqual(cost_cells("pred", {}, expected=True)[0], "missing cost")
        self.assertEqual(cost_cells("pred", {}, expected=False)[0], "0")


class SummaryCliTests(unittest.TestCase):
    def test_cli_writes_json_with_abmil_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            write_predictions(tmp / "predictions.csv", ROWS)
            with open(tmp / "slide_scores.tsv", "w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["patient_id", "slide_id", "split", "truth", "score_POS", "tiles"])
                w.writerows([["p0", "s0", "validation", "POS", 0.8, 10], ["p2", "s2", "validation", "NEG", 0.1, 10]])
            (tmp / "m.json").write_text(json.dumps({"settings": [
                {"name": "X", "kind": "pred", "path": str(tmp / "predictions.csv")},
                {"name": "Y", "kind": "pred", "path": str(tmp / "absent.csv"), "log": str(tmp / "absent.log")}]}))
            (tmp / "abmil.json").write_text(json.dumps({"settings": [
                {"name": "A", "kind": "abmil", "path": str(tmp / "slide_scores.tsv")},
                {"name": "B", "kind": "abmil", "path": str(tmp / "absent.tsv")}]}))
            result = subprocess.run([sys.executable, str(ROOT / "summarize_subtype_results.py"),
                                     "--settings-manifest", str(tmp / "m.json"),
                                     "--abmil-settings-manifest", str(tmp / "abmil.json"),
                                     "--output", str(tmp / "out.json")],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            out = json.loads((tmp / "out.json").read_text())
        self.assertEqual(out["__usage__"]["completed_runs"], 1)
        self.assertEqual(out["__usage__"]["failed_runs"], 1)
        self.assertEqual(out["__usage__"]["valid_prediction_count_by_run"], {"X": 3})
        self.assertEqual(out["__usage__"]["abmil_baselines_completed"], 1)
        self.assertEqual(out["abmil_baselines"]["B"]["status"], "missing")


class SummaryCostTests(unittest.TestCase):
    def test_from_log_splits_sites_and_rounds(self):
        text = "\n".join([
            "site=site-1, round=1, local_lora_rank=16, train_runtime_sec=3600.00, eval_runtime_sec=10.00, "
            "round_runtime_sec=3610.00, cuda_peak_allocated_mb=40960.00",
            "site=site-1, round=1, received adapter size: 100.00 MB",
            "site=site-1, round=1, sent updated adapter size: 120.00 MB",
            "site=site-2, round=1, local_lora_rank=16, train_runtime_sec=1800.00, eval_runtime_sec=10.00, "
            "round_runtime_sec=1810.00, cuda_peak_allocated_mb=20480.00",
            "site=site-2, round=1, received adapter size: 100.00 MB",
            "site=site-2, round=1, sent updated adapter size: 80.00 MB",
        ])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "client.log"
            path.write_text(text)
            cost = from_log(path)
        self.assertEqual(cost["rounds"], 1)
        self.assertEqual(cost["clients"], 2)
        self.assertEqual(cost["gpu_hours_train"], 1.5)
        self.assertEqual(cost["train_hours_by_site"], {"site-1": 1.0, "site-2": 0.5})
        self.assertEqual(cost["peak_gb_by_site"], {"site-1": 40.0, "site-2": 20.0})
        self.assertEqual(cost["upload_mb_by_round"], {"1": 200.0})
        self.assertEqual(cost["download_mb_by_round"], {"1": 200.0})
        self.assertEqual(cost["comm_gb_total"], 0.39)


if __name__ == "__main__":
    unittest.main()
