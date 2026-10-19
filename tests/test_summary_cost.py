"""Cost summary keeps per-site time and per-round traffic separate."""
import tempfile
import unittest
from pathlib import Path

from summarize_subtype_results import from_log


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
