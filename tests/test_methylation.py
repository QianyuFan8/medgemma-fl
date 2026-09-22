import csv
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_methylation import evaluate_rows
from methylation_utils import make_record, classification_metrics, format_methylation_example, encode_methylation_example
from methylation_validate import validate_clients
from prepare_methylation import export_sites


class MethylationTests(unittest.TestCase):
    def test_prompt_and_metrics(self):
        r = make_record("SECRET_PATIENT", "WT", ["cg00000001"], [0.2], ["CCSK", "WT"])
        self.assertNotIn("SECRET_PATIENT", r["prompt"])
        self.assertEqual(format_methylation_example(r)["messages"][1]["content"][0]["text"], "WT")
        result = classification_metrics(["WT", "CCSK"], ["WT", "<unparsed>"], ["CCSK", "WT"])
        self.assertEqual(result["balanced_accuracy"], .5)
        self.assertEqual(result["invalid_predictions"], 1)
        with self.assertRaises(ValueError):
            make_record("P", "WT", ["cg1"], [float("nan")], ["WT"])

    def test_exact_evaluation_cohort(self):
        ref = [{"patient_id": "P1", "label_name": "A"}, {"patient_id": "P2", "label_name": "B"}]
        pred = [{"patient_id": "P1", "truth": "A", "prediction": "A"}]
        with self.assertRaises(ValueError):
            evaluate_rows(pred, ref, ["A", "B"])

    def test_export_and_leakage_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ("train", "validation", "test"):
                with (root / f"{split}.csv").open("w") as f:
                    w = csv.writer(f); w.writerow(["patient_id", "diagnosis", "cg00000001"])
                    for dx in ("WT", "CCSK"):
                        for i in range(3):
                            w.writerow([f"{split}_{dx}_{i}", dx, .3])
            export_sites(root, 3)
            validate_clients(root / "clients", 3)
            test_patient = json.loads((root / "test.json").read_text())[0]["patient_id"]
            site = root / "clients/site-1/train.json"
            records = json.loads(site.read_text()); records[0]["patient_id"] = test_patient
            site.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_clients(root / "clients", 3)

    def test_assistant_mask_and_overflow(self):
        class Tokenizer:
            pad_token_id = 0
            def __call__(self, text, **kwargs):
                return {"input_ids": [ord(c) for c in text]}
        class Processor:
            tokenizer = Tokenizer()
            def apply_chat_template(self, messages, add_generation_prompt=False, **kwargs):
                prefix = "USER:" + messages[0]["content"][0]["text"] + "\nASSISTANT:"
                return prefix if add_generation_prompt else prefix + messages[1]["content"][0]["text"] + "!"
        record = format_methylation_example(make_record("P", "WT", ["cg1"], [.2], ["CCSK", "WT"]))
        ids, labels = encode_methylation_example(Processor(), record)
        supervised = [x for x in labels if x != -100]
        self.assertEqual(supervised, list(map(ord, "WT!")))
        with self.assertRaisesRegex(ValueError, "tokens"):
            encode_methylation_example(Processor(), record, 2)


if __name__ == "__main__":
    unittest.main()
