"""Prevent excluded subtypes entering teacher, gold or silver training."""
import copy
from pathlib import Path
import sys
import unittest
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ridge_teacher.cohort.split import allocate, audit
from ridge_teacher.cohort.policy import assert_training_policy, excluded_subtype
from ridge_teacher.image_processing.manifest import export_manifest
import test_core
import tempfile


class SubtypePolicyTests(unittest.TestCase):
    def test_swap_preserves_holdout_counts_labels_and_test_membership(self):
        make = test_core.PreparationTests().row
        rows = [make(f"p{i}", "POS" if i < 34 else "NEG") for i in range(59)]
        old, _ = audit(rows); allocate(old)
        old_by = {r["patient_id"]: r for r in old}
        spindle = next(r["patient_id"] for r in old if r["split"] == "train_gold" and r["gold_label"] == "NEG")
        rms_test = next(r["patient_id"] for r in old if r["split"] == "test" and r["gold_label"] == "NEG")
        for r in rows:
            if r["patient_id"] == spindle: r["subtype"] = "Spindle"
            if r["patient_id"] == rms_test: r["subtype"] = "RMS-NOS"
        for uid, label, slide, subtype in [("methyl_only", "NEG", False, "RMS NOS"),
                                          ("silver_blocked", "UNKNOWN", True, "Spindle"),
                                          ("silver_ok", "UNKNOWN", True, "ERMS")]:
            r = make(uid, label, slide=slide); r["subtype"] = subtype; rows.append(r)
        current, _ = audit(rows); allocate(current)
        by = {r["patient_id"]: r for r in current}
        self.assertEqual(by[spindle]["split"], "validation")
        self.assertFalse(by[spindle]["teacher_train"])
        self.assertEqual(by[rms_test]["split"], "test")
        changed = [uid for uid in old_by if old_by[uid]["split"] != by[uid]["split"]]
        self.assertEqual(len(changed), 2)
        replacement = next(uid for uid in changed if uid != spindle)
        self.assertEqual(by[replacement]["split"], "train_gold")
        self.assertEqual(by[replacement]["gold_label"], "NEG")
        self.assertTrue(by[replacement]["teacher_train"])
        for split in ["train_gold", "validation", "test"]:
            self.assertEqual(sorted(r["gold_label"] for r in old if r["split"] == split),
                             sorted(r["gold_label"] for r in current if r["split"] == split))
        self.assertEqual(by["methyl_only"]["split"], "prediction_only")
        self.assertEqual(by["silver_blocked"]["split"], "prediction_only")
        self.assertFalse(by["methyl_only"]["teacher_train"])
        self.assertEqual(by["silver_ok"]["split"], "train_silver_candidate")
        assert_training_policy(pd.DataFrame(current))
        reverse, _ = audit(list(reversed(rows))); allocate(reverse)
        self.assertEqual({r["patient_id"]:r["split"] for r in current},
                         {r["patient_id"]:r["split"] for r in reverse})

    def test_policy_rejects_all_training_routes_and_missing_metadata(self):
        for subtype in ["Spindle", "spindle cell RMS", "RMS-NOS", "RMS NOS", "Infantile", "Sclerosing", "Unknown", "", "new subtype"]:
            self.assertTrue(excluded_subtype(subtype))
            for split, teacher in [("excluded", True), ("train_gold", False), ("train_silver_candidate", False)]:
                p = pd.DataFrame([dict(subtype=subtype, teacher_train=teacher, split=split)])
                with self.assertRaisesRegex(ValueError, "Subtype exclusion"):
                    assert_training_policy(p)
        with self.assertRaisesRegex(ValueError, "Subtype metadata"):
            assert_training_policy(pd.DataFrame([dict(teacher_train=True, split="train_gold")]))
        # Previously generated manifests without the new exclusion flag still fail safely.
        assert_training_policy(pd.DataFrame([dict(subtype="RMS NOS",teacher_train=False,split="test")]))
        for subtype in ["ERMS", "ARMS", "E-RMS", "A-RMS", " erms "]:
            self.assertFalse(excluded_subtype(subtype))
            assert_training_policy(pd.DataFrame([dict(subtype=subtype,teacher_train=True,split="train_gold")]))

    def test_repeated_patient_cannot_hide_excluded_subtype(self):
        make = test_core.PreparationTests().row
        a = make("mixed", "NEG"); b = copy.deepcopy(a)
        b.update(subtype="Spindle", sentrix_id="another_array", **{"Sample ID":"z_other"})
        patients, _ = audit([a,b])
        self.assertEqual(patients[0]["subtype"], "ERMS")
        self.assertTrue(patients[0]["training_excluded"])
        patients[0].update(teacher_train=True, split="train_gold")
        with self.assertRaisesRegex(ValueError, "Subtype exclusion"):
            assert_training_policy(pd.DataFrame(patients))

    def test_image_manifest_rejects_prediction_only_silver(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            pd.DataFrame([dict(patient_id="p",subtype="Spindle",teacher_train=False,split="prediction_only")]).to_csv(d/"patients.csv",index=False)
            pd.DataFrame([dict(patient_id="p",included=True)]).to_csv(d/"silver.csv",index=False)
            with self.assertRaisesRegex(ValueError, "outside the allowed"):
                export_manifest(d,d/"silver.csv",d/"image.csv")
