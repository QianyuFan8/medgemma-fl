import argparse
import csv
import json
import os
from pathlib import Path
import random
import tempfile
import unittest

from PIL import Image

from methylation_utils import panel_fingerprint
from methylation_validate import validate_clients
from prepare_comet_subtype import allocate, assign_sites, audit, export, prepare
from prepare_shared_methylation import read_tsv, write_tsv
from subtype_utils import (SUBTYPE_CLASSES, binary_auprc, binary_auroc, format_subtype_example,
                           make_subtype_record, normalize_slide_id, patient_from_compbio, read_mapping,
                           subtype_metrics)
from evaluate_subtype import lenient_parse, positive_score


def synthetic_inputs(root, n_probes=400):
    """Mapping table, metadata, beta matrix (with detection p) and slide list."""
    rng = random.Random(7)
    rows, meta, sentrix = [], [], []
    spec = [("ERMS", 7), ("ARMS", 7), ("Spindle", 1)]
    k = 0
    for subtype, n in spec:
        for i in range(n):
            sid = f"20000000{k:04d}_R01C01"
            rows.append({"Sample ID": f"RMS_{k}", "Subtype": subtype, "Comp Bio ID": f"SJRHB{k:03d}_D1",
                         "X850k Tumor File Name": sid, "Slide Scan File": 7000 + k if k % 3 else ""})
            meta.append(dict(Sample_Name=sid, **{"UID.Subject": f"SJ{k}", "Disease": "RMS", "Sample.Type": "Patient tumor",
                                                "Final.QC": "PASS", "Pass.QC": "PASS"}))
            sentrix.append((sid, subtype))
            k += 1
    # Second sample of patient 0 (same subtype) with the slide, a QC failure, a shared slide, a missing array.
    rows.append({"Sample ID": "RMS_dup", "Subtype": "ERMS", "Comp Bio ID": "SJRHB000_D2",
                 "X850k Tumor File Name": "20000000dup_R01C01", "Slide Scan File": 7999})
    meta.append(dict(Sample_Name="20000000dup_R01C01", **{"UID.Subject": "SJ0", "Disease": "RMS", "Sample.Type": "Patient tumor",
                                                          "Final.QC": "PASS", "Pass.QC": "PASS"}))
    sentrix.append(("20000000dup_R01C01", "ERMS"))
    rows.append({"Sample ID": "RMS_qc", "Subtype": "ARMS", "Comp Bio ID": "SJRHB900_D1",
                 "X850k Tumor File Name": "20000000900_R01C01", "Slide Scan File": 7900})
    meta.append(dict(Sample_Name="20000000900_R01C01", **{"UID.Subject": "SJ900", "Disease": "RMS", "Sample.Type": "Patient tumor",
                                                          "Final.QC": "FAIL", "Pass.QC": "PASS"}))
    sentrix.append(("20000000900_R01C01", "ARMS"))
    rows.append({"Sample ID": "RMS_sh1", "Subtype": "ARMS", "Comp Bio ID": "SJRHB901_D1",
                 "X850k Tumor File Name": "20000000901_R01C01", "Slide Scan File": 8135})
    rows.append({"Sample ID": "RMS_sh2", "Subtype": "ARMS", "Comp Bio ID": "SJRHB902_D1",
                 "X850k Tumor File Name": "20000000902_R01C01", "Slide Scan File": 8135})
    for s in ("20000000901_R01C01", "20000000902_R01C01"):
        meta.append(dict(Sample_Name=s, **{"UID.Subject": s, "Disease": "RMS", "Sample.Type": "Patient tumor",
                                           "Final.QC": "PASS", "Pass.QC": "PASS"}))
        sentrix.append((s, "ARMS"))
    rows.append({"Sample ID": "RMS_missing", "Subtype": "ERMS", "Comp Bio ID": "SJRHB903_D1",
                 "X850k Tumor File Name": "20000000903_R01C01", "Slide Scan File": ""})
    with (root / "map.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_tsv(root / "meta.txt", meta)
    with (root / "beta.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        header = ["TargetID"]
        for s, _ in sentrix:
            header += [f"{s}.Ave_Beta", f"{s}.Detection.Pval"]
        w.writerow(header)
        for j in range(n_probes):
            row = [f"cg{j:08d}"]
            for s, subtype in sentrix:
                base = 0.3 if subtype == "ARMS" and j < 40 else 0.6
                row += [f"{base + rng.uniform(-0.05, 0.05):.6f}", "0.5" if (j == 5 and s.endswith("0001_R01C01")) else "0"]
            w.writerow(row)
    slides = [f"{7000 + k}.svs" for k in range(15) if k % 3] + ["7999.svs", "7900.svs", "8135.svs", "6217.svs"]
    (root / "slides.txt").write_text("\n".join(slides) + "\n")
    return rows


def fake_prepared(root, audit_dir, n_probes=12):
    """A prepared directory with the files export() expects, without running R."""
    rows = [r for r in read_tsv(audit_dir / "samples_audit.tsv") if r["status"] == "candidate_pending_label_review"]
    probes = [f"cg{j:08d}" for j in range(n_probes)]
    root.mkdir(parents=True)
    rng = random.Random(1)
    features = [dict(feature_order=i + 1, probe_id=p, training_median=0.5, p_value=1e-5, bh_fdr=1e-3) for i, p in enumerate(probes)]
    write_tsv(root / "features.tsv", features)
    import hashlib
    digest = hashlib.sha256((root / "features.tsv").read_bytes()).hexdigest()
    (root / "features.json").write_text(json.dumps(dict(schema_version=1, classes=SUBTYPE_CLASSES, probes=probes,
                                                      panel_id=panel_fingerprint(probes), selection={"tested_nonconstant_cpgs": 100},
                                                      features_sha256=digest, training_medians=[0.5] * n_probes, qc_mode="lenient")))
    splits = {}
    # Cycle splits within each (subtype, slide availability) group so every split keeps
    # both classes even after image exports drop slide-less patients.
    groups = {}
    for r in sorted(rows, key=lambda r: r["patient_id"]):
        groups.setdefault((r["diagnosis"], r["slide_status"] == "available"), []).append(r)
    for members in groups.values():
        for i, r in enumerate(members):
            split = ["train", "validation", "test", "train", "train"][i % 5]
            splits[r["patient_id"]] = dict(r, split=split, qc_pass="TRUE", include="true")
    # Ensure each split has both classes
    by_split = {}
    for r in splits.values():
        by_split.setdefault(r["split"], set()).add(r["diagnosis"])
    assert all(len(v) == 2 for v in by_split.values()), by_split
    write_tsv(root / "splits.tsv", list(splits.values()))
    for split in ("train", "validation", "test"):
        with (root / f"{split}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "diagnosis"] + probes)
            for r in splits.values():
                if r["split"] == split:
                    w.writerow([r["patient_id"], r["diagnosis"]] + [f"{rng.uniform(0.1, 0.9):.6f}" for _ in probes])
    for name in ("FEATURES.md", "selected_cpgs.tsv", "selection_summary.json", "preprocessing.rds", "config.json",
                 "cohort_provenance.json", "R_sessionInfo.txt"):
        (root / name).write_text("{}\n" if name.endswith(".json") else "x\n")
    return splits


class UtilsTests(unittest.TestCase):
    def test_ids(self):
        self.assertEqual(patient_from_compbio("SJRHB031412_D4"), "SJRHB031412")
        self.assertEqual(patient_from_compbio("SJRHB020_A"), "SJRHB020")
        self.assertEqual(normalize_slide_id(8135.0), "8135")
        self.assertEqual(normalize_slide_id("NA"), "")
        self.assertEqual(normalize_slide_id(None), "")

    def test_metrics(self):
        self.assertEqual(binary_auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)
        self.assertAlmostEqual(binary_auprc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), (1 + 2 / 3) / 2)
        m = subtype_metrics(["ERMS", "ARMS", "ARMS", "ERMS"], ["ERMS", "ARMS", "ERMS", "ERMS"], [0.2, 0.9, 0.4, 0.1], SUBTYPE_CLASSES)
        self.assertEqual(m["positive_class"], "ARMS")
        self.assertEqual(m["accuracy"], 0.75)
        self.assertEqual(m["roc_auc"], 1.0)
        with self.assertRaises(ValueError):
            binary_auroc([1, 1], [0.1, 0.2])

    def test_records_and_prompts(self):
        probes, values = ["cg1", "cg2"], [0.25, 0.5]
        meth = make_subtype_record("P1", "ERMS", probes, values, SUBTYPE_CLASSES, "methylation", image="x.png")
        self.assertIsNone(meth["image"])
        self.assertIn("cg1: 0.250000\ncg2: 0.500000", meth["prompt"])
        self.assertNotIn("[H&E]", meth["prompt"])
        self.assertNotIn("P1", meth["prompt"])
        self.assertNotIn("ERMS\n", meth["prompt"].split("Options:")[0])
        img = make_subtype_record("P1", "ARMS", probes, values, SUBTYPE_CLASSES, "image", image="x.png")
        self.assertEqual(img["panel_id"], panel_fingerprint([]))
        self.assertNotIn("cg1", img["prompt"])
        both = make_subtype_record("P1", "ARMS", probes, values, SUBTYPE_CLASSES, "multimodal", image="x.png")
        self.assertIn("[H&E]", both["prompt"])
        self.assertIn("[DNA methylation]", both["prompt"])
        msgs = format_subtype_example(both)["messages"]
        self.assertEqual([c["type"] for c in msgs[0]["content"]], ["image", "text"])
        self.assertEqual(msgs[1]["content"][0]["text"], "ARMS")
        self.assertEqual([c["type"] for c in format_subtype_example(meth)["messages"][0]["content"]], ["text"])
        with self.assertRaises(ValueError):
            make_subtype_record("P1", "ARMS", probes, values, SUBTYPE_CLASSES, "image")
        with self.assertRaises(ValueError):
            make_subtype_record("P1", "PRMS", probes, values, SUBTYPE_CLASSES, "methylation")

    def test_parse_and_scores(self):
        self.assertEqual(lenient_parse("The features favour alveolar rhabdomyosarcoma (ARMS).", SUBTYPE_CLASSES), "ARMS")
        self.assertEqual(lenient_parse("Embryonal, not alveolar", SUBTYPE_CLASSES), "ERMS")
        self.assertEqual(lenient_parse("I cannot tell.", SUBTYPE_CLASSES), "<unparsed>")
        self.assertEqual(positive_score({"score_ERMS": "0.3"}, "ARMS", SUBTYPE_CLASSES), 0.7)
        self.assertEqual(positive_score({"score_ARMS": "0.9", "score_ERMS": "0.1"}, "ARMS", SUBTYPE_CLASSES), 0.9)

    def test_allocation(self):
        self.assertEqual(allocate(10, [0.8, 0.2, 0.8]), [5, 1, 4])  # stable tie-break favours the earlier site
        self.assertEqual(sum(allocate(7, [1, 1, 1])), 7)
        rows = [dict(patient_id=f"E{i}", diagnosis="ERMS") for i in range(30)] + [dict(patient_id=f"A{i}", diagnosis="ARMS") for i in range(30)]
        sites = ["site-1", "site-2", "site-3"]
        skew = assign_sites(rows, sites, "skew", random.Random(0), 0.9)
        site2_arms = sum(1 for r in rows if r["diagnosis"] == "ARMS" and skew[r["patient_id"]] == "site-2")
        self.assertGreater(site2_arms, 20)
        iid = assign_sites(rows, sites, "iid", random.Random(0), 0.9)
        per_site = [sum(1 for r in rows if iid[r["patient_id"]] == s) for s in sites]
        self.assertEqual(max(per_site) - min(per_site) <= 2, True)


class PipelineTests(unittest.TestCase):
    def test_audit_and_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_inputs(root)
            mapping = read_mapping(root / "map.csv")
            self.assertEqual(mapping[0]["patient_id"], "SJRHB000")
            report = audit(argparse.Namespace(mapping=root / "map.csv", comet_matrix=root / "beta.tsv",
                                              comet_metadata=root / "meta.txt", slide_dir=None,
                                              slide_list=root / "slides.txt", output=root / "audit"))
            status = report["sample_status"]
            self.assertEqual(status["subtype_not_in_task"], 1)
            self.assertEqual(status["metadata_qc_failed"], 1)
            self.assertEqual(status["sentrix_not_in_matrix"], 1)
            self.assertEqual(status["additional_sample_same_patient"], 1)
            self.assertEqual(report["shared_slides"], ["8135"])
            self.assertEqual(report["slides_without_mapping_row"], ["6217"])
            self.assertEqual(report["candidate_patients_by_subtype"], {"ERMS": 7, "ARMS": 9})
            rows = {r["comet_sample_id"]: r for r in read_tsv(root / "audit/samples_audit.tsv")}
            # Patient 0: the duplicate array with a slide is preferred over the slide-less one.
            self.assertEqual(rows["RMS_dup"]["status"], "candidate_pending_label_review")
            self.assertEqual(rows["RMS_0"]["status"], "additional_sample_same_patient")
            self.assertEqual(rows["RMS_sh1"]["slide_status"], "slide_shared_by_multiple_samples")
            self.assertEqual(rows["RMS_1"]["slide_status"], "available")
            self.assertEqual(rows["RMS_3"]["slide_status"], "no_slide_id")
            with self.assertRaises(ValueError):
                prepare(argparse.Namespace(audit_dir=root / "audit", output=root / "prep", confirm_reviewed_labels=False,
                                           qc_mode="lenient", top_fraction=0.1, require_slide=False, exclude_probes=None,
                                           max_sample_missing=0.2, seed=1, rscript="Rscript", r_library=None))

            prepared = root / "prepared"
            splits = fake_prepared(prepared, root / "audit")
            montages = root / "montages"
            montages.mkdir()
            for r in splits.values():
                if r["slide_status"] == "available":
                    Image.new("RGB", (32, 32), "white").save(montages / f"{r['slide_id']}.png")

            def run(modality, partition, n_clients, name):
                return export(argparse.Namespace(prepared=prepared, output=root / name, modality=modality, partition=partition,
                                                 n_clients=n_clients, dominant_fraction=0.8, montage_dir=montages, seed=3))

            meth = run("methylation", "iid", 3, "meth_iid")
            task = json.loads((meth / "task.json").read_text())
            self.assertEqual(task["panel_mode"], "shared")
            self.assertEqual(task["training_regime"], "simulated_federated")
            validate_clients(meth / "clients", 3)
            train = [r for i in (1, 2, 3) for r in json.loads((meth / f"clients/site-{i}/train.json").read_text())]
            test = json.loads((meth / "test.json").read_text())
            self.assertEqual(len(train), sum(1 for r in splits.values() if r["split"] == "train"))
            self.assertTrue(all(r["image"] is None for r in train))
            self.assertFalse({r["patient_id"] for r in train} & {r["patient_id"] for r in test})
            self.assertTrue(all("site_id" in r for r in test))

            central = run("image", "iid", 1, "image_central")
            task = json.loads((central / "task.json").read_text())
            self.assertEqual(task["panel_mode"], "image")
            self.assertEqual(task["training_regime"], "centralized")
            validate_clients(central / "clients", 1)
            train = json.loads((central / "clients/site-1/train.json").read_text())
            self.assertTrue(all(r["image"] and (central / r["image"]).is_file() for r in train))
            self.assertTrue(all("cg" not in r["prompt"] for r in train))
            self.assertEqual(sum(task["dropped_for_modality"].values()),
                             sum(1 for r in splits.values() if r["slide_status"] != "available"))

            multi = run("multimodal", "skew", 3, "multi_skew")
            validate_clients(multi / "clients", 3)
            task = json.loads((multi / "task.json").read_text())
            self.assertEqual(task["dominant_fraction"], 0.8)
            rec = json.loads((multi / "validation.json").read_text())[0]
            self.assertIn("[H&E]", rec["prompt"])
            self.assertIn("[DNA methylation]", rec["prompt"])
            with self.assertRaises(FileExistsError):
                run("multimodal", "skew", 3, "multi_skew")

    @unittest.skipUnless(os.environ.get("RUN_R_INTEGRATION"), "set RUN_R_INTEGRATION=1 to run limma")
    def test_prepare_with_r(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic_inputs(root)
            audit(argparse.Namespace(mapping=root / "map.csv", comet_matrix=root / "beta.tsv", comet_metadata=root / "meta.txt",
                                     slide_dir=None, slide_list=root / "slides.txt", output=root / "audit"))
            for mode in ("lenient", "strict"):
                prepare(argparse.Namespace(audit_dir=root / "audit", output=root / f"prep_{mode}", confirm_reviewed_labels=True,
                                           qc_mode=mode, top_fraction=0.05, require_slide=False, exclude_probes=None,
                                           max_sample_missing=0.2, seed=1, rscript=os.environ.get("RSCRIPT", "Rscript"),
                                           r_library=os.environ.get("R_LIBRARY")))
                summary = json.loads((root / f"prep_{mode}/selection_summary.json").read_text())
                spec = json.loads((root / f"prep_{mode}/features.json").read_text())
                self.assertEqual(spec["qc_mode"], mode)
                self.assertEqual(len(spec["probes"]), summary["selected_cpgs"])
                self.assertEqual(summary["selected_cpgs"], -(-summary["tested_nonconstant_cpgs"] // 20))
                # The first 40 probes separate ARMS from ERMS by construction.
                self.assertTrue(all(int(p[2:]) < 40 for p in spec["probes"]))
                splits = read_tsv(root / f"prep_{mode}/splits.tsv")
                self.assertTrue({"slide_id", "slide_status"} <= set(splits[0]))
            lenient = json.loads((root / "prep_lenient/selection_summary.json").read_text())
            strict = json.loads((root / "prep_strict/selection_summary.json").read_text())
            self.assertGreaterEqual(lenient["passing_training_probe_qc"], strict["passing_training_probe_qc"])


if __name__ == "__main__":
    unittest.main()
