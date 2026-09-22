import csv
import json
from pathlib import Path
import tempfile
import unittest

from methylation_manifest import build_manifest, COHORTS


class ManifestTests(unittest.TestCase):
    def test_metadata_only_and_missing_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for c, cohort in enumerate(COHORTS["450k"]):
                samples = [f"TARGET-01-P{c}{i}-01A" for i in range(6)]
                with (root / f"TARGET-{cohort}__450k_beta.txt").open("w") as f:
                    csv.writer(f, delimiter="\t").writerow(["TargetID"]+[s+".Ave_Beta" for s in samples])
                # Last patient absent: it must NOT inherit the filename's diagnosis.
                entries.extend((s.replace("-", "."), cohort) for s in samples[:5])
            meta = root / "meta.tsv"
            with meta.open("w") as f:
                w=csv.writer(f, delimiter="\t"); w.writerow(["Sample","Dx"]); w.writerows(entries)
            build_manifest(meta,root,"450k",root/"out")
            audit=json.loads((root/"out/audit.json").read_text())
            self.assertEqual(audit["sample_status"]["unmatched_diagnosis"],5)
            self.assertEqual(audit["included_patients_by_diagnosis"],dict.fromkeys(COHORTS["450k"],5))
            cfg=json.loads((root/"out/config.json").read_text())
            self.assertFalse(cfg["metadata_reviewed"])
            # Conflicting metadata diagnoses must not be resolved by source filename.
            with meta.open("a") as f:
                csv.writer(f, delimiter="\t").writerow([entries[0][0],"WT"])
            build_manifest(meta,root,"450k",root/"conflict")
            audit=json.loads((root/"conflict/audit.json").read_text())
            self.assertEqual(audit["sample_status"]["conflicting_diagnosis"],1)
            self.assertNotIn("AML",audit["included_patients_by_diagnosis"])


if __name__ == "__main__":
    unittest.main()
