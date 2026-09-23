import csv
import json
from pathlib import Path
import tempfile
import unittest

from methylation_manifest import COHORTS
from prepare_cohort_labels import prepare


class CohortLabelTests(unittest.TestCase):
    def test_confirmation_and_exact_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):
                prepare(root,root/"out")
            for c, dx in enumerate(COHORTS["450k"]):
                samples=[f"TARGET-01-P{c}{i}-01A" for i in range(6)]
                cols=[s+".Ave_Beta" for s in samples]
                for file, data in ((root/f"{dx.lower()}.samples", cols[:5]),
                                   (root/f"TARGET-{dx}__450k_beta.txt",cols)):
                    with file.open("w") as f:
                        csv.writer(f,delimiter="\t").writerow(["TargetID"]+data)
            prepare(root,root/"out",True)
            audit=json.loads((root/"out/audit.json").read_text())
            self.assertEqual(audit["sample_status"]["not_in_corresponding_sample_list"],5)
            self.assertEqual(audit["included_patients_by_diagnosis"],dict.fromkeys(COHORTS["450k"],5))
            self.assertFalse(json.loads((root/"out/label_provenance.json").read_text())["independently_clinically_verified_by_software"])


if __name__ == "__main__":
    unittest.main()
