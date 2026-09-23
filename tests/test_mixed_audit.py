import csv
from pathlib import Path
import tempfile
import unittest

from audit_mixed_methylation import audit, identify
from methylation_manifest import COHORTS


class MixedAuditTests(unittest.TestCase):
    def test_identifiers(self):
        self.assertEqual(identify('TARGET.30.PAT1.01A.01D'), ('TARGET-30-PAT1', '01'))
        self.assertEqual(identify('SJMPAL001'), ('', ''))

    def test_audit_conflicts_and_unknowns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for platform, cohorts in COHORTS.items():
                for cohort in cohorts:
                    samples = [f'TARGET-01-{cohort.replace("-", "")}-01A']
                    if cohort in ('AML', 'ALL-P3'):
                        samples.append('TARGET-01-SHARED-01A')
                    if cohort == 'ALL-P3':
                        samples.append('SJMPAL001')
                    suffix = '450k' if platform == '450k' else 'EPIC'
                    listing = 'all.samples' if cohort == 'ALL-P3' else cohort.lower()+'.samples'
                    for name in (f'TARGET-{cohort}__{suffix}_beta.txt', listing):
                        with (root/name).open('w') as f:
                            csv.writer(f, delimiter='\t').writerow(['TargetID']+[s+'.Ave_Beta' for s in samples])
            result = audit(root, root/'audit')
            self.assertEqual(result['cross_platform_patient_ids'], ['TARGET-01-SHARED'])
            self.assertEqual(result['cross_cohort_patient_ids'], ['TARGET-01-SHARED'])
            self.assertEqual(result['sample_status']['unresolved_patient_identifier'], 1)
            self.assertEqual(result['candidate_unique_patients_by_diagnosis']['ALL'], 1)
            self.assertFalse(result['independently_clinically_verified'])
            with self.assertRaises(FileExistsError):
                audit(root, root/'audit')


if __name__ == '__main__':
    unittest.main()
