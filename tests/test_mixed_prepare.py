import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare_mixed_methylation import CLASSES, allocate, export
from methylation_validate import validate_clients


class MixedPreparationTests(unittest.TestCase):
    def test_allocation_and_deduplication(self):
        rows = []
        for dx in CLASSES:
            for i in range(12):
                rows.append(dict(patient_id=f'{dx}{i}', sample_id=f'{dx}{i}-01A',
                                 sample_type='01', proposed_diagnosis=dx,
                                 platform='epic' if dx in ('ALL', 'RT') else '450k',
                                 matrix_file=dx, status='candidate_pending_label_review'))
        rows.append(dict(rows[0], sample_id='duplicate-03A', sample_type='03'))
        sites, duplicates = allocate(rows, 42)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(sites['site-1']), 30)
        self.assertEqual(len(sites['site-3']), 24)
        self.assertEqual(allocate(rows, 42), (sites, duplicates))

    def test_local_panels_and_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for n in range(1, 4):
                local = root/'local'/f'site-{n}'
                local.mkdir(parents=True)
                classes = ['ALL', 'RT'] if n == 3 else ['AML', 'CCSK', 'NBL', 'OS', 'WT']
                for split in ('train', 'validation', 'test'):
                    with (local/f'{split}.csv').open('w') as f:
                        w = csv.writer(f)
                        w.writerow(['patient_id', 'diagnosis', f'cg{n:08d}'])
                        for dx in classes:
                            w.writerow([f'{n}-{split}-{dx}', dx, .3])
            export(root, 42)
            task = validate_clients(root/'clients', 3)
            self.assertEqual(task['classes'], CLASSES)
            self.assertEqual(len({p['panel_id'] for p in task['site_panels'].values()}), 3)
            file = root/'clients/site-1/train.json'
            rows = json.loads(file.read_text())
            self.assertEqual(rows[0]['classes'], CLASSES)
            rows[0]['panel_id'] = task['site_panels']['site-2']['panel_id']
            file.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'panel'):
                validate_clients(root/'clients', 3)


if __name__ == '__main__':
    unittest.main()
