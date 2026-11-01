import csv
import json
from pathlib import Path
import random
import tempfile
import unittest

from fusion_identity import validate_patient_rows
from fusion_threshold import apply, choose_threshold, fit, metrics
from prepare_comet_subtype import assign_sites, write_tsv
from scripts.check_fusion_splits import inspect, inspect_clients


class PatientClientsTests(unittest.TestCase):
    def test_repeated_specimens_stay_together_and_order_is_irrelevant(self):
        rows = [dict(patient_id=f'{label}{i}-a', canonical_uid=f'{label}{i}', diagnosis=label)
                for label in ('NEG', 'POS') for i in range(12)]
        rows += [dict(rows[0], patient_id='repeat-a'), dict(rows[0], patient_id='repeat-b')]
        sites = ['site-1', 'site-2', 'site-3']
        for partition in ('iid', 'skew'):
            assigned = assign_sites(rows, sites, partition, random.Random(7), 0.8)
            self.assertEqual(assigned, assign_sites(list(reversed(rows)), sites, partition, random.Random(7), 0.8))
            self.assertEqual(len({assigned[r['patient_id']] for r in rows if r['canonical_uid'] == 'NEG0'}), 1)
            # Adding specimens must not change other patients' site assignments.
            singles = assign_sites(rows[:-2], sites, partition, random.Random(7), 0.8)
            self.assertTrue(all(assigned[k] == v for k, v in singles.items()))
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            assign_sites(rows + [dict(rows[0], patient_id='conflict', diagnosis='POS')], sites, 'iid', random.Random(7), 0.8)
        with self.assertRaisesRegex(ValueError, 'Missing canonical_uid'):
            assign_sites([{'patient_id': 'sample-not-patient', 'diagnosis': 'NEG'}], sites, 'iid', random.Random(7), 0.8)

    def test_split_overlap_and_conflicting_labels_rejected(self):
        row = dict(patient_id='sample1', canonical_uid='patient1', diagnosis='POS', split='train')
        for changed in ({'split': 'test'}, {'diagnosis': 'NEG'}):
            with self.assertRaisesRegex(ValueError, 'cross-split|conflicting'):
                validate_patient_rows([row, dict(row, patient_id='sample2', **changed)])

    def test_guard_checks_actual_client_json(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            rows = [dict(patient_id=f'{s}{i}', canonical_uid=f'{s}patient', diagnosis='NEG', split=s,
                         site_id='site-1') for s in ('train', 'validation', 'test') for i in range(2)]
            write_tsv(root / 'splits.tsv', rows)
            write_tsv(root / 'site_assignments.tsv', rows)
            (root / 'task.json').write_text(json.dumps({'sites': ['site-1', 'site-2'], 'n_clients': 2}))
            for site in ('site-1', 'site-2'):
                (root / 'clients' / site).mkdir(parents=True)
            for split in ('train', 'validation', 'test'):
                selected = [r for r in rows if r['split'] == split]
                with (root / f'{split}.csv').open('w', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['patient_id', 'diagnosis'])
                    w.writeheader(); w.writerows({k: r[k] for k in w.fieldnames} for r in selected)
                records = [dict(r, label_name=r['diagnosis']) for r in selected]
                if split != 'train':
                    (root / f'{split}.json').write_text(json.dumps(records))
                if split != 'test':
                    (root / f'clients/site-1/{split}.json').write_text(json.dumps(records))
                    (root / f'clients/site-2/{split}.json').write_text('[]')
            signature = inspect(root)
            self.assertEqual(inspect_clients(root, signature), 3)
            # Intended assignment says site-1; move a specimen in actual JSON.
            path = root / 'clients/site-1/train.json'
            records = json.loads(path.read_text())
            moved = records.pop()
            path.write_text(json.dumps(records))
            (root / 'clients/site-2/train.json').write_text(json.dumps([moved]))
            with self.assertRaisesRegex(ValueError, 'placement'):
                inspect_clients(root, signature)
            # Also reject a manifest that itself splits the two specimens.
            rows[1]['site_id'] = 'site-2'
            write_tsv(root / 'site_assignments.tsv', rows)
            with self.assertRaisesRegex(ValueError, 'multiple clients'):
                inspect_clients(root, signature)


def threshold_fixture(root):
    data = root / 'data'; data.mkdir()
    rows = []
    for split in ('validation', 'test'):
        # POS patient has two records, to test patient aggregation explicitly.
        for i, (uid, truth, score) in enumerate([('n', 'NEG', 0.80), ('p', 'POS', 0.90), ('p', 'POS', 0.94)]):
            rows.append(dict(patient_id=f'{split}{i}', canonical_uid=split + uid,
                             split=split, diagnosis=truth, score_POS=score))
    write_tsv(data / 'splits.tsv', rows)
    (data / 'task.json').write_text(json.dumps({'classes': ['NEG', 'POS'], 'panel_id': 'fixed'}))
    for split in ('validation', 'test'):
        selected = [r for r in rows if r['split'] == split]
        (data / f'{split}.json').write_text(json.dumps([
            {'patient_id': r['patient_id'], 'label_name': r['diagnosis']} for r in selected]))
        out = root / split; out.mkdir()
        (out / 'evaluation.json').write_text(json.dumps({
            'mode': 'medgemma', 'split': split, 'panel_id': 'fixed', 'modality': 'multimodal',
            'arguments': {'data_dir': str(data), 'model_path': 'run/model.pt', 'base_model': 'base', 'parse': 'strict'}}))
        with (out / 'predictions.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['patient_id', 'truth', 'score_POS'])
            w.writeheader(); w.writerows({'patient_id': r['patient_id'], 'truth': r['diagnosis'], 'score_POS': r['score_POS']}
                                        for r in selected)
    return data


class ThresholdTests(unittest.TestCase):
    def test_extreme_scores_ties_and_nonfinite_rejection(self):
        y, p = ['NEG', 'NEG', 'POS', 'POS'], [1e-30, 2e-30, 4e-30, 6e-30]
        t = choose_threshold(y, p)
        self.assertGreater(t, 2e-30); self.assertLessEqual(t, 4e-30)
        self.assertEqual(metrics(y, p, t)['balanced_accuracy'], 1.0)
        self.assertEqual(metrics(y, p, t)['roc_auc'], metrics(y, p, 0.5)['roc_auc'])
        self.assertEqual(choose_threshold(['NEG', 'POS'], [0.9, 0.9]), 0.5)
        for bad in (float('nan'), float('inf'), -0.1):
            with self.assertRaises(ValueError):
                choose_threshold(['NEG', 'POS'], [0.2, bad])

    def test_fit_validation_apply_test_and_provenance(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = threshold_fixture(root)
            artifact = fit(root / 'validation', data)
            self.assertEqual(artifact['validation_provenance']['n_records'], 3)
            self.assertEqual(artifact['validation_provenance']['n_patients'], 2)
            self.assertAlmostEqual(artifact['threshold'], 0.86)
            self.assertIsNone(artifact['validation_loo_balanced_accuracy'])
            result = apply(root / 'test', data, artifact)
            self.assertEqual(result['default_0_5']['balanced_accuracy'], 0.5)
            self.assertEqual(result['validation_selected_threshold']['balanced_accuracy'], 1.0)
            with self.assertRaisesRegex(ValueError, 'never fit'):
                fit(root / 'test', data)
            for field in ('model_path', 'max_seq_length'):
                path = root / 'test/evaluation.json'
                report = json.loads(path.read_text())
                before = path.read_text()
                report['arguments'][field] = 'changed'
                path.write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'different model'):
                    apply(root / 'test', data, artifact)
                path.write_text(before)
            changed = dict(artifact, fitted_split='test')
            with self.assertRaisesRegex(ValueError, 'Invalid validation'):
                apply(root / 'test', data, changed)
            # Fitting is independent of test labels/scores, even when they change.
            path = root / 'test/predictions.csv'
            path.write_text(path.read_text().replace('0.8', '0.99'))
            self.assertEqual(fit(root / 'validation', data), artifact)


if __name__ == '__main__':
    unittest.main()
