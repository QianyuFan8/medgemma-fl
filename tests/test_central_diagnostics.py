import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
from evaluate_subtype import evaluation_reference, patient_score_metrics
from methylation_utils import panel_fingerprint
from prepare_comet_subtype import write_tsv
from prepare_central_diagnostics import make_plan, prepare, select_tiny_records
from run_central_diagnostics import train_command, verify_training
from scripts.check_fusion_splits import inspect, inspect_clients
from test_training_args import load_builder
from types import SimpleNamespace


def fixture(root):
    prepared, montages = root / 'prepared', root / 'montages'
    prepared.mkdir(); montages.mkdir()
    rows = []
    for split, n in [('train', 5), ('validation', 1), ('test', 1)]:
        for label in ('NEG', 'POS'):
            for i in range(n):
                rid = f'{split}-{label}-{i}'
                rows.append(dict(patient_id=rid, canonical_uid=rid, diagnosis=label, split=split,
                                 qc_pass='TRUE', slide_id=rid, slide_status='available'))
    rows.append(dict(rows[0], patient_id='another-specimen'))
    probes = ['cg1', 'cg2']
    write_tsv(prepared / 'splits.tsv', rows)
    write_tsv(prepared / 'features.tsv', [{'probe_id': p} for p in probes])
    (prepared / 'features.json').write_text(json.dumps({
        'classes': ['NEG', 'POS'], 'probes': probes, 'panel_id': panel_fingerprint(probes),
        'features_sha256': hashlib.sha256((prepared / 'features.tsv').read_bytes()).hexdigest(), 'selection': {}}))
    for split in ('train', 'validation', 'test'):
        with (prepared / f'{split}.csv').open('w', newline='') as f:
            w = csv.writer(f); w.writerow(['patient_id', 'diagnosis'] + probes)
            for r in rows:
                if r['split'] == split:
                    w.writerow([r['patient_id'], r['diagnosis'], '0.100000', '0.800000'])
    for name in ('FEATURES.md', 'selected_cpgs.tsv', 'selection_summary.json', 'preprocessing.rds',
                 'config.json', 'cohort_provenance.json', 'R_sessionInfo.txt'):
        (prepared / name).write_text('{}')
    for row in rows:
        Image.new('RGB', (32, 32), 'white').save(montages / (row['slide_id'] + '.png'))
    return prepared, montages, rows


class DiagnosticsTests(unittest.TestCase):
    def test_subset_is_balanced_unique_training_only_and_deterministic(self):
        with tempfile.TemporaryDirectory() as td:
            _, _, rows = fixture(Path(td))
            ids = select_tiny_records(rows)
            self.assertEqual(ids, select_tiny_records(list(reversed(rows))))
            selected = [r for r in rows if r['patient_id'] in ids]
            self.assertEqual(len(selected), 8)
            self.assertEqual(len({r['canonical_uid'] for r in selected}), 8)
            self.assertEqual({r['split'] for r in selected}, {'train'})
            self.assertEqual(sum(r['diagnosis'] == 'POS' for r in selected), 4)

    def test_prepared_modalities_retain_patients_betas_and_training_reference(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source, montages, rows = fixture(root)
            plan = prepare(source, montages, root / 'out', 'top16')
            for mode in ('multimodal', 'image', 'methylation'):
                data = root / 'out' / mode
                self.assertEqual(inspect(data), inspect(source))
                inspect_clients(data, inspect(data))
                task = json.loads((data / 'task.json').read_text())
                records = evaluation_reference(data, 'train', task)
                self.assertEqual(len(records), 11)
                self.assertTrue(all(bool(r['image']) == (mode != 'methylation') for r in records))
                self.assertTrue(all(('[DNA methylation]' in r['prompt']) == (mode != 'image') for r in records))
                self.assertEqual((data / 'train.csv').read_bytes(), (source / 'train.csv').read_bytes())
            tiny = root / 'out/tiny'
            self.assertEqual(sum(v[1] == 'train' for v in inspect(tiny).values()), 8)
            self.assertEqual((tiny / 'validation.csv').read_bytes(), (source / 'validation.csv').read_bytes())
            self.assertFalse(plan['test_evaluation_enabled'])
            with self.assertRaises(FileExistsError):
                prepare(source, montages, root / 'out', 'top16')

    def test_patient_metrics_do_not_double_weight_repeated_specimens(self):
        with tempfile.TemporaryDirectory() as td:
            source, _, _ = fixture(Path(td))
            rows = [dict(patient_id='train-NEG-0', truth='NEG', score_POS=0.1),
                    dict(patient_id='another-specimen', truth='NEG', score_POS=0.3),
                    dict(patient_id='train-POS-0', truth='POS', score_POS=0.9)]
            result = patient_score_metrics(rows, source)
            self.assertEqual(result['n_patients'], 2)
            self.assertEqual(result['n_records'], 3)
            self.assertEqual(result['balanced_accuracy'], 1.0)

    def test_plan_uses_one_continuous_round_fixed_modality_lr_and_no_test(self):
        plan = make_plan()
        self.assertEqual(len(plan), 5)
        for ex in plan:
            self.assertNotIn('test', ex['evaluation_splits'])
            cmd = train_command(Path('data'), Path('run'), ex, 42)
            self.assertEqual(cmd[cmd.index('--num_rounds') + 1], '1')
            self.assertEqual(cmd[cmd.index('--seed') + 1], '42')
        self.assertEqual({ex['learning_rate'] for ex in plan[2:]}, {5e-5})
        self.assertEqual({ex['epochs'] for ex in plan[1:]}, {10})
        smoke = train_command(Path('data'), Path('run'), plan[0], 42, smoke=True)
        self.assertNotIn('--num_train_epochs', smoke)
        self.assertEqual(smoke[smoke.index('--max_steps') + 1], '1')

    def test_explicit_seed_reaches_trainer(self):
        args = SimpleNamespace(task='subtype', num_train_epochs=10, per_device_train_batch_size=1,
                               per_device_eval_batch_size=1, gradient_accumulation_steps=4,
                               logging_steps=5, learning_rate=5e-5, report_to='none', max_steps=None,
                               max_seq_length=32768, eval_steps=10, seed=42)
        result = load_builder()(args, 'unused', True)
        self.assertEqual((result['seed'], result['data_seed']), (42, 42))

    def test_initial_checkpoint_does_not_count_as_completed_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'FL_global_model.pt').write_text('initial weights')
            with self.assertRaises(RuntimeError):
                verify_training(root, 10, False)
            path = root / 'training_diagnostics.json'
            path.write_text(json.dumps({'global_step': 1, 'epoch': 0.1, 'training_loss': 0.2}))
            with self.assertRaises(RuntimeError):
                verify_training(root, 10, False)
            self.assertEqual(verify_training(root, 10, True)['global_step'], 1)


if __name__ == '__main__':
    unittest.main()
