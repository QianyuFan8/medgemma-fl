import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from prepare_multiimage_diagnostics import convert_record, prepare, signature
from run_multiimage_diagnostics import training_command, main
from subtype_utils import make_subtype_record, format_subtype_example, load_example_images


def fixture(root):
    root.mkdir(parents=True)
    (root / 'images').mkdir()
    montage = Image.new('RGB', (896, 896))
    for i in range(16):
        montage.paste((i * 15, 255 - i * 15, i), (i % 4 * 224, i // 4 * 224, (i % 4 + 1) * 224, (i // 4 + 1) * 224))
    montage.save(root / 'images/fixture.png')
    site = root / 'clients/site-1'; site.mkdir(parents=True)
    records, manifest = {}, []
    for split in ('train', 'validation', 'test'):
        records[split] = []
        for label in ('NEG', 'POS'):
            rid = split + '_' + label
            row = make_subtype_record(rid, label, [], [], ['NEG', 'POS'], 'image', 'images/fixture.png', 'site-1')
            row['canonical_uid'] = 'UID_' + rid
            records[split].append(row)
            manifest.append(dict(patient_id=rid, canonical_uid=row['canonical_uid'], diagnosis=label, split=split, site_id='site-1'))
        path = site / f'{split}.json' if split == 'train' else root / f'{split}.json'
        path.write_text(json.dumps(records[split]))
        if split == 'validation':
            (site / 'validation.json').write_bytes(path.read_bytes())
        with (root / f'{split}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id', 'diagnosis'])
            writer.writeheader()
            writer.writerows({'patient_id': r['patient_id'], 'diagnosis': r['label_name']} for r in records[split])
    for filename in ('splits.tsv', 'site_assignments.tsv'):
        with (root / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0]), delimiter='\t')
            writer.writeheader(); writer.writerows(manifest)
    task = dict(task='comet_fusion', modality='image', panel_mode='image', classes=['NEG', 'POS'],
                sites=['site-1'], n_clients=1, probes=[], panel_id=records['train'][0]['panel_id'])
    (root / 'task.json').write_text(json.dumps(task))
    return records


class MultiimageTests(unittest.TestCase):
    def test_exact_pixels_order_and_training_inference_image_loading_match(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'source'; rows = fixture(root)
            row = convert_record(rows['train'][0])
            from_path = load_example_images(row, root)
            with Image.open(root / row['image']) as source:
                from_pil = load_example_images({**row, 'image': source.copy()})
                rebuilt = Image.new('RGB', source.size)
                for i, (a, b) in enumerate(zip(from_path, from_pil)):
                    self.assertEqual(a.size, (224, 224))
                    np.testing.assert_array_equal(a, b)
                    self.assertEqual(a.getpixel((0, 0)), (i * 15, 255 - i * 15, i))
                    rebuilt.paste(a, (i % 4 * 224, i // 4 * 224))
                np.testing.assert_array_equal(rebuilt, source)
            self.assertEqual(len(from_path), 16)
            msgs = format_subtype_example(row)['messages']
            self.assertEqual(sum(c['type'] == 'image' for c in msgs[0]['content']), 16)
            self.assertEqual(msgs[1]['content'], [{'type': 'text', 'text': 'NEG'}])
            self.assertEqual(len(load_example_images(rows['train'][0], root)), 1)
            with self.assertRaises(ValueError):
                load_example_images({**row, 'image': Image.new('RGB', (224, 224))})

    def test_export_retains_records_labels_splits_and_source_pixels_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); source = root / 'source'; rows = fixture(source)
            original = (source / 'clients/site-1/train.json').read_bytes()
            output = root / 'output'; prepare(source, output)
            self.assertEqual(signature(source), signature(output))
            converted = json.loads((output / 'clients/site-1/train.json').read_text())
            self.assertEqual(len(converted), len(rows['train']))
            self.assertEqual([r['patient_id'] for r in converted], [r['patient_id'] for r in rows['train']])
            self.assertEqual([r['label_name'] for r in converted], ['NEG', 'POS'])
            self.assertTrue(all(r['image_layout'] == 'tiles16' for r in converted))
            self.assertEqual((source / 'clients/site-1/train.json').read_bytes(), original)
            self.assertEqual((output / 'images/fixture.png').read_bytes(), (source / 'images/fixture.png').read_bytes())
            with self.assertRaises(FileExistsError): prepare(source, output)
            with self.assertRaises(ValueError): convert_record({**rows['train'][0], 'modality': 'multimodal'})

    def test_update_budget_and_checkpoint_arguments(self):
        cmd = training_command(Path('data'), Path('work'), 42)
        for key, expected in [('--num_train_epochs', '10'), ('--num_rounds', '1'), ('--n_clients', '1'),
                              ('--adapter_scope', 'language-only'), ('--learning_rate', '5e-05'),
                              ('--per_device_train_batch_size', '1'), ('--gradient_accumulation_steps', '4')]:
            self.assertEqual(cmd[cmd.index(key) + 1], expected)
        self.assertEqual(cmd[cmd.index('--save_adapter_epochs') + 1:], ['2', '5', '10'])
        smoke = training_command(Path('data'), Path('smoke'), 42, smoke=True)
        self.assertNotIn('--save_adapter_epochs', smoke)
        self.assertEqual(smoke[smoke.index('--max_steps') + 1], '1')

    def test_queue_smoke_is_separate_and_scores_only_six_train_validation_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / 'data'; fixture(data / 'image')
            (data / 'diagnostics_plan.json').write_text(json.dumps({'arm': 'top16', 'training_seed': 42, 'test_evaluation_enabled': False}))
            output = root / 'run'; calls = []
            language = 'model.language_model.layers.0.q_proj.lora_A.weight'
            def stage(cmd, log, name, *rest):
                calls.append(cmd)
                if 'job.py' in cmd:
                    smoke = '--max_steps' in cmd
                    ws = Path(cmd[cmd.index('--workspace') + 1]); ws.mkdir(parents=True)
                    (ws / 'training_diagnostics.json').write_text(json.dumps({'global_step': 1 if smoke else 10, 'epoch': 1 if smoke else 10, 'training_loss': .1}))
                    (ws / 'trainable_parameters.json').write_text(json.dumps({'adapter_scope': 'language-only', 'trainable_parameters': 16, 'vision_trainable_parameters': 0, 'trainable_tensors': [language]}))
                    if not smoke:
                        folder = ws / 'epoch_checkpoints'; folder.mkdir()
                        for epoch in (2, 5, 10):
                            cp = folder / f'epoch-{epoch:02d}.pt'; cp.write_bytes(b'fixture')
                            cp.with_suffix('.json').write_text(json.dumps({'epoch': epoch, 'planned_epochs': 10, 'global_step': epoch, 'checkpoint_sha256': hashlib.sha256(cp.read_bytes()).hexdigest()}))
                elif 'evaluate_subtype.py' in cmd:
                    self.assertIn('--skip-generation', cmd)
                    self.assertIn(cmd[cmd.index('--split') + 1], ('train', 'validation'))
                    dest = Path(cmd[cmd.index('--output') + 1]); dest.mkdir()
                    (dest / 'evaluation.json').write_text(json.dumps({'metrics': {'patient_level_score_0_5': {'n_patients': 2, 'accuracy': .5, 'balanced_accuracy': .5, 'roc_auc': .5, 'pr_auc': .5, 'recall': {'NEG': 0, 'POS': 1}}}}))
                else:
                    Path(cmd[cmd.index('--output') + 1]).write_text('{}')
            with patch('sys.argv', ['runner', '--data-root', str(data), '--output', str(output), '--baseline', str(root / 'baseline')]), \
                 patch('run_multiimage_diagnostics.check_baseline', return_value={'snapshot_steps': {'2': 2, '5': 5, '10': 10}}), \
                 patch('run_multiimage_diagnostics.importlib.metadata.version', return_value='fixture'), \
                 patch('run_multiimage_diagnostics.subprocess.check_output', return_value='fixture'), \
                 patch('run_multiimage_diagnostics.processor_preflight'), \
                 patch('run_multiimage_diagnostics.run_stage', side_effect=stage):
                main()
                with self.assertRaises(FileExistsError): main()
            jobs = [c for c in calls if 'job.py' in c]
            self.assertEqual(len(jobs), 2)
            self.assertNotEqual(jobs[0][jobs[0].index('--workspace') + 1], jobs[1][jobs[1].index('--workspace') + 1])
            self.assertEqual(len([c for c in calls if 'evaluate_subtype.py' in c]), 6)
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
