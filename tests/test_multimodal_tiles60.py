"""CPU regression checks for the 60-epoch multimodal comparison."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_multiimage_diagnostics import fixture
from check_methylation_context import measure_context
from methylation_validate import validate_clients
from prepare_multiimage_diagnostics import convert_record, prepare, signature
from run_multimodal_tiles60 import EPOCHS, SCOPES, main, training_command, verify_scope
from run_epoch_diagnostics import snapshot
from subtype_utils import make_subtype_record, format_subtype_example


def multimodal_fixture(root):
    fixture(root)
    probes = [f'cg{i:08d}' for i in range(849)]
    for path in root.rglob('*.json'):
        if path.name not in ('train.json', 'validation.json', 'test.json'):
            continue
        records = []
        for old in json.loads(path.read_text()):
            row = make_subtype_record(old['patient_id'], old['label_name'], probes,
                                      [i / 849 for i in range(849)], ['NEG', 'POS'],
                                      'multimodal', old['image'], 'site-1')
            row['canonical_uid'] = old['canonical_uid']
            records.append(row)
        path.write_text(json.dumps(records))
    task = json.loads((root / 'task.json').read_text())
    task.update(modality='multimodal', panel_mode='shared', probes=probes,
                panel_id=records[0]['panel_id'])
    features = 'probe\n' + '\n'.join(probes) + '\n'
    digest = hashlib.sha256(features.encode()).hexdigest()
    task['features_sha256'] = digest
    spec = {k: task[k] for k in ('probes', 'panel_id', 'classes', 'features_sha256')}
    for folder in (root, root / 'clients/site-1'):
        (folder / 'features.tsv').write_text(features)
        (folder / 'features.json').write_text(json.dumps(spec))
    (root / 'task.json').write_text(json.dumps(task))
    validate_clients(root / 'clients', 1, require_patient_groups=True)


def write_audit(workspace, scope):
    names = ['model.language_model.layers.0.q_proj.lora_A.weight']
    if scope == 'lora-only':
        names.append('model.vision_tower.layers.0.q_proj.lora_B.weight')
    audit = dict(adapter_scope=scope, trainable_parameters=16 * len(names),
                 vision_trainable_parameters=16 if scope == 'lora-only' else 0,
                 trainable_tensors=names)
    (workspace / 'trainable_parameters.json').write_text(json.dumps(audit))
    return audit


class Multimodal60Tests(unittest.TestCase):
    def test_export_keeps_full_cpg_text_order_pixels_and_patient_splits(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'source'; multimodal_fixture(source)
            output = Path(td) / 'out'
            prepare(source, output, modality='multimodal')
            self.assertEqual(signature(source), signature(output))
            for path in source.rglob('*.json'):
                if path.name not in ('train.json', 'validation.json', 'test.json'):
                    continue
                old = json.loads(path.read_text())
                new = json.loads((output / path.relative_to(source)).read_text())
                self.assertEqual(len(old), len(new))
                for a, b in zip(old, new):
                    self.assertEqual(a['prompt'].split('[DNA methylation]')[1],
                                     b['prompt'].split('[DNA methylation]')[1])
                    self.assertEqual(b['prompt'].count('[DNA methylation]'), 1)
                    self.assertEqual(len([l for l in b['prompt'].splitlines() if l.startswith('cg')]), 849)
                    for key in set(a) - {'prompt', 'prompt_style'}:
                        self.assertEqual(a[key], b[key])
                    messages = format_subtype_example(b)['messages']
                    self.assertEqual(sum(x['type'] == 'image' for x in messages[0]['content']), 16)
                    self.assertEqual(len(messages[1]['content']), 1)
            self.assertEqual((source / 'images/fixture.png').read_bytes(), (output / 'images/fixture.png').read_bytes())
            validate_clients(output / 'clients', 1, require_patient_groups=True)
            with self.assertRaises(FileExistsError): prepare(source, output, modality='multimodal')
            bad = json.loads((source / 'clients/site-1/train.json').read_text())[0]
            with self.assertRaises(ValueError): convert_record(bad)  # image-only callers remain strict
            bad['prompt'] = bad['prompt'].replace('[DNA methylation]', '[removed]')
            with self.assertRaises(ValueError): convert_record(bad, modality='multimodal')

    def test_context_reserves_all_sixteen_images(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'source'; multimodal_fixture(source)
            output = Path(td) / 'out'; prepare(source, output, modality='multimodal')
            processor = SimpleNamespace(apply_chat_template=lambda *a, **k: 'text',
                                        tokenizer=lambda *a, **k: {'input_ids': [1] * 100})
            with patch('check_methylation_context.encode_methylation_example', return_value=([1] * 110, None)):
                a = measure_context(source, processor)
                b = measure_context(output, processor)
            self.assertEqual(b['required_max_seq_length'] - a['required_max_seq_length'], 15 * 280)

    def test_scope_audits_reject_full_weights_and_missing_vision_adapters(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit = write_audit(root, 'lora-only')
            verify_scope(root, 'lora-only')
            audit['trainable_tensors'].append('model.language_model.lm_head.weight')
            (root / 'trainable_parameters.json').write_text(json.dumps(audit))
            with self.assertRaises(RuntimeError): verify_scope(root, 'lora-only')
            write_audit(root, 'language-only')
            verify_scope(root, 'language-only')
            with self.assertRaises(RuntimeError): verify_scope(root, 'lora-only')

    def test_queue_independent_smokes_and_two_full_runs_with_correct_checkpoints(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / 'source'; multimodal_fixture(data / 'multimodal')
            (data / 'diagnostics_plan.json').write_text(json.dumps(dict(arm='top16', training_seed=42, test_evaluation_enabled=False)))
            output = root / 'run'; calls = []
            def stage(cmd, log, name, *rest):
                calls.append(cmd)
                if 'job.py' in cmd:
                    smoke = '--max_steps' in cmd
                    scope = cmd[cmd.index('--adapter_scope') + 1]
                    ws = Path(cmd[cmd.index('--workspace') + 1]); ws.mkdir(parents=True)
                    self.assertNotIn('--resume', cmd)
                    self.assertEqual(cmd[cmd.index('--num_rounds') + 1], '1')
                    self.assertEqual(cmd[cmd.index('--gradient_accumulation_steps') + 1], '4')
                    write_audit(ws, scope)
                    (ws / 'training_diagnostics.json').write_text(json.dumps(dict(global_step=1 if smoke else 60, epoch=1 if smoke else 60, training_loss=.1)))
                    if smoke:
                        self.assertNotIn('--save_adapter_epochs', cmd)
                    else:
                        self.assertEqual(cmd[cmd.index('--num_train_epochs') + 1], '60')
                        self.assertEqual(cmd[cmd.index('--save_adapter_epochs') + 1:], [str(e) for e in EPOCHS])
                        cpdir = ws / 'epoch_checkpoints'; cpdir.mkdir()
                        for e in EPOCHS:
                            cp = cpdir / f'epoch-{e:02d}.pt'; cp.write_bytes(b'fixture')
                            cp.with_suffix('.json').write_text(json.dumps(dict(epoch=e, planned_epochs=60, global_step=e,
                                checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())))
                            snapshot(ws, e, planned_epochs=60)
                            with self.assertRaises(RuntimeError): snapshot(ws, e)  # old 10-epoch expectations remain enforced
                elif 'evaluate_subtype.py' in cmd:
                    self.assertIn(cmd[cmd.index('--split') + 1], ('train', 'validation'))
                    self.assertIn('--skip-generation', cmd)
                    self.assertEqual(cmd[cmd.index('--prediction-source') + 1], 'option_scores')
                    dest = Path(cmd[cmd.index('--output') + 1]); dest.mkdir()
                    (dest / 'evaluation.json').write_text(json.dumps({'metrics': {'patient_level_score_0_5': dict(
                        n_patients=2, accuracy=.5, balanced_accuracy=.5, roc_auc=.5, pr_auc=.5, recall={'NEG': 0, 'POS': 1})}}))
                else:
                    Path(cmd[cmd.index('--output') + 1]).write_text('{}')
            with patch('sys.argv', ['runner', '--data-root', str(data), '--output', str(output), '--arm', 'top16']), \
                 patch('run_multimodal_tiles60.importlib.metadata.version', return_value='fixture'), \
                 patch('run_multimodal_tiles60.subprocess.check_output', return_value='fixture'), \
                 patch('run_multimodal_tiles60.check_context', return_value={}), \
                 patch('run_multimodal_tiles60.processor_preflight'), \
                 patch('run_multimodal_tiles60.run_stage', side_effect=stage):
                main()
                with self.assertRaises(FileExistsError): main()
            jobs = [c for c in calls if 'job.py' in c]
            self.assertEqual(len(jobs), 4)
            self.assertTrue(all('--max_steps' in c for c in jobs[:2]))
            self.assertTrue(all('--max_steps' not in c for c in jobs[2:]))
            self.assertEqual(len({c[c.index('--workspace') + 1] for c in jobs}), 4)
            self.assertEqual([c[c.index('--adapter_scope') + 1] for c in jobs], list(SCOPES) * 2)
            self.assertEqual(len([c for c in calls if 'evaluate_subtype.py' in c]), 28)
            self.assertEqual(len(json.loads((output / 'summary.json').read_text())), 28)
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'completed')


if __name__ == '__main__':
    unittest.main()
