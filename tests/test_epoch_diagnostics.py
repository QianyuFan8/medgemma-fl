"""CPU checks for snapshot boundaries, provenance and unchanged training budget."""
import ast
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run_epoch_diagnostics import EPOCHS, experiments, snapshot
from run_epoch_diagnostics import main as run_main
from run_central_diagnostics import train_command, evaluation_command


REPO = Path(__file__).resolve().parents[1]


def callback_class():
    tree = ast.parse((REPO / 'client.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EpochAdapterCheckpoint')
    ns = {'TrainerCallback': object, 'get_adapter_state_dict': lambda model: {'layer.lora_A.weight': [1, 2]}}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'client.py', 'exec'), ns)
    return ns['EpochAdapterCheckpoint']


class EpochTests(unittest.TestCase):
    def test_full_queue_evaluates_six_snapshots_without_test_or_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data = root / 'data'; data.mkdir(); output = root / 'run'
            (data / 'diagnostics_plan.json').write_text(json.dumps({
                'arm': 'top16', 'training_seed': 42, 'test_evaluation_enabled': False}))
            calls = []
            def stage(cmd, log, name, status_path, status):
                calls.append(cmd)
                if 'job.py' in cmd:
                    ws = Path(cmd[cmd.index('--workspace')+1]); ws.mkdir()
                    (ws / 'training_diagnostics.json').write_text(json.dumps({
                        'epoch': 10, 'global_step': 100, 'training_loss': .01}))
                    folder = ws / 'epoch_checkpoints'; folder.mkdir()
                    for e in EPOCHS:
                        cp = folder / f'epoch-{e:02d}.pt'; cp.write_bytes(b'adapter')
                        cp.with_suffix('.json').write_text(json.dumps({
                            'epoch': e, 'planned_epochs': 10, 'global_step': e*10,
                            'checkpoint_sha256': hashlib.sha256(cp.read_bytes()).hexdigest()}))
                elif 'evaluate_subtype.py' in cmd:
                    self.assertIn(cmd[cmd.index('--split')+1], ('train', 'validation'))
                    dest = Path(cmd[cmd.index('--output')+1]); dest.mkdir()
                    (dest / 'evaluation.json').write_text(json.dumps({'metrics': {
                        'patient_level_score_0_5': {'n_patients': 10, 'balanced_accuracy': .6,
                                                  'roc_auc': .7, 'pr_auc': .7,
                                                  'recall': {'NEG': .6, 'POS': .6}}}}))
                else:
                    Path(cmd[cmd.index('--output')+1]).write_text('{}')
            with patch('sys.argv', ['runner', '--data-root', str(data), '--output', str(output)]), \
                 patch('run_epoch_diagnostics.validate_clients'), \
                 patch('run_epoch_diagnostics.importlib.metadata.version', return_value='test'), \
                 patch('run_epoch_diagnostics.subprocess.check_output', return_value='test'), \
                 patch('run_epoch_diagnostics.run_stage', side_effect=stage):
                run_main()
                with self.assertRaises(FileExistsError):
                    run_main()
            self.assertEqual(len(calls), 20)
            self.assertEqual(len(json.loads((output / 'summary.json').read_text())), 12)
            self.assertEqual(json.loads((output / 'status.json').read_text())['state'], 'completed')

    def test_only_exact_requested_epochs_are_saved_and_readable(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            args = SimpleNamespace(output_dir=td, num_train_epochs=10)
            cb = callback_class()(EPOCHS)
            fake_torch = SimpleNamespace(save=lambda obj, p: p.write_text(json.dumps(obj)))
            with patch.dict('sys.modules', {'torch': fake_torch}):
                for e in (1, 1.9, 2, 2, 3, 5, 10):
                    control = object()
                    self.assertIs(cb.on_epoch_end(args, SimpleNamespace(
                        epoch=e, global_step=round(e * 10), is_world_process_zero=True), control, model=None), control)
            self.assertEqual(cb.saved, {2, 5, 10})
            self.assertEqual(len(list(root.rglob('*.pt'))), 3)
            for e in EPOCHS:
                path = snapshot(root, e)
                state = json.loads(path.read_text())
                self.assertIn('model.layer.lora_A.weight', state['model'])
                self.assertFalse(path.with_suffix('.pt.tmp').exists())
            path.write_text('corrupt')
            with self.assertRaisesRegex(RuntimeError, 'checksum'):
                snapshot(root, 10)

    def test_worker_does_not_save_and_existing_checkpoint_not_overwritten(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(output_dir=td, num_train_epochs=10)
            state = SimpleNamespace(epoch=2, global_step=20, is_world_process_zero=False)
            cb = callback_class()(EPOCHS)
            fake_torch = SimpleNamespace(save=lambda obj, p: p.write_text(json.dumps(obj)))
            with patch.dict('sys.modules', {'torch': fake_torch}):
                cb.on_epoch_end(args, state, None)
                self.assertEqual(list(Path(td).iterdir()), [])
                state.is_world_process_zero = True
                cb.on_epoch_end(args, state, None)
                with self.assertRaises(FileExistsError):
                    callback_class()(EPOCHS).on_epoch_end(args, state, None)

    def test_missing_or_ambiguous_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(RuntimeError):
                snapshot(root, 2)
            for site in ('one', 'two'):
                p = root / site / 'epoch_checkpoints'; p.mkdir(parents=True)
                (p / 'epoch-02.pt').write_bytes(b'weights')
            with self.assertRaises(RuntimeError):
                snapshot(root, 2)

    def test_continuous_budget_and_evaluation_destinations(self):
        self.assertEqual(EPOCHS, (2, 5, 10))
        self.assertEqual([e['learning_rate'] for e in experiments()], [2e-4, 5e-5])
        for ex in experiments():
            cmd = train_command(Path('data'), Path('train'), ex, 42)
            for flag, value in [('--num_train_epochs', '10'), ('--num_rounds', '1'), ('--n_clients', '1')]:
                self.assertEqual(cmd[cmd.index(flag)+1], value)
            for split in ('train', 'validation'):
                cmd = evaluation_command(Path('data'), Path('epoch-02.pt'), Path('epoch-02'), split)
                self.assertEqual(cmd[cmd.index('--output')+1], f'epoch-02/eval_{split}')

    def test_snapshot_argument_reaches_client(self):
        import shlex
        tree = ast.parse((REPO / 'job.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_build_train_args')
        ns = {'shlex': shlex}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'job.py', 'exec'), ns)
        args = SimpleNamespace(task='subtype', max_seq_length=32768, model_name_or_path='model',
                               attn_implementation='sdpa', seed=42, logging_steps=5, eval_steps=10,
                               fl_method='fedavg', max_steps=None, num_train_epochs=10,
                               learning_rate=2e-4, per_device_train_batch_size=1,
                               per_device_eval_batch_size=1, gradient_accumulation_steps=4,
                               save_adapter_epochs=[2, 5, 10])
        cmd = shlex.split(ns['_build_train_args'](args, 'data', 'images', 'none', 16))
        start = cmd.index('--save_adapter_epochs')
        self.assertEqual(cmd[start+1:start+4], ['2', '5', '10'])


if __name__ == '__main__':
    unittest.main()
