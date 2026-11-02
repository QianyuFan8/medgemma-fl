"""CPU tests for candidate-only scoring and frozen-vision experiment wiring."""
import hashlib
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from adapter_scope import target_modules, infer_adapter_scope, trainable_summary
from evaluate_subtype import option_prediction, predict_one
from run_zero_language_diagnostics import main, score_command, verify_language_audit, BASE_MODEL


class ZeroLanguageTests(unittest.TestCase):
    def test_language_targets_exclude_vision_projector_and_output_head(self):
        pattern = target_modules('language-only')
        for name in ('model.language_model.layers.0.self_attn.q_proj',
                     'model.language_model.layers.0.mlp.down_proj'):
            self.assertIsNotNone(re.fullmatch(pattern, name))
        for name in ('model.vision_tower.encoder.layers.0.self_attn.q_proj',
                     'model.multi_modal_projector.linear', 'model.language_model.lm_head'):
            self.assertIsNone(re.fullmatch(pattern, name))
        language = 'base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight'
        vision = 'base_model.model.model.vision_tower.encoder.layers.0.self_attn.q_proj.lora_A.weight'
        self.assertEqual(infer_adapter_scope({language: None}), 'language-only')
        self.assertEqual(infer_adapter_scope({language: None, vision: None}), 'lora-only')
        param = SimpleNamespace(requires_grad=True, numel=lambda: 16)
        model = SimpleNamespace(named_parameters=lambda: [(language, param)])
        self.assertEqual(trainable_summary(model, 'language-only')['vision_trainable_parameters'], 0)
        model.named_parameters = lambda: [(language, param), (vision, param)]
        with self.assertRaisesRegex(ValueError, 'freeze'):
            trainable_summary(model, 'language-only')

    def test_option_labels_and_ties_do_not_require_generated_text(self):
        self.assertEqual(option_prediction(['NEG', 'POS'], [.7, .3]), 'NEG')
        self.assertEqual(option_prediction(['NEG', 'POS'], [.5, .5]), 'POS')
        with self.assertRaises(ValueError):
            option_prediction(['NEG', 'POS'], [float('nan'), .3])

    def test_score_only_never_generates_and_scores_full_multitoken_answers(self):
        from contextlib import nullcontext
        class Tensor(np.ndarray):
            def __new__(cls, value): return np.asarray(value).view(cls)
            def float(self): return self
            def unsqueeze(self, axis): return Tensor(np.expand_dims(self, axis))
            def gather(self, axis, index): return Tensor(np.take_along_axis(self, index, axis=axis))
            def softmax(self, axis):
                e = np.exp(np.asarray(self) - np.max(self, axis=axis, keepdims=True))
                return Tensor(e / e.sum(axis=axis, keepdims=True))
        def log_softmax(x, axis):
            a = np.asarray(x); shifted = a - a.max(axis=axis, keepdims=True)
            return Tensor(shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True)))
        torch = SimpleNamespace(inference_mode=nullcontext, equal=np.array_equal,
                                log_softmax=log_softmax, stack=lambda x: Tensor(x))
        class Model:
            def generate(self, **kwargs): raise AssertionError('Free generation must not run')
            def __call__(self, **kwargs):
                # POS has the larger first-token probability, but NEG has the
                # larger full-answer probability: .4*.9 versus .6*.1.
                probabilities = [[.000001, .4, .6, .000001, .000001],
                                 [.000001, .000001, .000001, .9, .1],
                                 [.2, .2, .2, .2, .2]]
                return SimpleNamespace(logits=Tensor([np.log(probabilities)]))
        def template(messages, **kwargs):
            self.assertEqual([m['role'] for m in messages], ['user'])
            return 'prompt'
        def encode(processor, text, images, device):
            return {'input_ids': Tensor([[0] if text=='prompt' else
                                         [0, 1, 3] if text.endswith('NEG') else [0, 2, 4]])}
        row = {'patient_id': 'fixture', 'label_name': 'POS', 'prompt': 'Classify', 'image': None}
        with patch.dict('sys.modules', {'torch': torch, 'inference_utils': SimpleNamespace(get_model_device=lambda m: 'cpu')}), \
             patch('evaluate_subtype.encode', side_effect=encode):
            result = predict_one(Model(), SimpleNamespace(apply_chat_template=template), row, '.',
                                 ['NEG', 'POS'], 100, 'strict', 'option_scores', True)
        self.assertEqual(result['prediction'], 'NEG')
        self.assertEqual(result['raw_response'], '')
        self.assertGreater(result['score_NEG'], result['score_POS'])

    def test_queue_has_three_base_model_controls_and_only_one_language_training(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); data=root/'data'; data.mkdir(); output=root/'out'
            (data/'diagnostics_plan.json').write_text(json.dumps({
                'test_evaluation_enabled': False, 'training_seed': 42, 'arm': 'top16'}))
            for mode in ('image', 'multimodal', 'methylation'):
                dest=data/mode; dest.mkdir(); (dest/'task.json').write_text(json.dumps({'modality':mode}))
            language = 'model.base_model.model.model.language_model.layers.0.q_proj.lora_A.weight'
            calls=[]
            def stage(cmd, log, name, status_path, status):
                calls.append(cmd)
                if 'job.py' in cmd:
                    self.assertEqual(cmd[cmd.index('--adapter_scope')+1], 'language-only')
                    self.assertEqual(cmd[cmd.index('--num_train_epochs')+1], '10')
                    self.assertEqual(cmd[cmd.index('--learning_rate')+1], '5e-05')
                    ws=Path(cmd[cmd.index('--workspace')+1]); ws.mkdir()
                    (ws/'training_diagnostics.json').write_text(json.dumps({'global_step':100,'epoch':10,'training_loss':.1}))
                    (ws/'trainable_parameters.json').write_text(json.dumps({
                        'adapter_scope':'language-only','trainable_parameters':16,
                        'vision_trainable_parameters':0,'trainable_tensors':[language]}))
                    folder=ws/'epoch_checkpoints'; folder.mkdir()
                    for e in (2,5,10):
                        cp=folder/f'epoch-{e:02d}.pt'; cp.write_bytes(b'weights')
                        cp.with_suffix('.json').write_text(json.dumps({'epoch':e,'planned_epochs':10,
                            'global_step':e*10,'checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest()}))
                elif 'evaluate_subtype.py' in cmd:
                    self.assertIn('--skip-generation',cmd)
                    self.assertEqual(cmd[cmd.index('--prediction-source')+1], 'option_scores')
                    self.assertNotEqual(cmd[cmd.index('--split')+1], 'test')
                    if name.startswith('zero_shot'):
                        self.assertEqual(cmd[cmd.index('--model-path')+1], BASE_MODEL)
                        self.assertEqual(cmd[cmd.index('--split')+1], 'validation')
                    dest=Path(cmd[cmd.index('--output')+1]); dest.mkdir()
                    (dest/'evaluation.json').write_text(json.dumps({'metrics':{'patient_level_score_0_5':{
                        'n_patients':10,'accuracy':.6,'balanced_accuracy':.6,'roc_auc':.7,'pr_auc':.7,
                        'recall':{'NEG':.6,'POS':.6}}}}))
                else:
                    Path(cmd[cmd.index('--output')+1]).write_text('{}')
            with patch('sys.argv',['runner','--data-root',str(data),'--output',str(output)]), \
                 patch('run_zero_language_diagnostics.validate_clients'), \
                 patch('run_zero_language_diagnostics.importlib.metadata.version',return_value='fixture'), \
                 patch('run_zero_language_diagnostics.subprocess.check_output',return_value='fixture'), \
                 patch('run_zero_language_diagnostics.run_stage',side_effect=stage), \
                 patch.dict('sys.modules',{'torch':SimpleNamespace(load=lambda *a,**k:{'model':{language:None}})}):
                main()
                with self.assertRaises(FileExistsError): main()
            self.assertEqual(len([c for c in calls if 'job.py' in c]),1)
            self.assertEqual(len([c for c in calls if 'evaluate_subtype.py' in c]),9)
            self.assertEqual(len(json.loads((output/'summary.json').read_text())),9)
            self.assertEqual(json.loads((output/'status.json').read_text())['state'],'completed')


if __name__ == '__main__': unittest.main()
