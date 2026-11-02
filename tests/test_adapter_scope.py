import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from adapter_scope import saved_modules, infer_adapter_scope, trainable_summary, target_modules

class AdapterScopeTests(unittest.TestCase):
    def test_legacy_and_lora_checkpoint_detection(self):
        adapter={'model.base_model.model.layers.0.q_proj.lora_A.weight':None,
                 'model.base_model.model.layers.0.q_proj.lora_B.weight':None}
        self.assertEqual(infer_adapter_scope(adapter),'lora-only')
        self.assertIsNone(saved_modules('lora-only'))
        for key in ['model.base_model.model.lm_head.weight',
                    'base_model.model.model.embed_tokens.weight',
                    'base_model.model.lm_head.modules_to_save.default.weight']:
            self.assertEqual(infer_adapter_scope({**adapter,key:None}),'legacy')
        self.assertEqual(saved_modules('legacy'),['lm_head','embed_tokens'])

    def test_reject_unintended_full_weight_training(self):
        param=SimpleNamespace(requires_grad=True,numel=lambda:16)
        model=SimpleNamespace(named_parameters=lambda:[('layer.lora_A.default.weight',param)])
        self.assertEqual(trainable_summary(model,'lora-only')['trainable_parameters'],16)
        model.named_parameters=lambda:[('lm_head.weight',param)]
        with self.assertRaisesRegex(ValueError,'non-adapter'):
            trainable_summary(model,'lora-only')

    def test_peft_config_keeps_rank_but_changes_full_modules(self):
        p=Path(__file__).resolve().parents[1]/'model.py'
        t=ast.parse(p.read_text())
        fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='create_lora_config')
        fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
        ns=dict(DEFAULT_LORA_R=16,DEFAULT_LORA_ALPHA=16,DEFAULT_LORA_DROPOUT=.05,
                DEFAULT_TARGET_MODULES='all-linear',saved_modules=saved_modules,target_modules=target_modules,
                LoraConfig=lambda **kw:kw)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
        legacy=ns['create_lora_config'](); pure=ns['create_lora_config'](adapter_scope='lora-only')
        self.assertEqual(legacy['r'],pure['r'])
        self.assertIsNone(pure['modules_to_save'])
        self.assertEqual(legacy['modules_to_save'],['lm_head','embed_tokens'])
        language=ns['create_lora_config'](adapter_scope='language-only')
        self.assertEqual(language['target_modules'], target_modules('language-only'))
        self.assertIsNone(language['modules_to_save'])
        self.assertEqual(language['r'], pure['r'])

if __name__=='__main__':unittest.main()
