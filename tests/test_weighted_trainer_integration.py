"""Actual TRL/Transformers CPU training: accumulation must preserve patient weights."""
import copy
import tempfile
import unittest
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import Gemma3TextConfig, Gemma3ForCausalLM, PreTrainedTokenizerFast
from trl import SFTTrainer, SFTConfig
from medgemma_fl.weighted_sft import make_weighted_trainer, weighted_forward


class TrainerIntegration(unittest.TestCase):
    def check_accumulation(self, count):
        torch.set_num_threads(1);torch.manual_seed(7)
        config=Gemma3TextConfig(vocab_size=8,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=1,head_dim=8,max_position_embeddings=64,
            sliding_window=32,pad_token_id=0,bos_token_id=1,eos_token_id=7,attention_dropout=0.)
        config._attn_implementation='eager'
        model=Gemma3ForCausalLM(config);reference=copy.deepcopy(model)
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {'[PAD]':0,'[BOS]':1,'[UNK]':2,'POS':3,'NEG':4,'a':5,'b':6,'[EOS]':7},unk_token='[UNK]')),
            pad_token='[PAD]',bos_token='[BOS]',eos_token='[EOS]',unk_token='[UNK]')
        rows=[dict(input_ids=[1,5,i,7],labels=[-100,-100,i,7],student_loss_weight=w)
              for i,w in [(3,1.),(4,.3),(3,1.),(4,.3)]][:count]
        def collate(examples):
            return {k:torch.tensor([r[k] for r in examples],dtype=torch.float32 if k=='student_loss_weight' else torch.long) for k in rows[0]}
        reference.train();loss=weighted_forward(reference,collate(rows));loss.backward()
        expected={n:p.detach()-.01*p.grad for n,p in reference.named_parameters()}
        with tempfile.TemporaryDirectory() as td:
            args=SFTConfig(output_dir=td,use_cpu=True,bf16=False,fp16=False,max_steps=1,
                per_device_train_batch_size=1,gradient_accumulation_steps=4,learning_rate=.01,
                optim='sgd',weight_decay=0.,max_grad_norm=0.,lr_scheduler_type='constant',warmup_steps=0,
                save_strategy='no',report_to='none',disable_tqdm=True,remove_unused_columns=False,
                dataset_kwargs={'skip_prepare_dataset':True},seed=7)
            trainer=make_weighted_trainer(SFTTrainer,model=model,args=args,train_dataset=Dataset.from_list(rows),
                processing_class=tokenizer,data_collator=collate)
            trainer.train()
            for name,p in model.named_parameters():
                self.assertTrue(torch.allclose(p,expected[name],atol=2e-6,rtol=1e-5),name)

    def test_real_trainer_accumulation_matches_full_batch_sgd(self):
        self.check_accumulation(4)

    def test_partial_final_accumulation_group(self):
        self.check_accumulation(3)


if __name__=='__main__':unittest.main()
