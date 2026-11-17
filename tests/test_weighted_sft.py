import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
import torch
import torch.multiprocessing as mp
from medgemma_fl.weighted_sft import patient_weighted_ce, weighted_forward, WeightedCollator, make_weighted_trainer


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Embedding(8,4);self.head=torch.nn.Linear(4,8)
    def forward(self,input_ids,logits_to_keep,**kwargs):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)[:,-logits_to_keep:]))


def batch():
    return {'input_ids':torch.tensor([[1,2,3,4],[1,3,5,4],[1,6,2,4],[1,5,3,4]]),
            'labels':torch.tensor([[-100,-100,3,4],[-100,-100,5,4],[-100,-100,2,4],[-100,-100,3,4]]),
            'student_loss_weight':torch.tensor([1.,.3,1.,.3])}


def worker(rank,port):
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
    torch.set_num_threads(1);torch.distributed.init_process_group('gloo',rank=rank,world_size=2)
    torch.manual_seed(10);base=Toy();reference=Toy();reference.load_state_dict(base.state_dict())
    ddp=torch.nn.parallel.DistributedDataParallel(base);b=batch()
    # Two accumulation microbatches on each of two ranks = a single batch of four.
    for j in range(2):
        i=rank+j*2
        (weighted_forward(ddp,{k:v[i:i+1] for k,v in b.items()})/2).backward()
    weighted_forward(reference,b).backward()
    for p,q in zip(base.parameters(),reference.parameters()):
        assert torch.allclose(p.grad,q.grad,atol=1e-6), 'DDP/accumulation changed weighted gradient'
    torch.distributed.destroy_process_group()


class WeightedTests(unittest.TestCase):
    def test_silver_weight_does_not_cancel_at_batch_one(self):
        torch.manual_seed(1);m=Toy();b={k:v[:1] for k,v in batch().items()}
        gold=weighted_forward(m,b);b['student_loss_weight']=torch.tensor([.3]);silver=weighted_forward(m,b)
        self.assertAlmostEqual(silver.item()/gold.item(),.3,places=6)

    def test_answer_only_and_patient_mean(self):
        logits=torch.zeros(2,3,8,requires_grad=True)
        labels=torch.tensor([[-100,-100,2,-100],[-100,-100,3,4]])
        loss=patient_weighted_ce(logits,labels,1,torch.tensor([1.,.3]))
        self.assertAlmostEqual(loss.item(),float(torch.log(torch.tensor(8.)))*.65,places=6)
        loss.backward();self.assertTrue(torch.equal(logits.grad[:,-1],torch.zeros_like(logits.grad[:,-1])))

    def test_invalid_weight_rejected(self):
        collator=WeightedCollator(lambda examples:{})
        with self.assertRaises(ValueError):collator([{'student_loss_weight':float('nan')}])
        with self.assertRaises(KeyError):collator([{}])

    def test_trainer_scaling_contract(self):
        class Stub:
            def __init__(self,**kw):self.model_accepts_loss_kwargs=True
        t=make_weighted_trainer(Stub)
        self.assertFalse(t.model_accepts_loss_kwargs)
        m=Toy();b=batch();loss=t.compute_loss(m,b,num_items_in_batch=999)
        self.assertTrue(torch.allclose(loss,weighted_forward(m,b)))

    def test_two_rank_ddp_and_accumulation(self):
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        ctx=mp.spawn(worker,args=(port,),nprocs=2,join=False)
        import time
        deadline=time.monotonic()+60
        while not ctx.join(timeout=1):
            if time.monotonic()>deadline:
                for p in ctx.processes:p.terminate()
                self.fail('DDP test timeout')


if __name__=='__main__':unittest.main()
