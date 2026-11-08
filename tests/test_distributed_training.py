"""Two real CPU/Gloo ranks exercise communication and FedLC gradient reduction."""
import json
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
import torch
import torch.multiprocessing as mp
from distributed_training import DistributedClient
from fl_methods import FLMethodState


class FakeFlare:
    def __init__(self):self.calls=[]
    def init(self):self.calls.append('init')
    def system_info(self):return {'site_name':'site-1'}
    def is_running(self):self.calls.append('is_running');return True
    def receive(self):self.calls.append('receive');return {'weights':torch.ones(3),'round':2}
    def send(self,model):self.calls.append('send')


class ToyBackbone(torch.nn.Module):
    def __init__(self):super().__init__();self.embed=torch.nn.Embedding(8,4)
    def forward(self,input_ids,**kw):return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class ToyGemma(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=ToyBackbone();self.lm_head=torch.nn.Linear(4,8)
        self.config=SimpleNamespace(get_text_config=lambda:SimpleNamespace(final_logit_softcapping=None))
        self.forward_calls=0
    def forward(self,input_ids,logits_to_keep=0,**kw):
        self.forward_calls+=1
        hidden=self.model(input_ids).last_hidden_state
        return SimpleNamespace(logits=self.lm_head(hidden[:,-logits_to_keep:]))


def ddp_worker(rank,port,folder):
    torch.set_num_threads(1)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port),RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2')
    api=FakeFlare();bridge=DistributedClient(api,backend='gloo')
    assert bridge.site_name()=='site-1' and bridge.is_running()
    message=bridge.receive();assert message['round']==2 and torch.equal(message['weights'],torch.ones(3))
    with bridge.gpu_slot(str(Path(folder)/'shared.lock'),'site-1','train'):
        torch.manual_seed(7)
        base=ToyGemma();ddp=torch.nn.parallel.DistributedDataParallel(base)
        state=FLMethodState('fedlc');state.token_ids={'NEG':2,'POS':3};state.margins={'NEG':.2,'POS':.1}
        for step in range(2):
            labels=torch.tensor([[-100,-100,2+rank,4]])
            inputs={'input_ids':torch.tensor([[1,rank+1,2,4]]),'labels':labels}
            loss=state.fedlc_loss(base,inputs,wrapped_model=ddp)
            loss.backward()
            # DDP gradients must equal the serial two-example reference.
            reference=ToyGemma();reference.load_state_dict(base.state_dict())
            both={'input_ids':torch.tensor([[1,1,2,4],[1,2,2,4]]),
                  'labels':torch.tensor([[-100,-100,2,4],[-100,-100,3,4]])}
            state.fedlc_loss(reference,both).backward()
            for p,q in zip(base.parameters(),reference.parameters()):
                assert torch.allclose(p.grad,q.grad,atol=1e-6)
            with torch.no_grad():
                for p in base.parameters():p.add_(p.grad,alpha=-.1)
            base.zero_grad(set_to_none=True)
        assert base.forward_calls==2
    bridge.send({'trained':True})
    (Path(folder)/f'{rank}.json').write_text(json.dumps(api.calls))
    bridge.close()


class DistributedTests(unittest.TestCase):
    def test_rank_zero_only_exchange_and_synchronized_fedlc(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as td:
            context=mp.spawn(ddp_worker,args=(port,td),nprocs=2,join=False)
            import time
            deadline=time.monotonic()+60
            while not context.join(timeout=1):
                if time.monotonic()>deadline:
                    for p in context.processes:p.terminate()
                    self.fail('DDP test timed out')
            self.assertEqual(json.loads((Path(td)/'0.json').read_text()),['init','is_running','receive','send'])
            self.assertEqual(json.loads((Path(td)/'1.json').read_text()),[])


if __name__=='__main__':unittest.main()
