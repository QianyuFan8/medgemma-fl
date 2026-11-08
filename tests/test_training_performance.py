import argparse
import ast
import os
from pathlib import Path
import shlex
from types import SimpleNamespace
import unittest
from copy import copy
import sys

from training_performance import (add_performance_args, performance_cli, client_gpu_plan,
    parse_client_gpu_groups, accumulation_for_world, distributed_layout)


class PerformanceTests(unittest.TestCase):
    def test_hardware_plans_use_all_requested_gpus(self):
        for n, sizes in [(1,[1,1,1]),(2,[1,1,1]),(4,[2,1,1]),(8,[4,2,2])]:
            ids=','.join(map(str,range(n)))
            value, threads=client_gpu_plan(ids)
            groups=parse_client_gpu_groups(value,3)
            self.assertEqual(list(map(len,groups)),sizes)
            self.assertEqual(set(sum(groups,[])),set(range(n)))
            self.assertEqual(threads,min(n,3))
            for group in groups:
                self.assertEqual(accumulation_for_world(4,1,len(group))*len(group),4)

    def test_invalid_hardware_and_batch_rejected(self):
        for ids in ['0,0','-1','0,foo','']:
            with self.assertRaises(ValueError):client_gpu_plan(ids)
        for groups in ['[0,1],[1,2],[3]','[0],[1]','0,1,2']:
            with self.assertRaises(ValueError):parse_client_gpu_groups(groups,3)
        with self.assertRaises(ValueError):accumulation_for_world(4,1,8)

    def test_defaults_and_options_round_trip(self):
        p=argparse.ArgumentParser();add_performance_args(p)
        original=p.parse_args([])
        self.assertTrue(original.quantized and original.gradient_checkpointing and original.local_eval)
        self.assertEqual(performance_cli(original),[])
        chosen=p.parse_args(['--no-quantized','--no-gradient_checkpointing','--no-local_eval',
                             '--preprocess_cache_dir','/tmp/test cache','--preprocess_cache_max_gb','2'])
        self.assertEqual(vars(p.parse_args(performance_cli(chosen))),vars(chosen))

    def test_ddp_rank_validation(self):
        self.assertEqual(distributed_layout({}),(1,0,0))
        self.assertEqual(distributed_layout({'WORLD_SIZE':'8','RANK':'7','LOCAL_RANK':'7'}),(8,7,7))
        with self.assertRaises(ValueError):distributed_layout({'WORLD_SIZE':'2','RANK':'2'})

    def test_job_configures_torchrun_and_scopes_shared_locks(self):
        path=Path(__file__).resolve().parents[1]/'job.py'
        tree=ast.parse(path.read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='gpu_site_settings')
        ns=dict(os=os,copy=copy,sys=sys,shlex=shlex,parse_client_gpu_groups=parse_client_gpu_groups,
                accumulation_for_world=accumulation_for_world)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
        args=SimpleNamespace(gpu='[0,1],[2],[3]',effective_batch_size=4,per_device_train_batch_size=1,
                             gradient_accumulation_steps=4,client_gpu_lock='/tmp/gpu.lock',workspace='/tmp/work')
        _,sites=ns['gpu_site_settings'](args,['site-1','site-2','site-3'])
        self.assertIn('--nproc_per_node=2',sites['site-1'][1])
        self.assertNotIn('torch.distributed.run',sites['site-2'][1])
        self.assertEqual(sites['site-1'][0].gradient_accumulation_steps,2)
        self.assertEqual(len({a.client_gpu_lock for a,c in sites.values()}),3)
        args.gpu='[0],[1],[0]'
        _,sites=ns['gpu_site_settings'](args,['site-1','site-2','site-3'])
        self.assertEqual(sites['site-1'][0].client_gpu_lock,sites['site-3'][0].client_gpu_lock)
        self.assertNotEqual(sites['site-1'][0].client_gpu_lock,sites['site-2'][0].client_gpu_lock)


if __name__=='__main__':unittest.main()
