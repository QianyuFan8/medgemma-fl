import hashlib
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from nvflare.app_common.abstract.fl_model import FLModel, ParamsType
    from federated_snapshots import SnapshotNaiveAggregator
    AVAILABLE=True
except ImportError:
    AVAILABLE=False


@unittest.skipUnless(AVAILABLE,'Requires the VM torch/NVFlare environment')
class SnapshotTests(unittest.TestCase):
    def test_shared_gpu_lock_is_forwarded_to_every_external_client(self):
        import shlex
        from unittest.mock import patch
        from job import define_parser, _build_train_args
        with tempfile.TemporaryDirectory(prefix='gpu lock ') as td:
            lock = str(Path(td)/'shared.lock')
            with patch('sys.argv', ['job.py', '--client_gpu_lock', lock]):
                args = define_parser()
            for site in ('site-1', 'site-2', 'site-3'):
                cmd = shlex.split(_build_train_args(args, f'/data/{site}', '/data', 'none', 16))
                self.assertEqual(cmd[cmd.index('--client_gpu_lock')+1], lock)

    def test_actual_flare_job_export_serializes_snapshot_configuration(self):
        from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
        with tempfile.TemporaryDirectory() as td:
            script=Path.cwd()/'client.py'
            self.assertTrue(script.is_file())
            agg=SnapshotNaiveAggregator(16,str(Path(td)/'snapshots'),[2,5,10,20,30,50,60],3)
            recipe=FedAvgRecipe(name='review',model={'class_path':'torch.nn.Linear','args':{'in_features':1,'out_features':1}},
                min_clients=3,num_rounds=60,train_script=str(script),aggregator=agg,launch_external_process=True,
                server_expected_format='pytorch',key_metric='')
            recipe.job.export_job(str(Path(td)/'export'))
            configs=list(Path(td).rglob('config_fed_server.json'))
            self.assertEqual(len(configs),1)
            text=configs[0].read_text()
            self.assertIn('federated_snapshots.SnapshotNaiveAggregator',text)
            self.assertIn('save_rounds',text)
            self.assertTrue(list(Path(td).rglob('federated_snapshots.py')))
            self.assertTrue(list(Path(td).rglob('client_gpu_lock.py')))

    def test_global_snapshots_are_weighted_post_aggregation_and_round_number_survives_reset(self):
        with tempfile.TemporaryDirectory() as td:
            agg=SnapshotNaiveAggregator(1,td,[1,2],3)
            agg.info=lambda *args:None
            for round_id in (1,2):
                for count,value in ((1,1.),(2,2.),(3,3.)):
                    params={'model.layer.lora_A.weight':torch.tensor([[value]]),
                            'model.layer.lora_B.weight':torch.tensor([[2*value]])}
                    agg.accept_model(FLModel(params=params,params_type=ParamsType.FULL,meta={'num_examples':count}))
                result=agg.aggregate_model()
                path=Path(td)/f'round-{round_id:03d}.pt'
                state=torch.load(path,weights_only=True)['model']
                self.assertAlmostEqual(state['model.layer.lora_A.weight'].item(),14/6,places=5)
                self.assertTrue(torch.equal(state['model.layer.lora_A.weight'],result.params['model.layer.lora_A.weight']))
                info=json.loads(path.with_suffix('.json').read_text())
                self.assertEqual(info['completed_rounds'],round_id)
                self.assertEqual(info['checkpoint_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(info['training_record_weights'],[1,2,3])
                agg.reset_stats()
            self.assertEqual(agg.completed_rounds,2)
            with self.assertRaisesRegex(ValueError,'incomplete-client'):agg.aggregate_model()


if __name__ == '__main__':unittest.main()
