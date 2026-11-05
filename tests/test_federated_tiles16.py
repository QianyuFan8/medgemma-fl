import csv
import json
from pathlib import Path
import tempfile
import unittest

from test_multimodal_tiles60 import multimodal_fixture, write_audit
from prepare_multiimage_diagnostics import prepare as separate_images, signature
from prepare_comet_subtype import write_tsv
from prepare_federated_tiles16 import prepare, verify
from run_federated_tiles16 import training_command, saved_rounds, verify_training, SEED
from run_spatial20_inference import digest


def source_fixture(root):
    montage = root / 'montage'
    multimodal_fixture(montage)
    manifest = []
    for split in ('train', 'validation', 'test'):
        path = montage / ('clients/site-1/train.json' if split == 'train' else f'{split}.json')
        old = json.loads(path.read_text())
        rows = []
        for row in old:
            for i in range(12):
                new = dict(row, patient_id=row['patient_id']+f'_{i}', canonical_uid=row['canonical_uid']+f'_{i}', image_selection='spatial')
                rows.append(new)
            if split == 'train':
                rows.append(dict(rows[-1], patient_id=rows[-1]['patient_id']+'_repeat'))
        path.write_text(json.dumps(rows))
        if split == 'validation':
            (montage/'clients/site-1/validation.json').write_text(json.dumps(rows))
        with (montage/f'{split}.csv').open('w', newline='') as f:
            w=csv.DictWriter(f,fieldnames=['patient_id','diagnosis']);w.writeheader()
            w.writerows(dict(patient_id=r['patient_id'],diagnosis=r['label_name']) for r in rows)
        manifest += [dict(patient_id=r['patient_id'],canonical_uid=r['canonical_uid'],diagnosis=r['label_name'],split=split,site_id='site-1') for r in rows]
    write_tsv(montage/'splits.tsv',manifest)
    write_tsv(montage/'site_assignments.tsv',manifest)
    source = root/'source'
    separate_images(montage,source,modality='multimodal')
    return source


class FederatedTilesTests(unittest.TestCase):
    def test_repartition_retains_inputs_and_patient_ownership(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); source=source_fixture(root)
            before={str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()}
            original={r['patient_id']:r for r in json.loads((source/'clients/site-1/train.json').read_text())}
            for layout in ('iid','skew08'):
                a,b=root/(layout+'a'),root/(layout+'b')
                report=prepare(source,a,layout); other=prepare(source,b,layout)
                self.assertEqual(report['assignments_sha256'],other['assignments_sha256'])
                self.assertEqual(report['content_sha256'],other['content_sha256'])
                self.assertEqual(signature(source),signature(a))
                owners={};seen=set()
                for site in ('site-1','site-2','site-3'):
                    for row in json.loads((a/'clients'/site/'train.json').read_text()):
                        self.assertNotIn(row['patient_id'],seen);seen.add(row['patient_id'])
                        self.assertEqual(owners.setdefault(row['canonical_uid'],site),site)
                        for k,v in original[row['patient_id']].items():
                            if k!='site_id':self.assertEqual(row[k],v)
                self.assertEqual(seen,set(original))
                verify(a)
                with self.assertRaises(FileExistsError):prepare(source,a,layout)
                path=a/'clients/site-1/train.json'
                path.write_text(path.read_text().replace('Interpret all 16','Consider all 16'))
                with self.assertRaisesRegex(ValueError,'Prepared data changed'):verify(a)
            self.assertEqual(before,{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file()})

    def test_training_budget_and_global_snapshots(self):
        self.assertEqual(saved_rounds(60),[2,5,10,20,30,50,60])
        for scope in ('language-only','lora-only'):
            cmd=training_command(Path('data'),Path('run'),scope,'fedlc',60,1)
            for flag,value in [('--num_rounds','60'),('--num_train_epochs','1'),('--n_clients','3'),
                               ('--adapter_scope',scope),('--learning_rate','5e-5'),('--sim_threads','1')]:
                self.assertEqual(cmd[cmd.index(flag)+1],value)
            self.assertNotIn('--max_steps',cmd)
            self.assertNotIn('--save_adapter_epochs',cmd)
            self.assertNotIn('--resume',cmd)
            smoke=training_command(Path('data'),Path('smoke'),scope,'fedavg',60,1,True)
            self.assertEqual(smoke[smoke.index('--num_rounds')+1],'1')
            self.assertEqual(smoke[smoke.index('--max_steps')+1],'1')

    def test_initial_server_checkpoint_is_not_completion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); data=root/'data'; dest=root/'run'
            for site in ('site-1','site-2','site-3'):
                folder=data/'clients'/site;folder.mkdir(parents=True)
                (folder/'train.json').write_text('[{},{}]')
                workspace=dest/'train/medgemma'/site;workspace.mkdir(parents=True)
                write_audit(workspace,'language-only')
                r=workspace/'round-0';r.mkdir()
                (r/'training_diagnostics.json').write_text(json.dumps(dict(global_step=1,epoch=1,training_loss=.1,
                    seed=SEED,learning_rate=5e-5,train_records=2,max_steps=None)))
            (dest/'global_checkpoints').mkdir()
            (dest/'global_checkpoints/round-060.pt').write_bytes(b'initial')
            with self.assertRaisesRegex(RuntimeError,'did not complete all 60'):
                verify_training(dest,'language-only',data,60,1)


if __name__ == '__main__':unittest.main()
