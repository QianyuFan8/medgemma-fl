import argparse
import csv
import json
import tempfile
import unittest
from pathlib import Path
from test_comet_subtype import synthetic_inputs, fake_prepared
from prepare_comet_subtype import audit, export, read_tsv, write_tsv
from plan_fusion_cv import make_folds
from evaluate_fusion_baselines import combine


class FusionLinkageTests(unittest.TestCase):
    def test_canonical_aliases_same_patient_slide_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);synthetic_inputs(root)
            with (root/'map.csv').open() as f:rows=list(csv.DictReader(f))
            # A different legacy prefix, same canonical UID and same slide.
            rows[0]['Slide Scan File']='7999'
            rows[0]['Comp Bio ID']='LEGACY_ALIAS_D1'
            with (root/'map.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            meta=read_tsv(root/'meta.txt')
            for r in meta:r['SJID']='';r['Slide.Scan.File']=''
            meta[1]['SJID']='DISAGREES_D1'
            meta[2]['Slide.Scan.File']='999999'
            write_tsv(root/'meta.txt',meta)
            audit(argparse.Namespace(mapping=root/'map.csv',fusion=root/'fusion.csv',comet_matrix=root/'beta.tsv',
                comet_metadata=root/'meta.txt',slide_dir=None,slide_list=root/'slides.txt',output=root/'audit'))
            records={r['comet_sample_id']:r for r in read_tsv(root/'audit/samples_audit.tsv')}
            self.assertEqual(records['RMS_0']['patient_id'],records['RMS_dup']['patient_id'])
            self.assertEqual(records['RMS_0']['slide_status'],'available')
            self.assertEqual(records['RMS_dup']['slide_status'],'available')
            self.assertEqual(sum(records[k]['status']=='candidate_pending_label_review' for k in ['RMS_0','RMS_dup']),1)
            self.assertEqual(records['RMS_1']['status'],'sample_identity_conflict')
            self.assertEqual(records['RMS_2']['slide_status'],'slide_mapping_conflict')
            self.assertEqual(records['RMS_sh1']['slide_status'],'slide_shared_by_multiple_samples')

    def test_cv_disjoint_complete_deterministic(self):
        rows=[dict(patient_id=f'{label}{i}',diagnosis=label) for label in ['NEG','POS'] for i in range(15)]
        plans=make_folds(rows)
        self.assertEqual(plans,make_folds(list(reversed(rows))))
        tests=[]
        for plan in plans:
            self.assertEqual(len({r['patient_id'] for r in plan}),len(rows))
            for split in ['train','validation','test']:
                self.assertEqual({r['diagnosis'] for r in plan if r['split']==split},{'NEG','POS'})
            tests.extend(r['patient_id'] for r in plan if r['split']=='test')
        self.assertEqual(len(tests),len(set(tests)))
        self.assertEqual(set(tests),{r['patient_id'] for r in rows})

    def test_old_panel_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'features.json').write_text(json.dumps({'classes':['ERMS','ARMS']}))
            with self.assertRaisesRegex(ValueError,'ERMS/ARMS'):
                export(argparse.Namespace(prepared=root,output=root/'out',modality='methylation',n_clients=1))

    def test_late_fusion_exact_patient_alignment(self):
        a=[dict(patient_id='a',truth='NEG',score_POS=0.2),dict(patient_id='b',truth='POS',score_POS=0.8)]
        b=[dict(patient_id='b',truth='POS',score_POS=0.6),dict(patient_id='a',truth='NEG',score_POS=0.4)]
        rows=combine(a,b)
        self.assertAlmostEqual(rows[0]['score_POS'],0.3)
        self.assertAlmostEqual(rows[1]['score_POS'],0.7)
        with self.assertRaises(ValueError):combine(a,b[:1])
        with self.assertRaises(ValueError):combine(a,b+b[:1])
        b[0]['truth']='NEG'
        with self.assertRaises(ValueError):combine(a,b)

if __name__=='__main__':unittest.main()
