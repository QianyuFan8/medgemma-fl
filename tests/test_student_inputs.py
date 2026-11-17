import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
import numpy as np
from pipelines.medgemma_student.common import digest, write_csv, write_json, read_csv
from pipelines.medgemma_student.plan import prepare_plan
from pipelines.medgemma_student.inputs import export_inputs, verify_export
from pipelines.medgemma_student.selector import strict_spatial16, fit_selector


def fixture(root):
    c=root/'cohort';r=root/'returned';c.mkdir();(r/'silver').mkdir(parents=True);(r/'limma/teacher').mkdir(parents=True)
    patients=[]
    for i,(split,label,subtype) in enumerate([('train_gold','POS','ARMS'),('train_gold','NEG','ERMS'),
                    ('train_silver_candidate','UNKNOWN','ERMS'),('validation','NEG','Spindle'),('test','POS','ARMS')]):
        patients.append(dict(patient_id=f'p{i}',sample_id=f's{i}',slide_id=str(i),subtype=subtype,
             split=split,gold_label=label,observed_subtypes=subtype,training_excluded=str(subtype=='Spindle')))
    write_csv(c/'patients.csv',patients)
    write_json(r/'silver/panel_provenance.json',{'cohort_sha256':digest(c/'patients.csv')})
    write_json(r/'teacher_selection.json',{'selected_selector':'limma','holdouts_used_for_selection':False})
    probes=[f'cg{i:08d}' for i in range(1000)];(r/'limma/teacher/selected_cpgs.txt').write_text('\n'.join(probes)+'\n')
    write_csv(r/'silver/paired_selected_cpg_beta_imputed.csv',[
        dict(patient_id=p['patient_id'],sample_id=p['sample_id'],split=p['split'],**{cg:.5 for cg in probes}) for p in patients])
    write_csv(r/'silver/image_input_manifest.csv',[dict(patient_id=p['patient_id'],sample_id=p['sample_id'],
        split=p['split'],slide_id=p['slide_id']) for p in patients])
    write_csv(r/'silver/student_train.csv',[dict(patient_id=p['patient_id'],sample_id=p['sample_id'],slide_id=p['slide_id'],
        training_label=p['gold_label'] if i<2 else 'NEG',label_source='gold' if i<2 else 'silver',student_loss_weight=1 if i<2 else .3)
        for i,p in enumerate(patients[:3])])
    write_csv(r/'silver/silver_labels.csv',[dict(patient_id='p2',included=True,predicted_label='NEG',confidence=.95)])
    return r,c


class Inputs(unittest.TestCase):
    def test_export_sixteen_images_weights_and_integrity(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,c=fixture(root);p=root/'plan'
            plan=prepare_plan(r,c,p);tiles=root/'tiles';tiles.mkdir();audit=[]
            for row in plan['rows']:
                name=row['slide_id']+'.png';image=Image.new('RGB',(896,896));image.save(tiles/name)
                audit.append(dict(patient_id=row['patient_id'],slide_id=row['slide_id'],file=name,image_sha256=digest(tiles/name)))
            write_json(tiles/'tiles.json',dict(plan_sha256=digest(p/'plan.json'),rows=audit))
            out=root/'export';export_inputs(p/'plan.json',tiles,out);verify_export(out)
            rows=json.loads((out/'clients/site-1/train.json').read_text())
            self.assertEqual([x['student_loss_weight'] for x in rows],[1,1,.3])
            self.assertTrue(all(x['image_layout']=='tiles16' for x in rows))
            self.assertIn('16 separate images',rows[0]['prompt'])
            rows[0]['student_loss_weight']=.3;write_json(out/'clients/site-1/train.json',rows)
            with self.assertRaisesRegex(ValueError,'changed'):verify_export(out)

    def test_training_subtype_blocked_and_holdout_preserved(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,c=fixture(root)
            patients=read_csv(c/'patients.csv')
            patients[0]['subtype']='RMS-NOS';write_csv(c/'patients.csv',patients)
            write_json(r/'silver/panel_provenance.json',{'cohort_sha256':digest(c/'patients.csv')})
            with self.assertRaisesRegex(ValueError,'Non-ERMS'):prepare_plan(r,c,root/'plan')

    def test_missing_features_fail_before_selector_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,c=fixture(root);prepare_plan(r,c,root/'plan')
            with self.assertRaisesRegex(FileNotFoundError,'Missing features'):
                fit_selector(root/'plan/plan.json',root/'selector',epochs=1)
            self.assertFalse((root/'selector').exists())

    def test_spatial_separation_and_no_relaxation(self):
        coords=np.array([(x*224,y*224) for y in range(8) for x in range(8)])
        indices=strict_spatial16(np.arange(64)[::-1],coords,224)
        self.assertEqual(len(indices),16)
        distances=np.linalg.norm(coords[indices,None]-coords[None,indices],axis=-1)
        np.fill_diagonal(distances,np.inf);self.assertGreaterEqual(distances.min(),448)
        with self.assertRaisesRegex(ValueError,'Cannot select'):
            strict_spatial16(np.arange(16),coords[:16],224)

    def test_holdout_labels_do_not_change_abmil_training(self):
        import h5py,torch
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);r,c=fixture(root);features=root/'features';features.mkdir()
            for i in range(5):
                with h5py.File(features/f'{i}.h5','w') as h:
                    h['feats']=np.random.default_rng(i).normal(size=(16,1536)).astype('float32')
                    h['coords']=np.array([(x*448,y*448) for y in range(4) for x in range(4)])
                    h.attrs.update(model='hf-hub:bioptimus/H-optimus-1',tile=224,target_mpp=.5,tile_level0=224)
            plan=prepare_plan(r,c,root/'plan',feature_dirs=[features]);torch.set_num_threads(1)
            fit_selector(root/'plan/plan.json',root/'fit1',epochs=1)
            for row in plan['rows']:
                if row['split']!='train':row['label_name']='POS' if row['label_name']=='NEG' else 'NEG'
            write_json(root/'plan/altered.json',plan)
            fit_selector(root/'plan/altered.json',root/'fit2',epochs=1)
            a=torch.load(root/'fit1/model.pt',weights_only=True)['state_dict']
            b=torch.load(root/'fit2/model.pt',weights_only=True)['state_dict']
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))


if __name__=='__main__':unittest.main()
