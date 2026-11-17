import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ridge_teacher.features.selection import LimmaCpGSelector, FoldCpGSelector
from ridge_teacher.train.search import make_search

RSCRIPT=str(Path(sys.executable).with_name('Rscript')) if Path(sys.executable).with_name('Rscript').exists() else 'Rscript'


class HandoffTests(unittest.TestCase):
    def test_separate_detection_mask_and_probe_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);cohort=d/'cohort';cohort.mkdir()
            rows=[dict(subtype='ERMS',patient_id=f'p{i}',sample_id=f's{i}',sentrix_id=f'a{i}',gold_label='POS' if i%2 else 'NEG',
                       split='train_gold',teacher_train=True) for i in range(6)]
            pd.DataFrame(rows).to_csv(cohort/'patients.csv',index=False)
            beta={'TargetID':[f'cg{i:08d}' for i in range(10)]};dp={'TargetID':beta['TargetID']}
            for i in range(6):
                beta[f'a{i}.Ave_Beta']=np.full(10,.4);dp[f'a{i}']=np.zeros(10)
            dp['a0'][0]=.02
            pd.DataFrame(beta).to_csv(d/'beta.tsv',sep='\t',index=False)
            pd.DataFrame(dp).to_csv(d/'dp.tsv',sep='\t',index=False)
            command=[sys.executable,str(ROOT/'cache_matrix.py'),'--cohort',str(cohort),'--beta',str(d/'beta.tsv'),
                     '--detection-file',str(d/'dp.tsv')]
            r=subprocess.run([*command,'--out',str(d/'matrix')],text=True,capture_output=True)
            self.assertEqual(r.returncode,0,r.stderr)
            x=np.load(d/'matrix/beta.npy');self.assertTrue(np.isnan(x[0,0]));self.assertTrue(np.isfinite(x[0,1]))
            pd.DataFrame(dp).iloc[::-1].to_csv(d/'dp.tsv',sep='\t',index=False)
            r=subprocess.run([*command,'--out',str(d/'bad')],text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0);self.assertIn('TargetID order',r.stderr)

    def test_limma_uses_between_class_signal_and_reports_bh(self):
        rng=np.random.default_rng(51);y=np.array([0,1]*20)
        X=rng.uniform(.1,.9,(40,50));X[:,0]=.45+.05*y+rng.normal(0,.002,40)
        limma=LimmaCpGSelector(top_k=1,rscript=RSCRIPT).fit(X,y)
        variance=FoldCpGSelector(top_k=1).fit(X,y)
        self.assertEqual(limma.indices_.tolist(),[0])
        self.assertNotIn(0,variance.indices_)
        stat=limma.statistics_.iloc[0]
        self.assertGreater(stat.delta_M,0)
        self.assertGreaterEqual(stat.p_adj_BH,stat.p_value)
        self.assertEqual(stat.n_tested,50)

    def test_limma_is_refit_inside_inner_folds(self):
        rng=np.random.default_rng(3);X=rng.uniform(.1,.9,(24,20));y=np.array([0,1]*12)
        seen=[];original=LimmaCpGSelector.fit
        def record(est,X,y):
            seen.append(len(X));return original(est,X,y)
        with patch.object(LimmaCpGSelector,'fit',record):
            search=make_search(1,top_k=5,cs=[.1,1],selector='limma',rscript=RSCRIPT)
            search.fit(X,y)
        self.assertEqual(seen.count(16),6)
        self.assertEqual(seen.count(24),1)

    def test_one_command_both_selectors_resume_and_explicit_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);data=d/'raw data';cohort=d/'cohort';data.mkdir();cohort.mkdir()
            patients=[];meta=[];clinical=[];fusion=[]
            rng=np.random.default_rng(17);raw={'TargetID':[f'cg{i:08d}' for i in range(20)]}
            for i in range(50):
                split='train_gold' if i<30 else 'validation' if i<36 else 'test' if i<42 else 'train_silver_candidate' if i<48 else 'prediction_only'
                label='UNKNOWN' if i>=42 else 'POS' if i%2 else 'NEG'
                if i == 48: label='NEG'
                subtype='Spindle' if i==30 else 'RMS NOS' if i in {36,48,49} else 'ERMS'
                patients.append(dict(subtype=subtype,patient_id=f'p{i}',sample_id=f's{i}',sentrix_id=f'a{i}',slide_id=f'i{i}',
                                     gold_label=label,split=split,teacher_train=i<30))
                meta.append({'Sentrix_ID':f'a{i}','UID.Subject':f'p{i}','Final.QC':'PASS','Sample.Type':'Patient tumor'})
                clinical.append({'Sample ID':f's{i}','X850k Tumor File Name':f'a{i}'})
                fusion.append({'Sample ID':f's{i}','Final Fusion Status':'' if label=='UNKNOWN' else label})
                raw[f'a{i}.Ave_Beta']=np.clip(rng.normal(.8 if i%2 else .2,.025,20),0,1)
                raw[f'a{i}.Detection.Pval']=np.zeros(20)
            pd.DataFrame(patients).to_csv(cohort/'patients.csv',index=False)
            pd.DataFrame(meta).to_csv(data/'RMSs_meta.txt',sep='\t',index=False)
            pd.DataFrame(clinical).to_excel(data/'COMET_RMS.xlsx',index=False)
            pd.DataFrame(fusion).to_excel(data/'COMET_RMS_fusion_status.xlsx',index=False)
            pd.DataFrame(raw).to_csv(data/'RMSs_beta.txt',sep='\t',index=False)
            env=os.environ.copy();env['CAPSTONE_USE_EXISTING_ENV']='1';env['CAPSTONE_PYTHON']=sys.executable
            command=['bash',str(ROOT/'run.sh'),'--data-dir',str(data),'--cohort',str(cohort),
                     '--out',str(d/'result'),'--top-k','5','--rscript',RSCRIPT]
            def run(extra=()):
                r=subprocess.run([*command,*extra],env=env,text=True,capture_output=True)
                self.assertEqual(r.returncode,0,r.stdout[-3000:]+r.stderr[-3000:])
                return r
            run()
            summary=pd.read_csv(d/'result/results_summary.csv')
            self.assertEqual(set(summary.selector),{'limma','variance'})
            self.assertNotIn('test',set(summary.split))
            self.assertEqual(len(summary),4)
            self.assertTrue((d/'result/return_results.zip').exists())
            features=pd.read_csv(d/'result/silver/paired_selected_cpg_beta_imputed.csv')
            self.assertEqual(features.shape,(48,8))
            student=pd.read_csv(d/'result/silver/student_train.csv')
            self.assertTrue(student.subtype.eq('ERMS').all())
            self.assertFalse(set(student.patient_id)&{'p30','p36','p48','p49'})
            validation=pd.read_csv(d/'result/limma/validation/validation_predictions.csv')
            self.assertIn('Spindle',set(validation.subtype))
            model=d/'result/limma/teacher/teacher.joblib';time=model.stat().st_mtime_ns
            run(['--resume','--include-test'])
            self.assertEqual(time,model.stat().st_mtime_ns)
            summary=pd.read_csv(d/'result/results_summary.csv');self.assertEqual(len(summary),6)
            self.assertEqual(len(summary[summary.split=='test']),2)
            self.assertFalse(json.loads((d/'result/teacher_selection.json').read_text())['holdouts_used_for_selection'])
            # A newly observed gold label cannot silently become silver.
            f=pd.DataFrame(fusion);f.loc[42,'Final Fusion Status']='POS';f.to_excel(data/'COMET_RMS_fusion_status.xlsx',index=False)
            r=subprocess.run([*command,'--resume'],env=env,text=True,capture_output=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('real label',r.stderr)


if __name__=='__main__': unittest.main()
