"""Patient leakage and nested preprocessing regression tests; synthetic data only."""
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PIPE = ROOT / "pipelines" / "silver_labels"
sys.path.insert(0, str(PIPE))
import prepare
import teacher
import numpy as np
import pandas as pd


class PreparationTests(unittest.TestCase):
    def row(self, uid, label="UNKNOWN", slide=True, bad=False):
        r = {k: "False" for k in prepare.CONFLICTS}
        r.update(patient_id=uid, fusion_status=label, **{"Sample ID": uid + "_sample"},
                 sentrix_id=uid+"_array", slide_id=uid+"_slide", subtype="ERMS", issues="",
                 is_patient_tumor="True", has_methylation_metadata="True", block_match="True",
                 live_he=str(slide), identity_conflict=str(bad),
                 methylation_fusion_eligible=str(label in prepare.GOLD and not bad))
        return r

    def test_patient_grouping_and_no_label_transfer(self):
        a = self.row("a", "POS")
        b = self.row("a")
        b["Sample ID"], b["sentrix_id"] = "a_repeat", "repeat_array"
        c = self.row("b", "NEG", slide=False)
        d = self.row("b")
        d["Sample ID"], d["sentrix_id"] = "b_second", "second_array"
        p, _ = prepare.audit([a,b,c,d,self.row("c")])
        self.assertEqual([r["category"] for r in p], ["paired_gold", "paired_cross_sample_review", "paired_silver_candidate"])
        self.assertEqual(len(p), 3)

    def test_conflict_never_becomes_silver(self):
        a=self.row("a", "POS");b=self.row("a", "NEG")
        b["Sample ID"], b["sentrix_id"] = "other", "other_array"
        p,_=prepare.audit([a,b])
        self.assertEqual(p[0]["category"], "label_conflict_review")
        self.assertFalse(p[0]["teacher_eligible"])

    def test_splits_stable_disjoint_and_all_arrays_follow_patient(self):
        rows=[self.row(f"p{i}", "POS" if i<34 else "NEG") for i in range(59)]
        rows.append(self.row("silver"))
        p,_=prepare.audit(rows);a=prepare.allocate(copy.deepcopy(p));b=prepare.allocate(list(reversed(copy.deepcopy(p))))
        self.assertEqual({r['patient_id']:r['split'] for r in a},{r['patient_id']:r['split'] for r in b})
        self.assertEqual(sum(r['split']=='train_gold' for r in a),19)
        self.assertFalse(any(r['teacher_train'] for r in a if r['split'] in ['test','validation']))


class TeacherTests(unittest.TestCase):
    def test_inner_preprocessing_refitted_and_outer_unseen(self):
        rng=np.random.default_rng(42);X=rng.uniform(.1,.9,(24,12));y=np.array([0,1]*12)
        seen=[];original=teacher.FoldCpGSelector.fit
        def record(est,X,y=None):
            seen.append(len(X));return original(est,X,y)
        with patch.object(teacher.FoldCpGSelector,'fit',record):
            search=teacher.make_search(1,top_k=3,cs=[.1,1])
            search.fit(X,y)
        self.assertEqual(seen.count(16),6)
        self.assertEqual(seen.count(24),1)
        means=search.best_estimator_.named_steps['scale'].mean_.copy()
        search.predict_proba(np.ones((4,12))*100)
        np.testing.assert_array_equal(means,search.best_estimator_.named_steps['scale'].mean_)

    def test_raw_to_nested_teacher_to_silver(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);cohort=d/'cohort';cohort.mkdir()
            rows=[]
            for i in range(48):
                split='train_gold' if i<30 else 'validation' if i<36 else 'test' if i<42 else 'train_silver_candidate'
                rows.append(dict(subtype='ERMS',patient_id=f'p{i}',sample_id=f's{i}',sentrix_id=f'a{i}',slide_id=f'i{i}',
                                 gold_label='UNKNOWN' if i>=42 else 'POS' if i%2 else 'NEG',split=split,
                                 teacher_train=i<30))
            pd.DataFrame(rows).to_csv(cohort/'patients.csv',index=False)
            rng=np.random.default_rng(17);raw={'TargetID':[f'cg{i:08d}' for i in range(16)]}
            for i in range(48):
                raw[f'a{i}.Ave_Beta']=np.clip(rng.normal(.8 if i%2 else .2,.025,16),0,1)
                raw[f'a{i}.Detection.Pval']=np.zeros(16)
            pd.DataFrame(raw).to_csv(d/'raw.tsv',sep='\t',index=False)
            def run(script,*args):
                result=subprocess.run([sys.executable,str(PIPE/script),*map(str,args)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            run('cache_matrix.py','--cohort',cohort,'--beta',d/'raw.tsv','--out',d/'matrix')
            run('teacher.py','fit','--cohort',cohort,'--matrix',d/'matrix','--out',d/'fit','--top-k',5)
            oof=pd.read_csv(d/'fit/teacher_oof.csv')
            self.assertEqual(len(oof),30)
            self.assertEqual(set(oof.patient_id),{f'p{i}' for i in range(30)})
            run('teacher.py','silver','--cohort',cohort,'--matrix',d/'matrix','--out',d/'silver','--model',d/'fit/teacher.joblib')
            silver=pd.read_csv(d/'silver/silver_labels.csv')
            self.assertEqual(len(silver),6)
            self.assertTrue(silver.gold_label.eq('UNKNOWN').all())
            train=pd.read_csv(d/'silver/student_train.csv')
            self.assertFalse(set(train.patient_id)&{f'p{i}' for i in range(30,42)})
            self.assertEqual(sum(train.label_source.eq('gold')),30)
            self.assertFalse((d/'fit/test_predictions.csv').exists())
            run('teacher.py','evaluate','--split','validation','--cohort',cohort,'--matrix',d/'matrix','--out',d/'validation','--model',d/'fit/teacher.joblib')
            self.assertEqual(json.loads((d/'validation/metrics.json').read_text())['n_patients'],6)
            # Mutating a split after preprocessing must be rejected, not silently accepted.
            manifest=pd.read_csv(cohort/'patients.csv');manifest.loc[30,'teacher_train']=True
            manifest.to_csv(cohort/'patients.csv',index=False)
            with self.assertRaisesRegex(ValueError,'different cohort'):
                teacher.load_inputs(cohort,d/'matrix')


if __name__ == '__main__':
    unittest.main()
