"""Patient-level ABMIL evaluation and pre-specified equal-weight late fusion.

No fitting or weight tuning on evaluation labels. Both scores must target POS.
Use same-fold ABMIL slide_scores.tsv and ridge predictions.csv.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from subtype_utils import subtype_metrics


def index_predictions(rows):
    out={}
    for row in rows:
        pid=row['patient_id'];score=float(row['score_POS'])
        if not pid or pid in out:raise ValueError('Missing or duplicate patient ID')
        if row['truth'] not in ('NEG','POS') or not math.isfinite(score) or not 0<=score<=1:
            raise ValueError('Expected verified NEG/POS labels and finite POS scores in [0,1]')
        out[pid]={**row,'score_POS':score}
    if not out:raise ValueError('Empty evaluation cohort')
    return out


def combine(image, methylation):
    a=index_predictions(image);b=index_predictions(methylation)
    if a.keys()!=b.keys():raise ValueError('Patient sets differ: do not silently intersect model predictions')
    result=[]
    for pid in sorted(a):
        if a[pid]['truth']!=b[pid]['truth']:raise ValueError('Conflicting reference labels')
        result.append(dict(patient_id=pid,truth=a[pid]['truth'],
                           score_POS=(a[pid]['score_POS']+b[pid]['score_POS'])/2))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--abmil-scores',required=True);p.add_argument('--ridge-predictions')
    p.add_argument('--split',choices=['validation','test'],default='test')
    p.add_argument('--output',required=True);a=p.parse_args()
    with open(a.abmil_scores) as f:rows=[r for r in csv.DictReader(f,delimiter='\t') if r['split']==a.split]
    rows=list(index_predictions(rows).values())
    if a.ridge_predictions:
        # evaluate_subtype writes this provenance next to predictions.csv.
        provenance=json.loads((Path(a.ridge_predictions).parent/'evaluation.json').read_text())
        if provenance.get('split')!=a.split or provenance.get('mode')!='ridge':
            raise ValueError('Ridge provenance must match requested evaluation split')
        with open(a.ridge_predictions) as f:rows=combine(rows,list(csv.DictReader(f)))
    for r in rows:r['prediction']='POS' if r['score_POS']>=0.5 else 'NEG'
    metrics=subtype_metrics([r['truth'] for r in rows],[r['prediction'] for r in rows],
                            [r['score_POS'] for r in rows],['NEG','POS'])
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    with (out/'predictions.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['patient_id','truth','prediction','score_POS'],extrasaction='ignore')
        w.writeheader();w.writerows(rows)
    (out/'evaluation.json').write_text(json.dumps(dict(mode='late_fusion' if a.ridge_predictions else 'abmil',
        split=a.split,threshold=0.5,image_weight=0.5 if a.ridge_predictions else 1.0,
        metrics=metrics,note='Equal-weight scores are not claimed to be calibrated probabilities.'),indent=2))

if __name__=='__main__':main()
