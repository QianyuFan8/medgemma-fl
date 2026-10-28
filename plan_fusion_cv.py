"""Create patient-disjoint outer CV manifests; run prepare separately for every fold."""
import argparse
import json
import random
from pathlib import Path
from prepare_comet_subtype import read_tsv, write_tsv, sha256_file


def make_folds(rows, folds=5, seed=20261027):
    if folds < 2 or len({r['patient_id'] for r in rows}) != len(rows):
        raise ValueError('Need >=2 folds and unique canonical patients')
    groups = {label: sorted([r for r in rows if r['diagnosis'] == label], key=lambda r:r['patient_id'])
              for label in ('NEG', 'POS')}
    if sum(map(len, groups.values())) != len(rows):
        raise ValueError('Only NEG/POS labels are supported')
    rng = random.Random(seed)
    for members in groups.values():
        if len(members) < max(folds, 6):
            raise ValueError('Too few patients per class for this fold plan')
        rng.shuffle(members)
    plans = []
    for fold in range(folds):
        plan = []
        for label, members in groups.items():
            test = [r for i,r in enumerate(members) if i % folds == fold]
            trainval = [r for i,r in enumerate(members) if i % folds != fold]
            random.Random(seed + fold + (1000 if label == 'POS' else 0)).shuffle(trainval)
            nval = max(1, round(len(trainval)*0.2))
            if len(trainval)-nval < 3:
                raise ValueError('Fewer than 3 training patients per class')
            for split, subset in [('test',test),('validation',trainval[:nval]),('train',trainval[nval:])]:
                plan.extend(dict(patient_id=r['patient_id'], diagnosis=label, split=split) for r in subset)
        plans.append(sorted(plan,key=lambda r:r['patient_id']))
    return plans


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir',required=True);p.add_argument('--output',required=True)
    p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--cohort',choices=['paired','methylation'],default='paired')
    a=p.parse_args();root=Path(a.audit_dir);out=Path(a.output)
    report=json.loads((root/'audit.json').read_text())
    if report.get('classes') != ['NEG','POS'] or report.get('patient_id_source') != 'RMSs_meta UID.Subject':
        raise ValueError('Regenerate a canonical fusion audit first')
    if sha256_file(root/'samples_audit.tsv') != report['samples_audit_sha256']:
        raise ValueError('Audit was modified; regenerate it')
    rows=[r for r in read_tsv(root/'samples_audit.tsv') if r['status']=='candidate_pending_label_review'
          and (a.cohort=='methylation' or r['slide_status']=='available')]
    plans=make_folds(rows,a.folds,a.seed)
    out.mkdir(parents=True,exist_ok=False)
    for i,plan in enumerate(plans,1):write_tsv(out/f'fold_{i:02d}.tsv',plan)
    (out/'plan.json').write_text(json.dumps(dict(cohort=a.cohort,folds=a.folds,seed=a.seed,
        patients=len(rows),audit_sha256=report['samples_audit_sha256'],
        note='Frozen cohort assignment. Run training-only QC/limma afresh per fold; review post-QC class counts.'),indent=2))

if __name__=='__main__':main()
