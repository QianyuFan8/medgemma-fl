"""Join canonical patients, accepted labels, CpG panel, and available WSI assets."""
from collections import Counter
import json
import math
from pathlib import Path
from .common import digest, indexed, read_csv, write_csv, write_json
from pipelines.silver_labels.src.ridge_teacher.cohort.policy import training_excluded


def file_index(directories, suffix):
    out = {}
    for directory in directories:
        root = Path(directory).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f'Not a directory: {root}')
        # Ordered directories are explicit precedence, e.g. newest backup first.
        for p in sorted(root.rglob('*' + suffix)):
            out.setdefault(p.stem, str(p.resolve()))
    return out


def prepare_plan(returned, cohort, output, feature_dirs=(), slide_dirs=()):
    returned, cohort, output = map(lambda p: Path(p).resolve(), (returned, cohort, output))
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((returned/'silver/panel_provenance.json').read_text())
    if digest(cohort/'patients.csv') != provenance['cohort_sha256']:
        raise ValueError('Returned teacher and frozen cohort disagree')
    patients = indexed(read_csv(cohort/'patients.csv'))
    student = indexed(read_csv(returned/'silver/student_train.csv'))
    images = indexed(read_csv(returned/'silver/image_input_manifest.csv'))
    beta = indexed(read_csv(returned/'silver/paired_selected_cpg_beta_imputed.csv'))
    silver = indexed(read_csv(returned/'silver/silver_labels.csv'))
    selection = json.loads((returned/'teacher_selection.json').read_text())
    if selection['holdouts_used_for_selection']:
        raise ValueError('Teacher selection used holdouts')
    panel_file = returned/selection['selected_selector']/'teacher/selected_cpgs.txt'
    probes = panel_file.read_text().splitlines()
    if len(probes) != 1000 or len(set(probes)) != len(probes):
        raise ValueError('Expected the returned 1000-CpG panel')
    holdout = {pid for pid, p in patients.items() if p['split'] in ('validation', 'test')}
    if set(student) & holdout or set(images) != set(student) | holdout:
        raise ValueError('Image cohort coverage or train/holdout overlap')
    expected_gold = {pid for pid,p in patients.items() if p['split']=='train_gold'}
    expected_silver = {pid for pid,s in silver.items() if s['included'].lower()=='true'}
    if set(student) != expected_gold | expected_silver:
        raise ValueError('Student membership differs from gold plus accepted silver')
    features = file_index(feature_dirs, '.h5')
    slides = file_index(slide_dirs, '.svs')
    rows = []
    for pid, image in images.items():
        p = patients[pid]; b = beta[pid]
        for other in (image, b):
            if other['sample_id'] != p['sample_id'] or other['split'] != p['split']:
                raise ValueError('Sample/split changed in returned files')
        if image['slide_id'] != p['slide_id']:
            raise ValueError('Slide changed')
        values = [float(b[cg]) for cg in probes]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Invalid/improperly imputed CpG values')
        if pid in student:
            if training_excluded(p):
                raise ValueError('Non-ERMS/ARMS patient in training')
            s = student[pid]
            source = 'gold' if p['split']=='train_gold' else 'silver'
            if s['sample_id'] != p['sample_id'] or s['slide_id'] != p['slide_id'] or s['label_source'] != source:
                raise ValueError('Student identity/provenance changed')
            label = s['training_label']; weight = 1.0 if source=='gold' else .3
            if float(s['student_loss_weight']) != weight:
                raise ValueError('Unexpected loss weight')
            if source=='gold' and label != p['gold_label']:
                raise ValueError('Gold label changed')
            if source=='silver':
                score = silver[pid]
                if score['predicted_label'] != label or float(score['confidence']) < .9:
                    raise ValueError('Invalid silver selection')
            split = 'train'
        else:
            split, label, source, weight = p['split'], p['gold_label'], 'gold', 1.0
        if label not in ('POS','NEG'):
            raise ValueError('Invalid fusion label')
        rows.append(dict(patient_id=pid, canonical_uid=pid, sample_id=p['sample_id'],
                         slide_id=p['slide_id'], subtype=p['subtype'], split=split,
                         label_name=label, label_source=source, student_loss_weight=weight,
                         feature_file=features.get(p['slide_id'], ''),
                         slide_file=slides.get(p['slide_id'], ''), beta=values))
    indexed(rows, 'slide_id')
    source_files = [cohort/'patients.csv', panel_file, returned/'teacher_selection.json',
                    returned/'silver/panel_provenance.json']
    source_files += [returned/'silver'/name for name in ('student_train.csv','silver_labels.csv',
                       'image_input_manifest.csv','paired_selected_cpg_beta_imputed.csv')]
    counts = {s: dict(Counter(r['label_name'] for r in rows if r['split']==s)) for s in ('train','validation','test')}
    plan = dict(schema=1, probes=probes, rows=rows, counts=counts,
                source_sha256={str(p):digest(p) for p in source_files},
                selector_training='fixed epochs, training patients only; gold=1.0, silver=0.3',
                missing_features=sum(not r['feature_file'] for r in rows),
                missing_wsi=sum(not r['slide_file'] for r in rows))
    output.mkdir(parents=True)
    write_json(output/'plan.json',plan)
    write_csv(output/'asset_inventory.csv',[{k:v for k,v in r.items() if k!='beta'} for r in rows])
    write_csv(output/'missing_assets.csv',[dict(patient_id=r['patient_id'],slide_id=r['slide_id'],
              split=r['split'],missing_feature=not bool(r['feature_file']),missing_wsi=not bool(r['slide_file']))
              for r in rows])
    write_json(output/'summary.json',{k:v for k,v in plan.items() if k not in ('rows','probes','source_sha256')})
    return plan
