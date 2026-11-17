"""Export the new patient-disjoint cohort to the existing MedGemma input contract."""
import json
from pathlib import Path
import shutil
from .common import digest, load_plan, write_csv, write_json
from medgemma_fl.methylation_utils import panel_fingerprint
from medgemma_fl.methylation_validate import validate_clients
from medgemma_fl.subtype_utils import make_subtype_record, load_example_images
from medgemma_fl.preprocessing.prepare_multiimage_diagnostics import convert_record


def export_inputs(plan_path, tiles, output):
    plan=load_plan(plan_path);tiles=Path(tiles);output=Path(output)
    if output.exists():raise FileExistsError(output)
    tile_receipt=json.loads((tiles/'tiles.json').read_text())
    if tile_receipt['plan_sha256']!=digest(plan_path):raise ValueError('Tiles belong to another split')
    tile_rows={r['patient_id']:r for r in tile_receipt['rows']}
    if len(tile_rows)!=len(tile_receipt['rows']) or set(tile_rows)!={r['patient_id'] for r in plan['rows']}:
        raise ValueError('Tiles do not cover complete cohort')
    for row in plan['rows']:
        t=tile_rows[row['patient_id']]
        if t['slide_id']!=row['slide_id'] or digest(tiles/t['file'])!=t['image_sha256']:
            raise ValueError('Tile image identity/content changed')
    output.mkdir(parents=True);(output/'images').mkdir();client=output/'clients/site-1';client.mkdir(parents=True)
    probes=plan['probes'];panel_id=panel_fingerprint(probes)
    write_csv(output/'features.tsv',[{'probe':p} for p in probes],delimiter='\t')
    spec=dict(probes=probes,panel_id=panel_id,classes=['NEG','POS'],features_sha256=digest(output/'features.tsv'))
    write_json(output/'features.json',spec)
    for name in ('features.json','features.tsv'):shutil.copy2(output/name,client/name)
    records={s:[] for s in ('train','validation','test')};manifest=[]
    for row in plan['rows']:
        t=tile_rows[row['patient_id']];rel='images/'+t['file'];shutil.copy2(tiles/t['file'],output/rel)
        rec=make_subtype_record(row['patient_id'],row['label_name'],probes,row['beta'],
                               ['NEG','POS'],'multimodal',image=rel,site_id='site-1')
        rec=convert_record(rec,'multimodal')
        rec.update(canonical_uid=row['canonical_uid'],label_source=row['label_source'],
                   student_loss_weight=row['student_loss_weight'],image_selection='spatial')
        if len(load_example_images(rec,output))!=16:raise ValueError('Wrong image count')
        records[row['split']].append(rec)
        manifest.append(dict(patient_id=row['patient_id'],canonical_uid=row['canonical_uid'],
            diagnosis=row['label_name'],split=row['split'],site_id='site-1',subtype=row['subtype'],
            slide_id=row['slide_id'],label_source=row['label_source'],student_loss_weight=row['student_loss_weight']))
    write_csv(output/'splits.tsv',manifest,delimiter='\t');write_csv(output/'site_assignments.tsv',manifest,delimiter='\t')
    for s,rows in records.items():
        write_csv(output/f'{s}.csv',[r for r in manifest if r['split']==s])
        if s!='test':write_json(client/f'{s}.json',rows)
        if s!='train':write_json(output/f'{s}.json',rows)
    task=dict(task='comet_fusion',classes=['NEG','POS'],modality='multimodal',n_clients=1,
        sites=['site-1'],panel_mode='shared',panel_id=panel_id,probes=probes,
        features_sha256=spec['features_sha256'],image_layout='tiles16',
        student_loss={'reduction':'patient_mean_of_weighted_answer_token_mean', 'gold':1.,'silver':.3},
        training_regime='centralized',cohort='silver_v4',image_selection='spatial')
    write_json(output/'task.json',task)
    shutil.copy2(tiles/'tiles.json',output/'tiles.json')
    write_json(output/'student_inputs.json',dict(plan_sha256=digest(plan_path),counts=plan['counts'],
        train_patients=[r['patient_id'] for r in plan['rows'] if r['split']=='train'],
        source_sha256=plan['source_sha256'],
        content_sha256={str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}))
    return verify_export(output)


def verify_export(root):
    root=Path(root);receipt=json.loads((root/'student_inputs.json').read_text())
    for name,expected in receipt['content_sha256'].items():
        if digest(root/name)!=expected:raise ValueError(f'Prepared input changed: {name}')
    task=validate_clients(root/'clients',1,require_patient_groups=True)
    if task.get('student_loss')!={'reduction':'patient_mean_of_weighted_answer_token_mean','gold':1.,'silver':.3}:
        raise ValueError('Incorrect student weighting contract')
    for split in ('train','validation','test'):
        p=root/('clients/site-1/train.json' if split=='train' else f'{split}.json')
        rows=json.loads(p.read_text())
        for r in rows:
            source=r['label_source'];w=r['student_loss_weight']
            if source not in ('gold','silver') or w != (1. if source=='gold' else .3):
                raise ValueError('Invalid label-source weight')
            if split!='train' and source!='gold':raise ValueError('Silver label in holdouts')
    return receipt
