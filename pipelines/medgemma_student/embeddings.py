"""Generate missing frozen H-Optimus features, optionally across disjoint GPUs."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .common import load_plan, digest, write_json


def embed_missing(plan_path, output, device='cpu', gpu_ids=None, resume=False):
    plan=load_plan(plan_path);output=Path(output).resolve()
    ids=gpu_ids.split(',') if gpu_ids else []
    if ids and (len(set(ids))!=len(ids) or any(not re.fullmatch(r'\d+',i) for i in ids)):
        raise ValueError('GPU IDs must be distinct nonnegative integers')
    rows=[]
    for row in plan['rows']:
        if row['feature_file']:continue
        if not row['slide_file'] or not Path(row['slide_file']).is_file():
            raise FileNotFoundError(f"WSI not supplied for {row['slide_id']}")
        rows.append({'slide_id':row['slide_id'],'slide_file':row['slide_file']})
    if not rows:raise ValueError('No missing features; skip embedding stage')
    configuration=dict(plan_sha256=digest(plan_path),gpu_ids=ids,device=device,
                       model='hf-hub:bioptimus/H-optimus-1',tile=224,target_mpp=.5,
                       max_tiles=4000,sampling_seed=20261011,sat_thresh=8,tissue_frac=.5,
                       normalization='bioptimus')
    if output.exists():
        if not resume or json.loads((output/'embedding_run.json').read_text())!=configuration:
            raise ValueError('Existing embedding run differs; resume only with identical plan/devices')
    else:
        output.mkdir(parents=True);write_json(output/'embedding_run.json',configuration)
    jobs=max(1,len(ids));repo=Path(__file__).resolve().parents[2]
    def run(rank):
        group=rows[rank::jobs]
        if not group:return
        worklist=output/f'worklist-{rank}.json';write_json(worklist,group)
        env=os.environ.copy()
        if ids:env['CUDA_VISIBLE_DEVICES']=ids[rank]
        with (output/f'worker-{rank}.log').open('a') as log:
            subprocess.run([sys.executable,'-m','pipelines.medgemma_student.embedding_worker',
                '--worklist',str(worklist),'--output',str(output/f'worker-{rank}'),
                '--device','cuda' if ids else device],check=True,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(run,range(jobs)))
    write_json(output/'completed.json',dict(**configuration,slides=len(rows)))
