#!/usr/bin/env bash
# Existing SVS are required; no downloads or GPU training are performed here.
set -euo pipefail
source .venv/bin/activate
export COMET_SLIDE_DIR=${COMET_SLIDE_DIR:-data/raw/comet_slides}
python - <<'PYCHECK'
import csv, os
from pathlib import Path
with Path('data/fixed849_disjoint_paired/splits.tsv').open() as f:
    rows=list(csv.DictReader(f,delimiter='\t'))
missing=[r['slide_id'] for r in rows if not (Path(os.environ['COMET_SLIDE_DIR'])/(r['slide_id']+'.svs')).is_file()]
if missing: raise FileNotFoundError(f'Missing {len(missing)} SVS files: {missing[:8]}')
for path in ['data/wsi/fixed849_spatial16_montages','data/fixed849_spatial16_central',
             'data/fixed849_spatial16_iid','data/fixed849_spatial16_skew08']:
    if Path(path).exists(): raise FileExistsError(f'Existing output; inspect before rerunning: {path}')
PYCHECK
python data_preprocessing/abmil_montage.py montage \
  --embeddings data/wsi/hoptimus1 --abmil runs/fixed849_disjoint/abmil \
  --slide-dir "${COMET_SLIDE_DIR:-data/raw/comet_slides}" --selection spatial \
  --candidate-pool 128 --min-distance-tiles 2 \
  --output data/wsi/fixed849_spatial16_montages
for layout in central iid skew08; do
  case "$layout" in
    central) clients=1; partition=iid;;
    iid) clients=3; partition=iid;;
    skew08) clients=3; partition=skew;;
  esac
  python prepare_comet_subtype.py export --prepared data/fixed849_disjoint_paired \
    --modality multimodal --n-clients "$clients" --partition "$partition" \
    --dominant-fraction 0.8 --seed 20261011 \
    --montage-dir data/wsi/fixed849_spatial16_montages \
    --output "data/fixed849_spatial16_$layout"
  python scripts/check_fusion_splits.py --prepared data/fixed849_disjoint_paired \
    --export "data/fixed849_spatial16_$layout" --check-client-groups
done
python - <<'PY'
import csv, json
from pathlib import Path
for layout in ['central','iid','skew08']:
    old=Path(f'data/fixed849_disjoint_{layout}')
    new=Path(f'data/fixed849_spatial16_{layout}')
    for split in ['train','validation','test']:
        if (old/f'{split}.csv').read_bytes() != (new/f'{split}.csv').read_bytes():
            raise ValueError(f'Beta/row changes detected: {layout}/{split}')
    def assignments(p):
        with (p/'site_assignments.tsv').open() as f:
            return {(r['patient_id'],r['split'],r['site_id']) for r in csv.DictReader(f,delimiter='\t')}
    if assignments(old) != assignments(new):
        old_task=json.loads((old/'task.json').read_text())
        if old_task.get('client_assignment_version') == 2:
            raise ValueError(f'Canonical client assignments changed: {layout}')
        print(f'NOTE {layout}: legacy sample-level assignments differ; regenerate both arms with regroup_fusion_inputs.sh.')
    for p in new.rglob('*.json'):
        if p.name not in ['train.json','validation.json','test.json']: continue
        for r in json.loads(p.read_text()):
            if r.get('image_selection') != 'spatial':
                raise ValueError(f'Spatial prompt metadata missing: {p}')
    print(f'PASS {layout}: same beta rows and splits; canonical grouping and spatial prompts verified.')
PY
