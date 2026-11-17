#!/usr/bin/env bash
# Run inside the portable bundle's repo. No training/evaluation is auto-started.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash scripts/prepare_student_vm.sh /absolute/large-disk/work 0,1,2,3' >&2
  exit 2
fi
WORK="$1"; GPUS="$2"
[[ "$WORK" == /* ]] || { echo 'Use an absolute work directory on the large disk' >&2; exit 2; }
BUNDLE="$(cd .. && pwd)"
INPUT="$BUNDLE/inputs"
python scripts/verify_student_bundle.py "$BUNDLE"
python - "$GPUS" <<'PY'
import sys,dxpy,h5py,torch,timm,openslide
ids=[int(i) for i in sys.argv[1].split(',')]
if len(set(ids))!=len(ids) or not ids or min(ids)<0 or max(ids)>=torch.cuda.device_count():
    raise SystemExit('Requested GPU IDs are not available')
print('GPU count:',torch.cuda.device_count())
PY
dx whoami >/dev/null
mkdir -p "$WORK"
export HF_HOME="${HF_HOME:-$WORK/hf}"
python - <<'PY'
from huggingface_hub import hf_hub_download
# Fail on missing gated-model access before downloading hundreds of GB of WSI.
for model in ('bioptimus/H-optimus-1','google/medgemma-1.5-4b-it'):
    hf_hub_download(model,'config.json')
print('Model config access verified; weights will download when needed')
PY
IMAGE_MANIFEST="$INPUT/returned/silver/image_input_manifest.csv"
if [[ ! -f "$WORK/cloud_files.json" ]]; then
  echo 'PHASE: read-only cloud inventory'
  python -m pipelines.medgemma_student dx-inventory --image-manifest "$IMAGE_MANIFEST" --output "$WORK/cloud_files.json"
fi
echo 'PHASE: download required WSI with dx (resume supported)'
python -m pipelines.medgemma_student dx-download --cloud-manifest "$WORK/cloud_files.json" \
  --image-manifest "$IMAGE_MANIFEST" --output "$WORK/wsi"
if [[ ! -e "$WORK/initial_plan" ]]; then
  python -m pipelines.medgemma_student plan --returned "$INPUT/returned" --cohort "$INPUT/cohort" \
    --feature-dir "$BUNDLE/features/hoptimus1" --slide-dir "$WORK/wsi" --output "$WORK/initial_plan"
fi
echo 'PHASE: missing H-Optimus-1 features (logs: new_features/worker-*.log)'
python -m pipelines.medgemma_student embed-missing --plan "$WORK/initial_plan/plan.json" \
  --output "$WORK/new_features" --gpu-ids "$GPUS" --resume
if [[ ! -e "$WORK/complete_plan" ]]; then
  python -m pipelines.medgemma_student plan --returned "$INPUT/returned" --cohort "$INPUT/cohort" \
    --feature-dir "$WORK/new_features" --feature-dir "$BUNDLE/features/hoptimus1" \
    --slide-dir "$WORK/wsi" --output "$WORK/complete_plan"
fi
if [[ ! -f "$WORK/abmil/selector.json" ]]; then
  echo 'PHASE: new train-only weighted ABMIL (fixed 50 epochs)'
  CUDA_VISIBLE_DEVICES="${GPUS%%,*}" python -m pipelines.medgemma_student fit-selector \
    --plan "$WORK/complete_plan/plan.json" --output "$WORK/abmil" --device cuda
fi
if [[ ! -f "$WORK/tiles/tiles.json" ]]; then
  echo 'PHASE: strict spatial16 extraction'
  python -m pipelines.medgemma_student extract --plan "$WORK/complete_plan/plan.json" \
    --selector "$WORK/abmil" --output "$WORK/tiles"
fi
if [[ ! -f "$WORK/medgemma_inputs/student_inputs.json" ]]; then
  echo 'PHASE: portable 16-independent-image training inputs'
  python -m pipelines.medgemma_student export --plan "$WORK/complete_plan/plan.json" \
    --tiles "$WORK/tiles" --output "$WORK/medgemma_inputs"
fi
python -m pipelines.medgemma_student verify --data "$WORK/medgemma_inputs"
printf 'READY_FOR_BENCHMARK: %s\n' "$WORK/medgemma_inputs"
