#!/usr/bin/env bash
# Fixed-checkpoint validation inference only. Never starts training or stops jobs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd -P)"
SOURCE_RUN="${1:-}"
if [[ -z "$SOURCE_RUN" ]]; then
    test -f .multimodal_tiles60_spatial16.latest
    SOURCE_RUN="$(cat .multimodal_tiles60_spatial16.latest)"
fi
EPOCH="${2:-60}"
test -x .venv/bin/python
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/shadeform/comet_storage/hf}"
export COMET_SLIDE_DIR="${COMET_SLIDE_DIR:-data/raw/comet_slides}"
test -d "$HF_HOME"
test -d "$COMET_SLIDE_DIR"
test -f data/wsi/fixed849_spatial16_montages/montage_index.tsv
test -d data/wsi/hoptimus1
test -d runs/fixed849_disjoint/abmil/attention
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
command -v flock >/dev/null
exec 9>"$REPO_ROOT/.epoch_diagnostics.lock"
if ! flock -n 9; then
    echo 'A diagnostics queue holds the lock; refusing another GPU job.' >&2
    exit 1
fi
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$GPU_PIDS" ]]; then
    echo 'GPU is busy. Inspect nvidia-smi before launching.' >&2
    exit 1
fi
.venv/bin/python -c 'import torch, h5py, openslide; assert torch.cuda.is_available()'
mkdir -p runs
RUN_ROOT="$REPO_ROOT/runs/spatial20_inference_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN_ROOT"
nohup bash -c '
    set +e
    .venv/bin/python -u run_spatial20_inference.py --source-run "$1" --epoch "$2" \
        --slide-dir "$3" --output "$4/full"
    result=$?
    printf "%s\n" "$result" > "$4/exit_code"
    printf "QUEUE_EXIT_CODE=%s\n" "$result"
    exit "$result"
' _ "$SOURCE_RUN" "$EPOCH" "$COMET_SLIDE_DIR" "$RUN_ROOT" > "$RUN_ROOT/pipeline.log" 2>&1 < /dev/null &
QUEUE_PID=$!
printf '%s\n' "$QUEUE_PID" > "$RUN_ROOT/pipeline.pid"
printf '%s\n' "$RUN_ROOT" > "$REPO_ROOT/.spatial20_inference.latest"
printf 'Started PID=%s\nOutput: %s\n' "$QUEUE_PID" "$RUN_ROOT"
printf 'Log: tail -f "%s/pipeline.log"\n' "$RUN_ROOT"
printf 'SSH may disconnect. Keep montage-new running. This queue does not train a model.\n'
