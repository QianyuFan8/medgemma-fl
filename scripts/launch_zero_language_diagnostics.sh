#!/usr/bin/env bash
# Run on the VM after git pull. Never stops another process or overwrites a run.
set -euo pipefail
ARM="${1:?Usage: bash scripts/launch_zero_language_diagnostics.sh top16|spatial16}"
case "$ARM" in top16|spatial16) ;; *) echo 'Unknown montage arm' >&2; exit 1 ;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd -P)"
DATA_ROOT="$REPO_ROOT/data/central_diagnostics_${ARM}_20261101"
test -x .venv/bin/python
source .venv/bin/activate
test -f "$DATA_ROOT/diagnostics_plan.json"
command -v flock >/dev/null
export HF_HOME=/home/shadeform/comet_storage/hf
test -d "$HF_HOME"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# Inherited lock keeps a second launch from creating another training queue.
exec 9>"$REPO_ROOT/.epoch_diagnostics.lock"
if ! flock -n 9; then
    echo 'A diagnostics queue is already running; refusing duplicate launch.' >&2
    exit 1
fi
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
if [[ -n "$GPU_PIDS" ]]; then
    echo 'GPU has an active compute process. Inspect nvidia-smi before starting.' >&2
    exit 1
fi
.venv/bin/python -c 'from transformers import TrainerCallback; from client import EpochAdapterCheckpoint; import torch; assert torch.cuda.is_available(); print("Preflight imports and CUDA passed")'
mkdir -p "$REPO_ROOT/runs"
RUN_ROOT="$REPO_ROOT/runs/zero_language_diagnostics_${ARM}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN_ROOT"
nohup bash -c '
    set +e
    .venv/bin/python -u run_zero_language_diagnostics.py --data-root "$1" --output "$2/full"
    result=$?
    printf "%s\n" "$result" > "$2/exit_code"
    printf "QUEUE_EXIT_CODE=%s\n" "$result"
    exit "$result"
' _ "$DATA_ROOT" "$RUN_ROOT" > "$RUN_ROOT/pipeline.log" 2>&1 < /dev/null &
QUEUE_PID=$!
printf '%s\n' "$QUEUE_PID" > "$RUN_ROOT/pipeline.pid"
printf '%s\n' "$RUN_ROOT" > "$REPO_ROOT/.zero_language_diagnostics_${ARM}.latest"
printf 'Started PID=%s\nRun directory: %s\n' "$QUEUE_PID" "$RUN_ROOT"
printf 'Log: tail -f "%s/pipeline.log"\n' "$RUN_ROOT"
printf 'The remote queue continues after SSH disconnect; keep the VM running.\n'
