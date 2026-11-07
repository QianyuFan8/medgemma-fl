#!/usr/bin/env bash
# Start two centralized 60-epoch multimodal sixteen-image experiments; never stops another job.
set -euo pipefail
ARM="${1:?Usage: bash scripts/launch_multimodal_tiles60.sh top16|spatial16}"
case "$ARM" in top16|spatial16) ;; *) echo 'Unknown montage arm' >&2; exit 1 ;; esac
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd -P)"
DATA_ROOT="$REPO_ROOT/data/central_diagnostics_${ARM}_20261101"
test -x .venv/bin/python
source .venv/bin/activate
test -f "$DATA_ROOT/diagnostics_plan.json"
export HF_HOME=/home/shadeform/comet_storage/hf
test -d "$HF_HOME"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
command -v flock >/dev/null
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
.venv/bin/python -c 'from run_multimodal_tiles60 import EPOCHS; import torch; assert torch.cuda.is_available(); print("Snapshot epochs:", EPOCHS)'
test -f "$DATA_ROOT/multimodal/task.json"
mkdir -p "$REPO_ROOT/runs"
RUN_ROOT="$REPO_ROOT/runs/multimodal_tiles60_${ARM}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN_ROOT"
nohup bash -c '
    set +e
    .venv/bin/python -u run_multimodal_tiles60.py --data-root "$1" --output "$2/full" --arm "$3"
    result=$?
    printf "%s\n" "$result" > "$2/exit_code"
    printf "QUEUE_EXIT_CODE=%s\n" "$result"
    exit "$result"
' _ "$DATA_ROOT" "$RUN_ROOT" "$ARM" > "$RUN_ROOT/pipeline.log" 2>&1 < /dev/null &
QUEUE_PID=$!
printf '%s\n' "$QUEUE_PID" > "$RUN_ROOT/pipeline.pid"
printf '%s\n' "$RUN_ROOT" > "$REPO_ROOT/.multimodal_tiles60_${ARM}.latest"
printf 'Started PID=%s\nRun directory: %s\n' "$QUEUE_PID" "$RUN_ROOT"
printf 'Log: tail -f "%s/pipeline.log"\n' "$RUN_ROOT"
printf 'The queue survives SSH disconnection. Keep the remote VM running.\n'
