#!/usr/bin/env bash
# One scope, four federated configurations, sequentially. User starts this queue.
set -Eeuo pipefail
trap 'echo "Launch failed at line $LINENO: $BASH_COMMAND" >&2' ERR
fail() { echo "ERROR: $*" >&2; exit 1; }
SCOPE="${1:?Usage: bash scripts/launch_federated_tiles16.sh language-only|lora-only ROUNDS LOCAL_EPOCHS}"
ROUNDS="${2:?Specify the number of federated rounds}"
LOCAL_EPOCHS="${3:?Specify local epochs per round}"
shift 3
# Remaining CLI flags configure hardware/performance without changing the queue.
[[ "$SCOPE" == language-only || "$SCOPE" == lora-only ]] || fail 'Invalid adapter scope'
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ && "$LOCAL_EPOCHS" =~ ^[1-9][0-9]*$ ]] || fail 'Training budgets must be positive integers'
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd -P)"
[[ -x .venv/bin/python ]] || fail 'Missing .venv/bin/python'
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/shadeform/comet_storage/hf}"
[[ -d "$HF_HOME" ]] || fail "Missing model cache: $HF_HOME"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
command -v flock >/dev/null
exec 9>"$REPO_ROOT/.epoch_diagnostics.lock"
flock -n 9 || fail 'Another experiment queue holds the GPU lock'
GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)"
[[ -z "$GPU_PIDS" ]] || fail 'GPU is busy; inspect nvidia-smi before starting'
DATA_ROOT="$REPO_ROOT/data/federated_spatial16_v1"
if [[ ! -d "$DATA_ROOT" ]]; then
    [[ -f .multimodal_tiles60_spatial16.latest ]] || fail 'Missing spatial source-data pointer'
    SOURCE_RUN="$(cat .multimodal_tiles60_spatial16.latest)"
    .venv/bin/python prepare_federated_tiles16.py --source "$SOURCE_RUN/full/data" --output "$DATA_ROOT"
fi
.venv/bin/python - "$DATA_ROOT" <<'PY'
import sys, torch
from pathlib import Path
from prepare_federated_tiles16 import verify
for layout in ('iid', 'skew08'):
    verify(Path(sys.argv[1])/layout)
assert torch.cuda.is_available(), 'CUDA unavailable'
print('Verified both patient-grouped spatial input layouts.')
PY
mkdir -p runs
RUN_ROOT="$REPO_ROOT/runs/federated_tiles16_${SCOPE}_r${ROUNDS}_e${LOCAL_EPOCHS}_$(date +%Y%m%d_%H%M%S)_$$"
.venv/bin/python run_federated_tiles16.py --data-root "$DATA_ROOT" --scope "$SCOPE" \
    --rounds "$ROUNDS" --local-epochs "$LOCAL_EPOCHS" --output "$RUN_ROOT/full" --validate-only "$@"
mkdir "$RUN_ROOT"
nohup bash -c '
    set +e
    data_root="$1"; scope="$2"; rounds="$3"; epochs="$4"; run_root="$5"
    shift 5
    .venv/bin/python -u run_federated_tiles16.py --data-root "$data_root" --scope "$scope" \
        --rounds "$rounds" --local-epochs "$epochs" --output "$run_root/full" "$@"
    result=$?
    printf "%s\n" "$result" > "$run_root/exit_code"
    printf "QUEUE_EXIT_CODE=%s\n" "$result"
    exit "$result"
' _ "$DATA_ROOT" "$SCOPE" "$ROUNDS" "$LOCAL_EPOCHS" "$RUN_ROOT" "$@" > "$RUN_ROOT/pipeline.log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$RUN_ROOT/pipeline.pid"
printf '%s\n' "$RUN_ROOT" > .federated_tiles16.latest
printf 'Started four-model queue. Scope: %s; rounds: %s; local epochs: %s\n' "$SCOPE" "$ROUNDS" "$LOCAL_EPOCHS"
printf 'Output: %s\nLog: tail -f "%s/pipeline.log"\n' "$RUN_ROOT" "$RUN_ROOT"
printf 'The queue survives SSH disconnection. Keep this VM running.\n'
