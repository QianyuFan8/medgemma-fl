#!/usr/bin/env bash
# Fixed-checkpoint validation inference only. Never starts training or stops jobs.
set -Eeuo pipefail
trap 'echo "Launch failed at line $LINENO: $BASH_COMMAND" >&2' ERR
fail() { echo "ERROR: $*" >&2; exit 1; }
cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO_ROOT="$(pwd -P)"
SOURCE_RUN="${1:-}"
if [[ -z "$SOURCE_RUN" ]]; then
    [[ -f .multimodal_tiles60_spatial16.latest ]] || fail 'Missing spatial run pointer; provide source run as argument 1.'
    SOURCE_RUN="$(cat .multimodal_tiles60_spatial16.latest)"
fi
EPOCH="${2:-60}"
SCOPE="${3:-lora-only}"
[[ "$SCOPE" == lora-only || "$SCOPE" == language-only ]] || fail 'Scope must be lora-only or language-only.'
PREPARED="${4:-${SPATIAL20_BUNDLE:-data/spatial20_validation_bundle}}"
[[ -x .venv/bin/python ]] || fail 'Missing Python environment: .venv/bin/python'
[[ -d "$SOURCE_RUN" ]] || fail "Missing spatial source run: $SOURCE_RUN"
source .venv/bin/activate
export HF_HOME="${HF_HOME:-/home/shadeform/comet_storage/hf}"
[[ -d "$HF_HOME" ]] || fail "Missing model cache: $HF_HOME"
if [[ -f "$PREPARED/prepared_inputs.json" ]]; then
    INPUT_ARGS=(--prepared-inputs "$PREPARED")
    echo "Using verified spatial tile bundle: $PREPARED"
else
    [[ $# -lt 4 && -z "${SPATIAL20_BUNDLE:-}" ]] || fail "Missing prepared bundle: $PREPARED"
    if [[ -z "${COMET_SLIDE_DIR:-}" ]]; then
        for candidate in /home/shadeform/comet_storage/comet_slides data/raw/comet_slides; do
            if [[ -d "$candidate" ]]; then COMET_SLIDE_DIR="$candidate"; break; fi
        done
    fi
    [[ -d "${COMET_SLIDE_DIR:-}" ]] || fail 'Raw slides not found. Set COMET_SLIDE_DIR to the actual SVS directory.'
    for required in data/wsi/fixed849_spatial16_montages/montage_index.tsv data/wsi/hoptimus1 runs/fixed849_disjoint/abmil/attention; do
        [[ -e "$required" ]] || fail "Missing preparation input: $required"
    done
    INPUT_ARGS=(--slide-dir "$COMET_SLIDE_DIR")
    echo "Using raw slides: $COMET_SLIDE_DIR"
fi
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
.venv/bin/python -c 'import torch; assert torch.cuda.is_available()'
if [[ "${INPUT_ARGS[0]}" == --slide-dir ]]; then
    .venv/bin/python -c 'import h5py, openslide'
fi
mkdir -p runs
RUN_ROOT="$REPO_ROOT/runs/spatial20_inference_${SCOPE}_$(date +%Y%m%d_%H%M%S)_$$"
mkdir "$RUN_ROOT"
nohup bash -c '
    set +e
    source_run=$1; epoch=$2; scope=$3; run_root=$4
    shift 4
    .venv/bin/python -u run_spatial20_inference.py --source-run "$source_run" --epoch "$epoch" \
        --adapter-scope "$scope" --output "$run_root/full" "$@"
    result=$?
    printf "%s\n" "$result" > "$run_root/exit_code"
    printf "QUEUE_EXIT_CODE=%s\n" "$result"
    exit "$result"
' _ "$SOURCE_RUN" "$EPOCH" "$SCOPE" "$RUN_ROOT" "${INPUT_ARGS[@]}" > "$RUN_ROOT/pipeline.log" 2>&1 < /dev/null &
QUEUE_PID=$!
printf '%s\n' "$QUEUE_PID" > "$RUN_ROOT/pipeline.pid"
printf '%s\n' "$RUN_ROOT" > "$REPO_ROOT/.spatial20_inference.latest"
printf 'Started PID=%s\nOutput: %s\n' "$QUEUE_PID" "$RUN_ROOT"
printf 'Log: tail -f "%s/pipeline.log"\n' "$RUN_ROOT"
printf 'SSH may disconnect. Keep this VM running. This queue does not train a model.\n'
