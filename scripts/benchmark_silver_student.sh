#!/usr/bin/env bash
# Sequential single/four-card smoke+timing runs from the SAME base model.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo 'Usage: bash scripts/benchmark_silver_student.sh {1|4} DATA_ROOT NEW_OUTPUT [lora-only|language-only]' >&2
  exit 2
fi
N="$1"; DATA="$2"; OUT="$3"; SCOPE="${4:-lora-only}"
[[ "$N" == 1 || "$N" == 4 ]] || { echo 'Choose 1 or 4 GPUs' >&2; exit 2; }
[[ "$SCOPE" == lora-only || "$SCOPE" == language-only ]] || exit 2
[[ ! -e "$OUT" ]] || { echo 'Output already exists; use a new path' >&2; exit 2; }
if [[ "$N" == 1 ]]; then export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=0,1,2,3; fi
python -m pipelines.medgemma_student verify --data "$DATA"
python - "$N" <<'PY'
import sys, torch
n=int(sys.argv[1])
if torch.cuda.device_count()!=n:raise SystemExit(f'Expected {n} visible CUDA GPUs')
for i in range(n):
    p=torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory/2**30:.1f} GiB')
    if p.major<8:raise SystemExit('BF16 benchmark requires Ampere or newer')
PY
# Separate preflight output; failure must occur before model training starts.
mkdir -p "$(dirname "$OUT")"
python - "$DATA" "${OUT}.processor.json" <<'PY'
import sys
from medgemma_fl.preprocessing.prepare_multiimage_diagnostics import processor_preflight
processor_preflight(sys.argv[1],sys.argv[2])
PY
# Fixed effective batch=4: single GPU accumulation=4; four GPUs accumulation=1.
# Default QLoRA + non-reentrant checkpointing are held constant in both runs.
torchrun --standalone --nproc_per_node="$N" train/centralized.py \
  --data-root "$DATA" --output "$OUT" --adapter_scope "$SCOPE" \
  --weighted_labels --per_device_train_batch_size 1 --effective_batch_size 4 \
  --max_steps 30 --learning_rate 5e-5 --seed 20261117 \
  --max_seq_length 32768 --attn_implementation sdpa \
  --no-local_eval --benchmark_timing \
  2>&1 | tee "${OUT}.log"
