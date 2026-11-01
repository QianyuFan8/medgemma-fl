#!/usr/bin/env bash
# Run from repo root. Requires the existing paired data / exports.
set -euo pipefail
source .venv/bin/activate
export MEDGEMMA_ATTN_IMPLEMENTATION=sdpa TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
scope=${COMET_ADAPTER_SCOPE:-lora-only}
rounds=${COMET_ROUNDS:-5}; epochs=${COMET_EPOCHS:-2}
layout=${COMET_FL_LAYOUT:-iid}; method=${COMET_FL_METHOD:-fedavg}
split=${COMET_SPLIT:-validation}; context=${COMET_CONTEXT:-32768}
case "$scope" in legacy|lora-only) ;; *) exit 2;; esac
case "$layout" in iid|skew08) ;; *) exit 2;; esac
case "$method" in fedavg|fedlc|fedprox|scaffold) ;; *) exit 2;; esac
case "$split" in validation|test) ;; *) exit 2;; esac
[[ $rounds =~ ^[1-9][0-9]*$ && $epochs =~ ^[1-9][0-9]*$ ]] || exit 2
# A distinct tag is required when input images or methylation representation changes.
data_prefix=${COMET_DATA_PREFIX:-data/fixed849_disjoint}
input_tag=${COMET_INPUT_TAG:-top16}
[[ $input_tag =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
if [[ $data_prefix != data/fixed849_disjoint && $input_tag == top16 ]]; then
  echo 'Set a distinct COMET_INPUT_TAG for changed input exports'; exit 2
fi
mode=${1:?Use central-smoke, central-train, central-eval, fl-smoke, fl-train, fl-eval}
case "$mode" in
 central-smoke|central-train|central-eval)
  [[ $method == fedavg ]] || { echo 'Centralized control uses fedavg (no client correction)'; exit 2; }
  data=${data_prefix}_central; prefix=central; clients=1; gpus='[0]'; ranks=16;;
 fl-smoke|fl-train|fl-eval)
  data=${data_prefix}_${layout}; prefix=fl_${layout}_${method}; clients=3; gpus='[0],[0],[0]'; ranks=16,16,16;;
 *) exit 2;;
esac
python scripts/check_fusion_splits.py --prepared data/fixed849_disjoint_paired --export "$data"
root="runs/ablation_${scope}_${input_tag}_r${rounds}_e${epochs}"
mkdir -p "$root/logs"
case "$mode" in
 *-smoke|*-train)
  if [[ $mode == *-smoke ]]; then schedule=(--num_rounds 1 --max_steps 1); suffix=smoke
  else schedule=(--num_rounds "$rounds" --num_train_epochs "$epochs"); suffix=train; fi
  target="$root/${prefix}_${suffix}"
  [[ ! -e $target ]] || { echo "Existing run: $target; refusing overwrite"; exit 1; }
  git rev-parse HEAD > "$root/logs/${prefix}_${suffix}_commit.txt"
  git diff > "$root/logs/${prefix}_${suffix}_code.patch"
  python job.py --task subtype --adapter_scope "$scope" --attn_implementation sdpa \
    --max_seq_length "$context" --data_dir "$data/clients" --n_clients "$clients" \
    --gpu "$gpus" --site_lora_ranks "$ranks" --global_lora_rank 16 --sim_threads 1 \
    --fl_method "$method" --lora_aggregation naive --learning_rate 2e-4 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
    "${schedule[@]}" --workspace "$target" 2>&1 | tee "$root/logs/${prefix}_${suffix}.log"
  test -s "$target/medgemma/server/simulate_job/app_server/FL_global_model.pt" || {
    echo "No global checkpoint produced; inspect $target before continuing"; exit 1;
  }
  ;;
 *-eval)
  checkpoint="$root/${prefix}_train/medgemma/server/simulate_job/app_server/FL_global_model.pt"
  test -f "$checkpoint"
  python evaluate_subtype.py medgemma --data-dir "$data" --split "$split" \
    --model-path "$checkpoint" --max-seq-length "$context" \
    --output "$root/${prefix}_${split}" 2>&1 | tee "$root/logs/${prefix}_${split}.log"
  ;;
esac
