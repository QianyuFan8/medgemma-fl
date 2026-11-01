#!/usr/bin/env bash
# One input arm, all five models, sequentially on one GPU.
set -euo pipefail
input=${1:?Use top16 or spatial16}; phase=${2:?Use smoke, train, or eval}
case "$input" in
 top16) export COMET_DATA_PREFIX=data/fixed849_disjoint;;
 spatial16) export COMET_DATA_PREFIX=data/fixed849_spatial16;;
 *) exit 2;;
esac
case "$phase" in smoke|train|eval) ;; *) exit 2;; esac
export COMET_INPUT_TAG=$input COMET_ADAPTER_SCOPE=lora-only
export COMET_ROUNDS=5 COMET_EPOCHS=2
export COMET_SPLIT=${COMET_SPLIT:-validation}
COMET_FL_METHOD=fedavg bash scripts/run_fusion_ablation.sh "central-$phase"
for arm in iid:fedavg iid:fedlc skew08:fedavg skew08:fedlc; do
  COMET_FL_LAYOUT=${arm%:*} COMET_FL_METHOD=${arm#*:} \
    bash scripts/run_fusion_ablation.sh "fl-$phase"
done
