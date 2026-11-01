#!/usr/bin/env bash
# Reuse existing montages and CpGs; create new canonical-patient client layouts.
# No SVS reads, downloads, training, or overwriting of completed-run inputs.
set -euo pipefail
source .venv/bin/activate
arm=${1:?Use top16 or spatial16}
case "$arm" in
 top16) montage=data/wsi/fixed849_disjoint_montages; old=data/fixed849_disjoint;;
 spatial16) montage=data/wsi/fixed849_spatial16_montages; old=data/fixed849_spatial16;;
 *) exit 2;;
esac
prefix="data/fixed849_${arm}_patient_v2"
for layout in central iid skew08; do
  [[ ! -e ${prefix}_${layout} ]] || { echo "Existing output: ${prefix}_${layout}"; exit 1; }
done
for layout in central iid skew08; do
  case "$layout" in
    central) clients=1; partition=iid;;
    iid) clients=3; partition=iid;;
    skew08) clients=3; partition=skew;;
  esac
  python prepare_comet_subtype.py export --prepared data/fixed849_disjoint_paired \
    --modality multimodal --n-clients "$clients" --partition "$partition" \
    --dominant-fraction 0.8 --seed 20261011 --montage-dir "$montage" --output "${prefix}_${layout}"
  python scripts/check_fusion_splits.py --prepared data/fixed849_disjoint_paired \
    --export "${prefix}_${layout}" --check-client-groups
  for split in train validation test; do
    cmp "${old}_${layout}/${split}.csv" "${prefix}_${layout}/${split}.csv"
  done
done
echo "Patient grouping verified; beta matrices and split records are unchanged."
echo "For subsequent runs: COMET_DATA_PREFIX=$prefix COMET_INPUT_TAG=${arm}_patient_v2"
