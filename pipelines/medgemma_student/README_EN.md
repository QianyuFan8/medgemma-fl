# Gold/silver MedGemma student

Use the [VM runbook](VM_DNANEXUS.md) for installation, DNAnexus authentication, data preparation, one/four-GPU benchmarks, and deployment.

## Frozen experiment definition

- Training: 91 patients, comprising 19 gold and 72 accepted silver labels. Validation and test each contain 20 gold-labeled patients.
- All training patients must be ERMS/ARMS. Other subtypes remain outside fitting.
- Use the returned LIMMA teacher's fixed 1,000-CpG panel. Do not fit feature selection on holdout patients.
- Freeze H-Optimus-1. Reuse only pretrained encoder features, not earlier supervised ABMIL weights or attention scores.
- Fit a fresh gated ABMIL on training patients only: 50 fixed epochs, Adam LR 1e-4, weight decay 1e-5, seed 20261117. Gold/silver loss weights are 1.0/0.3.
- Select 16 tiles by attention ranking with origin-to-origin Euclidean distance at least two tile widths. Do not relax spacing or duplicate tiles to fill a shortfall.
- Tiles are stored losslessly in a PNG container, then split by the loader into 16 independent images. CpG text appears once per patient.
- Student loss is the mean across patients of `label_weight * mean_answer_token_CE`. Do not divide by the batch's sum of weights; that would cancel the silver weight at batch size one.

## Data preparation

`python -m pipelines.medgemma_student --help` lists stages:

1. `dx-inventory` and `dx-download`: freeze cloud file IDs and verify downloads.
2. `plan`: verify identities, splits, panel provenance, and available image assets.
3. `embed-missing`: encode missing WSI on independent GPU workers.
4. `fit-selector`: train new ABMIL without holdout labels.
5. `extract`: select and extract strict spatial16 tiles.
6. `export` and `verify`: create self-contained weighted MedGemma inputs.

The current private bundle contains 48 reusable feature bags; another 83 must be encoded. All 131 original WSI are required for fresh tile extraction. Missing assets cause a failure rather than silently dropping patients.

## Training

Run the single-GPU and four-GPU 30-step benchmarks sequentially using `scripts/benchmark_silver_student.sh`. Both use effective batch size four and the same base model. The default scope `lora-only` means vision plus language LoRA. Use `language-only` explicitly for the alternative scope.

After hardware validation, the full training example is:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train/centralized.py \
  --data-root "$STUDENT_WORK/medgemma_inputs" --output "$STUDENT_WORK/student_vl_60epochs" \
  --adapter_scope lora-only --weighted_labels \
  --per_device_train_batch_size 1 --effective_batch_size 4 \
  --num_train_epochs 60 --save_adapter_epochs 2 5 10 20 30 50 60 \
  --learning_rate 5e-5 --seed 20261117 --max_seq_length 32768 \
  --attn_implementation sdpa --no-local_eval
```

Run inside tmux and use a new output directory. Candidate-answer validation/test scoring is a separate step. CPU tests and exported-job checks do not establish CUDA performance or four-GPU speedup.
