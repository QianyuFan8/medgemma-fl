# COMET RMS: FOXO1 fusion status from H&E montages and EPIC methylation

Labels are `Final Fusion Status` in `COMET_RMS_fusion_status.xlsx` (POS / NEG only),
joined to `COMET_RMS.xlsx` on Sample ID. Samples without a POS or NEG call are excluded.
All patient-level inputs, embeddings, montages, prompts, logs and checkpoints stay under
the ignored `data/`, `logs/` and `runs/` directories.

Pipeline (run from the repository root on the GPU VM):

1. Download the beta matrix, metadata and slides from DNAnexus.
2. `prepare_comet_subtype.py audit` joins mapping table, metadata, matrix header and slides.
3. `prepare_comet_subtype.py prepare` runs lenient or strict probe QC and the training-only
   limma top-0.1% panel for POS vs NEG (`methylation/prepare.R`).
4. `data_preprocessing/wsi_embed.py` tiles each slide (224 px at 0.5 um/px, CLAM-style tissue mask) and embeds
   tiles with H-optimus-1 (gated; H-optimus-0 is the same-spec fallback).
5. `abmil_montage.py train` fits gated ABMIL on training slides only; `montage` writes the
   16 highest-attention tiles of every slide as a 4 x 4 montage (896 x 896).
6. `prepare_comet_subtype.py export` writes a modality/partition layout: methylation, image
   or multimodal; 1 client (centralized) or 3 clients (IID or label-skewed).
7. MedGemma three levels: zero-shot (`evaluate_subtype.py medgemma` on the base model),
   centralized LoRA/QLoRA (`job.py --n_clients 1`), federated LoRA/QLoRA (`job.py --n_clients 3`).
8. `evaluate_subtype.py` reports ROC-AUC, PR-AUC, accuracy and balanced accuracy; client
   logs give runtime, peak memory and adapter bytes sent/received per round.

The `prepare` step decides the patient cohort and the CpG panel once. Montages for
validation/test slides depend only on the ABMIL model fitted on training slides.

## 0. Environment additions

```bash
source .venv/bin/activate
sudo apt-get install -y openslide-tools libopenslide0   # once, needs sudo
python -m pip install -r requirements-wsi.txt
python -c 'import openslide, h5py, timm; print("wsi deps ok")'
hf auth login      # MedGemma 1.5 and H-optimus-1 are both gated: accept terms on the same account
```

`bioptimus/H-optimus-1` is the default encoder. `bioptimus/H-optimus-0` (same tile
specification and normalization constants) and the ungated Lunit DINO ViT-S
(`--model hf-hub:1aurent/vit_small_patch16_224.lunit_dino`, ImageNet normalization) are
fallbacks while access is pending; embeddings from different encoders live in separate
directories and are never mixed.

## 1. Data on the VM

```bash
dx login
dx select --level VIEW project-JBXfX9Q0vQQYJ9k4BxGVyYzJ
mkdir -p data/raw/comet data/raw/comet_slides
dx download project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/RMSs_beta.txt -o data/raw/comet/RMS_beta.tsv
dx download project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/RMSs_meta.txt -o data/raw/comet/RMSs_meta.txt
dx ls project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/ | grep '\.svs$' > data/raw/comet/slide_list.txt
dx download project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/*.svs -o data/raw/comet_slides/   # ~181 GiB
```

Copy `COMET_RMS.xlsx` and `COMET_RMS_fusion_status.xlsx` to the VM. Do not commit them.

## 2. Audit

```bash
python prepare_comet_subtype.py audit \
  --mapping data/raw/comet/COMET_RMS.xlsx \
  --fusion data/raw/comet/COMET_RMS_fusion_status.xlsx \
  --comet-matrix data/raw/comet/RMS_beta.tsv \
  --comet-metadata data/raw/comet/RMSs_meta.txt \
  --slide-dir data/raw/comet_slides \
  --output data/manifests/comet_fusion_v1_audit
cat data/manifests/comet_fusion_v1_audit/audit.json
```

A row stays in the task only when Final Fusion Status is POS or NEG. Missing calls are
`fusion_not_called` and do not enter limma. Slide 8135 is `slide_shared_by_multiple_samples`.
One array per patient is kept (slide-bearing array preferred, then lexicographic ID).
Review `samples_audit.tsv` before confirming. An earlier ERMS-vs-ARMS panel cannot be reused.

## 3. Prepare the panel (lenient primary, strict sensitivity)

```bash
export R_LIBS_USER="$PWD/data/r-library"; unset R_HOME; mkdir -p logs; set -o pipefail
python prepare_comet_subtype.py prepare \
  --audit-dir data/manifests/comet_fusion_v1_audit \
  --output data/comet_fusion_lenient_v1 \
  --qc-mode lenient --top-fraction 0.001 --seed 20261011 \
  --confirm-reviewed-labels 2>&1 | tee logs/comet_fusion_lenient_v1_prepare.log
python prepare_comet_subtype.py prepare \
  --audit-dir data/manifests/comet_fusion_v1_audit \
  --output data/comet_fusion_strict_v1 \
  --qc-mode strict --top-fraction 0.001 --seed 20261011 \
  --confirm-reviewed-labels 2>&1 | tee logs/comet_fusion_strict_v1_prepare.log
cat data/comet_fusion_lenient_v1/selection_summary.json
```

- lenient: detection p > 0.01 masked, CpGs missing in > 5% of training patients dropped,
  training-median imputation (`max_probe_missing = 0.05`).
- strict: same masking, CpGs with any training missingness dropped (`max_probe_missing = 0`).
  Held-out failures are still median-imputed so every patient keeps the full panel.
- Ranking: limma moderated test of POS vs NEG on training M-values, top 0.1% of tested
  CpGs. Only training patients with a POS or NEG call enter the contrast. Splits are
  patient-level 60/20/20 with the same seed in both modes, so the two panels are
  directly comparable.
- `--require-slide` restricts the cohort to patients with an uploaded slide. The default
  keeps every POS/NEG patient with methylation and lets the image exports drop patients
  without a montage.

## 4. Tile and embed slides

```bash
python wsi_embed.py --slide-dir data/raw/comet_slides \
  --output data/wsi/hoptimus1_embeddings \
  --model hf-hub:bioptimus/H-optimus-1 --batch-size 128 --num-workers 8 \
  2>&1 | tee logs/wsi_embed_hoptimus1.log
cat data/wsi/hoptimus1_embeddings/embedding_log.json | head -20
```

If CLAM is preferred for segmentation, run its `create_patches_fp.py` at patch size 224
and pass `--coords-dir <CLAM_RESULTS>/patches` so only the coordinates come from CLAM.
Slides without microns-per-pixel metadata fail; pass `--assume-mpp` only for a known
scanner setting. `--max-tiles 4000` caps memory for very large slides.

## 5. ABMIL and montages

```bash
python data_preprocessing/abmil_montage.py train \
  --embeddings data/wsi/hoptimus1_embeddings \
  --prepared data/comet_fusion_lenient_v1 \
  --output runs/comet_subtype/abmil_hoptimus1_v1 --epochs 30 \
  2>&1 | tee logs/abmil_hoptimus1_v1.log
python data_preprocessing/abmil_montage.py montage \
  --embeddings data/wsi/hoptimus1_embeddings \
  --abmil runs/comet_subtype/abmil_hoptimus1_v1 \
  --slide-dir data/raw/comet_slides \
  --output data/wsi/montages_hoptimus1_v1
```

`training.json` records train/validation ROC-AUC of the ABMIL image baseline; it is a
report, not a model-selection signal. `slide_scores.tsv` is the H-optimus + ABMIL
reference result for Aim 1 (image-only). The montage of a slide is the same for every
MedGemma level and modality.

## 6. Export layouts

```bash
P=data/comet_fusion_lenient_v1; M=data/wsi/montages_hoptimus1_v1
python prepare_comet_subtype.py export --prepared $P --modality multimodal --n-clients 1 \
  --montage-dir $M --output data/comet_fusion_multimodal_central_v1
python prepare_comet_subtype.py export --prepared $P --modality multimodal --n-clients 3 --partition iid \
  --montage-dir $M --output data/comet_fusion_multimodal_iid_v1
python prepare_comet_subtype.py export --prepared $P --modality multimodal --n-clients 3 --partition skew \
  --montage-dir $M --output data/comet_fusion_multimodal_skew_v1
python prepare_comet_subtype.py export --prepared $P --modality image --n-clients 1 \
  --montage-dir $M --output data/comet_fusion_image_central_v1
python prepare_comet_subtype.py export --prepared $P --modality methylation --n-clients 1 \
  --output data/comet_fusion_meth_central_v1
cat data/comet_fusion_multimodal_iid_v1/site_counts.json
```

Image/multimodal exports keep only patients with a montage and copy each montage to
`images/<patient>.png`; `task.json` lists how many patients were dropped per split.
`--partition skew` gives site-1 and site-3 mostly NEG and site-2 mostly POS
(`--dominant-fraction 0.8`). Test patients never enter client folders.

User prompt (no identifiers, no label):

```text
Classify the FOXO1 fusion status of this pediatric rhabdomyosarcoma.

[H&E]
The attached image is a 4x4 montage of the 16 highest-attention tiles from one H&E whole-slide image (224 px at 0.5 um/px).

[DNA methylation]
cg00000001: 0.250000
...

What is the most likely fusion status?
Options: NEG, POS.
Reply with only POS or NEG.
```

## 7. Context check and the three MedGemma levels

```bash
export SUBTYPE_CONTEXT=12288
python check_methylation_context.py --data-dir data/comet_fusion_multimodal_iid_v1 --max-seq-length "$SUBTYPE_CONTEXT"
```

The check adds a 280-token reserve per image. Use the budget it reports for every run.

Baseline 1, zero-shot MedGemma (no training):

```bash
python evaluate_subtype.py medgemma --data-dir data/comet_fusion_multimodal_central_v1 \
  --model-path google/medgemma-1.5-4b-it --parse lenient --max-seq-length "$SUBTYPE_CONTEXT" \
  --output runs/comet_subtype/zero_shot_multimodal_validation_v1
```

Baseline 2, centralized LoRA/QLoRA (one client = pooled training):

```bash
python job.py --task subtype --data_dir data/comet_fusion_multimodal_central_v1/clients \
  --n_clients 1 --gpu "[0]" --num_rounds 3 --num_train_epochs 2 \
  --max_seq_length "$SUBTYPE_CONTEXT" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 --global_lora_rank 16 --site_lora_ranks 16 \
  --workspace runs/comet_subtype/central_multimodal_v1 2>&1 | tee logs/comet_central_multimodal_v1.log
```

Federated LoRA/QLoRA (same hyperparameters, IID then skew, each from the base model):

```bash
for layout in iid skew; do
  python job.py --task subtype --data_dir "data/comet_fusion_multimodal_${layout}_v1/clients" \
    --n_clients 3 --gpu "[0],[0],[0]" --num_rounds 3 --num_train_epochs 2 \
    --max_seq_length "$SUBTYPE_CONTEXT" --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 --global_lora_rank 16 --site_lora_ranks 16,16,16 \
    --workspace "runs/comet_subtype/fl_multimodal_${layout}_v1" 2>&1 | tee "logs/comet_fl_multimodal_${layout}_v1.log"
done
```

Start every configuration with `--num_rounds 1 --max_steps 1` as a smoke test. Three
clients on one GPU share memory; on a single A100 80GB the 4-bit base model is loaded
three times.

**Attention kernel.** The notebook default `eager` materialises a full
sequence x sequence attention matrix per layer; at ~17k tokens (800 CpGs plus an image)
that is 9 GiB per layer and three concurrent clients run out of memory on one 80 GB GPU.
Add `--attn_implementation sdpa` to every `job.py` command above and export
`MEDGEMMA_ATTN_IMPLEMENTATION=sdpa` before `evaluate_subtype.py`, so training and
evaluation use the same kernel. If memory is still short, `--sim_threads 1` runs the
clients one after another on the shared GPU, or prepare a smaller panel
(`--top-fraction 0.0005`, about 400 CpGs) under a new experiment name. Repeat the same commands with the image-only and methylation-only exports
to isolate each modality.

### 7.1 Non-IID methods beyond FedAvg

Under label skew the FedAvg global model keeps its ranking (ROC-AUC) but its decision
boundary drifts toward the majority class. Three client-side methods are available
through `job.py --fl_method`; the server aggregation stays FedAvg (`naive` or `hlora`):

| `--fl_method` | What changes locally | Extra flags | Notes |
| --- | --- | --- | --- |
| `fedprox` | loss += mu/2 · ‖w − w_global‖² over trainable parameters | `--fedprox_mu` (0.05) | cheapest; general drift control |
| `scaffold` | gradient correction (c − c_i); client control variate updated per round and averaged on the server | — | upload doubles (2 × adapter); requires uniform site ranks; AdamW makes the (x − y)/Σlr update heuristic |
| `fedlc` | logits of the class options at the first answer token reduced by tau · n_c^(−1/4) with local counts n_c (FedLC, Zhang et al. 2022) | `--fedlc_tau` (1.0) | targets label skew directly; cross-entropy restricted to answer tokens via backbone + lm_head |

Run each on the skew layout (and on iid as a control) with the same schedule, e.g.:

```bash
python job.py --task subtype --data_dir data/comet_fusion_multimodal_skew08_v1/clients \
  --n_clients 3 --gpu "[0],[0],[0]" --num_rounds 5 --num_train_epochs 2 \
  --max_seq_length "$SUBTYPE_CONTEXT" --attn_implementation sdpa \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 4 \
  --global_lora_rank 16 --site_lora_ranks 16,16,16 \
  --fl_method fedprox --fedprox_mu 0.05 \
  --workspace runs/comet_subtype/fl_multimodal_skew08_fedprox_v1 2>&1 | tee logs/comet_fl_multimodal_skew08_fedprox_v1.log
```

Smoke-test a new method with `--num_rounds 2 --max_steps 2` first: SCAFFOLD's control
variate is only exercised from the second round on. Evaluate with `evaluate_subtype.py`
as before; the checkpoint path pattern is unchanged.

## 8. Evaluation

```bash
python evaluate_subtype.py ridge --data-dir data/comet_fusion_multimodal_central_v1 \
  --output runs/comet_subtype/ridge_validation_v1
python evaluate_subtype.py medgemma --data-dir data/comet_fusion_multimodal_central_v1 \
  --model-path runs/comet_subtype/central_multimodal_v1/medgemma/server/simulate_job/app_server/FL_global_model.pt \
  --max-seq-length "$SUBTYPE_CONTEXT" --output runs/comet_subtype/central_multimodal_validation_v1
for layout in iid skew; do
  python evaluate_subtype.py medgemma --data-dir "data/comet_fusion_multimodal_${layout}_v1" \
    --model-path "runs/comet_subtype/fl_multimodal_${layout}_v1/medgemma/server/simulate_job/app_server/FL_global_model.pt" \
    --max-seq-length "$SUBTYPE_CONTEXT" --output "runs/comet_subtype/fl_multimodal_${layout}_validation_v1"
done
python evaluate_subtype.py compare --data-dir data/comet_fusion_multimodal_central_v1 \
  --ridge-predictions runs/comet_subtype/ridge_validation_v1/predictions.csv \
  --medgemma-predictions runs/comet_subtype/central_multimodal_validation_v1/predictions.csv \
  --output runs/comet_subtype/compare_central_validation_v1
```

`evaluation.json` holds ROC-AUC, PR-AUC (positive class POS), accuracy, balanced
accuracy, per-class recall, and the count of replies that are not exactly POS or NEG.
Those replies are wrong predictions; they are not relabeled as NEG.

```bash
python summarize_subtype_results.py --split validation
```

This prints two tables. The first is predictive performance. The second separates
computational cost (train time, GPU hours, peak GPU memory, and train time per site)
from communication (rounds, upload per round, download per round, and total traffic).
Both come from the client logs: `train_runtime_sec`, `cuda_peak_allocated_mb`,
`received adapter size`, and `sent updated adapter size`. Differential privacy is not
implemented. Run `--split test` only after every choice is frozen on validation.

## Developer checks

```bash
python -m unittest tests/test_comet_subtype.py -v
RUN_R_INTEGRATION=1 RSCRIPT=/opt/homebrew/bin/Rscript R_LIBRARY=../methylation_cpg_routes/.Rlib \
  python -m unittest tests/test_comet_subtype.py -v
```

The Python tests cover the audit join, patient deduplication, shared-slide exclusion,
modality exports, client validation, prompt content, metrics and parsing. The R test runs
real limma in lenient and strict modes on synthetic data. GPU training, the tokenizer and
WSI readers still need the VM smoke runs above.
