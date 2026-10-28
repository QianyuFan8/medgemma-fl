# Fusion experiment: corrected patient linkage and evaluation

These changes are maintained on the shared `comet-fusion-pos-neg` branch,
building on fusion commit `9b5d19e`. The CLI name `--task subtype` is retained for compatibility;
labels and prompts on this branch are NEG/POS. No model weights have been trained
by these code changes.

## Models

| Comparison | Input | Implementation |
|---|---|---|
| Frozen H-optimus-1 + gated ABMIL | H&E tile embeddings | `data_preprocessing/wsi_embed.py`, `abmil_montage.py train`, `evaluate_fusion_baselines.py` |
| Ridge logistic regression | Selected methylation CpGs, converted to M-values | `evaluate_subtype.py ridge` -> `methylation/ridge.R` |
| Equal-weight late fusion | ABMIL POS score and ridge POS score | `evaluate_fusion_baselines.py --ridge-predictions ...` |
| MedGemma without fine-tuning | Montage + CpG beta values as text | `evaluate_subtype.py medgemma` with base model |
| Centralized MedGemma LoRA/QLoRA | Same format | `job.py --task subtype --n_clients 1` |
| Simulated federated MedGemma LoRA/QLoRA | Same format, disjoint patient clients | `job.py --task subtype --n_clients 3` |

MedGemma also supports image-only and methylation-only exports for ablations.
Start with FedAvg; existing FedProx/SCAFFOLD/FedLC options are optional experiments,
not prerequisites. This is ABMIL, not a full CLAM classifier; CLAM-style refers to
slide tissue masking. MedGemma receives montage pixels and CpG text, not raw
H-optimus vectors. The methylation selector is centralized on training patients;
the experiment is not end-to-end federated feature selection.

“Zero-shot” means no MedGemma fine-tuning. The CpG panel and attention-based tile
selector are supervised on training data, so the complete pipeline is not
training-free. Describe it as frozen MedGemma with supervised input preparation.
An entirely training-free comparison would require a separate, label-independent
tile/panel protocol. Keep this distinction in papers and reports.

## Data required

Use the full original CpG-by-sample COMET beta matrix, matching methylation
metadata, original COMET mapping and final fusion-status file, and current slide
inventory. The supplied parent `Data/` directory is an old ERMS/ARMS preparation
with only 849 selected beta columns; it cannot reconstruct a new top-0.1% panel.
Do not overwrite it. Use new fold-specific output paths. Review the updated audit
before setting `--confirm-reviewed-labels`.

## Primary paired-cohort cross-validation

Run from the repository root after placing source files in the paths below.
These commands are templates; obtain the real original beta matrix first.

```bash
python prepare_comet_subtype.py audit \
  --mapping data/raw/comet/COMET_RMS.xlsx \
  --fusion data/raw/comet/COMET_RMS_fusion_status.xlsx \
  --comet-metadata data/raw/comet/RMSs_meta.txt \
  --comet-matrix data/raw/comet/RMS_beta.tsv \
  --slide-dir data/raw/comet_slides \
  --output data/manifests/fusion_uid_audit

python plan_fusion_cv.py --audit-dir data/manifests/fusion_uid_audit \
  --cohort paired --folds 5 --output data/manifests/fusion_paired_cv

for fold in 01 02 03 04 05; do
  python prepare_comet_subtype.py prepare \
    --audit-dir data/manifests/fusion_uid_audit \
    --require-slide --split-file "data/manifests/fusion_paired_cv/fold_${fold}.tsv" \
    --qc-mode lenient --top-fraction 0.001 --confirm-reviewed-labels \
    --output "data/fusion_cv/fold_${fold}"
done
```

The split planner assigns each canonical patient to the outer test set exactly
once. Each outer training pool has a stratified validation subset for tuning.
The R step applies fixed sample QC, training-only probe QC, imputation and limma
separately for each fold. Post-QC losses are explicit; stop and review if a split
loses a class. All models in a fold must use the resulting same eligible patients.
The generated manifests require an exact match to the selected cohort; do not
reuse paired manifests with the larger methylation cohort.

Within each fold, run ABMIL training and montage creation with that fold's
`--prepared` directory and distinct output directories. Reuse frozen encoder
embeddings, but do not reuse an ABMIL checkpoint/attention map across folds.
Export all modalities from the same paired prepared directory. For each layout,
use the existing commands in COMET_SUBTYPE.md with the fold paths substituted.
The exporter now fails on missing montages instead of silently reducing the
cohort and writes baseline CSVs for the same patients retained in MedGemma JSON.
Use the same seed for all layouts; use the same IID/skew assignment between
modalities. Never choose epochs, panel fractions, prompts or FL settings by test
performance. Retain the test set for final evaluation and pool out-of-fold
patient predictions with uncertainty intervals in the final analysis.

## Baseline evaluation

```bash
python evaluate_fusion_baselines.py \
  --abmil-scores runs/fusion/fold_01/abmil/slide_scores.tsv \
  --split test --output runs/fusion/fold_01/abmil_test

python evaluate_subtype.py ridge \
  --data-dir data/fusion_cv/fold_01_multimodal_central \
  --split test --output runs/fusion/fold_01/ridge_test

python evaluate_fusion_baselines.py \
  --abmil-scores runs/fusion/fold_01/abmil/slide_scores.tsv \
  --ridge-predictions runs/fusion/fold_01/ridge_test/predictions.csv \
  --split test --output runs/fusion/fold_01/late_fusion_test
```

Late fusion uses a pre-specified 0.5/0.5 average and threshold 0.5; it does not
optimize weights using evaluation labels or claim calibrated probabilities.
It requires identical patient IDs and labels; no silent intersection is allowed.
Use outputs from the same fold: patient matching alone cannot prove that a model
was trained on the correct fold. ABMIL and ridge provenance must be retained.

The larger methylation-only cohort is a supplementary analysis, not a replacement
for the paired head-to-head comparison. If extra methylation training patients
are used, explicitly exclude all outer test/validation patient samples and
document the expanded training cohort. Predicting missing fusion labels is a
separate inference workflow; these predictions must never become reference labels.
