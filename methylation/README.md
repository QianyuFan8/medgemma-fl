# DNA methylation: limma → ridge versus federated MedGemma

This research branch starts from `main`, not `scope-2`. No Scope 2 changes are merged. Existing histology commands retain their default behavior.

## What this experiment does

1. Read a **reviewed patient/sample manifest** and TARGET beta matrices (one platform per experiment).
2. Stratify patients into approximately 60% train / 20% validation / 20% test; require one sample per patient. Small classes may have slightly different ratios.
3. Mask failed detection p-values when provided, apply fixed sample missingness QC, then learn probe missingness filtering, median imputation and variance filtering **from training patients only**.
4. Convert training beta values to M-values and run limma (`lmFit`, `eBayes`). Test the diagnosis coefficient for binary outcomes, or jointly test diagnosis coefficients for multiclass outcomes. Optional covariates must be prespecified and the design must be full-rank.
5. Select the top K candidates meeting BH FDR < 0.05. Default candidate universe: 20,000 highest training-variance CpGs. No fallback to non-significant probes if fewer than K pass.
6. Freeze the panel. Export the same six-decimal, training-median-imputed beta values for both methods. Ridge transforms these to M-values and uses training-fitted standardization; MedGemma receives text `cg...: beta` in a fixed order.
7. Tune ridge's regularization on validation only. MedGemma uses QLoRA and simulated same-rank FedAvg clients. Compare balanced accuracy, macro-F1, per-class recall, accuracy, and unparsed outputs on identical patients.

**This is centralized feature selection followed by federated model training, not end-to-end federated feature selection or differential privacy.** The initial simulated client partition is stratified/random, not a real institution split. Ridge is pooled/centralized, so this comparison differs in both model family and training regime; it does not isolate the causal effect of federation.

## Local Mac: prepare the existing CCSK/WT dataset

No GPU or Python ML packages are needed for preparation/ridge; Python 3.10+ plus R packages `data.table`, `jsonlite`, `limma`, `glmnet` suffice. Use the same R installation that built the package library. On this machine the existing library is reusable:

```bash
cd /Users/wuzhentian/Desktop/Harvard_Fall/Capstone/medgemma-fl

/opt/homebrew/bin/python3.12 prepare_methylation.py \
  --config methylation/config.ccsk_wt.json \
  --rscript /opt/homebrew/bin/Rscript \
  --r-library ../methylation_cpg_routes/.Rlib \
  --n-clients 3

/opt/homebrew/bin/python3.12 evaluate_methylation.py ridge \
  --data-dir data/methylation_ccsk_wt_v1 \
  --output runs/methylation_ccsk_wt/ridge_validation \
  --rscript /opt/homebrew/bin/Rscript \
  --r-library ../methylation_cpg_routes/.Rlib
```

The existing CCSK/WT manifest has patient-matched diagnosis evidence, which is why that preset has `metadata_reviewed=true`. It retains only rows already marked `include=TRUE`. It does not reuse CpGs selected in the old full-development A/B/C run. **CCSK has only 11 patients**, so a holdout will contain only about two CCSK patients: useful as a pipeline pilot, not persuasive clinical validation.

For a new platform/cohort, edit `methylation/config.example.json`: set the manifest path, `matrix_root` (relative matrix paths are resolved against this), a fresh output directory, optional diagnoses, K, and `metadata_reviewed=true` only after inspecting label evidence. All config-relative paths are interpreted from the current working directory, so run from the repository root. No sample diagnosis is inferred from matrix filenames.

Optional covariates: `"covariates": ["batch", "age"]`, `"numeric_covariates": ["age"]`. They must exist and be nonmissing in the manifest. Perfect diagnosis/batch confounding causes a failure, not a misleading adjusted result. The current presets have no covariates because validated batch metadata have not been supplied.

Outputs:

- `selected_cpgs.tsv`, `limma_training_only.tsv`, `preprocessing.rds`: frozen panel, statistics, and training medians.
- `sample_audit.tsv`, `splits.tsv`: sample QC and patient splits.
- `train.csv`, `validation.csv`, `test.csv`: selected measurements and diagnoses.
- `clients/site-N/train.json`, optional `validation.json`: FL client inputs; **no test patients**.
- `validation.json`, `test.json`, `task.json`: held-out prompts, task vocabulary, ordered-panel fingerprint.
- `config.json`, `R_sessionInfo.txt`: preprocessing provenance.

Output directories are never overwritten. Change the output path for a new run. If preprocessing succeeded but client export failed (e.g. too many clients), fix the client count and use `prepare_methylation.py ... --export-only` before any clients directory exists. Large matrices are loaded into memory; this is not an out-of-core implementation.

On a new machine, install the R dependencies (if needed) using `Rscript methylation/install.R`. This branch's preprocessing does not require the old `methylation_cpg_routes` code: only the example paths refer to its data and optional existing R library. Move the data/manifest and adjust paths on the server. Do not commit patient data or model weights to Git.

## GPU server: federated MedGemma training

Use the repository's existing CUDA/Hugging Face environment setup and model-access instructions. Copy the prepared directory securely to the server; the FL stage requires `clients/`, `task.json`, and the held-out JSONs, not the raw large matrices. Keeping CSVs allows ridge evaluation there too.

```bash
python job.py \
  --task methylation \
  --data_dir ./data/methylation_ccsk_wt_v1/clients \
  --n_clients 3 \
  --num_rounds 3 \
  --lora_aggregation naive \
  --global_lora_rank 16 \
  --site_lora_ranks 16,16,16 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 4096 \
  --workspace ./runs/methylation_ccsk_wt/fl
```

By default the existing recipe assigns one GPU to each client (3 GPUs for this command). Set `--gpu` according to actual server resources; this implementation does not promise that concurrent clients fit on one GPU. Methylation defaults to uniform ranks if ranks are omitted. It does not enable HLoRA or DP-SGD. Existing multimodal model construction is reused, but inputs are text-only and no image features are supplied.

Training uses **assistant-answer-only loss**. It verifies chat-template prefix alignment and rejects token overflow rather than truncating CpGs. K=100/200 does not by itself guarantee a token budget; tokenizer behavior must be checked in the actual server environment.

## Evaluate and compare

Use the actual checkpoint path printed by NVFlare. Do not use the old histology evaluation scripts for methylation.

```bash
python evaluate_methylation.py medgemma \
  --data-dir data/methylation_ccsk_wt_v1 \
  --model-path /actual/path/to/FL_global_model.pt \
  --output runs/methylation_ccsk_wt/medgemma_validation

python evaluate_methylation.py compare \
  --data-dir data/methylation_ccsk_wt_v1 \
  --ridge-predictions runs/methylation_ccsk_wt/ridge_validation/predictions.csv \
  --medgemma-predictions runs/methylation_ccsk_wt/medgemma_validation/predictions.csv \
  --output runs/methylation_ccsk_wt/comparison_validation
```

Each evaluation produces `predictions.csv` and `evaluation.json` with metrics and arguments. Ridge also saves its tuning table/model. Compare refuses missing/extra patients, inconsistent truths, or different panel/split provenance. MedGemma receives **only the user prompt** at inference; exact diagnosis-code output is required. Extra explanations/unrecognized answers count as errors, not silently skipped patients.

During development use validation. Once K, prompts, hyperparameters, and the checkpoint are frozen, repeat each evaluation with **`--split test` and new output directories**, then compare with `--split test`. Do not select rounds or K using test scores. The test is an internal held-out subset, not an independent external cohort. Ridge's chosen lambda remains based only on validation; it does not refit on validation patients.

## Testing / current verification boundary

```bash
python -m unittest discover -s tests -p test_methylation.py -v
python tests/smoke_methylation.py --rscript Rscript
python tests/smoke_methylation.py --rscript Rscript --classes 2
```

On this Mac add `--rscript /opt/homebrew/bin/Rscript --r-library ../methylation_cpg_routes/.Rlib` to smoke commands. Synthetic tests run real limma, ridge and client export on CPU. They perturb held-out measurements and assert the training panel/data remain unchanged. Pure-Python tests check assistant label masking, overflow, cohort matching, and client leakage guards.

No real-data training result or GPU/TRL/NVFlare methylation run has been established by these tests. In particular, actual MedGemma tokenizer/template compatibility, CUDA memory and server dependency versions still require a short server smoke run. Existing histology functionality should be checked there too.

The beta matrices are already processed data; this code does not perform raw IDAT normalization or automatically exclude SNP/cross-reactive probes. It cannot remove unknown batch confounding, establish CpG causality, or show clinical utility. Probe annotation QC, independent cohorts, repeated splits and uncertainty estimates remain research follow-ups.
