# DNA methylation: limma → ridge versus federated MedGemma

This research branch starts from `main`. Existing histology commands retain their default behavior.

## Current 450K label source: user-confirmed `.samples` cohort membership

The current experiment uses the user's explicit confirmation that eligible tumor samples in `aml.samples`, `ccsk.samples`, `nbl.samples`, `os.samples`, and `wt.samples` carry the corresponding cohort diagnoses. This is a documented label-source assertion, **not independent clinical verification by the software**. It does not use `/RNAseq/TARGET_meta.txt` or the removed GDC clinical tables. Do not follow the older metadata-download instructions below for this mode.

After updating this branch on the VM and activating its environment, run:

```bash
python download_methylation.py --platform 450k --sample-lists
python download_methylation.py --platform 450k --sample-lists --download
python prepare_cohort_labels.py \
  --matrix-root data/raw/methylation \
  --output data/manifests/450k_cohort_v1 \
  --confirm-cohort-labels
cat data/manifests/450k_cohort_v1/audit.json
cat data/manifests/450k_cohort_v1/label_provenance.json
nano data/manifests/450k_cohort_v1/config.json
```

Review the five diagnoses and exclusions before setting `metadata_reviewed` to `true`. Each sample must appear in its corresponding `.samples` list and beta matrix. Non-primary samples, conflicting patient labels, duplicate patients, and insufficiently represented classes are excluded and logged. Source sample IDs and assigned diagnoses are saved in `450k_cohort_v1_label_evidence.tsv`. EPIC/`all.samples` is deliberately not supported by this labeling command and needs a separate review, given its mixed sample identifiers.

Then prepare the train-only panel and ridge baseline:

```bash
export R_LIBS_USER="$PWD/data/r-library"
unset R_HOME
mkdir -p logs
set -o pipefail
python prepare_methylation.py --config data/manifests/450k_cohort_v1/config.json \
  --n-clients 3 2>&1 | tee logs/methylation_450k_prepare.log
python evaluate_methylation.py ridge --data-dir data/methylation_450k_v1 \
  --output runs/methylation_450k/ridge_validation
```

Continue with VM steps 8–10 below for GPU smoke training, full FL training, and validation/test comparison. Use a new output path if an experiment directory already exists. Final accuracy requires actually running the aggregated MedGemma checkpoint and ridge on the same held-out patients; preparation or synthetic test scores are not a substitute.

## What this experiment does

1. Read a **reviewed patient/sample manifest** and TARGET beta matrices (one platform per experiment).
2. Stratify patients into approximately 60% train / 20% validation / 20% test; require one sample per patient. Small classes may have slightly different ratios.
3. Mask failed detection p-values when provided, apply fixed sample missingness QC, then learn probe missingness filtering, median imputation and variance filtering **from training patients only**.
4. Convert training beta values to M-values and run limma (`lmFit`, `eBayes`). Test the diagnosis coefficient for binary outcomes, or jointly test diagnosis coefficients for multiclass outcomes. Optional covariates must be prespecified and the design must be full-rank.
5. Select the top K candidates meeting BH FDR < 0.05. Default candidate universe: 20,000 highest training-variance CpGs. No fallback to non-significant probes if fewer than K pass.
6. Freeze the panel. Export the same six-decimal, training-median-imputed beta values for both methods. Ridge transforms these to M-values and uses training-fitted standardization; MedGemma receives text `cg...: beta` in a fixed order.
7. Tune ridge's regularization on validation only. MedGemma uses QLoRA and simulated same-rank FedAvg clients. Compare balanced accuracy, macro-F1, per-class recall, accuracy, and unparsed outputs on identical patients.

**This is centralized feature selection followed by federated model training, not end-to-end federated feature selection or differential privacy.** The initial simulated client partition is stratified/random, not a real institution split. Ridge is pooled/centralized, so this comparison differs in both model family and training regime; it does not isolate the causal effect of federation.

## Complete VM workflow: DNAnexus download → five-class 450K analysis → federated training → comparison

Run the following commands on the VM from the `medgemma-fl` repository root, except for the local publishing step. Neither the old `target_data/` directory nor a copy of `methylation_cpg_routes` is required on the VM. **Diagnosis labels come exclusively from the verified metadata downloaded from DNAnexus; missing labels are not filled from the old GDC clinical tables.**

### 1. Publish the local branch, then retrieve it on the VM

First run these commands in the local repository on the Mac. Publishing is an explicit user action; updating this document does not push the branch:

```bash
git status
git add download_methylation.py methylation_manifest.py methylation/prepare.R methylation/README.md tests/test_methylation_manifest.py
git add -u target_data
git commit -m "Use DNAnexus-only metadata and document VM methylation workflow"
git push -u origin research/dna-methylation
```

Review any other changes shown by `git status` separately instead of indiscriminately running `git add .`. If these changes are already committed, skip the staging and commit commands and only push.

For a new checkout on the VM:

```bash
git clone --branch research/dna-methylation https://github.com/QianyuFan8/medgemma-fl.git
cd medgemma-fl
```

If the repository already exists on the VM, inspect and preserve local changes first, then run `git fetch origin`, `git switch research/dna-methylation`, and `git pull --ff-only`. Do not overwrite ongoing experiments.

### 2. Set up the environment, GPUs, R packages, and model access

Prefer the CUDA/Python environment that already runs the original MedGemma workflow. First check:

```bash
python3 --version
nvidia-smi
Rscript --version
```

Python >=3.10 is required. To create a new Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install dxpy
python -m pip install -r requirements.txt
python -m pip check
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())'
```

`requirements-vm.txt` records versions from the original VM; it is not a universal installation lockfile. Resolve any PyTorch/CUDA installation or compatibility errors before training. The default three-client parallel configuration requires three GPUs with sufficient memory; a single-GPU VM is not guaranteed to accommodate three concurrent clients.

If R or build dependencies are missing on an Ubuntu VM, a user with sudo privileges can run:

```bash
sudo apt-get update
sudo apt-get install -y r-base r-base-dev gfortran libcurl4-openssl-dev libssl-dev libxml2-dev
```

Use a separate R package library inside the repository's ignored data directory; do not commit the installed packages:

```bash
mkdir -p data/r-library
export R_LIBS_USER="$PWD/data/r-library"
unset R_HOME
Rscript methylation/install.R
Rscript -e 'stopifnot(all(vapply(c("data.table","jsonlite","limma","glmnet"), requireNamespace, logical(1), quietly=TRUE))); sessionInfo()'
```

In each new terminal, reactivate `.venv` and set `R_LIBS_USER` as above. After accepting the license for `google/medgemma-4b-it` on Hugging Face, authenticate:

```bash
hf auth login
```

### 3. Log in to DNAnexus, select the read-only project, and verify metadata

```bash
dx login
dx select --level VIEW project-JBXfX9Q0vQQYJ9k4BxGVyYzJ
dx ls project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/
dx ls project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/methyl/
dx find data --project project-JBXfX9Q0vQQYJ9k4BxGVyYzJ --name TARGET_meta.txt --brief
```

**Do not assume the metadata is located in `/methyl/`.** The current local example is named `TARGET_meta.txt` and must contain `Sample` and `Dx` columns. If the search returns no results or multiple versions, confirm the correct file in the mentor's shared project rather than automatically choosing the first result. Inspect any newer file with a different name or schema and adapt the reader if necessary. Paste the verified `project-...:file-...` reference or full project path below:

```bash
mkdir -p data/raw/metadata
read -r -p "Paste the verified DNAnexus metadata reference: " META_DX_SOURCE
dx describe "$META_DX_SOURCE"
dx download "$META_DX_SOURCE" -o data/raw/metadata/TARGET_meta.txt
head -n 5 data/raw/metadata/TARGET_meta.txt
```

Never put tokens or passwords in code, configuration files, or Git. Run `dx login` again if you encounter `ExpiredToken`. The local token had expired when this document was updated, so the exact remote metadata path was not reverified and no large files were downloaded.

### 4. Download the five 450K beta matrices

```bash
python download_methylation.py --platform 450k
python download_methylation.py --platform 450k --download
```

The first command previews transfers; the second invokes `dx download`. The five source cohorts are AML, CCSK, NBL, OS, and WT. Completed files are stored in `data/raw/methylation/` and skipped on subsequent runs; interrupted downloads retain their `.part` files. Do not manually rename incomplete downloads to their final filenames. The paths use the previously listed `/methyl/idat_sample_sheet_TARGET-...` names; consult `dx ls` if the mentor has renamed files.

### 5. Match patient labels exclusively from DNAnexus metadata

```bash
python methylation_manifest.py \
  --platform 450k \
  --metadata data/raw/metadata/TARGET_meta.txt \
  --matrix-root data/raw/methylation \
  --output data/manifests/450k_v1

cat data/manifests/450k_v1/audit.json
head -n 10 data/manifests/450k_v1/samples.tsv
```

The script reads only matrix headers and matches patient IDs to the metadata's `Dx` field; matrix filenames are not used as diagnosis labels. Normal/non-primary samples, unmatched or conflicting diagnoses, additional samples from the same patient, and classes with fewer than five patients are recorded separately and excluded. Five patients is only a software threshold, not evidence of statistical reliability.

**The current local `TARGET_meta.txt` contains only 15 AML label records; do not assume it covers all five methylation matrices.** Inspect `unmatched_by_source`. If many samples are unmatched, request metadata covering the methylation cohort from the mentor rather than restoring the old clinical-table fallback or assigning labels from filenames. A successful run may still represent an incompletely covered subset, which must be disclosed in the report.

The generated `config.json` defaults to `metadata_reviewed=false`. Verify that all five diagnoses are represented and review the exclusions, then open it in an editor:

```bash
nano data/manifests/450k_v1/config.json
```

Set `metadata_reviewed` to `true` only after review. You can set `top_k=100` or `200`; use a new `output` path for a different experiment. Do not assign the source file's cancer type to unlabeled samples simply to obtain a five-class dataset.

### 6. Run limma selection and export data for three clients

```bash
mkdir -p logs
set -o pipefail
python prepare_methylation.py \
  --config data/manifests/450k_v1/config.json \
  --n-clients 3 2>&1 | tee logs/methylation_450k_prepare.log
```

The default output is `data/methylation_450k_v1/`. Inspect `splits.tsv`, `sample_audit.tsv`, and `selected_cpgs.tsv`. Only training patients contribute to limma selection; validation/test patients do not. Each of the three simulated sites contains multiple diagnosis classes. Increase VM RAM if needed; the current import is not an out-of-core implementation.

### 7. Run the ridge baseline on validation patients

```bash
python evaluate_methylation.py ridge \
  --data-dir data/methylation_450k_v1 \
  --output runs/methylation_450k/ridge_validation
```

### 8. Smoke-test federated MedGemma before the full training run

First confirm that `torch.cuda.device_count()` is >=3 and that the GPUs support BF16 and have sufficient memory. This example uses the default three-GPU configuration; do not launch it unchanged on a single-GPU machine.

```bash
python job.py --task methylation \
  --data_dir data/methylation_450k_v1/clients \
  --n_clients 3 --num_rounds 1 --max_steps 1 \
  --lora_aggregation naive --global_lora_rank 16 --site_lora_ranks 16,16,16 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 --max_seq_length 4096 \
  --workspace runs/methylation_450k/fl_smoke \
  2>&1 | tee logs/methylation_450k_fl_smoke.log
```

The smoke run checks the actual tokenizer, NVFlare integration, and GPU environment; it is not a performance experiment. After it succeeds:

```bash
python job.py --task methylation \
  --data_dir data/methylation_450k_v1/clients \
  --n_clients 3 --num_rounds 3 --num_train_epochs 1 \
  --lora_aggregation naive --global_lora_rank 16 --site_lora_ranks 16,16,16 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 --max_seq_length 4096 \
  --workspace runs/methylation_450k/fl \
  2>&1 | tee logs/methylation_450k_fl.log
```

### 9. Evaluate the final aggregated model on validation patients and compare

```bash
find runs/methylation_450k/fl -name FL_global_model.pt -print
read -r -p "Paste the final aggregated checkpoint path: " METHYLATION_MODEL
python evaluate_methylation.py medgemma \
  --data-dir data/methylation_450k_v1 --model-path "$METHYLATION_MODEL" \
  --output runs/methylation_450k/medgemma_validation
python evaluate_methylation.py compare \
  --data-dir data/methylation_450k_v1 \
  --ridge-predictions runs/methylation_450k/ridge_validation/predictions.csv \
  --medgemma-predictions runs/methylation_450k/medgemma_validation/predictions.csv \
  --output runs/methylation_450k/comparison_validation
cat runs/methylation_450k/comparison_validation/evaluation.json
```

If the checkpoint is not in the directory above, use the actual result path printed by NVFlare. Do not accidentally evaluate the smoke-run checkpoint.

### 10. Evaluate the test split only after all choices are frozen

```bash
python evaluate_methylation.py ridge --split test \
  --data-dir data/methylation_450k_v1 --output runs/methylation_450k/ridge_test
python evaluate_methylation.py medgemma --split test \
  --data-dir data/methylation_450k_v1 --model-path "$METHYLATION_MODEL" \
  --output runs/methylation_450k/medgemma_test
python evaluate_methylation.py compare --split test \
  --data-dir data/methylation_450k_v1 \
  --ridge-predictions runs/methylation_450k/ridge_test/predictions.csv \
  --medgemma-predictions runs/methylation_450k/medgemma_test/predictions.csv \
  --output runs/methylation_450k/comparison_test
cat runs/methylation_450k/comparison_test/evaluation.json
```

Do not use test results to select K, prompts, or training rounds. This is an internal holdout, not independent external validation.

### 11. Run EPIC as a separate subsequent experiment

```bash
python download_methylation.py --platform epic --download
python methylation_manifest.py --platform epic \
  --metadata data/raw/metadata/TARGET_meta.txt \
  --matrix-root data/raw/methylation --output data/manifests/epic_v1
```

Review the actual diagnoses and missing labels before editing the generated EPIC configuration. Repeat steps 6–10 with `epic` in place of `450k` in the relevant paths. The workflow does not automatically intersect the two platforms' top CpG panels.

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
