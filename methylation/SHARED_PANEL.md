# Eight-class shared-panel experiment: TARGET + COMET RMS (top 0.1%)

This workflow adds RMS as the eighth diagnosis. It does not overwrite the earlier
five-class or seven-class local-panel experiments. Run commands from the repository
root on the VM after committing/pushing the code on the Mac and pulling it on the VM.
All real patient data, derived feature files, prompts, logs and checkpoints belong
under the ignored `data/`, `logs/` and `runs/` directories, not in GitHub.

## Resuming after the earlier top 1% preparation

After pulling this revision on the VM, keep the existing downloads and
`data/manifests/shared_8class_v1_audit`. If inputs and the reviewed audit have not
changed, restart at **Section 4**, then run Sections 5 and 6 before smoke training.
Do not rerun the audit into its existing directory. All new prepared-data, log and
result paths include `top0p1pct`, preserving the earlier 1% experiment.
The commands explicitly use `--top-fraction 0.001` (0.1%, not 1% or 10%).
The Python CLI default remains 0.01 for compatibility; do not omit this argument.
Keep the previous seed, eligibility/QC settings and input files unchanged for a
matched comparison. The commands use the original default seed, 20260929; if your
previous config used a different seed or QC settings, reuse those instead.
This is a smaller-panel pilot, not evidence that 1% has no additional benefit.

## Design and interpretation

1. Audit TARGET 450K, TARGET EPIC and COMET methylation samples. COMET is a cohort,
   not a third array technology; verify the exact array version with the provider.
2. Use TARGET cohort-membership labels only after review. Use COMET metadata
   `Disease=RMS`, `Sample.Type=Patient tumor`, `Final.QC=PASS`, `Pass.QC=PASS`.
   Exclude cell lines, xenografts, unresolved records and extra arrays from the
   same patient. Unknown records are excluded only with an explicit flag.
3. Use `UID.Subject` for COMET patient identity. Select one eligible array per
   patient deterministically (lexicographic sample ID for COMET), not by beta
   values or model performance. This does **not** establish primary/relapse timing.
   Confirm this representative-sample policy with the data owner if timing matters.
4. Stratify unique patients approximately 60/20/20 into train/validation/test once.
   The two FL layouts reuse exactly these patients, split assignments and features.
5. Intersect canonical CpG IDs across **all eight cohort matrices**, including
   COMET. Canonical IDs must match the same biological loci; verify array
   annotation versions. Duplicate IDs are rejected, and suffixed EPIC-v2 probe
   IDs require explicit annotation-based harmonization upstream, not string guessing.
6. Apply detection-p masking when supplied; fixed sample missingness <=20%; optional
   prespecified probe exclusion list. Require <=5% missing training values in
   **each source group** (TARGET_450K, TARGET_EPIC, COMET_RMS), not just pooled data.
   Fit median imputation on pooled training data only; apply it unchanged elsewhere.
7. Transform training beta values to M-values and remove zero-variance/nonfinite
   training features. Test all remaining common features using limma's moderated
   omnibus diagnosis test. **There is no 20,000-feature variance cap.** Rank by
   raw p-value, with CpG ID breaking ties. Select
   `ceil(top_fraction * number_of_tested_CpGs)`; this guide explicitly uses
   `top_fraction=0.001` (top 0.1%).
   BH FDR is reported, but **there is no FDR cutoff in this new top-percent mode**.
   For example, 400,000 tested features produce 400 selected features. The current
   dataset is expected to yield about 436; verify the actual selection summary.
8. Freeze the panel and training medians; distribute identical feature artifacts
   to every client. All training/evaluation prompts use all eight options.

QC here means **processed-beta QC**, not a claim of complete raw-array QC. Without
IDATs, detection p-values or relevant annotations, the software cannot reconstruct
all assay QC, cross-reactive/SNP probe filters or normalization. An optional
`--exclude-probes` file supplies reviewed CpG exclusions, one ID per line, no header.

This remains **centralized training-only feature selection followed by simulated
federated model training**, not end-to-end federated limma. A shared panel removes
site-specific feature-name differences but does not remove batch effects.
RMS is unique to COMET, and other diagnoses are coupled to TARGET platforms;
diagnosis and source are therefore confounded. High internal accuracy alone does
not demonstrate cross-institution or cross-platform generalization. Do not blindly
add source/site as limma covariates when they are collinear with diagnosis.

## 1. Publish local changes, then update the VM environment

First, on the **Mac**, review and publish the implementation. These commands stage
the workflow files explicitly, not downloaded data or experiment outputs:

```bash
cd /Users/wuzhentian/Desktop/Harvard_Fall/Capstone/medgemma-fl
git branch --show-current
git status --short
git add README.md evaluate_methylation.py job.py methylation/README.md \
  methylation/prepare.R methylation_utils.py methylation_validate.py \
  check_methylation_context.py methylation/SHARED_PANEL.md \
  prepare_shared_methylation.py tests/test_shared_methylation.py
git diff --cached --stat
git commit -m "Add shared eight-class methylation panel and IID/non-IID workflows"
git push origin research/dna-methylation
```

Confirm the current branch is `research/dna-methylation` before committing. If the
changes were already committed/pushed, skip those steps. Reviewing or editing this
guide does not itself commit or push anything.

Then, on the **VM** (all remaining commands run there):

```bash
cd ~/medgemma-fl
git branch --show-current
git status --short
git pull --ff-only
source .venv/bin/activate
export R_LIBS_USER="$PWD/data/r-library"
unset R_HOME
mkdir -p logs
set -o pipefail
python -m pip install dxpy
Rscript -e 'stopifnot(all(vapply(c("data.table","jsonlite","limma","glmnet"), requireNamespace, logical(1), quietly=TRUE)))'
hf auth login
dx login
dx select --level VIEW project-JBXfX9Q0vQQYJ9k4BxGVyYzJ
```

Keep the working GPU environment. Only run `Rscript methylation/install.R` if the
R package check fails and the selected R interpreter/library are compatible.
Check that the VM is also on `research/dna-methylation`. Preserve any VM changes
and resolve a failed pull before proceeding; do not force-reset the checkout.

## 2. Download matrices and sample lists

Existing TARGET files are skipped by the downloader:

```bash
python download_methylation.py --platform 450k --sample-lists --download
python download_methylation.py --platform epic --sample-lists --download
dx ls project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/
mkdir -p data/raw/comet
df -h .
free -h
```

The supplied DNAnexus screenshot confirms `/COMET_RMS/RMSs_beta.txt` (about
11.3 GiB). It is a processed CpG-by-sample beta matrix with first column `TargetID`
and sample columns such as `201985320079_R06C01.Ave_Beta`. The existing parser
removes only the exact `.Ave_Beta` suffix and matches `Sample_Name` in the metadata;
no column-map file is needed for this naming convention. The screenshot verifies
the displayed columns, not coverage of every metadata sample: the audit checks all
columns after download.

The provided local `RMSs_meta (1).txt` is metadata; `(1)` is a local duplicate-download
suffix. Confirm that the remote metadata is named `RMSs_meta.txt` in the folder
listing above. If it is named differently, change only `COMET_META_SOURCE` below
to the listed path. Stop if either `dx describe` fails; do not substitute RNA-seq
metadata. No 11.3-GiB download to the Mac is necessary.

```bash
COMET_MATRIX_SOURCE='project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/RMSs_beta.txt'
COMET_META_SOURCE='project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/COMET_RMS/RMSs_meta.txt'
dx describe "$COMET_MATRIX_SOURCE"
dx describe "$COMET_META_SOURCE"
```

After both objects are verified, download directly on the VM:

```bash
dx download "$COMET_MATRIX_SOURCE" -o data/raw/comet/RMS_beta.tsv
dx download "$COMET_META_SOURCE" -o data/raw/comet/RMSs_meta.txt
head -n 2 data/raw/comet/RMS_beta.tsv | cut -f 1-5
```

`RMS_beta.tsv` is the local name used consistently in the commands below; the
remote name is `RMSs_beta.txt`. Do not force-overwrite an existing file. If already
downloaded, verify its source and successful completion before skipping the download.
Keep the original metadata unchanged. The disk must have room for this matrix and
the prepared outputs in addition to existing TARGET data. File size is not a RAM
estimate: R reads multiple matrices and creates numeric working copies, so monitor
host RAM separately. Successful downloads alone do not establish training readiness.

## 3. Audit patient identity and labels (no beta fitting)

```bash
python prepare_shared_methylation.py audit \
  --matrix-root data/raw/methylation \
  --comet-matrix data/raw/comet/RMS_beta.tsv \
  --comet-metadata data/raw/comet/RMSs_meta.txt \
  --output data/manifests/shared_8class_v1_audit
cat data/manifests/shared_8class_v1_audit/audit.json
```

Review `samples_audit.tsv`, exclusions and candidate unique-patient counts. The
supplied metadata has 547 arrays: 477 patient-tumor arrays, 47 xenografts and 23
cell lines. The 477 patient-tumor arrays represent 393 distinct `UID.Subject`
values; **393 is not the final included count**, which also depends on matrix
coverage and QC. Both metadata QC fields are PASS in the supplied version.

COMET beta columns must match `Sample_Name` exactly, optionally with `.Ave_Beta`.
If the matrix uses a different identifier, provide a reviewed TSV with columns
`beta_column` and `sample_name`, using `--comet-column-map PATH`. No fuzzy matching
or diagnosis inference from arbitrary column names is performed. Detection columns
are recognized by the corresponding `.Detection.Pval` suffix.

TARGET and COMET patient IDs use different naming systems. Distinct strings do
not prove the cohorts contain different people. Confirm non-overlap with the data
owner, or provide `--patient-map PATH` during audit: a TSV with columns `source`
(`TARGET` or `COMET`), `source_patient_id`, and canonical `patient_id`. Cross-source
diagnosis conflicts block preparation. Unmapped known IDs retain their original
TARGET identity or a `COMET:` namespace; unknown identities are never invented.

## 4. Prepare one common panel and one set of splits

Only use the confirmation flags after reviewing the audit and identity issue above:

```bash
python prepare_shared_methylation.py prepare \
  --audit-dir data/manifests/shared_8class_v1_audit \
  --output data/methylation_shared_8class_top0p1pct_v1 \
  --confirm-reviewed-labels \
  --confirm-patient-identity \
  --exclude-unresolved \
  --top-fraction 0.001 \
  --seed 20260929 \
  2>&1 | tee logs/shared_8class_top0p1pct_v1_prepare.log
cat data/methylation_shared_8class_top0p1pct_v1/selection_summary.json
head -n 6 data/methylation_shared_8class_top0p1pct_v1/features.tsv
```

The `--exclude-unresolved` flag explicitly excludes unresolved records reported
in the audit. It does not waive diagnosis conflicts. Preparation refuses to
overwrite output/config/manifest files: after an error, preserve them for diagnosis
and choose a new versioned output name when retrying.

Important artifacts:

| Artifact | Purpose |
| --- | --- |
| `features.tsv` | Ordered CpG IDs, training medians, limma p-values and BH FDR |
| `features.json` | Machine-readable panel, ordered-ID hash, TSV hash and preprocessing |
| `FEATURES.md` | Human-readable instructions for using the fixed panel |
| `eligible_cpgs.tsv` | Full tested universe defining the top-percent denominator |
| `limma_training_only.tsv` | Full training-only limma results |
| `splits.tsv`, `sample_audit.tsv` | Private patient splits and sample-level beta QC |
| `train.csv`, `validation.csv`, `test.csv` | Same rounded beta values for ridge and MedGemma |
| `preprocessing.rds` | Frozen preprocessing and selected CpGs |

Medians are training-data-derived statistics; a fixed feature artifact is not
automatically privacy-safe or approved for public release.

## 5. Export matched IID and non-IID experiments

```bash
python prepare_shared_methylation.py export \
  --prepared data/methylation_shared_8class_top0p1pct_v1 \
  --partition iid --output data/methylation_shared_8class_top0p1pct_iid_v1
python prepare_shared_methylation.py export \
  --prepared data/methylation_shared_8class_top0p1pct_v1 \
  --partition non-iid --output data/methylation_shared_8class_top0p1pct_noniid_v1
cat data/methylation_shared_8class_top0p1pct_iid_v1/site_counts.json
cat data/methylation_shared_8class_top0p1pct_noniid_v1/site_counts.json
```

- IID: shuffle within each diagnosis and distribute round-robin across three sites.
  Per-diagnosis counts differ by at most one; class frequencies remain unequal.
  This is approximately stratified IID, not eight equally sized classes.
- Non-IID default: site-1 = AML + CCSK; site-2 = NBL + OS + WT;
  site-3 = ALL + RT + RMS. Thus only site-3 has COMET, alongside TARGET EPIC.
  This is a strong diagnosis-disjoint simulation, not a measured hospital split.
- For a different diagnosis-disjoint layout, use `--site-map PATH` with JSON mapping
  `site-1`, `site-2`, `site-3` to nonempty diagnosis lists; each class must appear once.
- Both layouts copy the same split CSVs and feature artifacts. Only site assignment
  changes. Test examples stay outside client folders. Each site receives identical
  `features.tsv`, `features.json`, `FEATURES.md`, plus its own train/validation JSONs.

User prompt (patient ID, site/platform and true diagnosis are not input features):

```text
Classify this pediatric tumor using DNA methylation.

[DNA methylation]
cg00000001: 0.250000
...

What is the most likely diagnosis?
Options: ALL, AML, CCSK, NBL, OS, RT, WT, RMS.
Reply with only the diagnosis abbreviation.
```

The true label appears only in the assistant training target. Loss is masked on
the user prompt. Every site and evaluation sample uses the same ordered panel
and all eight options, even when a site has never seen some diagnoses.

## 6. Check actual input length BEFORE training

For approximately 436 features, the expected length is roughly 8,800 tokens,
not the earlier 87,112 tokens. This is an estimate; the actual tokenizer check
below is authoritative. Start with a 10,240-token budget, not the old 4,096 or
90,112 setting:

```bash
export METHYL_CONTEXT=10240
python check_methylation_context.py \
  --data-dir data/methylation_shared_8class_top0p1pct_iid_v1 \
  --max-seq-length "$METHYL_CONTEXT"
```

This loads only the processor/config, not model weights. Continue only if the
check passes. If it fails, inspect `required_max_seq_length` and
`model_context_capacity`, choose a supported budget and rerun. Both layouts use
identical input text, so the same budget applies. Re-export `METHYL_CONTEXT`
when opening a new terminal, and use it for both training and evaluation.

**Supported context is not a GPU memory guarantee.** Even this smaller panel must
pass the three-client smoke test on the single A100. Never silently truncate the
prompt. Top 1% (`--top-fraction 0.01`) and top 5% (`--top-fraction 0.05`) remain
available as separate experiments; preserve their outputs and do not choose the
feature budget based on test performance.

## 7. Federated smoke run, then matched full runs

The existing pipeline uses QLoRA and same-rank LoRA-factor FedAvg by default.
Shared-panel jobs automatically repeat the context check before launching clients.
The default below matches the previously reported **single A100 80GB VM**: all
three clients share GPU 0. This does **not** serialize clients or guarantee memory
availability. If the long-context smoke run runs out of memory, stop; do not launch
the full runs or truncate features. Use more resources or a separately implemented
sequential simulation. On a VM that actually has three available GPUs, set
`METHYL_GPU_MAP='[0],[1],[2]'` instead. Recheck the VM hardware before choosing.

```bash
nvidia-smi
METHYL_GPU_MAP='[0],[0],[0]'
```

```bash
python job.py --task methylation \
  --data_dir data/methylation_shared_8class_top0p1pct_iid_v1/clients \
  --n_clients 3 --num_rounds 1 --max_steps 1 \
  --max_seq_length "$METHYL_CONTEXT" \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --global_lora_rank 16 --site_lora_ranks 16,16,16 \
  --gpu "$METHYL_GPU_MAP" \
  --workspace runs/shared_8class_top0p1pct/iid_smoke_v1 \
  2>&1 | tee logs/shared_8class_top0p1pct_iid_smoke_v1.log
```

After successful smoke training, run each layout **from the same base model**, not
by continuing the IID checkpoint into the non-IID run. Use identical hyperparameters:

```bash
for layout in iid noniid; do
  python job.py --task methylation \
    --data_dir "data/methylation_shared_8class_top0p1pct_${layout}_v1/clients" \
    --n_clients 3 --num_rounds 3 --num_train_epochs 2 \
    --max_seq_length "$METHYL_CONTEXT" \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --global_lora_rank 16 --site_lora_ranks 16,16,16 \
    --gpu "$METHYL_GPU_MAP" \
    --workspace "runs/shared_8class_top0p1pct/${layout}_fl_v1" \
    2>&1 | tee "logs/shared_8class_top0p1pct_${layout}_fl_v1.log"
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then break; fi
done
```

Record software versions, GPU configuration and random seeds. One split/seed is a
pilot comparison; repeated seeds are needed to assess stability.

## 8. Validation comparison, then a final untouched-test comparison

The pooled ridge baseline uses the same panel/patients for both layouts, so fit it
once. Ridge trains on train only; lambda is selected using validation balanced
accuracy. It is a centralized benchmark, not federated ridge.

```bash
python evaluate_methylation.py ridge \
  --data-dir data/methylation_shared_8class_top0p1pct_iid_v1 \
  --output runs/shared_8class_top0p1pct/ridge_validation_v1

for layout in iid noniid; do
  python evaluate_methylation.py medgemma \
    --data-dir "data/methylation_shared_8class_top0p1pct_${layout}_v1" \
    --model-path "runs/shared_8class_top0p1pct/${layout}_fl_v1/medgemma/server/simulate_job/app_server/FL_global_model.pt" \
    --max-seq-length "$METHYL_CONTEXT" \
    --output "runs/shared_8class_top0p1pct/${layout}_validation_v1" || break
  python evaluate_methylation.py compare \
    --data-dir "data/methylation_shared_8class_top0p1pct_${layout}_v1" \
    --ridge-predictions runs/shared_8class_top0p1pct/ridge_validation_v1/predictions.csv \
    --medgemma-predictions "runs/shared_8class_top0p1pct/${layout}_validation_v1/predictions.csv" \
    --output "runs/shared_8class_top0p1pct/${layout}_comparison_validation_v1" || break
done
```

Inspect `evaluation.json` and `predictions.csv`. Report accuracy, balanced accuracy,
macro-F1, class recalls and per-site MedGemma results. Small classes need uncertainty
reporting; accuracy can hide complete failure on a rare class.

Only after finalizing both models/hyperparameters using validation:

```bash
python evaluate_methylation.py ridge --split test \
  --data-dir data/methylation_shared_8class_top0p1pct_iid_v1 \
  --output runs/shared_8class_top0p1pct/ridge_test_v1
for layout in iid noniid; do
  python evaluate_methylation.py medgemma --split test \
    --data-dir "data/methylation_shared_8class_top0p1pct_${layout}_v1" \
    --model-path "runs/shared_8class_top0p1pct/${layout}_fl_v1/medgemma/server/simulate_job/app_server/FL_global_model.pt" \
    --max-seq-length "$METHYL_CONTEXT" \
    --output "runs/shared_8class_top0p1pct/${layout}_test_v1" || break
  python evaluate_methylation.py compare --split test \
    --data-dir "data/methylation_shared_8class_top0p1pct_${layout}_v1" \
    --ridge-predictions runs/shared_8class_top0p1pct/ridge_test_v1/predictions.csv \
    --medgemma-predictions "runs/shared_8class_top0p1pct/${layout}_test_v1/predictions.csv" \
    --output "runs/shared_8class_top0p1pct/${layout}_comparison_test_v1" || break
done
```

Never rerun limma on validation/test. Do not tune on the final test scores. Preserve
the panel, splits, audit/configuration, logs, prediction files and global adapter in
approved backed-up storage; a Git commit alone does not preserve the experiment.

## Developer checks

```bash
python -m unittest discover -s tests -p 'test_*.py'
RUN_R_INTEGRATION=1 python -m unittest discover -s tests -p 'test_shared_methylation.py'
```

The integration test uses synthetic data, real R limma and both site layouts. It
checks common-probe filtering, per-source missingness, top-percent counts, exclusion
of non-patient samples, patient deduplication, feature/prompt integrity and unchanged
selection when only held-out beta values are perturbed. It is not a real-data
performance test and does not validate GPU memory capacity.
