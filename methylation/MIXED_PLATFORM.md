# Mixed-platform seven-class experiment

This is a new experiment, not a replacement for the five-class experiment.
Use `prepare_mixed_methylation.py` for local panels, not the single-panel exporter.
The existing training entry point and MedGemma evaluator support these exported
local panels. Pooled ridge/compare mode deliberately rejects mixed-panel tasks.

## Design

- Site 1: AML + CCSK (450K); site 2: NBL + OS + WT (450K);
  site 3: ALL + RT (EPIC). Diagnosis sets do not overlap across sites.
- All prompts will offer ALL, AML, CCSK, NBL, OS, RT, and WT.
- Split by globally resolved patient identity before fitting preprocessing.
- Each site selects its panel using its own training patients only.
- Validation and test reuse their site's frozen panel and preprocessing.
- Site/platform and diagnosis are confounded in this dataset. Results will not
  establish generalization to new sites or platform-independent diagnosis.

## VM: download and audit

Run from the repository root with the existing virtual environment activated.
First update the code using your normal reviewed Git workflow. Keep all existing
five-class data, manifests, and results. The shared raw directory is reused to avoid
downloading the five existing 450K matrices again; existing files are skipped.

```bash
python -m pip install dxpy
dx login
dx ls project-JBXfX9Q0vQQYJ9k4BxGVyYzJ:/methyl/
df -h .

# Preview two EPIC matrices and their sample-list transfers.
python download_methylation.py --platform epic --sample-lists

# Download only missing files. Do not delete existing matrices.
python download_methylation.py --platform epic --sample-lists --download
python download_methylation.py --platform 450k --sample-lists --download

python audit_mixed_methylation.py \
  --matrix-root data/raw/methylation \
  --output data/manifests/mixed_7class_v1_audit
```

The download helper skips existing destinations; this does not verify their
integrity. A previous interrupted `.part` file may need recovery before retrying.
The audit reads headers only, not the multi-GB matrix bodies.

## Review before proceeding

Inspect `audit.json` and `samples_audit.tsv` locally on the VM. Neither file should
be committed or publicly shared. The console prints aggregate counts without
patient identifiers. Confirm:

1. Lists correspond to matrix versions; investigate absent and unlisted samples.
2. Confirm the actual diagnoses represented by `all.samples` and `rt.samples`.
   The word `all` in a filename is not independently verified ALL diagnosis.
3. Resolve unknown identifiers (including non-TARGET identifiers) with an explicit
   sample-to-patient and diagnosis mapping. Do not silently exclude or relabel them.
4. Review sample-type exclusions and duplicate patients across all seven files.
   Cross-cohort labels must be reconciled; one patient must not span sites/splits.
5. Approve final eligible counts before fixing site allocation and splits.

## Prepare and train after the audit

The user authorized exclusion of unresolved identifiers and cohort-membership
labels. These are recorded assertions, not independently verified diagnoses.
The explicit flags below implement that decision. All exclusions are retained in
`exclusions.json`. Cross-cohort conflicts and cross-platform duplicate candidate
patients still stop preparation; they are not silently relabeled.

All AML/CCSK patients go to site 1, NBL/OS/WT to site 2, and ALL/RT to site 3.
This is complete label-support separation, with unequal site sizes. Before site
assignment, one eligible sample per
patient is selected deterministically (type 01, then 09, then 03, then sample ID).
Each site independently splits approximately 60/20/20 by diagnosis, fits QC and
limma on training patients, and freezes 100 CpGs for validation/test. All prompts
offer seven labels. Local panels can reveal site identity and therefore restrict
the likely diagnoses: this is a shortcut/confounding risk, not evidence of
platform-independent biology. CCSK local training counts remain tiny.
This v2 partition rejects resuming old v1 balanced-site preparation. Keep the
old outputs; regenerate all three panels and prompts in the new directory below.

```bash
export R_LIBS_USER="$PWD/data/r-library"
unset R_HOME
mkdir -p logs
set -o pipefail

python prepare_mixed_methylation.py \
  --audit-dir data/manifests/mixed_7class_v1_audit \
  --output data/methylation_mixed_7class_disjoint_v2 \
  --confirm-cohort-labels --exclude-unresolved \
  2>&1 | tee logs/methylation_mixed_7class_disjoint_prepare_v2.log
```

Preparation prints post-QC counts by site, split, and diagnosis. It loads one site
at a time, but still requires enough host RAM for dense methylation matrices.
If fewer than 100 CpGs meet FDR < 0.05 or class counts fail QC, preparation stops;
there is no nonsignificant fallback. A partial run can use `--resume` with the same
arguments to skip completed R stages. If a failed R stage left a partial output,
use a new experiment directory rather than overwriting it. Do not train unless
preparation succeeds and `task.json` exists.

On the existing A100 80GB VM, the following uses the same three-concurrent-client
GPU mapping as the five-class experiment. Available GPU memory is still required.
This starts a fresh model, not the old five-class checkpoint.

```bash
python job.py --task methylation \
  --data_dir data/methylation_mixed_7class_disjoint_v2/clients \
  --n_clients 3 --gpu '[0],[0],[0]' \
  --num_rounds 3 --num_train_epochs 1 \
  --lora_aggregation naive --global_lora_rank 16 --site_lora_ranks 16,16,16 \
  --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 --max_seq_length 4096 \
  --workspace runs/methylation_mixed_7class_disjoint/fl_gpu0_v2 \
  2>&1 | tee logs/methylation_mixed_7class_disjoint_fl_v2.log
```

## Global validation with frozen local panels

After successful training, confirm the checkpoint exists, then evaluate:

```bash
MIXED_MODEL=runs/methylation_mixed_7class_disjoint/fl_gpu0_v2/medgemma/server/simulate_job/app_server/FL_global_model.pt
test -f "$MIXED_MODEL" && python evaluate_methylation.py medgemma \
  --data-dir data/methylation_mixed_7class_disjoint_v2 \
  --model-path "$MIXED_MODEL" \
  --output runs/methylation_mixed_7class_disjoint/validation_v2
```

Each patient retains the source site's panel, while every prediction uses the
seven-class vocabulary. Metrics include pooled seven-class results and per-site
results (site macro metrics cover classes actually present there). Only evaluate
test after freezing settings, using `--split test` and a fresh output directory.
Do not compare seven-class scores directly with the old five-class experiment as
if they measured only the effect of non-IID partitioning: the task/cohort changed.

The old five-class data and results remain unchanged. Local selection is simulated
on one VM; this is not a deployment of private feature selection across real hospitals.
Patient-level files, local panels, and checkpoints should remain out of Git.

## Tests

```bash
python -m unittest discover -s tests -p 'test_m*.py'
python tests/smoke_mixed.py
```

Synthetic CPU tests exercise real three-site limma preparation, seven-class export,
deduplication, site/panel consistency and patient-disjoint checks. GPU training
and real mixed-cohort performance must be verified on the VM.
