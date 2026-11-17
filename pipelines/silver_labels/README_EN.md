# Ridge teacher and silver-label handoff

This self-contained CPU workflow prepares patient-disjoint splits, compares variance and LIMMA CpG selection, fits Ridge logistic teachers, and exports labeled student inputs.

## Cohort policy

The v4 policy permits only ERMS/ARMS patients in any training dataset. Unknown or other subtypes are excluded from fitting, including teacher, gold student, and silver student training.

- Ridge teacher training: 108 patients (53 POS / 55 NEG).
- Paired gold student training: 19 patients (10 POS / 9 NEG).
- Gold validation and test: 20 patients each (12 POS / 8 NEG each).
- Eligible unlabeled silver candidates: 79 patients, before confidence filtering.
- All 40 holdout patients are excluded from teacher fitting and feature selection.

The frozen split and private metadata must be supplied separately. Historical experimental results are not rewritten. These exploratory splits include previously examined cases and are not a new external blinded cohort.

## Run the private handoff

From the unpacked `ridge_teacher_handoff` directory:

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data"
```

Required source files:

```text
RMSs_beta.txt
RMSs_meta.txt
COMET_RMS.xlsx
COMET_RMS_fusion_status.xlsx
```

The beta TSV uses `TargetID` and `<Sentrix_ID>.Ave_Beta` columns. Detection p-values must be supplied as `<Sentrix_ID>.Detection.Pval` columns or via `--detection-file /path/to/detection_p.tsv`. A separate detection file must have the same probe order. Missing detection p-values stop the workflow unless `--allow-missing-detection-p` is explicitly selected and recorded as a different QC protocol.

The launcher builds an isolated Python/R environment. Linux and macOS are supported; use WSL2 for Windows. To check inputs without fitting, append `--preflight-only`.

## Statistical procedure

1. Verify the fixed split, patient identities, representative samples, and true labels against raw data.
2. Apply detection-p and beta-range QC. Excessive patient missingness stops the workflow; do not silently replace holdout patients.
3. Compare 1,000-CpG panels selected by training-fold variance or training-fold LIMMA fusion-status differences.
4. Use five outer folds and three inner folds. Probe coverage, imputation, scaling, and selection are fitted within each training fold. Tune Ridge C by inner-fold balanced accuracy.
5. Select the teacher using nested OOF balanced accuracy, then ROC AUC; ties favor LIMMA. Post-selection OOF performance is exploratory, not an unbiased final estimate.
6. Refit each teacher on all 108 teacher-training patients and score validation. Test scoring is disabled by default.
7. Predict eligible unlabeled patients. The initial inclusion rule is `max(p, 1-p) >= 0.90`; this model score is not a guarantee of 90% correctness.
8. Export accepted silver labels with provenance and suggested downstream weights gold=1.0, silver=0.3.

LIMMA uses clipped beta values to compute M-values, fits `M ~ fusion_status` with `lmFit`, applies `eBayes`, computes two-sided p-values and BH corrections, and ranks by raw p-value with deterministic ties. No FDR cutoff is required for the predictive top-k panel. Ridge inputs are standardized beta values for both selection methods.

## Outputs and resuming

Default output directory: `outputs/teacher_v1/`.

- `results_summary.csv`: OOF and validation metrics for both teachers.
- `teacher_selection.json`: selection rule and selected teacher.
- `variance/teacher/`, `limma/teacher/`: models, OOF predictions, and panel IDs.
- `silver/silver_labels.csv`: scores, predictions, eligibility, and label provenance.
- `silver/student_train.csv`: gold and accepted silver training rows.
- `silver/paired_selected_cpg_beta_imputed.csv`: selected features with split assignments; never combine holdout rows into training.
- `silver/image_input_manifest.csv`, `silver/panel_provenance.json`: downstream input contracts.
- `return_results.zip`: private return package, including models, selected inputs, results, and logs. Never commit it.

```bash
bash run.sh --data-dir "/path/to/COMET_RMS_Data" --resume
bash run.sh --data-dir "/path/to/COMET_RMS_Data" --resume --include-test
bash run.sh --self-test
```

Enable test scoring only after fixing selection rules. Changed data, splits, code, or parameters require a new output directory rather than mixing cached runs.

## Source layout

`src/ridge_teacher/cohort/` defines identity and training eligibility; `data/` handles raw inputs and contracts; `features/` implements fold-local selection; `train/` implements CV and fitting; `inference/` exports predictions and silver labels; `evaluation/` computes metrics. `image_processing/manifest.py` links images but does not encode WSI.

The MedGemma student workflow must fit a fresh supervised ABMIL selector under the new split. It must not reuse checkpoints trained on the new holdout patients.
