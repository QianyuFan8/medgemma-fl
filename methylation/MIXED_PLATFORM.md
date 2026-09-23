# Mixed-platform seven-class experiment: data audit stage

This is a new experiment, not a replacement for the five-class experiment.
The current entry point audits input headers and cohort membership only. It does
not yet prepare local panels, train a seven-class model, or evaluate mixed panels.
Do not use the existing single-panel preparation/evaluation commands for this experiment.

## Design

- Sites 1 and 2 will use 450K patients; site 3 will use EPIC patients.
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

No training configuration is emitted at this stage. Local-panel preparation,
seven-class export, and multi-panel evaluation follow after this audit is reviewed.
Never fit limma on validation or test labels. Use new processed-data and result
directories for those later stages, leaving `data/methylation_450k_v1` and
`runs/methylation_450k` unchanged.
