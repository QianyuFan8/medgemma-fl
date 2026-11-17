# MedGemma-FL

Centralized and federated MedGemma LoRA/QLoRA experiments for pediatric RMS FOXO1 fusion-status prediction, combining H&E tiles and DNA methylation.

## Start here

- [目录与代码阅读指南](docs/REPOSITORY_GUIDE_ZH.md)
- [Current gold/silver student pipeline](pipelines/medgemma_student/README.md)
- [VM setup, DNAnexus download and one/four-GPU tests](pipelines/medgemma_student/VM_DNANEXUS.md)
- [DDP and performance settings](docs/TRAINING_PERFORMANCE.md)
- [Ridge teacher handoff](pipelines/silver_labels/README.md)

## Layout

```text
train/                    Centralized, federated job and client CLI entry points
inference/                Candidate scoring, evaluation, threshold and summary CLIs
src/medgemma_fl/           Actual model, training, loss, DDP and shared implementation
  training/               Centralized loop, NVFlare job builder and local client loop
  evaluation/             Evaluation and scoring implementation
  preprocessing/          Cohort and input preparation implementation
  experiments/            Historical experiment queue implementation
pipelines/
  silver_labels/          Patient splits, fold-local CpG selection, Ridge and silver labels
  medgemma_student/       DX → H-Optimus → new ABMIL → spatial16 → student inputs
data_preprocessing/      WSI/ABMIL tools and cohort/input preparation CLIs
experiments/              Historical ablation queue CLIs
scripts/                  Shell launchers, VM preparation, checks and bundle creation
tests/                    Regression tests
docs/                     Usage, architecture and performance documentation
methylation/              Earlier R-based preprocessing/Ridge scripts
UMAP/                     Earlier exploratory analysis (outputs stay local)
assets/                   Documentation figures
```

## Install and use

Install the appropriate CPU/CUDA dependencies first (see the VM guide). In the activated environment, install this checkout:

```bash
python -m pip install -e . --no-deps
python train/centralized.py --help
python train/federated.py --help
python inference/medgemma.py --help
python -m pipelines.medgemma_student --help
```

Public entry points call the implementation in `src/medgemma_fl/`; they do not contain a second copy of the training logic. Imports use the `medgemma_fl` namespace. The editable install deliberately does not choose or replace your PyTorch/CUDA runtime. The Ridge handoff remains independently runnable with `pipelines/silver_labels/run.sh`.

## Migration

Root CLI paths have moved: `train_centralized.py` → `train/centralized.py`, `job.py` → `train/federated.py`, `client.py` → `train/client.py`, `evaluate_subtype.py` → `inference/medgemma.py`. Maintained launchers use the new paths. See the [complete mapping](docs/entrypoint_migration.json).

For a historical copied command, use the compatibility dispatcher:

```bash
python scripts/legacy_entrypoint.py train_centralized.py --help
```

The literal old command `python job.py ...` no longer works; root-level shim files are not retained. Historical external archives and saved results are unchanged. Use the **v2 student VM bundle** after this reorganization.

Patient inputs, credentials, model weights and VM backups are separate from source code. `data/`, `runs/` and `logs/` are ignored local runtime directories. CPU tests and job export checks do not replace actual CUDA execution.
