# Repository and code guide

## Entry points and implementations

Public CLI wrappers call the installed `medgemma_fl` package; they do not duplicate training logic.

| Directory | Purpose |
|---|---|
| `train/` | Centralized training, federated job creation, and client training entry points |
| `inference/` | Candidate scoring, evaluation, thresholding, and summaries |
| `src/medgemma_fl/` | Models, LoRA, loss functions, DDP, and shared implementations |
| `data_preprocessing/` | WSI tissue detection, patches, encoders, ABMIL, and cohort preparation |
| `pipelines/silver_labels/` | Patient splits, subtype policy, fold-local CpG selection, and Ridge teacher |
| `pipelines/medgemma_student/` | DX download, fresh ABMIL, spatial16, and weighted student inputs |
| `experiments/` | Historical ablation and evaluation queues |
| `scripts/` | Launchers, hardware benchmarks, checks, and private bundle construction |
| `tests/` | Regression tests for data isolation, training, DDP, and job export |
| `docs/` | English documentation and entry-point migration map |
| `methylation/` | Earlier R-based preprocessing and Ridge code |
| `UMAP/` | Exploratory analysis code; generated results remain local |
| `assets/` | Documentation illustrations |

`data/`, `runs/`, `logs/`, private inputs, weights, and backups are not source code and must not be committed.

## Training code to read

- `train/centralized.py` calls `src/medgemma_fl/training/centralized.py`.
- `train/federated.py` calls `src/medgemma_fl/training/federated.py`.
- `train/client.py` calls `src/medgemma_fl/training/client.py` and is normally launched by NVFlare.
- `weighted_sft.py` implements patient-weighted answer-token loss.
- `distributed_training.py` implements client-internal distributed training support.
- `training_performance.py`, `training_benchmark.py`, and `training_cache.py` handle hardware settings, timing, and preprocessing caches.
- `adapter_scope.py` controls trainable LoRA scope; `model.py` loads models.
- `job_packaging.py` includes package sources in exported NVFlare jobs.

The current gold/silver weighted workflow targets centralized student training; it is not automatically supported by every federated method.

## Installation and migration

Install the appropriate CPU/CUDA dependencies, then run:

```bash
python -m pip install -e . --no-deps
python train/centralized.py --help
python train/federated.py --help
python inference/medgemma.py --help
```

Root Python entry points have moved. See [entrypoint_migration.json](entrypoint_migration.json).
Historical copied commands can use `python scripts/legacy_entrypoint.py OLD_FILENAME.py ...`.
Use the [VM runbook](../pipelines/medgemma_student/VM_DNANEXUS.md) for the current deployment.

All committed documentation must be English. Chinese notes may remain in ignored local files. Run `python3 scripts/check_english_docs.py` before publishing.
