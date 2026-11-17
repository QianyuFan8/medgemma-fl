# Experiment pipelines

- [Ridge teacher and silver labels](silver_labels/README_EN.md): patient splits, fold-local CpG selection, Ridge training, inference, and label provenance. This CPU workflow has its own Python/R environment and `src/` layout.
- [Gold/silver MedGemma student](medgemma_student/README_EN.md): DNAnexus downloads, frozen H-Optimus features, fresh train-only ABMIL selection, spatial16 inputs, and weighted training.

MedGemma CLI entry points live in `train/` and `inference/`. Shared implementations live in `src/medgemma_fl/`. Image preprocessing tools live in `data_preprocessing/`.
