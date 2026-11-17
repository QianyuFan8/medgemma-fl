# Experiment pipelines

`silver_labels/` is a self-contained CPU workflow with its own environment and src layout for patient splits, fold-specific CpG selection, Ridge training, inference, and silver-label provenance. Start with its README and run.sh.

Existing MedGemma training, federated client/server entry points and data_preprocessing image tools remain in their original locations for compatibility. They are not required to run the Ridge handoff bundle.
