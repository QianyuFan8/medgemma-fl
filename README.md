# MedGemma-FL

Federated LoRA/QLoRA fine-tuning of MedGemma 1.5 4B for pediatric rhabdomyosarcoma subtype classification (ERMS vs ARMS).

The current experiment uses COMET H&E whole-slide images and DNA methylation. Image tiling, H-Optimus-1 embeddings, and the ABMIL montage are in `data_preprocessing/`. Cohort audit, limma CpG selection, and client export are in `prepare_comet_subtype.py`. Training and evaluation entry points are `job.py` and `evaluate_subtype.py`.

Run the steps in [methylation/COMET_SUBTYPE.md](methylation/COMET_SUBTYPE.md).
