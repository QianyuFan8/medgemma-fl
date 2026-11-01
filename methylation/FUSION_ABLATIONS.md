# Fixed-cohort LoRA-only and spatial-selection ablations

For new-VM installation, backup restoration, and all ten runs, use [the Chinese VM guide](VM_LORA_ONLY_10_RUNS_ZH.md).

Use the existing patient-disjoint 38/10/12 record split (35/10/11 patients). Keep the supplied 849-CpG panel and the documented feature-selection caveat. No labels, splits, beta values, or baseline predictions are changed by these tools. Patient grouping must also be considered when reporting metrics.

## A. First change only the trainable parameter scope

The previous clients used 4-bit base weights plus LoRA and fully trainable `lm_head` / `embed_tokens`. New `--adapter_scope lora-only` omits the latter two modules while keeping rank 16 and existing LoRA target layers. The base model, input montages, methylation text, scheduler, learning rate, seeds and 5 rounds × 2 local epochs remain unchanged for the first comparison. Quantization remains unchanged: this is still QLoRA-style training.

`legacy` remains the default for job.py/client.py, preserving older experiments. The ablation launcher explicitly defaults to `lora-only`. Clients fail if an unintended non-LoRA tensor is trainable and write `trainable_parameters.json`. Server and clients receive the same scope. The `.pt` inference loader detects whether full embedding/head weights were stored, so old and new checkpoints remain distinct and loadable.

Removing full matrix training reduces optimizer and communication costs but does not eliminate the long-context forward/backward computation. Measure wall-clock time, peak memory and communication rather than promising a speedup. Start from the base model, not from the old trained checkpoint. This comparison does not reduce LoRA rank or increase training time simultaneously.

### Update the VM code

Update the shared branch code, or transfer these files from this checkout to the matching VM checkout: `adapter_scope.py`, `model.py`, `client.py`, `job.py`, `inference_utils.py`, `data_preprocessing/tile_selection.py`, `data_preprocessing/abmil_montage.py`, `data_preprocessing/audit_tile_selection.py`, `scripts/run_fusion_ablation.sh`, `prepare_comet_subtype.py`. The standalone scripts/check_fusion_splits.py must also be included. Preserve VM data and checkpoints.

Run from `/home/ubuntu/medgemma-fl` (adjust if your checkout is elsewhere):

```bash
source .venv/bin/activate
export COMET_ADAPTER_SCOPE=lora-only
export COMET_ROUNDS=5 COMET_EPOCHS=2 COMET_CONTEXT=32768
export COMET_SPLIT=validation COMET_FL_METHOD=fedavg
bash scripts/run_fusion_ablation.sh central-smoke
bash scripts/run_fusion_ablation.sh central-train
bash scripts/run_fusion_ablation.sh central-eval

export COMET_FL_LAYOUT=iid COMET_FL_METHOD=fedavg
bash scripts/run_fusion_ablation.sh fl-smoke
bash scripts/run_fusion_ablation.sh fl-train
bash scripts/run_fusion_ablation.sh fl-eval
```

After smoke tests succeed, the remaining three arms use the same launcher:

```bash
for arm in iid:fedlc skew08:fedavg skew08:fedlc; do
  export COMET_FL_LAYOUT=${arm%:*} COMET_FL_METHOD=${arm#*:}
  bash scripts/run_fusion_ablation.sh fl-smoke
  bash scripts/run_fusion_ablation.sh fl-train
  bash scripts/run_fusion_ablation.sh fl-eval
done
```

Outputs are isolated in `runs/ablation_lora-only_top16_r5_e2/`; existing training directories are rejected. Centralized and FL can run on separate VMs with identical inputs. Do not run these jobs concurrently on one GPU unless capacity is verified.

Do not expand the training budget until this matched ablation is inspected. An optional predeclared longer-budget arm uses `COMET_ROUNDS=10 COMET_EPOCHS=2`, which automatically changes the output directory. More training is not guaranteed to help. Compare validation performance across seeds / budgets without choosing on test. Since the existing test has already informed development, repeated test results are exploratory; retain new independent patients for final confirmation.

## B. Audit the old selections without SVS or GPU

The backup contains H5 features/coordinates, attention arrays and montage indices. These suffice for a geometric audit. It does not contain original SVS; coordinates cannot establish tumor content or scanner/staining quality.

```bash
python -m pip install matplotlib
python data_preprocessing/audit_tile_selection.py \
  --embeddings data/wsi/hoptimus1 \
  --abmil runs/fixed849_disjoint/abmil \
  --prepared data/fixed849_disjoint_paired \
  --existing-index data/wsi/fixed849_disjoint_montages/montage_index.tsv \
  --output runs/tile_audit_top_vs_spatial
```

The audit verifies the old montage coordinates against top attention before comparing strategies. It uses no outcome labels. Plot examples come from train/validation only. Metrics use the sampled tile coordinate extent, not the full WSI or annotated tumor extent. An adjacent component is connected by edges <=1.5 tile widths; flagging 8 adjacent tiles is descriptive, not a validated clinical QC threshold.

Local 2026-10-31 audit: 60/60 old selections matched their recorded montage index. Top attention had 2/60 slides with >=8 adjacent tiles, versus 0/60 with the proposed spatial selector. The median nearest-neighbor distance per slide, summarized across slides, changed from 2.92 to 4.24 tile widths. This is a geometry result, not a classification improvement. Broad regional concentration can remain even when neighboring duplicates are removed.

## C. Generate a separate spatial montage arm

The proposed selector takes the top 128 attention candidates, greedily chooses up to 16 in attention order with a 2-tile-width minimum coordinate distance, then explicitly reports any half-distance or zero-distance fallback needed to fill slots. Defaults were not optimized using fusion outcomes. Candidate restriction avoids automatically selecting very low-attention regions merely to maximize spread. No spatial selector guarantees correct pathology or external-site generalization.

With SVS available, reuse the existing H-Optimus features and ABMIL attention; no encoder or ABMIL retraining is required for this selection-only comparison:

```bash
python data_preprocessing/abmil_montage.py montage \
  --embeddings data/wsi/hoptimus1 --abmil runs/fixed849_disjoint/abmil \
  --slide-dir data/raw/comet_slides --selection spatial \
  --candidate-pool 128 --min-distance-tiles 2 \
  --output data/wsi/fixed849_spatial16_montages
python prepare_comet_subtype.py export --prepared data/fixed849_disjoint_paired \
  --modality multimodal --n-clients 1 \
  --montage-dir data/wsi/fixed849_spatial16_montages --output data/fixed849_spatial16_central
python prepare_comet_subtype.py export --prepared data/fixed849_disjoint_paired \
  --modality multimodal --n-clients 3 --partition iid \
  --montage-dir data/wsi/fixed849_spatial16_montages --output data/fixed849_spatial16_iid
```

The updated exporter reads selection_audit.json and automatically describes the spatial-selection rule consistently in all exported JSON inputs. No manual prompt edits are needed. Check that exported record sets and FL client assignments match the corresponding top16 experiment. The selection rule and its factual prompt description change together; do not add outcome-specific descriptions.

```bash
export COMET_DATA_PREFIX=data/fixed849_spatial16 COMET_INPUT_TAG=spatial16
export COMET_ADAPTER_SCOPE=lora-only COMET_ROUNDS=5 COMET_EPOCHS=2
export COMET_FL_METHOD=fedavg COMET_SPLIT=validation
bash scripts/run_fusion_ablation.sh central-smoke
bash scripts/run_fusion_ablation.sh central-train
bash scripts/run_fusion_ablation.sh central-eval
```

Compare this against LoRA-only + top16 at 5×2, not only against legacy + top16, to separate the parameter-scope and selection effects. All MedGemma models compared within an arm should use the same montage set. Do not overwrite baseline montages.

## External-site checks

Persist slide ID, tile coordinates, attention, selection configuration, spacing statistics and fallback counts for every new slide. Automatically flag outlying concentration or insufficient candidates against training-defined reference distributions. Review selected patches and the WSI thumbnail for tissue/artifact quality. For real robustness evidence, evaluate held-out institutions/scanners or prospectively collected external slides; simulated label skew alone does not validate unseen-site robustness. Attention bias can affect all 128 candidates, so future quality-filtered feature-diversity or spatially stratified sampling is a useful separate control.

## Validation limits

CPU tests cover checkpoint-scope inference, rejecting unexpected trainable weights, configuration construction, spatial selection, fallback reporting and existing cohort/export behavior. Spatial auditing ran on all 60 saved coordinate/attention pairs. Cloud GPU training, PEFT/NVFlare integration and raw-SVS montage rendering require the VM smoke test; they were not executed locally.
