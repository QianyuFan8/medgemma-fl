# Weighted silver-label federated comparison

This experiment reuses the frozen centralized student inputs. It does not train
from a centralized checkpoint, change CpG selection, or move holdout patients
into training. Both validation and test retain their original gold labels.

## Primary comparisons

| Configuration | Sites | Local training method | Server aggregation |
|---|---:|---|---|
| IID + FedAvg | 3 | Patient-weighted answer-token cross-entropy | Record-count-weighted LoRA A/B averages |
| Skew 0.8 + FedAvg | 3 | Same loss | Same aggregation |
| Skew 0.8 + FedLC | 3 | Calibrate class logits, then apply patient-weighted loss | Same aggregation |

Use vision and language LoRA, rank 16, learning rate 5e-5, effective local batch
size 4, and initialization seed 20261117. Gold loss weight is 1.0 and silver loss
weight is 0.3. The reduction is mean(weight * patient mean answer-token loss),
not division by the sum of weights. Aggregation uses training record counts;
the silver discount is not applied a second time at the server. FedLC uses raw
local counts of the assigned training labels, including silver labels, and tau 1.

The initial budget is 60 rounds with one local epoch per round. All sites
participate. Save post-aggregation adapters at rounds 2, 5, 10, 20, 30, 50, and
60. Each local round creates a new AdamW optimizer and linear learning-rate
schedule, so this is not equivalent to continuous centralized training for
60 epochs. A later 12-round / 5-local-epoch comparison should be a separate run.

Patient ownership is frozen within each partition. Training assignments are
stratified by gold/silver provenance and label. Skew 0.8 denotes the existing
class-conditional allocation weights, not exactly 80% prevalence at every site;
inspect `site_patient_counts.json` for realized counts.

## Prepare and check

Run from the repository root after activating the CUDA environment. `SOURCE`
is the verified centralized student export, and `FL_DATA` must be a new path.

```bash
python data_preprocessing/prepare_federated_tiles16.py --source "$SOURCE" --output "$FL_DATA"
python experiments/run_silver_federated.py --data-root "$FL_DATA" --output "$SMOKE_OUTPUT" \
  --rounds 60 --local-epochs 1 --gpu-groups 0,1 2 3 --smoke
```

The smoke test performs one optimizer step at each site and verifies one global
aggregation for each experiment. It must complete successfully before a full run.

```bash
python experiments/run_silver_federated.py --data-root "$FL_DATA" --output "$RUN_OUTPUT" \
  --rounds 60 --local-epochs 1 --gpu-groups 0,1 2 3
```

Optional `--cache-dir /absolute/local/ssd/path` caches deterministic processor
tensors with a shared 20 GiB limit. It does not cache learned activations.

The three arguments to `--gpu-groups` correspond to the three table rows.
Experiments run concurrently on disjoint groups. Sites within an experiment
share its GPU group and take turns using a lock; a multi-GPU group uses DDP
inside each site. Effective batch size stays 4. DDP can pad a client's last
batch by repeating a record; per-client diagnostics record that distinction.
This is a single-VM simulation, not a measurement of inter-hospital networking.

## Outputs and evaluation

`manifest.json` records input hashes, source hashes, settings, and scheduling.
Each configuration contains its command, training log, status, per-site
diagnostics, and `global_checkpoints/round-NNN.pt`. A subprocess exit alone is
not accepted as success: every client round and global checkpoint is checked.

Holdout scoring is separate from training. Score all requested global
checkpoints on the unchanged validation set with POS/NEG candidate scoring and
threshold 0.5. Choose checkpoints using validation, then report the corresponding
test scores. Test labels must not influence training, calibration, or selection.
