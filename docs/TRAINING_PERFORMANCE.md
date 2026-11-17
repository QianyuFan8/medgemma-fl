# Single- and multi-GPU training

The same client/model code supports a single CUDA GPU or replicated DDP.
Existing launches keep 4-bit QLoRA, non-reentrant activation checkpointing,
and local validation enabled. Performance options are explicit and recorded.
These changes are for new runs; do not update code underneath a running queue.

## Federated experiments: one machine, 1 / 4 / 8 GPUs

Run from the repository on the VM after installing this code. Existing prepared
`data/federated_spatial16_v1` is reused, including the patient-disjoint split.

```bash
# One GPU: three clients share it, protected by the existing process lock.
bash scripts/launch_federated_tiles16.sh lora-only 60 1 --gpus 0

# Four GPUs: client groups [0,1], [2], [3].
bash scripts/launch_federated_tiles16.sh lora-only 60 1 --gpus 0,1,2,3

# Eight GPUs: client groups [0,1,2,3], [4,5], [6,7].
bash scripts/launch_federated_tiles16.sh language-only 60 1 --gpus 0,1,2,3,4,5,6,7
```

Choose ONE command. The launcher returns immediately, uses `nohup`, runs smoke
checks for the four FL configurations, and then runs the full queue. Hardware
and batch settings are checked before the background queue starts. Use `--gpus`
to choose devices in a normal shell without a pre-set `CUDA_VISIBLE_DEVICES` mask.
The VM must stay on; the laptop need not stay connected.

Each client with multiple GPUs gets its own `torchrun` group. Rank zero alone
receives/uploads adapters; all ranks train synchronously. Disjoint GPU groups
can run concurrently. Identical GPU groups share a lock; partially overlapping
groups are rejected. Global candidate scoring remains single-GPU and uses a
requested GPU between training jobs. This is one machine with several GPUs,
not DDP across the two existing VMs over SSH.

The effective batch per client defaults to 4. Accumulation is adjusted as
`effective batch / (per-device batch × GPUs in that client)`; invalid fractional
settings fail. For example, the 8-GPU layout uses accumulation 1 / 2 / 2 with
per-device batch 1. For custom experiments, `train/federated.py --gpu '[0,1],[2],[3]'`
also supports these groups with `--effective_batch_size 4`.

## Centralized training: same program, different launcher

`--data-root` must be an existing centralized export with `clients/site-1/train.json`,
`validation.json`, images, and the existing patient mapping. Do not pass a
three-client federated export here. `--output` must not exist.

```bash
source .venv/bin/activate
export HF_HOME=/home/shadeform/comet_storage/hf
CENTRAL_DATA="$(cat .multimodal_tiles60_spatial16.latest)/full/data"

# One GPU; effective batch = 1 × 4 × 1 = 4.
CUDA_VISIBLE_DEVICES=0 python train/centralized.py \
  --data-root "$CENTRAL_DATA" --output runs/central_perf_gpu1 \
  --adapter_scope lora-only --num_train_epochs 60 \
  --effective_batch_size 4 --save_adapter_epochs 2 5 10 20 30 50 60

# Four GPUs; effective batch = 1 × 1 × 4 = 4.
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 train/centralized.py \
  --data-root "$CENTRAL_DATA" --output runs/central_perf_gpu4 \
  --adapter_scope lora-only --num_train_epochs 60 \
  --effective_batch_size 4 --save_adapter_epochs 2 5 10 20 30 50 60

# Eight GPUs require an effective batch of at least 8 with one record/GPU.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 train/centralized.py \
  --data-root "$CENTRAL_DATA" --output runs/central_perf_gpu8 \
  --adapter_scope lora-only --num_train_epochs 60 \
  --effective_batch_size 8 --save_adapter_epochs 2 5 10 20 30 50 60
```

These centralized commands run in the foreground. To disconnect, run the chosen
command in tmux (detach with Ctrl-b d), or wrap it in `nohup bash ...` with output
redirected and stdin from `/dev/null`. Do not launch all three commands together.
`final.pt` and epoch snapshots contain adapters; they are for existing inference
loaders, not optimizer-state resume. This entry point performs optional validation
loss, not the separate candidate-answer classification evaluation. It does not
use test labels for training or checkpoint choice.

## Speed / memory options

Append these to either the federated launcher or `train/centralized.py`:

| Option | What changes | Trade-off |
|---|---|---|
| `--no-local_eval` | Skip local validation loss during/after local training | FL queue still evaluates global checkpoints with candidate scores |
| `--preprocess_cache_dir /local/ssd/medgemma-cache` | Cache processor output tensors and labels on local disk | Extra disk/I/O; often most useful for repeated batch-size-1 examples |
| `--preprocess_cache_max_gb 32` | Set the shared cache disk budget | When full, new batches are processed normally without being cached |
| `--no-gradient_checkpointing` | Store activations instead of recomputing them | More VRAM, potentially faster backward; may OOM |
| `--no-quantized` | BF16 base weights with LoRA rather than 4-bit QLoRA | More VRAM; speed must be benchmarked, not assumed |
| `--per_device_train_batch_size 2 --effective_batch_size 4` | Larger microbatch, less accumulation if divisible | More VRAM; only valid if the hardware group fits this effective batch |

Activation checkpointing was ALREADY enabled. It saves VRAM by repeating forward
computation during backward; it is not a disk-cache speed optimization. BF16,
fused AdamW, and SDPA were already used by the experiment queues. FlashAttention 2
is an existing optional backend, but installing it is not required for this change.

The new cache stores only deterministic CPU preprocessing (tokens, masks,
processed pixels, labels). It does not cache trainable vision embeddings, does
not freeze the vision encoder, and does not change prompts/tiles/CpG precision.
Its key covers the ordered batch, image pixels, labels, processor/tokenizer state,
package versions, and preprocessing code. Writes are atomic and disk-budgeted;
corrupt entries are recomputed. Never enable it with random image augmentation.
Keep caches local to the training site. Higher batch sizes produce fewer repeated
batch combinations, so cache reuse can fall; slow/network storage can be slower
than fresh preprocessing. Cache hit/miss and time counters are saved.

## What remains comparable, and what does not

- Patient splits, labels, LoRA scope, FL methods and original-record aggregation
  weights are preserved. Hardware settings are included in the run manifest.
- DDP may repeat records to give ranks equal batch counts. With only 38 training
  records, this matters: 38 on four ranks with microbatch 1 can become 40 draws.
  Client shards are even smaller. The performance reports record the expected
  padded epoch size. Equal effective batch does not imply identical sample order
  or exactly the same single-GPU trajectory.
- DDP replicates the base model per GPU; it does not pool VRAM. This implementation
  does not claim FSDP/ZeRO/tensor-parallel support. More GPUs are not guaranteed to
  be faster for a small cohort; communication and startup can dominate.
- The pre-existing single-GPU FedLC computation is retained. DDP FedLC must use
  the wrapped model forward to trigger gradient reduction; its custom mean loss
  is explicitly scaled by Trainer gradient accumulation. Treat new distributed
  runs as a new training configuration, not bitwise reproductions of old runs.

## Benchmark before a long run

Use a NEW output directory for each short `--max_steps` run, without epoch snapshot
arguments. Start with the default, then change one setting at a time. Compare
the same hardware, examples and effective batch. Allow warm-up and report both
cold-cache and warm-cache timing; do not quote the first step as steady-state speed.

Each rank writes training wall time, peak allocated/reserved VRAM, global step,
cache counters, and Trainer metrics. FL clients write these under round-specific
diagnostics, and the queue records per-client GPU/accumulation settings.
Before a full 60-round run, check finite loss, saved adapters, expected trainable
scope, and an unchanged validation evaluation procedure.

The CPU regression suite includes real two-process Gloo tests for rank-zero-only
FLARE exchange and FedLC gradients against a serial reference, plus exact cache
tensor/label checks and CLI/GPU allocation tests. Full MedGemma CUDA/NCCL runs on
4/8 GPUs still require hardware smoke tests; no speedup multiplier is claimed.

References: [PEFT multi-GPU QLoRA](https://github.com/huggingface/peft/blob/main/examples/sft/README.md),
[FLARE client API](https://nvflare.readthedocs.io/en/2.9.0/programming_guide/execution_api_type/client_api.html),
[Transformers parallelism](https://huggingface.co/docs/transformers/en/perf_train_gpu_many).
The implementation was also checked against the installed FLARE recipe API.
