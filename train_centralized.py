"""One training implementation for python (1 GPU) and torchrun (DDP).

Train on an existing one-client export; never read the test set for training.
Checkpoints use the existing adapter-only inference envelope. DDP replicates
the model; it does not combine GPUs' VRAM into one large memory pool.
"""
import argparse
import json
import math
from pathlib import Path
import time

from training_performance import add_performance_args, effective_batch_size, accumulation_for_world


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    add_performance_args(p)
    p.add_argument('--data-root', required=True, help='Existing centralized export with clients/site-1/train.json')
    p.add_argument('--output', required=True, help='New output directory (existing paths are refused)')
    p.add_argument('--model_name_or_path', default='google/medgemma-1.5-4b-it')
    p.add_argument('--adapter_scope', choices=('lora-only', 'language-only', 'legacy'), default='lora-only')
    p.add_argument('--attn_implementation', choices=('eager', 'sdpa', 'flash_attention_2'), default='sdpa')
    p.add_argument('--num_train_epochs', type=int, default=60)
    p.add_argument('--max_steps', type=int, default=None, help='Short timing/smoke run instead of epoch training')
    p.add_argument('--save_adapter_epochs', type=int, nargs='*', default=[])
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--per_device_eval_batch_size', type=int, default=1)
    p.add_argument('--gradient_accumulation_steps', type=int, default=4)
    p.add_argument('--effective_batch_size', type=int, default=None,
                   help='Optional target batch across ranks; overrides gradient accumulation')
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--max_seq_length', type=int, default=32768)
    p.add_argument('--lora_rank', type=int, default=16)
    p.add_argument('--seed', type=int, default=20261011)
    p.add_argument('--logging_steps', type=int, default=1)
    p.add_argument('--eval_steps', type=int, default=1000000)
    return p


def main():
    args = parser().parse_args()
    if min(args.num_train_epochs, args.per_device_train_batch_size,
           args.per_device_eval_batch_size, args.gradient_accumulation_steps) < 1:
        raise ValueError('Epochs and batch settings must be positive')
    if args.save_adapter_epochs and (args.max_steps is not None or
            any(e < 1 or e > args.num_train_epochs for e in args.save_adapter_epochs)):
        raise ValueError('Epoch snapshots require complete epochs within the training budget')
    import torch
    from transformers import AutoProcessor, set_seed
    from trl import SFTTrainer
    from client import _load_site_split, _build_training_args, EpochAdapterCheckpoint
    from distributed_training import DistributedClient
    from model import create_peft_medgemma_model, get_adapter_state_dict, MEDGEMMA_IMAGE_TOKEN_ID
    from adapter_scope import trainable_summary
    from methylation_validate import validate_clients
    from methylation_utils import build_methylation_collator
    from subtype_utils import build_multimodal_collator
    from training_cache import maybe_cache_collator
    from utils import require_supported_gpu, sync_cuda, get_peak_cuda_memory_usage_mb

    distributed = DistributedClient()
    require_supported_gpu()
    if args.effective_batch_size is not None:
        args.gradient_accumulation_steps = accumulation_for_world(
            args.effective_batch_size, args.per_device_train_batch_size, distributed.world)
    set_seed(args.seed)
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    task = validate_clients(root/'clients', 1, require_patient_groups=True)
    if distributed.rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    args.task, args.report_to = 'subtype', 'none'
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    processor.tokenizer.padding_side = 'right'
    train = _load_site_split(str(root/'clients/site-1/train.json'), str(root), 'subtype')
    validation_path = root/'validation.json'
    validation = (_load_site_split(str(validation_path), str(root), 'subtype')
                  if args.local_eval and validation_path.exists() else None)
    if train[0].get('image'):
        collator = build_multimodal_collator(processor, args.max_seq_length,
                    getattr(processor, 'image_token_id', None) or MEDGEMMA_IMAGE_TOKEN_ID)
    else:
        collator = build_methylation_collator(processor, args.max_seq_length)
    collator = maybe_cache_collator(collator, processor, args)
    model = create_peft_medgemma_model(args.model_name_or_path, quantized=args.quantized,
                device_map={'': distributed.local_rank}, lora_rank=args.lora_rank,
                adapter_scope=args.adapter_scope, attn_implementation=args.attn_implementation,
                gradient_checkpointing=args.gradient_checkpointing)
    audit = trainable_summary(model, args.adapter_scope)
    trainer = SFTTrainer(model=model, args=_build_training_args(args, str(output), validation is not None),
                         train_dataset=train, eval_dataset=validation,
                         processing_class=processor, data_collator=collator)
    callback = EpochAdapterCheckpoint(args.save_adapter_epochs)
    trainer.add_callback(callback)
    if distributed.rank == 0:
        manifest = dict(arguments=vars(args), world_size=distributed.world, train_records=len(train),
                        effective_batch_size=effective_batch_size(args, distributed.world),
                        trainable_parameters=audit, task=task,
                        sampler_note='DDP may repeat records to equalize rank batches; not exact single-GPU sample order.',
                        estimated_records_per_ddp_epoch=math.ceil(len(train)/(distributed.world*args.per_device_train_batch_size))*
                            distributed.world*args.per_device_train_batch_size if distributed.world > 1 else len(train))
        (output/'run_manifest.json').write_text(json.dumps(manifest, indent=2))
    sync_cuda()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = trainer.train()
    sync_cuda()
    elapsed = time.perf_counter()-started
    peak_allocated, peak_reserved = get_peak_cuda_memory_usage_mb()
    evaluation = trainer.evaluate() if validation is not None else None
    report = dict(rank=distributed.rank, world_size=distributed.world, train_runtime_sec=elapsed,
                  global_step=trainer.state.global_step, epoch=trainer.state.epoch,
                  training_loss=result.training_loss, trainer_metrics=result.metrics,
                  peak_allocated_mb=peak_allocated, peak_reserved_mb=peak_reserved,
                  cache=getattr(collator, 'stats', None), validation_loss=evaluation)
    (output/f'rank-{distributed.rank}_performance.json').write_text(json.dumps(report, indent=2))
    if distributed.rank == 0:
        if callback.saved != set(args.save_adapter_epochs):
            raise RuntimeError('Not all requested epoch checkpoints were saved')
        torch.save({'model': {'model.'+k: v for k,v in get_adapter_state_dict(model).items()}}, output/'final.pt')
        (output/'completed.json').write_text(json.dumps(report, indent=2))
    distributed.barrier()
    distributed.close()


if __name__ == '__main__':
    main()
