# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Client script for federated MedGemma fine-tuning with TRL SFTTrainer and LoRA-only exchange.

Adapted from the Google Health MedGemma Hugging Face fine-tuning notebook:
https://github.com/google-health/medgemma/blob/main/notebooks/fine_tune_with_hugging_face.ipynb
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import signal
import sys
import time

from training_performance import add_performance_args, effective_batch_size
from training_cache import maybe_cache_collator
from distributed_training import DistributedClient
from data_utils import DEFAULT_MODEL_NAME_OR_PATH, format_training_example, resolve_image_path
from datasets import Image, load_dataset
from methylation_utils import build_methylation_collator, format_methylation_example
from subtype_utils import build_multimodal_collator, format_subtype_example
from lora_utils import build_uniform_lora_rank_map, truncate_global_bank_for_site
from fl_methods import METHODS, FLMethodState, split_scaffold_params
from model import MEDGEMMA_IMAGE_TOKEN_ID, apply_adapter_state, create_peft_medgemma_model, get_adapter_state_dict
from transformers import AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer
from utils import (
    abs_path,
    free_memory,
    get_cuda_memory_usage_mb,
    get_peak_cuda_memory_usage_mb,
    params_size_mb,
    require_supported_gpu,
    reset_peak_cuda_memory_stats,
    sync_cuda,
)

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey


class EpochAdapterCheckpoint(TrainerCallback):
    """Save inference-only adapter snapshots without changing Trainer scheduling."""

    def __init__(self, epochs):
        self.epochs = set(epochs)
        self.saved = set()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        import hashlib
        import json
        from pathlib import Path
        import torch

        epoch = round(state.epoch or 0)
        if (epoch not in self.epochs or epoch in self.saved or
                abs((state.epoch or 0) - epoch) > 1e-6 or not state.is_world_process_zero):
            return control
        root = Path(args.output_dir) / 'epoch_checkpoints'
        root.mkdir(parents=True, exist_ok=True)
        target = root / f'epoch-{epoch:02d}.pt'
        if target.exists():
            raise FileExistsError(target)
        temporary = target.with_suffix('.pt.tmp')
        # Use the same tensor naming/envelope as the existing inference loader.
        weights = {'model.' + k: v for k, v in get_adapter_state_dict(model).items()}
        torch.save({'model': weights}, temporary)
        temporary.replace(target)
        metadata = {'epoch': epoch, 'global_step': state.global_step,
                    'planned_epochs': args.num_train_epochs,
                    'checkpoint_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                    'purpose': 'inference_only_adapter_snapshot_not_optimizer_resume'}
        target.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
        self.saved.add(epoch)
        print(f'SAVED_EPOCH_ADAPTER epoch={epoch} step={state.global_step} path={target}', flush=True)
        return control


def _load_site_split(json_path: str, image_root: str, task: str = "histology"):
    dataset = load_dataset("json", data_files=json_path, split="train")
    if task == "methylation":
        return dataset.map(format_methylation_example)
    if task == "subtype":
        dataset = dataset.map(format_subtype_example)
        if dataset[0].get("image"):
            dataset = dataset.map(lambda example: {"image": resolve_image_path(example["image"], image_root)})
            dataset = dataset.cast_column("image", Image())
        return dataset
    dataset = dataset.map(lambda example: {"image": resolve_image_path(example["image"], image_root)})
    dataset = dataset.cast_column("image", Image())
    return dataset.map(format_training_example)


def _build_collate_fn(processor):
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id
    boi_token = tokenizer.special_tokens_map.get("boi_token")
    boi_token_id = tokenizer.convert_tokens_to_ids(boi_token) if boi_token else None

    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            images.append([example["image"].convert("RGB")])
            texts.append(
                processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False).strip()
            )

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        if boi_token_id is not None:
            labels[labels == boi_token_id] = -100
        labels[labels == MEDGEMMA_IMAGE_TOKEN_ID] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def _build_training_args(args, output_dir: str, do_eval: bool) -> SFTConfig:
    training_args = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="no",
        learning_rate=args.learning_rate,
        bf16=True,
        max_grad_norm=0.3,
        warmup_steps=0.03,
        lr_scheduler_type="linear",
        report_to=args.report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        prediction_loss_only=True,
    )
    if args.max_steps is not None:
        training_args["max_steps"] = args.max_steps
    if getattr(args, "seed", None) is not None:
        training_args["seed"] = args.seed
        training_args["data_seed"] = args.seed
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        # Image adapters are unused in CpG-only batches. Non-reentrant
        # checkpointing above is compatible with unused-parameter detection.
        training_args['ddp_find_unused_parameters'] = True
        training_args['ddp_timeout'] = 3600
    if args.task in ("methylation", "subtype"):
        # Keep warmup_steps above: the VM's Transformers/TRL accepts a
        # fractional value here and no longer accepts warmup_ratio.
        training_args["max_length"] = args.max_seq_length
    if do_eval:
        training_args["eval_strategy"] = "steps"
        training_args["eval_steps"] = args.eval_steps
    else:
        training_args["eval_strategy"] = "no"
    return SFTConfig(**training_args)


def main():
    parser = argparse.ArgumentParser(description="Federated MedGemma client with QLoRA-based local training.")
    add_performance_args(parser)
    parser.add_argument("--task", choices=("histology", "methylation", "subtype"), default="histology")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--client_gpu_lock", default=None,
                        help="Shared absolute lock path for external clients using one GPU.")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--attn_implementation", choices=("eager", "sdpa", "flash_attention_2"), default=None,
                        help="Attention kernel; default eager (notebook). Use sdpa for long CpG prompts.")
    parser.add_argument("--fl_method", choices=METHODS, default="fedavg",
                        help="Client-side non-IID method: fedavg (plain), fedprox, scaffold or fedlc.")
    parser.add_argument("--fedprox_mu", type=float, default=0.05, help="FedProx proximal coefficient mu.")
    parser.add_argument("--fedlc_tau", type=float, default=1.0, help="FedLC calibration strength tau.")
    parser.add_argument(
        "--data_path", type=str, default="./data/site-1", help="Site data directory containing train.json."
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="./NCT-CRC-HE-100K",
        help="Root directory used to resolve image paths stored in train.json and validation.json.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_NAME_OR_PATH,
        help="MedGemma Hugging Face model ID or local path.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per round. If omitted, one local epoch is used.",
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=1, help="Local epochs per round when max_steps is not set."
    )
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate (default: 2e-4).")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=4,
        help="Training batch size per device (default: 4).",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=4,
        help="Evaluation batch size per device (default: 4).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4).",
    )
    parser.add_argument("--logging_steps", type=int, default=25, help="Logging interval for trainer.train().")
    parser.add_argument("--eval_steps", type=int, default=50, help="Evaluation interval when validation data exists.")
    parser.add_argument(
        "--eval_subset_size",
        type=int,
        default=200,
        help="Maximum number of validation samples evaluated per round (default: 200). Use 0 for full validation.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="none",
        help='Trainer reporting backend, e.g. "none", "tensorboard", or "wandb".',
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Working directory for round-specific trainer outputs (default: ./medgemma_checkpoints under the client workspace).",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="Local site LoRA rank. Incoming global banks are truncated to this rank before loading.",
    )
    parser.add_argument("--adapter_scope", choices=("legacy", "lora-only", "language-only"), default="legacy")
    parser.add_argument('--save_adapter_epochs', type=int, nargs='+', default=[],
                        help='Save adapter-only inference snapshots at selected local epoch ends.')
    args = parser.parse_args()
    if args.save_adapter_epochs and (args.max_steps is not None or
            any(e < 1 or e > args.num_train_epochs for e in args.save_adapter_epochs)):
        parser.error('Snapshot epochs must be within num_train_epochs; max_steps is unsupported.')
    if args.seed is not None:
        from transformers import set_seed
        set_seed(args.seed)

    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))

    example_dir = os.path.dirname(os.path.abspath(__file__))
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    data_path = abs_path(args.data_path)
    image_root = abs_path(args.image_root)
    train_json = os.path.join(data_path, "train.json")
    validation_json = os.path.join(data_path, "validation.json")
    if not os.path.isfile(train_json):
        raise FileNotFoundError(f"Expected train.json at {train_json}.")

    distributed = DistributedClient(flare)
    require_supported_gpu()
    client_name = distributed.site_name()

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    train_dataset = _load_site_split(train_json, image_root, args.task)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples found in {train_json}. Re-run prepare_data.py with a non-empty site split."
        )
    if args.task == "methylation" or (args.task == "subtype" and not train_dataset[0].get("image")):
        collate_fn = build_methylation_collator(processor, args.max_seq_length)
    elif args.task == "subtype":
        # Read the image soft-token id from the processor so a MedGemma version change cannot desync it.
        image_token_id = getattr(processor, "image_token_id", None) or MEDGEMMA_IMAGE_TOKEN_ID
        collate_fn = build_multimodal_collator(processor, args.max_seq_length, image_token_id)
    else:
        collate_fn = _build_collate_fn(processor)
    collate_fn = maybe_cache_collator(collate_fn, processor, args)
    eval_dataset = (_load_site_split(validation_json, image_root, args.task)
                    if args.local_eval and os.path.isfile(validation_json) else None)

    print(
        f"site={client_name}, train_samples={len(train_dataset)}, validation_samples={len(eval_dataset or [])}, "
        f"local_lora_rank={args.lora_rank}"
    )
    with distributed.gpu_slot(args.client_gpu_lock, client_name, "initialize"):
        model = create_peft_medgemma_model(
            model_name_or_path=args.model_name_or_path,
            quantized=args.quantized,
            device_map={"": distributed.local_rank},
            lora_rank=args.lora_rank,
            adapter_scope=args.adapter_scope,
            attn_implementation=args.attn_implementation,
            gradient_checkpointing=args.gradient_checkpointing,
        )
        print(f"site={client_name}")
        model.print_trainable_parameters()
        from adapter_scope import trainable_summary
        parameter_audit = trainable_summary(model, args.adapter_scope)
        print(f"adapter_scope={args.adapter_scope}, trainable_parameters={parameter_audit['trainable_parameters']}")
        free_memory()
    fl_state = FLMethodState(args.fl_method, mu=args.fedprox_mu, tau=args.fedlc_tau,
                             classes=train_dataset[0].get("classes"))
    print(f"site={client_name}, fl_method={args.fl_method}, fedprox_mu={args.fedprox_mu}, fedlc_tau={args.fedlc_tau}")

    if args.work_dir is None:
        work_dir = os.path.join(os.getcwd(), "medgemma_checkpoints")
    else:
        work_dir = abs_path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    import json
    if distributed.rank == 0:
        with open(os.path.join(work_dir, "trainable_parameters.json"), "w") as f:
            json.dump(parameter_audit, f, indent=2)

    while distributed.is_running():
        input_model = distributed.receive()
        if input_model is None:
            break

        current_round = input_model.current_round
        received_mb = params_size_mb(input_model.params)
        print(f"site={client_name}, round={current_round}, received adapter size: {received_mb:.2f} MB")
        adapter_params, server_control = split_scaffold_params(input_model.params)
        if args.fl_method == "scaffold":
            bad = [k for k, v in adapter_params.items() if ".lora_A." in k and v.shape[0] != args.lora_rank]
            if bad:
                raise ValueError("SCAFFOLD requires every site rank to equal the global LoRA rank (no bank truncation)")
        site_rank_map = build_uniform_lora_rank_map(adapter_params.keys(), args.lora_rank)
        local_adapter_state = truncate_global_bank_for_site(adapter_params, site_rank_map)
        with distributed.gpu_slot(args.client_gpu_lock, client_name, f"round-{current_round}"):
            apply_adapter_state(model, local_adapter_state)
            model.train()
            fl_state.start_round(model, server_control, train_labels=list(train_dataset["label_name"]),
                                 tokenizer=processor.tokenizer)

            if eval_dataset is not None and args.eval_subset_size > 0 and len(eval_dataset) > args.eval_subset_size:
                round_eval_dataset = eval_dataset.shuffle(seed=42 + current_round).select(range(args.eval_subset_size))
            else:
                round_eval_dataset = eval_dataset

            round_output_dir = os.path.join(work_dir, f"round-{current_round}")
            if distributed.rank == 0 and os.path.isdir(round_output_dir):
                shutil.rmtree(round_output_dir)
            distributed.barrier()

            sync_cuda()
            round_start_time = time.perf_counter()
            reset_peak_cuda_memory_stats()
            round_start_allocated_mb, round_start_reserved_mb = get_cuda_memory_usage_mb()

            trainer = fl_state.make_trainer(
                SFTTrainer,
                model=model,
                args=_build_training_args(args, round_output_dir, round_eval_dataset is not None),
                train_dataset=train_dataset,
                eval_dataset=round_eval_dataset,
                processing_class=processor,
                data_collator=collate_fn,
            )
            epoch_callback = EpochAdapterCheckpoint(args.save_adapter_epochs)
            if args.save_adapter_epochs:
                trainer.add_callback(epoch_callback)

            sync_cuda()
            train_start_time = time.perf_counter()
            train_result = trainer.train()
            if distributed.rank == 0 and epoch_callback.saved != set(args.save_adapter_epochs):
                raise RuntimeError('Training did not save every requested epoch checkpoint')
            sync_cuda()
            train_runtime_sec = time.perf_counter() - train_start_time
            train_loss = float(getattr(train_result, "training_loss", float("nan")))
            os.makedirs(round_output_dir, exist_ok=True)
            diagnostics_name = 'training_diagnostics.json' if distributed.rank == 0 else f'rank-{distributed.rank}_performance.json'
            with open(os.path.join(round_output_dir, diagnostics_name), "w") as f:
                json.dump({"seed": args.seed, "learning_rate": args.learning_rate,
                           "num_train_epochs": args.num_train_epochs, "max_steps": args.max_steps,
                           "global_step": trainer.state.global_step, "epoch": trainer.state.epoch, "training_loss": train_loss,
                           "train_records": len(train_dataset), "log_history": trainer.state.log_history,
                           "performance": {"gradient_checkpointing": args.gradient_checkpointing,
                                           "quantized": args.quantized, "local_eval": args.local_eval,
                                           "world_size": distributed.world,
                                           "effective_batch_size": effective_batch_size(args, distributed.world),
                                           "estimated_records_per_ddp_epoch":
                                               math.ceil(len(train_dataset)/(distributed.world*args.per_device_train_batch_size))*
                                               distributed.world*args.per_device_train_batch_size
                                               if distributed.world > 1 else len(train_dataset),
                                           "train_runtime_sec": train_runtime_sec,
                                           "cache_cumulative": getattr(collate_fn, 'stats', None)}}, f, indent=2)
            metrics = {"loss": train_loss}
            eval_runtime_sec = 0.0
            if round_eval_dataset is not None:
                sync_cuda()
                eval_start_time = time.perf_counter()
                eval_metrics = trainer.evaluate()
                sync_cuda()
                eval_runtime_sec = time.perf_counter() - eval_start_time
                eval_loss = float(eval_metrics["eval_loss"])
                metrics["eval_loss"] = eval_loss
                metrics["neg_eval_loss"] = -eval_loss

            params = {"model." + key: value for key, value in get_adapter_state_dict(model).items()}
            params.update(fl_state.end_round(model))  # SCAFFOLD control variate, empty otherwise
            sent_mb = params_size_mb(params)
            sync_cuda()
            round_runtime_sec = time.perf_counter() - round_start_time
            peak_allocated_mb, peak_reserved_mb = get_peak_cuda_memory_usage_mb()
            peak_allocated_delta_mb = max(0.0, peak_allocated_mb - round_start_allocated_mb)
            peak_reserved_delta_mb = max(0.0, peak_reserved_mb - round_start_reserved_mb)
            meta = {"num_examples": len(train_dataset)}
            if args.max_steps is not None:
                meta[FLMetaKey.NUM_STEPS_CURRENT_ROUND] = args.max_steps
            else:
                meta["NUM_TRAIN_EPOCHS_CURRENT_ROUND"] = args.num_train_epochs
            # Exported adapters are CPU tensors. Release optimizer, gradients and
            # cached activations BEFORE another client acquires the GPU slot.
            model.zero_grad(set_to_none=True)
            del trainer
            free_memory()
        distributed.send(flare.FLModel(params_type=flare.ParamsType.FULL, params=params, metrics=metrics, meta=meta))
        print(
            f"site={client_name}, round={current_round}, local_lora_rank={args.lora_rank}, "
            f"train_runtime_sec={train_runtime_sec:.2f}, eval_runtime_sec={eval_runtime_sec:.2f}, "
            f"round_runtime_sec={round_runtime_sec:.2f}, "
            f"cuda_allocated_start_mb={round_start_allocated_mb:.2f}, "
            f"cuda_reserved_start_mb={round_start_reserved_mb:.2f}, "
            f"cuda_peak_allocated_mb={peak_allocated_mb:.2f}, "
            f"cuda_peak_reserved_mb={peak_reserved_mb:.2f}, "
            f"cuda_peak_allocated_delta_mb={peak_allocated_delta_mb:.2f}, "
            f"cuda_peak_reserved_delta_mb={peak_reserved_delta_mb:.2f}"
        )
        print(f"site={client_name}, round={current_round}, sent updated adapter size: {sent_mb:.2f} MB")

        del params, input_model, local_adapter_state
        free_memory()
    distributed.close()


if __name__ == "__main__":
    main()
