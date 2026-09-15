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
import os
import random
import shutil
import signal
import sys
import time
from collections import Counter, defaultdict

import torch
from data_utils import (
    DEFAULT_MODEL_NAME_OR_PATH,
    TASK_RNASEQ,
    build_rnaseq_prompt,
    format_training_example,
    parse_diagnosis_label,
    parse_prediction_label,
    resolve_image_path,
)
from datasets import Image, load_dataset
from lora_utils import build_uniform_lora_rank_map, truncate_global_bank_for_site
from model import (
    DEFAULT_MODULES_TO_SAVE,
    MEDGEMMA_IMAGE_TOKEN_ID,
    apply_adapter_state,
    create_peft_medgemma_model,
    get_adapter_state_dict,
)
from transformers import AutoProcessor
from trl import SFTConfig, SFTTrainer
from utils import (
    abs_path,
    append_jsonl,
    free_memory,
    get_cuda_memory_usage_mb,
    get_peak_cuda_memory_usage_mb,
    lora_factor_params,
    params_size_bytes,
    params_size_mb,
    require_supported_gpu,
    reset_peak_cuda_memory_stats,
    sync_cuda,
)

import nvflare.client as flare
from nvflare.apis.fl_constant import FLMetaKey


def _parse_modules_to_save(raw_value: str | None, task: str) -> list[str] | None:
    if raw_value is None:
        return [] if task == TASK_RNASEQ else list(DEFAULT_MODULES_TO_SAVE)
    stripped = raw_value.strip()
    if not stripped:
        return []
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _load_site_split(json_path: str, image_root: str | None, task: str):
    dataset = load_dataset("json", data_files=json_path, split="train")
    if task == TASK_RNASEQ:
        return dataset.map(format_training_example)
    if not image_root:
        raise ValueError("--image_root is required for the histopathology task.")
    dataset = dataset.map(lambda example: {"image": resolve_image_path(example["image"], image_root)})
    dataset = dataset.cast_column("image", Image())
    return dataset.map(format_training_example)


def _label_counts(dataset) -> dict[int, int]:
    if "label" not in dataset.column_names:
        return {}
    return dict(sorted(Counter(int(label) for label in dataset["label"]).items()))


def _oversample_by_label(dataset, seed: int):
    """Duplicate minority classes in-site until each label matches the majority count."""
    if "label" not in dataset.column_names:
        return dataset
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset["label"]):
        by_label[int(label)].append(index)
    if len(by_label) <= 1:
        return dataset
    target = max(len(indices) for indices in by_label.values())
    rng = random.Random(seed)
    chosen: list[int] = []
    for indices in by_label.values():
        chosen.extend(indices[rng.randrange(len(indices))] for _ in range(target))
    rng.shuffle(chosen)
    return dataset.select(chosen)


def _build_text_collate_fn(processor):
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id

    def collate_fn(examples):
        texts = [
            processor.apply_chat_template(example["messages"], add_generation_prompt=False, tokenize=False).strip()
            for example in examples
        ]
        batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096)
        labels = batch["input_ids"].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


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


def _evaluate_classification_accuracy(model, processor, dataset, task: str, max_new_tokens: int) -> dict:
    tokenizer = processor.tokenizer
    device = next(model.parameters()).device
    model.eval()
    correct = 0
    unparsed = 0
    total = 0
    parse_fn = parse_diagnosis_label if task == TASK_RNASEQ else parse_prediction_label

    for example in dataset:
        if task == TASK_RNASEQ:
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": build_rnaseq_prompt(example["expression_text"])}],
                }
            ]
            prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False).strip()
            inputs = tokenizer(prompt_text, return_tensors="pt")
        else:
            prompt_text = processor.apply_chat_template(
                example["messages"][:1], add_generation_prompt=True, tokenize=False
            ).strip()
            inputs = processor(text=[prompt_text], images=[[example["image"].convert("RGB")]], return_tensors="pt")

        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response_text = tokenizer.decode(output_ids[0, prompt_length:], skip_special_tokens=True)
        predicted_index = parse_fn(response_text)
        if predicted_index < 0:
            unparsed += 1
        if predicted_index == int(example["label"]):
            correct += 1
        total += 1

    model.train()
    accuracy = correct / total if total else 0.0
    return {"accuracy": accuracy, "correct": correct, "total": total, "unparsed": unparsed}


def _build_training_args(args, output_dir: str, do_eval: bool) -> SFTConfig:
    training_args = dict(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
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
    )
    if args.max_steps is not None:
        training_args["max_steps"] = args.max_steps
    if do_eval:
        training_args["eval_strategy"] = "steps"
        training_args["eval_steps"] = args.eval_steps
    else:
        training_args["eval_strategy"] = "no"
    return SFTConfig(**training_args)


def main():
    parser = argparse.ArgumentParser(description="Federated MedGemma client with QLoRA-based local training.")
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
    parser.add_argument(
        "--task",
        type=str,
        choices=("histopathology", "rnaseq"),
        default="histopathology",
        help="Downstream fine-tuning task (default: histopathology).",
    )
    parser.add_argument(
        "--quantized",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the MedGemma base model in 4-bit (QLoRA). Use --no-quantized for bf16 LoRA.",
    )
    parser.add_argument(
        "--modules_to_save",
        type=str,
        default=None,
        help="Comma-separated extra PEFT modules to save. Empty string disables lm_head/embed_tokens.",
    )
    parser.add_argument(
        "--comm_log_dir",
        type=str,
        default=None,
        help="Directory for per-round JSONL communication and accuracy logs.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Optional experiment name written into communication logs.",
    )
    parser.add_argument(
        "--n_clients",
        type=int,
        default=3,
        help="Number of federated clients, used to compute bytes-per-round (default: 3).",
    )
    parser.add_argument(
        "--balance_labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Oversample minority diagnosis labels within each site before local SFT (default: off).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=16,
        help="Generated tokens used for per-round classification accuracy (default: 16).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda _signum, _frame: sys.exit(0))

    example_dir = os.path.dirname(os.path.abspath(__file__))
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)

    data_path = abs_path(args.data_path)
    image_root = abs_path(args.image_root) if args.image_root else None
    train_json = os.path.join(data_path, "train.json")
    validation_json = os.path.join(data_path, "validation.json")
    if not os.path.isfile(train_json):
        raise FileNotFoundError(f"Expected train.json at {train_json}. Run prepare_data.py or prepare_rnaseq.py first.")

    require_supported_gpu()
    flare.init()
    client_name = flare.system_info().get("site_name", "unknown")
    modules_to_save = _parse_modules_to_save(args.modules_to_save, args.task)

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True, use_fast=False)
    processor.tokenizer.padding_side = "right"
    collate_fn = _build_text_collate_fn(processor) if args.task == TASK_RNASEQ else _build_collate_fn(processor)

    train_dataset = _load_site_split(train_json, image_root, args.task)
    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples found in {train_json}. Re-run the matching prepare script with a non-empty site split."
        )
    eval_dataset = _load_site_split(validation_json, image_root, args.task) if os.path.isfile(validation_json) else None
    n_train_raw = len(train_dataset)
    train_label_counts = _label_counts(train_dataset)

    print(
        f"site={client_name}, task={args.task}, train_samples={n_train_raw}, "
        f"train_label_counts={train_label_counts}, "
        f"validation_samples={len(eval_dataset or [])}, local_lora_rank={args.lora_rank}, "
        f"quantized={args.quantized}, balance_labels={args.balance_labels}"
    )
    model = create_peft_medgemma_model(
        model_name_or_path=args.model_name_or_path,
        quantized=args.quantized,
        device_map={"": 0},
        lora_rank=args.lora_rank,
        modules_to_save=modules_to_save,
    )
    print(f"site={client_name}")
    model.print_trainable_parameters()

    if args.work_dir is None:
        work_dir = os.path.join(os.getcwd(), "medgemma_checkpoints")
    else:
        work_dir = abs_path(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    while flare.is_running():
        input_model = flare.receive()
        if input_model is None:
            break

        current_round = input_model.current_round
        downlink_bytes = params_size_bytes(input_model.params)
        downlink_lora_bytes = params_size_bytes(lora_factor_params(input_model.params))
        received_mb = params_size_mb(input_model.params)
        print(f"site={client_name}, round={current_round}, received adapter size: {received_mb:.2f} MB")
        site_rank_map = build_uniform_lora_rank_map(input_model.params.keys(), args.lora_rank)
        local_adapter_state = truncate_global_bank_for_site(input_model.params, site_rank_map)
        apply_adapter_state(model, local_adapter_state)
        model.train()

        if eval_dataset is not None and args.eval_subset_size > 0 and len(eval_dataset) > args.eval_subset_size:
            round_eval_dataset = eval_dataset.shuffle(seed=42 + current_round).select(range(args.eval_subset_size))
        else:
            round_eval_dataset = eval_dataset

        round_output_dir = os.path.join(work_dir, f"round-{current_round}")
        if os.path.isdir(round_output_dir):
            shutil.rmtree(round_output_dir)

        sync_cuda()
        round_start_time = time.perf_counter()
        reset_peak_cuda_memory_stats()
        round_start_allocated_mb, round_start_reserved_mb = get_cuda_memory_usage_mb()

        round_train_dataset = train_dataset
        if args.balance_labels:
            round_train_dataset = _oversample_by_label(train_dataset, seed=42 + current_round)
            print(
                f"site={client_name}, round={current_round}, balanced_train_samples={len(round_train_dataset)}, "
                f"balanced_label_counts={_label_counts(round_train_dataset)}"
            )

        trainer = SFTTrainer(
            model=model,
            args=_build_training_args(args, round_output_dir, round_eval_dataset is not None),
            train_dataset=round_train_dataset,
            eval_dataset=round_eval_dataset,
            processing_class=processor,
            data_collator=collate_fn,
        )

        sync_cuda()
        train_start_time = time.perf_counter()
        train_result = trainer.train()
        sync_cuda()
        train_runtime_sec = time.perf_counter() - train_start_time
        train_loss = float(getattr(train_result, "training_loss", float("nan")))
        metrics = {"loss": train_loss}
        eval_runtime_sec = 0.0
        accuracy_metrics = None
        if round_eval_dataset is not None:
            sync_cuda()
            eval_start_time = time.perf_counter()
            eval_metrics = trainer.evaluate()
            accuracy_metrics = _evaluate_classification_accuracy(
                model=model,
                processor=processor,
                dataset=round_eval_dataset,
                task=args.task,
                max_new_tokens=args.max_new_tokens,
            )
            sync_cuda()
            eval_runtime_sec = time.perf_counter() - eval_start_time
            eval_loss = float(eval_metrics["eval_loss"])
            metrics["eval_loss"] = eval_loss
            metrics["neg_eval_loss"] = -eval_loss
            metrics["accuracy"] = float(accuracy_metrics["accuracy"])

        params = {"model." + key: value for key, value in get_adapter_state_dict(model).items()}
        uplink_bytes = params_size_bytes(params)
        uplink_lora_bytes = params_size_bytes(lora_factor_params(params))
        sent_mb = params_size_mb(params)
        bytes_per_round = int(args.n_clients) * (downlink_bytes + uplink_bytes)
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
        flare.send(flare.FLModel(params_type=flare.ParamsType.FULL, params=params, metrics=metrics, meta=meta))
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
        print(
            f"site={client_name}, round={current_round}, sent updated adapter size: {sent_mb:.2f} MB, "
            f"downlink_bytes={downlink_bytes}, uplink_bytes={uplink_bytes}, bytes_per_round={bytes_per_round}"
        )
        if args.comm_log_dir:
            log_record = {
                "experiment": args.experiment_name or args.task,
                "site": client_name,
                "round": int(current_round),
                "task": args.task,
                "lora_rank": args.lora_rank,
                "quantized": bool(args.quantized),
                "downlink_bytes": downlink_bytes,
                "uplink_bytes": uplink_bytes,
                "downlink_lora_bytes": downlink_lora_bytes,
                "uplink_lora_bytes": uplink_lora_bytes,
                "bytes_per_round": bytes_per_round,
                "accuracy": None if accuracy_metrics is None else float(accuracy_metrics["accuracy"]),
                "eval_loss": metrics.get("eval_loss"),
                "loss": metrics.get("loss"),
                "unparsed": None if accuracy_metrics is None else int(accuracy_metrics["unparsed"]),
            }
            append_jsonl(os.path.join(abs_path(args.comm_log_dir), f"round_metrics.{client_name}.jsonl"), log_record)

        del trainer, params, input_model, local_adapter_state
        free_memory()


if __name__ == "__main__":
    main()
