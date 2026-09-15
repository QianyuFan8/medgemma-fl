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
"""Run homogeneous-rank LoRA/QLoRA FLARE jobs and summarize accuracy vs bytes-per-round."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys

from data_utils import DEFAULT_RNASEQ_DATA_DIR, DEFAULT_RNASEQ_EVAL_FILE
from plot_comm_study import DEFAULT_FULL_FT_BYTES_PER_ROUND, write_comm_study_svg

EXPERIMENTS = [
    {"name": "lora-r4", "rank": 4, "quantized": False},
    {"name": "lora-r8", "rank": 8, "quantized": False},
    {"name": "lora-r16", "rank": 16, "quantized": False},
    {"name": "lora-r32", "rank": 32, "quantized": False},
    {"name": "qlora-r16", "rank": 16, "quantized": True},
]


def _read_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _summarize_comm_logs(comm_log_dir: str) -> dict:
    records = []
    for path in sorted(glob.glob(os.path.join(comm_log_dir, "round_metrics.*.jsonl"))):
        records.extend(_read_jsonl(path))
    if not records:
        return {"bytes_per_round": None, "local_accuracy": None, "n_records": 0}

    last_round = max(int(record["round"]) for record in records)
    last_records = [record for record in records if int(record["round"]) == last_round]
    bytes_values = [int(record["bytes_per_round"]) for record in last_records if record.get("bytes_per_round") is not None]
    accuracies = [float(record["accuracy"]) for record in last_records if record.get("accuracy") is not None]
    mean_bytes = sum(bytes_values) / len(bytes_values) if bytes_values else None
    mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
    return {
        "bytes_per_round": mean_bytes,
        "local_accuracy": mean_accuracy,
        "n_records": len(records),
        "last_round": last_round,
    }


def _find_global_model(workspace_root: str, job_name: str) -> str | None:
    pattern = os.path.join(workspace_root, job_name, "**", "FL_global_model.pt")
    matches = glob.glob(pattern, recursive=True)
    return matches[0] if matches else None


def _parse_eval_accuracy(stdout: str) -> float | None:
    for line in stdout.splitlines():
        if "Fine-tuned model:" in line and "accuracy=" in line:
            token = line.split("accuracy=", 1)[1].split()[0].rstrip(",")
            return float(token)
    return None


def main():
    parser = argparse.ArgumentParser(description="Run the RNA-seq LoRA communication study.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_RNASEQ_DATA_DIR)
    parser.add_argument("--eval_file", type=str, default=DEFAULT_RNASEQ_EVAL_FILE)
    parser.add_argument("--workspace", type=str, default="/tmp/nvflare/simulation")
    parser.add_argument("--comm_log_root", type=str, default="./comm_logs")
    parser.add_argument("--num_rounds", type=int, default=3)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--n_clients", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--skip_eval", action="store_true", help="Skip global eval.json generation accuracy.")
    parser.add_argument(
        "--balance_labels",
        action="store_true",
        help="Oversample minority diagnosis labels within each site before local SFT.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting study_results.json under --comm_log_root.",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default=None,
        help="Optional comma-separated experiment names to run, e.g. lora-r4,qlora-r16.",
    )
    parser.add_argument(
        "--name_suffix",
        type=str,
        default="",
        help="Appended to each experiment/job name so new runs do not overwrite prior simulators.",
    )
    args = parser.parse_args()

    selected = {part.strip() for part in args.experiments.split(",")} if args.experiments else None
    experiments = [item for item in EXPERIMENTS if selected is None or item["name"] in selected]
    if not experiments:
        raise ValueError("No matching experiments to run.")

    python = sys.executable
    results = []
    results_path = os.path.abspath(os.path.join(args.comm_log_root, "study_results.json"))
    if os.path.isfile(results_path) and not args.overwrite:
        raise FileExistsError(
            f"{results_path} already exists. Use a new --comm_log_root (recommended) or pass --overwrite."
        )
    for experiment in experiments:
        name = experiment["name"] + (args.name_suffix or "")
        rank = experiment["rank"]
        ranks = ",".join([str(rank)] * args.n_clients)
        comm_log_dir = os.path.abspath(os.path.join(args.comm_log_root, name))
        os.makedirs(comm_log_dir, exist_ok=True)
        cmd = [
            python,
            "job.py",
            "--task",
            "rnaseq",
            "--data_dir",
            args.data_dir,
            "--n_clients",
            str(args.n_clients),
            "--num_rounds",
            str(args.num_rounds),
            "--global_lora_rank",
            str(rank),
            "--site_lora_ranks",
            ranks,
            "--lora_aggregation",
            "naive",
            "--modules_to_save",
            "",
            "--comm_log_dir",
            comm_log_dir,
            "--experiment_name",
            name,
            "--workspace",
            args.workspace,
        ]
        if args.max_steps is None:
            cmd.extend(["--num_train_epochs", str(args.num_train_epochs)])
        cmd.append("--quantized" if experiment["quantized"] else "--no-quantized")
        if args.balance_labels:
            cmd.append("--balance_labels")
        if args.max_steps is not None:
            cmd.extend(["--max_steps", str(args.max_steps)])
        if args.gpu:
            cmd.extend(["--gpu", args.gpu])
        print("Running:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        summary = _summarize_comm_logs(comm_log_dir)
        global_model = _find_global_model(args.workspace, name)
        global_accuracy = None
        if global_model and not args.skip_eval:
            eval_cmd = [
                python,
                "run_evaluation.py",
                "--task",
                "rnaseq",
                "--eval_file",
                args.eval_file,
                "--tuned_model_path",
                global_model,
                "--finetune_only",
                "--max_samples",
                "0",
            ]
            print("Evaluating:", " ".join(eval_cmd), flush=True)
            completed = subprocess.run(eval_cmd, check=True, capture_output=True, text=True)
            print(completed.stdout)
            global_accuracy = _parse_eval_accuracy(completed.stdout)
        result = {
            "name": name,
            "rank": rank,
            "quantized": experiment["quantized"],
            "bytes_per_round": summary["bytes_per_round"],
            "local_accuracy": summary["local_accuracy"],
            "accuracy": global_accuracy if global_accuracy is not None else summary["local_accuracy"],
            "global_model": global_model,
            "comm_log_dir": comm_log_dir,
        }
        results.append(result)
        with open(os.path.join(comm_log_dir, "job_summary.json"), "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)

    payload = {
        "experiments": results,
        "full_ft_bytes_per_round_theoretical": DEFAULT_FULL_FT_BYTES_PER_ROUND,
    }
    results_path = os.path.abspath(os.path.join(args.comm_log_root, "study_results.json"))
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    svg_path = os.path.abspath(os.path.join(args.comm_log_root, "accuracy_vs_bytes.svg"))
    try:
        write_comm_study_svg(results, svg_path, DEFAULT_FULL_FT_BYTES_PER_ROUND)
        print(f"Wrote plot to {svg_path}")
    except ValueError as exc:
        print(f"Skipping plot until accuracy and bytes are available: {exc}")
    print(f"Wrote study results to {results_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
