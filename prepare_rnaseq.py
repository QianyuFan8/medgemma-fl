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
"""Prepare TARGET RNA-seq count matrices as 3-site MedGemma text shards."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
from data_utils import DIAGNOSIS_CLASSES, DIAGNOSIS_TO_INDEX, TASK_RNASEQ, write_label_distribution_svg

BATCH_RE = re.compile(r"WU-TARGET-Batch(\d+)-(STRANDED|UNSTRANDED)_RSEM_gene_count", re.IGNORECASE)
PATIENT_RE = re.compile(r"^TARGET-\d+-([A-Z0-9]+)", re.IGNORECASE)
DEFAULT_SITE_BATCHES = {
    "site-1": {1, 2, 3},
    "site-2": {4, 5, 6},
    "site-3": {7, 8, 9},
}


def normalize_sample_id(sample_id: str) -> str:
    return sample_id.strip().replace(".", "-")


def patient_id_from_sample(sample_id: str) -> str:
    match = PATIENT_RE.match(sample_id)
    if match:
        return match.group(1).upper()
    return sample_id


def load_diagnosis_map(meta_path: str) -> dict[str, str]:
    diagnoses = {}
    with open(meta_path, newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "Sample" not in reader.fieldnames or "Dx" not in reader.fieldnames:
            raise ValueError(f"Expected Sample and Dx columns in {meta_path}, got {reader.fieldnames}")
        for row in reader:
            sample_id = normalize_sample_id(row["Sample"])
            diagnosis = (row["Dx"] or "").strip().upper()
            if diagnosis and diagnosis != "NA":
                diagnoses[sample_id] = diagnosis
    return diagnoses


def parse_count_filename(filename: str) -> tuple[int, str] | None:
    match = BATCH_RE.search(filename)
    if not match:
        return None
    return int(match.group(1)), match.group(2).upper()


def _read_count_table(path: str) -> tuple[list[str], list[str], list[str], list[str], np.ndarray]:
    with open(path, newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        sample_ids = [normalize_sample_id(value) for value in header[4:]]
        gene_ids = []
        gene_symbols = []
        bio_types = []
        rows = []
        for row in reader:
            if len(row) < 5:
                continue
            gene_ids.append(row[0])
            gene_symbols.append(row[1] or row[0])
            bio_types.append(row[2])
            rows.append([float(value) if value not in ("", "NA") else 0.0 for value in row[4 : 4 + len(sample_ids)]])
    return gene_ids, gene_symbols, bio_types, sample_ids, np.asarray(rows, dtype=np.float64)


def merge_count_matrices(rnaseq_dir: str, diagnoses: dict[str, str]) -> dict:
    files = []
    for name in sorted(os.listdir(rnaseq_dir)):
        parsed = parse_count_filename(name)
        if parsed is None or not name.endswith(".txt"):
            continue
        batch_id, strandedness = parsed
        files.append((strandedness != "STRANDED", batch_id, strandedness, os.path.join(rnaseq_dir, name)))
    files.sort()
    if not files:
        raise FileNotFoundError(f"No WU-TARGET-Batch*_RSEM_gene_count files found under {rnaseq_dir}")

    sample_ids: list[str] = []
    sample_batches: list[int] = []
    seen_samples: set[str] = set()
    gene_ids: list[str] | None = None
    gene_symbols: list[str] | None = None
    bio_types: list[str] | None = None
    count_blocks: list[np.ndarray] = []

    for _priority, batch_id, strandedness, path in files:
        file_gene_ids, file_symbols, file_bio_types, file_sample_ids, counts = _read_count_table(path)
        if gene_ids is None:
            gene_ids = file_gene_ids
            gene_symbols = file_symbols
            bio_types = file_bio_types
        elif file_gene_ids != gene_ids:
            raise ValueError(f"Gene row mismatch in {path}")

        keep_indices = []
        for col_idx, sample_id in enumerate(file_sample_ids):
            diagnosis = diagnoses.get(sample_id)
            if diagnosis not in DIAGNOSIS_TO_INDEX or sample_id in seen_samples:
                continue
            keep_indices.append(col_idx)
            sample_ids.append(sample_id)
            sample_batches.append(batch_id)
            seen_samples.add(sample_id)
        if keep_indices:
            count_blocks.append(counts[:, keep_indices])
            print(
                f"  {os.path.basename(path)}: kept {len(keep_indices)} labeled unique samples "
                f"(batch={batch_id}, {strandedness.lower()})"
            )

    if not count_blocks:
        raise ValueError("No labeled TARGET samples were found in the RNA-seq count matrices.")

    counts = np.concatenate(count_blocks, axis=1)
    return {
        "gene_ids": gene_ids,
        "gene_symbols": gene_symbols,
        "bio_types": bio_types,
        "sample_ids": sample_ids,
        "sample_batches": np.asarray(sample_batches, dtype=np.int32),
        "counts": counts,
        "labels": [diagnoses[sample_id] for sample_id in sample_ids],
    }


def tmm_norm_factors(counts: np.ndarray, log_ratio_trim: float = 0.3, sum_trim: float = 0.05) -> np.ndarray:
    """edgeR TMM normalization factors with a product-1 constraint."""
    library_sizes = np.maximum(counts.sum(axis=0), 1.0)
    upper_quartiles = np.percentile(counts, 75, axis=0)
    ref_idx = int(np.argmin(np.abs(upper_quartiles - np.mean(upper_quartiles))))
    ref_counts = counts[:, ref_idx]
    ref_lib = library_sizes[ref_idx]
    factors = np.ones(counts.shape[1], dtype=np.float64)

    for sample_idx in range(counts.shape[1]):
        if sample_idx == ref_idx:
            continue
        obs = counts[:, sample_idx]
        obs_lib = library_sizes[sample_idx]
        keep = (obs > 0) & (ref_counts > 0)
        if int(keep.sum()) < 50:
            continue
        obs_k = obs[keep]
        ref_k = ref_counts[keep]
        log_r = np.log2((obs_k / obs_lib) / (ref_k / ref_lib))
        abs_e = 0.5 * np.log2((obs_k / obs_lib) * (ref_k / ref_lib))
        var = (obs_lib - obs_k) / (obs_lib * obs_k) + (ref_lib - ref_k) / (ref_lib * ref_k)
        finite = np.isfinite(log_r) & np.isfinite(abs_e) & np.isfinite(var) & (var > 0)
        log_r, abs_e, var = log_r[finite], abs_e[finite], var[finite]
        if log_r.size < 50:
            continue
        n = log_r.size
        lo_m = int(np.floor(n * log_ratio_trim))
        hi_m = int(np.ceil(n * (1.0 - log_ratio_trim)))
        lo_a = int(np.floor(n * sum_trim))
        hi_a = int(np.ceil(n * (1.0 - sum_trim)))
        order_m = np.argsort(log_r)
        order_a = np.argsort(abs_e)
        keep_m = np.zeros(n, dtype=bool)
        keep_a = np.zeros(n, dtype=bool)
        keep_m[order_m[lo_m:hi_m]] = True
        keep_a[order_a[lo_a:hi_a]] = True
        selected = keep_m & keep_a
        if not np.any(selected):
            continue
        weights = 1.0 / var[selected]
        factors[sample_idx] = 2.0 ** (np.sum(weights * log_r[selected]) / np.sum(weights))

    factors = factors / np.exp(np.mean(np.log(np.maximum(factors, 1e-8))))
    return factors


def log2_tmm_cpm(counts: np.ndarray) -> np.ndarray:
    factors = tmm_norm_factors(counts)
    effective_lib = np.maximum(counts.sum(axis=0) * factors, 1.0)
    cpm = counts / effective_lib * 1e6
    return np.log2(cpm + 1.0), factors


def select_highly_variable_genes(log_cpm: np.ndarray, gene_mask: np.ndarray, n_genes: int) -> np.ndarray:
    variances = np.var(log_cpm[gene_mask], axis=1)
    n_keep = min(n_genes, int(gene_mask.sum()))
    local_order = np.argsort(variances)[::-1][:n_keep]
    return np.flatnonzero(gene_mask)[local_order]


def _site_for_batch(batch_id: int) -> str:
    for site_name, batches in DEFAULT_SITE_BATCHES.items():
        if batch_id in batches:
            return site_name
    raise ValueError(f"Batch {batch_id} is not mapped to a federated site.")


def _split_patients_for_eval(records: list[dict], eval_fraction: float, seed: int) -> tuple[set[str], set[str]]:
    rng = random.Random(seed)
    by_label: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["patient_id"] not in by_label[record["label_name"]]:
            by_label[record["label_name"]].append(record["patient_id"])
    eval_patients: set[str] = set()
    for patients in by_label.values():
        rng.shuffle(patients)
        n_eval = max(1, int(round(len(patients) * eval_fraction))) if patients else 0
        eval_patients.update(patients[:n_eval])
    all_patients = {record["patient_id"] for record in records}
    return eval_patients, all_patients - eval_patients


def _split_site_records(records: list[dict], validation_fraction: float, seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_patient[record["patient_id"]].append(record)
    patient_ids = list(by_patient)
    rng.shuffle(patient_ids)
    n_val = max(1, int(round(len(patient_ids) * validation_fraction))) if len(patient_ids) > 1 else 0
    val_patients = set(patient_ids[:n_val])
    train, validation = [], []
    for patient_id, patient_records in by_patient.items():
        target = validation if patient_id in val_patients else train
        target.extend(patient_records)
    rng.shuffle(train)
    rng.shuffle(validation)
    return {"train": train, "validation": validation}


def assign_sites(records: list[dict], split_strategy: str, num_clients: int, seed: int) -> dict[str, list[dict]]:
    if split_strategy == "batch":
        grouped: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            grouped[_site_for_batch(record["batch_id"])].append(record)
        return grouped

    rng = random.Random(seed)
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_patient[record["patient_id"]].append(record)
    patient_ids = list(by_patient)
    rng.shuffle(patient_ids)
    grouped = {f"site-{idx}": [] for idx in range(1, num_clients + 1)}
    for idx, patient_id in enumerate(patient_ids):
        site_name = f"site-{(idx % num_clients) + 1}"
        grouped[site_name].extend(by_patient[patient_id])
    return grouped


def build_records(merged: dict, hvg_indices: np.ndarray, log_cpm: np.ndarray) -> list[dict]:
    symbols = [merged["gene_symbols"][idx] for idx in hvg_indices]
    records = []
    for sample_idx, sample_id in enumerate(merged["sample_ids"]):
        values = log_cpm[hvg_indices, sample_idx]
        expression_text = "\n".join(f"{symbol}\t{value:.4f}" for symbol, value in zip(symbols, values))
        label_name = merged["labels"][sample_idx]
        records.append(
            {
                "task": TASK_RNASEQ,
                "sample_id": sample_id,
                "patient_id": patient_id_from_sample(sample_id),
                "batch_id": int(merged["sample_batches"][sample_idx]),
                "label": DIAGNOSIS_TO_INDEX[label_name],
                "label_name": label_name,
                "expression_text": expression_text,
            }
        )
    return records


def _public_record(record: dict) -> dict:
    return {
        "task": TASK_RNASEQ,
        "sample_id": record["sample_id"],
        "patient_id": record["patient_id"],
        "batch_id": record["batch_id"],
        "label": record["label"],
        "label_name": record["label_name"],
        "expression_text": record["expression_text"],
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare TARGET RNA-seq TMM-CPM shards for federated MedGemma SFT.")
    parser.add_argument("--rnaseq_dir", type=str, default="./RNAseq", help="Directory with TARGET count matrices.")
    parser.add_argument("--meta_path", type=str, default="./RNAseq/TARGET_meta.txt", help="Sample diagnosis table.")
    parser.add_argument("--output_dir", type=str, default="./data/rnaseq", help="Output directory for site shards.")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of federated sites (default: 3).")
    parser.add_argument("--n_hvg", type=int, default=200, help="Number of highly variable genes in each prompt.")
    parser.add_argument("--min_count", type=int, default=10, help="Minimum count for a gene to be considered expressed.")
    parser.add_argument("--min_samples", type=int, default=10, help="Minimum samples in which a gene must be expressed.")
    parser.add_argument("--eval_fraction", type=float, default=0.15, help="Patient fraction held out as global eval.json.")
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=0.15,
        help="Patient fraction reserved as per-site validation (default: 0.15).",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        choices=("batch", "random"),
        default="batch",
        help="batch: Batch01-03 / 04-06 / 07-09 sites; random: patient-level IID shards.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed (default: 42).")
    args = parser.parse_args()

    if args.split_strategy == "batch" and args.num_clients != 3:
        raise ValueError("The default batch site map is defined for --num_clients 3. Use --split_strategy random.")

    rnaseq_dir = os.path.abspath(args.rnaseq_dir)
    diagnoses = load_diagnosis_map(args.meta_path)
    print(f"Loaded {len(diagnoses)} labeled samples from {os.path.abspath(args.meta_path)}")
    print(f"Merging count matrices from {rnaseq_dir}")
    merged = merge_count_matrices(rnaseq_dir, diagnoses)
    counts = merged["counts"]
    bio_mask = np.array([bio == "protein_coding" for bio in merged["bio_types"]])
    expressed_mask = (counts >= args.min_count).sum(axis=1) >= args.min_samples
    gene_mask = bio_mask & expressed_mask
    print(
        f"Merged samples={counts.shape[1]}, genes={counts.shape[0]}, "
        f"protein_coding_expressed={int(gene_mask.sum())}"
    )
    print("Diagnosis counts: " + ", ".join(f"{name}={merged['labels'].count(name)}" for name in DIAGNOSIS_CLASSES))

    log_cpm, factors = log2_tmm_cpm(counts)
    print(f"TMM factors: min={factors.min():.3f}, median={np.median(factors):.3f}, max={factors.max():.3f}")

    patient_probe = [
        {"patient_id": patient_id_from_sample(sample_id), "label_name": label_name}
        for sample_id, label_name in zip(merged["sample_ids"], merged["labels"])
    ]
    eval_patients, _ = _split_patients_for_eval(patient_probe, args.eval_fraction, args.seed)
    train_sample_mask = np.array(
        [patient_id_from_sample(sample_id) not in eval_patients for sample_id in merged["sample_ids"]]
    )
    hvg_indices = select_highly_variable_genes(log_cpm[:, train_sample_mask], gene_mask, args.n_hvg)
    records = build_records(merged, hvg_indices, log_cpm)
    eval_records = [_public_record(record) for record in records if record["patient_id"] in eval_patients]
    remaining = [record for record in records if record["patient_id"] not in eval_patients]
    site_groups = assign_sites(remaining, args.split_strategy, args.num_clients, args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    eval_path = os.path.join(args.output_dir, "eval.json")
    with open(eval_path, "w") as handle:
        json.dump(eval_records, handle, indent=2)
    print(f"Global eval: {len(eval_records)} samples from {len(eval_patients)} patients -> {eval_path}")

    site_splits = {}
    for site_name in sorted(site_groups):
        site_dir = os.path.join(args.output_dir, site_name)
        os.makedirs(site_dir, exist_ok=True)
        splits = _split_site_records(
            site_groups[site_name],
            args.validation_fraction,
            args.seed + int.from_bytes(site_name.encode(), "little") % 1000,
        )
        batches = sorted({int(record["batch_id"]) for record in splits["train"]})
        train_counts = [sum(1 for record in splits["train"] if record["label_name"] == name) for name in DIAGNOSIS_CLASSES]
        total_train = max(sum(train_counts), 1)
        dominant = [
            name
            for name, count in zip(DIAGNOSIS_CLASSES, train_counts)
            if count / total_train >= 0.15
        ]
        site_splits[site_name] = {
            "train": splits["train"],
            "validation": splits["validation"],
            "dominant_labels": dominant or None,
            "batch_note": f"batches {', '.join(str(batch) for batch in batches)}",
        }
        train_path = os.path.join(site_dir, "train.json")
        validation_path = os.path.join(site_dir, "validation.json")
        with open(train_path, "w") as handle:
            json.dump([_public_record(record) for record in splits["train"]], handle, indent=2)
        with open(validation_path, "w") as handle:
            json.dump([_public_record(record) for record in splits["validation"]], handle, indent=2)
        label_summary = ", ".join(
            f"{name}={sum(1 for record in splits['train'] if record['label_name'] == name)}"
            for name in DIAGNOSIS_CLASSES
        )
        print(
            f"  {site_name}: train={len(splits['train'])} -> {train_path}, "
            f"validation={len(splits['validation'])} -> {validation_path}"
        )
        print(f"    train_labels: {label_summary}")

    plot_path = os.path.join(args.output_dir, "split_label_distribution.svg")
    write_label_distribution_svg(
        site_splits,
        plot_path,
        class_names=DIAGNOSIS_CLASSES,
        title="TARGET RNA-seq client train diagnosis distribution",
        subtitle="Batch-grouped 3-site split (site-1: B01–03, site-2: B04–06, site-3: B07–09). Bars are training samples only.",
    )
    print(f"Wrote label distribution plot to {os.path.abspath(plot_path)}")

    hvg_path = os.path.join(args.output_dir, "hvg_genes.json")
    with open(hvg_path, "w") as handle:
        json.dump(
            {
                "n_hvg": int(hvg_indices.size),
                "genes": [
                    {"gene_id": merged["gene_ids"][idx], "gene_symbol": merged["gene_symbols"][idx]}
                    for idx in hvg_indices
                ],
            },
            handle,
            indent=2,
        )
    print(f"Wrote HVG list to {os.path.abspath(hvg_path)}")
    print(f"Done. Client data written under {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
