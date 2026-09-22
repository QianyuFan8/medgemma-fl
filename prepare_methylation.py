"""Run R limma preprocessing and export patient-disjoint simulated FL clients."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess

from methylation_utils import make_record


def export_sites(directory, n_clients=3, seed=20260922):
    directory = Path(directory)
    if n_clients < 1:
        raise ValueError("n_clients must be positive")
    rows, panels = {}, {}
    for split in ("train", "validation", "test"):
        with (directory / f"{split}.csv").open() as f:
            reader = csv.DictReader(f)
            panels[split] = reader.fieldnames[2:]
            rows[split] = list(reader)
        if not rows[split]:
            raise ValueError(f"Empty split: {split}")
    if panels["train"] != panels["validation"] or panels["train"] != panels["test"]:
        raise ValueError("Panels/order differ across splits")
    ids = [r["patient_id"] for split in rows.values() for r in split]
    if len(set(ids)) != len(ids):
        raise ValueError("Patients overlap within/across splits")
    classes = sorted({r["diagnosis"] for r in rows["train"]})
    probes = panels["train"]
    records = {s: [make_record(r["patient_id"], r["diagnosis"], probes,
                              [float(r[p]) for p in probes], classes) for r in rs] for s, rs in rows.items()}
    client_root = directory / "clients"
    if client_root.exists():
        raise FileExistsError(client_root)
    allocation = {}
    rng = random.Random(seed)
    for split in ("train", "validation"):
        buckets = [[] for _ in range(n_clients)]
        for dx in classes:
            group = [r for r in records[split] if r["label_name"] == dx]
            if split == "train" and len(group) < n_clients:
                raise ValueError(f"Not enough training patients of {dx} for {n_clients} clients")
            rng.shuffle(group)
            offset = min(range(n_clients), key=lambda i: len(buckets[i]))
            for j, row in enumerate(group):
                buckets[(offset+j) % n_clients].append(row)
        allocation[split] = buckets
    client_root.mkdir()
    def dump(path, value):
        path.write_text(json.dumps(value, indent=2)+"\n")
    for i in range(n_clients):
        site = client_root / f"site-{i+1}"
        site.mkdir()
        for split in ("train", "validation"):
            if allocation[split][i]:
                dump(site / f"{split}.json", allocation[split][i])
    # Test patients never appear in client data directories.
    for split in ("validation", "test"):
        dump(directory / f"{split}.json", records[split])
    dump(directory / "task.json", {"classes": classes, "probes": probes,
                                   "panel_id": records["train"][0]["panel_id"],
                                   "n_clients": n_clients, "seed": seed,
                                   "selection": "centralized_train_only_limma",
                                   "partition": "simulated_stratified_not_real_institutions"})
    print(f"Exported {n_clients} simulated clients to {client_root}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--rscript", default="Rscript")
    p.add_argument("--r-library", help="Optional existing R package library; must match selected R runtime")
    p.add_argument("--n-clients", type=int, default=3)
    p.add_argument("--export-only", action="store_true", help="Export an already prepared directory without rerunning limma")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if not args.export_only:
        env = os.environ.copy()
        env.pop("R_HOME", None)
        if args.r_library:
            env["R_LIBS_USER"] = str(Path(args.r_library).resolve())
        subprocess.run([args.rscript, str(Path(__file__).parent / "methylation/prepare.R"), args.config],
                       env=env, check=True)
    export_sites(cfg["output"], args.n_clients, cfg.get("seed", 20260922))


if __name__ == "__main__":
    main()
