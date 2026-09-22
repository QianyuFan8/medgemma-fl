"""Fail before loading model weights if simulated clients have inconsistent data."""
import json
from pathlib import Path


def validate_clients(data_dir, n_clients):
    root = Path(data_dir)
    task = json.loads((root.parent / "task.json").read_text())
    if task["n_clients"] != n_clients:
        raise ValueError("Client count differs from prepared dataset")
    heldout = json.loads((root.parent / "test.json").read_text())
    seen = {r["patient_id"] for r in heldout}
    for i in range(1, n_clients+1):
        for split in ("train", "validation"):
            file = root / f"site-{i}" / f"{split}.json"
            if split == "validation" and not file.exists():
                continue
            records = json.loads(file.read_text())
            if not records:
                raise ValueError(f"Empty client split: {file}")
            for r in records:
                if r["patient_id"] in seen:
                    raise ValueError("Patient leakage across clients/splits")
                seen.add(r["patient_id"])
                if r["panel_id"] != task["panel_id"] or r["classes"] != task["classes"]:
                    raise ValueError("Clients must share CpG panel and label vocabulary")
                if r["label_name"] not in task["classes"]:
                    raise ValueError("Unknown diagnosis")
    return task
