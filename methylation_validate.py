"""Fail before loading model weights if simulated clients have inconsistent data."""
import json
from pathlib import Path


def validate_clients(data_dir, n_clients):
    root = Path(data_dir)
    task = json.loads((root.parent / "task.json").read_text())
    if task["n_clients"] != n_clients:
        raise ValueError("Client count differs from prepared dataset")
    heldout = json.loads((root.parent / "test.json").read_text())
    def check_record(r, site=None):
        expected = task['panel_id']
        if task.get('panel_mode') == 'local':
            name = r.get('site_id')
            if name not in task['site_panels'] or (site is not None and name != site):
                raise ValueError('Invalid record site identity')
            expected = task['site_panels'][name]['panel_id']
        if r['panel_id'] != expected or r['classes'] != task['classes']:
            raise ValueError('Wrong site panel or label vocabulary')
        if r['label_name'] not in task['classes']:
            raise ValueError('Unknown diagnosis')
    for r in heldout:
        check_record(r)
    seen = {r["patient_id"] for r in heldout}
    if len(seen) != len(heldout):
        raise ValueError('Duplicate test patients')
    validation = []
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
                check_record(r, f'site-{i}')
                if split == 'validation':
                    validation.append(r)
    global_validation = json.loads((root.parent/'validation.json').read_text())
    if sorted(validation, key=lambda r: r['patient_id']) != sorted(global_validation, key=lambda r: r['patient_id']):
        raise ValueError('Global validation differs from client validation')
    return task
