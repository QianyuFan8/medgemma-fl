"""Fail before loading model weights if simulated clients have inconsistent data."""
import json
import hashlib
import re
from pathlib import Path

from methylation_utils import panel_fingerprint


def validate_clients(data_dir, n_clients):
    root = Path(data_dir)
    task = json.loads((root.parent / "task.json").read_text())
    if task["n_clients"] != n_clients:
        raise ValueError("Client count differs from prepared dataset")
    if task.get('panel_mode') == 'shared':
        if task['panel_id'] != panel_fingerprint(task['probes']):
            raise ValueError('Panel fingerprint differs from ordered probes')
        root_spec = json.loads((root.parent / 'features.json').read_text())
        for directory in [root.parent] + [root / f'site-{i}' for i in range(1, n_clients+1)]:
            spec = json.loads((directory / 'features.json').read_text())
            digest = hashlib.sha256((directory / 'features.tsv').read_bytes()).hexdigest()
            if (spec['panel_id'] != task['panel_id'] or spec['probes'] != task['probes'] or
                    spec['classes'] != task['classes'] or digest != task['features_sha256'] or
                    spec['features_sha256'] != digest or spec != root_spec):
                raise ValueError('Distributed feature artifact mismatch')
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
        if task.get('panel_mode') == 'shared':
            if r.get('site_id') not in task['sites'] or (site and r['site_id'] != site):
                raise ValueError('Invalid shared-panel site identity')
            actual = re.findall(r'^(cg\d+): ', r['prompt'], flags=re.MULTILINE)
            if actual != task['probes']:
                raise ValueError('Prompt CpG order differs from shared panel')
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
