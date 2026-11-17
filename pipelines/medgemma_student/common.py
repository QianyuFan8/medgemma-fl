import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def indexed(rows, key='patient_id'):
    out = {r[key]: r for r in rows}
    if len(out) != len(rows) or '' in out:
        raise ValueError(f'Missing/duplicate {key}')
    return out


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def write_csv(path, rows, delimiter=','):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        w.writeheader(); w.writerows(rows)


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    for name, expected in plan['source_sha256'].items():
        if digest(name) != expected:
            raise ValueError(f'Source changed after planning: {name}')
    return plan
