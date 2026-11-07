"""Fail before model execution if prepared/exported cohorts are stale or overlap."""
import argparse
from collections import defaultdict
from pathlib import Path
import csv
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fusion_identity import validate_patient_rows

def read(path, sep="\t"):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=sep))


def inspect(root):
    root=Path(root)
    rows=read(root/'splits.tsv')
    if not rows:raise ValueError(f'Empty cohort: {root}')
    signature=validate_patient_rows(rows)
    for split in ['train','validation','test']:
        matrix=read(root/f'{split}.csv',',')
        actual=[(r['patient_id'],r['diagnosis']) for r in matrix]
        expected=[(r['patient_id'],r['diagnosis']) for r in rows if r['split']==split]
        if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
            raise ValueError(f'CSV/manifest mismatch: {root}/{split}.csv')
    return signature


def inspect_clients(root, signature):
    """Check actual JSON placement, not just the intended assignment table."""
    root = Path(root)
    task = json.loads((root / 'task.json').read_text())
    sites = task['sites']
    if len(sites) != task['n_clients'] or len(set(sites)) != len(sites):
        raise ValueError('Invalid client inventory')
    assignments, owners = {}, {}
    for row in read(root / 'site_assignments.tsv'):
        rid, site = row['patient_id'], row['site_id']
        if rid not in signature or rid in assignments or site not in sites:
            raise ValueError('Invalid or duplicate client assignment')
        uid, split, label = signature[rid]
        if (row.get('canonical_uid'), row['split'], row['diagnosis']) != (uid, split, label):
            raise ValueError('Client assignment disagrees with canonical manifest; regenerate exports')
        if uid in owners and owners[uid] != site:
            raise ValueError('Canonical patient is assigned to multiple clients; regenerate exports')
        owners[uid] = site
        assignments[rid] = site
    if set(assignments) != set(signature):
        raise ValueError('Client assignments do not cover the cohort')
    for split in ('train', 'validation', 'test'):
        actual = {}
        paths = [(root / 'clients' / site / f'{split}.json', site) for site in sites]
        if split == 'test':
            paths = [(root / 'test.json', None)]
        for path, folder_site in paths:
            if not path.exists() and split == 'validation':
                continue
            for row in json.loads(path.read_text()):
                rid = row['patient_id']
                if rid not in signature or rid in actual:
                    raise ValueError('Duplicate or unknown record in client JSON')
                uid, expected_split, label = signature[rid]
                site = row['site_id']
                if (expected_split != split or row['label_name'] != label or
                        row.get('canonical_uid') != uid or site != assignments[rid] or
                        (folder_site is not None and site != folder_site)):
                    raise ValueError('Client JSON placement or labels disagree with canonical manifest')
                actual[rid] = row
        if set(actual) != {rid for rid, (_, s, _) in signature.items() if s == split}:
            raise ValueError('Client JSON does not cover the requested split')
        if split == 'validation':
            pooled = json.loads((root / 'validation.json').read_text())
            if len(pooled) != len(actual) or {r['patient_id']: r for r in pooled} != actual:
                raise ValueError('Pooled validation JSON differs from client validation JSON')
    return len(owners)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--export',action='append',default=[])
    p.add_argument('--check-client-groups', action='store_true',
                   help='Required for new training; checks canonical UID ownership and actual client JSONs')
    a=p.parse_args()
    expected=inspect(a.prepared)
    for path in a.export:
        if inspect(path)!=expected:raise ValueError(f'Export differs from paired cohort: {path}')
        if a.check_client_groups:
            inspect_clients(path, expected)
    print(f'PASS: {len(expected)} records, patient-disjoint train/validation/test; exports match.')
    if a.check_client_groups:
        print('PASS: each canonical patient belongs to exactly one client; JSON placement verified.')

if __name__=='__main__':main()
