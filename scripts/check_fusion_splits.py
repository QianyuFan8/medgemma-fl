"""Fail before model execution if prepared/exported cohorts are stale or overlap."""
import argparse
from collections import defaultdict
from pathlib import Path
import csv

def read(path, sep="\t"):
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=sep))


def inspect(root):
    root=Path(root)
    rows=read(root/'splits.tsv')
    if not rows:raise ValueError(f'Empty cohort: {root}')
    owners=defaultdict(set)
    signature={}
    for r in rows:
        uid=r.get('canonical_uid','').strip()
        if not uid:raise ValueError(f'Missing canonical_uid; rerun updated link_panel.py: {root}')
        owners[uid].add(r['split'])
        if r['patient_id'] in signature:raise ValueError('Duplicate record ID')
        signature[r['patient_id']]=(uid,r['split'],r['diagnosis'])
    overlaps={u:sorted(s) for u,s in owners.items() if len(s)>1}
    if overlaps:raise ValueError(f'Patient overlap: {overlaps}')
    for split in ['train','validation','test']:
        matrix=read(root/f'{split}.csv',',')
        actual=[(r['patient_id'],r['diagnosis']) for r in matrix]
        expected=[(r['patient_id'],r['diagnosis']) for r in rows if r['split']==split]
        if len(actual)!=len(set(actual)) or set(actual)!=set(expected):
            raise ValueError(f'CSV/manifest mismatch: {root}/{split}.csv')
    return signature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True)
    p.add_argument('--export',action='append',default=[])
    a=p.parse_args()
    expected=inspect(a.prepared)
    for path in a.export:
        if inspect(path)!=expected:raise ValueError(f'Export differs from paired cohort: {path}')
    print(f'PASS: {len(expected)} records, patient-disjoint train/validation/test; exports match.')

if __name__=='__main__':main()
