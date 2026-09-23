"""Prepare three local limma panels and a shared seven-class FL task."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess

from methylation_utils import make_record, panel_fingerprint

CLASSES = ['ALL', 'AML', 'CCSK', 'NBL', 'OS', 'RT', 'WT']
SITE_CLASSES = {'site-1': ['AML', 'CCSK'], 'site-2': ['NBL', 'OS', 'WT'],
                'site-3': ['ALL', 'RT']}
PARTITION = 'diagnosis_disjoint_450k_epic_v2'


def check_partition(saved):
    if saved.get('partition') != PARTITION or saved.get('site_classes') != SITE_CLASSES:
        raise ValueError('Saved partition differs from diagnosis-disjoint v2; use a new output directory, not --resume')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def allocate(rows, seed):
    """Resolve one sample per patient globally, then assign simulated sites."""
    if any(r['status'] == 'cross_cohort_label_conflict' for r in rows):
        raise ValueError('Cross-cohort diagnosis conflicts must be resolved')
    candidates = [dict(r) for r in rows if r['status'] == 'candidate_pending_label_review']
    candidates.sort(key=lambda r: (r['patient_id'], {'01': 0, '09': 1, '03': 2}[r['sample_type']],
                                   r['sample_id'], r['matrix_file']))
    patients, duplicates = {}, []
    for row in candidates:
        pid = row['patient_id']
        if pid in patients:
            if row['platform'] != patients[pid]['platform']:
                raise ValueError('Cross-platform patient requires explicit adjudication')
            duplicates.append(row)
        else:
            patients[pid] = row
    rng = random.Random(seed)
    sites = {f'site-{i}': [] for i in range(1, 4)}
    for dx in CLASSES:
        group = [r for r in patients.values() if r['proposed_diagnosis'] == dx]
        expected = 'epic' if dx in ('ALL', 'RT') else '450k'
        if not group or any(r['platform'] != expected for r in group):
            raise ValueError(f'Missing diagnosis or unexpected platform: {dx}')
        rng.shuffle(group)
        name = next(name for name, diagnoses in SITE_CLASSES.items() if dx in diagnoses)
        sites[name].extend(group)
    for name, group in sites.items():
        counts = Counter(r['proposed_diagnosis'] for r in group)
        if len(counts) < 2 or min(counts.values()) < 5:
            raise ValueError(f'{name} needs >=5 patients per local class for the current holdout: {counts}')
    return sites, duplicates


def export(root, seed):
    if (root/'clients').exists():
        raise FileExistsError('Client export exists; use a new experiment directory')
    records = {s: [] for s in ('train', 'validation', 'test')}
    panels, counts = {}, {}
    for name in ('site-1', 'site-2', 'site-3'):
        local = root/'local'/name
        site_counts = {}
        probes = None
        for split in records:
            with (local/f'{split}.csv').open() as f:
                reader = csv.DictReader(f)
                current = reader.fieldnames[2:]
                rows = list(reader)
            if not rows or (probes is not None and current != probes):
                raise ValueError('Empty split or inconsistent local panel')
            if {r['diagnosis'] for r in rows} != set(SITE_CLASSES[name]):
                raise ValueError(f'{name}/{split} has the wrong diagnosis partition; prepare a fresh v2 dataset')
            probes = current
            converted = [dict(make_record(r['patient_id'], r['diagnosis'], probes,
                              [float(r[p]) for p in probes], CLASSES), site_id=name) for r in rows]
            records[split].extend(converted)
            site_counts[split] = dict(Counter(r['diagnosis'] for r in rows))
        panels[name] = {'probes': probes, 'panel_id': panel_fingerprint(probes)}
        counts[name] = site_counts
    ids = [r['patient_id'] for group in records.values() for r in group]
    if len(ids) != len(set(ids)):
        raise ValueError('Patient overlap across sites/splits')
    for split, group in records.items():
        if {r['label_name'] for r in group} != set(CLASSES):
            raise ValueError(f'{split} does not contain all seven classes')
    for name in panels:
        dest = root/'clients'/name
        dest.mkdir(parents=True)
        for split in ('train', 'validation'):
            dump(dest/f'{split}.json', [r for r in records[split] if r['site_id'] == name])
    for split in ('validation', 'test'):
        dump(root/f'{split}.json', records[split])
    bundle_id = hashlib.sha256(json.dumps(panels, sort_keys=True).encode()).hexdigest()
    dump(root/'task.json', {'classes': CLASSES, 'n_clients': 3, 'seed': seed,
                           'panel_mode': 'local', 'site_panels': panels, 'panel_id': bundle_id,
                           'selection': 'local_training_only_limma',
                           'partition': PARTITION, 'site_classes': SITE_CLASSES, 'counts': counts})
    from methylation_validate import validate_clients
    validate_clients(root/'clients', 3)
    print(json.dumps(counts, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--confirm-cohort-labels', action='store_true')
    p.add_argument('--exclude-unresolved', action='store_true')
    p.add_argument('--seed', type=int, default=20260924)
    p.add_argument('--top-k', type=int, default=100)
    p.add_argument('--rscript', default='Rscript')
    p.add_argument('--resume', action='store_true', help='Resume a partial run with the saved configuration')
    a = p.parse_args()
    root, audit_dir = Path(a.output).resolve(), Path(a.audit_dir)
    if not a.confirm_cohort_labels:
        p.error('--confirm-cohort-labels is required; cohort labels are user assertions')
    if a.resume:
        saved = json.loads((root/'preparation.json').read_text())
        check_partition(saved)
        seed = saved['seed']
    else:
        if root.exists():
            raise FileExistsError(root)
        report = json.loads((audit_dir/'audit.json').read_text())
        for source in report['sources'].values():
            if source['duplicate_list_entries'] or source['listed_but_absent_from_matrix']:
                raise ValueError('Sample-list duplicates/version mismatch require review')
        with (audit_dir/'samples_audit.tsv').open() as f:
            rows = list(csv.DictReader(f, delimiter='\t'))
        unresolved = [r for r in rows if r['status'] == 'unresolved_patient_identifier']
        if unresolved and not a.exclude_unresolved:
            p.error('Unresolved identifiers exist; explicit --exclude-unresolved required')
        sites, duplicates = allocate(rows, a.seed)
        root.mkdir(parents=True)
        dump(root/'exclusions.json', {'audit_exclusions': [r for r in rows if r['status'] != 'candidate_pending_label_review'],
                                     'additional_samples_same_patient': duplicates})
        dump(root/'preparation.json', {'seed': a.seed, 'top_k': a.top_k,
                                      'partition': PARTITION, 'site_classes': SITE_CLASSES,
                                      'audit_dir': str(audit_dir.resolve()),
                                      'label_basis': 'user_confirmed_cohort_membership',
                                      'exclude_unresolved': a.exclude_unresolved})
        for i, (name, group) in enumerate(sites.items()):
            manifest = root/f'{name}_samples.tsv'
            prepared = [dict(r, diagnosis=r['proposed_diagnosis'], include='TRUE', label_status='matched') for r in group]
            with manifest.open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(prepared[0]), delimiter='\t')
                w.writeheader(); w.writerows(prepared)
            dump(root/f'{name}_config.json', {'metadata': str(manifest), 'metadata_reviewed': True,
                 'output': str(root/'local'/name), 'seed': a.seed+i, 'top_k': a.top_k,
                 'fdr': .05, 'variance_cap': 20000, 'covariates': [],
                 'diagnoses': sorted({r['diagnosis'] for r in prepared})})
        seed = a.seed
    env = os.environ.copy()
    env.pop('R_HOME', None)
    for name in ('site-1', 'site-2', 'site-3'):
        marker = root/f'{name}.complete'
        if marker.exists():
            continue
        subprocess.run([a.rscript, str(Path(__file__).parent/'methylation/prepare.R'),
                        str(root/f'{name}_config.json')], env=env, check=True)
        marker.touch()
    export(root, seed)


if __name__ == '__main__':
    main()
