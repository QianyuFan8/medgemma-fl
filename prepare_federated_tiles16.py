"""Repartition the frozen spatial 16-image + CpG export by canonical patient."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import shutil

from methylation_validate import validate_clients
from prepare_comet_subtype import assign_sites, read_tsv, write_tsv
from prepare_multiimage_diagnostics import signature
from run_spatial20_inference import digest, write_json


def prepare(source, output, layout, seed=20261011):
    source, output = Path(source).resolve(), Path(output).resolve()
    if layout not in ('iid', 'skew08'):
        raise ValueError('Expected iid or skew08')
    task = validate_clients(source / 'clients', 1, require_patient_groups=True)
    if task['modality'] != 'multimodal' or task.get('image_layout') != 'tiles16' or len(task['probes']) != 849:
        raise ValueError('Expected the frozen multimodal 16-image/849-CpG export')
    original = signature(source)
    manifest = {r['patient_id']: r for r in read_tsv(source / 'splits.tsv')}
    records = {'train': json.loads((source / 'clients/site-1/train.json').read_text()),
               **{s: json.loads((source / f'{s}.json').read_text()) for s in ('validation', 'test')}}
    if any(r.get('image_selection') != 'spatial' for rows in records.values() for r in rows):
        raise ValueError('Every record must carry spatial selection provenance')
    sites = ['site-1', 'site-2', 'site-3']
    rng = random.Random(seed)
    partition = 'iid' if layout == 'iid' else 'skew'
    assignments, converted = [], {}
    for split, rows in records.items():
        owners = assign_sites([manifest[r['patient_id']] for r in rows], sites, partition, rng, .8)
        converted[split] = [{**r, 'site_id': owners[r['patient_id']]} for r in rows]
        for row in converted[split]:
            assignments.append(dict(patient_id=row['patient_id'], canonical_uid=row['canonical_uid'],
                                    diagnosis=row['label_name'], split=split, site_id=row['site_id'], image=row['image']))
        for site in sites:
            if split == 'train' and not any(r['site_id'] == site for r in converted[split]):
                raise ValueError(f'{site} has no training records')
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('images', 'clients'))
    (output / 'images').symlink_to(source / 'images', target_is_directory=True)
    for site in sites:
        dest = output / 'clients' / site
        dest.mkdir(parents=True)
        for name in ('features.json', 'features.tsv'):
            shutil.copy2(source / name, dest / name)
        for split in ('train', 'validation'):
            selected = [r for r in converted[split] if r['site_id'] == site]
            if selected:
                write_json(dest / f'{split}.json', selected)
    for split in ('validation', 'test'):
        write_json(output / f'{split}.json', converted[split])
    write_tsv(output / 'site_assignments.tsv', assignments)
    task.update(n_clients=3, sites=sites, partition=partition,
                dominant_fraction=.8 if layout == 'skew08' else None,
                seed=seed, training_regime='simulated_federated',
                client_assignment_unit='canonical_uid', client_assignment_version=2)
    write_json(output / 'task.json', task)
    patient_counts, record_counts = {}, {}
    for site in sites:
        patient_counts[site], record_counts[site] = {}, {}
        for split in records:
            selected = [r for r in converted[split] if r['site_id'] == site]
            record_counts[site][split] = dict(Counter(r['label_name'] for r in selected))
            patient_counts[site][split] = {label: len({r['canonical_uid'] for r in selected if r['label_name'] == label})
                                          for label in ('NEG', 'POS')}
    write_json(output / 'site_counts.json', record_counts)
    write_json(output / 'site_patient_counts.json', patient_counts)
    if signature(output) != original:
        raise ValueError('Cohort split signature changed')
    validate_clients(output / 'clients', 3, require_patient_groups=True)
    images = {r['image'] for rows in records.values() for r in rows}
    # Content-only fingerprints permit exact comparison across different VM paths.
    report = dict(layout=layout, source=str(source), assignment_seed=seed,
                  source_split_sha256=digest(source / 'splits.tsv'),
                  source_json_sha256={s: digest(source / ('clients/site-1/train.json' if s == 'train' else f'{s}.json'))
                                      for s in records},
                  assignments_sha256=digest(output / 'site_assignments.tsv'),
                  image_sha256={name: digest(source / name) for name in sorted(images)},
                  site_patient_counts=patient_counts, site_record_counts=record_counts,
                  unchanged_fields='all record fields except site_id; source CSVs, patient splits, prompts and image pixels unchanged',
                  aggregation_weight='training record count; repeated specimens remain in one client',
                  skew_definition='existing class-conditional allocation weights with dominant_fraction=0.8; realized client proportions are reported')
    critical = [output / name for name in ('task.json', 'validation.json', 'test.json', 'features.json', 'features.tsv',
                                           'train.csv', 'validation.csv', 'test.csv')]
    critical += sorted((output / 'clients').rglob('*.json'))
    critical += sorted((output / 'clients').rglob('features.tsv'))
    report['content_sha256'] = {str(p.relative_to(output)): digest(p) for p in critical}
    write_json(output / 'federated_inputs.json', report)
    return report


def verify(root):
    root = Path(root)
    validate_clients(root / 'clients', 3, require_patient_groups=True)
    report = json.loads((root / 'federated_inputs.json').read_text())
    if digest(root / 'splits.tsv') != report['source_split_sha256'] or digest(root / 'site_assignments.tsv') != report['assignments_sha256']:
        raise ValueError('Prepared split or client assignment changed')
    for name, expected in report['image_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Prepared image changed: {name}')
    for name, expected in report['content_sha256'].items():
        if digest(root / name) != expected:
            raise ValueError(f'Prepared data changed: {name}')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    for layout in ('iid', 'skew08'):
        report = prepare(a.source, out / layout, layout)
        print(json.dumps({layout: report['site_patient_counts']}, indent=2))


if __name__ == '__main__':
    main()
