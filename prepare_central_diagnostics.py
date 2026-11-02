"""Prepare matched centralized diagnostics without changing the frozen CpG panel."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

from fusion_identity import validate_patient_rows
from prepare_comet_subtype import export, read_tsv, write_tsv
from scripts.check_fusion_splits import inspect, inspect_clients


def select_tiny_records(rows, seed=20261101, per_class=4):
    signature = validate_patient_rows(rows)
    by_label = {'NEG': {}, 'POS': {}}
    for rid, (uid, split, label) in signature.items():
        if split == 'train':
            by_label[label].setdefault(uid, []).append(rid)
    rng = random.Random(seed)
    selected = []
    for label in ('NEG', 'POS'):
        patients = sorted(by_label[label])
        if len(patients) < per_class:
            raise ValueError('Not enough training patients for the balanced fit diagnostic')
        rng.shuffle(patients)
        selected.extend(sorted(by_label[label][uid])[0] for uid in patients[:per_class])
    return sorted(selected)


def make_plan():
    return [
        dict(name='01_tiny_fit', dataset='tiny', epochs=20, learning_rate=2e-4,
             evaluation_splits=['train'], purpose='training_fit_diagnostic_only'),
        dict(name='02_continuous_lr2e4', dataset='multimodal', epochs=10, learning_rate=2e-4,
             evaluation_splits=['train', 'validation'], purpose='continuous_centralized_control'),
        dict(name='03_continuous_lr5e5', dataset='multimodal', epochs=10, learning_rate=5e-5,
             evaluation_splits=['train', 'validation'], purpose='lower_learning_rate'),
        dict(name='04_image_lr5e5', dataset='image', epochs=10, learning_rate=5e-5,
             evaluation_splits=['train', 'validation'], purpose='modality_ablation'),
        dict(name='04_methylation_lr5e5', dataset='methylation', epochs=10, learning_rate=5e-5,
             evaluation_splits=['train', 'validation'], purpose='modality_ablation'),
    ]


def prepare(prepared, montage_dir, output, arm, seed=20261101):
    prepared, montage_dir, output = map(Path, (prepared, montage_dir, output))
    if output.exists():
        raise FileExistsError(output)
    signature = inspect(prepared)
    rows = read_tsv(prepared / 'splits.tsv')
    selected = select_tiny_records(rows, seed)
    if arm not in ('top16', 'spatial16'):
        raise ValueError('Unknown montage arm')
    audit_path = montage_dir / 'selection_audit.json'
    selection = json.loads(audit_path.read_text())['arguments']['selection'] if audit_path.exists() else 'top'
    if selection != {'top16': 'top', 'spatial16': 'spatial'}[arm]:
        raise ValueError('Montage selection metadata disagrees with the requested arm')
    output.mkdir(parents=True)
    tiny = output / 'tiny_prepared'
    shutil.copytree(prepared, tiny)
    tiny_rows = [r for r in rows if r['split'] != 'train' or r['patient_id'] in selected]
    write_tsv(tiny / 'splits.tsv', tiny_rows)
    with (prepared / 'train.csv').open(newline='') as f:
        reader = csv.DictReader(f)
        fields, matrix = reader.fieldnames, list(reader)
    with (tiny / 'train.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in matrix if r['patient_id'] in selected)
    (tiny / 'diagnostic_subset.json').write_text(json.dumps({
        'purpose': 'training_fit_diagnostic_not_generalization', 'selection_seed': seed,
        'source_prepared': str(prepared), 'selected_training_records': selected,
        'one_record_per_patient': True, 'panel_refit': False,
        'note': 'Reuse the existing fixed panel; never interpret training-fit performance as held-out accuracy.'}, indent=2))
    for name, source, modality in [('tiny', tiny, 'multimodal'), ('multimodal', prepared, 'multimodal'),
                                   ('image', prepared, 'image'), ('methylation', prepared, 'methylation')]:
        dest = output / name
        export(argparse.Namespace(prepared=source, output=dest, modality=modality,
                                 n_clients=1, partition='iid', dominant_fraction=0.8,
                                 montage_dir=montage_dir, seed=20261011))
        actual = inspect(dest)
        if actual != inspect(source):
            raise ValueError('Diagnostic export unexpectedly changed the source cohort')
        inspect_clients(dest, actual)
        for split in ('train', 'validation', 'test'):
            if (dest / f'{split}.csv').read_bytes() != (source / f'{split}.csv').read_bytes():
                raise ValueError('Diagnostic export changed beta values or record order')
    manifest = {'arm': arm, 'selection_seed': seed, 'training_seed': 42, 'rounds': 1,
                'adapter_scope': 'lora-only', 'lora_rank': 16, 'max_seq_length': 32768,
                'tiny_records': len(selected), 'full_records': len(signature),
                'full_patients': len({v[0] for v in signature.values()}),
                'tiny_selection_sha256': hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                'source_split_sha256': {s: hashlib.sha256((prepared / f'{s}.csv').read_bytes()).hexdigest()
                                        for s in ('train', 'validation', 'test')},
                'experiments': make_plan(), 'test_evaluation_enabled': False}
    (output / 'diagnostics_plan.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--montage-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', choices=('top16', 'spatial16'), required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.prepared, args.montage_dir, args.output, args.arm), indent=2))


if __name__ == '__main__':
    main()
