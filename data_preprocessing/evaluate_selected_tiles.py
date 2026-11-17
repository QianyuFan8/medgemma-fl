"""Audit montage provenance and evaluate frozen ABMIL on exactly its selected tiles.

Only train/validation are scored. Per-record details stay in the output directory;
stdout contains aggregate results only. No model fitting or threshold selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_tsv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_indices(coords, rows, expected=16):
    """Match actual montage positions to unique cached level-0 coordinates."""
    rows = sorted(rows, key=lambda r: int(r['position']))
    if [int(r['position']) for r in rows] != list(range(expected)):
        raise ValueError('Montage positions must cover exactly 0..15')
    lookup = defaultdict(list)
    for i, xy in enumerate(coords):
        lookup[tuple(int(v) for v in xy)].append(i)
    result = []
    for r in rows:
        matches = lookup[(int(r['x']), int(r['y']))]
        if len(matches) != 1:
            raise ValueError('Selected coordinate is missing or ambiguous in cached features')
        result.append(matches[0])
    if len(set(result)) != expected:
        raise ValueError('Repeated tile in montage')
    return np.asarray(result, dtype=int), rows


def reconstruct_matches(raw_path, attrs, rows, png):
    import openslide
    from PIL import Image

    tile, size = int(attrs['tile']), int(attrs['tile_level0'])
    canvas = Image.new('RGB', (4 * tile, 4 * tile), 'white')
    with openslide.OpenSlide(str(raw_path)) as slide:
        for r in rows:
            j = int(r['position'])
            patch = slide.read_region((int(r['x']), int(r['y'])), 0, (size, size))
            patch = patch.convert('RGB').resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(patch, ((j % 4) * tile, (j // 4) * tile))
    with Image.open(png) as im:
        return np.array_equal(np.asarray(canvas), np.asarray(im.convert('RGB')))


def main():
    import torch
    from PIL import Image
    from data_preprocessing.abmil_montage import build_model, load_bag
    from medgemma_fl.evaluation.evaluate_subtype import evaluation_reference, patient_score_metrics
    from medgemma_fl.fusion_identity import validate_patient_rows

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='Export parent containing image and multimodal')
    p.add_argument('--montage-root', required=True)
    p.add_argument('--embeddings', default='data/wsi/hoptimus1')
    p.add_argument('--abmil', default='runs/fixed849_disjoint/abmil')
    p.add_argument('--prepared', default='data/fixed849_disjoint_paired')
    p.add_argument('--raw-slides')
    p.add_argument('--extra-raw-montage', help='Also reconstruct another source montage on this VM')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    data, montage, abmil = Path(a.data_root), Path(a.montage_root), Path(a.abmil)
    manifest = read_tsv(data / 'image/splits.tsv')
    signature = validate_patient_rows(manifest)
    original_signature = validate_patient_rows(read_tsv(Path(a.prepared) / 'splits.tsv'))
    if signature != original_signature:
        raise ValueError('ABMIL and MedGemma patient/split/label signatures differ')
    original_slides = {r['patient_id']: r['slide_id'] for r in read_tsv(Path(a.prepared) / 'splits.tsv')}
    if any(original_slides[r['patient_id']] != r['slide_id'] for r in manifest):
        raise ValueError('ABMIL and MedGemma slide linkage differs')
    rows = [r for r in manifest if r['split'] in ('train', 'validation')]
    if len({r['slide_id'] for r in rows}) != len(rows):
        raise ValueError('Slide duplicated across records')
    references = {}
    for modality in ('image', 'multimodal'):
        root = data / modality
        if validate_patient_rows(read_tsv(root / 'splits.tsv')) != signature:
            raise ValueError('Export manifests disagree')
        task = json.loads((root / 'task.json').read_text())
        refs = [r for split in ('train', 'validation') for r in evaluation_reference(root, split, task)]
        if len(refs) != len(rows) or {r['patient_id'] for r in refs} != {r['patient_id'] for r in rows}:
            raise ValueError('Export records do not cover exact audit cohort')
        references[modality] = {r['patient_id']: r for r in refs}
    index = defaultdict(list)
    for r in read_tsv(montage / 'montage_index.tsv'):
        index[r['slide_id']].append(r)
    extra_index = defaultdict(list)
    if a.extra_raw_montage:
        if not a.raw_slides:
            raise ValueError('Extra reconstruction requires raw slides')
        for r in read_tsv(Path(a.extra_raw_montage) / 'montage_index.tsv'):
            extra_index[r['slide_id']].append(r)
    ckpt = torch.load(abmil / 'model.pt', map_location='cpu', weights_only=True)
    if ckpt['classes'] != ['NEG', 'POS']:
        raise ValueError('Unexpected checkpoint class order')
    model = build_model(ckpt['dim'], ckpt['hidden'], ckpt['attn'], ckpt['dropout'])
    model.load_state_dict(ckpt['state_dict'], strict=True)
    model.eval()
    old_scores = {r['slide_id']: r for r in read_tsv(abmil / 'slide_scores.tsv') if r['split'] in ('train', 'validation')}
    predictions, audits, fingerprints, extra_fingerprints = [], [], [], []
    for number, row in enumerate(rows, 1):
        sid, rid = row['slide_id'], row['patient_id']
        feats, coords, attrs = load_bag(a.embeddings, sid)
        if len(feats) != len(coords) or not np.isfinite(feats).all():
            raise ValueError('Invalid cached feature bag')
        if str(attrs.get('model')) != 'hf-hub:bioptimus/H-optimus-1':
            raise ValueError('Unexpected cached encoder')
        chosen, ordered = selected_indices(coords, index[sid])
        png = montage / f'{sid}.png'
        source_hash = sha(png)
        fingerprints.append((sid, source_hash))
        with Image.open(png) as im:
            if im.size != (4 * int(attrs['tile']), 4 * int(attrs['tile'])):
                raise ValueError('Montage size differs from tile metadata')
        for modality, refs in references.items():
            ref = refs[rid]
            if ref['canonical_uid'] != row['canonical_uid'] or ref['label_name'] != row['diagnosis']:
                raise ValueError('Export image identity or label mismatch')
            if sha(data / modality / ref['image']) != source_hash:
                raise ValueError('Actual MedGemma input differs from indexed source montage')
        with torch.inference_mode():
            logits, attention = model(torch.from_numpy(feats))
            full_score = logits.softmax(-1)[1].item()
            # Recompute and renormalize attention within the selected bag.
            logits16, attention16 = model(torch.from_numpy(feats[chosen]))
            selected_score = logits16.softmax(-1)[1].item()
        saved_attention = np.load(abmil / 'attention' / f'{sid}.npy')
        if saved_attention.shape != (len(feats),):
            raise ValueError('Saved attention has wrong bag length')
        att_diff = float(np.max(np.abs(attention.numpy() - saved_attention)))
        score_diff = abs(full_score - float(old_scores[sid]['score_POS']))
        if old_scores[sid]['truth'] != row['diagnosis'] or old_scores[sid]['split'] != row['split']:
            raise ValueError('Historical prediction identity differs')
        index_diff = max(abs(float(r['attention']) - float(saved_attention[i])) for i, r in zip(chosen, ordered))
        if max(att_diff, score_diff, index_diff) > 1e-5:
            raise ValueError('Frozen model does not reproduce historical scores or indexed attention')
        raw_match = extra_match = None
        if a.raw_slides:
            raw_match = reconstruct_matches(Path(a.raw_slides) / f'{sid}.svs', attrs, ordered, png)
        if a.extra_raw_montage:
            _, extra_ordered = selected_indices(coords, extra_index[sid])
            extra_png = Path(a.extra_raw_montage) / f'{sid}.png'
            extra_match = reconstruct_matches(Path(a.raw_slides) / f'{sid}.svs', attrs, extra_ordered, extra_png)
            extra_fingerprints.append((sid, sha(extra_png)))
        distances = np.linalg.norm(coords[chosen, None, :] - coords[chosen][None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        audit = dict(patient_id=rid, split=row['split'], cached_tiles=len(feats), selected_tiles=len(chosen),
                     selected_attention_mass=float(attention[chosen].sum()),
                     min_spacing_tile_widths=float(distances.min() / int(attrs['tile_level0'])),
                     historical_score_max_abs_difference=score_diff, attention_max_abs_difference=att_diff,
                     raw_pixel_match=raw_match, extra_raw_pixel_match=extra_match)
        audits.append(audit)
        for mode, score in [('all_cached', full_score), ('selected16', selected_score)]:
            predictions.append(dict(patient_id=rid, canonical_uid=row['canonical_uid'], split=row['split'],
                                    mode=mode, truth=row['diagnosis'], score_POS=score,
                                    prediction='POS' if score >= .5 else 'NEG'))
        if number % 8 == 0:
            print(f'Audited {number}/{len(rows)} train/validation records', flush=True)
    for filename, values in [('predictions.tsv', predictions), ('provenance.tsv', audits)]:
        with (out / filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]), delimiter='\t')
            writer.writeheader()
            writer.writerows(values)
    metrics = {}
    for split in ('train', 'validation'):
        metrics[split] = {}
        for mode in ('all_cached', 'selected16'):
            selected = [r for r in predictions if r['split'] == split and r['mode'] == mode]
            metrics[split][mode] = patient_score_metrics(selected, data / 'image')
    fingerprint = lambda values: hashlib.sha256(json.dumps(sorted(values)).encode()).hexdigest()
    summary = dict(status='completed', model_sha256=sha(abmil / 'model.pt'), records_audited=len(rows),
                   splits_scored=['train', 'validation'], no_training=True, threshold=.5,
                   image_and_multimodal_source_hash_matches=len(rows),
                   source_montage_cohort_digest=fingerprint(fingerprints),
                   extra_source_montage_cohort_digest=fingerprint(extra_fingerprints) if extra_fingerprints else None,
                   raw_pixel_matches=sum(r['raw_pixel_match'] is True for r in audits) if a.raw_slides else None,
                   extra_raw_pixel_matches=sum(r['extra_raw_pixel_match'] is True for r in audits) if a.extra_raw_montage else None,
                   historical_score_max_abs_difference=max(r['historical_score_max_abs_difference'] for r in audits),
                   metrics=metrics)
    for split in ('train', 'validation'):
        selected = [r for r in audits if r['split'] == split]
        summary[split + '_selection'] = {
            k: {'min': min(r[k] for r in selected), 'median': float(np.median([r[k] for r in selected])),
                'max': max(r[k] for r in selected)}
            for k in ('cached_tiles', 'selected_attention_mass', 'min_spacing_tile_widths')}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('AGGREGATE_JSON_BEGIN', flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print('AGGREGATE_JSON_END', flush=True)


if __name__ == '__main__':
    main()
