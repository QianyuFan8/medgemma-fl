"""Paired validation-only 16-vs-20-image inference with one fixed V+L adapter.

The first sixteen images are reused byte-for-byte from the historical export.
Four new raw-WSI regions must satisfy the requested spatial distance; no fallback
relaxes it. No fitting, threshold tuning, test scoring or source mutation occurs.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f, delimiter='\t'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def original_indices(coords, index_rows):
    rows = sorted(index_rows, key=lambda r: int(r['position']))
    if [int(r['position']) for r in rows] != list(range(16)):
        raise ValueError('Source montage index must contain exactly positions 0..15')
    lookup = {tuple(map(int, xy)): i for i, xy in enumerate(coords)}
    if len(lookup) != len(coords):
        raise ValueError('Duplicate cached coordinates')
    indices = [lookup[(int(r['x']), int(r['y']))] for r in rows]
    if len(set(indices)) != 16:
        raise ValueError('Repeated source tile')
    return indices, rows


def extend_selection(weights, coords, original, tile_level0, candidate_pool=128,
                     min_distance_tiles=2.0):
    """Append four ranked candidates, preserving the exact original order."""
    weights, coords = np.asarray(weights), np.asarray(coords)
    if (weights.ndim != 1 or coords.shape != (len(weights), 2) or
            not np.isfinite(weights).all() or not np.isfinite(coords).all()):
        raise ValueError('Invalid attention or coordinate arrays')
    if (len(original) != 16 or len(set(original)) != 16 or
            any(i < 0 or i >= len(weights) for i in original) or
            tile_level0 <= 0 or candidate_pool < 20 or min_distance_tiles < 2):
        raise ValueError('Invalid original tiles or selection settings')
    chosen = list(original)
    for index in np.argsort(-weights, kind='stable')[:candidate_pool]:
        index = int(index)
        if index in chosen:
            continue
        distance = np.linalg.norm(coords[chosen] - coords[index], axis=1) / tile_level0
        if float(distance.min()) >= min_distance_tiles:
            chosen.append(index)
        if len(chosen) == 20:
            return chosen
    raise ValueError('Cannot add four tiles at the required distance within the candidate pool. '
                     'No spacing relaxation performed. Review the audit or explicitly increase --candidate-pool.')


def prompt_for_twenty(prompt):
    marker = '[H&E]\n'
    if prompt.count(marker) != 1 or prompt.count('[DNA methylation]') != 1:
        raise ValueError('Expected one multimodal H&E section and one CpG section')
    prefix, section = prompt.split(marker)
    description, sep, suffix = section.partition('\n\n')
    if (not sep or description.count('16') != 2 or
            '16 separate images' not in description or 'all 16 images' not in description):
        raise ValueError('Unexpected source 16-image description')
    new = description.replace('16 separate images', '20 separate images').replace('all 16 images', 'all 20 images')
    return prefix + marker + new + sep + suffix


def prepare_pairs(data, montage_root, embeddings, abmil, slide_dir, output,
                  candidate_pool=128, min_distance_tiles=2.0):
    import h5py
    import openslide
    from PIL import Image
    from methylation_validate import validate_clients
    from fusion_identity import validate_patient_rows
    from subtype_utils import load_example_images
    from data_preprocessing.tile_selection import spatial_metrics

    task = validate_clients(data / 'clients', 1, require_patient_groups=True)
    if task['modality'] != 'multimodal' or task.get('image_layout') != 'tiles16' or len(task['probes']) != 849:
        raise ValueError('Expected the historical 16-image multimodal 849-CpG export')
    manifest = read_tsv(data / 'splits.tsv')
    signature = validate_patient_rows(manifest)
    by_record = {r['patient_id']: r for r in manifest}
    index = defaultdict(list)
    for row in read_tsv(montage_root / 'montage_index.tsv'):
        index[row['slide_id']].append(row)
    rows = json.loads((data / 'validation.json').read_text())
    pairs, audit = [], []
    for record in rows:
        rid = record['patient_id']
        if signature[rid][1:] != ('validation', record['label_name']):
            raise ValueError('Evaluation record is not in the validation split')
        sid = by_record[rid]['slide_id']
        if Path(sid).name != sid:
            raise ValueError('Invalid slide ID')
        with h5py.File(embeddings / f'{sid}.h5', 'r') as f:
            coords, attrs = f['coords'][:], dict(f.attrs)
        size, tile = int(attrs['tile_level0']), int(attrs['tile'])
        if tile != 224 or not math.isclose(float(attrs['target_mpp']), .5):
            raise ValueError('Expected 224px tiles at 0.5 microns/pixel')
        weights = np.load(abmil / 'attention' / f'{sid}.npy', allow_pickle=False)
        original, ordered = original_indices(coords, index[sid])
        if len(weights) != len(coords) or any(abs(float(r['attention']) - float(weights[i])) > 1e-5
                                              for i, r in zip(original, ordered)):
            raise ValueError('Historical attention/index provenance mismatch')
        chosen = extend_selection(weights, coords, original, size, candidate_pool, min_distance_tiles)
        images16 = load_example_images(record, data)
        raw = slide_dir / f'{sid}.svs'
        if not raw.is_file():
            raise FileNotFoundError(f'Missing raw WSI: {raw}; four extra tiles cannot be recovered from a 16-tile PNG')
        with openslide.OpenSlide(str(raw)) as slide:
            images20 = []
            for position, i in enumerate(chosen):
                region = slide.read_region(tuple(map(int, coords[i])), 0, (size, size))
                region = region.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
                if position < 16:
                    if region.tobytes() != images16[position].tobytes():
                        raise ValueError(f'Raw WSI and historical tile pixels disagree: {sid}, position {position}')
                    region = images16[position]  # Keep the actual historical input pixels.
                images20.append(region)
        folder = output / 'tiles' / sid
        folder.mkdir(parents=True, exist_ok=True)
        hashes = []
        for position, im in enumerate(images20):
            dest = folder / f'tile-{position:02d}.png'
            im.save(dest)
            hashes.append(digest(dest))
        for count, imgs, prompt in [(16, images16, record['prompt']),
                                    (20, images20, prompt_for_twenty(record['prompt']))]:
            pairs.append(dict(record=record, image_count=count, images=imgs, prompt=prompt))
        row_audit = dict(record_id=rid, slide_id=sid, original_indices=original, selected_indices=chosen,
                         first_16_pixels_and_order_unchanged=True, candidate_pool=candidate_pool,
                         extra_tiles_spacing_relaxed=False, tile_png_sha256=hashes,
                         source_image_sha256=digest(data / record['image']),
                         source_coordinates_sha256=digest(embeddings / f'{sid}.h5'),
                         source_attention_sha256=digest(abmil / 'attention' / f'{sid}.npy'),
                         source_wsi=str(raw.resolve()),
                         original_spatial_metrics=spatial_metrics(coords, original, size),
                         twenty_spatial_metrics=spatial_metrics(coords, chosen, size),
                         extra_tiles=[dict(position=j, x=int(coords[i, 0]), y=int(coords[i, 1]),
                                           attention=float(weights[i])) for j, i in enumerate(chosen) if j >= 16])
        audit.append(row_audit)
        write_json(output / 'tile_audit.json', audit)
        print(f'PREPARED {rid}: original 16 verified, four new spatial tiles added', flush=True)
    return pairs


def encode_pair(processor, pair):
    content = [{'type': 'image'} for _ in pair['images']]
    content.append({'type': 'text', 'text': pair['prompt']})
    # Ground-truth labels are not part of the model input.
    text = processor.apply_chat_template([{'role': 'user', 'content': content}],
                                         add_generation_prompt=True, tokenize=False)
    return text, processor(text=[text], images=[pair['images']], return_tensors='pt')


def preflight(processor, pairs, output, max_seq_length):
    checks = []
    soft = processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
    for pair in pairs:
        _, batch = encode_pair(processor, pair)
        n, count = batch['input_ids'].shape[1], pair['image_count']
        actual = int((batch['input_ids'] == soft).sum())
        if n + 32 > max_seq_length or batch['pixel_values'].shape[0] != count or actual != 256 * count:
            raise ValueError('Processor image count/context check failed; no truncation is allowed')
        checks.append(dict(record_id=pair['record']['patient_id'], image_count=count,
                           prompt_tokens=n, image_soft_tokens=actual,
                           pixel_values_shape=list(batch['pixel_values'].shape)))
    write_json(output / 'processor_preflight.json', checks)


def score_pair(model, processor, pair, max_seq_length):
    import torch
    from inference_utils import get_model_device
    from evaluate_subtype import option_prediction
    text, prefix = encode_pair(processor, pair)
    n = prefix['input_ids'].shape[1]
    device = get_model_device(model)
    logps = []
    with torch.inference_mode():
        for label in ('NEG', 'POS'):
            full = processor(text=[text + label], images=[pair['images']], return_tensors='pt')
            ids = full['input_ids']
            if (ids.shape[1] <= n or ids.shape[1] > max_seq_length or
                    not torch.equal(ids[0, :n], prefix['input_ids'][0])):
                raise ValueError('Candidate answer does not extend the exact prompt prefix')
            full = full.to(device)
            # Only materialize answer-position vocabulary logits, not the 17k+ prompt logits.
            answer_length = ids.shape[1] - n
            logits = model(**full, logits_to_keep=answer_length + 1, use_cache=False).logits[0, :-1].float()
            if logits.shape[0] != answer_length:
                raise ValueError('Model did not honor answer-position logits_to_keep')
            logps.append(torch.log_softmax(logits, -1).gather(1, full['input_ids'][0, n:].unsqueeze(1)).sum().item())
            del full, logits
    probs = torch.tensor(logps, dtype=torch.float64).softmax(0).tolist()
    return dict(patient_id=pair['record']['patient_id'], truth=pair['record']['label_name'],
                prediction=option_prediction(['NEG', 'POS'], probs), score_NEG=probs[0], score_POS=probs[1],
                logp_NEG=logps[0], logp_POS=logps[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', required=True, help='Existing multimodal_tiles60 run directory, or its full subdirectory')
    p.add_argument('--output', required=True)
    p.add_argument('--epoch', type=int, default=60, choices=(2, 5, 10, 20, 30, 50, 60))
    p.add_argument('--montage-root', default='data/wsi/fixed849_spatial16_montages')
    p.add_argument('--embeddings', default='data/wsi/hoptimus1')
    p.add_argument('--abmil', default='runs/fixed849_disjoint/abmil')
    p.add_argument('--slide-dir', default='data/raw/comet_slides')
    p.add_argument('--candidate-pool', type=int, default=128)
    p.add_argument('--max-seq-length', type=int, default=32768)
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    full = Path(args.source_run).resolve()
    if not (full / 'run_manifest.json').is_file():
        full = full / 'full'
    manifest = json.loads((full / 'run_manifest.json').read_text())
    if manifest['arm'] != 'spatial16' or manifest['base_model'] != 'google/medgemma-1.5-4b-it':
        raise ValueError('Source must be the existing spatial16 multimodal MedGemma 60-epoch run')
    source = full / 'multimodal_tiles16_lora-only_lr5e5'
    data = full / 'data'
    from run_epoch_diagnostics import snapshot
    from run_multimodal_tiles60 import verify_scope
    checkpoint = snapshot(source / 'train', args.epoch, planned_epochs=60)
    verify_scope(source / 'train', 'lora-only')
    expected_path = source / f'epoch-{args.epoch:02d}' / 'eval_validation' / 'predictions.csv'
    with expected_path.open(newline='') as f:
        expected = {r['patient_id']: r for r in csv.DictReader(f)}
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = dict(state='running', current_stage='prepare_tiles', updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    write_json(output / 'status.json', status)
    write_json(output / 'run_manifest.json', dict(arguments=vars(args), source_run=str(full),
               checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
               adapter_scope='lora-only', split='validation', fixed_threshold=.5,
               training_performed=False, test_evaluation_enabled=False,
               scoring='full_candidate_log_likelihood; answer-position logits only',
               source_split_sha256=digest(data / 'splits.tsv'),
               commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()))
    try:
        pairs = prepare_pairs(data, Path(args.montage_root), Path(args.embeddings), Path(args.abmil),
                              Path(args.slide_dir), output, args.candidate_pool)
        if args.prepare_only:
            status.update(state='prepared', current_stage=None)
            return
        os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false')
        from transformers import AutoProcessor
        from inference_utils import load_model_and_processor, load_nvflare_global_pt
        from adapter_scope import infer_adapter_scope
        from evaluate_subtype import patient_score_metrics
        processor = AutoProcessor.from_pretrained(manifest['base_model'], use_fast=False)
        processor.tokenizer.padding_side = 'left'
        preflight(processor, pairs, output, args.max_seq_length)
        weights = load_nvflare_global_pt(str(checkpoint))
        if (infer_adapter_scope(weights) != 'lora-only' or
                not any('.vision_tower.' in k and '.lora_' in k for k in weights) or
                not any('.language_model.' in k and '.lora_' in k for k in weights)):
            raise ValueError('Selected checkpoint does not contain both vision and language adapters')
        del weights
        model, processor = load_model_and_processor(str(checkpoint), manifest['base_model'], 'cuda')
        by_count = {16: [], 20: []}
        errors = []
        # Verify every baseline before scoring any 20-image input.
        for count in (16, 20):
            status['current_stage'] = f'score_{count}_images'
            write_json(output / 'status.json', status)
            for pair in [x for x in pairs if x['image_count'] == count]:
                prediction = score_pair(model, processor, pair, args.max_seq_length)
                if count == 16:
                    old = expected[prediction['patient_id']]
                    if old['truth'] != prediction['truth']:
                        raise ValueError('Historical validation label mismatch')
                    error = max(abs(float(old[f'score_{c}']) - prediction[f'score_{c}']) for c in ('NEG', 'POS'))
                    errors.append(error)
                    if error > 1e-5 or (float(old['score_POS']) >= .5) != (prediction['score_POS'] >= .5):
                        raise ValueError(f'Historical 16-image score parity failed: {error}; investigate before interpreting 20 images')
                by_count[count].append(prediction)
                write_csv(output / f'predictions_{count}.csv', by_count[count])
                print(f'SCORED tiles={count} record={prediction["patient_id"]}', flush=True)
            metrics = patient_score_metrics(by_count[count], data)
            write_json(output / f'metrics_{count}.json', metrics)
        write_json(output / 'baseline_parity.json', {'max_absolute_score_error': max(errors), 'passed': True})
        summary = []
        for count in (16, 20):
            m = patient_score_metrics(by_count[count], data)
            summary.append(dict(images=count, epoch=args.epoch, n_patients=m['n_patients'], threshold=.5,
                                **{k: m[k] for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc')},
                                recall_NEG=m['recall']['NEG'], recall_POS=m['recall']['POS']))
        write_json(output / 'summary.json', summary)
        write_csv(output / 'summary.csv', summary)
        paired = []
        for old, new in zip(by_count[16], by_count[20]):
            if old['patient_id'] != new['patient_id']:
                raise ValueError('Paired prediction order mismatch')
            paired.append(dict(record_id=old['patient_id'], truth=old['truth'], score16=old['score_POS'],
                               score20=new['score_POS'], delta_score=new['score_POS']-old['score_POS'],
                               prediction16=old['prediction'], prediction20=new['prediction']))
        write_csv(output / 'paired_predictions.csv', paired)
        status.update(state='completed', current_stage=None)
        print(json.dumps(summary, indent=2), flush=True)
        print('ALL_COMPLETED: fixed-checkpoint validation comparison; no training or test scoring.', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        write_json(output / 'status.json', status)


if __name__ == '__main__':
    main()
