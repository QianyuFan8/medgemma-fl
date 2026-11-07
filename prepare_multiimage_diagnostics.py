"""Derive a sixteen-images-per-record export without changing splits or CpG text."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

from fusion_identity import validate_patient_rows
from methylation_validate import validate_clients
from subtype_utils import load_example_images


def signature(root):
    with (Path(root) / 'splits.tsv').open(newline='') as f:
        return validate_patient_rows(list(csv.DictReader(f, delimiter='\t')))


def convert_record(record, modality='image'):
    if modality not in ('image', 'multimodal') or record.get('modality') != modality or not record.get('image'):
        raise ValueError('Expected image-bearing source records of the requested modality')
    if record.get('image_layout', 'montage') != 'montage':
        raise ValueError('Source must be the original montage export')
    prompt = record['prompt']
    marker = '[H&E]\n'
    if prompt.count(marker) != 1 or prompt.count('[DNA methylation]') != int(modality == 'multimodal'):
        raise ValueError('Unexpected source prompt; refusing implicit modality changes')
    before, section = prompt.split(marker)
    _, separator, after = section.partition('\n\n')
    if not separator or 'Reply with only POS or NEG.' not in after:
        raise ValueError('Unexpected source answer instructions')
    description = ('The attached 16 separate images are the selected tiles from one H&E '
                   'whole-slide image (224 px at 0.5 um/px). Interpret all 16 images '
                   'together to make one patient-level prediction.')
    return {**record, 'image_layout': 'tiles16', 'prompt_style': 'comet_fusion_tiles16_v1',
            'prompt': before + marker + description + '\n\n' + after}


def prepare(source_root, output, modality='image'):
    source, output = Path(source_root).resolve(), Path(output).resolve()
    task = validate_clients(source / 'clients', 1, require_patient_groups=True)
    if task['modality'] != modality or modality not in ('image', 'multimodal') or task['classes'] != ['NEG', 'POS']:
        raise ValueError('Expected a one-client fusion export of the requested modality')
    original = signature(source)
    if output.exists():
        raise FileExistsError(output)
    shutil.copytree(source, output, ignore=shutil.ignore_patterns('images'))
    (output / 'images').symlink_to(source / 'images', target_is_directory=True)
    files = [output / 'validation.json', output / 'test.json']
    files += sorted((output / 'clients').glob('site-*/*.json'))
    image_hashes = {}
    counts = {}
    for path in files:
        if path.name not in ('train.json', 'validation.json', 'test.json'):
            continue
        rows = json.loads(path.read_text())
        updated = [convert_record(r, modality) for r in rows]
        # Reading image dimensions/pixels does not evaluate held-out outcomes.
        for old, new in zip(rows, updated):
            for key in set(old) - {'prompt', 'prompt_style', 'image_layout'}:
                if old.get(key) != new.get(key):
                    raise ValueError('Record identity changed during export')
            rel = old['image']
            if rel not in image_hashes:
                tiles = load_example_images(new, source)
                if len(tiles) != 16:
                    raise ValueError('Expected exactly sixteen images')
                image_hashes[rel] = hashlib.sha256((source / rel).read_bytes()).hexdigest()
        path.write_text(json.dumps(updated, indent=2) + '\n')
        counts[str(path.relative_to(output))] = len(rows)
    task.update(image_layout='tiles16', prompt_style='comet_fusion_tiles16_v1')
    (output / 'task.json').write_text(json.dumps(task, indent=2) + '\n')
    if signature(output) != original:
        raise ValueError('Patient identity/split/label signature changed')
    validate_clients(output / 'clients', 1, require_patient_groups=True)
    report = {'source_image_export': str(source), 'modality': modality, 'records_per_file': counts,
              'source_split_sha256': hashlib.sha256((source / 'splits.tsv').read_bytes()).hexdigest(),
              'images_sha256': image_hashes, 'image_layout': 'tiles16',
              'one_record_one_target': True, 'split_or_label_changes': False,
              'source_pixels_reused_losslessly': True}
    (output / 'multiimage_provenance.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def processor_preflight(data, output):
    """CPU-only: exercise the real training collator and inference image loader."""
    import torch
    from transformers import AutoProcessor
    from evaluate_subtype import evaluation_reference
    from subtype_utils import format_subtype_example, build_multimodal_collator
    torch.set_num_threads(4)
    root = Path(data)
    task = json.loads((root / 'task.json').read_text())
    processor = AutoProcessor.from_pretrained('google/medgemma-1.5-4b-it', use_fast=False)
    processor.tokenizer.padding_side = 'right'
    soft_id = processor.tokenizer.convert_tokens_to_ids('<image_soft_token>')
    report = {'image_layout': 'tiles16', 'model_inference_performed': False, 'checks': []}
    for split in ('train', 'validation'):
        row = max(evaluation_reference(root, split, task),
                  key=lambda r: len(processor.tokenizer(r['prompt'], add_special_tokens=False)['input_ids']))
        training_row = format_subtype_example(row)
        training_row['image'] = str(root / row['image'])
        batch = build_multimodal_collator(processor, 32768)([training_row])
        prompt = processor.apply_chat_template(training_row['messages'][:1],
                                                add_generation_prompt=True, tokenize=False)
        inference = processor(text=[prompt], images=[load_example_images(row, root)], return_tensors='pt')
        n = inference['input_ids'].shape[1]
        if (batch['pixel_values'].shape[0] != 16 or
                int((batch['input_ids'] == soft_id).sum()) != 4096 or
                not torch.equal(batch['pixel_values'], inference['pixel_values']) or
                not torch.equal(batch['input_ids'][0, :n], inference['input_ids'][0]) or
                not torch.all(batch['labels'][0, :n] == -100)):
            raise ValueError('Multi-image training/inference tensor parity failed')
        target = processor.tokenizer.decode(batch['labels'][0][batch['labels'][0] != -100])
        if target != row['label_name'] + '<end_of_turn>\n':
            raise ValueError('Unexpected assistant-only training target')
        if not torch.isfinite(batch['pixel_values']).all():
            raise ValueError('Non-finite image tensor')
        report['checks'].append({'split': split, 'pixel_values_shape': list(batch['pixel_values'].shape),
                                 'sequence_length': batch['input_ids'].shape[1], 'image_soft_tokens': 4096,
                                 'training_inference_pixels_and_prefix_match': True,
                                 'assistant_only_target_verified': True})
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report
