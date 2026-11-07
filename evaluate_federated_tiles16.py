"""Candidate-answer scoring of a global checkpoint; test requires explicit --split test."""
import argparse
import json
import os
from pathlib import Path

from prepare_federated_tiles16 import verify
from run_spatial20_inference import digest, score_pair, validate_weight_scope, write_csv, write_json
from subtype_utils import load_example_images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--scope', required=True, choices=('lora-only', 'language-only'))
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=('validation', 'test'), default='validation')
    p.add_argument('--gpu-memory-fraction', type=float,
                   help='Optional allocator memory cap for evaluation alongside a running training job.')
    a = p.parse_args()
    if a.gpu_memory_fraction is not None and not 0 < a.gpu_memory_fraction <= 1:
        p.error('GPU memory fraction must be in (0, 1]')
    data, checkpoint, output = Path(a.data), Path(a.checkpoint), Path(a.output)
    verify(data)
    info = json.loads(checkpoint.with_suffix('.json').read_text())
    if info['stage'] != 'post_aggregation' or info['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Invalid global checkpoint metadata')
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false')
    if a.gpu_memory_fraction is not None:
        import torch
        torch.cuda.set_per_process_memory_fraction(a.gpu_memory_fraction, device=0)
    from inference_utils import load_model_and_processor, load_nvflare_global_pt
    from evaluate_subtype import patient_score_metrics
    weights = load_nvflare_global_pt(str(checkpoint))
    validate_weight_scope(weights, a.scope)
    del weights
    model, processor = load_model_and_processor(str(checkpoint), 'google/medgemma-1.5-4b-it', 'cuda')
    predictions = []
    reference = json.loads((data / f'{a.split}.json').read_text())
    for record in reference:
        images = load_example_images(record, data)
        if len(images) != 16:
            raise ValueError('Expected exactly 16 spatial images')
        predictions.append(score_pair(model, processor, dict(record=record, images=images,
                                      image_count=16, prompt=record['prompt']), 32768))
        write_csv(output / 'predictions.csv', predictions)
        print(f'SCORED split={a.split} count={len(predictions)}/{len(reference)}', flush=True)
    from evaluate_subtype import evaluate_rows
    evaluate_rows(predictions, reference, ['NEG', 'POS'], 'POS')
    metrics = patient_score_metrics(predictions, data)
    import torch
    write_json(output / 'evaluation.json', dict(split=a.split, adapter_scope=a.scope,
               completed_rounds=info['completed_rounds'], checkpoint=str(checkpoint.resolve()),
               checkpoint_sha256=info['checkpoint_sha256'], data=str(data.resolve()),
               prediction_source='full_candidate_log_likelihood', metrics=metrics,
               peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               peak_gpu_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
               gpu_memory_fraction=a.gpu_memory_fraction))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
