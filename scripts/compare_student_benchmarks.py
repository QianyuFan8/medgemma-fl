"""Compare verified same-input/same-settings 1-vs-4-GPU benchmark runs."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def report(path):
    p=Path(path);manifest=json.loads((p/'run_manifest.json').read_text())
    reports=[json.loads((p/f'rank-{i}_timing.json').read_text()) for i in range(manifest['world_size'])]
    perf=[json.loads((p/f'rank-{i}_performance.json').read_text()) for i in range(manifest['world_size'])]
    if not (p/'completed.json').is_file():raise ValueError('Run is incomplete')
    sec=max(r['mean_step_sec'] for r in reports)
    return manifest,dict(gpus=manifest['world_size'],mean_step_sec=sec,
        patients_per_sec=manifest['effective_batch_size']/sec,
        training_seconds=max(r['train_runtime_sec'] for r in perf),
        gpu_hours=max(r['train_runtime_sec'] for r in perf)*manifest['world_size']/3600,
        max_allocated_gib=max(r['peak_allocated_mb'] for r in perf)/1024)


def main():
    p=argparse.ArgumentParser();p.add_argument('single');p.add_argument('four');a=p.parse_args()
    x,one=report(a.single);y,four=report(a.four)
    if (x['world_size'],y['world_size'])!=(1,4):raise ValueError('Expected single and four GPU runs')
    keys=('model_name_or_path','adapter_scope','weighted_labels','learning_rate','seed','max_steps',
          'quantized','gradient_checkpointing','attn_implementation','max_seq_length','preprocess_cache_dir')
    if any(x['arguments'][k]!=y['arguments'][k] for k in keys) or x['effective_batch_size']!=y['effective_batch_size']:
        raise ValueError('Training settings differ; not a controlled scaling comparison')
    if x.get('student_input_sha256')!=y.get('student_input_sha256') or not x.get('student_input_sha256'):
        raise ValueError('Different or missing frozen input digests')
    if x.get('environment')!=y.get('environment') or x.get('base_model_commit')!=y.get('base_model_commit'):
        raise ValueError('Software environment or base model revision differs')
    print(json.dumps(dict(single=one,four=four,step_speedup=one['mean_step_sec']/four['mean_step_sec'],
        throughput_efficiency=(one['mean_step_sec']/four['mean_step_sec'])/4,
        note='Includes DDP sampler padding/repeats; not proof of identical optimization or final accuracy.'),indent=2))


if __name__=='__main__':main()
