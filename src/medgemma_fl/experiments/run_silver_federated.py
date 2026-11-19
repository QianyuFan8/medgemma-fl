"""Run independent IID/FedAvg and skew/FedAvg/FedLC jobs on disjoint GPU groups.

Each experiment has three sites and starts from the same seeded base model.
Gold/silver loss weights remain 1.0/0.3. Site aggregation uses record counts;
silver weights are applied exactly once, inside the patient-mean local loss.
"""
import argparse
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from medgemma_fl.experiments.run_federated_tiles16 import training_command, verify_training, saved_rounds
from medgemma_fl.preprocessing.prepare_federated_tiles16 import verify
from medgemma_fl.training_performance import accumulation_for_world, parse_gpu_ids

CONFIGS = [('iid', 'fedavg'), ('skew08', 'fedavg'), ('skew08', 'fedlc')]


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def command(data, dest, method, gpu_ids, rounds, epochs, seed, smoke, cache):
    ids = parse_gpu_ids(gpu_ids)
    cmd = training_command(data, dest, 'lora-only', method, rounds, epochs, smoke=smoke)
    group = '[' + ','.join(map(str, ids)) + ']'
    for flag, value in [('--gpu', ','.join([group]*3)), ('--sim_threads', '3'),
                        ('--seed', str(seed)), ('--gradient_accumulation_steps',
                         str(accumulation_for_world(4, 1, len(ids))))]:
        cmd[cmd.index(flag)+1] = value
    cmd += ['--weighted_labels', '--effective_batch_size', '4', '--no-local_eval']
    if cache:
        cmd += ['--preprocess_cache_dir', str(cache), '--preprocess_cache_max_gb', '20']
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--rounds', type=int, default=60)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--seed', type=int, default=20261117)
    p.add_argument('--gpu-groups', nargs=3, default=['0,1', '2', '3'])
    p.add_argument('--cache-dir')
    p.add_argument('--smoke', action='store_true', help='Only one round and one optimizer step per site.')
    a = p.parse_args()
    if min(a.rounds, a.local_epochs) < 1:
        p.error('Training budgets must be positive')
    groups = [parse_gpu_ids(g) for g in a.gpu_groups]
    if len(set(sum(groups, []))) != sum(map(len, groups)):
        p.error('Independent experiments must have disjoint GPU groups')
    import torch
    if max(sum(groups, [])) >= torch.cuda.device_count():
        p.error('Requested GPU is not visible')
    for group in groups:
        accumulation_for_world(4, 1, len(group))
    data, out = Path(a.data_root).resolve(), Path(a.output).resolve()
    inputs = {layout: verify(data/layout) for layout in ('iid', 'skew08')}
    if inputs['iid']['source_split_sha256'] != inputs['skew08']['source_split_sha256']:
        raise ValueError('Cohorts differ between partitions')
    for layout in inputs:
        if inputs[layout].get('student_loss') != dict(reduction='patient_mean_of_weighted_answer_token_mean', gold=1., silver=.3):
            raise ValueError('Expected the frozen weighted student inputs')
    out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
               WANDB_MODE='disabled', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2',
               MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', PYTHONHASHSEED=str(a.seed))
    write(out/'manifest.json', dict(arguments=vars(a), configurations=CONFIGS, inputs=inputs,
        runtime_source_sha256={str(f.relative_to(Path(__file__).resolve().parents[1])):
                               hashlib.sha256(f.read_bytes()).hexdigest()
                               for f in Path(__file__).resolve().parents[1].rglob('*.py')},
        initialization='same seeded base model, no centralized checkpoint', scope='vision_and_language_lora',
        seed=a.seed, lora_rank=16, learning_rate=5e-5, effective_batch_per_site=4,
        saved_rounds=[1] if a.smoke else saved_rounds(a.rounds), fedlc_tau=1.,
        fedlc_counts='raw local counts of gold and silver assigned training labels',
        optimizer_schedule='AdamW and linear learning rate restart each local round',
        aggregation='training-record-count-weighted LoRA factors; patient loss applies silver weight once',
        evaluation='No holdout scoring during training; validation checkpoint scoring is a separate step',
        simulated_sites=True, total_local_epochs=a.rounds*a.local_epochs))

    def run(index, layout, method):
        name = f'{layout}_{method}'
        dest = out/name
        dest.mkdir()
        cmd = command(data/layout, dest, method, a.gpu_groups[index], a.rounds,
                      a.local_epochs, a.seed, a.smoke, Path(a.cache_dir).resolve() if a.cache_dir else None)
        write(dest/'command.json', cmd)
        started = time.time()
        write(dest/'status.json', dict(state='running', started_unix=started))
        print(f'START {name} gpus={a.gpu_groups[index]}', flush=True)
        with (dest/'training.log').open('w') as log:
            result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            if result.returncode:
                raise RuntimeError(f'Training process exited {result.returncode}')
            report = verify_training(dest, 'lora-only', data/layout, 1 if a.smoke else a.rounds,
                                     a.local_epochs, smoke=a.smoke, seed=a.seed)
            if any(not r.get('weighted_labels') for r in report):
                raise RuntimeError('A client did not use weighted labels')
            write(dest/'verified_training.json', report)
            write(dest/'status.json', dict(state='completed', elapsed_sec=time.time()-started))
            print(f'DONE {name}', flush=True)
        except Exception as exc:
            write(dest/'status.json', dict(state='failed', error=str(exc)))
            raise
    errors = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = {pool.submit(run, i, *config): config for i, config in enumerate(CONFIGS)}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                errors.append(dict(configuration=jobs[job], error=str(exc)))
                print(f'FAILED {jobs[job]}: {exc}', flush=True)
    write(out/'status.json', dict(state='failed' if errors else 'completed', errors=errors))
    if errors:
        sys.exit(1)
    print('ALL_THREE_EXPERIMENTS_COMPLETED', flush=True)


if __name__ == '__main__':
    main()
