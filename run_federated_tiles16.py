"""Four federated 16-image + CpG runs for one LoRA scope on configurable GPUs.

Each run starts from the base model. All three simulated clients participate in
each round. Shared GPU groups serialize; disjoint groups run concurrently.
Optimizer and linear LR scheduler restart each local
round (existing client behavior); this is not centralized continuous training.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from prepare_federated_tiles16 import verify
from prepare_multiimage_diagnostics import processor_preflight
from run_central_diagnostics import run_stage
from run_multimodal_tiles60 import verify_scope
from run_spatial20_inference import digest, write_csv, write_json
from training_performance import (add_performance_args, performance_cli, client_gpu_plan,
                                  parse_client_gpu_groups, accumulation_for_world)

CONFIGS = [('iid', 'fedavg'), ('iid', 'fedlc'), ('skew08', 'fedavg'), ('skew08', 'fedlc')]
SEED = 20261011


def saved_rounds(rounds):
    return sorted({r for r in (2, 5, 10, 20, 30, 50, 60, rounds) if 1 <= r <= rounds})


def training_command(data, dest, scope, method, rounds, local_epochs, smoke=False, performance=None):
    if scope not in ('lora-only', 'language-only') or method not in ('fedavg', 'fedlc') or min(rounds, local_epochs) < 1:
        raise ValueError('Invalid federated training configuration')
    cmd = [sys.executable, '-u', 'job.py', '--task', 'subtype', '--adapter_scope', scope,
           '--model_name_or_path', 'google/medgemma-1.5-4b-it', '--attn_implementation', 'sdpa',
           '--max_seq_length', '32768', '--data_dir', str(data / 'clients'),
           '--n_clients', '3', '--gpu', '[0],[0],[0]', '--sim_threads', '1',
           '--client_gpu_lock', str((dest / 'client_gpu.lock').resolve()),
           '--site_lora_ranks', '16,16,16', '--global_lora_rank', '16', '--lora_aggregation', 'naive',
           '--fl_method', method, '--fedlc_tau', '1.0', '--learning_rate', '5e-5', '--seed', str(SEED),
           '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1',
           '--gradient_accumulation_steps', '4', '--logging_steps', '1', '--eval_steps', '1000000',
           '--num_rounds', '1' if smoke else str(rounds), '--num_train_epochs', str(local_epochs),
           '--workspace', str(dest / 'train'), '--global_snapshot_dir', str(dest / 'global_checkpoints'),
           '--save_global_rounds'] + [str(r) for r in ([1] if smoke else saved_rounds(rounds))]
    if smoke:
        cmd += ['--max_steps', '1']
    if performance is not None:
        groups, threads = client_gpu_plan(performance.gpus)
        for flag, value in [('--gpu', groups), ('--sim_threads', str(threads)),
                            ('--per_device_train_batch_size', str(performance.per_device_train_batch_size))]:
            cmd[cmd.index(flag)+1] = value
        cmd += ['--effective_batch_size', str(performance.effective_batch_size)] + performance_cli(performance)
    return cmd


def verify_training(dest, scope, data, rounds, local_epochs, smoke=False):
    workspace = dest / 'train'
    reports = []
    for site in ('site-1', 'site-2', 'site-3'):
        site_dir = workspace / 'medgemma' / site
        verify_scope(site_dir, scope)
        expected_n = len(json.loads((data / 'clients' / site / 'train.json').read_text()))
        files = list(site_dir.rglob('training_diagnostics.json'))
        by_round = {int(f.parent.name.removeprefix('round-')): f for f in files}
        if len(files) != rounds or set(by_round) != set(range(rounds)):
            log = dest / 'training.log'
            reason = ''
            if log.is_file():
                oom = next((line for line in log.read_text(errors='replace').splitlines()
                            if 'OutOfMemoryError:' in line), None)
                if oom:
                    reason = f'; CUDA out of memory: {oom.strip()}'
            raise RuntimeError(f'{site} did not complete all {rounds} rounds; inspect {log}{reason}')
        for r, path in sorted(by_round.items()):
            report = json.loads(path.read_text())
            if (report['global_step'] < 1 or not math.isfinite(report['training_loss']) or
                    report['seed'] != SEED or report['learning_rate'] != 5e-5 or
                    report['train_records'] != expected_n or
                    (not smoke and (report['epoch'] < local_epochs - 1e-6 or report['max_steps'] is not None))):
                raise RuntimeError(f'Incomplete or mismatched training at {site} round {r}')
            reports.append(dict(site=site, round=r + 1, **report))
    for r in ([1] if smoke else saved_rounds(rounds)):
        path = dest / 'global_checkpoints' / f'round-{r:03d}.pt'
        info = json.loads(path.with_suffix('.json').read_text())
        if (info['completed_rounds'] != r or info['stage'] != 'post_aggregation' or
                info['n_clients'] != 3 or info['checkpoint_sha256'] != digest(path)):
            raise RuntimeError('Invalid aggregated checkpoint')
        expected_weights = sorted(len(json.loads((data / 'clients' / s / 'train.json').read_text()))
                                  for s in ('site-1', 'site-2', 'site-3'))
        if sorted(info['training_record_weights']) != expected_weights:
            raise RuntimeError('Aggregation weights differ from client record counts')
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    add_performance_args(p)
    p.add_argument('--gpus', default='0', help='One or more GPU IDs, e.g. 0 or 0,1,2,3 or 0,1,2,3,4,5,6,7')
    p.add_argument('--effective_batch_size', type=int, default=4, help='Effective batch per FL client, across its GPUs')
    p.add_argument('--per_device_train_batch_size', type=int, default=1)
    p.add_argument('--data-root', required=True)
    p.add_argument('--scope', required=True, choices=('lora-only', 'language-only'))
    p.add_argument('--rounds', required=True, type=int)
    p.add_argument('--local-epochs', required=True, type=int)
    p.add_argument('--output', required=True)
    p.add_argument('--validate-only', action='store_true', help='Check CLI/hardware/data without creating a training run')
    a = p.parse_args()
    groups, threads = client_gpu_plan(a.gpus)
    for group in parse_client_gpu_groups(groups, 3):
        accumulation_for_world(a.effective_batch_size, a.per_device_train_batch_size, len(group))
    if min(a.rounds, a.local_epochs) < 1:
        p.error('rounds and local epochs must be positive')
    root, output = Path(a.data_root).resolve(), Path(a.output).resolve()
    inputs = {layout: verify(root / layout) for layout in ('iid', 'skew08')}
    if inputs['iid']['source_split_sha256'] != inputs['skew08']['source_split_sha256']:
        raise ValueError('IID and skew inputs have different cohorts')
    import torch
    used_gpus = {g for group in parse_client_gpu_groups(groups, 3) for g in group}
    if any(g >= torch.cuda.device_count() for g in used_gpus):
        raise ValueError(f'Requested GPUs {sorted(used_gpus)} exceed {torch.cuda.device_count()} visible GPUs')
    if a.validate_only:
        print(f'VALIDATED gpu_groups={groups} effective_batch_per_client={a.effective_batch_size}')
        return
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(SEED))
    write_json(output / 'run_manifest.json', dict(arguments=vars(a), configurations=CONFIGS,
               saved_rounds=saved_rounds(a.rounds), local_epochs_per_patient=a.rounds * a.local_epochs,
               initialization='fresh base model and seeded adapters for every experiment; no centralized checkpoint',
               training_seed=SEED, learning_rate=5e-5, local_batch_size=a.per_device_train_batch_size,
               effective_batch_per_client=a.effective_batch_size, client_gpu_groups=groups,
               gradient_accumulation_steps_by_site=[accumulation_for_world(a.effective_batch_size,
                   a.per_device_train_batch_size, len(g)) for g in parse_client_gpu_groups(groups, 3)],
               lora_rank=16, aggregation='record-count-weighted LoRA A/B averaging', fedlc_tau=1.0,
               gpu_scheduling='per-GPU-group lock; DDP within multi-GPU clients; concurrent disjoint groups',
               scheduler='AdamW and linear learning-rate schedule restart each client round',
               evaluation_split='validation', threshold=.5, test_evaluation_enabled=False,
               input_provenance=inputs, commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               versions={n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'trl', 'peft', 'nvflare')}))
    status, rows = dict(state='running', stages=[]), []
    status_path = output / 'status.json'
    try:
        for layout in ('iid', 'skew08'):
            processor_preflight(root / layout, output / f'{layout}_processor_check.json')
        # Exercise both FedLC and FedAvg before committing to the full queue.
        for layout, method in CONFIGS:
            name = f'{layout}_{method}'
            dest = output / 'smoke' / name; dest.mkdir(parents=True)
            run_stage(training_command(root / layout, dest, a.scope, method, 1, a.local_epochs, True, a),
                      dest / 'training.log', name + ':smoke', status_path, status)
            write_json(dest / 'verified_training.json', verify_training(dest, a.scope, root / layout, 1, a.local_epochs, True))
            print(f'VERIFIED {name}:smoke — all three clients and global checkpoint checked.', flush=True)
        for layout, method in CONFIGS:
            name = f'{layout}_{method}'
            dest = output / name; dest.mkdir()
            run_stage(training_command(root / layout, dest, a.scope, method, a.rounds, a.local_epochs, performance=a),
                      dest / 'training.log', name + ':train', status_path, status)
            write_json(dest / 'verified_training.json', verify_training(dest, a.scope, root / layout, a.rounds, a.local_epochs))
            print(f'VERIFIED {name}:train — all client rounds and global checkpoints checked.', flush=True)
            for r in saved_rounds(a.rounds):
                checkpoint = dest / 'global_checkpoints' / f'round-{r:03d}.pt'
                evaluation = dest / f'validation_round_{r:03d}'
                # Candidate evaluation is single-GPU and runs between training jobs.
                # Restrict it to a requested device rather than implicitly GPU 0.
                visible = os.environ.get('CUDA_VISIBLE_DEVICES')
                selected = visible.split(',')[min(used_gpus)] if visible else str(min(used_gpus))
                cmd = ['env', f'CUDA_VISIBLE_DEVICES={selected}', sys.executable, '-u', 'evaluate_federated_tiles16.py', '--data', str(root / layout),
                       '--checkpoint', str(checkpoint), '--scope', a.scope, '--output', str(evaluation)]
                run_stage(cmd, dest / f'validation_round_{r:03d}.log', name + f':validation_round_{r}', status_path, status)
                m = json.loads((evaluation / 'evaluation.json').read_text())['metrics']
                rows.append(dict(configuration=name, adapter_scope=a.scope, round=r, local_epochs=r*a.local_epochs,
                                 n_patients=m['n_patients'], threshold=.5,
                                 **{k: m[k] for k in ('accuracy', 'balanced_accuracy', 'roc_auc', 'pr_auc')},
                                 recall_NEG=m['recall']['NEG'], recall_POS=m['recall']['POS']))
                write_json(output / 'summary.json', rows)
                write_csv(output / 'summary.csv', rows)
        status.update(state='completed', current_stage=None)
        print('ALL_COMPLETED: four federated models and validation evaluations.', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        write_json(status_path, status)


if __name__ == '__main__':
    main()
