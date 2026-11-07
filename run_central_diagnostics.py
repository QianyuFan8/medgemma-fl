"""Run five predeclared centralized diagnostics sequentially on one GPU.

Start from repo root using .venv/bin/python. The two montage VMs run the same
plan. Test is never scored automatically. Completed outputs are never replaced.
"""
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from prepare_central_diagnostics import make_plan
from methylation_validate import validate_clients


def train_command(data, workspace, experiment, seed, smoke=False):
    cmd = [sys.executable, '-u', 'job.py', '--task', 'subtype', '--adapter_scope', 'lora-only',
           '--attn_implementation', 'sdpa', '--max_seq_length', '32768',
           '--data_dir', str(data / 'clients'), '--n_clients', '1', '--gpu', '[0]',
           '--site_lora_ranks', '16', '--global_lora_rank', '16', '--sim_threads', '1',
           '--fl_method', 'fedavg', '--lora_aggregation', 'naive', '--num_rounds', '1',
           '--learning_rate', str(experiment['learning_rate']), '--seed', str(seed),
           '--per_device_train_batch_size', '1', '--per_device_eval_batch_size', '1',
           '--gradient_accumulation_steps', '4', '--logging_steps', '5', '--eval_steps', '10',
           '--workspace', str(workspace)]
    return cmd + (['--max_steps', '1'] if smoke else ['--num_train_epochs', str(experiment['epochs'])])


def run_stage(command, log, stage, status_path, status):
    status.update(current_stage=stage, state='running', updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    print(f'START {stage}', flush=True)
    started = time.monotonic()
    with log.open('x') as f:
        f.write('COMMAND ' + json.dumps(command) + '\n'); f.flush()
        result = subprocess.run(command, stdout=f, stderr=subprocess.STDOUT)
    status.setdefault('stages', []).append({'stage': stage, 'exit_code': result.returncode,
                                           'seconds': time.monotonic() - started, 'log': str(log)})
    status_path.write_text(json.dumps(status, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError(f'{stage} failed with exit code {result.returncode}; inspect {log}')
    print(f'DONE {stage}', flush=True)


def verify_training(workspace, epochs, smoke):
    # NVFlare may return without a nonzero process code after a client failure.
    # An initial server checkpoint alone is not evidence of completed training.
    files = list(Path(workspace).rglob('training_diagnostics.json'))
    if len(files) != 1:
        raise RuntimeError('Expected exactly one completed client round with training diagnostics')
    report = json.loads(files[0].read_text())
    if report['global_step'] < 1 or not math.isfinite(report['training_loss']):
        raise RuntimeError('No successful finite-loss optimizer updates')
    if not smoke and report.get('epoch', 0) < epochs - 1e-6:
        raise RuntimeError('Client did not finish the declared epoch budget')
    return report


def evaluation_command(data, checkpoint, dest, split):
    # `train` is the NVFlare workspace, so evaluation must use another name.
    return [sys.executable, '-u', 'evaluate_subtype.py', 'medgemma', '--data-dir', str(data),
            '--split', split, '--model-path', str(checkpoint), '--max-seq-length', '32768',
            '--output', str(dest / f'eval_{split}')]


def run_experiment(ex, data, dest, seed, smoke, resume, status_path, status):
    name = ex['name']
    dest.mkdir(exist_ok=resume)
    workspace = dest / 'train'
    checkpoint = workspace / 'medgemma/server/simulate_job/app_server/FL_global_model.pt'
    if workspace.exists():
        if not resume or not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError('Existing training workspace has no reusable checkpoint; refusing overwrite')
        trained = verify_training(workspace, ex['epochs'], smoke)
        if trained.get('seed') != seed or trained.get('learning_rate') != ex['learning_rate']:
            raise ValueError('Existing checkpoint training settings differ from this experiment')
        print(f'SKIP {name}:fit (completed checkpoint verified)', flush=True)
    else:
        run_stage(train_command(data, workspace, ex, seed, smoke),
                  dest / 'training.log', name + ':fit', status_path, status)
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise RuntimeError('Training returned without a global checkpoint')
        trained = verify_training(workspace, ex['epochs'], smoke)
    verified = dest / 'verified_training.json'
    if not verified.exists():
        verified.write_text(json.dumps(trained, indent=2) + '\n')
    if smoke:
        return
    for split in ex['evaluation_splits']:
        evaluation = dest / f'eval_{split}'
        if evaluation.exists():
            report_path = evaluation / 'evaluation.json'
            if not resume or not report_path.is_file() or not (evaluation / 'predictions.csv').is_file():
                raise RuntimeError(f'Incomplete/existing evaluation at {evaluation}; refusing overwrite')
            report = json.loads(report_path.read_text())
            if (report['split'] != split or report['arguments']['model_path'] != str(checkpoint) or
                    report['arguments']['data_dir'] != str(data)):
                raise ValueError('Existing evaluation provenance differs from this experiment')
            print(f'SKIP {name}:evaluate_{split} (completed evaluation)', flush=True)
            continue
        log = dest / f'evaluate_{split}.log'
        attempt = 1
        while log.exists():
            attempt += 1
            log = dest / f'evaluate_{split}.attempt{attempt}.log'
        run_stage(evaluation_command(data, checkpoint, dest, split), log,
                  name + ':evaluate_' + split, status_path, status)
    if 'validation' in ex['evaluation_splits']:
        threshold = dest / 'validation_patient_threshold.json'
        if resume and threshold.exists():
            print(f'SKIP {name}:validation_threshold (existing artifact)', flush=True)
            return
        cmd = [sys.executable, 'fusion_threshold.py', 'fit', '--evaluation-dir', str(dest / 'eval_validation'),
               '--data-dir', str(data), '--output', str(threshold)]
        run_stage(cmd, dest / 'threshold.log', name + ':validation_threshold', status_path, status)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true', help='One-step checks of multimodal, image and methylation paths')
    p.add_argument('--resume', action='store_true', help='Reuse verified completed training; run only pending stages')
    args = p.parse_args()
    data_root, output = Path(args.data_root), Path(args.output)
    plan = json.loads((data_root / 'diagnostics_plan.json').read_text())
    if plan['experiments'] != make_plan() or plan['test_evaluation_enabled']:
        raise ValueError('Changed diagnostic plan; review before running')
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    if args.resume:
        saved = json.loads((output / 'run_manifest.json').read_text())
        if saved['plan'] != plan or saved['smoke'] != args.smoke or Path(saved['data_root']).resolve() != data_root.resolve():
            raise ValueError('Resume plan, mode, or data root differs from the existing run')
    for name in ('tiny', 'multimodal', 'image', 'methylation'):
        validate_clients(data_root / name / 'clients', 1, require_patient_groups=True)
    output.mkdir(parents=True, exist_ok=args.resume)
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(plan['training_seed']))
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'nvflare')}
    manifest = {
        'plan': plan, 'data_root': str(data_root), 'smoke': args.smoke,
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'versions': versions, 'python': sys.version}
    if args.resume:
        if saved['versions'] != versions:
            raise ValueError('Runtime package versions changed; review before resuming')
        history = output / 'resume_history.jsonl'
        with history.open('a') as f:
            f.write(json.dumps(manifest) + '\n')
    else:
        (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (output / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    status_path = output / 'status.json'; status = {'arm': plan['arm'], 'state': 'starting', 'stages': []}
    if args.resume and status_path.exists():
        status = json.loads(status_path.read_text())
        status.pop('error', None)
    experiments = plan['experiments']
    if args.smoke:
        experiments = [experiments[i] for i in (0, 3, 4)]
    try:
        for ex in experiments:
            run_experiment(ex, data_root / ex['dataset'], output / ex['name'], plan['training_seed'],
                           args.smoke, args.resume, status_path, status)
        status.update(state='completed', current_stage=None)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
