"""Train two centralized 10-epoch trajectories; evaluate epoch 2/5/10 adapters.

Uses existing diagnostic exports. Never scores test or selects a checkpoint on
test. Outputs are new and immutable; this runner does not resume optimizer state.
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from methylation_validate import validate_clients
from run_central_diagnostics import train_command, verify_training, run_stage, evaluation_command


EPOCHS = (2, 5, 10)


def experiments():
    return [dict(name='multimodal_lr2e4', dataset='multimodal', epochs=10, learning_rate=2e-4),
            dict(name='image_lr5e5', dataset='image', epochs=10, learning_rate=5e-5)]


def snapshot(workspace, epoch, planned_epochs=10):
    matches = list(workspace.rglob(f'epoch_checkpoints/epoch-{epoch:02d}.pt'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one epoch {epoch} adapter; found {len(matches)}')
    path = matches[0]
    info = json.loads(path.with_suffix('.json').read_text())
    if (info['epoch'] != epoch or info['planned_epochs'] != planned_epochs or info['global_step'] <= 0 or
            info['checkpoint_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest()):
        raise RuntimeError(f'Invalid checkpoint metadata or checksum: {path}')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    plan = json.loads((root / 'diagnostics_plan.json').read_text())
    if plan['test_evaluation_enabled']:
        raise ValueError('Expected diagnostic data with test scoring disabled')
    for ex in experiments():
        validate_clients(root / ex['dataset'] / 'clients', 1, require_patient_groups=True)
    output.mkdir(parents=True, exist_ok=False)
    seed = plan['training_seed']
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(seed))
    manifest = {'data_root': str(root), 'arm': plan['arm'], 'seed': seed,
                'experiments': experiments(), 'snapshot_epochs': EPOCHS,
                'test_evaluation_enabled': False, 'source_plan': plan,
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'versions': {n: importlib.metadata.version(n) for n in
                             ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'nvflare')}}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    status_path = output / 'status.json'
    status = {'state': 'starting', 'arm': plan['arm'], 'stages': []}
    summary = []
    try:
        for ex in experiments():
            data = root / ex['dataset']
            dest = output / ex['name']; dest.mkdir()
            workspace = dest / 'train'
            cmd = train_command(data, workspace, ex, seed)
            cmd += ['--save_adapter_epochs'] + [str(e) for e in EPOCHS]
            run_stage(cmd, dest / 'training.log', ex['name'] + ':fit_10_epochs', status_path, status)
            trained = verify_training(workspace, 10, False)
            (dest / 'verified_training.json').write_text(json.dumps(trained, indent=2) + '\n')
            checkpoints = {epoch: snapshot(workspace, epoch) for epoch in EPOCHS}
            for epoch, checkpoint in checkpoints.items():
                ep = dest / f'epoch-{epoch:02d}'; ep.mkdir()
                for split in ('train', 'validation'):
                    cmd = evaluation_command(data, checkpoint, ep, split)
                    run_stage(cmd, ep / f'evaluate_{split}.log',
                              f"{ex['name']}:epoch{epoch}:evaluate_{split}", status_path, status)
                    report = json.loads((ep / f'eval_{split}/evaluation.json').read_text())
                    metrics = report['metrics']['patient_level_score_0_5']
                    summary.append({'model': ex['name'], 'epoch': epoch, 'split': split,
                                    'n_patients': metrics['n_patients'], 'threshold': 0.5,
                                    'balanced_accuracy': metrics['balanced_accuracy'],
                                    'roc_auc': metrics['roc_auc'], 'pr_auc': metrics['pr_auc'],
                                    'recall_NEG': metrics['recall']['NEG'],
                                    'recall_POS': metrics['recall']['POS']})
                    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
                    with (output / 'summary.csv').open('w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
                        writer.writeheader(); writer.writerows(summary)
                cmd = [sys.executable, 'fusion_threshold.py', 'fit', '--evaluation-dir',
                       str(ep / 'eval_validation'), '--data-dir', str(data), '--output',
                       str(ep / 'validation_patient_threshold.json')]
                run_stage(cmd, ep / 'threshold.log', f"{ex['name']}:epoch{epoch}:validation_threshold",
                          status_path, status)
        status.update(state='completed', current_stage=None)
        print('ALL_COMPLETED: two trajectories, six checkpoints, train/validation only.', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
