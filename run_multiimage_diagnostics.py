"""One centralized language-only QLoRA trajectory with sixteen independent images.

Preserve the existing montage run's record order, splits and update budget.
Save epochs 2/5/10 and evaluate train/validation only. No automatic test scoring.
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

from prepare_multiimage_diagnostics import prepare, processor_preflight, signature
from run_central_diagnostics import train_command, verify_training, run_stage
from run_epoch_diagnostics import EPOCHS, snapshot
from run_zero_language_diagnostics import score_command, verify_language_audit, add_summary


def experiment():
    return dict(name='image_tiles16_language_only_lr5e5', dataset='image', epochs=10, learning_rate=5e-5)


def training_command(data, workspace, seed, smoke=False):
    cmd = train_command(data, workspace, experiment(), seed, smoke=smoke)
    cmd[cmd.index('--adapter_scope') + 1] = 'language-only'
    if not smoke:
        cmd += ['--save_adapter_epochs'] + [str(e) for e in EPOCHS]
    return cmd


def find_baseline(arm):
    candidates = []
    for root in Path('runs').glob(f'zero_language_diagnostics_{arm}_*'):
        for path in (root, root / 'full'):
            status = path / 'status.json'
            if status.is_file() and json.loads(status.read_text()).get('state') == 'completed':
                candidates.append(path.resolve())
    if not candidates:
        raise FileNotFoundError('No completed matching frozen-vision montage baseline found')
    return max(candidates, key=lambda p: (p / 'status.json').stat().st_mtime)


def check_baseline(baseline, root, plan, versions):
    manifest = json.loads((baseline / 'run_manifest.json').read_text())
    if (manifest['arm'] != plan['arm'] or manifest['seed'] != plan['training_seed'] or
            manifest['adapter_scope'] != 'language-only' or manifest['versions'] != versions or
            manifest['experiment']['epochs'] != 10 or
            manifest['experiment']['learning_rate'] != experiment()['learning_rate']):
        raise ValueError('Baseline settings/runtime differ; do not run an unmatched comparison')
    old_root = Path(manifest['data_root']) / 'image'
    if signature(old_root) != signature(root / 'image'):
        raise ValueError('Baseline and requested cohort differ')
    old_rows = json.loads((old_root / 'clients/site-1/train.json').read_text())
    new_rows = json.loads((root / 'image/clients/site-1/train.json').read_text())
    if old_rows != new_rows:
        raise ValueError('Baseline source records/order/prompt changed')
    n = len(new_rows)
    per_epoch = math.ceil(n / 4)  # batch=1, accumulation=4, one GPU
    workspace = baseline / 'image_language_only_lr5e5/train'
    verify_language_audit(workspace)
    old_steps = {}
    for epoch in EPOCHS:
        checkpoint = snapshot(workspace, epoch)
        step = json.loads(checkpoint.with_suffix('.json').read_text())['global_step']
        if step != epoch * per_epoch:
            raise ValueError('Baseline optimizer budget differs from the expected record budget')
        old_steps[str(epoch)] = step
    return {'baseline': str(baseline), 'n_training_records': n,
            'n_training_patients': len({r['canonical_uid'] for r in new_rows}),
            'batch_size': 1, 'gradient_accumulation_steps': 4,
            'optimizer_steps_per_epoch': per_epoch, 'snapshot_steps': old_steps,
            'tile_level_dataset_expansion': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--baseline', help='Completed zero_language run directory; default auto-detect')
    args = p.parse_args()
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    plan = json.loads((root / 'diagnostics_plan.json').read_text())
    if plan['test_evaluation_enabled']:
        raise ValueError('Test evaluation must be disabled')
    versions = {n: importlib.metadata.version(n) for n in
                ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'nvflare')}
    baseline = Path(args.baseline).resolve() if args.baseline else find_baseline(plan['arm'])
    budget = check_baseline(baseline, root, plan, versions)
    output.mkdir(parents=True, exist_ok=False)
    seed = plan['training_seed']
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(seed))
    manifest = {'source_data_root': str(root), 'arm': plan['arm'], 'training_seed': seed,
                'experiment': experiment(), 'adapter_scope': 'language-only', 'versions': versions,
                'snapshot_epochs': EPOCHS, 'budget_comparison': budget,
                'test_evaluation_enabled': False, 'prediction_source': 'option_scores',
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'smoke_is_separate_from_main_trajectory': True}
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    state = {'state': 'starting', 'arm': plan['arm'], 'stages': []}
    status = output / 'status.json'
    try:
        data = output / 'data'
        state.update(current_stage='prepare', state='running')
        status.write_text(json.dumps(state, indent=2) + '\n')
        prepare(root / 'image', data)
        processor_preflight(data, output / 'processor_preflight.json')
        print('PREPROCESSING_PASSED: 16 images per record; source cohort unchanged.', flush=True)
        smoke = output / 'smoke'
        run_stage(training_command(data, smoke, seed, smoke=True), output / 'smoke.log',
                  'smoke:one_optimizer_step', status, state)
        verify_training(smoke, 1, True)
        verify_language_audit(smoke)
        print('GPU_SMOKE_PASSED: restarting from base weights for the 10-epoch comparison.', flush=True)
        ex = experiment()
        dest = output / ex['name']; dest.mkdir()
        workspace = dest / 'train'
        run_stage(training_command(data, workspace, seed), dest / 'training.log',
                  'tiles16:fit_10_epochs', status, state)
        trained = verify_training(workspace, 10, False)
        if trained['global_step'] != budget['snapshot_steps']['10']:
            raise ValueError('Main trajectory optimizer count differs from montage baseline')
        (dest / 'verified_training.json').write_text(json.dumps(trained, indent=2) + '\n')
        (dest / 'verified_parameter_audit.json').write_text(json.dumps(verify_language_audit(workspace), indent=2) + '\n')
        rows = []
        for epoch in EPOCHS:
            checkpoint = snapshot(workspace, epoch)
            if json.loads(checkpoint.with_suffix('.json').read_text())['global_step'] != budget['snapshot_steps'][str(epoch)]:
                raise ValueError('Snapshot step count differs from the montage baseline')
            ep = dest / f'epoch-{epoch:02d}'; ep.mkdir()
            for split in ('train', 'validation'):
                run_stage(score_command(data, checkpoint, ep, split), ep / f'evaluate_{split}.log',
                          f'tiles16:epoch{epoch}:{split}', status, state)
                add_summary(output, rows, ep, ex['name'], epoch, split)
            cmd = [sys.executable, 'fusion_threshold.py', 'fit', '--evaluation-dir',
                   str(ep / 'eval_validation'), '--data-dir', str(data), '--output',
                   str(ep / 'validation_patient_threshold.json')]
            run_stage(cmd, ep / 'threshold.log', f'tiles16:epoch{epoch}:validation_threshold', status, state)
        state.update(state='completed', current_stage=None)
        print('ALL_COMPLETED: independent tiles, epochs 2/5/10, matched optimizer steps; no test scoring.', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=str(exc))
        raise
    finally:
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
