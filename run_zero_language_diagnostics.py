"""Zero-shot modality controls, then H&E with frozen vision and language LoRA.

Three zero-shot validation evaluations use the unadapted 4-bit base model.
One continuous 10-epoch image-only run saves epochs 2/5/10, evaluated on train
and validation. Candidate likelihoods determine labels; test is never scored.
"""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from adapter_scope import infer_adapter_scope
from methylation_validate import validate_clients
from run_central_diagnostics import train_command, verify_training, run_stage, evaluation_command
from run_epoch_diagnostics import EPOCHS, snapshot

BASE_MODEL = 'google/medgemma-1.5-4b-it'
ZERO_MODALITIES = ('image', 'multimodal', 'methylation')


def language_experiment():
    return dict(name='image_language_only_lr5e5', dataset='image', epochs=10, learning_rate=5e-5)


def score_command(data, checkpoint, dest, split):
    return evaluation_command(data, checkpoint, dest, split) + [
        '--prediction-source', 'option_scores', '--skip-generation']


def verify_language_audit(workspace):
    files = list(workspace.rglob('trainable_parameters.json'))
    if len(files) != 1:
        raise RuntimeError('Expected one trainable parameter audit')
    audit = json.loads(files[0].read_text())
    names = audit.get('trainable_tensors', [])
    if (audit.get('adapter_scope') != 'language-only' or not names or
            audit.get('vision_trainable_parameters') != 0 or
            audit.get('trainable_parameters', 0) <= 0 or
            any('.language_model.' not in n or ('.lora_A.' not in n and '.lora_B.' not in n) for n in names)):
        raise RuntimeError('Frozen-vision/language-only audit failed')
    return audit


def add_summary(output, rows, dest, name, epoch, split):
    report = json.loads((dest / f'eval_{split}/evaluation.json').read_text())
    metrics = report['metrics']['patient_level_score_0_5']
    rows.append({'model': name, 'epoch': epoch, 'split': split,
                 'n_patients': metrics['n_patients'], 'threshold': 0.5,
                 'prediction_source': 'option_scores', 'accuracy': metrics['accuracy'],
                 'balanced_accuracy': metrics['balanced_accuracy'], 'roc_auc': metrics['roc_auc'],
                 'pr_auc': metrics['pr_auc'], 'recall_NEG': metrics['recall']['NEG'],
                 'recall_POS': metrics['recall']['POS']})
    (output / 'summary.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    plan = json.loads((root / 'diagnostics_plan.json').read_text())
    if plan['test_evaluation_enabled']:
        raise ValueError('Test scoring must be disabled')
    for mode in ZERO_MODALITIES:
        validate_clients(root / mode / 'clients', 1, require_patient_groups=True)
        task = json.loads((root / mode / 'task.json').read_text())
        if task['modality'] != mode:
            raise ValueError('Export modality does not match the requested experiment')
    output.mkdir(parents=True, exist_ok=False)
    seed = plan['training_seed']
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(seed))
    manifest = dict(data_root=str(root), arm=plan['arm'], seed=seed, base_model=BASE_MODEL,
                    zero_shot_modalities=ZERO_MODALITIES, zero_shot_splits=['validation'],
                    experiment=language_experiment(), snapshot_epochs=EPOCHS,
                    prediction_source='option_scores', skip_generation=True,
                    adapter_scope='language-only', test_evaluation_enabled=False, source_plan=plan,
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    versions={n: importlib.metadata.version(n) for n in
                              ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'nvflare')})
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    status_path = output / 'status.json'
    status = dict(state='starting', arm=plan['arm'], stages=[])
    rows = []
    try:
        for mode in ZERO_MODALITIES:
            name = f'zero_shot_{mode}'; dest = output / name; dest.mkdir()
            cmd = score_command(root / mode, BASE_MODEL, dest, 'validation')
            run_stage(cmd, dest / 'evaluate_validation.log', name + ':validation', status_path, status)
            add_summary(output, rows, dest, name, 0, 'validation')
        ex = language_experiment(); data = root / 'image'
        dest = output / ex['name']; dest.mkdir(); workspace = dest / 'train'
        cmd = train_command(data, workspace, ex, seed)
        cmd[cmd.index('--adapter_scope') + 1] = 'language-only'
        cmd += ['--save_adapter_epochs'] + [str(e) for e in EPOCHS]
        run_stage(cmd, dest / 'training.log', ex['name'] + ':fit_10_epochs', status_path, status)
        trained = verify_training(workspace, 10, False)
        (dest / 'verified_training.json').write_text(json.dumps(trained, indent=2) + '\n')
        audit = verify_language_audit(workspace)
        (dest / 'verified_parameter_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        for epoch in EPOCHS:
            checkpoint = snapshot(workspace, epoch)
            # Check reload scope before allocating an inference model.
            import torch
            weights = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
            if infer_adapter_scope(weights) != 'language-only':
                raise RuntimeError('Snapshot must contain only language adapters')
            del weights
            ep = dest / f'epoch-{epoch:02d}'; ep.mkdir()
            for split in ('train', 'validation'):
                cmd = score_command(data, checkpoint, ep, split)
                run_stage(cmd, ep / f'evaluate_{split}.log',
                          f"{ex['name']}:epoch{epoch}:evaluate_{split}", status_path, status)
                add_summary(output, rows, ep, ex['name'], epoch, split)
            cmd = [sys.executable, 'fusion_threshold.py', 'fit', '--evaluation-dir',
                   str(ep / 'eval_validation'), '--data-dir', str(data), '--output',
                   str(ep / 'validation_patient_threshold.json')]
            run_stage(cmd, ep / 'threshold.log', f"{ex['name']}:epoch{epoch}:validation_threshold",
                      status_path, status)
        status.update(state='completed', current_stage=None)
        print('ALL_COMPLETED: three zero-shot controls and frozen-vision language LoRA; no test.', flush=True)
    except BaseException as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status_path.write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    main()
