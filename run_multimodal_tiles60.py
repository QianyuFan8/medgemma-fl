"""Two centralized 60-epoch, sixteen-image + CpG QLoRA trajectories per VM.

Both start independently from the base model, not smoke or previous adapters.
Save epochs 2/5/10/20/30/50/60; score train and validation, never test.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from check_methylation_context import check_context
from prepare_multiimage_diagnostics import prepare, processor_preflight, signature
from run_central_diagnostics import train_command, verify_training, run_stage
from run_epoch_diagnostics import snapshot
from run_zero_language_diagnostics import BASE_MODEL, score_command, verify_language_audit, add_summary

EPOCHS = (2, 5, 10, 20, 30, 50, 60)
SCOPES = ('lora-only', 'language-only')


def experiment(scope):
    if scope not in SCOPES:
        raise ValueError('Unsupported adapter scope')
    return dict(name=f'multimodal_tiles16_{scope}_lr5e5', dataset='multimodal',
                epochs=60, learning_rate=5e-5, adapter_scope=scope)


def training_command(data, workspace, seed, scope, smoke=False):
    cmd = train_command(data, workspace, experiment(scope), seed, smoke=smoke)
    cmd[cmd.index('--adapter_scope') + 1] = scope
    if not smoke:
        cmd += ['--save_adapter_epochs'] + [str(e) for e in EPOCHS]
    return cmd


def verify_scope(workspace, scope):
    if scope == 'language-only':
        return verify_language_audit(workspace)
    if scope != 'lora-only':
        raise ValueError('Unsupported adapter scope')
    files = list(Path(workspace).rglob('trainable_parameters.json'))
    if len(files) != 1:
        raise RuntimeError('Expected one parameter audit')
    audit = json.loads(files[0].read_text())
    names = audit.get('trainable_tensors', [])
    if (audit.get('adapter_scope') != scope or not names or
            audit.get('vision_trainable_parameters', 0) <= 0 or
            not any('.vision_tower.' in n for n in names) or
            not any('.language_model.' in n for n in names) or
            any('.lora_A.' not in n and '.lora_B.' not in n for n in names)):
        raise RuntimeError('Expected both vision and language adapters, with no full trainable weights')
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--arm', required=True, choices=('top16', 'spatial16'))
    args = p.parse_args()
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    plan = json.loads((root / 'diagnostics_plan.json').read_text())
    if plan['test_evaluation_enabled'] or plan['arm'] != args.arm:
        raise ValueError('Source arm differs or test scoring is enabled')
    source = root / 'multimodal'
    cohort = signature(source)
    task = json.loads((source / 'task.json').read_text())
    if len(task['probes']) != 849 or task['modality'] != 'multimodal':
        raise ValueError('Expected the existing 849-CpG multimodal export')
    train_rows = json.loads((source / 'clients/site-1/train.json').read_text())
    per_epoch = math.ceil(len(train_rows) / 4)
    output.mkdir(parents=True, exist_ok=False)
    seed = plan['training_seed']
    os.environ.update(MEDGEMMA_ATTN_IMPLEMENTATION='sdpa', TOKENIZERS_PARALLELISM='false',
                      WANDB_MODE='disabled', PYTHONHASHSEED=str(seed))
    manifest = dict(source_data_root=str(root), arm=args.arm, seed=seed, base_model=BASE_MODEL,
                    experiments=[experiment(s) for s in SCOPES], snapshot_epochs=EPOCHS,
                    batch_size=1, gradient_accumulation_steps=4,
                    n_training_records=len(train_rows),
                    n_training_patients=len({r['canonical_uid'] for r in train_rows}),
                    optimizer_steps_per_epoch=per_epoch,
                    snapshot_steps={str(e): e * per_epoch for e in EPOCHS},
                    test_evaluation_enabled=False, prediction_source='option_scores',
                    independent_base_initialization=True, smoke_is_separate_from_main_trajectory=True,
                    scheduler='linear over the full 60-epoch trajectory; not matched to the old 10-epoch schedule',
                    source_plan=plan, source_split_sha256=hashlib.sha256((source / 'splits.tsv').read_bytes()).hexdigest(),
                    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    versions={n: importlib.metadata.version(n) for n in
                              ('torch', 'transformers', 'trl', 'peft', 'bitsandbytes', 'nvflare')})
    (output / 'run_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'code.patch').write_text(subprocess.check_output(['git', 'diff'], text=True))
    state = dict(state='starting', arm=args.arm, stages=[])
    status = output / 'status.json'
    rows = []
    try:
        state.update(state='running', current_stage='prepare')
        status.write_text(json.dumps(state, indent=2) + '\n')
        data = output / 'data'
        prepare(source, data, modality='multimodal')
        if signature(data) != cohort:
            raise ValueError('Cohort changed during preparation')
        # Preserve every CpG value, ordering and patient/specimen record.
        for split in ('train', 'validation', 'test'):
            if (data / f'{split}.csv').read_bytes() != (source / f'{split}.csv').read_bytes():
                raise ValueError('CpG matrix changed')
        context = check_context(data, BASE_MODEL, 32768)
        (output / 'context_check.json').write_text(json.dumps(context, indent=2) + '\n')
        processor_preflight(data, output / 'processor_preflight.json')
        print('INPUTS_VERIFIED: 16 images plus unchanged CpG text; one target per record.', flush=True)
        # Check BOTH memory paths before spending hours on either full trajectory.
        for scope in SCOPES:
            smoke = output / f'smoke_{scope}'
            run_stage(training_command(data, smoke, seed, scope, smoke=True),
                      output / f'smoke_{scope}.log', f'{scope}:smoke', status, state)
            report = verify_training(smoke, 1, True)
            if report['global_step'] != 1:
                raise RuntimeError('Smoke must complete exactly one optimizer step')
            verify_scope(smoke, scope)
        print('BOTH_SMOKES_PASSED: each full run will reload the base model.', flush=True)
        for scope in SCOPES:
            ex = experiment(scope)
            dest = output / ex['name']; dest.mkdir()
            workspace = dest / 'train'
            run_stage(training_command(data, workspace, seed, scope), dest / 'training.log',
                      f'{scope}:fit_60_epochs', status, state)
            report = verify_training(workspace, 60, False)
            if report['global_step'] != 60 * per_epoch:
                raise RuntimeError('Unexpected patient/specimen update budget')
            (dest / 'verified_training.json').write_text(json.dumps(report, indent=2) + '\n')
            (dest / 'verified_parameter_audit.json').write_text(json.dumps(verify_scope(workspace, scope), indent=2) + '\n')
            for epoch in EPOCHS:
                checkpoint = snapshot(workspace, epoch, planned_epochs=60)
                if json.loads(checkpoint.with_suffix('.json').read_text())['global_step'] != epoch * per_epoch:
                    raise RuntimeError('Snapshot update count mismatch')
                ep = dest / f'epoch-{epoch:02d}'; ep.mkdir()
                for split in ('train', 'validation'):
                    run_stage(score_command(data, checkpoint, ep, split), ep / f'evaluate_{split}.log',
                              f'{scope}:epoch{epoch}:{split}', status, state)
                    add_summary(output, rows, ep, ex['name'], epoch, split)
                cmd = [sys.executable, 'fusion_threshold.py', 'fit', '--evaluation-dir',
                       str(ep / 'eval_validation'), '--data-dir', str(data), '--output',
                       str(ep / 'validation_patient_threshold.json')]
                run_stage(cmd, ep / 'threshold.log', f'{scope}:epoch{epoch}:validation_threshold', status, state)
        state.update(state='completed', current_stage=None)
        print('ALL_COMPLETED: two 60-epoch multimodal trajectories; seven snapshots each; no test scoring.', flush=True)
    except BaseException as exc:
        state.update(state='failed', error=str(exc))
        raise
    finally:
        state['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status.write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
