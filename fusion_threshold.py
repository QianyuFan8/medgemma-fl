"""Fit a validation-only decision threshold and apply it to saved test scores.

No GPU or retraining is needed. Original evaluation files are never modified.
Default unit is canonical patient (mean score over that patient's records).
This changes hard decisions, not probability calibration or ROC ranking.

Usage:
  python fusion_threshold.py fit --evaluation-dir RUN_validation --data-dir EXPORT --output threshold.json
  python fusion_threshold.py apply --evaluation-dir RUN_test --data-dir EXPORT --threshold threshold.json --output threshold_test.json
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from fusion_identity import validate_patient_rows
from subtype_utils import subtype_metrics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def identity(report):
    args = report['arguments']
    if report['mode'] != 'medgemma' or not args.get('model_path'):
        raise ValueError('This threshold tool requires a single MedGemma evaluation')
    return {key: report.get(key) for key in (
        'mode', 'panel_id', 'modality', 'training_regime', 'partition', 'checkpoint_sha256')} | {
        key: args.get(key) for key in ('model_path', 'base_model', 'max_seq_length', 'parse', 'data_dir')}


def load_evaluation(evaluation_dir, data_dir, required_split, unit):
    root, data = Path(evaluation_dir), Path(data_dir)
    report = json.loads((root / 'evaluation.json').read_text())
    if report['split'] != required_split:
        raise ValueError(f'Expected {required_split} evaluation; never fit thresholds on test')
    if Path(report['arguments']['data_dir']).resolve() != data.resolve():
        raise ValueError('Data directory differs from evaluation provenance')
    task = json.loads((data / 'task.json').read_text())
    if task.get('classes') != ['NEG', 'POS'] or task.get('panel_id') != report['panel_id']:
        raise ValueError('Task or CpG panel mismatch')
    with (data / 'splits.tsv').open(newline='') as f:
        manifest = validate_patient_rows(list(csv.DictReader(f, delimiter='\t')))
    expected = {rid: (uid, label) for rid, (uid, split, label) in manifest.items() if split == required_split}
    reference = json.loads((data / f'{required_split}.json').read_text())
    if (len(reference) != len(expected) or
            {r['patient_id']: r['label_name'] for r in reference} != {k: v[1] for k, v in expected.items()}):
        raise ValueError('Evaluation reference disagrees with patient manifest')
    with (root / 'predictions.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != len(expected) or {r['patient_id'] for r in rows} != set(expected):
        raise ValueError('Incomplete or duplicate evaluation predictions')
    groups = {}
    for row in rows:
        uid, label = expected[row['patient_id']]
        score = float(row['score_POS'])
        if row['truth'] != label or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid probability or ground-truth mismatch')
        key = uid if unit == 'patient' else row['patient_id']
        groups.setdefault(key, {'truth': label, 'scores': []})['scores'].append(score)
    truth = [r['truth'] for _, r in sorted(groups.items())]
    scores = [math.fsum(r['scores']) / len(r['scores']) for _, r in sorted(groups.items())]
    if set(truth) != {'NEG', 'POS'}:
        raise ValueError('Both classes are required for threshold fitting and balanced evaluation')
    return report, truth, scores, {
        'n_records': len(rows), 'n_patients': len({v[0] for v in expected.values()}),
        'prediction_sha256': digest(root / 'predictions.csv'),
        'evaluation_sha256': digest(root / 'evaluation.json'),
        'patient_hashes': sorted({hashlib.sha256(v[0].encode()).hexdigest() for v in expected.values()}),
    }


def balanced_accuracy(truth, scores, threshold):
    return sum(sum((s >= threshold) == (label == 'POS') for label, s in zip(truth, scores) if label == cls)
               / truth.count(cls) for cls in ('NEG', 'POS')) / 2


def choose_threshold(truth, scores):
    if len(truth) != len(scores) or set(truth) != {'NEG', 'POS'}:
        raise ValueError('Threshold fitting requires aligned scores and both classes')
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid probability')
    values = sorted(set(scores))
    # Include all-positive/all-negative decisions and the original 0.5 threshold.
    candidates = {0.5, values[0], math.nextafter(values[-1], math.inf)}
    for left, right in zip(values, values[1:]):
        mid = left + (right - left) / 2
        candidates.add(mid if mid > left else right)
    return max(candidates, key=lambda t: (balanced_accuracy(truth, scores, t), -abs(t - 0.5), -t))


def metrics(truth, scores, threshold):
    return subtype_metrics(truth, ['POS' if s >= threshold else 'NEG' for s in scores],
                           scores, ['NEG', 'POS'], 'POS')


def fit(evaluation_dir, data_dir, unit='patient'):
    report, truth, scores, provenance = load_evaluation(evaluation_dir, data_dir, 'validation', unit)
    threshold = choose_threshold(truth, scores)
    # LOO estimates how unstable threshold selection is in the small validation set.
    correct_by_class = {'NEG': [], 'POS': []}
    for i in range(len(truth)):
        y, p = truth[:i] + truth[i + 1:], scores[:i] + scores[i + 1:]
        if set(y) != {'NEG', 'POS'}:
            correct_by_class = None
            break
        t = choose_threshold(y, p)
        correct_by_class[truth[i]].append(int((scores[i] >= t) == (truth[i] == 'POS')))
    loo = (sum(sum(v) / len(v) for v in correct_by_class.values()) / 2
           if correct_by_class else None)
    return {
        'schema_version': 1, 'fitted_split': 'validation', 'positive_class': 'POS', 'unit': unit,
        'patient_aggregation': 'mean_score_POS' if unit == 'patient' else None,
        'criterion': 'max_balanced_accuracy; ties closest_to_0.5 then lower',
        'threshold': threshold, 'model_identity': identity(report), 'validation_provenance': provenance,
        'identity_verification': 'checkpoint_hash' if report.get('checkpoint_sha256') else 'evaluation_metadata_only',
        'validation_default': metrics(truth, scores, 0.5),
        'validation_threshold_resubstitution': metrics(truth, scores, threshold),
        'validation_loo_balanced_accuracy': loo,
        'note': 'Validation resubstitution is optimistic. LOO only diagnoses threshold instability; '
                'it is not an independent model-performance estimate. Scores are not recalibrated.',
    }


def apply(evaluation_dir, data_dir, artifact):
    if (artifact.get('schema_version') != 1 or artifact.get('fitted_split') != 'validation' or
            artifact.get('positive_class') != 'POS' or artifact.get('unit') not in ('patient', 'record')):
        raise ValueError('Invalid validation threshold artifact')
    report, truth, scores, provenance = load_evaluation(evaluation_dir, data_dir, 'test', artifact['unit'])
    if identity(report) != artifact['model_identity']:
        raise ValueError('Threshold belongs to a different model, input layout, or evaluation configuration')
    if set(provenance['patient_hashes']) & set(artifact['validation_provenance']['patient_hashes']):
        raise ValueError('Validation/test patient overlap')
    threshold = float(artifact['threshold'])
    if not math.isfinite(threshold) or not 0 <= threshold <= math.nextafter(1.0, math.inf):
        raise ValueError('Invalid threshold')
    return {'split': 'test', 'unit': artifact['unit'], 'threshold': threshold,
            'threshold_fitted_split': 'validation', 'model_identity': identity(report),
            'provenance': provenance, 'default_0_5': metrics(truth, scores, 0.5),
            'validation_selected_threshold': metrics(truth, scores, threshold),
            'note': 'Original generated-label metrics remain in evaluation.json. This report uses score-based decisions.'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=('fit', 'apply'))
    p.add_argument('--evaluation-dir', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--threshold', help='Validation artifact, required by apply')
    p.add_argument('--unit', choices=('patient', 'record'), default='patient', help='Fitting unit; apply uses the saved unit')
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if args.command == 'fit':
        if args.threshold:
            p.error('fit does not accept --threshold')
        result = fit(args.evaluation_dir, args.data_dir, args.unit)
    else:
        if not args.threshold:
            p.error('apply requires --threshold from validation')
        result = apply(args.evaluation_dir, args.data_dir, json.loads(Path(args.threshold).read_text()))
        result['threshold_artifact_sha256'] = digest(args.threshold)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        f.write(json.dumps(result, indent=2) + '\n')
    print(f'Wrote {args.command} report to {output}; original scores and metrics unchanged.')


if __name__ == '__main__':
    main()
