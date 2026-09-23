"""Audit seven-cohort matrix headers and sample lists without fitting any model.

Labels are cohort-membership proposals, not independent clinical verification.
Unknown identifiers and cross-cohort patients require review before training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re

from methylation_manifest import COHORTS


def identify(sample):
    normalized = sample.replace('.', '-')
    match = re.fullmatch(r'(TARGET-\d{2}-[A-Za-z0-9]+)-(\d{2})[A-Za-z0-9]*(?:-[A-Za-z0-9]+)*', normalized)
    return (match.group(1), match.group(2)) if match else ('', '')


def audit(matrix_root, output):
    root, out = Path(matrix_root), Path(output)
    if out.exists():
        raise FileExistsError(f'Use a new audit directory: {out}')
    records, sources = [], {}
    for platform, cohorts in COHORTS.items():
        for cohort in cohorts:
            suffix = '450k' if platform == '450k' else 'EPIC'
            matrix = root / f'TARGET-{cohort}__{suffix}_beta.txt'
            listing = root / ('all.samples' if cohort == 'ALL-P3' else cohort.lower() + '.samples')
            with matrix.open() as f:
                header = next(csv.reader(f, delimiter='\t'))
            beta = [c[:-9] for c in header if c.endswith('.Ave_Beta')]
            if not beta or len(beta) != len(set(beta)):
                raise ValueError(f'Missing or duplicate beta sample columns: {matrix}')
            with listing.open() as f:
                cells = [c.strip() for row in csv.reader(f, delimiter='\t') for c in row]
            listed = [c[:-9] for c in cells if c.endswith('.Ave_Beta')]
            if not listed:
                raise ValueError(f'No .Ave_Beta identifiers in {listing}; inspect its format, do not guess labels')
            key = f'{platform}/{cohort}'
            sources[key] = {
                'matrix': str(matrix.resolve()), 'sample_list': str(listing.resolve()),
                'matrix_samples': len(beta), 'listed_samples': len(listed),
                'duplicate_list_entries': sorted(s for s, n in Counter(listed).items() if n > 1),
                'listed_but_absent_from_matrix': sorted(set(listed) - set(beta)),
                'matrix_samples_not_listed': len(set(beta) - set(listed)),
            }
            for sample in beta:
                pid, sample_type = identify(sample)
                member = sample in set(listed)
                status = ('not_in_sample_list' if not member else
                          'unresolved_patient_identifier' if not pid else
                          'sample_type_excluded' if sample_type not in ('01', '03', '09') else
                          'candidate_pending_label_review')
                records.append(dict(sample_id=sample, patient_id=pid, sample_type=sample_type,
                                    platform=platform, source_cohort=cohort,
                                    proposed_diagnosis='ALL' if cohort == 'ALL-P3' else cohort,
                                    in_sample_list=member, status=status,
                                    matrix_file=str(matrix.resolve()), beta_column=sample+'.Ave_Beta',
                                    label_source=str(listing.resolve())))
    patients = defaultdict(list)
    for r in records:
        if r['patient_id']:
            patients[r['patient_id']].append(r)
    cross_platform, cross_cohort, repeated = [], [], []
    for pid, rows in patients.items():
        if len(rows) > 1:
            repeated.append(pid)
        if len({r['platform'] for r in rows}) > 1:
            cross_platform.append(pid)
        if len({r['proposed_diagnosis'] for r in rows}) > 1:
            cross_cohort.append(pid)
            for r in rows:
                r['status'] = 'cross_cohort_label_conflict'
    counts = defaultdict(set)
    for r in records:
        if r['status'] == 'candidate_pending_label_review':
            counts[r['proposed_diagnosis']].add(r['patient_id'])
    report = {
        'stage': 'audit_only_not_training_ready',
        'label_basis': 'proposed_sample_list_cohort_membership',
        'independently_clinically_verified': False,
        'sources': sources,
        'sample_status': dict(Counter(r['status'] for r in records)),
        'candidate_unique_patients_by_diagnosis': {k: len(v) for k, v in sorted(counts.items())},
        'repeated_patient_ids': sorted(repeated),
        'cross_platform_patient_ids': sorted(cross_platform),
        'cross_cohort_patient_ids': sorted(cross_cohort),
        'notes': [
            'No matrix values were read, no panels selected, and no train/test splits assigned.',
            'Candidate counts are before QC and are not final cohort sizes.',
            'Unknown identifiers require an explicit diagnosis and patient-ID mapping; do not infer ALL from all.samples.',
            'Resolve cross-platform duplicates globally before site assignment or splitting.',
            'Sample types 01/03/09 reuse the existing eligibility rule; review its suitability for every cohort.',
        ],
    }
    out.mkdir(parents=True)
    with (out / 'samples_audit.tsv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]), delimiter='\t')
        w.writeheader()
        w.writerows(records)
    (out / 'audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('sources', 'repeated_patient_ids', 'cross_platform_patient_ids', 'cross_cohort_patient_ids')}, indent=2))
    print(f'Cross-platform patients: {len(cross_platform)}; cross-cohort patients: {len(cross_cohort)}')
    print(f'Private audit files saved to {out}; do not commit patient-level files.')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix-root', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    audit(a.matrix_root, a.output)
