import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from check_methylation_context import measure_context
from methylation_manifest import COHORTS
from methylation_utils import make_record
from methylation_validate import validate_clients
from prepare_shared_methylation import (CLASSES, audit_all, comet_records, export_sites,
                                        prepare, read_tsv, write_tsv)


def synthetic(root, n_probes=1000):
    rng = random.Random(42)
    for platform, cohorts in COHORTS.items():
        for dx in cohorts:
            code = "ALL" if dx == "ALL-P3" else dx
            ids = [f'TARGET-01-{code}{i:03d}-01A' for i in range(10)]
            cols = [s + '.Ave_Beta' for s in ids]
            listing = 'all.samples' if dx == 'ALL-P3' else dx.lower()+'.samples'
            with (root/listing).open('w') as f:
                csv.writer(f, delimiter='\t').writerow(cols)
            suffix = '450k' if platform == '450k' else 'EPIC'
            with (root/f'TARGET-{dx}__{suffix}_beta.txt').open('w') as f:
                w = csv.writer(f, delimiter='\t')
                w.writerow(['TargetID']+cols)
                for j in range(n_probes):
                    w.writerow([f'cg{j:08d}'] + [0.5 if j == 2 else
                               .1 + .1*CLASSES.index(code) + rng.uniform(-.02, .02) for _ in ids])
    meta = [dict(Sample_Name=f'ARRAY{i:03d}', **{'UID.Subject': f'RMS{i:03d}',
             'Disease': 'RMS', 'Sample.Type': 'Patient tumor', 'Final.QC': 'PASS', 'Pass.QC': 'PASS'})
            for i in range(10)]
    meta += [{**meta[0], 'Sample_Name': 'ARRAY100'},
             {**meta[1], 'Sample_Name': 'ARRAY101', 'Sample.Type': 'Cell line'},
             {**meta[2], 'Sample_Name': 'ARRAY102', 'Sample.Type': 'Xenograft'},
             {**meta[3], 'Sample_Name': 'ARRAY103', 'Final.QC': 'FAIL'}]
    write_tsv(root/'rms_meta.txt', meta)
    with (root/'comet.tsv').open('w') as f:
        w = csv.writer(f, delimiter='\t')
        w.writerow(['probe'] + [r['Sample_Name'] for r in meta])
        # Last probe is absent in COMET and therefore must not be selected/tested.
        for j in range(n_probes-1):
            w.writerow([f'cg{j:08d}'] + [('NA' if j == 1 else .5 if j == 2 else rng.uniform(.78, .82)) for _ in meta])


class SharedTests(unittest.TestCase):
    def test_comet_ave_beta_header_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = '201985320079_R06C01'
            write_tsv(root/'meta.txt', [dict(Sample_Name=sample, **{
                'UID.Subject': 'SYNTHETIC_PATIENT', 'Disease': 'RMS',
                'Sample.Type': 'Patient tumor', 'Final.QC': 'PASS', 'Pass.QC': 'PASS'})])
            (root/'beta.txt').write_text(
                f'TargetID\t{sample}.Ave_Beta\t{sample}.Detection.Pval\n'
                'cg01763666\t0.8703868\t0.001\n')
            rows, report = comet_records(root/'beta.txt', root/'meta.txt')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['patient_id'], 'COMET:SYNTHETIC_PATIENT')
            self.assertEqual(rows[0]['beta_column'], sample+'.Ave_Beta')
            self.assertEqual(rows[0]['detection_column'], sample+'.Detection.Pval')
            self.assertEqual(rows[0]['status'], 'candidate_pending_label_review')
            self.assertEqual(report['metadata_samples_not_in_matrix'], [])

    def test_cross_source_label_conflict_is_not_deduplicated_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic(root, 5)
            write_tsv(root/'crosswalk.tsv', [
                {'source': 'TARGET', 'source_patient_id': 'TARGET-01-AML000', 'patient_id': 'SAME'},
                {'source': 'COMET', 'source_patient_id': 'RMS000', 'patient_id': 'SAME'}])
            report = audit_all(root, root/'comet.tsv', root/'rms_meta.txt', root/'audit',
                               patient_map=root/'crosswalk.tsv')
            self.assertEqual(report['sample_status']['cross_cohort_label_conflict'], 3)

    def test_context_measurement_counts_actual_template_and_reserve(self):
        class Processor:
            @staticmethod
            def tokenizer(text, **kwargs):
                return {'input_ids': list(map(ord, text))}

            @staticmethod
            def apply_chat_template(messages, add_generation_prompt=False, **kwargs):
                prefix = 'USER:' + messages[0]['content'][0]['text'] + '\nASSISTANT:'
                return prefix if add_generation_prompt else prefix + messages[1]['content'][0]['text'] + '!'
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'clients/site-1').mkdir(parents=True)
            row = make_record('P', 'RMS', ['cg00000001'], [.2], CLASSES, prompt_style='pediatric_v1')
            (root/'task.json').write_text(json.dumps({'n_clients': 1, 'probes': ['cg00000001'], 'panel_id': row['panel_id']}))
            for path in ['clients/site-1/train.json', 'validation.json', 'test.json']:
                (root/path).write_text(json.dumps([row]))
            with patch('check_methylation_context.validate_clients') as validator:
                result = measure_context(root, Processor())
                validator.assert_called_once()
            self.assertEqual(result['required_max_seq_length'], len('USER:'+row['prompt']+'\nASSISTANT:')+32)
            self.assertEqual(result['sample_counts'], dict(train=1, validation=1, test=1))

    def test_audit_metadata_not_filename_and_patient_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic(root, 10)
            report = audit_all(root, root/'comet.tsv', root/'rms_meta.txt', root/'audit')
            self.assertEqual(report['candidate_patients_by_diagnosis'], {dx: 10 for dx in CLASSES})
            self.assertEqual(report['sample_status']['non_patient_tumor'], 2)
            self.assertEqual(report['sample_status']['metadata_qc_failed'], 1)
            self.assertEqual(report['sample_status']['additional_sample_same_patient'], 1)
            records, _ = comet_records(root/'comet.tsv', root/'rms_meta.txt')
            # Plain beta column names must NEVER be treated as detection p-values.
            self.assertTrue(all(r['detection_column'] == '' for r in records))

    def test_unknown_comet_columns_require_explicit_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic(root, 5)
            (root/'other.tsv').write_text('probe\tUNKNOWN\ncg00000000\t0.5\n')
            rows, _ = comet_records(root/'other.tsv', root/'rms_meta.txt')
            self.assertEqual(rows[0]['status'], 'unresolved_comet_column')
            write_tsv(root/'map.tsv', [{'beta_column': 'UNKNOWN', 'sample_name': 'ARRAY000'}])
            rows, _ = comet_records(root/'other.tsv', root/'rms_meta.txt', root/'map.tsv')
            self.assertEqual(rows[0]['patient_id'], 'COMET:RMS000')

    def test_prompt(self):
        row = make_record('PRIVATE_ID', 'RMS', ['cg00000001'], [.25], CLASSES, prompt_style='pediatric_v1')
        self.assertEqual(row['prompt'], 'Classify this pediatric tumor using DNA methylation.\n\n'
                         '[DNA methylation]\ncg00000001: 0.250000\n\n'
                         'What is the most likely diagnosis?\nOptions: ALL, AML, CCSK, NBL, OS, RT, WT, RMS.\n'
                         'Reply with only the diagnosis abbreviation.')
        self.assertNotIn('PRIVATE_ID', row['prompt'])

    @unittest.skipUnless(os.environ.get('RUN_R_INTEGRATION') == '1', 'Set RUN_R_INTEGRATION=1 with compatible R packages')
    def test_real_limma_and_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            synthetic(root)
            audit_all(root, root/'comet.tsv', root/'rms_meta.txt', root/'audit')
            (root/'excluded.txt').write_text('cg00000003\n')
            args = argparse.Namespace(audit_dir=root/'audit', output=root/'prepared', confirm_reviewed_labels=True,
                                      confirm_patient_identity=True, exclude_unresolved=False, top_fraction=.01,
                                      seed=1, max_sample_missing=.2, max_probe_missing=.05,
                                      exclude_probes=root/'excluded.txt', r_library=None,
                                      rscript=os.environ.get('TEST_RSCRIPT', 'Rscript'))
            prepare(args)
            spec = json.loads((root/'prepared/features.json').read_text())
            summary = spec['selection']
            self.assertEqual(summary['common_cpgs_before_qc'], 999)
            self.assertEqual(summary['tested_nonconstant_cpgs'], 996)
            self.assertEqual(len(spec['probes']), math.ceil(996*.01))
            eligible = {r['probe_id'] for r in read_tsv(root/'prepared/eligible_cpgs.tsv')}
            self.assertTrue({'cg00000001', 'cg00000002', 'cg00000003', 'cg00000999'}.isdisjoint(eligible))
            for mode in ('iid', 'non-iid'):
                export_sites(root/'prepared', root/mode, mode)
                validate_clients(root/mode/'clients', 3)
            iid = json.loads((root/'iid/task.json').read_text())
            noniid = json.loads((root/'non-iid/task.json').read_text())
            self.assertEqual(iid['prepared_split_hashes'], noniid['prepared_split_hashes'])
            self.assertEqual(iid['panel_id'], noniid['panel_id'])
            for site in ('site-1', 'site-2', 'site-3'):
                records = json.loads((root/f'iid/clients/{site}/train.json').read_text())
                self.assertEqual({r['label_name'] for r in records}, set(CLASSES))
                records = json.loads((root/f'non-iid/clients/{site}/train.json').read_text())
                self.assertEqual({r['label_name'] for r in records}, set(noniid['site_classes'][site]))
            subprocess.run([os.environ.get('PYTHON', 'python3'), 'evaluate_methylation.py', 'ridge',
                            '--data-dir', str(root/'iid'), '--output', str(root/'ridge_validation'),
                            '--rscript', args.rscript], check=True)
            self.assertEqual(json.loads((root/'ridge_validation/evaluation.json').read_text())['metrics']['n_patients'], 16)
            # Change only held-out beta values: selected panel and training medians must be identical.
            manifest = read_tsv(root/'prepared/splits.tsv')
            for matrix in {r['matrix_file'] for r in manifest}:
                columns = {r['beta_column'] for r in manifest if r['matrix_file'] == matrix and r['split'] != 'train'}
                with open(matrix) as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    fields, rows = reader.fieldnames, list(reader)
                for row in rows:
                    for col in columns:
                        row[col] = '0.5'
                with open(matrix, 'w') as f:
                    w = csv.DictWriter(f, fieldnames=fields, delimiter='\t'); w.writeheader(); w.writerows(rows)
            config = json.loads((root/'prepared/config.json').read_text())
            config['output'] = str(root/'perturbed')
            (root/'perturbed_config.json').write_text(json.dumps(config))
            subprocess.run([args.rscript, 'methylation/prepare.R', str(root/'perturbed_config.json')], check=True)
            self.assertEqual((root/'prepared/features.tsv').read_text(), (root/'perturbed/features.tsv').read_text())
            # Shared artifact and prompt guards fail closed.
            file = root/'iid/clients/site-1/train.json'
            records = json.loads(file.read_text())
            records[0]['prompt'] = records[0]['prompt'].replace(spec['probes'][0], 'cg99999999')
            file.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, 'CpG order'):
                validate_clients(root/'iid/clients', 3)


if __name__ == '__main__':
    unittest.main()
