import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class SuiteTests(unittest.TestCase):
    def run_fixture(self, arm, fail=''):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'scripts').mkdir()
            (root/'scripts/run_fusion_ablation.sh').write_text('''#!/usr/bin/env bash
printf '%s|%s|%s|%s|%s|%s|%s\\n' "$1" "$COMET_DATA_PREFIX" "$COMET_INPUT_TAG" "$COMET_ROUNDS" "$COMET_EPOCHS" "${COMET_FL_LAYOUT:-}" "$COMET_FL_METHOD" >> calls
if [[ ${FAIL_METHOD:-} == "$COMET_FL_METHOD" ]]; then exit 9; fi
''')
            env={**os.environ,'FAIL_METHOD':fail,'COMET_ROUNDS':'99','COMET_EPOCHS':'99'}
            result=subprocess.run(['bash',str(ROOT/'scripts/run_fusion_suite.sh'),arm,'train'],cwd=root,env=env)
            return result.returncode,(root/'calls').read_text().splitlines()

    def test_five_groups_fixed_budget_and_input_arm(self):
        for arm,prefix in [('top16','data/fixed849_disjoint'),('spatial16','data/fixed849_spatial16')]:
            code,rows=self.run_fixture(arm)
            self.assertEqual(code,0);self.assertEqual(len(rows),5)
            for r in rows:self.assertEqual(r.split('|')[1:5],[prefix,arm,'5','2'])
            self.assertEqual([r.split('|')[-2:] for r in rows[1:]],
                             [['iid','fedavg'],['iid','fedlc'],['skew08','fedavg'],['skew08','fedlc']])

    def test_stops_on_failed_group(self):
        code,rows=self.run_fixture('top16','fedlc')
        self.assertNotEqual(code,0);self.assertEqual(len(rows),3)

if __name__=='__main__':unittest.main()
