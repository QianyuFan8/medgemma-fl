"""Public CLI and package boundaries after removing root-level Python scripts."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_all_migrated_implementations_and_entries_exist(self):
        mapping=json.loads((ROOT/'docs/entrypoint_migration.json').read_text())
        self.assertEqual(list(ROOT.glob('*.py')),[])
        for path in [*mapping['implementation'].values(),*mapping['entrypoints'].values()]:
            self.assertTrue((ROOT/path).is_file(),path)
        old=set(mapping['imports'])
        for file in (ROOT/'src').rglob('*.py'):
            for n in ast.walk(ast.parse(file.read_text())):
                if isinstance(n,ast.ImportFrom) and not n.level:self.assertNotIn(n.module,old,str(file))

    def test_public_cli_and_legacy_dispatch_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as td:
            for script,args in [('train/centralized.py',[]),('inference/medgemma.py',[]),
                                ('data_preprocessing/prepare_multiimage_diagnostics.py',[]),
                                ('scripts/legacy_entrypoint.py',['train_centralized.py'])]:
                result=subprocess.run([sys.executable,str(ROOT/script),*args,'--help'],cwd=td,
                    capture_output=True,text=True,timeout=60,env={**os.environ,'HF_HUB_OFFLINE':'1'})
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertIn('usage:',result.stdout.lower())


if __name__=='__main__':unittest.main()
