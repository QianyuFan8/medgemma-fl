"""Exercise launch argument forwarding and actionable preflight failures without a GPU."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


class LauncherTests(unittest.TestCase):
    def fixture(self, root):
        for name in ('scripts', '.venv/bin', 'mock-bin', 'source-run', 'hf', 'bundle'):
            (root / name).mkdir(parents=True)
        script = Path(__file__).resolve().parents[1] / 'scripts/launch_spatial20_inference.sh'
        shutil.copyfile(script, root / 'scripts/launch_spatial20_inference.sh')
        (root / '.venv/bin/activate').write_text('')
        mock = root / '.venv/bin/python'
        mock.write_text('#!/bin/bash\nif [[ "$1" == -c ]]; then exit 0; fi\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
        mock.chmod(0o755)
        for name in ('flock', 'nvidia-smi'):
            mock = root / 'mock-bin' / name
            mock.write_text('#!/bin/sh\nexit 0\n'); mock.chmod(0o755)
        (root / 'bundle/prepared_inputs.json').write_text('{}')
        env = {**os.environ, 'PATH': str(root / 'mock-bin') + ':' + os.environ['PATH'],
               'HF_HOME': str(root / 'hf'), 'CAPTURE_ARGS': str(root / 'args.txt')}
        env.pop('SPATIAL20_BUNDLE', None)
        return env

    def test_language_scope_and_bundle_reach_background_process(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); env = self.fixture(root)
            result = subprocess.run(['bash', 'scripts/launch_spatial20_inference.sh',
                                     str(root / 'source-run'), '60', 'language-only', str(root / 'bundle')],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            run = Path((root / '.spatial20_inference.latest').read_text().strip())
            for _ in range(100):
                if (run / 'exit_code').is_file(): break
                time.sleep(.02)
            self.assertEqual((run / 'exit_code').read_text().strip(), '0')
            args = (root / 'args.txt').read_text().splitlines()
            for flag, value in [('--source-run', str(root / 'source-run')), ('--epoch', '60'),
                                ('--adapter-scope', 'language-only'), ('--prepared-inputs', str(root / 'bundle'))]:
                self.assertEqual(args[args.index(flag) + 1], value)

    def test_missing_bundle_reports_reason_without_starting(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); env = self.fixture(root)
            result = subprocess.run(['bash', 'scripts/launch_spatial20_inference.sh',
                                     str(root / 'source-run'), '60', 'language-only', str(root / 'missing')],
                                    cwd=root, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing prepared bundle', result.stderr)
            self.assertFalse((root / '.spatial20_inference.latest').exists())


if __name__ == '__main__':
    unittest.main()
