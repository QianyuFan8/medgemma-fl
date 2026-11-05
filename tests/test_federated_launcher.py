import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


class FederatedLauncherTests(unittest.TestCase):
    def test_explicit_scope_and_sixty_round_budget_reach_background_runner(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name in ('scripts','.venv/bin','mock-bin','hf','data/federated_spatial16_v1'):
                (root/name).mkdir(parents=True)
            shutil.copy2(Path(__file__).resolve().parents[1]/'scripts/launch_federated_tiles16.sh',root/'scripts/launch.sh')
            (root/'.venv/bin/activate').write_text('')
            mock=root/'.venv/bin/python'
            mock.write_text('#!/bin/bash\nif [[ "$1" == - ]]; then cat >/dev/null; exit 0; fi\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n')
            mock.chmod(0o755)
            for name in ('flock','nvidia-smi'):
                mock=root/'mock-bin'/name;mock.write_text('#!/bin/sh\nexit 0\n');mock.chmod(0o755)
            env={**os.environ,'PATH':str(root/'mock-bin')+':'+os.environ['PATH'],
                 'HF_HOME':str(root/'hf'),'CAPTURE_ARGS':str(root/'args.txt')}
            for scope in ('language-only','lora-only'):
                run=subprocess.run(['bash','scripts/launch.sh',scope,'60','1'],cwd=root,env=env,capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                path=Path((root/'.federated_tiles16.latest').read_text().strip())
                for _ in range(100):
                    if (path/'exit_code').exists():break
                    time.sleep(.02)
                self.assertEqual((path/'exit_code').read_text().strip(),'0')
                args=(root/'args.txt').read_text().splitlines()
                for flag,value in (('--scope',scope),('--rounds','60'),('--local-epochs','1')):
                    self.assertEqual(args[args.index(flag)+1],value)
                self.assertIn('run_federated_tiles16.py',args)


if __name__ == '__main__':unittest.main()
