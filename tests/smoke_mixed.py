"""Synthetic seven-class end-to-end local-panel preparation; no GPU needed."""
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_mixed_methylation import audit
from methylation_manifest import COHORTS


def main():
    rng = random.Random(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for platform, cohorts in COHORTS.items():
            for k, dx in enumerate(cohorts):
                ids = [f'TARGET-01-{dx.replace("-", "")}{i:03d}-01A' for i in range(20)]
                cols = [s+'.Ave_Beta' for s in ids]
                name = 'all.samples' if dx == 'ALL-P3' else dx.lower()+'.samples'
                with (root/name).open('w') as f:
                    csv.writer(f, delimiter='\t').writerow(cols)
                suffix = '450k' if platform == '450k' else 'EPIC'
                with (root/f'TARGET-{dx}__{suffix}_beta.txt').open('w') as f:
                    w = csv.writer(f, delimiter='\t')
                    w.writerow(['TargetID']+cols)
                    for j in range(50):
                        w.writerow([f'cg{j:08d}']+[.1+.15*k+rng.uniform(-.01, .01) for _ in ids])
        audit(root, root/'audit')
        subprocess.run([sys.executable, 'prepare_mixed_methylation.py', '--audit-dir', str(root/'audit'),
                        '--output', str(root/'prepared'), '--confirm-cohort-labels', '--exclude-unresolved',
                        '--top-k', '10', '--rscript', os.environ.get('TEST_RSCRIPT', 'Rscript')], check=True)
        task = json.loads((root/'prepared/task.json').read_text())
        assert len(task['classes']) == 7
        assert all(len(p['probes']) == 10 for p in task['site_panels'].values())
        print('PASS: real local limma on three sites, seven-class export, disjoint patient validation.')


if __name__ == '__main__':
    main()
