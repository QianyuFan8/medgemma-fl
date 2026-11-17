"""Dispatch an old root CLI name to the new implementation without copying logic."""
import json
from pathlib import Path
import runpy
import sys

root = Path(__file__).resolve().parents[1]
entries = json.loads((root/'docs/entrypoint_migration.json').read_text())['entrypoints']
if len(sys.argv)<2 or sys.argv[1] not in entries:
    raise SystemExit('Usage: python scripts/legacy_entrypoint.py OLD_NAME.py [args]\nNames: '+', '.join(sorted(entries)))
target = root/entries[sys.argv[1]]
sys.argv = [str(target), *sys.argv[2:]]
runpy.run_path(str(target), run_name='__main__')
