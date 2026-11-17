"""Check the portable bundle before accessing cloud data or using cached features."""
import hashlib
import json
from pathlib import Path
import sys
root=Path(sys.argv[1]).resolve()
manifest=json.loads((root/'bundle_manifest.json').read_text())
for name,expected in manifest['sha256'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root):raise ValueError('Unsafe bundle path')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected:raise ValueError('Bundle file changed: '+name)
print('BUNDLE VERIFIED:',len(manifest['sha256']),'files;',manifest['cached_feature_count'],'cached WSI features')
