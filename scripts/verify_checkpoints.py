"""Verify local historical checkpoint checksums without deserializing tensors."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'docs/checkpoints.json').read_text())
failed=False
for entry in manifest['checkpoints']:
    p=root/entry['path']
    if not p.is_file():
        print('MISSING',entry['path']);failed=True;continue
    with p.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    ok=digest==entry['sha256']
    print('OK' if ok else 'MISMATCH',entry['path']);failed|=not ok
sys.exit(1 if failed else 0)
