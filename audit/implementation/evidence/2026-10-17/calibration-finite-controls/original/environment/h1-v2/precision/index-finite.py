"""Index only the newly completed finite evidence; do not touch prior indexes."""
import hashlib,json
from pathlib import Path

index=Path('environment/h1-v2/precision/finite-inventory.json')
assert not index.exists()
roots=[Path(value) for value in (
    'environment/h1-v2/sensitivities','environment/h1-v2/probe-calibration',
    'environment/h1-v2/precision','environment/h1-v2/resource-proposal',
    'artifacts/h1-v2/sensitivities','artifacts/h1-v2/probe-calibration',
    'artifacts/h1-v2/precision','artifacts/h1-v2/finite-summary')]
report=Path('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-calibration-phase-b-sensitivity-precision-report.md')
paths=sorted({report}|{path for root in roots for path in root.rglob('*') if path.is_file() and path!=index})
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
rows=[dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)) for path in paths]
value=dict(scope='new finite Phase-B sensitivity/probe/precision evidence only; no historical namespace mutation',
    source_sha256='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c',
    execution_sha256='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c',
    count=len(rows),bytes=sum(row['bytes'] for row in rows),files=rows)
index.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
errors=[row['path'] for row in rows if Path(row['path']).stat().st_size!=row['bytes'] or digest(Path(row['path']))!=row['sha256']]
assert not errors,errors
print(json.dumps(dict(index=str(index),sha256=digest(index),count=value['count'],bytes=value['bytes'],verification_errors=errors)))
