"""Hash the new actual full-resource evidence only, preserving earlier indexes."""
import hashlib,json,time
from pathlib import Path
base=Path('environment/h1-v2/complete-resource');index=base/'inventory.json';assert not index.exists()
report=Path('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-calibration-phase-b-complete-resource-report.md')
paths=sorted({report}|{p for root in (base,Path('artifacts/h1-v2/complete-resource')) for p in root.rglob('*') if p.is_file() and p!=index})
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
started=time.perf_counter_ns();files=[dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p)) for p in paths]
value=dict(scope='one completed full engineering R2 resource sample and proposed strict receipts; no genuine pilot',
    source_sha256='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c',
    count=len(files),bytes=sum(p['bytes'] for p in files),files=files)
index.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
errors=[p['path'] for p in files if Path(p['path']).stat().st_size!=p['bytes'] or digest(Path(p['path']))!=p['sha256']]
assert not errors,errors
print(json.dumps(dict(path=str(index),sha256=digest(index),count=value['count'],bytes=value['bytes'],verification_errors=errors,
    indexing_and_verification_seconds=(time.perf_counter_ns()-started)/1e9)))
