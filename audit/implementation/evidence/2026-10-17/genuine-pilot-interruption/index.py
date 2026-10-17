"""Terminal complete inventory, including retained failures and all raw files."""
import hashlib,json,time
from pathlib import Path
base=Path('environment/h1-v2/genuine-pilot');index=base/'inventory.json';assert not index.exists()
r=json.loads((base/'readout.json').read_text());assert r['status'] in ('completed','failed','interrupted')
roots=[base,Path('environment/h1-v2/sleep-observation')]+[Path(path) for path in r['attempt_roots']]
report=Path('.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-genuine-pilot-report.md')
root_checks={Path('environment/h1-v2/user-resume-root')/name for name in ('original-seal-verification.json','terminal-semantic-verification.json')}
paths=sorted({report}|root_checks|{p for root in roots for p in root.rglob('*') if p.is_file() and p!=index})
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
tick=time.perf_counter_ns();files=[dict(path=str(path),bytes=path.stat().st_size,sha256=digest(path)) for path in paths]
value=dict(scope='new actual genuine pilot evidence only, all attempts/controls/probes/failures; no test/cohort/freeze',
    source_sha256=r['source']['execution_sha256'],recipe_sha256=r['recipe_sha256'],count=len(files),
    bytes=sum(f['bytes'] for f in files),files=files)
index.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
errors=[f['path'] for f in files if Path(f['path']).stat().st_size!=f['bytes'] or digest(Path(f['path']))!=f['sha256']]
assert not errors,errors
print(json.dumps(dict(path=str(index),sha256=digest(index),count=value['count'],bytes=value['bytes'],verification_errors=errors,
    indexing_verification_seconds=(time.perf_counter_ns()-tick)/1e9)))
