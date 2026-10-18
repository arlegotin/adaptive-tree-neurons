"""Inventory complete stopped audit outputs, including failures and postcheck logs."""
import hashlib,json
from pathlib import Path
ROOT=Path.cwd();ENV=ROOT/'environment/long-probes-20261018/audit';OUT=ROOT/'artifacts/long-probes-20261018/audit'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1048576),b''):h.update(block)
 return h.hexdigest()
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
assert json.loads((OUT/'preservation.json').read_text())['status']=='completed'
paths=sorted(p for base in (ENV,OUT) for p in base.rglob('*') if p.is_file() and p not in (ENV/'inventory.json',ENV/'inventory-seal.json'))
paths.extend([ROOT/'artifacts/long-probes-20261018/cohort/analysis.json',ROOT/'.superpowers/sdd/2026-10-18-long-probe-allocation-diagnostic/task-3-report.md'])
files=[{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'bytes':p.stat().st_size} for p in paths]
value={'schema_version':1,'purpose':'Complete stopped Task3 audit/replay/restore/reconciliation/twice-render evidence, including both checker failures and all logs; no new efficacy units','artifacts':files,'artifact_count':len(files),'total_logical_bytes':sum(e['bytes'] for e in files),'excludes':['this inventory and seal to avoid a self-hash cycle'],'automatic_numerical_retries':0,'manual_checker_corrections':2}
write(ENV/'inventory.json',value);write(ENV/'inventory-seal.json',{'sha256':sha(ENV/'inventory.json')})
print(json.dumps({'artifacts':len(files),'bytes':value['total_logical_bytes'],'inventory_sha256':sha(ENV/'inventory.json')}))
