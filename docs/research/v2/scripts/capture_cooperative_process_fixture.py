"""Reopen actual tiny-worker fixtures without new processes, labels or models."""
import hashlib
import json
from pathlib import Path
import sys

from treeneuron.research import cooperative_process_v2
from treeneuron.research.deployment_worker_v2 import _source

fixture, output = map(Path, sys.argv[1:])
def forbidden(*args, **kwargs):
    raise AssertionError('Read-only process inspection must not spawn or probe')
cooperative_process_v2.subprocess.Popen = forbidden
rows = []
for path in sorted(fixture.rglob('process/process.json')):
    receipt = json.loads(path.read_text())
    if receipt.get('schema') != 'research/v2-cooperative-process-receipt':
        continue
    root = path.parent
    proof = cooperative_process_v2.inspect_cooperative_process(root=root,
        expected_receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_started_sha256=hashlib.sha256((root/'started.json').read_bytes()).hexdigest())
    rows.append(dict(root=str(root), receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        proof=proof))
assert len(rows) == 13
value = dict(schema='research/v2-cooperative-process-fixture-observation',
    scientific_source_sha256=_source(), actual_closed_root_episodes=len(rows),
    new_processes_for_capture=0, labels_read=False, native_models_fit=False,
    protected_access_authorized=False, native_workload_qualified=False,
    quality_eligible=False, production_execution_authorized=False, rows=rows)
with output.open('x') as stream:
    json.dump(value, stream, sort_keys=True, indent=2)
    stream.write('\n')
print(json.dumps(dict(rows=len(rows),scientific_source_sha256=value['scientific_source_sha256'],
    capture_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),bytes=output.stat().st_size)))
