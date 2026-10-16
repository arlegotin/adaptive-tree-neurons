"""Curate only final-fix engineering evidence; verify historical files read-only."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from treeneuron.backend import implementation_hash, lock_hash
from treeneuron.config import canonical_hash, load_experiment
from treeneuron.train import source_hash


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    payload=json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n'
    with path.open('x') as stream:
        stream.write(payload)


root=Path(sys.argv[1])
study=read(root/'study.json')
assert study['provenance']['source_sha256']==source_hash()
assert study['status']=='completed' and len(study['methods'])==17
original=Path('docs/research/2026-10-16-exploratory-paired-evidence.json')
assert sha(original)=='ad07a1d24a9c19b184ce996c518374ee15428fc137dc9763dcaf69e1d3d1d854'
historical=[]
for name,expected,revision in [
    ('environment/task12-artifact-hashes.json','fab7d7e0c0c73b9394eefc0e9cc8ed2eeff7fe42a84b87d0c7dd5c96fe3a9437','d852661'),
    ('environment/task12-fix1-artifact-hashes.json','6fa3e000e65d7c0b571f36d1477050518bb61d57d17c22481053eae4257711a9','97e3414')]:
    path=Path(name)
    assert sha(path)==expected
    files=read(path)['files']
    live=0
    archived=0
    for filename,digest in files.items():
        if filename.startswith(('environment/','artifacts/','docs/')):
            assert sha(Path(filename))==digest,filename
            live+=1
        else:
            payload=subprocess.check_output(['git','show',f'{revision}:{filename}'])
            assert hashlib.sha256(payload).hexdigest()==digest,filename
            archived+=1
    historical.append(dict(path=name,sha256=expected,unchanged_live_files=live,source_files_verified_at_original_revision=archived,revision=revision))
files=set()
for path in Path('environment').glob('core-final-fix*'):
    if path.is_file(): files.add(path)
    else: files.update(p for p in path.rglob('*') if p.is_file() and p.name!='.ledger.lock')
files.update(p for p in root.rglob('*') if p.is_file() and p.name!='.ledger.lock')
for name in ['src/treeneuron/experiments/paired.py','src/treeneuron/analysis/exploratory.py',
    'src/treeneuron/train.py','src/treeneuron/baselines/causal.py','tests/test_paired.py',
    'tests/test_paired_diagnostics.py','configs/explore-core-final-fix-check.yaml']:
    files.add(Path(name))
# Shell redirection can precreate this script's stdout file; it is not evidence until exit.
files.discard(Path('environment/core-final-fix-curation-stdout.txt'))
index=Path('environment/core-final-fix-artifact-hashes.json')
assert index not in files
write(index,dict(schema_version=1,scope='Final-fix-only code/tests/config/logs/CPU correctness diagnostics and one tiny actual engineering study; excludes self-index and curator stdout',files={str(p):sha(p) for p in sorted(files)}))
config=Path('configs/explore-core-final-fix-check.yaml')
causal=read(Path('environment/core-final-fix-causal.json'))
assert causal['implemented_checks_passed'] and len(causal['recipes'])==13
verification={}
for path in sorted(Path('environment').glob('core-final-fix-*.txt')):
    if any(label in path.name for label in ('red','green','focused','pytest')):
        verification[path.name]=dict(path=str(path),sha256=sha(path),summary=path.read_text().strip().splitlines()[-1])
evidence=dict(schema_version=1,scope='Whole-core final review I1/I2/I3/M1/M2/M3 engineering corrections only; no new predictive inventory or efficacy claim',
    base_commit='f0eeca8938ae89279bd8eee44b236d575cc1a822',implementation_commit=study['provenance']['checkout'],
    source_sha256=source_hash(),lock_sha256=lock_hash(),backend_producer_sha256=implementation_hash(),
    verification=verification,causal=dict(path='environment/core-final-fix-causal.json',sha256=sha(Path('environment/core-final-fix-causal.json')),implemented_checks_passed=True,recipes=13),
    math_backend_scope='Mathematical/kernel/scoring/allocation producer files unchanged. Relevant suite tests and causal CLI passed; no repeated math CLI or backend calibration. Existing CPU receipt guards remained active.',
    historical_preservation=dict(indices=historical,original_public_evidence_sha256=sha(original),large_inventories_rerun=False,old_checkpoints_loaded=False),
    small_study=read(Path('environment/core-final-fix-small-audit.json')),
    small_study_summary=dict(path='environment/core-final-fix-small-summary.json',sha256=sha(Path('environment/core-final-fix-small-summary.json'))),
    small_config=dict(path=str(config),sha256=sha(config),resolved_sha256=canonical_hash(load_experiment(config).to_dict())),
    index=dict(path=str(index),sha256=sha(index),files=len(files)))
path=Path('docs/research/2026-10-16-core-final-fix-evidence.json')
write(path,evidence)
print(json.dumps(dict(status='passed',evidence=str(path),evidence_sha256=sha(path),index_files=len(files),index_sha256=sha(index),historical_checks=historical)))
