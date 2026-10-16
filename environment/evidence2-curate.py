"""Retain bounded Evidence Task2 engineering receipts; no bulk assets committed."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
from treeneuron.backend import implementation_hash, lock_hash
from treeneuron.train import source_hash

root = Path(__file__).resolve().parents[1]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')

causal_path=root/'environment/evidence2-causal.json'
practical_path=root/'environment/evidence2-practical.json'
causal=json.loads(causal_path.read_text())
practical=json.loads(practical_path.read_text())
assert causal['implemented_checks_passed'] and not causal['track_ready']
assert causal['capabilities']['B7']['status']=='pending'
assert not practical['implemented_checks_passed']
assert causal['source_hash']==source_hash()
receipt=root/'environment/task11-parent-backend.json'
assert digest(receipt)=='44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197'
assert implementation_hash()=='237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e'
artifact_root=Path(causal['artifacts_root'])
inputs=set(p for p in artifact_root.rglob('*') if p.is_file())
inputs.update((root/'src/treeneuron').rglob('*.py'))
inputs.update(root/'tests'/name for name in ('test_greedy.py','test_adapters.py','test_causal_baselines.py'))
# Curation stdout describes these output hashes and cannot index itself.
inputs.update(p for p in (root/'environment').glob('evidence2-*.txt') if 'curation' not in p.name)
inputs.update((causal_path,practical_path,receipt,root/'uv.lock',Path(__file__)))
records=[dict(path=str(p.relative_to(root)),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(inputs)]
index_path=root/'environment/evidence2-artifact-hashes.json'
write(index_path,dict(schema_version=1,scope='Evidence Task2 fresh source engineering fixtures and exact verification outputs',files=records))
probes=causal['greedy_probes']
for trial in probes['trials']:
    child=Path(probes['artifact_directory'])/trial['path']
    manifest=json.loads((child/'manifest.json').read_text())
    assert manifest['identity']['status']=='completed'
    assert manifest['identity']['score_hash']==probes['score_hash']
    assert manifest['identity']['coordinator_run_id']==Path(probes['artifact_directory']).name
    assert trial['executed_updates']==2 and trial['nanoseconds']>0
assert len(probes['trials'])==3 and probes['complete']
curated=dict(schema_version=1,date='2026-10-16',task='Evidence Task2',
    status='B0–B6 implementation verified; independent review pending; B7 gate pending Task3',
    scope='CPU engineering correctness only; no predictive efficacy, throughput, full ANT reproduction or full causal readiness claim',
    checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
    package_source_sha256=source_hash(),lock_sha256=lock_hash(),backend_producer_sha256=implementation_hash(),
    backend_receipt=dict(path=str(receipt.relative_to(root)),sha256=digest(receipt)),
    verification=dict(focused='189 passed, 1 skipped in 35.01s',full_suite=(root/'environment/evidence2-final-pytest.txt').read_text().strip().splitlines()[-1],
        causal=dict(exit_status=0,path=str(causal_path.relative_to(root)),sha256=digest(causal_path),
                    neural_recipes=len(causal['recipes']),baseline_families='B0–B6',track_ready=False,B7='pending Task3'),
        practical=dict(exit_status=1,path=str(practical_path.relative_to(root)),sha256=digest(practical_path),
                       reason='Required B8/B9/B10/B11/B14 pending; B12/B13 optional unavailable')),
    fixture=dict(backend='CPU eager float32; CPU64 scores',cpu_threads=1,n_fit=128,n_growth=32,
        logical_batch_size=128,physical_batch_size=128,greedy_d=16,greedy_h=2,base_updates=2,
        probe_updates=2,joint_continuation_updates=2,requested_count=1,effective_count=probes['allocation']['effective_count'],
        knots='F-only per-feature quantiles; additive cubic truncated-power basis; externally selected alpha'),
    greedy=dict(report_path=str((Path(probes['artifact_directory'])/'greedy.json').relative_to(root)),
        complete=probes['complete'],eligible_ids=probes['eligible_ids'],allocation=probes['allocation'],
        nanoseconds=probes['nanoseconds'],cost_boundary=probes['cost_boundary'],
        trials=[{k:t[k] for k in ('candidate','outcome','nanoseconds','selected_update','executed_updates','g_mse','run_id')} for t in probes['trials']],
        no_V_or_T_access=True,joint_recipe=causal['recipes'][-1]),
    artifact_index=dict(path=str(index_path.relative_to(root)),sha256=digest(index_path),files=len(records)),
    limits=['Engineering fixture is not a scientific pilot, independent cohort, resource calibration or throughput benchmark.',
        'Probe cost nests child setup/training/checkpoint costs; do not sum nested durations. Prior parent fitting/shared scoring and final report publication are excluded explicitly.',
        'Joint continuation retains the separate adapter-fit timing boundary; future search runner must own complete enclosing execution cost.',
        'Failures abort exhaustive selection and retain not-attempted candidates. Patience-shortened probes reject instead of changing the shared policy.',
        'Spline supports numerical columns and F-fitted missing indicators; categorical inputs reject. V-search orchestration is future Task7.',
        'No backend-producer files or locks changed; historical receipts/checkpoints/evidence were neither rewritten nor loaded across source identities.'],
    next_gate='Independent Task2 spec/quality review, then faithful pinned NODE Task3; B7 remains required before full WP03/causal gate closure')
write(root/'docs/research/2026-10-16-evidence2-capabilities.json',curated)
print(json.dumps({'files':len(records),'index_sha256':digest(index_path),'curated_sha256':digest(root/'docs/research/2026-10-16-evidence2-capabilities.json')},sort_keys=True))
