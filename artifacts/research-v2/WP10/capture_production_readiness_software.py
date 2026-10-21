"""Actual fixed software profiles; no native qualification/readiness/T claim."""
from pathlib import Path
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[3]
from treeneuron.research.deployment_worker_v2 import _source
from treeneuron.research.production_verification_v2 import load_production_verification,production_verification_profiles
from treeneuron.research.workqueue import _inventory
from treeneuron.config import canonical_hash

source=_source();assert source=='eeeeb85725e9da424f35a25af96a8837ad8c555eb8b557abb10143f5e2097962'
records={};expected=dict(search=257,reporting=138,pipeline=159,protection=197)
for profile,count in expected.items():
    root=ROOT/'artifacts/research-v2/WP10'/('production-'+profile+'-eeeeb85')
    reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root)))
    proof=load_production_verification(reference,expected_profile=profile)
    result=json.loads((root/'result.json').read_bytes());process=result['process']
    assert proof['test_cases']==count and proof['identity']['scientific_source_sha256']==source
    assert proof['software_verified'] is True and proof['native_workload_qualified'] is proof['T_accessed'] is False
    records[profile]=dict(reference=reference,test_cases=count,files=production_verification_profiles()[profile],
        identity_sha256=canonical_hash(proof['identity']),process_elapsed_ns=process['elapsed_ns'],
        observed_peak_root_rss_bytes=process['observed_peak_root_rss_bytes'],
        root_only_memory_scope=process['rss_scope'],junit_sha256=hashlib.sha256((root/'worker/junit.xml').read_bytes()).hexdigest())
all_files=[name for row in records.values() for name in row['files']]
assert len(all_files)==44 and len(set(all_files))==44 and sum(r['test_cases'] for r in records.values())==751
base=ROOT/'src/treeneuron';previous_program=(ROOT/'artifacts/research-v2/WP10/readiness-pre-legacy-clock-program_process.py').read_bytes()
previous_cli=(ROOT/'artifacts/research-v2/WP10/readiness-pre-console-cli.py').read_bytes()
def reconstruct(previous_console):
    digest=hashlib.sha256()
    for p in sorted(base.rglob('*.py')):
        raw=p.read_bytes()
        if p==base/'research/program_process_v2.py':raw=previous_program
        if previous_console and p==base/'research/cli.py':raw=previous_cli
        digest.update(p.relative_to(base).as_posix().encode());digest.update(raw)
    return digest.hexdigest()
assert reconstruct(False)=='a60f50740c1d1d61770757cccb2c0cec50a84c7044429f19b80e163c055e5bd3'
assert reconstruct(True)=='e4028dd0c20aeec2c6dbea2b2caa1f079b2915265485e6834a37b3dfeb2765f3'
assert _source()==source
output=ROOT/'artifacts/research-v2/WP10/production-readiness-software';output.mkdir(parents=True,exist_ok=False)
value=dict(schema='research/v2-production-readiness-software-observation',source_sha256=source,profiles=records,
    passing_cases=751,fixed_test_files=44,all_actual_profile_loaders_reopened=True,previous_sources_reconstructed=True,
    actual_supervised_software_positive=True,actual_native_readiness_positive=False,
    fresh_final_branch_review_complete=False,production_source_ready=False,
    actual_benchmark_models_fit_or_predicted=0,actual_benchmark_labels_read=0,
    known_fixture_models_and_sentinel_arrays_used=True,T_accessed=False,protected_access_authorized=False,
    whole_repository_suite_passed=False,formal_efficiency_claim=False,
    scope='Four actual immutable current-source software profiles only; native qualification, branch review, readiness seal and practical benchmark remain separate')
path=output/'observation.json';path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),passing_cases=751),sort_keys=True))
