"""Build exact proposed strict receipts from the completed measured sample only.

No source imports, numerical jobs, inference approval or readiness validation.
Root must adjudicate proposed caps/support and validate before scientific work.
"""
import copy,hashlib,json,re,shutil,zipfile
from pathlib import Path

base=Path('environment/h1-v2/complete-resource');out=base/'proposed-readiness';out.mkdir(exist_ok=False)
m=json.loads((base/'measurement.json').read_text());s=json.loads((base/'complete-summary.json').read_text());inv=json.loads((base/'invocation.json').read_text())
assert m['status']=='completed' and inv['exit_code']==0 and s['teacher_count']==1 and len(s['repetitions'])==2
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def reference(path):return dict(path=str(path),sha256=sha(path))
def write(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

root=Path(m['attempt_root'])
# Sample retention covers final immutable producer files and all execution
# readouts/config/policy/host observations/timing/launcher; derived curation,
# proposed receipts/reports/index are separately retained administrative costs.
execution_names=['candidate.json','policy.json','measure.py','launch.py','run.stdout','run.stderr',
    'launcher.stdout','launcher.stderr','invocation.json','measurement.json','paired-timing.json','host-before.json','host-after.json']
sample_files=[p for p in root.rglob('*') if p.is_file()]+[base/name for name in execution_names]
sample_bytes=sum(path.stat().st_size for path in sample_files)
sample=dict(status='completed',teacher_count=1,enclosing_seconds=inv['outer_ns']/1e9,retained_bytes=sample_bytes,
    peak_memory_bytes=m['peak_rss_bytes'],recipe_sha256=m['recipe_sha256'],runtime_sha256=m['runtime_sha256'],source_sha256=m['source']['execution_sha256'])
sample_path=out/'sample.json';write(sample_path,sample)
write(out/'sample-files.json',dict(boundary='all final producer files plus named execution/readout files; excludes later curation/proposed receipts/reports/index',
    files=[dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)) for path in sorted(sample_files)],bytes=sample_bytes))

host=json.loads((base/'host-after.json').read_text())
pressure=next(c['stdout'] for c in host['commands'] if c['command']==['memory_pressure'])
physical=int(re.search(r'The system has (\d+)',pressure).group(1))
page_size=int(re.search(r'page size of (\d+)',pressure).group(1));free_pages=int(re.search(r'Pages free:\s+(\d+)',pressure).group(1))
free_page_bytes=free_pages*page_size
# Explicit chosen reserves/caps, not misrepresented as observations.
additional_memory_reserve=2*1024**3
available_memory=free_page_bytes-additional_memory_reserve
assert available_memory>sample['peak_memory_bytes']
reserved_memory=physical-available_memory
free_disk=host['disk']['free'];disk_reserve=10*1024**3;storage_cap=free_disk-disk_reserve
assert storage_cap>0
pilot_cap=14400.0;confirmation_cap=28800.0
measurement=dict(measured_seconds_per_teacher=sample['enclosing_seconds'],measured_bytes_per_teacher=sample_bytes,
    peak_memory_bytes=sample['peak_memory_bytes'],machine_memory_bytes=physical,reserved_memory_bytes=reserved_memory,
    free_disk_bytes=free_disk,max_seconds=pilot_cap,max_storage_bytes=storage_cap,retries=0,
    failure_policy='retain_and_block',source_sha256=sample['source_sha256'],samples=[reference(sample_path)])
measurement_path=out/'measurement.json';write(measurement_path,measurement)

# Nonoverlapping resource model at the measured E: setup residual, complete fit
# repetitions, then E+ledger costs/row. Conservative max per-repetition fit.
fit_clocks=[f['enclosing_ns']/1e9 for f in m['fit_readouts']]
fit_max=max(fit_clocks)
run_events=[event for event in m['clock_events'] if event['function']=='run_repetition']
fit_events=[event for event in m['clock_events'] if event['function']=='fit_repetition']
E_seconds=sum(a['elapsed_ns']-b['elapsed_ns'] for a,b in zip(run_events,fit_events))/1e9
ledger_seconds=sum(p['elapsed_ns'] for p in m['ledger_publications'] if Path(p['path']).name=='raw.json')/1e9
E_charge=E_seconds
fixed_seconds=max(1e-9,sample['enclosing_seconds']-sum(fit_clocks)-E_charge)
E_rows=m['config']['pilot']['evaluation_rows'];R=2
raw=json.loads((root/'raw.json').read_text())
def row_array_bytes(record):return sum(len(canonical(record[name])) for name in ('h1_rows','static_rows'))
E_bytes=sum(p.stat().st_size for p in root.rglob('E-predictions.npz'))
E_bytes+=sum(row_array_bytes(a['record']) for a in raw['attempts'] if 'record' in a)+sum(row_array_bytes(r) for r in raw['accepted_records'])
fit_bytes=[]
for repetition,record in enumerate(raw['accepted_records']):
    directory=root/'teacher-0'/f'repetition-{repetition}'
    E_bytes+=row_array_bytes(record)
    fit_bytes.append(sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())-
        (directory/'E-predictions.npz').stat().st_size-row_array_bytes(record))
with zipfile.ZipFile(root/'teacher-0/observations.npz') as archive:
    E_bytes+=sum(info.compress_size for info in archive.infolist() if info.filename.startswith('E_'))
fixed_bytes=sample_bytes-sum(fit_bytes)-E_bytes
assert fixed_bytes>0
constraints=dict(max_seconds=confirmation_cap,max_storage_bytes=storage_cap,available_memory_bytes=available_memory,
    peak_memory_bytes=sample['peak_memory_bytes'],setup_seconds_per_teacher=fixed_seconds,
    seconds_per_repetition=fit_max,storage_bytes_per_teacher=fixed_bytes,storage_bytes_per_repetition=max(fit_bytes),
    seconds_per_evaluation_row=E_charge/(R*E_rows),bytes_per_evaluation_row=E_bytes/(R*E_rows),
    retries=0,estimation_rows=E_rows,alpha=.05,families=1,inference_supported=False,
    power_iterations=16384,bootstrap_seed=261022106,endpoint_tolerance=m['config']['pilot']['endpoint_tolerance'],
    power_mc_tolerance=.01,required_controls=['B1','B2','B3','B4','B5','B6'],
    evidence_hashes=dict(resource_sample=sha(sample_path),finite_precision=sha(Path('environment/h1-v2/precision/monte-carlo.json')),
        complete_summary=sha(base/'complete-summary.json')),
    policies=dict(parent_updates=8000,continuation_updates=8000,patience=20,validation_interval=25,
        logical_batch=128,physical_batch=128,bootstrap_iterations=8000,probe_updates=320,
        hpo_seconds=0.0,hpo_trials=0,timing_blocks=27,timing_relative_precision=.05))
constraints_path=out/'constraints.json';write(constraints_path,constraints)
config=copy.deepcopy(m['config']);config['study_id']='h1-v2-genuine-pilot'
config['execution']['artifacts_root']='artifacts/h1-v2/scientific-pilot'
config['seeds'].update(teacher=261022100,data=261022101,optimizer=261022102,allocation=261022103,contrast=261022104,reference=261022105)
config['pilot'].update(purpose='scientific',initial_teachers=8,max_teachers=16,bootstrap_seed=261022106,
    max_seconds=pilot_cap,constraints_path=str(constraints_path),readiness_path=str(out/'readiness.json'))
write(out/'scientific-config.json',config)
reviews=Path('audit/implementation/evidence/2026-10-17')
ready=dict(schema_version=1,pinned_commit='fc63de6884ba5d7a911e0228d2c6707c3ce90a11',execution_sha256=sample['source_sha256'],
    reviews={task:reference(reviews/name) for task,name in [('9','task-9-acceptance-v5.json'),('10','task-10-acceptance-v4.json'),('13','task-13-acceptance-v3.json')]},
    config_sha256=hashlib.sha256(canonical(config)).hexdigest(),runtime=reference(Path(config['execution']['backend_report'])),
    measurement=reference(measurement_path),constraints=reference(constraints_path),
    seed_plan=dict(namespace='exploratory/pilot',seeds=config['seeds'],max_teachers=16,repetitions=2))
write(out/'readiness.json',ready)
write(out/'proposal-not-approval.json',dict(status='strict proposed receipts only; not readiness validation or execution authorization',
    observed=dict(physical_memory_bytes=physical,free_pages=free_pages,page_bytes=page_size,free_page_bytes=free_page_bytes,free_disk_bytes=free_disk),
    chosen=dict(extra_memory_reserve_bytes=additional_memory_reserve,disk_reserve_bytes=disk_reserve,pilot_max_seconds=pilot_cap,
        confirmation_max_seconds=confirmation_cap,retries=0,inference_supported=False),
    support_scope='false pending independent-teacher inference assessment; conditional E-block MC does not establish population coverage',
    forecast16=dict(seconds=16*sample['enclosing_seconds'],bytes=16*sample_bytes),
    forecast32=dict(seconds=32*sample['enclosing_seconds'],bytes=32*sample_bytes),
    cost_decomposition=dict(fit_seconds=fit_clocks,E_seconds=E_seconds,raw_ledger_publication_seconds=ledger_seconds,
        teacher_fixed_residual_seconds=fixed_seconds,fit_bytes=fit_bytes,E_plus_ledgers_bytes=E_bytes,teacher_fixed_residual_bytes=fixed_bytes),
    limitations=['single engineering teacher not a teacher-tail/resource guarantee',
        'raw.json cumulative rewrites grow faster than linear across an expanded cohort; one-teacher factor not guaranteed at16/32teachers',
        'JSON/Python row-ledger resident memory grows with teachers; peak RSS from n1 does not establish n16peak',
        'all raw-ledger publication time is assigned fixed teacher overhead at the measured E; its nonlinear larger-cohort/row scaling is not guaranteed by this one-point cost model',
        'E262144 is bounded variance-pilot workload; unresolved engineering row-SE target, final nT derived from genuine independentteacher variance',
        'all above caps/support/reserves are explicit proposed choices for root adjudication, not measured outputs',
        'no scientificpilot/readiness-validation/freeze/T run by this script']))
print('strict proposed receipts written; sample seconds',sample['enclosing_seconds'],'bytes',sample_bytes,'16teacher forecast',16*sample['enclosing_seconds'],16*sample_bytes)
