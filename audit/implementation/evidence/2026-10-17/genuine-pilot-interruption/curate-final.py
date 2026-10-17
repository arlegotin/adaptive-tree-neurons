"""Terminal stdlib evidence curation, no source imports or extra experiment."""
import array,ast,hashlib,json,shutil,time,zipfile
from datetime import datetime,timezone
from pathlib import Path

base=Path('environment/h1-v2/genuine-pilot');readout=json.loads((base/'readout.json').read_text())
assert readout['status'] in ('completed','failed','interrupted')
assert len(readout['attempt_roots'])==1
root=Path(readout['attempt_roots'][0]);raw=json.loads((root/'raw.json').read_text())
invocation=json.loads((base/'invocation.json').read_text())
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def output(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def floats(archive,name):
    content=archive.read(name+'.npy');assert content[:6]==b'\x93NUMPY'
    width=2 if content[6]==1 else 4;length=int.from_bytes(content[8:8+width],'little')
    header=ast.literal_eval(content[8+width:8+width+length].decode('latin1').strip())
    assert header['descr']=='<f8' and len(header['shape'])==1 and not header['fortran_order']
    values=array.array('d');values.frombytes(content[8+width+length:]);assert len(values)==header['shape'][0]
    return values
def stage(directory,max_updates):
    evaluations=[];selection=[];updates=examples=0
    with (directory/'events.jsonl').open() as stream:
        for line in stream:
            event=json.loads(line)
            if event['type']=='update':updates+=1;examples+=event['rows']
            elif event['type']=='evaluation':evaluations.append(event)
            elif event['type']=='selection':selection.append(event)
    bad=0
    for e in reversed(evaluations):
        if e['improved']:break
        bad+=1
    return dict(path=str(directory),selected_update=selection[-1]['selected_update'] if selection else None,
        executed_updates=updates,examples=examples,max_updates=max_updates,hit_ceiling=updates==max_updates,
        improving_at_ceiling=updates==max_updates and bool(evaluations[-1]['improved']) if evaluations else None,
        final_consecutive_nonimprovements=bad,
        G_curve=[dict(stage_update=e['stage_update'],update=e['update'],normalized_mse=e['metrics']['normalized_mse'],improved=e['improved']) for e in evaluations])
def category(path):
    if path.name=='raw.json':return 'raw_ledger'
    if path.name=='pair.json':return 'pair_ledgers'
    if path.name=='events.jsonl':return 'raw_events'
    if path.name.endswith('.meta.json'):return 'checkpoint_metadata'
    if path.suffix=='.pt':return 'tensor_payloads'
    if path.name=='E-predictions.npz':return 'E_predictions'
    if path.suffix=='.npz':return 'other_arrays'
    return 'other'
bytes_by_category={}
for path in root.rglob('*'):
    if path.is_file():bytes_by_category[category(path)]=bytes_by_category.get(category(path),0)+path.stat().st_size
result=dict(scope='actual independent-teacher variance pilot; no protected test/cohort/freeze',attempt_root=str(root),
    readout_sha256=sha(base/'readout.json'),invocation_sha256=sha(base/'invocation.json'),raw_sha256=sha(root/'raw.json'),
    source=readout['source'],recipe_sha256=readout['recipe_sha256'],config_sha256=readout['resolved_config_sha256'],
    runtime_sha256=readout['receipt_sha256'],readiness_sha256=readout['readiness_sha256'],constraints_sha256=readout['constraints_sha256'],
    status=raw['status'],failure=raw.get('failure'),failed_pairs=raw['failed_pairs'],
    perf_clock_external_seconds=invocation['external_seconds'],perf_clock_source_attempt_seconds=raw['enclosing_ns']/1e9,
    clock_scope='observed UTC/perf gaps; perf-clock execution is not unconditional calendar walltime; existing perf-clock cap unchanged',
    peak_rss_bytes=readout['peak_rss_bytes'],final_producer_bytes=sum(bytes_by_category.values()),byte_categories=bytes_by_category,
    accepted_records=len(raw['accepted_records']),completed_attempts=sum(a['status']=='completed' for a in raw['attempts']),
    attempts=[],teachers=[],raw_calibration_status=raw.get('calibration_status'),
    runtime_batch_seconds=raw['batch_runtimes'],batch_completed_counts=raw['batch_completed_counts'],
    raw_publications=readout['raw_publications'],resource_observations=readout['resource_observations'],
    resource_limited=readout['resource_limited'],numerical_STOP=True)
def utc(value):return datetime.strptime(value,'%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
obs=readout['resource_observations']
result['clock_observations']=dict(initial_utc=obs[0]['utc'],final_utc=obs[-1]['utc'],
    observed_utc_seconds=(utc(obs[-1]['utc'])-utc(obs[0]['utc'])).total_seconds(),
    corresponding_perf_seconds=obs[-1]['elapsed_seconds']-obs[0]['elapsed_seconds'],
    limitation='integer-second native observation endpoints; this interval excludes pre-initial imports and is not an exact full-process UTC timestamp pair',paired=[])
sleep_root=Path('environment/h1-v2/sleep-observation')
result['clock_observations']['root_sleep_evidence']=[dict(path=str(path),sha256=sha(path)) for path in sorted(sleep_root.rglob('*')) if path.is_file()]
for pair in readout['paired']:
    end_index=pair['resource_observation_index'];before=obs[end_index-1];after=obs[end_index]
    calendar=(utc(after['utc'])-utc(before['utc'])).total_seconds();perf=after['elapsed_seconds']-before['elapsed_seconds']
    result['clock_observations']['paired'].append(dict(path=pair['path'],before_utc=before['utc'],after_utc=after['utc'],
        observed_utc_seconds=calendar,corresponding_perf_seconds=perf,utc_minus_perf_seconds=calendar-perf,
        original_pair_perf_seconds=pair['wrapper_seconds']))
groups={};probe_count=0;roles=0
for entry in raw['attempts']:
    a=dict(teacher_index=entry['teacher_index'],repetition=entry['repetition'],status=entry['status'],enclosing_ns=entry.get('enclosing_ns'),failure=entry.get('failure'))
    directory=root/f"teacher-{entry['teacher_index']}"/f"repetition-{entry['repetition']}"
    if entry['status']=='completed':
        record=entry['record'];assert record['status']=='completed'
        assert set(record['methods'])=={'keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
        assert set(record['controls'])=={'B1','B2','B3','B4','B5','B6'}
        a.update(teacher_id=record['teacher_id'],optimizer_seed=record['optimizer_seed'],data_hash=record['data_hash'],
            evaluation_hash=record['evaluation_hash'],h1_mean=sum(record['h1_rows'])/len(record['h1_rows']),
            static_mean=sum(record['static_rows'])/len(record['static_rows']),controls=record['controls'],pairing_audit=record['pairing_audit'],
            E_rows=len(record['h1_rows']),parent=stage(directory/'runs'/record['parent_run_id'],8000),methods={})
        descriptor=directory/'runs'/record['parent_run_id']/'checkpoints/best.json';pointer=json.loads(descriptor.read_text())
        metadata=json.loads((descriptor.parent/pointer['metadata']).read_text());variance=metadata['state']['fit_target_variance']
        with zipfile.ZipFile(directory/'E-predictions.npz') as archive:
            y=floats(archive,'y')
            a['normalized_E_risks']={role:sum((p-target)**2 for p,target in zip(floats(archive,role),y))/len(y)/variance for role in record['methods']}
        assert abs(a['normalized_E_risks']['random']-a['normalized_E_risks']['top']-a['h1_mean'])<1e-12
        assert abs(a['normalized_E_risks']['ordinary_upper']-a['normalized_E_risks']['static_partial']-a['static_mean'])<1e-12
        for role,method in record['methods'].items():
            a['methods'][role]=dict(total_parameters=method['total_parameters'],selected_g_mse=method['selected_g_mse'],
                enclosing_seconds=method['enclosing_ns']/1e9,stages=[stage(directory/s['path'],s['max_updates']) for s in method['stages']])
        roles+=len(a['methods'])
        p=Path(record['greedy_report']);g=json.loads(p.read_text());assert g['complete'] and not g['not_attempted']
        assert len(g['trials'])==1+len(g['eligible_ids']) and all(t['outcome']=='completed' and t['executed_updates']==320 for t in g['trials'])
        a['greedy']=dict(path=str(p),sha256=sha(p),eligible_ids=g['eligible_ids'],probe_count=len(g['trials']),
            enclosing_seconds=g['nanoseconds']/1e9,trials=[{k:t[k] for k in ('candidate','outcome','selected_update','executed_updates','g_mse','nanoseconds','path')} for t in g['trials']])
        probe_count+=len(g['trials']);groups.setdefault(entry['teacher_index'],[]).append(a)
    result['attempts'].append(a)
for index,rows in sorted(groups.items()):
    result['teachers'].append(dict(index=index,teacher_id=rows[0]['teacher_id'],completed_repetitions=len(rows),
        h1_mean=sum(r['h1_mean'] for r in rows)/len(rows),static_mean=sum(r['static_mean'] for r in rows)/len(rows),
        repetitions=[dict(repetition=r['repetition'],optimizer_seed=r['optimizer_seed'],h1_mean=r['h1_mean'],static_mean=r['static_mean']) for r in rows]))
result['coverage']=dict(completed_teachers_with_R2=sum(t['completed_repetitions']==2 for t in result['teachers']),
    role_records=roles,actual_B6_probes=probe_count)
for name in ('summary.json','design_decision.json','resource_forecast.json'):
    if (root/name).exists():result[name.removesuffix('.json')]=json.loads((root/name).read_text())
if 'summary' in result:
    summary=result['summary'];diagnostics=summary['diagnostics']
    order=list(dict.fromkeys(record['teacher_id'] for record in raw['accepted_records']))
    inputs=[]
    for position,teacher_id in enumerate(order):
        teacher=next(t for t in result['teachers'] if t['teacher_id']==teacher_id)
        assert abs(teacher['h1_mean']-summary['teacher_means'][position])<1e-12
        rows=next(a['E_rows'] for a in result['attempts'] if a.get('teacher_id')==teacher_id and a['status']=='completed')
        static_values=[p['static_mean'] for p in teacher['repetitions']]
        inputs.append(dict(teacher_index=teacher['index'],teacher_id=teacher_id,
            h1_mean=summary['teacher_means'][position],static_mean=teacher['static_mean'],
            exact_pair_means=teacher['repetitions'],repetitions=teacher['completed_repetitions'],evaluation_rows=rows,
            within_teacher_H1_variance=diagnostics['within_teacher_variances'][position],
            paired_row_H1_variance=diagnostics['paired_row_variances'][position],
            within_teacher_static_mean_variance=sum((x-teacher['static_mean'])**2 for x in static_values)/(len(static_values)-1)))
    result['saved_scalar_reanalysis_inputs']=dict(scope='source-summary-aligned actual independent-teacher scalar inputs; no MC refinement performed',
        summary_sha256=sha(root/'summary.json'),raw_sha256=result['raw_sha256'],
        original_bootstrap_iterations=8000,original_bootstrap_seed=261022106,
        original_endpoint_gate=raw['config']['pilot']['endpoint_tolerance'],teachers=inputs)
result['final_disk']=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(zip(('total','used','free'),shutil.disk_usage('.'))))
output(base/'terminal-summary.json',result)
print(json.dumps(dict(status=result['status'],coverage=result['coverage'],bytes=result['final_producer_bytes'],perf_clock_external_seconds=result['perf_clock_external_seconds'],failed_pairs=result['failed_pairs'])))
