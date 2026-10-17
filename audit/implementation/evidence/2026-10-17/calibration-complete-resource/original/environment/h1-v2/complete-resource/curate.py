"""Stdlib postrun curation; preserve every original producer record unchanged."""
import array,ast,hashlib,json,shutil,struct,time,zipfile
from pathlib import Path

base=Path('environment/h1-v2/complete-resource')
measurement=json.loads((base/'measurement.json').read_text());assert measurement['status']=='completed'
invocation=json.loads((base/'invocation.json').read_text());assert invocation['exit_code']==0
root=Path(measurement['attempt_root']);raw=json.loads((root/'raw.json').read_text())
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def write(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def float_array(archive,name):
    content=archive.read(name+'.npy');assert content[:6]==b'\x93NUMPY'
    width=2 if content[6]==1 else 4;length=int.from_bytes(content[8:8+width],'little')
    header=ast.literal_eval(content[8+width:8+width+length].decode('latin1').strip())
    assert header['descr']=='<f8' and len(header['shape'])==1 and not header['fortran_order']
    values=array.array('d');values.frombytes(content[8+width+length:]);assert len(values)==header['shape'][0]
    return values
def category(path):
    if path.name=='raw.json':return 'raw_ledger'
    if path.name=='pair.json':return 'pair_ledgers'
    if path.name=='events.jsonl':return 'raw_events'
    if path.name.endswith('.meta.json'):return 'checkpoint_metadata'
    if path.suffix=='.pt':return 'tensor_payloads'
    if path.name=='E-predictions.npz':return 'E_predictions'
    if path.suffix=='.npz':return 'other_arrays'
    return 'other'
def bytes_by_category(directory):
    out={}
    for path in directory.rglob('*'):
        if path.is_file():out[category(path)]=out.get(category(path),0)+path.stat().st_size
    return dict(categories=out,total=sum(out.values()))
def stage(path):
    events=[json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    updates=[e for e in events if e['type']=='update'];evaluations=[e for e in events if e['type']=='evaluation'];selection=[e for e in events if e['type']=='selection']
    policy=json.loads((path/'resolved.json').read_text()) if (path/'resolved.json').exists() else None
    # The declared stage allowance is independently retained by the readout.
    obs=next(s for s in measurement['stage_readouts'] if Path(s['path'])==path)
    bad=0
    for event in reversed(evaluations):
        if event['improved']:break
        bad+=1
    return dict(path=str(path),events_sha256=hash_file(path/'events.jsonl'),selected_update=selection[-1]['selected_update'],
        executed_updates=len(updates),examples=sum(e['rows'] for e in updates),max_updates=obs['max_updates'],
        hit_ceiling=len(updates)==obs['max_updates'],improving_at_ceiling=len(updates)==obs['max_updates'] and evaluations[-1]['improved'],
        final_consecutive_nonimprovements=bad,normalized_selected_G=obs['normalized_G'],
        G_curve=[dict(stage_update=e['stage_update'],update=e['update'],normalized_mse=e['metrics']['normalized_mse'],improved=e['improved']) for e in evaluations],
        raw_row_id_commitments=[hashlib.sha256(struct.pack('<'+'q'*len(e['row_ids']),*e['row_ids'])).hexdigest() for e in updates],
        bytes=bytes_by_category(path))

report=dict(scope='one actual complete engineering teacher R2, never genuine variance pilot/coverage',
    original_measurement_sha256=hash_file(base/'measurement.json'),invocation_sha256=hash_file(base/'invocation.json'),
    attempt_root=str(root),raw_sha256=hash_file(root/'raw.json'),recipe_sha256=measurement['recipe_sha256'],
    source_sha256=measurement['source']['execution_sha256'],runtime_sha256=measurement['runtime_sha256'],
    status=raw['status'],teacher_count=1,repetitions=[],failed_pairs=raw['failed_pairs'],
    external_seconds=invocation['outer_ns']/1e9,peak_rss_bytes=measurement['peak_rss_bytes'],
    final_attempt_bytes=bytes_by_category(root),raw_ledger_duplication=measurement['raw_ledger_duplication'],
    tagged_clocks=measurement['tagged_clocks'],ledger_publications=measurement['ledger_publications'],
    raw_calibration_status=raw['calibration_status'],numerical_STOP=True)
assert len(raw['accepted_records'])==2 and raw['failed_pairs']==0
for repetition,record in enumerate(raw['accepted_records']):
    directory=root/'teacher-0'/f'repetition-{repetition}'
    assert set(record['methods'])=={'keep','top','random','greedy','widen_upper','ordinary_lower','ordinary_upper','static_partial'}
    rep=dict(repetition=repetition,teacher_id=record['teacher_id'],optimizer_seed=record['optimizer_seed'],
        parent=stage(directory/'runs'/record['parent_run_id']),methods={},pairing_audit=record['pairing_audit'],
        h1_mean=sum(record['h1_rows'])/len(record['h1_rows']),static_mean=sum(record['static_rows'])/len(record['static_rows']),
        enclosing_seconds=record['enclosing_ns']/1e9,fit_only_readout=measurement['fit_readouts'][repetition])
    descriptor=directory/'runs'/record['parent_run_id']/'checkpoints/best.json';pointer=json.loads(descriptor.read_text())
    metadata=json.loads((descriptor.parent/pointer['metadata']).read_text());variance=metadata['state']['fit_target_variance']
    with zipfile.ZipFile(directory/'E-predictions.npz') as archive:
        y=float_array(archive,'y')
        rep['normalized_E_risks']={role:sum((p-target)**2 for p,target in zip(float_array(archive,role),y))/len(y)/variance for role in record['methods']}
    assert abs(rep['normalized_E_risks']['random']-rep['normalized_E_risks']['top']-rep['h1_mean'])<1e-12
    assert abs(rep['normalized_E_risks']['ordinary_upper']-rep['normalized_E_risks']['static_partial']-rep['static_mean'])<1e-12
    for role,method in record['methods'].items():
        rep['methods'][role]=dict(stages=[stage(directory/s['path']) for s in method['stages']],
            selected_g_mse=method['selected_g_mse'],total_parameters=method['total_parameters'],enclosing_seconds=method['enclosing_ns']/1e9)
    probes_path=Path(record['greedy_report']);greedy=json.loads(probes_path.read_text())
    assert greedy['complete'] and not greedy['not_attempted'] and len(greedy['eligible_ids'])==64 and len(greedy['trials'])==65
    assert all(t['outcome']=='completed' and t['executed_updates']==320 for t in greedy['trials'])
    rep['greedy']=dict(path=str(probes_path),sha256=hash_file(probes_path),complete=True,eligible_ids=greedy['eligible_ids'],
        allocation=greedy['allocation'],probe_count=len(greedy['trials']),trials=[])
    for trial in greedy['trials']:
        rep['greedy']['trials'].append(dict(candidate=trial['candidate'],nanoseconds=trial['nanoseconds'],
            g_mse=trial['g_mse'],stage=stage(probes_path.parent/trial['path'])))
    report['repetitions'].append(rep)
report['current_disk']=dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(zip(('total','used','free'),shutil.disk_usage('.'))))
write(base/'complete-summary.json',report)
print('complete graph verified: 2 fresh parents,16 role records,130 actual probes; artifact bytes',report['final_attempt_bytes']['total'])
