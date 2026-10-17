"""Stdlib evidence curation only: no source imports, fitting or new trials."""
import ast
import hashlib
import json
import shutil
import struct
import time
import zipfile
from pathlib import Path

ROOT = Path.cwd()
BASE = Path('environment/h1-v2/precision')
OUT = Path('artifacts/h1-v2/finite-summary')
OUT.mkdir(exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

def category(path):
    if path.name == 'events.jsonl': return 'raw_events'
    if path.name.endswith('.meta.json'): return 'checkpoint_metadata'
    if path.suffix == '.pt': return 'tensor_payloads'
    if path.suffix == '.npz': return 'compressed_arrays'
    return 'other'

def bytes_by_category(root):
    out = {k: 0 for k in ('raw_events', 'checkpoint_metadata', 'tensor_payloads', 'compressed_arrays', 'other')}
    for path in root.rglob('*'):
        if path.is_file(): out[category(path)] += path.stat().st_size
    return dict(categories=out, total=sum(out.values()))

def array(archive, key):
    # Read existing simple 1D float64 NPY arrays without importing numpy/torch.
    content = archive.read(key + '.npy')
    assert content[:6] == b'\x93NUMPY'
    width = 2 if content[6] == 1 else 4
    length = int.from_bytes(content[8:8 + width], 'little')
    header = ast.literal_eval(content[8 + width:8 + width + length].decode('latin1').strip())
    assert header['descr'] == '<f8' and header['fortran_order'] is False
    assert len(header['shape']) == 1
    values = list(struct.unpack('<' + 'd' * header['shape'][0], content[8 + width + length:]))
    return values

summary = dict(scope='finite exploratory sensitivities, diagnostic B6 opportunity and conditional numerical precision; no scientific pilot or T',
    source_sha256='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c',
    execution_sha256='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c',
    source_pin='00d6a3a67af47412680ae14944902981aac0c9c8',
    curation='stdlib arithmetic on retained predictions; no refit/new numerical experiment',
    coordinates=[], numerical_STOP=True)
coordinates=json.loads(Path('environment/h1-v2/sensitivities/coordinates.json').read_text())
invocations=[json.loads(line) for line in Path('environment/h1-v2/sensitivities/launcher.stdout').read_text().splitlines()]
for coordinate in coordinates['coordinates']:
    name=coordinate['name'];root=Path('artifacts/h1-v2/sensitivities')/('h1-v2-'+name)
    path=root/'measurements.json';raw=json.loads(path.read_text())
    assert raw['status']=='completed' and not raw['failures']
    descriptor=Path(raw['parent_checkpoint']);d=json.loads(descriptor.read_text())
    metadata=json.loads((descriptor.parent/d['metadata']).read_text())
    variance=metadata['state']['fit_target_variance']
    with zipfile.ZipFile(root/'predictions.npz') as archive:
        y=array(archive,'E_y')
        losses={role:[(prediction-target)**2/variance for prediction,target in zip(array(archive,role+'_E'),y)]
            for role in raw['methods'] if role!='parent'}
    rowloss_path=OUT/(name+'-E-row-losses.json')
    write(rowloss_path,dict(scope='derived retained exploratory E observations, not teacher resamples',
        predictions_path=str(root/'predictions.npz'),predictions_sha256=digest(root/'predictions.npz'),
        F_target_variance=variance,rows=len(y),losses=losses,
        h1_rows=[random-top for random,top in zip(losses['random'],losses['top'])]))
    roles={}
    for role,method in raw['methods'].items():
        stages=[]
        for stage in method['stages']:
            assert not stage['hit_ceiling'] and stage['final_consecutive_nonimprovements']==20
            stages.append({**{k:stage[k] for k in ('path','selected_update','executed_updates','examples','max_updates','hit_ceiling','improving_at_ceiling','final_consecutive_nonimprovements')},
                'G_curve':[dict(stage_update=e['stage_update'],update=e['update'],mse=e['metrics']['mse'],normalized_mse=e['metrics']['normalized_mse'],improved=e['improved']) for e in stage['evaluations']]})
        roles[role]=dict(normalized_G=method['normalized_G'],normalized_E=method.get('normalized_E'),
            enclosing_ns=method['enclosing_ns'],stages=stages)
        if role in losses:
            assert abs(sum(losses[role])/len(y)-method['normalized_E'])<1e-12
    summary['coordinates'].append(dict(name=name,axes_changed=coordinate['axes_changed'],seeds=coordinate['seeds'],
        path=str(path),sha256=digest(path),config_sha256=raw['config_sha256'],teacher_id=raw['teacher_id'],split_sha256=raw['split_sha256'],
        E=raw['E'],E_row_losses=dict(path=str(rowloss_path),sha256=digest(rowloss_path)),roles=roles,
        h1_E=raw['h1_E'],oracle_normalized_E=raw['unfair_oracle_normalized_E'],pairing_audit=raw['pairing_audit'],
        sham_max_abs=raw.get('sham_max_abs'),eligible_count=raw['eligible_count'],effective_count=raw['effective_count'],
        enclosing_ns=raw['enclosing_ns'],outer_ns=next(v['outer_ns'] for v in invocations if v['coordinate']==name),
        peak_rss_bytes=raw['peak_rss_bytes'],bytes=bytes_by_category(root),failures=raw['failures']))
locator=json.loads(Path('environment/h1-v2/probe-calibration/result-locator.json').read_text())
probe=json.loads(Path(locator['path']).read_text())
assert probe['status']=='completed' and not probe['failures']
summary['probes']=dict(path=locator['path'],sha256=digest(Path(locator['path'])),scope=probe['scope'],trials=[])
for trial in probe['trials']:
    stage=trial['stage'];bad=0
    for event in reversed(stage['evaluations']):
        if event['improved']:break
        bad+=1
    summary['probes']['trials'].append({**{k:trial[k] for k in ('candidate','horizon','normalized_selected_G','selected_stage_update','enclosing_ns','retained_bytes')},
        'hit_ceiling':stage['hit_ceiling'],'improving_at_ceiling':stage['improving_at_ceiling'],
        'final_consecutive_nonimprovements':bad,'executed_updates':stage['executed_updates'],'examples':stage['examples'],
        'G_curve':[dict(stage_update=e['stage_update'],normalized_mse=e['metrics']['normalized_mse'],improved=e['improved']) for e in stage['evaluations']]})
summary['probes'].update(rankings=probe['rankings'],enclosing_ns=probe['enclosing_ns'],peak_rss_bytes=probe['peak_rss_bytes'],
    bytes=bytes_by_category(Path(locator['path']).parent),proposed_probe_updates=320,
    limitation='largest finite tested allowance; only four predeclared IDs; no rank convergence or plateau claim; full resource must execute keep plus all64 eligible IDs in each repetition')
summary['precision']={name:dict(path=str(BASE/name),sha256=digest(BASE/name)) for name in ('workload.json','loss-timing.json','monte-carlo.json','paired-timing.json')}
summary['current_disk']=dict(observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**dict(zip(('total','used','free'),shutil.disk_usage(ROOT))))
write(BASE/'finite-summary.json',summary)
print('curated',len(summary['coordinates']),'coordinates',len(summary['probes']['trials']),'probes; all53 fitting stages below ceilings with20 final nonimprovements')
