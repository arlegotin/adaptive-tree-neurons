"""Exactly the two prescribed never-started pairs; existing numerical APIs only."""
import time
local_started=time.perf_counter_ns()
import hashlib,json,os,re,resource,shutil,subprocess,sys
from pathlib import Path
started=int(os.environ['H1_CONTINUATION_STARTED_NS'])
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.backend import lock_hash,runtime_hash,validate_backend_report
from treeneuron.config import load_experiment,canonical_hash
from treeneuron.data.types import array_hash
from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition,_runtime_preflight
from treeneuron.protocol import execution_snapshot,recipe_hash,validate_pilot_readiness
from treeneuron.artifacts import _atomic_json

base=Path('environment/h1-v2/user-pause-continuation/attempt-2');root=Path(os.environ['H1_CONTINUATION_ROOT'])
original=Path('artifacts/h1-v2/scientific-pilot/pilot-fe9efe02afc2444f94bd4aa6511cb76d')
original_env=Path('environment/h1-v2/genuine-pilot')
config_path=Path('environment/h1-v2/scientific-ready/scientific-config.yaml')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as s:
        for b in iter(lambda:s.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):_atomic_json(path,value,replace=True)
def size(directory):return sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
def elapsed():return (time.perf_counter_ns()-started)/1e9
report=dict(status='starting',scope='Ruling18 exact missing teacher15 R0/R1, separate exploratory lineage, original timeout retained',
    artifact_root=str(root),started_perf_ns=started,worker_start_perf_ns=local_started,
    worker_pid=os.getpid(),source={},pairs=[],resource_observations=[],failures=[],timings={},original_members=[])
def publish():save(base/'readout.json',report)
def observe(boundary):
    observations=[]
    for command in [['memory_pressure'],['pmset','-g','batt']]:
        p=subprocess.run(command,capture_output=True,text=True)
        observations.append(dict(command=command,exit_code=p.returncode,stdout=p.stdout,stderr=p.stderr))
    pressure=observations[0]['stdout'];free=re.search(r'System-wide memory free percentage:\s+(\d+)%',pressure)
    swap=re.search(r'Swapouts:\s+(\d+)',pressure)
    value=dict(boundary=boundary,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),elapsed_seconds=elapsed(),
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,disk=dict(zip(('total','used','free'),shutil.disk_usage('.'))),
        native=observations,coarse_free_percent=int(free.group(1)) if free else None,swapouts=int(swap.group(1)) if swap else None,
        new_retained_bytes=size(base)+size(root))
    report['resource_observations'].append(value);publish();return value
def guard(observation,next_pair):
    reasons=[];projection=(908098595+1)//2 if next_pair else 0
    if elapsed()>7363.856268376:reasons.append('remaining original aggregate elapsed allowance exhausted')
    if observation['peak_rss_bytes']>6448447488:reasons.append('process peak exceeds available-memory cap')
    total=report['original_retained_guard_bytes']+observation['new_retained_bytes']
    observation.update(aggregate_retained_bytes=total,next_pair_projection_bytes=projection)
    if total+projection>30871367120:reasons.append('aggregate original/new retention plus conditional projection exceeds cap')
    if observation['disk']['free']<10737418240+projection:reasons.append('disk reserve plus next-pair projection unavailable')
    first=report['resource_observations'][0]
    if (observation['coarse_free_percent'] is not None and observation['coarse_free_percent']<=10 and
        observation['swapouts'] is not None and first['swapouts'] is not None and observation['swapouts']>first['swapouts']):
        reasons.append('native <=10% coarse availability together with new swapouts')
    publish()
    if reasons:raise RuntimeError('resource_limited: '+'; '.join(reasons))

try:
    root.mkdir(parents=True,exist_ok=False)
    config=load_experiment(config_path);fullkey=canonical_hash(config.to_dict())
    assert fullkey=='9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073'
    assert recipe_hash(config)=='4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a'
    assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
    source=execution_snapshot(Path.cwd(),'6cf06399736a94c281408a51e2cb6f10f79cea67')
    assert source['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
    verify=json.loads(Path('environment/h1-v2/user-resume-root/terminal-semantic-verification.json').read_text())
    assert not verify['errors'] and verify['completed_attempts']==30 and verify['failed_pairs']==0
    seal_verify=json.loads(Path('environment/h1-v2/user-resume-root/original-seal-verification.json').read_text());assert not seal_verify['errors']
    assert sha(original/'raw-seal.json')==seal_verify['seal_sha256']=='48fbb78a29c4caa1ffb776f929326481d4a911554e64bf6c83ae8304a78f9842'
    assert sha(original_env/'inventory.json')=='c0a46776b126cd4a30c1f586b25ea3c786155736b362322fccfb428f3c06f4ea'
    inventory=json.loads((original_env/'inventory.json').read_text())
    report.update(source=source,package_sha256=train.source_hash(),lock_sha256=lock_hash(),runtime_sha256=runtime_hash(),
        full_config_key=fullkey,scientific_recipe_sha256=recipe_hash(config),config_path=str(config_path),config_file_sha256=sha(config_path),
        original_inventory_sha256=sha(original_env/'inventory.json'),original_raw_sha256='7478e09a4369292642bf4de55c9fe8634558067e89b8ab87f7b5ea354123d716',
        original_seal_sha256=sha(original/'raw-seal.json'),original_retained_guard_bytes=inventory['bytes']+(original_env/'inventory.json').stat().st_size,
        script_sha256=sha(Path(__file__)),readiness_sha256=sha(Path(config.pilot.readiness_path)),
        receipt_sha256=sha(Path(config.execution.backend_report)),constraints_sha256=sha(Path(config.pilot.constraints_path)),
        original_members=[dict(teacher_index=i,repetition=j,path=str(original/f'teacher-{i}'/f'repetition-{j}'/'pair.json'),
            sha256=next(f['sha256'] for f in inventory['files'] if f['path']==str(original/f'teacher-{i}'/f'repetition-{j}'/'pair.json'))) for i in range(15) for j in range(2)])
    # Bind current identity, not a second bulk replay of the accepted 12GB seal.
    assert all(sha(Path(m['path']))==m['sha256'] for m in report['original_members'])
    report['runtime']=_runtime_preflight(config)
    receipt=json.loads(Path(config.execution.backend_report).read_text())
    for p in receipt['backend_probes']['probes'].values():
        if p.get('status')=='passed':validate_backend_report(p)
    validate_pilot_readiness(config,Path(config.pilot.readiness_path),root=Path.cwd())
    report['timings']['source_readiness_receipt_validation_seconds']=elapsed()
    report['status']='running';initial=observe('initial')
    guard(initial,True)
    tick=time.perf_counter_ns();teacher,data,e,tc=teacher_inputs(config,15)
    assert json.loads(json.dumps(teacher.to_dict(),allow_nan=False))==json.loads((original/'teacher-15/teacher.json').read_text())
    assert json.loads(json.dumps(data.manifest.to_dict(),allow_nan=False))==json.loads((original/'teacher-15/splits.json').read_text())
    arrays={'E_X':e.X,'E_y':e.y,'E_rows':e.row_id}
    arrays.update({f'{part}_{key}':getattr(getattr(data,part),key) for part in 'FGV' for key in ('X','y','row_id')})
    checks={}
    with np.load(original/'teacher-15/observations.npz',allow_pickle=False) as saved:
        assert set(saved.files)==set(arrays)
        for name,value in arrays.items():
            actual=saved[name]
            checks[name]=dict(shape=list(value.shape),dtype=str(value.dtype),original_sha256=array_hash(actual),regenerated_sha256=array_hash(value),
                exact_equal=bool(np.array_equal(actual,value)))
            assert checks[name]['exact_equal'] and checks[name]['original_sha256']==checks[name]['regenerated_sha256']
    report.update(teacher_id=teacher.teacher_id,teacher_config=tc.to_dict(),input_equality=checks,
        original_teacher_sha256=sha(original/'teacher-15/teacher.json'),original_splits_sha256=sha(original/'teacher-15/splits.json'),
        original_arrays_sha256=sha(original/'teacher-15/observations.npz'))
    report['timings']['regenerate_compare_seconds']=(time.perf_counter_ns()-tick)/1e9
    save(root/'original-config.json',config.to_dict());save(root/'teacher.json',teacher.to_dict());save(root/'splits.json',data.manifest.to_dict());publish()
    print('INPUT_EQUALITY',json.dumps(dict(teacher_id=teacher.teacher_id,arrays=len(checks),all_exact=True,elapsed_seconds=elapsed())),flush=True)
    for repetition in (0,1):
        before=observe(f'before repetition-{repetition}');guard(before,True)
        cfg=repetition_config(tc,repetition);directory=root/f'repetition-{repetition}'
        tick=time.perf_counter_ns();record=run_repetition(cfg,data,e,directory,fullkey)
        record['repetition']=repetition
        # Match the original caller: final completed schema exists before manifest reference.
        pub_tick=time.perf_counter_ns();save(directory/'pair.json',record)
        publication=(time.perf_counter_ns()-pub_tick)/1e9
        assert record['status']=='completed' and not record['failures'] and set(record['controls'])=={'B1','B2','B3','B4','B5','B6'}
        assert len(record['methods'])==8 and record['recipe_hash']==fullkey and record['teacher_id']==teacher.teacher_id
        greedy=json.loads(Path(record['greedy_report']).read_text())
        assert greedy['complete'] and len(greedy['trials'])==65 and not greedy['not_attempted']
        assert all(t['outcome']=='completed' and t['executed_updates']==320 for t in greedy['trials'])
        item=dict(teacher_index=15,repetition=repetition,path=str(directory/'pair.json'),sha256=sha(directory/'pair.json'),
            optimizer_seed=record['optimizer_seed'],full_config_key=record['recipe_hash'],
            h1_mean=sum(record['h1_rows'])/len(record['h1_rows']),static_mean=sum(record['static_rows'])/len(record['static_rows']),
            fit_evaluation_seconds=record['enclosing_ns']/1e9,wrapper_pair_seconds=(time.perf_counter_ns()-tick)/1e9,
            final_pair_publication_seconds=publication,pair_directory_bytes=size(directory),roles=sorted(record['methods']),controls=record['controls'],probes=65,
            costs=record['costs'])
        report['pairs'].append(item);after=observe(f'after repetition-{repetition}');publish()
        print('PAIR_COMPLETE',json.dumps(item),flush=True)
        del record
    report['status']='completed'
except BaseException as error:
    report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    report['failures'].append(f'{type(error).__name__}: {error}');raise
finally:
    report.update(enclosing_seconds=elapsed(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    observe('final');save(root/'continuation-manifest.json',report);publish()
