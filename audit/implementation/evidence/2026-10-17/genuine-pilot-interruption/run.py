"""Once-only genuine-pilot CLI with transparent progress/resource readouts."""
import copy,functools,hashlib,importlib,json,os,re,resource,shutil,subprocess,sys,time
from dataclasses import asdict
from pathlib import Path
started=time.perf_counter_ns()
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron.config import load_experiment,canonical_hash
from treeneuron import train
from treeneuron.protocol import execution_snapshot,recipe_hash
from treeneuron.cli import main

base=Path('environment/h1-v2/genuine-pilot');config_path=Path('environment/h1-v2/scientific-ready/scientific-config.yaml')
config=load_experiment(config_path);constraints=json.loads(Path(config.pilot.constraints_path).read_text())
assert config.pilot.purpose=='scientific' and config.pilot.initial_teachers==8 and config.pilot.max_teachers==16 and config.pilot.repetitions==2
assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
source=execution_snapshot(Path.cwd(),'6cf06399736a94c281408a51e2cb6f10f79cea67')
assert source['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
assert recipe_hash(config)=='4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a'
assert not (base/'readout.json').exists()
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(path)
report=dict(status='starting',scope='authorized genuine independent-teacher variance pilot8->16/R2; no freeze/cohort/T',
    source=source,package_sha256=train.source_hash(),recipe_sha256=recipe_hash(config),config_path=str(config_path),
    config_file_sha256=sha(config_path),resolved_config_sha256=canonical_hash(config.to_dict()),
    constraints_sha256=sha(Path(config.pilot.constraints_path)),readiness_sha256=sha(Path(config.pilot.readiness_path)),
    validation_sha256=sha(Path('environment/h1-v2/scientific-ready/validation.json')),
    receipt_sha256=sha(Path(config.execution.backend_report)),wrapper_sha256=sha(Path(__file__)),
    runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),environment={k:os.environ[k] for k in
      ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),
    paired=[],batches=[],teachers=[],raw_publications=[],resource_observations=[],failures=[],resource_limited=False)
def publish():save(base/'readout.json',report)
def native(boundary):
    observations=[]
    for command in [['memory_pressure'],['pmset','-g','batt']]:
        result=subprocess.run(command,capture_output=True,text=True)
        observations.append(dict(command=command,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
    pressure=observations[0]['stdout'];match=re.search(r'System-wide memory free percentage:\s+(\d+)%',pressure)
    swap=re.search(r'Swapouts:\s+(\d+)',pressure)
    item=dict(boundary=boundary,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        elapsed_seconds=(time.perf_counter_ns()-started)/1e9,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        disk=dict(zip(('total','used','free'),shutil.disk_usage('.'))),native=observations,
        coarse_free_percent=int(match.group(1)) if match else None,swapouts=int(swap.group(1)) if swap else None)
    report['resource_observations'].append(item);return item
def unsafe(observation):
    reasons=[]
    if observation['peak_rss_bytes']>constraints['available_memory_bytes']:reasons.append('process peak exceeds root available-memory constraint')
    if observation['disk']['free']<10*1024**3:reasons.append('free disk below root explicit10GiB reserve')
    initial=report['resource_observations'][0]
    if (observation['coarse_free_percent'] is not None and observation['coarse_free_percent']<=10
            and observation['swapouts'] is not None and initial['swapouts'] is not None
            and observation['swapouts']>initial['swapouts']):reasons.append('native <=10% coarse available memory together with new swapouts')
    return reasons
driver=importlib.import_module('treeneuron.pilot.calibrate')
original_pair=driver.run_repetition;original_teacher=driver.teacher_inputs;original_summary=driver.summarize_pilot;original_atomic=driver._atomic_json
roots_before=set(Path(config.execution.artifacts_root).glob('pilot-*'))
@functools.wraps(original_teacher)
def teacher_inputs(*args,**kwargs):
    tick=time.perf_counter_ns();value=original_teacher(*args,**kwargs)
    teacher,data,e,tc=value
    item=dict(index=args[1],teacher_id=teacher.teacher_id,seeds=tc.to_dict()['seeds'],split_sha256=data.manifest.assignment_hash,
        E_rows=len(e.y),setup_seconds=(time.perf_counter_ns()-tick)/1e9)
    report['teachers'].append(item);publish();print('TEACHER_START',json.dumps(item),flush=True);return value
@functools.wraps(original_pair)
def run_repetition(*args,**kwargs):
    directory=Path(args[3]);before=native('before '+str(directory));reasons=unsafe(before)
    attempt=directory.parents[1]
    retained=sum(path.stat().st_size for path in attempt.rglob('*') if path.is_file())
    raw_path=attempt/'raw.json';raw_bytes=raw_path.stat().st_size if raw_path.exists() else 0
    next_pair_budget=(908098595+1)//2+2*raw_bytes
    before.update(retained_attempt_bytes=retained,next_pair_projection_bytes=next_pair_budget,
        projection_scope='half observed fullR2 sample plus two current raw-ledger copies; conditional next-pair disk reserve, not a tail guarantee')
    if retained>constraints['max_storage_bytes'] or retained+next_pair_budget>constraints['max_storage_bytes']:
        reasons.append('retained pilot plus next-pair projection exceeds root storage cap')
    if before['disk']['free']<10*1024**3+next_pair_budget:reasons.append('insufficient explicit10GiB disk reserve plus next-pair projection')
    if reasons:
        report.update(resource_limited=True,resource_limit_reasons=reasons);publish()
        raise RuntimeError('resource_limited before next atomic paired fit: '+'; '.join(reasons))
    tick=time.perf_counter_ns();record=original_pair(*args,**kwargs)
    item=dict(path=str(directory),teacher_id=record['teacher_id'],optimizer_seed=record['optimizer_seed'],status=record['status'],
        wrapper_seconds=(time.perf_counter_ns()-tick)/1e9,original_enclosing_ns=record['enclosing_ns'],
        h1_mean=sum(record['h1_rows'])/len(record['h1_rows']),static_mean=sum(record['static_rows'])/len(record['static_rows']),
        controls=record['controls'],failed_pairs=len(record['failures']),
        final_pair_directory_bytes=sum(path.stat().st_size for path in directory.rglob('*') if path.is_file()))
    report['paired'].append(item);after=native('after '+str(directory));item['resource_observation_index']=len(report['resource_observations'])-1
    publish();print('PAIR_COMPLETE',json.dumps(item),flush=True)
    if directory.name=='repetition-1':print('TEACHER_COMPLETE',directory.parent.name,'paired',len(report['paired']),flush=True)
    return record
@functools.wraps(original_summary)
def summarize(*args,**kwargs):
    tick=time.perf_counter_ns();summary=original_summary(*args,**kwargs);value=asdict(summary)
    report['batches'].append(dict(call=len(report['batches'])+1,independent_teachers=summary.independent_teachers,
        batch_runtimes=list(summary.batch_runtimes),endpoint_mc_error=summary.endpoint_mc_error,sd_interval=summary.sd_interval,
        elapsed_seconds=(time.perf_counter_ns()-tick)/1e9,summary=value))
    save(base/f"summary-{summary.independent_teachers}-call{len(report['batches'])}.json",value)
    publish();print('BATCH_SUMMARY',json.dumps({k:v for k,v in report['batches'][-1].items() if k!='summary'}),flush=True);return summary
@functools.wraps(original_atomic)
def atomic(*args,**kwargs):
    tick=time.perf_counter_ns();value=original_atomic(*args,**kwargs)
    if Path(args[0]).name=='raw.json':
        record=args[1];item=dict(path=str(args[0]),bytes=Path(args[0]).stat().st_size,
            nanoseconds=time.perf_counter_ns()-tick,attempts=len(record['attempts']),accepted_records=len(record['accepted_records']),
            completed_attempts=sum(a['status']=='completed' for a in record['attempts']),status=record['status'],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['raw_publications'].append(item);publish();print('RAW_PUBLICATION',json.dumps(item),flush=True)
    return value
driver.teacher_inputs=teacher_inputs;driver.run_repetition=run_repetition;driver.summarize_pilot=summarize;driver._atomic_json=atomic
report['status']='running';native('initial');publish()
try:
    status=main(['calibrate','--config',str(config_path)])
    assert status==0,status
    report['status']='completed'
except BaseException as error:
    report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    report['failures'].append(f'{type(error).__name__}: {error}');raise
finally:
    roots=set(Path(config.execution.artifacts_root).glob('pilot-*'))-roots_before
    report['attempt_roots']=[str(path) for path in sorted(roots)]
    report.update(wrapper_enclosing_seconds=(time.perf_counter_ns()-started)/1e9,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    native('final');publish()
