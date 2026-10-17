"""Readout wrappers around the unchanged complete engineering producer, once."""
import copy,functools,hashlib,importlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train,artifacts,checkpoint
from treeneuron.baselines import causal,greedy
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.protocol import execution_snapshot,recipe_hash
from treeneuron.experiments.paired import file_hash,write_json
from treeneuron.evaluation.timing import TimingPolicy,ResidentCausalPredictor,measure_comparison,write_timing,read_timing

driver=importlib.import_module('treeneuron.pilot.calibrate')
base=Path('environment/h1-v2/complete-resource');policy=json.loads((base/'policy.json').read_text())
config=resolve_experiment(json.loads((base/'candidate.json').read_text()))
assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
source=execution_snapshot(Path.cwd(),'fc63de6884ba5d7a911e0228d2c6707c3ce90a11')
assert source['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
assert recipe_hash(config)==policy['recipe_sha256']
assert config.pilot.purpose=='engineering' and config.pilot.initial_teachers==config.pilot.max_teachers==1
assert config.pilot.repetitions==2 and config.pilot.max_seconds==7200
assert file_hash(Path(config.execution.backend_report))=='4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42'
assert not (base/'measurement.json').exists()
started=time.perf_counter_ns();stack=[];aggregates={};events=[];publications=[];fit_readouts=[];stage_readouts=[];teacher_readouts=[];cache={};phase='graph'
readout_atomic=artifacts._atomic_json
report=dict(status='running',scope='one complete engineering resource teacher; no scientificpilot/freeze/T',
    source=source,package_sha256=train.source_hash(),config=config.to_dict(),config_sha256=canonical_hash(config.to_dict()),
    recipe_sha256=recipe_hash(config),runtime_sha256=file_hash(Path(config.execution.backend_report)),
    policy_sha256=file_hash(base/'policy.json'),wrapper_sha256=file_hash(Path(__file__)),
    runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),env={k:os.environ[k] for k in
        ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),
    failures=[],fit_readouts=fit_readouts,stage_readouts=stage_readouts,teacher_readouts=teacher_readouts)

def publish():readout_atomic(base/'measurement.json',report,replace=True)
def wrap(fn,tag,name):
    key=(fn,tag)
    if key in cache:return cache[key]
    @functools.wraps(fn)
    def measured(*args,**kwargs):
        tick=time.perf_counter_ns();frame=dict(tag=tag,name=name,child_ns=0);stack.append(frame);status='failed'
        selected=None
        try:
            value=fn(*args,**kwargs);status='completed'
            if name=='teacher_inputs':
                teacher,data,e,tc=value
                teacher_readouts.append(dict(teacher_id=teacher.teacher_id,seeds=tc.to_dict()['seeds'],split=data.manifest.assignment_hash,E_rows=len(e.y)))
            if name=='fit_repetition':
                fit_readouts.append(dict(root=str(args[2]),enclosing_ns=value['enclosing_ns'],costs=copy.deepcopy(value['costs']),
                    status=value['status'],peak_rss_bytes=value['peak_rss_bytes'],retained_bytes_before_final_pair=value['retained_bytes'],
                    parent_run_id=value['parent_run_id'],greedy_report=value['greedy_report']))
                print('FIT_GRAPH_COMPLETE',args[2],value['enclosing_ns']/1e9,flush=True)
            if name in ('fit_parent','continue_fit'):
                selected=value
                paths=args[2] if name=='fit_parent' else args[3]
                record=dict(kind=tag,path=str(paths.directory),selected_update=value.update,
                    executed_updates=value.stage['updates_completed'],max_updates=value.stage['policy']['max_updates'],
                    normalized_G=value.stage['best_mse']/value.fit_target_variance,
                    checkpoint_selected_stage=value.stage.get('best_update'))
                stage_readouts.append(record)
                if tag!='probe_fit' or sum(r['kind']=='probe_fit' for r in stage_readouts)%16==0:
                    print('STAGE',json.dumps(record),flush=True)
            return value
        finally:
            elapsed=time.perf_counter_ns()-tick;stack.pop();exclusive=elapsed-frame['child_ns']
            if stack:stack[-1]['child_ns']+=elapsed
            slot=aggregates.setdefault(phase+'/'+tag,dict(calls=0,inclusive_ns=0,exclusive_ns=0,failed_calls=0))
            slot['calls']+=1;slot['inclusive_ns']+=elapsed;slot['exclusive_ns']+=exclusive;slot['failed_calls']+=status!='completed'
            if name in ('teacher_inputs','fit_repetition','run_repetition','calibrate','run_greedy_probes','fit_parent'):
                events.append(dict(function=name,tag=tag,started_offset_ns=tick-started,elapsed_ns=elapsed,exclusive_ns=exclusive,status=status))
            if name=='_atomic_json' and args and Path(args[0]).name in ('raw.json','pair.json','greedy.json'):
                path=Path(args[0]);publications.append(dict(path=str(path),elapsed_ns=elapsed,status=status,bytes=path.stat().st_size if path.exists() else None))
    cache[key]=measured;return measured

def patch(module,name,tag,readout_name=None):
    if hasattr(module,name):setattr(module,name,wrap(getattr(module,name),tag,readout_name or name))

# Every replacement calls its original exactly once, returns the same object,
# and changes no training/scoring/prediction output or producer ledger schema.
for module in (driver,artifacts,greedy,train):patch(module,'_atomic_json','ledger_publication')
for module in (artifacts,greedy,train):patch(module,'append_event','raw_event_publication')
for module in (checkpoint,greedy):patch(module,'save_checkpoint','checkpoint_publication')
patch(checkpoint,'save_best_checkpoint','checkpoint_publication')
for module in (checkpoint,driver):patch(module,'load_checkpoint','load')
for module in (train,causal):patch(module,'fit_parent','base_fit')
for module in (train,causal):patch(module,'continue_fit','continuation_fit')
patch(greedy,'continue_fit','probe_fit')
patch(driver,'teacher_inputs','setup')
patch(driver,'score_units','score')
patch(driver,'run_greedy_probes','probe_setup_ranking')
patch(driver,'fit_repetition','fit_graph_orchestration')
patch(driver,'run_repetition','E_loss_and_wrapper_residual')
patch(driver,'calibrate','attempt_orchestration_aggregation')
patch(np,'savez_compressed','compressed_array_publication')
patch(np,'savez','score_array_publication')
patch(causal.CausalPredictor,'predict','E_prediction')

root_before=set(Path(config.execution.artifacts_root).glob('pilot-*'))
publish()
try:
    summary=driver.calibrate(config)
    roots=set(Path(config.execution.artifacts_root).glob('pilot-*'))-root_before
    assert len(roots)==1
    root=roots.pop();raw=json.loads((root/'raw.json').read_text())
    assert raw['status']=='completed' and len(raw['accepted_records'])==2 and raw['failed_pairs']==0
    assert summary.independent_teachers==1 and summary.repetitions==2
    report.update(attempt_root=str(root),summary=summary.to_dict() if hasattr(summary,'to_dict') else __import__('dataclasses').asdict(summary),
        raw_sha256=file_hash(root/'raw.json'),raw_calibration_status=raw['calibration_status'])
    # Actual E/ledger duplication accounting; these sums describe serialized copies.
    report['raw_ledger_duplication']=dict(raw_file_bytes=(root/'raw.json').stat().st_size,
        completed_attempt_record_copies=sum('record' in a for a in raw['attempts']),accepted_record_copies=len(raw['accepted_records']),
        h1_plus_static_rows_per_completed_record=2*config.pilot.evaluation_rows,
        duplicated_numeric_E_values=sum(len(a['record']['h1_rows'])+len(a['record']['static_rows']) for a in raw['attempts'] if 'record' in a)+
            sum(len(r['h1_rows'])+len(r['static_rows']) for r in raw['accepted_records']),
        limitation='raw.json retains both completed attempt and accepted record copies; cumulative ledger rewrites and independent pair.json copies also charged; larger cohorts not measured by fake ledgers')
    # Retained timing policies, measured on repetition0 selected top/random.
    phase='timing'
    pair=raw['accepted_records'][0];models={}
    for name in ('top','random'):
        path=Path(pair['models'][name]['path']);descriptor=json.loads(path.read_text())
        metadata=json.loads((path.parent/descriptor['metadata']).read_text())
        models[name]=causal.CausalPredictor.load(path,expected_hashes=metadata['state']['identity_hashes'])
    with np.load(root/'teacher-0/observations.npz',allow_pickle=False) as data:X=data['E_X'][:policy['timing']['rows']].copy()
    t=policy['timing'];common=dict(cpu_threads=1,warmup_window=5,max_warmup=30,min_repetitions=t['min_repetitions'],
        max_repetitions=t['max_repetitions'],relative_precision=t['relative_precision'],warmup_relative_spread=t['warmup_relative_spread'],seed=t['seed'])
    host=TimingPolicy(batch_sizes=(128,),**common);resident=TimingPolicy(batch_sizes=(128,),mode='resident_kernel',**common);chunks=[]
    class BoundedResident:
        def __init__(self,predictor):self.view=ResidentCausalPredictor(predictor);self.state=predictor.state
        def resource_summary(self):return self.view.resource_summary()
        def prepare_timing_input(self,values):
            assert 128<=min(128,self.state.config.data.n_fit)
            return self.view.prepare_timing_input(values)
        def predict_resident(self,values):
            assert 0<len(values)<=min(128,self.state.config.data.n_fit)
            chunks.append(len(values));return self.view.predict_resident(values)
    tick=time.perf_counter_ns()
    timing=measure_comparison({'top-host':(models['top'],host),'random-host':(models['random'],host),
        'top-resident':(BoundedResident(models['top']),resident),'random-resident':(BoundedResident(models['random']),resident)},X)
    write_timing(base/'paired-timing.json',timing);read_timing(base/'paired-timing.json')
    report.update(timing_enclosing_ns=time.perf_counter_ns()-tick,timing_precision_status=timing['precision_status'],
        ruling15=dict(requested_batch=128,F=2306,all_actual_chunks=sorted(set(chunks)),chunk_count=len(chunks)))
    report['status']='completed'
except BaseException as error:
    report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    report['failures'].append(f'{type(error).__name__}: {error}')
    roots=set(Path(config.execution.artifacts_root).glob('pilot-*'))-root_before
    if len(roots)==1:report['attempt_root']=str(roots.pop())
    raise
finally:
    report.update(enclosing_ns_before_final_readout=time.perf_counter_ns()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        tagged_clocks=aggregates,tagged_clock_rule='exclusive_ns are disjoint within instrumented call tree; inclusive_ns are nested and must not be added; unhooked stage IO stays inside stage residual',
        clock_events=events,ledger_publications=publications,disk_after=dict(zip(('total','used','free'),shutil.disk_usage('.'))),
        observed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    if report.get('attempt_root'):
        root=Path(report['attempt_root']);report['attempt_final_bytes_before_external_index']=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    publish()
