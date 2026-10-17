"""Actual engineering recipe replay and durable post-growth recovery."""
import time
START=time.perf_counter()
import hashlib,importlib.metadata,json,os,resource,shutil,subprocess,sys,traceback
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.pilot.calibrate import teacher_inputs,repetition_config,run_repetition,_runtime_preflight
from treeneuron.protocol import execution_snapshot
from treeneuron.checkpoint import load_checkpoint
from treeneuron.artifacts import create_run,finish_run,update_run_observations
from treeneuron.data.types import array_hash
from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import score_units
from treeneuron.growth.insert import insert_growth

ROOT=Path(os.environ['WP09_ORIGINAL_ROOT']);BASE=ROOT/'environment/h1-v2/wp09';OUT=ROOT/'artifacts/h1-v2/wp09/clean-replay'
REF=Path('artifacts/h1-v2/complete-resource/pilot-5f6c0cc620bd441db53ed126157fbedf/teacher-0/repetition-0')
OUT.mkdir(parents=True,exist_ok=False)
report=dict(status='starting',failures=[],comparisons={},pid=os.getpid(),utc_start=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),timings={})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(): (BASE/'clean-readout.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
def equal(a,b,path=''):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape,path
        assert torch.equal(a.cpu(),b.cpu()),path
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),(path,a.keys(),b.keys())
        for k in a:equal(a[k],b[k],f'{path}/{k}')
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,f'{path}/{i}')
    else:assert a==b,(path,a,b)
def load(path):
    pointer=json.loads(path.read_text());meta=json.loads((path.parent/pointer['metadata']).read_text())
    return load_checkpoint(path,expected_hashes=meta['state']['identity_hashes'],device='cpu')
def compare_states(a,b,label,data,evaluation):
    sa=train.pack_state(a);sb=train.pack_state(b)
    equal(sa['tensors'],sb['tensors'],label+'/all_tensors_including_full_rng')
    for key in ('model','identity_hashes','update','epoch','cursor','n_fit','optimizer_kind','optimizer_groups','scheduler','preprocessing','fit_target_variance','data_hashes','stage'):
        equal(sa['metadata'][key],sb['metadata'][key],label+'/'+key)
    for split,rows in [('F',data.F),('G',data.G),('E',evaluation)]:equal(a.predict(rows.X),b.predict(rows.X),label+'/'+split+'predictions')
    equal(a.next_row_ids(data.F),b.next_row_ids(data.F),label+'/next_rows')
    report['comparisons'][label]=dict(exact=True,maximum_numeric_difference=0,update=a.update,epoch=a.epoch,cursor=a.cursor,
        next_row_ids_sha256=array_hash(a.next_row_ids(data.F)),tensor_groups=list(sa['tensors']),mps_rng_present=a.rng_states['torch_mps']['state'] is not None,
        model_topology=sa['metadata']['model'],stage=sa['metadata']['stage'])
    save()
try:
    assert str(Path(train.__file__).resolve()).startswith(str(Path.cwd()/'src/treeneuron'))
    assert torch.get_default_dtype()==torch.float32
    snapshot=execution_snapshot(Path.cwd(),'7a63114');assert len(snapshot['files'])==315 and snapshot['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
    assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
    assert sha('environment/h1-v2/backend.json')=='4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42'
    report['environment']=dict(executable=sys.executable,version=sys.version,import_path=train.__file__,torch=torch.__version__,numpy=np.__version__,
        dependencies={d.metadata['Name']:d.version for d in importlib.metadata.distributions()},mps_available=torch.backends.mps.is_available(),
        thread_env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')})
    report['source']=snapshot
    config=resolve_experiment(json.loads(Path('environment/h1-v2/complete-resource/candidate.json').read_text()));key=canonical_hash(config.to_dict())
    report['config_key']=key;report['runtime']=_runtime_preflight(config)
    teacher,data,e,tc=teacher_inputs(config,0);cfg=repetition_config(tc,0)
    obs=REF.parent/'observations.npz'
    arrays={'E_X':e.X,'E_y':e.y,'E_rows':e.row_id,**{f'{p}_{k}':getattr(getattr(data,p),k) for p in 'FGV' for k in ('X','y','row_id')}}
    with np.load(obs,allow_pickle=False) as saved:
        assert set(saved.files)==set(arrays)
        for name,value in arrays.items():equal(value,saved[name],'input/'+name)
    original=json.loads((REF/'pair.json').read_text())
    parent_reference=load(REF/'runs'/original['parent_run_id']/'checkpoints/best.json')
    assert (parent_reference.rng_states['torch_mps']['state'] is not None)==torch.backends.mps.is_available(),'Native/sandbox MPS visibility differs from original engineering reference'
    del parent_reference
    report['status']='running';report['disk_before']=shutil.disk_usage('.').free;assert report['disk_before']>10*1024**3;save()
    tick=time.perf_counter();fresh=run_repetition(cfg,data,e,OUT/'repetition-0',key);fresh['repetition']=0
    (OUT/'repetition-0/pair.json').write_text(json.dumps(fresh,separators=(',',':'),allow_nan=False)+'\n')
    report['timings']['full_replay_seconds']=time.perf_counter()-tick
    assert len(fresh['methods'])==8 and fresh['status']=='completed' and not fresh['failures']
    equal(json.loads((REF/'score-table.json').read_text()),json.loads((OUT/'repetition-0/score-table.json').read_text()),'score-table')
    report['comparisons']['scores']=dict(exact=True)
    for name,method in fresh['methods'].items():
        refmethod=original['methods'][name]
        for field in ('allocation','selected_g_mse','executed_row_ids','total_parameters','initial_pairing'):
            if field in method:equal(method[field],refmethod[field],name+'/'+field)
        a=load(OUT/'repetition-0/models'/f'{name}.json');b=load(REF/'models'/f'{name}.json')
        compare_states(a,b,name,data,e);del a,b
    oldg=json.loads(Path(original['greedy_report']).read_text());newg=json.loads(Path(fresh['greedy_report']).read_text())
    assert len(oldg['trials'])==len(newg['trials'])==65
    for k in ('allocation','eligible_ids','effective_count','improvements','keep_g_mse','probe_updates','complete'):equal(oldg[k],newg[k],'greedy/'+k)
    for i,(a,b) in enumerate(zip(oldg['trials'],newg['trials'])):
        for k in ('candidate','unit_id','outcome','selected_update','executed_updates','g_mse'):equal(a[k],b[k],f'probe{i}/{k}')
    report['comparisons']['all65_probes']=dict(exact=True,count=65)
    tick=time.perf_counter();parent=load(OUT/'repetition-0/runs'/fresh['parent_run_id']/'checkpoints/best.json')
    import copy
    table=score_units(copy.deepcopy(parent.model).double().eval(),torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64),torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64),epsilon=cfg.growth.epsilon,tau=cfg.model.tau,adam_epsilon=cfg.training.adam_epsilon,batch_size=cfg.training.batch_size,unit_block_size=cfg.growth.block_units)
    allocation=allocate(table,cfg.growth.m_requested,policy='adam',seed=cfg.seeds.allocation)
    # Parent best pointers contain the completed selected snapshot required by insertion.
    insert_growth(parent,allocation);insertion=parent.update;interrupted=create_run(OUT/'recovery-runs',cfg.to_dict());append=train.append_event
    def interrupt(paths,event):
        append(paths,event)
        if event['type']=='checkpoint' and event['update']==insertion+25:raise KeyboardInterrupt('WP09 controlled durable complete-update boundary after actual growth')
    train.append_event=interrupt
    try:train.continue_fit(parent,data,train.TrainingPolicy.from_config(cfg,continuation=True),interrupted)
    except KeyboardInterrupt as err:report['controlled_interruption']=dict(exception=str(err),path=str(interrupted.directory),insertion_update=insertion,update=insertion+25)
    else:raise AssertionError('Controlled interruption was not reached')
    finally:train.append_event=append
    assert json.loads((interrupted.directory/'manifest.json').read_text())['identity']['status']=='interrupted'
    durable=interrupted.directory/'checkpoints/last.json';resumed=load_checkpoint(durable,expected_hashes=parent.identity_hashes,device='cpu')
    assert resumed.update==insertion+25 and not resumed.stage['complete'] and resumed.stage['updates_completed']==25
    retry=create_run(OUT/'recovery-runs',cfg.to_dict());update_run_observations(retry,{'resources':{'retry_links':{'interrupted_run_id':interrupted.run_id,'durable_checkpoint':str(durable),'sha256':sha(durable)}}})
    resumed=train.continue_fit(resumed,data,train.TrainingPolicy.from_config(cfg,continuation=True),retry);finish_run(retry,{'selected_g_mse':resumed.stage['best_mse']},'completed')
    uninterrupted=load(OUT/'repetition-0/models/top.json')
    compare_states(resumed,uninterrupted,'post-growth-interrupted-recovery',data,e)
    report['controlled_interruption'].update(last_pointer_sha256=sha(durable),retry_path=str(retry.directory),original_policy=resumed.stage['policy'],executed_updates=resumed.stage['updates_completed'])
    report['timings']['growth_recovery_seconds']=time.perf_counter()-tick
    report.update(status='completed',roles=8,probes=65,disk_after=shutil.disk_usage('.').free)
except BaseException:report['status']='failed';report['failures'].append(traceback.format_exc());raise
finally:
    report.update(seconds=time.perf_counter()-START,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,utc_end=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=sha(__file__))
    save();print(json.dumps(dict(status=report['status'],seconds=report['seconds'],failures=report['failures'])),flush=True)
