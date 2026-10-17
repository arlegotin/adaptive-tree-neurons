"""Existing-API bounded probe-opportunity calibration; no full B6 selection."""
import os,json,time,copy,resource
from pathlib import Path
from dataclasses import replace,asdict
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment
from treeneuron.checkpoint import load_checkpoint,save_checkpoint
from treeneuron.artifacts import create_run,finish_run,_atomic_json,TERMINAL_STATUSES
from treeneuron.data.synthetic import make_from_config
from treeneuron.growth.score import score_units
from treeneuron.growth.insert import insert_growth
from treeneuron.baselines.causal import _selected_parent
from treeneuron.baselines.greedy import _allocation
from treeneuron.experiments.paired import file_hash,write_json,_stage_record,_score_record,pairing_snapshot
from treeneuron.protocol import execution_snapshot

base=Path('environment/h1-v2/probe-calibration');work=json.loads((base/'workload.json').read_text())
c=resolve_experiment(json.loads(Path(work['parent_config']).read_text()))
prior=json.loads(Path(work['parent_measurement']).read_text())
assert train.source_hash()==prior['package_sha256']=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
source=execution_snapshot(Path.cwd(),'00d6a3a67af47412680ae14944902981aac0c9c8')
assert source['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
train.BACKEND_REPORT_PATH=Path(c.execution.backend_report).resolve()
parent=load_checkpoint(Path(prior['parent_checkpoint']),expected_hashes=prior['parent_identity_hashes'],device='cpu')
teacher,data=make_from_config(c);assert data.manifest.assignment_hash==prior['split_sha256']
assert pairing_snapshot(parent)==prior['parent_pairing']
root=Path('artifacts/h1-v2/probe-calibration');root.mkdir(exist_ok=True)
coordinator=create_run(root,c.to_dict());started=time.perf_counter_ns()
report=dict(status='running',scope=work['scope'],source_sha256=train.source_hash(),execution_sha256=source['execution_sha256'],
    config_sha256=parent.identity_hashes['config'],parent_checkpoint=prior['parent_checkpoint'],
    parent_checkpoint_sha256=file_hash(Path(prior['parent_checkpoint'])),parent_pairing=pairing_snapshot(parent),
    receipt_sha256=file_hash(train.BACKEND_REPORT_PATH),wrapper_sha256=file_hash(Path(__file__)),
    workload_sha256=file_hash(base/'workload.json'),runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),env={k:os.environ[k] for k in
    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),
    trials=[],failures=[],selected_parent_G=parent.stage['best_mse'],F_variance=parent.fit_target_variance)
def publish():_atomic_json(coordinator.directory/'probe-calibration.json',report,replace=True)
publish();open_child=None
try:
    x=torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64);y=torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64)
    table=score_units(copy.deepcopy(parent.model).double().eval(),x,y,epsilon=c.growth.epsilon,
        tau=c.model.tau,adam_epsilon=c.training.adam_epsilon,batch_size=c.training.batch_size,unit_block_size=c.growth.block_units)
    write_json(coordinator.directory/'score-table.json',_score_record(table));snapshot=_selected_parent(parent)
    report['eligible_ids']=[int(table.ids[i]) for i in range(len(table.ids)) if bool(table.eligible[i])]
    save_checkpoint(parent,coordinator.directory/'parent.json')
    normal=train.TrainingPolicy.from_config(c,continuation=True)
    for horizon in work['horizons']:
        for candidate in work['candidates']:
            tick=time.perf_counter_ns();policy=replace(normal,max_updates=horizon)
            child=create_run(coordinator.directory/'trials',c.to_dict());open_child=child
            trial=dict(candidate=candidate,horizon=horizon,status='running',path=str(child.directory),policy=asdict(policy))
            report['trials'].append(trial);publish()
            state=train.unpack_state(snapshot,expected_hashes=parent.identity_hashes,device='cpu');state.history=copy.deepcopy(parent.history)
            assert pairing_snapshot(state)==report['parent_pairing']
            if candidate!='keep':
                if candidate not in report['eligible_ids']:raise ValueError('Predeclared probe ID ineligible; no substitution authorized')
                row=table.ids.tolist().index(candidate)
                insert_growth(state,_allocation(table,[row],1,c.seeds.allocation))
            state.growth_metadata['greedy_probe']=dict(candidate='keep' if candidate=='keep' else f'unit_{candidate}',
                coordinator_run_id=coordinator.run_id,parent_checkpoint_sha256=report['parent_checkpoint_sha256'],
                score_hash=table.table_hash,requested_count=c.growth.m_requested,policy=asdict(policy))
            selected=train.continue_fit(state,data,policy,child)
            stage=_stage_record(selected,child,coordinator.directory)
            stage['examples']=sum(json.loads(line)['rows'] for line in (child.directory/'events.jsonl').read_text().splitlines() if json.loads(line)['type']=='update')
            assert stage['executed_updates']==horizon
            trial.update(status='completed',stage=stage,selected_g_mse=selected.stage['best_mse'],
                normalized_selected_G=selected.stage['best_mse']/parent.fit_target_variance,
                selected_stage_update=selected.update-parent.update,initial_pairing=report['parent_pairing'])
            finish_run(child,{'normalized_G':trial['normalized_selected_G']},'completed');open_child=None
            trial.update(enclosing_ns=time.perf_counter_ns()-tick,retained_bytes=sum(p.stat().st_size for p in child.directory.rglob('*') if p.is_file()))
            publish();print(json.dumps({k:trial[k] for k in ('candidate','horizon','normalized_selected_G','selected_stage_update','enclosing_ns')}),flush=True)
    report['rankings']={str(h):[t['candidate'] for t in sorted((t for t in report['trials'] if t['horizon']==h),key=lambda t:(t['selected_g_mse'],str(t['candidate'])))] for h in work['horizons']}
    report['status']='completed';finish_run(coordinator,{'scope':work['scope']},'completed')
except BaseException as error:
    report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed';report['failures'].append(f'{type(error).__name__}: {error}')
    if open_child is not None:
        m=json.loads((open_child.directory/'manifest.json').read_text())
        if m['identity']['status'] not in TERMINAL_STATUSES:finish_run(open_child,{'failure':report['failures'][-1]},report['status'])
    finish_run(coordinator,{'failure':report['failures'][-1]},report['status']);raise
finally:
    report.update(enclosing_ns=time.perf_counter_ns()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        retained_bytes_before_final_report=sum(p.stat().st_size for p in coordinator.directory.rglob('*') if p.is_file()))
    publish();write_json(base/'result-locator.json',dict(path=str(coordinator.directory/'probe-calibration.json'),sha256=file_hash(coordinator.directory/'probe-calibration.json')))
