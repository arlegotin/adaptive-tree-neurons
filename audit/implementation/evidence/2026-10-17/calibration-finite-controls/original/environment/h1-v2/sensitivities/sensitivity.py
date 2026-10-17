"""Finite one-factor sensitivities using existing reviewed generators/APIs."""
import os,sys,json,time,copy,resource
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.artifacts import create_run,finish_run,_atomic_json,TERMINAL_STATUSES
from treeneuron.data.synthetic import make_from_config,TeacherSpec
from treeneuron.data.controls import sample_control
from treeneuron.data.controls import oracle_diagnostic_features
from treeneuron.checkpoint import load_checkpoint
from treeneuron.growth.score import score_units
from treeneuron.growth.allocate import allocate
from treeneuron.baselines.causal import build_causal
from treeneuron.baselines.api import FitRequest
from treeneuron.protocol import execution_snapshot
from treeneuron.experiments.paired import (write_json,file_hash,_stage_record,_score_record,
    _allocation_record,_gate_probe,_json_tree,pairing_snapshot,audit_pairing)

config_path=Path(sys.argv[1]);c=resolve_experiment(json.loads(config_path.read_text()))
source=execution_snapshot(Path.cwd(),'00d6a3a67af47412680ae14944902981aac0c9c8')
assert source['execution_sha256']=='07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c'
assert train.source_hash()=='d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c'
train.BACKEND_REPORT_PATH=Path(c.execution.backend_report).resolve()
assert file_hash(train.BACKEND_REPORT_PATH)=='4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42'
out=Path(c.execution.artifacts_root)/c.study_id;out.mkdir(exist_ok=False,parents=True)
report=dict(status='running',scope='predeclared exploratory recipe/truncation check; no genuinepilot/confirmation/T',
    source=source,package_sha256=train.source_hash(),config=c.to_dict(),config_sha256=canonical_hash(c.to_dict()),
    config_file_sha256=file_hash(config_path),wrapper_sha256=file_hash(Path(__file__)),
    receipt_sha256=file_hash(train.BACKEND_REPORT_PATH),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),env={k:os.environ[k] for k in
    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}),
    methods={},child_runs=[],costs={},failures=[],improving_ceilings=[],V_accessed=False,T_accessed=False)
started=time.perf_counter_ns();open_child=None
def publish():_atomic_json(out/'measurements.json',report,replace=True)
def register(paths,metadata):
    global open_child
    open_child=paths;report['child_runs'].append(dict(run_id=paths.run_id,path=str(paths.directory),growth_metadata=metadata));publish()
def finish(paths):
    global open_child
    finish_run(paths,{},'completed');open_child=None
def stage_summary(state,paths,name):
    stage=_stage_record(state,paths,out)
    stage['examples']=sum(json.loads(line)['rows'] for line in (paths.directory/'events.jsonl').read_text().splitlines() if json.loads(line)['type']=='update')
    bad=0
    for e in reversed(stage['evaluations']):
        if e['improved']:break
        bad+=1
    stage['final_consecutive_nonimprovements']=bad
    if stage['improving_at_ceiling']:
        report['improving_ceilings'].append(dict(role=name,path=stage['path'],max_updates=stage['max_updates'],selected_update=stage['selected_update']))
        print('IMPROVING_CEILING',name,json.dumps(report['improving_ceilings'][-1]),flush=True)
    return stage
publish()
try:
    tick=time.perf_counter_ns();teacher,data=make_from_config(c)
    E_seed=c.seeds.data+999;e=sample_control(teacher,c.pilot.evaluation_rows,E_seed,c.data.noise_sigma)
    assert not set(e.row_id)&set(data.manifest.row_id)
    write_json(out/'teacher.json',teacher.to_dict());write_json(out/'splits.json',data.manifest.to_dict())
    arrays=dict(E_X=e.X,E_y=e.y,E_rows=e.row_id,**{f'{s}_{k}':getattr(getattr(data,s),k) for s in 'FGV' for k in ('X','y','row_id','group_id')})
    if e.noise_free_target is not None:arrays['E_f']=e.noise_free_target
    np.savez_compressed(out/'observations.npz',**arrays)
    report.update(teacher_id=teacher.teacher_id,split_sha256=data.manifest.assignment_hash,
        E=dict(seed=E_seed,rows=len(e.y),role='independent exploratory evaluation only',disjoint_FGV=True))
    report['costs']['setup_ns']=time.perf_counter_ns()-tick
    paths=create_run(out/'runs',c.to_dict());register(paths,{})
    tick=time.perf_counter_ns();parent=train.fit_parent(data,c,paths);finish(paths)
    parent_path=paths.directory/'checkpoints/best.json';identities=parent.identity_hashes
    report['methods']['parent']=dict(method_id='B1',role='ordinary_pre_growth_parent',
        stages=[stage_summary(parent,paths,'parent')],normalized_G=parent.stage['best_mse']/parent.fit_target_variance,
        enclosing_ns=time.perf_counter_ns()-tick)
    report.update(parent_pairing=pairing_snapshot(parent),parent_checkpoint_sha256=file_hash(parent_path),
        parent_checkpoint=str(parent_path),parent_identity_hashes=identities,parent_run_id=paths.run_id)
    parent_predictions=dict(G=parent.predict(data.G.X),E=parent.predict(e.X))
    publish();print('parent completed',report['methods']['parent']['normalized_G'],flush=True)
    tick=time.perf_counter_ns()
    x=torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64)
    y=torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64)
    table=score_units(copy.deepcopy(parent.model).double().eval(),x,y,epsilon=c.growth.epsilon,
        tau=c.model.tau,adam_epsilon=c.training.adam_epsilon,batch_size=c.training.batch_size,unit_block_size=c.growth.block_units)
    write_json(out/'score-table.json',_score_record(table))
    allocations={name:allocate(table,c.growth.m_requested,policy=policy,seed=c.seeds.allocation)
        for name,policy in [('top','adam'),('random','random')]}
    report['tiny_gate_probe']=_gate_probe(parent.model,x,y,allocations['top'],c.training.adam_epsilon,table)
    report['eligible_count']=int(table.eligible.sum());report['effective_count']=allocations['top'].effective_count
    report['costs']['score_and_gate_probe_ns']=time.perf_counter_ns()-tick
    tick=time.perf_counter_ns()
    if isinstance(teacher,TeacherSpec):
        F_oracle=oracle_diagnostic_features(teacher,data.F.X).X;E_oracle=oracle_diagnostic_features(teacher,e.X).X
        beta=np.linalg.lstsq(F_oracle,data.F.y,rcond=None)[0]
        report['unfair_oracle_normalized_E']=float(np.mean((E_oracle@beta-e.y)**2)/np.var(data.F.y))
        del F_oracle,E_oracle
    else:report['unfair_oracle_normalized_E']=None
    report['costs']['unfair_oracle_ns']=time.perf_counter_ns()-tick
    target=(c.model.d+2)*c.model.h+1+2*allocations['top'].effective_count*(c.model.d+1)
    recipes=[('keep','B1',{},True),('top','B5',{'variant':'adam'},True),('random','B4',{},True)]
    if c.data.population=='pure_noise':recipes.insert(1,('sham','B1',{},True))
    del parent,x,y
    predictions=dict(parent_E=parent_predictions['E'],parent_G=parent_predictions['G'])
    for name,method,recipe,warm in recipes:
        tick=time.perf_counter_ns();record=dict(method_id=method,recipe=recipe,stages=[],status='running')
        report['methods'][name]=record;publish()
        if warm:
            parent=load_checkpoint(parent_path,expected_hashes=identities,device='cpu')
            assert pairing_snapshot(parent)==report['parent_pairing']
            adapter=build_causal(method,recipe,parent=parent,allocation=allocations.get(name))
            named=dict(adapter.state.model.named_parameters())
            for old_name,p in parent.model.named_parameters():
                q=named[old_name];assert p.data_ptr()!=q.data_ptr() and torch.equal(p,q)
                assert _json_tree(parent.optimizer.state.get(p,{}))==_json_tree(adapter.state.optimizer.state.get(q,{}))
            record.update(parent_run_id=report['parent_run_id'],parent_checkpoint_sha256=report['parent_checkpoint_sha256'],
                initial_pairing=pairing_snapshot(adapter.state),inherited_parameters_optimizer='exact values/moments/steps; independent parameter storage')
            assert record['initial_pairing']==report['parent_pairing']
            record['initial_prediction_max_abs']=max(float(np.max(np.abs(adapter.predict(X)-parent_predictions[key])))
                for key,X in [('G',data.G.X),('E',e.X)])
            assert record['initial_prediction_max_abs']==0
            if name in allocations:record['allocation']=_allocation_record(allocations[name])
            del parent
            allowances=[c.training.continuation_updates]
        else:
            adapter=build_causal(method,recipe);allowances=[c.training.base_updates,c.training.continuation_updates]
        for allowance in allowances:
            adapter.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,c.seeds.optimizer,
                {'development_data':data},{'max_updates':allowance},out/'runs'),on_run_created=register)
            record['stages'].append(stage_summary(adapter.state,adapter.run_paths,name));finish(adapter.run_paths);publish()
        model_path=out/'models'/f'{name}.json';model_path.parent.mkdir(exist_ok=True)
        adapter.save(model_path);E_prediction=adapter.predict(e.X);G_prediction=adapter.predict(data.G.X)
        predictions[name+'_E']=E_prediction;predictions[name+'_G']=G_prediction
        record.update(status='completed',enclosing_ns=time.perf_counter_ns()-tick,
            normalized_G=adapter.state.stage['best_mse']/float(np.var(data.F.y)),
            normalized_E=float(np.mean((E_prediction-e.y)**2)/np.var(data.F.y)),
            normalized_noise_free_E=None if e.noise_free_target is None else float(np.mean((E_prediction-e.noise_free_target)**2)/np.var(data.F.y)),
            model=dict(path=str(model_path),sha256=file_hash(model_path)),identity_hashes=adapter.state.identity_hashes,
            total_parameters=sum(p.numel() for p in adapter.state.model.parameters()),executed_row_ids=record['stages'][-1]['row_hashes'])
        publish();print(name,'completed G',record['normalized_G'],'E',record['normalized_E'],flush=True)
    report['pairing_audit']=audit_pairing(report)
    if 'sham' in report['methods']:
        report['sham_max_abs']=max(float(np.max(np.abs(predictions['keep_'+s]-predictions['sham_'+s]))) for s in ('G','E'))
        assert report['sham_max_abs']==0
    saved=dict(**predictions,E_y=e.y,E_rows=e.row_id,G_y=data.G.y,G_rows=data.G.row_id)
    if e.noise_free_target is not None:saved['E_f']=e.noise_free_target
    np.savez_compressed(out/'predictions.npz',**saved)
    report['h1_E']=report['methods']['random']['normalized_E']-report['methods']['top']['normalized_E']
    report['status']='completed';publish()
except BaseException as error:
    report['status']='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'
    report['failures'].append(f'{type(error).__name__}: {error}')
    if open_child is not None:
        manifest=json.loads((open_child.directory/'manifest.json').read_text())
        if manifest['identity']['status'] not in TERMINAL_STATUSES:finish_run(open_child,{'failure':report['failures'][-1]},report['status'])
    raise
finally:
    report.update(enclosing_ns=time.perf_counter_ns()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        retained_bytes_before_final_report=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()))
    publish()
