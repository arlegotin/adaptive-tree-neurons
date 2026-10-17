"""Cheap current-source diagnostics from the completed exploratory parent only."""
import os,time,json,copy,resource,sys
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment
from treeneuron.checkpoint import load_checkpoint
from treeneuron.data.synthetic import make_from_config
from treeneuron.data.controls import sample_control
from treeneuron.growth.score import score_units
from treeneuron.growth.allocate import allocate
from treeneuron.experiments.paired import _gate_probe,_score_record,write_json,file_hash
from treeneuron.baselines.causal import CausalPredictor
from treeneuron.evaluation.timing import (TimingPolicy,ResidentCausalPredictor,measure_comparison,
    measure_predictor,measure_cold_predictor,write_timing,read_timing)

directory=Path('environment/h1-v1/calibration')
root=Path('artifacts/h1-v1/development/primary-stopping-4000-a')
raw=json.loads((root/'measurements.json').read_text());config=resolve_experiment(raw['config'])
teacher,data=make_from_config(config)
parent_path=root/raw['methods']['parent']['stage']['path']/'checkpoints/best.json'
meta=json.loads((parent_path.parent/json.loads(parent_path.read_text())['metadata']).read_text())
identities=meta['state']['identity_hashes'];del meta
state=load_checkpoint(parent_path,expected_hashes=identities,device='cpu')
train.BACKEND_REPORT_PATH=Path(config.execution.backend_report).resolve()
report=dict(parent=str(parent_path),parent_pointer_sha256=file_hash(parent_path),
    source=train.source_hash(),runtime=raw['identity']['runtime'],scope='exploratory parent only; no refitting, no T, no scientific pilot',failures=[])
if sys.argv[1]=='epochs':
    x=torch.tensor(state.preprocessing.transform_X(data.F.X),dtype=torch.float32)
    y=torch.tensor(state.preprocessing.transform_y(data.F.y),dtype=torch.float32)
    table=score_units(copy.deepcopy(state.model).double().eval(),x.double(),y.double(),epsilon=.05,
        tau=1.,adam_epsilon=1e-8,batch_size=4096,unit_block_size=8)
    allocation=allocate(table,16,policy='adam',seed=config.seeds.allocation)
    report['cells']=[]
    for requested in (128,256,512,1024,2048,4096,8192,16384):
        model=copy.deepcopy(state.model);model.layer.grow(allocation.ids,allocation.contrasts.float())
        optimizer=train._optimizer(model,config);samples=[]
        for epoch in range(8):
            tick=time.perf_counter_ns();actual=[]
            order=torch.randperm(len(y),generator=torch.Generator().manual_seed(18000+epoch))
            for start in range(0,len(y),requested):
                ids=order[start:start+requested];optimizer.zero_grad(set_to_none=True)
                (model(x[ids])-y[ids]).square().mean().backward();optimizer.step();actual.append(len(ids))
            samples.append(dict(epoch=epoch,elapsed_ns=time.perf_counter_ns()-tick,actual_rows=actual,updates=len(actual)))
        score_samples=[]
        for repetition in range(4):
            tick=time.perf_counter_ns()
            score_units(copy.deepcopy(state.model).double().eval(),x.double(),y.double(),epsilon=.05,
                tau=1.,adam_epsilon=1e-8,batch_size=requested,unit_block_size=8)
            score_samples.append(time.perf_counter_ns()-tick)
        median=float(np.median([s['elapsed_ns'] for s in samples[1:]]))
        report['cells'].append(dict(requested_batch=requested,physical_batch=requested,F_rows=len(y),
            warmup=samples[0],samples=samples[1:],median_epoch_ns=median,rows_per_second=len(y)*1e9/median,
            score_warmup_ns=score_samples[0],score_samples_ns=score_samples[1:]))
    peak=max(c['rows_per_second'] for c in report['cells'])
    report.update(scope='actual current parent + scored16 growth, fresh AdamW per batch, complete optimizer epochs (one warmup/sevenretained); excludes G/checkpoint/reconstruction; score timings include CPU64 copy',
        selected_batch=min(c['requested_batch'] for c in report['cells'] if c['rows_per_second']>=.95*peak),
        peak_rows_per_second=peak,rule='smallest tested >=95% maximum median complete-epoch throughput')
    write_json(directory/'grown-epoch-score-sweep.json',report)
elif sys.argv[1]=='score':
    x=torch.tensor(state.preprocessing.transform_X(data.F.X),dtype=torch.float64)
    y=torch.tensor(state.preprocessing.transform_y(data.F.y),dtype=torch.float64)
    report['contrast_sensitivity']=[]
    for epsilon in (.01,.05,.2):
        table=score_units(copy.deepcopy(state.model).double().eval(),x,y,epsilon=epsilon,
            tau=config.model.tau,adam_epsilon=config.training.adam_epsilon,batch_size=config.training.batch_size,unit_block_size=8)
        allocation=allocate(table,16,policy='adam',seed=config.seeds.allocation)
        probe=_gate_probe(state.model,x,y,allocation,config.training.adam_epsilon,table)
        write_json(directory/f'score-epsilon-{epsilon}.json',_score_record(table))
        report['contrast_sensitivity'].append(dict(epsilon=epsilon,eligible_count=int(table.eligible.sum()),
            selected_ids=allocation.ids.tolist(),gate_probe=probe))
    write_json(directory/'contrast-sensitivity.json',report)
elif sys.argv[1]=='timing':
    # Fitted B1 view of completed parent. No fit/update/allocation is performed.
    predictor=CausalPredictor('B1',{},config=state.config,state=state)
    resident=ResidentCausalPredictor(predictor)
    actual=[]
    class BoundedResident:
        def resource_summary(self):return resident.resource_summary()
        def prepare_timing_input(self,X):
            assert 128<=min(128,config.data.n_fit)
            return resident.prepare_timing_input(X)
        def predict_resident(self,X):
            assert 0<len(X)<=min(128,config.data.n_fit)
            actual.append(len(X));return resident.predict_resident(X)
    X=sample_control(teacher,65536,261017899,.1).X
    common=dict(cpu_threads=1,warmup_window=5,max_warmup=30,min_repetitions=40,
        max_repetitions=200,relative_precision=.1,warmup_relative_spread=.1,seed=261017811)
    host=TimingPolicy(batch_sizes=(4096,),**common)
    host128=TimingPolicy(batch_sizes=(128,),**common)
    kernel=TimingPolicy(batch_sizes=(128,),mode='resident_kernel',**common)
    report['ruling15']=dict(requested_resident_batch=128,F_rows=config.data.n_fit,
        supported_upper=min(128,config.data.n_fit),assertions_before_preparation=True)
    paired=measure_comparison({'host4096':(predictor,host),'host128':(predictor,host128),
        'resident128':(BoundedResident(),kernel)},X[:4096])
    write_timing(directory/'paired-parent-timing.json',paired);read_timing(directory/'paired-parent-timing.json')
    report['paired_summary']=paired['summary'];report['ruling15'].update(actual_chunk_values=sorted(set(actual)),actual_chunk_count=len(actual))
    report['scaling']=[]
    for rows in (512,4096,16384,65536):
        result=measure_predictor(predictor,X[:rows],host)
        write_timing(directory/f'parent-host-rows-{rows}.json',result)
        read_timing(directory/f'parent-host-rows-{rows}.json')
        t=time.perf_counter_ns();prediction=predictor.predict(X[:rows]);prediction_ns=time.perf_counter_ns()-t
        save_path=directory/f'parent-E-rows-{rows}.npz';t=time.perf_counter_ns()
        np.savez_compressed(save_path,prediction=prediction,row_id=np.arange(rows,dtype=np.int64))
        report['scaling'].append(dict(rows=rows,summary=result['summary'],prediction_ns=prediction_ns,
            prediction_save_ns=time.perf_counter_ns()-t,retained_output_bytes=save_path.stat().st_size,
            serialization_scope='compressed parent prediction+int64 row IDs; no y/features and no eight-role claim'))
    report['cold']=measure_cold_predictor(predictor,X[:4096],host,directory/'cold-parent',timeout_seconds=120.)
    write_json(directory/'prediction-scaling.json',report)
print('completed',sys.argv[1],flush=True)
