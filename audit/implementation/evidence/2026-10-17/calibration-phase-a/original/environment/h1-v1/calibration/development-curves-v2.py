"""Source-fixed phase-A execution wrapper; generated evidence, not package code."""
import os,sys,json,time,copy,resource,subprocess,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron import train
from treeneuron.config import resolve_experiment,canonical_hash
from treeneuron.artifacts import create_run,finish_run
from treeneuron.data.synthetic import make_from_config,sample_teacher
from treeneuron.data.controls import sample_control,oracle_diagnostic_features
from treeneuron.data.transforms import FittedTransform
from treeneuron.protocol import execution_snapshot
from treeneuron.experiments.paired import file_hash,write_json

BASE=Path('environment/h1-v1/calibration')
ROOT=Path('artifacts/h1-v1/development')
train.BACKEND_REPORT_PATH=Path('environment/h1-v1/backend.json').resolve()

def config(name,**changes):
    value=dict(study_id=name,training=dict(base_updates=4000,continuation_updates=4000,
        validation_interval=100,checkpoint_interval=100,patience=10,batch_size=4096),
        seeds=dict(teacher=261017100,data=261017101,optimizer=261017102,
            allocation=261017103,contrast=261017104,reference=261017105),
        backend=dict(device='cpu',cpu_threads=1,interop_threads=1),
        execution=dict(backend_report='environment/h1-v1/backend.json',artifacts_root=str(ROOT)),
        pilot=dict(purpose='engineering',initial_teachers=1,max_teachers=1,repetitions=2,
            evaluation_rows=4096,probe_updates=20,bootstrap_iterations=2000,bootstrap_seed=261017106,
            endpoint_tolerance=.005,max_seconds=14400.))
    for key,item in changes.items():value.setdefault(key,{}).update(item)
    return resolve_experiment(value)

def identity():
    accepted=json.loads(Path('environment/h1-v1/accepted-source.json').read_text())
    actual=execution_snapshot(Path.cwd(),'680429908735a8077fbfab972468b9d420b7a9f1')
    assert actual['execution_sha256']==accepted['execution_sha256']
    assert train.source_hash()=='5c66c55693825e7dec51f0e59c74bd995da98b54c36f34acc445039d9ef85495'
    return dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        execution_sha256=actual['execution_sha256'],source_sha256=train.source_hash(),
        receipt_sha256=file_hash(train.BACKEND_REPORT_PATH),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        runtime=dict(cpu=torch.get_num_threads(),interop=torch.get_num_interop_threads(),
        env={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTORCH_ENABLE_MPS_FALLBACK')}))

def batches():
    result=dict(identity=identity(),scope='optimizer+all F rows in complete epochs; no G/checkpoint IO; one discarded warmup, seven retained epochs; serial CPU32',cells=[])
    c=config('h1-v1-batch-sweep');teacher,data=make_from_config(c)
    transform=FittedTransform.fit(data.F.X,data.F.y)
    x=torch.tensor(transform.transform_X(data.F.X),dtype=torch.float32)
    y=torch.tensor(transform.transform_y(data.F.y),dtype=torch.float32)
    for batch in (128,256,512,1024,2048,4096,8192,16384):
        torch.manual_seed(c.seeds.optimizer);model=train.SelectiveRegressor(16,64)
        optimizer=train._optimizer(model,c);samples=[]
        for epoch in range(8):
            tick=time.perf_counter_ns();updates=0;actual=[]
            order=torch.randperm(len(y),generator=torch.Generator().manual_seed(8000+epoch))
            for start in range(0,len(y),batch):
                ids=order[start:start+batch];optimizer.zero_grad(set_to_none=True)
                (model(x[ids])-y[ids]).square().mean().backward();optimizer.step()
                actual.append(len(ids));updates+=1
            samples.append(dict(epoch=epoch,elapsed_ns=time.perf_counter_ns()-tick,updates=updates,actual_rows=actual))
        median=float(np.median([s['elapsed_ns'] for s in samples[1:]]))
        result['cells'].append(dict(requested_batch=batch,physical_batch=batch,F_rows=len(y),
            warmup=samples[0],samples=samples[1:],median_epoch_ns=median,rows_per_second=len(y)*1e9/median))
    peak=max(c['rows_per_second'] for c in result['cells'])
    result.update(selected_batch=min(c['requested_batch'] for c in result['cells'] if c['rows_per_second']>=.95*peak),
        rule='smallest admitted safe logical=physical request reaching95% observed maximum median complete-epoch throughput',peak_rows_per_second=peak)
    write_json(BASE/'batch-sweep.json',result);print(json.dumps(result),flush=True)

def fit_boundary():
    result=dict(identity=identity(),scope='fresh fit_parent including receipt/optimizer/F conversion/G selection/checkpoint+ledger IO; one complete fit per requested batch in alternating order, three retained replicates',cells=[])
    for rep in range(3):
        for batch in ((2048,4096) if rep%2==0 else (4096,2048)):
            c=config(f'h1-v1-fit-boundary-{batch}-{rep}',training=dict(batch_size=batch,
                base_updates=5*((2306+batch-1)//batch),validation_interval=(2306+batch-1)//batch,
                checkpoint_interval=(2306+batch-1)//batch,patience=10))
            teacher,data=make_from_config(c);paths=create_run(ROOT/'fit-boundaries',c.to_dict())
            tick=time.perf_counter_ns();state=train.fit_parent(data,c,paths);finish_run(paths,{'selected_g_mse':state.stage['best_mse']},'completed')
            result['cells'].append(dict(requested_batch=batch,replicate=rep,enclosing_ns=time.perf_counter_ns()-tick,
                actual_rows=[e['rows'] for e in state.history if e['type']=='update'],stage=state.stage,path=str(paths.directory)))
    write_json(BASE/'fit-boundaries.json',result);print(json.dumps(result),flush=True)

def curves():
    from treeneuron.growth.score import score_units
    from treeneuron.growth.allocate import allocate
    from treeneuron.baselines.causal import build_causal,ordinary_width_brackets
    from treeneuron.baselines.api import FitRequest
    from treeneuron.checkpoint import load_checkpoint
    from treeneuron.experiments.paired import _stage_record,_score_record,_gate_probe,_allocation_record
    c=resolve_experiment(json.loads(Path(sys.argv[2]).read_text()))
    out=ROOT/c.study_id;out.mkdir(exist_ok=False)
    report=dict(identity=identity(),config=c.to_dict(),config_sha256=canonical_hash(c.to_dict()),status='running',methods={},failures=[])
    tick=time.perf_counter_ns()
    def publish():
        (out/'measurements.json').write_text(json.dumps(report,allow_nan=False))
    publish()
    try:
        teacher,data=make_from_config(c);e=sample_control(teacher,c.pilot.evaluation_rows,c.seeds.data+999,c.data.noise_sigma)
        write_json(out/'teacher.json',teacher.to_dict());write_json(out/'splits.json',data.manifest.to_dict())
        np.savez_compressed(out/'observations.npz',E_X=e.X,E_y=e.y,E_rows=e.row_id,
            **{f'{s}_{k}':getattr(getattr(data,s),k) for s in 'FGV' for k in ('X','y','row_id')})
        paths=create_run(out/'runs',c.to_dict());start=time.perf_counter_ns()
        parent=train.fit_parent(data,c,paths);finish_run(paths,{},'completed')
        parent_path=paths.directory/'checkpoints/best.json';identities=parent.identity_hashes
        report['methods']['parent']=dict(enclosing_ns=time.perf_counter_ns()-start,stage=_stage_record(parent,paths,out),
            normalized_G=parent.stage['best_mse']/float(np.var(data.F.y)))
        publish();print('parent completed',report['methods']['parent']['normalized_G'],flush=True)
        x=torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64)
        y=torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64)
        table=score_units(copy.deepcopy(parent.model).double().eval(),x,y,epsilon=c.growth.epsilon,
            tau=c.model.tau,adam_epsilon=c.training.adam_epsilon,batch_size=c.training.batch_size,unit_block_size=c.growth.block_units)
        write_json(out/'score-table.json',_score_record(table))
        alloc={k:allocate(table,c.growth.m_requested,policy=p,seed=c.seeds.allocation) for k,p in [('top','adam'),('random','random')]}
        report['gate_probe']=_gate_probe(parent.model,x,y,alloc['top'],c.training.adam_epsilon,table)
        if c.data.population=='primary':
            a=oracle_diagnostic_features(teacher,data.F.X).X;b=oracle_diagnostic_features(teacher,e.X).X
            beta=np.linalg.lstsq(a,data.F.y,rcond=None)[0]
            report['oracle_unfair_normalized_E']=float(np.mean((b@beta-e.y)**2)/np.var(data.F.y))
        target=(c.model.d+2)*c.model.h+1+2*alloc['top'].effective_count*(c.model.d+1)
        recipes=[('keep','B1',{},True),('sham','B1',{},True),('top','B5',{'variant':'adam'},True),('random','B4',{},True),
            ('ordinary_upper','B2',dict(variant='upper',target_parameters=target,experiment=c.to_dict()),False),
            ('static_partial','B3',dict(variant='partial',coordinates='centered',count=alloc['top'].effective_count,experiment=c.to_dict()),False)]
        del parent,x,y
        predictions={}
        for name,method,recipe,warm in recipes:
            start=time.perf_counter_ns()
            if warm:
                parent=load_checkpoint(parent_path,expected_hashes=identities,device='cpu')
                adapter=build_causal(method,recipe,parent=parent,allocation=alloc.get(name));del parent
                allowances=[c.training.continuation_updates]
            else:adapter=build_causal(method,recipe);allowances=[c.training.base_updates,c.training.continuation_updates]
            record=dict(stages=[]);report['methods'][name]=record;publish()
            for allowance in allowances:
                adapter.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,c.seeds.optimizer,{'development_data':data},{'max_updates':allowance},out/'runs'))
                record['stages'].append(_stage_record(adapter.state,adapter.run_paths,out));finish_run(adapter.run_paths,{},'completed')
                publish()
            adapter.save(out/f'{name}.json');prediction=adapter.predict(e.X);predictions[name]=prediction
            record.update(enclosing_ns=time.perf_counter_ns()-start,normalized_G=adapter.state.stage['best_mse']/float(np.var(data.F.y)),
                normalized_E=float(np.mean((prediction-e.y)**2)/np.var(data.F.y)),model=str(out/f'{name}.json'),
                identity_hashes=adapter.state.identity_hashes,
                actual_growth_count=alloc[name].effective_count if name in alloc else None)
            publish();print(name,'completed',record['normalized_G'],record['normalized_E'],flush=True)
        np.savez_compressed(out/'predictions.npz',**predictions,y=e.y,row_id=e.row_id)
        report['sham_max_abs']=float(np.max(np.abs(predictions['keep']-predictions['sham'])))
        assert report['sham_max_abs']==0
        report['h1_E']=report['methods']['random']['normalized_E']-report['methods']['top']['normalized_E']
        report['status']='completed'
    except BaseException as error:
        report['status']='failed';report['failures'].append(f'{type(error).__name__}: {error}');raise
    finally:
        report.update(enclosing_ns=time.perf_counter_ns()-tick,retained_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        publish()

if __name__=='__main__':
    {'batches':batches,'fit-boundary':fit_boundary,'curves':curves}[sys.argv[1]]()
