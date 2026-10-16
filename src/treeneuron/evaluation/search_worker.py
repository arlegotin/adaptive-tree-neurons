"""Private hashed JSON/NPZ worker protocol. No V labels or executable pickle."""
from __future__ import annotations
import json
from pathlib import Path
import resource
import sys
import time


def _development(request, arrays):
    import numpy as np
    from treeneuron.data.types import DevelopmentData, ObservationSet, SplitManifest, array_hash
    raw=request['manifest']; fields={k:v for k,v in raw.items() if k not in ('assignment_hash','feature_schema_hash')}
    manifest=SplitManifest(**fields)
    if manifest.to_dict()!=json.loads(json.dumps(raw)):
        # tuple/list normalization is checked canonically below.
        from treeneuron.config import canonical_hash
        if canonical_hash(manifest.to_dict())!=canonical_hash(raw):raise ValueError('Split manifest mismatch')
    if manifest.phase!='exploratory' or set(manifest.assignments)-set('FGV'):raise ValueError('Only F/G/V development manifests accepted')
    parts={}
    for p in 'FG':
        positions=[i for i,x in enumerate(manifest.assignments) if x==p]
        ids=np.array([manifest.row_id[i] for i in positions],dtype=np.int64)
        groups=np.array([manifest.group_id[i] for i in positions],dtype=np.int64)
        if not np.array_equal(ids,arrays[p+'_ids']):raise ValueError('Manifest row IDs differ')
        parts[p]=ObservationSet(arrays[p+'_X'],arrays[p+'_y'],ids,groups,feature_names=manifest.feature_names)
        for field in ('X','y'):
            if array_hash(getattr(parts[p],field))!=dict(manifest.hashes)[p+'.'+field]:raise ValueError('Manifest F/G hash mismatch')
    if array_hash(arrays['V_X'])!=dict(manifest.hashes)['V.X']:raise ValueError('Manifest V features mismatch')
    class SealedValidation:
        def __getattribute__(self,name):raise RuntimeError('Fit attempted to access sealed V partition')
    # Rehydrate a validated redacted view preserving ORIGINAL split identity.
    # Never substitute synthetic V labels, which would change checkpoint identity.
    data=object.__new__(DevelopmentData)
    for key,value in {**parts,'V':SealedValidation(),'teacher_id':manifest.teacher_id,'manifest':manifest}.items():
        object.__setattr__(data,key,value)
    from treeneuron.baselines.causal import _validate_context
    _validate_context(data)
    return data


def _fit(request,arrays,directory):
    import copy
    import numpy as np
    import torch
    from treeneuron.artifacts import create_run, finish_run, update_run_observations
    from treeneuron.baselines.api import FitRequest
    from treeneuron.config import resolve_experiment
    from treeneuron import train
    data=_development(request,arrays); method=request['method'];cfg=dict(request['config'])
    seed=request['seed']; experiment=copy.deepcopy(request['experiment'])
    train.BACKEND_REPORT_PATH=Path(request['backend_report'])
    run_root=directory/'runs'; run_root.mkdir()
    if method in ('Mean','Ridge','SplineRidge'):
        from treeneuron.baselines.simple import MeanPredictor,RidgePredictor,SplineRidgePredictor
        cls={'Mean':MeanPredictor,'Ridge':RidgePredictor,'SplineRidge':SplineRidgePredictor}[method]
        model=cls(**cfg)
        model.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,seed,{}, {},directory))
    elif method=='B7':
        from treeneuron.baselines.node import NodePredictor
        model=NodePredictor(cfg)
        model.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,seed,{}, {'max_updates':request['max_updates']},directory))
    else:
        from treeneuron.baselines.causal import build_causal
        from treeneuron.baselines.greedy import run_greedy_probes
        from treeneuron.growth.score import score_units
        from treeneuron.growth.allocate import allocate
        for key in ('learning_rate','weight_decay'):experiment['training'][key]=cfg.pop(key)
        config=resolve_experiment(experiment)
        seed=config.seeds.optimizer
        parent=None;allocation=None
        warm=method in ('B1','B4','B5','B6') or (method=='B2' and cfg.get('variant')=='widen')
        if warm:
            parent_paths=create_run(run_root,config.to_dict())
            parent=train.fit_parent(data,config,parent_paths)
            finish_run(parent_paths,{'role':'fresh_search_parent','update':parent.update},'completed')
        if method in ('B4','B5','B6'):
            g=config.growth
            # Full shared scorer is explicitly CHARGED even for random allocation;
            # this is not a claim of an implemented sampled-ID standalone scorer.
            table=score_units(copy.deepcopy(parent.model).double().eval(),
                torch.tensor(parent.preprocessing.transform_X(data.F.X),dtype=torch.float64),
                torch.tensor(parent.preprocessing.transform_y(data.F.y),dtype=torch.float64),
                epsilon=g.epsilon,tau=config.model.tau,adam_epsilon=config.training.adam_epsilon,
                batch_size=config.training.batch_size,unit_block_size=min(config.model.h,8))
            if method=='B6':
                probe_paths=create_run(run_root,config.to_dict())
                update_run_observations(probe_paths,{'method':{'scope':'coordinator'},'data':{'raw_hash':data.manifest.dataset_id,'split_hash':data.manifest.assignment_hash},'identity':{'parent_run_id':parent_paths.run_id,'code_revision':train.source_hash(),'lock_hash':train.lock_hash()}})
                try:
                    allocation=run_greedy_probes(parent,table,data,probe_updates=cfg.pop('probe_updates'),seed=request['seed'],paths=probe_paths)
                    finish_run(probe_paths,{'role':'search_probe_coordinator','status':'completed'},'completed')
                except BaseException as exc:
                    finish_run(probe_paths,{'error':dict(type=type(exc).__name__,message=str(exc))},'failed');raise
            else:allocation=allocate(table,g.m_requested,policy='random' if method=='B4' else cfg['variant'],seed=request['seed'])
        if method=='B2' and cfg.get('variant')=='widen':cfg['seed']=request['seed']
        if not warm:cfg['experiment']=config.to_dict()
        model=build_causal(method,cfg,parent=parent,allocation=allocation)
        updates=config.training.continuation_updates if warm else config.training.base_updates
        model.fit(FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,seed,{'development_data':data},{'max_updates':updates},run_root))
        finish_run(model.run_paths,{'role':'search_candidate','update':model.state.update},'completed')
    return model


def main():
    started=time.perf_counter_ns();directory=Path(sys.argv[1]).resolve().parent
    from treeneuron.artifacts import _atomic_json
    from treeneuron.config import canonical_hash
    from treeneuron.evaluation.search import file_hash
    response={};request={}
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        payload=json.loads(Path(sys.argv[1]).read_text());request=payload['payload']
        if set(payload)!={'payload','sha256'} or canonical_hash(request)!=payload['sha256']:raise ValueError('Request hash mismatch')
        expected={'schema_version','method','config','seed','experiment','max_updates','backend_report','identity','arrays','input_path','input_sha256','manifest'}
        if set(request)!=expected or request['schema_version']!=1:raise ValueError('Worker request schema mismatch')
        from treeneuron.train import source_hash,lock_hash
        if request['identity']['source']!=source_hash() or request['identity']['core_lock']!=lock_hash():raise ValueError('Worker source/lock identity mismatch')
        expected_receipt=request['identity'].get('backend_report_sha256')
        if expected_receipt is not None and file_hash(request['backend_report'])!=expected_receipt:raise ValueError('Backend receipt hash mismatch')
        if file_hash(request['input_path'])!=request['input_sha256']:raise ValueError('Worker input hash mismatch')
        from treeneuron.data.types import array_hash
        with np.load(request['input_path'],allow_pickle=False) as archive:
            if set(archive.files)!={'F_X','F_y','G_X','G_y','V_X','F_ids','G_ids','V_ids'}:raise ValueError('Unexpected worker arrays (V labels prohibited)')
            arrays={k:archive[k] for k in archive.files}
        if {k:array_hash(v) for k,v in arrays.items()}!=request['arrays']:raise ValueError('Worker array hashes mismatch')
        model=_fit(request,arrays,directory)
        summary=model.resource_summary()
        summary.update(threads={'cpu':torch.get_num_threads(),'interop':torch.get_num_interop_threads()},fit_label_partitions=['F','G'],
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024),
            scorer_cost_policy='Full score table charged for allocation methods, including random; no standalone subset scorer claimed')
        response['resource_summary']=summary
        model.save(directory/'predictor.json')
        prediction=model.predict(arrays['V_X']);np.save(directory/'prediction.npy',prediction,allow_pickle=False)
        eligible=summary.get('quality_selection_eligible',True)
        response.update(quality_selection_eligible=eligible,prediction_sha256=file_hash(directory/'prediction.npy'),
            predictor_sha256=file_hash(directory/'predictor.json'),worker_body_ns=time.perf_counter_ns()-started,
            identity=request['identity'])
    except BaseException as exc:
        response.update(error=dict(type=type(exc).__name__,message=str(exc)),identity=request.get('identity'),worker_body_ns=time.perf_counter_ns()-started)
        # Retain original causal child state, closing operational runs on failure.
        runs=directory/'runs'
        if runs.exists():
            from treeneuron.artifacts import RunPaths,finish_run
            for manifest_path in runs.glob('*/manifest.json'):
                try:
                    manifest=json.loads(manifest_path.read_text())
                    if manifest['identity']['status'] not in ('completed','failed','interrupted'):
                        finish_run(RunPaths(manifest_path.parent.name,manifest_path.parent),{'error':response['error']},'failed')
                except Exception as close_error:response.setdefault('publication_errors',[]).append(repr(close_error))
    _atomic_json(directory/'response.json',dict(payload=response,sha256=canonical_hash(response)))
    if 'error' in response:raise SystemExit(1)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--backend-receipt':
        import os
        if 'torch' in sys.modules or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='0':raise RuntimeError('Require fresh fallback-disabled process')
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        from treeneuron.backend import _run,validate_backend_report,select_backend
        from treeneuron.artifacts import _atomic_json
        result=_run('cpu',1602047,[128],evidence={})
        result.pop('active_stage',None)
        result=json.loads(json.dumps(result,allow_nan=False));validate_backend_report(result)
        probes={'cpu':result}
        _atomic_json(Path(sys.argv[2]),{'backend_probes':{'probes':probes,'comparison_blocks':[],'decision':select_backend(probes)}})
    else:main()
