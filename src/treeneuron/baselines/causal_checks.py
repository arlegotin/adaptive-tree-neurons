"""Executable diagnostic recipes, with actual shapes and pending capabilities."""
from __future__ import annotations
import copy
from dataclasses import asdict
import json
from contextlib import nullcontext
from pathlib import Path
import tempfile
import numpy as np
import torch
from treeneuron import train
from treeneuron.artifacts import create_run, finish_run
from treeneuron.baselines.api import FitRequest
from treeneuron.baselines.causal import build_causal, CausalPredictor
from treeneuron.baselines.simple import MeanPredictor, RidgePredictor, SplineRidgePredictor
from treeneuron.baselines.registry import capabilities
from treeneuron.baselines.greedy import run_greedy_probes
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData, ObservationSet
from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import score_units


def collect_causal_report(*, backend_report: Path | None=None, artifacts_dir: Path | None=None):
    """Run finite F/G-only synthetic correctness checks; no predictive study claim."""
    report=dict(schema_version=1,track='causal',implemented_checks_passed=False,
        source_hash=train.source_hash(),scope='CPU correctness diagnostics, not predictive or throughput evidence',
        capabilities={'B0':{'status':'implemented','recipes':['mean','ridge','spline_ridge'],'spline':'implemented'},
            **{f'B{i}':{'status':'implemented'} for i in range(1,6)},
            'B6':{'status':'implemented','recipe':'ANT-inspired restricted allocation'},
            'B7':{'status':'implemented','recipe':'Pinned shallow NODE'},
            'B2_duplication':{'status':'deferred','reason':'No duplication/state/symmetry recipe executed'}},
        adapter_matrix=[asdict(c) for c in capabilities() if c.track=='causal'],
        track_ready=False, implementation_scope='B0–B7 CPU correctness; default plateau/calibration is a later gate',recipes=[])
    old_path=train.BACKEND_REPORT_PATH
    before=train.capture_rng();threads=torch.get_num_threads()
    try:
        if backend_report is not None: train.BACKEND_REPORT_PATH=Path(backend_report)
        torch.set_num_threads(1)
        rng=np.random.default_rng(1602042)
        def rows(n,start):
            x=rng.normal(size=(n,16)); ids=np.arange(start,start+n,dtype=np.int64)
            return ObservationSet(x,10+2*x[:,0]-x[:,1]+.2*x[:,2]*x[:,3],ids,ids)
        data=DevelopmentData(rows(128,0),rows(32,200),rows(16,300),'causal-correctness-diagnostic')
        class Forbidden:
            def __getattribute__(self,name): raise AssertionError('Verification training accessed V')
        object.__setattr__(data,'V',Forbidden())
        config=resolve_experiment({'study_id':'verify-causal','data':{'n_fit':128,'n_growth':32,'n_validation':16},
            'training':{'base_updates':2,'continuation_updates':2,'validation_interval':1,'patience':10},
            'backend':{'device':'cpu','cpu_threads':1}})
        # Preserve every attempted recipe and checkpoint, including failures.
        with nullcontext(artifacts_dir or Path(tempfile.mkdtemp(prefix='treeneuron-causal-'))) as directory:
            root=Path(directory)
            root.mkdir(parents=True,exist_ok=True)
            report['artifacts_root']=str(root.resolve())
            request=FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,2,{'development_data':data},{'max_updates':2},root)
            for model in (MeanPredictor(),RidgePredictor(.3),SplineRidgePredictor(.3,n_knots=3)):
                model.fit(request)
                prediction=model.predict(data.G.X)
                assert prediction.shape==(32,) and np.isfinite(prediction).all()
                if isinstance(model,RidgePredictor):
                    transform=model._transform; x=transform.transform_X(data.F.X)
                    design=np.column_stack((np.ones(len(x)),x)); target=transform.transform_y(data.F.y)
                    penalty=np.diag([0.]+[.3]*x.shape[1])
                    coefficients=np.linalg.solve(design.T@design+penalty,design.T@target)
                    np.testing.assert_allclose(model._coefficients,coefficients,rtol=1e-10,atol=1e-10)
                elif type(model) is MeanPredictor: np.testing.assert_allclose(prediction,np.mean(data.F.y),rtol=0,atol=1e-12)
                path=root/(type(model).__name__+'.json'); model.save(path)
                np.testing.assert_array_equal(type(model).load(path).predict(data.G.X),prediction)
            parent=train.fit_parent(data,config,create_run(root,config.to_dict()))
            scoring=copy.deepcopy(parent.model).double().eval()
            table=score_units(scoring,torch.tensor(parent.preprocessing.transform_X(data.F.X)),
                torch.tensor(parent.preprocessing.transform_y(data.F.y)),epsilon=.05,tau=1.,adam_epsilon=1e-8,
                batch_size=128,unit_block_size=8)
            recipes=[('B1',{},parent,None)]
            for variant in ('lower','upper'):
                recipes.append(('B2',{'experiment':config.to_dict(),'variant':variant,'target_parameters':1697},None,None))
            for added in (30,31): recipes.append(('B2',{'variant':'widen','added_units':added,'seed':4},parent,None))
            for coords in ('centered','direct_leaf'):
                for variant in ('partial','full'):
                    cfg={'experiment':config.to_dict(),'variant':variant,'coordinates':coords}
                    if variant=='partial': cfg['count']=16
                    recipes.append(('B3',cfg,None,None))
            for method,policy in (('B4','random'),('B5','gaussian'),('B5','adam'),('B5','sgd')):
                allocation=allocate(table,16,policy=policy,seed=3)
                recipes.append((method,{'variant':policy} if method=='B5' else {},parent,allocation))
            # Two-unit B6 engineering fixture: all eligible probes plus keep,
            # same F128/G32 data, two updates each, then fresh joint continuation.
            probe_config=config.to_dict()
            probe_config['model']['h']=2
            probe_config['growth']['m_requested']=1
            probe_config=resolve_experiment(probe_config)
            probe_parent=train.fit_parent(data,probe_config,create_run(root,probe_config.to_dict()))
            probe_table=score_units(copy.deepcopy(probe_parent.model).double().eval(),
                torch.tensor(probe_parent.preprocessing.transform_X(data.F.X)),
                torch.tensor(probe_parent.preprocessing.transform_y(data.F.y)),epsilon=.05,tau=1.,adam_epsilon=1e-8,
                batch_size=128,unit_block_size=2)
            probe_paths=create_run(root,probe_config.to_dict())
            try:
                greedy=run_greedy_probes(probe_parent,probe_table,data,probe_updates=2,seed=3,paths=probe_paths)
                report['greedy_probes']=json.loads((probe_paths.directory/'greedy.json').read_text())
                report['greedy_probes']['artifact_directory']=str(probe_paths.directory.resolve())
            except BaseException as error:
                finish_run(probe_paths,{'failure':f'{type(error).__name__}: {error}'},'failed')
                raise
            else:
                finish_run(probe_paths,{'greedy_complete':True,'scope':'dedicated CLI probe coordinator'},'completed')
            recipes.append(('B6',{},probe_parent,greedy))
            for index,(method,cfg,source,allocation) in enumerate(recipes):
                adapter=build_causal(method,cfg,parent=source,allocation=allocation)
                if source is not None:
                    np.testing.assert_array_equal(adapter.predict(data.G.X),source.predict(data.G.X))
                adapter.fit(request)
                values=adapter.predict(data.G.X); state=adapter.state
                path=root/f'control-{index}.json'; adapter.save(path)
                restored=CausalPredictor.load(path,expected_hashes=state.identity_hashes)
                np.testing.assert_array_equal(restored.predict(data.G.X),values)
                # Exact resumed next-update model and Adam state, independent stores.
                x=torch.tensor(state.preprocessing.transform_X(data.F.X),dtype=torch.float32)
                for candidate in (state,restored.state):
                    candidate.optimizer.zero_grad(); candidate.model(x).square().mean().backward(); candidate.optimizer.step()
                for p,q in zip(state.model.parameters(),restored.state.model.parameters()):
                    torch.testing.assert_close(p,q,rtol=0,atol=0)
                    for key in ('step','exp_avg','exp_avg_sq'):
                        torch.testing.assert_close(state.optimizer.state[p][key],restored.state.optimizer.state[q][key],rtol=0,atol=0)
                report['recipes'].append(dict(method_id=method,recipe=adapter.recipe,
                    topology=state.model.export_config(),total_parameters=sum(p.numel() for p in state.model.parameters()),
                    actual_growth_count=adapter.resource_summary()['actual_growth_count'],
                    logical_batch_size=128,physical_batch_size=128,n_fit=128,n_select=32,
                    selected_update=state.update,executed_updates=state.history[-1]['executed_updates'],
                    reload_max_abs=0.,reload_next_optimizer_update_max_abs=0.,
                    checkpoint=str(path.resolve()),checkpoint_sha256=json.loads(path.read_text())['sha256'],
                    cpu_evidence=state.backend.get('causal_cpu_evidence'),
                    identities=state.identity_hashes,fit_cost=adapter.resource_summary()['costs']))
        for key in ('B0','B1','B2','B3','B4','B5','B6'):
            report['capabilities'][key]['status']='verified'
        report['capabilities']['B0']['spline']='verified'
        for row in report['adapter_matrix']:
            if row['status']=='implemented' and row['method']!='B7':row['status']='verified'
        from treeneuron.baselines.node_checks import verify_node
        report['node']=verify_node(root/'node')
        if report['node']['status']!='passed':raise RuntimeError('B7 NODE gate failed: '+str(report['node']))
        report['track_ready']=True
        for key in ('B0','B1','B2','B3','B4','B5','B6','B7'): report['capabilities'][key]['status']='verified'
        report['capabilities']['B0']['spline']='verified'
        for row in report['adapter_matrix']:
            if row['status']=='implemented': row['status']='verified'
        report['implemented_checks_passed']=True
    except Exception as error:
        report['failure']=f'{type(error).__name__}: {error}'
        for row in report['adapter_matrix']:
            if row['status']=='implemented':
                row.update(status='invalid',unavailable_reason='Verification did not complete: '+report['failure'])
                if row['method'] in report['capabilities']:
                    report['capabilities'][row['method']].update(status='invalid',reason=report['failure'])
    finally:
        train.BACKEND_REPORT_PATH=old_path;torch.set_num_threads(threads);train.restore_rng(before)
    return report
