from pathlib import Path
"""Causal controls must be active, independent, reconstructible and paired."""
import copy
import json
import numpy as np
import pytest
import torch
from test_train import cpu_training_evidence, training_case, assert_state_equal
from treeneuron.baselines.api import FitRequest
from treeneuron.artifacts import create_run
from treeneuron.train import fit_parent, pack_state, unpack_state, TrainingPolicy, continue_fit


def causal():
    from treeneuron.baselines import causal
    return causal


def test_nearest_ordinary_parameter_brackets():
    assert causal().ordinary_width_brackets(16, 1697) == (94, 95)
    assert causal().ordinary_width_brackets(16, 1693) == (94, 94)
    for d, target in [(0, 10), (16, 2), (True, 100), (16, 1.5)]:
        with pytest.raises(ValueError): causal().ordinary_width_brackets(d, target)


@pytest.mark.parametrize('coordinates', ['centered', 'direct_leaf'])
@pytest.mark.parametrize('m,total', [(0, 1153), (16, 1697), (64, 3329)])
def test_static_active_counts_and_formula(coordinates, m, total):
    model = causal().StaticRegressor(16, 64, m, coordinates, 1.).double()
    assert sum(p.numel() for p in model.parameters()) == total
    x = torch.randn(9, 16, dtype=torch.float64)
    layer = model.layer
    pieces = []
    if m:
        gate = torch.sigmoid(layer.gate(x))
        if coordinates == 'centered':
            a, q = layer.center(x), layer.contrast(x)
            selected = gate*(a+q) + (1-gate)*(a-q)
        else:
            selected = gate*layer.left(x) + (1-gate)*layer.right(x)
        pieces.append(selected)
    if m < 64: pieces.append(layer.ordinary(x))
    expected = model.head(torch.relu(torch.cat(pieces, dim=1))).squeeze(-1)
    torch.testing.assert_close(model(x), expected, rtol=1e-12, atol=1e-12)
    model(x).square().sum().backward()
    assert all(p.grad is not None for p in model.parameters())
    restored = causal().StaticRegressor.from_config(model.export_config()).double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_widen_preserves_function_moments_stream_and_schedule(training_case):
    data, config, paths = training_case
    parent = fit_parent(data, config, paths)
    assert parent.update > 0
    fork = causal().build_causal('B2', {'variant': 'widen', 'added_units': 3, 'seed': 17}, parent=parent)
    state = fork.state
    before = parent.predict(data.G.X)
    np.testing.assert_allclose(state.predict(data.G.X), before, rtol=0, atol=1e-6)
    assert state.config.model.h == 7 and state.identity_hashes['config'] != parent.identity_hashes['config']
    assert state.growth_metadata['parent_identity_hashes'] == parent.identity_hashes
    assert state.scheduler.position == parent.scheduler.position
    assert state.scheduler.horizon == parent.scheduler.horizon
    assert state.scheduler.base_lrs == parent.scheduler.base_lrs*2
    assert state.update == parent.update and state.cursor == parent.cursor
    assert torch.equal(state.order, parent.order)
    old = dict(parent.model.named_parameters())
    new = dict(state.model.named_parameters())
    for name, p in old.items():
        q = new[name.replace('head.','head.base.')]
        slices = tuple(slice(0, n) for n in p.shape)
        assert p.data_ptr() != q.data_ptr()
        torch.testing.assert_close(q[slices], p, rtol=0, atol=0)
        for key in ('exp_avg', 'exp_avg_sq'):
            torch.testing.assert_close(state.optimizer.state[q][key][slices], parent.optimizer.state[p][key], rtol=0, atol=0)
        assert state.optimizer.state[q]['step'] == parent.optimizer.state[p]['step']
    x = torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float32)
    state.model(x).sum().backward()
    assert torch.count_nonzero(state.model.head.added.weight.grad) > 0
    assert torch.count_nonzero(state.model.layer.added.weight.grad) == 0
    manifest = state.growth_metadata['widening_manifest']
    assert manifest['old_parameter_count'] == 17 and manifest['new_parameter_count'] == 29
    assert any(m['old_name'] == 'head.weight' and m['new_name'] == 'head.base.weight' for m in manifest['parameter_mappings'])
    fork.save(paths.directory / 'widened.json')
    from treeneuron.checkpoint import load_checkpoint
    loaded = load_checkpoint(paths.directory / 'widened.json', expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(state, loaded)
    a = continue_fit(state, data, TrainingPolicy.from_config(state.config, continuation=True), create_run(paths.directory.parent, state.config.to_dict()))
    b = continue_fit(loaded, data, TrainingPolicy.from_config(loaded.config, continuation=True), create_run(paths.directory.parent, loaded.config.to_dict()))
    assert_state_equal(a, b)


@pytest.mark.parametrize('method,extra', [('B2', {'variant':'lower','target_parameters':1697}), ('B2', {'variant':'upper','target_parameters':1697}), ('B3', {'variant':'partial','count':16,'coordinates':'centered'}), ('B3', {'variant':'full','coordinates':'direct_leaf'}), ('B3', {'variant':'partial','count':16,'coordinates':'direct_leaf'}), ('B3', {'variant':'full','coordinates':'centered'})])
def test_actual_large_shapes_fit_checkpoint_and_adapter(tmp_path, monkeypatch, cpu_training_evidence, method, extra):
    from treeneuron import train
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import DevelopmentData, ObservationSet
    monkeypatch.setattr(train, 'BACKEND_REPORT_PATH', cpu_training_evidence)
    rng = np.random.default_rng(72)
    def rows(n, start):
        x = rng.normal(size=(n,16)); ids = np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(x, 9 + x[:,0]*2, ids, ids)
    data = DevelopmentData(rows(128,0), rows(17,200), rows(3,300), 'causal-check')
    class Forbidden:
        def __getattribute__(self, name): raise AssertionError('V access')
    object.__setattr__(data, 'V', Forbidden())
    config = resolve_experiment({'study_id':'causal', 'data':{'n_fit':128,'n_growth':17,'n_validation':3}, 'training':{'base_updates':2,'continuation_updates':2,'validation_interval':1,'patience':10}})
    predictor = causal().build_causal(method, {'experiment':config.to_dict(), **extra})
    request = FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,2, {'development_data':data}, {'max_updates':2}, tmp_path)
    predictor.fit(request)
    manifest=json.loads((predictor.run_paths.directory/'manifest.json').read_text())
    assert manifest['data']['raw_hash']==data.manifest.dataset_id
    assert predictor.resource_summary()['run_id']==predictor.run_paths.run_id
    state = predictor.state
    assert predictor.predict(data.G.X).shape == (17,)
    x = torch.tensor(state.preprocessing.transform_X(data.G.X),dtype=torch.float32)
    with torch.no_grad(): expected = state.preprocessing.inverse_y(state.model(x).numpy())
    np.testing.assert_array_equal(predictor.predict(data.G.X),expected)
    evidence = state.backend['causal_cpu_evidence']
    assert evidence['topology'] == state.model.export_config()
    assert evidence['batch_size'] == 128 and evidence['status'] == 'passed'
    assert state.backend['performance_calibration_matches'] is False
    predictor.save(tmp_path/'control.json')
    restored = causal().CausalPredictor.load(tmp_path/'control.json', expected_hashes=state.identity_hashes)
    assert_state_equal(state,restored.state)
    a = continue_fit(state,data,TrainingPolicy.from_config(state.config,continuation=True),create_run(tmp_path,state.config.to_dict()))
    b = continue_fit(restored.state,data,TrainingPolicy.from_config(state.config,continuation=True),create_run(tmp_path,state.config.to_dict()))
    assert_state_equal(a,b)


def test_factory_rejects_role_confusion_and_bad_outputs(training_case):
    data, config, paths = training_case
    state = fit_parent(data,config,paths)
    for method, cfg in [('B6', {}), ('B2', {'variant':'duplicate'}), ('B3', {'variant':'partial','count':2,'coordinates':'bogus'})]:
        with pytest.raises(ValueError): causal().build_causal(method,cfg,parent=state)
    adapter = causal().build_causal('B1',{},parent=state)
    assert adapter.state is not state
    class Bad(torch.nn.Module):
        def __init__(self): super().__init__(); self.p=torch.nn.Parameter(torch.ones(1))
        def forward(self,x): return torch.ones(len(x),2)
    adapter.state.model = Bad()
    with pytest.raises(ValueError): adapter.predict(data.G.X)


def test_b4_b5_require_matching_allocation(training_case):
    from treeneuron.growth.allocate import Allocation
    data, config, paths = training_case
    state=fit_parent(data,config,paths)
    for method, policy in [('B4','random'),('B5','gaussian'),('B5','adam'),('B5','sgd')]:
        allocation=Allocation(1,1,torch.tensor([0]),torch.ones(1,3,dtype=torch.float64)*.05,policy,3,'a'*64,torch.full((1,),int(policy=='gaussian'),dtype=torch.int64),{'shared_scored_units':4,'standalone_contrast_units_required':int(policy=='random')})
        adapter=causal().build_causal(method,{'variant':policy} if method=='B5' else {},parent=state,allocation=allocation)
        assert adapter.state.model.layer.grown_ids.tolist()==[0]
        assert not state.model.layer.growth_applied
        np.testing.assert_array_equal(adapter.predict(data.G.X),state.predict(data.G.X))
    with pytest.raises(ValueError): causal().build_causal('B4',{},parent=state,allocation=allocation)


def test_widen_new_parameters_have_fresh_adam_steps(training_case):
    data,config,paths=training_case
    state=fit_parent(data,config,paths)
    before=state.predict(data.G.X)
    old_steps=[int(v['step']) for v in state.optimizer.state.values()]
    manifest=causal().widen_parent(state,3,42)
    np.testing.assert_array_equal(state.predict(data.G.X),before)
    assert manifest.new_parameter_names
    for name,p in state.model.named_parameters():
        step=int(state.optimizer.state[p]['step'])
        assert step==(0 if name in manifest.new_parameter_names else old_steps[0])


@pytest.mark.parametrize('mutation', ['receipt_shapes','receipt_error','moment_shape','moment_dtype','model_dtype','manifest','fg','context','budget'])
def test_strict_new_boundaries(training_case, mutation):
    data,config,paths=training_case
    parent=fit_parent(data,config,paths)
    adapter=causal().build_causal('B2',{'variant':'widen','added_units':2,'seed':8},parent=parent)
    if mutation in ('receipt_shapes','receipt_error','moment_shape','moment_dtype','model_dtype'):
        snap=pack_state(adapter.state)
        if mutation=='receipt_shapes': snap['metadata']['backend']['causal_cpu_evidence']['parameter_shapes']['head.base.weight']=[1,999]
        elif mutation=='receipt_error': snap['metadata']['backend']['causal_cpu_evidence']['forward_max_abs']=1e10
        elif mutation=='model_dtype': snap['tensors']['model']['head.base.weight']=snap['tensors']['model']['head.base.weight'].double()
        else:
            first=next(iter(snap['tensors']['optimizer']['state'].values()))
            first['exp_avg']=torch.ones(999) if mutation=='moment_shape' else first['exp_avg'].double()
        with pytest.raises(ValueError): unpack_state(snap,expected_hashes=adapter.state.identity_hashes,device='cpu')
    else:
        context={'development_data':data}; x=data.F.X; budget={'max_updates':6}
        if mutation=='manifest':
            from dataclasses import replace
            object.__setattr__(data,'manifest',replace(data.manifest,teacher_id='wrong'))
        elif mutation=='fg': x=x+1
        elif mutation=='context': context={}
        elif mutation=='budget': budget={'max_updates':1}
        request=FitRequest(x,data.F.y,data.G.X,data.G.y,2,context,budget,paths.directory.parent)
        with pytest.raises(ValueError): adapter.fit(request)


def test_shared_coordinates_forward_and_chain_rule():
    centered=causal().StaticRegressor(3,4,2,'centered',.7).double()
    direct=causal().StaticRegressor(3,4,2,'direct_leaf',.7).double()
    with torch.no_grad():
        direct.head.load_state_dict(centered.head.state_dict())
        direct.layer.ordinary.load_state_dict(centered.layer.ordinary.state_dict())
        direct.layer.gate.load_state_dict(centered.layer.gate.state_dict())
        for key in ('weight','bias'):
            a,q=getattr(centered.layer.center,key),getattr(centered.layer.contrast,key)
            getattr(direct.layer.left,key).copy_(a+q); getattr(direct.layer.right,key).copy_(a-q)
    x=torch.randn(11,3,dtype=torch.float64)
    torch.testing.assert_close(centered(x),direct(x),rtol=1e-12,atol=1e-12)
    centered(x).square().sum().backward();direct(x).square().sum().backward()
    for key in ('weight','bias'):
        left,right=getattr(direct.layer.left,key).grad,getattr(direct.layer.right,key).grad
        torch.testing.assert_close(getattr(centered.layer.center,key).grad,left+right,rtol=1e-12,atol=1e-12)
        torch.testing.assert_close(getattr(centered.layer.contrast,key).grad,left-right,rtol=1e-12,atol=1e-12)


def test_causal_cli_executes_recipes(tmp_path, cpu_training_evidence):
    import subprocess,sys
    path=tmp_path/'capabilities.json'
    run=subprocess.run([sys.executable,'-m','treeneuron.cli','verify-baselines','--track','causal',
        '--backend-report',str(cpu_training_evidence),'--artifacts-dir',str(tmp_path/'recipes'),'--output',str(path)],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    report=json.loads(path.read_text())
    assert report['implemented_checks_passed']
    assert report['track_ready'] is True
    assert {row['method'] for row in report['adapter_matrix'] if row['status']=='verified'} >= {'Mean','Ridge','SplineRidge','B6'}
    probes=report['greedy_probes']
    assert probes['complete'] and len(probes['trials'])==3
    assert all(t['nanoseconds']>0 for t in probes['trials'])
    assert report['capabilities']['B6']['status']=='verified'
    assert report['capabilities']['B7']['status']=='verified'
    coordinator=Path(probes['artifact_directory'])
    assert json.loads((coordinator/'manifest.json').read_text())['identity']['status']=='completed'
    assert (coordinator/'metrics.json').exists()
    assert report['node']['anchor']['unique_training_rows']==1025
    assert report['capabilities']['B0']['spline']=='verified'
    records=report['recipes']
    assert {r['total_parameters'] for r in records if r['method_id']=='B2'}=={1693,1711}
    assert all(r['reload_max_abs']==0 for r in records)
    assert all(r['logical_batch_size']==128 for r in records)


@pytest.mark.parametrize('variant', ['lower','upper','centered','direct_leaf','widen94','widen95'])
def test_actual_shapes_interrupted_recovery(tmp_path,monkeypatch,cpu_training_evidence,variant):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import DevelopmentData,ObservationSet
    monkeypatch.setattr(train,'BACKEND_REPORT_PATH',cpu_training_evidence)
    rng=np.random.default_rng(13)
    def rows(n,start):
        x=rng.normal(size=(n,16));ids=np.arange(start,start+n,dtype=np.int64)
        return ObservationSet(x,7+3*x[:,0],ids,ids)
    data=DevelopmentData(rows(129,0),rows(17,200),rows(3,300),'interrupted-causal')
    config=resolve_experiment({'study_id':'interrupted-causal','data':{'n_fit':129,'n_growth':17,'n_validation':3},
        'training':{'base_updates':3,'continuation_updates':3,'validation_interval':1,'patience':20,'scheduler':'cosine'}})
    if variant.startswith('widen'):
        parent=fit_parent(data,config,create_run(tmp_path,config.to_dict()))
        state=causal().build_causal('B2',{'variant':'widen','added_units':int(variant[-2:])-64,'seed':11},parent=parent).state
        config=state.config
        snapshot=pack_state(state)
        policy=TrainingPolicy.from_config(config,continuation=True)
        expected=continue_fit(state,data,policy,create_run(tmp_path,config.to_dict()))
        replay=unpack_state(snapshot,expected_hashes=state.identity_hashes,device='cpu')
        execute=lambda paths:continue_fit(replay,data,policy,paths)
        initial_update=replay.update
    else:
        recipe={'experiment':config.to_dict()}
        if variant in ('lower','upper'): recipe.update(variant=variant,target_parameters=1697);method='B2'
        else: recipe.update(variant='full',coordinates=variant);method='B3'
        config=causal().build_causal(method,recipe).config
        expected=fit_parent(data,config,create_run(tmp_path,config.to_dict()))
        execute=lambda paths:fit_parent(data,config,paths)
        policy=TrainingPolicy.from_config(config);initial_update=0
    paths=create_run(tmp_path,config.to_dict())
    append=train.append_event
    def interrupt(paths,event):
        append(paths,event)
        if event['type']=='checkpoint' and event['update']==initial_update+1:
            raise KeyboardInterrupt('durable new-shape update boundary')
    monkeypatch.setattr(train,'append_event',interrupt)
    with pytest.raises(KeyboardInterrupt):execute(paths)
    monkeypatch.setattr(train,'append_event',append)
    resumed=load_checkpoint(paths.directory/'checkpoints/last.json',expected_hashes=expected.identity_hashes,device='cpu')
    actual=continue_fit(resumed,data,policy,create_run(tmp_path,config.to_dict()))
    assert_state_equal(expected,actual)
    np.testing.assert_array_equal(expected.next_row_ids(data.F),actual.next_row_ids(data.F))
    assert [e['row_ids'] for e in expected.history if e['type']=='update'][-3:]==[e['row_ids'] for e in actual.history if e['type']=='update'][-3:]


@pytest.mark.parametrize('added',[0,31])
def test_widen_rejects_bad_state_atomically_and_preserves_rng(training_case,added):
    from treeneuron.train import capture_rng
    data,config,paths=training_case
    state=fit_parent(data,config,paths)
    before=pack_state(state);rng=capture_rng()
    for bad in (-1,True,100):
        with pytest.raises(ValueError):causal().widen_parent(state,bad,4)
        assert state.config.to_dict()==before['metadata']['config']
        np.testing.assert_array_equal(state.predict(data.G.X),unpack_state(before,expected_hashes=state.identity_hashes,device='cpu').predict(data.G.X))
    # Isolate process RNG from unpack_state's explicit restoration semantics.
    from treeneuron.train import restore_rng
    restore_rng(rng)
    causal().widen_parent(state,added,42)
    torch.testing.assert_close(torch.get_rng_state(),rng['torch_cpu'],rtol=0,atol=0)
    assert state.rng_states['python']==before['tensors']['rng_states']['python']
    with pytest.raises(ValueError):causal().widen_parent(state,1,42)


def test_fresh_adapter_rejects_forged_manifest_without_v_access(training_case):
    from dataclasses import replace
    data,config,paths=training_case
    adapter=causal().build_causal('B3',{'experiment':config.to_dict(),'variant':'full','coordinates':'direct_leaf'})
    object.__setattr__(data,'manifest',replace(data.manifest,teacher_id='wrong'))
    request=FitRequest(data.F.X,data.F.y,data.G.X,data.G.y,2,{'development_data':data},{'max_updates':6},paths.directory.parent)
    with pytest.raises(ValueError,match='provenance'): adapter.fit(request)


@pytest.mark.parametrize('defect',['method','variant','count'])
def test_adapter_load_rejects_recipe_topology_mismatch(training_case,defect):
    from treeneuron.checkpoint import save_checkpoint
    data,config,paths=training_case
    parent=fit_parent(data,config,paths)
    adapter=causal().build_causal('B2',{'variant':'widen','added_units':3,'seed':4},parent=parent)
    adapter.save(paths.directory/'good.json')
    metadata=adapter.state.growth_metadata['causal_adapter']
    if defect=='method':metadata['method_id']='B1';metadata['recipe']={}
    elif defect=='variant':metadata['recipe']['variant']='duplicate'
    else:metadata['recipe']['added_units']=2
    save_checkpoint(adapter.state,paths.directory/'bad.json')
    with pytest.raises(ValueError):causal().CausalPredictor.load(paths.directory/'bad.json',expected_hashes=adapter.state.identity_hashes)


def test_widen_enforces_float32_independent_of_process_default(training_case):
    data,config,paths=training_case
    state=fit_parent(data,config,paths)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        causal().widen_parent(state,2,42)
    finally:torch.set_default_dtype(previous)
    assert all(p.dtype==torch.float32 for p in state.model.parameters())
    assert all(v['step'].dtype==torch.float32 for v in state.optimizer.state.values())


def test_dedicated_greedy_coordinator_terminal_on_failure(tmp_path,monkeypatch,cpu_training_evidence):
    from treeneuron.baselines import causal_checks
    owned=[]
    def fail(*args,**kwargs):
        owned.append(kwargs['paths'].directory)
        raise RuntimeError('injected probe failure')
    monkeypatch.setattr(causal_checks,'run_greedy_probes',fail)
    report=causal_checks.collect_causal_report(backend_report=cpu_training_evidence,artifacts_dir=tmp_path)
    assert 'injected probe failure' in report['failure']
    assert len(owned)==1
    assert json.loads((owned[0]/'manifest.json').read_text())['identity']['status']=='failed'
    assert 'injected probe failure' in json.loads((owned[0]/'metrics.json').read_text())['failure']


def test_node_gate_failure_preserves_completed_b0_b6_checks(tmp_path,monkeypatch,cpu_training_evidence):
    from treeneuron.baselines import causal_checks,node_checks
    def fail(directory):raise RuntimeError('injected NODE gate failure')
    monkeypatch.setattr(node_checks,'verify_node',fail)
    report=causal_checks.collect_causal_report(backend_report=cpu_training_evidence,artifacts_dir=tmp_path)
    assert report['track_ready'] is False and report['implemented_checks_passed'] is False
    assert report['capabilities']['B0']['status']=='verified'
    assert report['capabilities']['B6']['status']=='verified'
    assert report['capabilities']['B7']['status']=='invalid'
    matrix={r['method']:r for r in report['adapter_matrix']}
    assert matrix['B6']['status']=='verified' and matrix['B7']['status']=='invalid'
