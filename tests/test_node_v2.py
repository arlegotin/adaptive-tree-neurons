"""Bounded NODE-v2 role, native recipe and portable endpoint checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from treeneuron.baselines.api import FitRequest


def config(mode='engineering_fixture',**parameters):
    return dict(schema='research/v2-node-config',family='P12',mode=mode,
        feature_schema=dict(columns=['x','z'],categorical=[]),parameters=parameters)


def test_native_search_and_fixture_are_disjoint_recipe_policies():
    from treeneuron.research.node_v2 import NodeV2Config
    anchor=NodeV2Config.parse(config('native_anchor')).resolved()
    assert {k:anchor[k] for k in ('trees','depth','layers','tree_dim','batch_size','report_frequency','patience','latest_checkpoints')}==dict(
        trees=2048,depth=6,layers=1,tree_dim=3,batch_size=512,report_frequency=100,patience=5000,latest_checkpoints=5)
    search=NodeV2Config.parse(config('search',trees=512,depth=4,lr_multiplier=.5)).resolved()
    assert search['optimizer']==dict(lr=.0005,betas=[.95,.998],nus=[.7,1.],eps=1e-8,weight_decay=0.,decouple_weight_decay=False)
    assert search['report_frequency']==100 and search['patience']==5000
    for bad in (config('native_anchor',trees=16),config('search',trees=32,depth=4,lr_multiplier=1),
            config('search',trees=512,depth=4,lr_multiplier=True),config('search',trees=512,depth=4,lr_multiplier=1,patience=2),
            config(trees=2048),config(depth=6),config(device='mps'),config(latest_checkpoints=3)):
        with pytest.raises(ValueError):NodeV2Config.parse(bad)


def test_unknown_or_coercive_config_and_schema_fail():
    from treeneuron.research.node_v2 import NodeV2Config
    for key,value in (('schema','old'),('family','B7'),('mode','anchor'),('parameters',[]),('extra',1)):
        row=config();row[key]=value
        with pytest.raises(ValueError):NodeV2Config.parse(row)
    for params in ({'trees':True},{'depth':2.},{'batch_size':'8'},{'report_frequency':3},{'patience':0}):
        with pytest.raises(ValueError):NodeV2Config.parse(config(**params))
    row=config();row['feature_schema']['columns']=['x','x']
    with pytest.raises(ValueError):NodeV2Config.parse(row)


def test_module_and_identity_helpers_do_not_import_torch_or_legacy_train(tmp_path):
    script="""
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','node','sklearn','qhoptim'} or name=='treeneuron.train':raise AssertionError(name)
sys.meta_path.insert(0,Block())
from treeneuron.research.node_v2 import NodeV2Config,source_identity
NodeV2Config.parse({'schema':'research/v2-node-config','family':'P12','mode':'native_anchor','feature_schema':{'columns':['x'],'categorical':[]},'parameters':{}})
assert source_identity()['package_sha256']
assert not any(x in sys.modules for x in ('torch','treeneuron.train','treeneuron.baselines.node'))
"""
    run=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_typed_one_hot_unknown_and_missing_columns_precede_all_column_quantiles():
    from treeneuron.research.node_v2 import NodeTransform
    from sklearn.preprocessing import QuantileTransformer
    schema=dict(columns=['amount','kind'],categorical=['kind'])
    X=pd.DataFrame({'amount':[1.,np.nan,4.,8.,2.,7.], 'kind':pd.Series([1,'1',True,1.,None,'unknown'],dtype=object)})
    y=np.array([7.,8.,12.,11.,8.,15.]);before=X.copy(deep=True)
    transform=NodeTransform.fit(X,y,schema,71)
    # Numeric1 + missing1 + five typed known categories + unknown1 =8.
    # Canonical token sort: bool, float, int, str:1, str:unknown; missing maps last.
    encoded=np.array([[1,0,0,0,1,0,0,0],[4,1,0,0,0,1,0,0],[4,0,1,0,0,0,0,0],
        [8,0,0,1,0,0,0,0],[2,0,0,0,0,0,0,1],[7,0,0,0,0,0,1,0]],dtype=np.float32)
    rng=np.random.RandomState(71);noisy=encoded.copy()
    noisy+=(.001/np.maximum(noisy.std(axis=0,keepdims=True),.001))*rng.randn(*noisy.shape)
    expected=QuantileTransformer(n_quantiles=1000,output_distribution='normal',ignore_implicit_zeros=False,
        subsample=10000,random_state=71,copy=True).fit(noisy)
    np.testing.assert_array_equal(transform.transform(X),expected.transform(encoded))
    np.testing.assert_array_equal(transform.to_state()['quantiles'],expected.quantiles_)
    assert transform.transform(X).dtype==np.float32
    pd.testing.assert_frame_equal(X,before)
    query=pd.DataFrame({'amount':[3.], 'kind':['new']})
    encoded_query=np.array([[3,0,0,0,0,0,0,1]],dtype=np.float32)
    np.testing.assert_array_equal(transform.transform(query),expected.transform(encoded_query))
    np.testing.assert_array_equal(transform.target(y),((y-y.mean())/y.std()).astype(np.float32))
    np.testing.assert_array_equal(transform.inverse(np.array([.2],dtype=np.float32)),np.array([.2],dtype=np.float32).astype(np.float64)*y.std()+y.mean())


def test_constant_target_adaptation_and_query_geometry_are_strict():
    from treeneuron.research.node_v2 import NodeTransform
    X=pd.DataFrame({'target':[1.,2.,3.],'row_id':['a','b','a']});schema=dict(columns=list(X),categorical=['row_id'])
    transform=NodeTransform.fit(X,np.full(3,12.),schema,9)
    assert transform.to_state()['preprocessing']['target_constant'] is True
    np.testing.assert_array_equal(transform.target([12.,12.]),[0.,0.])
    assert transform.transform(X).shape==(3,4)
    with pytest.raises(ValueError):transform.transform(X[['row_id','target']])
    with pytest.raises(ValueError):transform.transform(X.assign(extra=1))
    for bad in (True,'3',1j,np.inf):
        changed=X.copy();changed['target']=pd.Series([1.,bad,3.],dtype=object)
        with pytest.raises(ValueError):transform.transform(changed)


@pytest.fixture(scope='module')
def native_policy():
    import torch
    torch.set_num_threads(1)
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    return torch


def request(path,updates=12,cfg=None):
    rng=np.random.RandomState(9);X=rng.normal(size=(37,2)).astype(np.float32);G=rng.normal(size=(11,2)).astype(np.float32)
    return FitRequest(X,100.+3.*X[:,0].astype(np.float64),G,100.+3.*G[:,0].astype(np.float64),
        1337,cfg or config(),{'max_updates':updates},path)


def test_direct_official_init_training_rolling_selection_and_full_rows(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor,NodeV2Config
    req=request(tmp_path/'fit');model=NodeV2Predictor().fit(req)
    from node_v2_reference import reference
    expected=reference(req,NodeV2Config.parse(req.config).resolved(),tmp_path/'reference')
    torch=native_policy
    for key,value in model._state['initial_model'].items():torch.testing.assert_close(value,expected['initial'][key],rtol=1e-6,atol=1e-7)
    for key,value in model._model.state_dict().items():torch.testing.assert_close(value,expected['model'].state_dict()[key],rtol=1e-6,atol=1e-7)
    for key,value in model._state['raw_model'].items():torch.testing.assert_close(value,expected['raw'][key],rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose((model.predict(req.X_select)-req.y_fit.mean())/req.y_fit.std(),
        (expected['prediction']-req.y_fit.mean())/req.y_fit.std(),rtol=1e-6,atol=1e-7)
    summary=model.resource_summary()
    assert summary['selected_step']==expected['selected_step'] and summary['executed_updates']==12
    assert [r['rows'] for r in model._state['trace']]==expected['trace']
    assert set(sum(expected['trace'],[]))==set(range(37))
    assert any(len(row)==5 for row in expected['trace'])
    assert [r['snapshot_steps'] for r in model._state['reports']]==[r['snapshot_steps'] for r in expected['reports']]
    assert model._state['reports'][-1]['snapshot_steps']==[4,6,8,10,12]
    assert summary['quality_eligible'] is False and summary['checkpoint_available'] is True
    for group in model._state['raw_optimizer']['state'].values():
        assert np.isfinite(group['beta1_weight']) and native_policy.isfinite(group['exp_avg']).all()
    assert any(any(v>0 for v in row['gradient_norms'].values()) for row in model._state['trace'])


def test_actual_native_100_report_has_no_added_cap_report(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    model=NodeV2Predictor().fit(request(tmp_path/'fit',101,config(report_frequency=100)))
    assert [r['step'] for r in model._state['reports']]==[100]
    summary=model.resource_summary()
    assert summary['selected_step']==100 and summary['executed_updates']==101
    assert summary['stop_reason']=='max_updates' and summary['quality_eligible'] is False


def test_short_cap_is_raw_unselected_and_native_patience_boundary_is_strict(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor,_patience_reached
    model=NodeV2Predictor().fit(request(tmp_path/'fit',1,config(report_frequency=100)))
    assert model.resource_summary()['selected_step'] is None
    assert model.resource_summary()['checkpoint_available'] is False
    assert model.resource_summary()['selection_status']=='raw_unselected_at_cap'
    assert not _patience_reached(5100,100,5000) and _patience_reached(5101,100,5000)
    model.save(tmp_path/'raw-snapshot')
    loaded=NodeV2Predictor.load(tmp_path/'raw-snapshot',expected_manifest_sha256=trusted(tmp_path/'raw-snapshot'))
    assert loaded.resource_summary()['selection_status']=='raw_unselected_at_cap'
    assert loaded.resource_summary()['checkpoint_available'] is False
    np.testing.assert_array_equal(model.predict(np.array([[1.,2.]])),loaded.predict(np.array([[1.,2.]])))


def test_G_changes_do_not_fit_transform_or_initialize_and_caller_rng_is_restored(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    import random
    torch=native_policy;req=request(tmp_path/'first',2)
    before=(np.random.get_state(),random.getstate(),torch.get_rng_state().clone())
    first=NodeV2Predictor().fit(req)
    after=(np.random.get_state(),random.getstate(),torch.get_rng_state())
    assert before[0][0]==after[0][0] and before[0][2:]==after[0][2:]
    np.testing.assert_array_equal(before[0][1],after[0][1]);assert before[1]==after[1];assert torch.equal(before[2],after[2])
    changed=FitRequest(req.X_fit,req.y_fit,req.X_select*100,req.y_select+10000,req.seed,req.config,req.budget,tmp_path/'changed')
    second=NodeV2Predictor().fit(changed)
    np.testing.assert_array_equal(first._transform.to_state()['quantiles'],second._transform.to_state()['quantiles'])
    for k,v in first._state['initial_model'].items():torch.testing.assert_close(v,second._state['initial_model'][k],rtol=0,atol=0)
    held=copy.deepcopy(first._model.state_dict());first.predict(req.X_select*7)
    for k,v in held.items():assert torch.equal(v,first._model.state_dict()[k])


def test_failed_partial_fit_preserves_previous_model_and_original_error_receipt(tmp_path,native_policy,monkeypatch):
    from treeneuron.research import node_v2
    model=node_v2.NodeV2Predictor().fit(request(tmp_path/'first',2));before=model.predict(np.array([[1.,2.]]))
    original=node_v2._optimizer_step
    def fail_after_one(optimizer):
        if any(s for s in optimizer.state.values()):raise FloatingPointError('injected native numerical failure')
        return original(optimizer)
    monkeypatch.setattr(node_v2,'_optimizer_step',fail_after_one)
    with pytest.raises(FloatingPointError,match='native numerical'):model.fit(request(tmp_path/'failed',4))
    np.testing.assert_array_equal(model.predict(np.array([[1.,2.]])),before)
    receipt=json.loads((tmp_path/'failed/fit-receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['seed']==1337 and receipt['config']==config()
    assert receipt['executed_updates']==1 and len(receipt['trace'])==2
    assert receipt['trace'][0]['applied'] is True and receipt['trace'][1]['optimizer_entered'] is True
    assert receipt['trace'][1]['applied'] is False and receipt['elapsed_ns']>0
    def full(*args):raise OSError('receipt full')
    monkeypatch.setattr(node_v2,'_write',full)
    with pytest.raises(FloatingPointError,match='native numerical') as caught:model.fit(request(tmp_path/'failed-receipt',4))
    assert any('receipt full' in note for note in caught.value.__notes__)


def test_fixture_update_cap_cannot_silently_run_full_anchor(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    for cap in (0,True,129):
        with pytest.raises(ValueError):NodeV2Predictor().fit(request(tmp_path/str(cap),cap))


@pytest.fixture(scope='module')
def portable_case(tmp_path_factory,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    root=tmp_path_factory.mktemp('portable-case');req=request(root/'fit',12,config(trees=32,depth=4))
    return NodeV2Predictor().fit(req),req,root/'pristine'


def snapshot(case,path):
    import shutil
    model,req,pristine=case
    if not pristine.exists():model.save(pristine)
    shutil.copytree(pristine,path)
    return json.loads((path/'manifest.json').read_text())


def trusted(path):return hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()


def tree(node,*keys):
    for key in keys:
        if isinstance(key,int) and 'list' in node:node=node['list'][key]
        else:node=dict(node['mapping'])[key]
    return node


def set_tree(node,key,value):
    for pair in node['mapping']:
        if pair[0]==key:pair[1]=value;return
    raise AssertionError(key)


def test_portable_exact_reload_without_quantile_fit_or_data_initialization(tmp_path,portable_case,native_policy,monkeypatch):
    from treeneuron.research.node_v2 import NodeV2Predictor
    from sklearn.preprocessing import QuantileTransformer
    from treeneuron.baselines._node_vendor.odst import ODST
    model,req,_=portable_case;path=tmp_path/'snapshot';snapshot(portable_case,path)
    def forbidden(*args,**kwargs):raise AssertionError('load/query must not fit or initialize')
    monkeypatch.setattr(QuantileTransformer,'fit',forbidden);monkeypatch.setattr(ODST,'initialize',forbidden)
    import random
    before=(np.random.get_state(),random.getstate(),native_policy.get_rng_state().clone())
    loaded=NodeV2Predictor.load(path,expected_manifest_sha256=trusted(path))
    np.testing.assert_array_equal(loaded.predict(req.X_select),model.predict(req.X_select))
    assert loaded.predict(req.X_select).dtype==np.float64
    np.testing.assert_array_equal(before[0][1],np.random.get_state()[1]);assert before[0][2:]==np.random.get_state()[2:]
    assert before[1]==random.getstate() and native_policy.equal(before[2],native_policy.get_rng_state())
    assert loaded.resource_summary()==model.resource_summary()
    with pytest.raises(ValueError):loaded.predict(req.X_select,batch_size=True)


def test_external_manifest_hash_is_required_before_any_native_payload(tmp_path,portable_case,monkeypatch):
    from treeneuron.research import node_v2
    path=tmp_path/'snapshot';snapshot(portable_case,path)
    def forbidden(*args,**kwargs):raise AssertionError('NPZ opened before external commitment')
    monkeypatch.setattr(node_v2.np,'load',forbidden)
    with pytest.raises(ValueError,match='commitment'):node_v2.NodeV2Predictor.load(path,expected_manifest_sha256='0'*64)
    with pytest.raises(TypeError):node_v2.NodeV2Predictor.load(path)


@pytest.mark.parametrize('change',['unknown','config','runtime','source','quantile_dtype','selected_dtype','initialization',
    'selected_average','optimizer_dtype','optimizer_negative','optimizer_age','report_missing','report_negative',
    'selected_step','trace_rows','trace_time','F_identity','extra_array'])
def test_resealed_structured_endpoint_corruption_is_rejected(tmp_path,portable_case,native_policy,change):
    from treeneuron.research.node_v2 import NodeV2Predictor
    path=tmp_path/'snapshot';state=snapshot(portable_case,path)
    with np.load(path/'arrays.npz',allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
    payload=state['payload']
    selected=dict(tree(payload,'selected_model')['mapping'])
    if change=='unknown':state['extra']=True
    elif change=='config':state['config']['parameters']['trees']=16
    elif change=='runtime':state['identity']['runtime']['python']='wrong runtime'
    elif change=='source':state['identity']['source']['package_sha256']='0'*64
    elif change=='quantile_dtype':
        key=tree(payload,'transform','quantiles')['array'];arrays[key]=arrays[key].astype(np.float32)
    elif change=='selected_dtype':
        key=selected['0.0.response']['array'];arrays[key]=arrays[key].astype(np.float64)
    elif change=='initialization':arrays[selected['0.0._is_initialized_tensor']['array']][...]=0
    elif change=='selected_average':arrays[selected['0.0.response']['array']].flat[0]+=1
    elif change in ('optimizer_dtype','optimizer_negative','optimizer_age'):
        slot=tree(payload,'raw_optimizer','state')['mapping'][0][1]
        if change=='optimizer_age':set_tree(slot,'beta1_weight',123.)
        else:
            key=tree(slot,'exp_avg_sq')['array']
            if change=='optimizer_dtype':arrays[key]=arrays[key].astype(np.float64)
            else:arrays[key].flat[0]=-1
    elif change=='report_missing':tree(payload,'reports')['list'].pop()
    elif change=='report_negative':set_tree(tree(payload,'reports',0),'g_mse',-1.)
    elif change=='selected_step':set_tree(tree(payload,'summary'),'selected_step',1)
    elif change=='trace_rows':set_tree(tree(payload,'trace',0),'rows',{'list':[999]})
    elif change=='trace_time':set_tree(tree(payload,'trace',0),'elapsed_ns',-1)
    elif change=='F_identity':set_tree(tree(payload,'F_identity','target'),'sha256','z'*64)
    elif change=='extra_array':arrays['unexpected']=np.ones(1,dtype=np.float32)
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()) for k,v in arrays.items()}
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state,allow_nan=False))
    before=native_policy.get_rng_state().clone()
    errors=dict(unknown='schema',config='recipe',runtime='binding',source='binding',quantile_dtype='quantile arrays',
        selected_dtype='model tensor',initialization='Uninitialized',selected_average='rolling average',optimizer_dtype='slot tensor',
        optimizer_negative='slot tensor',optimizer_age='bias-weight age',report_missing='Missing or additional',report_negative='G/window',
        selected_step='summary',trace_rows='trace',trace_time='trace',F_identity='role-array',extra_array='Unreferenced')
    with pytest.raises(ValueError,match=errors[change]):NodeV2Predictor.load(path,expected_manifest_sha256=trusted(path))
    assert native_policy.equal(before,native_policy.get_rng_state())


def test_same_seed_short_native_patience_matches_direct_source_and_stops_after_strict_boundary(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor,NodeV2Config
    from node_v2_reference import reference
    req=request(tmp_path/'fit',12,config(patience=2))
    req=FitRequest(req.X_fit,req.y_fit,req.X_select,100.-100*(req.y_select-100.),req.seed,req.config,req.budget,req.artifact_dir)
    model=NodeV2Predictor().fit(req);expected=reference(req,NodeV2Config.parse(req.config).resolved(),tmp_path/'reference')
    summary=model.resource_summary()
    assert summary['stop_reason']=='G_patience' and summary['executed_updates']==summary['selected_step']+3
    assert summary['selected_step']==expected['selected_step'] and len(model._state['trace'])==len(expected['trace'])


@pytest.mark.parametrize('changed_identity',['source','runtime'])
def test_fit_identity_drift_retains_prior_prediction_and_failed_paid_trace(tmp_path,portable_case,native_policy,monkeypatch,changed_identity):
    from treeneuron.research import node_v2
    model=copy.deepcopy(portable_case[0]);before=model.predict(np.array([[1.,2.]]));changed=[False]
    def identity():return {'identity':'after' if changed[0] else 'before'}
    monkeypatch.setattr(node_v2,'source_identity' if changed_identity=='source' else '_runtime',identity)
    original=node_v2._optimizer_step
    def step(optimizer):original(optimizer);changed[0]=True
    monkeypatch.setattr(node_v2,'_optimizer_step',step)
    with pytest.raises(ValueError,match='changed'):model.fit(request(tmp_path/'drift',2))
    np.testing.assert_array_equal(model.predict(np.array([[1.,2.]])),before)
    receipt=json.loads((tmp_path/'drift/fit-receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['executed_updates']==2 and receipt['elapsed_ns']>0


@pytest.mark.parametrize('change',['target_mean','quantile_values','G_role_identity'])
def test_retained_prediction_binding_rejects_valid_geometry_transform_or_role_substitution(tmp_path,portable_case,native_policy,change):
    from treeneuron.research.node_v2 import NodeV2Predictor
    path=tmp_path/'snapshot';state=snapshot(portable_case,path);payload=state['payload']
    with np.load(path/'arrays.npz',allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
    if change=='target_mean':
        p=tree(payload,'transform','preprocessing');set_tree(p,'target_mean',dict(p['mapping'])['target_mean']+1.)
    elif change=='quantile_values':arrays[tree(payload,'transform','quantiles')['array']]+=1.
    else:set_tree(tree(payload,'G_identity','target'),'sha256','a'*64)
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='binding'):NodeV2Predictor.load(path,expected_manifest_sha256=trusted(path))


def test_failure_after_partial_parameter_mutation_is_explicitly_indeterminate(tmp_path,portable_case,native_policy,monkeypatch):
    from treeneuron.research import node_v2
    model=copy.deepcopy(portable_case[0]);before=model.predict(np.array([[1.,2.]]))
    def fail(optimizer):
        with native_policy.no_grad():next(p for g in optimizer.param_groups for p in g['params'] if p.requires_grad).add_(1.)
        raise FloatingPointError('partial native optimizer failure')
    monkeypatch.setattr(node_v2,'_optimizer_step',fail)
    with pytest.raises(FloatingPointError):model.fit(request(tmp_path/'failed',2))
    receipt=json.loads((tmp_path/'failed/fit-receipt.json').read_text())
    assert receipt['partial_optimizer_update_possible'] is True
    assert sum(receipt['phase_costs_ns'].values())==receipt['elapsed_ns']
    assert {'configuration','dependencies','preprocessing','initialization','training'}<=set(receipt['phase_costs_ns'])
    assert all(type(v) is int and v>=0 for v in receipt['phase_costs_ns'].values())
    assert receipt['executed_updates']==0 and receipt['trace'][0]['optimizer_entered'] is True
    np.testing.assert_array_equal(model.predict(np.array([[1.,2.]])),before)


def test_portable_snapshot_loads_in_fresh_one_thread_process(tmp_path,portable_case):
    path=tmp_path/'snapshot';snapshot(portable_case,path);model,req,_=portable_case
    np.save(tmp_path/'query.npy',req.X_select);np.save(tmp_path/'expected.npy',model.predict(req.X_select))
    script="""
from pathlib import Path
import sys,numpy as np,torch
from treeneuron.research.node_v2 import NodeV2Predictor
torch.set_num_threads(1);torch.set_num_interop_threads(1)
root=Path(sys.argv[1]);model=NodeV2Predictor.load(root/'snapshot',expected_manifest_sha256=sys.argv[2])
np.testing.assert_array_equal(model.predict(np.load(root/'query.npy',allow_pickle=False)),np.load(root/'expected.npy',allow_pickle=False))
"""
    run=subprocess.run([sys.executable,'-c',script,str(tmp_path),trusted(path)],capture_output=True,text=True)
    (tmp_path/'consumer.stdout.txt').write_text(run.stdout);(tmp_path/'consumer.stderr.txt').write_text(run.stderr)
    assert run.returncode==0,run.stdout+run.stderr


def test_source_identity_rejects_wrong_loaded_vendor_origin(native_policy,monkeypatch):
    from treeneuron.research import node_v2
    from treeneuron.baselines._node_vendor import arch
    monkeypatch.setattr(arch,'__file__','/private/tmp/wrong-native-origin/arch.py')
    with pytest.raises(ValueError,match='origin'):node_v2.source_identity()


def test_resealed_identical_arrays_preserve_scalar_geometry_and_load(tmp_path,portable_case,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    path=tmp_path/'snapshot';state=snapshot(portable_case,path)
    with np.load(path/'arrays.npz',allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
    assert all(state['arrays'][name]['shape']==list(value.shape) for name,value in arrays.items())
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state))
    loaded=NodeV2Predictor.load(path,expected_manifest_sha256=trusted(path))
    np.testing.assert_array_equal(loaded.predict(portable_case[1].X_select),portable_case[0].predict(portable_case[1].X_select))


def test_mixed_missing_unknown_and_constant_target_native_path_roundtrips(tmp_path,native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor,NodeV2Config
    from node_v2_reference import reference
    F=pd.DataFrame({'amount':[1.,np.nan,3.,4.,2.,5.],'kind':['a','b','a','b','a','b']})
    G=pd.DataFrame({'amount':[2.5,np.nan],'kind':['b','new']})
    cfg=config();cfg['feature_schema']=dict(columns=list(F),categorical=['kind'])
    req=FitRequest(F,np.full(6,12.),G,np.full(2,12.),1337,cfg,{'max_updates':6},tmp_path/'fit')
    model=NodeV2Predictor().fit(req)
    encoded_F=np.array([[1,0,1,0,0],[3,1,0,1,0],[3,0,1,0,0],[4,0,0,1,0],[2,0,1,0,0],[5,0,0,1,0]],dtype=np.float32)
    encoded_G=np.array([[2.5,0,0,1,0],[3,1,0,0,1]],dtype=np.float32)
    adapted=FitRequest(encoded_F,req.y_fit,encoded_G,req.y_select,req.seed,req.config,req.budget,tmp_path/'reference')
    expected=reference(adapted,NodeV2Config.parse(cfg).resolved(),tmp_path/'reference')
    np.testing.assert_allclose(model.predict(G)-12.,expected['prediction']-12.,rtol=1e-6,atol=1e-7)
    model.save(tmp_path/'snapshot');loaded=NodeV2Predictor.load(tmp_path/'snapshot',expected_manifest_sha256=trusted(tmp_path/'snapshot'))
    np.testing.assert_array_equal(loaded.predict(G),model.predict(G))
    assert loaded._transform.state['preprocessing']['target_constant'] is True
    assert loaded._transform.state['preprocessing']['target_scale']==1.
    assert loaded.resource_summary()['encoded_width']==5 and loaded.resource_summary()['quality_eligible'] is False
    receipt=json.loads((tmp_path/'fit/fit-receipt.json').read_text())
    assert sum(receipt['phase_costs_ns'].values())==receipt['elapsed_ns'] and len(receipt['trace'])==6


def test_cpu_node_fit_does_not_touch_other_device_rngs(tmp_path,native_policy,monkeypatch):
    from treeneuron.research.node_v2 import NodeV2Predictor
    def forbidden_backend(label):
        def forbidden(*args,**kwargs):raise AssertionError('CPU NODE touched an off-device RNG: '+label)
        return forbidden
    for module,name in ((native_policy.cuda,'manual_seed_all'),(native_policy.mps,'manual_seed'),
            (native_policy.xpu,'manual_seed_all'),(native_policy.mtia,'manual_seed_all')):
        monkeypatch.setattr(module,name,forbidden_backend(module.__name__))
    model=NodeV2Predictor().fit(request(tmp_path/'fit',2))
    assert model.resource_summary()['executed_updates']==2


@pytest.mark.parametrize('policy',['float64','meta'])
def test_non_native_caller_defaults_reject_before_constructor_draws(tmp_path,portable_case,native_policy,monkeypatch,policy):
    import contextlib,random
    from treeneuron.research import node_v2
    from treeneuron.baselines._node_vendor import arch
    torch=native_policy;model=copy.deepcopy(portable_case[0]);query=portable_case[1].X_select
    prediction=model.predict(query);calls=[];original_dtype=torch.get_default_dtype()
    before=(np.random.get_state(),random.getstate(),torch.get_rng_state().clone())
    def forbidden(*args,**kwargs):
        calls.append(True)
        raise AssertionError('NODE constructor reached under unsupported caller defaults')
    monkeypatch.setattr(arch,'DenseBlock',forbidden)
    try:
        if policy=='float64':torch.set_default_dtype(torch.float64)
        with torch.device('meta') if policy=='meta' else contextlib.nullcontext():
            expected_device=torch.get_default_device();expected_dtype=torch.get_default_dtype()
            with pytest.raises(ValueError,match='CPU.*float32'):
                model.fit(request(tmp_path/'rejected',2))
            assert calls==[]
            assert torch.get_default_device()==expected_device and torch.get_default_dtype()==expected_dtype
            assert torch.equal(before[2],torch.get_rng_state())
    finally:torch.set_default_dtype(original_dtype)
    np.testing.assert_array_equal(before[0][1],np.random.get_state()[1]);assert before[0][2:]==np.random.get_state()[2:]
    assert before[1]==random.getstate()
    np.testing.assert_array_equal(model.predict(query),prediction)
    receipt=json.loads((tmp_path/'rejected/fit-receipt.json').read_text())
    assert receipt['status']=='failed' and receipt['executed_updates']==0 and receipt['trace']==[]
    assert receipt['error_type']=='ValueError' and receipt['elapsed_ns']>0


@pytest.mark.parametrize('moment',['exp_avg','exp_avg_sq'])
def test_same_step_selected_and_raw_optimizer_moments_must_be_byte_identical(tmp_path,portable_case,native_policy,moment):
    from treeneuron.research.node_v2 import NodeV2Predictor
    path=tmp_path/'snapshot';state=snapshot(portable_case,path);payload=state['payload']
    summary=dict(tree(payload,'summary')['mapping'])
    assert summary['selected_step']==summary['executed_updates']==12
    selected_slot=tree(payload,'selected_optimizer','state')['mapping'][0][1]
    raw_slot=tree(payload,'raw_optimizer','state')['mapping'][0][1]
    selected_key=tree(selected_slot,moment)['array'];raw_key=tree(raw_slot,moment)['array']
    with np.load(path/'arrays.npz',allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
    assert selected_key!=raw_key and np.array_equal(arrays[selected_key],arrays[raw_key])
    arrays[selected_key].flat[0]+=np.float32(1.)
    assert not np.array_equal(arrays[selected_key],arrays[raw_key]) and np.isfinite(arrays[selected_key]).all()
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='same.step.*optimizer|optimizer.*same.step'):
        NodeV2Predictor.load(path,expected_manifest_sha256=trusted(path))
