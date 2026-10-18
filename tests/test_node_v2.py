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
    np.testing.assert_allclose(model.predict(req.X_select),expected['prediction'],rtol=1e-6,atol=1e-7)
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
