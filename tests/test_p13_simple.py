"""P13 bounded closed-form geometry, role isolation and portable state."""
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


ALPHAS=[1e-6,1e-4,1e-2,1.,1e2,1e4,1e6]


class Forbidden:
    def __array__(self,*args,**kwargs):raise AssertionError('G must never be inspected')


def config(method='ridge',alpha=1.,knots=4,schema=None):
    return dict(schema='research/v2-p13-config',family='P13',estimator=method,
        feature_schema=schema or dict(columns=['x','z'],categorical=[]),
        parameters={} if method=='mean' else dict(alpha=alpha,**({'n_knots':knots} if method=='spline' else {})))


def request(path,cfg=None,X=None,y=None):
    if X is None:
        rng=np.random.default_rng(81);X=rng.normal(size=(32,2))
    if y is None:y=11+np.arange(len(X))*.2
    return FitRequest(X,np.asarray(y),Forbidden(),Forbidden(),17,cfg or config(),{},path)


def fit(req):
    from treeneuron.research.p13_simple import P13Predictor
    return P13Predictor().fit(req)


def manifest(model,path):
    model.save(path)
    return json.loads((path/'manifest.json').read_text())


def test_exact_29_configs_and_no_duplicate_filler():
    from treeneuron.research.p13_simple import p13_configurations
    rows=p13_configurations(dict(columns=['x'],categorical=[]))
    expected=[config('mean',schema=dict(columns=['x'],categorical=[]))]
    expected += [config('ridge',a,schema=dict(columns=['x'],categorical=[])) for a in ALPHAS]
    expected += [config('spline',a,k,schema=dict(columns=['x'],categorical=[])) for k in (4,8,16) for a in ALPHAS]
    assert rows==expected and len({json.dumps(r,sort_keys=True) for r in rows})==29


@pytest.mark.parametrize('bad',[0.,-1.,True,'1',float('nan'),float('inf'),.3])
def test_alpha_outside_exact_support_fails(tmp_path,bad):
    with pytest.raises(ValueError):fit(request(tmp_path/'run',config(alpha=bad)))


def test_schema_is_explicit_not_a_feature_name_blacklist(tmp_path):
    X=pd.DataFrame({'target':[1.,2.,3.],'row_id':[3.,2.,4.]})
    cfg=config(schema=dict(columns=list(X),categorical=[]));model=fit(request(tmp_path/'run',cfg,X))
    assert model.predict(X).shape==(3,)
    with pytest.raises(ValueError):model.predict(X[['row_id','target']])
    with pytest.raises(ValueError):model.predict(X.assign(extra=1.))


@pytest.mark.parametrize('method',['mean','ridge','spline'])
def test_original_units_batch_invariance_and_no_G_access(tmp_path,method):
    req=request(tmp_path/'run',config(method));before=req.X_fit.copy();model=fit(req)
    prediction=model.predict(req.X_fit)
    np.testing.assert_array_equal(prediction,model.predict(req.X_fit,batch_size=1))
    np.testing.assert_array_equal(before,req.X_fit)
    if method=='mean':np.testing.assert_array_equal(prediction,np.full(len(before),req.y_fit.mean()))
    for size in (0,True,1.5):
        with pytest.raises(ValueError):model.predict(before,batch_size=size)
    summary=model.resource_summary()
    assert summary['capabilities']['selection_labels_used'] is False
    assert summary['capabilities']['dtype']=='float64'
    assert summary['full_workload_qualified'] is False


def test_typed_categories_unknown_and_F_only_geometry(tmp_path):
    X=pd.DataFrame({'x':[1.,np.nan,3.,5.,7.], 'kind':pd.Series([1,'1',True,1.,None],dtype=object)})
    schema=dict(columns=['x','kind'],categorical=['kind'])
    req=request(tmp_path/'run',config(schema=schema),X,np.array([8.,10.,12.,14.,16.]));before=X.copy(deep=True)
    model=fit(req);state=manifest(model,tmp_path/'model');cols=state['preprocessing']['columns']
    assert cols[0]['fill']==4. and cols[0]['indicator'] is True
    assert len(cols[1]['categories'])==4 and cols[1]['unknown_index']==4
    assert model.resource_summary()['encoded_width']==7
    query=pd.DataFrame({'x':[100.,100.],'kind':['unseen','__unknown__']})
    np.testing.assert_array_equal(model.predict(query)[:1],model.predict(query)[1:])
    pd.testing.assert_frame_equal(before,X)


@pytest.mark.parametrize('method',['mean','ridge','spline'])
def test_constants_missing_and_all_categorical_are_finite(tmp_path,method):
    X=pd.DataFrame({'kind':['a','b','a',None]})
    cfg=config(method,schema=dict(columns=['kind'],categorical=['kind']))
    model=fit(request(tmp_path/'run',cfg,X,np.full(4,7.)))
    np.testing.assert_allclose(model.predict(X),7.,atol=1e-12,rtol=0)
    state=manifest(model,tmp_path/'model')
    assert state['basis']['knots']==[]


def test_splines_never_expand_categorical_indicators(tmp_path):
    X=pd.DataFrame({'x':[0.,1.,2.,3.,4.,5.],'kind':['a','b','a','b','a','b']})
    model=fit(request(tmp_path/'run',config('spline',schema=dict(columns=list(X),categorical=['kind'])),X))
    state=manifest(model,tmp_path/'model')
    assert len(state['basis']['knots'])==1
    assert model.resource_summary()['basis_width']==1+3+4+3


def test_failed_refit_preserves_previous_predictions_and_paid_failure(tmp_path):
    req=request(tmp_path/'run');model=fit(req);expected=model.predict(req.X_fit)
    bad=request(tmp_path/'failure',X=req.X_fit.copy());bad.X_fit[0,0]=np.inf
    with pytest.raises(ValueError):model.fit(bad)
    np.testing.assert_array_equal(expected,model.predict(req.X_fit))
    receipt=json.loads((tmp_path/'failure/fit-receipt.json').read_text())
    assert receipt['status']!='completed' and receipt['elapsed_ns']>0 and receipt['error_type']=='ValueError'


@pytest.mark.parametrize('method',['mean','ridge','spline'])
def test_strict_snapshot_round_trip_requires_external_digest(tmp_path,method):
    from treeneuron.research.p13_simple import P13Predictor
    req=request(tmp_path/'run',config(method));model=fit(req);path=tmp_path/'model';manifest(model,path)
    digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    loaded=P13Predictor.load(path,expected_manifest_sha256=digest)
    np.testing.assert_array_equal(model.predict(req.X_fit),loaded.predict(req.X_fit))
    with pytest.raises(ValueError):P13Predictor.load(path,expected_manifest_sha256='0'*64)
    (path/'arrays.npz').write_bytes((path/'arrays.npz').read_bytes()+b'x')
    with pytest.raises(ValueError):P13Predictor.load(path,expected_manifest_sha256=digest)


@pytest.mark.parametrize('mutation',['unknown','dtype','width','basis','source','runtime','config'])
def test_rehashed_structured_corruption_is_rejected(tmp_path,mutation):
    from treeneuron.research.p13_simple import P13Predictor
    model=fit(request(tmp_path/'run'));path=tmp_path/'model';state=manifest(model,path)
    if mutation=='unknown':state['unknown']=1
    elif mutation=='dtype':state['arrays']['coefficients']['dtype']='float32'
    elif mutation=='width':state['arrays']['coefficients']['shape']=[1]
    elif mutation=='basis':state['basis']['kind']='quadratic'
    elif mutation=='source':state['source_identity']['sha256']='0'*64
    elif mutation=='runtime':state['runtime_identity']['python']='wrong'
    else:state['config']['parameters']['alpha']=.3
    (path/'manifest.json').write_text(json.dumps(state))
    digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError):P13Predictor.load(path,expected_manifest_sha256=digest)


def test_all29_numeric_configs_match_unchanged_legacy_reference(tmp_path):
    from treeneuron.research.p13_simple import p13_configurations
    from treeneuron.baselines.simple import MeanPredictor,RidgePredictor,SplineRidgePredictor
    rng=np.random.default_rng(31);X=rng.uniform(-1,1,size=(80,2));y=40+X[:,0]*3+X[:,1]**3
    for i,cfg in enumerate(p13_configurations(dict(columns=['x','z'],categorical=[]))):
        req=request(tmp_path/str(i),cfg,X,y);model=fit(req);p=cfg['parameters']
        old=MeanPredictor() if cfg['estimator']=='mean' else RidgePredictor(p['alpha']) if cfg['estimator']=='ridge' else SplineRidgePredictor(p['alpha'],n_knots=p['n_knots'])
        old.fit(req)
        np.testing.assert_allclose(model.predict(X),old.predict(X),atol=1e-11,rtol=1e-12)


def test_fresh_process_deployment_never_imports_torch_node_or_other_methods(tmp_path):
    script=r"""
import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'torch','node','qhoptim','entmax','sklearn','catboost','xgboost','lightgbm'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import hashlib,numpy as np
from pathlib import Path
from treeneuron.baselines.api import FitRequest
from treeneuron.research.p13_simple import P13Predictor,p13_configurations
root=Path(sys.argv[1]);X=np.arange(12,dtype=float).reshape(6,2)
cfg=p13_configurations({'columns':['x','z'],'categorical':[]})[1]
p=P13Predictor().fit(FitRequest(X,np.arange(6,dtype=float),None,None,1,cfg,{},root/'fit'))
p.save(root/'model');digest=hashlib.sha256((root/'model/manifest.json').read_bytes()).hexdigest()
q=P13Predictor.load(root/'model',expected_manifest_sha256=digest)
np.testing.assert_array_equal(p.predict(X),q.predict(X))
assert not any(n.split('.')[0] in {'torch','node'} for n in sys.modules)
"""
    completed=subprocess.run([sys.executable,'-c',script,str(tmp_path)],capture_output=True,text=True)
    assert completed.returncode==0,completed.stdout+completed.stderr


def test_mixed_ridge_matches_independent_original_unit_solve(tmp_path):
    X=pd.DataFrame({'x':[1.,2.,4.,8.],'kind':['a','b','a','b']});y=np.array([12.,15.,13.,23.])
    cfg=config(alpha=1e-2,schema=dict(columns=list(X),categorical=['kind']))
    model=fit(request(tmp_path/'run',cfg,X,y))
    z=(X['x'].to_numpy()-X['x'].mean())/X['x'].std(ddof=0)
    design=np.column_stack([np.ones(4),z,[1.,0.,1.,0.],[0.,1.,0.,1.],np.zeros(4)])
    penalty=np.diag([0.,.01,.01,.01,.01])
    coefficient=np.linalg.solve(design.T@design+penalty,design.T@y)
    np.testing.assert_allclose(model.predict(X),design@coefficient,atol=1e-11,rtol=1e-12)
    query=pd.DataFrame({'x':[3.],'kind':['unseen']})
    expected=np.array([1.,(3.-X['x'].mean())/X['x'].std(ddof=0),0.,0.,1.])@coefficient
    np.testing.assert_allclose(model.predict(query),[expected],atol=1e-11,rtol=1e-12)


@pytest.mark.parametrize('method',['mean','ridge','spline'])
def test_missing_numeric_columns_preserve_legacy_transform_conventions(tmp_path,method):
    from treeneuron.baselines.simple import MeanPredictor,RidgePredictor,SplineRidgePredictor
    X=np.array([[np.nan,2.],[1.,2.],[3.,2.],[4.,2.],[8.,2.]])
    req=request(tmp_path/'run',config(method),X,np.array([9.,12.,14.,11.,20.]));model=fit(req)
    old=MeanPredictor() if method=='mean' else RidgePredictor(1.) if method=='ridge' else SplineRidgePredictor(1.,n_knots=4)
    old.fit(req);query=np.array([[np.nan,np.nan],[7.,2.]])
    np.testing.assert_allclose(model.predict(query),old.predict(query),atol=1e-11,rtol=1e-12)


def test_rehashed_float32_array_rejects_silent_cast(tmp_path):
    from treeneuron.research.p13_simple import P13Predictor
    model=fit(request(tmp_path/'run'));path=tmp_path/'model';state=manifest(model,path)
    with np.load(path/'arrays.npz',allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
    arrays['coefficients']=arrays['coefficients'].astype(np.float32)
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']['coefficients']=dict(dtype='float32',shape=list(arrays['coefficients'].shape),
        sha256=hashlib.sha256(arrays['coefficients'].tobytes()).hexdigest())
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state))
    digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='dtype'):P13Predictor.load(path,expected_manifest_sha256=digest)


def test_failure_receipt_cannot_replace_original_input_error(tmp_path,monkeypatch):
    from treeneuron.research import p13_simple
    req=request(tmp_path/'failure');req.X_fit[0,0]=np.inf
    def disk_full(*args):raise OSError('mock receipt disk full')
    monkeypatch.setattr(p13_simple,'_write',disk_full)
    with pytest.raises(ValueError,match='finite') as caught:fit(req)
    assert any('disk full' in note for note in caught.value.__notes__)


@pytest.mark.parametrize('bad',[True,1+2j,'3',float('inf')])
def test_explicit_numeric_schema_rejects_invalid_values_without_mutating_request(tmp_path,bad):
    X=pd.DataFrame({'x':pd.Series([1.,bad,3.],dtype=object),'z':[1.,2.,3.]})
    with pytest.raises(ValueError):fit(request(tmp_path/'failure',X=X))


def test_constant_and_all_missing_numeric_features_remain_finite(tmp_path):
    X=np.array([[1.,np.nan],[1.,np.nan],[1.,np.nan]])
    model=fit(request(tmp_path/'run',config('spline'),X,np.full(3,4.)))
    np.testing.assert_array_equal(model.predict(X),[4.,4.,4.])
    state=manifest(model,tmp_path/'model')
    assert state['basis']['knots']==[[],[]]
    assert state['preprocessing']['columns'][1]['all_missing']


def test_prediction_allocates_rows_by_categories_not_category_squared_identity(tmp_path,monkeypatch):
    X=pd.DataFrame({'kind':['a','b','c','d']})
    model=fit(request(tmp_path/'run',config(schema=dict(columns=['kind'],categorical=['kind'])),X))
    def no_identity(*args,**kwargs):raise AssertionError('prediction must not allocate a category-squared identity')
    monkeypatch.setattr(np,'eye',no_identity)
    assert model.predict(pd.DataFrame({'kind':['unknown']})).shape==(1,)


def test_saved_constant_numeric_column_cannot_have_interior_knots(tmp_path):
    from treeneuron.research.p13_simple import P13Predictor
    model=fit(request(tmp_path/'run',config('spline'),np.ones((4,2)),np.arange(4,dtype=float)))
    path=tmp_path/'model';state=manifest(model,path)
    with np.load(path/'arrays.npz',allow_pickle=False) as values:arrays={k:values[k].copy() for k in values.files}
    state['basis']['knots']=[[1.],[]];state['basis_width']+=1
    arrays.update(coefficients=np.r_[arrays['coefficients'],0.],knots=np.array([1.]),knot_offsets=np.array([0,1,1],dtype=np.int64))
    np.savez(path/'arrays.npz',**arrays)
    state['arrays']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
    state['array_file']['sha256']=hashlib.sha256((path/'arrays.npz').read_bytes()).hexdigest()
    (path/'manifest.json').write_text(json.dumps(state));digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='knot'):P13Predictor.load(path,expected_manifest_sha256=digest)


def test_invalid_F_identity_container_rejects_as_snapshot_validation_error(tmp_path):
    from treeneuron.research.p13_simple import P13Predictor
    model=fit(request(tmp_path/'run'));path=tmp_path/'model';state=manifest(model,path)
    state['F_identity']=[];(path/'manifest.json').write_text(json.dumps(state))
    digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    with pytest.raises(ValueError):P13Predictor.load(path,expected_manifest_sha256=digest)


@pytest.mark.parametrize('changed',['RECORD','_linalg.py'])
def test_numerical_runtime_rejects_changed_pinned_record_or_python_source(monkeypatch,changed):
    from treeneuron.research import p13_simple
    original=p13_simple._sha
    def changed_bytes(path):
        path=Path(path)
        if path.name==changed:return '0'*64
        return original(path)
    monkeypatch.setattr(p13_simple,'_sha',changed_bytes)
    with pytest.raises(ValueError):p13_simple._runtime()
