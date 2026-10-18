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
