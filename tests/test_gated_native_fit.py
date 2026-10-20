"""Actual own-family F/G fitting and paired growth; no native quality admission."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def run_native(tmp_path,body):
    assert importlib.util.find_spec('treeneuron.research.gated_native') is not None, 'Missing v2 own-family fit adapter'
    script=tmp_path/'fixture.py'
    prefix='''import json, hashlib, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from treeneuron.baselines.api import FitRequest
from treeneuron.research.gated_native import GatedPredictor
root=Path(__file__).parent
rng=np.random.default_rng(17)
F=pd.DataFrame(rng.normal(size=(37,21)),columns=[f'x{i}' for i in range(20)]+['kind'])
F['kind']=np.arange(37)%2
G=pd.DataFrame(rng.normal(size=(17,21)),columns=F.columns)
G['kind']=np.arange(17)%3
Y=np.sin(F.x0.to_numpy())+.2*F.x1.to_numpy()
GY=np.sin(G.x0.to_numpy())+.2*G.x1.to_numpy()
schema=dict(columns=list(F.columns),categorical=['kind'])
def config(family,variant=None):
    cfg=dict(schema='research/v2-gated-native-config',family=family,mode='engineering_fixture',feature_schema=schema,parameters={'width':8},base_updates=3,continuation_updates=3,validation_interval=1,patience=2)
    if variant is not None:cfg['variant']=variant
    return cfg
def fit(family,name,variant=None):
    return GatedPredictor(family).fit(FitRequest(F,Y,G,GY,17,config(family,variant),{},root/name))
'''
    script.write_text(prefix+body)
    # Match the production fresh-worker policy; an upstream parent import may
    # set this default, which the native adapter correctly rejects on entry.
    env=dict(os.environ)
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    child=subprocess.run([sys.executable,'-B',str(script)],env=env,capture_output=True,text=True,timeout=120)
    (tmp_path/'stdout.log').write_text(child.stdout);(tmp_path/'stderr.log').write_text(child.stderr)
    assert child.returncode==0,child.stderr
    return json.loads(child.stdout)


def test_selected_and_random_fit_the_identical_parent_and_candidate_table_without_G_preprocessing(tmp_path):
    result=run_native(tmp_path,'''
a=fit('P00','selected');b=fit('P01','random')
sa=a.resource_summary();sb=b.resource_summary()
assert sa['parent_selected_state_sha256']==sb['parent_selected_state_sha256']
assert sa['candidate_table_sha256']==sb['candidate_table_sha256']
assert sa['effective_count']==sb['effective_count']<=2
assert sa['encoded_width']==sb['encoded_width']==23
assert sa['preprocessing']['columns'][-1]['categories']==['int:0','int:1']
assert sa['preprocessing']['columns'][0]['mean']==float(F.x0.mean())
assert sa['total_parameters']==8*25+1+2*sa['effective_count']*24
assert sa['quality_eligible'] is False and sa['checkpoint_eligible'] is False
assert sa['accessed_roles']==['F','G'] and sa['persistent_training_resume'] is False
assert len(sa['stages'])==2 and sa['growth_function_preserved_exact'] is True
with np.load(root/'selected/candidate-table.npz',allow_pickle=False) as x,np.load(root/'random/candidate-table.npz',allow_pickle=False) as y:
    assert set(x.files)==set(y.files)
    for key in x.files:assert np.array_equal(x[key],y[key],equal_nan=True)
with np.load(root/'selected/selected-state.npz',allow_pickle=False) as weights:
    assert set(weights.files)
assert np.isfinite(a.predict(G)).all()
print(json.dumps(dict(encoded_width=sa['encoded_width'],effective_count=sa['effective_count'])))
''')
    assert result['encoded_width']==23


@pytest.mark.parametrize('variant',['lower','upper'])
def test_ordinary_adapter_uses_actual_eligible_parameter_bracket_and_complete_parent(variant,tmp_path):
    r=run_native(tmp_path,f'''
g=fit('P00','gated');w=fit('P02','warm','{variant}')
a=g.resource_summary();b=w.resource_summary()
assert a['parent_selected_state_sha256']==b['parent_selected_state_sha256']
assert a['effective_count']==b['effective_count']
assert b['growth_function_preserved_exact'] and b['total_parameters']==b['ordinary_width']*25+1
assert (b['total_parameters']<=a['total_parameters']) if '{variant}'=='lower' else (b['total_parameters']>=a['total_parameters'])
assert all(p['age']==p['selected_absolute_update']-p['birth_update'] for p in b['parameter_ages'])
print(json.dumps(dict(width=b['ordinary_width'])))
''')
    assert r['width']>=8


@pytest.mark.parametrize('variant',['centered_partial','centered_full','direct_partial','direct_full'])
def test_static_adapter_runs_both_declared_stages_and_exact_capacity(variant,tmp_path):
    result=run_native(tmp_path,f'''
a=fit('P05','static','{variant}');s=a.resource_summary()
count=8 if '{variant}'.endswith('_full') else 2
assert s['total_parameters']==8*25+1+2*count*24
assert s['candidate_table_sha256'] is None and len(s['stages'])==2
assert s['effective_count']==count and s['encoded_width']==23
assert np.isfinite(a.predict(G)).all()
print(json.dumps(dict(parameters=s['total_parameters'])))
''')
    assert result['parameters']>201


def test_fit_preserves_callers_random_streams_and_rejects_repeat_fit(tmp_path):
    run_native(tmp_path,'''
random.seed(93);np.random.seed(93);torch.manual_seed(93)
py=random.getstate();n=np.random.get_state();t=torch.get_rng_state().clone()
a=fit('P00','own')
assert random.getstate()==py and np.array_equal(np.random.get_state()[1],n[1]) and torch.equal(torch.get_rng_state(),t)
try:a.fit(FitRequest(F,Y,G,GY,17,config('P00'),{},root/'forbidden'))
except RuntimeError:pass
else:raise AssertionError('second fit accepted')
assert not (root/'forbidden').exists()
print('{}')
''')


def test_high_cardinality_cannot_escape_encoded_workload_limit(tmp_path):
    run_native(tmp_path,'''
X=pd.DataFrame({'x':np.arange(130),'kind':np.arange(130)})
cfg=config('P00');cfg['feature_schema']=dict(columns=['x','kind'],categorical=['kind'])
try:GatedPredictor('P00').fit(FitRequest(X,np.arange(130.),X[:2],np.arange(2.),17,cfg,{},root/'too_wide'))
except ValueError:pass
else:raise AssertionError('unqualified encoded geometry accepted')
receipt=json.loads((root/'too_wide/attempt.json').read_text())
assert receipt['status']=='input_or_dependency_failure' and receipt['quality_eligible'] is False
assert not (root/'too_wide/events.jsonl').exists()
print('{}')
''')
