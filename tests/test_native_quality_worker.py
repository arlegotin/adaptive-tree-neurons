"""Explicit native budgets do not become hidden time caps or quality authority."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_method_worker import api,request_fixture,sha,write_request


@pytest.mark.parametrize('family',['P13','P03'])
def test_fresh_native_quality_budget_reaches_the_real_adapter_without_a_wall_cap(tmp_path,family):
    path,value=request_fixture(tmp_path/'input')
    value['budget']={'native_stopping':True}
    if family=='P03':
        from test_ordinary_neural_contract import config
        value['family_id']='P03';value['config']=config('P03','engineering_fixture',max_epochs=2,patience=2)
    expected=write_request(path,value);output=tmp_path/'worker'
    root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,'-B','-m','treeneuron.research.method_worker','--request',str(path),
        '--expected-request-sha256',expected,'--output',str(output)]
    result=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    (tmp_path/'command.json').write_text(json.dumps({'argv':argv,'returncode':result.returncode})+'\n')
    (tmp_path/'worker.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['status']=='completed_ineligible' and terminal['snapshot_reload_exact'] is True
    assert terminal['budget_resolution']==dict(requested={'native_stopping':True},adapter={},policy='native_stopping')
    assert terminal['quality_eligible'] is terminal['checkpoint_eligible'] is terminal['T_accessed'] is False
    assert terminal['accessed_roles']==['F_labels','G_labels','V_features']
    with np.load(output/'predictions.npz',allow_pickle=False) as archive:
        assert archive['prediction'].shape==(5,) and np.isfinite(archive['prediction']).all()
        if family=='P13':np.testing.assert_array_equal(archive['prediction'],np.full(5,14.))
    if family=='P03':
        manifest=json.loads((output/'fit/input-manifest.json').read_text())
        assert manifest['budget']=={} and manifest['config']['max_epochs']==2


@pytest.mark.parametrize('budget',[{'native_stopping':False},{'native_stopping':1},
    {'native_stopping':True,'max_seconds':7200},{}])
def test_implicit_or_mixed_native_budgets_fail_before_any_adapter_import(tmp_path,budget):
    path,value=request_fixture(tmp_path/'input');value['budget']=budget
    expected=write_request(path,value)
    with pytest.raises(ValueError):api().run_engineering_trial(path,expected_request_sha256=expected,output=tmp_path/'worker')
    terminal=json.loads((tmp_path/'worker/terminal.json').read_text())
    assert terminal['adapter_import_started'] is False and terminal['quality_eligible'] is False


def test_native_budget_does_not_grant_non_NODE_training_checkpoint_support(tmp_path):
    path,value=request_fixture(tmp_path/'input');value['budget']={'native_stopping':True}
    expected=write_request(path,value)
    with pytest.raises(ValueError):
        api().run_engineering_trial(path,expected_request_sha256=expected,output=tmp_path/'worker',training_checkpoints=tmp_path/'checkpoints')
    assert not (tmp_path/'checkpoints').exists()


def test_unbounded_NODE_still_requires_its_explicit_recoverable_checkpoint_route(tmp_path):
    path,value=request_fixture(tmp_path/'input');value['family_id']='P12'
    value['config']=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',
        execution='bounded_memory_v1',feature_schema=value['config']['feature_schema'],parameters={})
    value['budget']={'native_stopping':True};expected=write_request(path,value)
    with pytest.raises(ValueError):api()._validate_request(path,expected)
    parsed,_=api()._validate_request(path,expected,recoverable_training=True)
    assert parsed['budget']=={'native_stopping':True}
