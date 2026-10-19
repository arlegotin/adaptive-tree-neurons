"""Explicit recovery route reuses role-safe engineering input/reload checks."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_method_worker import request_fixture, write_request


def test_recovery_request_is_NODE_only_and_cannot_disable_legacy_budget(tmp_path):
    from treeneuron.research.method_worker import _validate_request
    path, value = request_fixture(tmp_path/'input')
    value['budget']={'native_stopping': True}
    digest=write_request(path,value)
    with pytest.raises(ValueError):_validate_request(path,digest)
    with pytest.raises(ValueError):_validate_request(path,digest,recoverable_training=True)


@pytest.mark.parametrize('budget', [{'native_stopping': True}, {'max_updates':3}])
def test_explicit_recovery_route_accepts_native_stopping_or_bounded_proof(tmp_path,budget):
    from treeneuron.research.method_worker import _validate_request
    path, value = request_fixture(tmp_path/'input')
    value['family_id']='P12'
    value['config']=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',
        execution='bounded_memory_v1',feature_schema=value['config']['feature_schema'],parameters={})
    value['budget']=budget
    digest=write_request(path,value)
    parsed,_=_validate_request(path,digest,recoverable_training=True)
    assert parsed['budget']==budget
    with pytest.raises(ValueError):_validate_request(path,digest)


def test_fresh_worker_checkpoint_path_retains_closed_model_and_exact_predictions(tmp_path):
    path,value=request_fixture(tmp_path/'input')
    value['family_id']='P12';value['budget']={'max_updates':3}
    value['config']=dict(schema='research/v2-node-config',family='P12',mode='engineering_fixture',
        execution='bounded_memory_v1',feature_schema=value['config']['feature_schema'],parameters={})
    digest=write_request(path,value)
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1';env.pop('KMP_DUPLICATE_LIB_OK',None)
    env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'src')
    common=[sys.executable,'-B','-m','treeneuron.research.method_worker','--request',str(path),
        '--expected-request-sha256',digest,'--training-checkpoints',str(tmp_path/'checkpoints')]
    first=subprocess.run(common+['--output',str(tmp_path/'first')],env=env,capture_output=True,text=True,timeout=45)
    assert first.returncode==0,first.stdout+first.stderr
    second=subprocess.run(common+['--resume','--output',str(tmp_path/'resumed')],env=env,capture_output=True,text=True,timeout=45)
    assert second.returncode==0,second.stdout+second.stderr
    terminal=json.loads((tmp_path/'resumed/terminal.json').read_text())
    assert terminal['status']=='completed_ineligible' and terminal['snapshot_reload_exact']
    assert terminal['quality_eligible'] is False and terminal['T_accessed'] is False
    assert json.loads((tmp_path/'resumed/fit/fit-receipt.json').read_text())['resume_step']==3
