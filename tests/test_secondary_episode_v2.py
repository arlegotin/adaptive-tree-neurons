"""Strict secondary request lineage cannot bypass actual global qualification."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.secondary_episode_v2'),'Missing owned secondary episode route'
    from treeneuron.research import secondary_episode_v2
    return secondary_episode_v2


def request(tmp_path):
    ref=lambda name,digest:dict(path=str(tmp_path/name),sha256=digest*64)
    return dict(schema='research/v2-secondary-program-episode-request',source_sha256='a'*64,
        practical_plan_reference=ref('forbidden-plan.json','b'),expected_practical_plan_hash='c'*64,
        job_id='energy/outer-1/repeat-0/P13',support_reference=ref('forbidden-support.json','d'),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='e'*64),
        global_barrier_reference=ref('forbidden-barrier.json','f'),resource_forecast_reference=ref('forbidden-forecast.json','1'),
        procedure_reference=ref('forbidden-procedure.json','2'),search_directory=str(tmp_path/'search'),rss_limit_bytes=4*1024**3,
        previous_episodes=[],previous_unstarted_episodes=[])


def test_secondary_bootstrap_is_strict_and_has_distinct_identity_without_numerical_imports(tmp_path):
    api()
    from treeneuron.research.primary_search_worker_v2 import _validate_request,primary_worker_identity
    value=request(tmp_path);assert _validate_request(value)==value
    path=tmp_path/'request.json';path.write_text(json.dumps(value));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert primary_worker_identity(path,digest)==dict(purpose='secondary_search_episode',request_sha256=digest,source_sha256='a'*64)
    code='import sys,json; from treeneuron.research.primary_search_worker_v2 import primary_worker_identity; primary_worker_identity(sys.argv[1],sys.argv[2]); print(json.dumps(sorted(set(sys.modules)&{"numpy","torch","scipy","pandas"})))'
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTHONDONTWRITEBYTECODE='1')
    result=subprocess.run([sys.executable,'-P','-B','-c',code,str(path),digest],env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[]


@pytest.mark.parametrize('field',['global_barrier_reference','resource_forecast_reference','procedure_reference'])
def test_secondary_request_cannot_omit_its_global_resource_or_procedure_pin(tmp_path,field):
    api();from treeneuron.research.primary_search_worker_v2 import _validate_request
    value=request(tmp_path);del value[field]
    with pytest.raises(ValueError):_validate_request(value)


def test_primary_request_cannot_acquire_secondary_permission_by_adding_a_barrier(tmp_path):
    api();from treeneuron.research.primary_search_worker_v2 import _validate_request
    value=request(tmp_path);value['schema']='research/v2-primary-program-episode-request'
    with pytest.raises(ValueError):_validate_request(value)


def test_secondary_previous_command_is_bound_to_the_same_global_barrier(tmp_path):
    api()
    from treeneuron.research.primary_search_worker_v2 import _check_episode_command,_job_binding,_episode_environment,primary_worker_identity,_ROOT
    value=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(value));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    start=dict(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2','--request',str(path),
        '--expected-request-sha256',digest,'--output',str(tmp_path/'worker')],cwd=str(_ROOT),environment_overrides=_episode_environment(),
        identity=primary_worker_identity(path,digest))
    assert _check_episode_command(start,_job_binding(value))==value
    changed=copy.deepcopy(value);changed['global_barrier_reference']['sha256']='0'*64
    with pytest.raises(ValueError):_check_episode_command(start,_job_binding(changed))


def test_secondary_worker_preparation_reopens_actual_global_gate_before_other_pins_or_output(tmp_path):
    api()
    from treeneuron.research.primary_search_worker_v2 import _prepare_qualified_episode
    with pytest.raises((ValueError,FileNotFoundError)) as error:_prepare_qualified_episode(request=request(tmp_path),output=tmp_path/'worker')
    assert not any(name in str(error.value) for name in ('forbidden-plan','forbidden-support','forbidden-procedure','forbidden-forecast'))
    assert list(tmp_path.iterdir())==[]


def test_secondary_caller_preflight_cannot_create_outputs_under_false_global_qualification(tmp_path):
    module=api();value=request(tmp_path)
    with pytest.raises((ValueError,FileNotFoundError)):
        module.prepare_secondary_invocation(barrier_reference=value['global_barrier_reference'],qualification_reference=value['qualification_reference'],
            resource_forecast_reference=value['resource_forecast_reference'],job_id=value['job_id'],request_path=tmp_path/'request.json',
            worker_output=tmp_path/'worker',control_output=tmp_path/'control',search_directory=tmp_path/'search')
    assert list(tmp_path.iterdir())==[]


def test_real_cold_secondary_worker_closes_a_failed_global_gate_without_staging_or_search(tmp_path):
    api()
    from treeneuron.research.primary_search_worker_v2 import primary_worker_identity,_episode_environment,_ROOT
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    value=request(tmp_path);path=tmp_path/'request.json';path.write_text(json.dumps(value));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2','--request',str(path),
        '--expected-request-sha256',digest,'--output',str(tmp_path/'worker')],cwd=str(_ROOT),env_overrides=_episode_environment(),
        output=tmp_path/'process',identity=primary_worker_identity(path,digest),admission_identity=dict(policy='software-only global-gate negative'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=512*1024**2)
    assert result['returncode']==1 and result['worker_protocol_closed'] and result['child_reaped']
    body=json.loads((tmp_path/'process/body-outcome.json').read_text())
    assert body['schema']=='research/v2-secondary-search-body-outcome' and body['status']=='body_failed'
    assert not any(name in body['traceback'] for name in ('forbidden-plan','forbidden-support','forbidden-procedure','forbidden-forecast'))
    assert body['quality_eligible'] is body['protected_access_authorized'] is body['T_accessed'] is False
    assert not (tmp_path/'worker').exists() and not (tmp_path/'search').exists()


@pytest.mark.parametrize('field',['source_sha256','practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference','qualification_reference'])
def test_pure_request_pin_join_refuses_changed_global_duplicates_without_granting_a_capability(tmp_path,field):
    module=api();value=request(tmp_path)
    names=['source_sha256','practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference','qualification_reference']
    payload={name:copy.deepcopy(value[name]) for name in names}
    assert module._check_request_binding(value,payload) is None
    changed=copy.deepcopy(value)
    if isinstance(changed[field],dict):changed[field][next(key for key in changed[field] if key.endswith('sha256'))]='0'*64
    else:changed[field]='0'*64
    with pytest.raises(ValueError):module._check_request_binding(changed,payload)


def test_secondary_resource_gate_freezes_its_binding_and_pauses_same_job_on_live_memory_changes(tmp_path):
    api()
    from test_native_resource_forecast_v2 import study
    from treeneuron.research.native_resource_forecast_v2 import _forecast_from_observations
    from treeneuron.research.primary_search_worker_v2 import _make_resource_gate
    plan,observations=study.__wrapped__(tmp_path)
    forecast=_forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['tranche']=='secondary')
    context=dict(forecast=forecast,job=job,reference=dict(path=str(tmp_path/'not-opened.json'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'not-opened-global.json'),sha256='b'*64))
    memory=[2*1024**3,1,2*1024**3]
    def observe(path):return dict(available_memory_bytes=memory.pop(0),free_disk_bytes=2*1024**3,observed_utc_ns=1,memory_basis='software fixture')
    gate,identity=_make_resource_gate(context,search_directory=tmp_path/'search',observe=observe)
    context['global_barrier_reference']['sha256']='c'*64
    rows=[gate() for _ in range(3)]
    assert [r['admitted'] for r in rows]==[True,False,True] and {r['job_id'] for r in rows}=={job['job_id']}
    assert identity['global_barrier_reference']['sha256']=='b'*64
    assert all(r['execution_authorized'] is r['protected_access_authorized'] is False for r in rows)
