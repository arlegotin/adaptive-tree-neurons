"""Actual preflight precedes caller output; prespawn denials fit nothing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_caller_v2'), 'Missing owned primary caller composition'
    from treeneuron.research import primary_caller_v2
    return primary_caller_v2


def test_public_caller_rejects_false_registry_before_any_output_or_plan_cache(tmp_path):
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        api().run_primary_search_episode(practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan'),sha256='a'*64),
            expected_practical_plan_hash='b'*64,job_id='airfoil/outer-0/repeat-0/P13',
            support_reference=dict(path=str(tmp_path/'forbidden-support'),sha256='c'*64),
            qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='d'*64),
            resource_forecast_reference=dict(path=str(tmp_path/'forbidden-forecast'),sha256='e'*64),
            episode_root=tmp_path/'episode',search_directory=tmp_path/'search')
    assert 'forbidden-plan' not in str(error.value) and 'forbidden-forecast' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


def test_prespawn_denial_is_measured_without_calling_Popen(tmp_path,monkeypatch):
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    import subprocess
    def forbidden(*args,**kwargs):raise AssertionError('Denied admission cannot start any process')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    root=tmp_path/'process'
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-c','raise AssertionError()'],cwd=str(tmp_path),
        env_overrides={},output=root,identity=dict(fixture='prespawn'),admission_identity=dict(policy='software'),
        admission_check=lambda:dict(admitted=False,reason='known RAM shortage'),rss_pause_bytes=128*1024**2,admit_before_spawn=True)
    assert result['execution_status']=='resource_exhausted' and result['stop_reason']=='prespawn_admission_denied'
    assert result['child_started'] is False and result['grant_publication_started'] is False
    sha=lambda name:hashlib.sha256((root/name).read_bytes()).hexdigest()
    proof=inspect_cooperative_process(root=root,expected_receipt_sha256=sha('process.json'),expected_started_sha256=sha('started.json'))
    assert proof['root_process_clock_authenticated'] and proof['elapsed_ns']>0


def test_primary_metadata_command_can_be_denied_before_spawn_and_paid_as_unstarted(tmp_path):
    api()
    from treeneuron.research.primary_search_worker_v2 import primary_worker_identity,_episode_environment,_unstarted_episode_state
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    value=dict(schema='research/v2-primary-search-resource-episode-request',source_sha256=_source(),
        practical_plan_reference=dict(path=str(tmp_path/'not-opened-plan'),sha256='a'*64),expected_practical_plan_hash='b'*64,
        job_id='airfoil/outer-0/repeat-0/P13',support_reference=dict(path=str(tmp_path/'not-opened-support'),sha256='c'*64),
        qualification_reference=dict(root=str(tmp_path/'not-opened-registry'),inventory_sha256='d'*64),
        resource_forecast_reference=dict(path=str(tmp_path/'not-opened-forecast'),sha256='e'*64),search_directory=str(tmp_path/'not-created-search'),
        rss_limit_bytes=4*1024**3,previous_episodes=[],previous_unstarted_episodes=[])
    path=tmp_path/'request.json';path.write_text(json.dumps(value));sha=hashlib.sha256(path.read_bytes()).hexdigest();root=tmp_path/'process'
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(path),'--expected-request-sha256',sha,'--output',str(tmp_path/'uncreated-worker')],
        cwd=str(Path(__file__).resolve().parents[1]),env_overrides=_episode_environment(),output=root,
        identity=primary_worker_identity(path,sha),admission_identity=dict(policy='software'),
        admission_check=lambda:dict(admitted=False),rss_pause_bytes=256*1024**2,admit_before_spawn=True)
    value['previous_unstarted_episodes']=[dict(process_root=str(root),process_receipt_sha256=hashlib.sha256((root/'process.json').read_bytes()).hexdigest(),
        process_started_sha256=hashlib.sha256((root/'started.json').read_bytes()).hexdigest())]
    known=_unstarted_episode_state(value)
    assert known['elapsed_ns']==result['elapsed_ns'] and known['episodes']==1 and known['model_attempts_paid']==0
    assert result['child_started'] is False and not (tmp_path/'not-created-search').exists()


def prepared(root,script):
    value=dict(schema='software-caller-only',source_sha256='a'*64);raw=(json.dumps(value)+'\n').encode()
    return dict(request_bytes=raw,request_sha256=hashlib.sha256(raw).hexdigest(),
        argv=[sys.executable,'-P','-B',str(script)],cwd=str(root.parent),
        environment_overrides=dict(PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'),PYTHONDONTWRITEBYTECODE='1'))


@pytest.fixture(scope='module')
def caller_episodes(tmp_path_factory):
    module=api();root=tmp_path_factory.mktemp('primary-caller-owned');rows={}
    for name in ('normal','denied','changed','failed','missingbody'):
        script=root/f'{name}.py'
        script.write_text("from treeneuron.research.cooperative_process_v2 import cooperative_worker\nwith cooperative_worker(identity=dict(fixture='caller')):\n    "+("raise RuntimeError('fixture failure')" if name=='failed' else "print('owned software worker')")+'\n')
        count=0
        def admission():
            nonlocal count
            count+=1
            return dict(admitted=name!='denied' and (name!='changed' or count==1),reason='software fixture')
        episode=root/name;value=prepared(episode,script)
        result=module._run_prepared_episode(prepared=value,episode_root=episode,identity=dict(fixture='caller'),
            admission_identity=dict(policy='software'),admission_check=admission,invocation_origin_ns=time.perf_counter_ns(),
            rss_limit_bytes=256*1024**2,expect_primary_body=name=='missingbody')
        rows[name]=dict(root=episode,result=result,admission_calls=count)
    return rows


def test_actual_compositor_checks_resources_twice_and_joins_success(caller_episodes):
    row=caller_episodes['normal'];result=row['result'];root=row['root']
    assert row['admission_calls']==2 and result['status']=='closed_generic_worker'
    assert result['observed_caller_elapsed_ns']>=result['cooperative_root_elapsed_ns']>0
    assert result['additional_caller_interval_ns']>=0
    assert result['whole_training_procedure_cost_authenticated'] is result['quality_eligible'] is False
    assert json.loads((root/'process/process.json').read_text())['child_reaped']


@pytest.mark.parametrize('name,started,calls',[('denied',False,1),('changed',True,2)])
def test_denial_before_or_after_spawn_never_authorizes_fitting(caller_episodes,name,started,calls):
    row=caller_episodes[name];result=row['result']
    assert result['status']=='closed_unstarted_episode' and result['child_started'] is started
    assert row['admission_calls']==calls and 'unstarted_reference' in result
    assert not (row['root']/'worker').exists()


@pytest.mark.parametrize('name',['failed','missingbody'])
def test_authorized_failure_or_missing_primary_outcome_has_no_unstarted_resume(caller_episodes,name):
    result=caller_episodes[name]['result']
    assert result['status']=='failed_authorized_body' and 'unstarted_reference' not in result


def test_reader_rejects_admitted_prespawn_record_with_denial_disposition(caller_episodes,tmp_path):
    from treeneuron.research.cooperative_process_v2 import inspect_cooperative_process
    root=tmp_path/'changed-record';shutil.copytree(caller_episodes['denied']['root']/'process',root)
    path=root/'pre-spawn-admission.json';value=json.loads(path.read_bytes());value['admitted']=True;path.write_text(json.dumps(value))
    receipt_path=root/'process.json';receipt=json.loads(receipt_path.read_bytes())
    for ref in receipt['files']:
        if ref['path']==path.name:ref.update(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):inspect_cooperative_process(root=root,
        expected_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        expected_started_sha256=hashlib.sha256((root/'started.json').read_bytes()).hexdigest())


@pytest.mark.parametrize('origin',[True,-1,10**30])
def test_invalid_caller_origin_rejects_before_request_or_output(tmp_path,origin):
    module=api();root=tmp_path/'episode';script=tmp_path/'unread-worker.py'
    with pytest.raises(ValueError):module._run_prepared_episode(prepared=prepared(root,script),episode_root=root,
        identity=dict(fixture='invalid-origin'),admission_identity=dict(policy='software'),
        admission_check=lambda:dict(admitted=False),invocation_origin_ns=origin,rss_limit_bytes=256*1024**2)
    assert not root.exists()
