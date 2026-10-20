"""Caller completion intervals are observed costs, never new native authority."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import pytest


def api():
    from treeneuron.research import primary_caller_v2
    assert hasattr(primary_caller_v2,'inspect_caller_boundary'), 'Missing caller completion boundary reader'
    return primary_caller_v2


@pytest.fixture(scope='module')
def episode(tmp_path_factory):
    module=api()
    from treeneuron.research.primary_search_worker_v2 import _episode_environment
    from treeneuron.research.deployment_worker_v2 import _source
    root=tmp_path_factory.mktemp('caller-boundary');directory=root/'episode'
    request=dict(schema='research/v2-primary-search-resource-episode-request',source_sha256=_source(),
        practical_plan_reference=dict(path=str(root/'not-opened-plan'),sha256='a'*64),expected_practical_plan_hash='b'*64,
        job_id='airfoil/outer-0/repeat-0/P13',support_reference=dict(path=str(root/'not-opened-support'),sha256='c'*64),
        qualification_reference=dict(root=str(root/'not-opened-registry'),inventory_sha256='d'*64),
        resource_forecast_reference=dict(path=str(root/'not-opened-forecast'),sha256='e'*64),search_directory=str(root/'not-created-search'),
        rss_limit_bytes=4*1024**3,previous_episodes=[],previous_unstarted_episodes=[])
    raw=(json.dumps(request)+'\n').encode();sha=hashlib.sha256(raw).hexdigest()
    prepared=dict(request_bytes=raw,request_sha256=sha,argv=[sys.executable,'-P','-B','-m',
        'treeneuron.research.primary_search_worker_v2','--request',str(directory/'request.json'),
        '--expected-request-sha256',sha,'--output',str(directory/'worker')],
        cwd=str(Path(__file__).resolve().parents[1]),environment_overrides=_episode_environment())
    result=module._run_prepared_episode(prepared=prepared,episode_root=directory,
        identity=dict(purpose='primary_search_episode',request_sha256=sha,source_sha256=request['source_sha256']),
        admission_identity=dict(policy='software no-child fixture'),admission_check=lambda:dict(admitted=False),
        invocation_origin_ns=time.perf_counter_ns(),rss_limit_bytes=256*1024**2,expect_primary_body=True)
    return dict(root=directory,request=request,result=result)


def test_complete_caller_interval_is_paid_before_the_eventual_first_fit(episode,monkeypatch):
    module=api()
    from treeneuron.research.primary_search_worker_v2 import _unstarted_episode_state,_previous_episode_state
    import subprocess
    def forbidden(*args,**kwargs):raise AssertionError('Boundary reader cannot spawn or probe a saved PID')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    reference=episode['result']['unstarted_reference']
    assert 'caller_reference' in reference
    proof=module.inspect_caller_boundary(reference['caller_reference'],process_reference=reference)
    assert proof['elapsed_ns']>proof['root_elapsed_ns']>0
    assert proof['additional_caller_interval_ns']==proof['elapsed_ns']-proof['root_elapsed_ns']
    assert proof['whole_training_procedure_cost_authenticated'] is proof['native_qualification_authenticated'] is False
    request=copy.deepcopy(episode['request']);request['previous_unstarted_episodes']=[reference]
    known=_unstarted_episode_state(request);prior=_previous_episode_state(request)
    assert known['elapsed_ns']==proof['elapsed_ns']==prior['previous_outer_tail_ns']
    assert known['root_only_episodes']==0 and known['caller_boundaries_checked']==1
    assert not Path(request['search_directory']).exists()


def test_legacy_root_scope_remains_explicit_instead_of_assigning_zero_caller_cost(episode):
    api()
    from treeneuron.research.primary_search_worker_v2 import _unstarted_episode_state
    reference=copy.deepcopy(episode['result']['unstarted_reference']);reference.pop('caller_reference')
    request=copy.deepcopy(episode['request']);request['previous_unstarted_episodes']=[reference]
    known=_unstarted_episode_state(request)
    assert known['root_only_episodes']==1 and known['caller_boundaries_checked']==0
    assert known['whole_training_procedure_cost_authenticated'] is False


def test_completion_clock_includes_episode_reference_hashing(tmp_path,monkeypatch):
    module=api()
    from test_primary_caller_v2 import prepared
    original=module._file_reference
    def delayed(path):
        if Path(path).name in ('request.json','caller-start.json','episode.json'):time.sleep(.025)
        return original(path)
    monkeypatch.setattr(module,'_file_reference',delayed)
    root=tmp_path/'episode'
    result=module._run_prepared_episode(prepared=prepared(root,tmp_path/'unused-script.py'),episode_root=root,
        identity=dict(fixture='hash-cost'),admission_identity=dict(policy='software'),admission_check=lambda:dict(admitted=False),
        invocation_origin_ns=time.perf_counter_ns(),rss_limit_bytes=256*1024**2)
    complete=json.loads((root/'caller-complete.json').read_bytes())
    assert complete['elapsed_ns']-result['observed_caller_elapsed_ns']>=75_000_000


def test_preflight_protects_the_complete_immutable_prior_caller_root(episode):
    from treeneuron.research.primary_invocation_v2 import _prior_input_locations
    roots=_prior_input_locations([episode['result']['unstarted_reference']])
    assert episode['root'] in roots


@pytest.mark.parametrize('target,fault',[('caller-start.json','schema'),('caller-start.json','extra'),('episode.json','schema'),('episode.json','extra')])
def test_complete_boundary_checks_the_linked_records_schemas(episode,tmp_path,target,fault):
    module=api();reference=copy.deepcopy(episode['result']['unstarted_reference']);caller=reference['caller_reference']
    complete_path=Path(caller['path']);before_complete=complete_path.read_bytes();complete=json.loads(before_complete)
    path=episode['root']/target;before=path.read_bytes();changed=json.loads(before)
    if fault=='schema':changed['schema']='wrong-record-kind'
    else:changed['undeclared_authority']=True
    raw=json.dumps(changed).encode();(tmp_path/'corrupt-linked-record.json').write_bytes(raw)
    path.write_bytes(raw)
    key='caller_start_reference' if target=='caller-start.json' else 'episode_reference'
    complete[key]['sha256']=hashlib.sha256(raw).hexdigest();complete_path.write_text(json.dumps(complete))
    caller['sha256']=hashlib.sha256(complete_path.read_bytes()).hexdigest()
    try:
        with pytest.raises(ValueError):module.inspect_caller_boundary(caller,process_reference=reference)
    finally:
        path.write_bytes(before);complete_path.write_bytes(before_complete)
    assert path.read_bytes()==before and complete_path.read_bytes()==before_complete


@pytest.mark.parametrize('fault',['hash','foreign','negative_additional','boolean_elapsed','authority'])
def test_caller_boundary_rejects_changed_or_foreign_clock_observations(episode,tmp_path,fault):
    module=api();reference=copy.deepcopy(episode['result']['unstarted_reference']);caller=reference['caller_reference']
    if fault=='hash':caller['sha256']='0'*64
    elif fault=='foreign':caller['path']=str(tmp_path/'foreign.json')
    else:
        # Controlled copy is allowed only to test rejection; it never creates a
        # complete native-qualified public positive or fits a model.
        original=Path(caller['path']);value=json.loads(original.read_bytes())
        if fault=='negative_additional':value['additional_caller_interval_ns']=-1
        elif fault=='boolean_elapsed':value['elapsed_ns']=True
        else:value['native_qualification_authenticated']=True
        raw=json.dumps(value).encode();(tmp_path/'retained-corrupt-copy.json').write_bytes(raw)
        before=original.read_bytes();original.write_bytes(raw)
        caller['sha256']=hashlib.sha256(raw).hexdigest()
        try:
            with pytest.raises(ValueError):module.inspect_caller_boundary(caller,process_reference=reference)
        finally:
            # This is a closed disposable software fixture, never research data
            # or a live producer. Restore exact bytes for every later reader.
            original.write_bytes(before)
        assert original.read_bytes()==before
        return
    with pytest.raises((ValueError,FileNotFoundError)):module.inspect_caller_boundary(caller,process_reference=reference)
