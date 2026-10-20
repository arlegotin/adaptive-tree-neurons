"""Actual ungranted roots cost time but authorize no fit or qualification."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


def api():
    from treeneuron.research import primary_search_worker_v2
    assert hasattr(primary_search_worker_v2,'_unstarted_episode_state'), 'Missing closed ungranted primary cost reader'
    return primary_search_worker_v2


@pytest.fixture(scope='module')
def episodes(tmp_path_factory):
    module=api()
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    root=tmp_path_factory.mktemp('unstarted-primary');previous=[]
    value=dict(schema='research/v2-primary-search-resource-episode-request',source_sha256=_source(),
        practical_plan_reference=dict(path=str(root/'not-opened-plan'),sha256='a'*64),expected_practical_plan_hash='b'*64,
        job_id='airfoil/outer-0/repeat-0/P13',support_reference=dict(path=str(root/'not-opened-support'),sha256='c'*64),
        qualification_reference=dict(root=str(root/'not-opened-registry'),inventory_sha256='d'*64),
        resource_forecast_reference=dict(path=str(root/'not-opened-forecast'),sha256='e'*64),search_directory=str(root/'not-created-search'),
        rss_limit_bytes=4*1024**3,previous_episodes=[],previous_unstarted_episodes=[])
    elapsed=0;rows=[];authorized=None
    for index in range(3):
        current=copy.deepcopy(value);current['previous_unstarted_episodes']=copy.deepcopy(previous)
        path=root/f'request-{index}.json';path.write_text(json.dumps(current));sha=hashlib.sha256(path.read_bytes()).hexdigest()
        output=root/f'process-{index}'
        result=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
            '--request',str(path),'--expected-request-sha256',sha,'--output',str(root/f'uncreated-worker-{index}')],
            cwd=str(Path(__file__).resolve().parents[1]),env_overrides=module._episode_environment(),output=output,
            identity=module.primary_worker_identity(path,sha),admission_identity=dict(policy='software-denial'),
            admission_check=lambda index=index:dict(admitted=index==2,reason='fixture phase test'),rss_pause_bytes=512*1024**2)
        reference=dict(process_root=str(output),process_receipt_sha256=hashlib.sha256((output/'process.json').read_bytes()).hexdigest(),
            process_started_sha256=hashlib.sha256((output/'started.json').read_bytes()).hexdigest())
        if index==2:
            assert result['grant_publication_started'] is True and result['returncode']==1
            authorized=reference
        else:
            assert result['execution_status']=='resource_exhausted' and result['grant_publication_started'] is False
            previous.append(reference);elapsed+=result['elapsed_ns'];rows.append(dict(reference=reference,result=result))
    value['previous_unstarted_episodes']=previous
    return dict(root=root,request=value,elapsed_ns=elapsed,rows=rows,authorized=authorized)


def test_two_closed_ungranted_episodes_are_paid_once_without_model_slots(episodes,monkeypatch):
    module=api()
    import subprocess
    def forbidden(*args,**kwargs):raise AssertionError('Reader cannot start processes or probe old PIDs')
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    result=module._unstarted_episode_state(episodes['request'])
    assert result['elapsed_ns']==episodes['elapsed_ns'] and result['episodes']==2
    assert result['native_qualification_authenticated'] is False and result['model_attempts_paid']==0
    prior=module._previous_episode_state(episodes['request'])
    assert prior['previous_outer_tail_ns']==episodes['elapsed_ns'] and prior['last'] is None
    assert not (episodes['root']/'not-created-search').exists()


@pytest.mark.parametrize('fault',['duplicate','wrong_job','wrong_source','changed_hash','reversed'])
def test_unstarted_history_cannot_change_lineage_or_double_charge(episodes,fault):
    module=api();request=copy.deepcopy(episodes['request'])
    if fault=='duplicate':request['previous_unstarted_episodes'].append(copy.deepcopy(request['previous_unstarted_episodes'][0]))
    elif fault=='wrong_job':request['job_id']='airfoil/outer-0/repeat-1/P13'
    elif fault=='wrong_source':request['source_sha256']='0'*64
    elif fault=='changed_hash':request['previous_unstarted_episodes'][0]['process_receipt_sha256']='0'*64
    else:request['previous_unstarted_episodes'].reverse()
    with pytest.raises(ValueError):module._unstarted_episode_state(request)


def test_closed_body_and_unstarted_ledgers_cannot_use_the_same_root(episodes):
    module=api();request=copy.deepcopy(episodes['request']);first=request['previous_unstarted_episodes'][0]
    request['previous_episodes']=[dict(**first,body_outcome_sha256='a'*64)]
    with pytest.raises(ValueError):module._unstarted_episode_state(request)


def test_authorized_failed_body_is_never_reclassified_as_an_unstarted_retry(episodes):
    module=api();request=copy.deepcopy(episodes['request'])
    request['previous_unstarted_episodes'].append(episodes['authorized'])
    with pytest.raises(ValueError,match='never-authorized'):
        module._unstarted_episode_state(request)


@pytest.mark.parametrize('fault',['boolean','extra_body_reference','legacy_schema'])
def test_bootstrap_rejects_ambiguous_operational_cost_history(episodes,tmp_path,fault):
    module=api();request=copy.deepcopy(episodes['request'])
    if fault=='boolean':request['previous_unstarted_episodes']=True
    elif fault=='extra_body_reference':request['previous_unstarted_episodes'][0]['body_outcome_sha256']='a'*64
    else:
        request['schema']='research/v2-primary-search-episode-request';request.pop('resource_forecast_reference')
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    with pytest.raises(ValueError):module.primary_worker_identity(path,hashlib.sha256(path.read_bytes()).hexdigest())
