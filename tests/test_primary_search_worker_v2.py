"""Cold bootstrap and real controller composition grant no fake qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_search_worker_v2'), 'Missing cold primary-search worker'
    from treeneuron.research import primary_search_worker_v2
    return primary_search_worker_v2


def test_request_identity_is_stdlib_only_and_never_decodes_fit_or_validation_data(tmp_path,monkeypatch):
    module=api()
    request=dict(schema='research/v2-primary-search-episode-request',source_sha256='a'*64,
        practical_plan_reference=dict(path=str(tmp_path/'not-opened-plan.json'),sha256='b'*64),
        expected_practical_plan_hash='c'*64,job_id='fixed-primary',
        support_reference=dict(path=str(tmp_path/'not-opened-support.json'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'not-opened-registry'),inventory_sha256='e'*64),
        search_directory=str(tmp_path/'not-created-search'),rss_limit_bytes=4*1024**3,
        previous_episodes=[])
    path=tmp_path/'request.json';path.write_text(json.dumps(request))
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    identity=module.primary_worker_identity(path,sha)
    assert identity==dict(purpose='primary_search_episode',request_sha256=sha,source_sha256='a'*64)
    assert not (tmp_path/'not-created-search').exists()


@pytest.mark.parametrize('fault',['extra','query_labels','rss_boolean','source','relative_plan','duplicate_field'])
def test_bootstrap_rejects_changed_or_ambiguous_request_fields_before_any_worker(tmp_path,fault):
    module=api()
    value=dict(schema='research/v2-primary-search-episode-request',source_sha256='a'*64,
        practical_plan_reference=dict(path=str(tmp_path/'plan.json'),sha256='b'*64),
        expected_practical_plan_hash='c'*64,job_id='fixed-primary',
        support_reference=dict(path=str(tmp_path/'support.json'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'registry'),inventory_sha256='e'*64),
        search_directory=str(tmp_path/'search'),rss_limit_bytes=4*1024**3,previous_episodes=[])
    if fault=='extra':value['ignored_field']=True
    elif fault=='query_labels':value['V_labels_path']=str(tmp_path/'V.npz')
    elif fault=='rss_boolean':value['rss_limit_bytes']=True
    elif fault=='source':value['source_sha256']=True
    elif fault=='relative_plan':value['practical_plan_reference']['path']='plan.json'
    path=tmp_path/'request.json';content=json.dumps(value)
    if fault=='duplicate_field':content=content[:-1]+',"source_sha256":"a"}'
    path.write_text(content)
    with pytest.raises(ValueError):module.primary_worker_identity(path,hashlib.sha256(path.read_bytes()).hexdigest())


def test_hand_filled_registry_cannot_reach_plan_roles_or_factory_output(tmp_path):
    module=api()
    from treeneuron.research.deployment_worker_v2 import _source
    value=dict(schema='research/v2-primary-search-episode-request',source_sha256=_source(),
        practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan.json'),sha256='b'*64),
        expected_practical_plan_hash='c'*64,job_id='fixed-primary',
        support_reference=dict(path=str(tmp_path/'forbidden-support.json'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='e'*64),
        search_directory=str(tmp_path/'search'),rss_limit_bytes=4*1024**3,previous_episodes=[])
    path=tmp_path/'request.json';path.write_text(json.dumps(value))
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module._prepare_qualified_episode(request=value,output=tmp_path/'worker')
    assert 'forbidden-plan' not in str(error.value) and 'forbidden-support' not in str(error.value)
    assert not (tmp_path/'worker').exists() and not (tmp_path/'search').exists()


def test_real_cold_controller_body_retains_a_closed_journal_without_qualification(tmp_path):
    module=api()
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process
    script=tmp_path/'software_worker.py'
    script.write_text('''
import hashlib, json, os
from pathlib import Path
from treeneuron.research.cooperative_process_v2 import cooperative_worker
with cooperative_worker(identity=dict(fixture='cold-search-controller')):
    from treeneuron.research.search_controller_v2 import make_search_plan
    from treeneuron.research.primary_search_worker_v2 import _run_prepared_search
    root=Path(os.environ['ATN_SEARCH_CONTROL_ROOT']).parent
    plan=make_search_plan(identity=dict(job_id='tiny/P13',family_id='P13',search_repeat=0,source_sha256='a'*64,runtime_sha256='b'*64,support_sha256='c'*64,split_sha256='d'*64),fit_seed=17,hpo_seed32=19,anchors=[dict(constant=0)],mode='finite',support=None,candidates=[dict(constant=1)],max_attempts=2,deadline_ns=None,prefixes_ns=[600000000000])
    def callback(request):
        p=Path(request['attempt_directory'])/'software-result.json';p.write_text('software-only')
        return dict(status='completed_eligible',V_score=1.,files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())],checkpoint_files=[],reason=None)
    _run_prepared_search(plan=plan,callback=callback,search_directory=root/'search',output=root/'worker',previous_outer_tail_ns=0,resume=False,expected_journal_hash=None)
''')
    root=tmp_path/'episode';root.mkdir()
    result=supervise_cooperative_process(argv=[sys.executable,'-P','-B',str(script)],cwd=str(root),
        env_overrides={'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src'),'PYTHONDONTWRITEBYTECODE':'1'},
        output=root/'process',identity=dict(fixture='cold-search-controller'),admission_identity=dict(policy='tiny-software'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=256*1024**2)
    assert result['returncode']==0
    saved=json.loads((root/'worker/search-result.json').read_text())
    proof=inspect_cooperative_process(root=root/'process',
        expected_receipt_sha256=hashlib.sha256((root/'process/process.json').read_bytes()).hexdigest(),
        expected_started_sha256=hashlib.sha256((root/'process/started.json').read_bytes()).hexdigest())
    assert proof['shared_monotonic_clock_observed']
    origin=json.loads((root/'process/started.json').read_text())['supervisor_started_monotonic_ns']
    events=[json.loads(line) for line in (root/'search/events.jsonl').read_text().splitlines()]
    assert events[0]['payload']['invocation_preparation_ns']>result['grant_elapsed_ns']
    assert result['elapsed_ns']>=saved['total_charged_ns']>0
    assert saved['attempts_paid']==2 and saved['whole_process_cost_authenticated'] is False
    assert events[-1]['kind']=='session_finished'


def test_actual_public_worker_rejects_fake_registry_before_any_plan_or_data(tmp_path):
    module=api()
    from treeneuron.research.deployment_worker_v2 import _source,_THREADS
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    project=Path(__file__).resolve().parents[1]
    value=dict(schema='research/v2-primary-search-episode-request',source_sha256=_source(),
        practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan.json'),sha256='b'*64),
        expected_practical_plan_hash='c'*64,job_id='fixed-primary',
        support_reference=dict(path=str(tmp_path/'forbidden-support.json'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='e'*64),
        search_directory=str(tmp_path/'search'),rss_limit_bytes=4*1024**3,previous_episodes=[])
    request=tmp_path/'request.json';request.write_text(json.dumps(value));sha=hashlib.sha256(request.read_bytes()).hexdigest()
    env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=str(project/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)
    r=supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(request),'--expected-request-sha256',sha,'--output',str(tmp_path/'worker')],
        cwd=str(project),env_overrides=env,output=tmp_path/'process',
        identity=module.primary_worker_identity(request,sha),admission_identity=dict(policy='software-only'),
        admission_check=lambda:dict(admitted=True),rss_pause_bytes=512*1024**2)
    assert r['returncode']==1 and r['worker_protocol_closed']
    outcome=json.loads((tmp_path/'process/body-outcome.json').read_text())
    assert outcome['status']=='body_failed'
    assert 'forbidden-plan' not in outcome['traceback'] and 'forbidden-support' not in outcome['traceback']
    assert not (tmp_path/'search').exists() and not (tmp_path/'worker').exists()
    assert outcome['quality_eligible'] is outcome['T_accessed'] is False


@pytest.mark.parametrize('field,value',[('argv',['/bin/echo']),('cwd','/private/tmp'),
    ('environment_overrides',{}),('identity',{'purpose':'unrelated'})])
def test_prior_episode_command_cannot_be_substituted_by_matching_flags(tmp_path,field,value):
    module=api()
    request=dict(schema='research/v2-primary-search-episode-request',source_sha256='a'*64,
        practical_plan_reference=dict(path=str(tmp_path/'not-opened-plan.json'),sha256='b'*64),
        expected_practical_plan_hash='c'*64,job_id='fixed-primary',
        support_reference=dict(path=str(tmp_path/'not-opened-support.json'),sha256='d'*64),
        qualification_reference=dict(root=str(tmp_path/'not-opened-registry'),inventory_sha256='e'*64),
        search_directory=str(tmp_path/'search'),rss_limit_bytes=4*1024**3,previous_episodes=[])
    path=tmp_path/'request.json';path.write_text(json.dumps(request));sha=hashlib.sha256(path.read_bytes()).hexdigest()
    body=dict(request_sha256=sha,job_binding=module._job_binding(request),
        search_result_reference=dict(path=str(tmp_path/'worker/body/search-result.json'),sha256='f'*64))
    start=dict(argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(path),'--expected-request-sha256',sha,'--output',str(tmp_path/'worker')],
        cwd=str(Path(__file__).resolve().parents[1]),environment_overrides=module._episode_environment(),
        identity=module.primary_worker_identity(path,sha))
    # Plain command metadata only. No process, search, registry, model or paid
    # clock is authenticated by this helper's otherwise valid return.
    assert module._check_episode_invocation(start,body,[]) is None
    start[field]=value
    with pytest.raises(ValueError):
        module._check_episode_invocation(start,body,[])


def test_two_actual_cold_controller_episodes_pay_closing_tail_before_resume(tmp_path):
    api()
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    script=tmp_path/'paired_software_worker.py'
    script.write_text('''
import hashlib, json, os, sys, time
from pathlib import Path
from treeneuron.research.cooperative_process_v2 import cooperative_worker
with cooperative_worker(identity=dict(fixture='two-cold-search-episodes')):
    from treeneuron.research.search_controller_v2 import make_search_plan
    from treeneuron.research.primary_search_worker_v2 import _run_prepared_search
    root=Path(sys.argv[1]); index=int(sys.argv[2]); episode=root/('episode'+str(index))
    time.sleep(.35)
    plan=make_search_plan(identity=dict(job_id='tiny/P13',family_id='P13',search_repeat=0,source_sha256='a'*64,runtime_sha256='b'*64,support_sha256='c'*64,split_sha256='d'*64),fit_seed=17,hpo_seed32=19,anchors=[dict(constant=0)],mode='finite',support=None,candidates=[dict(constant=1)],max_attempts=2,deadline_ns=None,prefixes_ns=[300000000])
    tail=0;previous=None
    if index:
        first=json.loads((root/'episode0/worker/search-result.json').read_text())
        outer=json.loads((root/'episode0/process/process.json').read_text())
        tail=outer['elapsed_ns']-first['total_charged_ns'];previous=first['journal_hash'];assert tail>=0
    def callback(request):
        p=Path(request['attempt_directory'])/'software-result.json';p.write_text('software-only')
        return dict(status='completed_eligible',V_score=1.,files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())],checkpoint_files=[],reason=None)
    _run_prepared_search(plan=plan,callback=callback,search_directory=root/'search',output=episode/'worker',previous_outer_tail_ns=tail,resume=bool(index),expected_journal_hash=previous,max_attempts_this_invocation=1)
''')
    rows=[]
    for index in range(2):
        episode=tmp_path/f'episode{index}';episode.mkdir()
        rows.append(supervise_cooperative_process(argv=[sys.executable,'-P','-B',str(script),str(tmp_path),str(index)],
            cwd=str(tmp_path),env_overrides={'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src'),'PYTHONDONTWRITEBYTECODE':'1'},
            output=episode/'process',identity=dict(fixture='two-cold-search-episodes'),
            admission_identity=dict(policy='tiny-software'),admission_check=lambda:dict(admitted=True),rss_pause_bytes=256*1024**2))
    assert all(row['returncode']==0 for row in rows)
    first=json.loads((tmp_path/'episode0/worker/search-result.json').read_text())
    final=json.loads((tmp_path/'episode1/worker/search-result.json').read_text())
    assert first['stop_reason']=='operator_pause' and final['attempts_paid']==2
    assert rows[0]['elapsed_ns']<final['total_charged_ns']<=sum(row['elapsed_ns'] for row in rows)
    assert final['prefixes']['300000000']['winner'] is None


@pytest.mark.parametrize('fault',[None,'endpoint','prefix_bytes','cost','record_count'])
def test_prior_episode_matches_exact_closed_journal_prefix(tmp_path,fault):
    module=api()
    from test_search_controller_v2 import Clock,plan,result,run
    clock=Clock();root=tmp_path/'search';config=plan(deadline_ns=None)
    def callback(request):clock.advance(5);return result(request)
    first=run(root,config,callback,clock,max_attempts_this_invocation=1)
    prefix=(root/'events.jsonl').read_bytes()
    run(root,config,callback,clock,resume=True,expected_journal_hash=first['journal_hash'],max_attempts_this_invocation=1)
    body=dict(journal_hash=first['journal_hash'],journal_sha256=hashlib.sha256(prefix).hexdigest(),
        total_charged_ns=first['total_charged_ns'])
    if fault=='endpoint':body['journal_hash']='0'*64
    elif fault=='prefix_bytes':body['journal_sha256']='0'*64
    elif fault=='cost':body['total_charged_ns']+=1
    elif fault=='record_count':first['records']=[]
    if fault is None:
        assert module._check_episode_journal_prefix(root,body,first)==len(prefix)
    else:
        with pytest.raises(ValueError):module._check_episode_journal_prefix(root,body,first)
