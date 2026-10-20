"""Serial primary program fixtures never impersonate native qualification."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

from test_primary_search_declaration_v2 import study


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_program_v2'),'Missing serial primary program'
    from treeneuron.research import primary_program_v2
    return primary_program_v2


@pytest.fixture
def declared(study,tmp_path):
    module=api();plan,support=study
    from treeneuron.research.practical_procedure_v2 import seal_practical_procedure
    from treeneuron.research.ledger import write_new_json
    write_new_json(tmp_path/'practical.json',plan.to_dict())
    ref=lambda p:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    procedure=seal_practical_procedure(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],support_reference=support,output=tmp_path/'procedure.json')
    args=dict(practical_plan_reference=ref(tmp_path/'practical.json'),expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        support_reference=support,procedure_reference=procedure,
        qualification_reference=dict(root=str(tmp_path/'unopened-registry'),inventory_sha256='a'*64),
        resource_forecast_reference=dict(path=str(tmp_path/'unopened-forecast'),sha256='b'*64),
        directory=tmp_path/'program',rss_limit_bytes=4*1024**3)
    program=module.build_primary_program_plan(practical_plan=plan,**args)
    return program,args


def test_program_plan_is_exactly_the_primary_tranche_and_carries_both_repeats(declared):
    program,args=declared
    assert len(program['jobs'])==168 and len(set(j['job_id'] for j in program['jobs']))==168
    assert sum(j['dataset']=='bike' for j in program['jobs'])==28
    assert all(j['outer_fold']==(4 if j['dataset']=='bike' else 0) for j in program['jobs'])
    assert {j['search_repeat'] for j in program['jobs']}=={0,1}
    assert program['procedure_reference']==args['procedure_reference']
    assert program['execution_authorized'] is program['protected_access_authorized'] is False
    assert not args['directory'].exists() and not Path(args['qualification_reference']['root']).exists()


def test_actual_public_program_rejects_false_registry_before_any_plan_or_root(tmp_path,monkeypatch):
    module=api()
    from treeneuron.research import practical_procedure_v2
    def forbidden(*args,**kwargs):raise AssertionError('False qualification cannot reach procedure/plan')
    monkeypatch.setattr(practical_procedure_v2,'load_practical_procedure',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.run_primary_program(practical_plan_reference=dict(path=str(tmp_path/'forbidden-plan'),sha256='a'*64),
            expected_practical_plan_hash='b'*64,support_reference=dict(path=str(tmp_path/'forbidden-support'),sha256='c'*64),
            procedure_reference=dict(path=str(tmp_path/'forbidden-procedure'),sha256='d'*64),
            qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='e'*64),
            resource_forecast_reference=dict(path=str(tmp_path/'forbidden-cache'),sha256='f'*64),directory=tmp_path/'program')
    assert 'forbidden-' not in str(error.value) and list(tmp_path.iterdir())==[]


def no_child_episode(program,job,episode_root,previous_episodes,previous_unstarted_episodes):
    from treeneuron.research.primary_caller_v2 import _run_prepared_episode
    from treeneuron.research.primary_search_worker_v2 import _episode_environment
    request=dict(schema='research/v2-primary-program-episode-request',source_sha256=program['source_sha256'],
        **{k:program[k] for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference',
            'qualification_reference','resource_forecast_reference','procedure_reference','rss_limit_bytes')},
        job_id=job['job_id'],search_directory=str(Path(program['directory'])/'searches'/job['job_id']),
        previous_episodes=previous_episodes,previous_unstarted_episodes=previous_unstarted_episodes)
    raw=(json.dumps(request)+'\n').encode();sha=hashlib.sha256(raw).hexdigest()
    prepared=dict(request_bytes=raw,request_sha256=sha,argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(episode_root/'request.json'),'--expected-request-sha256',sha,'--output',str(episode_root/'worker')],
        cwd=str(Path(__file__).resolve().parents[1]),environment_overrides=_episode_environment())
    return _run_prepared_episode(prepared=prepared,episode_root=episode_root,
        identity=dict(purpose='primary_search_episode',request_sha256=sha,source_sha256=program['source_sha256']),
        admission_identity=dict(policy='no-child-software-fixture'),admission_check=lambda:dict(admitted=False),
        invocation_origin_ns=time.perf_counter_ns(),rss_limit_bytes=256*1024**2,expect_primary_body=True)


def test_real_resource_denial_pauses_same_first_job_and_resume_keeps_every_paid_root(declared):
    module=api();program,args=declared;seen=[]
    def execute(**values):seen.append(values);return no_child_episode(**values)
    def forbidden(*args,**kwargs):raise AssertionError('A never-authorized episode cannot enter native search audit')
    first=module._run_program_queue(program,execute_episode=execute,audit_terminal=forbidden)
    assert first['status']=='resource_pause' and first['completed_jobs']==0
    second=module._run_program_queue(program,execute_episode=execute,audit_terminal=forbidden,
        expected_journal_sha256=first['journal_sha256'])
    assert second['status']=='resource_pause' and second['completed_jobs']==0
    assert len(seen)==2 and seen[0]['job']==seen[1]['job']
    assert seen[0]['previous_unstarted_episodes']==[] and len(seen[1]['previous_unstarted_episodes'])==1
    assert second['known_observed_caller_ns']>first['known_observed_caller_ns']>0
    assert second['whole_training_procedure_cost_authenticated'] is second['protected_access_authorized'] is False
    assert not (args['directory']/'searches').exists()
    assert not Path(args['qualification_reference']['root']).exists()


def test_changed_external_journal_pin_cannot_restart_or_skip_a_job(declared):
    module=api();program,args=declared
    first=module._run_program_queue(program,execute_episode=no_child_episode,audit_terminal=lambda **kw:None)
    before=(args['directory']/'events.jsonl').read_bytes()
    with pytest.raises(ValueError):module._run_program_queue(program,execute_episode=no_child_episode,
        audit_terminal=lambda **kw:None,expected_journal_sha256='0'*64)
    assert (args['directory']/'events.jsonl').read_bytes()==before


def test_interrupted_driver_reopens_the_closed_caller_without_starting_a_replacement(declared,monkeypatch):
    module=api();program,args=declared
    def interrupted(**values):
        no_child_episode(**values)
        raise RuntimeError('driver interrupted after caller closed')
    def forbidden(*args,**kwargs):raise AssertionError('Recovery must not spawn a replacement or audit a never-started fit')
    with pytest.raises(RuntimeError):module._run_program_queue(program,execute_episode=interrupted,audit_terminal=forbidden)
    root=args['directory'];journal=root/'events.jsonl';before=journal.read_bytes()
    assert len(before.splitlines())==1
    import subprocess
    original_popen=subprocess.Popen
    def no_replacement_workers(argv,*a,**kw):
        if argv==['/usr/sbin/sysctl','-n','kern.boottime']:return original_popen(argv,*a,**kw)
        return forbidden(argv,*a,**kw)
    monkeypatch.setattr(subprocess,'Popen',no_replacement_workers)
    recovered=module._run_program_queue(program,execute_episode=forbidden,audit_terminal=forbidden,
        expected_journal_sha256=hashlib.sha256(before).hexdigest())
    assert recovered['status']=='resource_pause' and recovered['completed_jobs']==0
    assert len(journal.read_bytes().splitlines())==2
    assert recovered['known_observed_caller_ns']>0


def test_unclosed_unknown_episode_cannot_be_replaced_or_assigned_a_zero_tail(declared):
    module=api();program,args=declared
    def interrupted(**values):raise RuntimeError('caller never returned evidence')
    with pytest.raises(RuntimeError):module._run_program_queue(program,execute_episode=interrupted,audit_terminal=lambda **kw:None)
    root=args['directory'];journal=root/'events.jsonl';before=journal.read_bytes()
    def forbidden(**values):raise AssertionError('An unknown attempt cannot be silently replaced')
    with pytest.raises(PermissionError):module._run_program_queue(program,execute_episode=forbidden,audit_terminal=forbidden,
        expected_journal_sha256=hashlib.sha256(before).hexdigest())
    assert journal.read_bytes()==before


def test_output_root_cannot_contain_working_or_qualified_proof_inputs(tmp_path):
    module=api();working=tmp_path/'working';working.mkdir();proof=tmp_path/'proof';proof.mkdir()
    for root in (working/'program',proof/'program',tmp_path):
        with pytest.raises(ValueError):module._check_program_location(root,working_roots=[str(working)],proof_roots=[str(proof)],input_files=[])
    assert not (working/'program').exists() and not (proof/'program').exists()
