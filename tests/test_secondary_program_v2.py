"""Secondary queue fixtures never supply a fabricated global/native success."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest

from test_primary_program_v2 import declared
from test_primary_search_declaration_v2 import study


def api():
    assert importlib.util.find_spec('treeneuron.research.secondary_program_v2'),'Missing serial secondary program'
    from treeneuron.research import secondary_program_v2
    return secondary_program_v2


@pytest.fixture
def secondary(declared):
    module=api();primary,args=declared
    from treeneuron.research.workload import PracticalPlan
    from treeneuron.config import canonical_json
    practical=PracticalPlan(canonical_json(json.loads(Path(args['practical_plan_reference']['path']).read_text())))
    global_ref=dict(path=str(args['directory'].parent/'unopened-global.json'),sha256='c'*64)
    program=module.build_secondary_program_plan(practical_plan=practical,global_barrier_reference=global_ref,**args)
    return program,args


def test_secondary_metadata_has_exact672_jobs_and_both_repeats_without_authority(secondary):
    program,args=secondary
    assert len(program['jobs'])==672 and len({j['job_id'] for j in program['jobs']})==672
    assert {j['search_repeat'] for j in program['jobs']}=={0,1}
    assert all(j['tranche']=='secondary' and j['outer_fold']!=(4 if j['dataset']=='bike' else 0) for j in program['jobs'])
    assert program['execution_authorized'] is program['native_workload_qualified'] is program['protected_access_authorized'] is False
    assert not args['directory'].exists() and not Path(program['global_barrier_reference']['path']).exists()


def no_child(program,job,episode_root,previous_episodes,previous_unstarted_episodes):
    from treeneuron.research.primary_program_v2 import _request_binding
    from treeneuron.research.primary_search_worker_v2 import _episode_environment,_episode_purpose,_ROOT
    from treeneuron.research.primary_caller_v2 import _run_prepared_episode
    value=dict(_request_binding(program,job),previous_episodes=previous_episodes,previous_unstarted_episodes=previous_unstarted_episodes)
    raw=(json.dumps(value)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    prepared=dict(request_bytes=raw,request_sha256=digest,argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
        '--request',str(episode_root/'request.json'),'--expected-request-sha256',digest,'--output',str(episode_root/'worker')],
        cwd=str(_ROOT),environment_overrides=_episode_environment())
    return _run_prepared_episode(prepared=prepared,episode_root=episode_root,
        identity=dict(purpose=_episode_purpose(value),request_sha256=digest,source_sha256=program['source_sha256']),
        admission_identity=dict(policy='unqualified software no-child fixture'),admission_check=lambda:dict(admitted=False),
        invocation_origin_ns=time.perf_counter_ns(),rss_limit_bytes=256*1024**2,expect_primary_body=True)


def test_real_no_child_secondary_queue_resumes_same_job_and_retains_all_known_clocks(secondary):
    program,args=secondary
    from treeneuron.research.primary_program_v2 import _run_program_queue
    calls=[]
    def execute(**kwargs):calls.append(kwargs);return no_child(**kwargs)
    def forbidden(**kw):raise AssertionError('Never-started job cannot be native-audited')
    first=_run_program_queue(program,execute_episode=execute,audit_terminal=forbidden,max_episode_starts=672)
    second=_run_program_queue(program,execute_episode=execute,audit_terminal=forbidden,max_episode_starts=672,expected_journal_sha256=first['journal_sha256'])
    assert first['status']==second['status']=='resource_pause' and second['completed_jobs']==0
    assert second['schema']=='research/v2-secondary-program-progress'
    assert calls[0]['job']==calls[1]['job'] and calls[0]['previous_unstarted_episodes']==[] and len(calls[1]['previous_unstarted_episodes'])==1
    assert second['known_observed_caller_ns']>first['known_observed_caller_ns']>0
    assert second['whole_training_procedure_cost_authenticated'] is second['protected_access_authorized'] is False
    assert not (args['directory']/'searches').exists()
    request=json.loads((calls[1]['episode_root']/'request.json').read_text())
    assert request['schema']=='research/v2-secondary-program-episode-request' and request['global_barrier_reference']==program['global_barrier_reference']
    before=(args['directory']/'events.jsonl').read_bytes()
    with pytest.raises(ValueError):_run_program_queue(program,execute_episode=execute,audit_terminal=forbidden,expected_journal_sha256='0'*64)
    assert (args['directory']/'events.jsonl').read_bytes()==before


def test_driver_interruption_reopens_closed_secondary_denial_without_spawning(secondary,monkeypatch):
    program,args=secondary
    from treeneuron.research.primary_program_v2 import _run_program_queue
    def interrupted(**kw):no_child(**kw);raise RuntimeError('driver interrupted after caller close')
    def forbidden(*a,**kw):raise AssertionError('A closed secondary denial cannot spawn or enter native audit')
    with pytest.raises(RuntimeError):_run_program_queue(program,execute_episode=interrupted,audit_terminal=forbidden)
    journal=args['directory']/'events.jsonl';raw=journal.read_bytes();assert len(raw.splitlines())==1
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',forbidden)
    result=_run_program_queue(program,execute_episode=forbidden,audit_terminal=forbidden,expected_journal_sha256=hashlib.sha256(raw).hexdigest())
    assert result['status']=='resource_pause' and result['completed_jobs']==0 and len(journal.read_bytes().splitlines())==2


def test_public_secondary_program_rejects_false_global_gate_before_plan_forecast_or_output(tmp_path,monkeypatch):
    module=api()
    from treeneuron.research import practical_procedure_v2
    def forbidden(*a,**kw):raise AssertionError('False global qualification cannot inspect the procedure')
    monkeypatch.setattr(practical_procedure_v2,'load_practical_procedure',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.run_secondary_program(barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='b'*64),
            resource_forecast_reference=dict(path=str(tmp_path/'forbidden-forecast'),sha256='c'*64),directory=tmp_path/'program')
    assert not any(s in str(error.value) for s in ('forbidden-global','forbidden-forecast')) and list(tmp_path.iterdir())==[]
