"""Live shortages pause before proposing/fitting and preserve paid search state."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from test_search_controller_v2 import Clock,plan,result,run


def test_initial_resource_denial_has_zero_attempts_and_paid_closed_resume(tmp_path):
    clock=Clock();calls=[];config=plan(deadline_ns=None,max_attempts=2)
    def admission():clock.advance(7);return dict(admitted=False,reason='insufficient_live_memory')
    def worker(request):calls.append(request);clock.advance(10);return result(request)
    first=run(tmp_path/'search',config,worker,clock,trial_admission=admission,
        trial_admission_identity=dict(forecast_hash='a'*64,policy='software-fixture'))
    assert calls==[] and first['attempts_paid']==0 and first['stop_reason']=='resource_pause'
    assert first['total_charged_ns']==7 and not list((tmp_path/'search').glob('attempt-*'))
    second=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'],
        trial_admission=lambda:dict(admitted=True,reason='resources_returned'),
        trial_admission_identity=dict(forecast_hash='a'*64,policy='software-fixture'))
    assert len(calls)==2 and calls[0]['config']==dict(width=17) and calls[1]['config']==dict(width=32)
    assert second['total_charged_ns']==27 and second['attempts_paid']==2
    events=[json.loads(line) for line in (tmp_path/'search/events.jsonl').read_text().splitlines()]
    assert len([e for e in events if e['kind']=='trial_admission'])==3


def test_resource_return_preserves_an_interrupted_candidates_identical_retry(tmp_path):
    clock=Clock();calls=[];config=plan(deadline_ns=None,max_attempts=3)
    def worker(request):
        calls.append(copy.deepcopy(request));clock.advance(10)
        return result(request,status='interrupted_recoverable' if len(calls)==1 else 'completed_eligible')
    first=run(tmp_path/'search',config,worker,clock)
    paused=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'],
        trial_admission=lambda:dict(admitted=False,reason='disk temporarily full'),trial_admission_identity=dict(policy='software'))
    assert paused['attempts_paid']==1 and paused['stop_reason']=='resource_pause' and len(calls)==1
    end=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=paused['journal_hash'],
        trial_admission=lambda:dict(admitted=True,reason='ready'),trial_admission_identity=dict(policy='software'))
    assert end['attempts_paid']==3 and calls[1]['retry_index']==1 and calls[1]['fit_seed']==calls[0]['fit_seed']
    assert calls[1]['config']==calls[0]['config'] and calls[1]['checkpoint_files']==first['records'][0]['files']


def test_denied_admission_does_not_advance_the_TPE_suggestion_stream(tmp_path):
    clock=Clock();calls=[];support=dict(lr=dict(kind='float',low=.001,high=.1,log=True))
    config=plan(mode='tpe',candidates=[],support=support,max_attempts=4,deadline_ns=None,prefixes_ns=[])
    def worker(request):calls.append(copy.deepcopy(request));clock.advance(2);return result(request,score=1.)
    first=run(tmp_path/'paused',config,worker,clock,trial_admission=lambda:dict(admitted=False),trial_admission_identity=dict(policy='software'))
    assert first['sampler_history']==[] and first['pending_suggestion'] is None
    end=run(tmp_path/'paused',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    reference=[]
    run(tmp_path/'reference',config,lambda request:(reference.append(copy.deepcopy(request)) or result(request,score=1.)),Clock())
    assert [r['config'] for r in calls]==[r['config'] for r in reference]
    assert end['sampler_history']


def test_independent_native_journal_reader_accepts_zero_attempt_resource_pause(tmp_path):
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.search_audit_v2 import audit_native_search_journal
    from treeneuron.research.ledger import write_new_json
    config=plan(deadline_ns=None,identity=dict(job_id='toy/P03',family_id='P03',search_repeat=0,
        source_sha256=_source(),runtime_sha256='b'*64,support_sha256='c'*64,split_sha256='d'*64))
    saved=run(tmp_path/'search',config,lambda request:pytest.fail('no fit may start'),Clock(),
        trial_admission=lambda:dict(admitted=False,reason='unknown live memory'),trial_admission_identity=dict(policy='software'))
    path=tmp_path/'saved.json';write_new_json(path,saved)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    proof=audit_native_search_journal(search_directory=tmp_path/'search',expected_plan_sha256=sha(tmp_path/'search/plan.json'),
        result_path=path,expected_result_sha256=sha(path),expected_journal_sha256=sha(tmp_path/'search/events.jsonl'))
    assert proof['attempts_paid']==0 and proof['completed_search'] is False


@pytest.mark.parametrize('fault',['without_hook','without_identity','boolean_not_strict'])
def test_malformed_resource_gate_cannot_be_ignored(tmp_path,fault):
    extra=dict(trial_admission=lambda:dict(admitted=True),trial_admission_identity=dict(policy='software'))
    if fault=='without_hook':extra['trial_admission']=None
    elif fault=='without_identity':extra['trial_admission_identity']=None
    else:extra['trial_admission']=lambda:dict(admitted=1)
    with pytest.raises(ValueError):run(tmp_path/'search',plan(deadline_ns=None),lambda request:pytest.fail('malformed admission cannot fit'),Clock(),**extra)


def test_explicit_null_admission_identity_cannot_masquerade_as_legacy_history(tmp_path):
    from treeneuron.config import canonical_hash
    from treeneuron.research.search_controller_v2 import _read_events,_Cursor,_checked_plan
    config=plan(deadline_ns=None,max_attempts=1);value=_checked_plan(config)
    run(tmp_path/'search',config,lambda request:result(request),Clock())
    path=tmp_path/'search/events.jsonl';events=[json.loads(line) for line in path.read_text().splitlines()]
    events[0]['payload']['trial_admission_identity']=None
    previous='0'*64
    for event in events:
        event['previous_hash']=previous;event['event_hash']=canonical_hash({k:v for k,v in event.items() if k!='event_hash'});previous=event['event_hash']
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    with pytest.raises(ValueError):_read_events(tmp_path/'search',value,_Cursor(value))


def test_admission_identity_is_an_immutable_snapshot_of_callers_metadata(tmp_path):
    from treeneuron.config import canonical_hash
    identity=dict(policy='original')
    def admission():identity['policy']='caller-mutated';return dict(admitted=True)
    run(tmp_path/'search',plan(deadline_ns=None,max_attempts=1),lambda request:result(request),Clock(),
        trial_admission=admission,trial_admission_identity=identity)
    events=[json.loads(line) for line in (tmp_path/'search/events.jsonl').read_text().splitlines()]
    assert events[1]['payload']['identity_sha256']==canonical_hash(dict(policy='original'))
