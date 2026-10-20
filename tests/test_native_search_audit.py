"""A search result needs exact proposal replay and every paid losing attempt."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash
from test_search_controller_v2 import Clock,plan,result,run
from test_search_v2 import support,objective


def api():
    assert importlib.util.find_spec('treeneuron.research.search_audit_v2') is not None,'Missing read-only native search audit'
    from treeneuron.research.search_audit_v2 import audit_native_search_journal
    return audit_native_search_journal


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n')


def fixture(tmp_path,*,tpe=False,paused=False,deadline=None):
    from treeneuron.research.deployment_worker_v2 import _source
    base=plan().to_dict();identity=base['identity']|dict(source_sha256=_source())
    cfg=plan(identity=identity,deadline_ns=deadline,max_attempts=18 if tpe else 4,
        mode='tpe' if tpe else 'finite',support=support() if tpe else None,candidates=[] if tpe else base['candidates'])
    clock=Clock();root=tmp_path/'search'
    def worker(request):
        clock.advance(5)
        return result(request,status='numerical_failure' if request['attempt_index']==2 else 'completed_eligible',
            score=objective(request['config']) if request['candidate_origin']=='tpe' else 1.)
    first=run(root,cfg,worker,clock,max_attempts_this_invocation=9 if tpe else 1 if paused else 32)
    out=first
    if tpe:
        out=run(root,cfg,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    path=tmp_path/'closed-result.json';publish(path,out)
    return root,path,out


def args(root,path):
    return dict(search_directory=root,expected_plan_sha256=sha(root/'plan.json'),
        result_path=path,expected_result_sha256=sha(path),expected_journal_sha256=sha(root/'events.jsonl'))


def test_exact_adaptive_replay_includes_failed_feedback_and_clean_resume_without_writes(tmp_path):
    from treeneuron.research.workqueue import _inventory
    fn=api();root,path,out=fixture(tmp_path,tpe=True);before=_inventory(root)
    audited=fn(**args(root,path))
    assert audited['search_journal_authenticated'] is True and audited['completed_search'] is True
    assert audited['attempts_paid']==18 and audited['sessions']==2
    assert audited['sampler_feedback_count']==17 and audited['failed_sampler_feedback_count']==1
    assert audited['winner']==out['winner'] and audited['prefixes']==out['prefixes']
    assert audited['total_charged_ns']==out['total_charged_ns']==90
    assert audited['native_workload_qualified'] is audited['V_arithmetic_authenticated'] is audited['T_accessed'] is False
    assert audited['whole_process_cost_authenticated'] is False and _inventory(root)==before


@pytest.mark.parametrize('change',['plan_hash','result_hash','journal_hash','file','records','pending','history','winner','prefix','cost','stop'])
def test_hash_or_self_rehashed_result_changes_cannot_override_the_actual_search(tmp_path,change):
    fn=api();root,path,out=fixture(tmp_path);kw=args(root,path)
    if change=='plan_hash':kw['expected_plan_sha256']='0'*64
    elif change=='result_hash':kw['expected_result_sha256']='0'*64
    elif change=='journal_hash':kw['expected_journal_sha256']='0'*64
    elif change=='file':(root/out['records'][0]['files'][0]['path']).write_bytes(b'changed model evidence')
    else:
        out=copy.deepcopy(out)
        if change=='records':out['records'].pop(2)
        elif change=='pending':out['pending_suggestion']={'new':'unpaid suggestion'}
        elif change=='history':out['sampler_history']=[{'number':0,'params':{},'value':0.}]
        elif change=='winner':out['winner']=None
        elif change=='prefix':out['prefixes'][next(iter(out['prefixes']))]['winner']=None
        elif change=='cost':out['total_charged_ns']+=1
        else:out['stop_reason']='support_exhausted'
        publish(path,out);kw['expected_result_sha256']=sha(path)
    with pytest.raises(ValueError):fn(**kw)


def test_rehashed_in_support_adaptive_proposal_still_fails_exact_TPE_law(tmp_path):
    fn=api();root,path,out=fixture(tmp_path,tpe=True)
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()];changed=False;previous='0'*64
    for event in events:
        if event['kind']=='attempt_started' and event['payload']['candidate_id']=='tpe-11':
            event['payload']['config']['x']=.123456789;changed=True
        event['previous_hash']=previous;event['event_hash']=canonical_hash({k:v for k,v in event.items() if k!='event_hash'});previous=event['event_hash']
    assert changed
    (root/'events.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
    out['journal_hash']=previous;publish(path,out)
    with pytest.raises(ValueError,match='exact search replay'):fn(**args(root,path))


def test_closed_pause_is_authenticated_without_claiming_a_completed_search(tmp_path):
    fn=api();root,path,_=fixture(tmp_path,paused=True)
    audited=fn(**args(root,path));assert audited['search_journal_authenticated'] is True
    assert audited['completed_search'] is False and audited['stop_reason']=='operator_pause'


def test_capped_legacy_search_and_active_writer_cannot_supply_this_native_quality_proof(tmp_path):
    from treeneuron.research.ledger import StudyLease
    fn=api();root,path,_=fixture(tmp_path,deadline=100)
    with pytest.raises(ValueError,match='native|deadline'):fn(**args(root,path))
    other=tmp_path/'other';other.mkdir();root,path,_=fixture(other)
    with StudyLease(root):
        with pytest.raises(PermissionError):fn(**args(root,path))


@pytest.mark.parametrize('change',['preparation','unknown_result','unexpected_stop','unexpected_status'])
def test_rehashed_journal_cannot_hide_paid_preparation_or_impossible_controller_transitions(tmp_path,change):
    fn=api();root,path,out=fixture(tmp_path)
    events=[json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()]
    if change=='preparation':events[0]['payload']['invocation_preparation_ns']=10**9
    elif change=='unknown_result':
        next(e for e in events if e['kind']=='attempt_finished')['payload']['result']['quality_eligible']=True
    else:
        last=next(e for e in reversed(events) if e['kind']=='attempt_finished')
        last['payload']['result']['unexpected_exception']=True
        last['payload']['record']['unexpected_exception']=True
        out['records'][-1]['unexpected_exception']=True
        if change=='unexpected_status':
            last['payload']['result'].update(status='numerical_failure',V_score=None,
                reason='Unexpected coordinator exception; diagnosis required')
            for row in (last['payload']['record'],out['records'][-1]):
                row.update(status='numerical_failure',V_score=None,reported_status='numerical_failure',reported_V_score=None)
            events[-1]['payload']['stop_reason']='unexpected_coordinator_error'
            out['stop_reason']='unexpected_coordinator_error'
            from treeneuron.analysis.practical_v2 import select_completed_trial
            out['winner']=select_completed_trial(out['records'],deadline_ns=out['total_charged_ns'])['winner']
            out['prefixes']={n:select_completed_trial(out['records'],deadline_ns=int(n)) for n in out['prefixes']}
    previous='0'*64
    for event in events:
        event['previous_hash']=previous;event['event_hash']=canonical_hash({k:v for k,v in event.items() if k!='event_hash'});previous=event['event_hash']
    (root/'events.jsonl').write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in events))
    out['journal_hash']=previous;publish(path,out)
    with pytest.raises(ValueError):fn(**args(root,path))


def test_interrupted_adaptive_candidate_remains_pending_until_identical_paid_recovery(tmp_path):
    from treeneuron.research.deployment_worker_v2 import _source
    fn=api();clock=Clock();root=tmp_path/'search'
    cfg=plan(identity=plan().to_dict()['identity']|dict(source_sha256=_source()),
        deadline_ns=None,max_attempts=4,mode='tpe',support=support(),candidates=[])
    def worker(request):
        clock.advance(5)
        return result(request,status='interrupted_recoverable' if request['attempt_index']==1 else 'completed_eligible')
    first=run(root,cfg,worker,clock,invocation_preparation_ns=7)
    path=tmp_path/'closed-result.json';publish(path,first)
    paused=fn(**args(root,path))
    assert paused['completed_search'] is False and paused['sampler_feedback_count']==0
    assert first['pending_suggestion'] is not None
    second=run(root,cfg,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    publish(path,second);complete=fn(**args(root,path))
    assert complete['completed_search'] is True and complete['attempts_paid']==4
    assert complete['sampler_feedback_count']==2 and complete['failed_sampler_feedback_count']==0
    assert complete['total_charged_ns']==27
