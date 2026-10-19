"""The outer budget includes native anchors, failures, recovery and preparation."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import search_controller_v2
    return search_controller_v2


class Clock:
    def __init__(self):self.value=0
    def __call__(self):return self.value
    def advance(self,n):self.value+=n


def plan(**overrides):
    values=dict(identity=dict(job_id='toy/P03',family_id='P03',search_repeat=0,
        source_sha256='a'*64,runtime_sha256='b'*64,support_sha256='c'*64,split_sha256='d'*64),
        fit_seed=487,hpo_seed32=19,max_attempts=4,deadline_ns=100,
        anchors=[dict(width=17)],mode='finite',support=None,
        candidates=[dict(width=32),dict(width=64),dict(width=128)],prefixes_ns=[20,40])
    return api().make_search_plan(**(values|overrides))


def result(request,*,status='completed_eligible',score=1.):
    path=Path(request['attempt_directory'])/'result.json'
    path.write_text(json.dumps(dict(config=request['config'],fit_seed=request['fit_seed'])))
    ref=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return dict(status=status,V_score=score if status=='completed_eligible' else None,
        files=[ref],checkpoint_files=[ref] if status=='interrupted_recoverable' else [],
        reason=None if status=='completed_eligible' else status)


def run(directory,config,worker,clock,**kwargs):
    return api().run_search(directory,plan=config,evaluate_trial=worker,clock_ns=clock,**kwargs)


def test_native_quality_search_keeps_late_completed_models_and_separate_budget_prefixes(tmp_path):
    minute=60*10**9;clock=Clock();requests=[]
    config=plan(deadline_ns=None,max_attempts=3,prefixes_ns=[10*minute,30*minute,120*minute])
    durations=[11*minute,120*minute,130*minute]
    def worker(request):
        requests.append(copy.deepcopy(request));clock.advance(durations[len(requests)-1])
        return result(request,score=1./len(requests))
    out=run(tmp_path/'quality',config,worker,clock)
    assert len(requests)==3 and all(r['budget_remaining_ns'] is None for r in requests)
    assert [r['status'] for r in out['records']]==['completed_eligible']*3
    assert out['winner']['attempt_index']==2 and out['stop_reason']=='attempt_limit'
    assert out['total_charged_ns']==261*minute
    assert out['prefixes'][str(10*minute)]['winner'] is None
    assert out['prefixes'][str(30*minute)]['winner']['attempt_index']==0
    assert out['prefixes'][str(120*minute)]['winner']['attempt_index']==0
    assert out['native_worker_qualified'] is False and out['protected_access_authorized'] is False


def test_native_quality_pause_resume_does_not_create_a_two_hour_deadline(tmp_path):
    hour=3600*10**9;clock=Clock();seen=[];config=plan(deadline_ns=None,max_attempts=2)
    def worker(request):
        seen.append(copy.deepcopy(request));clock.advance(3*hour)
        return result(request,score=1./len(seen))
    first=run(tmp_path/'quality',config,worker,clock,max_attempts_this_invocation=1)
    assert first['stop_reason']=='operator_pause' and first['total_charged_ns']==3*hour
    second=run(tmp_path/'quality',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    assert len(seen)==2 and second['winner']['attempt_index']==1 and second['total_charged_ns']==6*hour
    assert all(r['budget_remaining_ns'] is None for r in seen) and seen[1]['config']=={'width':32}


def test_native_quality_interruption_retains_same_seed_checkpoint_and_paid_attempt_limit(tmp_path):
    hour=3600*10**9;clock=Clock();seen=[];config=plan(deadline_ns=None,max_attempts=2)
    def worker(request):
        seen.append(copy.deepcopy(request));clock.advance(3*hour)
        return result(request,status='interrupted_recoverable' if len(seen)==1 else 'completed_eligible')
    first=run(tmp_path/'quality',config,worker,clock)
    assert first['stop_reason']=='interrupted_recoverable'
    second=run(tmp_path/'quality',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    assert len(seen)==2 and seen[0]['config']==seen[1]['config'] and seen[1]['retry_index']==1
    assert seen[1]['checkpoint_files'] and seen[0]['fit_seed']==seen[1]['fit_seed']==487
    assert second['attempts_paid']==2 and second['stop_reason']=='attempt_limit'
    with pytest.raises(ValueError,match='terminal'):
        run(tmp_path/'quality',config,worker,clock,resume=True,expected_journal_hash=second['journal_hash'])


def test_native_anchor_is_first_and_every_failure_consumes_the_outer_slot(tmp_path):
    clock=Clock();requests=[]
    def worker(request):
        requests.append(copy.deepcopy(request));clock.advance(5)
        return result(request,status='numerical_failure' if len(requests)==2 else 'completed_eligible',score=1./len(requests))
    out=run(tmp_path/'search',plan(max_attempts=3),worker,clock)
    assert [x['config']['width'] for x in requests]==[17,32,64]
    assert [x['candidate_origin'] for x in requests]==['anchor','finite','finite']
    assert all(x['fit_seed']==487 for x in requests)
    assert out['attempts_paid']==3 and out['total_charged_ns']==15
    assert out['stop_reason']=='attempt_limit' and out['winner']['attempt_index']==2
    assert out['records'][1]['status']=='numerical_failure'
    assert out['protected_access_authorized'] is False and out['native_worker_qualified'] is False


def test_unexpected_trial_failure_commits_partial_native_files_and_stops_the_search(tmp_path):
    clock=Clock()
    def worker(request):
        root=Path(request['attempt_directory'])/'native';root.mkdir()
        (root/'partial-state.bin').write_bytes(b'actual retained partial native bytes')
        clock.advance(7)
        raise RuntimeError('controlled coordinator failure after native output')
    out=run(tmp_path/'search',plan(),worker,clock)
    assert out['stop_reason']=='unexpected_coordinator_error' and out['attempts_paid']==1
    assert out['records'][0]['status']=='input_or_dependency_failure' and out['winner'] is None
    files={ref['path']:ref['sha256'] for ref in out['records'][0]['files']}
    assert files['attempt-000/native/partial-state.bin']==hashlib.sha256(b'actual retained partial native bytes').hexdigest()
    assert 'attempt-000/unexpected-error.json' in files
    assert out['total_charged_ns']==7


def test_deadline_includes_coordinator_work_and_late_predictions_cannot_win(tmp_path):
    clock=Clock();requests=[]
    def worker(request):
        requests.append(copy.deepcopy(request));clock.advance(15)
        return result(request,score=1./len(requests))
    out=run(tmp_path/'search',plan(deadline_ns=25,prefixes_ns=[10,20]),worker,clock)
    assert [x['budget_remaining_ns'] for x in requests]==[25,10]
    assert len(requests)==2 and out['total_charged_ns']==30 and out['stop_reason']=='deadline'
    assert out['records'][1]['status']=='timeout' and out['records'][1]['reported_status']=='completed_eligible'
    assert out['winner']['attempt_index']==0
    assert out['prefixes']['10']['winner'] is None
    assert out['prefixes']['20']['winner']['attempt_index']==0
    assert out['hard_worker_deadline_implemented'] is False


def test_same_candidate_recovery_pays_a_new_slot_and_preserves_checkpoint_and_seed(tmp_path):
    clock=Clock();seen=[];config=plan(max_attempts=3)
    def worker(request):
        seen.append(copy.deepcopy(request));clock.advance(7)
        return result(request,status='interrupted_recoverable' if len(seen)==1 else 'completed_eligible')
    first=run(tmp_path/'search',config,worker,clock)
    assert first['stop_reason']=='interrupted_recoverable' and first['attempts_paid']==1
    # Off-worker downtime does not become compute, nor erase already paid time.
    clock.advance(1000000)
    second=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    assert second['attempts_paid']==3 and second['total_charged_ns']==21
    assert seen[0]['config']==seen[1]['config']=={'width':17}
    assert seen[0]['candidate_id']==seen[1]['candidate_id']
    assert seen[1]['retry_index']==1 and seen[1]['checkpoint_files']
    assert seen[2]['config']=={'width':32} and seen[2]['retry_index']==0
    assert all(x['fit_seed']==487 for x in seen)


def test_second_interruption_closes_recovery_without_a_third_same_seed_attempt(tmp_path):
    clock=Clock();seen=[];config=plan()
    def worker(request):
        seen.append(request);clock.advance(3)
        return result(request,status='interrupted_recoverable')
    first=run(tmp_path/'search',config,worker,clock)
    out=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    assert len(seen)==2 and out['stop_reason']=='recovery_exhausted'
    assert out['winner'] is None and out['attempts_paid']==2
    with pytest.raises(ValueError,match='terminal'):
        run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=out['journal_hash'])


def test_finite_support_exhausts_without_filling_the_budget_with_duplicates(tmp_path):
    clock=Clock();seen=[]
    def worker(request):
        seen.append(request);clock.advance(1);return result(request)
    out=run(tmp_path/'search',plan(candidates=[dict(width=32)]),worker,clock)
    assert out['stop_reason']=='support_exhausted' and out['attempts_paid']==2


def test_tpe_pending_recovery_and_later_suggestions_match_an_uninterrupted_engine(tmp_path):
    from treeneuron.research.search_v2 import ReplayTPEV2
    clock=Clock();support=dict(lr=dict(kind='float',low=.001,high=.1,log=True))
    config=plan(mode='tpe',candidates=[],support=support,max_attempts=18,deadline_ns=1000,prefixes_ns=[])
    calls=[]
    def worker(request):
        calls.append(copy.deepcopy(request));clock.advance(1)
        if request['candidate_id']=='tpe-12' and request['retry_index']==0:
            return result(request,status='interrupted_recoverable')
        return result(request,score=1. if request['candidate_origin']=='anchor' else (request['config']['lr']-.03)**2)
    a=run(tmp_path/'search',config,worker,clock)
    assert a['attempts_paid']==14 and a['stop_reason']=='interrupted_recoverable'
    b=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=a['journal_hash'])
    assert b['attempts_paid']==18
    reference=ReplayTPEV2(seed=19,support=support,identity=config.to_dict()['identity'],max_suggestions=17)
    adaptive=[x for x in calls if x['candidate_origin']=='tpe' and x['retry_index']==0]
    for request in adaptive:
        assert reference.ask()==request['config']
        reference.tell((request['config']['lr']-.03)**2)
    assert calls[13]['candidate_id']==calls[14]['candidate_id']=='tpe-12'
    assert calls[13]['config']==calls[14]['config']
    assert len(b['sampler_history'])==16  # Anchor plus one paid retry leave16 adaptive outcomes.


def test_paused_search_replays_after_startup_and_rejects_changed_plan(tmp_path):
    clock=Clock();seen=[];config=plan()
    def worker(request):
        seen.append(request);clock.advance(4);return result(request)
    first=run(tmp_path/'search',config,worker,clock,max_attempts_this_invocation=2)
    assert first['stop_reason']=='operator_pause'
    changed=plan(fit_seed=488)
    with pytest.raises(ValueError,match='plan'):
        run(tmp_path/'search',changed,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    second=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])
    assert second['attempts_paid']==4 and second['total_charged_ns']==16
    assert [r['attempt_index'] for r in second['records']]==list(range(4))
    assert len({r['attempt_id'] for r in second['records']})==4
    with pytest.raises(ValueError):run(tmp_path/'search',config,worker,clock)


def test_raw_model_or_checkpoint_file_drift_blocks_resume_and_does_not_spend_a_slot(tmp_path):
    clock=Clock();config=plan()
    def worker(request):clock.advance(2);return result(request)
    first=run(tmp_path/'search',config,worker,clock,max_attempts_this_invocation=1)
    (tmp_path/'search'/'attempt-000'/'result.json').write_text('changed')
    with pytest.raises(ValueError,match='file commitment'):
        run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])


@pytest.mark.parametrize('change',[dict(max_attempts=33),dict(max_attempts=True),dict(deadline_ns=0),
    dict(deadline_ns=7200000000001),dict(anchors=[]),dict(mode='unknown'),dict(prefixes_ns=[101]),
    dict(prefixes_ns=[20,20]),dict(fit_seed=True),dict(hpo_seed32=-1)])
def test_invalid_budget_or_support_is_rejected_before_directory_creation(tmp_path,change):
    with pytest.raises(ValueError):plan(**change)
    assert not list(tmp_path.iterdir())


def test_coordinator_exception_is_retained_paid_and_cannot_trigger_seed_replacement(tmp_path):
    clock=Clock();seen=[]
    def worker(request):
        seen.append(request);clock.advance(9);raise RuntimeError('coordinator broke')
    out=run(tmp_path/'search',plan(),worker,clock)
    assert len(seen)==1 and out['attempts_paid']==1 and out['total_charged_ns']==9
    assert out['stop_reason']=='unexpected_coordinator_error'
    assert out['records'][0]['status']=='input_or_dependency_failure'
    assert out['records'][0]['unexpected_exception'] is True
    assert 'coordinator broke' in (tmp_path/'search'/'attempt-000'/'unexpected-error.json').read_text()


def test_worker_cannot_make_failed_predictions_eligible_or_report_uncommitted_files(tmp_path):
    clock=Clock()
    def worker(request):
        clock.advance(1);r=result(request,status='completed_ineligible');r['V_score']=0.;return r
    out=run(tmp_path/'search',plan(),worker,clock)
    assert out['stop_reason']=='unexpected_coordinator_error' and out['winner'] is None


def test_active_lease_prevents_a_second_controller(tmp_path):
    from treeneuron.research.ledger import StudyLease
    clock=Clock();config=plan()
    def worker(request):clock.advance(1);return result(request)
    first=run(tmp_path/'search',config,worker,clock,max_attempts_this_invocation=1)
    with StudyLease(tmp_path/'search'):
        with pytest.raises(ValueError,match='active'):
            run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=first['journal_hash'])


def test_interruption_at_last_paid_slot_cannot_advertise_an_available_recovery(tmp_path):
    clock=Clock();config=plan(max_attempts=1)
    def worker(request):clock.advance(2);return result(request,status='interrupted_recoverable')
    out=run(tmp_path/'search',config,worker,clock)
    assert out['stop_reason']=='recovery_budget_exhausted'
    with pytest.raises(ValueError,match='terminal'):
        run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=out['journal_hash'])


def test_invocation_preparation_is_paid_before_trials_and_again_on_recovery(tmp_path):
    clock=Clock();config=plan();seen=[]
    def worker(request):
        seen.append(request);clock.advance(5);return result(request)
    a=run(tmp_path/'search',config,worker,clock,max_attempts_this_invocation=1,invocation_preparation_ns=8)
    assert seen[0]['budget_remaining_ns']==92 and a['total_charged_ns']==13
    b=run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=a['journal_hash'],
        invocation_preparation_ns=6)
    assert seen[1]['budget_remaining_ns']==81 and b['total_charged_ns']==34


def test_resume_requires_an_external_journal_endpoint_even_after_coherent_rehash(tmp_path):
    clock=Clock();config=plan()
    def worker(request):clock.advance(1);return result(request)
    out=run(tmp_path/'search',config,worker,clock,max_attempts_this_invocation=1)
    with pytest.raises(ValueError,match='journal commitment'):
        run(tmp_path/'search',config,worker,clock,resume=True)
    path=tmp_path/'search'/'events.jsonl';events=[json.loads(line) for line in path.read_text().splitlines()]
    previous='0'*64
    for event in events:
        if event['kind']=='attempt_finished':
            event['payload']['result']['V_score']=.25
            event['payload']['record']['V_score']=.25
            event['payload']['record']['reported_V_score']=.25
        event['previous_hash']=previous
        event['event_hash']=canonical_hash({k:v for k,v in event.items() if k!='event_hash'})
        previous=event['event_hash']
    path.write_text(''.join(json.dumps(event)+'\n' for event in events))
    with pytest.raises(ValueError,match='journal commitment'):
        run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=out['journal_hash'])


def test_unclosed_process_death_cannot_be_assigned_a_free_zero_cost_resume(tmp_path):
    clock=Clock();config=plan()
    def worker(request):clock.advance(3);raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):run(tmp_path/'search',config,worker,clock)
    path=tmp_path/'search'/'events.jsonl';last=json.loads(path.read_text().splitlines()[-1])
    assert last['kind']=='attempt_started'
    with pytest.raises(ValueError,match='Unclosed.*external invocation cost'):
        run(tmp_path/'search',config,worker,clock,resume=True,expected_journal_hash=last['event_hash'])
