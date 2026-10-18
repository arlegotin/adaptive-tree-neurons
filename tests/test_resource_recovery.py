"""Remaining admission credits terminal whole transactions, never partial roles."""
import copy
import json
from types import SimpleNamespace

import pytest

from treeneuron.config import canonical_hash
from treeneuron.research.ledger import AttemptLedger


def setup_admission(tmp_path,monkeypatch):
    from treeneuron.research import resources
    output=tmp_path/'study';path=tmp_path/'admission.json'
    training=dict(base_updates=8000,continuation_updates=8000,batch_size=128)
    config=dict(kind='m0_reference',teachers=2,repetitions=1,evaluation_rows=65536,
        base_experiment=dict(training=training,model=dict(d=16,h=64),
            data=dict(n_fit=2306,n_growth=577,n_validation=577)),resource_admission=str(path))
    attempts=[dict(attempt_id=f't{t}/{role}',cell='primary',teacher_index=t,repetition=0,role=role)
        for t in range(2) for role in ('keep','top')]
    plan=dict(config=config,attempts=attempts,source_sha256='a'*64,execution_identity={})
    plan['plan_hash']=canonical_hash(plan)
    profiles=[]
    for phase in ('parent','continuation','singleton_probe'):
        p=tmp_path/(phase+'.json')
        p.write_text(json.dumps(dict(schema='research/v2-resource-measurement',source_sha256='a'*64,
            execution_identity_hash=canonical_hash({}),
            workload=dict(d=16,h=64,n_fit=2306,n_growth=577,n_validation=577,batch_size=128),
            phase=phase,elapsed_ns=1,peak_rss_bytes=1,retained_bytes=1,
            max_checkpoint_transaction_bytes=1,training_contract_hash=canonical_hash({'training':training}),
            status='completed_eligible')))
        import hashlib
        profiles.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    keys=[canonical_hash(dict(cell='primary',teacher_index=t,repetition=0)) for t in range(2)]
    record=dict(schema='research/v2-resource-admission',source_sha256='a'*64,
        config_hash=canonical_hash(config),execution_identity_hash=canonical_hash({}),measurements=profiles,
        forecast_worker_seconds_p50=1,forecast_worker_seconds_p90=2,forecast_remaining_bytes=1600,
        free_disk_margin_bytes=100,peak_rss_bytes=1,available_worker_seconds=3,max_checkpoint_transaction_bytes=1,
        case_retained_bytes={keys[0]:600,keys[1]:400},retry_reserve_bytes=600)
    path.write_text(json.dumps(record))
    free=[1700]
    monkeypatch.setattr(resources.shutil,'disk_usage',lambda _:SimpleNamespace(free=free[0]))
    return resources,plan,output,free


def terminal(ledger,ids,status='completed_eligible'):
    for name in ids:
        ledger.record(name,'running',elapsed_ns=0,recovery_attempt=0,started_utc_ns=1)
        ledger.record(name,status,elapsed_ns=123)


def create_ledger(resources,root,plan):
    decision=resources.validate_resource_admission(plan,root)['availability']
    ledger=AttemptLedger.create(root,plan)
    resources.publish_availability(root,decision,ledger)
    return ledger


def test_resume_uses_authenticated_terminal_group_credit_and_records_no_artifact_bytes(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    fresh=resources.validate_resource_admission(plan,root)
    assert fresh['availability']['required_additional_bytes']==1600
    ledger=create_ledger(resources,root,plan);terminal(ledger,['t0/keep','t0/top'])
    free[0]=900
    decision=resources.validate_resource_admission(plan,root,ledger=ledger)['availability']
    assert decision['required_additional_bytes']==800
    assert decision['terminal_transaction_count']==1 and decision['pending_transaction_count']==1
    assert decision['ledger_event_hash']==ledger.events[-1]['event_hash']
    assert ledger.charged_ns==246
    free[0]=899
    with pytest.raises(ValueError,match='free disk'):
        resources.validate_resource_admission(plan,root,ledger=ledger)


def test_partial_roles_and_interrupted_groups_receive_no_credit(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan);terminal(ledger,['t0/keep'])
    ledger.record('t0/top','running',elapsed_ns=0,recovery_attempt=0,started_utc_ns=1)
    ledger.record('t0/top','interrupted_recoverable',elapsed_ns=44,recovery_attempt=0)
    free[0]=1100
    with pytest.raises(ValueError,match='free disk'):
        resources.validate_resource_admission(plan,root,ledger=ledger)
    # Arbitrary large files cannot buy remaining-capacity credit.
    (root/'uncommitted.bin').write_bytes(b'x'*10000)
    with pytest.raises(ValueError,match='free disk'):
        resources.validate_resource_admission(plan,root,ledger=ledger)


def test_all_terminal_releases_worker_and_retry_reservations(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan);terminal(ledger,[a['attempt_id'] for a in plan['attempts']])
    free[0]=100
    decision=resources.validate_resource_admission(plan,root,ledger=ledger)['availability']
    assert decision['required_additional_bytes']==0 and decision['pending_transaction_count']==0


def test_changed_plan_or_tampered_ledger_cannot_release_capacity(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan);terminal(ledger,['t0/keep','t0/top'])
    changed=copy.deepcopy(plan);changed['attempts'][0]['role']='random'
    with pytest.raises(ValueError,match='plan'):
        resources.validate_resource_admission(changed,root,ledger=ledger)
    with (root/'events.jsonl').open('a') as stream:stream.write('{}\n')
    with pytest.raises((ValueError,KeyError)):
        resources.validate_resource_admission(plan,root,ledger=ledger)


def test_original_admission_cannot_be_replaced_at_same_path(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan)
    path=tmp_path/'admission.json';changed=json.loads(path.read_text())
    changed['case_retained_bytes']={k:1 for k in changed['case_retained_bytes']}
    changed.update(retry_reserve_bytes=1,forecast_remaining_bytes=3)
    path.write_text(json.dumps(changed));free[0]=103
    with pytest.raises(ValueError,match='admission commitment'):
        resources.validate_resource_admission(plan,root,ledger=ledger)


def test_interrupted_last_group_does_not_reserve_a_third_attempt(tmp_path,monkeypatch):
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan);terminal(ledger,['t1/keep','t1/top'])
    for role in ('keep','top'):
        ledger.record('t0/'+role,'running',elapsed_ns=0,recovery_attempt=0,started_utc_ns=1)
        ledger.record('t0/'+role,'interrupted_recoverable',elapsed_ns=44,recovery_attempt=0)
    free[0]=700
    decision=resources.validate_resource_admission(plan,root,ledger=ledger)['availability']
    assert decision['required_additional_bytes']==600 and decision['additional_retry_reserve_bytes']==0


def test_runner_authenticates_locked_resume_before_admission_and_publishes_failures(tmp_path,monkeypatch):
    import fcntl
    from treeneuron.research import runner,identity
    resources,plan,root,free=setup_admission(tmp_path,monkeypatch)
    ledger=create_ledger(resources,root,plan);terminal(ledger,['t0/keep','t0/top'])
    monkeypatch.setattr(runner.train,'source_hash',lambda:'a'*64)
    monkeypatch.setattr(identity,'execution_identity',lambda _: {})
    monkeypatch.setattr(runner,'_threads',lambda:None)
    calls=[];original=resources.validate_resource_admission
    def locked(*args,**kwargs):
        assert isinstance(kwargs['ledger'],AttemptLedger)
        with (root/'worker-lease.json').open('a+') as probe:
            with pytest.raises(BlockingIOError):fcntl.flock(probe.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        return original(*args,**kwargs)
    monkeypatch.setattr(resources,'validate_resource_admission',locked)
    monkeypatch.setattr(runner,'_run_locked',lambda *args: calls.append(True) or args[3])
    free[0]=900
    out=runner.run(SimpleNamespace(to_dict=lambda:copy.deepcopy(plan)),root,resume=True)
    assert out['availability']['required_additional_bytes']==800
    free[0]=899
    with pytest.raises(ValueError,match='free disk'):
        runner.run(SimpleNamespace(to_dict=lambda:copy.deepcopy(plan)),root,resume=True)
    assert len(calls)==1
    decisions=[json.loads(p.read_text()) for p in sorted(root.glob('resource-availability-*.json'))]
    assert decisions[-1]['status']=='insufficient_free_disk'
    assert len({d['admission_sha256'] for d in decisions})==1
    assert AttemptLedger.resume(root,plan).charged_ns==246
