"""Finite jobs retain failures, exact retry identity and authenticated inputs."""
from pathlib import Path

import pytest


def job(name,depends=()):
    return dict(job_id=name,kind='fixture',depends_on=list(depends),payload={'seed':871})


def test_graph_order_and_invalid_dependencies(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({'source':'fixed'},[job('child',('parent',)),job('parent')])
    visited=[]
    def handler(case,root,dependencies,attempt):
        visited.append(case['job_id']);(root/'value.txt').write_text(str(case['payload']['seed']))
        if dependencies:assert dependencies['parent']['value']['seed']==871
        return {'status':'completed_eligible','seed':871}
    result=run_queue(plan,tmp_path/'queue',handler,check_identity=lambda identity:None)
    assert visited==['parent','child'] and result['status']=='completed'
    with pytest.raises(ValueError,match='dependency'):
        make_queue_plan({},[job('a',('missing',))])
    with pytest.raises(ValueError,match='cycle'):
        make_queue_plan({},[job('a',('b',)),job('b',('a',))])


def test_unavailable_and_numerical_failure_keep_other_work_and_never_reseed(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('absent'),job('skip',('absent',)),job('bad'),job('missing',('bad',)),job('good')])
    seen=[]
    def handler(case,root,dependencies,attempt):
        seen.append(case['job_id'])
        if case['job_id']=='absent':return {'status':'completed_ineligible','reason':'actual snapshot not reached'}
        if case['job_id']=='bad':raise FloatingPointError('nonfinite fit')
        return {'status':'completed_eligible'}
    result=run_queue(plan,tmp_path/'queue',handler,check_identity=lambda identity:None)
    assert seen==['absent','bad','good'] and result['status']=='completed_with_failures'
    assert result['statuses']['skip']=='completed_ineligible'
    assert result['statuses']['missing']=='irrecoverable_missing'
    again=run_queue(plan,tmp_path/'queue',handler,check_identity=lambda identity:None,resume=True)
    assert seen==['absent','bad','good'] and again['statuses']==result['statuses']


def test_interruption_has_one_fresh_same_payload_retry_and_retains_paid_work(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    from treeneuron.research.ledger import AttemptLedger
    plan=make_queue_plan({},[job('one')]);seen=[];root=tmp_path/'queue'
    def handler(case,output,dependencies,attempt):
        seen.append((case['payload'],attempt,str(output)))
        (output/'seed.txt').write_text(str(case['payload']['seed']))
        if attempt==0:raise KeyboardInterrupt('predeclared interruption')
        return {'status':'completed_eligible'}
    with pytest.raises(KeyboardInterrupt):run_queue(plan,root,handler,check_identity=lambda identity:None)
    before=AttemptLedger.resume(root,plan).charged_ns
    result=run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)
    assert result['status']=='completed' and len(seen)==2
    assert seen[0][0]==seen[1][0] and seen[0][2]!=seen[1][2]
    assert Path(seen[0][2],'seed.txt').exists() and Path(seen[1][2],'seed.txt').exists()
    assert AttemptLedger.resume(root,plan).charged_ns>before>0


def test_dependency_tampering_and_changed_identity_fail_before_new_work(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('parent'),job('child',('parent',))]);writes=[]
    def handler(case,root,dependencies,attempt):
        if case['job_id']=='child':raise KeyboardInterrupt('stop after parent')
        path=root/'committed.txt';path.write_text('original');writes.append(path)
        return {'status':'completed_eligible'}
    root=tmp_path/'queue'
    with pytest.raises(KeyboardInterrupt):run_queue(plan,root,handler,check_identity=lambda identity:None)
    writes[0].write_text('changed')
    with pytest.raises(ValueError,match='commitment'):
        run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)
    def changed(identity):raise ValueError('source changed')
    with pytest.raises(ValueError,match='source changed'):
        run_queue(plan,tmp_path/'must-not-exist',handler,check_identity=changed)
    assert not (tmp_path/'must-not-exist').exists()


def test_each_dependency_reuse_rechecks_committed_bytes(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('parent'),job('mutator',('parent',)),job('child',('parent',))])
    called=[]
    def handler(case,root,dependencies,attempt):
        called.append(case['job_id'])
        if case['job_id']=='parent':
            path=root/'data.txt';path.write_text('original')
            return {'status':'completed_eligible','file':str(path)}
        if case['job_id']=='mutator':Path(dependencies['parent']['value']['file']).write_text('changed')
        return {'status':'completed_eligible'}
    with pytest.raises(ValueError,match='commitment'):
        run_queue(plan,tmp_path/'queue',handler,check_identity=lambda identity:None)
    assert called==['parent','mutator']


def test_second_interruption_closes_job_without_a_third_attempt(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('one')]);root=tmp_path/'queue';seen=[]
    def handler(case,output,dependencies,attempt):
        seen.append(attempt);raise KeyboardInterrupt('interrupt')
    with pytest.raises(KeyboardInterrupt):run_queue(plan,root,handler,check_identity=lambda identity:None)
    with pytest.raises(KeyboardInterrupt):run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)
    result=run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)
    assert seen==[0,1] and result['statuses']['one']=='irrecoverable_missing'


def test_orphan_recovery_retains_unknown_wall_time_as_an_upper_bound(tmp_path):
    import time
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    from treeneuron.research.ledger import AttemptLedger
    plan=make_queue_plan({},[job('one')]);root=tmp_path/'queue'
    ledger=AttemptLedger.create(root,plan)
    ledger.record('one','running',elapsed_ns=123,recovery_attempt=0,started_utc_ns=time.time_ns()-10000)
    result=run_queue(plan,root,lambda *args:{'status':'completed_eligible'},
        check_identity=lambda identity:None,resume=True)
    assert result['charged_job_ns']>10123
    assert result['cost_precision']=='contains_worker_loss_upper_bound'


def test_second_orphan_closure_can_recover_after_its_directory_was_created(tmp_path):
    import time
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    from treeneuron.research.ledger import AttemptLedger
    plan=make_queue_plan({},[job('one'),job('other')]);root=tmp_path/'queue'
    ledger=AttemptLedger.create(root,plan)
    ledger.record('one','running',elapsed_ns=0,recovery_attempt=0,started_utc_ns=time.time_ns())
    ledger.record('one','interrupted_recoverable',elapsed_ns=10)
    ledger.record('one','running',elapsed_ns=10,recovery_attempt=1,started_utc_ns=time.time_ns())
    interrupted_closure=root/'jobs'/canonical_hash({'job_id':'one'})/'lost-attempt-1'
    interrupted_closure.mkdir(parents=True)
    seen=[]
    def handler(case,*args):
        seen.append(case['job_id']);return {'status':'completed_eligible'}
    result=run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)
    assert seen==['other'] and result['statuses']['one']=='irrecoverable_missing'
    assert interrupted_closure.exists()


def test_added_uncommitted_dependency_file_is_rejected(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('parent'),job('child',('parent',))]);paths=[]
    def handler(case,root,dependencies,attempt):
        if case['job_id']=='child' and attempt==0:raise KeyboardInterrupt('pause')
        (root/'value.txt').write_text('committed');paths.append(root)
        return {'status':'completed_eligible'}
    root=tmp_path/'queue'
    with pytest.raises(KeyboardInterrupt):run_queue(plan,root,handler,check_identity=lambda identity:None)
    (paths[0]/'added.txt').write_text('not committed')
    with pytest.raises(ValueError,match='commitment'):
        run_queue(plan,root,handler,check_identity=lambda identity:None,resume=True)


def test_unsafe_worker_output_is_terminalized_without_blocking_unrelated_work(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    plan=make_queue_plan({},[job('unsafe'),job('good')]);target=tmp_path/'outside.txt';target.write_text('outside')
    seen=[]
    def handler(case,root,dependencies,attempt):
        seen.append(case['job_id'])
        if case['job_id']=='unsafe':(root/'link').symlink_to(target)
        return {'status':'completed_eligible'}
    root=tmp_path/'queue';result=run_queue(plan,root,handler,check_identity=lambda identity:None)
    assert seen==['unsafe','good'] and result['statuses']['unsafe']=='input_or_dependency_failure'
    assert result['statuses']['good']=='completed_eligible'
    assert list(root.glob('jobs/*/attempt-0/link'))[0].is_symlink()


def test_terminal_record_aggregation_can_read_ineligible_candidates(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    aggregate={**job('oracle',('good_pair','dead_pair')),'dependency_policy':'all_terminal'}
    plan=make_queue_plan({},[job('good_pair'),job('dead_pair'),aggregate]);seen=[]
    def handler(case,root,dependencies,attempt):
        seen.append(case['job_id'])
        if case['job_id']=='dead_pair':return {'status':'completed_ineligible','reason':'dead unit7'}
        if case['job_id']=='oracle':
            assert {d['value']['status'] for d in dependencies.values()}=={'completed_eligible','completed_ineligible'}
        return {'status':'completed_eligible'}
    result=run_queue(plan,tmp_path/'queue',handler,check_identity=lambda identity:None)
    assert seen==['good_pair','dead_pair','oracle']
    assert result['statuses']['oracle']=='completed_eligible'


def test_job_cost_includes_identity_checks_and_authenticated_dependency_reads(tmp_path,monkeypatch):
    from treeneuron.research import workqueue as queue
    from treeneuron.research.ledger import AttemptLedger
    ticks=[0]
    def now():ticks[0]+=1000;return ticks[0]
    monkeypatch.setattr(queue.time,'perf_counter_ns',now)
    original=queue._read
    def read(reference,root,job):
        if job['job_id']=='parent':ticks[0]+=50_000_000
        return original(reference,root,job)
    monkeypatch.setattr(queue,'_read',read)
    plan=queue.make_queue_plan({},[job('parent'),job('child',('parent',))]);root=tmp_path/'queue'
    queue.run_queue(plan,root,lambda *_:dict(status='completed_eligible'),check_identity=lambda _:None)
    latest={e['attempt_id']:e for e in AttemptLedger.resume(root,plan).events}
    assert latest['child']['elapsed_ns']>=50_000_000
    assert latest['child']['preparation_ns']>=50_000_000
