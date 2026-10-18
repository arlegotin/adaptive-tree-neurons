"""Final collection authenticates terminal files before using their metrics."""
import copy
import json
from pathlib import Path

import pytest


def queue(tmp_path):
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    jobs=[dict(job_id=n,kind='fixture',payload={},depends_on=[]) for n in ('first','second')]
    plan=make_queue_plan({'source':'fixture'},jobs);root=tmp_path/'queue'
    def handler(job,directory,*_):
        (directory/'values.json').write_text('{"value":17}')
        return dict(status='completed_eligible',value=17)
    run_queue(plan,root,handler,check_identity=lambda _:None)
    return plan,root


def test_read_only_collection_authenticates_complete_files_and_statuses(tmp_path):
    from treeneuron.research.workqueue import authenticated_queue
    plan,root=queue(tmp_path)
    before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
    with authenticated_queue(root,expected_plan_hash=plan['plan_hash']) as (saved,ledger,results):
        assert saved==plan and not ledger.pending_ids
        assert set(results)=={'first','second'} and results['first']['value']['value']==17
    assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['file','plan','event','status'])
def test_collection_rejects_changed_commitments_and_status_disagreement(tmp_path,mutation):
    from treeneuron.config import canonical_hash,canonical_json
    from treeneuron.research.workqueue import authenticated_queue
    plan,root=queue(tmp_path)
    if mutation=='file':next(root.rglob('values.json')).write_text('{"value":18}')
    elif mutation=='plan':
        value=json.loads((root/'plan.json').read_text());value['identity']['source']='changed'
        value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
        (root/'plan.json').write_text(canonical_json(value))
    else:
        lines=[json.loads(l) for l in (root/'events.jsonl').read_text().splitlines()]
        if mutation=='event':lines[-1]['elapsed_ns']+=1
        else:
            lines[-1]['status']='completed_ineligible'
            lines[-1]['event_hash']=canonical_hash({k:v for k,v in lines[-1].items() if k!='event_hash'})
        (root/'events.jsonl').write_text(''.join(canonical_json(l)+'\n' for l in lines))
    with pytest.raises(ValueError):
        with authenticated_queue(root,expected_plan_hash=plan['plan_hash']):pass


def test_collection_refuses_active_worker_without_mutating_lease(tmp_path):
    from treeneuron.research.workqueue import authenticated_queue
    from treeneuron.research.ledger import StudyLease
    plan,root=queue(tmp_path)
    with StudyLease(root):
        before=(root/'worker-lease.json').read_bytes()
        with pytest.raises(PermissionError,match='active'):
            with authenticated_queue(root,expected_plan_hash=plan['plan_hash']):pass
        assert (root/'worker-lease.json').read_bytes()==before


def screening():
    from treeneuron.research.config import expand,load_research_config
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    jobs=[];results={}
    for case in study['attempts']+study['conditional_attempts']:
        if case['matrix'] not in ('M2a','M2e_neural','M2e_head'):continue
        name='procedure:'+case['attempt_id']
        jobs.append(dict(job_id=name,kind='head' if case['matrix']=='M2e_head' else 'fit',payload=case))
        status='completed_ineligible' if 'fallback' in case['admission'] else 'completed_eligible'
        value=dict(status=status)
        if status=='completed_eligible':
            arm=(case['matrix']=='M2a' and case['role']=='top')
            risk=.97 if arm and case['teacher_index']<2 else .99 if arm else 1.
            value['metrics']={'selected':{'V':{'normalized_mse':risk},'E':{'forbidden':'do not inspect'}}}
        results['evaluate:'+case['attempt_id']]=dict(value=value)
    return dict(identity={'opportunity_decision_contract':study['opportunity_decision_contract']},attempts=jobs),results


def test_summary_collection_uses_V_and_preserves_fixed_initial_or_extended_scope():
    from treeneuron.research.opportunity_collection import screening_summaries
    plan,results=screening()
    initial,final=screening_summaries(plan,results)
    assert initial==final and initial['teacher_ids']==list(range(4))
    # No useful arm on the first4 requires the whole fixed next4, not a new seed.
    for value in results.values():
        if value['value']['status']=='completed_eligible':
            value['value']['metrics']['selected']['V']['normalized_mse']=1.
    initial,final=screening_summaries(plan,results)
    assert initial['extension_required'] and final['teacher_ids']==list(range(8))
    assert not final['extension_required'] and final['extension_exhausted']


def test_missing_primary_input_is_retained_as_missing_not_dropped():
    from treeneuron.research.opportunity_collection import screening_summaries
    plan,results=screening();del results[next(iter(results))]
    initial,final=screening_summaries(plan,results)
    assert initial['status']=='irrecoverable_missing' and final==initial


def test_production_collector_blocks_empty_real_graph_and_binds_audit(tmp_path):
    import hashlib
    from treeneuron.research.config import expand,load_research_config
    from treeneuron.research.opportunity_graph import compile_bounded_graph
    from treeneuron.research.ledger import AttemptLedger,StudyLease
    from treeneuron.research.opportunity_collection import collect_opportunity
    study=expand(load_research_config(Path('configs/research/v2/M2-opportunity.json'))).to_dict()
    plan=compile_bounded_graph(study);root=tmp_path/'queue'
    AttemptLedger.create(root,plan)
    with StudyLease(root):pass
    control=tmp_path/'control-audit.json'
    control.write_text(json.dumps(dict(plan_hash='c'*64,T_accessed=False,status='incomplete_or_ineligible')))
    kwargs=dict(expected_queue_hash=plan['plan_hash'],control_audit_path=control,
        expected_control_audit_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),expected_control_plan_hash='c'*64)
    result=collect_opportunity(root,study,**kwargs)
    assert result['gate_A']['decision']=='blocked'
    assert len(result['statuses'])==3853 and result['charged_cumulative_job_ns']==0
    assert result['M2_raw_prediction_arithmetic_independent'] is False
    assert result['T_accessed'] is False
    control.write_text(control.read_text()+' ')
    with pytest.raises(ValueError,match='audit commitment'):
        collect_opportunity(root,study,**kwargs)


@pytest.mark.parametrize('mutation',['failure_result','failed_raw'])
def test_collection_authenticates_paid_failure_history_after_successful_retry(tmp_path,mutation):
    from treeneuron.research.workqueue import make_queue_plan,run_queue,authenticated_queue
    from treeneuron.research.ledger import AttemptLedger
    plan=make_queue_plan({},[dict(job_id='retry',kind='fixture',payload={},depends_on=[])])
    root=tmp_path/'queue'
    def handler(job,directory,dependencies,attempt):
        (directory/'raw.txt').write_text('retained paid work')
        if attempt==0:raise OSError('transient competing process')
        return dict(status='completed_eligible')
    with pytest.raises(OSError):run_queue(plan,root,handler,check_identity=lambda _:None)
    run_queue(plan,root,handler,check_identity=lambda _:None,resume=True)
    ledger=AttemptLedger.resume(root,plan)
    failure=next(e for e in ledger.events if e['status']=='interrupted_recoverable')
    if mutation=='failure_result':Path(failure['result']['path']).unlink()
    else:
        value=json.loads(Path(failure['result']['path']).read_text())['value']
        Path(value['raw_attempt_evidence']['path'],'raw.txt').write_text('changed paid work')
    with pytest.raises(ValueError,match='commitment|inventory'):
        with authenticated_queue(root,expected_plan_hash=plan['plan_hash']):pass


def test_collection_preserves_worker_loss_cost_precision(tmp_path,monkeypatch):
    import hashlib,time
    from treeneuron.research import opportunity_collection as collection
    from treeneuron.research.workqueue import make_queue_plan,run_queue
    from treeneuron.research.ledger import AttemptLedger
    from treeneuron.research.opportunity_plan import opportunity_decision_contract
    contract=opportunity_decision_contract(dict(seed_root=2026101801,study_id='M2-opportunity'))
    jobs=[dict(job_id=n,kind='fixture',payload={},depends_on=[]) for n in ('lost','extension_decision:primary')]
    plan=make_queue_plan(dict(opportunity_decision_contract=contract),jobs);root=tmp_path/'queue'
    ledger=AttemptLedger.create(root,plan)
    ledger.record('lost','running',elapsed_ns=0,recovery_attempt=0,started_utc_ns=time.time_ns()-3600*10**9)
    def handler(job,*_):
        return dict(status='numerical_failure' if job['job_id']=='extension_decision:primary' else 'completed_eligible')
    run_queue(plan,root,handler,check_identity=lambda _:None,resume=True)
    monkeypatch.setattr(collection,'compile_bounded_graph',lambda _:plan)
    audit=tmp_path/'control.json';audit.write_text(json.dumps(dict(plan_hash='c'*64,T_accessed=False)))
    result=collection.collect_opportunity(root,dict(plan_hash='study',source_sha256='source'),
        expected_queue_hash=plan['plan_hash'],control_audit_path=audit,
        expected_control_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),expected_control_plan_hash='c'*64)
    assert result['charged_cumulative_job_ns']>=3600*10**9
    assert result['cost_precision']=='contains_worker_loss_upper_bound'
    assert result['measured_physical_elapsed_ns'] is None
    assert 'charged_physical_job_ns' not in result
    assert len(result['result_commitments'])==2
