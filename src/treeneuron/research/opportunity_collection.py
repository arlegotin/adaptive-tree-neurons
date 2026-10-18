"""Read-only authenticated M2 development collection; never a protected test."""
import hashlib
import json
from pathlib import Path

from treeneuron import train
from treeneuron.backend import runtime_hash
from treeneuron.config import canonical_hash
from .gate_a import gate_a_decision
from .opportunity_analysis import opportunity_summary
from .opportunity_graph import compile_bounded_graph
from .opportunity_workers import standalone_costs
from .workqueue import authenticated_queue


def screening_summaries(plan, results):
    """Consume authenticated producer V metrics, preserving missing coordinates."""
    cases=[j['payload'] for j in plan['attempts'] if j['kind'] in ('fit','head')
        and j['payload']['matrix'] in ('M2a','M2e_neural','M2e_head')]
    outcomes={}
    for case in cases:
        value=results.get('evaluate:'+case['attempt_id'],{}).get('value',{})
        row=dict(status=value.get('status','never_started'))
        if row['status']=='completed_eligible':
            row['V_NMSE']=value['metrics']['selected']['V']['normalized_mse']
        outcomes[case['attempt_id']]=row
    contract=plan['identity']['opportunity_decision_contract']
    initial_cases=[c for c in cases if c['teacher_index']<4]
    initial=opportunity_summary(initial_cases,{c['attempt_id']:outcomes[c['attempt_id']]
        for c in initial_cases},contract,teachers=range(4))
    final=(opportunity_summary(cases,outcomes,contract,teachers=range(8))
        if initial.get('extension_required') is True else initial)
    return initial,final


def collect_opportunity(directory, study, *, expected_queue_hash, control_audit_path,
                        expected_control_audit_sha256, expected_control_plan_hash):
    """Authenticate the frozen full DAG, its files, M1 audit and consumed metrics.

    M2 metrics are authenticated producer outputs, not independent raw arithmetic.
    A separate prediction reconstruction/report audit remains required.
    """
    expected=compile_bounded_graph(study)
    if expected['plan_hash']!=expected_queue_hash:
        raise ValueError('Collection study differs from the external queue commitment')
    audit_path=Path(control_audit_path)
    if audit_path.is_symlink() or not audit_path.is_file():raise ValueError('Unsafe control audit reference')
    audit_bytes=audit_path.read_bytes()
    if hashlib.sha256(audit_bytes).hexdigest()!=expected_control_audit_sha256:
        raise ValueError('Control audit commitment differs')
    control=json.loads(audit_bytes)
    if control.get('plan_hash')!=expected_control_plan_hash or control.get('T_accessed') is not False:
        raise ValueError('Control audit producer plan or protected-role scope differs')
    with authenticated_queue(directory,expected_plan_hash=expected_queue_hash) as (plan,ledger,results):
        if canonical_hash(plan)!=canonical_hash(expected):
            raise ValueError('Collected graph differs from the exact bounded current-source plan')
        initial,final=screening_summaries(plan,results)
        node='extension_decision:primary'
        decision=results.get(node,{}).get('value',{})
        if decision.get('status') in ('completed_eligible','completed_ineligible'):
            if canonical_hash(decision.get('summary',{}))!=canonical_hash(initial):
                raise ValueError('Stored extension decision differs from its authenticated V inputs')
        statuses={j['job_id']:'never_started' for j in plan['attempts']}
        statuses.update({e['attempt_id']:e['status'] for e in ledger.events})
        gate=gate_a_decision(control,initial,final,statuses,expected_job_ids=list(statuses),
            extension_status=statuses[node],opportunity_contract=plan['identity']['opportunity_decision_contract'])
        return dict(schema='research/v2-opportunity-collection',queue_plan_hash=expected_queue_hash,
            study_plan_hash=study['plan_hash'],producer_source_sha256=study['source_sha256'],
            collector_source_sha256=train.source_hash(),collector_runtime_sha256=runtime_hash(),
            control_audit_sha256=expected_control_audit_sha256,control_plan_hash=expected_control_plan_hash,
            ledger_event_hash=ledger.events[-1]['event_hash'] if ledger.events else '0'*64,
            result_commitments=[dict(job_id=e['attempt_id'],event_hash=e['event_hash'],
                status=e['status'],result=e['result']) for e in ledger.events if 'result' in e],
            initial_summary=initial,final_summary=final,gate_A=gate,statuses=statuses,
            standalone_costs=standalone_costs(plan,ledger),charged_cumulative_job_ns=ledger.charged_ns,
            cost_precision=('contains_worker_loss_upper_bound' if any(e.get('cost_precision')==
                'wall_time_upper_bound_after_worker_loss' for e in ledger.events) else 'measured_job_elapsed'),
            measured_physical_elapsed_ns=None,
            cost_scope='cumulative charged jobs may include downtime upper bounds; physical invocation/startup receipts separate; standalone sums share prerequisites',
            evidence_level='authenticated M2 producer metrics and independent M1 prediction arithmetic',
            M2_raw_prediction_arithmetic_independent=False,T_accessed=False,efficacy_claim=False)
