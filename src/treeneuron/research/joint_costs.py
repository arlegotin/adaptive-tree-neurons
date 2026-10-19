"""M3 experiment costs with shared development control kept outside each model.

These contended queue clocks include evidence I/O. They are neither deployment
measurements nor a predictive or quality-cost result.
"""
from collections import Counter

from treeneuron.config import canonical_hash
from .ledger import AttemptLedger,TERMINAL
from .workqueue import authenticated_queue


def describe_joint_costs(plan,ledger):
    """Recheck ledger arithmetic; this entry alone does not authenticate files."""
    if (type(plan) is not dict or plan.get('identity',{}).get('schema')!='research/v2-M3-queue-identity'
            or plan.get('plan_hash')!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'})
            or canonical_hash(plan)!=canonical_hash(ledger.plan)):
        raise ValueError('M3 cost plan differs from the committed attempt ledger')
    ledger=AttemptLedger.resume(ledger.directory,plan)
    latest={event['attempt_id']:event for event in ledger.events}
    jobs={job['job_id']:job for job in plan['attempts']}
    if (set(latest)!=set(jobs) or any(row['status'] not in TERMINAL-{'interrupted_recoverable'} for row in latest.values())):
        raise ValueError('Complete terminal M3 inventory required; no partial cost totals')
    ancestry={};procedures={};used=set();by_kind=Counter()
    for job in plan['attempts']:
        name=job['job_id'];kind=job['kind'];parents=set(job['depends_on'])
        if any(dep not in ancestry for dep in parents):raise ValueError('Cost dependency order differs')
        if kind=='data':
            if any(jobs[dep]['kind']!='extension_decision' for dep in parents):
                raise ValueError('Only the sampling/extension control barrier may be excluded at data creation')
            parents=set()
        for dep in tuple(parents):parents.update(ancestry[dep])
        ancestry[name]=parents;by_kind[kind]+=latest[name]['elapsed_ns']
        if kind not in ('fit','evaluate'):continue
        members=parents|{name};used.update(members)
        components=Counter()
        for member in members:components[jobs[member]['kind']]+=latest[member]['elapsed_ns']
        procedures[name]=dict(status=latest[name]['status'],payload=job['payload'],
            member_job_ids=sorted(members),charged_ns=sum(components.values()),by_kind_ns=dict(sorted(components.items())),
            dependency_execution_complete=all(latest[m]['status'] in ('completed_eligible','completed_ineligible') for m in members),
            cost_precision='contains_worker_loss_upper_bound' if any(e.get('cost_precision')=='wall_time_upper_bound_after_worker_loss'
                for e in ledger.events if e['attempt_id'] in members) else 'measured_job_elapsed')
    development=sorted(set(jobs)-used)
    return dict(schema='research/v2-M3-experimental-costs',queue_plan_hash=plan['plan_hash'],
        ledger_event_hash=ledger.events[-1]['event_hash'],producer_source_sha256=plan['identity']['source_sha256'],
        terminal_jobs=len(latest),terminal_statuses=dict(Counter(row['status'] for row in latest.values())),
        cohort_charged_job_ns=sum(row['elapsed_ns'] for row in latest.values()),cohort_by_kind_ns=dict(sorted(by_kind.items())),
        procedures=procedures,development_only_job_ids=development,
        development_only_charged_job_ns=sum(latest[name]['elapsed_ns'] for name in development),
        artifact_inventories_authenticated=False,Gate_B_decided=False,quality_cost_claim=False,T_accessed=False,
        scope='Standalone experimental fit/evaluation pays each actual prerequisite once; data creation excludes only its extension-decision control barrier. Full physical queue and diagnostic/control work remain separate. Standalone rows share physical work and must not be summed as cohort cost. Initial/resume authentication, outer process startup and final publication remain external; measured clocks are contended and include archive I/O.')


def collect_joint_costs(directory,*,expected_plan_hash,expected_ledger_hash):
    """Require an inactive queue, all file inventories and a trusted endpoint."""
    with authenticated_queue(directory,expected_plan_hash=expected_plan_hash) as (plan,ledger,results):
        if not ledger.events or ledger.events[-1]['event_hash']!=expected_ledger_hash:
            raise ValueError('External M3 cost ledger endpoint differs')
        if set(results)!={job['job_id'] for job in plan['attempts']}:
            raise ValueError('Every terminal M3 result must remain authenticated')
        result=describe_joint_costs(plan,ledger)
        result['artifact_inventories_authenticated']=True
        return result
