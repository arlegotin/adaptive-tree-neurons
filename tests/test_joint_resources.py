"""M3 forecasts resources without imposing a wall-time efficacy deadline."""
from copy import deepcopy

import pytest

from test_joint_qualification import target


def profiles(plan):
    from treeneuron.config import canonical_hash
    from treeneuron.research.joint_qualification import workload_key
    result={}
    for job in plan['attempts']:
        key=workload_key(job)
        result[key]=dict(workload_key=key,source_sha256=plan['identity']['source_sha256'],
            execution_identity_hash=canonical_hash(plan['identity']['execution_identity']),
            status='completed_ineligible' if job['kind']=='extension_decision' else 'completed_eligible',
            elapsed_ns=1000000000,executed_updates=job['payload'].get('max_updates',0) if job['kind'] in ('fit','probe','parent') else 0,
            retained_bytes=1024*1024,peak_rss_bytes=512*1024**2,max_checkpoint_transaction_bytes=128*1024,
            parent_selected_update=8000,provenance=dict(evidence_kind='analytic unit fixture only'))
    return list(result.values())


def test_forecast_covers_every_immutable_job_and_long_elapsed_time_does_not_stop_quality():
    from treeneuron.research.joint_resources import forecast_joint_resources,joint_resource_decision
    p=target();f=forecast_joint_resources(p,profiles(p))
    assert len(f['job_retained_bytes'])==3089
    assert f['maximum_planned_updates']==3440000
    assert f['peak_rss_bytes']==512*1024**2
    assert f['forecast_typical_worker_seconds']>=3089
    long_history=[dict(attempt_id=p['attempts'][-1]['job_id'],status='completed_eligible',elapsed_ns=1000000000000000000)]
    live=dict(free_disk_bytes=1024**4,available_memory_bytes=8*1024**3)
    decision=joint_resource_decision(p,f,p['attempts'][0],long_history,live)
    assert decision['admitted'] is True
    assert decision['charged_worker_seconds']>1000000
    low=joint_resource_decision(p,f,p['attempts'][0],[],dict(live,available_memory_bytes=1024**3))
    assert low['admitted'] is False and low['reason']=='insufficient_live_memory'
    disk=joint_resource_decision(p,f,p['attempts'][0],[],dict(live,free_disk_bytes=1))
    assert disk['admitted'] is False and disk['reason']=='insufficient_live_disk'
    assert decision['target_plan_hash']==p['plan_hash']


def test_parent_history_storage_is_extrapolated_without_counting_fresh_models_as_inherited():
    from treeneuron.research.joint_resources import forecast_joint_resources
    p=target();rows=profiles(p)
    for r in rows:
        if r['workload_key'].startswith(('fit:','probe:')):r['parent_selected_update']=0
    f=forecast_joint_resources(p,rows)
    jobs={j['payload']['role']:j['job_id'] for j in p['attempts'] if j['kind']=='fit'}
    assert f['job_retained_bytes'][jobs['keep']]>f['job_retained_bytes'][jobs['fresh_upper']]


@pytest.mark.parametrize('failure',['missing_class','stale_source','failed_model','zero_updates'])
def test_unqualified_profiles_cannot_enter_a_resource_forecast(failure):
    from treeneuron.research.joint_resources import forecast_joint_resources
    p=target();rows=profiles(p)
    if failure=='missing_class':rows.pop()
    elif failure=='stale_source':rows[0]['source_sha256']='0'*64
    elif failure=='failed_model':next(r for r in rows if r['workload_key']=='fit:joint')['status']='numerical_failure'
    else:next(r for r in rows if r['workload_key']=='fit:joint')['executed_updates']=0
    with pytest.raises(ValueError):forecast_joint_resources(p,rows)
