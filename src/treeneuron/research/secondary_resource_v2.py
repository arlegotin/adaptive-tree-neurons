"""Advisory live resource decisions after the actual primary/global barrier.

Resource arithmetic never qualifies a new workload or overrides owned RSS and
checkpoint-space guards. Legacy primary advice still refuses secondary jobs.
"""
import copy


def _secondary_resource_decision(*,forecast,job,live):
    from .native_resource_forecast_v2 import _native_resource_decision
    return _native_resource_decision(forecast=forecast,job=job,live=live,expected_tranche='secondary')


def secondary_resource_decision(*,barrier_reference,qualification_reference,job_id,resource_forecast_reference,live):
    from .practical_global_freeze_v2 import load_global_primary_barrier
    barrier=load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference)
    payload=barrier.payload
    if payload.get('secondary_development_access_authorized') is not True or payload.get('protected_access_authorized') is not False:
        raise ValueError('Actual frozen primary/global gates required before secondary advice')
    from treeneuron.config import canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .secondary_native_search_v2 import _bound_secondary_job
    from .native_resource_forecast_v2 import load_native_practical_resource_forecast
    plan=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    value,job,_=_bound_secondary_job(practical_plan=plan,expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        job_id=job_id,support_reference=payload['support_reference'])
    if value['source_sha256']!=payload['source_sha256']:raise ValueError('Frozen secondary resource source differs')
    forecast=load_native_practical_resource_forecast(resource_forecast_reference,practical_plan=plan,
        expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'],
        qualification_reference=qualification_reference)
    result=_secondary_resource_decision(forecast=forecast,job=job,live=live)
    if canonical_json(load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference).payload)!=canonical_json(payload):
        raise ValueError('Actual global resource binding changed during decision')
    return dict(result,global_barrier_reference=copy.deepcopy(barrier_reference),
        secondary_job_binding_authenticated=True,
        scope='Advisory resource proxies for the exact genuinely frozen secondary job. No secondary workload qualification, owned execution, protected access or resource upper-bound guarantee; pause and preserve exact state on shortage.')
