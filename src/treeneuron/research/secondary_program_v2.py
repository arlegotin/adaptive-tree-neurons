"""Serial secondary searches after genuine global freeze; no T/cost authority."""
import copy
from pathlib import Path
import time


def build_secondary_program_plan(*,practical_plan,global_barrier_reference,**kwargs):
    """Metadata only; supplied global/native references grant no authority."""
    from treeneuron.config import canonical_hash
    from .primary_program_v2 import build_primary_program_plan
    from .primary_selection_v2 import _primary
    from .primary_search_worker_v2 import _reference
    _reference(global_barrier_reference)
    primary=build_primary_program_plan(practical_plan=practical_plan,**kwargs)
    value,_=_primary(practical_plan,kwargs['expected_practical_plan_hash'])
    jobs=[job for job in value['jobs'] if job['tranche']=='secondary']
    if len(jobs)!=672:raise ValueError('Exactly672 fixed secondary jobs required')
    result=dict(primary,schema='research/v2-secondary-program-plan',jobs=copy.deepcopy(jobs),
        global_barrier_reference=copy.deepcopy(global_barrier_reference),
        scope='Immutable secondary-only metadata queue; actual native/global qualification, resources, terminal audits and complete840/T/cost barriers remain separate')
    result['program_hash']=canonical_hash({k:v for k,v in result.items() if k!='program_hash'})
    return result


def run_secondary_program(*,barrier_reference,qualification_reference,resource_forecast_reference,directory,
        rss_limit_bytes=4*1024**3,expected_journal_sha256=None,max_episode_starts=672):
    started=time.perf_counter_ns()
    from .practical_global_freeze_v2 import load_global_primary_barrier
    barrier=load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference)
    payload=barrier.payload
    if payload.get('secondary_development_access_authorized') is not True or payload.get('protected_access_authorized') is not False:
        raise ValueError('Actual frozen primary/global gate required for the secondary queue')
    from treeneuron.config import canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .native_qualification_registry_v2 import load_native_qualification_registry
    from .native_resource_forecast_v2 import load_native_practical_resource_forecast
    from .secondary_episode_v2 import _proof_roots,run_secondary_search_episode
    from .secondary_native_search_v2 import audit_secondary_qualified_search
    from .primary_program_v2 import _check_program_location,_run_program_queue
    registry=load_native_qualification_registry(qualification_reference)
    practical=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    program=build_secondary_program_plan(practical_plan=practical,global_barrier_reference=barrier_reference,
        **{k:payload[k] for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference')},
        qualification_reference=qualification_reference,resource_forecast_reference=resource_forecast_reference,
        directory=directory,rss_limit_bytes=rss_limit_bytes)
    load_native_practical_resource_forecast(resource_forecast_reference,practical_plan=practical,
        expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'],
        qualification_reference=qualification_reference)
    working,proofs=_proof_roots(payload,registry,qualification_reference,barrier_reference)
    _check_program_location(directory,working_roots=working,proof_roots=proofs,
        input_files=[_ref(payload[k],{}) for k in ('practical_plan_reference','support_reference','procedure_reference')]+[resource_forecast_reference['path']])
    def execute(*,program,job,episode_root,previous_episodes,previous_unstarted_episodes):
        run_secondary_search_episode(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
            resource_forecast_reference=resource_forecast_reference,job_id=job['job_id'],episode_root=episode_root,
            search_directory=Path(program['directory'])/'searches'/job['job_id'],previous_episodes=previous_episodes,
            previous_unstarted_episodes=previous_unstarted_episodes,rss_limit_bytes=rss_limit_bytes)
    def audit(**closed):
        body=closed['body'];job=closed['job']
        result=audit_secondary_qualified_search(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
            job_id=job['job_id'],search_directory=Path(program['directory'])/'searches'/job['job_id'],expected_search_plan_sha256=body['search_plan_sha256'],
            result_path=Path(body['search_result_reference']['path']),expected_result_sha256=body['search_result_reference']['sha256'],
            expected_journal_sha256=body['journal_sha256'],official_schedule_reference=body['official_schedule_reference'])
        if result.get('completed_search') is not True or any(result.get(k) is not True for k in
                ('HPO_journal_authenticated','all_attempt_native_evidence_authenticated','all_eligible_V_scores_authenticated','native_workload_qualified')):
            raise ValueError('Every terminal secondary search needs its genuine native/HPO/V audit')
    return _run_program_queue(program,execute_episode=execute,audit_terminal=audit,expected_journal_sha256=expected_journal_sha256,
        max_episode_starts=max_episode_starts,invocation_origin_ns=started)
