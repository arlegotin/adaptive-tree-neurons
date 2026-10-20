"""Owned secondary episodes; real global gates precede plan/data/output access.

The shared caller record format retains its historical names. The distinct
strict request, worker purpose, body schema and global reference bind the tranche.
No complete672/840 model, protected or whole-cost authority is provided here.
"""
import copy
import hashlib
from pathlib import Path
import sys
import time


def _binding(*,barrier_reference,qualification_reference,job_id):
    from .practical_global_freeze_v2 import load_global_primary_barrier
    barrier=load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference)
    payload=barrier.payload
    if payload.get('secondary_development_access_authorized') is not True or payload.get('protected_access_authorized') is not False:
        raise ValueError('Actual global primary barrier required for a secondary episode')
    from treeneuron.config import canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .secondary_native_search_v2 import _bound_secondary_job
    from .native_qualification_registry_v2 import load_native_qualification_registry
    plan=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    value,job,spec=_bound_secondary_job(practical_plan=plan,expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        job_id=job_id,support_reference=payload['support_reference'])
    registry=load_native_qualification_registry(qualification_reference)
    if (value['source_sha256']!=payload['source_sha256'] or registry['identity']['scientific_source_sha256']!=value['source_sha256']
            or registry['identity']['test_and_lock_files']['locks/research-v2/core.json']!=value['runtime_sha256']
            or registry['preparation']['datasets'][job['dataset']]['split_hashes'][str(job['outer_fold'])]!=job['split_hash']):
        raise ValueError('Actual secondary source/runtime/frozen split differs')
    return barrier,registry,plan,job,spec


def _proof_roots(payload,registry,qualification_reference,barrier_reference):
    from .native_qualification_registry_v2 import native_registry_storage_roots
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    proofs=[*roots['proof_roots'],barrier_reference['path'],payload['qualified_primary_models_reference']['path']]
    for row in payload['primary_models']:
        proofs.append(row['search_directory'])
        proofs.extend(view['model_binding']['snapshot_root'] for view in row['budget_view_models'].values() if view['model_binding'] is not None)
    return roots['working_roots'],proofs


def _resources(request,barrier,plan,job):
    from .native_resource_forecast_v2 import load_native_practical_resource_forecast
    forecast=load_native_practical_resource_forecast(request['resource_forecast_reference'],practical_plan=plan,
        expected_practical_plan_hash=barrier.payload['expected_practical_plan_hash'],support_reference=barrier.payload['support_reference'],
        qualification_reference=request['qualification_reference'])
    return dict(forecast=forecast,job=job,reference=copy.deepcopy(request['resource_forecast_reference']),
        global_barrier_reference=copy.deepcopy(request['global_barrier_reference']))


def _check_request_binding(request,payload):
    from treeneuron.config import canonical_json
    from .primary_search_worker_v2 import _validate_request,_is_secondary
    _validate_request(request)
    expected=dict(source_sha256=payload['source_sha256'],practical_plan_reference=payload['practical_plan_reference'],
        expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'],
        procedure_reference=payload['procedure_reference'],qualification_reference=payload['qualification_reference'])
    if not _is_secondary(request) or any(canonical_json(dict(value=request[k]))!=canonical_json(dict(value=v)) for k,v in expected.items()):
        raise ValueError('Secondary request cannot replace the frozen global plan/source/procedure/support/native pins')


def _audit_prior(request,prior,official):
    if prior['last'] is None:return
    from .secondary_native_search_v2 import audit_secondary_qualified_search
    body=prior['last']['body']
    audit_secondary_qualified_search(barrier_reference=request['global_barrier_reference'],qualification_reference=request['qualification_reference'],
        job_id=request['job_id'],search_directory=Path(request['search_directory']),expected_search_plan_sha256=body['search_plan_sha256'],
        result_path=Path(body['search_result_reference']['path']),expected_result_sha256=body['search_result_reference']['sha256'],
        expected_journal_sha256=body['journal_sha256'],official_schedule_reference=official)


def prepare_secondary_invocation(*,barrier_reference,qualification_reference,resource_forecast_reference,job_id,
        request_path,worker_output,control_output,search_directory,previous_episodes=None,previous_unstarted_episodes=None,rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns()
    barrier,registry,plan,job,spec=_binding(barrier_reference=barrier_reference,qualification_reference=qualification_reference,job_id=job_id)
    payload=barrier.payload
    from treeneuron.config import canonical_json
    from .primary_search_worker_v2 import _ROOT,_episode_environment,_previous_episode_state
    from .primary_invocation_v2 import _check_invocation_locations,_prior_input_locations
    from .primary_selection_v2 import _ref
    request=dict(schema='research/v2-secondary-program-episode-request',source_sha256=payload['source_sha256'],
        **{k:copy.deepcopy(payload[k]) for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference')},
        qualification_reference=copy.deepcopy(qualification_reference),global_barrier_reference=copy.deepcopy(barrier_reference),
        resource_forecast_reference=copy.deepcopy(resource_forecast_reference),job_id=job_id,search_directory=str(search_directory),rss_limit_bytes=rss_limit_bytes,
        previous_episodes=[] if previous_episodes is None else copy.deepcopy(previous_episodes),
        previous_unstarted_episodes=[] if previous_unstarted_episodes is None else copy.deepcopy(previous_unstarted_episodes))
    _check_request_binding(request,payload);prior=_previous_episode_state(request)
    working,proofs=_proof_roots(payload,registry,qualification_reference,barrier_reference)
    inputs=[_ref(payload[k],{}) for k in ('practical_plan_reference','support_reference','procedure_reference')]
    from .primary_search_worker_v2 import _reference
    inputs.extend([_reference(resource_forecast_reference),*_prior_input_locations(request['previous_episodes']),
        *_prior_input_locations(request['previous_unstarted_episodes'])])
    _check_invocation_locations(paths=dict(request=request_path,worker=worker_output,control=control_output,search=search_directory),
        working_roots=working,proof_roots=proofs,input_files=inputs,resume=bool(request['previous_episodes']))
    resources=_resources(request,barrier,plan,job);_audit_prior(request,prior,prior['official_schedule_reference'])
    if _binding(barrier_reference=barrier_reference,qualification_reference=qualification_reference,job_id=job_id)[0].payload!=payload:
        raise ValueError('Frozen secondary binding changed during invocation preflight')
    raw=(canonical_json(request)+'\n').encode();digest=hashlib.sha256(raw).hexdigest()
    return dict(schema='research/v2-secondary-invocation-preflight',request=request,request_bytes=raw,request_sha256=digest,
        argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2','--request',str(request_path),
            '--expected-request-sha256',digest,'--output',str(worker_output)],cwd=str(_ROOT),environment_overrides=_episode_environment(),
        control_output=str(control_output),invocation_origin_ns=started,preflight_elapsed_ns=time.perf_counter_ns()-started,
        resource_context=resources,native_qualification_authenticated=True,resource_admission_authenticated=False,
        execution_implemented=False,protected_access_authorized=False,T_accessed=False)


def _prepare_secondary_episode(*,request,output):
    barrier,registry,plan,job,spec=_binding(barrier_reference=request['global_barrier_reference'],
        qualification_reference=request['qualification_reference'],job_id=request['job_id'])
    payload=barrier.payload;_check_request_binding(request,payload)
    from .primary_search_worker_v2 import _path,_previous_episode_state,_sampler_runtime
    from .primary_native_preparation_v2 import _check_preparation_locations
    output=_path(str(output));search=_path(request['search_directory']);working,proofs=_proof_roots(payload,registry,request['qualification_reference'],request['global_barrier_reference'])
    _check_preparation_locations(output=output,search=search,working_roots=working,proof_roots=proofs,rss_limit_bytes=request['rss_limit_bytes'])
    if any(Path(p).is_relative_to(search) or Path(p).is_relative_to(output) for p in [*working,*proofs]):
        raise ValueError('Secondary worker output/search cannot contain frozen or qualified inputs')
    prior=_previous_episode_state(request);resources=_resources(request,barrier,plan,job)
    official=prior['official_schedule_reference'];_audit_prior(request,prior,official)
    output.mkdir(parents=True,exist_ok=False)
    if job['family_id']=='P06' and not request['previous_episodes']:
        from .native_schedule_v2 import prepare_realmlp_native_schedule
        interpreter,overlays=_sampler_runtime(registry,job)
        generated=prepare_realmlp_native_schedule(feature_schema=spec['feature_schema'],hpo_seed=job['hpo_seed'],output=output/'native-sampler',
            interpreter=interpreter,extra_pythonpath=overlays,rss_limit_bytes=request['rss_limit_bytes'])
        if generated['status']!='completed_native_schedule':raise ValueError('Official secondary sampler did not complete; paid evidence retained')
        official=generated['reference']
    from .secondary_native_search_v2 import prepare_secondary_native_callback
    prepared=prepare_secondary_native_callback(barrier_reference=request['global_barrier_reference'],qualification_reference=request['qualification_reference'],
        job_id=job['job_id'],output=output/'preparation',search_directory=search,official_schedule_reference=official,rss_limit_bytes=request['rss_limit_bytes'])
    return prepared,prior,official,resources


def run_secondary_search_episode(*,barrier_reference,qualification_reference,resource_forecast_reference,job_id,episode_root,
        search_directory,previous_episodes=None,previous_unstarted_episodes=None,rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns()
    if not sys.dont_write_bytecode:raise ValueError('Qualified caller requires bytecode writes disabled')
    root=Path(episode_root)
    prepared=prepare_secondary_invocation(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
        resource_forecast_reference=resource_forecast_reference,job_id=job_id,request_path=root/'request.json',worker_output=root/'worker',
        control_output=root/'process',search_directory=search_directory,previous_episodes=previous_episodes,
        previous_unstarted_episodes=previous_unstarted_episodes,rss_limit_bytes=rss_limit_bytes)
    from .primary_search_worker_v2 import _make_resource_gate,_episode_purpose
    from .primary_caller_v2 import _run_prepared_episode
    gate,identity=_make_resource_gate(prepared['resource_context'],search_directory=search_directory)
    return _run_prepared_episode(prepared=prepared,episode_root=root,
        identity=dict(purpose=_episode_purpose(prepared['request']),request_sha256=prepared['request_sha256'],source_sha256=prepared['request']['source_sha256']),
        admission_identity=identity,admission_check=gate,invocation_origin_ns=started,rss_limit_bytes=rss_limit_bytes,expect_primary_body=True)
