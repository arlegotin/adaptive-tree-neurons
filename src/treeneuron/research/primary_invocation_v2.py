"""Read-only actual-qualified primary invocation preflight.

This module prepares the exact command but creates no request, output directory
or process. Measured resource admission and execution are separate gates.
"""
import copy
import hashlib
from pathlib import Path
import sys
import time

from treeneuron.config import canonical_json
from .deployment_worker_v2 import _source
from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
from .primary_search_declaration_v2 import _bound_primary_job
from .primary_search_worker_v2 import _ROOT,_path,_reference,_json,_episode_environment
from .workload import PracticalPlan


def _check_invocation_locations(*,paths,working_roots,proof_roots,input_files,resume):
    """Path metadata only; this helper authenticates no registry or inputs."""
    if type(paths) is not dict or set(paths)!={'request','worker','control','search'} or type(resume) is not bool:
        raise ValueError('Exact primary invocation locations required')
    paths={key:_path(str(path)) for key,path in paths.items()}
    protected=[*(Path(p) for p in working_roots+proof_roots),*(Path(p) for p in input_files),
        Path(sys.prefix).resolve(),Path(sys.executable).resolve().parent,
        *((_ROOT/p).resolve() for p in ('src','tests','configs','locks','protocol','environment','.git','.agents','.codex','.aws'))]
    for key,path in paths.items():
        if any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
            raise ValueError('Invocation outputs must preserve all working, proof, interpreter and executable inputs')
        if key=='search' and resume:
            if not path.is_dir():raise ValueError('Resume requires an existing ordinary search directory')
        elif path.exists():raise ValueError('Fresh primary invocation outputs required')
        if any(path.is_relative_to(other) or other.is_relative_to(path) for name,other in paths.items() if name!=key):
            raise ValueError('Primary request, control, worker and search outputs must be disjoint')


def _prior_input_locations(previous):
    """Read command paths only; actual closed-episode authentication is separate."""
    paths=[]
    for episode in previous:
        root=_path(episode['process_root']);start=_json((root/'started.json').read_bytes());argv=start.get('argv')
        if type(argv) is not list or len(argv)!=11:
            raise ValueError('Exact prior invocation command required')
        paths.extend([root,_path(argv[6]),_path(argv[10])])
    return paths


def prepare_primary_invocation(*,practical_plan_reference,expected_practical_plan_hash,job_id,
        support_reference,qualification_reference,request_path,worker_output,control_output,search_directory,
        previous_episodes=None,rss_limit_bytes=4*1024**3):
    """Authenticate prerequisites before reading plan bytes or writing anything.

    The returned request bytes/hash and command are exact, but not permission to
    bypass measured resource admission. P06's official sampler remains inside
    the later worker's paid clock. Full-qualified public success needs actual
    completed current-source prerequisites; no metadata fixture can supply it.
    """
    started=time.perf_counter_ns()
    registry=load_native_qualification_registry(qualification_reference)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    plan_path=_reference(practical_plan_reference);support_path=_reference(support_reference)
    paths=dict(request=request_path,worker=worker_output,control=control_output,search=search_directory)
    previous=[] if previous_episodes is None else copy.deepcopy(previous_episodes)
    request=dict(schema='research/v2-primary-search-episode-request',source_sha256=_source(),
        practical_plan_reference=copy.deepcopy(practical_plan_reference),expected_practical_plan_hash=expected_practical_plan_hash,
        job_id=job_id,support_reference=copy.deepcopy(support_reference),qualification_reference=copy.deepcopy(qualification_reference),
        search_directory=str(search_directory),rss_limit_bytes=rss_limit_bytes,previous_episodes=previous)
    from .primary_search_worker_v2 import _validate_request,_previous_episode_state
    _validate_request(request)
    prior=_previous_episode_state(request)
    _check_invocation_locations(paths=paths,working_roots=roots['working_roots'],proof_roots=roots['proof_roots'],
        input_files=[plan_path,support_path,*_prior_input_locations(previous)],resume=bool(previous))
    raw=plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=practical_plan_reference['sha256']:
        raise ValueError('Practical plan file commitment changed')
    plan=PracticalPlan(canonical_json(_json(raw)))
    value,job,spec=_bound_primary_job(practical_plan=plan,expected_practical_plan_hash=expected_practical_plan_hash,
        job_id=job_id,support_reference=support_reference)
    dataset=job['dataset'];record=registry['preparation']['datasets'][dataset]
    if (registry['identity']['scientific_source_sha256']!=value['source_sha256']
            or registry['identity']['test_and_lock_files']['locks/research-v2/core.json']!=value['runtime_sha256']
            or record['split_hashes'][str(job['outer_fold'])]!=job['split_hash']):
        raise ValueError('Actual qualified source/runtime/primary split differs from declared job')
    if prior['last'] is not None:
        from .qualified_search_audit_v2 import audit_primary_qualified_search
        body=prior['last']['body']
        audit_primary_qualified_search(practical_plan=plan,expected_practical_plan_hash=expected_practical_plan_hash,
            job_id=job_id,support_reference=support_reference,qualification_reference=qualification_reference,
            search_directory=Path(search_directory),expected_search_plan_sha256=body['search_plan_sha256'],
            result_path=Path(body['search_result_reference']['path']),expected_result_sha256=body['search_result_reference']['sha256'],
            expected_journal_sha256=body['journal_sha256'],official_schedule_reference=prior['official_schedule_reference'])
    request_bytes=(canonical_json(request)+'\n').encode();sha=hashlib.sha256(request_bytes).hexdigest()
    if (plan_path.read_bytes()!=raw or _source()!=value['source_sha256']
            or _bound_primary_job(practical_plan=plan,expected_practical_plan_hash=expected_practical_plan_hash,
                job_id=job_id,support_reference=support_reference)!=(value,job,spec)):
        raise ValueError('Invocation source/plan/support changed during preflight')
    return dict(schema='research/v2-primary-invocation-preflight',request=request,request_bytes=request_bytes,
        request_sha256=sha,argv=[sys.executable,'-P','-B','-m','treeneuron.research.primary_search_worker_v2',
            '--request',str(request_path),'--expected-request-sha256',sha,'--output',str(worker_output)],
        cwd=str(_ROOT),environment_overrides=_episode_environment(),control_output=str(control_output),
        invocation_origin_ns=started,preflight_elapsed_ns=time.perf_counter_ns()-started,
        native_qualification_authenticated=True,resource_admission_authenticated=False,
        execution_implemented=False,protected_access_authorized=False,T_accessed=False)
