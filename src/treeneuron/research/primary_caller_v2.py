"""Owned primary caller composition; quality and whole-cost gates are separate.

Only standard-library imports precede entry into actual qualified preflight.
No caller-supplied true flag can replace the public qualification loader.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_prepared_episode(*,prepared,episode_root,identity,admission_identity,admission_check,
        invocation_origin_ns,rss_limit_bytes,expect_primary_body=False):
    """Generic owned composition only; public actual preflight is separate."""
    from .cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process,_write
    from .primary_search_worker_v2 import _path,_json
    if type(invocation_origin_ns) is not int or not 0<invocation_origin_ns<=time.perf_counter_ns():
        raise ValueError('Caller origin must precede entry before any publication')
    root=_path(str(episode_root));request=root/'request.json';control=root/'process'
    raw=prepared['request_bytes']
    if (root.exists() or root.is_symlink() or type(raw) is not bytes
            or hashlib.sha256(raw).hexdigest()!=prepared['request_sha256']):
        raise ValueError('Fresh caller root and exact prepared request bytes required')
    root.mkdir(parents=True,exist_ok=False)
    outcome=dict(schema='research/v2-primary-caller-outcome',status='caller_failed',
        request_sha256=prepared['request_sha256'],invocation_origin_ns=invocation_origin_ns,
        native_qualification_authenticated=False,quality_eligible=False,whole_training_procedure_cost_authenticated=False,
        protected_access_authorized=False,T_accessed=False)
    process=None
    try:
        _write(root/'caller-start.json',dict(schema='research/v2-primary-caller-start',argv=prepared['argv'],
            cwd=prepared['cwd'],environment_overrides=prepared['environment_overrides'],identity=identity,
            admission_identity=admission_identity,request_sha256=prepared['request_sha256'],
            invocation_origin_ns=invocation_origin_ns,expect_primary_body=expect_primary_body))
        with request.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        if _sha(request)!=prepared['request_sha256']:raise ValueError('Published request bytes changed')
        process=supervise_cooperative_process(argv=prepared['argv'],cwd=prepared['cwd'],
            env_overrides=prepared['environment_overrides'],output=control,identity=identity,
            admission_identity=admission_identity,admission_check=admission_check,rss_pause_bytes=rss_limit_bytes,
            invocation_origin_ns=invocation_origin_ns,admit_before_spawn=True)
        reference=dict(process_root=str(control),process_receipt_sha256=_sha(control/'process.json'),
            process_started_sha256=_sha(control/'started.json'))
        proof=inspect_cooperative_process(root=control,expected_receipt_sha256=reference['process_receipt_sha256'],
            expected_started_sha256=reference['process_started_sha256'])
        outcome.update(process_reference=reference,cooperative_root_elapsed_ns=proof['elapsed_ns'],
            process_status=proof['execution_status'],returncode=proof['returncode'],child_started=proof['child_started'])
        if proof['grant_publication_started'] is False:
            outcome.update(status='closed_unstarted_episode',unstarted_reference=reference)
        else:
            body=control/'body-outcome.json'
            if body.exists():
                value=_json(body.read_bytes())
                outcome['body_outcome_reference']=dict(path=str(body),sha256=_sha(body))
                if (proof['returncode']==0 and proof['worker_protocol_closed']
                        and value.get('status')=='closed_search_body' and value.get('request_sha256')==prepared['request_sha256']):
                    outcome.update(status='closed_primary_body',closed_body_reference=dict(reference,body_outcome_sha256=_sha(body)),
                        search_result_reference=value['search_result_reference'])
                else:outcome['status']='failed_authorized_body'
            elif not expect_primary_body and proof['returncode']==0 and proof['worker_protocol_closed']:
                outcome['status']='closed_generic_worker'
            else:outcome['status']='failed_authorized_body'
        if _sha(request)!=prepared['request_sha256']:raise ValueError('Caller request changed during execution')
    except BaseException as error:
        outcome.update(status='caller_failed',error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
    finally:
        elapsed=time.perf_counter_ns()-invocation_origin_ns
        outcome.update(observed_caller_elapsed_ns=elapsed,
            additional_caller_interval_ns=None if process is None else elapsed-process['elapsed_ns'],
            clock_scope='Declared caller entry through preflight, request publication, owned observer return and metadata readback. Own final outcome publication excluded. Root clock is nested, never add it twice. Additional caller interval is retained separately; complete outer cost and its prefix routing require the study ledger.')
        _write(root/'episode.json',outcome)
    return outcome


def run_primary_search_episode(*,practical_plan_reference,expected_practical_plan_hash,job_id,support_reference,
        qualification_reference,resource_forecast_reference,episode_root,search_directory,
        previous_episodes=None,previous_unstarted_episodes=None,rss_limit_bytes=4*1024**3):
    """Run one exact owned episode, or retain resource denial; never auto-retry.

    Actual complete qualified success remains separate from component fixtures.
    The global procedure/readiness and model/whole-cost audit are still required
    before scientific production or protected evaluation.
    """
    started=time.perf_counter_ns()
    if not sys.dont_write_bytecode:raise ValueError('Qualified caller requires bytecode writes disabled')
    from .primary_invocation_v2 import prepare_primary_invocation
    from .primary_search_worker_v2 import _path,_make_resource_gate
    root=_path(str(episode_root))
    prepared=prepare_primary_invocation(practical_plan_reference=practical_plan_reference,
        expected_practical_plan_hash=expected_practical_plan_hash,job_id=job_id,support_reference=support_reference,
        qualification_reference=qualification_reference,resource_forecast_reference=resource_forecast_reference,
        request_path=root/'request.json',worker_output=root/'worker',control_output=root/'process',
        search_directory=search_directory,previous_episodes=previous_episodes,
        previous_unstarted_episodes=[] if previous_unstarted_episodes is None else previous_unstarted_episodes,
        rss_limit_bytes=rss_limit_bytes)
    if root.exists() or root.is_symlink():raise ValueError('Fresh canonical caller root required')
    gate,admission_identity=_make_resource_gate(prepared['resource_context'],search_directory=search_directory)
    identity=dict(purpose='primary_search_episode',request_sha256=prepared['request_sha256'],
        source_sha256=prepared['request']['source_sha256'])
    return _run_prepared_episode(prepared=prepared,episode_root=root,identity=identity,
        admission_identity=admission_identity,admission_check=gate,invocation_origin_ns=started,
        rss_limit_bytes=rss_limit_bytes,expect_primary_body=True)
