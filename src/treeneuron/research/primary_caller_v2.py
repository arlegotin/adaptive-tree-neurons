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


_EPISODE_SCOPE='Declared caller entry through preflight, request publication, owned observer return and metadata readback. Own final outcome publication excluded. Root clock is nested, never add it twice. Additional caller interval is retained separately; complete outer cost and its prefix routing require the study ledger.'
_COMPLETE_SCOPE='Caller entry through episode publication/hash and completion preparation; own completion publication and return excluded. Root clock is nested. Final study clock must cover excluded bookkeeping; this is no native quality or complete-cost authority.'
_FLAGS=('native_qualification_authenticated','quality_eligible','whole_training_procedure_cost_authenticated','protected_access_authorized','T_accessed')


def _file_reference(path):return dict(path=str(path),sha256=_sha(path))


def _complete_caller(root,outcome,source,clock_domain):
    from .cooperative_process_v2 import _write
    from .deployment_worker_v2 import _source
    if 'process_reference' not in outcome:return outcome
    if _source()!=source:raise ValueError('Caller source changed during episode')
    references=dict(request_reference=_file_reference(root/'request.json'),caller_start_reference=_file_reference(root/'caller-start.json'),
        episode_reference=_file_reference(root/'episode.json'))
    end=time.perf_counter_ns();elapsed=end-outcome['invocation_origin_ns']
    complete=dict(schema='research/v2-primary-caller-complete-clock-1',root=str(root),caller_source_sha256=source,
        **references,process_reference=outcome['process_reference'],
        origin_ns=outcome['invocation_origin_ns'],end_ns=end,elapsed_ns=elapsed,clock_domain=clock_domain,
        root_elapsed_ns=outcome['cooperative_root_elapsed_ns'],
        additional_caller_interval_ns=elapsed-outcome['cooperative_root_elapsed_ns'],status=outcome['status'],
        scope=_COMPLETE_SCOPE,**{key:False for key in _FLAGS})
    path=root/'caller-complete.json';_write(path,complete);reference=_file_reference(path)
    returned=dict(outcome,caller_reference=reference,observed_complete_caller_elapsed_ns=elapsed)
    for key in ('unstarted_reference','closed_body_reference'):
        if key in returned:returned[key]=dict(returned[key],caller_reference=reference)
    return returned


def inspect_caller_boundary(reference,*,process_reference):
    """Reopen a declared observed boundary; no process, native/model authority."""
    from .cooperative_process_v2 import inspect_cooperative_process
    from .primary_search_worker_v2 import _path,_reference,_json
    from .deployment_worker_v2 import _source
    from .execution_clock_v2 import validate_clock_domain
    if type(process_reference) is not dict:raise ValueError('Exact owned process reference required')
    base={key:process_reference[key] for key in ('process_root','process_receipt_sha256','process_started_sha256')}
    control=_path(base['process_root']);root=control.parent;path=_reference(reference)
    if path!=root/'caller-complete.json' or not path.is_file() or _sha(path)!=reference['sha256']:
        raise ValueError('Caller boundary must be the exact committed local completion')
    value=_json(path.read_bytes())
    fields={'schema','root','caller_source_sha256','request_reference','caller_start_reference','episode_reference',
        'process_reference','origin_ns','end_ns','elapsed_ns','clock_domain','root_elapsed_ns','additional_caller_interval_ns','status','scope',*_FLAGS}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-primary-caller-complete-clock-1'
            or value['root']!=str(root) or value['caller_source_sha256']!=_source() or value['scope']!=_COMPLETE_SCOPE
            or value['process_reference']!=base or any(value.get(key) is not False for key in _FLAGS)):
        raise ValueError('Caller boundary source/schema/authority differs')
    validate_clock_domain(value['clock_domain'])
    loaded={}
    for name,filename in (('request_reference','request.json'),('caller_start_reference','caller-start.json'),('episode_reference','episode.json')):
        target=_reference(value[name])
        if target!=root/filename or not target.is_file() or _sha(target)!=value[name]['sha256']:
            raise ValueError('Caller boundary input bytes changed')
        loaded[name]=_json(target.read_bytes())
    proof=inspect_cooperative_process(root=control,expected_receipt_sha256=base['process_receipt_sha256'],expected_started_sha256=base['process_started_sha256'])
    start=_json((control/'started.json').read_bytes());caller=loaded['caller_start_reference'];episode=loaded['episode_reference']
    start_fields={'schema','argv','cwd','environment_overrides','identity','admission_identity','request_sha256',
        'invocation_origin_ns','expect_primary_body','caller_source_sha256','clock_domain'}
    episode_fields={'schema','status','request_sha256','invocation_origin_ns',*_FLAGS,'observed_caller_elapsed_ns',
        'additional_caller_interval_ns','clock_scope','process_reference','cooperative_root_elapsed_ns','process_status','returncode','child_started'}
    optional={'unstarted_reference','body_outcome_reference','closed_body_reference','search_result_reference','error_type','reason','traceback'}
    if (type(caller) is not dict or set(caller)!=start_fields or caller.get('schema')!='research/v2-primary-caller-start'
            or type(caller.get('expect_primary_body')) is not bool
            or type(episode) is not dict or not episode_fields<=set(episode)<=episode_fields|optional
            or episode.get('schema')!='research/v2-primary-caller-outcome' or episode.get('clock_scope')!=_EPISODE_SCOPE
            or episode.get('status') not in ('closed_unstarted_episode','closed_primary_body','closed_generic_worker','failed_authorized_body','caller_failed')
            or episode.get('process_status')!=proof['execution_status'] or episode.get('returncode')!=proof['returncode']
            or type(episode.get('returncode')) is not type(proof['returncode']) or episode.get('child_started') is not proof['child_started']):
        raise ValueError('Caller linked record schema or process disposition differs')
    for key in ('origin_ns','end_ns','elapsed_ns','root_elapsed_ns','additional_caller_interval_ns'):
        if type(value[key]) is not int or value[key]<0:raise ValueError('Strict caller clock quantities required')
    if (value['origin_ns']<=0 or value['end_ns']-value['origin_ns']!=value['elapsed_ns']
            or value['root_elapsed_ns']!=proof['elapsed_ns']
            or value['additional_caller_interval_ns']!=value['elapsed_ns']-proof['elapsed_ns']
            or type(episode.get('observed_caller_elapsed_ns')) is not int or value['elapsed_ns']<episode['observed_caller_elapsed_ns']
            or episode['observed_caller_elapsed_ns']<proof['elapsed_ns']
            or episode.get('additional_caller_interval_ns')!=episode['observed_caller_elapsed_ns']-proof['elapsed_ns']
            or episode.get('cooperative_root_elapsed_ns')!=proof['elapsed_ns']
            or episode.get('process_reference')!=base or episode.get('status')!=value['status']
            or any(episode.get(key) is not False for key in _FLAGS)
            or caller.get('caller_source_sha256')!=value['caller_source_sha256']
            or caller.get('clock_domain')!=value['clock_domain']
            or value['origin_ns']!=caller.get('invocation_origin_ns') or value['origin_ns']!=episode.get('invocation_origin_ns')
            or value['origin_ns']!=start.get('supervisor_started_monotonic_ns')
            or any(caller.get(key)!=start.get(key) for key in ('argv','cwd','environment_overrides','identity','admission_identity'))
            or caller.get('request_sha256')!=value['request_reference']['sha256']
            or episode.get('request_sha256')!=value['request_reference']['sha256']):
        raise ValueError('Caller clock, request, owned root or outcome boundary differs')
    if _sha(path)!=reference['sha256']:raise ValueError('Caller completion changed during reading')
    return dict(origin_ns=value['origin_ns'],end_ns=value['end_ns'],clock_domain=value['clock_domain'],
        elapsed_ns=value['elapsed_ns'],root_elapsed_ns=proof['elapsed_ns'],
        additional_caller_interval_ns=value['additional_caller_interval_ns'],caller_boundary_checked=True,
        **{key:False for key in _FLAGS},scope=_COMPLETE_SCOPE)


def _run_prepared_episode(*,prepared,episode_root,identity,admission_identity,admission_check,
        invocation_origin_ns,rss_limit_bytes,expect_primary_body=False):
    """Generic owned composition only; public actual preflight is separate."""
    from .cooperative_process_v2 import supervise_cooperative_process,inspect_cooperative_process,_write
    from .primary_search_worker_v2 import _path,_json
    from .deployment_worker_v2 import _source
    from .execution_clock_v2 import observe_clock_domain
    if type(invocation_origin_ns) is not int or not 0<invocation_origin_ns<=time.perf_counter_ns():
        raise ValueError('Caller origin must precede entry before any publication')
    root=_path(str(episode_root));request=root/'request.json';control=root/'process'
    source=_source();clock_domain=observe_clock_domain()
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
            invocation_origin_ns=invocation_origin_ns,expect_primary_body=expect_primary_body,caller_source_sha256=source,
            clock_domain=clock_domain))
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
            clock_scope=_EPISODE_SCOPE)
        _write(root/'episode.json',outcome)
    return _complete_caller(root,outcome,source,clock_domain)


def run_primary_search_episode(*,practical_plan_reference,expected_practical_plan_hash,job_id,support_reference,
        qualification_reference,resource_forecast_reference,episode_root,search_directory,
        previous_episodes=None,previous_unstarted_episodes=None,rss_limit_bytes=4*1024**3,procedure_reference=None):
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
        rss_limit_bytes=rss_limit_bytes,procedure_reference=procedure_reference)
    if root.exists() or root.is_symlink():raise ValueError('Fresh canonical caller root required')
    gate,admission_identity=_make_resource_gate(prepared['resource_context'],search_directory=search_directory)
    identity=dict(purpose='primary_search_episode',request_sha256=prepared['request_sha256'],
        source_sha256=prepared['request']['source_sha256'])
    return _run_prepared_episode(prepared=prepared,episode_root=root,identity=identity,
        admission_identity=admission_identity,admission_check=gate,invocation_origin_ns=started,
        rss_limit_bytes=rss_limit_bytes,expect_primary_body=True)
