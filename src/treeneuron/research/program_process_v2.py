"""Owned cold program envelope; operational clocks grant no scientific gate.

The public runner qualifies real inputs before output. The cold worker repeats
the public program gates after grant. An authorized root receives cooperative
cleanup only; nested supervisors must join their own workers before returning.
"""
import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .deployment_worker_v2 import _source,_digest,_json,_ROOT
from .execution_clock_v2 import observe_clock_domain,validate_clock_domain,same_clock_domain,_read_reference


_FLAGS=('quality_eligible','native_qualification_authenticated','whole_training_procedure_cost_authenticated',
    'protected_access_authorized','T_accessed','nested_program_closure_authenticated')
_SCOPE='Declared outer entry through cold program worker return/reap and owned log hashing. Includes worker imports, internal publication/return and awaited children. Observer final publication/return remains separate operator overhead; nested closure and complete procedure cost require the actual study audit.'


def _path(value):
    if type(value) is not str or not value:raise ValueError('Canonical absolute program path required')
    p=Path(value)
    if not p.is_absolute() or p.resolve()!=p or p.is_symlink():raise ValueError('Canonical absolute program path required')
    return p


def _reference(value):
    if type(value) is not dict or set(value)!={'path','sha256'} or not _digest(value.get('sha256')):
        raise ValueError('Exact program file reference required')
    return _path(value['path'])


def _file_reference(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _validate_request(value):
    fields={'schema','source_sha256','tranche','arguments','clock_domain','invocation_origin_ns'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-owned-program-request'
            or not _digest(value['source_sha256']) or value['tranche'] not in ('primary','secondary','confirmation')
            or type(value['invocation_origin_ns']) is not int or value['invocation_origin_ns']<=0):
        raise ValueError('Strict owned program request required')
    validate_clock_domain(value['clock_domain'])
    if value['tranche']=='confirmation':
        args=value['arguments'];refs={'programme_reference','protocol_reference','models_reference','global_barrier_reference','preT_costs_reference','readiness_reference'}
        if type(args) is not dict or set(args)!=refs|{'qualification_reference','directory','max_step_starts'}:
            raise ValueError('Exact fixed confirmation arguments required; no role paths or method selection')
        for key in refs:_reference(args[key])
        q=args['qualification_reference']
        if type(q) is not dict or set(q)!={'root','inventory_sha256'} or not _digest(q['inventory_sha256']):raise ValueError('Exact confirmation qualification inventory required')
        _path(q['root']);_path(args['directory'])
        if args['max_step_starts'] is not None and (type(args['max_step_starts']) is not int or not 1<=args['max_step_starts']<=7):
            raise ValueError('Only an operational prefix of the seven fixed steps is permitted')
        return value
    common={'qualification_reference','resource_forecast_reference','directory','rss_limit_bytes','expected_journal_sha256','max_episode_starts'}
    primary={'practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference'}
    args=value['arguments'];secondary=value['tranche']=='secondary'
    if type(args) is not dict or set(args)!=common|({'barrier_reference'} if secondary else primary):
        raise ValueError('Exact public program arguments required; no label paths')
    refs=['resource_forecast_reference',*(['barrier_reference'] if secondary else ['practical_plan_reference','support_reference','procedure_reference'])]
    for key in refs:_reference(args[key])
    q=args['qualification_reference']
    if type(q) is not dict or set(q)!={'root','inventory_sha256'} or not _digest(q.get('inventory_sha256')):
        raise ValueError('Exact actual qualification reference required')
    _path(q['root']);_path(args['directory'])
    if (type(args['rss_limit_bytes']) is not int or not 0<args['rss_limit_bytes']<=16*1024**3
            or type(args['max_episode_starts']) is not int or not 1<=args['max_episode_starts']<=(672 if secondary else 168)
            or args['expected_journal_sha256'] is not None and not _digest(args['expected_journal_sha256'])
            or not secondary and not _digest(args['expected_practical_plan_hash'])):
        raise ValueError('Typed program RSS/start/resume bounds required')
    return value


def _make_request(tranche,arguments,origin):
    return _validate_request(dict(schema='research/v2-owned-program-request',source_sha256=_source(),
        tranche=tranche,arguments=copy.deepcopy(arguments),clock_domain=observe_clock_domain(),invocation_origin_ns=origin))


def program_worker_identity(path,expected):
    _,_,value=_read_reference(dict(path=str(path),sha256=expected));_validate_request(value)
    return dict(purpose='owned_practical_program',request_sha256=expected,source_sha256=value['source_sha256'],tranche=value['tranche'])


def _command(path,digest):
    from .primary_search_worker_v2 import _episode_environment
    return dict(argv=[sys.executable,'-P','-B','-m','treeneuron.research.program_process_v2',
        '--request',str(path),'--expected-request-sha256',digest],cwd=str(_ROOT),environment_overrides=_episode_environment())


def _failed_program_invocation_references(error,request):
    """A published driver clock is metadata, never a qualification capability."""
    if request['tranche']=='confirmation':
        reference=getattr(error,'confirmation_invocation_reference',None)
        if reference is None:return []
        refs=_confirmation_invocation_references(request)
        if reference not in refs:raise ValueError('Failed confirmation marker is foreign to its exact programme')
        return [copy.deepcopy(reference)]
    reference=getattr(error,'program_invocation_reference',None)
    if reference is None:return []  # Before an actual queue: no program/plan read.
    path=_reference(reference);directory=_path(request['arguments']['directory'])
    if path.parent!=directory/'invocations' or request['source_sha256']!=_source():
        raise ValueError('Failed invocation marker must belong to this exact current program')
    from .execution_clock_v2 import inspect_program_invocation_clock
    inspect_program_invocation_clock(reference,program_reference=_file_reference(directory/'plan.json'))
    return [copy.deepcopy(reference)]


def _confirmation_invocation_references(request):
    """Source/byte-bound metadata only; not a qualification shortcut."""
    from .confirm_v2 import _checked_programme,_call_history,_invocation_history
    args=request['arguments'];reference=args['programme_reference'];_,_,plan=_read_reference(reference)
    if (request['source_sha256']!=_source() or plan.get('directory')!=args['directory']
            or plan.get('protocol_reference')!=args['protocol_reference']
            or plan.get('inputs')!={k:args[k] for k in ('models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference')}):
        raise ValueError('Exact current confirmation programme and frozen input bindings required')
    _checked_programme(reference,plan)
    return [r['reference'] for r in _invocation_history(reference,plan,_call_history(reference,plan))]


def _operational_location(output,request,*,working_roots=(),proof_roots=()):
    from .primary_program_v2 import _check_program_location
    args=request['arguments'];inputs=[]
    for key,value in args.items():
        if key.endswith('_reference') and key!='qualification_reference':inputs.append(str(_reference(value)))
    return _check_program_location(output,working_roots=[*working_roots,args['directory']],
        proof_roots=[*proof_roots,args['qualification_reference']['root']],input_files=inputs)


@contextmanager
def _confirmation_observer_slot(plan,output):
    """Serial immutable metadata slots only; genuine admission is external."""
    from .ledger import StudyLease
    base=_path(plan['observer_directory']);output=_path(str(output))
    if output.parent!=base:raise ValueError('Confirmation observer must use its bound complete archive')
    base.mkdir(parents=True,exist_ok=True)
    with StudyLease(base):
        paths=sorted(p for p in base.iterdir() if p.name!='worker-lease.json')
        if ([p.name for p in paths]!=[f'episode-{i:04}' for i in range(len(paths))]
                or any(p.is_symlink() or not p.is_dir() for p in paths)
                or output!=base/f'episode-{len(paths):04}'):
            raise ValueError('Only the next contiguous original observer slot may execute')
        yield output


def _run_program_envelope(*,request,output,admission_identity,admission_check,root_rss_pause_bytes=4*1024**3):
    """Owned composition only; private fixtures cannot grant real qualification."""
    from .cooperative_process_v2 import supervise_cooperative_process,_write
    _validate_request(request)
    if request['source_sha256']!=_source() or request['invocation_origin_ns']>time.perf_counter_ns():
        raise ValueError('Current source and preceding program origin required')
    root=_operational_location(output,request)
    if (root.exists() or root.is_symlink() or type(root_rss_pause_bytes) is not int
            or not 0<root_rss_pause_bytes<=16*1024**3):raise ValueError('Fresh program observer root and typed RSS required')
    root.mkdir(parents=True,exist_ok=False);path=root/'request.json';_write(path,request)
    reference=_file_reference(path);command=_command(path,reference['sha256']);identity=program_worker_identity(path,reference['sha256'])
    result=supervise_cooperative_process(**dict(argv=command['argv'],cwd=command['cwd'],env_overrides=command['environment_overrides']),
        output=root/'process',identity=identity,admission_identity=admission_identity,admission_check=admission_check,
        rss_pause_bytes=root_rss_pause_bytes,invocation_origin_ns=request['invocation_origin_ns'],admit_before_spawn=True)
    process_reference=dict(process_root=str(root/'process'),process_receipt_sha256=_file_reference(root/'process/process.json')['sha256'],
        process_started_sha256=_file_reference(root/'process/started.json')['sha256'])
    envelope=dict(schema='research/v2-owned-program-envelope',source_sha256=request['source_sha256'],request_reference=reference,
        process_reference=process_reference,**command,clock_domain=request['clock_domain'],origin_ns=request['invocation_origin_ns'],
        end_ns=request['invocation_origin_ns']+result['elapsed_ns'],elapsed_ns=result['elapsed_ns'],scope=_SCOPE,
        body_reference=_file_reference(root/'process/body-outcome.json') if (root/'process/body-outcome.json').is_file() else None,
        **{key:(None if key=='T_accessed' and request['tranche']=='confirmation' and result['child_started'] else False) for key in _FLAGS})
    _write(root/'envelope.json',envelope)
    return dict(result,envelope_reference=_file_reference(root/'envelope.json'))


def run_program_process(*,tranche,arguments,output,admission_identity,admission_check,root_rss_pause_bytes=4*1024**3):
    """Qualify real paths first, then observe an owned uncapped cold program."""
    origin=time.perf_counter_ns();request=_make_request(tranche,arguments,origin)
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    registry=load_native_qualification_registry(arguments['qualification_reference'])
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=arguments['qualification_reference'])
    working=roots['working_roots'];proofs=roots['proof_roots']
    if tranche=='secondary':
        from .practical_global_freeze_v2 import load_global_primary_barrier
        from .secondary_episode_v2 import _proof_roots
        barrier=load_global_primary_barrier(arguments['barrier_reference'],qualification_reference=arguments['qualification_reference'])
        working,proofs=_proof_roots(barrier.payload,registry,arguments['qualification_reference'],arguments['barrier_reference'])
    elif tranche=='confirmation':
        from .confirm_v2 import _actual_protocol,_checked_programme
        inputs={k:arguments[k] for k in ('models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference')}
        protocol=_actual_protocol(arguments['protocol_reference'],inputs)
        _,_,plan=_read_reference(arguments['programme_reference']);_checked_programme(arguments['programme_reference'],plan)
        if plan['inputs']!=inputs or plan['protocol_reference']!=protocol.reference or plan['directory']!=arguments['directory']:
            raise ValueError('Actual confirmation observer programme binding changed')
        from .protected_journal_v2 import _public_locations
        from .protected_prediction_gate_v2 import _publication_location
        study,owners=_public_locations(plan['protected_study_directory'],protocol)
        _publication_location(output,protocol=protocol,inspection=dict(study_directory=str(study),owner_root=str(owners),invocation_references=[]),kind='directory')
    _operational_location(output,request,working_roots=working,proof_roots=proofs)
    if tranche=='confirmation':
        with _confirmation_observer_slot(plan,output):
            return _run_program_envelope(request=request,output=output,admission_identity=admission_identity,
                admission_check=admission_check,root_rss_pause_bytes=root_rss_pause_bytes)
    return _run_program_envelope(request=request,output=output,admission_identity=admission_identity,
        admission_check=admission_check,root_rss_pause_bytes=root_rss_pause_bytes)


def inspect_program_process(reference):
    """Read actual closed root/handshake evidence, without native/cost authority."""
    from .cooperative_process_v2 import inspect_cooperative_process
    path,raw,value=_read_reference(reference);root=path.parent
    fields={'schema','source_sha256','request_reference','process_reference','argv','cwd','environment_overrides',
        'clock_domain','origin_ns','end_ns','elapsed_ns','scope','body_reference',*_FLAGS}
    if (path.name!='envelope.json' or type(value) is not dict or set(value)!=fields
            or value['schema']!='research/v2-owned-program-envelope' or value['source_sha256']!=_source()
            or value['scope']!=_SCOPE or any(value[k] is not False for k in _FLAGS if k!='T_accessed')):
        raise ValueError('Program envelope source/schema/authority differs')
    request_path,request_raw,request=_read_reference(value['request_reference']);_validate_request(request)
    expected_T=None if request['tranche']=='confirmation' else False
    if value['T_accessed'] is not expected_T and not (request['tranche']=='confirmation' and value['T_accessed'] is False):
        raise ValueError('Opaque confirmation access is unknown, never certified by a clock')
    if (request_path!=root/'request.json' or request['source_sha256']!=value['source_sha256']
            or request['clock_domain']!=value['clock_domain'] or request['invocation_origin_ns']!=value['origin_ns']):
        raise ValueError('Program envelope changed its request/clock identity')
    validate_clock_domain(value['clock_domain'])
    process=value['process_reference']
    if type(process) is not dict or set(process)!={'process_root','process_receipt_sha256','process_started_sha256'} or process['process_root']!=str(root/'process'):
        raise ValueError('Exact local owned program process required')
    proof=inspect_cooperative_process(root=root/'process',expected_receipt_sha256=process['process_receipt_sha256'],
        expected_started_sha256=process['process_started_sha256'])
    start=_json((root/'process/started.json').read_bytes());command=_command(request_path,value['request_reference']['sha256'])
    if value['T_accessed'] is not (None if request['tranche']=='confirmation' and proof['child_started'] else False):
        raise ValueError('Confirmation access metadata must match actual worker-start uncertainty')
    if (any(value[k]!=command[k] or start.get(k)!=command[k] for k in command)
            or start.get('identity')!=program_worker_identity(request_path,value['request_reference']['sha256'])
            or any(type(value[k]) is not int for k in ('origin_ns','end_ns','elapsed_ns'))
            or value['origin_ns']!=start.get('supervisor_started_monotonic_ns')
            or value['elapsed_ns']!=proof['elapsed_ns'] or value['end_ns']-value['origin_ns']!=value['elapsed_ns']):
        raise ValueError('Program command or observed cold interval differs')
    shared=False;body=value['body_reference']
    if body is not None:
        body_path,_,outcome=_read_reference(body)
        body_fields={'schema','status','request_sha256','source_sha256','clock_domain','origin_ns','end_ns',
            'program_progress','program_invocation_references','error_type','reason','traceback',*_FLAGS}
        if (body_path!=root/'process/body-outcome.json' or type(outcome) is not dict or set(outcome)!=body_fields
                or outcome['schema']!='research/v2-owned-program-body' or outcome['status'] not in ('program_failed','program_returned')
                or outcome['request_sha256']!=value['request_reference']['sha256'] or outcome['source_sha256']!=value['source_sha256']
                or any(outcome[k] is not False for k in _FLAGS if k!='T_accessed') or outcome['T_accessed'] is not expected_T
                or type(outcome['program_invocation_references']) is not list
                or any(type(outcome[k]) is not int for k in ('origin_ns','end_ns')) or outcome['end_ns']<outcome['origin_ns']):
            raise ValueError('Owned program body identity/schema differs')
        validate_clock_domain(outcome['clock_domain']);shared=same_clock_domain(value['clock_domain'],outcome['clock_domain'])
        if shared and not value['origin_ns']<=outcome['origin_ns']<=outcome['end_ns']<=value['end_ns']:
            raise ValueError('Owned program body lies outside its containing clock')
        if (outcome['status']=='program_returned' and (proof['returncode']!=0 or not proof['worker_protocol_closed']
                or type(outcome['program_progress']) is not dict or not outcome['program_invocation_references'])
                or outcome['status']=='program_failed' and outcome['program_progress'] is not None):
            raise ValueError('Owned program body contradicts its closed disposition')
        if request['tranche']=='confirmation' and outcome['program_invocation_references']:
            actual=_confirmation_invocation_references(request);seen=set()
            for ref in outcome['program_invocation_references']:
                if ref not in actual or ref['path'] in seen:raise ValueError('Confirmation body substituted or duplicated its programme invocation')
                seen.add(ref['path'])
    elif (root/'process/body-outcome.json').exists():raise ValueError('Envelope omitted a retained owned program body')
    if path.read_bytes()!=raw or request_path.read_bytes()!=request_raw or _source()!=value['source_sha256']:
        raise ValueError('Program envelope inputs changed while reading')
    return dict(origin_ns=value['origin_ns'],end_ns=value['end_ns'],elapsed_ns=value['elapsed_ns'],clock_domain=value['clock_domain'],
        owned_program_root_clock_checked=True,shared_observed_worker_clock_domain=shared,
        worker_protocol_closed=proof['worker_protocol_closed'],child_started=proof['child_started'],returncode=proof['returncode'],
        scope=_SCOPE,**{key:value[key] for key in _FLAGS})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',required=True)
    parser.add_argument('--expected-request-sha256',required=True);args=parser.parse_args(argv)
    identity=program_worker_identity(args.request,args.expected_request_sha256)
    from .cooperative_process_v2 import cooperative_worker,_write
    with cooperative_worker(identity=identity):
        _,_,request=_read_reference(dict(path=args.request,sha256=args.expected_request_sha256));_validate_request(request)
        origin=time.perf_counter_ns()
        outcome=dict(schema='research/v2-owned-program-body',status='program_failed',request_sha256=args.expected_request_sha256,
            source_sha256=request['source_sha256'],clock_domain=observe_clock_domain(),origin_ns=origin,end_ns=None,
            program_progress=None,program_invocation_references=[],error_type=None,reason=None,traceback=None,
            **{k:(None if k=='T_accessed' and request['tranche']=='confirmation' else False) for k in _FLAGS})
        code=1
        try:
            from .primary_search_worker_v2 import _episode_environment
            if (request['source_sha256']!=_source() or not sys.dont_write_bytecode
                    or any(os.environ.get(k)!=v for k,v in _episode_environment().items())):
                raise ValueError('Exact owned program source/core-only environment required')
            if request['tranche']=='primary':
                from .primary_program_v2 import run_primary_program
                progress=run_primary_program(**request['arguments'])
            elif request['tranche']=='confirmation':
                from .confirm_v2 import run_practical_confirmation
                confirmation_args=request['arguments']
                progress=run_practical_confirmation(confirmation_args['programme_reference'],
                    **{k:v for k,v in confirmation_args.items() if k not in ('programme_reference','directory')})
            else:
                from .secondary_program_v2 import run_secondary_program
                progress=run_secondary_program(**request['arguments'])
            if _read_reference(dict(path=args.request,sha256=args.expected_request_sha256))[2]!=request or _source()!=request['source_sha256']:
                raise ValueError('Program source/request changed during execution')
            directory=Path(request['arguments']['directory'])
            references=(_confirmation_invocation_references(request) if request['tranche']=='confirmation' else
                [_file_reference(p) for p in sorted((directory/'invocations').glob('*.json'))])
            if not references:raise ValueError('Returned program lacks an invocation receipt')
            outcome.update(program_progress=progress,program_invocation_references=references,status='program_returned');code=0
        except BaseException as error:
            outcome.update(error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
            try:outcome['program_invocation_references']=_failed_program_invocation_references(error,request)
            except (ValueError,OSError,KeyError,TypeError) as evidence_error:
                outcome['reason']+=f'; failed invocation evidence unavailable: {type(evidence_error).__name__}: {evidence_error}'
        finally:
            outcome['end_ns']=time.perf_counter_ns();_write(Path(os.environ['ATN_SEARCH_CONTROL_ROOT'])/'body-outcome.json',outcome)
    return code


if __name__=='__main__':raise SystemExit(main())
