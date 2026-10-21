"""Strict metadata controls for the existing practical public operations.

This module selects no method, reads no role arrays and grants no qualification.
The called public operations own their full scientific checks and publication.
"""
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
import sys

from .deployment_worker_v2 import _ROOT, _json, _source


# Names are code, never caller-selected import paths.  None means that the
# public operation returns a value for immutable JSON publication here.
_OPERATIONS={
    'support_lock':('practical_support_lock_v2','seal_practical_support_lock','output'),
    'practical_plan':(None,'_make_practical_plan',None),
    'procedure':('practical_procedure_v2','seal_practical_procedure','output'),
    'managed_preparation':('preparation_execution_v2','run_managed_preparation','output'),
    'qualification_registry':('native_qualification_registry_v2','seal_native_qualification_registry','output'),
    'resource_forecast':('native_resource_forecast_v2','forecast_native_practical_resources',None),
    'primary_inventory':('primary_selection_v2','seal_primary_selection_inventory','output'),
    'primary_models':('qualified_primary_models_v2','seal_qualified_primary_models','output'),
    'global_barrier':('practical_global_freeze_v2','seal_global_primary_barrier','output'),
    'practical_models':('qualified_practical_models_v2','seal_qualified_practical_models','output'),
    'preT_costs':('practical_preT_costs_v2','seal_practical_preT_costs','output'),
    'T_protocol':('practical_T_protocol_v2','seal_practical_T_protocol','output'),
    'primary_program':('primary_program_v2','run_primary_program',None),
    'secondary_program':('secondary_program_v2','run_secondary_program',None),
}
_PROGRAMS={'primary_program','secondary_program'}
_FREEZES={'support_lock','procedure','qualification_registry','primary_inventory',
    'primary_models','global_barrier','practical_models','preT_costs','T_protocol'}
_FROZEN_INPUTS=('models_reference','global_barrier_reference','qualification_reference','preT_costs_reference','readiness_reference')


def _read_control(path):
    original=Path(path)
    if original.is_symlink():raise ValueError('Practical command must be an ordinary JSON file')
    path=original.resolve()
    with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
    if len(raw)>16*1024**2:raise ValueError('Practical command exceeds metadata limit')
    value=_json(raw)
    if (type(value) is not dict or set(value)!={'schema','source_sha256','operation','arguments'}
            or value['schema']!='research/v2-practical-command' or value['source_sha256']!=_source()
            or type(value['operation']) is not str or value['operation'] not in _OPERATIONS
            or type(value['arguments']) is not dict or 'output' in value['arguments']):
        raise ValueError('Exact current-source practical command and known operation required')
    return value,dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest())


def _references(value):
    """Only locate declared metadata inputs; never decode a role archive."""
    if type(value) is dict:
        if set(value)=={'path','sha256'} and type(value['path']) is str:yield Path(value['path'])
        elif set(value)=={'root','inventory_sha256'} and type(value['root']) is str:yield Path(value['root'])
        else:
            for item in value.values():yield from _references(item)
    elif type(value) is list:
        for item in value:yield from _references(item)


def _output_path(output,control,reference):
    original=Path(output)
    if original.is_symlink():raise ValueError('Ordinary practical command output required')
    output=original.resolve()
    from treeneuron.protocol import _execution
    if output.is_relative_to(_ROOT) and (output==_ROOT or _execution(output.relative_to(_ROOT).as_posix())):
        raise ValueError('Practical output must preserve executable scientific inputs')
    protected=[Path(reference['path']),Path(sys.prefix).resolve(),
        *((_ROOT/p).resolve() for p in ('environment','.git','.agents','.codex','.aws')),
        *_references(control['arguments'])]
    if any(output.is_relative_to(p.resolve()) or p.resolve().is_relative_to(output) for p in protected):
        raise ValueError('Practical output must preserve all declared inputs')
    return output


def _target(operation):
    module,name,_=_OPERATIONS[operation]
    return globals()[name] if module is None else getattr(importlib.import_module('.'+module,__package__),name)


def _arguments(control,*,materialize=False):
    args=copy.deepcopy(control['arguments']);target=_target(control['operation']);params=inspect.signature(target).parameters
    # JSON binds immutable plan bytes rather than trying to serialize a live
    # Python dataclass.  Only these two public parameter names are adapted.
    for name in ('practical_plan','plan'):
        if name in params:
            key=name+'_reference'
            if name in args or key not in args:raise ValueError('An exact practical plan reference is required')
            reference=args.pop(key)
            if materialize:
                from .execution_clock_v2 import _read_reference
                from .workload import PracticalPlan
                from treeneuron.config import canonical_json
                args[name]=PracticalPlan(canonical_json(_read_reference(reference)[2]))
            else:args[name]=reference
    output_key=_OPERATIONS[control['operation']][2]
    checked=dict(args)
    if output_key:checked[output_key]=Path('/unused-metadata-output')
    try:inspect.signature(target).bind(**checked)
    except TypeError as error:raise ValueError(f'Practical {control["operation"]} arguments: {error}') from error
    return args


def plan_practical_command(path,*,output):
    control,reference=_read_control(path);output=_output_path(output,control,reference)
    _arguments(control)
    value=dict(schema='research/v2-practical-command-plan',source_sha256=control['source_sha256'],
        config_reference=reference,operation=control['operation'],arguments=control['arguments'],
        execution_authorized=False,protected_access_authorized=False,T_accessed=False,
        scope='Metadata command validation only; actual public native/data/source/resource gates remain required')
    from .ledger import write_new_json
    if _read_control(path)!=(control,reference):raise ValueError('Practical command changed during planning')
    write_new_json(output,value)
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def _make_practical_plan(*,data_manifest_references,support_reference,lock_sha256,study_id='P2-search'):
    from .execution_clock_v2 import _read_reference
    from .practical_support_lock_v2 import load_practical_support_lock
    from .workload import expand_practical
    law=load_practical_support_lock(support_reference)
    if type(data_manifest_references) is not dict:raise ValueError('Six exact dataset manifest references required')
    manifests={name:_read_reference(ref)[2] for name,ref in data_manifest_references.items()}
    return expand_practical(manifests,support_lock_sha256=support_reference['sha256'],
        source_sha256=law['source_sha256'],runtime_sha256=law['core_manifest_sha256'],
        lock_sha256=lock_sha256,study_id=study_id,native_quality=True).to_dict()


def _run_owned_program(operation,args,output):
    from .program_process_v2 import run_program_process,inspect_program_process
    from .execution_clock_v2 import _read_reference
    from .native_deployment_execution_v2 import _live_admission
    target=_target(operation);bound=inspect.signature(target).bind(**args);bound.apply_defaults()
    args=dict(bound.arguments);rss=args['rss_limit_bytes']
    identity=dict(purpose='practical_cli_owned_program',source_sha256=_source(),
        root_rss_pause_bytes=rss,required_memory_proxy_bytes=(5*rss+3)//4+512*1024**2,
        required_disk_proxy_bytes=512*1024**2,worker_concurrency=1,training_wall_time_limit_ns=None)
    result=run_program_process(tranche='primary' if operation=='primary_program' else 'secondary',
        arguments=args,output=output,admission_identity=identity,
        admission_check=lambda:_live_admission(identity,output),root_rss_pause_bytes=rss)
    inspect_program_process(result['envelope_reference'])
    envelope=_read_reference(result['envelope_reference'])[2];body=envelope['body_reference']
    progress=None if body is None else _read_reference(body)[2]['program_progress']
    if result['execution_status'] in ('resource_exhausted','interrupted_recoverable'):status='paused'
    elif result['execution_status']!='exited' or result['returncode']!=0:status='blocked'
    else:status='completed' if progress['status'] in ('primary_searches_terminal','secondary_searches_terminal') else 'paused'
    return dict(result,status=status)


def practical_command_exit_code(result):
    if type(result) is dict and result.get('status')=='preparation_failed':return 4
    if type(result) is dict and result.get('status') in ('completed','paused','blocked'):
        return {'completed':0,'paused':3,'blocked':4}[result['status']]
    return 0


def execute_practical_command(path,*,output,command='run',resume=False):
    control,reference=_read_control(path);output=_output_path(output,control,reference)
    operation=control['operation'];_arguments(control)
    if command not in ('run','freeze') or command=='freeze' and operation not in _FREEZES:
        raise ValueError('This practical operation is not a freeze publication')
    if resume and operation not in _PROGRAMS:
        raise ValueError('Only immutable programme queues support operational resume here')
    # Before materializing plan references, repeat the genuine gate.  Neither a
    # control file nor its planning receipt can turn a Boolean into authority.
    if 'qualification_reference' in control['arguments']:
        from .native_qualification_registry_v2 import load_native_qualification_registry
        load_native_qualification_registry(control['arguments']['qualification_reference'])
    args=_arguments(control,materialize=True)
    if _read_control(path)!=(control,reference):raise ValueError('Practical command changed before execution')
    if operation in _PROGRAMS:return _run_owned_program(operation,args,output)
    output_key=_OPERATIONS[operation][2]
    if output_key:return _target(operation)(**args,**{output_key:output})
    result=_target(operation)(**args)
    from .ledger import write_new_json
    if _read_control(path)!=(control,reference):raise ValueError('Practical command changed before publication')
    write_new_json(output,result)
    return dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest())


def _file_reference(path):
    path=Path(path)
    if path.is_symlink() or path.resolve()!=path or not path.is_file():raise ValueError('Ordinary absolute metadata file required')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _protocol_bootstrap(path):
    """Read only the JSON commitments needed to invoke the actual loader."""
    from .execution_clock_v2 import _read_reference
    from treeneuron.config import canonical_hash
    reference=_file_reference(Path(path).absolute());value=_read_reference(reference)[2]
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'}
            or value['schema']!='research/v2-final-practical-prediction-protocol' or type(value['payload']) is not dict
            or canonical_hash(value['payload'])!=value['payload_hash']
            or any(k not in value['payload'] for k in _FROZEN_INPUTS)):
        raise ValueError('Exact final practical protocol metadata required before actual loading')
    return reference,{k:copy.deepcopy(value['payload'][k]) for k in _FROZEN_INPUTS}


def _publication_arguments(protocol_reference,inputs,programme_reference,plan,costs_reference):
    return dict(schema='research/v2-practical-publication',protocol_reference=protocol_reference,
        programme_reference=programme_reference,analysis_reference=_file_reference(Path(plan['outputs']['analysis'])),
        raw_audit_reference=_file_reference(Path(plan['outputs']['raw_audit'])),
        prediction_stage_reference=_file_reference(Path(plan['outputs']['prediction_stage'])),costs_reference=costs_reference,
        directory=plan['protected_study_directory'],**inputs)


def _programme_complete(reference,plan):
    """Read-only metadata; complete clocks do not authenticate their science."""
    from .confirm_v2 import _checked_programme,_call_history,_invocation_history
    _checked_programme(reference,plan)
    history=_invocation_history(reference,plan,_call_history(reference,plan))
    return any(row['clock']['status']=='completed' and row['clock']['completed_steps']==plan['steps'] for row in history)


def confirm_practical_protocol(path,*,output):
    """Run/resume the fixed actual programme through its owned cold observer."""
    reference,inputs=_protocol_bootstrap(path)
    from .practical_T_protocol_v2 import load_practical_T_protocol
    load_practical_T_protocol(reference,**inputs)  # Genuine gates before output.
    root=_output_path(output,dict(arguments=dict(protocol_reference=reference,**inputs)),reference)
    from .confirm_v2 import seal_practical_confirmation_programme,_checked_programme
    from .execution_clock_v2 import _read_reference
    programme=seal_practical_confirmation_programme(reference,directory=root/'protected-study',output=root/'programme',**inputs)['reference']
    plan=_read_reference(programme)[2];_checked_programme(programme,plan)
    cost_path=root/'pipeline-costs.json';publication=root/'publication.json'
    from .practical_pipeline_costs_v2 import load_practical_pipeline_costs,seal_practical_pipeline_costs
    def cost_arguments():
        return dict(programme_reference=programme,analysis_reference=_file_reference(Path(plan['outputs']['analysis'])),
            raw_audit_reference=_file_reference(Path(plan['outputs']['raw_audit'])),
            prediction_stage_reference=_file_reference(Path(plan['outputs']['prediction_stage'])),
            directory=plan['protected_study_directory'],**inputs)
    if cost_path.exists():
        costs=load_practical_pipeline_costs(_file_reference(cost_path),protocol_reference=reference,**cost_arguments())
    else:
        if not _programme_complete(programme,plan):
            from .program_process_v2 import run_program_process,inspect_program_process
            from .native_deployment_execution_v2 import _live_admission
            archive=Path(plan['observer_directory'])
            slots=[] if not archive.exists() else [p for p in archive.iterdir() if p.name!='worker-lease.json']
            episode=archive/f'episode-{len(slots):04}'
            identity=dict(purpose='practical_cli_fixed_confirmation',source_sha256=_source(),
                **{k:plan[k] for k in ('rss_limit_bytes','required_memory_proxy_bytes','required_disk_proxy_bytes','worker_concurrency')})
            observed=run_program_process(tranche='confirmation',arguments=dict(programme_reference=programme,
                protocol_reference=reference,directory=plan['directory'],max_step_starts=None,**inputs),
                output=episode,admission_identity=identity,admission_check=lambda:_live_admission(identity,episode),
                root_rss_pause_bytes=plan['rss_limit_bytes'])
            inspect_program_process(observed['envelope_reference'])
            if observed['execution_status']!='exited' or observed['returncode']!=0:
                return dict(status='paused' if observed['execution_status'] in ('resource_exhausted','interrupted_recoverable') else 'blocked',
                    programme_reference=programme,envelope_reference=observed['envelope_reference'],protected_access_authorized=False)
            if not _programme_complete(programme,plan):
                return dict(status='paused',programme_reference=programme,envelope_reference=observed['envelope_reference'],protected_access_authorized=False)
        costs=seal_practical_pipeline_costs(reference,output=cost_path,**cost_arguments())
    value=_publication_arguments(reference,inputs,programme,plan,costs.reference)
    from .ledger import write_new_json
    if publication.exists():
        if _read_reference(_file_reference(publication))[2]!=value:raise ValueError('Existing publication references differ; never replace a study')
    else:write_new_json(publication,value)
    return dict(status='completed',publication_reference=_file_reference(publication),costs_reference=costs.reference)


def report_practical_study(directory,*,output):
    from .execution_clock_v2 import _read_reference
    path=Path(directory).absolute()/'publication.json';value=_read_reference(_file_reference(path))[2]
    fields={'schema','protocol_reference','programme_reference','analysis_reference','raw_audit_reference',
        'prediction_stage_reference','costs_reference','directory',*_FROZEN_INPUTS}
    if type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-practical-publication':
        raise ValueError('Exact completed practical publication references required')
    from .practical_report_v2 import write_protected_practical_report
    return write_protected_practical_report(value['protocol_reference'],output=Path(output).absolute(),
        **{k:v for k,v in value.items() if k not in ('schema','protocol_reference')})
