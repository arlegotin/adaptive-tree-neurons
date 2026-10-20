"""Containing program observations of complete method calls, without authority."""
import copy
import hashlib
from pathlib import Path
import time

from .deployment_worker_v2 import _source
from .execution_clock_v2 import validate_clock_domain,same_clock_domain,_read_reference


_FLAGS=('native_qualification_authenticated','whole_training_procedure_cost_authenticated','protected_access_authorized','T_accessed')
_SCOPE='Containing program clock around execute_episode argument preparation through actual return or raised exception. Own clock publication and queue bookkeeping are common program overhead, physically covered by the outer observer.'


def _program(program):
    from treeneuron.config import canonical_hash
    from .primary_search_worker_v2 import _path
    if (type(program) is not dict or program.get('schema') not in ('research/v2-primary-program-plan','research/v2-secondary-program-plan')
            or program.get('source_sha256')!=_source()
            or canonical_hash({k:v for k,v in program.items() if k!='program_hash'})!=program.get('program_hash')):
        raise ValueError('Exact current-source program metadata required')
    return _path(program['directory'])


def _location(program,job_id,index,episode_root):
    root=_program(program)
    if (type(job_id) is not str or job_id not in {job['job_id'] for job in program['jobs']}
            or type(index) is not int or index<0):raise ValueError('Exact program job and episode required')
    expected=root/'episodes'/job_id/f'episode-{index:04}'
    if str(episode_root)!=str(expected):raise ValueError('Full call must belong to its exact queue episode')
    return root/'call-clocks'/job_id/f'episode-{index:04}.json'


def observe_program_episode(*,program,job,index,episode_root,previous_episodes,previous_unstarted_episodes,
        execute_episode,clock_domain):
    """Private composition; injected fixture callbacks grant no native gate."""
    from .cooperative_process_v2 import _write
    path=_location(program,job['job_id'],index,episode_root);validate_clock_domain(clock_domain)
    if path.exists() or path.is_symlink():raise ValueError('Full call observation cannot be overwritten')
    domain=copy.deepcopy(clock_domain);origin=time.perf_counter_ns();error=None
    try:
        return execute_episode(program=program,job=copy.deepcopy(job),episode_root=episode_root,
            previous_episodes=copy.deepcopy(previous_episodes),previous_unstarted_episodes=copy.deepcopy(previous_unstarted_episodes))
    except BaseException as caught:
        error=dict(type=type(caught).__name__,reason=str(caught));raise
    finally:
        end=time.perf_counter_ns();caller=Path(episode_root)/'caller-complete.json'
        reference=dict(path=str(caller),sha256=hashlib.sha256(caller.read_bytes()).hexdigest()) if caller.is_file() else None
        value=dict(schema='research/v2-program-call-clock-1',source_sha256=program['source_sha256'],program_hash=program['program_hash'],
            job_id=job['job_id'],episode_index=index,episode_root=str(episode_root),clock_domain=domain,origin_ns=origin,end_ns=end,
            elapsed_ns=end-origin,disposition='returned' if error is None else 'raised',error=error,caller_reference=reference,
            scope=_SCOPE,**{key:False for key in _FLAGS})
        path.parent.mkdir(parents=True,exist_ok=True);_write(path,value)


def inspect_program_call_clock(reference,*,program):
    """Reopen source/file/caller observations; no real program/model gate."""
    from .primary_caller_v2 import inspect_caller_boundary
    path,raw,value=_read_reference(reference)
    fields={'schema','source_sha256','program_hash','job_id','episode_index','episode_root','clock_domain','origin_ns','end_ns',
        'elapsed_ns','disposition','error','caller_reference','scope',*_FLAGS}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-program-call-clock-1'
            or value['source_sha256']!=program.get('source_sha256') or value['program_hash']!=program.get('program_hash')
            or value['scope']!=_SCOPE or any(value[k] is not False for k in _FLAGS)):
        raise ValueError('Full call source/schema/authority differs')
    if _location(program,value['job_id'],value['episode_index'],value['episode_root'])!=path:
        raise ValueError('Exact local full call reference required')
    validate_clock_domain(value['clock_domain'])
    if (any(type(value[k]) is not int for k in ('origin_ns','end_ns','elapsed_ns')) or value['origin_ns']<=0
            or value['end_ns']<value['origin_ns'] or value['elapsed_ns']!=value['end_ns']-value['origin_ns']):
        raise ValueError('Strict complete call interval required')
    error=value['error']
    if value['disposition']=='returned':
        if error is not None:raise ValueError('Returned call cannot carry an exception')
    elif value['disposition']=='raised':
        if (type(error) is not dict or set(error)!={'type','reason'} or type(error['type']) is not str
                or not error['type'] or type(error['reason']) is not str):raise ValueError('Raised call needs its actual exception record')
    else:raise ValueError('Unknown full call disposition')
    checked=contained=False;caller_elapsed=None
    if value['caller_reference'] is not None:
        caller_path,_,caller=_read_reference(value['caller_reference'])
        if caller_path!=Path(value['episode_root'])/'caller-complete.json':raise ValueError('Full call substituted a foreign caller')
        proof=inspect_caller_boundary(value['caller_reference'],process_reference=caller['process_reference'])
        caller_elapsed=proof['elapsed_ns'];checked=True
        if value['elapsed_ns']<caller_elapsed:raise ValueError('Containing call is shorter than its observed caller')
        contained=same_clock_domain(value['clock_domain'],proof['clock_domain'])
        if contained and not value['origin_ns']<=proof['origin_ns']<=proof['end_ns']<=value['end_ns']:
            raise ValueError('Caller lies outside its observed containing call')
    if path.read_bytes()!=raw or _source()!=program['source_sha256']:raise ValueError('Full call inputs changed while reading')
    return dict(origin_ns=value['origin_ns'],end_ns=value['end_ns'],elapsed_ns=value['elapsed_ns'],clock_domain=value['clock_domain'],
        job_id=value['job_id'],episode_index=value['episode_index'],disposition=value['disposition'],
        error_type=None if error is None else error['type'],caller_boundary_checked=checked,
        caller_interval_containment_observed=contained,caller_elapsed_ns=caller_elapsed,scope=_SCOPE,**{key:False for key in _FLAGS})
