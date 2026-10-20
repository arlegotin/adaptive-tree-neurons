"""Read closed managed outcomes, including legitimate operational failures."""
import copy
import hashlib
from pathlib import Path

from .deployment_worker_v2 import _json
from .native_anchor_v2 import _closed_reference
from .native_workload_v2 import _bound_reader
from .workqueue import _inventory


_COMMAND=('argv','cwd','environment_overrides','identity','started_utc_ns','timeout_ns',
    'rss_limit_bytes','poll_interval_ns','terminate_grace_ns','log_limit_bytes')
_AUTHORITY=('strict_peak_memory_limit_qualified','eligible_model','protected_access_authorized',
    'native_recipe_runtime_qualified','whole_process_cost_authenticated')
_REASONS={
    'exited':{'process_exit'},
    'resource_exhausted':{'observed_root_rss_limit','log_bytes_limit'},
    'timeout':{'dedicated_worker_deadline','dedicated_worker_deadline_during_rss_probe',
        'dedicated_worker_deadline_during_rss_preflight','deadline_before_spawn'},
    'input_or_dependency_failure':{'rss_probe_unavailable','process_start_failure','unexpected_live_process_group'},
    'interrupted_recoverable':{'supervisor_interrupted'}}


def _positive(value):return type(value) is int and value>0


def inspect_closed_process(reference,*,expected_argv,expected_cwd,expected_environment_overrides,expected_identity):
    """Check saved ownership, files, limits and counters without probing a PID.

    Nonzero exit and resource/timeout stops remain operational outcomes. This
    does not authenticate a model, native recipe or external absolute clock.
    """
    reference=copy.deepcopy(reference)
    expected=copy.deepcopy(dict(argv=expected_argv,cwd=expected_cwd,
        environment_overrides=expected_environment_overrides,identity=expected_identity))
    root,inventory=_closed_reference(reference);read,refs=_bound_reader(root,inventory)
    process=read('process.json');start=read('started.json')
    fields={'schema',*_COMMAND,'execution_status','stop_reason','diagnostic','child_started','child_pid',
        'returncode','child_reaped','termination','observed_peak_root_rss_bytes','rss_sample_count',
        'rss_scope',*_AUTHORITY,'files','elapsed_ns','clock_scope','deadline_overshoot_ns'}
    if (type(process) is not dict or set(process)!=fields or process['schema']!='research/v2-process-receipt'
            or type(start) is not dict or set(start)!={'schema',*_COMMAND} or start['schema']!='research/v2-process-start'
            or any(start[k]!=process[k] for k in _COMMAND)
            or any(process[k]!=v for k,v in expected.items())
            or any(process[k] is not False for k in _AUTHORITY)):
        raise ValueError('Closed process schema/invocation/authority differs')
    if (not _positive(process['elapsed_ns']) or not _positive(process['started_utc_ns'])
            or any(not _positive(process[k]) for k in ('rss_limit_bytes','poll_interval_ns','log_limit_bytes'))
            or type(process['terminate_grace_ns']) is not int or process['terminate_grace_ns']<0
            or type(process['rss_sample_count']) is not int or process['rss_sample_count']<0):
        raise ValueError('Typed process clocks/limits/counters required')
    timeout=process['timeout_ns'];status=process['execution_status'];reason=process['stop_reason']
    if (timeout is not None and not _positive(timeout) or type(status) is not str or type(reason) is not str
            or status not in _REASONS or reason not in _REASONS[status]
            or status=='timeout' and timeout is None
            or process['deadline_overshoot_ns']!=(None if timeout is None else max(0,process['elapsed_ns']-timeout))):
        raise ValueError('Closed process deadline/status/stop reason differs')
    termination=process['termination']
    if (type(termination) is not dict or set(termination)!={'term_sent','kill_sent','group_absent_observed'}
            or any(type(v) is not bool for v in termination.values())
            or termination['kill_sent'] and not termination['term_sent']
            or type(process['child_started']) is not bool or type(process['child_reaped']) is not bool):
        raise ValueError('Typed process closure/termination required')
    names={'started.json','stdout.log','stderr.log','resource-samples.jsonl'}
    if process['child_started']:
        names.add('spawned.json');spawn=read('spawned.json')
        if (process['child_reaped'] is not True or not _positive(process['child_pid']) or type(process['returncode']) is not int
                or spawn.get('schema')!='research/v2-owned-process' or spawn.get('new_session') is not True
                or spawn.get('child_pid')!=process['child_pid'] or spawn.get('child_pgid')!=process['child_pid']
                or not _positive(spawn.get('observed_utc_ns')) or spawn['observed_utc_ns']<process['started_utc_ns']):
            raise ValueError('Managed process child was not closed/reaped or ownership differs')
        if status=='exited' and any(termination.values()):raise ValueError('Normal process exit contradicts termination flags')
    elif (process['child_pid'] is not None or process['returncode'] is not None or process['child_reaped'] is not False
            or any(termination.values()) or status in ('exited','resource_exhausted')
            or reason=='unexpected_live_process_group'):
        raise ValueError('Pre-spawn failure cannot claim an owned/exited child')
    files=process['files']
    if (type(files) is not list or len(files)!=len(names)
            or any(type(r) is not dict or set(r)!={'path','sha256','bytes'} or type(r['path']) is not str for r in files)
            or {r['path'] for r in files}!=names
            or set(refs)!=names|{'process.json'}):raise ValueError('Exact closed process file inventory required')
    for row in files:
        if (type(row) is not dict or set(row)!={'path','sha256','bytes'} or row['sha256']!=refs[row['path']]
                or type(row['bytes']) is not int or row['bytes']!=(root/row['path']).stat().st_size):
            raise ValueError('Closed raw process file commitment differs')
    count=0;peak=None;last=0;digest=hashlib.sha256()
    with (root/'resource-samples.jsonl').open('rb') as stream:
        for raw in stream:
            digest.update(raw);row=_json(raw)
            if (not raw.endswith(b'\n') or type(row) is not dict or set(row)!={'elapsed_ns','root_rss_bytes'}
                    or type(row['elapsed_ns']) is not int or not last<row['elapsed_ns']<=process['elapsed_ns']
                    or type(row['root_rss_bytes']) is not int or row['root_rss_bytes']<0):
                raise ValueError('Invalid raw process RSS observation')
            last=row['elapsed_ns'];count+=1;peak=row['root_rss_bytes'] if peak is None else max(peak,row['root_rss_bytes'])
    if (digest.hexdigest()!=refs['resource-samples.jsonl'] or count!=process['rss_sample_count']
            or type(process['observed_peak_root_rss_bytes']) is not type(peak) or process['observed_peak_root_rss_bytes']!=peak
            or not process['child_started'] and count):raise ValueError('Raw RSS peak/count differs from receipt')
    over=peak is not None and peak>process['rss_limit_bytes']
    if (over!=(status=='resource_exhausted' and reason=='observed_root_rss_limit')
            or reason=='log_bytes_limit' and sum((root/p).stat().st_size for p in ('stdout.log','stderr.log'))<=process['log_limit_bytes']):
        raise ValueError('Process resource outcome contradicts actual observations')
    if _inventory(root)!=inventory:raise ValueError('Process evidence changed during inspection')
    return dict(schema='research/v2-checked-closed-process',reference=reference,process_closure_authenticated=True,
        execution_status=status,stop_reason=reason,child_started=process['child_started'],child_reaped=process['child_reaped'],
        returncode=process['returncode'],elapsed_ns=process['elapsed_ns'],observed_peak_root_rss_bytes=peak,rss_sample_count=count,
        rss_peak_available=peak is not None,**{k:False for k in _AUTHORITY},
        scope='Trusted saved managed invocation/ownership/closed files/raw root-RSS and operational counters; no live PID probe, aggregate RAM, independent absolute clock, method recipe/model quality or protected authority')
