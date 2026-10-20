"""Observed monotonic epochs and closed records, without scientific authority.

Boot/host hashes identify an observed clock domain, not secure hardware identity.
Unknown or different domains cannot support absolute interval comparisons.
Closed elapsed durations remain readable after the observing machine reboots.
"""
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import time
import uuid

from .deployment_worker_v2 import _digest, _json, _source


PROGRAM_CLOCK_SCOPE='Actual invocation entry through preparation/queue/readback; own final receipt publication/return excluded'
_DOMAIN_FIELDS={'schema','clock','system','host_sha256','boot_sha256','status','reason','clock_implementation'}
_FLAGS=('whole_training_procedure_cost_authenticated','protected_access_authorized','T_accessed')


def _parse_boot_identity(system,raw):
    if type(raw) is not str:raise ValueError('Text kernel boot identity required')
    if system=='Linux':
        value=raw.strip()
        try:parsed=str(uuid.UUID(value))
        except (ValueError,AttributeError):raise ValueError('Invalid Linux boot identity') from None
        if value.lower()!=parsed:raise ValueError('Canonical Linux boot identity required')
        return 'linux-boot-id:'+parsed
    if system=='Darwin':
        match=re.fullmatch(r'\s*\{\s*sec\s*=\s*([0-9]+),\s*usec\s*=\s*([0-9]+)\s*\}[^\r\n]*\s*',raw)
        if not match:raise ValueError('Invalid Darwin boot identity')
        seconds,micros=map(int,match.groups())
        if seconds<=0 or micros>=1_000_000:raise ValueError('Invalid Darwin boot time')
        return f'darwin-boottime:{seconds}:{micros}'
    raise ValueError('No qualified kernel boot identity for this system')


def observe_clock_domain():
    """Read a bounded kernel probe; failures remain explicit unknown metadata."""
    system=platform.system();node=platform.node()
    host=hashlib.sha256(json.dumps([node,system,platform.machine()],ensure_ascii=True,separators=(',',':')).encode('ascii')).hexdigest()
    value=dict(schema='research/v2-monotonic-clock-domain',clock='perf_counter_ns',system=system,
        host_sha256=host,boot_sha256=None,status='unknown',reason='kernel_boot_identity_unavailable',
        clock_implementation=time.get_clock_info('perf_counter').implementation)
    try:
        if not node:raise OSError('Host identity unavailable')
        if system=='Linux':raw=Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii')
        elif system=='Darwin':
            raw=subprocess.run(['/usr/sbin/sysctl','-n','kern.boottime'],check=True,capture_output=True,
                text=True,timeout=2).stdout
        else:raise OSError('No supported kernel boot identity')
        identity=_parse_boot_identity(system,raw)
    except (OSError,ValueError,subprocess.SubprocessError):pass
    else:value.update(boot_sha256=hashlib.sha256(identity.encode('ascii')).hexdigest(),status='observed',reason=None)
    validate_clock_domain(value)
    return value


def validate_clock_domain(value):
    """Check stored metadata, deliberately without probing the current boot."""
    if (type(value) is not dict or set(value)!=_DOMAIN_FIELDS
            or value.get('schema')!='research/v2-monotonic-clock-domain' or value.get('clock')!='perf_counter_ns'
            or type(value.get('system')) is not str or not value['system']
            or not _digest(value.get('host_sha256')) or type(value.get('clock_implementation')) is not str
            or not value['clock_implementation']):raise ValueError('Strict observed clock domain required')
    if value.get('status')=='observed':
        if value['system'] not in ('Linux','Darwin') or not _digest(value['boot_sha256']) or value['reason'] is not None:
            raise ValueError('Observed clock domain needs a supported boot identity')
    elif value.get('status')=='unknown':
        if value['boot_sha256'] is not None or value['reason']!='kernel_boot_identity_unavailable':
            raise ValueError('Unknown clock domain must retain the unavailable identity')
    else:raise ValueError('Unknown clock domain status')


def same_clock_domain(first,second):
    """Compare metadata only; a match does not authenticate its provenance."""
    validate_clock_domain(first);validate_clock_domain(second)
    return first['status']=='observed' and second['status']=='observed' and first==second


def _read_reference(reference):
    if type(reference) is not dict or set(reference)!={'path','sha256'} or not _digest(reference.get('sha256')):
        raise ValueError('Exact clock file reference required')
    name=reference['path']
    if type(name) is not str or not name:raise ValueError('Canonical clock file path required')
    path=Path(name)
    if not path.is_absolute() or path.resolve()!=path or path.is_symlink():raise ValueError('Canonical clock file path required')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=reference['sha256']:raise ValueError('Clock input bytes changed')
    return path,raw,_json(raw)


def inspect_program_invocation_clock(reference,*,program_reference):
    """Reopen one source/byte-bound clock; no data, process or native gate."""
    saved,program_raw,program=_read_reference(program_reference)
    path,raw,value=_read_reference(reference)
    if (type(program) is not dict or program.get('schema') not in ('research/v2-primary-program-plan','research/v2-secondary-program-plan')
            or saved.name!='plan.json' or program.get('directory')!=str(saved.parent)
            or program.get('source_sha256')!=_source() or not _digest(program.get('program_hash'))):
        raise ValueError('Exact current-source program metadata required')
    declared=json.dumps({k:v for k,v in program.items() if k!='program_hash'},sort_keys=True,
        separators=(',',':'),ensure_ascii=False,allow_nan=False).encode('utf-8')
    if hashlib.sha256(declared).hexdigest()!=program['program_hash']:raise ValueError('Program metadata identity changed')
    secondary=program['schema']=='research/v2-secondary-program-plan'
    schema='research/v2-secondary-program-invocation-clock-1' if secondary else 'research/v2-primary-program-invocation-clock-1'
    if path.parent!=saved.parent/'invocations' or not re.fullmatch(r'[0-9a-f]{32}\.json',path.name):
        raise ValueError('Exact local program invocation required')
    fields={'schema','source_sha256','program_hash','journal_sha256','origin_ns','end_ns','elapsed_ns',
        'clock_domain','result','error','clock_scope',*_FLAGS}
    if (type(value) is not dict or set(value)!=fields or value['schema']!=schema
            or value['source_sha256']!=program['source_sha256'] or value['program_hash']!=program['program_hash']
            or not _digest(value['journal_sha256']) or value['clock_scope']!=PROGRAM_CLOCK_SCOPE
            or any(value[k] is not False for k in _FLAGS)):
        raise ValueError('Program clock source/schema/authority differs')
    validate_clock_domain(value['clock_domain'])
    if (any(type(value[k]) is not int for k in ('origin_ns','end_ns','elapsed_ns'))
            or value['origin_ns']<=0 or value['end_ns']<value['origin_ns']
            or value['elapsed_ns']!=value['end_ns']-value['origin_ns']):raise ValueError('Strict closed program interval required')
    result=value['result'];error=value['error']
    if error is None:
        result_fields={'schema','status','program_hash','journal_sha256','completed_jobs','known_observed_caller_ns',
            'native_model_barrier_authenticated','secondary_release_authorized','scope',*_FLAGS}
        result_schema='research/v2-secondary-program-progress' if secondary else 'research/v2-primary-program-progress'
        if (type(result) is not dict or set(result)!=result_fields or result['schema']!=result_schema
                or result['program_hash']!=program['program_hash'] or result['journal_sha256']!=value['journal_sha256']
                or type(result['completed_jobs']) is not int or not 0<=result['completed_jobs']<=len(program['jobs'])
                or type(result['known_observed_caller_ns']) is not int or result['known_observed_caller_ns']<0
                or type(result['status']) is not str or not result['status']
                or type(result['scope']) is not str or not result['scope']
                or any(result[k] is not False for k in (*_FLAGS,'native_model_barrier_authenticated','secondary_release_authorized'))):
            raise ValueError('Program clock result contradicts its source-bound invocation')
    elif (result is not None or type(error) is not dict or set(error)!={'type','reason'}
            or type(error['type']) is not str or not error['type'] or type(error['reason']) is not str):
        raise ValueError('Strict failed program invocation required')
    journal=saved.parent/'events.jsonl';content=journal.read_bytes();prefix=hashlib.sha256()
    committed=prefix.hexdigest()==value['journal_sha256']
    for line in content.splitlines(keepends=True):
        if not line.endswith(b'\n'):raise ValueError('Program journal has an incomplete event')
        prefix.update(line);committed=committed or prefix.hexdigest()==value['journal_sha256']
    if not committed:raise ValueError('Program clock journal prefix is absent')
    if (saved.read_bytes()!=program_raw or path.read_bytes()!=raw or _source()!=program['source_sha256']):
        raise ValueError('Program clock inputs changed during reading')
    return dict(origin_ns=value['origin_ns'],end_ns=value['end_ns'],elapsed_ns=value['elapsed_ns'],
        clock_domain=value['clock_domain'],program_hash=program['program_hash'],journal_sha256=value['journal_sha256'],
        invocation_clock_record_checked=True,scope=PROGRAM_CLOCK_SCOPE,**{key:False for key in _FLAGS})
