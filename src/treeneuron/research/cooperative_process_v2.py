"""Cold-process observation with an explicit no-spawn-before-grant contract.

Only trusted workers using ``cooperative_worker`` may run here. This is an
ownership protocol, not a sandbox for arbitrary programs. After grant the root
receives at most one SIGINT and must join its own nested supervisors. There is
no automatic outer hard kill, aggregate-memory bound or model qualification.

The worker handshake deliberately imports only the Python standard library.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time


_ROOT = 'ATN_SEARCH_CONTROL_ROOT'
_START = 'ATN_SEARCH_CONTROL_START_SHA256'
_CLOCK_SCOPE = 'Supervisor entry through root reap, log closure and file hashing; final receipt publication and caller overhead excluded'
_FALSE = ('eligible_model', 'native_recipe_runtime_qualified', 'protected_access_authorized',
          'strict_peak_memory_limit_qualified', 'aggregate_descendant_memory_measured',
          'whole_training_procedure_cost_authenticated')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _positive(value):
    return type(value) is int and value > 0


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('Regular cooperative evidence file required')
    content = path.read_bytes()
    value = json.loads(content)
    _canonical(value)
    return value, content


def _write(path, value):
    """Publish complete immutable JSON before another process can observe it."""
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.pending')
    with temporary.open('xb') as stream:
        stream.write((_canonical(value) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)  # Exclusive publication; never replace old evidence.
    temporary.unlink()


def _ref(path):
    return dict(path=path.name, bytes=path.stat().st_size, sha256=_sha(path.read_bytes()))


def _ready_expected(start, start_sha, pid, parent_pid, worker_ready_ns=None):
    value = dict(schema='research/v2-cooperative-ready', child_pid=pid, child_pgid=pid,
        parent_pid=parent_pid, started_sha256=start_sha, nonce=start['nonce'],
        identity_sha256=_sha(_canonical(start['identity']).encode()),
        argv_sha256=_sha(_canonical(start['argv']).encode()),
        no_nested_spawn_before_grant=True)
    if worker_ready_ns is not None:
        value['worker_ready_monotonic_ns'] = worker_ready_ns
    return value


def _validate_ready(ready,start,start_sha,pid,parent_pid,spawn_requested_ns,observed_elapsed_ns):
    origin = start.get('supervisor_started_monotonic_ns')
    worker = ready.get('worker_ready_monotonic_ns') if origin is not None else None
    if origin is not None and (not _positive(origin) or not _positive(spawn_requested_ns)
            or not _positive(observed_elapsed_ns) or not _positive(worker)
            or not origin+spawn_requested_ns <= worker <= origin+observed_elapsed_ns):
        raise ValueError('Worker and parent clock domains do not match observed ready bounds')
    if _canonical(ready) != _canonical(_ready_expected(start,start_sha,pid,parent_pid,worker)):
        raise ValueError('Cooperative ready identity differs')
    return origin is not None


def _grant_expected(ready_sha, nonce):
    return dict(schema='research/v2-cooperative-grant', ready_sha256=ready_sha, nonce=nonce)


@contextmanager
def cooperative_worker(*, identity):
    """Join before method imports or spawning; leave only after nested cleanup.

The body owns every nested worker it starts and must synchronously close it on
KeyboardInterrupt. Existing per-fit supervisors do this. A failed/unclosed body
is retained as an operational failure, never a resumable search by assumption.
"""
    root = Path(os.environ[_ROOT])
    start, raw = _read(root/'started.json')
    start_sha = _sha(raw)
    if (start_sha != os.environ[_START] or _canonical(identity) != _canonical(start['identity'])
            or _canonical(list(sys.orig_argv)) != _canonical(start['argv'])
            or os.getpgid(0) != os.getpid() or os.getppid() != start['supervisor_pid']):
        raise ValueError('Cooperative worker ownership/identity mismatch')
    ready = _ready_expected(start, start_sha, os.getpid(), os.getppid(),
        time.perf_counter_ns() if 'supervisor_started_monotonic_ns' in start else None)
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    entered = False
    exit_kind = 'returned'
    try:
        _write(root/'ready.json', ready)
        ready_sha = _sha((root/'ready.json').read_bytes())
        while not (root/'grant.json').exists():
            time.sleep(.01)
        grant, _ = _read(root/'grant.json')
        if _canonical(grant) != _canonical(_grant_expected(ready_sha, start['nonce'])):
            raise ValueError('Cooperative grant commitment mismatch')
        entered = True
        yield
    except BaseException as error:
        exit_kind = type(error).__name__
        raise
    finally:
        _write(root/'worker-closure.json', dict(schema='research/v2-cooperative-worker-closure',
            started_sha256=start_sha, nonce=start['nonce'], child_pid=os.getpid(),
            body_entered=entered, exit_kind=exit_kind,
            nested_cleanup_scope='Trusted worker must synchronously join its own nested supervisors'))
        signal.signal(signal.SIGINT, previous)


def supervise_cooperative_process(*, argv, cwd, env_overrides, output, identity,
        admission_identity, admission_check, rss_pause_bytes, poll_interval_ns=100_000_000,
        handshake_timeout_ns=10_000_000_000, pause_after_ns=None,
        log_limit_bytes=64*1024**2):
    """Retain one owned cold-worker episode, without automatic restart.

``pause_after_ns`` is an optional cooperative operational limit; None supplies
no wall cap. Root RSS is sampled, and excess requests cleanup rather than
enforcing a hard outer memory bound. Admission runs after ready, before grant.
"""
    started = time.perf_counter_ns()
    from .process_supervision_v2 import root_rss_bytes
    output, cwd = Path(output).absolute(), Path(cwd).resolve()
    if (not isinstance(argv, list) or not argv or any(type(v) is not str or not v or '\0' in v for v in argv)
            or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file() or not cwd.is_dir()
            or not isinstance(env_overrides, dict)
            or any(type(k) is not str or not k or '=' in k or '\0' in k
                or k.startswith('ATN_SEARCH_CONTROL_') or k in ('HOME','home','CODEX_HOME')
                or v is not None and (type(v) is not str or '\0' in v)
                for k,v in env_overrides.items())):
        raise ValueError('Invalid cooperative command/environment contract')
    if (not isinstance(identity, dict) or not identity or not isinstance(admission_identity, dict)
            or not admission_identity or not callable(admission_check)
            or any(not _positive(v) for v in (rss_pause_bytes,poll_interval_ns,handshake_timeout_ns,log_limit_bytes))
            or pause_after_ns is not None and not _positive(pause_after_ns)):
        raise ValueError('Invalid cooperative ownership/admission contract')
    identity_json = _canonical(identity)
    if output.exists() or output.is_symlink():
        raise ValueError('Fresh cooperative output required')
    output.mkdir(parents=True)
    start = dict(schema='research/v2-cooperative-start', argv=argv, cwd=str(cwd),
        environment_overrides=env_overrides, identity=identity, admission_identity=admission_identity,
        supervisor_pid=os.getpid(), nonce=secrets.token_hex(32), started_utc_ns=time.time_ns(),
        supervisor_started_monotonic_ns=started,
        rss_pause_bytes=rss_pause_bytes, poll_interval_ns=poll_interval_ns,
        handshake_timeout_ns=handshake_timeout_ns, pause_after_ns=pause_after_ns,
        log_limit_bytes=log_limit_bytes)
    _write(output/'started.json', start)
    start_sha = _sha((output/'started.json').read_bytes())
    environment = dict(os.environ)
    for key,value in env_overrides.items():
        if value is None:
            environment.pop(key, None)
        else:
            environment[key] = value
    environment.update({_ROOT:str(output), _START:start_sha})
    proc = None
    status, reason, diagnostic = 'input_or_dependency_failure', 'before_spawn', None
    granted = authorized = stop_attempted = False
    grant_elapsed = grant_started_elapsed = spawn_elapsed = spawn_requested = ready_observed = None
    stop_publication_error = None
    signals, samples = [], []

    def stop_before_grant():
        # Popen owns this unreaped root. Trusted protocol forbids descendants.
        if proc.poll() is None:
            proc.terminate()
            signals.append(dict(signal='SIGTERM', scope='owned_ungranted_root'))
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()
                signals.append(dict(signal='SIGKILL', scope='owned_ungranted_root'))
        proc.wait()

    def request_cleanup():
        nonlocal stop_attempted, stop_publication_error, status, reason, diagnostic
        if stop_attempted:
            return
        stop_attempted = True
        try:
            _write(output/'request-stop.json', dict(schema='research/v2-cooperative-stop',
                reason=reason, elapsed_ns=time.perf_counter_ns()-started, nonce=start['nonce']))
        except Exception as error:
            # A full disk cannot prevent cleanup. An absent/partial receipt stays
            # absent/partial; never retry by overwriting a producer's evidence.
            stop_publication_error = f'{type(error).__name__}: {error}'
            status, reason = 'input_or_dependency_failure', 'stop_receipt_publication_failed'
            diagnostic = stop_publication_error
        finally:
            if proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGINT)  # Root only; nested supervisors own their groups.
                except ProcessLookupError:
                    if proc.poll() is None:
                        raise
                else:
                    signals.append(dict(signal='SIGINT', scope='owned_authorized_root'))

    with (output/'stdout.log').open('xb') as stdout, (output/'stderr.log').open('xb') as stderr, \
            (output/'resource-samples.jsonl').open('x') as observations:
        try:
            root_rss_bytes(os.getpid(), 250_000_000)
            spawn_requested = time.perf_counter_ns()-started
            proc = subprocess.Popen(argv,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,
                stdout=stdout,stderr=stderr,start_new_session=True)
            spawn_elapsed = time.perf_counter_ns()-started
            _write(output/'spawned.json', dict(schema='research/v2-cooperative-owned-root',
                child_pid=proc.pid, child_pgid=proc.pid, new_session=True))
            while True:
                elapsed = time.perf_counter_ns()-started
                if proc.poll() is not None:
                    if status == 'input_or_dependency_failure' and reason == 'before_spawn':
                        status, reason = 'exited', 'root_exit'
                    break
                if not granted and (output/'ready.json').exists():
                    ready, raw = _read(output/'ready.json')
                    ready_observed = time.perf_counter_ns()-started
                    _validate_ready(ready,start,start_sha,proc.pid,os.getpid(),spawn_requested,ready_observed)
                    admission = admission_check()
                    if not isinstance(admission,dict) or type(admission.get('admitted')) is not bool:
                        raise ValueError('Admission must return an explicit boolean and retained evidence')
                    _write(output/'admission.json', admission)
                    if not admission['admitted']:
                        status, reason = 'resource_exhausted', 'pregrant_admission_denied'
                        break
                    # Publication can succeed just before an interrupt/error.
                    # From this point onward descendants MAY exist, even if the
                    # publisher never gets to set its success flag.
                    authorized = True
                    grant_started_elapsed = time.perf_counter_ns()-started
                    _write(output/'grant.json', _grant_expected(_sha(raw), start['nonce']))
                    granted = True
                    grant_elapsed = time.perf_counter_ns()-started
                try:
                    rss = root_rss_bytes(proc.pid,250_000_000)
                except ProcessLookupError:
                    if proc.poll() is not None:
                        continue
                    raise
                sample = dict(elapsed_ns=time.perf_counter_ns()-started,root_rss_bytes=rss)
                samples.append(sample)
                observations.write(_canonical(sample)+'\n')
                observations.flush()
                os.fsync(observations.fileno())
                if not granted and elapsed >= handshake_timeout_ns:
                    reason = 'handshake_timeout'
                    break
                if granted and not stop_attempted:
                    if rss > rss_pause_bytes:
                        status, reason = 'resource_exhausted', 'observed_root_rss_pause'
                        request_cleanup()
                    elif pause_after_ns is not None and elapsed >= pause_after_ns:
                        status, reason = 'interrupted_recoverable', 'declared_cooperative_pause'
                        request_cleanup()
                    elif (output/'stdout.log').stat().st_size+(output/'stderr.log').stat().st_size > log_limit_bytes:
                        status, reason = 'resource_exhausted', 'observed_log_bytes_pause'
                        request_cleanup()
                time.sleep(poll_interval_ns/1e9)
        except BaseException as error:
            diagnostic = f'{type(error).__name__}: {error}'
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                status, reason = 'interrupted_recoverable', 'supervisor_interrupted'
            else:
                status, reason = 'input_or_dependency_failure', 'supervision_failure'
        finally:
            if proc is not None:
                if authorized:
                    if proc.poll() is None:
                        request_cleanup()
                    # Never hard kill an outer coordinator with nested sessions.
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=.1)
                        except (subprocess.TimeoutExpired,KeyboardInterrupt):
                            pass
                    proc.wait()
                else:
                    stop_before_grant()
            for stream in (stdout,stderr):
                stream.flush()
                os.fsync(stream.fileno())
    closed = False
    if (output/'worker-closure.json').exists():
        closure, _ = _read(output/'worker-closure.json')
        closed = (proc is not None and closure.get('started_sha256') == start_sha
            and closure.get('nonce') == start['nonce'] and type(closure.get('child_pid')) is int
            and closure['child_pid'] == proc.pid)
    if status == 'exited' and (not granted or not closed):
        status, reason = 'input_or_dependency_failure', 'worker_protocol_not_closed'
    granted = (output/'grant.json').exists()
    if granted and grant_elapsed is None:
        grant_elapsed = time.perf_counter_ns()-started
    files = [_ref(path) for path in sorted(output.iterdir()) if path.is_file()]
    receipt = dict(schema='research/v2-cooperative-process-receipt', started_sha256=start_sha,
        execution_status=status,stop_reason=reason,diagnostic=diagnostic,child_started=proc is not None,
        child_pid=None if proc is None else proc.pid,returncode=None if proc is None else proc.returncode,
        child_reaped=proc is not None and proc.returncode is not None,grant_issued=granted,
        grant_publication_started=authorized,grant_publication_started_elapsed_ns=grant_started_elapsed,
        spawn_elapsed_ns=spawn_elapsed,grant_elapsed_ns=grant_elapsed,signals=signals,
        spawn_requested_elapsed_ns=spawn_requested,ready_observed_elapsed_ns=ready_observed,
        stop_publication_error=stop_publication_error,
        cooperative_stop_requested=(output/'request-stop.json').exists(),worker_protocol_closed=closed,
        rss_sample_count=len(samples),observed_peak_root_rss_bytes=max((s['root_rss_bytes'] for s in samples),default=None),
        files=files,elapsed_ns=time.perf_counter_ns()-started,clock_scope=_CLOCK_SCOPE,
        **{key:False for key in _FALSE})
    if _canonical(identity) != identity_json:
        raise ValueError('Cooperative process identity mutated')
    _write(output/'process.json',receipt)
    return receipt


def inspect_cooperative_process(*, root, expected_receipt_sha256, expected_started_sha256):
    """Reopen closed raw observations without spawning or probing a saved PID."""
    root = Path(root)
    receipt, receipt_raw = _read(root/'process.json')
    start, start_raw = _read(root/'started.json')
    if (_sha(receipt_raw) != expected_receipt_sha256 or _sha(start_raw) != expected_started_sha256
            or receipt.get('started_sha256') != expected_started_sha256):
        raise ValueError('Cooperative process external commitment changed')
    if (receipt.get('schema') != 'research/v2-cooperative-process-receipt'
            or start.get('schema') != 'research/v2-cooperative-start'
            or receipt.get('clock_scope') != _CLOCK_SCOPE
            or any(receipt.get(k) is not False for k in _FALSE)
            or not _positive(receipt.get('elapsed_ns'))
            or receipt.get('execution_status') not in ('exited','input_or_dependency_failure','resource_exhausted','interrupted_recoverable')):
        raise ValueError('Invalid cooperative process receipt')
    files = receipt.get('files')
    if not isinstance(files,list) or any(not isinstance(r,dict) for r in files):
        raise ValueError('Cooperative file commitments required')
    names = [r.get('path') for r in files]
    if (any(type(n) is not str or Path(n).name != n for n in names)
            or len(names) != len(set(names))
            or set(names) != {p.name for p in root.iterdir()}-{'process.json'}
            or not {'started.json','stdout.log','stderr.log','resource-samples.jsonl'} <= set(names)):
        raise ValueError('Exact cooperative file inventory required')
    for ref in files:
        path = root/ref['path']
        if path.is_symlink() or not path.is_file() or _canonical(_ref(path)) != _canonical(ref):
            raise ValueError('Cooperative raw file commitment changed')
    samples = [json.loads(line) for line in (root/'resource-samples.jsonl').read_text().splitlines()]
    last = 0
    for sample in samples:
        if (set(sample) != {'elapsed_ns','root_rss_bytes'} or not _positive(sample['elapsed_ns'])
                or sample['elapsed_ns'] <= last or sample['elapsed_ns'] > receipt['elapsed_ns']
                or type(sample['root_rss_bytes']) is not int or sample['root_rss_bytes'] < 0):
            raise ValueError('Invalid raw cooperative resource sample')
        last = sample['elapsed_ns']
    if (type(receipt.get('rss_sample_count')) is not int or receipt['rss_sample_count'] != len(samples)
            or receipt['observed_peak_root_rss_bytes'] != max((s['root_rss_bytes'] for s in samples),default=None)):
        raise ValueError('Cooperative resource summaries disagree')
    spawned = 'spawned.json' in names
    if receipt.get('child_started') is not spawned or receipt.get('child_reaped') is not spawned:
        raise ValueError('Closed owned root required')
    if spawned:
        pid = receipt.get('child_pid')
        owned, _ = _read(root/'spawned.json')
        if (not _positive(pid) or type(receipt.get('returncode')) is not int
                or not _positive(receipt.get('spawn_elapsed_ns'))
                or receipt['spawn_elapsed_ns'] >= receipt['elapsed_ns']
                or _canonical(owned) != _canonical(dict(schema='research/v2-cooperative-owned-root',child_pid=pid,child_pgid=pid,new_session=True))):
            raise ValueError('Invalid owned cooperative root')
        if 'supervisor_started_monotonic_ns' in start and (not _positive(start['supervisor_started_monotonic_ns'])
                or not _positive(receipt.get('spawn_requested_elapsed_ns'))
                or not receipt['spawn_requested_elapsed_ns']<=receipt['spawn_elapsed_ns']):
            raise ValueError('Invalid cold-root monotonic origin/spawn bounds')
    elif any(receipt.get(k) is not None for k in ('child_pid','returncode','spawn_elapsed_ns')):
        raise ValueError('Unstarted root cannot have process counters')
    granted = 'grant.json' in names
    if receipt.get('grant_issued') is not granted:
        raise ValueError('Grant summary disagrees')
    authorized = receipt.get('grant_publication_started')
    if type(authorized) is not bool or granted and not authorized:
        raise ValueError('Grant publication phase required')
    shared_clock = False
    if authorized:
        ready, ready_raw = _read(root/'ready.json')
        admission, _ = _read(root/'admission.json')
        if (not spawned or admission.get('admitted') is not True
                or not _positive(receipt.get('grant_publication_started_elapsed_ns'))
                or not receipt['spawn_elapsed_ns'] < receipt['grant_publication_started_elapsed_ns'] < receipt['elapsed_ns']):
            raise ValueError('Cooperative grant/ready/admission disagree')
        shared_clock = _validate_ready(ready,start,expected_started_sha256,pid,start['supervisor_pid'],
            receipt.get('spawn_requested_elapsed_ns'),receipt.get('ready_observed_elapsed_ns'))
    elif receipt.get('grant_publication_started_elapsed_ns') is not None:
        raise ValueError('Unauthorized root cannot have grant publication clock')
    if granted:
        grant, _ = _read(root/'grant.json')
        if (_canonical(grant) != _canonical(_grant_expected(_sha(ready_raw),start['nonce']))
                or not _positive(receipt.get('grant_elapsed_ns'))
                or not receipt['grant_publication_started_elapsed_ns'] <= receipt['grant_elapsed_ns'] < receipt['elapsed_ns']):
            raise ValueError('Grant byte/clock commitment disagrees')
    elif receipt.get('grant_elapsed_ns') is not None:
        raise ValueError('Ungrant cannot have a grant clock')
    closure_present = 'worker-closure.json' in names
    if receipt.get('worker_protocol_closed') is not closure_present:
        raise ValueError('Worker closure summary disagrees')
    if closure_present:
        closure, _ = _read(root/'worker-closure.json')
        if (not spawned or closure.get('schema') != 'research/v2-cooperative-worker-closure'
                or closure.get('started_sha256') != expected_started_sha256 or closure.get('nonce') != start['nonce']
                or type(closure.get('child_pid')) is not int or closure['child_pid'] != pid
                or type(closure.get('body_entered')) is not bool or type(closure.get('exit_kind')) is not str):
            raise ValueError('Worker closure commitment disagrees')
    signals = receipt.get('signals')
    if authorized:
        if signals not in ([],[dict(signal='SIGINT',scope='owned_authorized_root')]):
            raise ValueError('Granted worker may only receive one cooperative interrupt')
    elif signals not in ([],[dict(signal='SIGTERM',scope='owned_ungranted_root')],
            [dict(signal='SIGTERM',scope='owned_ungranted_root'),dict(signal='SIGKILL',scope='owned_ungranted_root')]):
        raise ValueError('Invalid ungranted owned-root signal')
    stop_present = 'request-stop.json' in names
    if receipt.get('cooperative_stop_requested') is not stop_present:
        raise ValueError('Cooperative stop commitment disagrees')
    if stop_present:
        stop, _ = _read(root/'request-stop.json')
        if (not authorized or stop.get('nonce') != start['nonce'] or not _positive(stop.get('elapsed_ns'))
                or stop['elapsed_ns'] > receipt['elapsed_ns']):
            raise ValueError('Invalid cooperative stop request')
    stop_error = receipt.get('stop_publication_error')
    if stop_error is not None and (type(stop_error) is not str or not stop_error
            or receipt['execution_status'] != 'input_or_dependency_failure'):
        raise ValueError('Invalid stop-publication failure record')
    if signals == [dict(signal='SIGINT',scope='owned_authorized_root')] and not stop_present and stop_error is None:
        raise ValueError('Cooperative interrupt needs a stop request or retained publication failure')
    if receipt['execution_status'] == 'exited' and (not granted or not closure_present):
        raise ValueError('Exited protocol requires closed granted worker')
    return dict(receipt,root_process_clock_authenticated=True,
        shared_monotonic_clock_observed=shared_clock,
        nested_cleanup_independently_authenticated=False,
        scope='Closed root clock and handshake only; caller must authenticate native nested closure, search history, model and source')
