"""Own-child process deadline/log/RSS observation, without model qualification.

Only a new process group created here is signaled. RSS is sampled for its root
worker; this is neither aggregate descendant memory nor a strict peak guarantee.
An exit code never grants predictive/checkpoint eligibility or protected access.
"""
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import time

from treeneuron.config import canonical_json
from .ledger import write_new_json


def _positive(value):return type(value) is int and value>0


def root_rss_bytes(pid,timeout_ns):
    if not _positive(pid) or not _positive(timeout_ns):raise ValueError('Invalid owned-PID probe')
    result=subprocess.run(['/bin/ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,
        timeout=timeout_ns/1e9,check=False)
    if result.returncode!=0:
        if result.returncode==1 and not result.stdout.strip():raise ProcessLookupError('Worker exited during RSS probe')
        raise RuntimeError(f'RSS probe exited {result.returncode}: {result.stderr.strip()}')
    parts=result.stdout.split()
    if len(parts)!=1 or not parts[0].isdigit():raise RuntimeError('RSS probe returned no unambiguous byte quantity')
    return int(parts[0])*1024


def _group_exists(pid):
    try:os.killpg(pid,0)
    except ProcessLookupError:return False
    return True


def _terminate(proc,grace_ns):
    record=dict(term_sent=False,kill_sent=False,group_absent_observed=False)
    def remaining_group():
        try:return _group_exists(proc.pid)
        except PermissionError:
            # Observed on macOS for a zombie-only group before wait/reap. Never
            # interpret EPERM as absence. TERM may still be in progress: a short
            # wait timeout means keep the grace/kill path active, not abandon it.
            try:proc.wait(timeout=.01)
            except subprocess.TimeoutExpired:return True
            return _group_exists(proc.pid)
    try:os.killpg(proc.pid,signal.SIGTERM);record['term_sent']=True
    except ProcessLookupError:pass
    end=time.perf_counter_ns()+grace_ns
    while time.perf_counter_ns()<end:
        proc.poll()
        if not remaining_group():break
        time.sleep(min(.01,max(0,(end-time.perf_counter_ns())/1e9)))
    if remaining_group():
        try:os.killpg(proc.pid,signal.SIGKILL);record['kill_sent']=True
        except ProcessLookupError:pass
    proc.wait(timeout=5)
    record['group_absent_observed']=not remaining_group()
    return record


def _file_ref(path,root):
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    return dict(path=str(path.relative_to(root)),sha256=digest,bytes=path.stat().st_size)


def supervise_process(*,argv,cwd,env_overrides,output,timeout_ns,rss_limit_bytes,
                      poll_interval_ns=100000000,terminate_grace_ns=250000000,
                      log_limit_bytes=64*1024**2,identity):
    """Retain one process attempt; its caller decides native terminal eligibility.

    ``argv`` is executed without a shell. An explicit None environment override
    removes that variable in the child only; the removal is retained in receipts.
    No retries or model selection occur.
    This does not recover a supervisor that itself is killed, nor authenticate
    the caller's native recipe/runtime/source identities.
    """
    started=time.perf_counter_ns();started_utc=time.time_ns()
    cwd=Path(cwd);output=Path(output)
    if (not isinstance(argv,list) or not argv or any(type(s) is not str or not s or '\0' in s for s in argv)
            or not Path(argv[0]).is_absolute() or not Path(argv[0]).is_file()
            or not cwd.is_dir() or not isinstance(env_overrides,dict)
            or any(type(k) is not str or not k or '=' in k or '\0' in k
                or v is not None and (type(v) is not str or '\0' in v)
                or k in ('HOME','home','CODEX_HOME') for k,v in env_overrides.items())
            or timeout_ns is not None and not _positive(timeout_ns)
            or any(not _positive(n) for n in (rss_limit_bytes,poll_interval_ns,log_limit_bytes))
            or type(terminate_grace_ns) is not int or terminate_grace_ns<0
            or not isinstance(identity,dict) or not identity):raise ValueError('Invalid supervised process execution contract')
    identity_json=canonical_json(identity)
    def expired(elapsed):return timeout_ns is not None and elapsed>=timeout_ns
    def probe_budget(elapsed=0):return 250000000 if timeout_ns is None else max(1,min(250000000,timeout_ns-elapsed))
    child_environment=dict(os.environ)
    for key,value in env_overrides.items():
        if value is None:child_environment.pop(key,None)
        else:child_environment[key]=value
    if output.is_symlink() or output.exists():raise ValueError('Fresh process output required; no silent retry/overwrite')
    output.mkdir(parents=True)
    command=dict(schema='research/v2-process-start',argv=argv,cwd=str(cwd.resolve()),
        environment_overrides=env_overrides,identity=identity,started_utc_ns=started_utc,
        timeout_ns=timeout_ns,rss_limit_bytes=rss_limit_bytes,poll_interval_ns=poll_interval_ns,
        terminate_grace_ns=terminate_grace_ns,log_limit_bytes=log_limit_bytes)
    write_new_json(output/'started.json',command)
    proc=None;status='input_or_dependency_failure';reason='rss_probe_unavailable';diagnostic=None
    peak=None;samples=0;termination=dict(term_sent=False,kill_sent=False,group_absent_observed=False)
    with (output/'stdout.log').open('xb') as stdout,(output/'stderr.log').open('xb') as stderr,(output/'resource-samples.jsonl').open('x') as observations:
        try:
            # Unknown measurement capability fails before starting an expensive fit.
            try:root_rss_bytes(os.getpid(),probe_budget())
            except subprocess.TimeoutExpired as error:
                diagnostic=f'{type(error).__name__}: {error}'
                if expired(time.perf_counter_ns()-started):
                    status='timeout';reason='dedicated_worker_deadline_during_rss_preflight'
            except Exception as error:diagnostic=f'{type(error).__name__}: {error}'
            else:
                if expired(time.perf_counter_ns()-started):status='timeout';reason='deadline_before_spawn'
                else:
                    try:proc=subprocess.Popen(argv,cwd=cwd,env=child_environment,stdin=subprocess.DEVNULL,
                        stdout=stdout,stderr=stderr,start_new_session=True)
                    except OSError as error:reason='process_start_failure';diagnostic=f'{type(error).__name__}: {error}'
                    if proc is not None:
                        write_new_json(output/'spawned.json',dict(schema='research/v2-owned-process',
                            child_pid=proc.pid,child_pgid=proc.pid,new_session=True,
                            observed_utc_ns=time.time_ns()))
                        while True:
                            elapsed=time.perf_counter_ns()-started;code=proc.poll()
                            if expired(elapsed):status='timeout';reason='dedicated_worker_deadline';break
                            if (output/'stdout.log').stat().st_size+(output/'stderr.log').stat().st_size>log_limit_bytes:
                                status='resource_exhausted';reason='log_bytes_limit';break
                            if code is not None:
                                status='exited';reason='process_exit'
                                if _group_exists(proc.pid):status='input_or_dependency_failure';reason='unexpected_live_process_group'
                                break
                            try:rss=root_rss_bytes(proc.pid,probe_budget(elapsed))
                            except ProcessLookupError:
                                if proc.poll() is not None:continue
                                status='input_or_dependency_failure';reason='rss_probe_unavailable';diagnostic='Owned worker disappeared from RSS probe';break
                            except subprocess.TimeoutExpired as error:
                                diagnostic=f'{type(error).__name__}: {error}'
                                if expired(time.perf_counter_ns()-started):
                                    status='timeout';reason='dedicated_worker_deadline_during_rss_probe'
                                else:status='input_or_dependency_failure';reason='rss_probe_unavailable'
                                break
                            except Exception as error:
                                status='input_or_dependency_failure';reason='rss_probe_unavailable';diagnostic=f'{type(error).__name__}: {error}';break
                            if type(rss) is not int or rss<0:
                                status='input_or_dependency_failure';reason='rss_probe_unavailable';diagnostic='Invalid RSS quantity';break
                            peak=rss if peak is None else max(peak,rss);samples+=1
                            observations.write(canonical_json(dict(elapsed_ns=time.perf_counter_ns()-started,root_rss_bytes=rss))+'\n')
                            observations.flush();os.fsync(observations.fileno())
                            if rss>rss_limit_bytes:status='resource_exhausted';reason='observed_root_rss_limit';break
                            remaining=poll_interval_ns if timeout_ns is None else timeout_ns-(time.perf_counter_ns()-started)
                            if remaining>0:time.sleep(min(poll_interval_ns,remaining)/1e9)
        except (KeyboardInterrupt,SystemExit) as error:
            status='interrupted_recoverable';reason='supervisor_interrupted';diagnostic=type(error).__name__
        finally:
            if proc is not None:
                if proc.poll() is None or _group_exists(proc.pid):termination=_terminate(proc,terminate_grace_ns)
                else:proc.wait()
            stdout.flush();stderr.flush();os.fsync(stdout.fileno());os.fsync(stderr.fileno())
    names=['started.json','stdout.log','stderr.log','resource-samples.jsonl']
    if (output/'spawned.json').exists():names.append('spawned.json')
    files=[_file_ref(output/name,output) for name in names]
    receipt=dict(schema='research/v2-process-receipt',**{k:v for k,v in command.items() if k!='schema'},
        execution_status=status,stop_reason=reason,diagnostic=diagnostic,child_started=proc is not None,
        child_pid=None if proc is None else proc.pid,returncode=None if proc is None else proc.returncode,
        child_reaped=proc is not None and proc.returncode is not None,termination=termination,
        observed_peak_root_rss_bytes=peak,rss_sample_count=samples,
        rss_scope='Observed root worker RSS samples; descendants and unobserved peaks are not measured',
        strict_peak_memory_limit_qualified=False,eligible_model=False,protected_access_authorized=False,
        native_recipe_runtime_qualified=False,files=files,elapsed_ns=time.perf_counter_ns()-started,
        clock_scope='Supervisor entry through log closure/file hashing before final receipt publication; child startup included',
        whole_process_cost_authenticated=False)
    receipt['deadline_overshoot_ns']=None if timeout_ns is None else max(0,receipt['elapsed_ns']-timeout_ns)
    # Verify the serialized caller identity was not mutated during the callback-free run.
    if canonical_json(identity)!=identity_json:raise ValueError('Caller process identity changed during execution')
    write_new_json(output/'process.json',receipt)
    return receipt
