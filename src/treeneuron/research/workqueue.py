"""Finite immutable prerequisite execution, paid failures and one exact retry."""
import copy
import hashlib
import heapq
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

from treeneuron.config import canonical_hash,canonical_json
from .ledger import AttemptLedger,StudyLease,TERMINAL,write_new_json


def make_queue_plan(identity,jobs):
    jobs=json.loads(canonical_json({'jobs':jobs}))['jobs'];identity=json.loads(canonical_json(identity))
    if not isinstance(identity,dict) or not isinstance(jobs,list):raise ValueError('Invalid queue identity/jobs')
    ids={}
    for i,job in enumerate(jobs):
        if (not isinstance(job,dict) or set(job)-{'dependency_policy'}!={'job_id','kind','depends_on','payload'}
                or any(type(job[k]) is not str or not job[k] for k in ('job_id','kind'))
                or not isinstance(job['payload'],dict) or not isinstance(job['depends_on'],list)
                or any(type(dep) is not str for dep in job['depends_on'])
                or len(set(job['depends_on']))!=len(job['depends_on']) or job['job_id'] in ids):
            raise ValueError('Invalid or duplicate queue job')
        job.setdefault('dependency_policy','all_eligible')
        if job['dependency_policy'] not in ('all_eligible','all_terminal'):
            raise ValueError('Invalid queue dependency policy')
        ids[job['job_id']]=i
    degrees=[len(job['depends_on']) for job in jobs];children={name:[] for name in ids}
    for i,job in enumerate(jobs):
        for dep in job['depends_on']:
            if dep not in ids:raise ValueError('Unknown queue dependency')
            children[dep].append(i)
    ready=[i for i,n in enumerate(degrees) if not n];heapq.heapify(ready);ordered=[]
    while ready:
        i=heapq.heappop(ready);ordered.append(jobs[i])
        for child in children[jobs[i]['job_id']]:
            degrees[child]-=1
            if not degrees[child]:heapq.heappush(ready,child)
    if len(ordered)!=len(jobs):raise ValueError('Queue dependency cycle')
    value=dict(schema='research/v2-workqueue',identity=identity,
        attempts=[dict(attempt_id=job['job_id'],**job) for job in ordered])
    value['plan_hash']=canonical_hash(value)
    return value


def _inventory(directory,*,exclude_marker=False):
    if directory.is_symlink() or not directory.is_dir():raise ValueError('Unsafe queue output root')
    entries=[];directories=[]
    for path in sorted(directory.rglob('*')):
        if exclude_marker and path==directory/'job-result.json':continue
        if path.is_symlink():raise ValueError('Queue outputs cannot contain symbolic links')
        if path.is_dir():
            directories.append(path.relative_to(directory).as_posix());continue
        if not path.is_file():raise ValueError('Queue outputs must be ordinary files')
        entries.append(dict(path=path.relative_to(directory).as_posix(),bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(files=entries,directories=directories)


def _publish(directory,job,attempt,value):
    result=dict(job_id=job['job_id'],job_hash=canonical_hash(job),attempt=attempt,
        value=value,**_inventory(directory))
    path=directory/'job-result.json';write_new_json(path,result)
    return dict(path=path.as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _read(reference,root,job):
    path=Path(reference['path']).resolve()
    if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest()!=reference['sha256']):
        raise ValueError('Queue result commitment mismatch')
    result=json.loads(path.read_text())
    if result['job_id']!=job['job_id'] or result['job_hash']!=canonical_hash(job):
        raise ValueError('Queue job identity commitment mismatch')
    try:inventory=_inventory(path.parent,exclude_marker=True)
    except (ValueError,OSError) as error:raise ValueError('Queue output inventory commitment invalid') from error
    if inventory!={key:result[key] for key in ('files','directories')}:
        raise ValueError('Queue output inventory commitment mismatch')
    return result


def _control_directory(root,category,name):
    base=root/'control'/category/canonical_hash({'job_id':name})
    base.mkdir(parents=True,exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='publication-',dir=base))


def _latest(ledger):
    return {event['attempt_id']:event for event in ledger.events}


def _retryable(error):
    return isinstance(error,(KeyboardInterrupt,MemoryError)) or (
        isinstance(error,OSError) and not isinstance(error,FileNotFoundError))


def _loop(plan,root,ledger,handler,check_identity):
    results={};latest=_latest(ledger);jobs={job['job_id']:job for job in plan['attempts']}
    for job in plan['attempts']:
        event=latest.get(job['job_id'])
        if event and event['status'] in TERMINAL-{'interrupted_recoverable'}:
            results[job['job_id']]=_read(event['result'],root,job)
    for job in plan['attempts']:
        name=job['job_id']
        if name in results:continue
        check_identity(copy.deepcopy(plan['identity']))
        previous=_latest(ledger).get(name);paid=0 if previous is None else previous['elapsed_ns']
        attempt=0 if previous is None else 1
        if previous and previous['status']=='running':
            bound=max(0,time.time_ns()-previous['started_utc_ns']);paid+=bound
            if previous.get('recovery_attempt')==1:
                directory=_control_directory(root,'second-orphan-closure',name)
                result=_publish(directory,job,1,dict(status='irrecoverable_missing',reason='Second worker loss'))
                ledger.record(name,'irrecoverable_missing',elapsed_ns=paid,result=result,
                    cost_precision='wall_time_upper_bound_after_worker_loss')
                results[name]=_read(result,root,job);continue
            ledger.record(name,'interrupted_recoverable',elapsed_ns=paid,
                reason='Exclusive lease acquired after orphaned running event',
                cost_precision='wall_time_upper_bound_after_worker_loss')
        latest=_latest(ledger)
        dependencies={dep:_read(latest[dep]['result'],root,jobs[dep]) for dep in job['depends_on']}
        directory=root/'jobs'/canonical_hash({'job_id':name})/f'attempt-{attempt}'
        ledger.record(name,'running',elapsed_ns=paid,recovery_attempt=attempt,started_utc_ns=time.time_ns())
        started=time.perf_counter_ns();directory.mkdir(parents=True,exist_ok=False)
        try:
            statuses=[result['value']['status'] for result in dependencies.values()]
            if job['dependency_policy']=='all_eligible' and any(
                    status not in ('completed_eligible','completed_ineligible') for status in statuses):
                value=dict(status='irrecoverable_missing',reason='A required prerequisite failed')
            elif job['dependency_policy']=='all_eligible' and any(status=='completed_ineligible' for status in statuses):
                value=dict(status='completed_ineligible',reason='A required prerequisite is unavailable')
            else:
                value=handler(copy.deepcopy(job),directory,copy.deepcopy(dependencies),attempt)
                if (not isinstance(value,dict) or value.get('status') not in TERMINAL-{'interrupted_recoverable'}):
                    raise ValueError('Handler must return a terminal JSON result or raise an interruption')
                canonical_json(value)
            check_identity(copy.deepcopy(plan['identity']))
            result=_publish(directory,job,attempt,value)
        except BaseException as error:
            retry=_retryable(error)
            status=('interrupted_recoverable' if retry and attempt==0 else 'irrecoverable_missing' if retry
                else 'numerical_failure' if isinstance(error,FloatingPointError) else 'input_or_dependency_failure')
            value=dict(status=status,error_type=type(error).__name__,reason=str(error),traceback=traceback.format_exc())
            try:
                try:
                    value['raw_attempt_evidence']=dict(path=str(directory),inventory=_inventory(directory),
                        status='inventory_committed_at_failure')
                except (ValueError,OSError) as inventory_error:
                    value['raw_attempt_evidence']=dict(path=str(directory),status='rejected_retained_unused',
                        reason=str(inventory_error))
                failure_directory=_control_directory(root,'failure-results',name)
                result=_publish(failure_directory,job,attempt,value)
                ledger.record(name,status,elapsed_ns=paid+time.perf_counter_ns()-started,result=result,
                    recovery_attempt=attempt,attempt_terminal_status=(
                        'resource_exhausted' if isinstance(error,(MemoryError,OSError)) else status))
            except BaseException as publication_error:
                error.add_note(f'Queue failure publication also failed: {publication_error}')
                raise error
            if retry or not isinstance(error,Exception):raise
        else:
            ledger.record(name,value['status'],elapsed_ns=paid+time.perf_counter_ns()-started,result=result,
                recovery_attempt=attempt)
        results[name]=_read(result,root,job)
    return results


def run_queue(plan,output,handler,*,check_identity,resume=False):
    """Callers supply identity/resource policy and model-specific handlers."""
    started=time.perf_counter_ns();plan=json.loads(canonical_json(plan));root=Path(output).resolve()
    if plan.get('plan_hash')!=canonical_hash({k:v for k,v in plan.items() if k!='plan_hash'}):
        raise ValueError('Queue plan digest differs')
    check_identity(copy.deepcopy(plan['identity']))
    if not resume:ledger=AttemptLedger.create(root,plan)
    with StudyLease(root):
        if resume:ledger=AttemptLedger.resume(root,plan)
        try:
            _loop(plan,root,ledger,handler,check_identity)
            statuses={name:e['status'] for name,e in _latest(ledger).items()}
            result=dict(status='incomplete' if ledger.pending_ids else 'completed'
                if all(s in ('completed_eligible','completed_ineligible') for s in statuses.values())
                else 'completed_with_failures',statuses=statuses,charged_job_ns=ledger.charged_ns)
            return result
        except BaseException as error:
            result=dict(status='interrupted_or_failed',error_type=type(error).__name__,reason=str(error),
                statuses={name:e['status'] for name,e in _latest(ledger).items()},charged_job_ns=ledger.charged_ns)
            raise
        finally:
            active=sys.exc_info()[1]
            result['cost_precision']=('contains_worker_loss_upper_bound' if any(
                e.get('cost_precision')=='wall_time_upper_bound_after_worker_loss' for e in ledger.events)
                else 'measured_job_elapsed')
            result['invocation_ns']=time.perf_counter_ns()-started
            result['cost_boundary']='run_queue call; job timers nest; startup/imports external; no standalone procedure claim'
            index=len(list(root.glob('invocation-*.json')))
            try:write_new_json(root/f'invocation-{index}.json',result)
            except BaseException as error:
                if active is None:raise
                active.add_note(f'Queue invocation publication also failed: {error}')
