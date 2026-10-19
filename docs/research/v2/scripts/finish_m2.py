"""Wait for complete inactive M2, then collect and independently audit once.

This operator imports only the standard library. It cannot fit, resume, change
the frozen queue, or evaluate protected T. Outputs live outside that queue.
"""
import argparse
from contextlib import contextmanager
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


FIELDS = {'schema', 'study', 'producer', 'auditor', 'expected_queue_hash',
          'producer_config', 'producer_config_sha256', 'auditor_helper_sha256',
          'control_audit', 'control_audit_sha256', 'control_plan_hash', 'interpreter'}
COMPLETE = {'completed_eligible', 'completed_ineligible'}
FAILED = {'timeout', 'resource_exhausted', 'numerical_failure',
          'input_or_dependency_failure', 'irrecoverable_missing'}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ordinary(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected an ordinary file: {path}')
    return path


def package_digest(root):
    source = Path(root)/'src/treeneuron'
    if source.is_symlink() or not source.is_dir():
        raise ValueError('Actor has no ordinary package source directory')
    digest = hashlib.sha256()
    paths = sorted(source.rglob('*.py'))
    if not paths:
        raise ValueError('Actor source is empty')
    for path in paths:
        ordinary(path)
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _inputs(config):
    if set(config) != FIELDS or config['schema'] != 'research/v2-M2-delayed-collection':
        raise ValueError('Unknown delayed collection configuration fields/schema')
    for name in ('producer', 'auditor'):
        actor = config[name]
        if set(actor) != {'root', 'source_sha256'} or not Path(actor['root']).is_absolute():
            raise ValueError('Actor must name an absolute root and package commitment')
        if package_digest(actor['root']) != actor['source_sha256']:
            raise ValueError(f'{name} package commitment differs')
    relative = Path(config['producer_config'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Producer configuration must stay in its actor root')
    references = [(Path(config['producer']['root'])/relative, 'producer_config_sha256'),
        (Path(config['auditor']['root'])/'docs/research/v2/scripts/audit_m2_predictions.py', 'auditor_helper_sha256'),
        (Path(config['control_audit']), 'control_audit_sha256')]
    for path, key in references:
        if sha(ordinary(path)) != config[key]:
            raise ValueError(f'{key} external commitment differs')
    for key in ('study', 'control_audit', 'interpreter'):
        if not Path(config[key]).is_absolute():
            raise ValueError(f'{key} must be absolute')


@contextmanager
def _lease(config):
    root = Path(config['study'])
    if root.is_symlink() or not root.is_dir():
        raise ValueError('Unsafe queue root')
    with ordinary(root/'worker-lease.json').open('r') as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _status_locked(config):
    root = Path(config['study'])
    plan = json.loads(ordinary(root/'plan.json').read_text())
    if (plan.get('schema') != 'research/v2-workqueue'
            or plan.get('plan_hash') != config['expected_queue_hash']
            or canonical_hash({k:v for k,v in plan.items() if k != 'plan_hash'}) != config['expected_queue_hash']):
        raise ValueError('Queue plan commitment differs')
    ids = [job['job_id'] for job in plan['attempts']]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError('Empty/duplicate job inventory')
    statuses = dict.fromkeys(ids, 'never_started')
    previous = '0'*64
    for line in ordinary(root/'events.jsonl').read_text().splitlines():
        event = json.loads(line)
        digest = event.pop('event_hash')
        if event['previous_hash'] != previous or canonical_hash(event) != digest:
            raise ValueError('Queue event chain commitment differs')
        name, status = event['attempt_id'], event['status']
        if name not in statuses or status not in COMPLETE | FAILED | {'running', 'interrupted_recoverable'}:
            raise ValueError('Unknown event job/status')
        if statuses[name] in COMPLETE | FAILED:
            raise ValueError('Terminal queue event rewritten')
        statuses[name] = status
        previous = digest
    status = ('failed_queue' if any(s in FAILED for s in statuses.values()) else
              'ready' if all(s in COMPLETE for s in statuses.values()) else 'incomplete_queue')
    return dict(status=status, planned_jobs=len(ids), statuses=dict(Counter(statuses.values())),
                ledger_event_hash=previous, T_accessed=False)


def probe(config):
    _inputs(config)
    with _lease(config) as held:
        if not held:
            return dict(status='producer_active', T_accessed=False)
        return _status_locked(config)


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


COLLECT_CODE = '''import json,sys
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from treeneuron.research.config import load_research_config,expand
from treeneuron.research.opportunity_collection import collect_opportunity
request=json.loads(Path(sys.argv[1]).read_text());c=request['config']
study=expand(load_research_config(Path(c['producer']['root'])/c['producer_config'])).to_dict()
result=collect_opportunity(Path(c['study']),study,expected_queue_hash=c['expected_queue_hash'],
    control_audit_path=Path(c['control_audit']),expected_control_audit_sha256=c['control_audit_sha256'],
    expected_control_plan_hash=c['control_plan_hash'])
if result['ledger_event_hash']!=request['endpoint'] or result['T_accessed'] is not False:
    raise ValueError('Frozen collector endpoint or protected-role scope differs')
from treeneuron.research.ledger import write_new_json
write_new_json(Path(request['output'])/'collection.json',result)
'''


def _run_stage(stage, *, actor, endpoint, config, output):
    stage_root = output/stage
    stage_root.mkdir()
    request = stage_root/'request.json'
    write_new(request, dict(config=config, endpoint=endpoint, output=str(stage_root)))
    if stage == 'producer_collection':
        argv = [config['interpreter'], '-B', '-c', COLLECT_CODE, str(request)]
        result_path = stage_root/'collection.json'
    else:
        argv = [config['interpreter'], '-B', str(Path(actor['root'])/
            'docs/research/v2/scripts/audit_m2_predictions.py'), '--study', config['study'],
            '--expected-plan-hash', config['expected_queue_hash'], '--expected-ledger-hash', endpoint,
            '--output', str(stage_root/'audit')]
        result_path = stage_root/'audit/audit.json'
    overrides = {key:'1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}
    overrides.update(PYTHONPATH=str(Path(actor['root'])/'src'), PYTORCH_ENABLE_MPS_FALLBACK='0')
    env = dict(os.environ, **overrides)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    start = time.perf_counter_ns()
    write_new(stage_root/'started.json', dict(argv=argv, cwd=actor['root'], environment_overrides=overrides,
        environment_unset=['KMP_DUPLICATE_LIB_OK'], actor=actor, started_utc_ns=time.time_ns(),
        helper_sha256=sha(__file__), request_sha256=sha(request), T_accessed=False))
    child = None; code = None; failure = None
    try:
        with (stage_root/'child.log').open('xb') as log:
            child = subprocess.Popen(argv, cwd=actor['root'], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            code = child.wait(timeout=4*3600)
        if code != 0:
            raise RuntimeError(f'{stage} exited {code}; retained child.log')
        result = json.loads(ordinary(result_path).read_text())
        if result.get('ledger_event_hash') != endpoint or result.get('T_accessed') is not False:
            raise ValueError('Postprocessing endpoint/protected-role commitment differs')
        if stage == 'independent_prediction_audit' and result.get('status') != 'completed_arithmetic':
            raise ValueError('Independent prediction audit is incomplete')
        return result
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        raise
    finally:
        write_new(stage_root/'terminal.json', dict(exit_code=code, pid=None if child is None else child.pid,
            reaped=child is None or child.poll() is not None, failure=failure,
            process_elapsed_ns=time.perf_counter_ns()-start, T_accessed=False))


def collect(config, output, *, run_stage=None):
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(Path(config['study']).resolve()):
        raise ValueError('Collection output must be fresh and outside the producer queue')
    _inputs(config)
    with _lease(config) as held:
        if not held:
            raise ValueError('Producer is active')
        ready = _status_locked(config)
        if ready['status'] != 'ready':
            raise ValueError('Queue is incomplete or failed; no collection admitted')
        endpoint = ready['ledger_event_hash']
        output.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter_ns()
        write_new(output/'config.json', config)
        result = dict(schema='research/v2-M2-delayed-collection-result', ledger_event_hash=endpoint,
            readiness=ready, T_accessed=False, efficacy_claim=False, autonomous_fitting=False)
        runner = _run_stage if run_stage is None else run_stage
        try:
            for stage, actor in [('producer_collection', config['producer']),
                                 ('independent_prediction_audit', config['auditor'])]:
                _inputs(config)
                value = runner(stage, actor=actor, endpoint=endpoint, config=config, output=output)
                if value.get('ledger_event_hash') != endpoint:
                    raise ValueError('Postprocessing returned a different ledger endpoint')
                _inputs(config)
                if _status_locked(config) != ready:
                    raise ValueError('Queue changed during postprocessing')
            result['status'] = 'collected_and_audited'
        except BaseException as error:
            result.update(status='postprocessing_failed', failure=dict(type=type(error).__name__,
                message=str(error), traceback=traceback.format_exc()))
            raise
        finally:
            result['elapsed_ns'] = time.perf_counter_ns()-started
            write_new(output/'terminal.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--expected-config-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch-record', type=Path, required=True)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--max-wait-seconds', type=float, default=86400)
    args = parser.parse_args()
    if not (0 < args.poll_seconds <= 60 and 0 < args.max_wait_seconds <= 7*86400):
        raise ValueError('Invalid finite polling policy')
    if sha(ordinary(args.config)) != args.expected_config_sha256:
        raise ValueError('Operator configuration commitment differs')
    config = json.loads(args.config.read_text())
    if args.watch_record.resolve().is_relative_to(Path(config['study']).resolve()):
        raise ValueError('Watch receipt must stay outside the queue')
    args.watch_record.mkdir(parents=True, exist_ok=False)
    helper_hash = sha(__file__); started = time.perf_counter_ns(); failure = None
    write_new(args.watch_record/'started.json', dict(argv=sys.argv, cwd=str(Path.cwd()),
        config_sha256=args.expected_config_sha256, config=config, helper_sha256=helper_hash,
        started_utc_ns=time.time_ns(), T_accessed=False))
    result = None
    try:
        with (args.watch_record/'checks.jsonl').open('x') as checks:
            while True:
                if sha(args.config) != args.expected_config_sha256 or sha(__file__) != helper_hash:
                    raise ValueError('Operator source/config changed while waiting')
                result = probe(config)
                checks.write(json.dumps(dict(observed_utc_ns=time.time_ns(), **result), sort_keys=True)+'\n')
                checks.flush()
                if result['status'] == 'ready':
                    result = collect(config, args.output); break
                if result['status'] == 'failed_queue':
                    break
                remaining = args.max_wait_seconds-(time.perf_counter_ns()-started)/1e9
                if remaining <= 0:
                    result = dict(status='watch_window_expired', last_probe=result, T_accessed=False); break
                time.sleep(min(args.poll_seconds, remaining))
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        raise
    finally:
        write_new(args.watch_record/'terminal.json', dict(result=result, failure=failure,
            elapsed_ns=time.perf_counter_ns()-started, T_accessed=False))
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'collected_and_audited' else 3


if __name__ == '__main__':
    raise SystemExit(main())
