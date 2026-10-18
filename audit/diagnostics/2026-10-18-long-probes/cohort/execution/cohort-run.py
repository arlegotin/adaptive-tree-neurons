"""One locked cohort invocation; operational counters only; zero retries."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

ROOT = Path.cwd()
ENV = ROOT / 'environment/long-probes-20261018'
OUT = ROOT / 'artifacts/long-probes-20261018/cohort'
PYTHON = str(ROOT / '.venv/bin/python')
EXPECTED_SOURCE = '089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543'
THREADS = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTORCH_ENABLE_MPS_FALLBACK='0')


def write(name, value):
    with (ENV / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def sealed(path):
    assert read(path.with_name(path.stem + '-seal.json')) == {'sha256': sha(path)}
    return read(path)


def native():
    result = {'free_disk_bytes': shutil.disk_usage(ROOT).free}
    for label, argv in [('power', ['pmset', '-g', 'batt']),
            ('memory_pressure', ['memory_pressure', '-Q']), ('vm_stat', ['vm_stat'])]:
        p = subprocess.run(argv, capture_output=True, text=True)
        result[label] = dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    return result


def tracked():
    names = subprocess.check_output(['git', 'ls-files', '-z']).split(b'\0')
    return {os.fsdecode(p): sha(ROOT / os.fsdecode(p)) for p in names
        if p and (ROOT / os.fsdecode(p)).is_file()}


def usage():
    r = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {key: getattr(r, key) for key in ('ru_utime', 'ru_stime', 'ru_maxrss',
        'ru_ixrss', 'ru_idrss', 'ru_isrss', 'ru_minflt', 'ru_majflt', 'ru_nswap',
        'ru_inblock', 'ru_oublock', 'ru_msgsnd', 'ru_msgrcv', 'ru_nsignals', 'ru_nvcsw', 'ru_nivcsw')}


def progress():
    value = dict(utc=utc(), completed_members=0, expected_members=16,
        completed_probe_stages=0, created_probe_stages=0, running_member=None)
    ledger_path = OUT / 'members.json'
    if ledger_path.exists():
        ledger = read(ledger_path)
        value['ledger_status'] = ledger['status']
        value['completed_members'] = sum(m['status'] == 'completed' for m in ledger['members'])
        running = [dict(teacher_index=m['teacher_index'], repetition=m['repetition'],
            status=m['status']) for m in ledger['members'] if m['status'] != 'completed']
        value['running_member'] = running[-1] if running else None
        value['ledger_failure_count'] = len(ledger['failures'])
    for path in OUT.glob('teacher-*/repetition-*/runs/*/probes/*/manifest.json'):
        manifest = read(path)
        value['created_probe_stages'] += 1
        value['completed_probe_stages'] += manifest['identity']['status'] == 'completed'
    return value


assert not OUT.exists(), 'New cohort output required; no automatic retry'
assert sha(ENV / 'cpu128-receipt.json') == 'e506511d50107138c0e94b1341687c5d9ff97e2fecac5b500a0d1946d1a93f43'
decision_path = ROOT / 'pilot/long-probes-20261018/resource-decision.json'
declaration_path = ROOT / 'protocol/diagnostics/2026-10-18-long-probes-lock.json'
decision = sealed(decision_path)
declaration = sealed(declaration_path)
assert decision['allowed'] is True
assert declaration['resource_decision_sha256'] == sha(decision_path)
assert declaration['execution_provenance'] == decision['provenance']
assert declaration['expected_members'] == [dict(teacher_index=i, repetition=j)
    for i in range(8) for j in range(2)]
freeze = ROOT / 'audit/diagnostics/2026-10-18-long-probes/cost/controller/source-freeze.json'
assert sha(freeze) == declaration['source_freeze_sha256']
for rel, record in read(freeze)['files'].items():
    assert sha(ROOT / rel) == record['sha256'], rel
assert sha(ROOT / 'src/treeneuron/pilot/probe_diagnostic.py') == declaration['analysis_source_sha256']
prior_inventory = ENV / 'cost-only-inventory.json'
for record in read(prior_inventory)['artifacts']:
    assert sha(ROOT / record['path']) == record['sha256'], record['path']
prior_paths = {p for base in (ENV, ROOT / 'artifacts/long-probes-20261018/cost')
    for p in base.rglob('*') if p.is_file() and not p.name.startswith('cohort-')}
prior_paths.update(ROOT / record['path'] for record in read(prior_inventory)['artifacts'])
prior = {str(p.relative_to(ROOT)): sha(p) for p in sorted(prior_paths)}
baseline = tracked()
bootstrap = ("import runpy,torch; torch.set_num_threads(1); torch.set_num_interop_threads(1); "
    "from treeneuron import train; assert train.source_hash()=='" + EXPECTED_SOURCE + "'; "
    "runpy.run_module('treeneuron.pilot.probe_diagnostic',run_name='__main__')")
argv = [PYTHON, '-c', bootstrap, 'run', '--spec', 'protocol/diagnostics/2026-10-18-long-probes.json',
    '--resource-decision', 'pilot/long-probes-20261018/resource-decision.json',
    '--output', 'artifacts/long-probes-20261018/cohort']
assert declaration['wrapper_arguments'] == dict(command='run',
    output=argv[-1], resource_decision=argv[-3], spec=argv[-5])
write('cohort-invocation.json', dict(argv=argv, cwd=str(ROOT), environment=THREADS,
    logical_cli=[PYTHON, '-m', 'treeneuron.pilot.probe_diagnostic', *argv[3:]],
    expected_source=EXPECTED_SOURCE, receipt_reused=True, receipt_producer_rerun=False,
    automatic_retries=0, fixed_teachers=8, fixed_repetitions=2,
    declaration_sha256=sha(declaration_path), resource_decision_sha256=sha(decision_path),
    bootstrap='Fresh interpreter CPU1/inter-op1 before module entry; all five BLAS1/fallback0 before imports.',
    effect_inspection=False, analysis_run=False, elapsed_time_kill=False))
preflight = native()
assert all(preflight[key]['returncode'] == 0 for key in ('power', 'memory_pressure', 'vm_stat'))
assert "'AC Power'" in preflight['power']['stdout']
assert preflight['free_disk_bytes'] >= decision['forecast']['retained_bytes'] + 10 * 1024**3
write('cohort-preflight.json', dict(utc=utc(), native=preflight, tracked_sha256=baseline,
    prior_evidence_sha256=prior, prior_cost_inventory_sha256=sha(prior_inventory),
    source_freeze_sha256=sha(freeze), declaration_sha256=sha(declaration_path),
    commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    git_status=subprocess.check_output(['git', 'status', '--porcelain'], text=True)))
awake = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
status = 1
observations = []
monitor_errors = []
try:
    before = native()
    with (ENV / 'cohort-stdout.txt').open('xb') as stdout, (ENV / 'cohort-stderr.txt').open('xb') as stderr:
        usage_before = usage()
        started_utc = utc()
        tick = time.perf_counter_ns()
        child = subprocess.Popen(argv, env=dict(os.environ, **THREADS), stdout=stdout, stderr=stderr)
        while True:
            try:
                status = child.wait(timeout=55)
                break
            except subprocess.TimeoutExpired:
                pass
            try:
                observation = progress()
                observation.update(elapsed_ns=time.perf_counter_ns() - tick, native=native())
                observations.append(observation)
                with (ENV / 'cohort-progress.jsonl').open('a') as stream:
                    stream.write(json.dumps(observation, sort_keys=True, allow_nan=False) + '\n')
                print(json.dumps({k: v for k, v in observation.items() if k != 'native'}, sort_keys=True), flush=True)
            except Exception as error:
                monitor_errors.append(dict(utc=utc(), error=f'{type(error).__name__}: {error}'))
                print(json.dumps(dict(monitor_error=monitor_errors[-1], action='continue fixed numerical child')), flush=True)
        elapsed = time.perf_counter_ns() - tick
        finished_utc = utc()
        usage_after = usage()
    files = sorted(p for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else []
    post_bytes = sum(p.stat().st_size for p in files)
    allocated_bytes = sum(p.stat().st_blocks * 512 for p in files)
    after = native()
    write('cohort-outer-measurement.json', dict(returncode=status, started_utc=started_utc,
        finished_utc=finished_utc, outer_elapsed_ns=elapsed,
        elapsed_boundary='perf_counter_ns immediately before Popen child launch through wait completion; all final run publication included; 55-second waits interrupt on exit; operational monitor overhead overlaps execution; initial/final native counters excluded',
        final_post_publication_retained_bytes=post_bytes,
        final_post_publication_allocated_bytes=allocated_bytes,
        final_post_publication_file_count=len(files),
        bytes_boundary='All regular files below new cohort output after child exit; environment execution evidence excluded',
        children_usage_before=usage_before, children_usage_after=usage_after,
        child_peak_rss_bytes=int(usage_after['ru_maxrss']) * (1 if sys.platform == 'darwin' else 1024),
        child_peak_rss_boundary='RUSAGE_CHILDREN maximum over completed children of this fresh wrapper; prior baseline retained; includes lightweight monitor native subprocesses; actual numerical process peak independently retained by internal resource observations',
        child_user_seconds=usage_after['ru_utime'] - usage_before['ru_utime'],
        child_system_seconds=usage_after['ru_stime'] - usage_before['ru_stime'],
        before=before, after=after, operational_observations=len(observations),
        monitor_errors=monitor_errors, automatic_retries=0, elapsed_time_kill=False))
    print(json.dumps(dict(phase='cohort', returncode=status, outer_elapsed_ns=elapsed,
        retained_bytes=post_bytes, allocated_bytes=allocated_bytes, files=len(files))), flush=True)
finally:
    awake.terminate()
    awake.wait()
    now = tracked()
    changed_prior = [p for p, h in prior.items() if not (ROOT / p).is_file() or sha(ROOT / p) != h]
    write('cohort-stop.json', dict(utc=utc(), returncode=status,
        receipt_producer_rerun=False, automatic_retries=0,
        tracked_unchanged=now == baseline,
        changed_tracked=[p for p in sorted(set(now) | set(baseline)) if now.get(p) != baseline.get(p)],
        prior_evidence_unchanged=not changed_prior, prior_evidence_files=len(prior),
        changed_prior_evidence=changed_prior, caffeinate_process_scoped=True,
        effect_inspection=False, analysis_run=False, elapsed_time_kill=False,
        numerical_stop=True, next_step='Independent audit; no further fitting or analysis by cohort executor'))
sys.exit(status)
