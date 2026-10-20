"""Real nested workers exercise cooperative ownership and cold-process costs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.cooperative_process_v2'), 'Missing cooperative cold-process supervisor'
    from treeneuron.research import cooperative_process_v2
    return cooperative_process_v2


WORKER = '''
import json, os, sys
from pathlib import Path
from treeneuron.research.cooperative_process_v2 import cooperative_worker
mode, target = sys.argv[1], Path(sys.argv[2])
identity = dict(fixture='nested-cold-worker', source='engineering-only')
if mode == 'wrong': identity['source'] = 'wrong-source'
if mode == 'spoof':
    import time
    from treeneuron.research.cooperative_process_v2 import _read, _write, _ready_expected
    control = Path(os.environ['ATN_SEARCH_CONTROL_ROOT'])
    start, raw = _read(control/'started.json')
    ready = _ready_expected(start,os.environ['ATN_SEARCH_CONTROL_START_SHA256'],os.getpid(),os.getppid())
    ready['identity_sha256'] = '0'*64
    _write(control/'ready.json',ready)
    time.sleep(1)
    raise SystemExit(13)
with cooperative_worker(identity=identity):
    assert not any(k in sys.modules for k in ('torch','numpy','yaml'))
    if mode == 'nonzero': raise SystemExit(7)
    if mode == 'abrupt': os._exit(9)
    from treeneuron.research.process_supervision_v2 import supervise_process
    r = supervise_process(argv=[sys.executable, '-B', '-c', 'import time; print("nested-ready",flush=True); time.sleep(' + ('30' if mode == 'pause' else '.1') + ')'],
        cwd=str(target.parent), env_overrides={}, output=target,
        timeout_ns=None, rss_limit_bytes=64*1024**2, poll_interval_ns=10_000_000,
        identity=dict(fixture='owned-nested-process'))
    print(json.dumps(dict(nested_status=r['execution_status'], nested_reaped=r['child_reaped'])), flush=True)
'''


@pytest.fixture(scope='module')
def observations(tmp_path_factory):
    cached = {}
    def get():
        if cached:
            return cached
        module = api()
        root = tmp_path_factory.mktemp('cooperative-cold-root')
        script = root / 'worker.py'
        script.write_text(WORKER)
        for name, mode, admitted, threshold, pause in (
            ('normal', 'normal', True, 256*1024**2, None),
            ('denied', 'normal', False, 256*1024**2, None),
            ('wrong', 'wrong', True, 256*1024**2, None),
            ('spoof', 'spoof', True, 256*1024**2, None),
            ('paused', 'pause', True, 256*1024**2, 1_500_000_000),
            ('rss', 'pause', True, 1, None),
            ('nonzero', 'nonzero', True, 256*1024**2, None),
            ('abrupt', 'abrupt', True, 256*1024**2, None),
        ):
            directory = root / name
            directory.mkdir()
            kwargs = dict(argv=[sys.executable, '-P', '-B', str(script), mode, str(directory/'nested')],
                cwd=str(root), env_overrides={'PYTHONPATH': str(Path(__file__).resolve().parents[1]/'src'),
                    'PYTHONDONTWRITEBYTECODE': '1'}, output=directory/'process',
                identity=dict(fixture='nested-cold-worker', source='engineering-only'),
                admission_identity=dict(policy='fixed-engineering-admission'),
                admission_check=lambda admitted=admitted: dict(admitted=admitted, reason='fixture'),
                rss_pause_bytes=threshold, poll_interval_ns=20_000_000,
                handshake_timeout_ns=10_000_000_000, pause_after_ns=pause)
            receipt = module.supervise_cooperative_process(**kwargs)
            cached[name] = dict(kwargs=kwargs, receipt=receipt, root=directory)
        return cached
    return get


def reopen(row):
    root = row['root'] / 'process'
    sha = lambda name: hashlib.sha256((root/name).read_bytes()).hexdigest()
    return api().inspect_cooperative_process(root=root, expected_receipt_sha256=sha('process.json'),
        expected_started_sha256=sha('started.json'))


def test_normal_nested_process_is_joined_and_external_clock_includes_startup(observations):
    row = observations()['normal']
    result = row['receipt']
    assert result['execution_status'] == 'exited' and result['returncode'] == 0
    assert result['grant_issued'] and result['child_reaped'] and result['worker_protocol_closed']
    assert result['elapsed_ns'] >= result['grant_elapsed_ns'] > result['spawn_elapsed_ns'] > 0
    nested = json.loads((row['root']/'nested/process.json').read_text())
    assert nested['child_reaped'] and nested['elapsed_ns'] < result['elapsed_ns']
    assert nested['child_pid'] != result['child_pid']
    assert result['signals'] == []
    proof = reopen(row)
    assert proof['root_process_clock_authenticated'] is True
    assert proof['native_recipe_runtime_qualified'] is proof['protected_access_authorized'] is False
    assert proof['whole_training_procedure_cost_authenticated'] is False


@pytest.mark.parametrize('name', ['denied', 'wrong', 'spoof'])
def test_no_grant_means_no_nested_worker_and_only_own_root_can_be_stopped(observations, name):
    row = observations()[name]
    result = row['receipt']
    assert result['grant_issued'] is False and result['child_reaped']
    assert not (row['root']/'nested').exists()
    assert not (row['root']/'process/grant.json').exists()
    assert all(s['scope'] == 'owned_ungranted_root' for s in result['signals'])
    assert reopen(row)['worker_protocol_closed'] is False


def test_granted_pause_joins_separate_nested_session_before_root_exit(observations):
    row = observations()['paused']
    result = row['receipt']
    assert result['execution_status'] == 'interrupted_recoverable'
    assert result['returncode'] == 0 and result['worker_protocol_closed']
    assert result['signals'] == [dict(signal='SIGINT', scope='owned_authorized_root')]
    nested = json.loads((row['root']/'nested/process.json').read_text())
    assert nested['execution_status'] == 'interrupted_recoverable'
    assert nested['child_reaped'] and nested['termination']['group_absent_observed']
    assert reopen(row)['cooperative_stop_requested'] is True


def test_RSS_pause_uses_cooperative_signal_and_claims_no_strict_or_aggregate_memory_bound(observations):
    row = observations()['rss']
    result = row['receipt']
    assert result['grant_issued'] and result['execution_status'] == 'resource_exhausted'
    assert result['observed_peak_root_rss_bytes'] > 1
    assert result['signals'] == [dict(signal='SIGINT', scope='owned_authorized_root')]
    assert result['child_reaped'] and result['worker_protocol_closed']
    proof = reopen(row)
    assert proof['strict_peak_memory_limit_qualified'] is proof['aggregate_descendant_memory_measured'] is False


@pytest.mark.parametrize('name,code,closed', [('nonzero',7,True), ('abrupt',9,False)])
def test_failures_keep_costs_and_protocol_closure_limits_without_a_predictive_score(observations,name,code,closed):
    row = observations()[name]
    assert row['receipt']['returncode'] == code and row['receipt']['worker_protocol_closed'] is closed
    proof = reopen(row)
    assert proof['root_process_clock_authenticated'] and proof['eligible_model'] is False
    assert 'V_score' not in proof


def test_reader_rejects_changed_raw_RSS_samples(observations):
    row = observations()['normal']; path = row['root']/'process/resource-samples.jsonl'
    before = path.read_bytes()
    try:
        path.write_bytes(before + b'{}\n')
        with pytest.raises(ValueError, match='commitment'):
            reopen(row)
    finally:
        path.write_bytes(before)


@pytest.mark.parametrize('field,value', [('child_reaped',False), ('rss_sample_count',9999),
    ('grant_issued',False), ('eligible_model',True), ('elapsed_ns',True)])
def test_reader_rejects_recommitted_inconsistent_receipts(observations,field,value):
    row = observations()['normal']; path = row['root']/'process/process.json'; before = path.read_bytes()
    try:
        result = json.loads(before); result[field] = value; path.write_text(json.dumps(result))
        with pytest.raises(ValueError): reopen(row)
    finally:
        path.write_bytes(before)


def test_reserved_handshake_environment_cannot_be_overridden(tmp_path):
    with pytest.raises(ValueError, match='environment'):
        api().supervise_cooperative_process(argv=[sys.executable, '-c', 'pass'], cwd=str(tmp_path),
            env_overrides={'ATN_SEARCH_CONTROL_ROOT':'foreign'}, output=tmp_path/'run', identity={'a':1},
            admission_identity={'policy':'fixture'}, admission_check=lambda: {'admitted':True},
            rss_pause_bytes=64*1024**2)


@pytest.mark.parametrize('boundary', ['grant_before', 'grant_after', 'stop_before', 'stop_after'])
def test_publication_errors_still_join_the_owned_worker_without_outer_hard_kill(tmp_path,monkeypatch,boundary):
    module = api(); script = tmp_path/'worker.py'; script.write_text(WORKER)
    original_write = module._write; original_popen = module.subprocess.Popen; owned = []
    def publish(path,value):
        if Path(path).name == 'grant.json' and boundary == 'grant_before':
            raise OSError(28, 'controlled ENOSPC before grant publication')
        if Path(path).name == 'request-stop.json' and boundary == 'stop_before':
            raise OSError(28, 'controlled ENOSPC before stop publication')
        original_write(path,value)
        if Path(path).name == 'grant.json' and boundary == 'grant_after':
            raise OSError(28, 'controlled ENOSPC after grant publication')
        if Path(path).name == 'request-stop.json' and boundary == 'stop_after':
            raise OSError(28, 'controlled ENOSPC after stop publication')
    def start(*args,**kwargs):
        process = original_popen(*args,**kwargs)
        if kwargs.get('start_new_session'): owned.append(process)
        return process
    monkeypatch.setattr(module,'_write',publish)
    monkeypatch.setattr(module.subprocess,'Popen',start)
    try:
        r = module.supervise_cooperative_process(argv=[sys.executable,'-P','-B',str(script),'normal',str(tmp_path/'nested')],
            cwd=str(tmp_path),env_overrides={'PYTHONPATH':str(Path(__file__).resolve().parents[1]/'src')},
            output=tmp_path/'process',identity=dict(fixture='nested-cold-worker',source='engineering-only'),
            admission_identity=dict(policy='fixture'), admission_check=lambda:dict(admitted=True),
            rss_pause_bytes=1 if boundary.startswith('stop') else 256*1024**2,poll_interval_ns=20_000_000)
        assert r['child_reaped'] and r['grant_publication_started']
        assert r['signals'] == [dict(signal='SIGINT',scope='owned_authorized_root')]
        assert r['execution_status'] == 'input_or_dependency_failure'
        assert reopen(dict(root=tmp_path))['root_process_clock_authenticated']
    finally:
        # Fixtures are bounded even when exercising a defective implementation.
        for process in owned: process.wait(timeout=5)


def test_timeout_without_ready_does_not_admit_or_spawn_nested_workers(tmp_path):
    r = api().supervise_cooperative_process(argv=[sys.executable,'-c','import time; time.sleep(1)'],
        cwd=str(tmp_path),env_overrides={},output=tmp_path/'process',identity=dict(fixture='no-ready'),
        admission_identity=dict(policy='fixture'),admission_check=lambda:dict(admitted=True),
        rss_pause_bytes=64*1024**2,handshake_timeout_ns=100_000_000,poll_interval_ns=10_000_000)
    assert r['stop_reason']=='handshake_timeout' and not r['grant_issued'] and r['child_reaped']
    assert reopen(dict(root=tmp_path))['root_process_clock_authenticated']


def test_reopening_a_receipt_never_starts_a_process_or_probes_an_old_PID(observations,monkeypatch):
    row = observations()['normal']
    def forbidden(*args,**kwargs): raise AssertionError('Read-only receipt must not touch processes')
    monkeypatch.setattr(api().subprocess,'Popen',forbidden)
    assert reopen(row)['root_process_clock_authenticated']


@pytest.mark.parametrize('name,field,value', [('ready.json','identity_sha256','0'*64),
    ('grant.json','nonce','wrong'),('worker-closure.json','child_pid',True)])
def test_recommitted_protocol_files_cannot_change_ownership(observations,name,field,value):
    row = observations()['normal'];root = row['root']/'process'
    path = root/name;receipt_path = root/'process.json'
    before, receipt_before = path.read_bytes(), receipt_path.read_bytes()
    try:
        value_dict = json.loads(before);value_dict[field] = value;path.write_text(json.dumps(value_dict))
        receipt = json.loads(receipt_before)
        next(r for r in receipt['files'] if r['path']==name).update(
            bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(ValueError):reopen(row)
    finally:
        path.write_bytes(before);receipt_path.write_bytes(receipt_before)
