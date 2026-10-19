"""Native NODE time budgets do not become shortened training recipes."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from test_method_worker import request_fixture, sha, write_request
from test_node_v2 import config


def api():
    from treeneuron.research import node_v2
    return node_v2


def test_fresh_worker_explicitly_initializes_neural_threads_without_importing_NODE(tmp_path):
    script = '''
import json,sys
from treeneuron.research.method_worker import _initialize_native_threads
assert _initialize_native_threads('P13') is None
assert _initialize_native_threads('P08') is None
assert 'torch' not in sys.modules
record=_initialize_native_threads('P12')
import torch
assert record==dict(intra_op=1,inter_op=1)
assert torch.get_num_threads()==torch.get_num_interop_threads()==1
assert 'treeneuron.baselines._node_vendor.arch' not in sys.modules
print(json.dumps(record,sort_keys=True))
'''
    argv = [sys.executable, '-B', '-c', script]
    run = subprocess.run(argv, env=os.environ|{'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=30)
    (tmp_path/'thread-setup-process.json').write_text(json.dumps(dict(argv=argv, returncode=run.returncode,
        stdout=run.stdout, stderr=run.stderr), indent=2)+'\n')
    assert run.returncode == 0, run.stdout+run.stderr


def test_fresh_validated_NODE_worker_sets_threads_before_factory(tmp_path):
    path, value = request_fixture(tmp_path/'input')
    value.update(family_id='P12', config=dict(config('native_anchor'),
        feature_schema=value['config']['feature_schema']))
    write_request(path, value)
    script = '''
import json,sys
from pathlib import Path
from treeneuron.research import method_worker
def inspect(family):
 import torch
 assert family=='P12'
 assert torch.get_num_threads()==torch.get_num_interop_threads()==1
 raise RuntimeError('verified process-owned thread treatment before native factory')
method_worker._factory=inspect
try:method_worker.run_engineering_trial(Path(sys.argv[1]),expected_request_sha256=sys.argv[2],output=Path(sys.argv[3]))
except RuntimeError as error:assert str(error)=='verified process-owned thread treatment before native factory'
else:raise AssertionError('Expected controlled factory sentinel')
record=json.loads((Path(sys.argv[3])/'terminal.json').read_text())
assert record['torch_threads']==dict(intra_op=1,inter_op=1)
assert record['quality_eligible'] is False
'''
    argv = [sys.executable, '-B', '-c', script, str(path), sha(path), str(tmp_path/'worker')]
    run = subprocess.run(argv, env=os.environ|{'PYTHONDONTWRITEBYTECODE': '1'},
        capture_output=True, text=True, timeout=30)
    (tmp_path/'worker-thread-process.json').write_text(json.dumps(dict(argv=argv, returncode=run.returncode,
        stdout=run.stdout, stderr=run.stderr), indent=2)+'\n')
    assert run.returncode == 0, run.stdout+run.stderr


@pytest.mark.parametrize('mode', ['native_anchor', 'search'])
def test_time_budget_has_no_update_ceiling_and_uses_absolute_monotonic_deadline(mode):
    module = api()
    parameters = dict(trees=512, depth=4, lr_multiplier=1) if mode == 'search' else {}
    cfg = module.NodeV2Config.parse(config(mode, **parameters))
    cap, deadline = module._training_budget(cfg, dict(max_seconds=120.), started_ns=19)
    assert cap is None and deadline == 120_000_000_019
    assert cfg.resolved()['patience'] == 5000 and cfg.resolved()['report_frequency'] == 100


@pytest.mark.parametrize('budget', [dict(max_seconds=True), dict(max_seconds=0),
    dict(max_seconds=float('inf')), dict(max_seconds=7201), dict(max_seconds=1, max_updates=1)])
def test_invalid_native_time_budget_is_not_coerced(budget):
    module = api()
    cfg = module.NodeV2Config.parse(config('native_anchor'))
    with pytest.raises(ValueError): module._training_budget(cfg, budget, started_ns=0)


def test_explicit_engineering_update_ceiling_stays_bounded():
    module = api();cfg = module.NodeV2Config.parse(config())
    assert module._training_budget(cfg, dict(max_updates=128), started_ns=0) == (128, None)
    for budget in (dict(max_updates=129), dict(max_updates=True), dict(max_seconds=1)):
        with pytest.raises(ValueError): module._training_budget(cfg, budget, started_ns=0)


def test_native_shared_worker_expiry_is_timeout_before_native_allocation(tmp_path, monkeypatch):
    from treeneuron.research import method_worker
    module = api();path, value = request_fixture(tmp_path/'input')
    value.update(family_id='P12', config=dict(config('native_anchor'),
        feature_schema=value['config']['feature_schema']), budget=dict(max_seconds=1e-12))
    write_request(path, value)
    def forbidden(*args, **kwargs): pytest.fail('Expired NODE trial allocated native model/runtime')
    monkeypatch.setattr(module, '_native', forbidden)
    with pytest.raises(TimeoutError):
        method_worker.run_engineering_trial(path, expected_request_sha256=sha(path), output=tmp_path/'worker')
    terminal = json.loads((tmp_path/'worker/terminal.json').read_text())
    assert terminal['status'] == 'timeout' and terminal['quality_eligible'] is False
    assert terminal['adapter_import_started'] and terminal['T_accessed'] is False
    receipt = json.loads((tmp_path/'worker/fit/fit-receipt.json').read_text())
    assert receipt['budget'] == dict(max_seconds=1e-12) and receipt['executed_updates'] == 0
    assert receipt['error_type'] == 'TimeoutError' and receipt['phase'] == 'configuration'


def test_actual_small_NODE_unbounded_loop_preserves_native_patience_and_deadline():
    module = api()
    import torch
    cfg = module.NodeV2Config.parse(config(report_frequency=1, patience=1))
    resolved = cfg.resolved()
    rng = np.random.RandomState(91)
    F = rng.normal(size=(13, 2)).astype('float32');G = rng.normal(size=(5, 2)).astype('float32')
    model = module._build_model(2, resolved)
    with torch.no_grad(): model(torch.from_numpy(F))
    optimizer = module._optimizer(model, resolved)
    trace = []
    # Predetermined G evaluations make the expected strict patience event exact;
    # actual NODE forward/backward/QHAdam and row stream still execute.
    original = module._predict_normalized
    try:
        module._predict_normalized = lambda model, features, chunk: np.zeros(len(features), dtype='float32')
        state = module._train(model, optimizer, F, np.ones(len(F), dtype='float32'),
            G, np.ones(len(G), dtype='float32'), resolved, None, trace,
            deadline_ns=time.perf_counter_ns()+10_000_000_000)
    finally:
        module._predict_normalized = original
    assert state['summary']['selected_step'] == 1 and state['summary']['executed_updates'] == 3
    assert state['summary']['stop_reason'] == 'G_patience'
    assert len(trace) == 3 and all(row['applied'] for row in trace)
    assert trace[0]['rows']+trace[1]['rows'] == list(dict.fromkeys(trace[0]['rows']+trace[1]['rows']))
    failed_trace = []
    with pytest.raises(TimeoutError):
        module._train(model, optimizer, F, np.ones(len(F), dtype='float32'),
            G, np.ones(len(G), dtype='float32'), resolved, None, failed_trace,
            deadline_ns=time.perf_counter_ns()-1)
    assert failed_trace == []


def test_expiry_during_update_prevents_a_new_G_evaluation(monkeypatch):
    module = api()
    import torch
    resolved = module.NodeV2Config.parse(config(report_frequency=1, patience=1)).resolved()
    F = np.random.RandomState(7).normal(size=(13, 2)).astype('float32')
    model = module._build_model(2, resolved)
    with torch.no_grad(): model(torch.from_numpy(F))
    optimizer = module._optimizer(model, resolved)
    original = module._optimizer_step;clock = [0]
    def step(optimizer):
        original(optimizer);clock[0] = 1_000_000_000
    def forbidden(*args): pytest.fail('Expired update started a new G evaluation')
    monkeypatch.setattr(module, '_optimizer_step', step)
    monkeypatch.setattr(module, '_predict_normalized', forbidden)
    monkeypatch.setattr(module.time, 'perf_counter_ns', lambda: clock[0])
    trace = []
    with pytest.raises(TimeoutError):
        module._train(model, optimizer, F, np.ones(len(F), dtype='float32'),
            F, np.ones(len(F), dtype='float32'), resolved, None, trace, deadline_ns=100)
    assert len(trace) == 1 and trace[0]['applied'] and trace[0]['elapsed_ns'] == 1_000_000_000
