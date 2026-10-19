"""Actual NODE training resumes raw state, not its rolling-average predictor."""
import copy
import json

import numpy as np
import pytest

from treeneuron.baselines.api import FitRequest
from test_node_v2 import config, request, native_policy


def same(a, b, torch):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            if key != 'elapsed_ns':
                same(a[key], b[key], torch)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y, torch)
    else:
        assert a == b


def test_native_stopping_is_explicit_and_not_a_large_fake_deadline():
    from treeneuron.research.node_v2 import NodeV2Config, _training_budget
    cfg = NodeV2Config.parse(config('native_anchor'))
    assert _training_budget(cfg, {'native_stopping': True}, started_ns=123) == (None, None)
    for bad in ({'native_stopping': 1}, {'native_stopping': False},
                {'native_stopping': True, 'max_seconds': 7200}):
        with pytest.raises(ValueError):
            _training_budget(cfg, bad, started_ns=123)


def test_same_runtime_interruption_resume_matches_every_training_state(tmp_path, native_policy, monkeypatch):
    from treeneuron.research import node_v2 as node
    cfg = config(report_frequency=2)
    base = request(tmp_path/'uninterrupted', 17, cfg)
    reference = node.NodeV2Predictor().fit(base)
    checkpoints = tmp_path/'checkpoints'
    original = node._optimizer_step
    calls = 0

    def interrupt(optimizer):
        nonlocal calls
        calls += 1
        if calls == 8:
            raise KeyboardInterrupt('controlled after checkpoint6, before update8')
        return original(optimizer)

    monkeypatch.setattr(node, '_optimizer_step', interrupt)
    with pytest.raises(KeyboardInterrupt):
        node.NodeV2Predictor().fit(request(tmp_path/'interrupted', 17, cfg),
                                 training_checkpoints=checkpoints)
    assert json.loads((checkpoints/'latest.json').read_text())['step'] == 6
    monkeypatch.setattr(node, '_optimizer_step', original)
    monkeypatch.setattr(node, '_initialize_model', lambda *a: pytest.fail('Resume redrew data-aware initialization'))
    recovered = node.NodeV2Predictor().fit(request(tmp_path/'resumed', 17, cfg),
                                         training_checkpoints=checkpoints, resume=True)
    same(reference._state, recovered._state, native_policy)
    np.testing.assert_array_equal(reference.predict(base.X_select), recovered.predict(base.X_select))
    assert json.loads((tmp_path/'resumed/fit-receipt.json').read_text())['resume_step'] == 6
    # Histories occur in external immutable segments, never every tensor payload.
    manifests = list(checkpoints.glob('checkpoint-*/manifest.json'))
    assert manifests and all('trace' not in json.loads(p.read_text())['state_fields'] for p in manifests)


@pytest.mark.parametrize('change', ['seed', 'F', 'G', 'config', 'budget'])
def test_resume_rejects_changed_identity(tmp_path, native_policy, change):
    from treeneuron.research.node_v2 import NodeV2Predictor
    checkpoints = tmp_path/'checkpoints'
    req = request(tmp_path/'first', 7)
    NodeV2Predictor().fit(req, training_checkpoints=checkpoints)
    values = dict(X_fit=req.X_fit.copy(), y_fit=req.y_fit.copy(), X_select=req.X_select.copy(),
                  y_select=req.y_select.copy(), seed=req.seed, config=copy.deepcopy(req.config),
                  budget=copy.deepcopy(req.budget), artifact_dir=tmp_path/'changed')
    if change == 'seed': values['seed'] += 1
    elif change == 'F': values['y_fit'][0] += .1
    elif change == 'G': values['y_select'][0] += .1
    elif change == 'config': values['config']['parameters']['batch_size'] = 16
    else: values['budget']['max_updates'] = 9
    with pytest.raises(ValueError, match='binding|identity'):
        NodeV2Predictor().fit(FitRequest(**values), training_checkpoints=checkpoints, resume=True)


def test_incomplete_publication_keeps_previous_recoverable_prefix(tmp_path, native_policy, monkeypatch):
    from treeneuron.research import node_checkpoints as store
    from treeneuron.research.node_v2 import NodeV2Predictor
    checkpoints = tmp_path/'checkpoints'
    real = store._publish_pointer

    def fail(path, pointer):
        if pointer['step'] == 4:
            raise OSError('controlled pointer interruption')
        real(path, pointer)

    monkeypatch.setattr(store, '_publish_pointer', fail)
    with pytest.raises(OSError):
        NodeV2Predictor().fit(request(tmp_path/'broken', 9), training_checkpoints=checkpoints)
    assert json.loads((checkpoints/'latest.json').read_text())['step'] == 2
    monkeypatch.setattr(store, '_publish_pointer', real)
    result = NodeV2Predictor().fit(request(tmp_path/'recovered', 9), training_checkpoints=checkpoints, resume=True)
    reference = NodeV2Predictor().fit(request(tmp_path/'reference', 9))
    same(reference._state, result._state, native_policy)
    assert len(list(checkpoints.glob('checkpoint-*'))) >= 6  # Failed publication retained.


@pytest.fixture(scope='module')
def valid_training_state(tmp_path_factory, native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    root = tmp_path_factory.mktemp('recovery-semantic-state')
    p = NodeV2Predictor().fit(request(root/'fit', 6), training_checkpoints=root/'checkpoints')
    state = copy.deepcopy(p._state)
    state['summary']['stop_reason'] = 'checkpoint_boundary'
    return state, p._config, json.loads((root/'checkpoints/binding.json').read_text())


@pytest.mark.parametrize('change', ['negative_moment', 'moment_age', 'dtype', 'averaged_raw', 'rng'])
def test_corrupt_training_state_is_rejected_before_pointer_publication(tmp_path, native_policy, valid_training_state, change):
    from treeneuron.research.node_checkpoints import NodeCheckpointStore
    state, cfg, binding = copy.deepcopy(valid_training_state)
    if change == 'negative_moment':
        next(iter(state['raw_optimizer']['state'].values()))['exp_avg_sq'].flatten()[0] = -1
    elif change == 'moment_age':
        next(iter(state['raw_optimizer']['state'].values()))['beta1_weight'] = 0.
    elif change == 'dtype':
        key = next(k for k,v in state['raw_model'].items() if v.is_floating_point())
        state['raw_model'][key] = state['raw_model'][key].double()
    elif change == 'averaged_raw':
        # At a report boundary the raw snapshot must equal the newest raw member.
        state['raw_model'] = copy.deepcopy(state['selected_model'])
    else:
        state['rng']['torch'] = state['rng']['torch'].double()
    store = NodeCheckpointStore(tmp_path/'store', binding)
    with pytest.raises(ValueError):
        store.publish(state, cfg)
    assert not (tmp_path/'store/latest.json').exists()


def test_valid_looking_changed_F_commitment_still_rejects_binding(tmp_path,native_policy,valid_training_state):
    from treeneuron.research.node_checkpoints import NodeCheckpointStore
    from treeneuron.research.node_v2 import _prediction_binding
    state,cfg,binding=copy.deepcopy(valid_training_state)
    state['F_identity']['target']['sha256']='0'*64
    state['prediction_binding']=_prediction_binding(state,cfg)
    store=NodeCheckpointStore(tmp_path/'store',binding)
    with pytest.raises(ValueError,match='binding'):store.publish(state,cfg)
    assert not (tmp_path/'store/latest.json').exists()


def test_recovery_store_refuses_a_second_writer(tmp_path,valid_training_state):
    from treeneuron.research.node_checkpoints import NodeCheckpointStore
    _,_,binding=valid_training_state
    first=NodeCheckpointStore(tmp_path/'store',binding)
    try:
        with pytest.raises(ValueError,match='active|locked'):
            NodeCheckpointStore(tmp_path/'store',binding,resume=True)
    finally:
        first.close()


def test_checkpoint_payload_retention_keeps_two_and_all_row_history(tmp_path, native_policy):
    from treeneuron.research.node_v2 import NodeV2Predictor
    root = tmp_path/'checkpoints'
    NodeV2Predictor().fit(request(tmp_path/'fit', 13), training_checkpoints=root)
    assert len(list(root.glob('checkpoint-*/arrays.npz'))) == 2
    assert len(list(root.glob('checkpoint-*/trace.json'))) == 7
    assert len(list(root.glob('checkpoint-*/manifest.json'))) == 8
