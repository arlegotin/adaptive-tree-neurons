"""Malformed recovery state must never replace a working checkpoint pointer."""
import copy
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from test_checkpoint import assert_nested_equal


@pytest.fixture
def snapshot():
    """An untrained local software state, with no backend qualification claim."""
    from treeneuron import train
    from treeneuron.config import resolve_experiment
    from treeneuron.data.transforms import FittedTransform
    from treeneuron.data.types import DevelopmentData, ObservationSet
    config = resolve_experiment({'study_id': 'snapshot-boundary', 'model': {'d': 2, 'h': 4},
        'data': {'n_fit': 13, 'n_growth': 7, 'n_validation': 5},
        'training': {'batch_size': 4, 'base_updates': 6, 'continuation_updates': 6,
                     'validation_interval': 2, 'patience': 20}, 'backend': {'cpu_threads': 1}})
    rng = np.random.default_rng(83)
    def rows(n, start):
        x = rng.normal(size=(n, 2)); ids = np.arange(start, start+n, dtype=np.int64)
        return ObservationSet(x, 7+3*x[:, 0]-x[:, 1], ids, ids)
    data = DevelopmentData(rows(13, 0), rows(7, 100), rows(5, 200), 'local-fixture')
    transform = FittedTransform.fit(data.F.X, data.F.y)
    model = train.SelectiveRegressor(2, 4).float().eval()
    optimizer = train._optimizer(model, config)
    schedule = train.UpdateSchedule(optimizer, kind=config.training.scheduler, horizon=12,
        min_lr=config.training.min_learning_rate, base_lrs=[config.training.learning_rate])
    policy = train.TrainingPolicy.from_config(config)
    stage = dict(kind='base', start_update=0, updates_completed=0, bad_evaluations=0,
                 best_mse=None, policy=policy.__dict__, complete=False)
    state = train.TrainingState(model, optimizer, schedule, 0, 0, torch.arange(13), 0,
        torch.Generator().manual_seed(3), train.capture_rng(), transform,
        train._identities(data, config, transform), config,
        dict(device='cpu', dtype='float32', fallback=False, cpu_threads=1),
        float(np.var(data.F.y)), train._fg_hashes(data), stage)
    return train.pack_state(state)


def damage(snapshot, fault):
    tensors = snapshot['tensors']
    if fault == 'order_dtype': tensors['order'] = tensors['order'].double()
    elif fault == 'order_duplicates': tensors['order'].zero_()
    elif fault == 'generator': tensors['sampler_generator'] = torch.zeros(1, dtype=torch.uint8)
    else: tensors['optimizer']['param_groups'][0]['lr'] *= 2


@pytest.mark.parametrize('fault', ['order_dtype', 'order_duplicates', 'generator', 'optimizer_options'])
@pytest.mark.parametrize('location', ['current', 'best'])
def test_invalid_snapshot_preserves_existing_pointer_and_rng(snapshot, tmp_path, fault, location):
    from treeneuron import checkpoint, train
    current = copy.deepcopy(snapshot); best = copy.deepcopy(snapshot)
    pointer = tmp_path/'last.json'
    checkpoint._publish_snapshot(current, best, [], pointer)
    previous = pointer.read_bytes(); files = sorted(p.name for p in tmp_path.iterdir())
    damage(current if location == 'current' else best, fault)
    before = train.capture_rng()
    with pytest.raises(ValueError): checkpoint._publish_snapshot(current, best, [], pointer)
    assert pointer.read_bytes() == previous
    assert sorted(p.name for p in tmp_path.iterdir()) == files
    assert_nested_equal(train.capture_rng(), before)
    loaded = checkpoint.load_checkpoint(pointer, expected_hashes=snapshot['metadata']['identity_hashes'], device='cpu')
    torch.testing.assert_close(loaded.order, torch.arange(13), rtol=0, atol=0)


@pytest.mark.parametrize('fault', ['order_dtype', 'order_duplicates', 'generator', 'optimizer_options'])
@pytest.mark.parametrize('location', ['current', 'best'])
def test_loader_rejects_bad_snapshot_before_optimizer_or_rng_restoration(snapshot, tmp_path, monkeypatch, fault, location):
    from treeneuron import checkpoint, train
    pointer = tmp_path/'last.json'
    checkpoint._publish_snapshot(snapshot, copy.deepcopy(snapshot), [], pointer)
    top = json.loads(pointer.read_bytes()); metadata = tmp_path/top['metadata']
    descriptor = json.loads(metadata.read_bytes()); payload = tmp_path/descriptor['payload']
    tensors = torch.load(payload, weights_only=True)
    target = dict(tensors=tensors['state' if location == 'current' else 'best_state'])
    damage(target, fault)
    stream = io.BytesIO(); torch.save(tensors, stream); payload.write_bytes(stream.getvalue())
    descriptor['sha256'] = hashlib.sha256(payload.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(descriptor)); top['sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(top))
    def forbidden(*args, **kwargs): raise AssertionError('Corruption reached optimizer reconstruction')
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', forbidden)
    before = train.capture_rng()
    with pytest.raises(ValueError):
        checkpoint.load_checkpoint(pointer, expected_hashes=snapshot['metadata']['identity_hashes'], device='cpu')
    assert_nested_equal(train.capture_rng(), before)


def test_complete_snapshot_roundtrip_preserves_permutation_generator_and_options(snapshot, tmp_path):
    from treeneuron import checkpoint, train
    pointer = tmp_path/'last.json'
    before = train.capture_rng()
    checkpoint._publish_snapshot(snapshot, copy.deepcopy(snapshot), [], pointer)
    assert_nested_equal(train.capture_rng(), before)
    loaded = checkpoint.load_checkpoint(pointer, expected_hashes=snapshot['metadata']['identity_hashes'], device='cpu')
    actual = train.pack_state(loaded)
    from treeneuron.config import canonical_json
    assert canonical_json(actual['metadata']) == canonical_json(snapshot['metadata'])
    assert_nested_equal(actual['tensors'], snapshot['tensors'])
