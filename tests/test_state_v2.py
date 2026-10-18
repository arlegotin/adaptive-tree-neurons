"""Reject semantically corrupt state before loading or publishing tensors."""
import copy

import pytest
import torch

from test_train import training_case, cpu_training_evidence, assert_state_equal


def corrupt(snapshot, kind):
    tensors = snapshot['tensors']
    first = next(iter(tensors['optimizer']['state'].values()))
    if kind == 'incumbent_age':
        first['step'].zero_()
    elif kind == 'negative_second_moment':
        first['exp_avg_sq'].flatten()[0] = -1
    elif kind == 'model_dtype':
        tensors['model']['layer.base.weight'] = tensors['model']['layer.base.weight'].double()
    elif kind == 'moment_dtype':
        first['exp_avg'] = first['exp_avg'].double()
    elif kind == 'model_shape':
        tensors['model']['layer.base.weight'] = tensors['model']['layer.base.weight'][:1]


@pytest.mark.parametrize('kind', ['incumbent_age', 'negative_second_moment',
                                  'model_dtype', 'moment_dtype', 'model_shape'])
@pytest.mark.parametrize('location', ['current', 'best'])
def test_corrupt_state_rejected_before_pointer_publication(training_case, kind, location):
    from treeneuron.train import fit_parent, pack_state
    from treeneuron.checkpoint import _publish_snapshot
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    assert state.update > 0
    current = pack_state(state)
    best = copy.deepcopy(current)
    corrupt(current if location == 'current' else best, kind)
    pointer = paths.directory / 'corrupt.json'
    with pytest.raises(ValueError, match='Adam|tensor'):
        _publish_snapshot(current, best, [], pointer)
    assert not pointer.exists()


@pytest.mark.parametrize('kind', ['incumbent_age', 'negative_second_moment',
                                  'model_dtype', 'moment_dtype'])
def test_in_memory_unpack_rejects_corruption(training_case, kind):
    from treeneuron.train import fit_parent, pack_state, unpack_state
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    snapshot = pack_state(state)
    corrupt(snapshot, kind)
    with pytest.raises(ValueError, match='Adam|tensor'):
        unpack_state(snapshot, expected_hashes=state.identity_hashes, device='cpu')


def test_live_negative_moment_cannot_be_packed(training_case):
    from treeneuron.train import fit_parent, pack_state
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    next(iter(state.optimizer.state.values()))['exp_avg_sq'].flatten()[0] = -1
    with pytest.raises(ValueError, match='second Adam moment'):
        pack_state(state)


def test_legitimate_snapshot_roundtrip_is_exact(training_case):
    from treeneuron.train import fit_parent, pack_state, unpack_state
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    assert state.cursor not in (0, len(state.order))
    loaded = unpack_state(pack_state(state), expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(state, loaded)


def test_missing_birth_contract_rejected(training_case):
    from treeneuron.train import fit_parent, pack_state, unpack_state
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    snapshot = pack_state(state)
    snapshot['metadata'].pop('parameter_birth_updates', None)
    with pytest.raises(ValueError, match='metadata|birth'):
        unpack_state(snapshot, expected_hashes=state.identity_hashes, device='cpu')


def test_float64_process_default_does_not_change_checkpoint_recipe(training_case):
    from treeneuron.train import fit_parent, pack_state
    data, config, paths = training_case
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        state = fit_parent(data, config, paths)
    finally:
        torch.set_default_dtype(before)
    snapshot = pack_state(state)
    assert all(record['step'].dtype == torch.float32
               for record in snapshot['tensors']['optimizer']['state'].values())
