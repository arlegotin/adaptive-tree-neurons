"""Growth preserves selected training history and paired continuation streams."""
import copy
from dataclasses import replace
import json

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence, assert_state_equal, events
from test_checkpoint import assert_nested_equal
from treeneuron import train
from treeneuron.artifacts import create_run
from treeneuron.checkpoint import load_checkpoint, save_checkpoint
from treeneuron.growth.allocate import Allocation
from treeneuron.growth.types import GrowthManifest


@pytest.fixture
def parent(training_case):
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    assert state.update > 0  # Exercise actual old Adam history, not empty state.
    return state, data, config, paths


def allocation(empty=False):
    return Allocation(2, 0 if empty else 2,
        torch.tensor([] if empty else [0, 2], dtype=torch.int64),
        torch.empty((0, 3), dtype=torch.float64) if empty else torch.tensor(
            [[.1, -.2, .3], [.3, .1, -.2]], dtype=torch.float64),
        'keep' if empty else 'adam', 37, 'a'*64,
        torch.zeros(0 if empty else 2, dtype=torch.int64),
        {'shared_scored_units': 4, 'standalone_contrast_units_required': 0})


def clone(state):
    return train.unpack_state(train.pack_state(state), expected_hashes=state.identity_hashes, device='cpu')


def ordinary_step(state, data):
    ids = state.order[state.cursor:state.cursor+4]
    x = torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float32)[ids]
    y = torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float32)[ids]
    state.optimizer.zero_grad(set_to_none=True)
    (state.model(x)-y).square().mean().backward()
    gradients = {n: p.grad.clone() for n, p in state.model.named_parameters()}
    state.optimizer.step()
    state.scheduler.step()
    return gradients


def test_growth_keeps_old_moments_and_predictions(parent):
    from treeneuron.growth.insert import insert_growth
    state, data, _, _ = parent
    prediction = state.predict(data.G.X)
    old = dict(state.model.named_parameters())
    moments = {n: copy.deepcopy(state.optimizer.state[p]) for n, p in old.items()}
    live_moments = {n: dict(state.optimizer.state[p]) for n, p in old.items()}
    group = state.optimizer.param_groups[0]
    options = {k: copy.deepcopy(v) for k, v in group.items() if k != 'params'}
    saved = train.pack_state(state)
    rng = train.capture_rng()
    manifest = insert_growth(state, allocation())
    np.testing.assert_array_equal(state.predict(data.G.X), prediction)
    for n, p in old.items():
        assert dict(state.model.named_parameters())[n] is p
        assert_nested_equal(state.optimizer.state[p], moments[n])
        for key in live_moments[n]:
            assert state.optimizer.state[p][key] is live_moments[n][key]
    assert state.optimizer.param_groups[0] is group
    assert {k: v for k, v in group.items() if k != 'params'} == options
    assert sum(p.numel() for p in state.model.parameters()) == 29
    assert manifest.old_parameter_count == 12 and manifest.added_parameter_count == 12
    assert state.scheduler.position == saved['metadata']['update']
    assert state.scheduler.base_lrs == [.02, .02]
    assert state.optimizer.param_groups[1]['lr'] == group['lr']
    assert_nested_equal(train.capture_rng(), rng)
    for key in ('order', 'sampler_generator', 'rng_states'):
        assert_nested_equal(train.pack_state(state)['tensors'][key], saved['tensors'][key])
    assert state.stage == saved['metadata']['stage']
    assert state.identity_hashes == saved['metadata']['identity_hashes']
    assert GrowthManifest.from_dict(json.loads(json.dumps(state.growth_metadata['manifest']))) == manifest
    assert state.growth_metadata['allocation']['contrasts'] == allocation().contrasts.tolist()
    assert state.growth_metadata['inserted_contrasts'] == allocation().contrasts.float().tolist()


def test_new_gate_step_starts_at_zero_and_old_update_matches_manual_reference(parent):
    from treeneuron.growth.insert import insert_growth
    state, data, _, _ = parent
    reference = clone(state)
    old_steps = {n: s['step'].item() for n, p in state.model.named_parameters() if (s := state.optimizer.state[p])}
    reference.model.layer.grow(allocation().ids, allocation().contrasts.float())
    added = [(n, p) for n, p in reference.model.named_parameters() if n not in old_steps]
    options = {k: v for k, v in reference.optimizer.param_groups[0].items() if k not in ('params', 'param_names')}
    reference.optimizer.add_param_group({'params': [p for _, p in added], 'param_names': [n for n, _ in added], **options})
    reference.scheduler.base_lrs.append(.02)
    insert_growth(state, allocation())
    for n, p in state.model.named_parameters():
        if n not in old_steps:
            assert state.optimizer.state[p]['step'].item() == 0
            assert torch.count_nonzero(state.optimizer.state[p]['exp_avg']) == 0
            assert torch.count_nonzero(state.optimizer.state[p]['exp_avg_sq']) == 0
    gradients = ordinary_step(state, data)
    ordinary_step(reference, data)
    assert torch.count_nonzero(gradients['layer.contrast.weight']) == 0
    assert torch.count_nonzero(gradients['layer.gate.weight']) > 0
    for n, p in state.model.named_parameters():
        assert state.optimizer.state[p]['step'].item() == old_steps.get(n, 0) + 1
        torch.testing.assert_close(p, dict(reference.model.named_parameters())[n], rtol=0, atol=0)
    assert_nested_equal(state.optimizer.state_dict(), reference.optimizer.state_dict())
    assert state.scheduler.state_dict() == reference.scheduler.state_dict()


def test_empty_allocation_preserves_run(parent):
    from treeneuron.growth.insert import insert_growth
    state, data, config, paths = parent
    reference = clone(state)
    before = train.pack_state(state)
    manifest = insert_growth(state, allocation(empty=True))
    assert manifest.added_parameter_count == 0
    assert state.model.layer.growth_applied and state.model.layer.gate is None
    assert len(state.optimizer.param_groups) == 1
    assert_nested_equal(state.optimizer.state_dict(), before['tensors']['optimizer'])
    policy = train.TrainingPolicy.from_config(config, continuation=True)
    a = train.continue_fit(state, data, policy, create_run(paths.directory.parent, config.to_dict()))
    b = train.continue_fit(reference, data, policy, create_run(paths.directory.parent, config.to_dict()))
    assert_state_equal(a, b)
    assert a.model.layer.growth_applied


@pytest.mark.parametrize('boundary', ['before', 'after', 'during'])
def test_resume_through_growth_matches_uninterrupted(parent, monkeypatch, boundary):
    from treeneuron.growth.insert import insert_growth
    state, data, config, paths = parent
    recovered = clone(state)
    policy = train.TrainingPolicy.from_config(config, continuation=True)
    insert_growth(state, allocation())
    uninterrupted_paths = create_run(paths.directory.parent, config.to_dict())
    expected = train.continue_fit(state, data, policy, uninterrupted_paths)
    if boundary == 'before':
        pointer = paths.directory / 'before.json'
        save_checkpoint(recovered, pointer)
        recovered = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    insert_growth(recovered, allocation())
    if boundary == 'after':
        pointer = paths.directory / 'after.json'
        save_checkpoint(recovered, pointer)
        recovered = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    if boundary == 'during':
        append = train.append_event
        def interrupt(paths, event):
            append(paths, event)
            if event['type'] == 'checkpoint' and event['update'] == recovered.update:
                # Interrupt after the first actual continuation update.
                if recovered.stage['updates_completed'] == 1: raise KeyboardInterrupt()
        attempt = create_run(paths.directory.parent, config.to_dict())
        monkeypatch.setattr(train, 'append_event', interrupt)
        with pytest.raises(KeyboardInterrupt): train.continue_fit(recovered, data, policy, attempt)
        monkeypatch.setattr(train, 'append_event', append)
        recovered = load_checkpoint(attempt.directory / 'checkpoints/last.json', expected_hashes=state.identity_hashes, device='cpu')
    retry = create_run(paths.directory.parent, config.to_dict())
    actual = train.continue_fit(recovered, data, policy, retry)
    assert_state_equal(actual, expected)
    assert_nested_equal(actual.growth_metadata, expected.growth_metadata)
    np.testing.assert_array_equal(actual.next_row_ids(data.F), expected.next_row_ids(data.F))
    assert actual.model.layer.growth_applied
    assert [e['row_ids'] for e in actual.history if e['type'] == 'update'][-6:] == [e['row_ids'] for e in expected.history if e['type'] == 'update'][-6:]


@pytest.mark.parametrize('defect', ['count', 'ids', 'dtype', 'shape', 'nan', 'overflow', 'zero', 'hash', 'policy', 'seed', 'attempts', 'cost'])
def test_bad_allocation_rejected_without_mutation(parent, defect):
    from treeneuron.growth.insert import insert_growth
    state, _, _, _ = parent
    changes = {
        'count': {'effective_count': 3}, 'ids': {'ids': torch.tensor([2, 0])},
        'dtype': {'contrasts': allocation().contrasts.float()}, 'shape': {'contrasts': torch.ones(2, 2, dtype=torch.float64)},
        'nan': {'contrasts': torch.full((2, 3), float('nan'), dtype=torch.float64)},
        'overflow': {'contrasts': torch.full((2, 3), 1e300, dtype=torch.float64)},
        'zero': {'contrasts': torch.zeros(2, 3, dtype=torch.float64)},
        'hash': {'table_hash': 'bad'}, 'policy': {'policy': 'bad'}, 'seed': {'seed': -1},
        'attempts': {'direction_attempts': torch.tensor([1])}, 'cost': {'cost': {'shared_scored_units': -1}}}
    before, rng = train.pack_state(state), train.capture_rng()
    with pytest.raises((ValueError, TypeError)): insert_growth(state, replace(allocation(), **changes[defect]))
    assert_nested_equal(train.pack_state(state), before)
    assert_nested_equal(train.capture_rng(), rng)


@pytest.mark.parametrize('defect', ['unmapped', 'duplicate', 'recipe', 'active', 'unselected'])
def test_invalid_parent_rejected_before_growth(parent, defect):
    from treeneuron.growth.insert import insert_growth
    state, _, _, _ = parent
    if defect == 'unmapped': state.optimizer.param_groups[0]['param_names'][0] = 'unknown'
    if defect == 'duplicate': state.optimizer.param_groups[0]['param_names'][0] = 'head.bias'
    if defect == 'recipe': state.optimizer.param_groups[0]['eps'] = 1e-4
    if defect == 'active': state.stage['complete'] = False
    if defect == 'unselected': state.best_snapshot = train.pack_state(state)
    tensors = copy.deepcopy(state.model.state_dict())
    opt = copy.deepcopy(state.optimizer.state_dict())
    rng = train.capture_rng()
    with pytest.raises(ValueError): insert_growth(state, allocation())
    assert not state.model.layer.growth_applied
    assert_nested_equal(state.model.state_dict(), tensors)
    assert_nested_equal(state.optimizer.state_dict(), opt)
    assert_nested_equal(train.capture_rng(), rng)


@pytest.mark.parametrize('empty', [False, True])
def test_second_insertion_and_duplicate_migration_rejected(parent, empty):
    from treeneuron.growth.insert import insert_growth, migrate_optimizer
    state, _, _, _ = parent
    manifest = insert_growth(state, allocation(empty))
    before = train.pack_state(state)
    with pytest.raises((ValueError, RuntimeError)): insert_growth(state, allocation(empty))
    with pytest.raises((ValueError, RuntimeError)): migrate_optimizer(state, manifest)
    assert_nested_equal(train.pack_state(state), before)


@pytest.mark.parametrize('defect', ['unmapped', 'duplicate', 'shape', 'slice', 'count'])
def test_manifest_rejected_before_optimizer_mutation(parent, defect):
    from treeneuron.growth.insert import migrate_optimizer
    state, _, _, _ = parent
    manifest = state.model.layer.grow(allocation().ids, allocation().contrasts.float())
    maps = manifest.parameter_mappings
    changes = {
        'unmapped': {'parameter_mappings': maps[:1]},
        'duplicate': {'parameter_mappings': (maps[0], maps[0])},
        'shape': {'parameter_mappings': (replace(maps[0], old_shape=(1, 8)), maps[1])},
        'slice': {'parameter_mappings': (replace(maps[0], new_slice=((0, 1), (0, 2))), maps[1])},
        'count': {'added_parameter_count': 999}}
    before = copy.deepcopy(state.optimizer.state_dict())
    schedule = state.scheduler.state_dict()
    with pytest.raises(ValueError): migrate_optimizer(state, replace(manifest, **changes[defect]))
    assert_nested_equal(state.optimizer.state_dict(), before)
    assert state.scheduler.state_dict() == schedule


def test_insertion_rolls_back_if_migration_fails(parent, monkeypatch):
    from treeneuron.growth import insert
    state, _, _, _ = parent
    before, rng = train.pack_state(state), train.capture_rng()
    migrate = insert.migrate_optimizer
    def fail(*args):
        migrate(*args)
        raise RuntimeError('injected after migration')
    monkeypatch.setattr(insert, 'migrate_optimizer', fail)
    with pytest.raises(RuntimeError, match='injected'): insert.insert_growth(state, allocation())
    assert_nested_equal(train.pack_state(state), before)
    assert_nested_equal(train.capture_rng(), rng)


@pytest.mark.parametrize('policy', ['adam', 'random', 'sgd', 'gaussian', 'keep'])
def test_real_allocator_keeps_training_stream_and_selected_contrasts(parent, policy):
    from treeneuron.growth.allocate import allocate
    from treeneuron.growth.insert import insert_growth
    from treeneuron.growth.score import score_units
    state, data, _, paths = parent
    score = score_units(copy.deepcopy(state.model).double().eval(),
        torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float64),
        torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float64),
        epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=4, unit_block_size=2)
    rng = train.capture_rng()
    local_rng = copy.deepcopy(state.rng_states)
    rows = state.next_row_ids(data.F)
    choice = allocate(score, 2, policy=policy, seed=137)
    assert choice.effective_count == (0 if policy == 'keep' else 2)
    insert_growth(state, choice)
    assert_nested_equal(train.capture_rng(), rng)
    assert_nested_equal(state.rng_states, local_rng)
    np.testing.assert_array_equal(state.next_row_ids(data.F), rows)
    if choice.effective_count:
        actual = torch.cat((state.model.layer.contrast.weight, state.model.layer.contrast.bias[:, None]), dim=1)
        torch.testing.assert_close(actual, choice.contrasts.float(), rtol=0, atol=0)
    pointer = paths.directory / 'allocation.json'
    save_checkpoint(state, pointer)
    recovered = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert_nested_equal(recovered.growth_metadata, state.growth_metadata)
    assert_state_equal(recovered, state)


def test_growth_from_selected_update_zero_preserves_lazy_old_state(training_case):
    from treeneuron.data.transforms import FittedTransform
    from treeneuron.data.types import DevelopmentData, ObservationSet
    from treeneuron.growth.insert import insert_growth
    data, config, paths = training_case
    torch.manual_seed(config.seeds.optimizer)
    initial = train.SelectiveRegressor(2, 4).eval()
    transform = FittedTransform.fit(data.F.X, data.F.y)
    with torch.no_grad():
        y = transform.inverse_y(initial(torch.tensor(transform.transform_X(data.G.X), dtype=torch.float32)).numpy())
    selected_data = DevelopmentData(data.F, ObservationSet(data.G.X, y, data.G.row_id, data.G.group_id), data.V, data.teacher_id)
    state = train.fit_parent(selected_data, config, paths)
    assert state.update == 0 and not state.optimizer.state
    old = dict(state.model.named_parameters())
    insert_growth(state, allocation())
    assert all(p not in state.optimizer.state for p in old.values())
    ordinary_step(state, data)
    assert all(saved['step'].item() == 1 for saved in state.optimizer.state.values())


def test_initial_coordinate_counts_and_checkpoint_reconstruction(training_case):
    from treeneuron.config import resolve_experiment
    from treeneuron.data.types import DevelopmentData, ObservationSet
    from treeneuron.growth.insert import insert_growth
    data, config, paths = training_case
    value = config.to_dict()
    value['model'].update(d=16, h=64, tau=1.)
    config = resolve_experiment(value)
    def expand(rows):
        return ObservationSet(np.pad(rows.X, ((0, 0), (0, 14))), rows.y, rows.row_id, rows.group_id)
    data = DevelopmentData(expand(data.F), expand(data.G), expand(data.V), data.teacher_id)
    paths = create_run(paths.directory.parent, config.to_dict())
    state = train.fit_parent(data, config, paths)
    before = state.predict(data.G.X)
    assert sum(p.numel() for p in state.model.parameters()) == 1153
    choice = Allocation(16, 16, torch.arange(16), torch.full((16, 17), .05, dtype=torch.float64),
        'adam', 37, 'a'*64, torch.zeros(16, dtype=torch.int64),
        {'shared_scored_units': 64, 'standalone_contrast_units_required': 0})
    manifest = insert_growth(state, choice)
    assert manifest.added_parameter_count == 544
    assert sum(p.numel() for p in state.model.parameters()) == 1697
    np.testing.assert_array_equal(state.predict(data.G.X), before)
    pointer = paths.directory / 'coordinate.json'
    save_checkpoint(state, pointer)
    recovered = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(recovered, state)
