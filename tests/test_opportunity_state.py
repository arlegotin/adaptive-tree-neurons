"""Early intervention is an explicit current-state boundary, never fake G-best."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence
from test_checkpoint import assert_nested_equal


def test_actual_snapshot_boundary_preserves_state_and_requires_research_insertion(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.growth.allocate import allocate
    from treeneuron.growth.score import score_units
    from treeneuron.growth.insert import insert_growth
    from treeneuron.research.boundaries import current_boundary, insert_at_current_boundary
    from treeneuron.research.roles import fitting_view
    data, config, paths = training_case
    snapshots = {}
    train.fit_parent(data, config, paths, snapshot_updates=(3,),
                     on_snapshot=lambda label, snapshot: snapshots.setdefault(label, snapshot))
    original = copy.deepcopy(snapshots['current'])
    view = fitting_view(data)
    state = current_boundary(original, view, requested_update=3)
    assert state.update == 3 and state.stage['kind'] == 'research_boundary'
    assert state.stage['complete'] and state.best_snapshot is None
    assert state.growth_metadata['research_boundary']['selection_policy'] == 'actual_update'
    packed = train.pack_state(state)
    assert_nested_equal(packed['tensors'], original['tensors'])
    assert packed['metadata']['scheduler'] == original['metadata']['scheduler']
    for name, tensor in original['tensors']['model'].items():
        torch.testing.assert_close(packed['tensors']['model'][name], tensor, atol=0, rtol=0)
    assert state.scheduler.position == 3 and state.cursor == original['metadata']['cursor']
    torch.testing.assert_close(state.order, original['tensors']['order'], atol=0, rtol=0)
    assert state.parameter_birth_updates == original['metadata']['parameter_birth_updates']
    with pytest.raises(ValueError):
        current_boundary(original, view, requested_update=4)
    model = copy.deepcopy(state.model).double().eval()
    table = score_units(model, torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float64),
        torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float64),
        epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=4, unit_block_size=4)
    allocation = allocate(table, 2, policy='adam', seed=7)
    prediction = state.predict(data.G.X)
    with pytest.raises(ValueError, match='research'):
        insert_growth(state, allocation)
    insert_at_current_boundary(state, allocation)
    np.testing.assert_array_equal(state.predict(data.G.X), prediction)
    assert state.growth_metadata['research_boundary']['actual_update'] == 3
    result = train.continue_fit(state, view, train.TrainingPolicy.from_config(config, continuation=True),
                                create_run(tmp_path/'continuation', config.to_dict()))
    assert result.stage['start_update'] == 3
    assert result.growth_metadata['research_boundary']['actual_update'] == 3


def test_warm_ordinary_bracket_uses_same_declared_actual_boundary(training_case):
    from treeneuron import train
    from treeneuron.baselines.causal import widen_parent
    from treeneuron.research.boundaries import current_boundary, widen_at_current_boundary
    from treeneuron.research.roles import fitting_view
    data, config, paths = training_case
    snapshots = {}
    train.fit_parent(data, config, paths, snapshot_updates=(3,),
                     on_snapshot=lambda label, snapshot: snapshots.setdefault(label, snapshot))
    state = current_boundary(snapshots['current'], fitting_view(data), requested_update=3)
    before = state.predict(data.G.X)
    with pytest.raises(ValueError, match='research'):
        widen_parent(state, 2, 7)
    widen_at_current_boundary(state, 2, 7)
    np.testing.assert_array_equal(state.predict(data.G.X), before)
    assert state.scheduler.position == state.update == 3
    assert state.growth_metadata['research_boundary']['selection_policy'] == 'actual_update'
    assert state.parameter_birth_updates['layer.added.weight'] == 3


def test_fixed_budget_disables_only_patience_and_keeps_zero_selection(training_case, tmp_path, monkeypatch):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.config import resolve_experiment
    data, config, _ = training_case
    raw = config.to_dict()
    raw['training'].update(base_updates=6, continuation_updates=6, validation_interval=2, patience=1)
    config = resolve_experiment(raw)
    monkeypatch.setattr(train, 'regression_metrics', lambda *a, **k: {'mse': 1.})
    parent = train.fit_parent(data, config, create_run(tmp_path/'parent', config.to_dict()))
    assert parent.update == 0 and parent.stage['updates_completed'] == 2
    snapshots = []
    fixed_parent = train.fit_parent(data, config, create_run(tmp_path/'fixed-parent', config.to_dict()),
        stop_by_patience=False, snapshot_updates=(3,), on_snapshot=lambda label, snapshot: snapshots.append((label, snapshot['metadata']['update'])))
    assert fixed_parent.stage['updates_completed'] == 6 and ('current', 3) in snapshots
    policy = replace(train.TrainingPolicy.from_config(config, continuation=True), stop_by_patience=False)
    result = train.continue_fit(parent, data, policy, create_run(tmp_path/'fixed', config.to_dict()))
    assert result.update == 0 and result.stage['updates_completed'] == 6
    assert result.stage['policy']['stop_by_patience'] is False
    with pytest.raises(ValueError):
        replace(policy, stop_by_patience=0)


def test_interrupted_fixed_budget_rejects_capture_plan_drift(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    policy = replace(train.TrainingPolicy.from_config(config, continuation=True), stop_by_patience=False)
    requested = tuple(parent.update+i for i in (2, 4, 6))
    interrupted_paths = create_run(tmp_path/'interrupted', config.to_dict())
    def interrupt(label, snapshot):
        if label == 'current':
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        train.continue_fit(parent, data, policy, interrupted_paths,
                           snapshot_updates=requested, on_snapshot=interrupt)
    pointer = interrupted_paths.directory/'checkpoints/last.json'
    restore = lambda: load_checkpoint(pointer, expected_hashes=parent.identity_hashes, device='cpu')
    for updates in ((), requested[:-1]):
        with pytest.raises(ValueError, match='policy|capture'):
            train.continue_fit(restore(), data, policy,
                create_run(tmp_path/f'drift-{len(updates)}', config.to_dict()),
                snapshot_updates=updates, on_snapshot=lambda *args: None)
    seen = []
    resumed = train.continue_fit(restore(), data, policy,
        create_run(tmp_path/'resumed', config.to_dict()), snapshot_updates=requested,
        on_snapshot=lambda label, snapshot: seen.append((label, snapshot['metadata']['update'])))
    assert resumed.stage['updates_completed'] == policy.max_updates
    assert ('current', requested[-1]) in seen


@pytest.mark.parametrize('operation', ['insert', 'widen'])
def test_failed_research_migration_rolls_back_complete_boundary(training_case, monkeypatch, operation):
    from treeneuron import train
    from treeneuron.research.boundaries import current_boundary, insert_at_current_boundary, widen_at_current_boundary
    from treeneuron.research.roles import fitting_view
    from treeneuron.growth.allocate import allocate
    from treeneuron.growth.score import score_units
    import treeneuron.growth.insert as insertion
    import treeneuron.baselines.causal as causal
    data, config, paths = training_case
    snapshots = {}
    train.fit_parent(data, config, paths, snapshot_updates=(3,),
        on_snapshot=lambda label, snapshot: snapshots.setdefault(label, snapshot))
    state = current_boundary(snapshots['current'], fitting_view(data), requested_update=3)
    if operation == 'insert':
        table = score_units(copy.deepcopy(state.model).double().eval(),
            torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float64),
            torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float64),
            epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=4, unit_block_size=4)
        allocation = allocate(table, 2, policy='adam', seed=7)
        migrate = insertion.migrate_optimizer
        def fail_after_migration(*args):
            migrate(*args)
            raise RuntimeError('injected migration failure')
        monkeypatch.setattr(insertion, 'migrate_optimizer', fail_after_migration)
        perform = lambda: insert_at_current_boundary(state, allocation)
    else:
        validate = causal.validate_tensor_state
        calls = []
        def fail_after_candidate(value):
            validate(value)
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('injected candidate publication failure')
        monkeypatch.setattr(causal, 'validate_tensor_state', fail_after_candidate)
        perform = lambda: widen_at_current_boundary(state, 2, 7)
    before, rng = train.pack_state(state), train.capture_rng()
    with pytest.raises(RuntimeError, match='injected'):
        perform()
    assert_nested_equal(train.pack_state(state), before)
    assert_nested_equal(train.capture_rng(), rng)
