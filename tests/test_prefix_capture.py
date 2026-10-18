"""Nested horizons retain the G-best full state of one unchanged trajectory."""
from dataclasses import replace
import copy

import numpy as np
import pytest

from test_train import training_case, cpu_training_evidence, events, assert_state_equal
from test_checkpoint import assert_nested_equal


def test_prefix_selection_uses_existing_G_grid_and_preserves_trajectory(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    saved = train.pack_state(parent)
    clone = lambda: train.unpack_state(saved, expected_hashes=parent.identity_hashes, device='cpu')
    policy = replace(train.TrainingPolicy.from_config(config, continuation=True), stop_by_patience=False)
    baseline = train.continue_fit(clone(), data, policy, create_run(tmp_path/'baseline', config.to_dict()))
    seen = {}
    def capture(prefix, snapshot):
        seen[prefix] = copy.deepcopy(snapshot)
        # Callback RNG use and mutation of its copy cannot alter training.
        __import__('torch').rand(3)
        snapshot['tensors']['model']['head.bias'].fill_(999)
    captured_paths = create_run(tmp_path/'captured', config.to_dict())
    result = train.continue_fit(clone(), data, policy, captured_paths,
        selection_prefixes=(0, 2, 4, 6), on_prefix=capture)
    assert_state_equal(result, baseline)
    assert tuple(seen) == (0, 2, 4, 6)
    evaluations = [e for e in events(captured_paths) if e['type'] == 'evaluation']
    assert [e['stage_update'] for e in evaluations] == [0, 2, 4, 6]
    for prefix, snapshot in seen.items():
        eligible = [e for e in evaluations if e['stage_update'] <= prefix]
        selected = min(eligible, key=lambda e: e['metrics']['mse'])
        assert snapshot['metadata']['update'] == selected['update']
        assert snapshot['metadata']['stage']['updates_completed'] == selected['stage_update']
        assert snapshot['metadata']['stage']['best_mse'] == selected['metrics']['mse']
    assert result.stage['policy']['selection_prefixes'] == [0, 2, 4, 6]


def test_prefix_plan_rejects_non_nested_or_stopping_treatments(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.artifacts import create_run
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    fixed = replace(train.TrainingPolicy.from_config(config, continuation=True), stop_by_patience=False)
    for prefixes in ((0, 1, 6), (0, 2, 2), (0, 8), (2, 0, 6)):
        with pytest.raises(ValueError):
            replace(fixed, selection_prefixes=prefixes)
    with pytest.raises(ValueError):
        replace(fixed, stop_by_patience=True, selection_prefixes=(0, 2, 6))
    attempt = create_run(tmp_path/'interrupt', config.to_dict())
    def interrupt(prefix, snapshot):
        if prefix == 2:
            raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        train.continue_fit(parent, data, fixed, attempt,
            selection_prefixes=(0, 2, 6), on_prefix=interrupt)
    restore = lambda: load_checkpoint(attempt.directory/'checkpoints/last.json',
        expected_hashes=parent.identity_hashes, device='cpu')
    for changed in ((), (0, 2, 4, 6)):
        with pytest.raises(ValueError, match='policy|capture'):
            train.continue_fit(restore(), data, fixed,
                create_run(tmp_path/f'drift-{len(changed)}', config.to_dict()),
                selection_prefixes=changed, on_prefix=lambda *a: None)
    seen = []
    restored = train.continue_fit(restore(), data, fixed,
        create_run(tmp_path/'resume', config.to_dict()), selection_prefixes=(0, 2, 6),
        on_prefix=lambda p, s: seen.append((p, s['metadata']['update'])))
    assert [p for p, _ in seen] == [2, 6]
    assert restored.stage['updates_completed'] == 6


@pytest.fixture
def prefix_snapshot(training_case):
    from treeneuron import train
    data, config, paths = training_case
    seen = {}
    train.fit_parent(data, config, paths, stop_by_patience=False,
        selection_prefixes=(0, 2, 6), on_prefix=lambda p, s: seen.setdefault(p, s))
    return seen[0]


def test_sink_rejects_a_late_parent_labeled_as_prefix_zero(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.research.snapshots import PrefixSnapshotSink
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    sink = PrefixSnapshotSink(tmp_path/'prefixes', (0, 2, 6), trajectory_id='case')
    with pytest.raises(ValueError, match='prefix|selection'):
        sink(0, train.pack_state(parent))


def test_prefix_sink_is_durable_idempotent_and_rejects_changed_boundary(prefix_snapshot, tmp_path):
    from treeneuron import train
    from treeneuron.research.snapshots import PrefixSnapshotSink
    snapshot = prefix_snapshot
    sink = PrefixSnapshotSink(tmp_path/'prefixes', (0, 2, 6), trajectory_id='fixed-case')
    sink(0, snapshot)
    resumed = PrefixSnapshotSink(tmp_path/'prefixes', (0, 2, 6), trajectory_id='fixed-case')
    resumed(0, snapshot)
    assert len(resumed.records) == 1
    changed = copy.deepcopy(snapshot)
    changed['tensors']['model']['head.bias'] += 1
    with pytest.raises(ValueError, match='changed'):
        resumed(0, changed)
    late = copy.deepcopy(snapshot)
    late['metadata']['stage']['start_update'] += 1
    late['metadata']['update'] += 1
    # Malformed or different-stage snapshots cannot join a valid prefix series.
    with pytest.raises(ValueError):
        resumed(2, late)
    wrong_dtype = copy.deepcopy(snapshot)
    wrong_dtype['tensors']['model']['head.bias'] = wrong_dtype['tensors']['model']['head.bias'].double()
    with pytest.raises(ValueError):
        resumed(2, wrong_dtype)
    with pytest.raises(ValueError):
        PrefixSnapshotSink(tmp_path/'prefixes', (0, 6), trajectory_id='fixed-case')
    report = resumed.finalize()
    assert report['unavailable'] == [2, 6]
    assert resumed.finalize() == report


@pytest.mark.parametrize('after_link', [False, True])
@pytest.mark.parametrize('same_sink', [False, True])
def test_prefix_record_interrupt_is_recoverable_after_atomic_pointer(prefix_snapshot, tmp_path, monkeypatch, after_link, same_sink):
    from treeneuron import train
    from treeneuron.research import snapshots
    snapshot = prefix_snapshot
    root = tmp_path/'prefixes'
    sink = snapshots.PrefixSnapshotSink(root, (0, 2, 6), trajectory_id='fixed')
    original_link = snapshots.os.link
    def interrupt(source, destination):
        if destination.name == 'prefix-0.record.json':
            if after_link:
                original_link(source, destination)
            raise KeyboardInterrupt
        return original_link(source, destination)
    monkeypatch.setattr(snapshots.os, 'link', interrupt)
    with pytest.raises(KeyboardInterrupt):
        sink(0, snapshot)
    monkeypatch.setattr(snapshots.os, 'link', original_link)
    recovered = sink if same_sink else snapshots.PrefixSnapshotSink(root, (0, 2, 6), trajectory_id='fixed')
    recovered(0, snapshot)
    assert len(recovered.records) == 1
    assert recovered.finalize()['unavailable'] == [2, 6]
