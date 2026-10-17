"""Checkpoint row histories commit to full, separately retained stage events."""
import copy
import hashlib
import json

import numpy as np
import pytest

from test_train import cpu_training_evidence, training_case, events, assert_state_equal
from treeneuron import train
from treeneuron.artifacts import create_run
from treeneuron.checkpoint import load_checkpoint
from treeneuron.config import resolve_experiment
from treeneuron.data.types import DevelopmentData, ObservationSet


def assert_committed(history, raw):
    assert len(history) == len(raw)
    for saved, event in zip(history, raw):
        if event['type'] != 'update':
            assert saved == event
            continue
        # Independently reconstruct the existing dtype/shape/ordered-byte hash.
        ids = np.asarray(event['row_ids'], dtype=np.int64)
        digest = hashlib.sha256(b'int64' + str(ids.shape).encode() + ids.tobytes()).hexdigest()
        assert saved == {
            **{k: v for k, v in event.items() if k != 'row_ids'},
            'row_ids_commitment': {'schema_version': 1, 'encoding': 'ordered_int64_array_hash',
                                   'count': len(ids), 'sha256': digest}}
        assert event['rows'] == len(ids)


def test_minibatch_history_commits_without_changing_raw_events_or_checkpoint_resume(training_case):
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    raw = [e for e in events(paths) if e['type'] in {'update', 'evaluation', 'checkpoint', 'selection'}]
    assert_committed(state.history, raw)
    for name in ('last', 'best'):
        loaded = load_checkpoint(paths.directory / f'checkpoints/{name}.json',
                                 expected_hashes=state.identity_hashes, device='cpu')
        assert loaded.history == state.history
    restored = load_checkpoint(paths.directory / 'checkpoints/best.json',
                               expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(state, restored)
    assert state.next_row_ids(data.F).tolist() == restored.next_row_ids(data.F).tolist()
    policy = train.TrainingPolicy.from_config(config, continuation=True)
    a = create_run(paths.directory.parent, config.to_dict())
    b = create_run(paths.directory.parent, config.to_dict())
    direct = train.continue_fit(state, data, policy, a)
    resumed = train.continue_fit(restored, data, policy, b)
    assert_state_equal(direct, resumed)
    assert_committed(resumed.history[len(raw):],
                     [e for e in events(b) if e['type'] in {'update', 'evaluation', 'checkpoint', 'selection'}])


def test_history_bytes_scale_with_updates_instead_of_past_batch_rows(training_case):
    _, config, original = training_case
    rng = np.random.default_rng(83)
    def rows(n, start):
        x = rng.normal(size=(n, 2))
        ids = np.arange(start, start+n, dtype=np.int64)
        return ObservationSet(x, x[:, 0] - x[:, 1], ids, ids)
    data = DevelopmentData(rows(257, 100000000), rows(7, 200000000), rows(5, 300000000), 'storage')
    sizes = []
    for batch in (4, 128):
        raw_config = config.to_dict()
        raw_config['training']['batch_size'] = batch
        raw_config['data']['n_fit'] = 257
        config = resolve_experiment(raw_config)
        paths = create_run(original.directory.parent, config.to_dict())
        state = train.fit_parent(data, config, paths)
        updates = [e for e in state.history if e['type'] == 'update']
        raw_updates = [e for e in events(paths) if e['type'] == 'update']
        assert_committed(updates, raw_updates)
        sizes.append((len(json.dumps(updates)), len(json.dumps(raw_updates))))
    assert abs(sizes[0][0] - sizes[1][0]) < 100
    # Fixed event overhead and epoch-tail batches limit the raw JSON ratio.
    assert sizes[1][1] > 4 * sizes[0][1]


def test_growth_keeps_compact_parent_prefix_and_full_new_stage_events(training_case):
    from test_optimizer_growth import allocation
    from treeneuron.growth.insert import insert_growth
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    prefix = copy.deepcopy(parent.history)
    assert all('row_ids_commitment' in e for e in prefix if e['type'] == 'update')
    insert_growth(parent, allocation())
    child_paths = create_run(paths.directory.parent, config.to_dict())
    child = train.continue_fit(parent, data, train.TrainingPolicy.from_config(config, continuation=True), child_paths)
    assert child.history[:len(prefix)] == prefix
    assert_committed(child.history[len(prefix):],
                     [e for e in events(child_paths) if e['type'] in {'update', 'evaluation', 'checkpoint', 'selection'}])
