"""Durable pointer publication protects complete optimizer and RNG state."""
import json
import random

import numpy as np
import pytest
import torch

from test_train import training_case, cpu_training_evidence, assert_state_equal


def test_roundtrip_named_optimizer_rng_and_topology(training_case):
    from treeneuron.train import fit_parent, restore_rng
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoint.json'
    digest = save_checkpoint(state, pointer)
    restore_rng(state.rng_states)
    expected = (random.random(), np.random.random(), torch.rand(3))
    loaded = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)
    assert_state_equal(loaded, state)
    assert len(digest) == 64
    assert loaded.optimizer.param_groups[0]['param_names'] == list(dict(loaded.model.named_parameters()))


def test_failed_atomic_write_keeps_last_valid(training_case, monkeypatch):
    from treeneuron.train import fit_parent
    from treeneuron import checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'last.json'
    checkpoint.save_checkpoint(state, pointer)
    before = pointer.read_bytes()
    original = checkpoint.os.replace
    def fail(source, destination):
        if destination == pointer: raise OSError('injected before pointer rename')
        return original(source, destination)
    monkeypatch.setattr(checkpoint.os, 'replace', fail)
    with pytest.raises(OSError, match='injected'): checkpoint.save_checkpoint(state, pointer)
    assert pointer.read_bytes() == before
    restored = checkpoint.load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(restored, state)


@pytest.mark.parametrize('key', ['split', 'config', 'code', 'lock'])
def test_resume_rejects_identity_drift_before_optimizer_load(training_case, key, monkeypatch):
    from treeneuron.train import fit_parent
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoint.json'
    save_checkpoint(state, pointer)
    expected = {**state.identity_hashes, key: '0'*64}
    def forbidden(*a, **k): raise AssertionError('optimizer loaded before identity validation')
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', forbidden)
    with pytest.raises(ValueError, match='identity'): load_checkpoint(pointer, expected_hashes=expected, device='cpu')


def test_corrupted_checksum_rejected(training_case):
    from treeneuron.train import fit_parent
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoint.json'
    save_checkpoint(state, pointer)
    descriptor = json.loads((pointer.parent / json.loads(pointer.read_text())['metadata']).read_text())
    payload = pointer.parent / descriptor['payload']
    payload.write_bytes(payload.read_bytes() + b'corrupt')
    with pytest.raises(ValueError, match='checksum'): load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')


def test_bounded_retention_preserves_last_and_best(training_case):
    from treeneuron.train import fit_parent
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    directory = paths.directory / 'checkpoints'
    assert len(list(directory.glob('*.pt'))) <= 2
    for name in ('last', 'best'):
        loaded = load_checkpoint(directory / f'{name}.json', expected_hashes=state.identity_hashes, device='cpu')
        assert loaded.update >= state.update


def test_active_stage_has_durable_best_pointer_at_interruption(training_case, monkeypatch):
    from treeneuron import train
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    append = train.append_event
    def interrupt(paths, event):
        append(paths, event)
        if event['type'] == 'checkpoint' and event['update'] == 2: raise KeyboardInterrupt()
    monkeypatch.setattr(train, 'append_event', interrupt)
    with pytest.raises(KeyboardInterrupt): train.fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoints/best.json'
    assert pointer.exists()
    descriptor = json.loads((pointer.parent / json.loads(pointer.read_text())['metadata']).read_text())
    best = load_checkpoint(pointer, expected_hashes=descriptor['state']['identity_hashes'], device='cpu')
    assert best.update in (0, 2)


def test_topology_drift_rejected_before_tensor_load(training_case, monkeypatch):
    from treeneuron.train import fit_parent
    from treeneuron import checkpoint
    import hashlib
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoint.json'
    checkpoint.save_checkpoint(state, pointer)
    p = json.loads(pointer.read_text())
    metadata = pointer.parent / p['metadata']
    descriptor = json.loads(metadata.read_text())
    descriptor['state']['model']['layer']['d'] = 999999
    metadata.write_text(json.dumps(descriptor))
    p['sha256'] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(p))
    def forbidden(*a, **k): raise AssertionError('tensors loaded before topology validation')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='topology'):
        checkpoint.load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')


def test_grown_topology_and_named_multiple_groups_roundtrip(training_case):
    from treeneuron.train import fit_parent, capture_rng, UpdateSchedule
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    old_names = list(dict(state.model.named_parameters()))
    state.model.layer.grow(torch.tensor([1, 3]), torch.full((2, 3), .02))
    named = dict(state.model.named_parameters())
    new_names = [name for name in named if name not in old_names]
    # This checks checkpoint reconstruction, not Task10 optimizer migration.
    state.optimizer.add_param_group({'params': [named[n] for n in new_names],
                                      'param_names': new_names, 'lr': state.optimizer.param_groups[0]['lr']})
    # Added growth parameters own explicit fresh records, including at age zero.
    for name in new_names:
        parameter = named[name]
        state.optimizer.state[parameter] = {
            'step': torch.tensor(0., dtype=torch.float32),
            'exp_avg': torch.zeros_like(parameter), 'exp_avg_sq': torch.zeros_like(parameter)}
    schedule = state.scheduler.state_dict()
    schedule['base_lrs'].append(config.training.learning_rate)
    state.scheduler = UpdateSchedule(state.optimizer, **schedule)
    state.rng_states = capture_rng()
    pointer = paths.directory / 'grown.json'
    save_checkpoint(state, pointer)
    loaded = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(loaded, state)
    assert loaded.model.layer.grown_ids.tolist() == [1, 3]
    assert loaded.optimizer.param_groups[1]['param_names'] == new_names
    np.testing.assert_array_equal(loaded.predict(data.G.X), state.predict(data.G.X))


def test_named_parameter_association_required_at_publication(training_case):
    from treeneuron.train import fit_parent
    from treeneuron.checkpoint import save_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    state.optimizer.param_groups[0]['param_names'].reverse()
    with pytest.raises(ValueError, match='parameter names'):
        save_checkpoint(state, paths.directory / 'wrong.json')
    assert not (paths.directory / 'wrong.json').exists()


def test_checkpoint_rejects_cross_backend_resume(training_case):
    from treeneuron.train import fit_parent
    from treeneuron.checkpoint import load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    with pytest.raises(ValueError, match='Same-backend'):
        load_checkpoint(paths.directory / 'checkpoints/best.json', expected_hashes=state.identity_hashes, device='mps')


@pytest.mark.parametrize('phase', ['.pt', '.meta.json'])
def test_failed_version_write_keeps_prior_pointer_and_failed_evidence(training_case, monkeypatch, phase):
    from treeneuron.train import fit_parent
    from treeneuron import checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    pointer = paths.directory / 'checkpoints/last.json'
    before = pointer.read_bytes()
    write = checkpoint._immutable
    def fail(path, payload):
        write(path, payload)
        if path.name.endswith(phase): raise OSError('injected fsync/publication failure')
    monkeypatch.setattr(checkpoint, '_immutable', fail)
    with pytest.raises(OSError): checkpoint.save_checkpoint(state, pointer)
    assert pointer.read_bytes() == before
    checkpoint.load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    before_cleanup = len(list(pointer.parent.glob('*.pt')))
    checkpoint.prune_checkpoints(pointer.parent)
    assert len(list(pointer.parent.glob('*.pt'))) == before_cleanup


def assert_nested_equal(actual, expected):
    """Compare saved and live RNG collections without consuming any stream."""
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected: assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, e in zip(actual, expected): assert_nested_equal(a, e)
    else:
        assert actual == expected


def corrupt_checkpoint_tensor(tensors, target, value):
    if target == 'model':
        tensors['model']['head.weight'].fill_(value)
    else:
        next(iter(tensors['optimizer']['state'].values()))[target].fill_(value)


@pytest.mark.parametrize('scope', ['current', 'embedded_best'])
@pytest.mark.parametrize('target', ['model', 'exp_avg', 'exp_avg_sq', 'step'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_invalid_tensor_save_preserves_last_valid_and_rng(training_case, scope, target, value):
    from treeneuron.train import fit_parent, pack_state, capture_rng
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    state.best_snapshot = pack_state(state)
    pointer = paths.directory / 'checked.json'
    save_checkpoint(state, pointer)
    valid = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    before_pointer = pointer.read_bytes()
    before_files = set(paths.directory.iterdir())
    if scope == 'embedded_best':
        corrupt_checkpoint_tensor(state.best_snapshot['tensors'], target, value)
    elif target == 'model':
        with torch.no_grad(): state.model.head.weight.fill_(value)
    else:
        next(iter(state.optimizer.state.values()))[target].fill_(value)
    before_rng = capture_rng()
    with pytest.raises(ValueError, match='Nonfinite'):
        save_checkpoint(state, pointer)
    assert pointer.read_bytes() == before_pointer
    assert set(paths.directory.iterdir()) == before_files
    assert_nested_equal(capture_rng(), before_rng)
    restored = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert_state_equal(restored, valid)


@pytest.mark.parametrize('scope', ['state', 'best_state'])
@pytest.mark.parametrize('target', ['model', 'exp_avg', 'exp_avg_sq', 'step'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')], ids=['nan', 'inf'])
def test_valid_checksum_nonfinite_load_rejected_before_optimizer_or_rng(training_case, monkeypatch, scope, target, value):
    from treeneuron.train import fit_parent, pack_state, capture_rng
    from treeneuron.checkpoint import save_checkpoint, load_checkpoint
    import hashlib
    data, config, paths = training_case
    state = fit_parent(data, config, paths)
    state.best_snapshot = pack_state(state)
    pointer = paths.directory / 'malformed-fixture.json'
    save_checkpoint(state, pointer)
    # This intentionally malformed fixture exists only in pytest's temporary
    # run directory. No actual backend evidence/artifact digest is modified.
    p = json.loads(pointer.read_text())
    metadata_path = pointer.parent / p['metadata']
    descriptor = json.loads(metadata_path.read_text())
    payload_path = pointer.parent / descriptor['payload']
    tensors = torch.load(payload_path, map_location='cpu', weights_only=True)
    corrupt_checkpoint_tensor(tensors[scope], target, value)
    torch.save(tensors, payload_path)
    descriptor['sha256'] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(descriptor))
    p['sha256'] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    pointer.write_text(json.dumps(p))
    before_pointer = pointer.read_bytes()
    before_rng = capture_rng()
    def forbidden(*args, **kwargs):
        raise AssertionError('Nonfinite tensor state reached optimizer application')
    monkeypatch.setattr(torch.optim.AdamW, 'load_state_dict', forbidden)
    with pytest.raises(ValueError, match='Nonfinite'):
        load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert pointer.read_bytes() == before_pointer
    assert_nested_equal(capture_rng(), before_rng)
