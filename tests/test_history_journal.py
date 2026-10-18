"""Checkpoint history uses immutable, portable committed journal prefixes."""
import copy
import hashlib
import json
import shutil
import linecache
import sys

import pytest

from test_train import training_case, cpu_training_evidence, assert_state_equal


def test_history_records_are_append_only_and_defensive():
    from treeneuron.history import History
    event = {'type': 'update', 'nested': {'ids': [1, 2]}}
    history = History([event])
    event['nested']['ids'].append(3)
    first = history[0]
    first['nested']['ids'].append(4)
    assert history[0]['nested']['ids'] == [1, 2]
    fork = copy.deepcopy(history)
    fork.append({'type': 'selection'})
    assert len(history) == 1 and len(fork) == 2
    assert fork[:1] == history and history == [history[0]]
    with pytest.raises(TypeError):
        history[0] = {}


def test_interruption_during_append_cannot_leave_partial_cached_history():
    from treeneuron.history import History
    import treeneuron.history as module
    history = History([{'type': 'start'}])
    before = list(history)
    def interrupt(frame, event, arg):
        if frame.f_code.co_filename == module.__file__ and frame.f_code.co_name == 'append' and event == 'line':
            line = linecache.getline(module.__file__, frame.f_lineno).strip()
            if line.startswith('self._bytes +=') or line.startswith('self._state = ('):
                raise KeyboardInterrupt('injected within history append')
        return interrupt
    try:
        sys.settrace(interrupt)
        with pytest.raises(KeyboardInterrupt):
            history.append({'type': 'pending'})
    finally:
        sys.settrace(None)
    assert list(history) == before


def test_owned_unchanged_prefix_uses_incremental_publication_but_changed_file_is_checked(tmp_path, monkeypatch):
    from treeneuron.history import History
    import treeneuron.history as module
    history = History([{'type': 'start'}])
    reference = history.publish(tmp_path)
    read = module._read_prefix
    reads = []
    def count(*args):
        reads.append(args)
        return read(*args)
    monkeypatch.setattr(module, '_read_prefix', count)
    history.append({'type': 'next'})
    reference = history.publish(tmp_path)
    history.publish(tmp_path)
    assert not reads
    path = tmp_path/reference['file']
    path.write_bytes(b'!'+path.read_bytes()[1:])
    with pytest.raises(ValueError, match='history'):
        history.publish(tmp_path)
    assert reads


@pytest.mark.parametrize('corruption', ['traversal', 'symlink', 'count', 'noncanonical_json'])
def test_history_references_reject_malformed_or_unsafe_inputs(tmp_path, corruption):
    from treeneuron.history import History
    history = History([{'type': 'start'}])
    reference = history.publish(tmp_path)
    path = tmp_path/reference['file']
    if corruption == 'traversal':
        reference['file'] = '../'+reference['file']
    elif corruption == 'symlink':
        original = path.read_bytes()
        path.unlink()
        other = tmp_path/'other.jsonl'
        other.write_bytes(original)
        path.symlink_to(other)
    elif corruption == 'count':
        reference['count'] += 1
    else:
        raw = b'{"type":"first","type":"second"}\n'
        path.write_bytes(raw)
        reference.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        History.load(tmp_path, reference)


def test_retention_checks_existing_versions_without_rescanning_absent_names(tmp_path, monkeypatch):
    import treeneuron.checkpoint as module
    journal = tmp_path/'published.jsonl'
    entries = [dict(metadata=f'checkpoint-{i:032x}.meta.json', payload=f'checkpoint-{i:032x}.pt')
               for i in range(1000)]
    journal.write_text(''.join(json.dumps(e)+'\n' for e in entries))
    unpublished = tmp_path/('checkpoint-'+'f'*32+'.pt')
    unpublished.write_bytes(b'failed unpublished evidence')
    original = module._version_path
    checks = []
    def count(*args):
        checks.append(args)
        return original(*args)
    monkeypatch.setattr(module, '_version_path', count)
    module.prune_checkpoints(tmp_path)
    assert unpublished.read_bytes() == b'failed unpublished evidence'
    assert len(checks) <= 2
    checks.clear()
    # Unexpected journal replacement must rebuild the cache and still respect
    # the immutable journal as the only authority for eligible deletion.
    journal.write_text(json.dumps(dict(metadata=unpublished.stem+'.meta.json', payload=unpublished.name))+'\n')
    module.prune_checkpoints(tmp_path)
    assert not unpublished.exists()


def test_unterminated_publication_entry_is_retained_and_rejected(tmp_path):
    from treeneuron.checkpoint import prune_checkpoints
    raw = json.dumps(dict(metadata='checkpoint-'+'a'*32+'.meta.json', payload='checkpoint-'+'a'*32+'.pt')).encode()
    journal = tmp_path/'published.jsonl'
    journal.write_bytes(raw)
    with pytest.raises(ValueError, match='journal'):
        prune_checkpoints(tmp_path)
    assert journal.read_bytes() == raw


def test_native_checkpoint_uses_compact_history_and_portable_prefix(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.checkpoint import _descriptor, save_checkpoint, load_checkpoint
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    root = tmp_path/'journal'
    pointer = root/'first.json'
    save_checkpoint(parent, pointer)
    _, before, _ = _descriptor(pointer)
    assert before['schema_version'] == 2
    assert before['history']['schema_version'] == 2
    first_history = copy.deepcopy(parent.history)
    metadata_size = len(json.dumps(before))
    for i in range(200):
        parent.history.append({'type': 'diagnostic_marker', 'index': i})
    save_checkpoint(parent, root/'later.json')
    _, later, _ = _descriptor(root/'later.json')
    assert later['history']['file'] == before['history']['file']
    assert len(json.dumps(later)) < metadata_size+100
    assert before['history']['count'] == len(first_history)
    restored = load_checkpoint(pointer, expected_hashes=parent.identity_hashes, device='cpu')
    assert restored.history == first_history
    assert_state_equal(restored, parent)
    clone = tmp_path/'portable'
    shutil.copytree(root, clone)
    moved = load_checkpoint(clone/'later.json', expected_hashes=parent.identity_hashes, device='cpu')
    assert moved.history == parent.history


def test_history_corruption_rejects_before_model_reconstruction(training_case, tmp_path, monkeypatch):
    from treeneuron import train
    from treeneuron.checkpoint import _descriptor, save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    pointer = tmp_path/'journal'/'last.json'
    save_checkpoint(state, pointer)
    _, descriptor, _ = _descriptor(pointer)
    journal = pointer.parent/descriptor['history']['file']
    original = journal.read_bytes()
    journal.write_bytes(b'!'+original[1:])
    calls = []
    monkeypatch.setattr(train, 'unpack_state', lambda *a, **kw: calls.append(1))
    with pytest.raises(ValueError, match='history|History'):
        load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert not calls


def test_uncommitted_tail_is_retained_and_resume_writes_fresh_journal(training_case, tmp_path):
    from treeneuron import train
    from treeneuron.checkpoint import _descriptor, save_checkpoint, load_checkpoint
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    pointer = tmp_path/'journal'/'last.json'
    save_checkpoint(state, pointer)
    _, descriptor, _ = _descriptor(pointer)
    journal = pointer.parent/descriptor['history']['file']
    with journal.open('ab') as stream:
        stream.write(b'{interrupted uncommitted tail')
    retained = journal.read_bytes()
    restored = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert restored.history == state.history
    restored.history.append({'type': 'recovery_marker'})
    save_checkpoint(restored, pointer)
    _, new, _ = _descriptor(pointer)
    assert new['history']['file'] != descriptor['history']['file']
    assert journal.read_bytes() == retained
    loaded = load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert loaded.history == restored.history


def test_interrupted_journal_commit_keeps_old_pointer_and_all_attempt_bytes(training_case, tmp_path, monkeypatch):
    from treeneuron import train, checkpoint
    data, config, paths = training_case
    state = train.fit_parent(data, config, paths)
    pointer = tmp_path/'journal'/'last.json'
    checkpoint.save_checkpoint(state, pointer)
    old_pointer, old_history = pointer.read_bytes(), copy.deepcopy(state.history)
    _, descriptor, _ = checkpoint._descriptor(pointer)
    journal = pointer.parent/descriptor['history']['file']
    state.history.append({'type': 'pending_marker'})
    sync = checkpoint._sync_directory
    def fail(directory):
        if directory == pointer.parent:
            raise OSError('injected interrupted history directory sync')
        return sync(directory)
    monkeypatch.setattr(checkpoint, '_sync_directory', fail)
    with pytest.raises(OSError, match='injected'):
        checkpoint.save_checkpoint(state, pointer)
    monkeypatch.setattr(checkpoint, '_sync_directory', sync)
    assert pointer.read_bytes() == old_pointer
    failed_bytes = journal.read_bytes()
    recovered = checkpoint.load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu')
    assert recovered.history == old_history
    checkpoint.save_checkpoint(state, pointer)
    _, new, _ = checkpoint._descriptor(pointer)
    assert new['history']['file'] != descriptor['history']['file']
    assert journal.read_bytes() == failed_bytes
    assert checkpoint.load_checkpoint(pointer, expected_hashes=state.identity_hashes, device='cpu').history == state.history
