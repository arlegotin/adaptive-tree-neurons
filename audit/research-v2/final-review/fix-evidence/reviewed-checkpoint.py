"""Local complete-state checkpoints with immutable versions and durable pointers.

A pointer JSON names one immutable metadata JSON and its SHA256. Metadata names
one immutable tensor payload and its SHA256. Payload/metadata are fsynced before
atomic pointer replacement and directory fsync. Failure at any earlier boundary
leaves the preceding pointer intact. Tensor files are self-created local artifacts
loaded with weights_only=True; this is not an external checkpoint import API.

Retention: the run's ``last.json`` and ``best.json`` protect their complete payloads.
A last payload also includes the active stage's complete G-best snapshot. Only
superseded versions recorded in published.jsonl may be deleted, after durable
publication. Failed/unpublished files and every event/curve/run are retained. A
caller-supplied additional *.json pointer protects an interruption/final artifact.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import io
import json
import os
from pathlib import Path
import re
import uuid

import torch

from .config import canonical_json

_VERSION = re.compile(r'checkpoint-[0-9a-f]{32}\.(?:pt|meta\.json)\Z')
_PUBLICATION_CACHE = {}
_PUBLICATION_CACHE_LIMIT = 32


def _journal_stamp(path):
    if path.is_symlink():
        raise ValueError('Publication journal cannot be a symlink')
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _published_versions(journal):
    """Cached journal membership; only existing versions need filesystem checks."""
    key, stamp = str(journal.resolve()), _journal_stamp(journal)
    cached = _PUBLICATION_CACHE.get(key)
    if cached is not None and cached['stamp'] == stamp:
        _PUBLICATION_CACHE.pop(key)
        _PUBLICATION_CACHE[key] = cached
        return cached
    names = set()
    if stamp is not None:
        try:
            raw = journal.read_text()
            if raw and not raw.endswith('\n'):
                raise ValueError('Incomplete publication journal tail; retain and recover into a fresh run')
            for line in raw.splitlines():
                entry = json.loads(line)
                if not isinstance(entry, dict) or set(entry) != {'metadata', 'payload'}:
                    raise ValueError('Invalid publication journal entry')
                for key_name, suffix in (('metadata', '.meta.json'), ('payload', '.pt')):
                    name = entry[key_name]
                    if type(name) is not str or not _VERSION.fullmatch(name) or not name.endswith(suffix):
                        raise ValueError('Invalid publication journal version name')
                    names.add(name)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError('Invalid publication journal JSON') from error
        if _journal_stamp(journal) != stamp:
            raise ValueError('Publication journal changed during validation')
    cached = dict(stamp=stamp, names=names)
    _PUBLICATION_CACHE[key] = cached
    if len(_PUBLICATION_CACHE) > _PUBLICATION_CACHE_LIMIT:
        _PUBLICATION_CACHE.pop(next(iter(_PUBLICATION_CACHE)))
    return cached


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def _immutable(path, payload):
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result: raise ValueError('Duplicate checkpoint metadata key')
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(), object_pairs_hook=unique)
        canonical_json(value)
        return value
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Invalid checkpoint JSON metadata') from error


def _version_path(directory, name, suffix):
    if type(name) is not str or not _VERSION.fullmatch(name) or not name.endswith(suffix):
        raise ValueError('Invalid checkpoint version path')
    path = directory / name
    if path.is_symlink(): raise ValueError('Checkpoint version cannot be a symlink')
    return path


def _descriptor(path):
    pointer = _read_json(path)
    if set(pointer) != {'schema_version', 'metadata', 'sha256'} or pointer['schema_version'] != 1:
        raise ValueError('Invalid checkpoint pointer')
    metadata = _version_path(path.parent, pointer['metadata'], '.meta.json')
    if _digest(metadata.read_bytes()) != pointer['sha256']:
        raise ValueError('Checkpoint metadata checksum mismatch')
    descriptor = _read_json(metadata)
    if set(descriptor) != {'schema_version', 'payload', 'sha256', 'state', 'best_state', 'history'} or descriptor['schema_version'] not in (1, 2):
        raise ValueError('Invalid checkpoint descriptor')
    if descriptor['schema_version'] == 2:
        from .history import reference_path
        reference_path(path.parent, descriptor['history'])
    elif not isinstance(descriptor['history'], list):
        raise ValueError('Invalid inline checkpoint history')
    payload = _version_path(path.parent, descriptor['payload'], '.pt')
    if _digest(payload.read_bytes()) != descriptor['sha256']:
        raise ValueError('Checkpoint payload checksum mismatch')
    return pointer, descriptor, payload


def save_checkpoint(state, path: Path) -> str:
    """Publish a complete-update state; return the immutable metadata digest."""
    from .train import pack_state
    return _publish_snapshot(pack_state(state), state.best_snapshot, state.history, Path(path))


def save_best_checkpoint(state, path: Path) -> str:
    """Publish the embedded best snapshot without reconstruction or RNG use."""
    if state.best_snapshot is None:
        raise ValueError('No active-stage best snapshot to publish')
    # The active selection point is its own complete best state. Keeping that
    # snapshot makes best.json a resumable stage checkpoint, without reevaluating
    # a tied G risk or resetting the original allowance/patience counters.
    return _publish_snapshot(state.best_snapshot, state.best_snapshot, state.history, Path(path))


def _publish_snapshot(snapshot, best, history, path):
    from .train import validate_metadata, validate_tensor_state, validate_saved_adam_history
    if path.suffix != '.json' or _VERSION.fullmatch(path.name):
        raise ValueError('Checkpoint path must name a JSON pointer, not a version')
    meta = snapshot['metadata']
    validate_metadata(meta, meta['identity_hashes'], meta['device'])
    validate_tensor_state(snapshot['tensors'])
    validate_saved_adam_history(meta, snapshot['tensors'])
    if best is not None:
        validate_metadata(best['metadata'], meta['identity_hashes'], meta['device'])
        validate_tensor_state(best['tensors'])
        validate_saved_adam_history(best['metadata'], best['tensors'])
    payload = io.BytesIO()
    torch.save({'state': snapshot['tensors'], 'best_state': None if best is None else best['tensors']}, payload)
    raw = payload.getvalue()
    version = f'checkpoint-{uuid.uuid4().hex}'
    from .history import History
    journal = isinstance(history, History)
    history_record = history.publish(path.parent) if journal else history
    descriptor = {'schema_version': 2 if journal else 1, 'payload': version + '.pt', 'sha256': _digest(raw),
                  'state': snapshot['metadata'], 'best_state': None if best is None else best['metadata'],
                  'history': history_record}
    encoded = (canonical_json(descriptor) + '\n').encode()
    digest = _digest(encoded)
    pointer = {'schema_version': 1, 'metadata': version + '.meta.json', 'sha256': digest}
    pointer_bytes = (canonical_json(pointer) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    _immutable(path.parent / (version + '.pt'), raw)
    _immutable(path.parent / (version + '.meta.json'), encoded)
    _sync_directory(path.parent)
    temporary = path.parent / f'.{path.name}.{uuid.uuid4().hex}.tmp'
    _immutable(temporary, pointer_bytes)
    os.replace(temporary, path)
    _sync_directory(path.parent)
    # Journal only durable, successful publications; unjournaled attempts remain
    # evidence even when another save succeeds later.
    journal = path.parent/'published.jsonl'
    cache = _published_versions(journal)
    entry = {'metadata': pointer['metadata'], 'payload': descriptor['payload']}
    with journal.open('ab') as stream:
        stream.write((canonical_json(entry) + '\n').encode())
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    # An interruption before the final stamp leaves a stale cache and forces a
    # complete journal rebuild. Cached names never authorize unpublished files.
    cache['names'].update(entry.values())
    cache['stamp'] = _journal_stamp(journal)
    return digest


def load_checkpoint(path: Path, *, expected_hashes: Mapping[str, str], device: str):
    """Validate safe metadata and identity before deserializing local tensors."""
    from .train import unpack_state, validate_metadata, validate_tensor_state, validate_saved_adam_history
    path = Path(path)
    _, descriptor, payload = _descriptor(path)
    validate_metadata(descriptor['state'], expected_hashes, device)
    if descriptor['best_state'] is not None:
        validate_metadata(descriptor['best_state'], expected_hashes, device)
    from .history import History
    history = (History.load(path.parent, descriptor['history']) if descriptor['schema_version'] == 2
               else History(descriptor['history']))
    tensors = torch.load(payload, map_location='cpu', weights_only=True)
    if not isinstance(tensors, dict) or set(tensors) != {'state', 'best_state'}:
        raise ValueError('Invalid checkpoint tensor schema')
    if (descriptor['best_state'] is None) != (tensors['best_state'] is None):
        raise ValueError('Incomplete best state')
    # Validate both snapshots before any reconstruction can restore/consume RNG
    # or apply tensors to the optimizer. A valid checksum is not a finite-state
    # guarantee, and an invalid embedded best must reject the whole checkpoint.
    validate_tensor_state(tensors['state'])
    validate_saved_adam_history(descriptor['state'], tensors['state'])
    if tensors['best_state'] is not None:
        validate_tensor_state(tensors['best_state'])
        validate_saved_adam_history(descriptor['best_state'], tensors['best_state'])
    state = unpack_state({'metadata': descriptor['state'], 'tensors': tensors['state']},
                         expected_hashes=expected_hashes, device=device)
    state.best_snapshot = None if descriptor['best_state'] is None else {
        'metadata': descriptor['best_state'], 'tensors': tensors['best_state']}
    state.history = history
    return state


def prune_checkpoints(directory: Path) -> dict:
    """Bound successfully superseded storage while preserving failed evidence."""
    directory = Path(directory)
    referenced = set()
    pointers = []
    history_prefixes = {}
    for path in directory.glob('*.json'):
        if _VERSION.fullmatch(path.name): continue
        pointer, descriptor, payload = _descriptor(path)
        referenced.update((pointer['metadata'], payload.name))
        if descriptor['schema_version'] == 2:
            ref = descriptor['history']
            history_prefixes[ref['file']] = max(ref['bytes'], history_prefixes.get(ref['file'], 0))
        pointers.append(path.name)
    journal = directory / 'published.jsonl'
    published = _published_versions(journal)['names']
    for path in directory.iterdir():
        if _VERSION.fullmatch(path.name) and path.name in published and path.name not in referenced:
            suffix = '.meta.json' if path.name.endswith('.meta.json') else '.pt'
            _version_path(directory, path.name, suffix).unlink(missing_ok=True)
    _sync_directory(directory)
    versions = [path for path in directory.iterdir() if _VERSION.fullmatch(path.name)]
    return {'policy': 'retain all durable pointer targets; delete only journaled superseded versions; preserve failed/unpublished attempts',
            'pointers': sorted(pointers), 'referenced_bytes': sum((directory / name).stat().st_size for name in referenced)+sum(history_prefixes.values()),
            'history_committed_prefix_bytes': sum(history_prefixes.values()),
            'history_files': sorted(history_prefixes),
            'retained_version_bytes': sum(p.stat().st_size for p in versions),
            'directory_bytes': sum(p.stat().st_size for p in directory.iterdir() if p.is_file()),
            'retained_payloads': sum(p.suffix == '.pt' for p in versions),
            'active_best_embedded_in_last': True}
