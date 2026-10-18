"""Immutable event values with portable append-only checkpoint journal prefixes."""
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .config import canonical_json

_NAME = re.compile(r'history-[0-9a-f]{32}\.jsonl\Z')


def reference_path(directory, reference):
    if (not isinstance(reference, dict)
            or set(reference) != {'schema_version', 'encoding', 'file', 'bytes', 'count', 'sha256'}
            or type(reference['schema_version']) is not int or reference['schema_version'] != 2
            or reference['encoding'] != 'canonical_json_lines_utf8'
            or type(reference['file']) is not str or not _NAME.fullmatch(reference['file'])
            or any(type(reference[k]) is not int or reference[k] < 0 for k in ('bytes', 'count'))
            or type(reference['sha256']) is not str
            or not re.fullmatch(r'[0-9a-f]{64}', reference['sha256'])):
        raise ValueError('Invalid checkpoint history reference')
    path = Path(directory)/reference['file']
    if path.is_symlink() or not path.is_file() or path.stat().st_size < reference['bytes']:
        raise ValueError('Missing, short or unsafe checkpoint history file')
    return path


def _read_prefix(path, length):
    with path.open('rb') as stream:
        raw = stream.read(length)
    if len(raw) != length:
        raise ValueError('Short checkpoint history prefix')
    return raw


def _stamp(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class History(Sequence):
    """Append-only records; cached values are canonical bytes, never mutable dicts."""
    def __init__(self, records=()):
        self._state = (None, 0, 0, hashlib.sha256())
        self._journals = {}
        for record in records:
            self.append(record)

    def append(self, record):
        if not isinstance(record, dict):
            raise ValueError('History events must be JSON objects')
        raw = (canonical_json(record)+'\n').encode()
        tail, count, size, previous_hash = self._state
        digest = previous_hash.copy()
        digest.update(raw)
        self._state = ((raw, tail), count+1, size+len(raw), digest)

    def _lines(self, start=0):
        tail, count, _, _ = self._state
        values = []
        for _ in range(count-start):
            raw, tail = tail
            values.append(raw)
        return reversed(values)

    def __len__(self):
        return self._state[1]

    def __getitem__(self, index):
        if index == -1 and len(self):
            return json.loads(self._state[0][0])
        values = list(self._lines())
        if isinstance(index, slice):
            return [json.loads(raw) for raw in values[index]]
        return json.loads(values[index])

    def __iter__(self):
        return (json.loads(raw) for raw in self._lines())

    def __eq__(self, other):
        if isinstance(other, History):
            return list(self._lines()) == list(other._lines())
        if isinstance(other, (list, tuple)):
            return list(self) == list(other)
        return NotImplemented

    def __deepcopy__(self, memo):
        result = type(self)()
        result._state = self._state  # Immutable tail; append commits a new head.
        memo[id(self)] = result
        return result

    def publish(self, directory):
        from .checkpoint import _sync_directory
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        key = str(directory.resolve())
        cached = self._journals.get(key)
        previous = None if cached is None else cached['reference']
        if previous is not None:
            path = reference_path(directory, previous)
            changed = _stamp(path) != cached['stamp']
            if changed and hashlib.sha256(_read_prefix(path, previous['bytes'])).hexdigest() != previous['sha256']:
                raise ValueError('Changed committed checkpoint history prefix')
            if path.stat().st_size != previous['bytes']:
                previous = None  # Preserve uncommitted tail; never truncate it.
        if previous is None:
            path = directory/f'history-{uuid.uuid4().hex}.jsonl'
            count, mode = 0, 'xb'
        else:
            count, mode = previous['count'], 'ab'
        with path.open(mode) as stream:
            for raw in self._lines(count):
                stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(directory)
        reference = dict(schema_version=2, encoding='canonical_json_lines_utf8', file=path.name,
            bytes=self._state[2], count=len(self), sha256=self._state[3].hexdigest())
        self._journals[key] = dict(reference=reference, stamp=_stamp(path))
        return dict(reference)

    @classmethod
    def load(cls, directory, reference):
        path = reference_path(directory, reference)
        stamp = _stamp(path)
        raw = _read_prefix(path, reference['bytes'])
        if _stamp(path) != stamp:
            raise ValueError('Checkpoint history changed during validation')
        if hashlib.sha256(raw).hexdigest() != reference['sha256']:
            raise ValueError('Checkpoint history checksum mismatch')
        result = cls()
        try:
            for line in raw.splitlines(keepends=True):
                value = json.loads(line)
                if not isinstance(value, dict) or (canonical_json(value)+'\n').encode() != line:
                    raise ValueError('Noncanonical checkpoint history record')
                result.append(value)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as error:
            raise ValueError('Invalid checkpoint history JSON') from error
        if len(result) != reference['count'] or result._state[2] != reference['bytes']:
            raise ValueError('Checkpoint history count/length mismatch')
        result._journals[str(Path(directory).resolve())] = dict(reference=dict(reference), stamp=stamp)
        return result
