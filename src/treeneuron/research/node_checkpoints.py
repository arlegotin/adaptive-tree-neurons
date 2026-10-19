"""Atomic raw-training recovery, separate from NODE prediction snapshots.

Each payload commits an external immutable row-history prefix. Failed publication
leaves the previous pointer usable and keeps the failed transaction's files.
This module grants no quality eligibility or protected-data authority.
"""
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import uuid

import numpy as np

from . import node_v2 as node


def _publish_pointer(path, pointer):
    temporary = path.parent / ('pointer-' + uuid.uuid4().hex + '.tmp')
    node._write(temporary, pointer)
    os.replace(temporary, path)
    _sync(path.parent)


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ref(path, root):
    return dict(path=str(path.relative_to(root)), sha256=node._sha(path), bytes=path.stat().st_size)


def _checked(root, ref, pattern):
    if (type(ref) is not dict or set(ref) != {'path', 'sha256', 'bytes'}
            or type(ref['path']) is not str or not re.fullmatch(pattern, ref['path'])
            or not node._digest_string(ref['sha256']) or type(ref['bytes']) is not int or ref['bytes'] < 0):
        raise ValueError('Invalid NODE training checkpoint file commitment')
    path = root / ref['path']
    if any(p.is_symlink() for p in (root, path.parent, path)) or not path.is_file() or _ref(path, root) != ref:
        raise ValueError('NODE training checkpoint file identity differs')
    return path


class NodeCheckpointStore:
    def __init__(self, root, binding, *, resume=False):
        self.root = Path(root)
        self.binding = copy.deepcopy(binding)
        self.segments = []
        self.committed_rows = 0
        self.head = None
        self.retained = []
        if self.root.is_symlink():raise ValueError('NODE checkpoint root cannot be a symlink')
        if resume:
            if self.root.is_symlink() or node._read(self.root/'binding.json') != binding:
                raise ValueError('NODE training resume binding differs')
        else:
            self.root.mkdir(parents=True, exist_ok=False)
            node._write(self.root/'binding.json', binding)
            _sync(self.root)
        self._lease=os.open(self.root/'lease.lock',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:fcntl.flock(self._lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self._lease);self._lease=None
            raise ValueError('NODE checkpoint writer already active/locked') from None

    def close(self):
        if self._lease is not None:
            fcntl.flock(self._lease,fcntl.LOCK_UN);os.close(self._lease);self._lease=None

    def _check_binding(self,state,cfg):
        if self._lease is None:raise ValueError('NODE checkpoint store lease is closed')
        if node._read(self.root/'binding.json')!=self.binding or cfg.to_state()!=self.binding['config']:
            raise ValueError('NODE checkpoint source/configuration binding changed')
        keys=('transform','seed','max_updates','n_fit','n_select','F_identity','G_identity','initialization')
        arrays={};packed=node._pack({k:state[k] for k in keys},arrays)
        if (packed!=self.binding['static'] or {k:node._array_identity(v) for k,v in arrays.items()}!=self.binding['arrays']):
            raise ValueError('NODE checkpoint data/seed/transform binding differs')

    def publish(self, state, cfg):
        torch = node._native()[0]
        before = node._rng(torch)
        try:
            self._check_binding(state,cfg)
            node._validate_endpoint(state, cfg, allow_running=True)
            step = state['summary']['executed_updates']
            if len(state['trace']) != step or step < self.committed_rows:
                raise ValueError('NODE checkpoint history regressed')
            directory = self.root / f'checkpoint-{step:012d}-{uuid.uuid4().hex}'
            directory.mkdir()
            segments = copy.deepcopy(self.segments)
            if step > self.committed_rows:
                tracepath = directory/'trace.json'
                node._write(tracepath, state['trace'][self.committed_rows:])
                segments.append(_ref(tracepath, self.root))
            compact = {k:v for k,v in state.items() if k != 'trace'}
            arrays = {}
            payload = node._pack(compact, arrays)
            required=sum(v.nbytes for v in arrays.values())*2+64*1024**2
            if shutil.disk_usage(self.root).free<required:
                raise OSError(28,'Insufficient disk for complete NODE checkpoint transaction; prior pointer retained')
            arraypath = directory/'arrays.npz'
            with arraypath.open('xb') as stream:
                np.savez(stream, **arrays)
                stream.flush();os.fsync(stream.fileno())
            manifest = dict(schema='research/v2-node-training-checkpoint',
                binding_sha256=node._digest(self.binding), step=step, state_fields=sorted(compact),
                payload=payload, arrays={k:node._array_identity(v) for k,v in arrays.items()},
                array_file=_ref(arraypath, self.root), history=segments,
                previous=self.head,eligible_model=False, protected_access_authorized=False)
            node._write(directory/'manifest.json', manifest)
            _sync(directory)
            pointer = dict(schema='research/v2-node-training-pointer',step=step,
                manifest=_ref(directory/'manifest.json', self.root))
            _publish_pointer(self.root/'latest.json', pointer)
            self.segments, self.committed_rows = segments, step
            self.head=pointer['manifest']
            self.retained.append(self.head)
            # Only committed operational tensor payloads are pruned. Histories,
            # manifests, failed transactions, and latest/previous remain intact.
            while len(self.retained)>2:
                old=self.retained.pop(0)
                oldpath=_checked(self.root,old,r'checkpoint-\d{12}-[0-9a-f]{32}/manifest\.json')
                oldmanifest=node._read(oldpath)
                payloadpath=_checked(self.root,oldmanifest['array_file'],r'checkpoint-\d{12}-[0-9a-f]{32}/arrays\.npz')
                node._write(oldpath.parent/'payload-pruned.json',dict(schema='research/v2-node-operational-retention',
                    removed=oldmanifest['array_file'],after_published=self.head,keep_latest_payloads=2))
                payloadpath.unlink();_sync(oldpath.parent)
        finally:
            node._restore_rng(before, torch)

    def load(self, cfg):
        torch = node._native()[0]
        before = node._rng(torch)
        try:
            pointer = node._read(self.root/'latest.json')
            if (type(pointer) is not dict or set(pointer) != {'schema','step','manifest'}
                    or pointer['schema'] != 'research/v2-node-training-pointer'
                    or type(pointer['step']) is not int or pointer['step'] < 0):
                raise ValueError('Invalid NODE training pointer')
            prefix = r'checkpoint-\d{12}-[0-9a-f]{32}/'
            path = _checked(self.root, pointer['manifest'], prefix+r'manifest\.json')
            value = node._read(path)
            fields = {'schema','binding_sha256','step','state_fields','payload','arrays','array_file','history',
                      'previous','eligible_model','protected_access_authorized'}
            if (type(value) is not dict or set(value) != fields
                    or value['schema'] != 'research/v2-node-training-checkpoint'
                    or value['binding_sha256'] != node._digest(self.binding) or value['step'] != pointer['step']
                    or value['eligible_model'] is not False or value['protected_access_authorized'] is not False
                    or type(value['arrays']) is not dict or type(value['history']) is not list):
                raise ValueError('NODE checkpoint schema/binding differs')
            arraypath = _checked(self.root, value['array_file'], prefix+r'arrays\.npz')
            if arraypath.parent != path.parent:
                raise ValueError('NODE checkpoint payload transaction differs')
            with np.load(arraypath, allow_pickle=False) as archive:
                if len(archive.files) != len(set(archive.files)) or set(archive.files) != set(value['arrays']):
                    raise ValueError('NODE checkpoint array inventory differs')
                arrays = {name:archive[name].copy() for name in archive.files}
            for name,array in arrays.items():
                if array.dtype.kind not in 'fiub' or not np.isfinite(array).all() or node._array_identity(array) != value['arrays'][name]:
                    raise ValueError('NODE checkpoint array dtype/finiteness differs')
            seen = set()
            state = node._unpack(value['payload'], arrays, seen)
            if seen != set(arrays) or type(state) is not dict or sorted(state) != value['state_fields'] or 'trace' in state:
                raise ValueError('NODE checkpoint payload fields differ')
            trace = []
            names = set()
            for ref in value['history']:
                segment = _checked(self.root, ref, prefix+r'trace\.json')
                if ref['path'] in names:
                    raise ValueError('Repeated NODE history segment')
                names.add(ref['path'])
                rows = node._read(segment)
                if type(rows) is not list or not rows:
                    raise ValueError('Invalid NODE history segment')
                trace.extend(rows)
            state['trace'] = trace
            if len(trace) != value['step'] or state['summary']['executed_updates'] != value['step']:
                raise ValueError('NODE checkpoint/history boundary differs')
            self._check_binding(state,cfg)
            node._validate_endpoint(state, cfg, allow_running=True)
            self.segments = copy.deepcopy(value['history'])
            self.committed_rows = len(trace)
            self.head=pointer['manifest'];self.retained=[self.head]
            if value['previous'] is not None:
                _checked(self.root,value['previous'],prefix+r'manifest\.json')
                self.retained.insert(0,value['previous'])
            return state
        finally:
            node._restore_rng(before, torch)
