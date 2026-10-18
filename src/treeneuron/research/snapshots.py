"""Exact snapshot consumers and explicit finite-horizon selection grids."""
from pathlib import Path
import json
import os
import uuid

from .ledger import write_new_json


def _commit_json(path, value):
    """Atomic immutable record; incomplete temporary files remain evidence."""
    from treeneuron.config import canonical_json
    from treeneuron.checkpoint import _sync_directory
    encoded = (canonical_json(value)+'\n').encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError('Cannot replace a changed capture record')
        _sync_directory(path.parent)
        return
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    with temporary.open('xb') as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise ValueError('Concurrent changed capture record')
    _sync_directory(path.parent)
    temporary.unlink()
    _sync_directory(path.parent)


def nested_grid(horizon):
    if type(horizon) is not int or horizon <= 0 or horizon % 25:
        raise ValueError('Diagnostic horizons must be positive multiples of25')
    return tuple(range(0, horizon+1, 25))


class SnapshotSink:
    def __init__(self, root, requested):
        self.root, self.requested, self.captured = Path(root), tuple(requested), set()
        self.records = []

    def __call__(self, label, snapshot):
        update = snapshot['metadata']['update']
        self.root.mkdir(parents=True, exist_ok=True)
        from treeneuron.checkpoint import _publish_snapshot
        path = self.root/f'{label}-{update}.json'
        digest = _publish_snapshot(snapshot, None, [], path)
        self.records.append(dict(label=label, update=update, pointer=str(path), metadata_sha256=digest))
        if label == 'current':
            self.captured.add(update)

    def finalize(self):
        write_new_json(self.root/'capture.json', dict(requested=list(self.requested),
            captured=sorted(self.captured), unavailable=sorted(set(self.requested)-self.captured), records=self.records))


class PrefixSnapshotSink:
    """Commit one G-best full snapshot per prefix; replay may repeat a boundary."""
    def __init__(self, root, requested, *, trajectory_id):
        from treeneuron.config import canonical_json
        self.root, self.requested = Path(root), tuple(requested)
        if (not self.requested or any(type(p) is not int or p < 0 for p in self.requested)
                or list(self.requested) != sorted(set(self.requested))
                or type(trajectory_id) is not str or not trajectory_id):
            raise ValueError('Invalid prefix capture identity/plan')
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = dict(schema='research/v2-prefix-capture', trajectory_id=trajectory_id,
                         prefixes=list(self.requested))
        path = self.root/'plan.json'
        if path.exists():
            if canonical_json(json.loads(path.read_text())) != canonical_json(self.plan):
                raise ValueError('Cannot resume changed prefix capture plan')
        else:
            _commit_json(path, self.plan)
        binding_path = self.root/'binding.json'
        self._binding = json.loads(binding_path.read_text()) if binding_path.exists() else None
        self.records = []
        for prefix in self.requested:
            record = self.root/f'prefix-{prefix}.record.json'
            if record.exists():
                value = json.loads(record.read_text())
                self._validate_record(prefix, value)
                self.records.append(value)

    def _validate_snapshot(self, prefix, snapshot):
        from treeneuron import train
        from treeneuron.config import canonical_json
        if not isinstance(snapshot, dict) or set(snapshot) != {'metadata', 'tensors'}:
            raise ValueError('Invalid prefix selection snapshot')
        meta = snapshot['metadata']
        train.validate_metadata(meta, meta['identity_hashes'], 'cpu')
        train.validate_tensor_state(snapshot['tensors'])
        train.validate_saved_adam_history(meta, snapshot['tensors'])
        stage = meta['stage']
        policy = train.TrainingPolicy(**stage['policy'])
        if (stage['complete'] is not False or stage['kind'] not in ('base', 'continuation')
                or policy.stop_by_patience or policy.selection_prefixes != self.requested
                or not 0 <= stage['updates_completed'] <= prefix
                or meta['update'] != stage['start_update']+stage['updates_completed']
                or stage['best_mse'] is None):
            raise ValueError('Snapshot is not a selected state within its declared fixed-budget prefix')
        binding = dict(identity_hashes=meta['identity_hashes'], model=meta['model'],
            growth_metadata=meta['growth_metadata'], stage_kind=stage['kind'],
            start_update=stage['start_update'], policy=stage['policy'])
        if self._binding is not None and canonical_json(binding) != canonical_json(self._binding):
            raise ValueError('Prefix selection belongs to a changed trajectory')
        return binding

    def _snapshot(self, path):
        import torch
        from treeneuron.checkpoint import _descriptor
        pointer, descriptor, payload = _descriptor(path)
        if descriptor['best_state'] is not None or descriptor['history'] != []:
            raise ValueError('Prefix checkpoint must contain one retained selection')
        tensors = torch.load(payload, map_location='cpu', weights_only=True)
        if set(tensors) != {'state', 'best_state'} or tensors['best_state'] is not None:
            raise ValueError('Invalid prefix tensor payload')
        return pointer, dict(metadata=descriptor['state'], tensors=tensors['state'])

    def _validate_record(self, prefix, record):
        from treeneuron.config import canonical_hash
        from treeneuron.experiments.paired import _json_tree
        if (set(record) != {'prefix', 'selected_update', 'snapshot_sha256', 'metadata_sha256', 'pointer'}
                or record['prefix'] != prefix or record['pointer'] != f'prefix-{prefix}.json'):
            raise ValueError('Invalid prefix capture record')
        pointer, snapshot = self._snapshot(self.root/record['pointer'])
        if self._binding is None:
            raise ValueError('Prefix record has no trajectory binding')
        self._validate_snapshot(prefix, snapshot)
        if (pointer['sha256'] != record['metadata_sha256']
                or snapshot['metadata']['update'] != record['selected_update']
                or canonical_hash(_json_tree(snapshot)) != record['snapshot_sha256']):
            raise ValueError('Prefix capture commitment mismatch')

    def __call__(self, prefix, snapshot):
        from treeneuron.config import canonical_hash, canonical_json
        from treeneuron.experiments.paired import _json_tree
        from treeneuron.checkpoint import _publish_snapshot, _sync_directory
        if type(prefix) is not int or prefix not in self.requested:
            raise ValueError('Unplanned prefix')
        binding = self._validate_snapshot(prefix, snapshot)
        if self._binding is None:
            _commit_json(self.root/'binding.json', binding)
            self._binding = json.loads(canonical_json(binding))
        digest = canonical_hash(_json_tree(snapshot))
        record_path = self.root/f'prefix-{prefix}.record.json'
        previous = json.loads(record_path.read_text()) if record_path.exists() else None
        if previous is not None:
            self._validate_record(prefix, previous)
            if previous['snapshot_sha256'] != digest:
                raise ValueError('Cannot replace a changed prefix selection')
            if not any(r['prefix'] == prefix for r in self.records):
                self.records.append(previous)
                self.records.sort(key=lambda r: r['prefix'])
            _sync_directory(self.root)
            return
        path = self.root/f'prefix-{prefix}.json'
        if path.exists():
            # A crash may publish the pointer before its small commit record.
            pointer, orphan = self._snapshot(path)
            self._validate_snapshot(prefix, orphan)
            if canonical_hash(_json_tree(orphan)) != digest:
                raise ValueError('Cannot replace a changed orphan prefix selection')
            metadata_digest = pointer['sha256']
        else:
            metadata_digest = _publish_snapshot(snapshot, None, [], path)
        record = dict(prefix=prefix, selected_update=snapshot['metadata']['update'],
            snapshot_sha256=digest, metadata_sha256=metadata_digest, pointer=path.name)
        _commit_json(record_path, record)
        self.records.append(record)
        self.records.sort(key=lambda r: r['prefix'])

    def finalize(self):
        for record in self.records:
            self._validate_record(record['prefix'], record)
        report = dict(**self.plan, records=self.records,
            unavailable=sorted(set(self.requested)-{r['prefix'] for r in self.records}))
        _commit_json(self.root/'capture.json', report)
        return report
