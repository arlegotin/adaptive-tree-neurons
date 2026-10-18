"""Exact snapshot consumers and explicit finite-horizon selection grids."""
from pathlib import Path

from .ledger import write_new_json


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
