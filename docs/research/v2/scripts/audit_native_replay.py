#!/usr/bin/env python3
"""Invoke only the pure native raw-event reader; never restore a model."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from treeneuron.research import replay

here = Path(__file__).resolve().parent
main = here.parents[3]
records = []
for name in ('M0-native-qualification', 'M0-journal-qualification'):
    pair_root = main / 'artifacts/research-v2/WP02' / name / 'pair'
    pair = json.loads((pair_root / 'pair.json').read_text())
    result = replay.reference_replay(pair_root, pair,
        SimpleNamespace(validation_interval=25, patience=20, base_updates=8000))
    assert result['classification'] == 'verified_patience_tail_replay'
    assert result['matching_updates'] == 500 and result['matching_G_evaluations'] == 21
    records.append(dict(root=str(pair_root.relative_to(main)), evidence=result))
receipt = dict(schema='research/v2-storage-equivalence-replay-reader',
    status='passed', records=records, state_restored=False, fitting=False,
    reader_sha256=hashlib.sha256(Path(replay.__file__).read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    interpretation='same exploratory parent losing-tail execution diagnostic; no headroom or independent-teacher inference')
with (here / 'replay-result.json').open('x') as stream:
    json.dump(receipt, stream, indent=2, sort_keys=True)
    stream.write('\n')
print(json.dumps(receipt, sort_keys=True))
