"""Freeze the actual declared practical search law without granting execution."""
import hashlib
from pathlib import Path

from treeneuron.config import canonical_json
from treeneuron.data.tabular import DATASETS
from .deployment_worker_v2 import _json
from .ledger import write_new_json
from .method_worker import _ROOT, _sha, _digest, scientific_source_sha256
from .search_supports_v2 import family_search_spec
from .workload import practical_search_budget


def _current_law():
    source=scientific_source_sha256();core=_sha(_ROOT/'locks/research-v2/core.json');families={}
    for dataset,schema in DATASETS.items():
        features=dict(columns=list(schema.features),categorical=list(schema.categorical))
        families[dataset]={f'P{i:02}':family_search_spec(f'P{i:02}',features) for i in range(14)}
    body=dict(schema='research/v2-practical-support-lock',source_sha256=source,core_manifest_sha256=core,
        families=families,budgets={f'P{i:02}':practical_search_budget(f'P{i:02}',native_quality=True) for i in range(14)},
        native_workload_qualified=False,protected_access_authorized=False,
        scope='Complete declared source/core-bound native anchors, semantic search supports and finite native-quality budgets; no data, native qualification, sampling, fitting, selection or protected authority')
    if source!=scientific_source_sha256() or core!=_sha(_ROOT/'locks/research-v2/core.json'):
        raise ValueError('Practical support source/core changed during collection')
    return body


def seal_practical_support_lock(output):
    output=Path(output)
    if (output.exists() or output.is_symlink() or output.suffix!='.json'
            or any(output.resolve().is_relative_to((_ROOT/name).resolve()) for name in ('src','tests','configs'))):
        raise ValueError('Fresh JSON support lock required; no overwrite or executable output')
    value=_current_law();write_new_json(output,value)
    return dict(path=str(output.resolve()),sha256=_sha(output))


def load_practical_support_lock(reference):
    if (type(reference) is not dict or set(reference)!={'path','sha256'}
            or type(reference['path']) is not str or not _digest(reference['sha256'])):
        raise ValueError('Exact practical support file reference required')
    path=Path(reference['path'])
    if not path.is_absolute() or path.resolve()!=path or path.is_symlink() or not path.is_file():
        raise ValueError('Ordinary canonical practical support lock required')
    with path.open('rb') as stream:raw=stream.read(16*1024**2+1)
    if len(raw)>16*1024**2 or hashlib.sha256(raw).hexdigest()!=reference['sha256']:
        raise ValueError('Practical support lock bytes differ')
    value=_json(raw);actual=_current_law()
    if canonical_json(value)!=canonical_json(actual) or _sha(path)!=reference['sha256']:
        raise ValueError('Practical support law/schema/source/core differs')
    return actual
