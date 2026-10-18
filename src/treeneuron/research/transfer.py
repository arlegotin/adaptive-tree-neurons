"""Prepare immutable transfer roles and expose primary F/G-only worker views.

Sealing is an audited API/data-role boundary, not a claim that an operator with
filesystem access is cryptographically unable to read original public targets.
Protected evaluation and secondary-development release remain WP09–WP10 work.
"""
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from treeneuron.config import canonical_hash
from treeneuron.data.tabular import DATASETS, load_table
from treeneuron.data.types import ObservationSet, _readonly
from .ledger import write_new_json
from .splits import grouped_splits, bike_splits


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal_table(table, output):
    root = Path(output)
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise ValueError('Transfer output exists; preserve immutable prepared data') from error
    splits = bike_splits(table.timestamps) if table.name == 'bike' else grouped_splits(table.name, table.group_id)
    manifest = dict(schema='research/v2-sealed-transfer', dataset=table.name,
        table_id=table.table_id, raw_sha256=table.raw_sha256,
        raw_identity_verified=table.raw_sha256 == DATASETS[table.name].raw_sha256,
        semantic_schema=asdict(DATASETS[table.name]), feature_names=list(table.feature_names),
        categorical_features=list(table.categorical_features), rows=len(table.y),
        groups=len(np.unique(table.group_id)), transformed_dimensions='adapter_F_fit_pending',
        splits=splits, files={}, protected_labels_read_for_modeling=False,
        labels_parsed_for='deterministic_preparation_and_partition_serialization_only',
        targets_inspected_for_split_assignment=False, prepared_utc_ns=time.time_ns())

    def save(relative, **arrays):
        path = root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        manifest['files'][relative] = dict(sha256=_file_hash(path), bytes=path.stat().st_size)

    for split in splits:
        fold = split['outer_fold']
        roles = np.asarray(split['roles'])
        for role in 'FGVT':
            mask = roles == role
            prefix = f'outer-{fold}/{role}'
            fields = dict(X=table.X[mask], row_id=table.row_id[mask], group_id=table.group_id[mask])
            if table.timestamps is not None:
                fields['timestamps'] = table.timestamps[mask]
            if role in 'FG':
                save(prefix+'.npz', **fields, y=table.y[mask])
            else:
                save(prefix+'-features.npz', **fields)
                save(('protected/' if role == 'T' else 'coordinator/')+prefix+'-labels.npz',
                     y=table.y[mask], row_id=table.row_id[mask])
    manifest['manifest_hash'] = canonical_hash(manifest)
    write_new_json(root/'manifest.json', manifest)
    return manifest


@dataclass(frozen=True)
class TransferFitView:
    F: ObservationSet
    G: ObservationSet
    validation_features: np.ndarray
    validation_row_ids: np.ndarray
    categorical_features: tuple[str, ...]
    manifest_hash: str
    split_hash: str

    @property
    def V(self):
        raise PermissionError('V labels remain with the transfer search coordinator')

    @property
    def T(self):
        raise PermissionError('T labels require the frozen transfer evaluator')


class TransferStore:
    def __init__(self, root):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        value = {k: v for k, v in manifest.items() if k != 'manifest_hash'}
        if (value['schema'] != 'research/v2-sealed-transfer'
                or value['dataset'] not in DATASETS
                or canonical_hash(value) != manifest['manifest_hash']):
            raise ValueError('Transfer manifest identity changed')
        self._manifest_json = json.dumps(manifest, sort_keys=True)

    @property
    def manifest(self):
        """A caller may inspect metadata but cannot mutate validated access rules."""
        return json.loads(self._manifest_json)

    def _split(self, fold):
        if type(fold) is not int or not 0 <= fold < 5:
            raise ValueError('Transfer outer fold must be an integer0–4')
        split = self.manifest['splits'][fold]
        primary = 4 if self.manifest['dataset'] == 'bike' else 0
        if fold != primary or not split['primary']:
            raise PermissionError('Secondary F/G/V release requires the frozen global v2 procedure (WP09)')
        return split

    def _read(self, relative):
        path = self.root/relative
        if (not path.resolve().is_relative_to(self.root.resolve()) or relative not in self.manifest['files']
                or _file_hash(path) != self.manifest['files'][relative]['sha256']):
            raise ValueError('Transfer role file changed or escaped sealed root')
        with np.load(path, allow_pickle=False) as saved:
            return {key: _readonly(saved[key], saved[key].dtype) for key in saved.files}

    def _access(self, fold, roles):
        # One lock covers reading the previous commitment and appending/fsync.
        with (self.root/'access.jsonl').open('a+', encoding='utf-8') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.seek(0)
            records = stream.read().splitlines()
            previous = '0'*64
            for line in records:
                old = json.loads(line)
                digest = old.pop('event_hash')
                if old['previous_hash'] != previous or canonical_hash(old) != digest:
                    raise ValueError('Transfer access commitments changed')
                previous = digest
            event = dict(schema='research/v2-transfer-access', outer_fold=fold,
                roles=roles, actor='declared_pre_freeze_coordinator', utc_ns=time.time_ns(),
                manifest_hash=self.manifest['manifest_hash'], previous_hash=previous)
            event['event_hash'] = canonical_hash(event)
            stream.seek(0, os.SEEK_END)
            stream.write(json.dumps(event, sort_keys=True, separators=(',', ':'))+'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def development(self, fold):
        split = self._split(fold)
        parts = {}
        for role in 'FG':
            data = self._read(f'outer-{fold}/{role}.npz')
            parts[role] = ObservationSet(data['X'], data['y'], data['row_id'], data['group_id'],
                dates=data.get('timestamps'), feature_names=tuple(self.manifest['feature_names']),
                stream_id=f"transfer-v2/{self.manifest['table_id']}/{fold}/{role}")
        validation = self._read(f'outer-{fold}/V-features.npz')
        self._access(fold, ['F_labels', 'G_labels', 'V_features'])
        return TransferFitView(parts['F'], parts['G'], validation['X'], validation['row_id'],
            tuple(self.manifest['categorical_features']), self.manifest['manifest_hash'], split['split_hash'])

    def labels(self, fold, role):
        self._split(fold)
        if role != 'V':
            raise PermissionError('Only primary V labels are available to this coordinator; T requires WP09/WP10 freeze/evaluator')
        return self.coordinator_validation(fold).y

    def coordinator_validation(self, fold):
        self._split(fold)
        features = self._read(f'outer-{fold}/V-features.npz')
        labels = self._read(f'coordinator/outer-{fold}/V-labels.npz')
        if not np.array_equal(features['row_id'], labels['row_id']):
            raise ValueError('Coordinator validation labels/features have different row identities')
        self._access(fold, ['V_labels', 'V_features', 'V_group_or_day_ids'])
        return ObservationSet(features['X'], labels['y'], features['row_id'], features['group_id'],
            dates=features.get('timestamps'), feature_names=tuple(self.manifest['feature_names']),
            stream_id=f"transfer-v2/{self.manifest['table_id']}/{fold}/V")


def _data_environment():
    required = {'openpyxl': '3.1.5', 'xlrd': '2.0.2', 'et-xmlfile': '2.0.0'}
    evidence = {}
    for name in ('pandas', 'numpy', *required):
        matches = [dist for dist in importlib.metadata.distributions()
                   if dist.metadata['Name'].lower().replace('_', '-') == name]
        if len(matches) != 1:
            raise ValueError(f'Data dependency {name}: require one installed distribution; install the locked isolated data overlay')
        dist = matches[0]
        module = importlib.import_module(name.replace('-', '_'))
        version = getattr(module, '__version__', dist.version)
        if version != dist.version or (name in required and dist.version != required[name]):
            raise ValueError(f'Data dependency {name}: import/metadata/version differs from the qualified lock')
        evidence[name] = dict(version=dist.version, import_path=str(Path(module.__file__).resolve()),
            distribution_path=str(dist.locate_file('')), record_sha256=hashlib.sha256(
                (dist.read_text('RECORD') or '').encode()).hexdigest())
    return evidence


def prepare_data(config_path, output):
    """Prepare all six exact acquired tables; no fitting or protected metrics."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'Duplicate data configuration key: {key}')
            result[key] = value
        return result
    value = json.loads(Path(config_path).read_text(), object_pairs_hook=unique)
    if (set(value) != {'schema', 'acquisition_manifest', 'acquisition_manifest_sha256', 'datasets'}
            or value['schema'] != 'research/v2-data-preparation'
            or value['datasets'] != list(DATASETS)):
        raise ValueError('Data preparation requires exactly the frozen six-dataset schema')
    acquisition = Path(value['acquisition_manifest'])
    if _file_hash(acquisition) != value['acquisition_manifest_sha256']:
        raise ValueError('Acquisition manifest differs from pinned bytes')
    acquired = json.loads(acquisition.read_text())
    entries = {entry['dataset_id']: entry for entry in acquired['datasets']}
    output = Path(output)
    if output.exists():
        raise ValueError('Prepared suite output already exists; no silent overwrite')
    output.mkdir(parents=True)
    result = dict(schema='research/v2-prepared-suite', config=value, datasets={}, models_fitted=False,
                  protected_labels_read_for_modeling=False, status='running', dependencies={},
                  argv=sys.argv, cwd=str(Path.cwd()), environment_overrides={name: os.environ.get(name)
                    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTORCH_ENABLE_MPS_FALLBACK')})
    source_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob('*.py')):
        digest.update(path.relative_to(source_root).as_posix().encode())
        digest.update(path.read_bytes())
    result['source_sha256'] = digest.hexdigest()
    try:
        result['dependencies'] = _data_environment()
        for name, schema in DATASETS.items():
            entry = entries[schema.uci_id]
            matches = entry['matched_members']
            if entry['status'] != 'raw_hash_matched' or len(matches) != 1 or matches[0]['sha256'] != schema.raw_sha256:
                raise ValueError(f'{name}: acquisition did not retain the expected exact raw asset')
            path = acquisition.parent/matches[0]['path']
            if not path.resolve().is_relative_to(acquisition.parent.resolve()):
                raise ValueError('Raw acquisition path escaped its root')
            table = load_table(name, path)
            record = seal_table(table, output/name)
            result['datasets'][name] = dict(manifest_hash=record['manifest_hash'],
                rows=record['rows'], groups=record['groups'], raw_sha256=record['raw_sha256'])
        result['status'] = 'completed'
        result['suite_hash'] = canonical_hash(result)
        write_new_json(output/'suite.json', result)
    except BaseException as error:
        result.update(status='failed_or_interrupted', failure=f'{type(error).__name__}: {error}')
        raise
    finally:
        write_new_json(output/'preparation-status.json', result)
    return result
