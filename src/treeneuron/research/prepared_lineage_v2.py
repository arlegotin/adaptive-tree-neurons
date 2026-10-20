"""Recorded preparation/source and opaque archive bytes; no target decoding."""
from collections import Counter
from dataclasses import asdict
import hashlib
import io
from pathlib import Path
import re
import subprocess
import tarfile
import time

from treeneuron.config import canonical_hash,canonical_json
from treeneuron.data.tabular import DATASETS
from .deployment_worker_v2 import _ROOT,_source,_json,_path
from .workload import _checked_manifest


def _sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def archived_scientific_source(commit):
    if type(commit) is not str or re.fullmatch('[0-9a-f]{40}',commit) is None:
        raise ValueError('Exact reachable preparation commit required')
    result=subprocess.run(['git','archive',commit,'src/treeneuron'],cwd=_ROOT,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=30,check=False)
    if result.returncode:raise ValueError('Preparation source commit is unavailable in retained Git history')
    digest=hashlib.sha256();count=0
    with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:') as archive:
        for member in sorted(archive.getmembers(),key=lambda row:Path(row.name)):
            if not member.name.endswith('.py'):continue
            if not member.isfile() or not member.name.startswith('src/treeneuron/'):
                raise ValueError('Preparation Python source is not an ordinary retained file')
            digest.update(member.name.removeprefix('src/treeneuron/').encode())
            digest.update(archive.extractfile(member).read());count+=1
    if not count:raise ValueError('Preparation source contains no scientific Python files')
    return dict(commit=commit,source_sha256=digest.hexdigest(),python_files=count)


def audit_prepared_transfer_suite(*,prepared_root,expected_suite_sha256,acquisition_manifest,
        expected_acquisition_sha256,preparation_commit):
    started=time.perf_counter_ns();source=_source();prepared=Path(prepared_root);acquisition=Path(acquisition_manifest)
    if any(type(v) is not str or re.fullmatch('[0-9a-f]{64}',v) is None for v in (expected_suite_sha256,expected_acquisition_sha256)):
        raise ValueError('External preparation/acquisition SHA256 commitments required')
    if prepared.is_symlink() or not prepared.is_dir() or acquisition.is_symlink():raise ValueError('Ordinary retained data roots required')
    prepared=prepared.resolve();acquisition=acquisition.resolve();references={}
    def observe(path,expected,expected_bytes=None):
        if (path.is_symlink() or not path.is_file() or type(expected) is not str
                or re.fullmatch('[0-9a-f]{64}',expected) is None
                or expected_bytes is not None and (type(expected_bytes) is not int or expected_bytes<=0
                    or path.stat().st_size!=expected_bytes) or _sha(path)!=expected):
            raise ValueError('Prepared/raw artifact differs from its external byte commitment')
        if references.get(str(path),expected)!=expected:raise ValueError('Conflicting data artifact references')
        references[str(path)]=expected
    def decode(path,expected=None):
        if path.is_symlink() or not path.is_file():raise ValueError('Ordinary retained metadata required')
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if expected is not None and digest!=expected:raise ValueError('Decoded metadata differs from its external bytes')
        value=_json(raw);observe(path,digest)
        return value
    suite_path=prepared/'suite.json';suite=decode(suite_path,expected_suite_sha256)
    acquired=decode(acquisition,expected_acquisition_sha256);origin=archived_scientific_source(preparation_commit)
    fields={'schema','status','datasets','models_fitted','protected_labels_read_for_modeling','source_sha256',
        'dependencies','argv','cwd','environment_overrides','config','suite_hash'}
    if (type(suite) is not dict or set(suite)!=fields or suite['schema']!='research/v2-prepared-suite'
            or suite['status']!='completed' or suite['models_fitted'] is not False
            or suite['protected_labels_read_for_modeling'] is not False or suite['source_sha256']!=origin['source_sha256']
            or canonical_hash({k:v for k,v in suite.items() if k!='suite_hash'})!=suite['suite_hash']
            or type(suite['datasets']) is not dict or set(suite['datasets'])!=set(DATASETS)):
        raise ValueError('Completed six-dataset preparation/source identity differs')
    cfg=suite['config']
    if (type(cfg) is not dict or set(cfg)!={'schema','datasets','acquisition_manifest','acquisition_manifest_sha256'}
            or cfg['schema']!='research/v2-data-preparation' or cfg['datasets']!=list(DATASETS)
            or type(cfg['acquisition_manifest']) is not str or cfg['acquisition_manifest_sha256']!=expected_acquisition_sha256):
        raise ValueError('Preparation acquisition/schema declaration differs')
    entries=acquired.get('datasets') if type(acquired) is dict else None
    if (type(entries) is not list or len(entries)!=6 or any(type(row) is not dict or type(row.get('dataset_id')) is not int for row in entries)
            or {r['dataset_id'] for r in entries}!={s.uci_id for s in DATASETS.values()}):
        raise ValueError('Exactly six distinct official acquisition entries required')
    entries={r['dataset_id']:r for r in entries};datasets={};role_count=0;protected_count=0
    for name,schema in DATASETS.items():
        entry=entries[schema.uci_id];matched=entry.get('matched_members')
        if entry.get('status')!='raw_hash_matched' or type(matched) is not list or len(matched)!=1:
            raise ValueError('Required exact raw dataset asset is unavailable')
        raw=matched[0]
        if raw.get('sha256')!=schema.raw_sha256:raise ValueError('Raw identity differs from the fixed dataset schema')
        raw_path=_path(acquisition.parent,raw['path']);observe(raw_path,schema.raw_sha256,raw.get('bytes'))
        manifest_path=_path(prepared,name+'/manifest.json');manifest=decode(manifest_path)
        manifest_hash,_=_checked_manifest(name,manifest)
        if (suite['datasets'][name]!={k:manifest[k] for k in ('manifest_hash','rows','groups','raw_sha256')}
                or manifest_hash!=suite['datasets'][name]['manifest_hash'] or type(manifest['rows']) is not int
                or type(manifest['groups']) is not int or manifest['rows']!=schema.rows or manifest['groups']!=schema.groups
                or manifest['raw_sha256']!=schema.raw_sha256 or manifest['feature_names']!=list(schema.features)
                or manifest['categorical_features']!=list(schema.categorical)
                or canonical_json(manifest['semantic_schema'])!=canonical_json(asdict(schema))):
            raise ValueError('Prepared schema/rows/groups differ from the declared raw dataset')
        required=set()
        for fold,split in enumerate(manifest['splits']):
            roles=split['roles'];counts=split['actual_rows']
            if (split['outer_fold']!=fold or type(roles) is not list or len(roles)!=schema.rows
                    or any(type(role) is not str or role not in ('F','G','V','T','embargo','unused_future') for role in roles)
                    or type(counts) is not dict or any(type(v) is not int or v<0 for v in counts.values())
                    or Counter(roles)!={k:v for k,v in counts.items() if v}):
                raise ValueError('Stored split order or row-count declaration differs')
            required.update(f'outer-{fold}/{role}.npz' for role in 'FG')
            required.update(f'outer-{fold}/{role}-features.npz' for role in 'VT')
            required.update((f'coordinator/outer-{fold}/V-labels.npz',f'protected/outer-{fold}/T-labels.npz'))
        if type(manifest['files']) is not dict or set(manifest['files'])!=required:raise ValueError('Complete fixed role-archive inventory required')
        for relative,ref in manifest['files'].items():
            if type(ref) is not dict or set(ref)!={'sha256','bytes'}:raise ValueError('Exact sealed archive byte reference required')
            observe(_path(prepared/name,relative),ref['sha256'],ref['bytes']);role_count+=1
            protected_count+=relative.startswith('protected/')
        datasets[name]=dict(manifest_hash=manifest_hash,manifest_file_sha256=references[str(manifest_path)],
            raw_sha256=schema.raw_sha256,rows=schema.rows,groups=schema.groups,
            primary_fold=4 if name=='bike' else 0,split_hashes={str(s['outer_fold']):s['split_hash'] for s in manifest['splits']})
    for path,digest in references.items():observe(Path(path),digest)
    if source!=_source():raise ValueError('Data byte-audit implementation changed during inspection')
    return dict(schema='research/v2-prepared-data-byte-source-audit',source_sha256=source,
        preparation_source=origin,recorded_preparation_source_matches_commit=True,
        expected_suite_sha256=expected_suite_sha256,expected_acquisition_sha256=expected_acquisition_sha256,
        suite_hash=suite['suite_hash'],datasets=datasets,raw_assets_hashed=6,role_archives_hashed=role_count,
        protected_label_archives_hashed=protected_count,arrays_or_labels_decoded=False,data_content_rederived=False,
        preparation_execution_independently_reexecuted=False,preparation_runtime_independently_authenticated=False,
        models_fit=0,protected_metrics_computed=0,native_workload_qualified=False,protected_access_authorized=False,
        artifact_references=[dict(path=p,sha256=d) for p,d in sorted(references.items())],
        elapsed_ns=time.perf_counter_ns()-started,
        scope='Consistency of trusted recorded preparation, retained commit and actual raw/sealed byte commitments. Opaque protected archives are hashed, never decoded; no fresh raw-to-row content reconstruction, modeling, native qualification or protected evaluation.')
