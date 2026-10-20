"""Read-only pre-freeze API history evidence; no arrays or access capability.

An authenticated log describes recorded API calls. It cannot authenticate the
absence of unlogged direct filesystem access. No freeze readiness is granted.
"""
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re

from treeneuron.config import canonical_hash


_DATASETS={'airfoil','concrete','ccpp','parkinsons','bike','energy'}
_ROLES={'F_labels','G_labels','V_features','V_labels','V_group_or_day_ids'}
_EVENT_FIELDS={'schema','outer_fold','roles','actor','utc_ns','manifest_hash','previous_hash','event_hash'}


def _digest(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _json(raw):
    def unique(pairs):
        value={}
        for key,item in pairs:
            if key in value:raise ValueError('Duplicate transfer audit field')
            value[key]=item
        return value
    def forbidden(value):raise ValueError('Nonfinite transfer audit JSON')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=forbidden)


def _bytes(path):
    if path.is_symlink() or not path.is_file():raise ValueError('Unsafe transfer audit reference')
    return path.read_bytes()


def _manifest(path,expected):
    raw=_bytes(path)
    if not _digest(expected) or hashlib.sha256(raw).hexdigest()!=expected:
        raise ValueError('External transfer manifest commitment differs')
    manifest=_json(raw)
    if (type(manifest) is not dict or manifest.get('schema')!='research/v2-sealed-transfer'
            or type(manifest.get('dataset')) is not str or manifest['dataset'] not in _DATASETS
            or manifest.get('protected_labels_read_for_modeling') is not False
            or canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest.get('manifest_hash')):
        raise ValueError('Transfer metadata schema/scope/commitment differs')
    primary=4 if manifest['dataset']=='bike' else 0
    splits=manifest.get('splits')
    if type(splits) is not list or len(splits)!=5:raise ValueError('Five fixed transfer folds required')
    for fold,split in enumerate(splits):
        if (type(split) is not dict or split.get('schema')!='research/v2-transfer-split'
                or split.get('dataset')!=manifest['dataset'] or type(split.get('outer_fold')) is not int
                or split['outer_fold']!=fold or split.get('primary') is not (fold==primary)
                or canonical_hash({k:v for k,v in split.items() if k!='split_hash'})!=split.get('split_hash')):
            raise ValueError('Transfer primary/fold identity differs')
    return raw,manifest,primary


def _history(raw,manifest,primary):
    if raw and not raw.endswith(b'\n'):raise ValueError('Partial transfer access record')
    previous='0'*64;roles=Counter();count=0
    for line in raw.splitlines():
        event=_json(line)
        if type(event) is not dict or set(event)!=_EVENT_FIELDS:raise ValueError('Unknown transfer access event schema')
        digest=event.pop('event_hash')
        if (event['previous_hash']!=previous or canonical_hash(event)!=digest
                or event['schema']!='research/v2-transfer-access'
                or event['actor']!='declared_pre_freeze_coordinator'
                or type(event['utc_ns']) is not int or event['utc_ns']<=0
                or event['manifest_hash']!=manifest['manifest_hash']
                or type(event['outer_fold']) is not int or event['outer_fold']!=primary
                or type(event['roles']) is not list or not event['roles']
                or any(type(role) is not str or role not in _ROLES for role in event['roles'])
                or len(set(event['roles']))!=len(event['roles'])):
            raise ValueError('Corrupt or forbidden pre-freeze transfer access')
        roles.update(event['roles']);count+=1;previous=digest
    return dict(event_count=count,role_counts=dict(sorted(roles.items())),access_event_hash=previous)


@contextmanager
def primary_access_snapshot(root,*,expected_manifest_sha256,expected_access_sha256=None):
    """Hold a shared existing-log lease through the caller's read-only checks.

    Log absence cannot be leased without creating a file: it is explicitly a
    weaker absence snapshot, rechecked at exit. This function creates no files.
    """
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('Unsafe sealed transfer root')
    original,manifest,primary=_manifest(root/'manifest.json',expected_manifest_sha256)
    path=root/'access.jsonl';stream=None;held=False
    try:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file():raise ValueError('Unsafe transfer access log')
            stream=path.open('rb')
            try:fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
            except BlockingIOError as error:raise PermissionError('Transfer access append is active') from error
            held=True;raw=stream.read();digest=hashlib.sha256(raw).hexdigest()
        else:raw=b'';digest=None
        if expected_access_sha256 is not None and (not _digest(expected_access_sha256) or digest!=expected_access_sha256):
            raise ValueError('External transfer access commitment differs')
        result=dict(schema='research/v2-pre-freeze-transfer-access-audit',
            status='valid_recorded_primary_access' if held else 'access_log_absent_snapshot',
            dataset=manifest['dataset'],primary_fold=primary,manifest_file_sha256=expected_manifest_sha256,
            manifest_hash=manifest['manifest_hash'],access_file_sha256=digest,
            raw_identity_verified=manifest.get('raw_identity_verified') is True,
            shared_lease_held=held,lease_scope='During this audit/context only; result is not a live capability',
            **_history(raw,manifest,primary),arrays_or_labels_opened=False,T_accessed=False,
            protected_access_authorized=False,freeze_readiness_qualified=False,
            unlogged_filesystem_access_authenticated=False)
        yield result
        if _bytes(root/'manifest.json')!=original:raise ValueError('Transfer manifest changed during access audit')
        if held:
            if _bytes(path)!=raw:raise ValueError('Transfer access changed despite shared lease')
        elif path.exists() or path.is_symlink():raise ValueError('Previously absent transfer log appeared during audit')
    finally:
        if stream is not None:
            if held:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
            stream.close()


def audit_pre_freeze_access(root,*,expected_manifest_sha256,expected_access_sha256=None):
    with primary_access_snapshot(root,expected_manifest_sha256=expected_manifest_sha256,
                                 expected_access_sha256=expected_access_sha256) as result:
        return result


@contextmanager
def frozen_primary_access_prefix(root,*,expected_manifest_sha256,prefix=None):
    """Preserve a recorded boundary while permitting later valid primary audits.

    This is not a data-release capability. A separate post-freeze journal must
    carry any authorized secondary/protected access; it cannot enter this log.
    """
    fields={'schema','dataset','primary_fold','manifest_file_sha256','manifest_hash','prefix_bytes',
        'prefix_sha256','prefix_event_count','prefix_event_hash','access_log_absent_at_freeze',
        'protected_access_authorized','unlogged_filesystem_access_authenticated','arrays_or_labels_opened','T_accessed','scope'}
    with primary_access_snapshot(root,expected_manifest_sha256=expected_manifest_sha256) as snapshot:
        path=Path(root)/'access.jsonl';raw=_bytes(path) if path.exists() else b''
        if prefix is None:
            length=len(raw);absent=snapshot['status']=='access_log_absent_snapshot'
        else:
            if (type(prefix) is not dict or set(prefix)!=fields or type(prefix.get('prefix_bytes')) is not int
                    or not 0<=prefix['prefix_bytes']<=len(raw)
                    or type(prefix.get('prefix_event_count')) is not int or prefix['prefix_event_count']<0
                    or type(prefix.get('primary_fold')) is not int
                    or any(prefix.get(k) is not False for k in ('protected_access_authorized',
                        'unlogged_filesystem_access_authenticated','arrays_or_labels_opened','T_accessed'))
                    or type(prefix.get('access_log_absent_at_freeze')) is not bool):
                raise ValueError('Exact frozen prefix record and integer byte boundary required')
            length=prefix['prefix_bytes'];absent=prefix['access_log_absent_at_freeze']
        if absent and length:raise ValueError('An absent frozen log cannot contain recorded access bytes')
        _,manifest,primary=_manifest(Path(root)/'manifest.json',expected_manifest_sha256)
        history=_history(raw[:length],manifest,primary)
        record=dict(schema='research/v2-frozen-primary-access-prefix',dataset=snapshot['dataset'],
            primary_fold=primary,manifest_file_sha256=expected_manifest_sha256,manifest_hash=snapshot['manifest_hash'],
            prefix_bytes=length,prefix_sha256=hashlib.sha256(raw[:length]).hexdigest(),
            prefix_event_count=history['event_count'],prefix_event_hash=history['access_event_hash'],
            access_log_absent_at_freeze=absent,protected_access_authorized=False,
            unlogged_filesystem_access_authenticated=False,arrays_or_labels_opened=False,T_accessed=False,
            scope='Exact recorded primary API prefix; later valid primary-only events allowed. No secondary/protected capability or unlogged filesystem nonaccess proof.')
        if prefix is not None and record!=prefix:raise ValueError('Frozen primary API prefix identity or bytes changed')
        yield record
