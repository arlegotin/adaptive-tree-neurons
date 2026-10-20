"""Separate secondary F/G/V roles after the actual global primary barrier.

No secondary native-workload or protected/T capability is provided here.
Legacy TransferStore's primary-only guards remain unchanged.
"""
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time


_ACTORS={
    'frozen_secondary_fit_worker':['F_labels','G_labels','V_features'],
    'frozen_secondary_V_coordinator':['V_labels','V_features','V_group_or_day_ids'],
}


def _check_access(binding,manifest,fold,actor,roles):
    from .primary_search_worker_v2 import _digest
    fields={'protocol_sha256','source_sha256','manifest_file_sha256','dataset','manifest_hash','created_utc_ns'}
    if (type(binding) is not dict or set(binding)!=fields
            or any(not _digest(binding[k]) for k in ('protocol_sha256','source_sha256','manifest_file_sha256','manifest_hash'))
            or type(binding['created_utc_ns']) is not int or binding['created_utc_ns']<=0
            or binding['dataset']!=manifest['dataset'] or binding['manifest_hash']!=manifest['manifest_hash']
            or type(fold) is not int or not 0<=fold<5 or fold==(4 if manifest['dataset']=='bike' else 0)
            or manifest['splits'][fold].get('primary') is not False
            or actor not in _ACTORS or type(roles) is not list or roles!=_ACTORS[actor]):
        raise ValueError('Exact frozen secondary source/protocol/manifest/actor/roles required; no primary/T access')


def _record_access(path,*,binding,manifest,outer_fold,actor,roles):
    """Append recorded API metadata only; this function grants no data access."""
    from treeneuron.config import canonical_hash,canonical_json
    from .primary_search_worker_v2 import _path,_json
    _check_access(binding,manifest,outer_fold,actor,roles)
    path=_path(str(path))
    if path.exists() and not path.is_file():raise ValueError('Ordinary released access journal required')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        stream.seek(0);raw=stream.read()
        if raw and not raw.endswith(b'\n'):raise ValueError('Partial released access event cannot be reset')
        previous='0'*64
        for line in raw.splitlines():
            event=_json(line)
            fields={'schema','binding','outer_fold','split_hash','actor','roles','utc_ns','previous_hash','event_hash'}
            if (type(event) is not dict or set(event)!=fields or event['schema']!='research/v2-released-transfer-access'
                    or canonical_json(event['binding'])!=canonical_json(binding) or event['previous_hash']!=previous
                    or canonical_hash({k:v for k,v in event.items() if k!='event_hash'})!=event['event_hash']
                    or type(event['utc_ns']) is not int or event['utc_ns']<binding['created_utc_ns']):
                raise ValueError('Released access source/protocol/hash history changed')
            _check_access(binding,manifest,event['outer_fold'],event['actor'],event['roles'])
            if event['split_hash']!=manifest['splits'][event['outer_fold']]['split_hash']:
                raise ValueError('Released split identity changed')
            previous=event['event_hash']
        now=time.time_ns()
        if now<binding['created_utc_ns']:raise ValueError('Released access cannot precede global freeze')
        event=dict(schema='research/v2-released-transfer-access',binding=copy.deepcopy(binding),outer_fold=outer_fold,
            split_hash=manifest['splits'][outer_fold]['split_hash'],actor=actor,roles=list(roles),utc_ns=now,previous_hash=previous)
        event['event_hash']=canonical_hash(event)
        stream.seek(0,os.SEEK_END);stream.write((canonical_json(event)+'\n').encode());stream.flush();os.fsync(stream.fileno())


class _ReleasedRoles:
    """Private data-only engine, also tested on unqualified toy tables."""
    def __init__(self,root,*,expected_manifest_sha256,record):
        from .transfer_access_v2 import _manifest
        from .transfer import TransferStore
        root=Path(root);_manifest(root/'manifest.json',expected_manifest_sha256)
        self._store=TransferStore(root);self._record=record

    @property
    def manifest(self):return self._store.manifest

    def _split(self,fold):
        primary=4 if self.manifest['dataset']=='bike' else 0
        if type(fold) is not int or not 0<=fold<5:raise ValueError('Integer fixed secondary fold required')
        split=self.manifest['splits'][fold]
        if fold==primary or split['primary'] is not False:raise PermissionError('Released route admits secondary development folds only')
        return split

    def _observation(self,data,fold,role):
        from treeneuron.data.types import ObservationSet
        return ObservationSet(data['X'],data['y'],data['row_id'],data['group_id'],dates=data.get('timestamps'),
            feature_names=tuple(self.manifest['feature_names']),stream_id=f"transfer-v2/{self.manifest['table_id']}/{fold}/{role}")

    def development(self,fold):
        from .transfer import TransferFitView
        split=self._split(fold);parts={r:self._observation(self._store._read(f'outer-{fold}/{r}.npz'),fold,r) for r in 'FG'}
        validation=self._store._read(f'outer-{fold}/V-features.npz')
        self._record(outer_fold=fold,actor='frozen_secondary_fit_worker',roles=list(_ACTORS['frozen_secondary_fit_worker']))
        return TransferFitView(parts['F'],parts['G'],validation['X'],validation['row_id'],
            tuple(self.manifest['categorical_features']),self.manifest['manifest_hash'],split['split_hash'])

    def coordinator_validation(self,fold):
        import numpy as np
        self._split(fold);data=self._store._read(f'outer-{fold}/V-features.npz');labels=self._store._read(f'coordinator/outer-{fold}/V-labels.npz')
        if not np.array_equal(data['row_id'],labels['row_id']):raise ValueError('Released V features/labels row identity differs')
        observation=self._observation(dict(data,y=labels['y']),fold,'V')
        self._record(outer_fold=fold,actor='frozen_secondary_V_coordinator',roles=list(_ACTORS['frozen_secondary_V_coordinator']))
        return observation

    def labels(self,fold,role):
        if role!='V':raise PermissionError('Global primary barrier does not authorize T; released labels are V coordinator only')
        return self.coordinator_validation(fold).y


class ReleasedTransferStore:
    """Load real native/model gates before any secondary metadata, arrays or log."""
    def __init__(self,dataset,*,barrier_reference,qualification_reference):
        from .practical_global_freeze_v2 import load_global_primary_barrier
        barrier=load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference)
        payload=barrier.payload
        if (payload.get('secondary_development_access_authorized') is not True
                or payload.get('protected_access_authorized') is not False
                or type(dataset) is not str or dataset not in payload['prepared_datasets']):
            raise ValueError('Actual global primary barrier and fixed dataset required')
        entry=payload['prepared_datasets'][dataset];prefix=payload['frozen_primary_access_prefixes'][dataset]
        from .primary_search_worker_v2 import _path,_ROOT
        root=_path(entry['root']);manifest_path=root/'manifest.json'
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=entry['manifest_file_sha256']:
            raise ValueError('Actual frozen prepared manifest changed')
        binding=dict(protocol_sha256=barrier_reference['sha256'],source_sha256=payload['source_sha256'],
            manifest_file_sha256=entry['manifest_file_sha256'],dataset=dataset,
            manifest_hash=prefix['manifest_hash'],created_utc_ns=payload['created_utc_ns'])
        journal=_ROOT/'artifacts/research-v2/released-access'/payload['expected_practical_plan_hash']/(dataset+'.jsonl')
        def record(**kwargs):return _record_access(journal,binding=binding,manifest=self.manifest,**kwargs)
        self._roles=_ReleasedRoles(root,expected_manifest_sha256=entry['manifest_file_sha256'],record=record)
        self.root=root

    @property
    def manifest(self):return self._roles.manifest

    def development(self,fold):return self._roles.development(fold)
    def coordinator_validation(self,fold):return self._roles.coordinator_validation(fold)
    def labels(self,fold,role):return self._roles.labels(fold,role)
