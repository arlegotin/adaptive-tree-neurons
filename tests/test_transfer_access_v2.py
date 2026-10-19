"""Recorded access evidence is read-only and never grants a protected role."""
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash


def api():
    from treeneuron.research import transfer_access_v2
    return transfer_access_v2


def setup(tmp_path,dataset='energy'):
    from treeneuron.research.transfer import TransferStore
    root=tmp_path/'sealed';root.mkdir()
    splits=[]
    for fold in range(5):
        split=dict(schema='research/v2-transfer-split',dataset=dataset,outer_fold=fold,primary=fold==(4 if dataset=='bike' else 0))
        split['split_hash']=canonical_hash(split);splits.append(split)
    manifest=dict(schema='research/v2-sealed-transfer',dataset=dataset,splits=splits,
        raw_identity_verified=False,protected_labels_read_for_modeling=False)
    manifest['manifest_hash']=canonical_hash(manifest)
    (root/'manifest.json').write_text(json.dumps(manifest))
    return root,hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),TransferStore(root)


def append(root,**changes):
    records=(root/'access.jsonl').read_text().splitlines() if (root/'access.jsonl').exists() else []
    previous=json.loads(records[-1])['event_hash'] if records else '0'*64
    manifest=json.loads((root/'manifest.json').read_text())
    event=dict(schema='research/v2-transfer-access',outer_fold=0,roles=['F_labels','G_labels','V_features'],
        actor='declared_pre_freeze_coordinator',utc_ns=123,manifest_hash=manifest['manifest_hash'],previous_hash=previous)
    event.update(changes);event['event_hash']=canonical_hash(event)
    with (root/'access.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')


def test_existing_primary_API_events_authenticate_without_arrays_or_authority(tmp_path,monkeypatch):
    module=api();root,digest,store=setup(tmp_path)
    store._access(0,['F_labels','G_labels','V_features'])
    store._access(0,['V_labels','V_features','V_group_or_day_ids'])
    before={p.name:p.read_bytes() for p in root.iterdir()}
    def forbidden(*args,**kwargs):raise AssertionError('No arrays/labels are needed for access history')
    import numpy as np
    monkeypatch.setattr(np,'load',forbidden)
    result=module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
    assert result['status']=='valid_recorded_primary_access' and result['event_count']==2
    assert result['role_counts']['V_features']==2 and result['role_counts']['V_labels']==1
    assert result['primary_fold']==0 and result['shared_lease_held'] is True
    assert result['protected_access_authorized'] is result['freeze_readiness_qualified'] is False
    assert result['unlogged_filesystem_access_authenticated'] is False
    assert before=={p.name:p.read_bytes() for p in root.iterdir()}


def test_absent_log_remains_absent_and_is_only_an_unleased_snapshot(tmp_path):
    module=api();root,digest,store=setup(tmp_path)
    result=module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
    assert result['event_count']==0 and result['access_file_sha256'] is None
    assert result['shared_lease_held'] is False and result['status']=='access_log_absent_snapshot'
    assert not (root/'access.jsonl').exists()


@pytest.mark.parametrize('change',[dict(outer_fold=1),dict(outer_fold=True),dict(roles=['T_labels']),
    dict(roles=['V_labels','T_features']),dict(roles=['F_labels','F_labels']),dict(roles=[]),
    dict(actor='unqualified_post_freeze_actor'),dict(manifest_hash='0'*64),dict(utc_ns=True),dict(extra=True)])
def test_even_rehashed_invalid_or_forbidden_records_fail(tmp_path,change):
    module=api();root,digest,store=setup(tmp_path);append(root,**change)
    before=(root/'access.jsonl').read_bytes()
    with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
    assert (root/'access.jsonl').read_bytes()==before


def test_corrupt_chain_partial_and_duplicate_JSON_cannot_pass(tmp_path):
    module=api();root,digest,store=setup(tmp_path);append(root)
    path=root/'access.jsonl';good=path.read_text()
    for changed in (good.replace('F_labels','V_labels'),good.rstrip('\n'),good.replace('"utc_ns": 123','"utc_ns": 123,"utc_ns": 123')):
        path.write_text(changed)
        with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)


def test_shared_snapshot_excludes_active_append_and_preserves_metadata(tmp_path):
    module=api();root,digest,store=setup(tmp_path);append(root)
    path=root/'access.jsonl';before=path.read_bytes()
    with path.open('r') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(PermissionError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
    with module.primary_access_snapshot(root,expected_manifest_sha256=digest) as result:
        with path.open('r') as stream:
            with pytest.raises(BlockingIOError):fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert path.read_bytes()==before


def test_external_manifest_and_access_frontiers_reject_drift(tmp_path):
    module=api();root,digest,store=setup(tmp_path);append(root)
    result=module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
    append(root)
    with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest,
        expected_access_sha256=result['access_file_sha256'])
    with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256='0'*64)


def test_changed_file_during_shared_snapshot_is_detected(tmp_path):
    module=api();root,digest,store=setup(tmp_path);append(root)
    with pytest.raises(ValueError):
        with module.primary_access_snapshot(root,expected_manifest_sha256=digest):
            # A direct filesystem writer can ignore flock: detection is required.
            append(root)


def test_bike_primary_is_four_and_zero_remains_forbidden(tmp_path):
    module=api();root,digest,store=setup(tmp_path,dataset='bike')
    store._access(4,['F_labels','G_labels','V_features'])
    assert module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)['primary_fold']==4
    append(root,outer_fold=0)
    with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)


@pytest.mark.parametrize('dataset',[[],{}])
def test_malformed_dataset_identity_is_rejected_as_invalid_metadata(tmp_path,dataset):
    module=api();root,digest,store=setup(tmp_path)
    path=root/'manifest.json';manifest=json.loads(path.read_text());manifest['dataset']=dataset
    manifest['manifest_hash']=canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})
    path.write_text(json.dumps(manifest));digest=hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):module.audit_pre_freeze_access(root,expected_manifest_sha256=digest)
