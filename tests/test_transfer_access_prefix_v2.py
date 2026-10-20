"""Frozen recorded prefixes allow only later valid primary-only API events."""
import copy
import hashlib

import pytest

from test_transfer_access_v2 import setup,append


def api():
    from treeneuron.research import transfer_access_v2
    assert hasattr(transfer_access_v2,'frozen_primary_access_prefix'),'Missing frozen primary access prefix'
    return transfer_access_v2


def test_later_primary_audits_keep_the_original_frozen_prefix_and_no_arrays_are_opened(tmp_path,monkeypatch):
    module=api();root,digest,store=setup(tmp_path)
    store._access(0,['V_labels','V_features','V_group_or_day_ids']);before=(root/'access.jsonl').read_bytes()
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest) as record:prefix=copy.deepcopy(record)
    assert prefix['prefix_bytes']==len(before) and prefix['prefix_sha256']==hashlib.sha256(before).hexdigest()
    assert prefix['prefix_event_count']==1 and not prefix['protected_access_authorized']
    store._access(0,['V_labels','V_features','V_group_or_day_ids'])
    assert hashlib.sha256((root/'access.jsonl').read_bytes()).hexdigest()!=prefix['prefix_sha256']
    import numpy as np
    def forbidden(*args,**kwargs):raise AssertionError('No arrays or labels are required by a log-prefix audit')
    monkeypatch.setattr(np,'load',forbidden)
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest,prefix=prefix) as checked:
        assert checked==prefix and checked['unlogged_filesystem_access_authenticated'] is False


def test_absent_prefix_stays_absent_at_capture_and_allows_later_valid_primary_log(tmp_path):
    module=api();root,digest,store=setup(tmp_path)
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest) as record:prefix=copy.deepcopy(record)
    assert not (root/'access.jsonl').exists() and prefix['access_log_absent_at_freeze']
    assert prefix['prefix_bytes']==0 and prefix['prefix_event_count']==0
    store._access(0,['F_labels'])
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest,prefix=prefix) as checked:assert checked==prefix


@pytest.mark.parametrize('fault',['hash','count','partial','boolean_size','boolean_count','boolean_fold','integer_flag','authority','extra','false_absence'])
def test_changed_or_forged_prefix_cannot_be_reaccepted(tmp_path,fault):
    module=api();root,digest,store=setup(tmp_path);append(root)
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest) as record:prefix=copy.deepcopy(record)
    if fault=='hash':prefix['prefix_sha256']='0'*64
    elif fault=='count':prefix['prefix_event_count']+=1
    elif fault=='partial':prefix['prefix_bytes']-=1
    elif fault=='boolean_size':prefix['prefix_bytes']=True
    elif fault=='boolean_count':prefix['prefix_event_count']=True
    elif fault=='boolean_fold':prefix['primary_fold']=False
    elif fault=='integer_flag':prefix['T_accessed']=0
    elif fault=='authority':prefix['protected_access_authorized']=True
    elif fault=='extra':prefix['secondary_release_authorized']=True
    else:prefix['access_log_absent_at_freeze']=True
    with pytest.raises(ValueError):
        with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest,prefix=prefix):pass


@pytest.mark.parametrize('changes',[dict(outer_fold=1),dict(roles=['T_labels']),dict(roles=['T_features'])])
def test_valid_old_prefix_never_admits_forbidden_events_in_the_later_suffix(tmp_path,changes):
    module=api();root,digest,store=setup(tmp_path);append(root)
    with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest) as record:prefix=copy.deepcopy(record)
    append(root,**changes)
    with pytest.raises(ValueError):
        with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest,prefix=prefix):pass


def test_prefix_context_excludes_appends_and_detects_a_writer_ignoring_the_lease(tmp_path):
    module=api();root,digest,store=setup(tmp_path);append(root)
    import fcntl
    with pytest.raises(ValueError):
        with module.frozen_primary_access_prefix(root,expected_manifest_sha256=digest):
            with (root/'access.jsonl').open('r') as stream:
                with pytest.raises(BlockingIOError):fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            append(root)
