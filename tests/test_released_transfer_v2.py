"""Toy released secondary data engines never fabricate native qualification."""
import copy
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.released_transfer_v2'),'Missing released secondary transfer API'
    from treeneuron.research import released_transfer_v2
    return released_transfer_v2


@pytest.fixture
def toy(tmp_path):
    from treeneuron.data.tabular import table_from_frame,DATASETS
    from treeneuron.research.transfer import seal_table,TransferStore
    schema=DATASETS['energy'];frame=pd.DataFrame({n:np.arange(24,dtype=float) for n in schema.raw_columns})
    for n in ('X1','X2','X3','X4','X5'):frame[n]=np.repeat(np.arange(12),2)
    root=tmp_path/'toy';manifest=seal_table(table_from_frame('energy',frame),root)
    assert not manifest['raw_identity_verified']
    return root,hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),TransferStore(root)


def test_toy_secondary_engine_preserves_split_ids_and_never_hands_V_or_T_labels_to_fit_view(toy,monkeypatch):
    module=api();root,digest,legacy=toy;events=[]
    engine=module._ReleasedRoles(root,expected_manifest_sha256=digest,record=lambda **kw:events.append(kw))
    with pytest.raises(PermissionError):legacy.development(1)
    original=np.load
    def no_T(path,*args,**kwargs):
        assert '/protected/' not in str(path),'Secondary engine must never decode a T archive'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',no_T)
    view=engine.development(1);validation=engine.coordinator_validation(1)
    np.testing.assert_array_equal(view.validation_row_ids,validation.row_id)
    np.testing.assert_array_equal(view.validation_features,validation.X)
    assert not set(view.F.row_id)&set(view.G.row_id) and not set(view.F.group_id)&set(validation.group_id)
    assert view.split_hash==legacy.manifest['splits'][1]['split_hash']
    assert view.F.feature_names==tuple(legacy.manifest['feature_names'])
    with pytest.raises(PermissionError):_ = view.V
    with pytest.raises(PermissionError):_ = view.T
    with pytest.raises(PermissionError):engine.labels(1,'T')
    assert [e['actor'] for e in events]==['frozen_secondary_fit_worker','frozen_secondary_V_coordinator']
    assert not (root/'access.jsonl').exists()
    with pytest.raises(PermissionError):engine.development(0)


def test_changed_role_bytes_fail_before_a_view_or_access_record_is_returned(toy):
    module=api();root,digest,legacy=toy;events=[]
    engine=module._ReleasedRoles(root,expected_manifest_sha256=digest,record=lambda **kw:events.append(kw))
    (root/'outer-1/F.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):engine.development(1)
    assert events==[]


def binding(digest,manifest):
    return dict(protocol_sha256='a'*64,source_sha256='b'*64,manifest_file_sha256=digest,
        dataset='energy',manifest_hash=manifest['manifest_hash'],created_utc_ns=1)


def test_secondary_journal_is_separate_and_rejects_another_protocol_or_forbidden_roles(toy,tmp_path):
    module=api();root,digest,legacy=toy;path=tmp_path/'journal.jsonl';meta=binding(digest,legacy.manifest)
    args=dict(binding=meta,manifest=legacy.manifest,outer_fold=1,actor='frozen_secondary_fit_worker',roles=['F_labels','G_labels','V_features'])
    module._record_access(path,**args);module._record_access(path,**dict(args,actor='frozen_secondary_V_coordinator',roles=['V_labels','V_features','V_group_or_day_ids']))
    before=path.read_bytes();assert len(before.splitlines())==2 and not (root/'access.jsonl').exists()
    for changed in (dict(args,binding=dict(meta,protocol_sha256='c'*64)),dict(args,outer_fold=0),dict(args,roles=['T_labels']),dict(args,actor='unknown')):
        with pytest.raises(ValueError):module._record_access(path,**changed)
        assert path.read_bytes()==before


def test_corrupt_secondary_journal_is_not_overwritten_or_reset(toy,tmp_path):
    module=api();root,digest,legacy=toy;path=tmp_path/'journal.jsonl'
    args=dict(binding=binding(digest,legacy.manifest),manifest=legacy.manifest,outer_fold=1,
        actor='frozen_secondary_fit_worker',roles=['F_labels','G_labels','V_features'])
    module._record_access(path,**args);raw=path.read_bytes();path.write_bytes(raw.replace(b'F_labels',b'T_labels'))
    before=path.read_bytes()
    with pytest.raises(ValueError):module._record_access(path,**args)
    assert path.read_bytes()==before


def test_public_released_store_rejects_false_global_qualification_before_any_data_or_log(tmp_path,monkeypatch):
    module=api()
    from treeneuron.research.transfer import TransferStore
    def forbidden(*args,**kwargs):raise AssertionError('False qualification cannot inspect transfer data')
    monkeypatch.setattr(TransferStore,'__init__',forbidden)
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.ReleasedTransferStore('energy',barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='b'*64))
    assert 'forbidden-global' not in str(error.value) and list(tmp_path.iterdir())==[]
