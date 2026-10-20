"""A metadata digest cannot substitute for the complete frozen support law."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.practical_support_lock_v2') is not None,'Missing actual practical support lock'
    from treeneuron.research import practical_support_lock_v2
    return practical_support_lock_v2


def reference(path):return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_sealed_law_covers_every_dataset_family_and_matched_random_support(tmp_path):
    module=api();path=tmp_path/'support.json';ref=module.seal_practical_support_lock(path)
    assert ref==reference(path);body=module.load_practical_support_lock(ref)
    assert set(body['families'])=={'airfoil','concrete','ccpp','parkinsons','bike','energy'}
    for families in body['families'].values():
        assert set(families)=={f'P{i:02}' for i in range(14)}
        assert families['P00']['support']==families['P01']['support']
        assert len(families['P13']['finite_configurations'])==29
        assert sum(len(v['anchors']) for v in families.values())==20
    assert body['families']['bike']['P07']['feature_schema']['categorical']==['season','mnth','hr','weekday','weathersit','holiday','workingday']
    assert body['native_workload_qualified'] is body['protected_access_authorized'] is False


def test_existing_lock_is_never_overwritten(tmp_path):
    module=api();path=tmp_path/'support.json';ref=module.seal_practical_support_lock(path);before=path.read_bytes()
    with pytest.raises(ValueError,match='exists|fresh|overwrite'):module.seal_practical_support_lock(path)
    assert path.read_bytes()==before and module.load_practical_support_lock(ref)


@pytest.mark.parametrize('fault',['source','core','missing_family','partial_anchors','paired_support','unexpected_field','authority'])
def test_self_rehashed_changed_support_law_cannot_be_admitted(tmp_path,fault):
    module=api();path=tmp_path/'support.json';module.seal_practical_support_lock(path);body=json.loads(path.read_text())
    if fault=='source':body['source_sha256']='0'*64
    elif fault=='core':body['core_manifest_sha256']='0'*64
    elif fault=='missing_family':body['families']['energy'].pop('P08')
    elif fault=='partial_anchors':body['families']['energy']['P05']['anchors'].pop()
    elif fault=='paired_support':
        from treeneuron.config import canonical_hash
        spec=body['families']['energy']['P00'];spec['support']['lr']['low']=5e-4
        spec['support_hash']=canonical_hash({k:v for k,v in spec.items() if k!='support_hash'})
    elif fault=='unexpected_field':body['best_dataset']='energy'
    else:body['native_workload_qualified']=True
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='law|source|core|support|schema'):
        module.load_practical_support_lock(reference(path))


def test_old_external_digest_and_relative_reference_fail(tmp_path):
    module=api();path=tmp_path/'support.json';ref=module.seal_practical_support_lock(path)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):module.load_practical_support_lock(ref)
    with pytest.raises(ValueError):module.load_practical_support_lock(dict(path='support.json',sha256=ref['sha256']))
