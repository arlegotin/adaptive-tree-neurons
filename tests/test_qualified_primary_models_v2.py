"""Byte joins stay unqualified; public model sealing needs genuine searches."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.qualified_primary_models_v2'), 'Missing strong primary model barrier'
    from treeneuron.research import qualified_primary_models_v2
    return qualified_primary_models_v2


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def snapshot(root):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    root.mkdir();(root/'manifest.json').write_text('{"fixture":"byte-only-snapshot"}')
    (root/'arrays.bin').write_bytes(bytes(range(32)))
    return ref(root/'manifest.json'),canonical_hash(_inventory(root))


def test_exact_metadata_and_all_snapshot_files_join_without_any_native_authority(tmp_path):
    model,digest=snapshot(tmp_path/'snapshot')
    result=api().inspect_selected_snapshot_binding(selected_model=model,expected_metadata=model,
        snapshot_root=tmp_path/'snapshot',expected_inventory_sha256=digest)
    assert result['metadata']==model and result['snapshot_inventory_sha256']==digest
    assert result['native_workload_qualified'] is result['protected_access_authorized'] is False


@pytest.mark.parametrize('fault',['stderr','different_snapshot','tensor_bytes','extra_file',
    'metadata_hash','inventory_hash','relative_path','symlink'])
def test_any_winner_artifact_is_insufficient_for_the_selected_model(tmp_path,fault):
    model,digest=snapshot(tmp_path/'snapshot');chosen=model.copy();expected=model.copy()
    if fault=='stderr':
        path=tmp_path/'stderr.log';path.write_text('eligible winner log');chosen=ref(path)
    elif fault=='different_snapshot':chosen,_=snapshot(tmp_path/'other')
    elif fault=='tensor_bytes':(tmp_path/'snapshot/arrays.bin').write_bytes(b'changed')
    elif fault=='extra_file':(tmp_path/'snapshot/uncommitted.bin').write_bytes(b'changed')
    elif fault=='metadata_hash':expected['sha256']='0'*64
    elif fault=='inventory_hash':digest='0'*64
    elif fault=='relative_path':chosen['path']='manifest.json'
    else:
        (tmp_path/'linked.json').symlink_to(tmp_path/'snapshot/manifest.json');chosen['path']=str(tmp_path/'linked.json')
    with pytest.raises(ValueError):
        api().inspect_selected_snapshot_binding(selected_model=chosen,expected_metadata=expected,
            snapshot_root=tmp_path/'snapshot',expected_inventory_sha256=digest)


@pytest.mark.parametrize('operation',['seal','load'])
def test_false_registry_is_rejected_before_inventory_or_search_access(tmp_path,monkeypatch,operation):
    module=api()
    from treeneuron.research import primary_selection_v2
    def forbidden(*args,**kwargs):raise AssertionError('False registry must never reach inventory')
    monkeypatch.setattr(primary_selection_v2,'load_primary_selection_inventory',forbidden)
    arguments=dict(practical_plan=None,expected_practical_plan_hash='a'*64,
        support_reference=dict(path=str(tmp_path/'not-opened-support.json'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64))
    with pytest.raises((ValueError,FileNotFoundError)):
        if operation=='seal':
            module.seal_qualified_primary_models(**arguments,
                primary_inventory_reference=dict(path=str(tmp_path/'not-opened-inventory.json'),sha256='d'*64),
                search_proofs={},output=tmp_path/'not-created.json')
        else:
            module.load_qualified_primary_models(**arguments,path=tmp_path/'not-opened-seal.json',expected_file_sha256='e'*64)
    assert not (tmp_path/'not-created.json').exists()


@pytest.mark.parametrize('fault',[None,'extra_file','changed_loser','symlink'])
def test_full_search_inventory_cannot_gain_uncommitted_evidence_after_audit(tmp_path,fault):
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    root=tmp_path/'search';root.mkdir();(root/'losing-result.json').write_text('{"fixture":"not-qualified"}')
    expected=canonical_hash(_inventory(root))
    if fault=='extra_file':(root/'unpaid-result.json').write_text('{}')
    elif fault=='changed_loser':(root/'losing-result.json').write_text('{"changed":true}')
    elif fault=='symlink':(root/'alias').symlink_to(root/'losing-result.json')
    if fault is None:
        result=api().inspect_search_inventory_binding(search_directory=root,expected_inventory_sha256=expected)
        assert result['inventory_sha256']==expected and result['native_workload_qualified'] is False
    else:
        with pytest.raises(ValueError):
            api().inspect_search_inventory_binding(search_directory=root,expected_inventory_sha256=expected)
