"""Uncapped native fits retain owned-process guards and immutable recovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from treeneuron.config import canonical_hash
from test_method_worker import request_fixture,write_request
from test_node_v2 import native_policy


def api():
    assert importlib.util.find_spec('treeneuron.research.native_fit_v2') is not None,'Missing native-fit supervisor'
    from treeneuron.research import native_fit_v2
    return native_fit_v2


@pytest.mark.parametrize('family',['P13','P03'])
def test_actual_uncapped_native_process_has_no_deadline_and_keeps_its_source_and_rows(tmp_path,family):
    module=api();path,value=request_fixture(tmp_path/'input');value['budget']={'native_stopping':True}
    if family=='P03':
        from test_ordinary_neural_contract import config
        value['family_id']=family;value['config']=config(family,'engineering_fixture',max_epochs=2,patience=2)
    expected=write_request(path,value);out=tmp_path/'native'
    result=module.run_native_fit(path,expected_request_sha256=expected,output=out,interpreter=sys.executable,rss_limit_bytes=1024**3)
    assert result['status']=='completed_ineligible' and result['quality_eligible'] is result['T_accessed'] is False
    process=json.loads((out/'process/process.json').read_text())
    assert process['timeout_ns'] is None and process['child_reaped'] is True and process['returncode']==0
    assert process['identity']['request_sha256']==expected
    assert process['environment_overrides']['MPLCONFIGDIR']==str(out/'cache/mpl')
    assert process['environment_overrides']['XDG_CACHE_HOME']==str(out/'cache/xdg')
    with np.load(out/'worker/predictions.npz',allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['row_id'],np.arange(200,205))
        if family=='P13':np.testing.assert_array_equal(archive['prediction'],np.full(5,14.))
    assert result['recovery_reference']['request_sha256']==expected
    assert result['recovery_reference']['inventory_sha256']


def reference(root,request_sha='a'*64,checkpoint=True,active=False):
    root.mkdir();files={
        'result.json':json.dumps(dict(schema='research/v2-native-fit-result',child_reaped=not active,
            status='running' if active else 'interrupted_recoverable',request_sha256=request_sha)).encode(),
        'training-checkpoints/binding.json':b'{"source":"fixture-integrity-only"}',
        'training-checkpoints/arrays.bin':b'raw fixture bytes; no model qualification'}
    if checkpoint:files['training-checkpoints/latest.json']=b'{"step":4}'
    for name,value in files.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value)
    # External commitment independently built from the known literal fixture.
    inventory=dict(directories=['training-checkpoints'],files=[dict(path=k,bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
        for k,v in sorted(files.items())])
    return dict(root=str(root.resolve()),inventory_sha256=canonical_hash(inventory),request_sha256=request_sha)


def test_recovery_copies_verified_bytes_into_independent_inodes_without_mutating_old_attempt(tmp_path):
    module=api();ref=reference(tmp_path/'previous');target=tmp_path/'new/checkpoints'
    before={p:p.read_bytes() for p in Path(ref['root']).rglob('*') if p.is_file()}
    result=module.clone_native_checkpoint(ref,target,expected_request_sha256='a'*64)
    assert result['resume'] is True
    for src in (Path(ref['root'])/'training-checkpoints').rglob('*'):
        if src.is_file():
            dst=target/src.relative_to(Path(ref['root'])/'training-checkpoints')
            assert dst.read_bytes()==src.read_bytes() and dst.stat().st_ino!=src.stat().st_ino
    assert before=={p:p.read_bytes() for p in Path(ref['root']).rglob('*') if p.is_file()}


@pytest.mark.parametrize('change',['request','late_change','active','self_destination'])
def test_recovery_rejects_changed_request_or_old_bytes_before_copying(tmp_path,change):
    module=api();ref=reference(tmp_path/'previous',active=change=='active');target=tmp_path/'new/checkpoints';expected='a'*64
    if change=='request':expected='b'*64
    elif change=='late_change':(tmp_path/'previous/training-checkpoints/arrays.bin').write_bytes(b'changed')
    elif change=='self_destination':target=tmp_path/'previous/training-checkpoints/copy'
    with pytest.raises(ValueError):module.clone_native_checkpoint(ref,target,expected_request_sha256=expected)
    assert not target.exists()


def test_interruption_before_first_checkpoint_is_an_explicit_same_seed_restart(tmp_path):
    module=api();ref=reference(tmp_path/'previous',checkpoint=False);target=tmp_path/'new/checkpoints'
    result=module.clone_native_checkpoint(ref,target,expected_request_sha256='a'*64)
    assert result['resume'] is False and result['reason']=='no_committed_checkpoint_same_seed_restart'
    assert not target.exists()


def test_full_NODE_native_launch_always_requests_recovery_and_still_enforces_owned_RSS(tmp_path):
    module=api();path,value=request_fixture(tmp_path/'input');value['family_id']='P12';value['budget']={'native_stopping':True}
    value['config']=dict(schema='research/v2-node-config',family='P12',mode='native_anchor',execution='bounded_memory_v1',
        feature_schema=value['config']['feature_schema'],parameters={})
    result=module.run_native_fit(path,expected_request_sha256=write_request(path,value),output=tmp_path/'native',
        interpreter=sys.executable,rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['quality_eligible'] is False
    p=json.loads((tmp_path/'native/process/process.json').read_text())
    assert p['timeout_ns'] is None and p['child_reaped'] is True
    assert '--training-checkpoints' in p['argv']


def test_native_supervisor_refuses_capped_request_instead_of_silently_changing_it(tmp_path):
    module=api();path,value=request_fixture(tmp_path/'input')
    with pytest.raises(ValueError):
        module.run_native_fit(path,expected_request_sha256=write_request(path,value),output=tmp_path/'native',interpreter=sys.executable)
    assert not (tmp_path/'native').exists()


def test_external_owned_child_termination_retains_a_recoverable_attempt(tmp_path):
    import os
    import signal
    import threading
    import time
    module=api();path,value=request_fixture(tmp_path/'input')
    from test_ordinary_neural_contract import config
    value['family_id']='P03';value['budget']={'native_stopping':True}
    value['config']=config('P03','engineering_fixture',max_epochs=2,patience=2)
    expected=write_request(path,value);output=tmp_path/'native';observed=[]
    def terminate_only_owned_child():
        limit=time.monotonic()+20;spawn=output/'process/spawned.json'
        while time.monotonic()<limit:
            if spawn.exists():
                owned=json.loads(spawn.read_text())
                assert owned['new_session'] is True and owned['child_pid']==owned['child_pgid']
                os.kill(owned['child_pid'],signal.SIGTERM);observed.append(owned['child_pid']);return
            time.sleep(.01)
    thread=threading.Thread(target=terminate_only_owned_child,daemon=True);thread.start()
    try:
        result=module.run_native_fit(path,expected_request_sha256=expected,output=output,interpreter=sys.executable)
    finally:thread.join(timeout=21)
    assert observed and result['status']=='interrupted_recoverable'
    assert result['child_reaped'] is True and result['quality_eligible'] is False
    assert result['recovery_reference']['request_sha256']==expected


def test_copied_real_NODE_checkpoint_resumes_the_exact_uninterrupted_state(tmp_path,native_policy,monkeypatch):
    from treeneuron.research.node_v2 import NodeV2Predictor
    from treeneuron.research import node_checkpoints
    from treeneuron.research.workqueue import _inventory
    from test_node_v2 import request
    from test_node_training_recovery import same
    module=api();old=tmp_path/'previous';old.mkdir()
    reference_model=NodeV2Predictor();reference_model.fit(request(tmp_path/'reference',7))
    real=node_checkpoints._publish_pointer
    def interrupt(path,pointer):
        if pointer['step']==4:raise KeyboardInterrupt('Controlled copy-recovery fixture before pointer publication')
        real(path,pointer)
    monkeypatch.setattr(node_checkpoints,'_publish_pointer',interrupt)
    with pytest.raises(KeyboardInterrupt):
        NodeV2Predictor().fit(request(old/'fit',7),training_checkpoints=old/'training-checkpoints')
    monkeypatch.setattr(node_checkpoints,'_publish_pointer',real)
    (old/'result.json').write_text(json.dumps(dict(schema='research/v2-native-fit-result',status='interrupted_recoverable',
        child_started=False,request_sha256='a'*64,scope='Direct tiny capped fixture, no full native qualification')))
    ref=dict(root=str(old),inventory_sha256=canonical_hash(_inventory(old)),request_sha256='a'*64)
    before=_inventory(old);new=tmp_path/'copied'
    module.clone_native_checkpoint(ref,new,expected_request_sha256='a'*64)
    resumed=NodeV2Predictor();resumed.fit(request(tmp_path/'resumed',7),training_checkpoints=new,resume=True)
    same(reference_model._state,resumed._state,native_policy)
    np.testing.assert_array_equal(reference_model.predict(request(tmp_path/'query',7).X_select),
        resumed.predict(request(tmp_path/'query-again',7).X_select))
    assert _inventory(old)==before
