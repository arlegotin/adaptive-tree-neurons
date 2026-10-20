"""Full-size recovery is a numerical capability, never a quality result."""
import importlib.util
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_recovery_proof_v2') is not None,'Missing closed full-size recovery proof'
    from treeneuron.research import native_recovery_proof_v2
    return native_recovery_proof_v2


@pytest.fixture(scope='module')
def actual_recovery(tmp_path_factory):
    result=api().qualify_fullsize_node_recovery(output=tmp_path_factory.mktemp('fullsize-recovery')/'execution',interpreter=sys.executable)
    assert result['status']=='completed_recovery_qualification',result
    return result


def test_actual_fullsize_recovery_has_exact_state_and_paid_attempt_evidence_without_new_fit_or_prediction(actual_recovery,monkeypatch):
    from treeneuron.research.node_v2 import NodeV2Predictor,NodeTransform
    def forbidden(*a,**kw):raise AssertionError('Recovery proof reader cannot fit, predict or refit quantiles')
    monkeypatch.setattr(NodeV2Predictor,'fit',forbidden);monkeypatch.setattr(NodeV2Predictor,'predict',forbidden)
    monkeypatch.setattr(NodeTransform,'fit',forbidden)
    proof=api().load_fullsize_node_recovery(actual_recovery['reference'])
    assert proof['recovery_qualified'] is True and proof['state_bytes_independently_compared'] is True
    assert proof['trees']==2048 and proof['depth']==6 and proof['effective_batch']==512
    assert proof['reference_updates']==105 and proof['interrupted_applied_updates']==102 and proof['resume_step']==100
    assert proof['resumed_new_updates']==5 and proof['total_applied_updates']==212 and proof['paid_attempts']==3
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['protected_access_authorized'] is False


def test_closed_recovery_inventory_cannot_be_replaced_by_an_earlier_unbound_proof(actual_recovery):
    reference=dict(actual_recovery['reference'],inventory_sha256='0'*64)
    with pytest.raises(ValueError,match='inventory|bytes|commitment'):
        api().load_fullsize_node_recovery(reference)


@pytest.mark.parametrize('fault',['state_digest','recovery_counter','model_dtype','interpreter_join'])
def test_recommitted_inconsistent_fullsize_recovery_proofs_are_rejected(actual_recovery,fault):
    import json,hashlib
    import numpy as np
    from treeneuron.config import canonical_hash
    from treeneuron.research.native_node_workload_v2 import _metadata
    from treeneuron.research.node_v2 import _array_identity
    from treeneuron.research.workqueue import _inventory
    module=api();root=Path(actual_recovery['reference']['root']);outer=root/'result.json';result=root/'workload/result.json'
    snapshots=[root/'workload'/name for name in ('reference-snapshot','resumed-snapshot')]
    paths=[outer,result,root/'process/process.json',root/'process/started.json',
        *[s/name for s in snapshots for name in ('manifest.json','arrays.npz')]]
    before={p:p.read_bytes() for p in paths}
    try:
        value=json.loads(result.read_text())
        if fault=='state_digest':value['state_digest_without_elapsed_times']='0'*64
        elif fault=='recovery_counter':value['resumed_final_updates']=104
        elif fault=='interpreter_join':
            process=json.loads((root/'process/process.json').read_text());process['argv'][0]='/bin/echo'
            start=json.loads((root/'process/started.json').read_text());start['argv']=process['argv']
            (root/'process/started.json').write_text(json.dumps(start))
            for ref in process['files']:
                p=root/'process'/ref['path'];ref.update(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
            (root/'process/process.json').write_text(json.dumps(process))
        else:
            for snapshot in snapshots:
                path=snapshot/'manifest.json';manifest=json.loads(path.read_text())
                with np.load(snapshot/'arrays.npz',allow_pickle=False) as saved:arrays={k:saved[k].copy() for k in saved.files}
                changed=0
                for key,array in arrays.items():
                    if array.dtype==np.dtype('uint8') and array.shape==():
                        arrays[key]=array.astype(np.int64);manifest['arrays'][key]=_array_identity(arrays[key]);changed+=1
                assert changed>=3
                np.savez(snapshot/'arrays.npz',**arrays);manifest['array_file']['sha256']=hashlib.sha256((snapshot/'arrays.npz').read_bytes()).hexdigest()
                path.write_text(json.dumps(manifest));state=_metadata(manifest['payload'],manifest['arrays'],set())
                value['state_digest_without_elapsed_times']=module._state_digest(state,manifest['arrays'])
            value['final_snapshot_sha256']=hashlib.sha256((snapshots[1]/'manifest.json').read_bytes()).hexdigest()
        result.write_text(json.dumps(value));record=json.loads(outer.read_text())
        record['workload_result_sha256']=hashlib.sha256(result.read_bytes()).hexdigest()
        record['process_sha256']=hashlib.sha256((root/'process/process.json').read_bytes()).hexdigest();outer.write_text(json.dumps(record))
        reference=dict(root=str(root),inventory_sha256=canonical_hash(_inventory(root)))
        with pytest.raises(ValueError,match='state|recovery|Recovery|geometry|interpreter|runtime|invocation'):
            module.load_fullsize_node_recovery(reference)
    finally:
        for p,raw in before.items():p.write_bytes(raw)
        assert canonical_hash(_inventory(root))==actual_recovery['reference']['inventory_sha256']
