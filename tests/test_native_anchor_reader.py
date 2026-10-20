"""Closed anchor proofs must remain checked when joined into qualification."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from treeneuron.config import canonical_hash
from test_native_anchor_execution import evidence, qualify


def api():
    from treeneuron.research import native_anchor_v2
    assert hasattr(native_anchor_v2, 'load_native_anchor_execution'), 'Missing checked native anchor proof reader'
    return native_anchor_v2


def reference(root):
    from treeneuron.research.workqueue import _inventory
    return dict(root=str(root.resolve()), inventory_sha256=canonical_hash(_inventory(root)))


def prepared(module, tmp_path, evidence):
    args, old, software=evidence()
    out=tmp_path/'anchor'; result=qualify(module,args,old,software,out)
    assert result['native_anchor_execution_authenticated'] is True
    return args,out


def load(module,root,args,**changes):
    kwargs=dict(prepared_root=args['prepared_root'],expected_manifest_sha256=args['expected_manifest_sha256'],
        expected_family='P13',expected_anchor_id='mean',expected_dataset='energy')
    kwargs.update(changes)
    return module.load_native_anchor_execution(reference(root),**kwargs)


def test_actual_closed_anchor_reopens_generation_software_replay_and_independent_primary_arithmetic(tmp_path,evidence):
    module=api();args,root=prepared(module,tmp_path,evidence)
    proof=load(module,root,args)
    assert proof['native_anchor_execution_authenticated'] is True
    assert proof['family_id']=='P13' and proof['anchor_id']=='mean' and proof['dataset']=='energy'
    assert proof['prediction_reload_exact'] is True and proof['validation_recomputed'] is True
    assert proof['V_score_used_for_qualification'] is False
    assert proof['native_workload_qualified'] is proof['quality_eligible'] is proof['T_accessed'] is False


@pytest.mark.parametrize('change',['family','anchor','dataset','source'])
def test_wrong_expected_coordinate_or_changed_source_rejects_before_V_access(tmp_path,evidence,change):
    module=api();args,root=prepared(module,tmp_path,evidence)
    access=Path(args['prepared_root'])/'access.jsonl';before=access.read_bytes();kwargs={}
    if change=='family':kwargs['expected_family']='P03'
    elif change=='anchor':kwargs['expected_anchor_id']='ridge'
    elif change=='dataset':kwargs['expected_dataset']='airfoil'
    else:
        saved=json.loads((root/'result.json').read_text());saved['source_sha256']='0'*64
        (root/'result.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError):load(module,root,args,**kwargs)
    assert access.read_bytes()==before


def test_self_recommitted_wrong_validation_value_does_not_authenticate(tmp_path,evidence):
    module=api();args,root=prepared(module,tmp_path,evidence)
    inspection=root/'inspection/result.json';saved=json.loads(inspection.read_text())
    saved['validation']['score']+=.5;inspection.write_text(json.dumps(saved))
    anchor=json.loads((root/'result.json').read_text())
    anchor['inspection_reference']=reference(root/'inspection')
    (root/'result.json').write_text(json.dumps(anchor))
    with pytest.raises(ValueError,match='arithmetic'):load(module,root,args)


def test_recommitted_failed_replay_cannot_supply_an_anchor_proof(tmp_path,evidence):
    module=api();args,root=prepared(module,tmp_path,evidence)
    from treeneuron.research.workqueue import _inventory
    path=root/'inspection/replay/terminal.json';terminal=json.loads(path.read_text())
    terminal['prediction_reload_exact']=False;path.write_text(json.dumps(terminal))
    inspection=root/'inspection/result.json';saved=json.loads(inspection.read_text())
    saved['replay']=copy.deepcopy(terminal)
    saved['replay_artifact_inventory_sha256']=canonical_hash(_inventory(root/'inspection/replay'))
    inspection.write_text(json.dumps(saved))
    anchor=json.loads((root/'result.json').read_text());anchor['inspection_reference']=reference(root/'inspection')
    (root/'result.json').write_text(json.dumps(anchor))
    access=Path(args['prepared_root'])/'access.jsonl';before=access.read_bytes()
    with pytest.raises(ValueError,match='replay'):load(module,root,args)
    assert access.read_bytes()==before


def test_recommitted_feature_only_replay_must_still_use_the_original_generation_query(tmp_path,evidence):
    import numpy as np
    from treeneuron.research.workqueue import _inventory
    module=api();args,root=prepared(module,tmp_path,evidence)
    replay_root=root/'inspection';query=replay_root/'input/query.npz'
    with np.load(query,allow_pickle=False) as saved:arrays={key:saved[key].copy() for key in saved.files}
    arrays['X'][:,0]+=42
    # Break the staged hard link; the original immutable generation stays intact.
    query.unlink();np.savez(query,**arrays)
    request_path=replay_root/'input/request.json';request=json.loads(request_path.read_text())
    request['query']['sha256']=hashlib.sha256(query.read_bytes()).hexdigest()
    request_path.write_text(json.dumps(request));request_sha=hashlib.sha256(request_path.read_bytes()).hexdigest()
    terminal=json.loads((replay_root/'replay/terminal.json').read_text())
    terminal.update(request_sha256=request_sha,query_sha256=request['query']['sha256'])
    (replay_root/'replay/terminal.json').write_text(json.dumps(terminal))
    process=json.loads((replay_root/'process/process.json').read_text())
    process['identity']['request_sha256']=request_sha
    process['argv'][process['argv'].index('--expected-request-sha256')+1]=request_sha
    (replay_root/'process/process.json').write_text(json.dumps(process))
    inspection=json.loads((replay_root/'result.json').read_text())
    inspection.update(replay=terminal,process=process,
        replay_artifact_inventory_sha256=canonical_hash(_inventory(replay_root/'replay')),
        replay_process_inventory_sha256=canonical_hash(_inventory(replay_root/'process')))
    (replay_root/'result.json').write_text(json.dumps(inspection))
    anchor=json.loads((root/'result.json').read_text());anchor['inspection_reference']=reference(replay_root)
    (root/'result.json').write_text(json.dumps(anchor))
    with pytest.raises(ValueError,match='original generation'):
        load(module,root,args)
