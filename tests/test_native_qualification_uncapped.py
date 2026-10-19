"""Native engineering qualification keeps fixed anchors and immutable recovery."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from treeneuron.config import canonical_hash
from test_native_qualification_runner import api,config


def rewrite(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')


def test_explicit_native_qualification_preserves_all_120_primary_anchors_without_deadline(tmp_path):
    module=api();path,value=config(tmp_path,all_cases=True)
    value['native_stopping']=True;rewrite(path,value)
    plan=module.qualification_plan(path)
    assert len(plan['attempts'])==120 and plan['identity']['per_anchor_deadline_ns'] is None
    assert plan['identity']['native_stopping'] is True
    assert plan['identity']['fit_seed']==17 and plan['identity']['Gate_D'] is False
    assert all(row['payload']['outer_fold']==(4 if row['payload']['dataset']=='bike' else 0)
        for row in plan['attempts'])


@pytest.mark.parametrize('flag',[False,1,None,'true'])
def test_explicit_native_qualification_rejects_coercive_or_ambiguous_flag(tmp_path,flag):
    module=api();path,value=config(tmp_path);value['native_stopping']=flag;rewrite(path,value)
    with pytest.raises(ValueError):module.qualification_plan(path)


def prepared_config(root):
    from treeneuron.data.tabular import DATASETS,table_from_frame
    from treeneuron.research.transfer import seal_table
    root.mkdir();module=api();schema=DATASETS['energy']
    frame=pd.DataFrame({name:np.arange(24,dtype=float) for name in schema.raw_columns})
    for name in ('X1','X2','X3','X4','X5'):frame[name]=np.repeat(np.arange(12),2)
    # Synthetic role fixture only: the supplied metadata digest is not an
    # acquisition proof. Qualification outputs deliberately grant no Gate D.
    table=table_from_frame('energy',frame,raw_sha256=schema.raw_sha256)
    sealed=root/'sealed';seal_table(table,sealed);manifest=sealed/'manifest.json'
    value=dict(schema='research/v2-native-qualification',study_id='native-stopping-fixture',
        source_sha256=module.qualification_source_sha256(),locks_sha256=module.qualification_locks_sha256(),
        interpreter=sys.executable,families=['P13'],overlays={},native_stopping=True,
        datasets=dict(energy=dict(root=str(sealed),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())))
    path=root/'qualification.json';rewrite(path,value);return path,value


def resources(monkeypatch):
    # Isolate outer admission from unrelated machine activity; actual child RSS
    # supervision still runs, and the real child/arrays/artifacts are exercised.
    monkeypatch.setattr(api(),'_resources',lambda out:dict(available_memory_bytes=8*1024**3,free_disk_bytes=8*1024**3))


def test_real_native_qualification_closes_actual_model_with_no_deadline_or_V_labels(tmp_path,monkeypatch):
    module=api();path,value=prepared_config(tmp_path/'input');resources(monkeypatch)
    output=tmp_path/'output';result=module.qualify(path,output)
    assert result['status']=='completed' and result['native_anchor_executions_complete'] is True
    assert result['Gate_D'] is result['quality_eligible'] is result['V_labels_read'] is result['T_accessed'] is False
    process_path=next((output/'queue/jobs').glob('*/attempt-0/native/process/process.json'))
    process=json.loads(process_path.read_text())
    assert process['timeout_ns'] is None and process['child_reaped'] is True and process['returncode']==0
    attempt=process_path.parents[2]
    request=json.loads((attempt/'input/request.json').read_text())
    assert request['budget']=={'native_stopping':True} and set(request['files'])=={'F','G','V_features'}
    with np.load(attempt/'input/F.npz',allow_pickle=False) as fit:
        expected=float(np.mean(fit['y']))
    with np.load(attempt/'native/worker/predictions.npz',allow_pickle=False) as predictions:
        np.testing.assert_array_equal(predictions['prediction'],np.full(len(predictions['prediction']),expected))
    access=[json.loads(s) for s in (Path(value['datasets']['energy']['root'])/'access.jsonl').read_text().splitlines()]
    assert all(row['roles']==['F_labels','G_labels','V_features'] for row in access)


@pytest.mark.parametrize('change_old',[False,True])
def test_native_resource_retry_authenticates_old_attempt_and_preserves_seed_and_paid_time(tmp_path,monkeypatch,change_old):
    from treeneuron.research import native_fit_v2
    from treeneuron.research.workqueue import _inventory
    module=api();path,value=prepared_config(tmp_path/'input');resources(monkeypatch)
    real=native_fit_v2.run_native_fit;calls=[]
    def first_attempt_tiny_RSS(*args,**kwargs):
        kwargs['rss_limit_bytes']=1 if not calls else 1024**3
        calls.append(kwargs['rss_limit_bytes']);return real(*args,**kwargs)
    monkeypatch.setattr(native_fit_v2,'run_native_fit',first_attempt_tiny_RSS)
    output=tmp_path/'output'
    with pytest.raises(MemoryError):module.qualify(path,output)
    old=next((output/'queue/jobs').glob('*/attempt-0'))
    assert json.loads((old/'native/result.json').read_text())['status']=='resource_exhausted'
    if change_old:(old/'native/process/stderr.log').write_text('changed previous evidence')
    before=_inventory(old);result=module.qualify(path,output,resume=True)
    assert _inventory(old)==before
    if change_old:
        assert result['status']=='completed_with_failures'
        assert not list((output/'queue/jobs').glob('*/attempt-1/native/process/spawned.json'))
        return
    assert result['status']=='completed'
    new=next((output/'queue/jobs').glob('*/attempt-1'))
    old_request=(old/'input/request.json').read_bytes();new_request=(new/'input/request.json').read_bytes()
    assert old_request==new_request and json.loads(new_request)['fit_seed']==17
    recovered=json.loads((new/'native/result.json').read_text())
    assert recovered['recovery']['reason']=='same_seed_restart_without_optimizer_checkpoint'
    events=[json.loads(s) for s in (output/'queue/events.jsonl').read_text().splitlines()]
    assert [e['status'] for e in events]==['running','interrupted_recoverable','running','completed_ineligible']
    assert events[-1]['elapsed_ns']>events[1]['elapsed_ns']>0
    assert recovered['quality_eligible'] is recovered['T_accessed'] is False
