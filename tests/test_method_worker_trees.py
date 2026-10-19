"""Opt-in native workers must reopen each adapter's actual snapshot contract."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('family,package',[
    ('P08','catboost'),('P09','xgboost'),('P10','lightgbm'),('P11','lightgbm')])
def test_supervised_tree_worker_reopens_state_snapshot_without_neural_manifest(family,package,tmp_path):
    overlay=os.environ.get('ATN_TREE_OVERLAY')
    if not overlay:pytest.skip('Requires the separately pinned native tree overlay')
    from treeneuron.config import canonical_hash
    from treeneuron.research.ledger import write_new_json
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.search_controller_v2 import make_search_plan,run_search
    from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
    import subprocess
    vm=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True).stdout
    page=int(re.search(r'page size of (\d+) bytes',vm)[1])
    counts=[int(re.search(name+r':\s+(\d+)',vm)[1]) for name in ('Pages free','Pages inactive')]
    admitted=sum(counts)*page>=1024**3 and shutil.disk_usage(tmp_path).free>=256*1024**2
    write_new_json(tmp_path/'admission.json',dict(admitted=admitted,vm_stat_stdout=vm,
        proxy_is_not_allocation_guarantee=True,free_disk_bytes=shutil.disk_usage(tmp_path).free))
    if not admitted:pytest.skip('Retained resource admission below the declared fixture floor')
    source=scientific_source_sha256();input_root=tmp_path/'input';input_root.mkdir();files={}
    rng=np.random.default_rng(1729)
    for role,n,offset in (('F',192,0),('G',64,1000),('V_features',17,2000)):
        X=rng.uniform(-1,1,(n,3));X[:,2]=np.arange(n)%2
        if role=='V_features':X[::7,2]=2
        rows=np.arange(offset,offset+n,dtype=np.int64)
        arrays=dict(X=X,row_id=rows,group_id=rows)
        if role!='V_features':arrays['y']=2+X[:,0]+X[:,1]**2+.4*X[:,2]
        p=input_root/(role+'.npz');np.savez(p,**arrays)
        files[role]=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    config=dict(schema='research/v2-native-tree-config',family=family,mode='engineering_fixture',
        feature_schema=dict(columns=['x1','x2','kind'],categorical=['kind']),parameters={},
        max_rounds=16,patience=2)
    split=canonical_hash(dict(files=files))
    template=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id='engineering/'+family+'/snapshot-boundary',family_id=family,fit_seed=17,
        scientific_source_sha256=source,config=config,budget=dict(max_seconds=60.),
        observation_identity=dict(dataset='synthetic_tree_snapshot_fixture',split_kind='grouped',split_sha256=split),files=files)
    path=input_root/'request.json';write_new_json(path,template)
    identity=dict(job_id=template['job_id'],family_id=family,search_repeat=0,source_sha256=source,
        runtime_sha256=canonical_hash(dict(overlay=str(Path(overlay).resolve()))),
        support_sha256=canonical_hash(dict(anchor=config)),split_sha256=split)
    plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=19,anchors=[config],
        mode='finite',support=None,candidates=[],max_attempts=1,deadline_ns=60_000_000_000,prefixes_ns=[])
    bridge=EngineeringTrialBridge(plan=plan,template_path=path,
        expected_template_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        search_directory=tmp_path/'search',interpreter=sys.executable,
        extra_pythonpath=(Path(overlay),),rss_limit_bytes=512*1024**2)
    result=run_search(tmp_path/'search',plan=plan,evaluate_trial=bridge,
        invocation_preparation_ns=bridge.preparation_ns)
    attempt=tmp_path/'search/attempt-000';worker=attempt/'worker'
    terminal=json.loads((worker/'terminal.json').read_text())
    assert [r['status'] for r in result['records']]==['completed_ineligible'],terminal.get('reason')
    assert result['winner'] is None and result['attempts_paid']==1
    assert terminal['snapshot_reload_exact'] and terminal['prediction_rows']==17
    assert terminal['quality_eligible'] is False and terminal['checkpoint_eligible'] is False
    assert terminal['T_accessed'] is False and terminal['accessed_roles']==['F_labels','G_labels','V_features']
    assert terminal['loaded_heavy_modules']==[package]
    entry=worker/'snapshot/state.json'
    assert entry.is_file() and not (worker/'snapshot/manifest.json').exists()
    assert terminal['snapshot_commitment']==dict(path='snapshot/state.json',sha256=hashlib.sha256(entry.read_bytes()).hexdigest())
    with np.load(worker/'predictions.npz',allow_pickle=False) as z:
        assert set(z.files)=={'prediction','row_id','group_id'}
        assert np.isfinite(z['prediction']).all()
        np.testing.assert_array_equal(z['row_id'],np.arange(2000,2017))
    process=json.loads((attempt/'process/process.json').read_text())
    assert process['child_reaped'] and process['returncode']==0
