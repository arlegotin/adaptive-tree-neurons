"""Actual fresh-process own-family fitting/reload without quality promotion."""
import hashlib
import json
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.parametrize('family,variant',[
    ('P00',None),('P01',None),('P02','lower'),('P02','upper'),
    ('P05','centered_partial'),('P05','centered_full'),('P05','direct_partial'),('P05','direct_full')])
def test_supervised_own_family_preserves_roles_closed_predictions_and_ineligibility(family,variant,tmp_path):
    from treeneuron.config import canonical_hash
    from treeneuron.research.ledger import write_new_json
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.search_controller_v2 import make_search_plan,run_search
    from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
    vm=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True).stdout
    page=int(re.search(r'page size of (\d+) bytes',vm)[1])
    counts=[int(re.search(name+r':\s+(\d+)',vm)[1]) for name in ('Pages free','Pages inactive')]
    admitted=sum(counts)*page>=int(1.75*2**30) and shutil.disk_usage(tmp_path).free>=512*1024**2
    write_new_json(tmp_path/'admission.json',dict(admitted=admitted,vm_stat_stdout=vm,
        proxy_is_not_allocation_guarantee=True,free_disk_bytes=shutil.disk_usage(tmp_path).free))
    if not admitted:pytest.skip('Retained live admission below declared own-worker fixture floor')
    source=scientific_source_sha256();data=tmp_path/'input';data.mkdir();files={};rng=np.random.default_rng(1729)
    for role,n,offset in (('F',37,0),('G',17,1000),('V_features',11,2000)):
        X=rng.normal(size=(n,21));X[:,-1]=np.arange(n)%2
        if role=='V_features':X[::3,-1]=2
        rows=np.arange(offset,offset+n,dtype=np.int64);arrays=dict(X=X,row_id=rows,group_id=rows)
        if role!='V_features':arrays['y']=np.sin(X[:,0])+.2*X[:,1]
        p=data/(role+'.npz');np.savez(p,**arrays);files[role]=dict(path=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    cfg=dict(schema='research/v2-gated-native-config',family=family,mode='engineering_fixture',
        feature_schema=dict(columns=[f'x{i}' for i in range(20)]+['kind'],categorical=['kind']),parameters={'width':8},
        base_updates=3,continuation_updates=3,validation_interval=1,patience=2)
    if variant is not None:cfg['variant']=variant
    split=canonical_hash(dict(files=files));identity=dict(job_id='gated-fixture/'+family+'/'+str(variant),
        family_id=family,search_repeat=0,source_sha256=source,runtime_sha256=canonical_hash(dict(interpreter=sys.executable)),
        support_sha256=canonical_hash(dict(anchor=cfg)),split_sha256=split)
    template=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',job_id=identity['job_id'],
        family_id=family,fit_seed=17,scientific_source_sha256=source,config=cfg,budget=dict(max_seconds=60.),
        observation_identity=dict(dataset='gated_worker_fixture',split_kind='grouped',split_sha256=split),files=files)
    path=data/'request.json';write_new_json(path,template)
    plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=19,anchors=[cfg],mode='finite',support=None,
        candidates=[],max_attempts=1,deadline_ns=60_000_000_000,prefixes_ns=[])
    bridge=EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        search_directory=tmp_path/'search',interpreter=sys.executable,rss_limit_bytes=1024**3)
    result=run_search(tmp_path/'search',plan=plan,evaluate_trial=bridge,invocation_preparation_ns=bridge.preparation_ns)
    attempt=tmp_path/'search/attempt-000';worker=attempt/'worker';terminal=json.loads((worker/'terminal.json').read_text())
    assert [r['status'] for r in result['records']]==['completed_ineligible'],terminal.get('reason')
    assert result['winner'] is None and result['attempts_paid']==1
    assert terminal['snapshot_reload_exact'] and terminal['prediction_rows']==11
    assert terminal['checkpoint_eligible'] is False and terminal['quality_eligible'] is False
    assert terminal['T_accessed'] is False and terminal['accessed_roles']==['F_labels','G_labels','V_features']
    assert terminal['loaded_heavy_modules']==['torch']
    assert terminal['adapter_resources']['encoded_width']==23
    assert terminal['adapter_resources']['persistent_training_resume'] is False
    with np.load(worker/'predictions.npz',allow_pickle=False) as saved:
        assert np.isfinite(saved['prediction']).all();np.testing.assert_array_equal(saved['row_id'],np.arange(2000,2011))
    process=json.loads((attempt/'process/process.json').read_text())
    assert process['child_reaped'] and process['returncode']==0
