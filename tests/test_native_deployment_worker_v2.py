"""Closed original-snapshot measurements are software evidence, never native gates."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_deployment_worker_v2'), 'Missing owned original-snapshot deployment leaf'
    from treeneuron.research import native_deployment_worker_v2
    return native_deployment_worker_v2


def ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def measurement_fixture(root):
    import pandas as pd
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research.p13_simple import P13Predictor
    from treeneuron.research.deployment_worker_v2 import _source
    from treeneuron.research.protected_query_v2 import _snapshot_inventory
    from treeneuron.research.protected_journal_v2 import _hash
    F=pd.DataFrame(dict(x=np.linspace(-1,1,24),kind=np.arange(24)%2));y=1+2*F.x.to_numpy()+.5*F.kind.to_numpy()
    schema=dict(columns=['x','kind'],categorical=['kind'])
    config=dict(schema='research/v2-p13-config',family='P13',estimator='ridge',feature_schema=schema,parameters=dict(alpha=1e-4))
    model=P13Predictor().fit(FitRequest(F,y,F.iloc[:4],y[:4],17,config,dict(max_seconds=30),root/'fit'))
    snapshot=root/'original-snapshot';model.save(snapshot)
    data=root/'outer-0';data.mkdir();Ffile=data/'F.npz'
    np.savez(Ffile,X=F.to_numpy(dtype=np.float64),y=y,row_id=np.arange(24,dtype=np.int64),group_id=np.arange(24,dtype=np.int64))
    values=F.to_numpy(dtype=np.float32)[np.arange(1024)%24];query=data/'F-deployment-features.npz'
    np.savez(query,X=values,row_id=np.arange(1024,dtype=np.int64))
    provenance=dict(schema='research/v2-F-derived-deployment-query',source_sha256=_source(),source_F_reference=ref(Ffile),
        query_reference=ref(query),row_policy='cycle_original_F_order_to_1024',source_rows=24,workload_rows=1024,
        source_role='F',original_role_verified_by_native_controller=False)
    provenance_path=data/'provenance.json';provenance_path.write_text(json.dumps(provenance,sort_keys=True)+'\n')
    episode=root/'episode';episode.mkdir();request=episode/'request.json'
    value=dict(schema='research/v2-original-snapshot-measurement-request',source_sha256=_source(),family_id='P13',
        logical_job_id='known-fixture-only',logical_fit_seed=2**61+17,feature_schema=schema,
        model_binding=dict(attempt_index=0,selected_metadata=ref(snapshot/'manifest.json'),snapshot_root=str(snapshot),
            snapshot_inventory_sha256=_hash(_snapshot_inventory(snapshot)),generation_sha256='c'*64),
        query_reference=ref(query),query_provenance_reference=ref(provenance_path),batch_sizes=[1,1024],warmup_calls=3,samples_per_batch=32)
    request.write_text(json.dumps(value,sort_keys=True)+'\n')
    frame=pd.DataFrame(values,columns=schema['columns'])
    expected={n:model.predict(frame.iloc[:n],batch_size=n) for n in (1,1024)}
    return request,value,expected


def launch(module,path):
    from treeneuron.research.cooperative_process_v2 import supervise_cooperative_process
    project=Path(__file__).resolve().parents[1];episode=path.parent
    env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
    env.update(PYTHONPATH=str(project/'src'),KMP_DUPLICATE_LIB_OK=None)
    supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.native_deployment_worker_v2',
        '--request',str(path),'--expected-request-sha256',ref(path)['sha256'],'--output',str(episode/'worker')],
        cwd=project,env_overrides=env,output=episode/'process',identity=module.native_measurement_identity(path,ref(path)['sha256']),
        admission_identity=dict(purpose='known_local_fixture_not_native_qualification'),
        admission_check=lambda:dict(admitted=True,reason='known local fixture only'),rss_pause_bytes=1024**3,admit_before_spawn=True)
    return dict(process_root=str(episode/'process'),process_receipt_sha256=ref(episode/'process/process.json')['sha256'],
        process_started_sha256=ref(episode/'process/started.json')['sha256'])


def test_unowned_cli_rejects_before_request_features_or_method_import(tmp_path):
    module=api();env=os.environ.copy()
    for key in tuple(env):
        if key.startswith('ATN_SEARCH_CONTROL_'):env.pop(key)
    p=subprocess.run([sys.executable,'-P','-B','-m','treeneuron.research.native_deployment_worker_v2',
        '--request',str(tmp_path/'forbidden-request.json'),'--expected-request-sha256','a'*64,'--output',str(tmp_path/'output')],
        env=env,capture_output=True,text=True,timeout=15)
    assert p.returncode!=0 and 'forbidden-request' not in p.stderr and not (tmp_path/'output').exists()


@pytest.mark.parametrize('fault',['source','model_hash','model_inventory','labels','dtype','rows','query_hash','unknown','warmup_bool'])
def test_changed_request_or_arrays_fail_before_native_method_import(tmp_path,fault):
    module=api();path,value,_=measurement_fixture(tmp_path)
    if fault=='source':value['source_sha256']='0'*64
    elif fault=='model_hash':value['model_binding']['selected_metadata']['sha256']='0'*64
    elif fault=='model_inventory':value['model_binding']['snapshot_inventory_sha256']='0'*64
    elif fault in ('labels','dtype','rows'):
        query=Path(value['query_reference']['path'])
        with np.load(query,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if fault=='labels':arrays['y']=np.zeros(1024)
        elif fault=='dtype':arrays['X']=arrays['X'].astype(np.float64)
        else:arrays['row_id'][1]=arrays['row_id'][0]
        np.savez(query,**arrays);value['query_reference']=ref(query)
        provenance=Path(value['query_provenance_reference']['path']);p=json.loads(provenance.read_text());p['query_reference']=ref(query)
        provenance.write_text(json.dumps(p)+'\n');value['query_provenance_reference']=ref(provenance)
    elif fault=='query_hash':value['query_reference']['sha256']='0'*64
    elif fault=='unknown':value['native_qualified']=True
    else:value['warmup_calls']=True
    path.write_text(json.dumps(value)+'\n')
    with pytest.raises((ValueError,FileNotFoundError)):module._checked_measurement_request(path,ref(path)['sha256'])
    assert not (path.parent/'worker').exists()


def test_actual_owned_leaf_measures_original_saved_ridge_without_copy_or_Torch(tmp_path):
    module=api();path,value,expected=measurement_fixture(tmp_path);snapshot=Path(value['model_binding']['snapshot_root'])
    from treeneuron.research.protected_query_v2 import _snapshot_inventory
    before=_snapshot_inventory(snapshot);process=launch(module,path)
    result=module.inspect_owned_native_measurement(process)
    assert result['finite_repeated_measurement_checked'] is True
    assert result['native_models_authenticated'] is result['cost_claim_authorized'] is result['T_accessed'] is False
    terminal=json.loads((path.parent/'worker/measurement.json').read_text())
    assert terminal['saved_native_seed']==17 and terminal['logical_fit_seed']==2**61+17
    assert terminal['loaded_method_modules']==[]
    for row in terminal['batches']:
        assert len(row['warmup_prediction_ns'])==3 and len(row['resident_prediction_ns'])==32
        assert row['prediction_values_sha256']==hashlib.sha256(expected[row['batch_size']].tobytes()).hexdigest()
    assert _snapshot_inventory(snapshot)==before
    assert terminal['model_binding']['snapshot_root']==str(snapshot)


def test_changed_closed_measurement_is_rejected_by_readonly_owned_join(tmp_path):
    module=api();path,value,_=measurement_fixture(tmp_path);process=launch(module,path)
    target=path.parent/'worker/measurement.json'
    with target.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):module.inspect_owned_native_measurement(process)


@pytest.mark.parametrize('fault',['closure','body_clock','saved_seed','extra_body','extra_measurement','impossible_clock','input_precision'])
def test_consistently_rehashed_but_semantically_invalid_measurement_still_rejects(tmp_path,fault):
    module=api();path,value,_=measurement_fixture(tmp_path);process=launch(module,path);root=Path(process['process_root'])
    body_path=root/'body-outcome.json';body=json.loads(body_path.read_text())
    if fault=='closure':
        target=root/'worker-closure.json';closure=json.loads(target.read_text());closure['exit_kind']='raised';target.write_text(json.dumps(closure)+'\n')
    elif fault=='body_clock':body['body_elapsed_ns']=-1
    elif fault=='extra_body':body['native_override_allowed']=True
    else:
        target=path.parent/'worker/measurement.json';measured=json.loads(target.read_text())
        if fault=='saved_seed':measured['saved_native_seed']=999
        elif fault=='extra_measurement':measured['native_override_allowed']=True
        elif fault=='impossible_clock':measured['model_load_ns']=10**30
        else:measured['input_dtype']='float64'
        target.write_text(json.dumps(measured)+'\n');body['measurement_reference']=ref(target)
    if fault!='closure':body_path.write_text(json.dumps(body)+'\n')
    receipt_path=root/'process.json';receipt=json.loads(receipt_path.read_text())
    for record in receipt['files']:
        target=root/record['path'];record.update(bytes=target.stat().st_size,sha256=ref(target)['sha256'])
    receipt_path.write_text(json.dumps(receipt)+'\n');process['process_receipt_sha256']=ref(receipt_path)['sha256']
    with pytest.raises(ValueError):module.inspect_owned_native_measurement(process)
