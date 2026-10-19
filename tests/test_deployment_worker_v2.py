"""Closed-snapshot CPU deployment fixtures; no protected labels or cost claims."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.deployment_worker_v2') is not None, 'Missing method-specific deployment worker'
    from treeneuron.research import deployment_worker_v2
    return deployment_worker_v2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n');return sha(path)


def prepare(root,family='P13',_isolated=True):
    # A prior NODE fixture can load sklearn's second OpenMP image into pytest.
    # The production guard correctly rejects that image set in a fresh P00
    # deployment. Fit P00 in its own process, as production already does.
    if family=='P00' and _isolated:
        project=Path(__file__).resolve().parents[1]
        env=os.environ|dict(PYTHONPATH=str(project/'src'),PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        env.pop('KMP_DUPLICATE_LIB_OK',None)
        code='import runpy,sys; from pathlib import Path; from treeneuron.research.method_worker import _initialize_native_threads; _initialize_native_threads("P00"); runpy.run_path(sys.argv[1])["prepare"](Path(sys.argv[2]),"P00",_isolated=False)'
        argv=[sys.executable,'-B','-c',code,str(Path(__file__).resolve()),str(root)]
        result=subprocess.run(argv,cwd=project,env=env,capture_output=True,text=True,timeout=60)
        publish(root/'fixture-command.json',dict(argv=argv,cwd=str(project),returncode=result.returncode))
        (root/'fixture-stdout.log').write_text(result.stdout);(root/'fixture-stderr.log').write_text(result.stderr)
        assert result.returncode==0,result.stderr
        path=root/'input/request.json';return path,json.loads(path.read_text()),sha(path)
    from treeneuron.research.p13_simple import P13Predictor
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research.method_worker import scientific_source_sha256
    data=root/'input';data.mkdir();X=pd.DataFrame(dict(x=np.arange(12,dtype=float),kind=np.arange(12)%2))
    config=dict(schema='research/v2-p13-config',family='P13',estimator='mean',
        feature_schema=dict(columns=['x','kind'],categorical=['kind']),parameters={})
    cls=P13Predictor
    if family=='P00':
        from treeneuron.research.gated_native import GatedPredictor
        cls=lambda:GatedPredictor('P00')
        config=dict(schema='research/v2-gated-native-config',family='P00',mode='engineering_fixture',
            feature_schema=config['feature_schema'],parameters=dict(width=8),base_updates=3,
            continuation_updates=3,validation_interval=1,patience=2)
    if family=='P12':
        from treeneuron.research.method_worker import _initialize_native_threads
        _initialize_native_threads('P12')
        from treeneuron.research.node_v2 import NodeV2Predictor
        cls=NodeV2Predictor
        config=dict(schema='research/v2-node-config',family='P12',mode='engineering_fixture',
            execution='bounded_memory_v1',feature_schema=config['feature_schema'],parameters={})
    model=cls().fit(FitRequest(X,3.+2.*X.x.to_numpy(),X.iloc[:4],np.ones(4),17,config,
        dict(max_updates=2) if family=='P12' else dict(max_seconds=30.),root/'fit'))
    model.save(data/'snapshot');query=data/'query.npz'
    np.savez(query,X=np.column_stack((np.arange(1024),np.arange(1024)%2)).astype(np.float32),
        row_id=np.arange(1024,dtype=np.int64))
    values=np.column_stack((np.arange(1024),np.arange(1024)%2)).astype(np.float32)
    frame=pd.DataFrame(values,columns=config['feature_schema']['columns'])
    np.savez(data/'expected-predictions.npz',**{f'batch{n}':model.predict(frame.iloc[:n],batch_size=n) for n in (1,1024)})
    request=dict(schema='research/v2-deployment-worker-request',purpose='engineering_qualification',family_id=family,
        scientific_source_sha256=scientific_source_sha256(),feature_schema=config['feature_schema'],
        snapshot=dict(path='snapshot',metadata_sha256=sha(data/'snapshot/manifest.json')),
        query=dict(path='query.npz',sha256=sha(query)),batch_sizes=[1,1024],warmup_calls=1,samples_per_batch=3)
    path=data/'request.json';return path,request,publish(path,request)


@pytest.mark.parametrize('kind',['request_hash','source','family','feature_schema','snapshot_hash','query_bytes','query_labels',
    'query_row_order_duplicates','query_dtype','query_size','unknown_field','batch_sizes','warmup_bool','sample_bool'])
def test_invalid_request_rejects_before_method_import(kind,tmp_path):
    module=api();path,value,digest=prepare(tmp_path);query=path.parent/'query.npz'
    if kind=='source':value['scientific_source_sha256']='0'*64
    elif kind=='family':value['family_id']='P14'
    elif kind=='feature_schema':value['feature_schema']['columns']=['different','kind']
    elif kind=='snapshot_hash':value['snapshot']['metadata_sha256']='0'*64
    elif kind=='query_bytes':query.write_bytes(query.read_bytes()+b'changed')
    elif kind in ('query_labels','query_row_order_duplicates','query_dtype','query_size'):
        with np.load(query,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if kind=='query_labels':arrays['y']=np.zeros(1024)
        elif kind=='query_row_order_duplicates':arrays['row_id'][1]=arrays['row_id'][0]
        elif kind=='query_dtype':arrays['X']=arrays['X'].astype(np.float64)
        else:arrays={k:v[:31] for k,v in arrays.items()}
        np.savez(query,**arrays);value['query']['sha256']=sha(query)
    elif kind=='unknown_field':value['quality_eligible']=True
    elif kind=='batch_sizes':value['batch_sizes']=[1,512]
    elif kind=='warmup_bool':value['warmup_calls']=True
    elif kind=='sample_bool':value['samples_per_batch']=True
    if kind not in ('request_hash','query_bytes'):digest=publish(path,value)
    if kind=='request_hash':digest='0'*64
    output=tmp_path/'output'
    with pytest.raises(ValueError):module.run_deployment(path,expected_request_sha256=digest,output=output)
    result=json.loads((output/'terminal.json').read_text())
    assert result['status']=='input_or_dependency_failure' and result['adapter_import_started'] is False
    assert result['T_accessed'] is result['cost_claim_authorized'] is False


@pytest.mark.parametrize('family',['P13','P00','P12'])
def test_actual_fresh_deployment_has_only_its_method_imports_separates_clocks_and_keeps_all_samples(family,tmp_path):
    api();path,value,digest=prepare(tmp_path,family);output=tmp_path/'output';root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    # Match the qualified fresh-worker launch policy. Parent-side native fitting
    # may have let an upstream dependency set its own OpenMP environment default.
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,'-B','-m','treeneuron.research.deployment_worker_v2','--request',str(path),
        '--expected-request-sha256',digest,'--output',str(output)]
    start=time.perf_counter_ns();result=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    outer=time.perf_counter_ns()-start
    (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),returncode=result.returncode,outer_wall_ns=outer,
        environment_overrides={k:env.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')}),indent=2)+'\n')
    (tmp_path/'stdout.log').write_text(result.stdout);(tmp_path/'stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    record=json.loads((output/'terminal.json').read_text())
    expected_modules=[] if family=='P13' else ['torch','treeneuron.baselines._node_vendor'] if family=='P12' else ['torch']
    assert record['status']=='completed_ineligible' and record['loaded_heavy_modules']==expected_modules
    assert record['torch_threads']==(None if family=='P13' else dict(intra_op=1,inter_op=1))
    assert record['cost_claim_authorized'] is record['protected_access_authorized'] is False
    assert record['whole_fresh_process_wall_authenticated'] is False
    assert record['query_rows']==1024 and record['input_dtype']=='float32'
    assert record['startup_environment']['KMP_DUPLICATE_LIB_OK'] is None
    assert record['scientific_source_sha256']==value['scientific_source_sha256']
    for name in ('common_import_ns','feature_decode_and_frame_ns','adapter_import_ns','model_load_ns'):
        assert type(record[name]) is int and record[name]>0
    assert record['inner_elapsed_ns']<outer and record['prediction_includes_native_preprocessing'] is True
    assert [b['batch_size'] for b in record['batches']]==[1,1024]
    for batch in record['batches']:
        assert len(batch['resident_prediction_ns'])==3 and len(batch['warmup_prediction_ns'])==1
        assert all(type(v) is int and v>0 for v in batch['resident_prediction_ns'])
        assert batch['predictions_equal_exact'] and batch['prediction_dtype']=='float64'
        with np.load(path.parent/'expected-predictions.npz',allow_pickle=False) as saved:
            expected=saved[f'batch{batch["batch_size"]}'].copy()
        assert batch['prediction_values_sha256']==hashlib.sha256(expected.tobytes()).hexdigest()
    before=(output/'terminal.json').read_bytes()
    with pytest.raises(ValueError):api().run_deployment(path,expected_request_sha256=digest,output=output)
    assert (output/'terminal.json').read_bytes()==before


def test_deployment_never_fits_and_rejects_late_input_drift(tmp_path,monkeypatch):
    module=api();path,value,digest=prepare(tmp_path)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*args,**kwargs):raise AssertionError('Deployment must never fit')
    original=P13Predictor.predict;calls=0
    def predict(model,*args,**kwargs):
        nonlocal calls
        calls+=1
        result=original(model,*args,**kwargs)
        query=path.parent/'query.npz';query.write_bytes(query.read_bytes()+b'changed during deployment')
        return result
    monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'predict',predict)
    with pytest.raises(ValueError):module.run_deployment(path,expected_request_sha256=digest,output=tmp_path/'output')
    assert calls>0
    assert json.loads((tmp_path/'output/terminal.json').read_text())['status']=='input_or_dependency_failure'
