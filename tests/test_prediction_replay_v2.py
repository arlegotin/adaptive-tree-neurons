"""Independent label-free replay; fixtures do not authorize scientific claims."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.prediction_replay_v2') is not None, 'Missing independent prediction replay'
    from treeneuron.research import prediction_replay_v2
    return prediction_replay_v2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n');return sha(path)


def prepare(root,family='P13',temporal=False,_isolated=True):
    if family=='P00' and _isolated:
        project=Path(__file__).resolve().parents[1]
        env=os.environ|dict(PYTHONPATH=str(project/'src'),PYTHONDONTWRITEBYTECODE='1')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        env.pop('KMP_DUPLICATE_LIB_OK',None)
        code='import runpy,sys; from pathlib import Path; from treeneuron.research.method_worker import _initialize_native_threads; _initialize_native_threads("P00"); runpy.run_path(sys.argv[1])["prepare"](Path(sys.argv[2]),"P00",temporal=sys.argv[3]=="True",_isolated=False)'
        argv=[sys.executable,'-B','-c',code,str(Path(__file__).resolve()),str(root),str(temporal)]
        result=subprocess.run(argv,cwd=project,env=env,capture_output=True,text=True,timeout=60)
        publish(root/'fixture-command.json',dict(argv=argv,cwd=str(project),returncode=result.returncode))
        (root/'fixture-stdout.log').write_text(result.stdout);(root/'fixture-stderr.log').write_text(result.stderr)
        assert result.returncode==0,result.stderr
        path=root/'input/request.json';return path,json.loads(path.read_text()),sha(path)
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research.method_worker import scientific_source_sha256,_initialize_native_threads
    from treeneuron.research.p13_simple import P13Predictor
    cls=P13Predictor;schema=dict(columns=['x','kind'],categorical=['kind'])
    config=dict(schema='research/v2-p13-config',family='P13',estimator='mean',feature_schema=schema,parameters={})
    if family=='P00':
        from treeneuron.research.gated_native import GatedPredictor
        cls=lambda:GatedPredictor('P00')
        config=dict(schema='research/v2-gated-native-config',family=family,mode='engineering_fixture',
            feature_schema=schema,parameters=dict(width=8),base_updates=3,continuation_updates=3,validation_interval=1,patience=2)
    if family=='P12':
        _initialize_native_threads(family)
        from treeneuron.research.node_v2 import NodeV2Predictor
        cls=NodeV2Predictor
        config=dict(schema='research/v2-node-config',family=family,mode='engineering_fixture',
            execution='bounded_memory_v1',feature_schema=schema,parameters={})
    X=pd.DataFrame(dict(x=np.arange(12,dtype=float),kind=np.arange(12)%2))
    model=cls().fit(FitRequest(X,3.+2.*X.x.to_numpy(),X.iloc[:4],np.ones(4),17,config,
        dict(max_updates=2) if family=='P12' else dict(max_seconds=30.),root/'fit'))
    data=root/'input';data.mkdir();model.save(data/'snapshot')
    rows=np.arange(37,dtype=np.int64)+1000
    query=dict(X=np.column_stack((np.arange(37)*.137,np.arange(37)%2)),row_id=rows,group_id=rows//4)
    if temporal:query['timestamps']=np.datetime64('2020-01-01T00','h')+np.arange(37).astype('timedelta64[h]')
    prediction=dict(prediction=np.asarray(model.predict(pd.DataFrame(query['X'],columns=schema['columns']))),
        **{k:v for k,v in query.items() if k!='X'})
    np.savez(data/'query.npz',**query);np.savez(data/'prediction.npz',**prediction)
    request=dict(schema='research/v2-prediction-replay-request',purpose='engineering_qualification',family_id=family,
        scientific_source_sha256=scientific_source_sha256(),feature_schema=schema,
        snapshot=dict(path='snapshot',metadata_sha256=sha(data/'snapshot/manifest.json')),
        query=dict(path='query.npz',sha256=sha(data/'query.npz')),
        prediction=dict(path='prediction.npz',sha256=sha(data/'prediction.npz')))
    path=data/'request.json';return path,request,publish(path,request)


def rewrite_archive(path,change):
    with np.load(path,allow_pickle=False) as saved:values={k:saved[k].copy() for k in saved.files}
    change(values);np.savez(path,**values)


@pytest.mark.parametrize('kind',['request_hash','source','family','schema','snapshot_hash','query_bytes','prediction_bytes',
    'query_labels','prediction_labels','query_dtype','duplicate_rows','row_order','groups','timestamps','nonfinite',
    'escaped_path','symlink_parent','unknown_field'])
def test_invalid_evidence_rejected_before_adapter_import(kind,tmp_path):
    module=api();path,value,digest=prepare(tmp_path,temporal=kind=='timestamps')
    query=path.parent/'query.npz';prediction=path.parent/'prediction.npz'
    if kind=='request_hash':digest='0'*64
    elif kind=='source':value['scientific_source_sha256']='0'*64
    elif kind=='family':value['family_id']='P14'
    elif kind=='schema':value['feature_schema']['columns']=['other','kind']
    elif kind=='snapshot_hash':value['snapshot']['metadata_sha256']='0'*64
    elif kind=='query_bytes':query.write_bytes(query.read_bytes()+b'drift')
    elif kind=='prediction_bytes':prediction.write_bytes(prediction.read_bytes()+b'drift')
    elif kind=='query_labels':rewrite_archive(query,lambda a:a.update(y=np.zeros(37)))
    elif kind=='prediction_labels':rewrite_archive(prediction,lambda a:a.update(y=np.zeros(37)))
    elif kind=='query_dtype':rewrite_archive(query,lambda a:a.update(X=a['X'].astype(np.float32)))
    elif kind=='duplicate_rows':rewrite_archive(query,lambda a:a['row_id'].__setitem__(1,a['row_id'][0]))
    elif kind=='row_order':rewrite_archive(prediction,lambda a:a.update(row_id=a['row_id'][::-1]))
    elif kind=='groups':rewrite_archive(prediction,lambda a:a['group_id'].__setitem__(0,-1))
    elif kind=='timestamps':rewrite_archive(prediction,lambda a:a.update(timestamps=a['timestamps']+np.timedelta64(1,'h')))
    elif kind=='nonfinite':rewrite_archive(prediction,lambda a:a['prediction'].__setitem__(0,np.nan))
    elif kind=='escaped_path':value['query']['path']='../input/query.npz'
    elif kind=='symlink_parent':
        (path.parent/'alias').symlink_to(path.parent,target_is_directory=True);value['query']['path']='alias/query.npz'
    elif kind=='unknown_field':value['quality_eligible']=True
    if kind not in ('query_bytes','prediction_bytes'):
        value['query']['sha256']=sha(query);value['prediction']['sha256']=sha(prediction)
    if kind!='request_hash':digest=publish(path,value)
    output=tmp_path/'output'
    with pytest.raises(ValueError):module.run_prediction_replay(path,expected_request_sha256=digest,output=output)
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['adapter_import_started'] is False and terminal['status']=='input_or_dependency_failure'
    assert terminal['quality_eligible'] is terminal['protected_access_authorized'] is terminal['T_accessed'] is False


@pytest.mark.parametrize('family,temporal',[('P13',False),('P13',True),('P00',False),('P12',False)])
def test_actual_fresh_replay_preserves_native_query_precision_and_method_isolation(family,temporal,tmp_path):
    api();path,value,digest=prepare(tmp_path,family,temporal);root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    output=tmp_path/'output';argv=[sys.executable,'-B','-m','treeneuron.research.prediction_replay_v2',
        '--request',str(path),'--expected-request-sha256',digest,'--output',str(output)]
    result=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=90)
    publish(tmp_path/'command.json',dict(argv=argv,cwd=str(root),returncode=result.returncode,
        environment_overrides={k:env.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')}))
    (tmp_path/'stdout.log').write_text(result.stdout);(tmp_path/'stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['status']=='completed_ineligible' and terminal['prediction_reload_exact'] is True
    assert terminal['query_rows']==37 and terminal['query_dtype']=='float64' and terminal['prediction_dtype']=='float64'
    assert terminal['loaded_heavy_modules']==([] if family=='P13' else ['torch','treeneuron.baselines._node_vendor'] if family=='P12' else ['torch'])
    assert terminal['torch_threads']==(None if family=='P13' else dict(intra_op=1,inter_op=1))
    assert terminal['labels_read'] is terminal['quality_eligible'] is terminal['model_generation_authenticated'] is False
    assert terminal['query_sha256']==value['query']['sha256'] and terminal['prediction_sha256']==value['prediction']['sha256']
    assert terminal['snapshot_files'] and all(sha(path.parent/'snapshot'/f['path'])==f['sha256'] for f in terminal['snapshot_files'])
    before=(output/'terminal.json').read_bytes()
    with pytest.raises(ValueError):api().run_prediction_replay(path,expected_request_sha256=digest,output=output)
    assert (output/'terminal.json').read_bytes()==before


@pytest.mark.parametrize('kind',['values','dtype'])
def test_replay_rejects_finite_predictions_that_are_not_the_saved_model_output(kind,tmp_path):
    module=api();path,value,_=prepare(tmp_path);prediction=path.parent/'prediction.npz'
    rewrite_archive(prediction,lambda a:a.update(prediction=a['prediction']+1) if kind=='values' else a.update(prediction=a['prediction'].astype(np.float32)))
    value['prediction']['sha256']=sha(prediction);digest=publish(path,value)
    with pytest.raises(FloatingPointError):module.run_prediction_replay(path,expected_request_sha256=digest,output=tmp_path/'output')
    result=json.loads((tmp_path/'output/terminal.json').read_text())
    assert result['status']=='numerical_failure' and result['prediction_reload_exact'] is False


@pytest.mark.parametrize('kind',['query','prediction','request','snapshot'])
def test_replay_never_fits_and_rejects_late_bound_file_drift(kind,tmp_path,monkeypatch):
    module=api();path,value,digest=prepare(tmp_path)
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Replay must never fit')
    original=P13Predictor.predict
    def predict(model,*a,**kw):
        result=original(model,*a,**kw)
        target=path if kind=='request' else path.parent/'snapshot/late-extra' if kind=='snapshot' else path.parent/(kind+'.npz')
        target.write_bytes((target.read_bytes() if target.exists() else b'')+b'drift')
        return result
    monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'predict',predict)
    with pytest.raises(ValueError):module.run_prediction_replay(path,expected_request_sha256=digest,output=tmp_path/'output')
    assert json.loads((tmp_path/'output/terminal.json').read_text())['status']=='input_or_dependency_failure'
