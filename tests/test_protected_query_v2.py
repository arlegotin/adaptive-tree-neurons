"""Saved-state feature inference fixtures never grant protected data authority."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_prediction_replay_v2 import prepare,sha,publish,rewrite_archive


def api():
    assert importlib.util.find_spec('treeneuron.research.protected_query_v2'), 'Missing saved T-feature query primitive'
    from treeneuron.research import protected_query_v2
    return protected_query_v2


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


@pytest.fixture
def saved_query(tmp_path):
    return query_fixture(tmp_path)


def query_fixture(tmp_path,*,family='P13',temporal=False):
    from treeneuron.research.workqueue import _inventory
    path,previous,_=prepare(tmp_path,family=family,temporal=temporal)
    role=path.parent/'outer-0';role.mkdir();feature=role/'T-features.npz';feature.write_bytes((path.parent/'query.npz').read_bytes())
    model=dict(attempt_index=0,selected_metadata=dict(path=str(path.parent/'snapshot/manifest.json'),sha256=previous['snapshot']['metadata_sha256']),
        snapshot_root=str(path.parent/'snapshot'),snapshot_inventory_sha256=canonical_hash(_inventory(path.parent/'snapshot')),generation_sha256='c'*64)
    query=dict(job_id='unqualified-toy-job',fit_seed=2**61+17,dataset='toy',outer_fold=0,family_id=family,
        split_hash='d'*64,manifest_hash='e'*64,query_role='T_features',feature_reference=dict(path=str(feature),sha256=sha(feature)),model_binding=model)
    identity={k:v for k,v in query.items() if k!='model_binding'}
    identity['model_content']=dict(attempt_index=0,generation_sha256=model['generation_sha256'],snapshot_inventory_sha256=model['snapshot_inventory_sha256'],
        selected_metadata_sha256=model['selected_metadata']['sha256'],selected_metadata_name='manifest.json')
    query.update(query_id=canonical_hash(identity),content_identity=identity)
    request=dict(schema='research/v2-saved-protected-feature-query',source_sha256=previous['scientific_source_sha256'],
        protocol_sha256='a'*64,prediction_manifest_sha256='b'*64,query=query,feature_schema=previous['feature_schema'],split_kind='temporal' if temporal else 'grouped')
    return path,previous,request


def rebind(request):
    q=request['query'];model=q['model_binding'];identity={k:v for k,v in q.items() if k not in ('query_id','content_identity','model_binding')}
    identity['model_content']=dict(attempt_index=model['attempt_index'],generation_sha256=model['generation_sha256'],snapshot_inventory_sha256=model['snapshot_inventory_sha256'],
        selected_metadata_sha256=model['selected_metadata']['sha256'],selected_metadata_name=Path(model['selected_metadata']['path']).name)
    q['content_identity']=identity;q['query_id']=canonical_hash(identity)


def test_private_forward_restores_actual_model_and_preserves_dtype_ids_without_training(saved_query,tmp_path):
    m=api();path,old,request=saved_query
    before=m._snapshot_inventory(Path(request['query']['model_binding']['snapshot_root']))
    result=m._saved_query(request,output=tmp_path/'output')
    with np.load(path.parent/'prediction.npz',allow_pickle=False) as expected,np.load(tmp_path/'output/prediction.npz',allow_pickle=False) as actual:
        assert actual.files==['prediction','row_id','group_id']
        for name in actual.files:
            assert actual[name].dtype==expected[name].dtype and np.array_equal(actual[name],expected[name])
    assert result['status']=='completed_ineligible' and result['finite_aligned_prediction_artifact'] is True
    assert result['logical_fit_seed']==2**61+17
    assert result['protected_access_authorized'] is result['native_qualification_authenticated'] is result['T_accessed'] is False
    assert before==m._snapshot_inventory(Path(request['query']['model_binding']['snapshot_root']))
    with pytest.raises(ValueError):m._saved_query(request,output=tmp_path/'output')


@pytest.mark.parametrize('fault',['source','query_id','content','role','label_archive','labels','float32','duplicate_rows','group_dtype','nan',
    'timestamp_missing','timestamp_order','family','feature_schema','snapshot_hash','inventory_hash','unknown_field','query_hash'])
def test_invalid_input_stops_before_method_import(tmp_path,fault):
    m=api();path,old,request=query_fixture(tmp_path,temporal=fault.startswith('timestamp'))
    q=request['query'];feature=Path(q['feature_reference']['path'])
    if fault=='source':request['source_sha256']='0'*64
    elif fault=='query_id':q['query_id']='0'*64
    elif fault=='content':q['content_identity']['fit_seed']+=1
    elif fault=='role':q['query_role']='V_features'
    elif fault=='label_archive':q['feature_reference']['path']=str(feature.with_name('T-labels.npz'))
    elif fault=='labels':rewrite_archive(feature,lambda x:x.update(y=np.zeros(37)))
    elif fault=='float32':rewrite_archive(feature,lambda x:x.update(X=x['X'].astype(np.float32)))
    elif fault=='duplicate_rows':rewrite_archive(feature,lambda x:x['row_id'].__setitem__(1,x['row_id'][0]))
    elif fault=='group_dtype':rewrite_archive(feature,lambda x:x.update(group_id=x['group_id'].astype(np.float64)))
    elif fault=='nan':rewrite_archive(feature,lambda x:x['X'].__setitem__((0,0),np.nan))
    elif fault=='timestamp_missing':rewrite_archive(feature,lambda x:x.pop('timestamps'))
    elif fault=='timestamp_order':rewrite_archive(feature,lambda x:x.update(timestamps=x['timestamps'][::-1]))
    elif fault=='family':q['family_id']='P08'
    elif fault=='feature_schema':request['feature_schema']['columns']=['other','kind']
    elif fault=='snapshot_hash':q['model_binding']['selected_metadata']['sha256']='0'*64
    elif fault=='inventory_hash':q['model_binding']['snapshot_inventory_sha256']='0'*64
    elif fault=='unknown_field':request['T_labels']='forbidden'
    elif fault=='query_hash':feature.write_bytes(feature.read_bytes()+b'drift')
    if fault in ('labels','float32','duplicate_rows','group_dtype','nan','timestamp_missing','timestamp_order'):
        q['feature_reference']['sha256']=sha(feature)
    if fault not in ('content','query_id'):rebind(request)
    with pytest.raises((ValueError,FileNotFoundError)):m._saved_query(request,output=tmp_path/'output')
    terminal=json.loads((tmp_path/'output/terminal.json').read_text())
    assert terminal['adapter_import_started'] is False
    assert not (tmp_path/'output/prediction.npz').exists()


def test_temporal_query_retains_actual_timestamps_exactly(tmp_path):
    m=api();path,_,request=query_fixture(tmp_path,temporal=True)
    result=m._saved_query(request,output=tmp_path/'output')
    with np.load(tmp_path/'output/prediction.npz',allow_pickle=False) as actual,np.load(path.parent/'prediction.npz',allow_pickle=False) as old:
        assert np.array_equal(actual['timestamps'],old['timestamps']) and actual['timestamps'].dtype==old['timestamps'].dtype
    assert result['query_rows']==37


def test_nonfinite_forward_or_mutated_snapshot_never_publishes_prediction(saved_query,tmp_path,monkeypatch):
    m=api();_,_,request=saved_query
    from treeneuron.research.p13_simple import P13Predictor
    monkeypatch.setattr(P13Predictor,'predict',lambda *a,**kw:np.full(37,np.nan))
    with pytest.raises(FloatingPointError):m._saved_query(request,output=tmp_path/'output')
    terminal=json.loads((tmp_path/'output/terminal.json').read_text());assert terminal['status']=='numerical_failure'
    assert not (tmp_path/'output/prediction.npz').exists()


def test_query_output_cannot_modify_snapshot_query_source_or_interpreter(saved_query,tmp_path):
    m=api();path,_,request=saved_query
    for out in (path.parent/'snapshot/output',Path(request['query']['feature_reference']['path']).parent/'output',Path(sys.executable).resolve().parent/'output'):
        with pytest.raises(ValueError):m._saved_query(request,output=out)
        assert not out.exists()


def test_public_preflight_checks_real_protocol_before_query_study_or_output(tmp_path):
    m=api();kwargs=dict(models_reference=dict(path=str(tmp_path/'forbidden-model'),sha256='a'*64),
        global_barrier_reference=dict(path=str(tmp_path/'forbidden-global'),sha256='b'*64),
        qualification_reference=dict(root=str(tmp_path/'false-registry'),inventory_sha256='c'*64),
        preT_costs_reference=dict(path=str(tmp_path/'forbidden-cost'),sha256='d'*64),
        readiness_reference=dict(path=str(tmp_path/'forbidden-readiness'),sha256='e'*64))
    with pytest.raises((ValueError,FileNotFoundError)) as error:m.prepare_protected_query(dict(path=str(tmp_path/'forbidden-protocol'),sha256='f'*64),
        directory=tmp_path/'forbidden-study',query_id='0'*64,episode_root=tmp_path/'output',**kwargs)
    assert all(s not in str(error.value) for s in ('forbidden-model','forbidden-protocol','forbidden-study','forbidden-cost'))
    assert list(tmp_path.iterdir())==[]


def test_snapshot_change_during_forward_is_detected_before_publication(saved_query,tmp_path,monkeypatch):
    m=api();_,_,request=saved_query
    from treeneuron.research.p13_simple import P13Predictor
    original=P13Predictor.predict
    def corrupt(model,*args,**kwargs):
        result=original(model,*args,**kwargs)
        with (Path(request['query']['model_binding']['snapshot_root'])/'arrays.npz').open('ab') as stream:stream.write(b'changed')
        return result
    monkeypatch.setattr(P13Predictor,'predict',corrupt)
    with pytest.raises(ValueError,match='changed'):m._saved_query(request,output=tmp_path/'output')
    assert not (tmp_path/'output/prediction.npz').exists()


def test_forward_does_not_call_fit_with_logical_large_seed(saved_query,tmp_path,monkeypatch):
    m=api();_,_,request=saved_query
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*args,**kwargs):raise AssertionError('Saved inference must never fit or refit')
    monkeypatch.setattr(P13Predictor,'fit',forbidden)
    out=m._saved_query(request,output=tmp_path/'output')
    assert out['logical_fit_seed']>2**32


@pytest.mark.parametrize('family',['P00','P12','P13'])
def test_saved_forward_in_fresh_process_keeps_native_snapshot_and_exact_values(tmp_path,family):
    import os
    import subprocess
    project=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(project/'src'),PYTHONDONTWRITEBYTECODE='1')
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    code='''import runpy,sys,json
from pathlib import Path
import numpy as np
from treeneuron.research.method_worker import _initialize_native_threads
_initialize_native_threads(sys.argv[3])
sys.path.insert(0,str(Path(sys.argv[1]).parent))
namespace=runpy.run_path(sys.argv[1]);root=Path(sys.argv[2])
path,old,request=namespace["query_fixture"](root,family=sys.argv[3])
from treeneuron.research.protected_query_v2 import _saved_query
out=_saved_query(request,output=root/"output")
with np.load(path.parent/"prediction.npz",allow_pickle=False) as expected,np.load(root/"output/prediction.npz",allow_pickle=False) as actual:
 for name in actual.files:assert actual[name].dtype==expected[name].dtype and np.array_equal(actual[name],expected[name])
assert out["finite_aligned_prediction_artifact"] is True and out["protected_access_authorized"] is False
if sys.argv[3]=="P13":assert out["loaded_heavy_modules"]==[]
print(json.dumps(dict(family=sys.argv[3],query_rows=out["query_rows"],T_accessed=out["T_accessed"])))
'''
    done=subprocess.run([sys.executable,'-P','-B','-c',code,str(Path(__file__).resolve()),str(tmp_path),family],
        env=env,cwd=project,capture_output=True,text=True,timeout=180)
    assert done.returncode==0,done.stdout+'\n'+done.stderr
    assert json.loads(done.stdout.splitlines()[-1])==dict(family=family,query_rows=37,T_accessed=False)


def test_metadata_compilation_does_not_open_arrays_and_refuses_repeated_ambiguous_query(saved_query,tmp_path,monkeypatch):
    m=api();_,_,request=saved_query;q=request['query']
    data=dict(feature_names=request['feature_schema']['columns'],categorical_features=request['feature_schema']['categorical'],manifest_hash=q['manifest_hash'],
        folds=[dict(outer_fold=0,split_hash=q['split_hash'],roles={'T_features':dict(reference=q['feature_reference'])})])
    payload=dict(source_sha256=request['source_sha256'],prediction_manifest=dict(manifest_hash=request['prediction_manifest_sha256'],queries=[q]),data={'toy':data})
    state=dict(phase='owners_ready',active=None,committed_queries=[],last_failures={},reruns={},binding=dict(protocol_sha256=request['protocol_sha256']))
    def forbidden(*args,**kwargs):raise AssertionError('Metadata compiler must not read feature/model bytes')
    monkeypatch.setattr(Path,'read_bytes',forbidden)
    assert m._compile_query_request(payload,state,query_id=q['query_id'])==request
    for changed in (dict(state,phase='labels_started'),dict(state,active=(q['query_id'],0)),dict(state,committed_queries=[q['query_id']]),
        dict(state,last_failures={q['query_id']:'numerical_failure'}),dict(state,last_failures={q['query_id']:'input_or_dependency_failure'},reruns={q['query_id']:1})):
        with pytest.raises(ValueError):m._compile_query_request(payload,changed,query_id=q['query_id'])


def test_disk_full_while_serializing_never_publishes_partial_prediction(saved_query,tmp_path,monkeypatch):
    m=api();_,_,request=saved_query
    def no_space(stream,**arrays):
        stream.write(b'incomplete archive')
        raise OSError(28,'No space left on device')
    monkeypatch.setattr(np,'savez',no_space)
    with pytest.raises(OSError):m._saved_query(request,output=tmp_path/'output')
    assert not (tmp_path/'output/prediction.npz').exists()
    terminal=json.loads((tmp_path/'output/terminal.json').read_text())
    assert terminal['status']=='resource_exhausted' and terminal['finite_aligned_prediction_artifact'] is False


@pytest.mark.parametrize('estimator',['ridge','spline'])
def test_nonconstant_reference_forward_preserves_fitted_geometry_and_unseen_category(saved_query,tmp_path,estimator):
    import pandas as pd
    m=api();_,_,request=saved_query
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research.p13_simple import P13Predictor
    schema=request['feature_schema'];x=np.linspace(-2.,2.,24);kind=np.arange(24)%2
    F=pd.DataFrame(dict(x=x,kind=kind));y=3+2*x+.3*np.sin(x)+1.5*kind
    G=pd.DataFrame(dict(x=[-.7,.3,.8,1.1],kind=[0,1,0,1]));yg=3+2*G.x.to_numpy()+.3*np.sin(G.x.to_numpy())+1.5*G.kind.to_numpy()
    parameters=dict(alpha=.01)
    if estimator=='spline':parameters['n_knots']=4
    config=dict(schema='research/v2-p13-config',family='P13',estimator=estimator,feature_schema=schema,parameters=parameters)
    model=P13Predictor().fit(FitRequest(F,y,G,yg,17,config,dict(max_seconds=30.),tmp_path/'nonconstant-fit'))
    snapshot=tmp_path/'nonconstant-snapshot';model.save(snapshot)
    q=request['query'];binding=q['model_binding'];binding.update(snapshot_root=str(snapshot),snapshot_inventory_sha256=canonical_hash(m._snapshot_inventory(snapshot)),
        selected_metadata=dict(path=str(snapshot/'manifest.json'),sha256=sha(snapshot/'manifest.json')))
    feature=Path(q['feature_reference']['path']);rewrite_archive(feature,lambda a:a['X'].__setitem__((0,1),2.))
    q['feature_reference']['sha256']=sha(feature);rebind(request)
    with np.load(feature,allow_pickle=False) as z:expected=np.asarray(model.predict(pd.DataFrame(z['X'],columns=schema['columns'])))
    assert np.ptp(expected)>1
    result=m._saved_query(request,output=tmp_path/'output')
    with np.load(tmp_path/'output/prediction.npz',allow_pickle=False) as actual:assert np.array_equal(actual['prediction'],expected)
    assert result['finite_aligned_prediction_artifact'] is True and result['protected_access_authorized'] is False
