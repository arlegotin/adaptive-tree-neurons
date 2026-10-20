"""Large feature-only queries require complete bounded, source-bound inference."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_native_generation_chain import actual


def api():
    assert importlib.util.find_spec('treeneuron.research.native_query_v2') is not None, 'Missing bounded native query prediction worker'
    from treeneuron.research import native_query_v2
    return native_query_v2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def native_model(tmp_path_factory):
    saved=[]
    def get():
        if not saved:
            root=tmp_path_factory.mktemp('native-query-model')
            with pytest.MonkeyPatch.context() as patch:args,old=actual(root,patch)
            saved.append((args,old))
        return saved[0]
    return get


def inputs(tmp_path,native_model,rows=65536):
    from treeneuron.research.immutable_inputs import stage_immutable_file
    from treeneuron.research.method_worker import scientific_source_sha256
    args,old=native_model();root=tmp_path/'inputs';root.mkdir()
    native=Path(args['native_reference']['root']);snapshot=native/'worker/snapshot'
    target=root/'snapshot';target.mkdir()
    for source in snapshot.rglob('*'):
        if source.is_dir():(target/source.relative_to(snapshot)).mkdir(exist_ok=True)
        else:stage_immutable_file(source,target/source.relative_to(snapshot),expected_sha256=sha(source))
    cfg=args['expected_configuration'];d=len(cfg['feature_schema']['columns'])
    X=np.arange(rows*d,dtype=np.float64).reshape(rows,d)/(rows*d)
    ids=np.arange(rows,dtype=np.int64)+1000000
    query=root/'query.npz';np.savez(query,X=X,row_id=ids,group_id=ids//17)
    request=dict(schema='research/v2-native-query-request',purpose='engineering_qualification',family_id='P13',
        scientific_source_sha256=scientific_source_sha256(),feature_schema=cfg['feature_schema'],
        snapshot=dict(path='snapshot',metadata_sha256=sha(target/'manifest.json')),
        query=dict(path='query.npz',sha256=sha(query)),chunk_rows=1024)
    path=root/'request.json';path.write_text(json.dumps(request))
    return path,request,args,old


def test_chunk_loop_preserves_every_row_and_bounded_call_sizes():
    module=api();calls=[];X=np.arange(15,dtype=np.float64).reshape(5,3)
    def predict(part):
        calls.append(len(part));return part[:,0]*2+part[:,1]**2
    values,count=module._predict_chunks(predict,X,chunk_rows=2)
    np.testing.assert_array_equal(values,X[:,0]*2+X[:,1]**2)
    assert calls==[2,2,1] and count==3


def test_real_closed_native_model_predicts_all_65536_rows_without_labels_or_heavy_imports(tmp_path,native_model):
    module=api();path,request,args,old=inputs(tmp_path,native_model)
    from treeneuron.research.workqueue import _inventory
    before=_inventory(old)
    result=module.predict_native_query(path,expected_request_sha256=sha(path),output=tmp_path/'prediction',interpreter=sys.executable)
    assert result['status']=='completed_ineligible' and result['child_reaped'] is True
    assert result['prediction_rows']==65536 and result['chunks_completed']==64
    assert result['labels_read'] is result['quality_eligible'] is result['protected_access_authorized'] is False
    with np.load(tmp_path/'prediction/worker/predictions.npz',allow_pickle=False) as out:
        assert set(out.files)=={'prediction','row_id','group_id'}
        assert out['prediction'].shape==(65536,) and np.isfinite(out['prediction']).all()
        np.testing.assert_array_equal(out['row_id'],np.arange(65536,dtype=np.int64)+1000000)
        with np.load(Path(args['native_reference']['root'])/'worker/predictions.npz',allow_pickle=False) as golden:
            assert np.all(out['prediction']==golden['prediction'][0])
    worker=json.loads((tmp_path/'prediction/worker/terminal.json').read_text())
    assert worker['fitting_performed'] is False and 'torch' not in worker['loaded_heavy_modules']
    assert _inventory(old)==before
    process=json.loads((tmp_path/'prediction/process/process.json').read_text())
    assert process['timeout_ns'] is None


@pytest.mark.parametrize('change',['labels','source','metadata','unknown','object'])
def test_forbidden_or_changed_inputs_fail_before_native_adapter_import(tmp_path,native_model,change):
    module=api();path,request,_,_=inputs(tmp_path,native_model,rows=3)
    if change in ('labels','object'):
        query=path.parent/'query.npz'
        with np.load(query,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if change=='labels':arrays['y']=np.zeros(3)
        else:arrays['X']=arrays['X'].astype(object)
        np.savez(query,**arrays);request['query']['sha256']=sha(query)
    elif change=='source':request['scientific_source_sha256']='0'*64
    elif change=='metadata':request['snapshot']['metadata_sha256']='0'*64
    else:request['target_path']='hidden-labels.npz'
    path.write_text(json.dumps(request))
    result=module.predict_native_query(path,expected_request_sha256=sha(path),output=tmp_path/'invalid',interpreter=sys.executable)
    assert result['status']=='input_or_dependency_failure' and result['quality_eligible'] is False
    worker=json.loads((tmp_path/'invalid/worker/terminal.json').read_text())
    assert worker['adapter_import_started'] is False and worker['fitting_performed'] is False
    assert not (tmp_path/'invalid/worker/predictions.npz').exists()


def test_owned_memory_stop_is_retained_without_any_predictions(tmp_path,native_model):
    module=api();path,request,_,_=inputs(tmp_path,native_model,rows=3)
    result=module.predict_native_query(path,expected_request_sha256=sha(path),output=tmp_path/'stopped',
        interpreter=sys.executable,rss_limit_bytes=1)
    assert result['status']=='resource_exhausted' and result['child_reaped'] is True
    assert result['quality_eligible'] is False and not (tmp_path/'stopped/worker/predictions.npz').exists()


@pytest.mark.parametrize('changed',['query','metadata','weights'])
def test_late_model_or_query_byte_change_cannot_publish_a_successful_outer_result(tmp_path,native_model,monkeypatch,changed):
    module=api();path,request,_,_=inputs(tmp_path,native_model,rows=3)
    from treeneuron.research import process_supervision_v2
    original=process_supervision_v2.supervise_process
    def mutate_only_staged_bytes_after_child_closed(**kwargs):
        result=original(**kwargs)
        assert result['execution_status']=='exited' and result['returncode']==0
        target=(path.parent/'query.npz' if changed=='query' else path.parent/'snapshot/manifest.json'
            if changed=='metadata' else next(p for p in (path.parent/'snapshot').rglob('*') if p.is_file() and p.name!='manifest.json'))
        raw=target.read_bytes();target.unlink();target.write_bytes(raw+b'\n')
        return result
    monkeypatch.setattr(process_supervision_v2,'supervise_process',mutate_only_staged_bytes_after_child_closed)
    with pytest.raises(ValueError):
        module.predict_native_query(path,expected_request_sha256=sha(path),output=tmp_path/'changed',interpreter=sys.executable)
    assert not (tmp_path/'changed/result.json').exists()
