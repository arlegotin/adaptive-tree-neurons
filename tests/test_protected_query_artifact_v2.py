"""Independent numerical readback, without qualification or test-label access."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from test_protected_query_v2 import saved_query,api,sha
from test_prediction_replay_v2 import rewrite_archive


def terminal_ref(path):return dict(path=str(path),sha256=sha(path))


def completed(tmp_path,saved_query):
    m=api();_,_,request=saved_query;root=tmp_path/'output';m._saved_query(request,output=root)
    return m,request,root,terminal_ref(root/'terminal.json')


def test_actual_saved_artifact_reader_rebuilds_rows_dtype_and_file_binding(saved_query,tmp_path):
    m,request,root,reference=completed(tmp_path,saved_query)
    assert hasattr(m,'inspect_saved_query_artifact'),'Missing independent saved-query artifact reader'
    out=m.inspect_saved_query_artifact(request,terminal_reference=reference)
    assert out['finite_aligned_prediction_artifact_checked'] is True and out['query_rows']==37
    assert out['prediction_dtype']=='float64'
    assert out['native_qualification_authenticated'] is out['protected_access_authorized'] is out['T_accessed'] is False
    assert out['terminal_reference']==reference and out['prediction_reference']['path']==str(root/'prediction.npz')


@pytest.mark.parametrize('fault',['terminal_hash','query_id','source','protocol','manifest','request','row_count','dtype','status','authority','labels_read',
    'elapsed_bool','unknown_field','prediction_bytes','prediction_labels','row_order','group_change','nonfinite','prediction_dtype','snapshot_after'])
def test_changed_or_nonfinite_committed_output_is_rejected(saved_query,tmp_path,fault):
    m,request,root,reference=completed(tmp_path,saved_query)
    assert hasattr(m,'inspect_saved_query_artifact'),'Missing independent saved-query artifact reader'
    path=root/'terminal.json';value=json.loads(path.read_text());pred=root/'prediction.npz'
    if fault=='terminal_hash':reference['sha256']='0'*64
    elif fault=='query_id':value['query_id']='0'*64
    elif fault=='source':value['source_sha256']='0'*64
    elif fault=='protocol':value['protocol_sha256']='0'*64
    elif fault=='manifest':value['prediction_manifest_sha256']='0'*64
    elif fault=='request':value['request_sha256']='0'*64
    elif fault=='row_count':value['query_rows']=36
    elif fault=='dtype':value['prediction_dtype']='float32'
    elif fault=='status':value['status']='completed_eligible'
    elif fault=='authority':value['protected_access_authorized']=True
    elif fault=='labels_read':value['labels_read']=True
    elif fault=='elapsed_bool':value['worker_body_ns']=True
    elif fault=='unknown_field':value['extra']=True
    elif fault=='prediction_bytes':pred.write_bytes(pred.read_bytes()+b'drift')
    elif fault=='prediction_labels':rewrite_archive(pred,lambda x:x.update(y=np.zeros(37)))
    elif fault=='row_order':rewrite_archive(pred,lambda x:x.update(row_id=x['row_id'][::-1]))
    elif fault=='group_change':rewrite_archive(pred,lambda x:x['group_id'].__setitem__(0,-1))
    elif fault=='nonfinite':rewrite_archive(pred,lambda x:x['prediction'].__setitem__(0,np.nan))
    elif fault=='prediction_dtype':rewrite_archive(pred,lambda x:x.update(prediction=x['prediction'].astype(np.float32)))
    elif fault=='snapshot_after':
        with (Path(request['query']['model_binding']['snapshot_root'])/'arrays.npz').open('ab') as stream:stream.write(b'drift')
    if fault in ('prediction_labels','row_order','group_change','nonfinite','prediction_dtype'):value['prediction_reference']['sha256']=sha(pred)
    if fault!='terminal_hash':
        path.write_text(json.dumps(value)+'\n');reference=terminal_ref(path)
    with pytest.raises(ValueError):m.inspect_saved_query_artifact(request,terminal_reference=reference)


def test_independent_reader_does_not_reload_or_predict_model(saved_query,tmp_path,monkeypatch):
    m,request,root,reference=completed(tmp_path,saved_query)
    assert hasattr(m,'inspect_saved_query_artifact'),'Missing independent saved-query artifact reader'
    from treeneuron.research.p13_simple import P13Predictor
    def forbidden(*a,**kw):raise AssertionError('Numerical file audit cannot fit, load, or predict again')
    monkeypatch.setattr(P13Predictor,'load',forbidden);monkeypatch.setattr(P13Predictor,'fit',forbidden);monkeypatch.setattr(P13Predictor,'predict',forbidden)
    result=m.inspect_saved_query_artifact(request,terminal_reference=reference)
    assert result['finite_aligned_prediction_artifact_checked'] is True


def test_temporal_readback_retains_clock_hours_and_rejects_shifted_predictions(tmp_path):
    from test_protected_query_v2 import query_fixture
    m=api();_,_,request=query_fixture(tmp_path,temporal=True);root=tmp_path/'output';m._saved_query(request,output=root)
    reference=terminal_ref(root/'terminal.json');out=m.inspect_saved_query_artifact(request,terminal_reference=reference)
    assert out['query_rows']==37
    rewrite_archive(root/'prediction.npz',lambda a:a.update(timestamps=a['timestamps']+np.timedelta64(1,'h')))
    v=json.loads((root/'terminal.json').read_text());v['prediction_reference']['sha256']=sha(root/'prediction.npz')
    (root/'terminal.json').write_text(json.dumps(v)+'\n')
    with pytest.raises(ValueError,match='row/group/time'):m.inspect_saved_query_artifact(request,terminal_reference=terminal_ref(root/'terminal.json'))


def test_reader_rejects_label_filename_before_any_label_bytes_are_read(saved_query,tmp_path,monkeypatch):
    m,request,root,reference=completed(tmp_path,saved_query)
    labels=tmp_path/'T-labels.npz';np.savez(labels,y=np.zeros(37))
    blocked=dict(path=str(labels),sha256=sha(labels));original=Path.read_bytes
    def no_label_read(path):
        if path.name=='T-labels.npz':raise AssertionError('Readback must reject label role before opening bytes')
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',no_label_read)
    with pytest.raises(ValueError):m.inspect_saved_query_artifact(request,terminal_reference=blocked)
    value=json.loads((root/'terminal.json').read_text());value['prediction_reference']=blocked
    (root/'terminal.json').write_text(json.dumps(value)+'\n')
    with pytest.raises(ValueError):m.inspect_saved_query_artifact(request,terminal_reference=terminal_ref(root/'terminal.json'))
