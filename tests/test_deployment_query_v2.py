"""F-only deterministic workload fixtures never authenticate native roles."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.deployment_query_v2'), 'Missing target-free F workload projection'
    from treeneuron.research import deployment_query_v2
    return deployment_query_v2


def fixture(root):
    from treeneuron.research.deployment_worker_v2 import _source
    from test_native_deployment_worker_v2 import ref
    directory=root/'data/outer-0';directory.mkdir(parents=True);path=directory/'F.npz'
    np.savez(path,X=np.array([[.1],[.2],[.3]],dtype=np.float64),y=np.array([1.,9.,7.]),
        row_id=np.array([11,12,13],dtype=np.int64),group_id=np.array([0,0,1],dtype=np.int64))
    meta=dict(schema='research/v2-F-deployment-input',source_sha256=_source(),role='F',dataset='known-fixture',outer_fold=0,
        manifest_hash='a'*64,split_hash='b'*64,F_reference=ref(path),F_bytes=path.stat().st_size,
        feature_schema=dict(columns=['x'],categorical=[]),temporal=False)
    return path,meta


def test_three_source_rows_cycle_deterministically_without_decoding_any_labels(tmp_path,monkeypatch):
    module=api();path,meta=fixture(tmp_path)
    original=np.lib.npyio.NpzFile.__getitem__
    def features_only(self,key):
        assert key in ('X','row_id'),'Workload projection must not decode labels/group/time'
        return original(self,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',features_only)
    result=module.build_F_workload_query(meta,output=tmp_path/'projection')
    with np.load(result['query_reference']['path'],allow_pickle=False) as z:
        assert z.files==['X','row_id'] and z['X'].dtype==np.float32
        np.testing.assert_array_equal(z['X'][:6,0],np.array([.1,.2,.3,.1,.2,.3],dtype=np.float32))
        assert [int(np.sum(z['X'][:,0]==np.float32(x))) for x in (.1,.2,.3)]==[342,341,341]
        np.testing.assert_array_equal(z['row_id'],np.arange(1024,dtype=np.int64))
    readback=module.inspect_F_workload_query(result['manifest_reference'],expected_metadata=meta)
    assert readback['numeric_projection_checked'] is True
    assert readback['label_values_decoded'] is readback['original_F_role_authenticated'] is readback['T_accessed'] is False


@pytest.mark.parametrize('fault',['source','role','hash','schema','dtype','duplicates','nonfinite','float32_overflow','extra_field'])
def test_invalid_source_or_feature_geometry_rejects_before_output(tmp_path,fault):
    module=api();path,meta=fixture(tmp_path)
    from test_native_deployment_worker_v2 import ref
    if fault=='source':meta['source_sha256']='0'*64
    elif fault=='role':meta['role']='T'
    elif fault=='hash':meta['F_reference']['sha256']='0'*64
    elif fault=='schema':meta['feature_schema']['columns']=['other','extra']
    else:
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
        if fault=='dtype':arrays['X']=arrays['X'].astype(np.float32)
        elif fault=='duplicates':arrays['row_id'][1]=arrays['row_id'][0]
        elif fault=='nonfinite':arrays['X'][0,0]=np.nan
        elif fault=='float32_overflow':arrays['X'][0,0]=1e40
        else:arrays['proxy_target']=np.ones(3)
        np.savez(path,**arrays);meta['F_reference']=ref(path);meta['F_bytes']=path.stat().st_size
    with pytest.raises(ValueError):module.build_F_workload_query(meta,output=tmp_path/'projection')
    assert not (tmp_path/'projection').exists()


def test_projection_cannot_write_inside_source_data_or_overwrite_existing_output(tmp_path):
    module=api();path,meta=fixture(tmp_path);before=path.read_bytes()
    with pytest.raises(ValueError):module.build_F_workload_query(meta,output=path.parent.parent/'projection')
    output=tmp_path/'existing';output.mkdir();(output/'user.txt').write_text('keep')
    with pytest.raises(ValueError):module.build_F_workload_query(meta,output=output)
    assert path.read_bytes()==before and (output/'user.txt').read_text()=='keep'


def test_changed_projection_or_provenance_fails_independent_readback(tmp_path):
    module=api();path,meta=fixture(tmp_path);result=module.build_F_workload_query(meta,output=tmp_path/'projection')
    query=Path(result['query_reference']['path'])
    with query.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):module.inspect_F_workload_query(result['manifest_reference'],expected_metadata=meta)


def test_wrong_projected_values_cannot_be_hidden_by_rehashing_all_records(tmp_path):
    module=api();path,meta=fixture(tmp_path);result=module.build_F_workload_query(meta,output=tmp_path/'projection')
    from test_native_deployment_worker_v2 import ref
    query=Path(result['query_reference']['path'])
    with np.load(query,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['X']=np.roll(arrays['X'],1,axis=0);np.savez(query,**arrays)
    manifest_path=Path(result['manifest_reference']['path']);manifest=json.loads(manifest_path.read_text());manifest['query_reference']=ref(query)
    provenance_path=Path(result['provenance_reference']['path']);provenance=json.loads(provenance_path.read_text());provenance['query_reference']=ref(query)
    provenance_path.write_text(json.dumps(provenance)+'\n');manifest['provenance_reference']=ref(provenance_path)
    manifest_path.write_text(json.dumps(manifest)+'\n')
    with pytest.raises(ValueError,match='source projection'):module.inspect_F_workload_query(ref(manifest_path),expected_metadata=meta)


def test_changing_F_targets_does_not_change_projected_features(tmp_path):
    import copy
    module=api();path,meta=fixture(tmp_path);a=module.build_F_workload_query(meta,output=tmp_path/'first')
    from test_native_deployment_worker_v2 import ref
    with np.load(path,allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    arrays['y'][:]=100000.;np.savez(path,**arrays);changed=copy.deepcopy(meta);changed.update(F_reference=ref(path),F_bytes=path.stat().st_size)
    b=module.build_F_workload_query(changed,output=tmp_path/'second')
    with np.load(a['query_reference']['path'],allow_pickle=False) as left,np.load(b['query_reference']['path'],allow_pickle=False) as right:
        for name in ('X','row_id'):np.testing.assert_array_equal(left[name],right[name])


def test_disk_full_never_publishes_a_partial_query_archive(tmp_path,monkeypatch):
    import errno
    module=api();path,meta=fixture(tmp_path);before=path.read_bytes()
    def broken(stream,**arrays):stream.write(b'incomplete');raise OSError(errno.ENOSPC,'known fixture disk full')
    monkeypatch.setattr(np,'savez',broken)
    with pytest.raises(OSError):module.build_F_workload_query(meta,output=tmp_path/'projection')
    assert not (tmp_path/'projection/F-deployment-features.npz').exists()
    assert (tmp_path/'projection/.F-deployment-features.npz.pending').is_file() and path.read_bytes()==before


def test_projection_provenance_is_accepted_by_original_snapshot_worker_without_native_authority(tmp_path):
    module=api()
    from test_native_deployment_worker_v2 import measurement_fixture,ref
    from treeneuron.research import native_deployment_worker_v2 as worker
    known=tmp_path/'known';known.mkdir();request,value,_=measurement_fixture(known);F=known/'outer-0/F.npz'
    meta=dict(schema='research/v2-F-deployment-input',source_sha256=value['source_sha256'],role='F',dataset='known-fixture',outer_fold=0,
        manifest_hash='a'*64,split_hash='b'*64,F_reference=ref(F),F_bytes=F.stat().st_size,feature_schema=value['feature_schema'],temporal=False)
    query=module.build_F_workload_query(meta,output=tmp_path/'projection')
    value.update(query_reference=query['query_reference'],query_provenance_reference=query['provenance_reference'])
    request.write_text(json.dumps(value)+'\n');checked=worker._checked_measurement_request(request,ref(request)['sha256'])
    assert checked[3]['X'].shape==(1024,2)
    assert checked[4]['original_role_verified_by_native_controller'] is False
