"""Fresh-process engineering fits with explicit label and artifact boundaries."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def api():
    from treeneuron.research import method_worker
    return method_worker


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loaded_method_telemetry_uses_the_real_NODE_vendor_namespace():
    import treeneuron.baselines._node_vendor
    assert 'treeneuron.baselines._node_vendor' in api().loaded_method_modules()


def request_fixture(root,temporal=False):
    root.mkdir();files={}
    for role,n,offset in (('F',12,0),('G',4,100),('V_features',5,200)):
        rows=np.arange(n,dtype=np.int64)+offset
        arrays=dict(X=np.column_stack((np.arange(n,dtype=np.float64),np.arange(n)%2)),row_id=rows,group_id=rows)
        if temporal:arrays['timestamps']=np.datetime64('2020-01-01T00','h')+rows.astype('timedelta64[h]')
        if role!='V_features':arrays['y']=3.+2*arrays['X'][:,0]
        path=root/(role+'.npz');np.savez(path,**arrays)
        files[role]=dict(path=path.name,sha256=sha(path))
    value=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id='fixture/P13/mean',family_id='P13',fit_seed=17,
        scientific_source_sha256=api().scientific_source_sha256(),
        config=dict(schema='research/v2-p13-config',family='P13',estimator='mean',
            feature_schema=dict(columns=['x','kind'],categorical=['kind']),parameters={}),
        budget=dict(max_seconds=30.),observation_identity=dict(dataset='synthetic_fixture',split_kind='temporal' if temporal else 'grouped',split_sha256='a'*64),files=files)
    path=root/'request.json';path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return path,value


def write_request(path,value):path.write_text(json.dumps(value,sort_keys=True)+'\n');return sha(path)


@pytest.mark.parametrize('change',['source','request_digest','forbidden_V_labels','file_bytes',
    'row_overlap','group_overlap','object_array','escaped_path','unknown_role','seed_bool','unsupported_family',
    'bad_time_gap','time_reversal'])
def test_invalid_engineering_request_rejects_before_adapter_import(change,tmp_path):
    path,value=request_fixture(tmp_path/'input',temporal=change in ('bad_time_gap','time_reversal'))
    if change=='source':value['scientific_source_sha256']='0'*64
    elif change=='forbidden_V_labels':
        file=path.parent/'V_features.npz'
        with np.load(file,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        np.savez(file,**arrays,y=np.zeros(5));value['files']['V_features']['sha256']=sha(file)
    elif change=='file_bytes':(path.parent/'F.npz').write_bytes(b'not the committed array')
    elif change in ('row_overlap','group_overlap','object_array'):
        file=path.parent/'G.npz'
        with np.load(file,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if change=='row_overlap':arrays['row_id'][0]=0
        elif change=='group_overlap':arrays['group_id'][0]=0
        else:arrays['X']=arrays['X'].astype(object)
        np.savez(file,**arrays);value['files']['G']['sha256']=sha(file)
    elif change=='escaped_path':value['files']['F']['path']='../F.npz'
    elif change=='unknown_role':value['files']['T']=value['files']['F']
    elif change=='seed_bool':value['fit_seed']=True
    elif change=='unsupported_family':value['family_id']='P14'
    elif change in ('bad_time_gap','time_reversal'):
        file=path.parent/'G.npz'
        with np.load(file,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if change=='bad_time_gap':arrays['timestamps']=np.datetime64('2020-01-01T12','h')+np.arange(4).astype('timedelta64[h]')
        else:arrays['timestamps']=arrays['timestamps'][::-1]
        np.savez(file,**arrays);value['files']['G']['sha256']=sha(file)
    expected=write_request(path,value)
    if change=='request_digest':expected='0'*64
    output=tmp_path/'attempt'
    with pytest.raises(ValueError):api().run_engineering_trial(path,expected_request_sha256=expected,output=output)
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['status']=='input_or_dependency_failure' and terminal['adapter_import_started'] is False
    assert terminal['quality_eligible'] is False and terminal['T_accessed'] is False


@pytest.mark.parametrize('temporal',[False,True])
def test_actual_P13_worker_preserves_query_rows_and_closed_reload_without_torch(temporal,tmp_path):
    path,value=request_fixture(tmp_path/'input',temporal=temporal);output=tmp_path/'attempt'
    root=Path(__file__).resolve().parents[1]
    env=os.environ|dict(PYTHONPATH=str(root/'src'),PYTHONDONTWRITEBYTECODE='1')
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    argv=[sys.executable,'-B','-m','treeneuron.research.method_worker','--request',str(path),
        '--expected-request-sha256',sha(path),'--output',str(output)]
    result=subprocess.run(argv,cwd=root,env=env,capture_output=True,text=True,timeout=60)
    (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),returncode=result.returncode),indent=2)+'\n')
    (tmp_path/'stdout.log').write_text(result.stdout);(tmp_path/'stderr.log').write_text(result.stderr)
    assert result.returncode==0,result.stderr
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['status']=='completed_ineligible' and terminal['quality_eligible'] is False
    assert terminal['checkpoint_eligible'] is False and terminal['T_accessed'] is False
    assert terminal['scientific_source_sha256']==value['scientific_source_sha256']
    assert terminal['loaded_heavy_modules']==[] and terminal['snapshot_reload_exact'] is True
    assert terminal['accessed_roles']==['F_labels','G_labels','V_features']
    with np.load(output/'predictions.npz',allow_pickle=False) as z:
        np.testing.assert_array_equal(z['row_id'],np.arange(200,205))
        np.testing.assert_array_equal(z['group_id'],np.arange(200,205))
        np.testing.assert_array_equal(z['prediction'],np.full(5,14.))
        assert set(z.files)=={'row_id','group_id','prediction'}|({'timestamps'} if temporal else set())
        if temporal:np.testing.assert_array_equal(z['timestamps'],np.datetime64('2020-01-01T00','h')+np.arange(200,205).astype('timedelta64[h]'))
    for file in terminal['files']:
        p=output/file['path'];assert sha(p)==file['sha256'] and p.stat().st_size==file['bytes']
    previous=(output/'terminal.json').read_bytes()
    with pytest.raises(ValueError,match='Fresh'):api().run_engineering_trial(path,expected_request_sha256=sha(path),output=output)
    assert (output/'terminal.json').read_bytes()==previous


@pytest.mark.parametrize('failure',['input_replaced','storage_full'])
def test_late_worker_fault_preserves_a_failed_attempt_and_never_publishes_success(failure,tmp_path,monkeypatch):
    import errno
    from treeneuron.research.p13_simple import P13Predictor
    path,value=request_fixture(tmp_path/'input');original=P13Predictor.save
    def save(model,snapshot):
        if failure=='storage_full':raise OSError(errno.ENOSPC,'controlled no-space fixture')
        original(model,snapshot)
        (path.parent/'F.npz').write_bytes(b'changed after the original fit')
    monkeypatch.setattr(P13Predictor,'save',save)
    output=tmp_path/'attempt'
    expected=OSError if failure=='storage_full' else ValueError
    with pytest.raises(expected):api().run_engineering_trial(path,expected_request_sha256=sha(path),output=output)
    terminal=json.loads((output/'terminal.json').read_text())
    assert terminal['status']==('resource_exhausted' if failure=='storage_full' else 'input_or_dependency_failure')
    assert terminal['quality_eligible'] is False and terminal['adapter_import_started'] is True
    assert (output/'fit/fit-receipt.json').exists()
    assert terminal['files'] and 'traceback' in terminal
