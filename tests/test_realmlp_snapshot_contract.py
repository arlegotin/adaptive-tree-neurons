"""Fresh native snapshot boundary checks; tiny F/G fixture, no quality anchor."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def test_realmlp_snapshot_config_graph_payload_and_runtime_bindings(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),
        PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,str(Path(__file__).resolve()),'--worker',str(tmp_path/'worker')]
    started=time.perf_counter_ns();result=None;error=None
    try:result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
    except BaseException as caught:error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),
            environment={key:env.get(key) for key in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','XDG_CACHE_HOME',
                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')},
            returncode=result.returncode if result else None,error=error,wall_ns=time.perf_counter_ns()-started),indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result else str(error))
    assert result.returncode==0,result.stdout


def worker(output):
    import copy
    import hashlib
    import shutil
    import numpy as np
    import pandas as pd
    from treeneuron.baselines.api import FitRequest
    from treeneuron.research import realmlp_native as api
    from treeneuron.research.realmlp_snapshot import walk
    from test_realmlp_contract import config
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026101819)
    def draw(n):
        x=rng.normal(size=n)
        return pd.DataFrame(dict(x=x,kind=['a' if i%2 else 'b' for i in range(n)])),(2+x+.2*x*x).astype(np.float32)
    F,yF=draw(96);G,yG=draw(19)
    cfg=config('engineering_fixture',max_epochs=3)
    model=api.RealMLPPredictor().fit(FitRequest(F,yF,G,yG,17,cfg,{},output/'fit'))
    model.save(output/'snapshot');original=json.loads((output/'snapshot/manifest.json').read_text())
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    expected=sha(output/'snapshot/manifest.json')
    restored=api.RealMLPPredictor.load(output/'snapshot',expected_manifest_sha256=expected)
    np.testing.assert_array_equal(model.predict(G),restored.predict(G))
    checks=[]
    # The native image inspection must run on load, not merely at original fit.
    original_runtime=api._runtime
    calls=[]
    def refuse():
        calls.append('actual_native_image_authentication')
        raise ValueError('simulated actual-native-image authentication failure')
    api._runtime=refuse
    try:
        try:api.RealMLPPredictor.load(output/'snapshot',expected_manifest_sha256=expected)
        except ValueError as error:assert 'actual-native-image' in str(error)
        else:raise AssertionError('Load skipped actual native runtime authentication')
    finally:api._runtime=original_runtime
    assert calls==['actual_native_image_authentication']
    checks.append(dict(case='actual_runtime_authentication',rejected=True))
    original_dependencies=api._dependencies
    api._dependencies=lambda:(_ for _ in ()).throw(AssertionError('Native imports before trusted digest'))
    try:
        try:api.RealMLPPredictor.load(output/'snapshot',expected_manifest_sha256='0'*64)
        except ValueError as error:assert 'trusted' in str(error)
        else:raise AssertionError('Untrusted native snapshot accepted')
    finally:api._dependencies=original_dependencies
    checks.append(dict(case='external_trusted_digest_before_import',rejected=True))
    cases=['config_lr','native_lr','promoted_quality','negative_MSE','wrong_selected_epoch','float64_payload',
        'unknown_function','NTP_factor','reordered_vocabulary','changed_hidden_layout','changed_source',
        'negative_target_std','unknown_graph_field','state_identity_mismatch','fixture_to_search',
        'retained_runtime_image','retained_executable','retained_python','retained_platform',
        'wrong_preprocessing_function','reordered_preprocessing']
    accepted=[]
    for name in cases:
        directory=output/'rejected'/name;directory.mkdir(parents=True)
        value=copy.deepcopy(original);state=value['state'];resources=value['resources']
        shutil.copyfile(output/'snapshot/weights.npz',directory/'weights.npz')
        if name=='retained_runtime_image':resources['runtime']['native_images'][0]['sha256']='0'*64
        elif name=='retained_executable':resources['dependencies']['executable_sha256']='0'*64
        elif name=='retained_python':resources['dependencies']['python']='changed-python'
        elif name=='retained_platform':resources['dependencies']['platform']='changed-platform'
        elif name=='wrong_preprocessing_function':
            next(n for n in walk(state['model']) if n['type']=='FunctionLayer')['attributes']['function']={'name':'relu'}
        elif name=='reordered_preprocessing':
            group=next(n for n in walk(state['model']) if n['type']=='SequentialLayer' and len(n['children'])>=4
                and [x['type'] for x in n['children']][1:3]==['BiasLayer','ScaleLayer'])
            group['children'][1],group['children'][2]=group['children'][2],group['children'][1]
        elif name=='config_lr':value['config']['parameters']['lr']=.1
        elif name=='native_lr':resources['native_parameters']['lr']=.1
        elif name=='promoted_quality':resources.update(quality_eligible=True,checkpoint_eligible=True,status='completed_eligible')
        elif name=='negative_MSE':resources['history'][0]['G_mse']=-1.
        elif name=='wrong_selected_epoch':resources['selected_epoch']=1+(resources['selected_epoch']%3)
        elif name=='unknown_function':
            next(n for n in walk(state['model']) if n['type']=='ParametricActivationLayer')['attributes']['function']={'name':'eval'}
        elif name=='NTP_factor':
            next(n for n in walk(state['model']) if n['type']=='WeightLayer')['attributes']['factor']*=2
        elif name=='reordered_vocabulary':state['vocabulary']['categories'][0].reverse()
        elif name=='changed_hidden_layout':
            value['config']['parameters']['hidden_sizes']=[512]
            resources['config']=copy.deepcopy(value['config']);resources['native_parameters']['hidden_sizes']=[512]
        elif name=='changed_source':value['source']['package_sha256']='0'*64
        elif name=='unknown_graph_field':state['model']['eval']='arbitrary_function'
        elif name=='state_identity_mismatch':resources['selected_inference_state_sha256']='0'*64
        elif name=='fixture_to_search':
            value['config'].pop('max_epochs');value['config']['mode']='search'
            resources['config']=copy.deepcopy(value['config']);resources['mode']='search';resources['native_parameters']['n_epochs']=256
        else:
            with np.load(directory/'weights.npz',allow_pickle=False) as archive:arrays={k:archive[k].copy() for k in archive.files}
            if name=='float64_payload':arrays[next(iter(arrays))]=arrays[next(iter(arrays))].astype(np.float64)
            else:
                node=next(n for n in walk(state['model']) if n['type']=='NormalizeOutputLayer')
                arrays[node['parameters']['std']['tensor']].fill(-1.)
            np.savez(directory/'weights.npz',**arrays)
            state['payload']={k:dict(dtype=str(v.dtype),shape=list(v.shape),sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k,v in arrays.items()}
            value['weights_sha256']=sha(directory/'weights.npz')
        if name!='state_identity_mismatch':resources['selected_inference_state_sha256']=api._canonical(state)
        (directory/'manifest.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
        started=time.perf_counter_ns()
        try:api.RealMLPPredictor.load(directory,expected_manifest_sha256=sha(directory/'manifest.json'))
        except ValueError as error:
            checks.append(dict(case=name,rejected=True,reason=str(error),wall_ns=time.perf_counter_ns()-started))
            (directory/'rejection.json').write_text(json.dumps(checks[-1],indent=2)+'\n')
        else:
            accepted.append(name);checks.append(dict(case=name,rejected=False))
    (output/'checks.json').write_text(json.dumps(dict(rejections=checks,count=len(checks),full_quality_anchor_executed=False,incorrectly_accepted=accepted),indent=2)+'\n')
    assert not accepted, 'Incoherent native snapshots accepted: '+repr(accepted)


if __name__=='__main__':
    assert sys.argv[1]=='--worker';worker(Path(sys.argv[2]))
