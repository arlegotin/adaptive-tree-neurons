"""Small official native F/G fits; safe snapshot representation remains gated."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_realmlp_contract import api


@pytest.mark.parametrize('categorical,constant_target,missing',[(False,False,False),(True,False,False),(False,True,False),(True,False,True)])
def test_tiny_official_fit_predict_and_sampler_exposure(categorical,constant_target,missing,tmp_path):
    assert hasattr(api().RealMLPPredictor,'fit'),'Native RealMLP fitting is missing'
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),
        PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,str(Path(__file__).resolve()),'--worker',str(int(categorical)),str(int(constant_target)),str(int(missing)),str(tmp_path/'worker')]
    started=time.perf_counter_ns();result=None;error=None
    try:
        result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),
            environment={k:env.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','XDG_CACHE_HOME',
                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')},
            wall_ns=time.perf_counter_ns()-started,error=error,returncode=result.returncode if result else None),indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result else str(error))
    assert result.returncode==0,result.stdout


def worker(categorical,constant_target,output,variant=None,missing=False):
    import numpy as np
    import pandas as pd
    import torch
    from treeneuron.baselines.api import FitRequest
    from test_realmlp_contract import config
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(2026101819)
    def draw(n,query=False):
        x=rng.uniform(-2,2,n)
        frame=pd.DataFrame(dict(x=x,constant=np.full(n,7.)))
        if categorical:
            frame['binary']=['a' if i%2 else 'b' for i in range(n)]
            frame['many']=[('new' if query and i%2 else 'k'+str(i%11)) for i in range(n)]
            if missing:
                frame.loc[frame.index%37==0,'binary']=None
                frame.loc[frame.index%41==0,'many']=np.nan
        return frame,np.full(n,7.,np.float32) if constant_target else np.asarray(4.+x+.3*x*x,np.float32)
    F,yF=draw(300);G,yG=draw(37);Q,_=draw(1031,True)
    schema=dict(columns=list(F),categorical=['binary','many'] if categorical else [])
    parameters={}
    if variant is not None:
        parameters=dict(num_emb_type=['none','pl','pbld','plr'][variant//3],act=['relu','selu','mish'][variant%3],
            hidden_sizes=[[256]*3,[64]*5,[512]][variant%3],add_front_scale=bool(variant%2))
    cfg=config('engineering_fixture',max_epochs=3,feature_schema=schema,parameters=parameters)
    model=api().RealMLPPredictor()
    try:model.fit(FitRequest(F,yF,G,yG,17,cfg,{},output/'fit'))
    except FloatingPointError as error:
        if variant!=10:raise
        summary=model.resource_summary()
        assert summary['status']=='numerical_failure' and summary['checkpoint_eligible'] is False
        observed=summary['native_observation']
        assert observed['applied_updates']==0 and len(observed['F_row_batches'])==1
        visited=set(sum(observed['F_row_batches'],[]))
        assert len(visited)==256 and observed['unvisited_F_row_ids']==sorted(set(range(300))-visited)
        (output/'numerical-failure.json').write_text(json.dumps(dict(variant=variant,parameters=parameters,
            seed=17,reason=str(error),summary=summary,native_source_reproduction='docs/research/v2/scripts/realmlp_case10_native_diagnosis.py',
            status='numerical_failure',eligible=False,support_removed=False),indent=2)+'\n')
        return
    pred=model.predict(Q);summary=model.resource_summary()
    assert pred.shape==(1031,) and np.isfinite(pred).all()
    assert np.all(pred>=float(yF.min())) and np.all(pred<=float(yF.max()))
    assert summary['checkpoint_eligible'] is False and summary['quality_eligible'] is False
    initialization=summary['initialization_observation']
    assert initialization['scope']=='native_creator' and initialization['creator_calls']==1
    assert initialization['root_F_positions']==list(range(300)) and initialization['subsamples']==[]
    assert initialization['status']=='completed' and initialization['calls']
    if constant_target:
        assert summary['selected_epoch']==3
        assert all(row['G_native_rmse']==0. and row['G_mse']==0. for row in summary['history'])
        np.testing.assert_array_equal(pred,np.full(1031,7.))
    assert summary['native_epochs_executed']==3 and summary['applied_updates']==3
    assert summary['epoch_rows_admitted']==300 and summary['rows_per_epoch']==256
    assert all(len(x)==44 for x in summary['omitted_row_ids'])
    admitted={i for rows in summary['admitted_row_ids'] for i in rows}
    assert summary['cumulative_visited_rows']==len(admitted)
    assert sorted(admitted|set(summary['unvisited_F_row_ids']))==list(range(300))
    # Direct pinned factory invocation with independently built F-only boundaries.
    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Regressor
    from pytabkit.models.data.conversion import ToDictDatasetConverter
    from pytabkit.models.data.data import DictDataset,TensorInfo
    from pytabkit.models.alg_interfaces.base import SplitIdxs,InterfaceResources
    from pytabkit.models.training.logging import StdoutLogger
    frames=[]
    for frame in (F,G,Q):
        frame=frame.copy()
        for name in schema['categorical']:
            frame[name]=frame[name].map(lambda value:np.nan if pd.isna(value) else 'str:'+value).astype(object)
        frames.append(frame)
    converter=ToDictDatasetConverter(cat_features=[x in schema['categorical'] for x in schema['columns']],verbosity=0)
    fd=converter.fit_transform(frames[0]);gd=converter.transform(frames[1])
    ds=DictDataset(tensors={**{k:torch.cat([fd.tensors[k],gd.tensors[k]]) for k in fd.tensors},
        'y':torch.from_numpy(np.concatenate([yF,yG])[:,None])},
        tensor_infos=fd.tensor_infos|{'y':TensorInfo(feat_shape=[1])})
    split=SplitIdxs(torch.arange(300)[None],torch.arange(300,337)[None],None,
        split_seed=17,sub_split_seeds=[int(np.random.RandomState(17).randint(0,2**31-1))],split_id=0)
    reference=RealMLP_TD_Regressor(device='cpu',random_state=17,n_cv=1,n_refit=0,n_repeats=1,n_threads=1,
        verbosity=0,n_epochs=3,batch_size=256,predict_batch_size=1024,n_ens=1,
        train_metric_name='mse',val_metric_name='rmse',use_early_stopping=False,**parameters)._create_alg_interface(n_cv=1)
    reference.fit(ds,[split],InterfaceResources(1,[]),StdoutLogger(0),[None],name='direct_realmlp_fixture')
    expected=reference.predict(converter.transform(frames[2])).numpy().reshape(-1).astype(np.float64)
    np.testing.assert_array_equal(pred,expected)
    assert summary['selected_epoch']==reference.fit_params[0]['stop_epoch']['rmse']
    model.save(output/'snapshot')
    import hashlib
    manifest_sha=hashlib.sha256((output/'snapshot/manifest.json').read_bytes()).hexdigest()
    (output/'query.json').write_text(Q.to_json(orient='table',double_precision=15))
    argv=[sys.executable,str(Path(__file__).resolve()),'--load',str(output/'snapshot'),manifest_sha,
        str(output/'query.json'),str(output/'loaded')]
    load_started=time.perf_counter_ns()
    load_env=dict(os.environ);load_env.pop('KMP_DUPLICATE_LIB_OK',None)
    result=subprocess.run(argv,env=load_env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    (output/'load-command.json').write_text(json.dumps(dict(argv=argv,returncode=result.returncode,
        wall_ns=time.perf_counter_ns()-load_started,caller_KMP_DUPLICATE_LIB_OK=load_env.get('KMP_DUPLICATE_LIB_OK')),indent=2)+'\n')
    (output/'load-worker.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    loaded=np.load(output/'loaded/predictions.npy',allow_pickle=False)
    np.testing.assert_array_equal(loaded,pred)
    np.savez(output/'predictions.npz',observed=pred,reference=expected,loaded=loaded,yF=yF,yG=yG)

    (output/'fixture-features.json').write_text(json.dumps(dict(F=F.astype(object).where(F.notna(),None).to_dict('list'),G=G.astype(object).where(G.notna(),None).to_dict('list'),
        Q=Q.astype(object).where(Q.notna(),None).to_dict('list'),schema=schema),indent=2)+'\n')
    (output/'parity.json').write_text(json.dumps(dict(categorical=categorical,constant_target=constant_target,missing=missing,variant=variant,parameters=parameters,
        prediction_max_abs_error=0.,reload_max_abs_error=0.,external_manifest_sha256=manifest_sha,
        official_selected_epoch=reference.fit_params[0]['stop_epoch']['rmse'],summary=summary,
        portable_snapshot_qualified=True,full_quality_anchor_executed=False),indent=2)+'\n')


def load_worker(snapshot,expected,query,output):
    import numpy as np
    import pandas as pd
    output.mkdir(parents=True,exist_ok=False)
    model=api().RealMLPPredictor.load(snapshot,expected_manifest_sha256=expected)
    frame=pd.read_json(query,orient='table',precise_float=True)
    np.save(output/'predictions.npy',model.predict(frame),allow_pickle=False)
    (output/'resources.json').write_text(json.dumps(model.resource_summary(),indent=2)+'\n')
    try:model.fit(None)
    except RuntimeError as error:assert 'inference-only' in str(error)
    else:raise AssertionError('Loaded native snapshot allowed training')


if __name__=='__main__':
    if sys.argv[1]=='--load':load_worker(Path(sys.argv[2]),sys.argv[3],Path(sys.argv[4]),Path(sys.argv[5]))
    elif sys.argv[1]=='--graph':
        variant=int(sys.argv[2]);worker(bool(variant%2),False,Path(sys.argv[3]),variant)
    else:
        assert sys.argv[1]=='--worker';worker(bool(int(sys.argv[2])),bool(int(sys.argv[3])),Path(sys.argv[5]),missing=bool(int(sys.argv[4])))
