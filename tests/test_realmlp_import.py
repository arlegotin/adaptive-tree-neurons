"""Fresh method-specific import and F/G converter qualification, without fitting."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from test_realmlp_contract import api, config


def test_actual_native_import_config_and_F_only_converter(tmp_path):
    assert hasattr(api(), '_dependencies'), 'RealMLP import authentication is missing'
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'src'),
        str(root/'environment/research-v2-realmlp-site')]), PYTHONDONTWRITEBYTECODE='1',
        MPLCONFIGDIR=str(tmp_path/'mpl'), XDG_CACHE_HOME=str(tmp_path/'cache'))
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): env[k]='1'
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    argv=[sys.executable,str(Path(__file__).resolve()),'--worker',str(tmp_path/'worker')]
    start=time.perf_counter_ns(); result=None;error=None
    try:
        result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,timeout=90)
    except BaseException as caught:
        error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),
            environment={k:env.get(k) for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR',
                'XDG_CACHE_HOME','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')},
            error=error,returncode=result.returncode if result else None,
            wall_ns=time.perf_counter_ns()-start),indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result else str(error))
    assert result.returncode==0,result.stdout


def worker(output):
    import numpy as np
    import pandas as pd
    output.mkdir(parents=True,exist_ok=False)
    module=api();deps=module._dependencies()
    assert deps['core_distribution_count']==27 and deps['overlay_distribution_count']==15
    cfg=module.RealMLPConfig.parse(config())
    native=module.native_config(cfg,7)
    assert native['val_metric_name']=='rmse', 'Pinned regression validation default is rmse, not the training mse alias'
    assert native['hidden_sizes']==[256]*3 and native['act']=='mish'
    assert native['use_parametric_act'] and native['weight_param']=='ntk'
    assert native['opt']=='adam' and native['lr']==.2 and native['sq_mom']==.95
    assert native['tfms']==['one_hot','median_center','robust_scale','smooth_clip','embedding']
    assert native['normalize_output'] and native['clamp_output'] and native['n_epochs']==256
    F=pd.DataFrame(dict(x=np.arange(96.),kind=['a','b']*48))
    G=pd.DataFrame(dict(x=[1e6,-1e6],kind=['new','a']))
    prepared=module.prepare_native_data(F,np.arange(96.),G,np.array([1e9,-1e9]),cfg,7)
    assert prepared['cardinalities']==[3]
    assert prepared['F_cat_codes']==[[1],[2]]*48
    assert prepared['G_cat_codes']==[[0],[1]]
    assert prepared['target_precision']=='float32'
    assert prepared['accessed_roles']==['F','G']
    assert prepared['split'].train_idxs.tolist()==[list(range(96))]
    assert prepared['split'].val_idxs.tolist()==[[96,97]]
    assert prepared['split'].test_idxs is None
    # Prove the installed observation callback rejects before native scoring;
    # this invokes no model constructor, optimizer, loader, or actual fit.
    import torch
    from types import SimpleNamespace
    from pytabkit.models.training.lightning_modules import TabNNModule
    from pytabkit.models.alg_interfaces.nn_interfaces import NNAlgInterface
    from pytabkit.models.training.metrics import Metrics
    bad=torch.tensor([[[float('nan')]]],dtype=torch.float32)
    assert torch.isfinite(Metrics.apply(bad[0],torch.ones((1,1)),'rmse'))
    original_callbacks,original_fit=TabNNModule.create_callbacks,NNAlgInterface.fit
    callback_invocations=[]
    TabNNModule.create_callbacks=lambda ignored:[]
    def callback_probe(interface,*args,**kwargs):
        observed=TabNNModule.create_callbacks(SimpleNamespace(val_preds=[bad]))[-1]
        callback_invocations.append('raw_validation_epoch_callback')
        observed.on_validation_epoch_end(None,SimpleNamespace(val_preds=[bad]))
        raise AssertionError('Nonfinite cached G reached the downstream native metric phase')
    NNAlgInterface.fit=callback_probe
    try:
        try:module._fit_native(dict(split=SimpleNamespace(n_train=2),dataset=None),{})
        except FloatingPointError as error:assert 'raw G' in str(error)
        else:raise AssertionError('Native raw-G callback admitted sanitized nonfinite predictions')
        assert TabNNModule.create_callbacks(SimpleNamespace())==[]
    finally:
        TabNNModule.create_callbacks,NNAlgInterface.fit=original_callbacks,original_fit
    assert callback_invocations==['raw_validation_epoch_callback']
    runtime=module._runtime()
    (output/'import.json').write_text(json.dumps(dict(dependencies=deps,runtime=runtime,
        native_config=native,cardinalities=prepared['cardinalities'],models_constructed=False,
        fits_executed=False,raw_G_callback_rejected_before_metric=True),indent=2)+'\n')


if __name__=='__main__':
    assert sys.argv[1]=='--worker';worker(Path(sys.argv[2]))
