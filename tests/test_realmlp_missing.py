"""Pinned native missing-category/F-vocabulary boundary without a neural fit."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest


@pytest.mark.parametrize('mode',['F_missing','query_only_missing','F_all_missing'])
def test_native_missing_ordinal_zero_and_carrier(mode,tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),
        PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,str(Path(__file__).resolve()),'--worker',mode,str(tmp_path/'worker')]
    start=time.perf_counter_ns();result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),returncode=result.returncode,wall_ns=time.perf_counter_ns()-start),indent=2)+'\n')
    (tmp_path/'worker.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout


def worker(mode,output):
    import numpy as np
    import pandas as pd
    from treeneuron.research import realmlp_native as api
    from treeneuron.research.realmlp_snapshot import converter_state,restore_converter
    from test_realmlp_contract import config
    output.mkdir(parents=True,exist_ok=False)
    categories={'F_missing':['a',None,'b',np.nan,pd.NA,'a'],
        'query_only_missing':['a','b','a','b','a','b'],
        'F_all_missing':[None,np.nan,pd.NA,None,np.nan,pd.NA]}[mode]
    F=pd.DataFrame(dict(x=np.arange(6.),kind=categories));G=pd.DataFrame(dict(x=[100.,101.,102.,103.],kind=[None,np.nan,pd.NA,'new']))
    cfg=api.RealMLPConfig.parse(config())
    prepared=api.prepare_native_data(F,np.arange(6.),G,np.arange(4.),cfg,17)
    assert prepared['G_cat_codes']==[[0]]*4
    expected={'F_missing':[[1],[0],[2],[0],[0],[1]],'query_only_missing':[[1],[2],[1],[2],[1],[2]],'F_all_missing':[[0]]*6}[mode]
    assert prepared['F_cat_codes']==expected
    state=converter_state(prepared['converter'],cfg.feature_schema)
    assert state['F_missing']==[mode!='query_only_missing']
    assert state['categories']==[[] if mode=='F_all_missing' else ['str:a','str:b']]
    assert state['cardinalities']==[1 if mode=='F_all_missing' else 3]
    converter=restore_converter(state)
    query=api.native_frame(G,cfg.feature_schema)
    assert converter.transform(query).tensors['x_cat'].tolist()==[[0]]*4
    (output/'missing.json').write_text(json.dumps(dict(mode=mode,converter=state,F_codes=expected,
        G_codes=prepared['G_cat_codes'],native_models_constructed=False,fits_executed=False),indent=2)+'\n')


if __name__=='__main__':
    assert sys.argv[1]=='--worker';worker(sys.argv[2],Path(sys.argv[3]))
