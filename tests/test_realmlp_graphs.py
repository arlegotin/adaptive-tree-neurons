"""Predeclared bounded native embedding/activation/layout graph qualification."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import pytest


@pytest.mark.parametrize('variant',range(12))
def test_predeclared_native_graph_direct_and_fresh_snapshot(variant,tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),
        PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,str(root/'tests/test_realmlp_native.py'),'--graph',str(variant),str(tmp_path/'worker')]
    start=time.perf_counter_ns();result=None;error=None
    try:result=subprocess.run(argv,cwd=root,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
    except BaseException as caught:error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),
            environment={key:env.get(key) for key in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','XDG_CACHE_HOME',
                'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')},
            returncode=result.returncode if result else None,error=error,wall_ns=time.perf_counter_ns()-start),indent=2)+'\n')
        (tmp_path/'worker.log').write_text(result.stdout if result else str(error))
    assert result.returncode==0,result.stdout
