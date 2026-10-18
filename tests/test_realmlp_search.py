"""P06 suggestions preserve the pinned native distribution, including probabilities."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def api():
    from treeneuron.research import realmlp_search
    return realmlp_search


@pytest.mark.parametrize('field,value',[('attempt_index',-1),('attempt_index',32),('attempt_index',True),
    ('hpo_seed',-1),('hpo_seed',2**63),('hpo_seed',False)])
def test_invalid_attempt_or_hpo_seed_fails_before_native_import(field,value):
    kwargs=dict(hpo_seed=901,attempt_index=1);kwargs[field]=value
    with pytest.raises(ValueError):api().suggest_realmlp(dict(columns=['x'],categorical=[]),**kwargs)
    assert 'pytabkit' not in sys.modules


def test_exact_native_anchor_and_31_native_probability_draws_in_fresh_worker(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(root/'src'),str(root/'environment/research-v2-realmlp-site')]),
        PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR=str(tmp_path/'mpl'),XDG_CACHE_HOME=str(tmp_path/'cache'))
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[name]='1'
    env.pop('KMP_DUPLICATE_LIB_OK',None)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',str(tmp_path/'native')]
    import time
    start=time.perf_counter_ns();result=None;error=None
    try:result=subprocess.run(argv,cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
    except BaseException as caught:error=dict(type=type(caught).__name__,message=str(caught));raise
    finally:
        (tmp_path/'invocation.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),elapsed_ns=time.perf_counter_ns()-start,
            error=error,returncode=None if result is None else result.returncode),indent=2)+'\n')
    (tmp_path/'command.json').write_text(json.dumps(dict(argv=argv,cwd=str(root),environment_overrides={k:env.get(k)
        for k in ('PYTHONPATH','PYTHONDONTWRITEBYTECODE','MPLCONFIGDIR','XDG_CACHE_HOME','OMP_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','KMP_DUPLICATE_LIB_OK')},
        returncode=result.returncode),indent=2)+'\n')
    (tmp_path/'worker.log').write_text(result.stdout)
    assert result.returncode==0,result.stdout
    proof=json.loads((tmp_path/'native/result.json').read_text())
    assert proof['officially_matched_draws']==31 and proof['native_fits']==0
    assert proof['protected_access_authorized'] is False


def worker(output):
    import numpy as np
    from treeneuron.config import canonical_json
    from treeneuron.research.realmlp_native import RealMLPConfig
    from treeneuron.research.seeds import seed64
    output.mkdir(parents=True,exist_ok=False)
    schema=dict(columns=['x','category'],categorical=['category'])
    schedule=api().realmlp_schedule(schema,hpo_seed=901)
    assert len(schedule['suggestions'])==32
    anchor=schedule['suggestions'][0]
    assert anchor['config']['mode']=='native_anchor' and anchor['config']['parameters']=={}
    assert anchor['native_sampler_seed'] is None and anchor['eligible_predictive_result'] is False
    from pytabkit.models.alg_interfaces.nn_interfaces import RealMLPParamSampler
    from pytabkit.models.sklearn.default_params import DefaultParams
    def plain(v):
        if isinstance(v,np.generic):return v.item()
        if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [plain(x) for x in v]
        return v
    assert anchor['official_sampled_recipe']==DefaultParams.RealMLP_TD_REG
    draws=[]
    for index in range(1,32):
        actual=schedule['suggestions'][index]
        derived=seed64(901,'P06','official-default-sampler',str(index))
        reference=plain(RealMLPParamSampler(is_classification=False,hpo_space_name='default').sample_params(derived))
        assert actual['native_sampler_seed']==derived
        assert actual['official_sampled_recipe']==reference
        assert set(actual['config']['parameters'])=={'num_emb_type','add_front_scale','lr','p_drop','wd','plr_sigma','act','hidden_sizes'}
        for key,value in actual['config']['parameters'].items():assert value==reference[key]
        assert actual['config']['mode']=='search' and RealMLPConfig.parse(actual['config']).max_epochs==256
        assert actual['native_distribution']==dict(is_classification=False,hpo_space_name='default')
        assert actual['native_fits']==0 and actual['labels_read']==[]
        canonical_json(actual)
        (output/f'draw-{index:02}.json').write_text(canonical_json(actual)+'\n');draws.append(actual['config'])
    replay=api().suggest_realmlp(schema,hpo_seed=901,attempt_index=11)
    assert replay['config']==draws[10]
    changed=api().suggest_realmlp(schema,hpo_seed=902,attempt_index=11)
    assert changed['config']!=draws[10]
    (output/'anchor.json').write_text(canonical_json(anchor)+'\n')
    (output/'result.json').write_text(json.dumps(dict(officially_matched_draws=31,native_fits=0,
        protected_access_authorized=False,full_quality_anchor=False,source=schedule['source'],schedule=schedule,
        numpy_version=np.__version__,sampler_space='regression/default'),indent=2)+'\n')


if __name__=='__main__':
    assert sys.argv[1]=='--worker';worker(Path(sys.argv[2]))
