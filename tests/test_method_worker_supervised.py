"""Native own-process qualification; requires a working real RSS probe."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from test_method_worker import request_fixture,sha


def test_real_method_worker_owned_process_guards_and_identical_rerun(tmp_path):
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.process_supervision_v2 import supervise_process
    path,value=request_fixture(tmp_path/'input');source=scientific_source_sha256()
    project=Path(__file__).resolve().parents[1]
    env=dict(PYTHONPATH=str(project/'src'),PYTHONDONTWRITEBYTECODE='1')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
    observations=[]
    def launch(label,timeout_ns=30000000000,rss_limit_bytes=512*1024**2):
        root=tmp_path/label;root.mkdir();started=time.perf_counter_ns()
        process=supervise_process(argv=[sys.executable,'-B','-m','treeneuron.research.method_worker',
            '--request',str(path),'--expected-request-sha256',sha(path),'--output',str(root/'worker')],
            cwd=project,env_overrides=env,output=root/'supervisor',timeout_ns=timeout_ns,
            rss_limit_bytes=rss_limit_bytes,poll_interval_ns=10000000,terminate_grace_ns=100000000,
            identity=dict(purpose='engineering_qualification',scientific_source_sha256=source,
                request_sha256=sha(path),fit_seed=value['fit_seed']))
        outer=dict(elapsed_ns=time.perf_counter_ns()-started,
            scope='Around complete supervise_process including its final receipt publication',
            process_receipt_sha256=sha(root/'supervisor/process.json'))
        (root/'outer-clock.json').write_text(json.dumps(outer,indent=2)+'\n')
        observations.append(dict(label=label,process=process,outer_clock=outer))
        assert process['eligible_model'] is False and process['protected_access_authorized'] is False
        assert process['child_reaped'] and outer['elapsed_ns']>=process['elapsed_ns']
        return root,process
    first,normal=launch('normal')
    assert normal['execution_status']=='exited' and normal['returncode']==0
    _,deadline=launch('deadline',timeout_ns=200000000)
    assert deadline['execution_status']=='timeout' and deadline['child_started']
    assert deadline['termination']['term_sent'] and deadline['returncode']!=0
    _,memory=launch('rss_limit',rss_limit_bytes=1)
    assert memory['execution_status']=='resource_exhausted' and memory['observed_peak_root_rss_bytes']>1
    assert memory['termination']['term_sent'] and memory['returncode']!=0
    rerun,recovered=launch('same_seed_rerun')
    assert recovered['execution_status']=='exited' and recovered['returncode']==0
    for root in (first,rerun):
        terminal=json.loads((root/'worker/terminal.json').read_text())
        assert terminal['status']=='completed_ineligible' and terminal['quality_eligible'] is False
        assert terminal['fit_seed']==value['fit_seed'] and terminal['snapshot_reload_exact']
        assert terminal['loaded_heavy_modules']==[]
    with np.load(first/'worker/predictions.npz',allow_pickle=False) as a,np.load(rerun/'worker/predictions.npz',allow_pickle=False) as b:
        assert set(a.files)==set(b.files)
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])
    assert scientific_source_sha256()==source
    proof=dict(purpose='Actual tiny engineering method worker under owned-process guards',
        source_sha256=source,request_sha256=sha(path),observations=observations,
        same_seed_rerun_predictions_exact=True,paid_process_attempts=4,
        summed_disjoint_outer_attempt_ns=sum(r['outer_clock']['elapsed_ns'] for r in observations),
        native_training_checkpoint_resume=False,strict_peak_memory_qualified=False,
        quality_eligible=False,Gate_D=False,prepared_transfer_experiments=0,T_accessed=False)
    (tmp_path/'proof.json').write_text(json.dumps(proof,indent=2,sort_keys=True)+'\n')
