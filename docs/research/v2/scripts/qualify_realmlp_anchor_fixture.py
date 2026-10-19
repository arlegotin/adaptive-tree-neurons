"""One predeclared full-recipe P06 fixture; no V labels or quality promotion."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter_ns()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1':raise ValueError('Set qualified numerical thread variables before running')
    if not sys.dont_write_bytecode:raise ValueError('Run with -B')
    import numpy as np
    from treeneuron.config import canonical_hash
    from treeneuron.research.ledger import write_new_json
    from treeneuron.research.method_worker import scientific_source_sha256
    from treeneuron.research.search_controller_v2 import make_search_plan,run_search
    from treeneuron.research.seeds import seed64
    from treeneuron.research.trial_bridge_v2 import EngineeringTrialBridge
    project=Path(__file__).resolve().parents[4]
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    source=scientific_source_sha256();input_root=root/'input';input_root.mkdir();files={};seeds={}
    for role,n,offset in (('F',300,0),('G',37,1000),('V_features',1031,2000)):
        seed=seed64(2026101801,'WP07','P06-native256-fixture',role);seeds[role]=seed
        rng=np.random.default_rng(seed);x=rng.uniform(-2,2,n);kind=rng.integers(0,2,n)
        if role=='V_features':kind[np.arange(n)%17==0]=2
        rows=np.arange(offset,offset+n,dtype=np.int64)
        arrays=dict(X=np.column_stack((x,kind)),row_id=rows,group_id=rows)
        if role!='V_features':arrays['y']=4+x+.3*x*x+.2*(kind==1)
        path=input_root/(role+'.npz');np.savez(path,**arrays)
        files[role]=dict(path=path.name,sha256=sha(path))
    config=dict(schema='research/v2-realmlp-config',family='P06',mode='native_anchor',
        feature_schema=dict(columns=['x','kind'],categorical=['kind']),parameters={})
    split=canonical_hash(dict(files=files,seeds=seeds,rows=dict(F=300,G=37,V_features=1031)))
    template=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id='engineering/P06/native256-fixture',family_id='P06',fit_seed=17,
        scientific_source_sha256=source,config=config,budget=dict(max_seconds=180.),
        observation_identity=dict(dataset='synthetic_native_anchor_fixture',split_kind='grouped',split_sha256=split),files=files)
    path=input_root/'request.json';write_new_json(path,template)
    runtime=dict(scope='Prelaunch metadata; actual native images/imports audited inside the method worker',
        python=sys.version,platform=platform.platform(),executable=sys.executable,executable_sha256=sha(sys.executable),
        core_manifest_sha256=sha(project/'locks/research-v2/core.json'),
        private_installed_manifest_sha256=sha(project/'locks/research-v2/realmlp/installed-files.json'))
    identity=dict(job_id=template['job_id'],family_id='P06',search_repeat=0,source_sha256=source,
        runtime_sha256=canonical_hash(runtime),support_sha256=canonical_hash(dict(anchor=config)),split_sha256=split)
    plan=make_search_plan(identity=identity,fit_seed=17,hpo_seed32=73,anchors=[config],mode='finite',
        support=None,candidates=[],max_attempts=1,deadline_ns=180000000000,prefixes_ns=[])
    write_new_json(root/'prelaunch.json',dict(schema='research/v2-P06-full-recipe-fixture',source_sha256=source,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
        argv=sys.argv,cwd=str(Path.cwd()),script_sha256=sha(__file__),runtime_metadata=runtime,
        data_seeds=seeds,fit_seed=17,plan=plan.to_dict(),request_sha256=sha(path),
        rss_limit_bytes=1024**3,qualification_only=True,quality_eligible=False,T_accessed=False))
    shutil.copyfile(__file__,root/'launcher.py')
    vm=subprocess.run(['/usr/bin/vm_stat'],capture_output=True,text=True,check=True)
    page=re.search(r'page size of (\d+) bytes',vm.stdout)
    counts={name:int(value) for name,value in re.findall(r'^([^:\n]+):\s+(\d+)\.',vm.stdout,re.M)}
    names=('Pages free','Pages inactive','Pages speculative')
    available=int(page.group(1))*sum(counts[k] for k in names) if page and all(k in counts for k in names) else None
    disk=shutil.disk_usage(root).free;memory_floor=7*1024**3//4
    admitted=available is not None and available>=memory_floor and disk>=1024**3
    write_new_json(root/'admission.json',dict(admitted=admitted,vm_stat_stdout=vm.stdout,
        available_memory_proxy_bytes=available,proxy_is_not_an_allocation_guarantee=True,
        required_memory_proxy_bytes=memory_floor,free_disk_bytes=disk,required_disk_bytes=1024**3,
        rss_limit_bytes=1024**3,observed_utc_ns=time.time_ns()))
    if not admitted:
        write_new_json(root/'terminal.json',dict(status='paused_resources',attempts_started=0,quality_eligible=False,T_accessed=False))
        return 3
    prep_started=time.perf_counter_ns()
    bridge=EngineeringTrialBridge(plan=plan,template_path=path,expected_template_sha256=sha(path),
        search_directory=root/'search',interpreter=sys.executable,
        extra_pythonpath=(project/'environment/research-v2-realmlp-site',),rss_limit_bytes=1024**3)
    result=run_search(root/'search',plan=plan,evaluate_trial=bridge,
        invocation_preparation_ns=time.perf_counter_ns()-prep_started)
    write_new_json(root/'search-result.json',result)
    if scientific_source_sha256()!=source:raise ValueError('Source changed during native fixture')
    record=result['records'][0] if result['records'] else None
    success=record is not None and record['status']=='completed_ineligible'
    summary={}
    if success:
        worker=json.loads((root/'search/attempt-000/worker/terminal.json').read_text());resources=worker['adapter_resources']
        assert resources['native_epochs_executed']==256 and resources['applied_updates']==256
        assert resources['checkpoint_eligible'] is False and resources['quality_eligible'] is False
        assert worker['snapshot_reload_exact'] is True and result['winner'] is None
        summary=dict(native_epochs=256,applied_updates=256,selected_epoch=resources['selected_epoch'],
            cumulative_F_rows=resources['cumulative_visited_rows'],unvisited_F_row_ids=resources['unvisited_F_row_ids'],
            initialization=resources['initialization_observation'],native_source=resources['source'],
            native_dependencies=resources['dependencies'],native_runtime=resources['runtime'],
            worker_process_lifetime_peak_rss_bytes=resources['process_peak_rss_bytes'],
            snapshot_reload_exact=True,full_native_recipe_on_this_fixture=True)
    terminal=dict(schema='research/v2-P06-full-recipe-fixture-result',
        status='completed_native_recipe_fixture' if success else 'retained_failed_or_capped_fixture',
        source_sha256=source,attempts_paid=result['attempts_paid'],record=record,summary=summary,
        charged_search_ns=result['total_charged_ns'],launcher_body_ns=time.perf_counter_ns()-started,
        quality_eligible=False,Gate_D=False,prepared_transfer_fits=0,V_labels_created_or_read=False,T_accessed=False,
        scope='One synthetic native-recipe/resource fixture, not transfer workload qualification or predictive evidence')
    write_new_json(root/'terminal.json',terminal)
    print(json.dumps({k:v for k,v in terminal.items() if k not in ('summary','record')},sort_keys=True))
    return 0 if success else 4


if __name__=='__main__':raise SystemExit(main())
