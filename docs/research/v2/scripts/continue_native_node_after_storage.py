"""Serial operational continuation of the held c33 engineering inventory.

The outer coordinator never hard-kills its nested worker supervisors. Any
failed admission, fit, audit or interrupted coordinator needs explicit recovery.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--plan',type=Path,required=True)
parser.add_argument('--expected-plan-sha256',required=True)
parser.add_argument('--validate-only',action='store_true')
args=parser.parse_args();started=time.perf_counter_ns()
assert not args.plan.is_symlink() and sha(args.plan)==args.expected_plan_sha256
plan=json.loads(args.plan.read_text());root=Path(plan['output']);source=Path(plan['source_root'])
assert plan['schema']=='research/v2-NODE-storage-serial-continuation'
assert plan['T_accessed'] is plan['V_labels_read'] is plan['quality_eligible'] is False
assert plan['maximum_simultaneous_native_fits']==1 and plan['wall_deadline'] is None
assert plan['fit_seed']==17 and [case['dataset'] for case in plan['cases']]==['ccpp','bike']
assert str(Path(sys.executable))==plan['interpreter'] and sha(sys.executable)==plan['interpreter_sha256']


def unchanged():
    assert sha(args.plan)==args.expected_plan_sha256
    for path,digest in plan['operators'].items():
        assert not Path(path).is_symlink() and sha(path)==digest
    for item in plan['references']:
        assert not Path(item['path']).is_symlink() and sha(item['path'])==item['sha256']
    for case in plan['cases']:
        assert sha(Path(case['prepared'])/'manifest.json')==case['manifest_sha256']
    assert sha(Path(plan['waiting_for_study'])/'study.json')==plan['waiting_study_sha256']


unchanged()
assert str(Path(__file__).resolve()) in plan['operators']
sys.path.insert(0,str(source/'src'))
from treeneuron.research.method_worker import scientific_source_sha256
assert scientific_source_sha256()==plan['source_sha256']
if args.validate_only:
    assert not root.exists()
    print('validated_immutable_operational_plan_no_fit_or_query_labels')
    raise SystemExit(0)

from treeneuron.research.process_supervision_v2 import supervise_process
from treeneuron.research.opportunity_execution import live_resources

root.mkdir(parents=True,exist_ok=False)
lease=(root/'worker-lease.json').open('a+')
fcntl.flock(lease.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)


def write(name,value):
    with (root/name).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')


environment={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','PYTHONDONTWRITEBYTECODE')}
environment.update(PYTHONPATH=str(source/'src'),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None)
child_env=dict(os.environ)
for key,value in environment.items():
    if value is None:child_env.pop(key,None)
    else:child_env[key]=value
write('invocation.json',dict(argv=sys.argv,cwd=str(Path.cwd()),plan_sha256=args.expected_plan_sha256,
    operator_sha256=sha(__file__),started_utc_ns=time.time_ns(),environment_overrides=environment,
    no_outer_group_hard_kill=True,T_accessed=False,quality_eligible=False))


def audit(dataset,study,attempt_index):
    unchanged();live=live_resources(root)
    admitted=(live['available_memory_bytes'] is not None
        and live['available_memory_bytes']>=plan['memory_proxy_admission_bytes']
        and live['free_disk_bytes']>=plan['free_disk_admission_bytes'])
    write(dataset+'-audit-admission.json',dict(live,admitted=admitted))
    if not admitted:raise RuntimeError('Audit paused before model loading for RAM/disk admission')
    argv=[plan['interpreter'],'-P','-B',plan['audit_operator'],'--source',str(source),
        '--study',str(study),'--output',str(root/(dataset+'-completion-audit.json')),
        '--attempt-index',str(attempt_index)]
    if attempt_index:
        argv+=['--evidence-root',plan['evidence_root']]
        for flag in ('interruption-receipt','checkpoint-validation-receipt','resume-invocation'):
            ref=plan['resume_references'][flag]
            argv+=['--'+flag,ref['path'],'--expected-'+flag+'-sha256',ref['sha256']]
    process=supervise_process(argv=argv,cwd=source,env_overrides=environment,
        output=root/(dataset+'-audit-process'),timeout_ns=None,rss_limit_bytes=plan['worker_owned_RSS_cap_bytes'],
        identity=dict(purpose='native_completion_audit',dataset=dataset,source_sha256=plan['source_sha256'],
            operator_sha256=sha(plan['audit_operator']),plan_sha256=args.expected_plan_sha256))
    write(dataset+'-audit-invocation.json',dict(argv=argv,cwd=str(source),execution_status=process['execution_status'],
        returncode=process['returncode'],T_accessed=False))
    if process['execution_status']!='exited' or process['returncode']!=0:
        raise RuntimeError('Native completion audit did not pass: '+dataset)
    print(json.dumps(dict(dataset=dataset,status='native_completion_audited')),flush=True)


try:
    waiting=Path(plan['waiting_for_study'])
    while True:
        try:
            with (waiting/'worker-lease.json').open('rb') as stream:
                fcntl.flock(stream.fileno(),fcntl.LOCK_SH|fcntl.LOCK_NB)
                terminal=waiting/'attempt-001/terminal.json'
                if not terminal.is_file():raise RuntimeError('Native predecessor stopped without a terminal; retain and diagnose')
                if json.loads(terminal.read_text())['status']!='completed_ineligible':
                    raise RuntimeError('Native predecessor stopped without an eligible engineering completion')
            break
        except BlockingIOError:time.sleep(30)
    audit('parkinsons',waiting,1)
    for case in plan['cases']:
        unchanged();dataset=case['dataset'];output=Path(case['output'])
        assert not output.exists()
        argv=[plan['interpreter'],'-B',plan['launcher'],'--dataset',dataset,'--prepared',case['prepared'],
            '--expected-manifest-sha256',case['manifest_sha256'],'--expected-source-sha256',plan['source_sha256'],
            '--output',case['output']]
        print(json.dumps(dict(dataset=dataset,status='starting_full_native_anchor_no_wall_cap')),flush=True)
        with (root/(dataset+'-launch.log')).open('xb') as log:
            result=subprocess.run(argv,cwd=source,env=child_env,stdout=log,stderr=subprocess.STDOUT)
        write(dataset+'-launch-invocation.json',dict(argv=argv,cwd=str(source),returncode=result.returncode,T_accessed=False))
        if result.returncode:raise RuntimeError('Native fit paused or failed; explicit recovery required: '+dataset)
        audit(dataset,output,0)
    write('terminal.json',dict(status='remaining_three_native_anchors_completed_and_audited',
        elapsed_ns=time.perf_counter_ns()-started,timers_overlap_existing_resume=True,missing_prior_terminal_cost_tail_retained=True,
        new_native_fits=2,existing_resumed_fit_reused=True,V_labels_read=False,T_accessed=False,quality_eligible=False))
except BaseException as error:
    write('terminal.json',dict(status='paused_or_failed_requires_explicit_recovery',error_type=type(error).__name__,
        reason=str(error),elapsed_ns=time.perf_counter_ns()-started,all_prior_attempts_retained=True,T_accessed=False,quality_eligible=False))
    raise
