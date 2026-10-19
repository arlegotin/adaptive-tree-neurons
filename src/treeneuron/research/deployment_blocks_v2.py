"""Finite paired owned-process deployment measurements; no cost-claim authority."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .deployment_worker_v2 import _request,_source,_THREADS
from .ledger import TERMINAL
from .opportunity_execution import live_resources
from .process_supervision_v2 import supervise_process
from .seeds import seed64
from .workqueue import authenticated_queue,make_queue_plan,run_queue


_ROOT=Path(__file__).resolve().parents[3]


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_deployment_plan(requests,*,seed,interpreter,timeout_ns,rss_limit_bytes):
    """Bind exactly30 blocks, paired raw inputs and one explicit launch policy."""
    if (type(requests) is not dict or set(requests)!={'selected','baseline'} or type(seed) is not int
            or not 0<=seed<2**63 or type(timeout_ns) is not int or not 0<timeout_ns<=1200_000_000_000
            or type(rss_limit_bytes) is not int or not 1024**2<=rss_limit_bytes<=16*1024**3):
        raise ValueError('Finite paired deployment identity/budgets required')
    executable=Path(interpreter)
    if not executable.is_absolute() or not executable.is_file():raise ValueError('Absolute qualified interpreter required')
    references={};values={}
    for role,ref in requests.items():
        if type(ref) is not dict or set(ref)!={'path','sha256'}:raise ValueError('External deployment request commitments required')
        path=Path(ref['path']).absolute();value,*_=_request(path,ref['sha256'])
        references[role]=dict(path=str(path),sha256=ref['sha256']);values[role]=value
    for key in ('scientific_source_sha256','feature_schema','batch_sizes','warmup_calls','samples_per_batch'):
        if values['selected'][key]!=values['baseline'][key]:raise ValueError('Paired deployment treatment differs')
    if values['selected']['query']['sha256']!=values['baseline']['query']['sha256']:
        raise ValueError('Paired deployment query bytes differ')
    identity=dict(schema='research/v2-paired-deployment-identity',source_sha256=_source(),seed=seed,block_count=30,
        requests=references,interpreter=str(executable),interpreter_sha256=_sha(executable),
        timeout_ns=timeout_ns,rss_limit_bytes=rss_limit_bytes,free_disk_floor_bytes=512*1024**2,
        available_memory_proxy_floor_bytes=(5*rss_limit_bytes+3)//4+512*1024**2,
        cost_claim_authorized=False,protected_access_authorized=False,
        pairing='NumPy default_rng(seed64(seed,paired-deployment-timing,order)).integers(0,2,size=30)',
        resident_statistic='rounded_arithmetic_mean_of_all_declared_calls_per_process',
        recovery='No automatic measurement rerun; retain interrupted members and all costs')
    order=np.random.default_rng(seed64(seed,'paired-deployment-timing','order')).integers(0,2,size=30)
    jobs=[];previous=None
    for block,first in enumerate(order):
        roles=('selected','baseline') if first==0 else ('baseline','selected')
        for role in roles:
            name=f'block-{block:02}/{role}'
            jobs.append(dict(job_id=name,kind='deployment_measurement',depends_on=[] if previous is None else [previous],
                dependency_policy='all_terminal',payload=dict(block_id=block,role=role,
                    order='selected_first' if first==0 else 'baseline_first')))
            previous=name
    return make_queue_plan(identity,jobs)


def _checked(plan):
    if type(plan) is not dict or type(plan.get('identity')) is not dict:raise ValueError('Bound deployment plan required')
    identity=plan['identity']
    try:
        regenerated=make_deployment_plan(identity['requests'],seed=identity['seed'],interpreter=identity['interpreter'],
            timeout_ns=identity['timeout_ns'],rss_limit_bytes=identity['rss_limit_bytes'])
    except (KeyError,TypeError) as error:raise ValueError('Invalid deployment plan identity') from error
    if regenerated!=plan:raise ValueError('Deployment plan/source/inputs/launch identity changed')
    return identity


def run_deployment_blocks(plan,output,*,resume=False):
    """Run/continue an unchanged engineering queue with measured live admission."""
    identity=_checked(plan);root=Path(output)
    def check(value):
        if value!=identity:raise ValueError('Deployment queue identity changed')
        _checked(plan)
    def before_start(job,ledger):
        live=live_resources(root);available=live['available_memory_bytes']
        return dict(admitted=type(available) is int and available>=identity['available_memory_proxy_floor_bytes']
            and live['free_disk_bytes']>=identity['free_disk_floor_bytes'],live=live,
            memory_proxy_is_not_allocation_guarantee=True,required_disk_bytes=identity['free_disk_floor_bytes'],
            required_memory_proxy_bytes=identity['available_memory_proxy_floor_bytes'])
    def handler(job,directory,dependencies,recovery_attempt):
        if recovery_attempt:
            return dict(status='input_or_dependency_failure',reason='Interrupted timing member cannot automatically be remeasured',
                block_id=job['payload']['block_id'],role=job['payload']['role'])
        ref=identity['requests'][job['payload']['role']]
        env={k:'1' for k in _THREADS};env.update(PYTHONPATH=str(_ROOT/'src'),PYTHONDONTWRITEBYTECODE='1',
            PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None,
            MPLCONFIGDIR=str(directory/'cache/mpl'),XDG_CACHE_HOME=str(directory/'cache/xdg'))
        begin=time.perf_counter_ns()
        process=supervise_process(argv=[identity['interpreter'],'-B','-m','treeneuron.research.deployment_worker_v2',
            '--request',ref['path'],'--expected-request-sha256',ref['sha256'],'--output',str(directory/'worker')],
            cwd=_ROOT,env_overrides=env,output=directory/'process',timeout_ns=identity['timeout_ns'],
            rss_limit_bytes=identity['rss_limit_bytes'],identity=dict(plan_hash=plan['plan_hash'],job_id=job['job_id'],
                source_sha256=identity['source_sha256'],request_sha256=ref['sha256']))
        enclosing=time.perf_counter_ns()-begin;terminal=directory/'worker/terminal.json'
        status=process['execution_status']
        if status=='interrupted_recoverable':raise KeyboardInterrupt('Deployment supervisor was interrupted; no automatic timing rerun')
        if status=='exited':
            if process['returncode']==0 and terminal.is_file():
                worker=json.loads(terminal.read_text());status=worker['status']
                if status!='completed_ineligible':raise ValueError('Unexpected deployment worker eligibility')
            else:
                worker=json.loads(terminal.read_text()) if terminal.is_file() else {}
                failure=worker.get('status')
                status=failure if type(failure) is str and failure in TERMINAL-{
                    'completed_eligible','completed_ineligible','interrupted_recoverable'} else 'input_or_dependency_failure'
        return dict(status=status,block_id=job['payload']['block_id'],role=job['payload']['role'],
            process_path='process/process.json',worker_path='worker/terminal.json' if terminal.is_file() else None,
            process_enclosing_ns=enclosing,request_sha256=ref['sha256'],cost_claim_authorized=False)
    return run_queue(plan,root,handler,check_identity=check,resume=resume,before_start=before_start)


def collect_deployment_blocks(directory,*,expected_plan_hash):
    """Authenticate inactive results; any missing member blocks paired inference."""
    root=Path(directory);records={};missing=[];process_ns=0;all_reaped=True
    with authenticated_queue(root,expected_plan_hash=expected_plan_hash) as (plan,ledger,results):
        identity=_checked(plan)
        latest={event['attempt_id']:event for event in ledger.events}
        for job in plan['attempts']:
            name=job['job_id'];payload=job['payload'];entry=results.get(name)
            base=Path(latest[name]['result']['path']).parent if entry is not None else None
            value=entry['value'] if entry is not None else {}
            if base is not None and value.get('process_path') is not None:
                process=json.loads((base/value['process_path']).read_text())
                if process['identity']!=dict(plan_hash=plan['plan_hash'],job_id=name,source_sha256=identity['source_sha256'],
                        request_sha256=identity['requests'][payload['role']]['sha256']):
                    raise ValueError('Owned deployment process identity differs')
                all_reaped=all_reaped and (not process['child_started'] or process['child_reaped'])
                amount=value['process_enclosing_ns']
                if type(amount) is not int or amount<process['elapsed_ns']:raise ValueError('Invalid enclosing process clock')
                process_ns+=amount
            elif entry is not None or name in latest:all_reaped=None
            if entry is None or entry['value']['status']!='completed_ineligible':
                missing.append(dict(job_id=name,status=latest.get(name,{}).get('status','absent')
                    if entry is None else entry['value']['status']));continue
            if value.get('process_path') is None:raise ValueError('Completed deployment lacks its process receipt')
            process=json.loads((base/value['process_path']).read_text());worker=json.loads((base/value['worker_path']).read_text())
            if (not process['child_reaped'] or process['returncode']!=0 or process['execution_status']!='exited'
                    or worker['status']!='completed_ineligible' or worker['scientific_source_sha256']!=identity['source_sha256']
                    or worker['request_sha256']!=identity['requests'][payload['role']]['sha256']
                    or worker['cost_claim_authorized'] is not False or worker['T_accessed'] is not False):
                raise ValueError('Deployment process/worker identities or terminal status differ')
            request,*_=_request(Path(identity['requests'][payload['role']]['path']),identity['requests'][payload['role']]['sha256'])
            if [b['batch_size'] for b in worker['batches']]!=[1,1024]:raise ValueError('Deployment batches differ')
            durations={}
            for batch in worker['batches']:
                resident=batch['resident_prediction_ns'];warmups=batch['warmup_prediction_ns']
                if (len(resident)!=request['samples_per_batch'] or len(warmups)!=request['warmup_calls']
                        or not batch['predictions_equal_exact'] or any(type(n) is not int or n<=0 for n in resident+warmups+[batch['first_prediction_ns']])):
                    raise ValueError('Incomplete/nonfinite deployment samples')
                durations[str(batch['batch_size'])]=int(round(sum(resident)/len(resident)))
            records[payload['block_id'],payload['role']]=dict(resident=durations,
                process=amount,load=worker['model_load_ns'],order=payload['order'])
        charged=ledger.charged_ns;last=ledger.events[-1]['event_hash'] if ledger.events else None
    def blocks(kind,size=None):
        values=[]
        for i in range(30):
            a=records[i,'selected'];b=records[i,'baseline']
            values.append(dict(block_id=i,order=a['order'],selected_ns=a[kind][size] if size else a[kind],
                baseline_ns=b[kind][size] if size else b[kind]))
        return values
    if process_ns>charged:raise ValueError('Nested process clocks exceed their enclosing queue jobs')
    return dict(schema='research/v2-paired-deployment-measurements',status='incomplete_measurements' if missing else 'complete_engineering_measurements',
        plan_hash=expected_plan_hash,ledger_hash=last,completed_processes=len(records),missing_members=missing,
        resident_blocks=None if missing else {size:blocks('resident',size) for size in ('1','1024')},
        cold_process_blocks=None if missing else blocks('process'),model_load_blocks=None if missing else blocks('load'),
        all_children_reaped=all_reaped,charged_queue_job_ns=charged,sum_disjoint_process_enclosing_ns=process_ns,
        cost_claim_authorized=False,dedicated_timing_qualified=False,quality_bound_authenticated=False,
        protected_access_authorized=False,T_accessed=False,
        scope='Contended engineering process blocks; no complete-case inference, native workload or quality/cost claim. Fit/search prerequisites are separate.')
