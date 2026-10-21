"""Serial original-model timing, with paid failures and no timing resampling.

Private queue/collector results carry no native, test-access or efficiency
capability. Only public entry points load the actual full frozen plan first.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time

from .deployment_worker_v2 import _source,_ROOT,_THREADS,_json
from .protected_journal_v2 import _path,_reference,_hash,_canonical,_write_new
from .native_deployment_worker_v2 import native_measurement_identity,inspect_owned_native_measurement,_checked_measurement_request
from .workqueue import make_queue_plan,run_queue,authenticated_queue

_FALSE=dict(native_models_authenticated=False,original_F_role_authenticated=False,dedicated_timing_qualified=False,
    quality_precision_equivalence_authenticated=False,cost_claim_authorized=False,protected_access_authorized=False,T_accessed=False)
_CALL_SCOPE='Full measurement executor call through return/exception; own receipt publication/readback and queue overhead excluded'


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _untrusted_plan(reference):
    _,raw=_reference(reference);value=_json(raw)
    if value.get('schema')=='research/v2-sealed-native-deployment-plan':
        if set(value)!={'schema','payload','payload_sha256'} or _hash(value['payload'])!=value['payload_sha256']:raise ValueError('Timing plan envelope changed')
        value=value['payload']
    if (type(value) is not dict or value.get('source_sha256')!=_source() or type(value.get('schedule')) is not dict
            or value['schedule'].get('source_sha256')!=_source() or value['schedule'].get('worker_concurrency')!=1):
        raise ValueError('Exact current-source serial timing metadata required')
    return value


def _make_execution_queue(reference,payload,*,rss_limit_bytes):
    if type(rss_limit_bytes) is not int or not 64*1024**2<=rss_limit_bytes<=16*1024**3:raise ValueError('Measured root RSS policy must be between64MiB and16GiB')
    if _untrusted_plan(reference)!=payload:raise ValueError('Exact byte-bound timing metadata required')
    schedule=payload['schedule'];models={m['job_id']:m for m in schedule['models']};jobs=[];previous=None
    if len(models)!=len(schedule['models']) or not models or not schedule['jobs']:raise ValueError('Unique nonempty original model and timing jobs required')
    for item in schedule['jobs']:
        model=models.get(item['model_job_id'])
        if (model is None or item['request_reference']!=model['request_reference'] or item['role'] not in ('selected','baseline')
                or type(item['block_id']) is not int or not 0<=item['block_id']<30
                or item['depends_on']!=([] if previous is None else [previous]) or item['dependency_policy']!='all_terminal'):
            raise ValueError('Original request or fixed serial member order changed')
        jobs.append(dict(job_id=item['job_id'],kind='original_snapshot_measurement',depends_on=item['depends_on'],
            dependency_policy='all_terminal',payload=copy.deepcopy(item)));previous=item['job_id']
    identity=dict(schema='research/v2-native-timing-execution-identity',source_sha256=_source(),measurement_plan_reference=reference,
        measurement_plan_payload_sha256=_hash(payload),interpreter=str(Path(sys.executable).absolute()),interpreter_sha256=_ref(Path(sys.executable))['sha256'],
        rss_limit_bytes=rss_limit_bytes,required_memory_proxy_bytes=(5*rss_limit_bytes+3)//4+512*1024**2,
        required_disk_proxy_bytes=512*1024**2,worker_concurrency=1,pause_after_ns=None,
        recovery='No automatic timing resampling; one recovery may adopt an already closed exact measurement only',**_FALSE)
    return make_queue_plan(identity,jobs)


def _checked_queue(queue):
    try:
        identity=queue['identity'];payload=_untrusted_plan(identity['measurement_plan_reference'])
        expected=_make_execution_queue(identity['measurement_plan_reference'],payload,rss_limit_bytes=identity['rss_limit_bytes'])
    except (KeyError,TypeError) as error:raise ValueError('Exact original measurement queue required') from error
    if expected!=queue:raise ValueError('Timing queue/source/executable/resource policy changed')
    return payload


def _live_admission(identity,output):
    from .native_resource_forecast_v2 import observe_native_host_resources
    live=observe_native_host_resources(output);available=live['available_memory_bytes']
    return dict(admitted=type(available) is int and available>=identity['required_memory_proxy_bytes']
        and live['free_disk_bytes']>=identity['required_disk_proxy_bytes'],live_observation=live,
        required_memory_proxy_bytes=identity['required_memory_proxy_bytes'],required_disk_proxy_bytes=identity['required_disk_proxy_bytes'],
        memory_proxy_is_not_allocation_guarantee=True)


def _process_reference(episode):
    control=episode/'process'
    if not (control/'process.json').is_file() or not (control/'started.json').is_file():return None
    return dict(process_root=str(control),process_receipt_sha256=_ref(control/'process.json')['sha256'],process_started_sha256=_ref(control/'started.json')['sha256'])


def _execute_owned_measurement(*,request_path,expected_request_sha256,episode,rss_limit_bytes,admission_check):
    from .cooperative_process_v2 import supervise_cooperative_process
    overrides={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')};overrides.update(PYTHONPATH=str(_ROOT/'src'),KMP_DUPLICATE_LIB_OK=None)
    supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.native_deployment_worker_v2','--request',str(request_path),
        '--expected-request-sha256',expected_request_sha256,'--output',str(episode/'worker')],cwd=_ROOT,env_overrides=overrides,
        output=episode/'process',identity=native_measurement_identity(request_path,expected_request_sha256),
        admission_identity=dict(purpose='original_frozen_snapshot_timing',root_rss_pause_bytes=rss_limit_bytes),
        admission_check=admission_check,rss_pause_bytes=rss_limit_bytes,admit_before_spawn=True)
    return _process_reference(episode)


def _observe_call(queue,job,episode,execute_episode,admission_check):
    from .execution_clock_v2 import observe_clock_domain
    domain=observe_clock_domain();begin=time.perf_counter_ns();error=None;process=None
    try:
        process=execute_episode(request_path=episode/'request.json',expected_request_sha256=_ref(episode/'request.json')['sha256'],
            episode=episode,rss_limit_bytes=queue['identity']['rss_limit_bytes'],admission_check=admission_check)
        return process
    except BaseException as caught:
        error=dict(type=type(caught).__name__,reason=str(caught));raise
    finally:
        end=time.perf_counter_ns()
        if process is None:process=_process_reference(episode)
        _write_new(episode/'call-clock.json',dict(schema='research/v2-native-timing-call',source_sha256=_source(),queue_plan_hash=queue['plan_hash'],
            job_id=job['job_id'],episode=str(episode),request_reference=_ref(episode/'request.json'),origin_ns=begin,end_ns=end,elapsed_ns=end-begin,
            clock_domain=domain,process_reference=process,error=error,clock_scope=_CALL_SCOPE,standalone_authority=False))


def _read_call(reference,queue,job,episode):
    from .execution_clock_v2 import validate_clock_domain
    path,raw=_reference(reference);value=_json(raw)
    fields={'schema','source_sha256','queue_plan_hash','job_id','episode','request_reference','origin_ns','end_ns','elapsed_ns',
        'clock_domain','process_reference','error','clock_scope','standalone_authority'}
    if (path!=episode/'call-clock.json' or type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-native-timing-call'
            or value['source_sha256']!=_source() or value['queue_plan_hash']!=queue['plan_hash'] or value['job_id']!=job['job_id']
            or value['episode']!=str(episode) or value['request_reference']!=_ref(episode/'request.json')
            or value['request_reference']['sha256']!=job['payload']['request_reference']['sha256']
            or any(type(value[k]) is not int or value[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
            or value['end_ns']-value['origin_ns']!=value['elapsed_ns'] or value['clock_scope']!=_CALL_SCOPE or value['standalone_authority'] is not False
            or value['error'] is not None and (type(value['error']) is not dict or set(value['error'])!={'type','reason'}
                or any(type(value['error'][k]) is not str for k in ('type','reason')))):
        raise ValueError('Strict exact-member observed timing call required')
    validate_clock_domain(value['clock_domain'])
    if value['process_reference'] is not None:
        from .cooperative_process_v2 import inspect_cooperative_process
        process=value['process_reference']
        if process.get('process_root')!=str(episode/'process'):raise ValueError('Timing call substituted another process')
        proof=inspect_cooperative_process(root=episode/'process',expected_receipt_sha256=process['process_receipt_sha256'],expected_started_sha256=process['process_started_sha256'])
        start=_json((episode/'process/started.json').read_bytes());argv=start['argv']
        if (argv!=[sys.executable,'-P','-B','-m','treeneuron.research.native_deployment_worker_v2','--request',str(episode/'request.json'),
                '--expected-request-sha256',value['request_reference']['sha256'],'--output',str(episode/'worker')]
                or start['identity']!=native_measurement_identity(episode/'request.json',value['request_reference']['sha256'])
                or start['cwd']!=str(_ROOT) or start['rss_pause_bytes']!=queue['identity']['rss_limit_bytes']
                or start['pause_after_ns'] is not None or start.get('pre_spawn_admission_required') is not True
                or any(start['environment_overrides'].get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))
                or start['environment_overrides'].get('KMP_DUPLICATE_LIB_OK') is not None
                or not value['origin_ns']<=start['supervisor_started_monotonic_ns']<start['supervisor_started_monotonic_ns']+proof['elapsed_ns']<=value['end_ns']):
            raise ValueError('Owned timing clock/command/thread/RSS/episode must match its containing full call')
    return value


def _measurement_value(queue,job,episode):
    call=_read_call(_ref(episode/'call-clock.json'),queue,job,episode);process=call['process_reference']
    if process is None:raise ValueError('No closed original measurement process')
    actual=inspect_owned_native_measurement(process)
    if actual['request_reference']!=call['request_reference']:raise ValueError('Measurement substituted another member request')
    return dict(status='completed_ineligible',measurement_readback=actual,call_reference=_ref(episode/'call-clock.json'),
        adopted_from_closed_original=episode.name=='attempt-0',**_FALSE)


def _run_measurement_queue(queue,output,*,resume=False,max_jobs_this_invocation=None,admission_check=None,execute_episode=None,guard=None):
    _checked_queue(queue);root=_path(output)
    if max_jobs_this_invocation is not None and (type(max_jobs_this_invocation) is not int or max_jobs_this_invocation<=0):raise ValueError('Positive operational prefix required')
    guard=(lambda:None) if guard is None else guard;execute_episode=_execute_owned_measurement if execute_episode is None else execute_episode
    admission_check=(lambda:_live_admission(queue['identity'],root)) if admission_check is None else admission_check
    launched=0
    def check(identity):
        if identity!=queue['identity']:raise ValueError('Timing queue identity changed')
        _checked_queue(queue)
    def before(job,events):
        nonlocal launched
        if max_jobs_this_invocation is not None and launched>=max_jobs_this_invocation:return dict(admitted=False,reason='operational_prefix_only')
        decision=admission_check()
        if decision.get('admitted') is True:launched+=1
        return decision
    def handler(job,directory,dependencies,recovery_attempt):
        if recovery_attempt:
            original=directory.parent/'attempt-0'
            try:return _measurement_value(queue,job,original)
            except (ValueError,OSError,KeyError):
                return dict(status='irrecoverable_missing',reason='Interrupted/ambiguous original timing member cannot be automatically resampled',**_FALSE)
        template=job['payload']['request_reference'];_,raw=_reference(template)
        with (directory/'request.json').open('xb') as stream:stream.write(raw);stream.flush();__import__('os').fsync(stream.fileno())
        _checked_measurement_request(directory/'request.json',template['sha256'])
        _observe_call(queue,job,directory,execute_episode,admission_check)
        return dict(_measurement_value(queue,job,directory),adopted_from_closed_original=False)
    guard()
    from .execution_clock_v2 import observe_clock_domain
    domain=observe_clock_domain();begin=time.perf_counter_ns();error=None;result=None
    prior={str(p) for p in root.glob('invocation-*.json')}
    try:
        result=run_queue(queue,root,handler,check_identity=check,resume=resume,before_start=before)
        guard()
        return result
    except BaseException as caught:
        error=dict(type=type(caught).__name__,reason=str(caught));raise
    finally:
        if root.is_dir():
            invocations=[_ref(p) for p in root.glob('invocation-*.json') if str(p) not in prior]
            end=time.perf_counter_ns();index=len(list(root.glob('context-*.json')))
            _write_new(root/f'context-{index}.json',dict(schema='research/v2-native-timing-invocation-context',source_sha256=queue['identity']['source_sha256'],
                queue_plan_hash=queue['plan_hash'],origin_ns=begin,end_ns=end,elapsed_ns=end-begin,clock_domain=domain,
                queue_invocation_references=invocations,status='interrupted_or_failed' if error is not None else result['status'],error=error,
                clock_scope='Queue call plus final actual-context guard; own context publication/return and outer admission/startup excluded',standalone_authority=False))


def _collect_measurement_queue(queue,output):
    _checked_queue(queue);root=_path(output);records={};calls={};missing=[];total_root=0;unknown=[];references=[]
    with authenticated_queue(root,expected_plan_hash=queue['plan_hash']) as (saved,ledger,results):
        if saved!=queue:raise ValueError('Exact original timing queue required')
        jobs={j['job_id']:j for j in queue['attempts']}
        for name,job in jobs.items():
            parent=root/'jobs'/_hash(dict(job_id=name))
            for episode in sorted(parent.glob('attempt-*')):
                callpath=episode/'call-clock.json'
                if not callpath.is_file():
                    # Recovery never invokes a second measurement. Its work is
                    # covered by the enclosing queue, with no new call clock.
                    recovery=results.get(name,{}).get('value',{})
                    if (episode.name=='attempt-1' and not (episode/'process').exists()
                            and recovery.get('status') in ('completed_ineligible','irrecoverable_missing')):continue
                    unknown.append(str(episode));continue
                call=_read_call(_ref(callpath),queue,job,episode);calls[str(callpath)]=call;references.append(_ref(callpath))
                if call['process_reference'] is not None:
                    from .cooperative_process_v2 import inspect_cooperative_process
                    p=call['process_reference'];proof=inspect_cooperative_process(root=p['process_root'],expected_receipt_sha256=p['process_receipt_sha256'],expected_started_sha256=p['process_started_sha256'])
                    total_root+=proof['elapsed_ns']
            entry=results.get(name);value={} if entry is None else entry['value']
            if value.get('status')!='completed_ineligible':
                missing.append(dict(job_id=name,status=value.get('status','unstarted')));continue
            callref=value['call_reference'];episode=_path(callref['path']).parent
            if episode.parent!=parent:raise ValueError('A result adopted another timing member')
            actual=_measurement_value(queue,job,episode)
            if any(value[k]!=actual[k] for k in ('status','measurement_readback','call_reference')) or any(value[k] is not False for k in _FALSE):
                raise ValueError('Saved original measurement readback changed')
            _,raw=_reference(actual['measurement_readback']['measurement_reference']);measured=_json(raw)
            call=_read_call(callref,queue,job,episode)
            records[name]=dict(job=job['payload'],measurement=measured,call=call,root_ns=actual['measurement_readback']['observed_root_elapsed_ns'])
        invocation_paths=sorted(root.glob('invocation-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
        if [p.name for p in invocation_paths]!=[f'invocation-{i}.json' for i in range(len(invocation_paths))]:raise ValueError('Complete timing queue invocation history required')
        total_invocations=0;invocations=[]
        for path in invocation_paths:
            value=_json(path.read_bytes());amount=value.get('invocation_ns')
            if type(amount) is not int or amount<=0 or value.get('admitted_job_ids')!=list(jobs):raise ValueError('Complete positive queue invocation clock required')
            total_invocations+=amount;invocations.append(_ref(path))
        context_paths=sorted(root.glob('context-*.json'),key=lambda p:int(p.stem.split('-')[-1]))
        if [p.name for p in context_paths]!=[f'context-{i}.json' for i in range(len(context_paths))]:raise ValueError('Complete timing context history required')
        context_refs=[];covered=[];total_context=0;failed_contexts=[]
        from .execution_clock_v2 import validate_clock_domain
        for path in context_paths:
            value=_json(path.read_bytes())
            fields={'schema','source_sha256','queue_plan_hash','origin_ns','end_ns','elapsed_ns','clock_domain','queue_invocation_references',
                'status','error','clock_scope','standalone_authority'}
            if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-native-timing-invocation-context'
                    or value['source_sha256']!=_source() or value['queue_plan_hash']!=queue['plan_hash']
                    or any(type(value[k]) is not int or value[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
                    or value['end_ns']-value['origin_ns']!=value['elapsed_ns'] or value['standalone_authority'] is not False
                    or value['clock_scope']!='Queue call plus final actual-context guard; own context publication/return and outer admission/startup excluded'
                    or type(value['queue_invocation_references']) is not list or len(value['queue_invocation_references'])!=1
                    or type(value['status']) is not str or not value['status']
                    or value['error'] is not None and (type(value['error']) is not dict or set(value['error'])!={'type','reason'}
                        or any(type(value['error'][k]) is not str for k in ('type','reason')))):
                raise ValueError('Strict closed actual-context invocation required')
            validate_clock_domain(value['clock_domain']);ref=value['queue_invocation_references'][0]
            if ref not in invocations or ref in covered:raise ValueError('Each queue invocation needs exactly one containing context')
            _,raw=_reference(ref)
            if _json(raw)['invocation_ns']>value['elapsed_ns']:raise ValueError('Queue duration exceeds its actual context')
            covered.append(ref);total_context+=value['elapsed_ns'];context_refs.append(_ref(path))
            if value['error'] is not None:failed_contexts.append(_ref(path))
        if covered!=invocations:raise ValueError('Missing containing timing context cannot be ignored')
        total_calls=sum(v['elapsed_ns'] for v in calls.values())
        if total_calls>total_invocations or total_root>total_calls:raise ValueError('Nested measurement clocks exceed their enclosing calls/queue invocations')
        ledger_ref=_ref(root/'events.jsonl');charged=ledger.charged_ns
        if any(e.get('cost_precision')=='wall_time_upper_bound_after_worker_loss' for e in ledger.events):unknown.append('worker_loss_interval_is_only_an_upper_bound')
        comparisons={}
        for item in _untrusted_plan(queue['identity']['measurement_plan_reference'])['schedule']['comparisons']:
            name=item['comparison_id'];members={}
            for record in records.values():
                j=record['job']
                if j['comparison_id']==name:members[j['block_id'],j['role']]=record
            incomplete=[i for i in range(30) if any((i,r) not in members for r in ('selected','baseline'))]
            def blocks(kind,size=None):
                out=[]
                for i in range(30):
                    selected=members[i,'selected'];baseline=members[i,'baseline']
                    def amount(v):
                        if kind=='root':return v['root_ns']
                        if kind=='call':return v['call']['elapsed_ns']
                        if kind=='load':return v['measurement']['model_load_ns']
                        row=next(b for b in v['measurement']['batches'] if b['batch_size']==size)
                        return row['first_prediction_ns'] if kind=='first' else int(round(sum(row['resident_prediction_ns'])/32))
                    out.append(dict(block_id=i,order=selected['job']['order'],selected_ns=amount(selected),baseline_ns=amount(baseline)))
                return out
            comparisons[name]=dict(missing_blocks=incomplete,resident_blocks=None if incomplete else {str(s):blocks('resident',s) for s in (1,1024)},
                first_prediction_blocks=None if incomplete else {str(s):blocks('first',s) for s in (1,1024)},
                model_load_blocks=None if incomplete else blocks('load'),cold_process_blocks=None if incomplete else blocks('root'),
                full_executor_call_blocks=None if incomplete else blocks('call'),**_FALSE)
        # Read-only rechecks detect concurrent mutation without another forward pass.
        _checked_queue(queue)
        for reference in references+invocations+context_refs+[ledger_ref]:_reference(reference)
    complete=sum(not v['missing_blocks'] for v in comparisons.values())
    return dict(schema='research/v2-original-native-timing-readback',status='complete_descriptive_measurements' if not missing and complete==len(comparisons) else 'incomplete_measurements',
        queue_plan_hash=queue['plan_hash'],ledger_reference=ledger_ref,invocation_references=invocations,call_references=references,
        completed_measurements=len(records),missing_members=missing,comparisons=comparisons,complete_30_block_comparisons=complete,
        sum_full_executor_call_ns=total_calls,sum_owned_root_ns=total_root,sum_queue_invocation_ns=total_invocations,charged_queue_job_ns=charged,
        context_references=context_refs,sum_context_invocation_ns=total_context,failed_context_references=failed_contexts,
        all_attempt_costs_observed=not unknown,unobserved_costs=unknown,**_FALSE,
        cost_boundary='Calls and roots nest inside queue invocations; never add them. Outer publication/return/cold startup, native fitting and standalone prerequisites remain separate.',
        scope='Exact owned original-model samples and paid local failures only; no native/F-role/dedicated/precision or whole-cost/quality claim')


def _actual(reference,protocol_reference,directory,kwargs):
    from .native_deployment_plan_v2 import load_native_deployment_plan
    return load_native_deployment_plan(reference,protocol_reference=protocol_reference,directory=directory,**kwargs)


def run_native_deployment_measurements(reference,*,protocol_reference,directory,output,models_reference,global_barrier_reference,
        qualification_reference,preT_costs_reference,readiness_reference,resume=False,max_jobs_this_invocation=None,rss_limit_bytes=4*1024**3):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    plan=_actual(reference,protocol_reference,directory,kwargs)
    from .native_deployment_plan_v2 import _actual_protocol
    from .protected_prediction_gate_v2 import _publication_location
    protocol=_actual_protocol(protocol_reference,kwargs)
    target=_publication_location(output,protocol=protocol,inspection=dict(study_directory=plan.payload['study_directory'],owner_root=plan.payload['owner_root'],invocation_references=[]))
    protected=_path(reference['path']).parent
    if target.is_relative_to(protected) or protected.is_relative_to(target):raise ValueError('Timing output must preserve its original immutable plan')
    queue=_make_execution_queue(reference,plan.payload,rss_limit_bytes=rss_limit_bytes)
    def guard():
        if _actual(reference,protocol_reference,directory,kwargs).payload!=plan.payload:raise ValueError('Actual frozen native timing context changed')
    result=_run_measurement_queue(queue,target,resume=resume,max_jobs_this_invocation=max_jobs_this_invocation,guard=guard)
    return dict(result,actual_plan_binding_authenticated=True,**_FALSE)


def collect_native_deployment_measurements(reference,*,protocol_reference,directory,output,models_reference,global_barrier_reference,
        qualification_reference,preT_costs_reference,readiness_reference,rss_limit_bytes=4*1024**3):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    plan=_actual(reference,protocol_reference,directory,kwargs);queue=_make_execution_queue(reference,plan.payload,rss_limit_bytes=rss_limit_bytes)
    result=_collect_measurement_queue(queue,output)
    if _actual(reference,protocol_reference,directory,kwargs).payload!=plan.payload:raise ValueError('Actual native timing context changed during readback')
    return dict(result,actual_plan_binding_authenticated=True)
