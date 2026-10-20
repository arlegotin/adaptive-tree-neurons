"""Genuine-protocol prediction scheduling; this module never opens test labels.

Private queue fixtures authenticate no native or protected procedure. Only the
public entry constructs its in-invocation context through actual loaders.
"""
import copy
import hashlib
import json
from pathlib import Path
import secrets
import sys
import time
from types import SimpleNamespace

from .deployment_worker_v2 import _source,_ROOT,_THREADS,_json
from .protected_journal_v2 import (_path,_reference,_canonical,_hash,_write_new,_check_storage,
    _prepare_metadata_study,_claim_metadata_owners,_read_metadata_study,_append_metadata_event)
from .protected_query_v2 import _compile_query_request
from .protected_query_worker_v2 import inspect_owned_saved_query

_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False)


def _context_payload(payload,reference):
    payload=json.loads(_canonical(payload));_,raw=_reference(reference);stored=_json(raw)
    if (payload.get('source_sha256')!=_source() or (stored.get('payload') if stored.get('schema')=='research/v2-final-practical-prediction-protocol' else stored)!=payload):
        raise ValueError('Exact immutable current-source protocol payload/bytes required')
    queries=payload['prediction_manifest']['queries'];ids=[q['query_id'] for q in queries]
    if not ids or ids!=sorted(set(ids)):raise ValueError('Exact ordered frozen query inventory required')
    return payload


def _admitted_metadata(payload,reference):
    from .protected_journal_v2 import _admitted_metadata as metadata
    return metadata(SimpleNamespace(payload=payload,reference=reference))


def _event_rows(root):
    return [_json(line) for line in (root/'events.jsonl').read_bytes().splitlines()]


def _checked_commit(context,reference,*,event,root):
    _,raw=_reference(reference);value=_json(raw);details=event['details'];qid=details['query_id'];index=details['attempt_index']
    episode=root/'queries'/qid/f'episode-{index:04}'
    fields={'schema','source_sha256','protocol_sha256','genesis_sha256','query_id','attempt_index','core_request','core_request_sha256',
        'owned_readback','process_reference','call_reference','standalone_authority'}
    if (type(value) is not dict or set(value)!=fields or value['schema']!='research/v2-owned-prediction-commit'
            or value['standalone_authority'] is not False or reference['path']!=str(episode/'prediction-commit.json')
            or value['source_sha256']!=context['payload']['source_sha256'] or value['protocol_sha256']!=context['reference']['sha256']
            or value['genesis_sha256']!=context['binding']['genesis_sha256'] or value['query_id']!=qid
            or type(value['attempt_index']) is not int or value['attempt_index']!=index
            or value['core_request_sha256']!=_hash(value['core_request'])):
        raise ValueError('Exact frozen owned numerical prediction commit required')
    query=next((q for q in context['payload']['prediction_manifest']['queries'] if q['query_id']==qid),None)
    if query is None or value['core_request']['query']!=query:raise ValueError('Committed query differs from the frozen inventory')
    process=value['process_reference']
    if process.get('process_root')!=str(episode/'process'):raise ValueError('Prediction commit substituted another owned episode')
    actual=inspect_owned_saved_query(process)
    if (actual!=value['owned_readback'] or actual['core_request_sha256']!=value['core_request_sha256']
            or actual['query_id']!=qid or actual['attempt_index']!=index or actual['genesis_sha256']!=value['genesis_sha256']):
        raise ValueError('Prediction commit lacks its actual owned/numerical evidence')
    _,_,call=_read_call(value['call_reference'],context=context,episode=episode,query_id=qid,attempt_index=index)
    if call['elapsed_ns']!=details['elapsed_ns'] or call['process_reference']!=process or call['error'] is not None:
        raise ValueError('Prediction commit and observed full-call cost disagree')
    return value


def _read_call(reference,*,context,episode,query_id,attempt_index):
    path,raw=_reference(reference);value=_json(raw)
    fields={'schema','source_sha256','protocol_sha256','genesis_sha256','query_id','attempt_index','origin_ns','end_ns','elapsed_ns',
        'clock_domain','process_reference','error','clock_scope','standalone_authority'}
    if (path!=episode/'call-clock.json' or type(value) is not dict or set(value)!=fields
            or value['schema']!='research/v2-protected-query-call-clock' or value['source_sha256']!=context['payload']['source_sha256']
            or value['protocol_sha256']!=context['reference']['sha256'] or value['genesis_sha256']!=context['binding']['genesis_sha256']
            or value['query_id']!=query_id or type(value['attempt_index']) is not int or value['attempt_index']!=attempt_index
            or any(type(value[k]) is not int or value[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
            or value['end_ns']-value['origin_ns']!=value['elapsed_ns'] or value['standalone_authority'] is not False
            or value['clock_scope']!='Full executor call through return/exception; own clock publication and queue bookkeeping excluded'):
        raise ValueError('Exact observed protected query call required')
    from .execution_clock_v2 import validate_clock_domain
    validate_clock_domain(value['clock_domain'])
    return path,raw,value




def _checked_attempt_process(context,episode,process,call):
    from .cooperative_process_v2 import inspect_cooperative_process
    from .protected_query_worker_v2 import protected_query_worker_identity,_checked_delegate_prefix
    proof=inspect_cooperative_process(root=episode/'process',expected_receipt_sha256=process['process_receipt_sha256'],
        expected_started_sha256=process['process_started_sha256'])
    start=_json((episode/'process/started.json').read_bytes());origin=start['supervisor_started_monotonic_ns']
    if not call['origin_ns']<=origin<=origin+proof['elapsed_ns']<=call['end_ns']:
        raise ValueError('Actual owned root clock must lie inside the full observed executor call')
    argv=start['argv']
    if (type(argv) is not list or len(argv)!=11 or argv[:6]!=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2','--request']
            or argv[6]!=str(episode/'request.json') or argv[7]!='--expected-request-sha256' or argv[9]!='--output' or argv[10]!=str(episode/'worker')
            or start['cwd']!=str(_ROOT) or start['identity']!=protected_query_worker_identity(Path(argv[6]),argv[8])
            or any(start['environment_overrides'].get(k)!='1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE'))):
        raise ValueError('Attempt cost requires its exact owned source/request/thread command')
    delegate=_checked_delegate_prefix(Path(argv[6]),argv[8],output=Path(argv[10]))
    if (delegate['query_id']!=call['query_id'] or delegate['attempt_index']!=call['attempt_index']
            or delegate['genesis_sha256']!=context['binding']['genesis_sha256']
            or delegate['protocol_reference']['sha256']!=context['reference']['sha256']):
        raise ValueError('Attempt process substituted another protocol/query/genesis')
    return proof,delegate


def _failed_disposition(proof,episode,delegate):
    if proof['grant_publication_started'] is False:
        if proof['execution_status']=='resource_exhausted':
            return 'resource_pause','Known admission/resource denial before grant'
        if proof['execution_status']=='interrupted_recoverable':
            return 'interrupted_recoverable','Known interruption before grant'
        return 'input_or_dependency_failure',proof.get('diagnostic') or 'Known failed launch/handshake before grant'
    if not proof['worker_protocol_closed']:
        return 'irrecoverable_missing','Granted worker lacks a complete owned closure'
    committed={r['path']:r for r in proof['files']}
    if 'body-outcome.json' not in committed:return 'irrecoverable_missing','Closed worker body evidence is missing'
    _,raw=_reference(dict(path=str(episode/'process/body-outcome.json'),sha256=committed['body-outcome.json']['sha256']))
    body=_json(raw)
    if (body.get('schema')!='research/v2-owned-saved-query-body' or body.get('status')!='failed_saved_query_body'
            or body.get('source_sha256')!=delegate['source_sha256']
            or body.get('request_reference')!=dict(path=str(episode/'request.json'),sha256=hashlib.sha256((episode/'request.json').read_bytes()).hexdigest())
            or any(body.get(k) is not False for k in _FALSE)):
        raise ValueError('Failed body must match its actual closed source/delegation')
    reference=body.get('terminal_reference')
    if reference is None:return 'input_or_dependency_failure',body.get('reason') or 'Delegation failed before native forward'
    if reference.get('path')!=str(episode/'worker/terminal.json'):raise ValueError('Failed terminal belongs to another query')
    _,raw=_reference(reference);terminal=_json(raw)
    if (terminal.get('schema')!='research/v2-saved-feature-query-terminal' or terminal.get('request_sha256')!=delegate['core_request_sha256']
            or any(terminal.get(k) is not False for k in _FALSE)):
        raise ValueError('Failed terminal/source request or authority differs')
    status={'resource_exhausted':'resource_pause','interrupted_recoverable':'interrupted_recoverable','numerical_failure':'numerical_failure',
        'input_or_dependency_failure':'input_or_dependency_failure'}.get(terminal.get('status'),'irrecoverable_missing')
    return status,terminal.get('reason') or 'Closed failed saved-query body'


def _validate_attempt_costs(context,root,state):
    """Reconcile every completed attempt, including paid losing/denied work."""
    from .cooperative_process_v2 import inspect_cooperative_process
    total=0
    for event in _event_rows(root):
        if event['kind'] not in ('query_committed','query_failed','query_failed_with_evidence'):continue
        detail=event['details'];qid=detail['query_id'];index=detail['attempt_index']
        episode=root/'queries'/qid/f'episode-{index:04}'
        if event['kind']=='query_failed':
            raise ValueError('Legacy metadata failure lacks an actual immutable attempt-evidence reference')
        if event['kind']=='query_committed':
            _,raw=_reference(detail['prediction_reference']);reference=_json(raw)['call_reference']
        else:reference=detail['evidence_reference']
        _,_,call=_read_call(reference,context=context,episode=episode,query_id=qid,attempt_index=index)
        if call['elapsed_ns']!=detail['elapsed_ns']:raise ValueError('Recorded failed/successful cost differs from its actual full call')
        total+=call['elapsed_ns'];process=call['process_reference']
        if process is not None:
            if process.get('process_root')!=str(episode/'process'):raise ValueError('Attempt cost substituted a process')
            proof,delegate=_checked_attempt_process(context,episode,process,call)
            if event['kind']=='query_failed_with_evidence':
                expected,_=_failed_disposition(proof,episode,delegate)
                recovery=reference if expected in ('resource_pause','interrupted_recoverable') else None
                if detail['status']!=expected or detail['recovery_reference']!=recovery:
                    raise ValueError('Failed attempt classification/recovery contradicts actual owned evidence')
        elif event['kind']=='query_committed' or detail['status']!='irrecoverable_missing':
            raise ValueError('Missing process evidence cannot be treated as complete or recoverable')
    if total!=state['known_paid_ns'] or not state['execution_cost_complete']:
        raise ValueError('Every charged query attempt requires known, matching call evidence')


def _validate_commits(context,root,state):
    done={}
    _validate_attempt_costs(context,root,state)
    for event in _event_rows(root):
        if event['kind']=='query_committed':
            qid=event['details']['query_id'];reference=event['details']['prediction_reference']
            done[qid]=_checked_commit(context,reference,event=event,root=root)
    if set(done)!=set(state['committed_queries']):raise ValueError('Every recorded query needs its actual finite owned commit')
    return done


def _publish_delegate(context,root,state,query):
    qid=query['query_id'];index=state['attempts'].get(qid,0);failure=state['last_failures'].get(qid)
    mode='initial' if index==0 else 'same_state_resume' if failure in ('resource_pause','interrupted_recoverable') else 'same_seed_rerun'
    core=_compile_query_request(context['payload'],state,query_id=qid)
    episode=root/'queries'/qid/f'episode-{index:04}'
    if episode.exists():raise ValueError('Fresh exact query episode required; pending histories cannot be reset')
    _append_metadata_event(root,kind='query_started',details=dict(query_id=qid,attempt_index=index,resume_kind=mode))
    current=_read_metadata_study(root);episode.mkdir(parents=True,exist_ok=False)
    delegate=dict(schema='research/v2-owned-saved-query-delegate',source_sha256=context['payload']['source_sha256'],
        protocol_reference=context['reference'],study_directory=str(root),
        study_binding_reference=dict(path=str(root/'binding.json'),sha256=hashlib.sha256((root/'binding.json').read_bytes()).hexdigest()),
        genesis_sha256=context['binding']['genesis_sha256'],query_id=qid,attempt_index=index,journal_sha256=current['journal_sha256'],
        query_started_event_hash=current['last_event_hash'],core_request=core,core_request_sha256=_hash(core),worker_output=str(episode/'worker'))
    path=episode/'request.json';_write_new(path,delegate)
    return episode,delegate,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _run_prediction_queue(*,protocol_payload,protocol_reference,directory,owner_root,execute_episode,guard,
        max_queries_this_invocation=None,rss_limit_bytes=4*1024**3,invocation_origin_ns=None):
    """Private immutable in-call composition; outputs grant no native/T authority."""
    origin=time.perf_counter_ns() if invocation_origin_ns is None else invocation_origin_ns
    payload=_context_payload(protocol_payload,protocol_reference)
    if (type(rss_limit_bytes) is not int or not 0<rss_limit_bytes<=16*1024**3
            or max_queries_this_invocation is not None and (type(max_queries_this_invocation) is not int or max_queries_this_invocation<=0)):
        raise ValueError('Bounded root RSS and explicit operational query count required')
    guard()
    protected=[protocol_reference['path'],*(d['root'] for d in payload['data'].values()),
        *(q['model_binding']['snapshot_root'] for q in payload['prediction_manifest']['queries'])]
    root,owners=_check_storage(directory,owner_root,protected_paths=protected)
    _prepare_metadata_study(directory=root,owner_root=owners,**_admitted_metadata(payload,protocol_reference));_claim_metadata_owners(root)
    from .ledger import StudyLease
    from .execution_clock_v2 import observe_clock_domain
    domain=observe_clock_domain();invocation=secrets.token_hex(16);calls=[];error=None;status='operator_pause';attempted=0
    context=dict(payload=payload,reference=copy.deepcopy(protocol_reference),binding=_read_metadata_study(root)['binding'])
    start_file=root/'invocations'/f'{invocation}-start.json'
    _write_new(start_file,dict(schema='research/v2-protected-prediction-invocation-start',source_sha256=payload['source_sha256'],
        protocol_reference=protocol_reference,genesis_sha256=context['binding']['genesis_sha256'],origin_ns=origin,clock_domain=domain,
        standalone_authority=False))
    try:
        with StudyLease(root):
            state=_read_metadata_study(root);done=_validate_commits(context,root,state)
            if state['active'] is not None:status='blocked_ambiguous_active_query'
            elif state['phase'] in ('all_predictions_committed','labels_started','metrics_committed'):status='all_predictions_committed'
            else:
                for query in payload['prediction_manifest']['queries']:
                    qid=query['query_id']
                    if qid in done:continue
                    if max_queries_this_invocation is not None and attempted>=max_queries_this_invocation:break
                    state=_read_metadata_study(root)
                    failure=state['last_failures'].get(qid)
                    if failure in ('numerical_failure','irrecoverable_missing') or failure=='input_or_dependency_failure' and state['reruns'].get(qid,0):
                        status='blocked_terminal_query_failure';break
                    _context_payload(payload,protocol_reference)
                    episode,delegate,request_reference=_publish_delegate(context,root,state,query)
                    attempted+=1;call_begin=time.perf_counter_ns();returned=None;call_error=None
                    try:
                        returned=execute_episode(query_id=qid,request_path=episode/'request.json',
                            expected_request_sha256=request_reference['sha256'],worker_output=episode/'worker',
                            control_output=episode/'process',rss_limit_bytes=rss_limit_bytes)
                    except BaseException as caught:
                        call_error=dict(type=type(caught).__name__,reason=str(caught))
                    call_end=time.perf_counter_ns()
                    process=returned.get('process_reference') if type(returned) is dict else None
                    call=dict(schema='research/v2-protected-query-call-clock',source_sha256=payload['source_sha256'],
                        protocol_sha256=protocol_reference['sha256'],genesis_sha256=context['binding']['genesis_sha256'],
                        query_id=qid,attempt_index=delegate['attempt_index'],origin_ns=call_begin,end_ns=call_end,elapsed_ns=call_end-call_begin,
                        clock_domain=domain,process_reference=process,error=call_error,
                        clock_scope='Full executor call through return/exception; own clock publication and queue bookkeeping excluded',standalone_authority=False)
                    call_path=episode/'call-clock.json';_write_new(call_path,call)
                    call_reference=dict(path=str(call_path),sha256=hashlib.sha256(call_path.read_bytes()).hexdigest());calls.append(call_reference)
                    if process is None or call_error is not None:
                        _append_metadata_event(root,kind='query_failed_with_evidence',details=dict(query_id=qid,attempt_index=delegate['attempt_index'],
                            status='irrecoverable_missing',error='Missing closed executor evidence; no automatic retry',elapsed_ns=call['elapsed_ns'],recovery_reference=None,evidence_reference=call_reference))
                        status='blocked_missing_attempt_evidence';break
                    if process.get('process_root')!=str(episode/'process'):raise ValueError('Executor changed its owned query episode')
                    from .cooperative_process_v2 import inspect_cooperative_process
                    proof=inspect_cooperative_process(root=episode/'process',expected_receipt_sha256=process['process_receipt_sha256'],
                        expected_started_sha256=process['process_started_sha256'])
                    if proof['returncode']==0 and proof['worker_protocol_closed']:
                        joined=inspect_owned_saved_query(process)
                        if joined['query_id']!=qid or joined['core_request_sha256']!=delegate['core_request_sha256']:
                            raise ValueError('Owned query does not match this frozen scheduled request')
                        commit=dict(schema='research/v2-owned-prediction-commit',source_sha256=payload['source_sha256'],
                            protocol_sha256=protocol_reference['sha256'],genesis_sha256=context['binding']['genesis_sha256'],
                            query_id=qid,attempt_index=delegate['attempt_index'],core_request=delegate['core_request'],
                            core_request_sha256=delegate['core_request_sha256'],owned_readback=joined,process_reference=process,
                            call_reference=call_reference,standalone_authority=False)
                        path=episode/'prediction-commit.json';_write_new(path,commit)
                        reference=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                        _append_metadata_event(root,kind='query_committed',details=dict(query_id=qid,attempt_index=delegate['attempt_index'],
                            prediction_reference=reference,elapsed_ns=call['elapsed_ns']))
                        done[qid]=commit
                    else:
                        state_name,reason=_failed_disposition(proof,episode,delegate)
                        recoverable=state_name in ('resource_pause','interrupted_recoverable')
                        _append_metadata_event(root,kind='query_failed_with_evidence',details=dict(query_id=qid,attempt_index=delegate['attempt_index'],
                            status=state_name,error=reason or 'Closed failed query',elapsed_ns=call['elapsed_ns'],
                            recovery_reference=call_reference if recoverable else None,evidence_reference=call_reference))
                        status='resource_pause' if recoverable else 'blocked_query_failure';break
                else:status='all_predictions_committed'
            guard();_context_payload(payload,protocol_reference)
            state=_read_metadata_study(root);done=_validate_commits(context,root,state)
            if len(done)==len(payload['prediction_manifest']['queries']) and state['active'] is None:
                if state['phase']=='owners_ready':_append_metadata_event(root,kind='all_predictions_committed',details={})
                status='all_predictions_committed'
    except BaseException as caught:
        error=dict(type=type(caught).__name__,reason=str(caught));status='blocked_coordinator_failure'
    finally:
        end=time.perf_counter_ns();state=_read_metadata_study(root)
        clock=dict(schema='research/v2-protected-prediction-invocation-clock',source_sha256=payload['source_sha256'],
            protocol_reference=protocol_reference,genesis_sha256=context['binding']['genesis_sha256'],
            start_reference=dict(path=str(start_file),sha256=hashlib.sha256(start_file.read_bytes()).hexdigest()),
            origin_ns=origin,end_ns=end,elapsed_ns=end-origin,clock_domain=domain,call_references=calls,
            journal_sha256=state['journal_sha256'],status=status,error=error,standalone_authority=False,
            clock_scope='Invocation entry through queue/full guards; own final clock publication/return excluded; call clocks nested')
        final=root/'invocations'/f'{invocation}-end.json';_write_new(final,clock)
    return dict(schema='research/v2-protected-prediction-queue-progress',status=status,completed_queries=len(state['committed_queries']),
        planned_queries=len(payload['prediction_manifest']['queries']),attempted_queries=attempted,journal_sha256=state['journal_sha256'],
        invocation_reference=dict(path=str(final),sha256=hashlib.sha256(final.read_bytes()).hexdigest()),
        labels_read=False,**_FALSE,scope='Private queue/process/numerical composition only; no native/protected/label or whole-cost authority')


def _execute_owned_episode(*,query_id,request_path,expected_request_sha256,worker_output,control_output,rss_limit_bytes):
    from .cooperative_process_v2 import supervise_cooperative_process
    from .protected_query_worker_v2 import protected_query_worker_identity
    from .native_resource_forecast_v2 import observe_native_host_resources
    def admission():
        live=observe_native_host_resources(control_output.parent);available=live['available_memory_bytes']
        required=(5*rss_limit_bytes+3)//4+512*1024**2
        return dict(admitted=type(available) is int and available>=required and live['free_disk_bytes']>=512*1024**2,
            live_observation=live,required_memory_proxy_bytes=required,required_disk_proxy_bytes=512*1024**2)
    overrides={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')};overrides.update(PYTHONPATH=str(_ROOT/'src'),KMP_DUPLICATE_LIB_OK=None)
    supervise_cooperative_process(argv=[sys.executable,'-P','-B','-m','treeneuron.research.protected_query_worker_v2','--request',
        str(request_path),'--expected-request-sha256',expected_request_sha256,'--output',str(worker_output)],cwd=_ROOT,
        env_overrides=overrides,output=control_output,identity=protected_query_worker_identity(request_path,expected_request_sha256),
        admission_identity=dict(purpose='frozen_saved_query',query_id=query_id,root_rss_pause_bytes=rss_limit_bytes),
        admission_check=admission,rss_pause_bytes=rss_limit_bytes,admit_before_spawn=True)
    return dict(process_reference=dict(process_root=str(control_output),
        process_receipt_sha256=hashlib.sha256((control_output/'process.json').read_bytes()).hexdigest(),
        process_started_sha256=hashlib.sha256((control_output/'started.json').read_bytes()).hexdigest()))


def run_protected_predictions(reference,*,directory,models_reference,global_barrier_reference,qualification_reference,
        preT_costs_reference,readiness_reference,max_queries_this_invocation=None,rss_limit_bytes=4*1024**3):
    origin=time.perf_counter_ns()
    from .practical_T_protocol_v2 import load_practical_T_protocol
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=load_practical_T_protocol(reference,**kwargs);payload=protocol.payload
    from .protected_journal_v2 import _public_locations
    root,owners=_public_locations(directory,protocol)
    def guard():
        actual=load_practical_T_protocol(reference,**kwargs)
        if _canonical(actual.payload)!=_canonical(payload):raise ValueError('Actual frozen native/model/protocol/cost/readiness context changed')
    result=_run_prediction_queue(protocol_payload=payload,protocol_reference=protocol.reference,directory=root,owner_root=owners,
        execute_episode=_execute_owned_episode,guard=guard,max_queries_this_invocation=max_queries_this_invocation,
        rss_limit_bytes=rss_limit_bytes,invocation_origin_ns=origin)
    return dict(result,genuine_protocol_invocation_authenticated=result['status']!='blocked_coordinator_failure',
        scope='Actual full protocol gates surround this queue; label access and complete pipeline cost remain separate')
