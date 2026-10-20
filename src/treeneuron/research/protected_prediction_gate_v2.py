"""Complete readonly prediction evidence; label access remains a later gate."""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import time

from .deployment_worker_v2 import _source,_json
from .protected_journal_v2 import (_path,_reference,_canonical,_hash,_read_metadata_study,_assets,
    _initial_state,_transition,_write_new)
from .protected_prediction_queue_v2 import (_context_payload,_admitted_metadata,_validate_commits,_read_call)

_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False,
    whole_training_procedure_cost_authenticated=False,cost_claim_authorized=False)


def _barrier_history(root,state):
    binding=state['binding'];prefix=hashlib.sha256();past=_initial_state(binding);rows={}
    barrier=None
    for line in (root/'events.jsonl').read_bytes().splitlines(keepends=True):
        prefix.update(line);event=_json(line);_transition(past,event,binding);rows[prefix.hexdigest()]=copy.deepcopy(past)
        if event['kind']=='all_predictions_committed':
            if barrier is not None:raise ValueError('Exactly one prediction completion barrier required')
            barrier=dict(journal_prefix_sha256=prefix.hexdigest(),event_hash=event['event_hash'],state=copy.deepcopy(past))
    if barrier is None:raise ValueError('Complete witnessed prediction barrier is absent')
    return barrier,rows


def _invocations(context,root,history):
    from .execution_clock_v2 import validate_clock_domain,same_clock_domain
    directory=root/'invocations';starts={};ends={}
    for p in directory.iterdir():
        if p.is_symlink() or not p.is_file():raise ValueError('Ordinary immutable invocation files required')
        matched=re.fullmatch(r'([0-9a-f]{32})-(start|end)\.json',p.name)
        if not matched:raise ValueError('Unknown invocation record cannot be ignored')
        (starts if matched[2]=='start' else ends)[matched[1]]=p
    if not starts or set(starts)!=set(ends):raise ValueError('Every invocation start needs its complete end; orphans cannot be ignored')
    references=[];calls={};intervals=[];completed=False
    start_fields={'schema','source_sha256','protocol_reference','genesis_sha256','origin_ns','clock_domain','standalone_authority'}
    end_fields={'schema','source_sha256','protocol_reference','genesis_sha256','start_reference','origin_ns','end_ns','elapsed_ns',
        'clock_domain','call_references','journal_sha256','status','error','standalone_authority','clock_scope'}
    scope='Invocation entry through queue/full guards; own final clock publication/return excluded; call clocks nested'
    for identity in sorted(starts):
        p=starts[identity];raw=p.read_bytes();start=_json(raw);end_path=ends[identity];end_raw=end_path.read_bytes();end=_json(end_raw)
        if (type(start) is not dict or set(start)!=start_fields or start['schema']!='research/v2-protected-prediction-invocation-start'
                or type(end) is not dict or set(end)!=end_fields or end['schema']!='research/v2-protected-prediction-invocation-clock'
                or any(v['source_sha256']!=context['payload']['source_sha256'] or v['protocol_reference']['sha256']!=context['reference']['sha256']
                    or v['genesis_sha256']!=context['binding']['genesis_sha256'] or v['standalone_authority'] is not False for v in (start,end))
                or end['start_reference']!=dict(path=str(p),sha256=hashlib.sha256(raw).hexdigest()) or end['clock_scope']!=scope
                or any(type(end[k]) is not int or end[k]<=0 for k in ('origin_ns','end_ns','elapsed_ns'))
                or type(start['origin_ns']) is not int or start['origin_ns']!=end['origin_ns']
                or end['end_ns']-end['origin_ns']!=end['elapsed_ns'] or start['clock_domain']!=end['clock_domain']
                or type(end['call_references']) is not list or end['journal_sha256'] not in history):
            raise ValueError('Strict source/protocol/genesis/clock/journal-bound invocation required')
        validate_clock_domain(end['clock_domain'])
        if type(end['status']) is not str or not end['status'] or (end['error'] is not None and
                (type(end['error']) is not dict or set(end['error'])!={'type','reason'}
                    or any(type(end['error'][k]) is not str for k in ('type','reason')))):
            raise ValueError('Explicit closed invocation status/error required')
        interval=dict(reference=dict(path=str(end_path),sha256=hashlib.sha256(end_raw).hexdigest()),
            origin_ns=end['origin_ns'],end_ns=end['end_ns'],elapsed_ns=end['elapsed_ns'],clock_domain=end['clock_domain'])
        intervals.append(interval);previous_end=end['origin_ns'];nested=0
        for reference in end['call_references']:
            path=_path(reference['path'])
            if not path.is_relative_to(root/'queries') or path.name!='call-clock.json' or str(path) in calls:
                raise ValueError('Each local full call must be charged by exactly one invocation')
            _,raw=_reference(reference);value=_json(raw);episode=path.parent
            _,_,call=_read_call(reference,context=context,episode=episode,query_id=value['query_id'],attempt_index=value['attempt_index'])
            if (episode!=root/'queries'/call['query_id']/f"episode-{call['attempt_index']:04}" or call['clock_domain']!=end['clock_domain']
                    or not previous_end<=call['origin_ns']<call['end_ns']<=end['end_ns']):
                raise ValueError('Serial calls must fit within their own observed invocation without overlap')
            previous_end=call['end_ns'];nested+=call['elapsed_ns'];calls[str(path)]=dict(reference=reference,call=call)
        if nested>end['elapsed_ns']:raise ValueError('Nested calls cannot exceed containing invocation')
        past=history[end['journal_sha256']]
        if end['status']=='all_predictions_committed':
            if end['error'] is not None or past['phase'] not in ('all_predictions_committed','labels_started','metrics_committed'):
                raise ValueError('Complete invocation must retain its actual prediction barrier')
            completed=True
        references.append(interval['reference'])
    if not completed:raise ValueError('Prediction metadata alone lacks a complete successful invocation')
    all_files={str(p) for p in (root/'queries').glob('*/episode-*/call-clock.json')}
    if set(calls)!=all_files:raise ValueError('Every actual full call needs its containing invocation; no orphan/forgotten work')
    return references,calls,intervals


def _inspect_complete_stage(*,protocol_payload,protocol_reference,directory,owner_root):
    """Untrusted fixture/file inspector only; actual public gates are separate."""
    payload=_context_payload(protocol_payload,protocol_reference);root=_path(directory)
    state=_read_metadata_study(root);binding=state['binding'];metadata=_admitted_metadata(payload,protocol_reference)
    expected=dict(source_sha256=payload['source_sha256'],protocol_sha256=protocol_reference['sha256'],
        prediction_manifest_sha256=payload['prediction_manifest']['manifest_hash'],query_ids=sorted(q['query_id'] for q in payload['prediction_manifest']['queries']),
        assets=_assets(metadata['assets']),owner_root=str(_path(owner_root)))
    if any(binding[k]!=v for k,v in expected.items()) or state['active'] is not None:
        raise ValueError('Complete prediction stage differs from frozen source/protocol/owners/query inventory')
    if state['phase'] not in ('all_predictions_committed','labels_started','metrics_committed'):
        raise ValueError('Every planned prediction is required before complete stage inspection')
    barrier,history=_barrier_history(root,state);past=barrier['state']
    context=dict(payload=payload,reference=copy.deepcopy(protocol_reference),binding=binding)
    done=_validate_commits(context,root,past)
    ids=[q['query_id'] for q in payload['prediction_manifest']['queries']]
    if set(done)!=set(ids):raise ValueError('Missing planned prediction prevents a complete claim')
    references,calls,intervals=_invocations(context,root,history)
    terminal_calls={}
    for line in (root/'events.jsonl').read_bytes().splitlines():
        event=_json(line);kind=event['kind']
        if kind=='query_committed':
            _,raw=_reference(event['details']['prediction_reference']);reference=_json(raw)['call_reference']
        elif kind=='query_failed_with_evidence':reference=event['details']['evidence_reference']
        else:continue
        if reference['path'] in terminal_calls:raise ValueError('An attempt call cannot be charged twice')
        terminal_calls[reference['path']]=reference
    if terminal_calls!={k:v['reference'] for k,v in calls.items()}:raise ValueError('All successful and failed attempt calls must reconcile exactly')
    worker_ns=sum(i['elapsed_ns'] for i in intervals);call_ns=sum(v['call']['elapsed_ns'] for v in calls.values())
    if call_ns!=past['known_paid_ns'] or call_ns>worker_ns:raise ValueError('Invocation/call/journal totals differ')
    from .execution_clock_v2 import same_clock_domain
    same=all(same_clock_domain(
        intervals[0]['clock_domain'],i['clock_domain']) for i in intervals)
    ordered=sorted(intervals,key=lambda i:i['origin_ns']);overlap=same and any(a['end_ns']>b['origin_ns'] for a,b in zip(ordered,ordered[1:]))
    result=dict(schema='research/v2-complete-prediction-file-inspection',source_sha256=payload['source_sha256'],
        protocol_sha256=protocol_reference['sha256'],study_directory=str(root),owner_root=str(_path(owner_root)),genesis_sha256=binding['genesis_sha256'],
        prediction_manifest_sha256=payload['prediction_manifest']['manifest_hash'],prediction_journal_prefix_sha256=barrier['journal_prefix_sha256'],
        prediction_barrier_event_hash=barrier['event_hash'],query_count=len(ids),completed_query_count=len(done),query_ids=ids,
        commits=[done[q] for q in ids],invocation_references=references,invocation_count=len(references),call_count=len(calls),
        additive_invocation_elapsed_ns=worker_ns,nested_executor_call_ns=call_ns,common_invocation_overhead_ns=worker_ns-call_ns,
        same_observed_clock_domain=same,invocations_overlap=overlap if same else None,
        containing_span_ns=max(i['end_ns'] for i in intervals)-min(i['origin_ns'] for i in intervals) if same else None,
        all_prediction_evidence_checked=True,labels_read=False,**_FALSE,
        scope='Complete retained prediction/attempt/invocation file joins only; actual native/protocol/access gates and containing final-publication cost remain separate')
    for reference in references:
        _,raw=_reference(reference);_reference(_json(raw)['start_reference'])
    for entry in calls.values():_reference(entry['reference'])
    expected_names={Path(r['path']).name for r in references}|{Path(r['path']).name.replace('-end.json','-start.json') for r in references}
    if {p.name for p in (root/'invocations').iterdir()}!=expected_names:
        raise ValueError('Invocation history changed during complete stage inspection')
    _context_payload(payload,protocol_reference);_read_metadata_study(root)
    return result


def _actual_protocol(reference,kwargs):
    from .practical_T_protocol_v2 import load_practical_T_protocol
    return load_practical_T_protocol(reference,**kwargs)


def _actual_stage(reference,directory,kwargs):
    protocol=_actual_protocol(reference,kwargs)
    from .protected_journal_v2 import _public_locations
    root,owners=_public_locations(directory,protocol)
    value=_inspect_complete_stage(protocol_payload=protocol.payload,protocol_reference=protocol.reference,directory=root,owner_root=owners)
    if _canonical(_actual_protocol(reference,kwargs).payload)!=_canonical(protocol.payload):raise ValueError('Actual complete native/protocol context changed')
    return protocol,value


@dataclass(frozen=True)
class ProtectedPredictionStage:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)



def _check_stage_output(output,*,inspection,protocol_payload,protected_roots,input_files):
    """Path-only guard, never a qualification or label capability."""
    from .primary_program_v2 import _check_program_location
    working=[inspection['study_directory'],inspection['owner_root'],*(v['root'] for v in protocol_payload['data'].values()),
        *(q['model_binding']['snapshot_root'] for q in protocol_payload['prediction_manifest']['queries'])]
    path=_check_program_location(output,working_roots=working,proof_roots=protected_roots,
        input_files=[*input_files,*(r['path'] for r in inspection['invocation_references'])])
    if path.exists() or path.suffix!='.json':raise ValueError('Fresh complete stage JSON required')
    return path


def _publication_location(output,*,protocol,inspection):
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    payload=protocol.payload;q=payload['qualification_reference'];registry=load_native_qualification_registry(q)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=q)
    _,raw=_reference(payload['preT_costs_reference']);costs=_json(raw)['payload']
    files=[protocol.reference['path'],*(payload[k]['path'] for k in ('models_reference','global_barrier_reference','preT_costs_reference',
        'readiness_reference','practical_plan_reference','procedure_reference'))]
    return _check_stage_output(output,inspection=inspection,protocol_payload=payload,
        protected_roots=[*roots['working_roots'],*roots['proof_roots'],*costs['proof_roots']],input_files=files)


def seal_protected_prediction_stage(reference,*,directory,output,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol,value=_actual_stage(reference,directory,kwargs)
    path=_publication_location(output,protocol=protocol,inspection=value)
    payload=dict(schema='research/v2-genuine-complete-prediction-stage',created_utc_ns=time.time_ns(),
        protocol_reference=protocol.reference,directory=str(_path(directory)),inspection=value,
        genuine_frozen_prediction_evidence_authenticated=True,protected_access_authorized=False,T_accessed=False,
        scope='Actual complete native/protocol/source/owner/model/prediction history authenticated; label access still requires its own once-only evaluator')
    _write_new(path,dict(schema='research/v2-sealed-complete-prediction-stage',payload=payload,payload_sha256=_hash(payload)))
    return ProtectedPredictionStage(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),_canonical(payload))


def load_protected_prediction_stage(reference,*,protocol_reference,directory,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol,actual=_actual_stage(protocol_reference,directory,kwargs)
    _,raw=_reference(reference);saved=_json(raw)
    if (type(saved) is not dict or set(saved)!={'schema','payload','payload_sha256'} or saved['schema']!='research/v2-sealed-complete-prediction-stage'
            or type(saved['payload']) is not dict or _hash(saved['payload'])!=saved['payload_sha256']):
        raise ValueError('Exact immutable complete prediction stage required')
    payload=saved['payload']
    if (set(payload)!={'schema','created_utc_ns','protocol_reference','directory','inspection','genuine_frozen_prediction_evidence_authenticated',
            'protected_access_authorized','T_accessed','scope'} or payload['schema']!='research/v2-genuine-complete-prediction-stage'
            or type(payload['created_utc_ns']) is not int or payload['created_utc_ns']<=0 or payload['protocol_reference']!=protocol.reference
            or payload['directory']!=str(_path(directory)) or payload['inspection']!=actual
            or payload['genuine_frozen_prediction_evidence_authenticated'] is not True or payload['protected_access_authorized'] is not False
            or payload['T_accessed'] is not False
            or payload['scope']!='Actual complete native/protocol/source/owner/model/prediction history authenticated; label access still requires its own once-only evaluator'):
        raise ValueError('Complete prediction stage no longer matches actual native/source/owner/evidence')
    _reference(reference)
    return ProtectedPredictionStage(copy.deepcopy(reference),_canonical(payload))
