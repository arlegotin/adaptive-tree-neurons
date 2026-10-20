"""Once-only workspace ownership and witnessed protected-stage metadata.

Private metadata operations never authorize reading data or running a model.
The public preparation/inspection routes require the actual final protocol first.
This is retained workspace/bundle governance, not isolation from an OS owner.
"""
from contextlib import ExitStack,contextmanager
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import time

from .deployment_worker_v2 import _digest,_json,_source,_ROOT

_FALSE=dict(protected_access_authorized=False,native_qualification_authenticated=False,T_accessed=False)
_BINDING_FIELDS={'schema','source_sha256','protocol_sha256','prediction_manifest_sha256','query_ids','assets',
    'study_directory','owner_root','genesis','genesis_sha256'}
_ACTORS={'prepared':'coordinator','owners_ready':'coordinator','query_started':'coordinator',
    'query_committed':'prediction_worker','query_failed':'prediction_worker','all_predictions_committed':'coordinator',
    'labels_started':'evaluator','metrics_committed':'evaluator'}
_FAILURES={'resource_pause','interrupted_recoverable','input_or_dependency_failure','numerical_failure','irrecoverable_missing'}


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)


def _hash(value):return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _path(value):
    path=Path(value)
    if not path.is_absolute() or path.resolve()!=path or path.is_symlink():raise ValueError('Canonical protected storage path required')
    return path


def _reference(value):
    if type(value) is not dict or set(value)!={'path','sha256'} or not _digest(value['sha256']):
        raise ValueError('Exact immutable protected file reference required')
    path=_path(value['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=value['sha256']:raise ValueError('Protected committed file bytes changed')
    return path,raw


def _write_new(path,value):
    path=_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as stream:
        stream.write(_canonical(value)+'\n');stream.flush();os.fsync(stream.fileno())


@contextmanager
def _lock(path):
    import fcntl
    path=_path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream.fileno(),fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def _asset_key(asset):
    if (type(asset) is not dict or set(asset)!={'dataset','raw_sha256','target'} or not _digest(asset['raw_sha256'])
            or any(type(asset[k]) is not str or not asset[k] for k in ('dataset','target'))):
        raise ValueError('Stable official dataset/raw-byte/target identity required')
    return _hash(asset)


def _assets(value):
    if type(value) is not list or not value:raise ValueError('Explicit protected asset inventory required')
    rows=sorted(copy.deepcopy(value),key=_asset_key)
    if len({_asset_key(row) for row in rows})!=len(rows):raise ValueError('Duplicate protected asset identity')
    return rows


def _check_storage(directory,owner_root,*,protected_paths):
    root,owners=_path(directory),_path(owner_root)
    protected=[_path(p) for p in protected_paths]
    protected.extend((_ROOT/p).resolve() for p in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))
    protected.extend([Path(sys.prefix).resolve(),Path(sys.executable).resolve().parent])
    def overlap(a,b):return a.is_relative_to(b) or b.is_relative_to(a)
    if overlap(root,owners) or any(overlap(r,p) for r in (root,owners) for p in protected):
        raise ValueError('Protected study/owner storage must preserve source/runtime/data/model/proof paths and each other')
    return root,owners


def _validate_binding(binding,root):
    if (type(binding) is not dict or set(binding)!=_BINDING_FIELDS or binding['schema']!='research/v2-protected-study-binding'
            or binding['source_sha256']!=_source() or binding['study_directory']!=str(root)
            or any(not _digest(binding[k]) for k in ('protocol_sha256','prediction_manifest_sha256','genesis','genesis_sha256'))
            or hashlib.sha256(bytes.fromhex(binding['genesis'])).hexdigest()!=binding['genesis_sha256']):
        raise ValueError('Protected binding source/genesis/schema/directory changed')
    _path(binding['owner_root'])
    if _assets(binding['assets'])!=binding['assets']:raise ValueError('Stable sorted protected asset inventory required')
    ids=binding['query_ids']
    if type(ids) is not list or not ids or any(not _digest(q) for q in ids) or ids!=sorted(set(ids)):
        raise ValueError('Exact sorted frozen query inventory required')


def _prepare_metadata_study(*,directory,owner_root,protocol_reference,source_sha256,prediction_manifest_sha256,queries,assets):
    """Private metadata transaction only; cannot open protected roles or models."""
    root,owners=_check_storage(directory,owner_root,protected_paths=[_path(protocol_reference['path'])])
    _,raw=_reference(protocol_reference)
    if source_sha256!=_source() or not _digest(prediction_manifest_sha256):raise ValueError('Exact current source/prediction metadata required')
    if type(queries) is not list or not queries:raise ValueError('Frozen query list required')
    ids=[q['query_id'] for q in queries]
    if any(not _digest(q) for q in ids) or len(set(ids))!=len(ids):raise ValueError('Unique frozen queries required')
    expected=dict(schema='research/v2-protected-study-binding',source_sha256=source_sha256,
        protocol_sha256=hashlib.sha256(raw).hexdigest(),prediction_manifest_sha256=prediction_manifest_sha256,
        query_ids=sorted(ids),assets=_assets(assets),study_directory=str(root),owner_root=str(owners))
    if root.exists():
        result=_read_metadata_study(root);binding=result['binding']
        if {k:v for k,v in binding.items() if k not in ('genesis','genesis_sha256')}!=expected:
            raise ValueError('Cannot resume changed protected source/protocol/assets/query/storage binding')
        return binding
    root.mkdir(parents=True,exist_ok=False);owners.mkdir(parents=True,exist_ok=True)
    genesis=secrets.token_hex(32)
    binding=dict(expected,genesis=genesis,genesis_sha256=hashlib.sha256(bytes.fromhex(genesis)).hexdigest())
    _write_new(root/'binding.json',binding)
    # If interrupted between binding/journal/witness writes, never invent a fresh history.
    with (root/'events.jsonl').open('xb') as stream:stream.flush();os.fsync(stream.fileno())
    _append_metadata_event(root,kind='prepared',details={},_initial=True)
    return copy.deepcopy(binding)


def _owner(binding,asset):
    return dict(schema='research/v2-protected-asset-owner',asset=asset,asset_key=_asset_key(asset),
        protocol_sha256=binding['protocol_sha256'],source_sha256=binding['source_sha256'],
        study_directory=binding['study_directory'],genesis_sha256=binding['genesis_sha256'],binding_sha256=_hash(binding))


def _claims(binding,*,required):
    owners=_path(binding['owner_root'])
    for asset in binding['assets']:
        path=owners/'claims'/f'{_asset_key(asset)}.json'
        if path.exists():
            if _json(_path(path).read_bytes())!=_owner(binding,asset):raise PermissionError('Protected asset already claimed by another owner/protocol/genesis')
        elif required:raise ValueError('Protected owner claim is missing; no access')


def _claim_metadata_owners(directory):
    """Claim all stable assets; partial I/O preserves prior claims for exact resume."""
    root=_path(directory)
    with _lock(root/'journal.lock'):
        state=_read_metadata_study(root);binding=state['binding'];owners=_path(binding['owner_root'])
        with ExitStack() as stack:
            for asset in binding['assets']:stack.enter_context(_lock(owners/'locks'/f'{_asset_key(asset)}.lock'))
            _claims(binding,required=False)  # Check every conflict before publishing any missing owner.
            for asset in binding['assets']:
                path=owners/'claims'/f'{_asset_key(asset)}.json'
                if not path.exists():_write_new(path,_owner(binding,asset))
            _claims(binding,required=True)
            if state['phase']=='prepared':_append_unlocked(root,binding,state,kind='owners_ready',details={},actor='coordinator')
    return _read_metadata_study(root)


def _initial_state(binding):
    return dict(phase=None,committed_queries=[],attempts={},last_failures={},reruns={},active=None,
        known_paid_ns=0,execution_cost_complete=True,event_count=0,last_event_hash=binding['genesis_sha256'])


def _details(event,fields):
    if type(event['details']) is not dict or set(event['details'])!=set(fields):raise ValueError('Strict protected event detail fields required')
    return event['details']


def _cost(state,value):
    if value is None:state['execution_cost_complete']=False
    elif type(value) is not int or value<0:raise ValueError('Protected consumed time must be explicit nonnegative ns or unknown')
    else:state['known_paid_ns']+=value


def _transition(state,event,binding):
    kind=event['kind']
    if kind not in _ACTORS or event['actor']!=_ACTORS[kind]:raise ValueError('Unknown protected event or actor')
    phase=state['phase']
    if kind=='prepared':
        _details(event,())
        if state['event_count'] or phase is not None:raise ValueError('Protected history cannot be reset')
        state['phase']='prepared'
    elif kind=='owners_ready':
        _details(event,())
        if phase!='prepared':raise ValueError('Owner claims precede query stage exactly once')
        state['phase']='owners_ready'
    elif kind=='query_started':
        detail=_details(event,('query_id','attempt_index','resume_kind'));q=detail['query_id'];index=detail['attempt_index'];mode=detail['resume_kind']
        if (phase!='owners_ready' or state['active'] is not None or q not in binding['query_ids']
                or q in state['committed_queries'] or type(index) is not int or index!=state['attempts'].get(q,0)):
            raise ValueError('Frozen query cannot change, overlap, repeat or bypass stage barrier')
        failure=state['last_failures'].get(q)
        if index==0:
            if mode!='initial':raise ValueError('First query attempt must be initial')
        elif failure in ('resource_pause','interrupted_recoverable'):
            if mode!='same_state_resume':raise ValueError('Recoverable query must resume identical saved state')
        elif failure=='input_or_dependency_failure' and state['reruns'].get(q,0)==0:
            if mode!='same_seed_rerun':raise ValueError('Dependency recovery uses one deterministic same-seed rerun')
            state['reruns'][q]=1
        else:raise ValueError('Terminal/ambiguous query failure cannot be rerun or hidden')
        state['attempts'][q]=index+1;state['active']=(q,index)
    elif kind in ('query_committed','query_failed'):
        fields=('query_id','attempt_index','prediction_reference','elapsed_ns') if kind=='query_committed' else (
            'query_id','attempt_index','status','error','elapsed_ns','recovery_reference')
        detail=_details(event,fields);q=detail['query_id'];index=detail['attempt_index']
        if type(index) is not int or phase!='owners_ready' or state['active']!=(q,index):raise ValueError('Query outcome requires its exact active attempt')
        if kind=='query_committed':
            _reference(detail['prediction_reference']);state['committed_queries'].append(q)
        else:
            status=detail['status']
            if status not in _FAILURES or type(detail['error']) is not str or not detail['error']:
                raise ValueError('Explicit protected failure classification/reason required')
            recovery=detail['recovery_reference']
            if status in ('resource_pause','interrupted_recoverable'):_reference(recovery)
            elif recovery is not None:raise ValueError('Terminal failure cannot invent a recovery state')
            state['last_failures'][q]=status
        _cost(state,detail['elapsed_ns']);state['active']=None
    elif kind=='all_predictions_committed':
        _details(event,())
        if phase!='owners_ready' or state['active'] is not None or set(state['committed_queries'])!=set(binding['query_ids']):
            raise ValueError('Every frozen finite aligned prediction must commit before label stage')
        state['phase']=kind
    elif kind=='labels_started':
        _details(event,())
        if phase!='all_predictions_committed':raise ValueError('Labels require all predictions and can start only once')
        state['phase']=kind
    else:
        detail=_details(event,('metrics_reference','elapsed_ns'))
        if phase!='labels_started':raise ValueError('Metrics commit follows the sole label stage')
        _reference(detail['metrics_reference']);_cost(state,detail['elapsed_ns']);state['phase']=kind
    state['event_count']+=1;state['last_event_hash']=event['event_hash']


def _read_metadata_study(directory,*,_allow_initial=False):
    """Reopen strict source, witness, owner and referenced metadata bytes only."""
    root=_path(directory);binding_path=_path(root/'binding.json');binding_raw=binding_path.read_bytes()
    binding=_json(binding_raw);_validate_binding(binding,root)
    journal=_path(root/'events.jsonl');raw=journal.read_bytes();owners=_path(binding['owner_root'])
    history=_path(owners/'histories'/binding['genesis_sha256'])
    if (raw and not raw.endswith(b'\n')) or not raw and not _allow_initial:raise ValueError('Missing/partial protected journal cannot be reconstructed as fresh')
    lines=raw.splitlines();witnesses=sorted(history.glob('*.json')) if history.exists() else []
    if len(witnesses)!=len(lines):raise ValueError('Protected witness/journal count differs; truncation or incomplete transaction')
    state=_initial_state(binding)
    fields={'schema','source_sha256','protocol_sha256','genesis_sha256','binding_sha256','sequence','kind','actor','utc_ns','details','previous_hash','event_hash'}
    for index,line in enumerate(lines):
        event=_json(line);witness=history/f'{index:08}.json'
        if (type(event) is not dict or set(event)!=fields or event['schema']!='research/v2-protected-journal-event'
                or any(event[k]!=binding[k] for k in ('source_sha256','protocol_sha256','genesis_sha256')) or event['binding_sha256']!=_hash(binding)
                or type(event['sequence']) is not int or event['sequence']!=index or type(event['utc_ns']) is not int or event['utc_ns']<=0
                or event['previous_hash']!=state['last_event_hash'] or event['event_hash']!=_hash({k:v for k,v in event.items() if k!='event_hash'})
                or not witness.exists() or _json(_path(witness).read_bytes())!=event):
            raise ValueError('Protected journal schema/hash/source/sequence/witness changed')
        _transition(state,event,binding)
    if state['phase'] not in (None,'prepared'):_claims(binding,required=True)
    if binding_path.read_bytes()!=binding_raw or journal.read_bytes()!=raw or binding['source_sha256']!=_source():
        raise ValueError('Protected history/source changed during inspection')
    if state['active'] is not None:state['execution_cost_complete']=False
    state['committed_queries'].sort()
    return dict(state,binding=copy.deepcopy(binding),journal_sha256=hashlib.sha256(raw).hexdigest(),**_FALSE,
        scope='Metadata history only; predictions not numerically authenticated and no protected data access authorized')


def _append_unlocked(root,binding,state,*,kind,details,actor):
    event=dict(schema='research/v2-protected-journal-event',source_sha256=binding['source_sha256'],protocol_sha256=binding['protocol_sha256'],
        genesis_sha256=binding['genesis_sha256'],binding_sha256=_hash(binding),sequence=state['event_count'],kind=kind,actor=actor,utc_ns=time.time_ns(),
        details=copy.deepcopy(details),previous_hash=state['last_event_hash'])
    event['event_hash']=_hash(event);_transition(copy.deepcopy(state),event,binding)
    if kind=='owners_ready':_claims(binding,required=True)
    witness=_path(binding['owner_root'])/'histories'/binding['genesis_sha256']/f"{state['event_count']:08}.json"
    _write_new(witness,event)
    with _path(root/'events.jsonl').open('ab') as stream:
        stream.write((_canonical(event)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    return copy.deepcopy(event)


def _append_metadata_event(directory,*,kind,details,actor=None,_initial=False):
    root=_path(directory)
    with _lock(root/'journal.lock'):
        state=_read_metadata_study(root,_allow_initial=_initial)
        return _append_unlocked(root,state['binding'],state,kind=kind,details=details,actor=_ACTORS.get(kind) if actor is None else actor)


def _actual_protocol(reference,kwargs):
    from .practical_T_protocol_v2 import load_practical_T_protocol
    return load_practical_T_protocol(reference,**kwargs)


def _admitted_metadata(protocol):
    payload=protocol.payload;assets=[]
    for name,item in payload['data'].items():
        _,raw=_reference(item['manifest_reference']);manifest=_json(raw)
        assets.append(dict(dataset=name,raw_sha256=manifest['raw_sha256'],target=manifest['semantic_schema']['target']))
    prediction=payload['prediction_manifest']
    return dict(protocol_reference=protocol.reference,source_sha256=payload['source_sha256'],
        prediction_manifest_sha256=prediction['manifest_hash'],queries=prediction['queries'],assets=assets)


def _protected_storage_paths(payload,*,proof_roots,protocol_path):
    """Pure path inventory; contributes no input/source/native authority."""
    protected=[*proof_roots,protocol_path]
    protected.extend(item['root'] for item in payload['data'].values())
    protected.extend(query['model_binding']['snapshot_root'] for query in payload['prediction_manifest']['queries'])
    protected.extend(payload[k]['path'] for k in ('models_reference','preT_costs_reference','readiness_reference',
        'global_barrier_reference','practical_plan_reference','procedure_reference'))
    return protected


def _public_locations(directory,protocol):
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    payload=protocol.payload;q=payload['qualification_reference'];registry=load_native_qualification_registry(q)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=q)
    # The actual protocol loader already authenticated these costs, including every proof root.
    _,cost_raw=_reference(payload['preT_costs_reference']);costs=_json(cost_raw)['payload']
    protected=_protected_storage_paths(payload,proof_roots=[*roots['working_roots'],*roots['proof_roots'],*costs['proof_roots']],
        protocol_path=protocol.reference['path'])
    return _check_storage(directory,_ROOT/'artifacts/research-v2/protected-owners',protected_paths=protected)


def prepare_protected_study(reference,*,directory,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    """Actual protocol first; claims assets without reading any protected role."""
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(reference,kwargs);root,owners=_public_locations(directory,protocol)
    _prepare_metadata_study(directory=root,owner_root=owners,**_admitted_metadata(protocol))
    _claim_metadata_owners(root)
    return _read_metadata_study(root)


def inspect_protected_study(reference,*,directory,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(reference,kwargs);root,owners=_public_locations(directory,protocol)
    metadata=_admitted_metadata(protocol);out=_read_metadata_study(root);binding=out['binding']
    expected=dict(source_sha256=metadata['source_sha256'],protocol_sha256=metadata['protocol_reference']['sha256'],
        prediction_manifest_sha256=metadata['prediction_manifest_sha256'],query_ids=sorted(q['query_id'] for q in metadata['queries']),
        assets=_assets(metadata['assets']),owner_root=str(owners))
    if any(binding[k]!=v for k,v in expected.items()):raise ValueError('Protected study differs from actual frozen protocol/owner storage')
    return out
