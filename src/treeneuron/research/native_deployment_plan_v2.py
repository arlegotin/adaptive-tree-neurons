"""Bind primary original-model timing requests before protected labels.

The pure compiler grants no native/execution authority. Public routes require
the actual complete protocol, F roles, snapshots and witnessed owner history.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .deployment_worker_v2 import _source,_digest,_json
from .protected_journal_v2 import _path,_reference,_hash,_canonical,_write_new,_read_metadata_study,_append_metadata_event

_NAMES=('airfoil','concrete','ccpp','parkinsons','bike','energy')
_FAMILIES=tuple(f'P{i:02}' for i in range(14))
_FALSE=dict(native_models_authenticated=False,execution_authorized=False,original_F_role_authenticated=False,
    quality_precision_equivalence_authenticated=False,dedicated_timing_qualified=False,cost_claim_authorized=False,T_accessed=False)


def _primary_queries(payload):
    if payload.get('source_sha256')!=_source():raise ValueError('Exact current-source timing metadata required')
    rows=payload['prediction_manifest']['queries'];queries={q['query_id']:q for q in rows}
    if len(queries)!=len(rows):raise ValueError('Duplicate frozen query identity')
    indexed={}
    for v in payload['prediction_manifest']['view_bindings']:
        if v['view']!='native_quality' or v['dataset'] not in _NAMES or v['outer_fold']!=(4 if v['dataset']=='bike' else 0):continue
        key=(v['dataset'],v['search_repeat'],v['family_id'])
        q=queries.get(v['query_id'])
        if (key in indexed or type(v['search_repeat']) is not int or v['search_repeat'] not in (0,1) or v['family_id'] not in _FAMILIES
                or v['status']!='retained_model_bytes' or v['deadline_ns'] is not None or q is None
                or any(q[k]!=v[k] for k in ('dataset','outer_fold','family_id','job_id'))
                or v['job_id']!=f"{v['dataset']}/outer-{v['outer_fold']}/repeat-{v['search_repeat']}/{v['family_id']}"
                or type(q.get('model_binding')) is not dict):raise ValueError('Complete unique original primary native-quality model binding required')
        indexed[key]=q
    expected={(n,r,f) for n in _NAMES for r in (0,1) for f in _FAMILIES}
    if set(indexed)!=expected:raise ValueError('All168 original primary native-quality models required; unavailable models are not losses')
    return indexed


def _compile_timing_schedule(payload,reference,requests):
    """Supplied metadata only; no source arrays/models/qualification are opened."""
    import numpy as np
    from .seeds import seed64
    indexed=_primary_queries(payload)
    if (type(reference) is not dict or set(reference)!={'path','sha256'} or not _digest(reference['sha256'])
            or type(requests) is not dict or set(requests)!={q['job_id'] for q in indexed.values()}):
        raise ValueError('Exact frozen protocol and168 request references required')
    _path(reference['path']);models=[]
    for name in _NAMES:
        for r in (0,1):
            for family in _FAMILIES:
                q=indexed[name,r,family];ref=requests[q['job_id']]
                if type(ref) is not dict or set(ref)!={'path','sha256'} or not _digest(ref['sha256']):raise ValueError('Exact source-bound template reference required')
                _path(ref['path']);models.append(dict(job_id=q['job_id'],query_id=q['query_id'],dataset=name,outer_fold=q['outer_fold'],
                    search_repeat=r,family_id=family,model_binding=copy.deepcopy(q['model_binding']),request_reference=copy.deepcopy(ref)))
    comparisons=[];jobs=[];previous=None
    for name in _NAMES:
        fold=4 if name=='bike' else 0
        for r in (0,1):
            selected=indexed[name,r,'P00']['job_id']
            for family in _FAMILIES[1:]:
                baseline=indexed[name,r,family]['job_id'];identity=f'{name}/outer-{fold}/repeat-{r}/P00-vs-{family}'
                seed=seed64(2026101802,reference['sha256'],'deployment',name,str(r),family)
                coins=np.random.default_rng(seed).integers(0,2,size=30)
                orders=['selected_first' if int(c)==0 else 'baseline_first' for c in coins]
                comparisons.append(dict(comparison_id=identity,dataset=name,outer_fold=fold,search_repeat=r,baseline_family=family,
                    selected_job_id=selected,baseline_job_id=baseline,seed=seed,orders=orders,blocks=30,energy_descriptive=name=='energy'))
                for block,order in enumerate(orders):
                    roles=('selected','baseline') if order=='selected_first' else ('baseline','selected')
                    for role in roles:
                        model_job=selected if role=='selected' else baseline;job=f'{identity}/block-{block:02}/{role}'
                        jobs.append(dict(job_id=job,comparison_id=identity,dataset=name,outer_fold=fold,search_repeat=r,
                            block_id=block,role=role,order=order,model_job_id=model_job,request_reference=copy.deepcopy(requests[model_job]),
                            depends_on=[] if previous is None else [previous],dependency_policy='all_terminal'))
                        previous=job
    result=dict(schema='research/v2-native-timing-schedule-metadata',source_sha256=payload['source_sha256'],protocol_reference=copy.deepcopy(reference),
        models=models,comparisons=comparisons,jobs=jobs,worker_concurrency=1,models_copied_or_relocated=False,
        batch_sizes=[1,1024],warmup_calls=3,resident_calls_per_batch=32,**_FALSE,
        scope='Untrusted fixed timing metadata only; actual native/F-role/owner/request/resource and measurement gates remain required')
    result['schedule_hash']=_hash(result)
    return result


def _actual_protocol(reference,kwargs):
    from .protected_prediction_gate_v2 import _actual_protocol as load
    return load(reference,kwargs)


def _study(protocol,directory,*,prepare):
    from .protected_journal_v2 import (_public_locations,_prepare_metadata_study,_claim_metadata_owners)
    from .protected_prediction_queue_v2 import _admitted_metadata
    root,owners=_public_locations(directory,protocol)
    metadata=_admitted_metadata(protocol.payload,protocol.reference)
    if prepare:
        _prepare_metadata_study(directory=root,owner_root=owners,**metadata);_claim_metadata_owners(root)
    state=_read_metadata_study(root)
    # Reopening validates the exact binding without modifying existing history.
    _prepare_metadata_study(directory=root,owner_root=owners,**metadata)
    if state['binding']['owner_root']!=str(owners):raise ValueError('Timing plan substituted another owner archive')
    return root,owners,state


def _F_metadata(payload,name):
    item=payload['data'][name];index=4 if name=='bike' else 0
    folds=[f for f in item['folds'] if f['outer_fold']==index and f['primary'] is True]
    if len(folds)!=1:raise ValueError('One exact primary F role required for timing')
    fold=folds[0];record=fold['roles']['F']
    return dict(schema='research/v2-F-deployment-input',source_sha256=payload['source_sha256'],role='F',dataset=name,outer_fold=index,
        manifest_hash=item['manifest_hash'],split_hash=fold['split_hash'],F_reference=copy.deepcopy(record['reference']),F_bytes=record['bytes'],
        feature_schema=dict(columns=item['feature_names'],categorical=item['categorical_features']),temporal=name=='bike')


def _request(payload,q,projection):
    item=payload['data'][q['dataset']]
    return dict(schema='research/v2-original-snapshot-measurement-request',source_sha256=payload['source_sha256'],family_id=q['family_id'],
        logical_job_id=q['job_id'],logical_fit_seed=q['fit_seed'],feature_schema=dict(columns=item['feature_names'],categorical=item['categorical_features']),
        model_binding=copy.deepcopy(q['model_binding']),query_reference=copy.deepcopy(projection['query_reference']),
        query_provenance_reference=copy.deepcopy(projection['provenance_reference']),batch_sizes=[1,1024],warmup_calls=3,samples_per_batch=32)


def _ref(path):return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def _inspect_plan_files(saved,*,protocol,root,owners,state):
    """Full actual context is supplied only by public actual loaders."""
    from .deployment_query_v2 import inspect_F_workload_query
    from .native_deployment_worker_v2 import _checked_measurement_request
    fields={'schema','protocol_reference','source_sha256','study_directory','owner_root','genesis_sha256','projections','requests','schedule',
        'actual_frozen_model_and_F_bindings_checked','protected_access_authorized','execution_authorized','quality_precision_equivalence_authenticated',
        'dedicated_timing_qualified','cost_claim_authorized','T_accessed','scope'}
    if (type(saved) is not dict or set(saved)!=fields or saved['schema']!='research/v2-bound-native-deployment-plan'
            or saved['source_sha256']!=protocol.payload['source_sha256'] or saved['protocol_reference']!=protocol.reference
            or saved['study_directory']!=str(root) or saved['owner_root']!=str(owners) or saved['genesis_sha256']!=state['binding']['genesis_sha256']
            or saved['actual_frozen_model_and_F_bindings_checked'] is not True
            or any(saved[k] is not False for k in ('protected_access_authorized','execution_authorized','quality_precision_equivalence_authenticated',
                'dedicated_timing_qualified','cost_claim_authorized','T_accessed')) or set(saved['projections'])!=set(_NAMES)):
        raise ValueError('Exact actual source/owner/model/F bound timing plan required')
    projections={}
    for name,ref in saved['projections'].items():projections[name]=inspect_F_workload_query(ref,expected_metadata=_F_metadata(protocol.payload,name))
    indexed=_primary_queries(protocol.payload)
    if set(saved['requests'])!={q['job_id'] for q in indexed.values()}:raise ValueError('Every168 native model template required')
    for q in indexed.values():
        ref=saved['requests'][q['job_id']];value,*_=_checked_measurement_request(_path(ref['path']),ref['sha256'])
        if value!=_request(protocol.payload,q,projections[q['dataset']]):raise ValueError('Timing template changed its actual frozen model/F query/seed/config binding')
    actual=_compile_timing_schedule(protocol.payload,protocol.reference,saved['requests'])
    if saved['schedule']!=actual:raise ValueError('Timing schedule/order changed from its frozen protocol')
    return saved


@dataclass(frozen=True)
class NativeDeploymentPlan:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_native_deployment_plan(reference,*,directory,output,models_reference,global_barrier_reference,qualification_reference,
        preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(reference,kwargs);root,owners,state=_study(protocol,directory,prepare=True)
    from .ledger import StudyLease
    from .protected_prediction_gate_v2 import _publication_location
    from .deployment_query_v2 import build_F_workload_query
    from .native_deployment_worker_v2 import _checked_measurement_request
    output=_path(output);path=output/'deployment-plan.json'
    with StudyLease(root):
        state=_read_metadata_study(root)
        if state['deployment_plan_reference'] is not None:
            if state['deployment_plan_reference']['path']!=str(path):raise ValueError('Existing timing plan cannot be replaced or resampled')
            return load_native_deployment_plan(state['deployment_plan_reference'],protocol_reference=reference,directory=directory,**kwargs)
        if state['phase'] not in ('owners_ready','all_predictions_committed') or state['active'] is not None:
            raise ValueError('Timing plan must freeze before labels with no active query')
        _publication_location(path,protocol=protocol,inspection=dict(study_directory=str(root),owner_root=str(owners),invocation_references=[]))
        if output.exists():raise ValueError('Fresh immutable timing-plan output required; preserve incomplete attempts')
        indexed=_primary_queries(protocol.payload);output.mkdir(parents=True,exist_ok=False)
        projections={}
        for name in _NAMES:
            built=build_F_workload_query(_F_metadata(protocol.payload,name),output=output/'F-projections'/name)
            projections[name]=built
        requests={}
        for q in indexed.values():
            value=_request(protocol.payload,q,projections[q['dataset']]);target=output/'requests'/q['job_id']/'request.json'
            _write_new(target,value);requests[q['job_id']]=_ref(target);_checked_measurement_request(target,requests[q['job_id']]['sha256'])
        schedule=_compile_timing_schedule(protocol.payload,protocol.reference,requests)
        payload=dict(schema='research/v2-bound-native-deployment-plan',protocol_reference=protocol.reference,source_sha256=protocol.payload['source_sha256'],
            study_directory=str(root),owner_root=str(owners),genesis_sha256=state['binding']['genesis_sha256'],
            projections={n:p['manifest_reference'] for n,p in projections.items()},requests=requests,schedule=schedule,
            actual_frozen_model_and_F_bindings_checked=True,protected_access_authorized=False,execution_authorized=False,
            quality_precision_equivalence_authenticated=False,dedicated_timing_qualified=False,cost_claim_authorized=False,T_accessed=False,
            scope='Actual original native-model/F-query/source/owner binding only; pre-T witness and actual measurement/resource/cost/precision/quality qualification remain separate')
        _inspect_plan_files(payload,protocol=protocol,root=root,owners=owners,state=state)
        if _canonical(_actual_protocol(reference,kwargs).payload)!=_canonical(protocol.payload):raise ValueError('Actual native context changed before timing-plan freeze')
        for ref in requests.values():_reference(ref)
        for built in projections.values():
            for key in ('manifest_reference','query_reference','provenance_reference'):_reference(built[key])
        _write_new(path,dict(schema='research/v2-sealed-native-deployment-plan',payload=payload,payload_sha256=_hash(payload)))
        ref=_ref(path);_append_metadata_event(root,kind='deployment_plan_frozen',details=dict(plan_reference=ref))
    return NativeDeploymentPlan(ref,_canonical(payload))


def load_native_deployment_plan(reference,*,protocol_reference,directory,models_reference,global_barrier_reference,qualification_reference,
        preT_costs_reference,readiness_reference):
    kwargs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    protocol=_actual_protocol(protocol_reference,kwargs);root,owners,state=_study(protocol,directory,prepare=False)
    if state['deployment_plan_reference']!=reference:raise ValueError('Actual timing plan needs its immutable pre-label witness')
    _,raw=_reference(reference);saved=_json(raw)
    if (type(saved) is not dict or set(saved)!={'schema','payload','payload_sha256'} or saved['schema']!='research/v2-sealed-native-deployment-plan'
            or type(saved['payload']) is not dict or _hash(saved['payload'])!=saved['payload_sha256']):raise ValueError('Exact immutable timing plan required')
    value=_inspect_plan_files(saved['payload'],protocol=protocol,root=root,owners=owners,state=state)
    if _canonical(_actual_protocol(protocol_reference,kwargs).payload)!=_canonical(protocol.payload):raise ValueError('Actual native timing context changed')
    if _read_metadata_study(root)['deployment_plan_reference']!=reference:raise ValueError('Timing witness changed during readback')
    _reference(reference)
    return NativeDeploymentPlan(copy.deepcopy(reference),_canonical(value))
