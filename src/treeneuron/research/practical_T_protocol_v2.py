"""Bind final practical prediction identities; label access is a later stage.

The public protocol requires actual complete model, pre-T cost and readiness
loaders. A frozen protocol is not a stand-alone label-reading capability.
"""
import copy
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

from .deployment_worker_v2 import _source,_digest
from .execution_clock_v2 import _read_reference


_VIEWS=('native_quality','600000000000','1800000000000','7200000000000')
_FALSE=dict(native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False)


def _prediction_manifest(*,practical_plan,expected_practical_plan_hash,models,primary_models,feature_bindings):
    """Untrusted metadata compiler only: no arrays, models or gates are read."""
    from treeneuron.config import canonical_hash
    from .qualified_practical_models_v2 import inspect_practical_model_inventory
    from .primary_search_worker_v2 import _reference
    inspect_practical_model_inventory(practical_plan=practical_plan,expected_practical_plan_hash=expected_practical_plan_hash,
        models=models,primary_models=primary_models)
    expected={f"{r['dataset']}/outer-{r['outer_fold']}" for r in models}
    if type(feature_bindings) is not dict or set(feature_bindings)!=expected:
        raise ValueError('Exact thirty frozen T feature roles required')
    for row in models:
        key=f"{row['dataset']}/outer-{row['outer_fold']}";feature=feature_bindings[key]
        fields={'dataset','outer_fold','manifest_hash','split_hash','feature_reference'}
        if (type(feature) is not dict or set(feature)!=fields or any(type(feature[k]) is not type(row[k]) or feature[k]!=row[k]
                for k in ('dataset','outer_fold','manifest_hash','split_hash'))):
            raise ValueError('Prediction cannot substitute a dataset/split or label role')
        path=_reference(feature['feature_reference'])
        if path.name!='T-features.npz' or path.parent.name!=f"outer-{row['outer_fold']}":
            raise ValueError('Prediction query must be the explicit T feature archive')
    queries={};bindings=[];unavailable=0
    for row in models:
        feature=feature_bindings[f"{row['dataset']}/outer-{row['outer_fold']}"]
        for name in _VIEWS:
            view=row['budget_view_models'][name];query_id=None
            if view['model_binding'] is not None:
                query=dict(job_id=row['job_id'],fit_seed=row['fit_seed'],dataset=row['dataset'],outer_fold=row['outer_fold'],family_id=row['family_id'],
                    split_hash=row['split_hash'],manifest_hash=row['manifest_hash'],query_role='T_features',
                    feature_reference=copy.deepcopy(feature['feature_reference']),model_binding=copy.deepcopy(view['model_binding']))
                model=view['model_binding']
                identity={k:v for k,v in query.items() if k!='model_binding'}
                identity['model_content']=dict(attempt_index=model['attempt_index'],generation_sha256=model['generation_sha256'],
                    snapshot_inventory_sha256=model['snapshot_inventory_sha256'],selected_metadata_sha256=model['selected_metadata']['sha256'],
                    selected_metadata_name=Path(model['selected_metadata']['path']).name)
                query_id=canonical_hash(identity)
                queries.setdefault(query_id,dict(query_id=query_id,content_identity=identity,**query))
            else:unavailable+=1
            bindings.append(dict(job_id=row['job_id'],dataset=row['dataset'],outer_fold=row['outer_fold'],family_id=row['family_id'],
                search_repeat=row['search_repeat'],view=name,status=view['status'],deadline_ns=view['deadline_ns'],query_id=query_id))
    result=dict(schema='research/v2-protected-prediction-metadata',prediction_count=len(queries),planned_view_count=len(bindings),
        unavailable_views=unavailable,queries=[queries[k] for k in sorted(queries)],view_bindings=bindings,
        deduplication='Same frozen job/fit-seed/dataset/split/family/features/attempt and complete generation/snapshot/selected-metadata byte identities',**_FALSE,
        scope='Supplied model/view/split metadata only; no protected prediction or label access authorized')
    result['manifest_hash']=canonical_hash(result)
    return result


def _load_readiness(reference,qualification_reference):
    try:module=importlib.import_module('treeneuron.research.production_readiness_v2')
    except ModuleNotFoundError as error:
        raise ValueError('Required production readiness gate is not implemented/available; protected freeze remains unavailable') from error
    loader=getattr(module,'load_production_readiness',None)
    if not callable(loader):raise ValueError('Required production readiness loader is not implemented; no protected freeze')
    return loader(reference,qualification_reference=qualification_reference)


def _actual_inputs(*,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    from .qualified_practical_models_v2 import load_qualified_practical_models
    models=load_qualified_practical_models(models_reference,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference)
    from .practical_preT_costs_v2 import load_practical_preT_costs
    costs=load_practical_preT_costs(preT_costs_reference,models_reference=models_reference,
        global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference)
    readiness=_load_readiness(readiness_reference,qualification_reference)
    p=models.payload;c=costs.payload;r=readiness.payload
    if (c.get('preT_training_execution_cost_authenticated') is not True or c.get('models_reference')!=models_reference
            or r.get('production_source_ready') is not True or r.get('source_sha256')!=p['source_sha256']
            or r.get('qualification_reference')!=qualification_reference or r.get('protected_access_authorized') is not False
            or c.get('source_sha256')!=p['source_sha256']):
        raise ValueError('Actual complete source/models/training-cost/readiness gates must agree')
    return models,costs,readiness


def _data_bindings(prepared):
    """Read only genuine bound manifest metadata; never decode role archives."""
    from treeneuron.config import canonical_hash
    from .primary_search_worker_v2 import _path
    data={};features={}
    for name,item in prepared.items():
        root=_path(item['root']);ref=dict(path=str(root/'manifest.json'),sha256=item['manifest_file_sha256'])
        _,_,manifest=_read_reference(ref)
        if (manifest.get('dataset')!=name or manifest.get('schema')!='research/v2-sealed-transfer'
                or canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})!=manifest.get('manifest_hash')):
            raise ValueError('Actual sealed manifest metadata changed')
        folds=[]
        for fold,split in enumerate(manifest['splits']):
            roles={}
            for role,relative in [('F',f'outer-{fold}/F.npz'),('G',f'outer-{fold}/G.npz'),
                    ('T_features',f'outer-{fold}/T-features.npz'),('T_labels',f'protected/outer-{fold}/T-labels.npz')]:
                record=manifest['files'][relative]
                if not _digest(record['sha256']) or type(record['bytes']) is not int or record['bytes']<=0:
                    raise ValueError('Exact sealed role byte commitments required')
                roles[role]=dict(reference=dict(path=str(root/relative),sha256=record['sha256']),bytes=record['bytes'])
            folds.append(dict(outer_fold=fold,split_hash=split['split_hash'],primary=split['primary'],roles=roles))
            features[f'{name}/outer-{fold}']=dict(dataset=name,outer_fold=fold,manifest_hash=manifest['manifest_hash'],
                split_hash=split['split_hash'],feature_reference=roles['T_features']['reference'])
        data[name]=dict(root=str(root),manifest_reference=ref,manifest_hash=manifest['manifest_hash'],
            feature_names=manifest['feature_names'],categorical_features=manifest['categorical_features'],folds=folds)
    return data,features


def _assemble(*,inputs,references,created_utc_ns):
    from treeneuron.config import canonical_hash,canonical_json
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    from .practical_global_freeze_v2 import load_global_primary_barrier
    from .practical_procedure_v2 import load_practical_procedure
    models,costs,readiness=inputs;payload=models.payload
    if type(created_utc_ns) is not int or created_utc_ns<=0:raise ValueError('Exact final protocol publication time required')
    barrier=load_global_primary_barrier(references['global_barrier_reference'],qualification_reference=references['qualification_reference'])
    plan=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    procedure=load_practical_procedure(payload['procedure_reference'],practical_plan=plan,
        expected_practical_plan_hash=payload['expected_practical_plan_hash'],support_reference=payload['support_reference'])
    data,features=_data_bindings(barrier.payload['prepared_datasets'])
    predictions=_prediction_manifest(practical_plan=plan,expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        models=payload['models'],primary_models=barrier.payload['primary_models'],feature_bindings=features)
    result=dict(schema='research/v2-final-practical-prediction-payload',created_utc_ns=created_utc_ns,
        source_sha256=payload['source_sha256'],**copy.deepcopy(references),
        practical_plan_reference=payload['practical_plan_reference'],procedure_reference=payload['procedure_reference'],
        presearch_declaration_hash=procedure['declaration_hash'],policy=copy.deepcopy(procedure['policy']),
        model_payload_sha256=canonical_hash(payload),preT_cost_payload_sha256=canonical_hash(costs.payload),
        readiness_payload_sha256=canonical_hash(readiness.payload),data=data,prediction_manifest=predictions,
        frozen_primary_champions=copy.deepcopy(barrier.payload['primary_champions']),
        protected_prediction_plan_frozen=True,all_prediction_commits_required_before_labels=True,
        protected_access_authorized=False,T_accessed=False,whole_training_procedure_cost_authenticated=False,cost_claim_authorized=False,
        scope='Actual source/models/pre-T costs/readiness and unchanged presearch policy bound. Access still requires the once-only evaluator journal; protocol objects and flags alone never open labels.')
    if _source()!=payload['source_sha256']:raise ValueError('Final protocol source changed during assembly')
    return result


def _check_protocol_output(output,*,protected_paths):
    from .primary_search_worker_v2 import _path,_ROOT
    protected=[*(Path(p) for p in protected_paths),Path(sys.prefix).resolve(),Path(sys.executable).resolve().parent]
    protected.extend((_ROOT/n).resolve() for n in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))
    path=_path(str(output))
    if path.exists() or path.suffix!='.json' or any(path.is_relative_to(p) or p.is_relative_to(path) for p in protected):
        raise ValueError('Fresh final protocol JSON must preserve every source/data/runtime/proof input')
    return path


def _publication_location(output,*,payload,inputs):
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    q=payload['qualification_reference'];registry=load_native_qualification_registry(q)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=q)
    protected=[*roots['working_roots'],*roots['proof_roots'],*inputs[1].payload['proof_roots']]
    protected.extend(row['search_directory'] for row in inputs[0].payload['models'])
    protected.extend(payload[k]['path'] for k in ('models_reference','global_barrier_reference','preT_costs_reference','readiness_reference',
        'practical_plan_reference','procedure_reference'))
    return _check_protocol_output(output,protected_paths=protected)


@dataclass(frozen=True)
class PracticalTProtocol:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


def seal_practical_T_protocol(*,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference,output):
    refs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    inputs=_actual_inputs(**refs);payload=_assemble(inputs=inputs,references=refs,created_utc_ns=time.time_ns())
    path=_publication_location(output,payload=payload,inputs=inputs)
    from treeneuron.config import canonical_hash,canonical_json
    from .ledger import write_new_json
    if any(canonical_json(old.payload)!=canonical_json(new.payload) for old,new in zip(inputs,_actual_inputs(**refs),strict=True)):
        raise ValueError('Actual final protocol gates changed before publication')
    write_new_json(path,dict(schema='research/v2-final-practical-prediction-protocol',payload=payload,payload_hash=canonical_hash(payload)))
    return PracticalTProtocol(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()),canonical_json(payload))


def load_practical_T_protocol(reference,*,models_reference,global_barrier_reference,qualification_reference,preT_costs_reference,readiness_reference):
    refs=dict(models_reference=models_reference,global_barrier_reference=global_barrier_reference,
        qualification_reference=qualification_reference,preT_costs_reference=preT_costs_reference,readiness_reference=readiness_reference)
    inputs=_actual_inputs(**refs)
    _,_,value=_read_reference(reference)
    from treeneuron.config import canonical_hash,canonical_json
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'}
            or value['schema']!='research/v2-final-practical-prediction-protocol' or type(value['payload']) is not dict
            or canonical_hash(value['payload'])!=value['payload_hash']):raise ValueError('Exact immutable final practical protocol required')
    saved=value['payload']
    try:actual=_assemble(inputs=inputs,references=refs,created_utc_ns=saved['created_utc_ns'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed final practical protocol payload') from error
    if canonical_json(saved)!=canonical_json(actual):raise ValueError('Final policy/model/data/source/cost/readiness binding changed')
    _read_reference(reference)
    return PracticalTProtocol(copy.deepcopy(reference),canonical_json(actual))
