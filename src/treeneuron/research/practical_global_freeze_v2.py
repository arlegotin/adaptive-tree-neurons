"""Freeze actual primary models/procedure before secondary development access.

This is not the later protected evaluator or a cost/confirmation authorization.
Public loading always repeats actual native and primary-model validation.
"""
import copy
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from .validation_invocation_v2 import _within_validation, _load_once, _recheck_validation


_DATASETS=('airfoil','concrete','ccpp','parkinsons','bike','energy')
_VIEWS=('native_quality','600000000000','1800000000000','7200000000000')


def inspect_primary_champion_choices(models):
    """V-only complete-portfolio arithmetic on untrusted supplied metadata."""
    if type(models) is not list or len(models)!=168:raise ValueError('Exactly168 primary model rows required')
    indexed={}
    for row in models:
        if type(row) is not dict:raise ValueError('Explicit primary model rows required')
        dataset=row.get('dataset');repeat=row.get('search_repeat');family=row.get('family_id');fold=row.get('outer_fold')
        if (type(dataset) is not str or dataset not in _DATASETS or type(repeat) is not int or repeat not in (0,1)
                or type(family) is not str or family not in tuple(f'P{i:02}' for i in range(14))
                or type(fold) is not int or fold!=(4 if dataset=='bike' else 0)
                or row.get('job_id')!=f'{dataset}/outer-{fold}/repeat-{repeat}/{family}'):
            raise ValueError('Exact primary dataset/fold/family/R2 identity required')
        key=(dataset,repeat,family);views=row.get('budget_view_models')
        if key in indexed or type(views) is not dict or set(views)!=set(_VIEWS):
            raise ValueError('Complete unique primary model view inventory required')
        for name,view in views.items():
            if (type(view) is not dict or view.get('deadline_ns')!=(None if name=='native_quality' else int(name))):
                raise ValueError('Budget model view deadline differs')
            if view.get('status')=='retained_model_bytes':
                score=view.get('V_score')
                if (type(score) not in (int,float) or not math.isfinite(score) or score<0
                        or type(view.get('model_binding')) is not dict):raise ValueError('Finite V score and supplied model binding required')
            elif (view.get('status') not in ('unavailable_at_prefix','unavailable_no_eligible_model')
                    or view.get('V_score') is not None or view.get('model_binding') is not None):
                raise ValueError('Unavailable views cannot carry imputed scores/models')
        indexed[key]=row
    choices={}
    for dataset in _DATASETS:
        choices[dataset]={}
        for repeat in (0,1):
            choices[dataset][str(repeat)]={}
            for view in _VIEWS:
                rows=[indexed[dataset,repeat,f'P{i:02}'] for i in range(2,14)]
                missing=[r['family_id'] for r in rows if r['budget_view_models'][view]['status']!='retained_model_bytes']
                if missing:
                    selected=dict(status='unavailable_portfolio',family_id=None,job_id=None,V_score=None,model_binding=None,missing_families=missing)
                else:
                    best=min(r['budget_view_models'][view]['V_score'] for r in rows);tolerance=1e-12*max(1.,abs(best))
                    winner=min((r for r in rows if r['budget_view_models'][view]['V_score']<=best+tolerance),key=lambda r:r['family_id'])
                    selected=dict(status='V_selected_champion',family_id=winner['family_id'],job_id=winner['job_id'],
                        V_score=winner['budget_view_models'][view]['V_score'],model_binding=copy.deepcopy(winner['budget_view_models'][view]['model_binding']),missing_families=[])
                choices[dataset][str(repeat)][view]=selected
    return dict(schema='research/v2-primary-champion-choice-metadata',choices=choices,
        native_model_generation_authenticated=False,protected_access_authorized=False,
        scope='Complete supplied primary V/model-view metadata only; both repeats retained, P02-P13 portfolio fixed, no T selection or native authentication')


def _model_bytes(payload):
    from .qualified_primary_models_v2 import inspect_selected_snapshot_binding,inspect_search_inventory_binding
    seen=set()
    for row in payload['models']:
        inspect_search_inventory_binding(search_directory=Path(row['search_directory']),expected_inventory_sha256=row['search_inventory_sha256'])
        for view in row['budget_view_models'].values():
            model=view['model_binding']
            if model is None:continue
            key=(model['snapshot_root'],model['snapshot_inventory_sha256'])
            if key in seen:continue
            inspect_selected_snapshot_binding(selected_model=model['selected_metadata'],expected_metadata=model['selected_metadata'],
                snapshot_root=Path(model['snapshot_root']),expected_inventory_sha256=model['snapshot_inventory_sha256'])
            seen.add(key)


def _assemble(*,registry,qualification_reference,practical_plan_reference,expected_practical_plan_hash,
        support_reference,procedure_reference,qualified_primary_models_reference,created_utc_ns,prefixes=None):
    from treeneuron.config import canonical_json
    from .primary_selection_v2 import _ref,_primary
    from .workload import PracticalPlan
    from .practical_procedure_v2 import load_practical_procedure
    from .qualified_primary_models_v2 import load_qualified_primary_models
    from .transfer_access_v2 import frozen_primary_access_prefix
    from .native_anchor_v2 import _closed_reference
    from .deployment_worker_v2 import _source
    if type(created_utc_ns) is not int or created_utc_ns<=0:raise ValueError('Immutable global barrier publication time required')
    plan=PracticalPlan(canonical_json(_ref(practical_plan_reference,{},decode=True)));value,jobs=_primary(plan,expected_practical_plan_hash)
    procedure=load_practical_procedure(procedure_reference,practical_plan=plan,expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference)
    _ref(qualified_primary_models_reference,{})
    models=load_qualified_primary_models(practical_plan=plan,expected_practical_plan_hash=expected_practical_plan_hash,
        support_reference=support_reference,qualification_reference=qualification_reference,
        path=qualified_primary_models_reference['path'],expected_file_sha256=qualified_primary_models_reference['sha256']).payload
    if (models.get('qualified_primary_168_models_complete') is not True or models.get('presearch_procedure_semantics_authenticated') is not True
            or models.get('global_procedure')!=procedure_reference or models.get('plan_hash')!=expected_practical_plan_hash
            or models.get('source_sha256')!=value['source_sha256'] or registry['identity']['scientific_source_sha256']!=value['source_sha256']):
        raise ValueError('Actual complete primary models and same pre-search procedure/source required')
    champions=inspect_primary_champion_choices(models['models'])
    if prefixes is not None and (type(prefixes) is not dict or set(prefixes)!=set(_DATASETS)):
        raise ValueError('All six exact frozen primary access prefixes required')
    # Native/model audits above may append primary V events. Never perform those
    # label-reading audits while holding their own shared access-log leases.
    with ExitStack() as stack:
        fixed={}
        for dataset in _DATASETS:
            entry=registry['preparation']['datasets'][dataset];root=Path(registry['inputs']['prepared_roots'][dataset])
            fixed[dataset]=copy.deepcopy(stack.enter_context(frozen_primary_access_prefix(root,
                expected_manifest_sha256=entry['manifest_file_sha256'],prefix=None if prefixes is None else prefixes[dataset])))
            if fixed[dataset]['prefix_event_count']==0 or fixed[dataset]['access_log_absent_at_freeze']:
                raise ValueError('Actual qualified primary roles must retain recorded API access before global freeze')
        _model_bytes(models);_closed_reference(qualification_reference)
        _ref(qualified_primary_models_reference,{})
        if (_source()!=value['source_sha256'] or load_practical_procedure(procedure_reference,practical_plan=plan,
                expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference)!=procedure):
            raise ValueError('Global barrier source/procedure changed during publication')
        payload=dict(schema='research/v2-global-primary-barrier-payload',created_utc_ns=created_utc_ns,
            practical_plan_reference=copy.deepcopy(practical_plan_reference),expected_practical_plan_hash=expected_practical_plan_hash,
            support_reference=copy.deepcopy(support_reference),procedure_reference=copy.deepcopy(procedure_reference),
            qualification_reference=copy.deepcopy(qualification_reference),qualified_primary_models_reference=copy.deepcopy(qualified_primary_models_reference),
            source_sha256=value['source_sha256'],primary_models=copy.deepcopy(models['models']),primary_champions=champions,
            prepared_datasets={name:dict(root=str(Path(registry['inputs']['prepared_roots'][name])),
                manifest_file_sha256=registry['preparation']['datasets'][name]['manifest_file_sha256']) for name in _DATASETS},
            frozen_primary_access_prefixes=fixed,global_procedure_semantics_authenticated=True,
            qualified_primary_models_authenticated=True,secondary_development_access_authorized=True,
            secondary_native_workload_qualified=False,protected_access_authorized=False,protected_evaluation_authorized=False,
            T_accessed=False,whole_training_procedure_cost_authenticated=False,cost_claim_authorized=False,
            scope='Actual all168 primary models and all budget views plus unchanged global procedure frozen. Secondary F/G/V release only; new workload qualification, full post-secondary model inventory, protected evaluator and cost/readiness gates remain required.')
    return payload


def _publication_location(output,registry,qualification_reference,model_payload,references):
    from .native_qualification_registry_v2 import native_registry_storage_roots
    from .method_worker import _ROOT
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    protected=[*(Path(p) for p in roots['working_roots']+roots['proof_roots']),Path(sys.prefix).resolve(),
        *((_ROOT/n).resolve() for n in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws')),
        *(Path(r['path']) for r in references),
        *(Path(row[k]) for row in model_payload['models'] for k in ('snapshot_root','search_directory'))]
    protected.extend(Path(view['model_binding']['snapshot_root']) for row in model_payload['models']
        for view in row['budget_view_models'].values() if view['model_binding'] is not None)
    output=Path(output)
    if (not output.is_absolute() or output.resolve()!=output or output.is_symlink() or output.exists() or output.suffix!='.json'
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected)):
        raise ValueError('Fresh global barrier must preserve all qualified/source/model/search/input paths')
    return output


def _publish_after_validating(output,prepare):
    """Publish only after all validation contexts have completed their exits."""
    from treeneuron.config import canonical_hash
    from .ledger import write_new_json
    payload=prepare()
    write_new_json(output,dict(schema='research/v2-global-primary-barrier',payload=payload,payload_hash=canonical_hash(payload)))
    return payload


@dataclass(frozen=True)
class GlobalPrimaryBarrier:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


@_within_validation
def seal_global_primary_barrier(*,practical_plan_reference,expected_practical_plan_hash,support_reference,
        procedure_reference,qualification_reference,qualified_primary_models_reference,output):
    from .native_qualification_registry_v2 import load_native_qualification_registry
    registry=load_native_qualification_registry(qualification_reference)
    def prepare():
        payload=_assemble(registry=registry,qualification_reference=qualification_reference,
            practical_plan_reference=practical_plan_reference,expected_practical_plan_hash=expected_practical_plan_hash,
            support_reference=support_reference,procedure_reference=procedure_reference,
            qualified_primary_models_reference=qualified_primary_models_reference,created_utc_ns=time.time_ns())
        _publication_location(output,registry,qualification_reference,dict(models=payload['primary_models']),
            [practical_plan_reference,support_reference,procedure_reference,qualified_primary_models_reference])
        from .deployment_worker_v2 import _source
        if _source()!=payload['source_sha256']:raise ValueError('Source changed after validation contexts closed')
        _recheck_validation()
        return payload
    payload=_publish_after_validating(output,prepare)
    from treeneuron.config import canonical_json
    return GlobalPrimaryBarrier(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest()),canonical_json(payload))


def _guard_global_context(value):
    """Recheck retained bytes/prefixes; semantic reconstruction already ran."""
    from .primary_selection_v2 import _ref
    from .native_anchor_v2 import _closed_reference
    from .transfer_access_v2 import frozen_primary_access_prefix
    payload = value['payload']
    _ref(value['reference'], {})
    _closed_reference(payload['qualification_reference'])
    for key in ('practical_plan_reference', 'support_reference', 'procedure_reference', 'qualified_primary_models_reference'):
        _ref(payload[key], {})
    _model_bytes(dict(models=payload['primary_models']))
    for dataset, entry in payload['prepared_datasets'].items():
        with frozen_primary_access_prefix(Path(entry['root']),
                expected_manifest_sha256=entry['manifest_file_sha256'],
                prefix=payload['frozen_primary_access_prefixes'][dataset]):
            pass


@_within_validation
def load_global_primary_barrier(reference,*,qualification_reference):
    from .primary_selection_v2 import _ref
    from treeneuron.config import canonical_json
    reference = copy.deepcopy(reference); qualification_reference = copy.deepcopy(qualification_reference)
    def load():
        barrier = _load_global_primary_barrier(reference, qualification_reference=qualification_reference)
        return dict(reference=barrier.reference, payload=barrier.payload)
    value = _load_once('global_primary', dict(reference=reference, qualification_reference=qualification_reference),
        load, guard=_guard_global_context, check=lambda value: _ref(reference, {}))
    return GlobalPrimaryBarrier(copy.deepcopy(value['reference']), canonical_json(value['payload']))


def _load_global_primary_barrier(reference,*,qualification_reference):
    from .native_qualification_registry_v2 import load_native_qualification_registry
    registry=load_native_qualification_registry(qualification_reference)
    from treeneuron.config import canonical_json,canonical_hash
    from .primary_selection_v2 import _ref
    value=_ref(reference,{},decode=True)
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'} or value['schema']!='research/v2-global-primary-barrier'
            or type(value['payload']) is not dict or canonical_hash(value['payload'])!=value['payload_hash']
            or value['payload'].get('qualification_reference')!=qualification_reference):
        raise ValueError('Exact global primary barrier envelope and actual qualification required')
    saved=value['payload']
    try:
        rebuilt=_assemble(registry=registry,qualification_reference=qualification_reference,
            **{k:saved[k] for k in ('practical_plan_reference','expected_practical_plan_hash','support_reference','procedure_reference',
                'qualified_primary_models_reference','created_utc_ns')},prefixes=saved['frozen_primary_access_prefixes'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed global barrier payload') from error
    if canonical_json(saved)!=canonical_json(rebuilt):raise ValueError('Global primary barrier semantics or authority changed')
    _ref(reference,{})
    return GlobalPrimaryBarrier(copy.deepcopy(reference),canonical_json(rebuilt))
