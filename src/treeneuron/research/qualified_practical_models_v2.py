"""Actual all840 retained models, without protected or whole-cost authority."""
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from .validation_invocation_v2 import _within_validation, _recheck_validation


_VIEWS={'native_quality':None,'600000000000':600000000000,'1800000000000':1800000000000,'7200000000000':7200000000000}
_ROW={'job_id','dataset','outer_fold','family_id','search_repeat','fit_seed','split_hash','manifest_hash','V_score','winner_attempt_index',
    'selected_metadata','snapshot_root','snapshot_inventory_sha256','search_directory','search_inventory_sha256','generation_sha256',
    'budget_view_models','qualified_search_audit_sha256'}
_PROOF={'search_plan','search_result','journal_sha256','search_inventory_sha256','official_schedule_reference'}


def _score(value):return type(value) in (int,float) and math.isfinite(value) and value>=0


def _model_metadata(model,search):
    """Path/type metadata only; no file is opened or native authority inferred."""
    from .primary_search_worker_v2 import _reference,_path,_digest
    fields={'attempt_index','selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256'}
    if (type(model) is not dict or set(model)!=fields or type(model['attempt_index']) is not int or not 0<=model['attempt_index']<32
            or not _digest(model['snapshot_inventory_sha256']) or not _digest(model['generation_sha256'])):
        raise ValueError('Exact retained native model metadata required')
    path=_reference(model['selected_metadata']);root=_path(model['snapshot_root'])
    if path.parent!=root or path.name not in ('manifest.json','state.json') or not root.is_relative_to(search):
        raise ValueError('Retained model metadata must belong to its exact search snapshot')


def inspect_practical_model_inventory(*,practical_plan,expected_practical_plan_hash,models,primary_models):
    """Join untrusted identities only; public sealing separately checks bytes."""
    from treeneuron.config import canonical_json,canonical_hash
    from .primary_selection_v2 import _primary
    from .primary_search_worker_v2 import _path,_digest
    value,primary_jobs=_primary(practical_plan,expected_practical_plan_hash);jobs=value['jobs']
    if type(models) is not list or len(models)!=840 or type(primary_models) is not list or len(primary_models)!=168:
        raise ValueError('All840 models and the immutable168 primary rows required')
    expected={j['job_id']:j for j in jobs};indexed={}
    identity=('job_id','dataset','outer_fold','family_id','search_repeat','fit_seed','split_hash','manifest_hash')
    for row in models:
        if type(row) is not dict or set(row)!=_ROW or type(row['job_id']) is not str or row['job_id'] not in expected or row['job_id'] in indexed:
            raise ValueError('Complete unique fixed model rows required')
        job=expected[row['job_id']]
        if any(type(row[k]) is not type(job[k]) or row[k]!=job[k] for k in identity):
            raise ValueError('Model dataset/fold/repeat/seed/split/manifest identity changed')
        if (not _score(row['V_score']) or type(row['winner_attempt_index']) is not int or not 0<=row['winner_attempt_index']<32
                or any(not _digest(row[k]) for k in ('snapshot_inventory_sha256','search_inventory_sha256','generation_sha256','qualified_search_audit_sha256'))):
            raise ValueError('Exact finite native winner and evidence hashes required')
        search=_path(row['search_directory']);native=dict(attempt_index=row['winner_attempt_index'],
            **{k:row[k] for k in ('selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256')})
        _model_metadata(native,search)
        views=row['budget_view_models']
        if type(views) is not dict or set(views)!=set(_VIEWS):raise ValueError('All four fixed retained-model views required')
        for name,view in views.items():
            deadline=_VIEWS[name]
            if (type(view) is not dict or set(view)!={'status','deadline_ns','V_score','model_binding'}
                    or type(view['deadline_ns']) is not type(deadline) or view['deadline_ns']!=deadline):
                raise ValueError('Model view deadline or schema changed')
            if view['status']=='retained_model_bytes':
                if not _score(view['V_score']):raise ValueError('Retained model needs finite V risk')
                _model_metadata(view['model_binding'],search)
            elif deadline is None or view['status']!='unavailable_at_prefix' or view['V_score'] is not None or view['model_binding'] is not None:
                raise ValueError('Only fixed prefixes may be explicitly unavailable; no imputed loss/model')
        native_view=views['native_quality']
        if (native_view['V_score']!=row['V_score'] or canonical_json(native_view['model_binding'])!=canonical_json(native)):
            raise ValueError('Native view differs from its actual winner binding')
        indexed[row['job_id']]=row
    prior={}
    for row in primary_models:
        if type(row) is not dict or type(row.get('job_id')) is not str or row['job_id'] in prior:
            raise ValueError('Unique immutable primary rows required')
        prior[row['job_id']]=row
    if set(prior)!={j['job_id'] for j in primary_jobs} or any(canonical_json(indexed[name])!=canonical_json(row) for name,row in prior.items()):
        raise ValueError('Secondary results cannot replace or reselect frozen primary models')
    return dict(schema='research/v2-practical-model-identity-join',declared_models=840,primary_models=168,secondary_models=672,
        metadata_identity_sha256=canonical_hash(dict(models=[indexed[j['job_id']] for j in jobs])),
        native_qualification_authenticated=False,protected_access_authorized=False,T_accessed=False,
        scope='Supplied full model/view identities only; no file bytes, native/HPO/V qualification or protected authority')


def _frozen(global_barrier_reference,qualification_reference):
    from .practical_global_freeze_v2 import load_global_primary_barrier
    barrier=load_global_primary_barrier(global_barrier_reference,qualification_reference=qualification_reference)
    if (barrier.payload.get('secondary_development_access_authorized') is not True
            or barrier.payload.get('protected_access_authorized') is not False):
        raise ValueError('Actual frozen primary/global native models required before complete840 sealing')
    return barrier


def _bound_snapshots(row):
    from .qualified_primary_models_v2 import inspect_selected_snapshot_binding
    seen=set()
    for view in row['budget_view_models'].values():
        model=view['model_binding']
        if model is None:continue
        key=(model['snapshot_root'],model['snapshot_inventory_sha256'])
        if key in seen:continue
        inspect_selected_snapshot_binding(selected_model=model['selected_metadata'],expected_metadata=model['selected_metadata'],
            snapshot_root=Path(model['snapshot_root']),expected_inventory_sha256=model['snapshot_inventory_sha256'])
        seen.add(key)


def _assemble(*,barrier,global_barrier_reference,qualification_reference,secondary_search_proofs,created_utc_ns):
    from treeneuron.config import canonical_json,canonical_hash
    from .primary_selection_v2 import _primary,_ref,_digest
    from .workload import PracticalPlan
    from .secondary_native_search_v2 import audit_secondary_qualified_search
    from .qualified_primary_models_v2 import inspect_search_inventory_binding
    from .deployment_worker_v2 import _source
    payload=barrier.payload;practical=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    value,_=_primary(practical,payload['expected_practical_plan_hash']);jobs=[j for j in value['jobs'] if j['tranche']=='secondary']
    if (type(created_utc_ns) is not int or created_utc_ns<=0 or type(secondary_search_proofs) is not dict
            or set(secondary_search_proofs)!={j['job_id'] for j in jobs}):
        raise ValueError('Exact immutable timestamp and all672 secondary search proofs required')
    proofs=copy.deepcopy(secondary_search_proofs);models=copy.deepcopy(payload['primary_models'])
    for job in jobs:
        proof=proofs[job['job_id']]
        if (type(proof) is not dict or set(proof)!=_PROOF
                or not _digest(proof['journal_sha256']) or not _digest(proof['search_inventory_sha256'])):
            raise ValueError('Exact secondary search plan/result/journal/inventory/sampler commitments required')
        plan_path=Path(_ref(proof['search_plan'],{}));result_path=Path(_ref(proof['search_result'],{}));root=plan_path.parent
        if plan_path.name!='plan.json':raise ValueError('Exact native search plan location required')
        inspect_search_inventory_binding(search_directory=root,expected_inventory_sha256=proof['search_inventory_sha256'])
        actual=audit_secondary_qualified_search(barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
            job_id=job['job_id'],search_directory=root,expected_search_plan_sha256=proof['search_plan']['sha256'],result_path=result_path,
            expected_result_sha256=proof['search_result']['sha256'],expected_journal_sha256=proof['journal_sha256'],
            official_schedule_reference=proof['official_schedule_reference'])
        if (actual.get('job_id')!=job['job_id'] or actual.get('practical_plan_hash')!=payload['expected_practical_plan_hash']
                or actual.get('global_barrier_reference')!=global_barrier_reference or any(actual.get(k) is not True for k in
                    ('completed_search','complete_selected_model','HPO_journal_authenticated','all_attempt_native_evidence_authenticated',
                    'all_eligible_V_scores_authenticated','native_workload_qualified'))):
            raise ValueError('Every secondary model needs its actual complete native/HPO/V search')
        winner=actual['selected_model'];views=actual['search_view_models']['views'];native=views['native_quality']
        if canonical_json(native['model_binding'])!=canonical_json(winner):raise ValueError('Secondary native winner/view differs')
        row=dict(**{k:job[k] for k in ('job_id','dataset','outer_fold','family_id','search_repeat','fit_seed','split_hash','manifest_hash')},
            V_score=native['V_score'],winner_attempt_index=winner['attempt_index'],
            **{k:winner[k] for k in ('selected_metadata','snapshot_root','snapshot_inventory_sha256','generation_sha256')},
            search_directory=str(root),search_inventory_sha256=proof['search_inventory_sha256'],budget_view_models=views,
            qualified_search_audit_sha256=canonical_hash(actual))
        _bound_snapshots(row);models.append(row)
    joined=inspect_practical_model_inventory(practical_plan=practical,expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        models=models,primary_models=payload['primary_models'])
    if canonical_json(_frozen(global_barrier_reference,qualification_reference).payload)!=canonical_json(payload):
        raise ValueError('Frozen primary/global evidence changed during complete model sealing')
    for row in models:
        inspect_search_inventory_binding(search_directory=Path(row['search_directory']),expected_inventory_sha256=row['search_inventory_sha256'])
        _bound_snapshots(row)
    for proof in proofs.values():_ref(proof['search_plan'],{});_ref(proof['search_result'],{})
    if _source()!=payload['source_sha256']:raise ValueError('Frozen source changed during complete model sealing')
    by_id={row['job_id']:row for row in models}
    return dict(schema='research/v2-qualified-practical-model-payload',created_utc_ns=created_utc_ns,
        global_barrier_reference=copy.deepcopy(global_barrier_reference),qualification_reference=copy.deepcopy(qualification_reference),
        source_sha256=payload['source_sha256'],expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        practical_plan_reference=payload['practical_plan_reference'],support_reference=payload['support_reference'],procedure_reference=payload['procedure_reference'],
        primary_models_reference=payload['qualified_primary_models_reference'],secondary_search_proofs=proofs,
        models=[by_id[j['job_id']] for j in value['jobs']],identity_join=joined,qualified_practical_840_models_complete=True,
        primary_models_unchanged=True,global_procedure_semantics_authenticated=True,protected_access_authorized=False,
        protected_evaluation_authorized=False,T_accessed=False,whole_training_procedure_cost_authenticated=False,cost_claim_authorized=False,
        scope='Actual primary168 plus secondary672 native/HPO/V models and every retained budget snapshot. Final protected protocol/evaluation and whole cost remain separate.')


def _check_model_output_location(output,*,protected_paths):
    """Pure no-write path boundary, not qualification or publication authority."""
    from .primary_search_worker_v2 import _path
    protected=[*(Path(p) for p in protected_paths),Path(sys.prefix).resolve(),Path(sys.executable).resolve().parent]
    output=_path(str(output))
    if output.exists() or any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('Fresh complete model seal must preserve all native/global/model/search/source inputs')
    return output


def _publication_location(output,*,barrier,qualification_reference,models,proofs):
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    from .primary_search_worker_v2 import _path,_ROOT
    registry=load_native_qualification_registry(qualification_reference);roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    protected=[*(Path(p) for p in roots['working_roots']+roots['proof_roots']),Path(sys.prefix).resolve(),Path(barrier.reference['path'])]
    protected.extend(Path(barrier.payload[k]['path']) for k in ('practical_plan_reference','support_reference','procedure_reference','qualified_primary_models_reference'))
    protected.extend((_ROOT/name).resolve() for name in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))
    protected.extend(Path(row['search_directory']) for row in models)
    protected.extend(Path(view['model_binding']['snapshot_root']) for row in models for view in row['budget_view_models'].values() if view['model_binding'] is not None)
    protected.extend(Path(proof[k]['path']) for proof in proofs.values() for k in ('search_plan','search_result'))
    return _check_model_output_location(output,protected_paths=protected)


@dataclass(frozen=True)
class QualifiedPracticalModels:
    reference:dict
    _payload_json:str

    @property
    def payload(self):return json.loads(self._payload_json)


@_within_validation
def seal_qualified_practical_models(*,global_barrier_reference,qualification_reference,secondary_search_proofs,output):
    barrier=_frozen(global_barrier_reference,qualification_reference)
    payload=_assemble(barrier=barrier,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        secondary_search_proofs=secondary_search_proofs,created_utc_ns=time.time_ns())
    output=_publication_location(output,barrier=barrier,qualification_reference=qualification_reference,models=payload['models'],proofs=payload['secondary_search_proofs'])
    from .deployment_worker_v2 import _source
    if _source()!=payload['source_sha256']:raise ValueError('Source changed before complete model publication')
    from treeneuron.config import canonical_hash,canonical_json
    from .ledger import write_new_json
    _recheck_validation()
    write_new_json(output,dict(schema='research/v2-qualified-practical-models',payload=payload,payload_hash=canonical_hash(payload)))
    return QualifiedPracticalModels(dict(path=str(output),sha256=hashlib.sha256(output.read_bytes()).hexdigest()),canonical_json(payload))


@_within_validation
def load_qualified_practical_models(reference,*,global_barrier_reference,qualification_reference):
    barrier=_frozen(global_barrier_reference,qualification_reference)
    from treeneuron.config import canonical_hash,canonical_json
    from .primary_selection_v2 import _ref
    value=_ref(reference,{},decode=True)
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'} or value['schema']!='research/v2-qualified-practical-models'
            or type(value['payload']) is not dict or canonical_hash(value['payload'])!=value['payload_hash']):
        raise ValueError('Complete model envelope changed')
    saved=value['payload']
    try:actual=_assemble(barrier=barrier,global_barrier_reference=global_barrier_reference,qualification_reference=qualification_reference,
        secondary_search_proofs=saved['secondary_search_proofs'],created_utc_ns=saved['created_utc_ns'])
    except (KeyError,TypeError) as error:raise ValueError('Malformed complete model payload') from error
    if canonical_json(saved)!=canonical_json(actual):raise ValueError('Complete model bytes, identities or authority changed')
    _ref(reference,{})
    return QualifiedPracticalModels(copy.deepcopy(reference),canonical_json(actual))
