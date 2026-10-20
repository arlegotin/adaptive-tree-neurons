"""Join all actual qualified primary searches to their complete winning models.

The old selection inventory authenticates metadata and files. This layer also
requires genuine native/journal/V evidence for every primary job. It does not
freeze global procedure semantics, qualify whole costs or release any labels.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import time

from treeneuron.config import canonical_hash,canonical_json
from .deployment_worker_v2 import _source
from .ledger import write_new_json
from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
from .primary_selection_v2 import _primary,_ref,_digest
from .qualified_search_audit_v2 import audit_primary_qualified_search
from .workqueue import _inventory


_FALSE=('global_procedure_semantics_authenticated','secondary_release_authorized',
        'protected_access_authorized','whole_training_procedure_cost_authenticated','T_accessed')


def inspect_selected_snapshot_binding(*,selected_model,expected_metadata,snapshot_root,expected_inventory_sha256):
    """Byte/path join only; supplied references confer no native authority."""
    references={};chosen=_ref(selected_model,references);expected=_ref(expected_metadata,references)
    root=Path(snapshot_root)
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or not _digest(expected_inventory_sha256) or chosen!=expected
            or selected_model!=expected_metadata or Path(chosen).parent!=root
            or Path(chosen).name not in ('manifest.json','state.json')
            or canonical_hash(_inventory(root))!=expected_inventory_sha256):
        raise ValueError('Selected model must match the actual winning metadata and full snapshot inventory')
    return dict(metadata=selected_model,snapshot_root=str(root),snapshot_inventory_sha256=expected_inventory_sha256,
        native_workload_qualified=False,protected_access_authorized=False,
        scope='Byte-bound metadata/full-file join only; actual model generation and search qualification are separate')


def inspect_search_inventory_binding(*,search_directory,expected_inventory_sha256):
    """Full byte inventory only; this does not qualify the search's contents."""
    root=Path(search_directory)
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink() or not root.is_dir()
            or not _digest(expected_inventory_sha256) or canonical_hash(_inventory(root))!=expected_inventory_sha256):
        raise ValueError('Complete closed search inventory changed after its audit')
    return dict(search_directory=str(root),inventory_sha256=expected_inventory_sha256,
        native_workload_qualified=False,protected_access_authorized=False)


def _publication_location(output,registry,inventory,qualification_reference):
    from .method_worker import _ROOT
    output=Path(output)
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    protected=[*(Path(p) for p in roots['working_roots']+roots['proof_roots']),Path(sys.prefix).resolve(),
        *(Path(e['search_plan']['path']).parent for e in inventory['selections'].values()),
        *((_ROOT/name).resolve() for name in ('src','tests','configs','locks','environment','.git','.agents','.codex','.aws'))]
    if (not output.is_absolute() or output.resolve()!=output or output.is_symlink() or output.exists()
            or any(output.is_relative_to(root) for root in protected)):
        raise ValueError('Fresh primary model seal must preserve all working/proof/search/executable inputs')


def _assemble(*,registry,practical_plan,expected_practical_plan_hash,support_reference,
        qualification_reference,primary_inventory_reference,search_proofs,created_utc_ns):
    from .primary_selection_v2 import load_primary_selection_inventory
    plan,jobs=_primary(practical_plan,expected_practical_plan_hash)
    if type(created_utc_ns) is not int or created_utc_ns<=0:
        raise ValueError('Immutable primary publication time required')
    if type(search_proofs) is not dict or set(search_proofs)!={j['job_id'] for j in jobs}:
        raise ValueError('All168 exact primary search proofs required; no missing job or seed selection')
    references={};_ref(primary_inventory_reference,references)
    loaded=load_primary_selection_inventory(primary_inventory_reference['path'],
        expected_file_sha256=primary_inventory_reference['sha256'],plan=practical_plan,
        expected_plan_hash=expected_practical_plan_hash)
    inventory=loaded.payload;models=[];snapshots=[];searches=[]
    if (plan.get('budget_policy')!='native_quality' or inventory['source_sha256']!=plan['source_sha256']
            or type(support_reference) is not dict or support_reference.get('sha256')!=plan['support_lock_sha256']):
        raise ValueError('Actual primary source/support/native-quality scope differs')
    from .practical_procedure_v2 import load_practical_procedure
    procedure=load_practical_procedure(inventory['global_procedure'],practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference)
    model_rows={row['job_id']:row for row in inventory['models']}
    for job in jobs:
        name=job['job_id'];entry=inventory['selections'][name];proof=search_proofs[name]
        if (type(proof) is not dict or set(proof)!={'journal_sha256','official_schedule_reference'}
                or not _digest(proof['journal_sha256'])):
            raise ValueError('Exact primary journal and official-sampler references required')
        root=Path(entry['search_plan']['path']).parent
        search_digest=canonical_hash(_inventory(root))
        actual=audit_primary_qualified_search(practical_plan=practical_plan,
            expected_practical_plan_hash=expected_practical_plan_hash,job_id=name,
            support_reference=support_reference,qualification_reference=qualification_reference,
            search_directory=root,expected_search_plan_sha256=entry['search_plan']['sha256'],
            result_path=Path(entry['search_result']['path']),expected_result_sha256=entry['search_result']['sha256'],
            expected_journal_sha256=proof['journal_sha256'],official_schedule_reference=proof['official_schedule_reference'])
        searches.append(inspect_search_inventory_binding(search_directory=root,expected_inventory_sha256=search_digest))
        if (actual.get('complete_selected_model') is not True or actual.get('completed_search') is not True
                or actual.get('job_id')!=name or actual.get('practical_plan_hash')!=expected_practical_plan_hash
                or any(actual.get(k) is not True for k in ('HPO_journal_authenticated',
                    'all_attempt_native_evidence_authenticated','all_eligible_V_scores_authenticated','native_workload_qualified'))):
            raise ValueError('Every primary model needs actual complete native search qualification')
        winner=actual['selected_model'];row=model_rows[name]
        if type(winner['attempt_index']) is not int or winner['attempt_index']!=row['winner_attempt_index']:
            raise ValueError('Primary model winner differs from actual qualified search')
        bound=inspect_selected_snapshot_binding(selected_model=entry['selected_model'],
            expected_metadata=winner['selected_metadata'],snapshot_root=Path(winner['snapshot_root']),
            expected_inventory_sha256=winner['snapshot_inventory_sha256'])
        snapshots.append(bound)
        views=actual.get('search_view_models')
        if (type(views) is not dict or views.get('schema')!='research/v2-primary-budget-model-bindings'
                or type(views.get('views')) is not dict
                or set(views['views'])!={'native_quality','600000000000','1800000000000','7200000000000'}
                or views['views']['native_quality'].get('model_binding')!=winner):
            raise ValueError('Every actual retained budget model must match the qualified search')
        for view in views['views'].values():
            model=view['model_binding']
            if model is None:continue
            view_bound=inspect_selected_snapshot_binding(selected_model=model['selected_metadata'],expected_metadata=model['selected_metadata'],
                snapshot_root=Path(model['snapshot_root']),expected_inventory_sha256=model['snapshot_inventory_sha256'])
            if view_bound not in snapshots:snapshots.append(view_bound)
        models.append(dict(job_id=name,dataset=job['dataset'],outer_fold=job['outer_fold'],
            family_id=job['family_id'],search_repeat=job['search_repeat'],fit_seed=job['fit_seed'],
            split_hash=job['split_hash'],manifest_hash=job['manifest_hash'],V_score=row['V_score'],
            winner_attempt_index=winner['attempt_index'],selected_metadata=bound['metadata'],
            snapshot_root=bound['snapshot_root'],snapshot_inventory_sha256=bound['snapshot_inventory_sha256'],
            search_directory=str(root),search_inventory_sha256=search_digest,
            generation_sha256=winner['generation_sha256'],budget_view_models=views['views'],qualified_search_audit_sha256=canonical_hash(actual)))
    # All mutable access logs are outside the pinned proof inventories. Recheck
    # the genuine registry, old inventory and full model bytes before returning.
    if load_native_qualification_registry(qualification_reference)!=registry:
        raise ValueError('Actual native qualification changed during primary model sealing')
    again=load_primary_selection_inventory(primary_inventory_reference['path'],
        expected_file_sha256=primary_inventory_reference['sha256'],plan=practical_plan,
        expected_plan_hash=expected_practical_plan_hash)
    if again.payload!=inventory or _source()!=plan['source_sha256']:
        raise ValueError('Primary inventory or source changed during model sealing')
    for bound in snapshots:
        inspect_selected_snapshot_binding(selected_model=bound['metadata'],expected_metadata=bound['metadata'],
            snapshot_root=Path(bound['snapshot_root']),expected_inventory_sha256=bound['snapshot_inventory_sha256'])
    for bound in searches:
        inspect_search_inventory_binding(search_directory=Path(bound['search_directory']),
            expected_inventory_sha256=bound['inventory_sha256'])
    if load_practical_procedure(inventory['global_procedure'],practical_plan=practical_plan,
            expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference)!=procedure:
        raise ValueError('Pre-search practical semantics changed during primary model sealing')
    return dict(schema='research/v2-qualified-primary-model-payload',created_utc_ns=created_utc_ns,
        plan_hash=expected_practical_plan_hash,source_sha256=plan['source_sha256'],support_reference=support_reference,
        qualification_reference=qualification_reference,primary_inventory_reference=primary_inventory_reference,
        global_procedure=inventory['global_procedure'],search_proofs=search_proofs,models=models,
        qualified_primary_168_models_complete=True,presearch_procedure_semantics_authenticated=True,**{flag:False for flag in _FALSE},
        scope='Actual complete primary native searches, precise winning metadata/full snapshots and exact pre-search semantic declaration. Post-search global freeze, complete costs, secondary release and protected authorization remain separate.')


@dataclass(frozen=True)
class QualifiedPrimaryModels:
    path:Path
    file_sha256:str
    _payload_json:str

    @property
    def payload(self):
        import json
        return json.loads(self._payload_json)


def seal_qualified_primary_models(*,practical_plan,expected_practical_plan_hash,support_reference,
        qualification_reference,primary_inventory_reference,search_proofs,output):
    registry=load_native_qualification_registry(qualification_reference)
    # Registry is deliberately first: caller-written inventory/flags cannot
    # cause plan, model or search inspection under an unqualified environment.
    payload=_assemble(registry=registry,practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,
        qualification_reference=qualification_reference,primary_inventory_reference=primary_inventory_reference,
        search_proofs=search_proofs,created_utc_ns=time.time_ns())
    from .primary_selection_v2 import load_primary_selection_inventory
    inventory=load_primary_selection_inventory(primary_inventory_reference['path'],
        expected_file_sha256=primary_inventory_reference['sha256'],plan=practical_plan,
        expected_plan_hash=expected_practical_plan_hash).payload
    output=Path(output);_publication_location(output,registry,inventory,qualification_reference)
    write_new_json(output,dict(schema='research/v2-qualified-primary-models',payload=payload,payload_hash=canonical_hash(payload)))
    digest=hashlib.sha256(output.read_bytes()).hexdigest()
    _ref(dict(path=str(output),sha256=digest),{})
    return QualifiedPrimaryModels(output,digest,canonical_json(payload))


def load_qualified_primary_models(*,practical_plan,expected_practical_plan_hash,support_reference,
        qualification_reference,path,expected_file_sha256):
    registry=load_native_qualification_registry(qualification_reference)
    value=_ref(dict(path=str(path),sha256=expected_file_sha256),{},decode=True)
    if (type(value) is not dict or set(value)!={'schema','payload','payload_hash'}
            or value['schema']!='research/v2-qualified-primary-models' or type(value['payload']) is not dict
            or canonical_hash(value['payload'])!=value['payload_hash']):
        raise ValueError('Qualified primary model envelope changed')
    saved=value['payload']
    try:
        rebuilt=_assemble(registry=registry,practical_plan=practical_plan,
            expected_practical_plan_hash=expected_practical_plan_hash,support_reference=support_reference,
            qualification_reference=qualification_reference,primary_inventory_reference=saved['primary_inventory_reference'],
            search_proofs=saved['search_proofs'],created_utc_ns=saved['created_utc_ns'])
    except (KeyError,TypeError) as error:
        raise ValueError('Malformed qualified primary model payload') from error
    if canonical_json(saved)!=canonical_json(rebuilt):raise ValueError('Qualified model semantics or authority changed')
    return QualifiedPrimaryModels(Path(path),expected_file_sha256,canonical_json(rebuilt))
