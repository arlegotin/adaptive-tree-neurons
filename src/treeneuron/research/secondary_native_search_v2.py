"""Unchanged native search procedures on genuinely released secondary folds.

This factory prepares callbacks, not an owned production program, complete paid
search audit, protected protocol or cost claim. Public entry reloads real gates.
"""
import copy
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time

from treeneuron.config import canonical_json
from .method_worker import _ROOT,_sha,scientific_source_sha256
from .ledger import write_new_json
from .primary_search_declaration_v2 import _bound_job,_compile_search_declaration
from .primary_native_preparation_v2 import _stage_native_template,_check_preparation_locations
from .native_trial_bridge_v2 import InspectedNativeTrialBridge
from .qualified_native_trial_v2 import _inspect_completed_native_trial
from .secondary_native_evidence_v2 import _candidate,_inspection,_workload
from .search_controller_v2 import SearchPlan
from .search_supports_v2 import family_search_spec
from .trial_bridge_v2 import EngineeringTrialBridge


_ADMISSION_CLOCK=('Whole global/native prerequisite checks, fitting, replay, workload and V arithmetic are paid here; '
    'caller includes publication/return. Nested spans are not added twice.')
_ADMISSION_SCOPE=('Companion eligibility after actual global/native gates and this secondary workload. No complete paid search, '
    'whole cost or protected authority; upstream engineering flags retained.')


def _bound_secondary_job(**kwargs):return _bound_job(**kwargs,role_scope='secondary')


def _secondary_declaration(**kwargs):
    """Metadata only; this private join cannot release data or execute a job."""
    return _compile_search_declaration(**kwargs,bind_job=_bound_secondary_job,role_scope='secondary')


def _frozen_declaration(*,barrier_reference,qualification_reference,job_id,official_schedule_reference=None):
    from .practical_global_freeze_v2 import load_global_primary_barrier
    barrier=load_global_primary_barrier(barrier_reference,qualification_reference=qualification_reference)
    payload=barrier.payload
    if (payload.get('secondary_development_access_authorized') is not True
            or payload.get('protected_access_authorized') is not False):
        raise ValueError('Actual global primary barrier required for secondary development')
    from .primary_selection_v2 import _ref
    from .workload import PracticalPlan
    plan=PracticalPlan(canonical_json(_ref(payload['practical_plan_reference'],{},decode=True)))
    declaration=_secondary_declaration(practical_plan=plan,expected_practical_plan_hash=payload['expected_practical_plan_hash'],
        job_id=job_id,support_reference=payload['support_reference'],official_schedule_reference=official_schedule_reference)
    if declaration['job']['source_sha256']!=payload['source_sha256']:
        raise ValueError('Secondary declaration cannot change the frozen global source')
    return barrier,declaration


def compile_secondary_search_declaration(**kwargs):
    _,declaration=_frozen_declaration(**kwargs)
    return declaration


def _stage(roles,**kwargs):
    def store_factory(root,fold):
        if Path(root)!=roles._store.root:raise ValueError('Secondary template cannot replace its admitted data root')
        roles._split(fold)
        return roles
    return _stage_native_template(**kwargs,store_factory=store_factory,role_scope='secondary',
        record_before_staging=lambda fold:roles._access(fold,['F_labels','G_labels','V_features']))


class _SecondaryInspectedBridge(InspectedNativeTrialBridge):
    """Private engineering bridge; it never promotes a candidate by itself."""
    def __init__(self,*,role_store,**kwargs):
        self._secondary_roles=role_store
        if Path(kwargs['prepared_root'])!=role_store._store.root:
            raise ValueError('Secondary bridge must preserve its admitted data root')
        self._initialize(**kwargs,role_store=role_store)

    def _inspect_candidate(self,**kwargs):return _candidate(self._secondary_roles,**kwargs)


def _secondary_component(roles,**kwargs):
    return _inspect_completed_native_trial(**kwargs,
        inspection_loader=lambda reference,**values:_inspection(roles,reference,**values),
        workload_inspector=lambda **values:_workload(roles,values),role_scope='secondary')


def _paid_attempt(roles,**kwargs):
    from .qualified_search_audit_v2 import _inspect_paid_native_attempt,_failed_inspection
    from .secondary_native_evidence_v2 import _generation
    def store_factory(root,fold):
        if Path(root)!=roles._store.root:raise ValueError('Paid secondary attempt must preserve its frozen prepared root')
        roles._split(fold)
        return roles
    return _inspect_paid_native_attempt(**kwargs,store_factory=store_factory,
        component_inspector=lambda **values:_secondary_component(roles,**values),
        failed_inspector=lambda root,generation,args:_failed_inspection(root,generation,args,audit_generation=lambda **values:_generation(roles,**values)))


def _check_secondary_admission(packet,outcome,*,qualification_reference,barrier_reference,charged_ns):
    """Validate supplied companion metadata; this function grants no capability."""
    component=outcome['candidate_component'];score=outcome['inspected_V_score'];trial=outcome['expected_trial']
    admitted=component is not None and type(score) in (int,float) and math.isfinite(score) and score>=0
    fields={'schema','qualification_reference','global_barrier_reference','source_sha256','plan_hash','candidate_id','upstream_status',
        'status','V_score','component','native_workload_qualified','quality_eligible','HPO_journal_authenticated','protected_access_authorized',
        'T_accessed','callback_body_before_publication_ns','clock_scope','scope'}
    if (type(packet) is not dict or set(packet)!=fields or packet['schema']!='research/v2-qualified-secondary-native-candidate-admission'
            or canonical_json(packet['qualification_reference'])!=canonical_json(qualification_reference)
            or canonical_json(packet['global_barrier_reference'])!=canonical_json(barrier_reference)
            or any(packet[k]!=trial[k] for k in ('source_sha256','plan_hash','candidate_id'))
            or packet['upstream_status']!=outcome['status'] or packet['status']!=('completed_eligible' if admitted else outcome['status'])
            or type(packet['V_score']) is not type(score if admitted else None) or packet['V_score']!=(score if admitted else None)
            or canonical_json(packet['component'])!=canonical_json(component)
            or any(type(packet[k]) is not bool or packet[k]!=admitted for k in ('quality_eligible','native_workload_qualified'))
            or any(packet[k] is not False for k in ('HPO_journal_authenticated','protected_access_authorized','T_accessed'))
            or packet['clock_scope']!=_ADMISSION_CLOCK or packet['scope']!=_ADMISSION_SCOPE
            or type(charged_ns) is not int or type(packet['callback_body_before_publication_ns']) is not int
            or not charged_ns>=packet['callback_body_before_publication_ns']>=max(1,outcome['callback_body_ns'])):
        raise ValueError('Secondary companion contradicts its frozen gate, native evidence, score, status or paid clock')


class _QualifiedSecondaryCallback:
    def __init__(self,*,inner,roles,barrier_reference,qualification_reference,declaration):
        self._inner=inner;self._roles=roles
        self._frozen=canonical_json(dict(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
            declaration=declaration))
        self.native_sampler_minimum_prerequisite_ns=inner.native_sampler_minimum_prerequisite_ns
        self.preparation_ns=inner.preparation_ns

    def _unchanged(self):
        frozen=json.loads(self._frozen);declaration=frozen['declaration']
        barrier,current=_frozen_declaration(barrier_reference=frozen['barrier_reference'],qualification_reference=frozen['qualification_reference'],
            job_id=declaration['job']['job_id'],official_schedule_reference=declaration['official_schedule_reference'])
        if canonical_json(current)!=canonical_json(declaration):raise ValueError('Frozen secondary procedure changed')
        return frozen,barrier

    def __call__(self,request):
        started=time.perf_counter_ns();frozen,_=self._unchanged()
        result=self._inner(request);attempt=Path(request['attempt_directory']);search=Path(request['search_directory'])
        native=json.loads(self._inner._binding);evidence=None
        if result['status']=='completed_ineligible':
            expected=dict(plan_hash=native['plan']['plan_hash'],source_sha256=native['plan']['identity']['source_sha256'],
                candidate_id=request['candidate_id'],candidate_origin=request['candidate_origin'],raw_configuration=request['config'],
                compiled_configuration=self._inner._configuration(native,request),official_schedule_reference=native['official_schedule_reference'],
                native_hpo_seed=native['native_hpo_seed'],fit_seed=native['plan']['fit_seed'])
            evidence=_secondary_component(self._roles,attempt_directory=attempt,expected_trial=expected,
                prepared_root=Path(native['prepared_root']),expected_manifest_sha256=native['expected_manifest_sha256'],outer_fold=native['outer_fold'])
        self._unchanged()
        score=None if evidence is None else evidence['inspected_V_score']
        admitted=type(score) in (int,float) and math.isfinite(score) and score>=0
        status='completed_eligible' if admitted else result['status']
        write_new_json(attempt/'admission.json',dict(schema='research/v2-qualified-secondary-native-candidate-admission',
            qualification_reference=frozen['qualification_reference'],global_barrier_reference=frozen['barrier_reference'],
            source_sha256=native['plan']['identity']['source_sha256'],plan_hash=native['plan']['plan_hash'],candidate_id=request['candidate_id'],
            upstream_status=result['status'],status=status,V_score=score if admitted else None,component=evidence,
            native_workload_qualified=admitted,quality_eligible=admitted,HPO_journal_authenticated=False,
            protected_access_authorized=False,T_accessed=False,callback_body_before_publication_ns=time.perf_counter_ns()-started,
            clock_scope=_ADMISSION_CLOCK,scope=_ADMISSION_SCOPE))
        return dict(status=status,V_score=score if admitted else None,reason=None if admitted else result['reason'],
            files=EngineeringTrialBridge._files(attempt,search),checkpoint_files=result['checkpoint_files'])


@dataclass(frozen=True)
class PreparedSecondaryNativeCallback:
    plan:SearchPlan
    callback:_QualifiedSecondaryCallback
    manifest_path:Path
    manifest_sha256:str
    preparation_before_return_ns:int
    native_sampler_minimum_prerequisite_ns:int


def prepare_secondary_native_callback(*,barrier_reference,qualification_reference,job_id,output,search_directory,
        official_schedule_reference=None,rss_limit_bytes=4*1024**3):
    started=time.perf_counter_ns()
    barrier,declaration=_frozen_declaration(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
        job_id=job_id,official_schedule_reference=official_schedule_reference)
    payload=barrier.payload;job=declaration['job'];dataset=job['dataset']
    from .released_transfer_v2 import ReleasedTransferStore
    store=ReleasedTransferStore(dataset,barrier_reference=barrier_reference,qualification_reference=qualification_reference)
    from .native_qualification_registry_v2 import load_native_qualification_registry,native_registry_storage_roots
    from .native_verification_v2 import verification_identity
    registry=load_native_qualification_registry(qualification_reference);prepared=store.root;entry=payload['prepared_datasets'][dataset]
    manifest=entry['manifest_file_sha256'];plan=SearchPlan(canonical_json(declaration['search_plan']));spec=family_search_spec(job['family_id'],
        dict(columns=store.manifest['feature_names'],categorical=store.manifest['categorical_features']))
    record=registry['preparation']['datasets'][dataset]
    if (registry['identity']!=verification_identity() or registry['identity']['scientific_source_sha256']!=payload['source_sha256']
            or str(prepared)!=registry['inputs']['prepared_roots'][dataset] or manifest!=record['manifest_file_sha256']
            or job['split_hash']!=record['split_hashes'][str(job['outer_fold'])]
            or declaration['search_plan']['identity']['support_sha256']!=spec['support_hash']):
        raise ValueError('Secondary source/native implementation/support/split differs from actual qualification')
    store._roles._split(job['outer_fold'])
    roots=native_registry_storage_roots(inputs=registry['inputs'],registry_reference=qualification_reference)
    primary_paths=[row['search_directory'] for row in payload['primary_models']]
    primary_paths.extend(view['model_binding']['snapshot_root'] for row in payload['primary_models']
        for view in row['budget_view_models'].values() if view['model_binding'] is not None)
    proofs=[*roots['proof_roots'],*primary_paths,barrier_reference['path']]
    output=Path(output);search=Path(search_directory)
    _check_preparation_locations(output=output,search=search,working_roots=roots['working_roots'],proof_roots=proofs,rss_limit_bytes=rss_limit_bytes)
    protected=[Path(p) for p in [*roots['working_roots'],*proofs]]
    if any(p.is_relative_to(search) or p.is_relative_to(output) for p in protected):
        raise ValueError('Secondary output/search cannot contain qualified or frozen primary evidence')
    anchor=next(row for row in registry['anchors'] if row['dataset']==dataset and row['family_id']==job['family_id']
        and row['anchor_id']==spec['anchors'][0]['anchor_id'])
    from .native_anchor_v2 import load_native_anchor_execution
    proof=load_native_anchor_execution(anchor['reference'],prepared_root=prepared,expected_manifest_sha256=manifest,
        expected_family=job['family_id'],expected_anchor_id=anchor['anchor_id'],expected_dataset=dataset)
    process=json.loads((Path(proof['generation']['native_inventory']['root'])/'process/process.json').read_text())
    interpreter=process['argv'][0];paths=process['environment_overrides']['PYTHONPATH'].split(os.pathsep)
    if (Path(interpreter).resolve()!=Path(sys.executable).resolve() or not paths or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).is_absolute() or not Path(p).is_dir() or not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Actual qualified native interpreter/import overlays changed')
    staged=_stage(store._roles,plan=plan,prepared_root=prepared,expected_manifest_sha256=manifest,outer_fold=job['outer_fold'],output=output/'template')
    inner=_SecondaryInspectedBridge(role_store=store._roles,plan=plan,template_path=Path(staged['request_path']),
        expected_template_sha256=staged['request_sha256'],prepared_root=prepared,expected_manifest_sha256=manifest,outer_fold=job['outer_fold'],
        search_directory=search,interpreter=interpreter,extra_pythonpath=paths[1:],rss_limit_bytes=rss_limit_bytes,
        official_schedule_reference=official_schedule_reference,native_hpo_seed=declaration['native_hpo_seed'])
    callback=_QualifiedSecondaryCallback(inner=inner,roles=store._roles,barrier_reference=barrier_reference,
        qualification_reference=qualification_reference,declaration=declaration)
    callback._unchanged()
    value=dict(schema='research/v2-prepared-secondary-native-callback',declaration=declaration,
        qualification_reference=copy.deepcopy(qualification_reference),global_barrier_reference=copy.deepcopy(barrier_reference),
        template=staged,search_directory=str(search),interpreter=interpreter,interpreter_sha256=_sha(Path(interpreter)),extra_pythonpath=paths[1:],
        rss_limit_bytes=rss_limit_bytes,actual_primary_anchor_reference=anchor['reference'],
        native_sampler_minimum_prerequisite_ns=callback.native_sampler_minimum_prerequisite_ns,
        elapsed_before_publication_ns=time.perf_counter_ns()-started,search_executed=False,secondary_native_workload_qualified=False,
        whole_process_cost_authenticated=False,production_execution_authorized=False,protected_access_authorized=False,T_accessed=False,
        scope='Prepared unchanged-support secondary callback after actual primary/global gates. Each new completed native workload still needs its own companion admission; owned program/paid-search/whole cost/T remain separate.')
    target=output/'factory.json';write_new_json(target,value)
    if scientific_source_sha256()!=payload['source_sha256']:raise ValueError('Frozen source changed during secondary callback preparation')
    return PreparedSecondaryNativeCallback(plan,callback,target,_sha(target),time.perf_counter_ns()-started,
        callback.native_sampler_minimum_prerequisite_ns)


def audit_secondary_qualified_search(*,barrier_reference,qualification_reference,job_id,search_directory,
        expected_search_plan_sha256,result_path,expected_result_sha256,expected_journal_sha256,official_schedule_reference=None):
    """Reopen real gates, native outcomes, paid HPO law and every retained view."""
    barrier,declaration=_frozen_declaration(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
        job_id=job_id,official_schedule_reference=official_schedule_reference)
    from .released_transfer_v2 import ReleasedTransferStore
    store=ReleasedTransferStore(declaration['job']['dataset'],barrier_reference=barrier_reference,qualification_reference=qualification_reference)
    prepared=store.root;manifest=barrier.payload['prepared_datasets'][declaration['job']['dataset']]['manifest_file_sha256']
    store._roles._split(declaration['job']['outer_fold'])
    spec=family_search_spec(declaration['job']['family_id'],dict(columns=store.manifest['feature_names'],categorical=store.manifest['categorical_features']))
    from .qualified_search_audit_v2 import _audit_qualified_search
    def compile_again():
        _,current=_frozen_declaration(barrier_reference=barrier_reference,qualification_reference=qualification_reference,
            job_id=job_id,official_schedule_reference=official_schedule_reference)
        return current
    result=_audit_qualified_search(declaration=declaration,prepared=prepared,manifest=manifest,spec=spec,search_directory=search_directory,
        expected_search_plan_sha256=expected_search_plan_sha256,result_path=result_path,expected_result_sha256=expected_result_sha256,
        expected_journal_sha256=expected_journal_sha256,qualification_reference=qualification_reference,
        attempt_inspector=lambda **kwargs:_paid_attempt(store._roles,**kwargs),
        admission_checker=lambda packet,outcome,**kwargs:_check_secondary_admission(packet,outcome,
            qualification_reference=qualification_reference,barrier_reference=barrier_reference,**kwargs),
        unchanged=compile_again,compile_again=compile_again)
    return dict(result,schema='research/v2-qualified-secondary-search-audit',global_barrier_reference=copy.deepcopy(barrier_reference),
        native_workload_qualified=True,secondary_672_model_barrier_satisfied=False,
        scope='Actual global/native gates, unchanged secondary declaration, complete paid HPO/native/V evidence and retained budget models. No complete840 model seal, whole cost or protected authority.')
