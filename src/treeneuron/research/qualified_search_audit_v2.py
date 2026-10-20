"""Compose actual qualification, paid search law and every native attempt."""
import copy
import os
from pathlib import Path
import sys

from treeneuron.config import canonical_hash, canonical_json
from .deployment_worker_v2 import _bytes, _json
from .method_worker import _ROOT, _THREADS, _read_json, _sha, scientific_source_sha256
from .native_generation_v2 import audit_native_generation
from .native_outcome_reader_v2 import inspect_native_fit_outcome
from .native_qualification_registry_v2 import load_native_qualification_registry
from .native_trial_bridge_v2 import InspectedNativeTrialBridge
from .primary_search_declaration_v2 import compile_primary_search_declaration
from .process_receipt_v2 import inspect_closed_process
from .qualified_native_trial_v2 import QualifiedNativeTrialBridge, _bind_procedure, inspect_completed_native_trial
from .search_audit_v2 import _inactive, audit_native_search_journal
from .search_controller_v2 import SearchPlan, _checked_plan, _files
from .search_supports_v2 import family_search_spec
from .transfer import TransferStore
from .workqueue import _inventory


_ADMISSION_CLOCK=('All registry byte-integrity scans, native callback, candidate workload and coordinator checks paid; '
    'outer search pays publication/return. Nested native/inspection spans must not be added twice.')
_ADMISSION_SCOPE=('Companion eligibility for declared V selection only. Original worker/model/inspection/bridge flags '
    'unchanged; complete paid search, primary-model seal and protected protocol remain separate.')
_BRIDGE_CLOCK=('Callback entry through packet preparation; native fit and inspection spans are nested; '
    'outer search pays publication/return')
_TRIAL={'plan_hash','source_sha256','candidate_id','candidate_origin','raw_configuration',
    'compiled_configuration','official_schedule_reference','native_hpo_seed','fit_seed'}
_BRIDGE_FIELDS=_TRIAL|{'schema','native_request_sha256','status','reason','inspected_V_score',
    'compilation_ns','inspection_span_ns','callback_body_ns','clock_scope','quality_eligible',
    'native_workload_qualified','HPO_journal_authenticated','T_accessed'}


def _exact_attempt_files(search,index,rows):
    attempt=search/f'attempt-{index:03}'
    checked=_files(search,rows)
    actual=[dict(path=f'attempt-{index:03}/'+r['path'],sha256=r['sha256']) for r in _inventory(attempt)['files']]
    if sorted(checked,key=lambda r:r['path'])!=actual:
        raise ValueError('Every paid attempt file must be retained exactly; no foreign/omitted losing evidence')


def _failed_inspection(root,generation,args):
    """A supervised replay failure reads no V labels and supplies no score."""
    saved=_read_json(root/'result.json');process=_read_json(root/'process/process.json');source=scientific_source_sha256()
    native=audit_native_generation(**args);request=_read_json(args['request_path']);path=root/'input/request.json'
    expected=dict(schema='research/v2-prediction-replay-request',purpose='engineering_qualification',
        family_id=request['family_id'],scientific_source_sha256=source,feature_schema=request['config']['feature_schema'],
        snapshot=dict(path='snapshot',metadata_sha256=native['snapshot']['sha256']),
        query=dict(path='query.npz',sha256=request['files']['V_features']['sha256']),
        prediction=dict(path='prediction.npz',sha256=native['prediction']['sha256']))
    if (_read_json(path)!=expected or _sha(root/'input/query.npz')!=expected['query']['sha256']
            or _sha(root/'input/prediction.npz')!=expected['prediction']['sha256']
            or _inventory(root/'input/snapshot')!=_inventory(Path(native['snapshot']['path']).parent)
            or saved.get('schema')!='research/v2-native-candidate-inspection' or saved.get('generation')!=native
            or saved.get('immutable_generation_root')!=str(generation)
            or saved.get('immutable_generation_inventory_sha256')!=canonical_hash(_inventory(generation))
            or saved.get('source_sha256')!=source or saved.get('process')!=process
            or any(saved.get(k) is not False for k in ('V_coordinator_invoked','V_labels_read_by_coordinator',
                'native_workload_qualified','HPO_journal_authenticated','quality_eligible','T_accessed',
                'model_training_independently_reexecuted'))
            or saved.get('replay') is not None or saved.get('validation') is not None):
        raise ValueError('Failed inspection generation/query/authority differs')
    argv=process.get('argv',[]);env=process.get('environment_overrides',{})
    if (type(argv) is not list or not argv or type(argv[0]) is not str or type(env) is not dict
            or type(env.get('PYTHONPATH')) is not str):raise ValueError('Managed failed replay invocation required')
    paths=env['PYTHONPATH'].split(os.pathsep)
    if (Path(argv[0]).resolve()!=Path(sys.executable).resolve() or paths[0]!=str(_ROOT/'src')
            or any(not Path(p).is_absolute() or not Path(p).is_dir()
                or not Path(p).resolve().is_relative_to((_ROOT/'environment').resolve()) for p in paths[1:])):
        raise ValueError('Failed replay interpreter/overlay differs')
    expected_env={k:'1' for k in (*_THREADS,'PYTHONDONTWRITEBYTECODE')}
    expected_env.update(PYTHONPATH=os.pathsep.join(paths),PYTORCH_ENABLE_MPS_FALLBACK='0',KMP_DUPLICATE_LIB_OK=None,
        MPLCONFIGDIR=str(root/'cache/mpl'),XDG_CACHE_HOME=str(root/'cache/xdg'))
    observed=inspect_closed_process(dict(root=str(root/'process'),inventory_sha256=canonical_hash(_inventory(root/'process'))),
        expected_argv=[argv[0],'-B','-m','treeneuron.research.prediction_replay_v2','--request',str(path),
            '--expected-request-sha256',_sha(path),'--output',str(root/'replay')],expected_cwd=str(_ROOT),
        expected_environment_overrides=expected_env,
        expected_identity=dict(purpose='candidate_inspection',request_sha256=_sha(path),source_sha256=source))
    if (process['timeout_ns'] is not None or observed['execution_status']=='exited'
            or saved.get('status')!=observed['execution_status']
            or type(saved.get('elapsed_ns')) is not int or type(saved.get('replay_supervisor_span_ns')) is not int
            or not saved['elapsed_ns']>=saved['replay_supervisor_span_ns']>=observed['elapsed_ns']):
        raise ValueError('Failed inspection status/cost contradicts process')
    return observed


def inspect_paid_native_attempt(*,search_directory,search_plan,spec,request,prepared_root,
        expected_manifest_sha256,outer_fold,official_schedule_reference,native_hpo_seed):
    """Actual component evidence only; this supplies no registry eligibility."""
    search=Path(search_directory);plan=_checked_plan(SearchPlan(canonical_json(search_plan)));request=copy.deepcopy(request)
    keys={'candidate_id','candidate_origin','config','retry_index','checkpoint_files','attempt_index','fit_seed','budget_remaining_ns'}
    if (type(request) is not dict or set(request)!=keys or type(request['attempt_index']) is not int
            or not 0<=request['attempt_index']<plan['max_attempts'] or type(request['retry_index']) is not int
            or request['retry_index'] not in (0,1)):
        raise ValueError('Exact typed paid native controller request required')
    index=request['attempt_index'];attempt=search/f'attempt-{index:03}'
    if (not search.is_absolute() or search.resolve()!=search or search.is_symlink()
            or type(index) is not int or index<0):raise ValueError('Exact paid attempt location required')
    before=_inventory(attempt);source=scientific_source_sha256();packet=_read_json(attempt/'bridge.json')
    compiled=InspectedNativeTrialBridge._configuration(dict(plan=plan,spec=spec),request)
    trial=dict(plan_hash=plan['plan_hash'],source_sha256=plan['identity']['source_sha256'],
        candidate_id=request['candidate_id'],candidate_origin=request['candidate_origin'],raw_configuration=request['config'],
        compiled_configuration=compiled,official_schedule_reference=official_schedule_reference,
        native_hpo_seed=native_hpo_seed,fit_seed=plan['fit_seed'])
    if (type(packet) is not dict or set(packet)!=_BRIDGE_FIELDS or packet['schema']!='research/v2-inspected-native-trial'
            or source!=trial['source_sha256'] or any(canonical_json(dict(v=packet[k]))!=canonical_json(dict(v=v)) for k,v in trial.items())
            or any(packet[k] is not False for k in ('quality_eligible','native_workload_qualified','HPO_journal_authenticated','T_accessed'))
            or packet['clock_scope']!=_BRIDGE_CLOCK
            or any(type(packet[k]) is not int or packet[k]<0 for k in ('compilation_ns','inspection_span_ns','callback_body_ns'))
            or not packet['callback_body_ns']>=packet['compilation_ns']+packet['inspection_span_ns']):
        raise ValueError('Paid native proposal/source/authority/clock differs')
    prepared=Path(prepared_root);manifest=prepared/'manifest.json'
    if manifest.is_symlink() or _sha(manifest)!=expected_manifest_sha256:raise ValueError('Official prepared manifest differs')
    store=TransferStore(prepared)
    if type(outer_fold) is not int or outer_fold!=(4 if store.manifest['dataset']=='bike' else 0):
        raise ValueError('Only primary development inputs admitted before the global freeze')
    split=store._split(outer_fold);role_files={}
    for role in ('F','G','V_features'):
        relative=f'outer-{outer_fold}/'+('V-features.npz' if role=='V_features' else role+'.npz')
        digest=store.manifest['files'][relative]['sha256']
        if _sha(prepared/relative)!=digest:raise ValueError('Official prepared role bytes differ')
        role_files[role]=dict(path=role+'.npz',sha256=digest)
    generation=attempt/'generation';path=generation/'input/request.json';worker=_read_json(path)
    initial=index
    if request['retry_index']:
        refs=_files(search,request['checkpoint_files'])
        if len(refs)!=1 or refs[0]['path']!=f'attempt-{index-1:03}/recovery.json':
            raise ValueError('Paid retry requires preceding exact native recovery packet')
        previous=_read_json(search/refs[0]['path'])
        if (set(previous)!={'plan_hash','initial_attempt_index','request','request_canonical_hash','native_reference'}
                or type(previous['initial_attempt_index']) is not int or previous['initial_attempt_index']!=index-1
                or previous['plan_hash']!=plan['plan_hash']):
            raise ValueError('Paid retry lineage differs')
        initial=previous['initial_attempt_index']
    elif request['retry_index']!=0 or request['checkpoint_files']!=[]:
        raise ValueError('Fresh attempt cannot reuse undeclared checkpoint state')
    expected_worker=dict(schema='research/v2-method-worker-request',purpose='engineering_qualification',
        job_id=plan['identity']['job_id']+f'/attempt-{initial:03}',family_id=plan['identity']['family_id'],
        fit_seed=plan['fit_seed'],scientific_source_sha256=source,config=compiled,budget={'native_stopping':True},
        observation_identity=dict(dataset=store.manifest['dataset'],split_kind='temporal' if store.manifest['dataset']=='bike' else 'grouped',
            split_sha256=split['split_hash']),files=role_files)
    if (worker!=expected_worker or packet['native_request_sha256']!=_sha(path)
            or request['fit_seed']!=plan['fit_seed'] or request['budget_remaining_ns'] is not None
            or plan['identity']['split_sha256']!=split['split_hash']):
        raise ValueError('Paid worker request/data/configuration/seed differs')
    returned=_read_json(generation/'native-fit-result.json');native=returned.get('recovery_reference')
    if type(native) is not dict or native.get('root')!=str(generation/'native'):
        raise ValueError('Paid native generation location differs')
    outcome=inspect_native_fit_outcome(request_path=path,expected_request_sha256=_sha(path),native_reference=native)
    saved=_read_json(generation/'native/result.json')
    if (set(returned)!=set(saved)|{'recovery_reference','callback_elapsed_ns'}
            or any(returned[k]!=v for k,v in saved.items()) or type(returned['callback_elapsed_ns']) is not int
            or returned['callback_elapsed_ns']<saved['elapsed_before_recovery_inventory_ns']):
        raise ValueError('Paid returned native outcome differs from its closed generation')
    if request['retry_index']:
        prior=_read_json(search/f'attempt-{index-1:03}/generation/native-fit-result.json')['recovery_reference']
        if (previous['native_reference']!=prior or previous['request']!=worker
                or previous['request_canonical_hash']!=canonical_hash(worker)):
            raise ValueError('Paid retry does not preserve its preceding request/generation')
        recovery=outcome['recovery']
        if recovery['reason']=='fresh_native_attempt' or recovery['resume'] and recovery['source']!=prior:
            raise ValueError('Paid retry secretly restarted or resumed a different native state')
    elif outcome['recovery']!={'resume':False,'reason':'fresh_native_attempt'}:
        raise ValueError('Fresh paid attempt has an undeclared recovery source')
    component=None;failure_inspection=None;upstream=outcome['status']
    if upstream=='completed_ineligible':
        inspected=_read_json(attempt/'inspection/result.json')
        if inspected.get('status')=='completed_candidate_inspection':
            component=inspect_completed_native_trial(attempt_directory=attempt,expected_trial=trial,
                prepared_root=prepared,expected_manifest_sha256=expected_manifest_sha256,outer_fold=outer_fold)
        else:
            args=dict(request_path=path,expected_request_sha256=_sha(path),native_reference=native,
                prepared_root=prepared,expected_manifest_sha256=expected_manifest_sha256,outer_fold=outer_fold,
                expected_fit_seed=plan['fit_seed'],expected_configuration=compiled)
            failure_inspection=_failed_inspection(attempt/'inspection',generation,args)
            upstream=failure_inspection['execution_status']
    status='interrupted_recoverable' if upstream in ('resource_exhausted','interrupted_recoverable') else upstream
    score=component['inspected_V_score'] if component is not None else None
    if packet['status']!=status or packet['inspected_V_score']!=score:
        raise ValueError('Paid bridge disposition/score contradicts native outcome or inspection')
    if status=='interrupted_recoverable':
        recovery=_read_json(attempt/'recovery.json')
        if canonical_json(recovery)!=canonical_json(dict(plan_hash=plan['plan_hash'],initial_attempt_index=initial,request=worker,
                request_canonical_hash=canonical_hash(worker),native_reference=native)):
            raise ValueError('Outgoing paid recovery packet differs from this native generation')
    if (_inventory(attempt)!=before or scientific_source_sha256()!=source
            or _sha(manifest)!=expected_manifest_sha256):raise ValueError('Paid native attempt/source/data changed during audit')
    return dict(schema='research/v2-checked-paid-native-attempt',attempt_evidence_authenticated=True,
        expected_trial=trial,status=status,reason=packet['reason'],inspected_V_score=score,candidate_component=component,
        native_outcome=outcome,failed_inspection_process=failure_inspection,callback_body_ns=packet['callback_body_ns'],
        quality_eligible=False,native_workload_qualified=False,T_accessed=False,
        scope='Actual paid proposal/request/native outcome and optional candidate component. No registry, whole-search or protected authority.')


def _check_admission(packet,outcome,reference,*,charged_ns):
    admitted=outcome['candidate_component'] is not None;trial=outcome['expected_trial']
    fields={'schema','qualification_reference','source_sha256','plan_hash','candidate_id','upstream_status',
        'status','V_score','component','native_workload_qualified','quality_eligible','HPO_journal_authenticated',
        'protected_access_authorized','T_accessed','callback_body_before_publication_ns','clock_scope','scope'}
    if (type(packet) is not dict or set(packet)!=fields or packet['schema']!='research/v2-qualified-native-candidate-admission'
            or packet['qualification_reference']!=reference
            or any(packet[k]!=trial[k] for k in ('source_sha256','plan_hash','candidate_id'))
            or packet['upstream_status']!=outcome['status']
            or packet['status']!=('completed_eligible' if admitted else outcome['status'])
            or type(packet['V_score']) is not type(outcome['inspected_V_score'])
            or packet['V_score']!=outcome['inspected_V_score'] or packet['component']!=outcome['candidate_component']
            or any(type(packet[k]) is not bool or packet[k]!=admitted for k in ('quality_eligible','native_workload_qualified'))
            or any(packet[k] is not False for k in ('HPO_journal_authenticated','protected_access_authorized','T_accessed'))
            or packet['clock_scope']!=_ADMISSION_CLOCK or packet['scope']!=_ADMISSION_SCOPE
            or type(charged_ns) is not int or type(packet['callback_body_before_publication_ns']) is not int
            or not charged_ns>=packet['callback_body_before_publication_ns']>=max(1,outcome.get('callback_body_ns',1))):
        raise ValueError('Qualified admission contradicts actual candidate/failure/registry/cost evidence')


def _check_sampler_preparation(events,minimum):
    sessions=[e['payload']['invocation_preparation_ns'] for e in events if e['kind']=='session_started']
    if (type(minimum) is not int or minimum<0 or not sessions or type(sessions[0]) is not int
            or sessions[0]<minimum):
        raise ValueError('Native sampler prerequisite must be paid in the first search session before prefix eligibility')


def audit_primary_qualified_search(*,practical_plan,expected_practical_plan_hash,job_id,support_reference,
        qualification_reference,search_directory,expected_search_plan_sha256,result_path,expected_result_sha256,
        expected_journal_sha256,official_schedule_reference=None):
    """Read a closed primary search; no new fit/inference or protected access.

    Every actual complete prerequisite is reopened before any search component.
    No caller-provided success flag substitutes for that capability. Whole outer
    process time and the all-168-model/global freeze barrier remain separate.
    """
    registry=load_native_qualification_registry(qualification_reference)
    declaration=compile_primary_search_declaration(practical_plan=practical_plan,
        expected_practical_plan_hash=expected_practical_plan_hash,job_id=job_id,support_reference=support_reference,
        official_schedule_reference=official_schedule_reference)
    plan=declaration['search_plan'];job=declaration['job'];dataset=job['dataset']
    prepared=Path(registry['inputs']['prepared_roots'][dataset]);manifest=registry['preparation']['datasets'][dataset]['manifest_file_sha256']
    _bind_procedure(registry,dict(plan=SearchPlan(canonical_json(plan)),prepared_root=prepared,
        expected_manifest_sha256=manifest,outer_fold=job['outer_fold']))
    binding=dict(registry=registry,qualification_reference=qualification_reference)
    root=Path(search_directory);result_path=Path(result_path)
    if (not root.is_absolute() or root.resolve()!=root or root.is_symlink()
            or not result_path.is_absolute() or result_path.resolve()!=result_path or result_path.is_symlink()):
        raise ValueError('Exact closed search/result paths required')
    spec=family_search_spec(job['family_id'],dict(columns=TransferStore(prepared).manifest['feature_names'],
        categorical=TransferStore(prepared).manifest['categorical_features']))
    with _inactive(root):
        before=_inventory(root);raw_plan=_bytes(root/'plan.json',expected_search_plan_sha256)
        if _json(raw_plan)!=plan:raise ValueError('Actual primary search differs from frozen global declaration')
        journal=audit_native_search_journal(search_directory=root,expected_plan_sha256=expected_search_plan_sha256,
            result_path=result_path,expected_result_sha256=expected_result_sha256,expected_journal_sha256=expected_journal_sha256)
        raw_result=_bytes(result_path,expected_result_sha256);saved=_json(raw_result)
        events=[_json(line) for line in _bytes(root/'events.jsonl',expected_journal_sha256).splitlines()]
        _check_sampler_preparation(events,declaration['native_sampler_minimum_prerequisite_ns'])
        starts=[e['payload'] for e in events if e['kind']=='attempt_started']
        finishes=[e['payload'] for e in events if e['kind']=='attempt_finished']
        if {p.name for p in root.glob('attempt-*')}!={f'attempt-{i:03}' for i in range(len(starts))}:
            raise ValueError('Actual attempt directories differ from the paid search history')
        attempts=[];winner=None
        for request,finish in zip(starts,finishes,strict=True):
            row=finish['record'];index=request['attempt_index'];attempt=root/f'attempt-{index:03}'
            _exact_attempt_files(root,index,row['files'])
            component=inspect_paid_native_attempt(search_directory=root,search_plan=plan,spec=spec,request=request,
                prepared_root=prepared,expected_manifest_sha256=manifest,outer_fold=job['outer_fold'],
                official_schedule_reference=official_schedule_reference,native_hpo_seed=declaration['native_hpo_seed'])
            admission=_read_json(attempt/'admission.json');_check_admission(admission,component,qualification_reference,charged_ns=row['charged_ns'])
            result=finish['result']
            if (row['status']!=admission['status'] or row['V_score']!=admission['V_score']
                    or result['reason']!=(None if admission['quality_eligible'] else component['reason'])
                    or result['checkpoint_files']!=([dict(path=f'attempt-{index:03}/recovery.json',sha256=_sha(attempt/'recovery.json'))]
                        if row['status']=='interrupted_recoverable' else [])):
                raise ValueError('Paid journal changes native eligibility/failure/score/recovery evidence')
            attempts.append(dict(attempt_index=index,status=row['status'],V_score=row['V_score'],
                evidence_sha256=canonical_hash(component),admission_sha256=_sha(attempt/'admission.json')))
            if saved['winner'] is not None and saved['winner']['attempt_index']==index:
                generation=component['candidate_component']['generation'];snapshot=Path(generation['snapshot']['path']).parent
                winner=dict(attempt_index=index,selected_metadata=generation['snapshot'],snapshot_root=str(snapshot),
                    snapshot_inventory_sha256=canonical_hash(_inventory(snapshot)),generation_sha256=canonical_hash(generation))
        QualifiedNativeTrialBridge._unchanged(binding)
        again=compile_primary_search_declaration(practical_plan=practical_plan,
            expected_practical_plan_hash=expected_practical_plan_hash,job_id=job_id,support_reference=support_reference,
            official_schedule_reference=official_schedule_reference)
        if (_inventory(root)!=before or _bytes(result_path,expected_result_sha256)!=raw_result
                or _bytes(root/'plan.json',expected_search_plan_sha256)!=raw_plan or again!=declaration):
            raise ValueError('Qualified search evidence changed during audit')
    return dict(schema='research/v2-qualified-primary-search-audit',practical_plan_hash=expected_practical_plan_hash,
        job_id=job_id,declaration_hash=declaration['declaration_hash'],qualification_reference=copy.deepcopy(qualification_reference),
        journal=journal,attempts=attempts,selected_model=winner,completed_search=journal['completed_search'],
        complete_selected_model=journal['completed_search'] and winner is not None,
        HPO_journal_authenticated=True,all_attempt_native_evidence_authenticated=True,
        all_eligible_V_scores_authenticated=True,native_workload_qualified=True,
        whole_process_cost_authenticated=False,primary_168_model_barrier_satisfied=False,
        protected_access_authorized=False,T_accessed=False,new_fit_or_inference=False,
        scope='Actual complete registry, exact primary global declaration, native proposal law, all paid native outcomes/candidate companions and selected snapshot. No whole outer process clock, all168/global freeze, secondary release or protected authority.')
