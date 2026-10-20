"""Companion candidate admission after actual complete native qualification."""
import copy
import json
from pathlib import Path
import time

from treeneuron.config import canonical_hash, canonical_json
from treeneuron.data.tabular import DATASETS
from .ledger import write_new_json
from .method_worker import _ROOT, _digest, _read_json, _sha, scientific_source_sha256
from .native_anchor_v2 import _closed_reference
from .native_inspection_reader_v2 import load_primary_native_inspection
from .native_qualification_registry_v2 import (load_native_qualification_registry,
    _all_references, _prepared_datasets)
from .native_trial_bridge_v2 import InspectedNativeTrialBridge
from .native_verification_v2 import verification_identity
from .native_workload_v2 import inspect_native_workload
from .search_controller_v2 import _checked_plan
from .search_supports_v2 import family_search_spec
from .trial_bridge_v2 import EngineeringTrialBridge
from .workqueue import _inventory


_TRIAL={'plan_hash','source_sha256','candidate_id','candidate_origin','raw_configuration',
    'compiled_configuration','official_schedule_reference','native_hpo_seed','fit_seed'}


def _check_references(references):
    for reference in references:_closed_reference(reference)


def _bind_procedure(registry,native_arguments):
    """Reject mismatched metadata before the inner bridge decodes worker input."""
    identity=registry['identity'];plan=_checked_plan(native_arguments['plan'])
    root=Path(native_arguments['prepared_root']);fold=native_arguments['outer_fold']
    matches=[name for name,path in registry['inputs']['prepared_roots'].items() if str(root)==path]
    if (not root.is_absolute() or root.is_symlink() or root.resolve()!=root or len(matches)!=1):
        raise ValueError('Exact qualified official working root required')
    dataset=matches[0];record=registry['preparation']['datasets'][dataset]
    if (type(fold) is not int or fold!=(4 if dataset=='bike' else 0)
            or native_arguments['expected_manifest_sha256']!=record['manifest_file_sha256']):
        raise ValueError('Qualified manifest/primary role differs')
    schema=DATASETS[dataset];spec=family_search_spec(plan['identity']['family_id'],
        dict(columns=list(schema.features),categorical=list(schema.categorical)))
    if (identity!=verification_identity() or plan['identity']['source_sha256']!=identity['scientific_source_sha256']
            or plan['identity']['runtime_sha256']!=identity['test_and_lock_files']['locks/research-v2/core.json']
            or plan['identity']['support_sha256']!=spec['support_hash']
            or plan['identity']['split_sha256']!=record['split_hashes'][str(fold)]):
        raise ValueError('Qualified source/runtime/support/split differs')
    return dataset


def inspect_completed_native_trial(*,attempt_directory,expected_trial,prepared_root,
        expected_manifest_sha256,outer_fold):
    """Check an actual trial component; this helper grants no global admission."""
    expected=copy.deepcopy(expected_trial);attempt=Path(attempt_directory)
    if (type(expected) is not dict or set(expected)!=_TRIAL
            or any(not _digest(expected[k]) for k in ('plan_hash','source_sha256'))
            or expected['source_sha256']!=scientific_source_sha256()
            or type(expected['fit_seed']) is not int or expected['fit_seed']<0
            or expected['candidate_origin'] not in ('anchor','finite','tpe')
            or not attempt.is_absolute() or attempt.is_symlink() or attempt.resolve()!=attempt):
        raise ValueError('Exact current trial proposal/source identity required')
    before=_inventory(attempt);packet=_read_json(attempt/'bridge.json')
    if (packet.get('schema')!='research/v2-inspected-native-trial' or packet.get('status')!='completed_ineligible'
            or any(packet.get(k)!=v or type(packet.get(k)) is not type(v) for k,v in expected.items())
            or any(packet.get(k) is not False for k in ('quality_eligible','native_workload_qualified',
                'HPO_journal_authenticated','T_accessed'))):
        raise ValueError('Inspected trial proposal/configuration/authority differs')
    generation=attempt/'generation';path=generation/'input/request.json';request=_read_json(path)
    if (packet.get('native_request_sha256')!=_sha(path) or request['config']!=expected['compiled_configuration']
            or request['fit_seed']!=expected['fit_seed']):raise ValueError('Native trial request/configuration differs')
    inspected=attempt/'inspection';saved=_read_json(inspected/'result.json')
    reference=dict(root=str(inspected),inventory_sha256=canonical_hash(_inventory(inspected)))
    args=dict(request_path=path,expected_request_sha256=_sha(path),native_reference=saved['generation']['native_inventory'],
        prepared_root=Path(prepared_root),expected_manifest_sha256=expected_manifest_sha256,
        outer_fold=outer_fold,expected_fit_seed=expected['fit_seed'],expected_configuration=expected['compiled_configuration'])
    inspection=load_primary_native_inspection(reference,generation_arguments=args,immutable_generation_root=generation)
    workload=inspect_native_workload(**args)
    if (workload['generation']!=inspection['generation'] or workload['source_sha256']!=expected['source_sha256']
            or packet.get('inspected_V_score')!=inspection['validation']['score']):
        raise ValueError('Trial workload/inspection belongs to a different generation or score')
    if _inventory(attempt)!=before or scientific_source_sha256()!=expected['source_sha256']:
        raise ValueError('Native trial evidence/source changed during inspection')
    return dict(schema='research/v2-checked-native-trial-component',candidate_evidence_authenticated=True,
        expected_trial=expected,inspection_reference=reference,inspection_sha256=canonical_hash(inspection),
        workload_sha256=canonical_hash(workload),generation=inspection['generation'],
        inspected_V_score=inspection['validation']['score'],resources=workload['process_observation'],
        quality_eligible=False,native_workload_qualified=False,T_accessed=False,
        scope='Actual native proposal/generation/full workload/closed replay/primary V component only; no complete registry, paid journal or protected authority')


class QualifiedNativeTrialBridge:
    """Keep original engineering packets; publish separate technical admission.

    Construction requires the complete actual registry. No caller-provided
    success flag can substitute for reopening its prerequisites.
    """
    def __init__(self,*,qualification_reference,**native_arguments):
        started=time.perf_counter_ns()
        registry=load_native_qualification_registry(qualification_reference)
        # Keep this ordering: invalid registries cannot reach worker preparation.
        dataset=_bind_procedure(registry,native_arguments)
        inner=InspectedNativeTrialBridge(**native_arguments)
        binding=json.loads(inner._binding)
        self._inner=inner
        self._binding=canonical_json(dict(registry=registry,qualification_reference=copy.deepcopy(qualification_reference),
            native=binding,dataset=dataset))
        self._unchanged(json.loads(self._binding))
        self.native_sampler_minimum_prerequisite_ns=inner.native_sampler_minimum_prerequisite_ns
        self.preparation_ns=time.perf_counter_ns()-started

    @staticmethod
    def _unchanged(binding):
        registry=binding['registry']
        if registry['identity']!=verification_identity():raise ValueError('Qualified source/test/lock identity changed')
        _check_references([binding['qualification_reference'],*_all_references(registry['inputs'])])
        _prepared_datasets(registry['inputs'],registry['preparation'])

    def __call__(self,request):
        started=time.perf_counter_ns();binding=json.loads(self._binding);native=binding['native']
        self._unchanged(binding)
        result=self._inner(request);attempt=Path(request['attempt_directory']);search=Path(request['search_directory'])
        evidence=None;upstream=result['status']
        if upstream=='completed_ineligible':
            expected=dict(plan_hash=native['plan']['plan_hash'],source_sha256=native['plan']['identity']['source_sha256'],
                candidate_id=request['candidate_id'],candidate_origin=request['candidate_origin'],raw_configuration=request['config'],
                compiled_configuration=self._inner._configuration(native,request),
                official_schedule_reference=native['official_schedule_reference'],native_hpo_seed=native['native_hpo_seed'],
                fit_seed=native['plan']['fit_seed'])
            evidence=inspect_completed_native_trial(attempt_directory=attempt,expected_trial=expected,
                prepared_root=Path(native['prepared_root']),expected_manifest_sha256=native['expected_manifest_sha256'],
                outer_fold=native['outer_fold'])
        self._unchanged(binding)
        admitted=evidence is not None
        status='completed_eligible' if admitted else upstream
        score=evidence['inspected_V_score'] if admitted else None
        write_new_json(attempt/'admission.json',dict(schema='research/v2-qualified-native-candidate-admission',
            qualification_reference=binding['qualification_reference'],source_sha256=native['plan']['identity']['source_sha256'],
            plan_hash=native['plan']['plan_hash'],candidate_id=request['candidate_id'],upstream_status=upstream,
            status=status,V_score=score,component=evidence,native_workload_qualified=admitted,quality_eligible=admitted,
            HPO_journal_authenticated=False,protected_access_authorized=False,T_accessed=False,
            callback_body_before_publication_ns=time.perf_counter_ns()-started,
            clock_scope='All registry byte-integrity scans, native callback, candidate workload and coordinator checks paid; outer search pays publication/return. Nested native/inspection spans must not be added twice.',
            scope='Companion eligibility for declared V selection only. Original worker/model/inspection/bridge flags unchanged; complete paid search, primary-model seal and protected protocol remain separate.'))
        return dict(status=status,V_score=score,reason=None if admitted else result['reason'],
            files=EngineeringTrialBridge._files(attempt,search),checkpoint_files=result['checkpoint_files'])
