"""Immutable finite practical-program metadata; no fitting or data access.

This planner authenticates supplied manifest commitments, not their raw files or
native support semantics. Qualification, measured admission and the future frozen
protocol must pass before any job can execute. A hash is never an access token.
"""
from dataclasses import dataclass
import json
import re

from treeneuron.config import canonical_hash, canonical_json
from .seeds import seed64, seed32, seed31

_DATASETS=('airfoil','concrete','ccpp','parkinsons','bike','energy')
_FAMILIES=tuple(f'P{i:02}' for i in range(14))


def _digest(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}',value) is None:
        raise ValueError('Practical implementation and data pins require lowercase SHA256 digests')
    return value


def _checked_manifest(dataset,manifest):
    if (not isinstance(manifest,dict) or manifest.get('schema')!='research/v2-sealed-transfer'
            or manifest.get('dataset')!=dataset or manifest.get('raw_identity_verified') is not True
            or manifest.get('protected_labels_read_for_modeling') is not False):
        raise ValueError('Practical plan requires unchanged, verified, uninspected sealed transfer metadata')
    digest=_digest(manifest.get('manifest_hash'))
    if canonical_hash({k:v for k,v in manifest.items() if k!='manifest_hash'})!=digest:
        raise ValueError('Sealed transfer manifest commitment changed')
    splits=manifest.get('splits')
    if not isinstance(splits,list) or len(splits)!=5:
        raise ValueError('Exactly five fixed outer splits required')
    checked={}
    for split in splits:
        if (not isinstance(split,dict) or split.get('schema')!='research/v2-transfer-split'
                or split.get('dataset')!=dataset):raise ValueError('Invalid transfer split metadata')
        fold=split.get('outer_fold')
        if type(fold) is not int or fold not in range(5) or fold in checked:
            raise ValueError('Missing or duplicate transfer outer fold')
        primary=fold==(4 if dataset=='bike' else 0)
        if type(split.get('primary')) is not bool or split['primary']!=primary:
            raise ValueError('Transfer primary split changed')
        split_hash=_digest(split.get('split_hash'))
        if canonical_hash({k:v for k,v in split.items() if k!='split_hash'})!=split_hash:
            raise ValueError('Sealed transfer split commitment changed')
        checked[fold]=split_hash
    return digest,checked


@dataclass(frozen=True)
class PracticalPlan:
    resolved_json: str

    def to_dict(self):
        return json.loads(self.resolved_json)


def practical_search_budget(family,*,native_quality=False):
    if family not in _FAMILIES or type(native_quality) is not bool:
        raise ValueError('Known family and explicit quality-view flag required')
    return dict(max_attempts=32,unique_configuration_limit=29 if family=='P13' else 32,
        deadline_ns=None if native_quality else 7200000000000,
        descriptive_prefixes_ns=[600000000000,1800000000000]+([7200000000000] if native_quality else []),
        official_anchor_first=True,failures_and_caps_paid=True)


def _paired_search_seeds(study_id,seed_root,*,native_quality):
    if (type(study_id) is not str or re.fullmatch(r'[A-Za-z0-9_-]+',study_id) is None
            or type(seed_root) is not int or seed_root!=2026101802 or type(native_quality) is not bool):
        raise ValueError('Fixed practical seed namespace required')
    used32=set();seeds={}
    for dataset in _DATASETS:
        for fold in range(5):
            for repeat in range(2):
                labels=(study_id,'practical',dataset,'outer',str(fold),'repeat',str(repeat),'HPO')
                library,counter=seed32(seed_root,*labels,used=used32)
                seeds[dataset,fold,repeat]=dict(hpo_seed=seed64(seed_root,*labels),hpo_seed32=library,
                    seed32_collision_counter=counter)
    if native_quality:
        # Keep every established HPO number; model seeds exclude all HPO
        # values and earlier model labels, with intentional family pairing.
        for (dataset,fold,repeat),record in seeds.items():
            labels=(study_id,'practical',dataset,'outer',str(fold),'repeat',str(repeat),'fit')
            # Pair every family numerically in the narrowest native range.
            # Tree adapters require signed31; keep HPO/library32 untouched.
            library,counter=seed31(seed_root,*labels,used=used32)
            record.update(fit_seed64=seed64(seed_root,*labels),fit_seed=library,
                fit_seed_bits=31,fit_seed31_collision_counter=counter)
    return seeds


def expand_practical(data_manifests,*,support_lock_sha256,source_sha256,runtime_sha256,
                     lock_sha256,study_id='P2-search',seed_root=2026101802,native_quality=False):
    """Return840 fixed jobs without opening role arrays or authorizing workers.

    Support/source/runtime locks are immutable external prerequisites. They are
    bound here but must be qualified and rechecked by the eventual executor.
    The primary/secondary partition cannot be used to select favorable folds.
    """
    if (type(study_id) is not str or re.fullmatch(r'[A-Za-z0-9_-]+',study_id) is None
            or type(seed_root) is not int or seed_root!=2026101802 or type(native_quality) is not bool):
        raise ValueError('Invalid fixed practical study identity or seed root')
    if not isinstance(data_manifests,dict) or set(data_manifests)!=set(_DATASETS):
        raise ValueError('All six fixed datasets, including separate simulation Energy, required')
    bindings={name:_digest(value) for name,value in dict(support_lock_sha256=support_lock_sha256,
        source_sha256=source_sha256,runtime_sha256=runtime_sha256,lock_sha256=lock_sha256).items()}
    checked={name:_checked_manifest(name,data_manifests[name]) for name in _DATASETS}
    # Seeds label60 independent searches. Reusing a label across14 families is
    # intentional pairing, whereas an accidental collision across labels is not.
    seeds=_paired_search_seeds(study_id,seed_root,native_quality=native_quality)
    jobs=[]
    for tranche in ('primary','secondary'):
        for dataset in _DATASETS:
            manifest_hash,splits=checked[dataset]
            for fold in range(5):
                primary=fold==(4 if dataset=='bike' else 0)
                if primary!=(tranche=='primary'):continue
                for repeat in range(2):
                    for family in _FAMILIES:
                        jobs.append(dict(job_id=f'{dataset}/outer-{fold}/repeat-{repeat}/{family}',
                            dataset=dataset,outer_fold=fold,search_repeat=repeat,family_id=family,
                            tranche=tranche,simulation_only=dataset=='energy',
                            manifest_hash=manifest_hash,split_hash=splits[fold],**bindings,
                            **seeds[dataset,fold,repeat],seed_selection_allowed=False,
                            budget=practical_search_budget(family,native_quality=native_quality),
                            fit_worker_access=['F_labels','G_labels','V_features'],
                            coordinator_access=['V_labels'],protected_access_authorized=False,
                            execution_prerequisites=(['qualified_native_anchor','measured_resource_admission',
                                'frozen_search_support'] if primary else
                                ['frozen_global_procedure','all_168_primary_selected_model_hashes'])))
    value=dict(schema='research/v2-practical-plan',study_id=study_id,seed_root=seed_root,
        **bindings,data_manifest_hashes={name:checked[name][0] for name in _DATASETS},jobs=jobs,
        tranches=dict(primary_jobs=168,secondary_jobs=672),
        limits=dict(conservative_attempt_ceiling=26880,finite_unique_attempt_ceiling=26700,
            primary_conservative_attempt_ceiling=5376,primary_finite_unique_attempt_ceiling=5340,
            dedicated_worker_hours_ceiling=None if native_quality else 1680,
            primary_worker_hours_ceiling=None if native_quality else 336),
        forecast=None,forecast_status='Requires measured complete native anchors; ceilings are not forecasts',
        execution_implemented=False,protected_access_authorized=False,
        qualification_scope='Metadata commitment validation only; raw data, native supports and workers unqualified here',
        secondary_barrier='All168primary models and global procedure must be frozen by confirmation/v2; this plan grants no release')
    if native_quality:
        value['budget_policy']='native_quality'
        value['native_fit_seed_policy']='paired-signed31-v1'
    value['plan_hash']=canonical_hash(value)
    return PracticalPlan(canonical_json(value))
