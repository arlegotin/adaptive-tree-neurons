"""Compile one exact primary job's metadata; this does not admit execution."""
import copy

from treeneuron.config import canonical_hash
from .method_worker import scientific_source_sha256
from .practical_support_lock_v2 import load_practical_support_lock
from .primary_selection_v2 import _primary
from .qualify import qualification_locks_sha256
from .search_controller_v2 import make_search_plan


def compile_primary_search_declaration(*,practical_plan,expected_practical_plan_hash,
        job_id,support_reference,official_schedule_reference=None):
    """Bind full study/job seeds and actual supports without reading data.

    In particular, native RealMLP uses the job's 63-bit HPO namespace, while
    generic library replay retains the distinct collision-managed 32-bit seed.
    Qualification, resources, worker execution and final freeze remain separate.
    """
    value,jobs=_primary(practical_plan,expected_practical_plan_hash)
    matches=[job for job in jobs if job['job_id']==job_id]
    if len(matches)!=1:raise ValueError('Exactly one declared primary job required; no secondary release')
    law=load_practical_support_lock(support_reference);locks=qualification_locks_sha256()
    if (value.get('budget_policy')!='native_quality' or value['support_lock_sha256']!=support_reference['sha256']
            or value['source_sha256']!=law['source_sha256'] or value['runtime_sha256']!=law['core_manifest_sha256']
            or value['lock_sha256']!=locks):
        raise ValueError('Native-quality study source/support/core/runtime/adapter lock pins differ')
    job=matches[0];family=job['family_id'];spec=law['families'][job['dataset']][family]
    anchors=[row['config'] for row in spec['anchors']];candidates=[];mode=spec['search_kind'];support=spec['support']
    native_seed=None;minimum=0
    if family=='P06':
        from .native_schedule_v2 import load_realmlp_native_schedule
        if official_schedule_reference is None:raise ValueError('Official native RealMLP sampler reference required')
        native_seed=job['hpo_seed']
        schedule=load_realmlp_native_schedule(official_schedule_reference,
            expected_hpo_seed=native_seed,expected_feature_schema=spec['feature_schema'])
        if schedule['configurations'][:len(anchors)]!=anchors:raise ValueError('Official native schedule anchor differs')
        candidates=schedule['configurations'][len(anchors):];mode='finite';support=None
        minimum=schedule['paid_preparation_ns']
    elif official_schedule_reference is not None:
        raise ValueError('Official native sampler reference belongs only to RealMLP')
    elif family=='P13':candidates=spec['finite_configurations'][len(anchors):]
    budget=job['budget']
    search=make_search_plan(identity=dict(job_id=job_id,family_id=family,search_repeat=job['search_repeat'],
        source_sha256=value['source_sha256'],runtime_sha256=value['runtime_sha256'],support_sha256=spec['support_hash'],
        split_sha256=job['split_hash']),fit_seed=job['fit_seed'],hpo_seed32=job['hpo_seed32'],
        anchors=anchors,mode=mode,support=support,candidates=candidates,max_attempts=budget['max_attempts'],
        deadline_ns=budget['deadline_ns'],prefixes_ns=budget['descriptive_prefixes_ns'])
    if (load_practical_support_lock(support_reference)!=law or qualification_locks_sha256()!=locks
            or scientific_source_sha256()!=value['source_sha256']):
        raise ValueError('Primary declaration source/support/lock changed during compilation')
    body=dict(schema='research/v2-primary-search-declaration',practical_plan_hash=expected_practical_plan_hash,
        job=copy.deepcopy(job),support_reference=copy.deepcopy(support_reference),search_plan=search.to_dict(),
        official_schedule_reference=copy.deepcopy(official_schedule_reference),native_hpo_seed=native_seed,
        native_sampler_minimum_prerequisite_ns=minimum,native_workload_qualified=False,
        protected_access_authorized=False,execution_implemented=False,
        scope='Exact primary metadata/support/model and HPO seed declaration only; full actual data/qualification/resources/execution/journal/global freeze and protected authority remain required. Native schedule minimum is not its whole paid preparation cost.')
    body['declaration_hash']=canonical_hash(body)
    return body
