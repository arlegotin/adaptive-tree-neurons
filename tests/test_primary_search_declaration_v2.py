"""Global primary job seeds/support pins must reach the actual search plan."""
import importlib.util
from pathlib import Path
import sys

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.primary_search_declaration_v2') is not None,'Missing exact primary job search declaration'
    from treeneuron.research import primary_search_declaration_v2
    return primary_search_declaration_v2


@pytest.fixture
def study(tmp_path):
    from test_practical_workload import fixtures
    from treeneuron.research.practical_support_lock_v2 import seal_practical_support_lock
    from treeneuron.research.qualify import qualification_locks_sha256
    from treeneuron.research.method_worker import scientific_source_sha256,_ROOT,_sha
    from treeneuron.research.workload import expand_practical
    support=seal_practical_support_lock(tmp_path/'supports.json')
    plan=expand_practical(fixtures(),support_lock_sha256=support['sha256'],source_sha256=scientific_source_sha256(),
        runtime_sha256=_sha(_ROOT/'locks/research-v2/core.json'),lock_sha256=qualification_locks_sha256(),native_quality=True)
    return plan,support


@pytest.mark.parametrize('family',['P00','P01','P02','P03','P07','P08','P13'])
def test_actual_primary_namespace_compiles_exact_job_seed_and_support_without_execution(study,family):
    module=api();plan,support=study;job_id=f'airfoil/outer-0/repeat-1/{family}'
    result=module.compile_primary_search_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        job_id=job_id,support_reference=support)
    expected=next(j for j in plan.to_dict()['jobs'] if j['job_id']==job_id);search=result['search_plan']
    assert search['fit_seed']==expected['fit_seed'] and search['hpo_seed32']==expected['hpo_seed32']
    assert search['identity']['job_id']==job_id and search['identity']['split_sha256']==expected['split_hash']
    assert search['deadline_ns'] is None and search['max_attempts']==32
    assert result['native_hpo_seed'] is None and result['official_schedule_reference'] is None
    assert result['native_workload_qualified'] is result['protected_access_authorized'] is False
    if family=='P13':assert len(search['anchors'])+len(search['candidates'])==29


@pytest.mark.parametrize('fault',['plan_hash','secondary','support_pin','runtime_pin','adapter_pin','unexpected_schedule'])
def test_wrong_primary_job_or_self_rehashed_global_pin_cannot_compile(study,fault):
    module=api();plan,support=study;value=plan.to_dict();job_id='airfoil/outer-0/repeat-0/P00';extra={}
    if fault in ('support_pin','runtime_pin','adapter_pin'):
        from treeneuron.config import canonical_hash,canonical_json
        from treeneuron.research.workload import PracticalPlan
        field={'support_pin':'support_lock_sha256','runtime_pin':'runtime_sha256','adapter_pin':'lock_sha256'}[fault]
        value[field]='0'*64
        for row in value['jobs']:row[field]='0'*64
        value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'})
        plan=PracticalPlan(canonical_json(value))
    elif fault=='secondary':job_id='airfoil/outer-1/repeat-0/P00'
    elif fault=='unexpected_schedule':extra['official_schedule_reference']={'root':'/unused','inventory_sha256':'0'*64}
    with pytest.raises((ValueError,PermissionError)):
        module.compile_primary_search_declaration(practical_plan=plan,expected_practical_plan_hash='0'*64 if fault=='plan_hash' else value['plan_hash'],
            job_id=job_id,support_reference=support,**extra)


def test_real_native_RealMLP_schedule_uses_declared_63_bit_seed_separately_from_library32(study,tmp_path):
    module=api();plan,support=study;job_id='airfoil/outer-0/repeat-0/P06'
    job=next(j for j in plan.to_dict()['jobs'] if j['job_id']==job_id)
    assert job['hpo_seed']!=job['hpo_seed32']
    from treeneuron.research.native_schedule_v2 import prepare_realmlp_native_schedule
    from treeneuron.data.tabular import DATASETS
    schema=DATASETS['airfoil'];prepared=prepare_realmlp_native_schedule(
        feature_schema=dict(columns=list(schema.features),categorical=list(schema.categorical)),hpo_seed=job['hpo_seed'],
        output=tmp_path/'native-schedule',interpreter=sys.executable,
        extra_pythonpath=[Path(__file__).resolve().parents[1]/'environment/research-v2-realmlp-site'])
    assert prepared['status']=='completed_native_schedule'
    result=module.compile_primary_search_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
        job_id=job_id,support_reference=support,official_schedule_reference=prepared['reference'])
    assert result['native_hpo_seed']==job['hpo_seed']
    assert result['search_plan']['hpo_seed32']==job['hpo_seed32']
    assert result['search_plan']['mode']=='finite' and len(result['search_plan']['candidates'])==31
    assert result['native_sampler_minimum_prerequisite_ns']>0 and result['native_workload_qualified'] is False
    with pytest.raises(ValueError,match='seed|schedule|source'):
        module.compile_primary_search_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
            job_id='airfoil/outer-0/repeat-1/P06',support_reference=support,official_schedule_reference=prepared['reference'])


def test_P06_cannot_substitute_generic_support_without_official_sampler(study):
    module=api();plan,support=study
    with pytest.raises(ValueError):
        module.compile_primary_search_declaration(practical_plan=plan,expected_practical_plan_hash=plan.to_dict()['plan_hash'],
            job_id='airfoil/outer-0/repeat-0/P06',support_reference=support)
