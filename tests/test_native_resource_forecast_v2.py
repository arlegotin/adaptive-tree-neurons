"""Measured envelopes remain explicit heuristics, never statistical quantiles."""
import importlib.util
from pathlib import Path

import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.native_resource_forecast_v2'), 'Missing measured native practical forecast'
    from treeneuron.research import native_resource_forecast_v2
    return native_resource_forecast_v2


def test_public_forecast_requires_actual_registry_before_any_plan_or_support(tmp_path):
    module=api()
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.forecast_native_practical_resources(practical_plan=None,expected_practical_plan_hash='a'*64,
            support_reference=dict(path=str(tmp_path/'forbidden-support.json'),sha256='b'*64),
            qualification_reference=dict(root=str(tmp_path/'fake-registry'),inventory_sha256='c'*64))
    assert 'forbidden-support' not in str(error.value)
    assert list(tmp_path.iterdir())==[]


def test_closed_footprint_includes_external_generation_and_counts_nested_files_once(tmp_path):
    module=api()
    from treeneuron.config import canonical_hash
    from treeneuron.research.workqueue import _inventory
    generation=tmp_path/'native-generation';generation.mkdir();(generation/'weights.bin').write_bytes(b'x'*200)
    inspection=generation/'inspection';inspection.mkdir();(inspection/'receipt.json').write_bytes(b'{}')
    qualification=tmp_path/'anchor-proof';qualification.mkdir();(qualification/'result.json').write_bytes(b'y'*20)
    references=[dict(root=str(p),inventory_sha256=canonical_hash(_inventory(p))) for p in (generation,inspection,qualification)]
    result=module._closed_logical_footprint(references)
    assert result['retained_logical_bytes']==222 and result['unique_files']==3
    assert result['native_qualification_authenticated'] is False
    (generation/'weights.bin').write_bytes(b'z'*200)
    with pytest.raises(ValueError):module._closed_logical_footprint(references)


@pytest.fixture
def study(tmp_path):
    from test_practical_workload import fixtures
    from treeneuron.research.workload import expand_practical
    from treeneuron.research.deployment_worker_v2 import _source
    plan=expand_practical(fixtures(),support_lock_sha256='a'*64,source_sha256=_source(),
        runtime_sha256='b'*64,lock_sha256='c'*64,native_quality=True)
    observations=[]
    for dataset in sorted({j['dataset'] for j in plan.to_dict()['jobs']}):
        for index in range(14):
            family=f'P{index:02}'
            for kind,seconds,rss,size in [('anchor',10,100_000_000,10_000),('anchor',20,120_000_000,20_000),('support',1000,150_000_000,30_000)]:
                observations.append(dict(dataset=dataset,family_id=family,kind=kind,
                    process_elapsed_ns=seconds*10**9,observed_peak_root_rss_bytes=rss,retained_logical_bytes=size))
    return plan,observations


def test_full_counts_paid_retries_and_no_retry_finite_support_are_distinct(study):
    module=api();plan,observations=study
    result=module._forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    assert result['counts']==dict(jobs=840,primary=168,secondary=672,paid_attempt_cap=26880,
        primary_paid_attempt_cap=5376,secondary_paid_attempt_cap=21504,
        no_retry_configurations=26700,primary_no_retry_configurations=5340,secondary_no_retry_configurations=21360)
    assert result['projection']['whole_native_anchor_proxy_ns']==840*32*20*10**9
    assert result['projection']['primary_native_anchor_proxy_ns']==168*32*20*10**9
    assert result['projection']['whole_retained_proxy_bytes']==840*32*30_000
    assert result['confidence_quantiles_available'] is False
    assert result['native_qualification_authenticated'] is False
    assert result['protected_access_authorized'] is False
    assert len(result['jobs'])==840 and len(result['envelopes'])==84
    assert not any('p90' in k or 'p50' in k for k in result['projection'])


def test_short_supports_contribute_rss_and_storage_but_never_convergence_time(study):
    module=api();plan,observations=study
    result=module._forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    row=result['envelopes']['airfoil/P13']
    assert row['native_anchor_min_ns']==10*10**9 and row['native_anchor_max_ns']==20*10**9
    assert row['largest_observed_root_rss_bytes']==150_000_000
    assert row['largest_retained_logical_bytes']==30_000
    assert row['transaction_reserve_proxy_bytes']==2*30_000+64*1024**2
    assert row['required_memory_proxy_bytes']==187_500_000+512*1024**2


@pytest.mark.parametrize('fault',['missing','duplicate','bad_elapsed','bad_size','bad_kind','source','plan'])
def test_incomplete_or_changed_metadata_cannot_forecast(study,fault):
    module=api();plan,observations=study
    if fault=='missing':observations=[r for r in observations if (r['dataset'],r['family_id'])!=('airfoil','P13')]
    elif fault=='duplicate':observations[0]['unexpected']=True
    elif fault=='bad_elapsed':observations[0]['process_elapsed_ns']=True
    elif fault=='bad_size':observations[0]['retained_logical_bytes']=-1
    elif fault=='bad_kind':observations[0]['kind']='partial-fit'
    elif fault=='source':
        from treeneuron.research.workload import PracticalPlan
        from treeneuron.config import canonical_json,canonical_hash
        value=plan.to_dict();value['source_sha256']='0'*64
        for job in value['jobs']:job['source_sha256']='0'*64
        value['plan_hash']=canonical_hash({k:v for k,v in value.items() if k!='plan_hash'});plan=PracticalPlan(canonical_json(value))
    with pytest.raises(ValueError):module._forecast_from_observations(plan,'0'*64 if fault=='plan' else plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)


@pytest.mark.parametrize('shortage,reason',[('disk','insufficient_live_disk'),('memory','insufficient_live_memory'),('unknown','unknown_live_memory'),('none','ready_for_next_immutable_job')])
def test_live_decision_preserves_the_job_and_has_no_elapsed_hour_cutoff(study,shortage,reason):
    module=api();plan,observations=study
    forecast=module._forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['job_id']=='airfoil/outer-0/repeat-0/P13')
    live=dict(free_disk_bytes=10*1024**3,available_memory_bytes=2*1024**3,observed_utc_ns=1,
        memory_basis='software fixture; no real RAM guarantee')
    if shortage=='disk':live['free_disk_bytes']=0
    if shortage=='memory':live['available_memory_bytes']=1
    if shortage=='unknown':live['available_memory_bytes']=None
    result=module.native_resource_decision(forecast=forecast,job=job,live=live)
    assert result['reason']==reason and result['admitted']==(shortage=='none')
    assert result['job_id']==job['job_id'] and result['forecast_hash']==forecast['forecast_hash']
    assert result['elapsed_time_cutoff_ns'] is None
    assert result['native_qualification_authenticated'] is False


def test_secondary_projection_does_not_authorize_its_execution(study):
    module=api();plan,observations=study
    forecast=module._forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['tranche']=='secondary')
    with pytest.raises(PermissionError):module.native_resource_decision(forecast=forecast,job=job,
        live=dict(free_disk_bytes=10**15,available_memory_bytes=10**15,observed_utc_ns=1,memory_basis='fixture'))
