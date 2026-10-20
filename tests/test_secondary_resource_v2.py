"""Released secondary resource advice remains separate from workload qualification."""
import importlib.util

import pytest

from test_native_resource_forecast_v2 import study


def api():
    assert importlib.util.find_spec('treeneuron.research.secondary_resource_v2'),'Missing secondary resource route'
    from treeneuron.research import secondary_resource_v2
    return secondary_resource_v2


@pytest.mark.parametrize('memory,disk,reason',[
    (2*1024**3,2*1024**3,'ready_for_next_immutable_job'),
    (1,2*1024**3,'insufficient_live_memory'),
    (None,2*1024**3,'unknown_live_memory'),
    (2*1024**3,1,'insufficient_live_disk')])
def test_secondary_advice_uses_same_proxies_without_20GiB_requirement_or_changing_job(study,memory,disk,reason):
    module=api();plan,observations=study
    from treeneuron.research.native_resource_forecast_v2 import _forecast_from_observations,native_resource_decision
    forecast=_forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    job=next(j for j in plan.to_dict()['jobs'] if j['job_id']=='airfoil/outer-1/repeat-0/P13')
    live=dict(available_memory_bytes=memory,free_disk_bytes=disk,observed_utc_ns=1,memory_basis='unqualified software fixture')
    result=module._secondary_resource_decision(forecast=forecast,job=job,live=live)
    assert result['reason']==reason and result['admitted']==(reason=='ready_for_next_immutable_job')
    assert result['required_memory_proxy_bytes']==724370912 and result['required_disk_proxy_bytes']==1140940688
    assert result['job_id']==job['job_id'] and result['forecast_hash']==forecast['forecast_hash']
    assert result['elapsed_time_cutoff_ns'] is None and result['execution_authorized'] is result['native_qualification_authenticated'] is False
    assert result['protected_access_authorized'] is result['T_accessed'] is False
    with pytest.raises(PermissionError):native_resource_decision(forecast=forecast,job=job,live=live)


def test_secondary_advice_cannot_use_primary_or_changed_forecast_job(study):
    module=api();plan,observations=study
    from treeneuron.research.native_resource_forecast_v2 import _forecast_from_observations
    forecast=_forecast_from_observations(plan,plan.to_dict()['plan_hash'],observations,free_disk_margin_bytes=1024**3)
    primary=next(j for j in plan.to_dict()['jobs'] if j['tranche']=='primary')
    live=dict(available_memory_bytes=2*1024**3,free_disk_bytes=2*1024**3,observed_utc_ns=1,memory_basis='fixture')
    with pytest.raises(PermissionError):module._secondary_resource_decision(forecast=forecast,job=primary,live=live)
    secondary=next(j for j in plan.to_dict()['jobs'] if j['tranche']=='secondary')
    with pytest.raises(ValueError):module._secondary_resource_decision(forecast=forecast,job=dict(secondary,fit_seed=19),live=live)


def test_public_resource_route_needs_actual_global_gate_before_forecast_or_live_data(tmp_path):
    module=api()
    with pytest.raises((ValueError,FileNotFoundError)) as error:
        module.secondary_resource_decision(barrier_reference=dict(path=str(tmp_path/'absent-barrier'),sha256='a'*64),
            qualification_reference=dict(root=str(tmp_path/'absent-registry'),inventory_sha256='b'*64),job_id='airfoil/outer-1/repeat-0/P13',
            resource_forecast_reference=dict(path=str(tmp_path/'forbidden-forecast'),sha256='c'*64),live=None)
    assert 'forbidden-forecast' not in str(error.value) and list(tmp_path.iterdir())==[]
