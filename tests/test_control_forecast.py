"""Finite control forecasts use measured speeds, update ceilings and whole cases."""
import importlib.util
from pathlib import Path

import pytest


def recipe():
    path=Path(__file__).resolve().parents[1]/'docs/research/v2/scripts/admit_control_matrix.py'
    spec=importlib.util.spec_from_file_location('control_forecast',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def fixture():
    attempts=[]
    for population,roles in [('positive',['keep','top','random','oracle_current','oracle_true']),
            ('globally_affine',['keep','top','random'])]:
        for role in roles:attempts.append(dict(cell=population,population=population,
            teacher_index=0,repetition=0,role=role,attempt_id=population+'/'+role))
    plan=dict(config=dict(kind='m1_controls',base_experiment=dict(training=dict(base_updates=10,
        continuation_updates=10))),attempts=attempts,source_sha256='a'*64,execution_identity={})
    samples={key:dict(retained_bytes=retained,max_checkpoint_transaction_bytes=5,peak_rss_bytes=rss,
        overhead_ns=5,stages=[dict(executed_updates=updates,elapsed_ns=elapsed) for _ in range(n)])
        for key,retained,rss,updates,elapsed,n in [('positive',100,123,2,4,5),('negative',200,345,5,15,4)]}
    return plan,samples


def test_forecast_arithmetic_is_not_nested_timer_or_case_count_inflation():
    plan,samples=fixture();record,detail=recipe().forecast(plan,samples,publication_allowance=0)
    assert sorted(record['case_retained_bytes'].values())==[505,630]
    assert record['retry_reserve_bytes']==630 and record['forecast_remaining_bytes']==1765
    assert record['peak_rss_bytes']==345
    assert record['forecast_worker_seconds_p50']==pytest.approx(230/1e9)
    assert record['forecast_worker_seconds_p90']==pytest.approx(285/1e9)
    assert detail['case_counts']==dict(positive=1,negative=1)
    assert detail['future_quantiles_validated'] is False


def test_zero_update_or_missing_native_cases_cannot_invent_a_forecast():
    plan,samples=fixture();samples['positive']['stages'][0]['executed_updates']=0
    with pytest.raises(ValueError,match='measured'):recipe().forecast(plan,samples)
