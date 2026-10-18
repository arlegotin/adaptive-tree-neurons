"""Prospective variance calibration must never borrow a favorable pilot mean."""
import math
import numpy as np
import pytest


def api():
    from treeneuron.analysis import design_v2
    return design_v2


def test_teacher_log_risk_averages_two_risks_before_log_not_repeat_effects():
    rows=[dict(teacher_id='t0',repetition=r,role=role,risk=value,status='completed_eligible')
        for r,pair in enumerate(((1.,1.),(9.,3.))) for role,value in zip(('random','selected'),pair)]
    result=api().teacher_log_benefits(rows)
    assert len(result)==1 and result[0]['teacher_id']=='t0'
    assert result[0]['random_risk']==5. and result[0]['selected_risk']==2.
    assert result[0]['log_benefit']==pytest.approx(math.log(2.5),abs=1e-15)


@pytest.mark.parametrize('change',['missing','duplicate','failure','zero','badrepeat'])
def test_missing_or_undefined_teacher_pair_cannot_be_hidden(change):
    rows=[dict(teacher_id='t',repetition=r,role=role,risk=1.,status='completed_eligible')
        for r in range(2) for role in ('random','selected')]
    if change=='missing':rows.pop()
    elif change=='duplicate':rows.append(rows[0])
    elif change=='failure':rows[0]['status']='numerical_failure'
    elif change=='zero':rows[0]['risk']=0.;rows[2]['risk']=0.
    else:rows[0]['repetition']=True
    with pytest.raises(ValueError):api().teacher_log_benefits(rows)


def test_calibration_and_starting_size_ignore_effect_sign_and_mean():
    values=np.array([-.03,-.01,.01,.03]*4)
    module=api();forecast=dict(p50_seconds=100.,p90_seconds=150.)
    a=module.calibrate_variance(values,runtime_prediction=forecast,seed=42)
    b=module.calibrate_variance(values+.2,runtime_prediction=forecast,seed=42)
    c=module.calibrate_variance(-values,runtime_prediction=forecast,seed=42)
    assert a['teacher_count']==16 and a['bootstrap_resamples']==10000
    assert a['s_U']==pytest.approx(b['s_U'],abs=1e-15)
    assert a['s_U']==pytest.approx(c['s_U'],abs=1e-15)
    assert a['extension_required']==b['extension_required']==c['extension_required']
    assert a['s_U']==max(a['normal_theory_SD_upper_90'],a['bootstrap_SD_percentile_90'])
    assert module.starting_teacher_count(a['s_U'])==module.starting_teacher_count(b['s_U'])
    assert module.starting_teacher_count(.01)['K']==32
    assert module.starting_teacher_count(.1)['K']==112
    assert module.starting_teacher_count(1.)['K'] is None
    assert a['freeze_eligible'] is False


def test_zero_variance_and_runtime_failure_get_one_extension_then_estimation():
    module=api()
    for n in (16,32):
        zero=module.calibrate_variance(np.zeros(n),runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=7)
        assert zero['extension_required'] is (n==16)
        assert zero['status']==('extend_once' if n==16 else 'estimation_only')
        assert zero['variance_identifiable'] is False
        noisy_runtime=module.calibrate_variance(np.linspace(-.02,.02,n),
            runtime_prediction=dict(p50_seconds=1.,p90_seconds=2.01),seed=8)
        assert noisy_runtime['extension_required'] is (n==16)
        assert noisy_runtime['status']==('extend_once' if n==16 else 'estimation_only')


@pytest.mark.parametrize('effects,forecast',[(np.ones(15),dict(p50_seconds=1.,p90_seconds=1.)),
    (np.full(16,np.nan),dict(p50_seconds=1.,p90_seconds=1.)),
    (np.ones(16),dict(p50_seconds=2.,p90_seconds=1.)),(np.ones(16),dict(p50_seconds=0.,p90_seconds=1.))])
def test_invalid_calibration_cannot_reach_sample_size(effects,forecast):
    with pytest.raises(ValueError):api().calibrate_variance(effects,runtime_prediction=forecast,seed=1)


def test_exact_binomial_bounds_do_not_accept_point_estimate_alone():
    module=api()
    interval=module.binomial_interval(0,100000)
    assert interval[0]==0.
    assert interval[1]==pytest.approx(1-.025**(1/100000),rel=1e-11)
    assert module.binomial_interval(100000,100000)[1]==1.
    assert module.binomial_interval(90000,100000)[0]<.9


def test_full_fixed_power_simulation_has_declared_laws_margin_and_chunk_invariance():
    module=api();effects=[-.01,.01]*8
    a=module.simulate_t_gate(effects,sd_upper=.01,K=32,seed=123,chunk_size=4096)
    b=module.simulate_t_gate(effects,sd_upper=.01,K=32,seed=123,chunk_size=777)
    assert a['simulated_cohorts_per_law']==100000
    assert set(a['laws'])=={'empirical','normal','t5'}
    for law,row in a['laws'].items():
        assert row==b['laws'][law]
        assert row['power']['lower95']>=.9 and row['null']['upper95']<=.03
    assert a['finite_t_power']>.99 and a['validation_passed'] is True
    assert a['freeze_eligible'] is False


def test_sparse_noop_mixture_can_fail_t_validation_and_cannot_change_analysis():
    result=api().simulate_t_gate([0.]*15+[-1.],sd_upper=.1,K=32,seed=88)
    assert result['laws']['empirical']['null']['upper95']>.03
    assert result['validation_passed'] is False
    assert result['status']=='failed_at_fixed_K'
    assert result['alternative_analysis_admitted'] is False


@pytest.mark.parametrize('kwargs',[dict(K=16,sd_upper=.01),dict(K=32,sd_upper=0.),dict(K=True,sd_upper=.01)])
def test_power_rejects_unfrozen_size_or_unidentifiable_variance(kwargs):
    with pytest.raises(ValueError):api().simulate_t_gate([-.01,.01]*8,seed=7,**kwargs)


def test_size_selection_validates_fixed_grid_and_cannot_freeze_or_shrink_margin():
    module=api();effects=np.array([-.007,-.002,.002,.007]*4)
    calibration=module.calibrate_variance(effects,runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=3)
    result=module.choose_teacher_count(calibration,effects,seed=4,feasible_maximum=32)
    assert result['K']==32 and result['status']=='power_validated_pending_resources'
    assert [s['K'] for s in result['simulations']]==[32]
    assert result['freeze_eligible'] is False and result['resource_validated'] is False
    large=module.calibrate_variance(effects*100,runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=3)
    estimation=module.choose_teacher_count(large,effects*100,seed=4,feasible_maximum=32)
    assert estimation['K']==32 and estimation['status']=='estimation_only' and estimation['simulations']==[]
    changed={**calibration,'s_U':calibration['s_U']/2}
    with pytest.raises(ValueError):module.choose_teacher_count(changed,effects,seed=4)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_adjacent_values_have_centered_empirical_null_and_translation_safe_SD(dtype):
    module=api();lo=dtype(1.);hi=np.nextafter(lo,dtype(2.),dtype=dtype)
    values=np.array([lo,hi]*8,dtype=dtype)
    expected_effects=np.array([-1.,1.]*8)*(float(hi)-float(lo))/2
    forecast=dict(p50_seconds=1.,p90_seconds=1.)
    got=module.calibrate_variance(values,runtime_prediction=forecast,seed=42)
    expected=module.calibrate_variance(expected_effects,runtime_prediction=forecast,seed=42)
    assert got['sample_SD']==pytest.approx(expected['sample_SD'],rel=1e-14)
    assert got['s_U']==pytest.approx(expected['s_U'],rel=1e-14)
    a=module.simulate_t_gate(values,sd_upper=.01,K=32,seed=123)
    b=module.simulate_t_gate([-1.,1.]*8,sd_upper=.01,K=32,seed=123)
    assert a['laws']['empirical']==b['laws']['empirical']
    assert a['laws']['empirical']['null']['upper95']<=.03


def test_unneeded_or_uncommitted_calibration_extension_cannot_be_admitted():
    module=api();forecast=dict(p50_seconds=1.,p90_seconds=1.)
    effects=np.array([-.007,-.002,.002,.007]*8)
    ready16=module.calibrate_variance(effects[:16],runtime_prediction=forecast,seed=3)
    assert ready16['extension_required'] is False
    complete=module.calibrate_variance(effects,runtime_prediction=forecast,seed=3)
    with pytest.raises(ValueError):module.choose_teacher_count(complete,effects,seed=4)
    with pytest.raises(ValueError):module.choose_teacher_count(complete,effects,seed=4,initial_calibration=ready16)
    authorized=module.calibrate_variance(effects[:16],runtime_prediction=dict(p50_seconds=1.,p90_seconds=3.),seed=3)
    result=module.choose_teacher_count(complete,effects,seed=4,initial_calibration=authorized,feasible_maximum=32)
    assert result['extension_lineage_validated'] is True
    # Variance-identical changed prefixes are still a different calibration.
    changed=-effects
    other=module.calibrate_variance(changed,runtime_prediction=forecast,seed=3)
    with pytest.raises(ValueError):module.choose_teacher_count(other,changed,seed=4,initial_calibration=authorized)


def test_teacher_aggregation_preserves_declared_prefix_order_without_lexical_ID_guessing():
    rows=[dict(teacher_id=t,repetition=r,role=role,risk=1.,status='completed_eligible')
        for t in ('t2','t10') for r in range(2) for role in ('random','selected')]
    assert [r['teacher_id'] for r in api().teacher_log_benefits(rows)]==['t2','t10']


def test_zero_variance_does_not_relabel_constant_nonzero_effects_as_noops():
    result=api().calibrate_variance(np.ones(16),runtime_prediction=dict(p50_seconds=1.,p90_seconds=1.),seed=7)
    assert result['variance_identifiable'] is False
    assert result['all_teacher_effects_exactly_zero'] is False
