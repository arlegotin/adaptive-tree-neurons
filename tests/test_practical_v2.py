"""Finite-suite arithmetic and V selection use clusters and paired repetitions."""
import copy
import math

import numpy as np
import pytest


def api():
    from treeneuron.analysis import practical_v2
    return practical_v2


def test_group_equal_risk_differs_from_row_equal_and_preserves_both_repeats():
    y=np.zeros(4);pred={'P00':np.array([[2.,0.],[0.,2.],[0.,2.],[0.,2.]])}
    actual=api().grouped_risks(y,pred,np.array(['a','b','b','b']),fit_target_variance=2.)
    assert actual['group_ids']==['str:a','str:b']
    assert actual['group_sizes']==[1,3]
    np.testing.assert_array_equal(actual['methods']['P00']['group_repeat_nmse'],[[2.,0.],[0.,2.]])
    assert actual['methods']['P00']['repeat_risk']==[1.,1.]
    assert actual['methods']['P00']['mean_repeat_risk']==1.
    assert actual['methods']['P00']['row_repeat_nmse']==[.5,1.5]
    assert actual['methods']['P00']['row_repeat_mae']==[.5,1.5]


@pytest.mark.parametrize('change',['missing_repeat','nonfinite','zero_variance','bad_groups'])
def test_invalid_metric_input_cannot_silently_drop_rows_or_repeats(change):
    y=np.array([0.,1.]);pred={'P00':np.zeros((2,2))};groups=np.array([1,2]);variance=1.
    if change=='missing_repeat':pred['P00']=np.zeros((2,1))
    elif change=='nonfinite':pred['P00'][0,0]=float('nan')
    elif change=='zero_variance':variance=0.
    else:groups=np.array([1.,float('nan')])
    with pytest.raises(ValueError):api().grouped_risks(y,pred,groups,fit_target_variance=variance)


def records():
    return [dict(attempt_id='a',attempt_index=0,family_id='P03',search_repeat=0,status='completed_eligible',
        completion_ns=20,V_score=1.+5e-13,charged_ns=20),
        dict(attempt_id='b',attempt_index=1,family_id='P03',search_repeat=0,status='completed_eligible',
        completion_ns=40,V_score=1.,charged_ns=20),
        dict(attempt_id='c',attempt_index=2,family_id='P03',search_repeat=0,status='completed_eligible',
        completion_ns=100,V_score=.1,charged_ns=60),
        dict(attempt_id='d',attempt_index=3,family_id='P03',search_repeat=0,status='numerical_failure',
        completion_ns=110,V_score=None,charged_ns=10)]


def test_prefix_does_not_admit_late_winner_or_erase_failed_cost():
    actual=api().select_completed_trial(records(),deadline_ns=40)
    assert actual['winner']['attempt_id']=='a'  # Earliest trial within tolerance of actual minimum.
    assert actual['total_charged_ns']==110 and actual['attempts_paid']==4
    assert actual['prefix_charged_ns']==40 and actual['eligible_at_prefix']==2
    assert api().select_completed_trial(records(),deadline_ns=100)['winner']['attempt_id']=='c'
    assert api().select_completed_trial(records(),deadline_ns=19)['winner'] is None


def test_repetitions_cannot_become_candidates_for_best_seed_selection():
    rows=records();rows[1]['search_repeat']=1
    with pytest.raises(ValueError,match='repeat'):api().select_completed_trial(rows,deadline_ns=100)


def test_portfolio_uses_V_lowest_family_ties_and_pays_all_twelve_searches():
    families={f'P{i:02}':dict(status='completed_eligible',V_score=1.,total_charged_ns=i,
        search_repeat=0,attempt_id=f'trial-{i}') for i in range(2,14)}
    result=api().select_champion(families)
    assert result['family_id']=='P02' and result['portfolio_charged_ns']==90
    assert result['portfolio_family_count']==12 and result['equal_total_cost_comparison'] is False
    missing=copy.deepcopy(families);missing['P08']['status']='completed_ineligible'
    blocked=api().select_champion(missing)
    assert blocked['status']=='incomplete_portfolio' and blocked['family_id'] is None
    assert blocked['portfolio_charged_ns']==90


def test_suite_averages_R2_risks_before_logs_and_keeps_energy_separate():
    data={name:dict(selected=[1.,3.],baseline=[2.,8.]) for name in
        ('airfoil','concrete','ccpp','parkinsons','bike')}
    data['energy']=dict(selected=[100.,100.],baseline=[1.,1.])
    actual=api().practical_benefit(data)
    assert actual['primary_mean_log_benefit']==pytest.approx(math.log(2.5))
    assert actual['repeat_mean_log_benefits']==pytest.approx([math.log(2.),math.log(8./3.)])
    assert actual['energy_log_benefit']==pytest.approx(math.log(.01))
    assert actual['nonnegative_task_count']==5 and not actual['task_degradation_over_5_percent']
    assert actual['confirmed_useful_win'] is False  # Point arithmetic cannot confirm an interval claim.
    del data['bike']
    with pytest.raises(ValueError,match='five'):api().practical_benefit(data)


def test_zero_risk_is_not_hidden_by_an_epsilon_in_log_claim():
    data={name:dict(selected=[0.,0.],baseline=[1.,1.]) for name in
        ('airfoil','concrete','ccpp','parkinsons','bike')}
    with pytest.raises(ValueError,match='positive'):api().practical_benefit(data)


def test_typed_cluster_ids_do_not_merge_integer_with_string():
    actual=api().grouped_risks(np.zeros(2),{'P00':np.zeros((2,2))},[1,'1'],fit_target_variance=1.)
    assert actual['group_ids']==['int:1','str:1'] and actual['group_sizes']==[1,1]


def test_missing_paid_attempt_cannot_be_disguised_as_complete_search():
    with pytest.raises(ValueError,match='attempt inventory'):
        api().select_completed_trial(records()[1:],deadline_ns=110)


def test_finite_inputs_whose_residual_overflows_fail_instead_of_reporting_infinite_risk():
    with pytest.raises(ValueError,match='overflows'):
        api().grouped_risks(np.array([-1e308]),{'P00':np.array([[1e308,1e308]])},[1],fit_target_variance=1.)


def test_exact_five_percent_degradation_is_not_labeled_more_than_five_percent():
    data={name:dict(selected=[1.05,1.05],baseline=[1.,1.]) for name in
        ('airfoil','concrete','ccpp','parkinsons','bike')}
    assert api().practical_benefit(data)['task_degradation_over_5_percent']==[]
    data['bike']['selected']=[1.05001,1.05001]
    assert api().practical_benefit(data)['task_degradation_over_5_percent']==['bike']


def bootstrap_fixture():
    result={}
    for dataset in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        n=10 if dataset=='bike' else 3
        baseline=np.repeat(np.arange(1,n+1,dtype=float)[:,None],2,axis=1)
        result[dataset]=dict(group_ids=[str(i) for i in range(n)],selected=baseline/2,baseline=baseline)
        if dataset=='bike':result[dataset]['day_ordinals']=np.arange(100,110)
    return result


def test_paired_conditional_bootstrap_keeps_constant_risk_ratio_and_energy_descriptive():
    result=api().paired_group_bootstrap(bootstrap_fixture(),seed=7,iterations=64,chunk_size=9)
    assert result['aggregate_interval']==pytest.approx([math.log(2),math.log(2)])
    assert result['aggregate_point_log_benefit']==pytest.approx(math.log(2))
    assert result['energy']['bootstrap_interval'] is None
    assert result['energy']['descriptive_group_effect_range']==pytest.approx([math.log(2),math.log(2)])
    assert result['resampling_units']['parkinsons']==3
    assert result['population_superiority_confirmed'] is False
    assert 'unverified_exchangeability' in result['dataset_assumptions']['ccpp']


def test_bootstrap_pairing_is_chunk_invariant_and_exactly_equal_models_have_zero_interval():
    data=bootstrap_fixture()
    for value in data.values():value['selected']=value['baseline'].copy()
    a=api().paired_group_bootstrap(data,seed=31,iterations=63,chunk_size=1)
    b=api().paired_group_bootstrap(data,seed=31,iterations=63,chunk_size=17)
    assert a['aggregate_interval']==[0.,0.] and a['pairing_index_sha256']==b['pairing_index_sha256']
    np.testing.assert_array_equal(a['aggregate_bootstrap_log_benefits'],b['aggregate_bootstrap_log_benefits'])
    assert a['dataset_intervals']==b['dataset_intervals']


def test_bootstrap_uses_actual_days_and_does_not_fill_calendar_gaps():
    data=bootstrap_fixture();data['bike']['day_ordinals'][5:]+=1
    with pytest.raises(ValueError,match='consecutive'):
        api().paired_group_bootstrap(data,seed=1,iterations=32)
    data=bootstrap_fixture()
    with pytest.raises(ValueError,match='block'):
        api().paired_group_bootstrap(data,seed=1,iterations=32,bike_block_days=14)


def test_bootstrap_averages_repeats_before_log_and_never_discards_undefined_resamples():
    data=bootstrap_fixture()
    for value in data.values():
        n=len(value['group_ids']);value['selected']=np.tile([1.,3.],(n,1));value['baseline']=np.tile([2.,8.],(n,1))
    result=api().paired_group_bootstrap(data,seed=2,iterations=32)
    assert result['aggregate_interval']==pytest.approx([math.log(2.5),math.log(2.5)])
    data=bootstrap_fixture();data['airfoil']['selected'][:]=0.
    result=api().paired_group_bootstrap(data,seed=2,iterations=32)
    assert result['status']=='undefined_log_risk' and result['aggregate_interval'] is None
    assert result['undefined_aggregate_resamples']==32 and result['discarded_resamples']==0


def test_calendar_blocks_are_contiguous_and_leave_one_group_sensitivity_is_reported():
    data=bootstrap_fixture()
    data['airfoil']['selected']=np.repeat(np.array([1.,3.,9.])[:,None],2,axis=1)
    data['airfoil']['baseline']=np.repeat(np.array([2.,3.,9.])[:,None],2,axis=1)
    result=api().paired_group_bootstrap(data,seed=19,iterations=32)
    indices=result['first_resampled_group_indices']['bike']
    assert len(indices)==10 and np.array_equal(np.diff(indices[:7]),np.ones(6))
    assert np.array_equal(np.diff(indices[7:]),np.ones(2))
    assert indices[0]<=3 and indices[7]<=3
    assert result['leave_one_group_out']['airfoil']['effect_range']==pytest.approx([0.,math.log(1.25)])
    assert result['leave_one_group_out']['airfoil']['undefined_count']==0


def test_declared_default_bootstrap_retains_twenty_thousand_artificial_draws():
    result=api().paired_group_bootstrap(bootstrap_fixture(),seed=11)
    assert result['iterations']==20000 and len(result['aggregate_bootstrap_log_benefits'])==20000
    assert result['discarded_resamples']==0 and not result['protected_access_authorized']
