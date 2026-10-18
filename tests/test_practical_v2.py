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
