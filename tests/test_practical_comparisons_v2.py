"""Finite paired conditional comparisons; fixtures grant no model authority."""
import importlib.util
import math

import numpy as np
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.analysis.practical_comparisons_v2'), 'Missing complete practical comparison arithmetic'
    from treeneuron.analysis import practical_comparisons_v2
    return practical_comparisons_v2


def fixture():
    losses={};champions={}
    for name in ('airfoil','concrete','ccpp','parkinsons','bike','energy'):
        n=21 if name=='bike' else 3
        families={f'P{i:02}':np.tile([2.,8.],(n,1)) for i in range(14)}
        families['P00']=np.tile([1.,3.],(n,1));families['P03']=np.tile([20.,6.],(n,1))
        packet=dict(group_ids=list(range(n)),family_losses=families)
        if name=='bike':packet['day_ordinals']=np.arange(100,121)
        losses[name]=packet;champions[name]=['P02','P03']
    return losses,champions


def test_champion_is_the_declared_validation_choice_for_each_repeat_and_energy_stays_separate():
    losses,champions=fixture()
    result=api().compare_practical_families(losses,champions=champions,seed=7,iterations=999,chunk_size=23)
    # Champion columns are [2,6]; average risks4/2 gives log2, not log of
    # repeatwise ratios, and never the test-best family (P03 mean13).
    assert result['champion']['aggregate_point_log_benefit']==pytest.approx(math.log(2))
    assert result['champion']['aggregate_interval']==pytest.approx([math.log(2)]*2)
    assert result['family_comparisons']['P02']['aggregate_point_log_benefit']==pytest.approx(math.log(2.5))
    assert result['primary_conditions']['positive_in_both_repeats'] is True
    assert result['primary_conditions']['nonnegative_task_count']==5
    assert result['primary_conditions']['task_degradation_over_5_percent']==[]
    assert result['multiplicity']['holm']['family_size']==13
    assert result['champion']['energy']['bootstrap_interval'] is None
    assert result['claim_authorized'] is result['protected_access_authorized'] is False
    assert result['champion_V_selection_authenticated'] is False


def test_every_method_and_champion_reuses_identical_group_indices_and_chunking():
    losses,champions=fixture()
    for name,packet in losses.items():
        n=len(packet['group_ids']);packet['family_losses']['P04'][:,0]=np.arange(n)+1
    a=api().compare_practical_families(losses,champions=champions,seed=31,iterations=79,chunk_size=1)
    b=api().compare_practical_families(losses,champions=champions,seed=31,iterations=79,chunk_size=17)
    for key in a['family_comparisons']:
        row=a['family_comparisons'][key]
        assert row['pairing_index_sha256']==a['champion']['pairing_index_sha256']
        np.testing.assert_array_equal(row['aggregate_bootstrap_log_benefits'],b['family_comparisons'][key]['aggregate_bootstrap_log_benefits'])
    assert a['multiplicity']['status']=='insufficient_resampling_resolution'


@pytest.mark.parametrize('fault',['family','champion_family','champion_repeat','negative','missing_task','extra_field'])
def test_incomplete_or_invalid_comparison_cannot_shrink_family_or_select_test_champion(fault):
    losses,champions=fixture()
    if fault=='family':losses['airfoil']['family_losses'].pop('P13')
    elif fault=='champion_family':champions['airfoil'][0]='P01'
    elif fault=='champion_repeat':champions['airfoil'].pop()
    elif fault=='negative':losses['ccpp']['family_losses']['P08'][0,0]=-1
    elif fault=='missing_task':losses.pop('concrete');champions.pop('concrete')
    else:losses['parkinsons']['test_best']='P07'
    with pytest.raises(ValueError):api().compare_practical_families(losses,champions=champions,seed=1,iterations=99)


def test_zero_risk_resamples_make_the_whole_multiplicity_family_unavailable():
    losses,champions=fixture();losses['airfoil']['family_losses']['P13'][:]=0
    result=api().compare_practical_families(losses,champions=champions,seed=1,iterations=999)
    assert result['multiplicity']['status']=='undefined_family'
    assert result['multiplicity']['holm'] is None
    assert result['multiplicity']['unavailable_hypotheses']==['P13']
    assert len(result['family_comparisons'])==13


def test_centered_bootstrap_margin_test_counts_ties_and_has_no_zero_monte_carlo_pvalue():
    module=api();margin=math.log(1/.98)
    # errors[-1,0,0,1], point-margin1 => tied tailcount1, plusone p2/5.
    value=module.bootstrap_margin_family({'A':dict(point=margin+1,draws=[margin,margin+1,margin+1,margin+2])},planned_hypotheses=['A'])
    assert value['marginals']['A']['p']==pytest.approx(.4)
    assert value['status']=='insufficient_resampling_resolution'
    high=module.bootstrap_margin_family({'A':dict(point=1.,draws=[1.]*999)},planned_hypotheses=['A'])
    assert high['holm']['raw_p']['A']==.001
    assert high['bounds']['A']['lower']==1.
    assert high['bounds']['A']['lower_endpoint_excluded'] is False


def test_partial_holm_rejection_yields_open_margin_without_an_invented_effect_size_gap():
    module=api();margin=math.log(1/.98)
    records={'A':dict(point=.2,draws=[.2]*999),'B':dict(point=margin,draws=[margin]*999)}
    value=module.bootstrap_margin_family(records,planned_hypotheses=['A','B'])
    assert value['holm']['rejected']==['A']
    assert value['bounds']['A']==dict(lower=margin,lower_endpoint_excluded=True,rejects_margin_null=True,strictly_sharper_than_margin=False)
    assert value['bounds']['B']['lower']==margin
    assert value['bounds']['B']['lower_endpoint_excluded'] is False


def test_all_rejected_regions_use_the_common_weakest_unadjusted_bound():
    records={'A':dict(point=.2,draws=[.2]*999),'B':dict(point=.1,draws=[.1]*999)}
    value=api().bootstrap_margin_family(records,planned_hypotheses=['A','B'])
    assert value['all_hypotheses_rejected'] is True
    assert value['bounds']['A']['lower']==value['bounds']['B']['lower']==.1
    assert not value['bounds']['A']['lower_endpoint_excluded']


def test_missing_declared_hypothesis_blocks_all_family_inference():
    value=api().bootstrap_margin_family({'A':dict(point=.2,draws=[.2]*999)},planned_hypotheses=['A','B'])
    assert value['status']=='undefined_family'
    assert value['family_size']==2 and value['holm'] is None
    assert value['unavailable_hypotheses']==['B']


def test_discrete_endpoint_inversion_agrees_with_strict_margin_test_at_equality():
    # Integer inputs make the intended tie exactly representable. At the
    # prospective log margin, adding/subtracting48 loses one ULP in draws.
    module=api();margin=1.
    # B99: alpha(B+1)=2.5, largest rejecting tailcount1. The second-largest
    # error97 sets the closed lower endpoint. At equality p3/100 >= .025.
    errors=np.arange(99,dtype=float)-49
    point=margin+48
    value=module.bootstrap_margin_family({'A':dict(point=point,draws=(errors+point).tolist())},planned_hypotheses=['A'],margin=margin)
    assert value['marginals']['A']['p']==pytest.approx(.03)
    assert value['holm']['rejected']==[]
    assert value['bounds']['A']['lower']==pytest.approx(margin)
    above=module.bootstrap_margin_family({'A':dict(point=point+.001,draws=(errors+point+.001).tolist())},planned_hypotheses=['A'],margin=margin)
    assert above['holm']['rejected']==['A']


def test_important_single_task_regression_is_retained_despite_positive_aggregate():
    losses,champions=fixture();losses['airfoil']['family_losses']['P00'][:]=[4.,8.]
    result=api().compare_practical_families(losses,champions=champions,seed=9,iterations=999)
    assert result['champion']['aggregate_point_log_benefit']>0
    assert result['primary_conditions']['task_degradation_over_5_percent']==['airfoil']
    assert not result['primary_conditions']['robustness_conditions_pass']
    assert not result['claim_authorized']


def test_one_bad_repeat_cannot_be_hidden_by_averaging_the_risks():
    losses,champions=fixture()
    for packet in losses.values():packet['family_losses']['P00'][:]=[3.,1.]
    result=api().compare_practical_families(losses,champions=champions,seed=9,iterations=999)
    assert result['champion']['aggregate_point_log_benefit']>0
    assert result['primary_conditions']['positive_in_both_repeats'] is False
    assert result['primary_conditions']['statistical_useful_conditions_pass'] is False


def test_energy_zero_risk_does_not_contaminate_observational_family_inference():
    losses,champions=fixture();losses['energy']['family_losses']['P00'][:]=0
    result=api().compare_practical_families(losses,champions=champions,seed=9,iterations=999)
    assert result['multiplicity']['status']=='completed'
    assert result['champion']['energy']['descriptive_group_effect_range'] is None
    assert result['champion']['aggregate_point_log_benefit']==pytest.approx(math.log(2))


def test_default_method_uses_twenty_thousand_draws_and_sensitivity_cannot_become_primary():
    losses,champions=fixture()
    result=api().compare_practical_families(losses,champions=champions,seed=11,bike_block_days=3)
    assert result['champion']['iterations']==20000
    assert not result['primary_conditions']['primary_seven_day_block']
    assert not result['primary_conditions']['statistical_useful_conditions_pass']
