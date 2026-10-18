"""Shared prerequisites and nested clocks must not manufacture cheap procedures."""
import copy

import numpy as np
import pytest


def api():
    from treeneuron.analysis import cost_v2
    return cost_v2


def timers():
    return [dict(timer_id='cohort',parent_id=None,start_ns=0,end_ns=110,requires=[]),
        dict(timer_id='parent',parent_id='cohort',start_ns=10,end_ns=40,requires=[]),
        dict(timer_id='selected',parent_id='cohort',start_ns=40,end_ns=95,requires=['parent']),
        dict(timer_id='score',parent_id='selected',start_ns=40,end_ns=45,requires=[]),
        dict(timer_id='continue',parent_id='selected',start_ns=45,end_ns=95,requires=[]),
        dict(timer_id='random',parent_id='cohort',start_ns=95,end_ns=110,requires=['parent'])]


def test_standalone_procedures_each_pay_shared_parent_but_physical_cohort_counts_once():
    result=api().attribute_costs(timers(),{'selected':['selected'],'random':['random']},cohort_timer='cohort')
    assert result['physical_cohort_wall_ns']==110
    assert result['standalone_procedures']['selected']['charged_ns']==85
    assert result['standalone_procedures']['random']['charged_ns']==45
    assert result['standalone_attributed_sum_ns']==130
    assert result['unattributed_cohort_overhead_ns']==10
    assert result['full_pipeline_cost_qualified'] is False


def test_repeated_prerequisite_is_paid_once_within_each_standalone_procedure():
    values=timers();values[-1]['requires']=['parent','score']
    # Score is inside selected and random runs afterward; selected includes it.
    result=api().attribute_costs(values,{'combined':['selected','random','parent']},cohort_timer='cohort')
    assert result['standalone_procedures']['combined']['charged_ns']==100
    assert set(result['standalone_procedures']['combined']['charged_roots'])=={'parent','selected','random'}


@pytest.mark.parametrize('change',['missing_timer','cycle','outside_parent','double_count','overlap','negative_clock'])
def test_contradictory_cost_graph_cannot_be_used_for_efficiency(change):
    values=timers();procedures={'selected':['selected']}
    if change=='missing_timer':values[2]['requires']=['absent']
    elif change=='cycle':values[1]['requires']=['selected']
    elif change=='outside_parent':values[3]['end_ns']=120
    elif change=='double_count':procedures['selected']=['selected','continue']
    elif change=='overlap':values[-1]['start_ns']=90;procedures['selected']=['selected','random']
    else:values[0]['start_ns']=-1
    with pytest.raises(ValueError):api().attribute_costs(values,procedures,cohort_timer='cohort')


def timing_packet(selected=50,baseline=100,n=30,seed=71):
    from treeneuron.research.seeds import seed64
    rng=np.random.default_rng(seed64(seed,'paired-deployment-timing','order'))
    order=rng.integers(0,2,size=n)
    return [dict(block_id=i,order='selected_first' if order[i]==0 else 'baseline_first',
        selected_ns=selected,baseline_ns=baseline) for i in range(n)]


def test_paired_savings_requires_lower_bound_above_twenty_percent_and_quality_below_1_02():
    result=api().paired_timing_savings(timing_packet(),seed=71,quality_risk_ratio_upper=1.01,iterations=64)
    assert result['savings_fraction']==.5 and result['savings_fraction_interval']==[.5,.5]
    assert result['paired_time_saving_interval_ns']==[50.,50.]
    assert result['statistical_savings_conditions_pass'] is True
    assert result['cost_claim_authorized'] is False
    boundary=api().paired_timing_savings(timing_packet(selected=80),seed=71,quality_risk_ratio_upper=1.01,iterations=64)
    assert boundary['statistical_savings_conditions_pass'] is False
    quality=api().paired_timing_savings(timing_packet(),seed=71,quality_risk_ratio_upper=1.02,iterations=64)
    assert quality['statistical_savings_conditions_pass'] is False


def test_cost_savings_is_ratio_of_means_with_paired_chunk_invariant_resampling():
    rows=timing_packet()
    for i,row in enumerate(rows):row.update(selected_ns=5,baseline_ns=10 if i%2==0 else 90)
    a=api().paired_timing_savings(rows,seed=71,quality_risk_ratio_upper=1.,iterations=127,chunk_size=1)
    b=api().paired_timing_savings(rows,seed=71,quality_risk_ratio_upper=1.,iterations=127,chunk_size=23)
    assert a['savings_fraction']==.9
    assert a['bootstrap_index_sha256']==b['bootstrap_index_sha256']
    np.testing.assert_array_equal(a['bootstrap_savings_fractions'],b['bootstrap_savings_fractions'])
    assert a['savings_fraction_interval']==b['savings_fraction_interval']
    assert a['block_count']==30


@pytest.mark.parametrize('change',['too_few','changed_order','duplicate_block','zero_cost','noninteger_cost','missing_field','invalid_quality'])
def test_unqualified_timing_inputs_cannot_pass_the_cost_criterion(change):
    rows=timing_packet();quality=1.
    if change=='too_few':rows.pop()
    elif change=='changed_order':rows[0]['order']='baseline_first' if rows[0]['order']=='selected_first' else 'selected_first'
    elif change=='duplicate_block':rows[1]['block_id']=0
    elif change=='zero_cost':rows[0]['selected_ns']=0
    elif change=='noninteger_cost':rows[0]['selected_ns']=50.
    elif change=='missing_field':del rows[0]['baseline_ns']
    else:quality=float('nan')
    with pytest.raises(ValueError):api().paired_timing_savings(rows,seed=71,quality_risk_ratio_upper=quality,iterations=64)


def test_nested_scoring_prerequisite_is_charged_even_without_duplicate_edge_on_container():
    values=timers();values[2]['requires']=[];values[3]['requires']=['parent']
    result=api().attribute_costs(values,{'selected':['selected']},cohort_timer='cohort')
    assert result['standalone_procedures']['selected']['charged_ns']==85
    assert set(result['standalone_procedures']['selected']['charged_roots'])=={'parent','selected'}
