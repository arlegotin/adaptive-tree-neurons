"""Pure design fixtures, not independent scientific pilot evidence."""
from dataclasses import replace
import math
import numpy as np
import pytest

from treeneuron.pilot.design import PilotSummary, choose_design, summarize_pilot


def constraints():
    return dict(max_seconds=1e8, max_storage_bytes=10**12, available_memory_bytes=10**10,
        peak_memory_bytes=10**8, setup_seconds_per_teacher=2., seconds_per_repetition=1.,
        storage_bytes_per_teacher=1000, storage_bytes_per_repetition=500,
        seconds_per_evaluation_row=0.001, bytes_per_evaluation_row=16,
        retries=1, estimation_rows=100, alpha=.05, families=1,
        inference_supported=True, power_iterations=1000, bootstrap_seed=31,
        endpoint_tolerance=.1, power_mc_tolerance=.05,
        required_controls=['B1','B2','B3','B4','B5','B6'],
        policies=dict(parent_updates=10,continuation_updates=10,patience=5,
            validation_interval=1,logical_batch=8,physical_batch=8,
            bootstrap_iterations=2000,probe_updates=1,hpo_seconds=0.,hpo_trials=0,
            timing_blocks=20,timing_relative_precision=.1),
        evidence_hashes={'fixture':'a'*64})


def summary(**overrides):
    base=dict(independent_teachers=8,repetitions=2,teacher_means=tuple(.1+x*.001 for x in range(8)),
        repetition_variance=.00001,paired_row_variance_upper=.01,
        between_teacher_variance_lower=.00001,sd_interval=(.005,.01),
        batch_runtimes=(10.,11.),h1_mean=.1,h1_interval=(.09,.11),
        static_mean=.02,static_interval=(-.01,.05),endpoint_mc_error=.001,
        controls=('B1','B2','B3','B4','B5','B6'),failed_pairs=0,
        evidence_hashes={'fixture':'a'*64},purpose='scientific')
    return PilotSummary(**(base|overrides))


@pytest.mark.parametrize('sign',[1,-1])
def test_resolved_h1_absolute_effect_and_completed_pilot_floor(sign):
    s=summary(h1_mean=sign*.1,h1_interval=(.09,.11) if sign==1 else (-.11,-.09))
    d=choose_design(s,constraints())
    assert d.delta_plan==.05
    assert d.K_calc==1 and d.K==d.pilot_teachers==8
    assert d.n_T==100000
    assert d.forecast['floor_extra_seconds']>0
    assert d.branch=='powered' and d.power['iterations']==1000


def test_static_fallback_and_both_unresolved_estimation():
    s=summary(h1_mean=0.,h1_interval=(-.01,.01),static_interval=(-.05,-.01),static_mean=-.03)
    assert choose_design(s,constraints()).delta_plan==.015
    d=choose_design(replace(s,static_interval=(-.05,.01)),constraints())
    assert d.branch=='estimation' and d.K==8 and d.repetitions==2 and d.delta_plan is None


@pytest.mark.parametrize('change',[
    {'independent_teachers':1,'teacher_means':(.1,)}, {'independent_teachers':7,'teacher_means':(.1,)*7},
    {'repetitions':1}, {'sd_interval':(.001,.01)}, {'batch_runtimes':(10.,13.)},
    {'batch_runtimes':(10.,)}, {'endpoint_mc_error':1.}, {'failed_pairs':1}])
def test_unaccepted_calibration_cannot_authorize_powered(change):
    assert choose_design(summary(**change),constraints()).branch in ('calibrate','blocked')


@pytest.mark.parametrize('change',[{'sd_interval':(0.,0.)}, {'sd_interval':None},
    {'between_teacher_variance_lower':0.}, {'between_teacher_variance_lower':None}])
def test_zero_or_undefined_variance_routes_estimation(change):
    d=choose_design(summary(**change),constraints())
    assert d.branch=='estimation' and d.K_calc is None and d.n_T==100


def test_infeasible_floor_does_not_truncate_and_missing_costs_block():
    c=constraints(); c['max_seconds']=1.
    d=choose_design(summary(),c)
    assert d.branch=='blocked' and d.K==8 and d.K_calc==1
    c=constraints(); del c['seconds_per_repetition']
    assert choose_design(summary(),c).branch=='blocked'
    assert choose_design(summary(controls=('B5','B4')),constraints()).branch=='blocked'


def test_missing_policies_and_adjusted_level_reject_and_unsupported_inference_estimates():
    c=constraints(); del c['policies']['patience']
    assert choose_design(summary(),c).branch=='blocked'
    c=constraints(); c['families']=2
    with pytest.raises(ValueError,match='adjusted'): choose_design(summary(),c)
    c=constraints(); c['inference_supported']=False
    assert choose_design(summary(),c).branch=='estimation'


def records():
    return [dict(teacher_id=f't{i}',repetition=j,data_hash=f'd{i}',evaluation_hash=f'e{i}',
        recipe_hash='common',h1_rows=[i*.1+j*.02+.01*k for k in range(5)],
        static_rows=[i*.05+j*.01+.02*k for k in range(5)],
        controls=['B1','B2','B3','B4','B5','B6'],status='completed')
        for i in range(8) for j in range(2)]


def test_hierarchy_averages_repetitions_and_bootstraps_whole_teachers():
    r=records(); s=summarize_pilot(r,batch_runtimes=(1.,1.),bootstrap_iterations=2000,seed=7)
    assert s.independent_teachers==8 and s.repetitions==2
    assert s.teacher_means==pytest.approx([i*.1+.03 for i in range(8)])
    assert s.repetition_variance==pytest.approx(.0002)
    assert s.h1_mean==pytest.approx(.38)
    assert s.sd_interval[0]>0 and s.endpoint_mc_error>=0
    assert s==summarize_pilot(r,batch_runtimes=(1.,1.),bootstrap_iterations=2000,seed=7)
    assert s.diagnostics['student_t_df']==7
    assert s.diagnostics['effective_independent_count']==8


@pytest.mark.parametrize('mutation',['duplicate','data','recipe','missing'])
def test_duplicate_units_or_changed_inner_data_and_recipe_rejected(mutation):
    r=records()
    if mutation=='duplicate': r.append(r[0])
    elif mutation=='data': r[1]['evaluation_hash']='different'
    elif mutation=='recipe': r[1]['recipe_hash']='different'
    else: r.pop()
    with pytest.raises(ValueError): summarize_pilot(r,batch_runtimes=(1.,1.),bootstrap_iterations=2000,seed=7)


def test_engineering_is_never_freeze_eligible():
    d=choose_design(summary(purpose='engineering'),constraints())
    assert d.branch=='blocked' and not d.freeze_eligible


@pytest.mark.parametrize('field,value',[('independent_teachers',True),('repetitions',2.5),
    ('failed_pairs',-1),('paired_row_variance_upper',-.1),('repetition_variance',-.1),
    ('h1_interval',(.2,.1)),('purpose','invented')])
def test_invalid_summary_cannot_reach_numeric_design(field,value):
    with pytest.raises(ValueError): choose_design(summary(**{field:value}),constraints())


@pytest.mark.parametrize('section,key,value',[
    ('policies','parent_updates',1.5),('policies','physical_batch',4),
    ('policies','timing_relative_precision',2.),('constraints','endpoint_tolerance',0.),
    ('constraints','inference_supported','yes'),('constraints','power_mc_tolerance',-1.)])
def test_numeric_policy_domains_are_enforced(section,key,value):
    c=constraints()
    if section=='policies': c['policies'][key]=value
    else: c[key]=value
    with pytest.raises(ValueError): choose_design(summary(),c)


def test_shared_E_correlation_does_not_fake_extra_rows():
    r=records()
    for record in r:
        # Identical row shape in repeated evaluations: average retains variance.
        record['h1_rows']=[float(k) for k in range(5)]
    s=summarize_pilot(r,batch_runtimes=(1.,1.),bootstrap_iterations=1000,seed=1)
    assert s.paired_row_variance_upper==pytest.approx(2.5)
    assert s.between_teacher_variance_lower==0.


def test_direct_numeric_decision_rejects_missing_selected_policies():
    d=choose_design(summary(),constraints())
    with pytest.raises(ValueError,match='polic'): replace(d,numeric_policies={})


def test_failed_rows_are_retained_as_failure_count():
    r=records()+[dict(teacher_id='failed',repetition=0,status='failed')]
    s=summarize_pilot(r,batch_runtimes=(1.,1.),bootstrap_iterations=1000,seed=1)
    assert s.failed_pairs==1 and s.independent_teachers==8


def test_repetition_tradeoff_records_variance_per_cost_recommendation():
    d=choose_design(summary(repetition_variance=.1),constraints())
    trade=d.forecast['repetition_tradeoff']
    assert trade['recommended_repetitions']>2
    assert trade['selected']==2
    assert trade['requires_new_calibration']


def test_design_retains_numeric_inference_and_retry_policy_even_for_estimation():
    c=constraints(); d=choose_design(summary(h1_interval=(-1.,1.),static_interval=(-1.,1.)),c)
    assert d.statistical_policy['alpha']==.05
    assert d.statistical_policy['bootstrap_seed']==31
    assert d.statistical_policy['retries']==1
    assert d.statistical_policy['power_iterations']==1000
