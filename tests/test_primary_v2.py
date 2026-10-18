"""Prospective teacher inference never treats repeats as independent problems."""
import copy
import math

import numpy as np
import pytest
from scipy.stats import t


def api():
    from treeneuron.analysis import primary_v2
    return primary_v2


def risks(effects,*,baseline='random'):
    ids=[f'teacher-{i}' for i in range(len(effects))];rows=[]
    for name,effect in zip(ids,effects):
        for repeat in (0,1):
            for role,risk in [('selected',1.),(baseline,math.exp(effect))]:
                rows.append(dict(teacher_id=name,repetition=repeat,role=role,risk=risk,status='completed_eligible'))
    return ids,rows


def test_teacher_t_interval_and_margin_p_use_four_teachers_not_eight_repeats():
    ids,rows=risks([0.,.02,.04,.06])
    result=api().analyze_teacher_risks(ids,rows,seed=3,iterations=64)
    # Mean .03, unbiased variance .002/3, SE sqrt(.002/12), df3.
    se=math.sqrt(.002/12);delta=math.log(1/.98)
    expected=[.03-float(t.ppf(.975,3))*se,.03+float(t.ppf(.975,3))*se]
    assert result['teacher_count']==4 and result['student_t_df']==3
    assert result['student_t_interval']==pytest.approx(expected)
    assert result['one_sided_margin_p']==pytest.approx(t.sf((.03-delta)/se,3))
    assert result['useful_margin_decision']=='inconclusive'
    assert result['teacher_signs']==dict(positive=3,zero=1,negative=0)
    assert result['confirmation_count_valid'] is False
    assert result['protected_access_authorized'] is False and result['claim_authorized'] is False


def test_risks_are_averaged_before_log_and_planned_teacher_order_is_retained():
    ids,rows=risks([.1,.2,.3]);ids=[ids[2],ids[0],ids[1]]
    for row in rows:
        if row['teacher_id']=='teacher-0':
            row['risk']=[1.,3.][row['repetition']] if row['role']=='selected' else [2.,8.][row['repetition']]
    result=api().analyze_teacher_risks(ids,list(reversed(rows)),seed=2,iterations=64)
    assert [r['teacher_id'] for r in result['teachers']]==ids
    assert result['teachers'][1]['log_benefit']==pytest.approx(math.log(2.5))
    assert result['teacher_count']==3


@pytest.mark.parametrize('change',['missing_repeat','missing_teacher','failed_cell'])
def test_missing_planned_risk_blocks_inference_instead_of_dropping_a_case(change):
    ids,rows=risks([.1,.2,.3,.4])
    if change=='missing_repeat':rows.pop()
    elif change=='missing_teacher':rows=[r for r in rows if r['teacher_id']!=ids[-1]]
    else:rows[-1].update(status='numerical_failure',risk=None)
    result=api().analyze_teacher_risks(ids,rows,seed=4,iterations=64)
    assert result['status']=='incomplete_predictions' and result['student_t_interval'] is None
    assert result['teacher_count']==4 and result['bootstrap_interval'] is None
    assert result['useful_margin_decision']=='unavailable' and result['missing_or_failed_cells']


@pytest.mark.parametrize('change',['duplicate','undeclared','wrong_role','nonfinite','negative','extra_repeat','duplicate_plan'])
def test_invalid_or_unplanned_risks_do_not_enter_the_analysis(change):
    ids,rows=risks([.1,.2,.3])
    if change=='duplicate':rows.append(copy.deepcopy(rows[-1]))
    elif change=='undeclared':rows[-1]['teacher_id']='unplanned'
    elif change=='wrong_role':rows[-1]['role']='test_best'
    elif change=='nonfinite':rows[-1]['risk']=float('nan')
    elif change=='negative':rows[-1]['risk']=-1.
    elif change=='extra_repeat':rows[-1]['repetition']=2
    else:ids.append(ids[-1])
    with pytest.raises(ValueError):api().analyze_teacher_risks(ids,rows,seed=3,iterations=64)


def test_exact_no_op_teachers_stay_in_mixed_samples_but_all_equal_t_variance_is_undefined():
    ids,rows=risks([0.,0.,.1,.2])
    result=api().analyze_teacher_risks(ids,rows,seed=1,iterations=64)
    assert result['teacher_count']==4 and result['teacher_signs']['zero']==2
    assert result['student_t_interval'] is not None
    ids,rows=risks([0.]*32)
    result=api().analyze_teacher_risks(ids,rows,seed=1,iterations=64)
    assert result['status']=='degenerate_teacher_variance' and result['one_sided_margin_p'] is None
    assert result['student_t_interval'] is None and result['bootstrap_interval']==[0.,0.]
    assert result['useful_margin_decision']=='undefined_degenerate_variance'


def test_zero_risk_is_not_floored_into_a_finite_log_effect():
    ids,rows=risks([.1,.2,.3])
    for row in rows:
        if row['teacher_id']==ids[0] and row['role']=='selected':row['risk']=0.
    result=api().analyze_teacher_risks(ids,rows,seed=1,iterations=64)
    assert result['status']=='undefined_log_risk' and result['student_t_interval'] is None
    assert result['undefined_teacher_ids']==[ids[0]]


def test_whole_teacher_bootstrap_draws_and_indices_are_chunk_invariant():
    ids,rows=risks([0.,.01,.03,.06])
    a=api().analyze_teacher_risks(ids,rows,seed=912,iterations=127,chunk_size=1)
    b=api().analyze_teacher_risks(ids,rows,seed=912,iterations=127,chunk_size=33)
    assert a['bootstrap_index_sha256']==b['bootstrap_index_sha256']
    np.testing.assert_array_equal(a['bootstrap_mean_log_benefits'],b['bootstrap_mean_log_benefits'])
    assert a['bootstrap_interval']==b['bootstrap_interval']
    assert a['bootstrap_unit']=='whole teacher with both repetition risks averaged first'
    assert a['student_t_interval']!=a['bootstrap_interval']


def test_holm_adjustment_has_monotone_stepdown_and_does_not_infer_a_smaller_family():
    # Ordered p=.01,.02,.021; multiplied by3,2,1 => .03,.04,.021,
    # stepdown running max => .03,.04,.04; at alpha.025 nothing rejects.
    actual=api().holm_family({'b':.02,'c':.021,'a':.01},planned_hypotheses=['a','b','c'])
    assert actual['adjusted_p']==pytest.approx(dict(a=.03,b=.04,c=.04))
    assert actual['rejected']==[]
    success=api().holm_family({'a':.001,'b':.01,'c':.9},planned_hypotheses=['a','b','c'])
    assert success['adjusted_p']==pytest.approx(dict(a=.003,b=.02,c=.9))
    assert success['rejected']==['a','b']
    assert actual['claim_authorized'] is False


@pytest.mark.parametrize('change',['missing','null','nonfinite','negative','too_large','duplicate_plan','extra'])
def test_multiplicity_requires_every_declared_hypothesis_even_when_one_is_unavailable(change):
    values=dict(a=.001,b=.01,c=.2);planned=['a','b','c']
    if change=='missing':del values['c']
    elif change=='null':values['c']=None
    elif change=='nonfinite':values['c']=float('nan')
    elif change=='negative':values['c']=-.1
    elif change=='too_large':values['c']=1.01
    elif change=='duplicate_plan':planned.append('a')
    else:values['d']=.01
    with pytest.raises(ValueError):api().holm_family(values,planned_hypotheses=planned)


def test_favorable_bootstrap_cannot_replace_inconclusive_primary_t_interval():
    ids,rows=risks([.021,.1,.1])
    result=api().analyze_teacher_risks(ids,rows,seed=431,iterations=20000)
    assert result['bootstrap_interval'][0]>math.log(1/.98)
    assert result['student_t_interval'][0]<math.log(1/.98)
    assert result['useful_margin_decision']=='inconclusive'


def test_valid_confirmation_count_does_not_grant_protected_or_claim_authority():
    ids,rows=risks([.045,.055]*16,baseline='warm_ordinary')
    result=api().analyze_teacher_risks(ids,rows,baseline_role='warm_ordinary',seed=1,iterations=64)
    assert result['confirmation_count_valid'] is True
    assert result['useful_margin_decision']=='above_useful_margin'
    assert result['baseline_role']=='warm_ordinary' and result['one_sided_margin_p']<.025
    assert result['claim_authorized'] is False and result['raw_metrics_authenticated'] is False


def test_exact_holm_alpha_boundary_cannot_satisfy_strict_useful_margin():
    actual=api().holm_family({'only':.025},planned_hypotheses=['only'])
    assert actual['adjusted_p']=={'only':.025} and actual['rejected']==[]
