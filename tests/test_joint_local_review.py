"""Saved local probes cannot turn repeated fits or chosen fractions into evidence."""
import copy
import importlib

import pytest


ROLES=('local_singleton','joint','unconditioned','pure_span')


def records():
    rows=[]
    for teacher in range(8):
        for repeat in (0,1):
            for role in ROLES:
                # Average repetition losses before computing each teacher ratio.
                base=1. if repeat==0 else 9.
                after=2. if repeat==0 else 4.
                values={k:dict(linearized_MSE=after/2,nonlinear_MSE=after,
                    ReLU_mask_changed_fraction=.1,gate_saturation_fraction=.2) for k in ('F','G')}
                diagnostic=dict(current_parent_MSE={'F':base,'G':base},
                    fractions=[1/16,1/4,1],fraction_selected_on_G=False,
                    local_updates_applied_to_parent=False,
                    linear_solve={'matched_Schur_objective_error':1e-15},
                    evaluations=[dict(fraction=f,**copy.deepcopy(values)) for f in (1/16,1/4,1)])
                rows.append(dict(teacher_index=teacher,repetition=repeat,role=role,diagnostic=diagnostic))
    return rows


def review(rows):
    module=importlib.import_module('treeneuron.research.joint_analysis')
    function=getattr(module,'summarize_joint_local_checks',None)
    assert callable(function),'Missing saved-local-model review'
    return function(rows,teacher_ids=list(range(8)))


def test_local_review_retains_all_fractions_and_averages_repetitions_before_ratios():
    result=review(records())
    assert result['local_diagnostic_count']==64 and result['teachers']==8
    assert result['T_accessed'] is result['Gate_B_decided'] is result['fresh_nonlinear_replay'] is False
    for row in result['by_role']['joint']:
        assert row['F']['mean_teacher_nonlinear_relative_reduction']==pytest.approx(.4)
        assert row['G']['mean_teacher_linear_relative_reduction']==pytest.approx(.7)
        assert row['G']['nonlinear_improving_teachers']==8
        assert row['G']['mean_gate_saturation_fraction']==pytest.approx(.2)
    assert len(result['teacher_rows'])==8*4*3


@pytest.mark.parametrize('mutation',['missing','duplicate','fraction','selected','applied','nan','role','repeat','teacher'])
def test_unpaired_changed_or_nonfinite_local_checks_cannot_silently_enter_the_review(mutation):
    rows=records()
    if mutation=='missing':rows.pop()
    elif mutation=='duplicate':rows.append(copy.deepcopy(rows[0]))
    elif mutation=='fraction':rows[0]['diagnostic']['evaluations'].pop()
    elif mutation=='selected':rows[0]['diagnostic']['fraction_selected_on_G']=True
    elif mutation=='applied':rows[0]['diagnostic']['local_updates_applied_to_parent']=True
    elif mutation=='nan':rows[0]['diagnostic']['evaluations'][0]['G']['nonlinear_MSE']=float('nan')
    elif mutation=='role':rows[0]['role']='random'
    elif mutation=='repeat':rows[0]['repetition']=2
    else:rows[0]['teacher_index']=8
    with pytest.raises(ValueError):review(rows)
