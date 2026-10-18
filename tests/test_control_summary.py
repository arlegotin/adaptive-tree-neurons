"""Complete planted/negative cohorts keep teacher means and failures visible."""
import copy
import math

import pytest


def audit():
    from treeneuron.research.control_analysis import positive_sensitivity
    rows=[]
    cells=[('positive',g,s,('keep','top','random','oracle_current','oracle_true'))
        for g in (.5,2.) for s in (0.,.1)]
    cells += [(p,None,None,('keep','top','random')) for p in
        ('globally_affine','additive_sine_quadratic','pure_noise','random_label_permutation')]
    for population,gamma,sigma,roles in cells:
        for t in range(4):
            for r in range(2):
                for role in roles:
                    ratio={'keep':1.2,'top':.4 if r==0 else 1.54,'random':1.,'oracle_current':.94,'oracle_true':.9}[role]
                    risk=(t+1)*ratio
                    row=dict(population=population,teacher_index=t,repetition=r,role=role,
                        status='completed_eligible',V_NMSE=risk,
                        metrics={k:dict(normalized_mse=risk) for k in ('V','E')})
                    if gamma is not None:row.update(gamma=gamma,sigma=sigma)
                    rows.append(row)
    return dict(schema='research/v2-control-audit',status='completed',rows=rows,
        evidence_level='retained_raw_prediction_arithmetic',T_accessed=False,
        positive_sensitivity=positive_sensitivity(rows))


def test_complete_summary_averages_repeats_before_teacher_risks_and_placement():
    from treeneuron.research.control_summary import summarize_control_audit
    result=summarize_control_audit(audit())
    assert result['status']=='completed_development' and result['planned_final_fits']==256
    assert result['status_counts']=={'completed_eligible':256}
    assert len(result['cells'])==8
    first=result['cells'][0]
    assert first['mean_risk']['E']['top']==pytest.approx(2.5*.97)
    assert first['placement']['E']['mean_teacher_log_benefit']==pytest.approx(math.log(1/.97))
    assert len(first['teachers'])==4 and first['independent_base_instances']==4
    assert result['positive_base_instances_reused_across_cells'] is True
    assert result['efficacy_claim'] is False and result['Gate_A_decided'] is False


def test_failure_remains_in_inventory_and_cannot_become_complete_case_result():
    from treeneuron.research.control_summary import summarize_control_audit
    value=audit();value['status']='incomplete_or_ineligible'
    value['rows'][0]=dict(value['rows'][0],status='numerical_failure')
    result=summarize_control_audit(value)
    assert result['status']=='incomplete_development'
    assert result['status_counts']['numerical_failure']==1
    assert result['cells'][0]['status']=='incomplete'
    assert result['cells'][0]['mean_risk'] is None
    assert result['cells'][0]['missing_coordinates'][0]['role']=='keep'


@pytest.mark.parametrize('mutation',['duplicate','missing','unexpected','inconsistent_metric','protected'])
def test_summary_rejects_inventory_or_audit_contradictions(mutation):
    from treeneuron.research.control_summary import summarize_control_audit
    value=audit()
    if mutation=='duplicate':value['rows'].append(copy.deepcopy(value['rows'][0]))
    elif mutation=='missing':value['rows'].pop()
    elif mutation=='unexpected':value['rows'][0]['teacher_index']=4
    elif mutation=='protected':value['T_accessed']=True
    else:value['rows'][0]['V_NMSE']*=2
    with pytest.raises(ValueError):summarize_control_audit(value)


def test_zero_risks_remain_undefined_log_effects_without_epsilon_imputation():
    from treeneuron.research.control_summary import summarize_control_audit
    value=audit()
    for row in value['rows']:
        if row['role']=='random':
            row['V_NMSE']=0.
            for role in ('V','E'):row['metrics'][role]['normalized_mse']=0.
    from treeneuron.research.control_analysis import positive_sensitivity
    value['positive_sensitivity']=positive_sensitivity(value['rows'])
    result=summarize_control_audit(value)
    assert result['cells'][0]['placement']['V']['mean_teacher_log_benefit'] is None
    assert result['cells'][0]['placement']['V']['undefined_teacher_count']==4
