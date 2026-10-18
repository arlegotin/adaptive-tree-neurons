"""Gate A is a development screen with exact integrity and one-extension rules."""
import copy

import pytest


def contract():
    from treeneuron.research.opportunity_plan import opportunity_decision_contract
    return opportunity_decision_contract({'seed_root':2026101801,'study_id':'M2-opportunity'})


def summary(*, n=4, gated=2, warm=1):
    from treeneuron.research.opportunity_analysis import rebuild_opportunity_summary
    contrasts=[]
    for contrast in contract()['contrasts']:
        count = gated if contrast['role']=='top' else warm if contrast['role']=='warm_lower' else 0
        rows=[dict(teacher_index=t,arm_V_NMSE=.97 if t<count else .99,
            reference_V_NMSE=1.,reference='matched_keep',origins=['original','original']) for t in range(n)]
        contrasts.append(dict(**contrast,teachers=rows))
    return rebuild_opportunity_summary(dict(teacher_ids=list(range(n)),contrasts=contrasts),contract())


def inputs():
    from treeneuron.research.control_analysis import positive_sensitivity
    rows=[dict(population='positive',gamma=2.,sigma=.1,teacher_index=t,repetition=r,
        role=role,status='completed_eligible',V_NMSE=1. if role=='random' else .9)
        for t in range(4) for r in range(2) for role in ('random','oracle_true')]
    control=dict(status='completed',evidence_level='retained_raw_prediction_arithmetic',
        positive_sensitivity=positive_sensitivity(rows),rows=rows)
    statuses={'nominal_dead_pair':'completed_ineligible','actual_pair':'completed_eligible',
        'local_diagnostic':'completed_eligible','extension_decision:primary':'completed_ineligible'}
    return control,summary(),statuses


def decide(control,initial,final,statuses,extension_status='completed_ineligible'):
    from treeneuron.research.gate_a import gate_a_decision
    return gate_a_decision(control,initial,final,statuses,
        expected_job_ids=['nominal_dead_pair','actual_pair','local_diagnostic','extension_decision:primary'],
        extension_status=extension_status,opportunity_contract=contract())


def test_gate_passes_only_screen_and_warm_only_does_not_claim_gated_opportunity():
    control,initial,statuses=inputs()
    result=decide(control,initial,initial,statuses)
    assert result['decision']=='screen_passed' and result['advance_joint_selector']
    assert result['scope']=='gated_growth_opportunity' and result['efficacy_claim'] is False
    warm=summary(gated=0,warm=2)
    result=decide(control,warm,warm,statuses)
    assert result['decision']=='screen_passed' and result['advance_joint_selector']
    assert result['scope']=='ordinary_capacity_only' and not result['gated_growth_opportunity']


def test_final_gate_requires_completed_declared_extension_and_never_a_third():
    control,_,statuses=inputs();initial=summary(gated=1,warm=0)
    assert initial['extension_required']
    statuses['extension_decision:primary']='completed_eligible'
    assert decide(control,initial,initial,statuses,extension_status='completed_eligible')['decision']=='blocked'
    extended=summary(n=8)
    assert decide(control,initial,extended,statuses,extension_status='completed_eligible')['decision']=='screen_passed'
    original=summary()
    assert decide(control,original,extended,statuses,extension_status='completed_eligible')['decision']=='blocked'
    assert decide(control,initial,dict(extended,extension_required=True),statuses,
        extension_status='completed_eligible')['decision']=='blocked'


def test_any_diagnostic_failure_or_missing_record_blocks_not_complete_case_selection():
    control,initial,statuses=inputs()
    for changed in (dict(statuses,local_diagnostic='numerical_failure'),{'actual_pair':'completed_eligible'}):
        result=decide(control,initial,initial,changed)
        assert result['decision']=='blocked' and not result['advance_joint_selector']
    assert decide(dict(control,status='incomplete_or_ineligible'),initial,initial,statuses)['decision']=='blocked'


def test_insensitive_control_or_absent_capacity_has_its_prescribed_disposition():
    from treeneuron.research.control_analysis import positive_sensitivity
    control,initial,statuses=inputs()
    for row in control['rows']:row['V_NMSE']=1.
    control['positive_sensitivity']=positive_sensitivity(control['rows'])
    result=decide(control,initial,initial,statuses)
    assert result['decision']=='insensitive_control' and not result['advance_joint_selector']
    control,_,statuses=inputs();initial=summary(gated=1,warm=0)
    statuses['extension_decision:primary']='completed_eligible'
    absent=summary(n=8,gated=1,warm=0)
    result=decide(control,initial,absent,statuses,extension_status='completed_eligible')
    assert result['decision']=='retired_regime' and not result['advance_joint_selector']


def test_extension_argument_cannot_override_denied_frozen_node():
    control,_,statuses=inputs()
    result=decide(control,summary(gated=1,warm=0),summary(n=8),statuses,
        extension_status='completed_eligible')
    assert result['decision']=='blocked'


def test_missing_required_extension_cannot_be_hidden_by_edited_summary_flag():
    control,_,statuses=inputs()
    edited=summary(gated=1,warm=0)
    edited['extension_required']=False
    result=decide(control,edited,edited,statuses)
    assert result['decision']=='blocked'


def test_control_summary_must_match_retained_true_oracle_rows():
    control,initial,statuses=inputs()
    control['positive_sensitivity'].update(sensitive=False,qualifying_instances=0)
    assert decide(control,initial,initial,statuses)['decision']=='blocked'


@pytest.mark.parametrize('mutation',[
    lambda s:s.update(maximum_qualifying_teachers=3),
    lambda s:s.update(best_fixed_interval=[.1,.2]),
    lambda s:s['contrasts'][0]['teachers'][0].update(relative_benefit=.7),
    lambda s:s['contrasts'][0]['teachers'][0].update(reference='unregistered'),
    lambda s:s['contrasts'][0]['teachers'][0].update(arm_V_NMSE=-1.),
    lambda s:s['contrasts'][0]['teachers'].pop(),
    lambda s:s['contrasts'].reverse(),
])
def test_derived_arithmetic_or_incomplete_fixed_registry_cannot_pass(mutation):
    control,initial,statuses=inputs();edited=copy.deepcopy(initial);mutation(edited)
    assert decide(control,edited,edited,statuses)['decision']=='blocked'


def test_result_binds_contract_and_extension_input():
    from treeneuron.config import canonical_hash
    control,initial,statuses=inputs();result=decide(control,initial,initial,statuses)
    assert result['input_bindings']['opportunity_contract_sha256']==canonical_hash(contract())
    assert result['input_bindings']['extension_status_sha256']==canonical_hash({'status':'completed_ineligible'})


def test_nonfinite_input_is_rejected_before_canonical_evidence_binding():
    control,initial,statuses=inputs()
    initial['contrasts'][0]['teachers'][0]['arm_V_NMSE']=float('inf')
    with pytest.raises(ValueError,match='finite JSON'):
        decide(control,initial,initial,statuses)
