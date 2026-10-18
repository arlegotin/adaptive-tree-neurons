"""M1's privileged control remains distinct from trained-pair oracle evidence."""
import copy


def rows():
    return [dict(population='positive',gamma=2.,sigma=.1,teacher_index=t,repetition=r,
        role=role,status='completed_eligible',V_NMSE=risk)
        for t,benefit in enumerate((.01,.03,.04,-.02)) for r in range(2)
        for role,risk in (('random',1.),('oracle_true',1-benefit))]


def test_control_sensitivity_uses_all_four_R2_means_and_exact_coordinate():
    from treeneuron.research.control_analysis import positive_sensitivity
    data=rows();before=copy.deepcopy(data)
    # Additional coordinates and another type of oracle cannot change this gate.
    data+= [dict(row,role='pair_oracle',V_NMSE=0.) for row in before]
    data+= [dict(row,gamma=.5,V_NMSE=0.) for row in before]
    result=positive_sensitivity(data)
    assert result['status']=='completed_eligible' and result['sensitive'] is True
    assert result['qualifying_instances']==2 and len(result['instances'])==4
    assert result['information_advantaged'] is True and not result['efficacy_claim']
    assert before==data[:16]


def test_control_never_selects_better_repeat_or_substitutes_an_oracle():
    from treeneuron.research.control_analysis import positive_sensitivity
    data=rows()
    for row in data:
        if row['role']=='oracle_true':row['V_NMSE']=.1 if row['repetition']==0 else 2.
    result=positive_sensitivity(data)
    assert result['qualifying_instances']==0 and result['sensitive'] is False
    data=[dict(row,role='pair_oracle') if row['role']=='oracle_true' else row for row in data]
    assert positive_sensitivity(data)['status']=='irrecoverable_missing'


def test_missing_failed_duplicate_or_invalid_risk_blocks_control_gate():
    from treeneuron.research.control_analysis import positive_sensitivity
    data=rows()
    for bad in (data[:-1],data+[data[0]], [dict(row,status='numerical_failure') if i==0 else row
            for i,row in enumerate(data)], [dict(row,V_NMSE=float('nan')) if i==0 else row
            for i,row in enumerate(data)]):
        result=positive_sensitivity(bad)
        assert result['status']=='irrecoverable_missing' and result['sensitive'] is None


def test_inclusive_two_percent_sensitivity_boundary_is_not_lost_by_subtraction():
    from treeneuron.research.control_analysis import positive_sensitivity
    data=rows();reference=11/997
    for row in data:
        row['V_NMSE']=reference if row['role']=='random' else reference*(.98 if row['teacher_index']<2 else .99)
    result=positive_sensitivity(data)
    assert result['qualifying_instances']==2 and result['sensitive'] is True
