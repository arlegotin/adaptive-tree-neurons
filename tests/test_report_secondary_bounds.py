"""Secondary regions retain endpoint meaning without changing primary gates."""
import copy

import pytest
from test_report_v2 import audit,api


def family():
    return dict(family_id='growth_vs_ordinary',construction='teacher_t_guilbaud2008_choice2_strict',
        planned_hypotheses=['strong','weak'],independent_unit='teacher',n_units=32,
        alpha=.025,margin=.02,status='completed',unavailable_reason=None,
        rows=[dict(hypothesis_id='strong',raw_p=.001,adjusted_p=.002,lower=.02,
            lower_endpoint_excluded=True,rejected=True),
            dict(hypothesis_id='weak',raw_p=.5,adjusted_p=.5,lower=-.11,
            lower_endpoint_excluded=False,rejected=False)])


def test_secondary_open_margin_is_printed_as_strict_assertion_not_a_numeric_gap():
    value=audit();value['secondary_families']=[family()]
    text=api().render_report(value)
    assert 'effect > 0.02' in text and 'effect >= -0.11' in text
    assert 'No numeric gap beyond the margin' in text
    assert 'No protected confirmation' in text
    assert 'growth_vs_ordinary' in text and '32' in text
    assert value['comparisons'][0]['decision']=='not_tested'
    same=copy.deepcopy(value);same['secondary_families'][0]['rows'].reverse()
    assert text==api().render_report(same)


def test_unavailable_family_keeps_every_hypothesis_without_imputed_bounds():
    value=audit();f=family();f.update(status='unavailable_family',unavailable_reason='weak contrast missing')
    for row in f['rows']:
        for key in ('raw_p','adjusted_p','lower','lower_endpoint_excluded','rejected'):row[key]=None
    value['secondary_families']=[f];text=api().render_report(value)
    assert 'weak contrast missing' in text and 'unavailable' in text and 'strong' in text and 'weak' in text


@pytest.mark.parametrize('kind',['missing_row','duplicate_family','unclosed_nonrejection','unopened_rejection',
    'invented_partial_gap','wrong_adjusted_p','wrong_rejection','wrong_unit','unknown_construction','incomplete_unavailable'])
def test_invalid_secondary_regions_cannot_turn_into_confidence_claims(kind):
    value=audit();f=family();value['secondary_families']=[f]
    if kind=='missing_row':f['rows'].pop()
    elif kind=='duplicate_family':value['secondary_families'].append(copy.deepcopy(f))
    elif kind=='unclosed_nonrejection':f['rows'][1].update(lower=.02,lower_endpoint_excluded=True)
    elif kind=='unopened_rejection':f['rows'][0]['lower_endpoint_excluded']=False
    elif kind=='invented_partial_gap':f['rows'][0]['lower']=.021
    elif kind=='wrong_adjusted_p':f['rows'][0]['adjusted_p']=.0001
    elif kind=='wrong_rejection':f['rows'][0]['rejected']=False
    elif kind=='wrong_unit':f['independent_unit']='training repetition'
    elif kind=='unknown_construction':f['construction']='observed_rank_t_bounds'
    else:f.update(status='unavailable_family',unavailable_reason='missing')
    with pytest.raises(ValueError):api().render_report(value)


def test_secondary_support_does_not_relax_the_strict_primary_lower_bound_gate():
    value=audit();value['secondary_families']=[family()];value.update(status='completed',evidence_scope='confirmation',protocol_sha256='4'*64)
    value['provenance']['protected_protocol_authenticated']=True
    value['comparisons'][0].update(decision='passed',estimate=.03,lower=.02,upper=.05,margin=.02,n_units=32)
    with pytest.raises(ValueError):api().render_report(value)


def test_teacher_region_bridge_projects_the_actual_result_without_changing_primary_scope():
    from treeneuron.analysis.primary_v2 import holm_compatible_teacher_bounds
    assert hasattr(api(),'teacher_family_report_record'), 'Missing compatible-region report bridge'
    v=[-3.,-2.,-1.,0.,0.,1.,2.,3.];ids=[f't{i}' for i in range(8)]
    result=holm_compatible_teacher_bounds(ids,{'strong':[x+4 for x in v],'weak':v},planned_hypotheses=['strong','weak'],margin=.02)
    record=api().teacher_family_report_record('ordinary',result);value=audit();value['secondary_families']=[record]
    assert record['n_units']==8 and record['rows'][0]['lower']==.02
    assert 'effect > 0.02' in api().render_report(value)
    assert value['comparisons'][0]['decision']=='not_tested'
    unavailable=holm_compatible_teacher_bounds(ids,{'strong':v},planned_hypotheses=['strong','weak'])
    record=api().teacher_family_report_record('ordinary',unavailable)
    assert record['status']=='unavailable_family' and len(record['rows'])==2
    assert all(row['lower'] is None for row in record['rows'])
