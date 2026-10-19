"""Holm compatibility requires regions, not rank-substituted t intervals."""
import math
import json

import numpy as np
import pytest
from scipy.stats import t


def api():
    from treeneuron.analysis import primary_v2
    assert hasattr(primary_v2,'holm_compatible_teacher_bounds'), 'Missing compatible secondary-family confidence bounds'
    return primary_v2.holm_compatible_teacher_bounds


def contrasts():
    # Literal residual vectors sum to zero and have sum of squares28; n8, SD2.
    residual=np.array([-3.,-2.,-1.,0.,0.,1.,2.,3.]);se=math.sqrt(4/8)
    return residual,se


def test_partly_rejected_family_has_open_margin_bound_and_refined_unrejected_bound():
    v,se=contrasts();effects={'strong':(v+4).tolist(),'weak':v.tolist(),'harmful':(v-1).tolist()}
    result=api()([f't{i}' for i in range(8)],effects,planned_hypotheses=['strong','weak','harmful'],margin=.02)
    assert result['holm']['rejected']==['strong']
    assert result['bounds']['strong']['lower']==.02 and result['bounds']['strong']['lower_endpoint_excluded'] is True
    assert result['bounds']['weak']['lower']==pytest.approx(-t.ppf(1-.025/2,7)*se)
    assert result['bounds']['harmful']['lower']==pytest.approx(-1-t.ppf(1-.025/2,7)*se)
    assert result['bounds']['strong']['strictly_sharper_than_margin'] is False
    assert result['familywise_confidence']==.975 and result['claim_authorized'] is False


def test_all_rejected_uses_a_joint_minimum_not_unadjusted_separate_intervals():
    v,se=contrasts();effects={'a':(v+5).tolist(),'b':(v+4).tolist()}
    result=api()([f't{i}' for i in range(8)],effects,planned_hypotheses=['a','b'],margin=.02)
    expected=4-float(t.ppf(.975,7))*se
    assert result['holm']['rejected']==['a','b']
    assert result['bounds']['a']['lower']==pytest.approx(expected)
    assert result['bounds']['b']['lower']==pytest.approx(expected)
    assert result['bounds']['a']['strictly_sharper_than_margin'] is True


def test_no_rejection_reduces_to_bonferroni_and_one_contrast_to_ordinary_t():
    v,se=contrasts();ids=[f't{i}' for i in range(8)]
    result=api()(ids,{'a':v.tolist(),'b':(v-1).tolist()},planned_hypotheses=['b','a'],margin=.02)
    assert result['holm']['rejected']==[] and result['bounds']['a']['lower']==pytest.approx(-t.ppf(.9875,7)*se)
    single=api()(ids,{'a':(v+4).tolist()},planned_hypotheses=['a'],margin=.02)
    assert single['bounds']['a']['lower']==pytest.approx(4-t.ppf(.975,7)*se)


def test_missing_or_degenerate_contrast_never_shrinks_the_family():
    v,_=contrasts();ids=[f't{i}' for i in range(8)]
    for effects in ({'a':v.tolist()},{'a':v.tolist(),'b':None},{'a':v.tolist(),'b':[0.]*8}):
        result=api()(ids,effects,planned_hypotheses=['a','b'])
        assert result['status']=='unavailable_family' and result['family_size']==2
        assert result['bounds'] is None and result['holm'] is None


@pytest.mark.parametrize('kind',['bad_length','duplicate_teacher','extra_hypothesis','nonfinite','repeat_matrix','bad_alpha','bad_margin'])
def test_unpaired_or_invalid_teacher_contrasts_are_rejected(kind):
    ids=[f't{i}' for i in range(8)];v,_=contrasts();effects={'a':v.tolist()};kwargs={}
    if kind=='bad_length':effects['a'].pop()
    elif kind=='duplicate_teacher':ids[-1]=ids[0]
    elif kind=='extra_hypothesis':effects['b']=v.tolist()
    elif kind=='nonfinite':effects['a'][0]=float('inf')
    elif kind=='repeat_matrix':effects['a']=[[1.,2.]]*8
    elif kind=='bad_alpha':kwargs['alpha']=True
    else:kwargs['margin']=float('nan')
    with pytest.raises(ValueError):api()(ids,effects,planned_hypotheses=['a'],**kwargs)


def test_correlated_normal_fixture_has_simultaneous_coverage_without_independence_between_contrasts(tmp_path):
    # Known population effects independently determine coverage; expected bounds
    # are not computed by duplicating the region formula. Fixed MC check is a
    # software sensitivity fixture, not the prospective calibration/power gate.
    rng=np.random.default_rng(3801);ids=[f't{i}' for i in range(8)]
    covered=0;count=3000;truth=np.array([.09,.02,-.03]);names=['a','b','c']
    for _ in range(count):
        common=rng.normal(size=(8,1));individual=rng.normal(size=(8,3))
        values=truth+.04*(np.sqrt(.7)*common+np.sqrt(.3)*individual)
        result=api()(ids,{name:values[:,i].tolist() for i,name in enumerate(names)},planned_hypotheses=names,margin=.02)
        covered+=all((truth[i]>result['bounds'][name]['lower'] if result['bounds'][name]['lower_endpoint_excluded'] else truth[i]>=result['bounds'][name]['lower']) for i,name in enumerate(names))
    (tmp_path/'coverage.json').write_text(json.dumps(dict(seed=3801,cohorts=count,teacher_count=8,effects=truth.tolist(),cross_contrast_correlation=.7,covered=covered,coverage=covered/count,required_fixture_floor=.965,not_prospective_power_gate=True),sort_keys=True)+'\n')
    assert covered/count>=.965


def test_strict_holm_uses_closed_unrejected_endpoints_and_only_opens_the_rejection_margin():
    v,_=contrasts();ids=[f't{i}' for i in range(8)]
    partial=api()(ids,{'a':(v+4).tolist(),'b':v.tolist()},planned_hypotheses=['a','b'],margin=0.)
    assert partial['bounds']['a']['lower_endpoint_excluded'] is True
    assert partial['bounds']['b']['lower_endpoint_excluded'] is False
    # At a strict p==alpha boundary there is no rejection; its region must not
    # imply positivity merely by excluding an equal lower margin endpoint.
    boundary=api()(ids,{'b':v.tolist()},planned_hypotheses=['b'],margin=0.,alpha=.5)
    assert boundary['holm']['raw_p']['b']==.5 and boundary['holm']['rejected']==[]
    assert boundary['bounds']['b']['lower_endpoint_excluded'] is False
    all_reject=api()(ids,{'a':(v+5).tolist(),'b':(v+4).tolist()},planned_hypotheses=['a','b'])
    assert all_reject['bounds']['a']['lower']>all_reject['margin']
    assert all_reject['bounds']['a']['lower_endpoint_excluded'] is False
