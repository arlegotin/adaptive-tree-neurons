"""Canonical practical-family presentation; no raw or protocol authentication."""
import copy

import pytest

from test_practical_comparisons_v2 import fixture
from test_report_v2 import audit


def api():
    from treeneuron.analysis import report_v2
    assert hasattr(report_v2,'practical_family_report_record'),'Missing practical family report projection'
    return report_v2


@pytest.fixture(scope='module')
def result():
    from treeneuron.analysis.practical_comparisons_v2 import compare_practical_families
    losses,champions=fixture()
    return compare_practical_families(losses,champions=champions,seed=7,iterations=999)


def test_canonical_report_preserves_all_thirteen_tests_and_actual_conditional_units(result):
    module=api();record=module.practical_family_report_record('fixed-practical-families',result)
    value=audit();value['secondary_families']=[record]
    text=module.render_report(value)
    assert 'airfoil=3' in text and 'parkinsons=3' in text and 'bike=21' in text
    assert 'calendar-day block: 7' in text and '999 bootstrap' in text
    assert 'unverified_exchangeability' in text and 'few_subject_clusters' in text
    assert 'teacher t inference' not in text
    assert 'not retraining or dataset-population uncertainty' in text
    assert 'P01' in text and 'P13' in text
    assert len(record['rows'])==13
    assert record['rows'][0]['lower']>0 and not record['rows'][0]['lower_endpoint_excluded']
    assert 'No protected confirmation' in text


def test_unavailable_family_keeps_every_row_without_adjusted_claims(result):
    module=api();value=copy.deepcopy(result)
    value['multiplicity'].update(status='insufficient_resampling_resolution',holm=None,bounds=None)
    record=module.practical_family_report_record('unavailable',value)
    assert record['status']=='unavailable_family'
    assert len(record['rows'])==13 and all(r['lower'] is None for r in record['rows'])
    report=audit();report['secondary_families']=[record]
    text=module.render_report(report)
    assert 'insufficient_resampling_resolution' in text and 'P13' in text


@pytest.mark.parametrize('fault',['family_shrink','pairing','iterations','missing_task','boolean_count'])
def test_projection_rejects_inconsistent_family_or_sampling_metadata(result,fault):
    module=api();value=copy.deepcopy(result)
    if fault=='family_shrink':value['multiplicity']['planned_hypotheses'].pop()
    elif fault=='pairing':value['family_comparisons']['P13']['pairing_index_sha256']['bike']='0'*64
    elif fault=='iterations':value['family_comparisons']['P13']['iterations']+=1
    elif fault=='missing_task':value['champion']['resampling_units'].pop('ccpp')
    else:value['champion']['resampling_units']['parkinsons']=True
    with pytest.raises(ValueError):module.practical_family_report_record('bad',value)


@pytest.mark.parametrize('fault',['endpoint','p','units','assumptions','teacher_label','extra'])
def test_renderer_rejects_false_interpretations_of_practical_region(result,fault):
    module=api();record=module.practical_family_report_record('bad',result)
    if fault=='endpoint':record['rows'][0]['lower_endpoint_excluded']=True
    elif fault=='p':record['rows'][0]['adjusted_p']=1
    elif fault=='units':record['n_units']=5
    elif fault=='assumptions':record['dataset_assumptions'].pop('ccpp')
    elif fault=='teacher_label':record['independent_unit']='teacher'
    else:record['T_authorized']=True
    value=audit();value['secondary_families']=[record]
    with pytest.raises(ValueError):module.render_report(value)


def test_partial_rejection_renders_open_margin_instead_of_positive_gap(result):
    module=api();value=copy.deepcopy(result)
    from treeneuron.analysis.practical_comparisons_v2 import bootstrap_margin_family,MARGIN
    records={f'P{i:02}':dict(point=.1 if i==1 else 0.,draws=[.1 if i==1 else 0.]*999) for i in range(1,14)}
    value['multiplicity']=bootstrap_margin_family(records,planned_hypotheses=list(records))
    record=module.practical_family_report_record('partial',value)
    row=record['rows'][0];assert row['lower']==MARGIN and row['lower_endpoint_excluded']
    report=audit();report['secondary_families']=[record]
    text=module.render_report(report)
    assert 'effect > 0.02020270732' in text
    assert 'No numeric gap beyond the margin' in text
