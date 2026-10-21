"""Canonical report fixtures; no real study, protocol, labels or models."""
import copy
import math

import pytest


def api():
    from treeneuron.analysis import report_v2
    return report_v2


def audit():
    return dict(schema='research/v2-canonical-audit',study_id='fixture-report',status='estimation',
        evidence_scope='development',source=dict(commit='1'*40,package_sha256='2'*64),
        provenance=dict(source_authenticated=True,raw_predictions_authenticated=True,
            protected_protocol_authenticated=False),protocol_sha256=None,
        independent_units='One artificial teacher; two paired training repetitions are not new teachers.',
        planned_prediction_ids=['teacher0/r0/grown','teacher0/r0/random'],
        predictions=[dict(prediction_id='teacher0/r0/random',method_id='random',task_id='teacher0',
            repetition=0,status='completed_eligible',normalized_risk=1.5,normalization_status='defined',
            primary_MSE=1.5,row_RMSE=math.sqrt(5/3),row_MAE=1.),
            dict(prediction_id='teacher0/r0/grown',method_id='grown',task_id='teacher0',
            repetition=0,status='completed_eligible',normalized_risk=.25,normalization_status='defined',
            primary_MSE=.25,row_RMSE=math.sqrt(1/3),row_MAE=1/3)],
        attempts=[dict(attempt_id='paid-anchor',status='completed_ineligible'),
                  dict(attempt_id='paid-failed-trial',status='numerical_failure')],
        comparisons=[dict(comparison_id='placement',decision='not_tested',explanation='No population inference from this fixture.',
            estimate=None,lower=None,upper=None,margin=None,independent_unit='teacher',n_units=1)],
        costs=dict(physical_elapsed_ns=10_000_000_000,standalone_attributed_ns=25_000_000_000,
            scope='Separate containing physical and shared-prerequisite standalone totals; never add them.'),
        limitations=['Artificial arithmetic only.'],sources=[dict(path='audit/input.json',sha256='3'*64)])


@pytest.mark.parametrize('status',['completed','estimation','retired','blocked'])
def test_every_study_status_retains_all_methods_failures_and_evidence_limits(status):
    assert hasattr(api(),'render_report')
    value=audit();value['status']=status
    text=api().render_report(value)
    assert status in text and 'teacher0/r0/grown' in text and 'teacher0/r0/random' in text
    assert 'paid-failed-trial' in text and 'numerical_failure' in text and 'paid-anchor' in text
    assert '0.25' in text and '1.5' in text and 'Artificial arithmetic only.' in text
    assert 'No protected confirmation' in text
    assert '10.000000' in text and '25.000000' in text and '35.000000' not in text
    assert value=={**audit(),'status':status}


def test_rendering_is_byte_stable_and_uses_declared_inventory_order():
    assert hasattr(api(),'render_report')
    first=audit();second=copy.deepcopy(first);second['predictions'].reverse();second['source']=dict(reversed(list(second['source'].items())))
    a=api().render_report(first);b=api().render_report(second)
    assert a==b and a.index('teacher0/r0/grown')<a.index('teacher0/r0/random')


def test_missing_final_prediction_blocks_a_positive_primary_decision_but_remains_reportable():
    assert hasattr(api(),'render_report')
    value=audit();value['status']='blocked';p=value['predictions'][0]
    p.update(status='irrecoverable_missing',normalized_risk=None,primary_MSE=None,row_RMSE=None,row_MAE=None,
        normalization_status='unavailable_prediction')
    text=api().render_report(value)
    assert 'irrecoverable_missing' in text and 'Incomplete planned predictions' in text
    value['evidence_scope']='confirmation';value['protocol_sha256']='4'*64
    value['provenance']['protected_protocol_authenticated']=True
    value['comparisons'][0]['decision']='passed'
    value['comparisons'][0].update(estimate=.05,lower=.03,upper=.07,margin=.02,n_units=32)
    with pytest.raises(ValueError):api().render_report(value)


@pytest.mark.parametrize('change',['missing_inventory_entry','duplicate_prediction','unplanned_prediction','failed_risk','unqualified_confirmation'])
def test_invalid_or_unqualified_inputs_cannot_become_a_complete_report(change):
    assert hasattr(api(),'render_report')
    value=audit()
    if change=='missing_inventory_entry':value['predictions'].pop()
    elif change=='duplicate_prediction':value['predictions'].append(copy.deepcopy(value['predictions'][0]))
    elif change=='unplanned_prediction':value['predictions'][0]['prediction_id']='outside-plan'
    elif change=='failed_risk':value['predictions'][0]['status']='timeout'
    else:value['evidence_scope']='confirmation';value['protocol_sha256']='4'*64
    with pytest.raises(ValueError):api().render_report(value)


def test_publication_is_immutable_and_failed_validation_creates_no_report(tmp_path):
    assert hasattr(api(),'write_report')
    target=tmp_path/'result.md';result=api().write_report(audit(),target)
    first=target.read_bytes();assert result['bytes']==len(first) and len(result['sha256'])==64
    with pytest.raises((ValueError,FileExistsError)):api().write_report(audit(),target)
    assert target.read_bytes()==first
    value=audit();value['status']='invented'
    bad=tmp_path/'bad.md'
    with pytest.raises(ValueError):api().write_report(value,bad)
    assert not bad.exists()


def confirmation_fixture():
    value=audit();value.update(status='completed',evidence_scope='confirmation',protocol_sha256='4'*64)
    value['provenance']['protected_protocol_authenticated']=True
    value['comparisons'][0].update(decision='passed',estimate=.05,lower=.03,upper=.07,margin=.02,n_units=32)
    return value


def test_report_retains_the_supplied_interval_margin_and_independent_unit():
    assert hasattr(api(),'render_report')
    text=api().render_report(confirmation_fixture())
    assert '0.05' in text and '0.03' in text and '0.07' in text and '0.02' in text and 'teacher' in text and '32' in text
    assert 'supplied audit' in text


@pytest.mark.parametrize('change',['equal_margin','below_margin','missing_interval','reversed_interval'])
def test_positive_comparison_requires_a_strict_compatible_supplied_lower_bound(change):
    assert hasattr(api(),'render_report')
    value=confirmation_fixture();c=value['comparisons'][0]
    if change=='equal_margin':c['lower']=.02
    elif change=='below_margin':c['lower']=.01
    elif change=='missing_interval':c['lower']=c['upper']=None
    else:c['lower']=.08
    with pytest.raises(ValueError):api().render_report(value)


def test_undefined_normalization_keeps_valid_predictions_and_original_unit_metrics_visible():
    assert hasattr(api(),'render_report')
    value=audit();p=value['predictions'][0]
    p.update(normalized_risk=None,normalization_status='undefined_F_variance')
    text=api().render_report(value)
    assert 'undefined_F_variance' in text and 'Undefined normalization' in text
    assert 'completed_eligible' in text and '1.5' in text and 'Row RMSE' in text and 'Row MAE' in text
    value.update(status='completed',evidence_scope='confirmation',protocol_sha256='4'*64)
    value['provenance']['protected_protocol_authenticated']=True
    value['comparisons'][0].update(decision='passed',estimate=.05,lower=.03,upper=.07,margin=.02,n_units=32)
    with pytest.raises(ValueError):api().render_report(value)


def test_unverified_source_and_prediction_provenance_are_visible_to_the_report_reader():
    assert hasattr(api(),'render_report')
    value=audit();value['provenance']['source_authenticated']=False
    value['provenance']['raw_predictions_authenticated']=False
    text=api().render_report(value)
    assert 'Source authentication: false.' in text and 'Raw prediction authentication: false.' in text
    assert 'Protected protocol authentication: false.' in text


@pytest.mark.parametrize('empty',['predictions','attempts'])
def test_an_empty_confirmation_inventory_cannot_support_a_positive_decision(empty):
    assert hasattr(api(),'render_report')
    value=confirmation_fixture()
    if empty=='predictions':value['planned_prediction_ids']=[];value['predictions']=[]
    else:value['attempts']=[]
    with pytest.raises(ValueError):api().render_report(value)


def test_completed_report_retains_explicitly_resolved_historical_interruption():
    value=audit();value['status']='completed'
    value['attempts']=[dict(attempt_id='initial',status='interrupted_recoverable'),dict(attempt_id='resume',status='completed_eligible')]
    value['attempt_resolutions']={'initial':'resume'}
    text=api().render_report(value)
    assert 'interrupted_recoverable' in text and 'initial' in text and 'resume' in text
    assert 'Resolved' in text or 'resolved' in text
    value['attempt_resolutions']={'initial':'missing'}
    with pytest.raises(ValueError):api().render_report(value)


def test_resolution_cannot_hide_a_running_or_unresolved_attempt():
    value=audit();value['status']='completed'
    value['attempts']=[dict(attempt_id='initial',status='interrupted_recoverable'),dict(attempt_id='resume',status='running')]
    value['attempt_resolutions']={'initial':'resume'}
    with pytest.raises(ValueError):api().render_report(value)
    value['attempts'][1]['status']='interrupted_recoverable'
    with pytest.raises(ValueError):api().render_report(value)
