"""Hand-calculated sentinel vectors, with no protected file access or models."""
import math

import numpy as np
import pytest


def api():
    from treeneuron.analysis import prediction_arithmetic_v2
    return prediction_arithmetic_v2


def example():
    return dict(F_y=np.array([0.,2.]),y=np.array([0.,2.,1.]),
        row_id=np.array([8,9,10],dtype=np.int64),group_id=np.array([101,101,202],dtype=np.int64),
        grouping='group',timestamps=None,planned_methods=['A','B'],
        statuses={'A':'completed_eligible','B':'completed_eligible'},
        predictions={'A':dict(row_id=np.array([8,9,10],dtype=np.int64),values=np.array([0.,0.,0.])),
                     'B':dict(row_id=np.array([8,9,10],dtype=np.int64),values=np.array([1.,2.,1.]))})


def test_group_risk_weights_groups_equally_and_recomputes_original_unit_metrics():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();out=api().audit_prediction_arrays(**x)
    assert out['status']=='complete_arithmetic' and out['F_target_variance']==1.
    assert out['groups']==2 and out['rows']==3
    assert out['methods']['A']['row_MSE']==pytest.approx(5/3)
    assert out['methods']['A']['primary_MSE']==1.5
    assert out['methods']['A']['normalized_risk']==1.5
    assert out['methods']['A']['row_RMSE']==pytest.approx(math.sqrt(5/3))
    assert out['methods']['A']['row_MAE']==1.
    assert out['methods']['B']['row_MSE']==pytest.approx(1/3)
    assert out['methods']['B']['primary_MSE']==.25
    assert out['risk_inventory_complete'] is True
    assert out['protected_access_authenticated'] is False and out['claim_authorized'] is False


def test_temporal_risk_uses_actual_days_and_preserves_missing_clock_hours():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x.update(grouping='day',group_id=np.array([0,0,1],dtype=np.int64),
        timestamps=np.array(['1970-01-01T00','1970-01-01T02','1970-01-02T00'],dtype='datetime64[h]'))
    out=api().audit_prediction_arrays(**x)
    assert out['groups']==2 and out['methods']['A']['primary_MSE']==1.5
    assert out['timestamps_sha256'] is not None
    x['group_id']=np.array([0,1,1],dtype=np.int64)
    with pytest.raises(ValueError):api().audit_prediction_arrays(**x)


def test_missing_prediction_remains_a_missing_planned_method_without_imputation():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();del x['predictions']['B'];x['statuses']['B']='timeout'
    out=api().audit_prediction_arrays(**x)
    assert out['status']=='incomplete_predictions' and out['risk_inventory_complete'] is False
    assert out['missing_methods']==[{'method_id':'B','status':'timeout'}]
    assert set(out['methods'])=={'A'} and out['methods']['A']['normalized_risk']==1.5


@pytest.mark.parametrize('change',['reordered_rows','duplicate_rows','nonfinite_prediction','extra_method','failed_scored_method'])
def test_misaligned_invalid_or_unplanned_predictions_fail_instead_of_silently_changing_the_comparison(change):
    assert hasattr(api(),'audit_prediction_arrays')
    x=example()
    if change=='reordered_rows':x['predictions']['A']['row_id']=np.array([10,9,8],dtype=np.int64)
    elif change=='duplicate_rows':x['row_id']=np.array([8,8,10],dtype=np.int64)
    elif change=='nonfinite_prediction':x['predictions']['A']['values'][0]=np.nan
    elif change=='extra_method':x['predictions']['C']=x['predictions']['A']
    else:x['statuses']['A']='numerical_failure'
    with pytest.raises(ValueError):api().audit_prediction_arrays(**x)


def test_constant_F_targets_preserve_original_unit_metrics_and_mark_normalization_undefined():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x['F_y']=np.array([1.,1.]);out=api().audit_prediction_arrays(**x)
    assert out['status']=='undefined_F_variance' and out['F_target_variance']==0
    assert out['methods']['A']['primary_MSE']==1.5 and out['methods']['A']['normalized_risk'] is None
    assert out['risk_inventory_complete'] is False


def test_unrepresentable_loss_fails_without_publishing_an_infinite_risk():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x['predictions']['A']['values'][0]=1e308
    with pytest.raises(ValueError):api().audit_prediction_arrays(**x)


def test_row_grouping_requires_distinct_observations_and_matches_plain_MSE():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x['grouping']='row';x['group_id']=x['row_id'].copy()
    out=api().audit_prediction_arrays(**x)
    assert out['groups']==3 and out['methods']['A']['primary_MSE']==pytest.approx(5/3)
    x['group_id']=np.array([101,101,202],dtype=np.int64)
    with pytest.raises(ValueError):api().audit_prediction_arrays(**x)


def test_representable_subnormal_loss_is_not_reported_as_perfect_prediction():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x['predictions']['A']['values']=np.full(3,math.sqrt(5e-324));x['y']=np.zeros(3)
    out=api().audit_prediction_arrays(**x)
    assert out['methods']['A']['row_MSE']==5e-324
    assert out['methods']['A']['primary_MSE']==5e-324


def test_finite_average_survives_an_overflowing_unscaled_sum():
    assert hasattr(api(),'audit_prediction_arrays')
    x=example();x['predictions']['A']['values']=np.full(3,1e154);x['y']=np.zeros(3)
    out=api().audit_prediction_arrays(**x)
    assert out['methods']['A']['row_MSE']==pytest.approx(1e308)
    assert out['methods']['A']['primary_MSE']==pytest.approx(1e308)
