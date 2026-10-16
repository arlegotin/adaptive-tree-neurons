import numpy as np
import pytest

from treeneuron.evaluation.metrics import regression_metrics


def test_normalized_risk_uses_fitting_population_variance():
    result = regression_metrics(np.array([1., 3.]), np.array([2., 2.]), fit_target_variance=4.)
    assert result['mse'] == 1.
    assert result['normalized_mse'] == .25
    assert result['rmse'] == result['mae'] == 1.
    assert result['normalized_mse'] - result['normalized_mse'] == 0.


def test_zero_variance_retains_raw_error_with_reason():
    result = regression_metrics(np.array([1., 1.]), np.array([2., 2.]), fit_target_variance=0.)
    assert result['mse'] == 1.
    assert result['normalized_mse'] is None
    assert result['normalized_mse_reason']


def test_group_macro_and_tail_errors_are_separate_from_row_risk():
    result = regression_metrics(np.zeros(4), np.array([0., 0., 0., 4.]),
                                fit_target_variance=1., groups=np.array(['a', 'a', 'a', 'b']),
                                noisefree=np.array([0., 0., 0., 4.]))
    assert result['mse'] == 4.
    assert result['group_macro_mse'] == 8.
    assert result['worst_group_mse'] == 16.
    assert result['max_absolute_error'] == 4.
    assert result['absolute_error_p95'] == pytest.approx(3.4)
    assert result['noise_free_mse'] == 0.
    assert [g['n'] for g in result['group_metrics']] == [3, 1]


@pytest.mark.parametrize('prediction', [np.zeros((2, 1)), np.zeros(1), [0., np.inf], [0., np.nan]])
def test_invalid_predictions_do_not_broadcast(prediction):
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(2), np.array(prediction), fit_target_variance=1.)


@pytest.mark.parametrize('variance', [-1., np.nan, np.inf, True])
def test_invalid_fitting_variance_rejected(variance):
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(2), np.zeros(2), fit_target_variance=variance)


@pytest.mark.parametrize('groups', [np.array([1.1, 1.2]), np.array([1, '1'], dtype=object),
                                    np.array([None, 'a']), np.array([[1], [2]]), np.array([1])])
def test_unsuitable_group_labels_rejected(groups):
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(2), np.ones(2), fit_target_variance=1., groups=groups)


def test_invalid_targets_and_noise_free_targets_rejected():
    for y in (np.array([]), np.array([[1.]]), np.array([np.nan])):
        with pytest.raises(ValueError):
            regression_metrics(y, np.zeros_like(y), fit_target_variance=1.)
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(2), np.zeros(2), fit_target_variance=1., noisefree=np.zeros(1))


def test_mixed_python_group_labels_are_not_coerced_to_the_same_string():
    with pytest.raises(ValueError):
        regression_metrics(np.zeros(2), np.zeros(2), fit_target_variance=1., groups=[1, '1'])


def test_large_finite_losses_have_finite_aggregate_metrics():
    result = regression_metrics(np.zeros(2), np.full(2, 1e154), fit_target_variance=0.)
    assert np.isfinite(result['mse'])
    assert result['mse'] == pytest.approx(1e308)
