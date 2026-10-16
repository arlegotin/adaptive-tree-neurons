"""Fitted statistics must never depend on selection or protected labels."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from treeneuron.data.transforms import FittedTransform


def test_population_scaling_and_original_target_units():
    X = np.array([[1., 7.], [3., 7.]])
    transform = FittedTransform.fit(X, np.array([10., 14.]))
    np.testing.assert_allclose(transform.transform_X(X), [[-1., 0.], [1., 0.]])
    np.testing.assert_allclose(transform.transform_y(np.array([10., 14.])), [-1., 1.])
    np.testing.assert_allclose(transform.inverse_y(np.array([-1., 1.])), [10., 14.])
    assert transform.to_state()['columns'][1]['constant'] is True
    assert transform.to_state()['columns'][1]['scale'] == 1.


def test_missingness_policy_preserves_complete_synthetic_width():
    transform = FittedTransform.fit(np.ones((3, 16)), np.ones(3))
    assert transform.transform_X(np.ones((2, 16))).shape == (2, 16)
    future = np.ones((1, 16))
    future[0, 0] = np.nan
    np.testing.assert_array_equal(transform.transform_X(future), np.zeros((1, 16)))
    np.testing.assert_array_equal(transform.transform_y(np.ones(3)), np.zeros(3))


def test_median_imputation_all_missing_and_fixed_indicators():
    X = np.array([[1., np.nan], [np.nan, np.nan], [5., np.nan]])
    transform = FittedTransform.fit(X, np.arange(3.))
    result = transform.transform_X(X)
    assert result.shape == (3, 4)
    np.testing.assert_allclose(result[:, 0], [-np.sqrt(1.5), 0., np.sqrt(1.5)])
    np.testing.assert_array_equal(result[:, 1:], [[0., 0., 1.], [0., 1., 1.], [0., 0., 1.]])
    col = transform.to_state()['columns'][1]
    assert col['fill'] == 0. and col['all_missing'] and col['indicator']


def test_categories_unknowns_and_exact_schema_roundtrip():
    X = pd.DataFrame({'amount': pd.Series([1, None, 3], dtype='Float64'),
                      'kind': pd.Categorical(['b', 'a', None])})
    transform = FittedTransform.fit(X, np.arange(3.))
    future = pd.DataFrame({'amount': [2.], 'kind': ['unseen']})
    assert transform.transform_X(future)[0, 1] == 0.
    state = json.loads(json.dumps(transform.to_state(), allow_nan=False))
    restored = FittedTransform.from_state(state)
    np.testing.assert_array_equal(restored.transform_X(X), transform.transform_X(X))
    with pytest.raises(ValueError):
        restored.transform_X(X[['kind', 'amount']])
    with pytest.raises(ValueError):
        restored.transform_X(X.to_numpy())
    state['columns'][0]['fill'] = 999.
    assert restored.to_state()['columns'][0]['fill'] == 2.


def test_transform_state_depends_only_on_fitting_arrays():
    # These are dummy local copies, not observations from a protected evaluator.
    partitions = {p: {'X': np.array([[1.], [3.]]), 'y': np.array([4., 8.])}
                  for p in 'FGVT'}
    before = FittedTransform.fit(partitions['F']['X'], partitions['F']['y']).to_state()
    changed = copy.deepcopy(partitions)
    for p in 'GVT':
        changed[p]['y'][:] = [1e9, -1e9]
    assert before == FittedTransform.fit(changed['F']['X'], changed['F']['y']).to_state()


@pytest.mark.parametrize('y', [[1., np.nan], [1., np.inf], [[1.], [2.]], [1.]])
def test_bad_fitting_labels_rejected(y):
    with pytest.raises(ValueError):
        FittedTransform.fit(np.ones((2, 1)), np.array(y))


@pytest.mark.parametrize('X', [np.empty((0, 1)), np.ones(2), np.array([[np.inf]])])
def test_bad_feature_arrays_rejected(X):
    with pytest.raises(ValueError):
        FittedTransform.fit(X, np.ones(len(X)))


@pytest.mark.parametrize('change', ['extra', 'scale', 'mean', 'version'])
def test_malformed_transform_state_rejected(change):
    state = FittedTransform.fit(np.array([[1.], [3.]]), np.array([1., 3.])).to_state()
    if change == 'extra':
        state['unexpected'] = 1
    elif change == 'scale':
        state['columns'][0]['scale'] = 0.
    elif change == 'mean':
        state['target_mean'] = np.nan
    else:
        state['version'] = 42
    with pytest.raises(ValueError):
        FittedTransform.from_state(state)


def test_transform_rejects_malformed_targets_features_and_categories():
    transform = FittedTransform.fit(np.ones((2, 1)), np.array([1., 2.]))
    for y in (np.array([[1.]]), np.array([np.nan])):
        with pytest.raises(ValueError):
            transform.transform_y(y)
        with pytest.raises(ValueError):
            transform.inverse_y(y)
    with pytest.raises(ValueError):
        transform.transform_X(np.ones((2, 2)))
    with pytest.raises(ValueError):
        FittedTransform.fit(pd.DataFrame({'a': [{'nested': 1}]}), np.ones(1))


def test_numpy_integer_categories_keep_typed_codes_and_empty_prediction_shape():
    X = pd.DataFrame({'kind': pd.Categorical([np.int64(2), np.int64(3)])})
    transform = FittedTransform.fit(X, np.array([1., 3.]))
    future = pd.DataFrame({'kind': pd.Categorical([np.int64(99), np.int64(2)])})
    np.testing.assert_array_equal(transform.transform_X(future), [[0.], [1.]])
    assert transform.transform_X(future.iloc[:0]).shape == (0, 1)


def test_numeric_schema_does_not_silently_coerce_string_features():
    X = pd.DataFrame({'x': [1., 2.]})
    transform = FittedTransform.fit(X, np.array([1., 2.]))
    with pytest.raises(ValueError):
        transform.transform_X(pd.DataFrame({'x': ['1', '2']}))


@pytest.mark.parametrize('name', ['y', 'noise_free_target', 'gates', 'teacher', 'reference_mean',
                                   'reference_std', 'row_id', 'group_id', 'dates'])
def test_project_record_metadata_cannot_be_passed_as_predictors(name):
    with pytest.raises(ValueError, match='metadata'):
        FittedTransform.fit(pd.DataFrame({'amount': [1., 2.], name: [3., 4.]}), np.array([1., 2.]))


def test_saved_schema_cannot_reintroduce_project_metadata():
    state = FittedTransform.fit(pd.DataFrame({'x': [1., 2.]}), np.array([1., 2.])).to_state()
    state['columns'][0]['name'] = 'row_id'
    with pytest.raises(ValueError, match='metadata'):
        FittedTransform.from_state(state)
