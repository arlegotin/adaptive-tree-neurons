import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from treeneuron.baselines.api import FitRequest, Predictor
from treeneuron.baselines.simple import MeanPredictor, RidgePredictor


class ForbiddenSelection:
    def __array__(self, *args, **kwargs):
        raise AssertionError('closed-form controls must not inspect G labels')


def request(tmp_path, X=None, y=None):
    return FitRequest(X_fit=np.array([[1.], [2.], [3.]]) if X is None else X,
                      y_fit=np.array([10., 14., 18.]) if y is None else y,
                      X_select=ForbiddenSelection(), y_select=ForbiddenSelection(),
                      seed=17, config={}, budget={}, artifact_dir=tmp_path)


def test_exact_request_fields_and_mean_original_units(tmp_path):
    assert [f.name for f in dataclasses.fields(FitRequest)] == [
        'X_fit', 'y_fit', 'X_select', 'y_select', 'seed', 'config', 'budget', 'artifact_dir']
    model: Predictor = MeanPredictor().fit(request(tmp_path))
    np.testing.assert_array_equal(model.predict(np.array([[8.], [9.]]), batch_size=1), [14., 14.])
    summary = model.resource_summary()
    assert summary['capabilities']['output_units'] == 'original_target'
    assert summary['capabilities']['internal_validation'] is False
    assert summary['rss_bytes']['value'] is None and summary['rss_bytes']['reason']


@pytest.mark.parametrize('alpha', [0., .7, 100.])
def test_ridge_matches_independent_augmented_solve(tmp_path, alpha):
    X = np.array([[1., 3.], [2., -1.], [4., 2.], [8., 4.]])
    y = np.array([101., 103., 107., 120.])
    model = RidgePredictor(alpha).fit(request(tmp_path, X, y))
    standardized = (X - X.mean(0)) / X.std(0, ddof=0)
    design = np.column_stack([np.ones(4), standardized])
    augmented = np.vstack([design, np.column_stack([np.zeros(2), np.sqrt(alpha)*np.eye(2)])])
    coefficients = np.linalg.lstsq(augmented, np.r_[y, np.zeros(2)], rcond=None)[0]
    np.testing.assert_allclose(model.predict(X), design @ coefficients, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('factory', [MeanPredictor, lambda: RidgePredictor(.1)])
def test_json_save_load_preserves_predictions_and_transform(tmp_path, factory):
    X = pd.DataFrame({'x': [1., None, 5.], 'kind': ['a', 'b', 'a']})
    model = factory().fit(request(tmp_path, X, np.array([2., 4., 9.])))
    path = tmp_path / 'model.json'
    model.save(path)
    payload = json.loads(path.read_text())
    assert payload['transform']['columns'][0]['indicator']
    restored = type(model).load(path)
    future = pd.DataFrame({'x': [None, 8.], 'kind': ['unknown', 'b']})
    np.testing.assert_array_equal(restored.predict(future), model.predict(future))
    payload['coefficients'] = [float('nan')]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        type(model).load(path)


@pytest.mark.parametrize('alpha', [-1., np.nan, np.inf, True, '1'])
def test_invalid_ridge_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        RidgePredictor(alpha)


def test_rank_deficient_unregularized_ridge_and_constant_targets(tmp_path):
    X = np.ones((3, 2))
    model = RidgePredictor(0.).fit(request(tmp_path, X, np.full(3, 7.)))
    np.testing.assert_allclose(model.predict(X), np.full(3, 7.))


def test_unfitted_and_invalid_batch_size_rejected(tmp_path):
    for model in (MeanPredictor(), RidgePredictor(1.)):
        with pytest.raises(RuntimeError):
            model.predict(np.ones((1, 1)))
        with pytest.raises(RuntimeError):
            model.save(tmp_path / 'unfitted.json')
        model.fit(request(tmp_path))
        for size in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                model.predict(np.ones((1, 1)), batch_size=size)


@pytest.mark.parametrize('mutation', ['unknown_field', 'wrong_width', 'bad_alpha', 'wrong_method'])
def test_ridge_snapshot_rejects_inconsistent_parameters(tmp_path, mutation):
    path = tmp_path / 'ridge.json'
    RidgePredictor(1.).fit(request(tmp_path)).save(path)
    state = json.loads(path.read_text())
    if mutation == 'unknown_field':
        state['unknown'] = 1
    elif mutation == 'wrong_width':
        state['coefficients'] = [0.]
    elif mutation == 'bad_alpha':
        state['alpha'] = -1.
    else:
        state['method'] = 'MeanPredictor'
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        RidgePredictor.load(path)
