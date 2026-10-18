"""F-only recipe boundaries; tiny official-reference checks live separately."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def api():
    assert importlib.util.find_spec('treeneuron.research.ordinary_neural') is not None, \
        'Versioned ordinary-neural adapter is missing'
    return importlib.import_module('treeneuron.research.ordinary_neural')


def config(family='P03', mode='native_anchor', **changes):
    value = dict(schema='research/v2-ordinary-neural-config', family=family, mode=mode,
        feature_schema=dict(columns=['x', 'kind'], categorical=['kind']), parameters={})
    value.update(changes)
    return value


def test_F_quantile_and_reserved_unknown_do_not_fit_query_features_or_labels():
    module = api()
    F = pd.DataFrame(dict(x=np.arange(30.)/7, kind=['a', 'b']*15))
    before = F.copy(deep=True)
    transform = module.NeuralTransform.fit(F, config()['feature_schema'])
    encoded = transform.transform(F)
    assert encoded.dtype == np.float32 and encoded.shape == (30, 2)
    assert transform.cardinalities == [3]
    assert set(encoded[:, 1]) == {0., 1.}
    saved = json.dumps(transform.to_state(), sort_keys=True)
    query = transform.transform(pd.DataFrame(dict(x=[1e6, -1e6], kind=['unseen', 'a'])))
    assert query[:, 1].tolist() == [2., 0.]
    assert np.isfinite(query).all()
    assert json.dumps(transform.to_state(), sort_keys=True) == saved
    restored = module.NeuralTransform.from_state(json.loads(saved))
    np.testing.assert_array_equal(restored.transform(F), encoded)
    pd.testing.assert_frame_equal(F, before)


def test_category_only_numeric_only_and_typed_vocabulary_roundtrip():
    module = api()
    F = np.array([[1], ['1'], [True]]*4, object)
    transform = module.NeuralTransform.fit(F, dict(columns=['c'], categorical=['c']))
    assert transform.cardinalities == [4]
    assert len(set(transform.transform(F)[:, 0])) == 3
    assert transform.transform(np.array([['new']], object))[0, 0] == 3
    numeric = module.NeuralTransform.fit(np.arange(20.).reshape(-1, 1),
        dict(columns=['x'], categorical=[]))
    assert numeric.cardinalities == []
    assert numeric.transform(np.empty((0, 1))).shape == (0, 1)


@pytest.mark.parametrize('family', ['P03', 'P04'])
def test_anchor_cannot_be_silently_shortened_or_given_search_settings(family):
    module = api()
    anchor = module.OrdinaryNeuralConfig.parse(family, config(family))
    assert (anchor.max_epochs, anchor.patience) == (1000, 20)
    tiny = module.OrdinaryNeuralConfig.parse(family, config(family, 'engineering_fixture',
        max_epochs=3, patience=2))
    assert (tiny.max_epochs, tiny.patience) == (3, 2)
    for bad in (config(family, max_epochs=3), config(family, parameters={'lr': .001})):
        with pytest.raises(ValueError):
            module.OrdinaryNeuralConfig.parse(family, bad)


@pytest.mark.parametrize('value', [
    config(mode='search', parameters={'hidden_dims': [128, 127]}),
    config(mode='search', parameters={'lr': .1}),
    config(mode='search', parameters={'weight_decay': 1e-8}),
    config(mode='search', parameters={'batch_size': True}),
    config(mode='search', parameters={'scheduler': 'cosine'}),
    config(mode='engineering_fixture', max_epochs=9, patience=2),
    config(feature_schema=dict(columns=['x', 'x'], categorical=[])), config(extra=True),
])
def test_out_of_support_and_unknown_settings_fail(value):
    with pytest.raises(ValueError):
        api().OrdinaryNeuralConfig.parse('P03', value)


def test_F_target_population_scaling_is_portable_and_constant_safe():
    module = api()
    scaler = module.TargetTransform.fit([10., 12., 14.])
    np.testing.assert_allclose(scaler.transform([10., 12., 14.]), [-1.2247449, 0., 1.2247449], rtol=1e-7)
    assert scaler.to_state()['mean'] == 12.
    assert scaler.to_state()['scale'] == np.std([10., 12., 14.], ddof=0)
    np.testing.assert_allclose(scaler.inverse(scaler.transform([10., 12., 14.])), [10., 12., 14.], atol=1e-6)
    assert module.TargetTransform.fit([4., 4.]).to_state()['scale'] == 1.


@pytest.mark.parametrize('field,value', [
    ('quantiles', [[float('nan')]]), ('vocabularies', [['str:a', 'str:a']]),
    ('quantile_random_state', 1), ('extra', True),
])
def test_corrupt_transform_cannot_be_loaded(field, value):
    module = api()
    F = pd.DataFrame(dict(x=np.arange(30.), kind=['a', 'b']*15))
    state = module.NeuralTransform.fit(F, config()['feature_schema']).to_state()
    state[field] = value
    with pytest.raises(ValueError):
        module.NeuralTransform.from_state(state)


@pytest.mark.parametrize('X', [
    pd.DataFrame(dict(kind=['a'], x=[1.])), np.array([[np.inf, 'a']], object),
    np.array([[1., None]], object), np.array([[1., 'a', 0]], object),
])
def test_bad_query_rejects_without_mutating_F_transform(X):
    module = api()
    tr = module.NeuralTransform.fit(pd.DataFrame(dict(x=np.arange(30.), kind=['a', 'b']*15)),
        config()['feature_schema'])
    before = tr.to_state()
    with pytest.raises(ValueError):
        tr.transform(X)
    assert tr.to_state() == before


def test_import_is_lazy_and_does_not_import_training_or_preprocessor_packages():
    api()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    program = ('import sys; import treeneuron.research.ordinary_neural; '
        'assert not set(("torch", "sklearn", "skorch", "pytabkit", "catboost", "xgboost", "lightgbm")) & set(sys.modules)')
    result = subprocess.run([sys.executable, '-c', program], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=30)
    assert result.returncode == 0, result.stdout


@pytest.mark.parametrize('family,parameters', [
    ('P03', {'hidden_dims': [128, 256, 128], 'lr': .002}),
    ('P03', {'hidden_dims': [128, 256, 128], 'dropout': .2}),
    ('P04', {'hidden_dropout': .25, 'width': 512}),
    ('P04', {'hidden_dropout': .25, 'lr': .002}),
])
def test_exceptional_anchor_coordinates_cannot_expand_generic_search_support(family, parameters):
    with pytest.raises(ValueError, match='anchor'):
        api().OrdinaryNeuralConfig.parse(family, config(family, 'search', parameters=parameters))


def test_generic_search_defaults_and_explicit_complete_published_anchors_are_admitted():
    module = api()
    mlp = module.OrdinaryNeuralConfig.parse('P03', config('P03', 'search'))
    resnet = module.OrdinaryNeuralConfig.parse('P04', config('P04', 'search'))
    assert mlp.resolved()['hidden_dims'] == [256, 256, 256]
    assert resnet.resolved()['hidden_dropout'] == .1
    for family, parameters in [('P03', {'hidden_dims': [128, 256, 128]}),
            ('P04', {'hidden_dropout': .25})]:
        published = module.OrdinaryNeuralConfig.parse(family, config(family, 'search', parameters=parameters))
        anchor = module.OrdinaryNeuralConfig.parse(family, config(family))
        assert published.resolved() == anchor.resolved()
