"""F-only native tree boundaries, without importing optional native packages."""
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
    spec = importlib.util.find_spec('treeneuron.research.native_trees')
    assert spec is not None, 'Versioned native tree adapter is not implemented'
    return importlib.import_module('treeneuron.research.native_trees')


def config(family='P09', mode='native_anchor', **changes):
    value = dict(schema='research/v2-native-tree-config', family=family, mode=mode,
        feature_schema=dict(columns=['number', 'kind'], categorical=['kind']), parameters={})
    value.update(changes)
    return value


def test_linear_transform_fits_statistics_and_vocabulary_on_F_only():
    module = api()
    F = pd.DataFrame({'number': [1., 3.], 'kind': ['__ATN_UNKNOWN__', 'a']})
    before = F.copy(deep=True)
    transform = module.FeatureTransform.fit(F, config()['feature_schema'], linear=True)
    assert transform.to_state()['means'] == [2., 0.]
    assert transform.to_state()['scales'] == [1., 1.]
    native_F = transform.transform(F, 'P11')
    np.testing.assert_array_equal(native_F['f0'], [-1., 1.])
    saved = json.dumps(transform.to_state(), sort_keys=True)
    G = pd.DataFrame({'number': [100., 0.], 'kind': ['new', '__ATN_UNKNOWN__']})
    encoded = transform.transform(G, 'P11')
    assert encoded['f1'].cat.codes.tolist() == [-1, 0]
    assert json.dumps(transform.to_state(), sort_keys=True) == saved
    catboost = transform.transform(G, 'P08')
    assert catboost['f1'][0] == '__ATN_UNKNOWN__'
    assert catboost['f1'][1] != '__ATN_UNKNOWN__'
    pd.testing.assert_frame_equal(F, before)


def test_native_categories_distinguish_raw_types_and_restore_exact_transform():
    module = api()
    F = np.array([[1., 1], [2., '1'], [3., True]], dtype=object)
    schema = dict(columns=['x', 'category'], categorical=['category'])
    transform = module.FeatureTransform.fit(F, schema, linear=False)
    encoded = transform.transform(F, 'P09')
    assert len(encoded['f1'].cat.categories) == 3
    assert len(set(encoded['f1'].cat.codes)) == 3
    restored = module.FeatureTransform.from_state(json.loads(json.dumps(transform.to_state())))
    pd.testing.assert_frame_equal(encoded, restored.transform(F, 'P09'))
    assert transform.transform(np.empty((0, 2), object), 'P09').shape == (0, 2)


@pytest.mark.parametrize('X', [
    pd.DataFrame({'kind': ['a'], 'number': [1.]}),
    np.array([[np.inf, 'a']], object), np.array([[1., None]], object),
    np.array([[1., 'a', 4.]], object),
])
def test_bad_or_reordered_query_is_rejected(X):
    module = api()
    transform = module.FeatureTransform.fit(
        pd.DataFrame({'number': [1., 2.], 'kind': ['a', 'b']}),
        config()['feature_schema'], linear=False)
    with pytest.raises(ValueError):
        transform.transform(X, 'P10')


@pytest.mark.parametrize('family,rounds,patience', [
    ('P08', 10000, 200), ('P09', 10000, 100), ('P10', 10000, 100), ('P11', 5000, 100)])
def test_native_limits_and_fixture_ineligibility_are_distinct(family, rounds, patience):
    module = api()
    anchor = module.NativeTreeConfig.parse(family, config(family))
    assert (anchor.max_rounds, anchor.patience) == (rounds, patience)
    assert anchor.mode == 'native_anchor'
    tiny = module.NativeTreeConfig.parse(family,
        config(family, 'engineering_fixture', max_rounds=32, patience=4))
    assert (tiny.max_rounds, tiny.patience) == (32, 4)
    with pytest.raises(ValueError):
        module.NativeTreeConfig.parse(family, config(family, max_rounds=32))


@pytest.mark.parametrize('value', [
    config(parameters={'nthread': 9}), config(parameters={'eta': .1}),
    config(mode='search', parameters={'max_depth': 4}),
    config(mode='search', parameters={'subsample': .5}),
    config(mode='search', parameters={'reg_alpha': -1}),
    config(mode='engineering_fixture', max_rounds=129, patience=8),
    config(feature_schema=dict(columns=['number', 'number'], categorical=[])),
    config(extra=True),
])
def test_unknown_out_of_support_or_implicit_budget_config_is_rejected(value):
    with pytest.raises(ValueError):
        api().NativeTreeConfig.parse('P09', value)


@pytest.mark.parametrize('field,value', [('scales', [0., 1.]), ('means', [float('nan'), 0.]),
    ('vocabularies', [[], ['str:a', 'str:a']]), ('extra', True)])
def test_transform_snapshot_rejects_corruption(field, value):
    module = api()
    state = module.FeatureTransform.fit(np.array([[1., 'a'], [3., 'b']], object),
        config()['feature_schema'], linear=True).to_state()
    state[field] = value
    with pytest.raises(ValueError):
        module.FeatureTransform.from_state(state)


def test_import_is_lazy_and_does_not_import_torch_or_any_tree_package():
    api()
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]/'src')
    program = ('import sys; import treeneuron.research.native_trees; '
        'assert not set(("torch", "catboost", "xgboost", "lightgbm")) & set(sys.modules)')
    result = subprocess.run([sys.executable, '-c', program], env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert result.returncode == 0, result.stdout


def test_failed_fit_retains_input_error_and_cost_before_native_import(tmp_path):
    from treeneuron.baselines.api import FitRequest
    module = api()
    assert hasattr(module, 'NativeTreePredictor'), 'Native fit adapter is missing'
    request = FitRequest(np.array([[1., 'a'], [2., 'b']], object), np.array([1., np.nan]),
        np.array([[3., 'a']], object), np.array([3.]), 7, config(), {}, tmp_path/'attempt')
    model = module.NativeTreePredictor('P09')
    with pytest.raises(ValueError, match='target'):
        model.fit(request)
    receipt = json.loads((request.artifact_dir/'attempt.json').read_text())
    assert receipt['status'] == 'input_or_dependency_failure'
    assert receipt['exception'] == 'ValueError'
    assert receipt['total_fit_wall_ns'] > 0
    assert receipt['checkpoint_eligible'] is False
    with pytest.raises(RuntimeError, match='fitted'):
        model.predict(np.array([[1., 'a']], object))


def test_constructed_parameters_preserve_native_stopping_and_linear_policy():
    module = api()
    assert hasattr(module, 'native_parameters'), 'Native parameter construction is missing'
    cat = module.native_parameters(module.NativeTreeConfig.parse('P08', config('P08')), 7)
    assert 'learning_rate' not in cat  # Keep the pinned factory's automatic LR.
    assert cat['counter_calc_method'] == 'SkipTest' and cat['thread_count'] == 1
    linear = module.native_parameters(module.NativeTreeConfig.parse('P11', config('P11')), 7)
    assert linear['linear_tree'] is True and linear['num_threads'] == 1
    constant = module.native_parameters(module.NativeTreeConfig.parse('P10', config('P10')), 7)
    assert constant['linear_tree'] is False and constant['bagging_freq'] == 1
    for seed in (-1, True, 2**31):
        with pytest.raises(ValueError):
            module.native_parameters(module.NativeTreeConfig.parse('P09', config()), seed)


def test_caller_supplied_openmp_suppression_is_rejected(tmp_path, monkeypatch):
    from treeneuron.baselines.api import FitRequest
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        monkeypatch.setenv(key, '1')
    monkeypatch.setenv('KMP_DUPLICATE_LIB_OK', 'True')
    request = FitRequest(np.array([[1., 'a'], [2., 'b']], object), np.array([1., 2.]),
        np.array([[3., 'a']], object), np.array([3.]), 7, config(), {}, tmp_path/'attempt')
    with pytest.raises(ValueError, match='suppression'):
        api().NativeTreePredictor('P09').fit(request)


@pytest.mark.parametrize('failure,status', [
    (ImportError('native API dependency unavailable'), 'input_or_dependency_failure'),
    (OSError('native runtime inspection denied'), 'input_or_dependency_failure'),
    (TypeError('pinned API argument unsupported'), 'input_or_dependency_failure'),
    (MemoryError('native allocation failed'), 'resource_exhausted'),
    (FloatingPointError('nonfinite native prediction'), 'numerical_failure'),
    (KeyboardInterrupt('native worker interrupted'), 'interrupted_recoverable'),
])
def test_native_exception_domain_is_preserved_without_any_native_fit(tmp_path, monkeypatch, failure, status):
    from treeneuron.baselines.api import FitRequest
    module = api()
    monkeypatch.setattr(module, '_dependencies', lambda family: (None, None, {}))
    def failed_native(*args):
        raise failure
    monkeypatch.setattr(module, '_fit_native', failed_native)
    request = FitRequest(np.array([[1., 'a'], [2., 'b']], object), np.array([1., 2.]),
        np.array([[3., 'a']], object), np.array([3.]), 7, config(), {}, tmp_path/'attempt')
    with pytest.raises(type(failure)) as caught:
        module.NativeTreePredictor('P09').fit(request)
    assert caught.value is failure
    receipt = json.loads((request.artifact_dir/'attempt.json').read_text())
    assert receipt['status'] == status
    assert receipt['exception'] == type(failure).__name__
    assert receipt['total_fit_wall_ns'] > 0 and receipt['traceback']
