"""P07 protocol boundaries; official-package numerical fixtures are separate."""
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
    assert importlib.util.find_spec('treeneuron.research.tabm_native') is not None, \
        'Versioned official TabM adapter is missing'
    return importlib.import_module('treeneuron.research.tabm_native')


def config(variant='plain', mode='native_anchor', **changes):
    value = dict(schema='research/v2-tabm-config', family='P07', variant=variant,
        mode=mode, feature_schema=dict(columns=['x', 'kind'], categorical=['kind']), parameters={})
    value.update(changes)
    return value


@pytest.mark.parametrize('variant,blocks,start', [('plain', 3, 'random-signs'),
    ('piecewise', 2, 'normal')])
def test_actual_native_factories_and_patience_remain_distinct(variant, blocks, start):
    cfg = api().TabMConfig.parse(config(variant))
    p = cfg.resolved()
    assert p['n_blocks'] == blocks and p['start_scaling_init'] == start
    assert p['d_block'] == 512 and p['k'] == 32 and p['batch_size'] == 256
    assert p['lr'] == .002 and p['weight_decay'] == .0003
    assert cfg.max_epochs == 1_000_000_000 and cfg.patience == 16
    assert p['n_bins'] == (48 if variant == 'piecewise' else None)
    assert p['d_embedding'] == (16 if variant == 'piecewise' else None)
    assert p['mean_fallback'] == 'none' and p['prediction_order'] == 'inverse_each_member_then_mean_float32'


@pytest.mark.parametrize('changes', [
    dict(max_epochs=3), dict(mode='native_anchor', parameters={'lr': .002}),
    dict(mode='search', parameters={'n_blocks': 1}),
    dict(mode='search', parameters={'d_block': 65}),
    dict(mode='search', parameters={'k': 4}),
    dict(mode='search', parameters={'batch_size': 128}),
    dict(mode='search', parameters={'weight_decay': 1e-5}),
    dict(mode='search', parameters={'dropout': .6}),
    dict(mode='search', parameters={'n_bins': 48}),
    dict(mode='engineering_fixture', max_epochs=4), dict(extra=True),
])
def test_unsupported_and_implicit_native_recipe_changes_fail(changes):
    with pytest.raises(ValueError):
        api().TabMConfig.parse(config(**changes))


def test_embedded_support_and_tiny_mode_are_explicit():
    module = api()
    cfg = module.TabMConfig.parse(config('piecewise', 'search', parameters=dict(
        n_blocks=4, d_block=80, lr=.0001, weight_decay=0., dropout=0., n_bins=2, d_embedding=8)))
    assert cfg.resolved()['n_bins'] == 2
    tiny = module.TabMConfig.parse(config(mode='engineering_fixture', max_epochs=3))
    assert tiny.max_epochs == 3 and tiny.patience == 16
    for bad in (dict(n_bins=129), dict(d_embedding=9), dict(activation=True)):
        with pytest.raises(ValueError):
            module.TabMConfig.parse(config('piecewise', 'search', parameters=bad))


def test_quantile_uses_exact_native_absolute_float32_noise_F_only():
    from sklearn.preprocessing import QuantileTransformer
    module = api()
    F = pd.DataFrame(dict(x=np.linspace(-4, 7, 96), kind=['a', 'b']*48))
    raw = F[['x']].to_numpy(dtype=np.float32)
    noisy = raw + np.random.default_rng(0).normal(0., 1e-5, raw.shape).astype(np.float32)
    qt = QuantileTransformer(n_quantiles=10, output_distribution='normal', subsample=10**9).fit(noisy)
    tr = module.TabMTransform.fit(F, config()['feature_schema'])
    state = tr.to_state()
    num, cat = tr.transform(F)
    np.testing.assert_array_equal(num, qt.transform(raw))
    assert num.dtype == np.float32 and cat.dtype == np.int64 and tr.cardinalities == [3]
    assert state['noisy_fit_dtype'] == 'float32' and state['noise_std'] == 1e-5
    assert state['quantile_random_state'] is None and state['noise_seed'] == 0
    before = json.dumps(state, sort_keys=True)
    qnum, qcat = tr.transform(pd.DataFrame(dict(x=[-1e6, 1e6], kind=['new', 'a'])))
    assert np.isfinite(qnum).all() and qcat.tolist() == [[2], [0]]
    assert json.dumps(tr.to_state(), sort_keys=True) == before
    restored = module.TabMTransform.from_state(json.loads(before))
    np.testing.assert_array_equal(restored.transform(F)[0], num)
    np.testing.assert_array_equal(restored.transform(F)[1], cat)


def test_constant_detection_uses_float32_values_and_cannot_learn_from_query():
    module = api()
    F = pd.DataFrame(dict(x=np.arange(96.), round_constant=1.+np.arange(96.)*1e-11,
        kind=['a', 'b']*48))
    schema = dict(columns=list(F), categorical=['kind'])
    tr = module.TabMTransform.fit(F, schema)
    assert tr.numeric_names == ['x']
    assert tr.to_state()['numeric_keep_mask'] == [True, False]
    assert tr.to_state()['dropped_numeric_values'] == {'round_constant': 1.}
    assert tr.transform(pd.DataFrame(dict(x=[1.], round_constant=[999.], kind=['a'])))[0].shape == (1, 1)
    assert module.TabMTransform.from_state(tr.to_state()).to_state() == tr.to_state()


def test_typed_categories_unknown_and_no_numeric_shape():
    module = api()
    F = np.array([[1], ['1'], [True]]*32, object)
    tr = module.TabMTransform.fit(F, dict(columns=['c'], categorical=['c']))
    assert tr.cardinalities == [4]
    num, cat = tr.transform(F)
    assert num.shape == (96, 0) and len(set(cat[:, 0])) == 3
    assert tr.transform(np.array([['unknown']], object))[1].item() == 3
    assert tr.transform(np.empty((0, 1), object))[0].shape == (0, 0)


def test_target_native_float32_mean_std_and_actual_constant_policy():
    module = api()
    y = np.array([1e6, 1e6+1., 1e6+4.], np.float32)
    tr = module.TabMTarget.fit(y)
    assert tr.mean == y.mean().item() and tr.scale == y.std().item()
    np.testing.assert_array_equal(tr.transform(y), (y-y.mean())/y.std())
    constant = module.TabMTarget.fit([1., 1.+1e-10])
    assert constant.scale == 1. and constant.to_state()['constant_float32_F'] is True
    with pytest.raises(FloatingPointError, match='underflow'):
        module.TabMTarget.fit(np.array([1e-30, 2e-30], np.float32))
    with pytest.raises((ValueError, FloatingPointError)):
        module.TabMTarget.fit([1e100, 2e100])


@pytest.mark.parametrize('field,value', [('numeric_keep_mask', [False]),
    ('quantiles', [[float('nan')]]), ('vocabularies', [['str:a', 'str:a']]),
    ('noise_std', .001), ('quantile_random_state', 0), ('extra', 1)])
def test_corrupt_F_transform_cannot_load(field, value):
    module = api()
    tr = module.TabMTransform.fit(pd.DataFrame(dict(x=np.arange(96.), kind=['a', 'b']*48)),
        config()['feature_schema'])
    state = tr.to_state()
    state[field] = value
    with pytest.raises(ValueError):
        module.TabMTransform.from_state(state)


def test_native_patience_allows_seventeen_misses_and_strict_first_best():
    tracker = api().TabMSelection(16)
    assert tracker.observe(3., 0) == (True, False)
    for epoch in range(1, 17):
        assert tracker.observe(3., epoch) == (False, False)
    assert tracker.observe(3., 17) == (False, True)
    assert tracker.best_epoch == 0
    with pytest.raises(FloatingPointError):
        tracker.observe(float('nan'), 18)


def test_member_loss_is_not_loss_on_averaged_prediction():
    import torch
    pred = torch.tensor([[2., -2.]], requires_grad=True)
    loss = api().member_loss(pred, torch.tensor([0.]))
    assert loss.item() == 4.
    loss.backward()
    assert pred.grad.tolist() == [[2., -2.]]


def test_prediction_inverse_before_member_mean_preserves_native_precision():
    module = api()
    values = np.array([[-2.3250308, -.21879166, -1.245911]], np.float32)
    target = module.TabMTarget.fit(np.array([1000000., 1000001., 1000004.], np.float32))
    expected = (values*target.scale+target.mean).mean(1)
    observed = module.original_mean(values, target)
    np.testing.assert_array_equal(observed, expected)
    assert observed.dtype == np.float32


def test_module_import_is_lazy():
    api()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    code = ('import sys; import treeneuron.research.tabm_native; '
        'assert not set(("torch","sklearn","tabm","rtdl_num_embeddings")) & set(sys.modules)')
    result = subprocess.run([sys.executable, '-c', code], env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, result.stdout


def test_bin_boundary_and_no_numeric_embedding_reject_without_silent_adaptation():
    module = api()
    embedded_numeric = module.TabMConfig.parse(config('piecewise',
        feature_schema=dict(columns=['x'], categorical=[])))
    with pytest.raises(ValueError, match='n_bins'):
        module.validate_workload(embedded_numeric, 48, 10, 1, [])
    embedded = module.TabMConfig.parse(config('piecewise'))
    with pytest.raises(ValueError, match='numeric'):
        module.validate_workload(embedded, 96, 10, 0, [3])
    plain = module.TabMConfig.parse(config())
    module.validate_workload(plain, 96, 10, 0, [3])
    with pytest.raises(ValueError):
        module.validate_workload(plain, 96, 10, 0, [])


def test_snapshot_requires_external_trusted_manifest_before_payload_access(tmp_path):
    module = api()
    assert hasattr(module, 'TabMPredictor'), 'Portable native TabM predictor is missing'
    with pytest.raises(ValueError, match='trusted'):
        module.TabMPredictor.load(tmp_path/'absent', expected_manifest_sha256=None)


def test_native_torch_concatenation_preserves_mean_reduction_layout():
    import torch
    class ColumnMajorMembers(torch.nn.Module):
        def forward(self, x, cat):
            return torch.arange(160, dtype=torch.float32).reshape(32, 5).T.unsqueeze(-1)
    observed = api()._members(ColumnMajorMembers(), np.ones((5, 1), np.float32),
        np.empty((5, 0), np.int64), 8096)
    assert observed.flags.c_contiguous, 'Notebook torch.cat makes member rows contiguous before NumPy mean'


def test_categorical_only_plain_passes_native_None_in_training_and_query():
    import torch
    class NativeCategoryBoundary(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Parameter(torch.tensor(.1))
        def forward(self, x_num, x_cat):
            assert x_num is None, 'Official TabM requires x_num=None when n_num_features=0'
            return self.head.expand(len(x_cat), 32, 1)
    module = api()
    cfg = module.TabMConfig.parse(config(mode='engineering_fixture', max_epochs=1,
        feature_schema=dict(columns=['c'], categorical=['c'])))
    inputs = np.empty((10, 0), np.float32), np.zeros((10, 1), np.int64)
    target = module.TabMTarget.fit(np.arange(10, dtype=np.float32))
    model = NativeCategoryBoundary()
    module._train_model(model, cfg, inputs, target.transform(np.arange(10, dtype=np.float32)),
        inputs, np.arange(10, dtype=np.float32), target)
    assert module._members(model, *inputs, 8096).shape == (10, 32)


@pytest.mark.parametrize('error,status,retry', [
    (ImportError('missing native dependency'), 'input_or_dependency_failure', True),
    (OSError('native runtime inspection I/O'), 'input_or_dependency_failure', True),
    (ValueError('native API error'), 'input_or_dependency_failure', False),
    (FloatingPointError('nonfinite loss'), 'numerical_failure', False),
    (MemoryError('allocation exhausted'), 'resource_exhausted', False),
    (KeyboardInterrupt('interrupted'), 'interrupted_recoverable', False),
])
def test_failure_receipt_preserves_domain_phase_original_exception_and_cost(error, status, retry,
        monkeypatch, tmp_path):
    from treeneuron.baselines.api import FitRequest
    module = api()
    def reject():
        raise error
    monkeypatch.setattr(module, '_dependencies', reject)
    F = pd.DataFrame(dict(x=np.arange(96.), kind=['a', 'b']*48))
    request = FitRequest(F, np.arange(96.), F.iloc[:10], np.arange(10.), 5,
        config(mode='engineering_fixture', max_epochs=1), {}, tmp_path/'attempt')
    predictor = module.TabMPredictor()
    with pytest.raises(type(error)) as caught:
        predictor.fit(request)
    assert caught.value is error
    summary = predictor.resource_summary()
    assert summary['status'] == status and summary['infrastructure_retry_permitted'] is retry
    assert summary['failure_phase'] == 'dependency_validation' and summary['total_fit_wall_ns'] > 0
    assert summary == json.loads((tmp_path/'attempt/attempt.json').read_text())


def test_nonfinite_member_training_is_an_explicit_failed_fit():
    import torch
    class NonfiniteModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head = torch.nn.Parameter(torch.tensor(float('nan')))
        def forward(self, x_num, x_cat):
            return self.head.expand(len(x_num), 32, 1)
    module = api()
    cfg = module.TabMConfig.parse(config(mode='engineering_fixture', max_epochs=1,
        feature_schema=dict(columns=['x'], categorical=[])))
    data = np.arange(10, dtype=np.float32)[:, None], np.empty((10, 0), np.int64)
    target = module.TabMTarget.fit(np.arange(10, dtype=np.float32))
    with pytest.raises(FloatingPointError, match='training loss'):
        module._train_model(NonfiniteModel(), cfg, data, np.arange(10, dtype=np.float32),
            data, np.arange(10, dtype=np.float32), target)


def test_bytecode_enabled_worker_refuses_before_any_native_import(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(root/'src'),
        str(root/'environment/research-v2-tabm-site')]))
    env.pop('PYTHONDONTWRITEBYTECODE', None)
    env.pop('KMP_DUPLICATE_LIB_OK', None)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '1'
    # -B protects source/core during setup. Toggle only after all helpers exist;
    # trap dynamic imports so the RED case cannot create overlay bytecode.
    program = '''import sys, importlib
from treeneuron.research import tabm_native, ordinary_neural
def forbidden(name):
    raise AssertionError('Reached native import before bytecode policy refusal: '+name)
importlib.import_module = forbidden
sys.dont_write_bytecode = False
try:
    tabm_native._dependencies()
except ValueError as error:
    assert 'bytecode' in str(error), str(error)
else:
    raise AssertionError('Bytecode-enabled worker was admitted')
'''
    result = subprocess.run([sys.executable, '-B', '-c', program], cwd=root, env=env,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert result.returncode == 0, result.stdout
