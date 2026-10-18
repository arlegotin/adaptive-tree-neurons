"""Tiny architecture/update and stopping tests, not native quality anchors."""
import copy
import importlib
import json

import numpy as np
import pytest

from ordinary_neural_reference import official_definitions, official_model
from test_ordinary_neural_contract import api, config


def test_noisy_quantile_matches_pinned_author_definition_exactly():
    module = api()
    rng = np.random.default_rng(7621)
    F = (rng.normal(size=(96, 3))*[.01, 1., 1000.]).astype(np.float32)
    G = rng.normal(size=(17, 3)).astype(np.float32)
    author = official_definitions()['TabrQuantileTransformer'](random_state=0).fit(F)
    transform = module.NeuralTransform.fit(F, dict(columns=['a', 'b', 'c'], categorical=[]))
    np.testing.assert_array_equal(transform.quantiles, author.normalizer_.quantiles_)
    np.testing.assert_array_equal(transform.transform(F), author.transform(F))
    np.testing.assert_array_equal(transform.transform(G), author.transform(G))


@pytest.mark.parametrize('family', ['P03', 'P04'])
@pytest.mark.parametrize('categorical', [False, True])
def test_initialized_forward_gradients_and_AdamW_step_match_author(family, categorical):
    import torch
    module = api()
    assert hasattr(module, 'build_model'), 'Native architecture construction is missing'
    schema = dict(columns=['x', 'c'] if categorical else ['x', 'z'],
        categorical=['c'] if categorical else [])
    parsed = module.OrdinaryNeuralConfig.parse(family, config(family, feature_schema=schema))
    with torch.random.fork_rng():
        torch.manual_seed(387)
        native = module.build_model(parsed, [3] if categorical else [])
        torch.manual_seed(387)
        author = official_model(family, schema, [3] if categorical else [], parsed.resolved())
        assert native.state_dict().keys() == author.state_dict().keys()
        for name, tensor in native.state_dict().items():
            assert torch.equal(tensor, author.state_dict()[name]), name
        X = torch.linspace(-1, 1, 32).reshape(16, 2)
        if categorical:
            X[:, 1] = torch.arange(16) % 3
            offset = native.category_offsets[0]
            assert torch.count_nonzero(native.category_embeddings.weight[offset+2]) == 0
        y = torch.sin(X[:, :1])
        for model in (native, author):
            model.train()
            torch.manual_seed(18)
            loss = (model(X.clone())-y).square().mean()
            loss.backward()
        for (name, p), (_, q) in zip(native.named_parameters(), author.named_parameters()):
            assert torch.equal(p.grad, q.grad), name
        for model in (native, author):
            torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01).step()
            model.eval()
        assert torch.equal(native(X), author(X))


def test_native_patience_progress_and_strict_checkpoint_are_distinct():
    module = api()
    assert hasattr(module, 'SelectionTracker'), 'Native stopping/checkpoint policy is missing'
    tracker = module.SelectionTracker(patience=2)
    assert tracker.observe(1., 1) == (True, False)
    assert tracker.observe(.99999, 2) == (True, False)
    assert tracker.observe(.99998, 3) == (True, True)
    assert tracker.best_epoch == 3
    with pytest.raises(FloatingPointError):
        tracker.observe(float('nan'), 4)


def test_BatchNorm_singleton_is_an_explicit_workload_failure_without_row_drop():
    module = api()
    assert hasattr(module, 'validate_workload'), 'Native workload validation is missing'
    resnet = module.OrdinaryNeuralConfig.parse('P04', config('P04'))
    with pytest.raises(ValueError, match='singleton'):
        module.validate_workload(resnet, 129, 20, [3])
    module.validate_workload(resnet, 130, 20, [3])


def test_target_or_schema_failure_retains_attempt_cost_before_native_import(tmp_path):
    from treeneuron.baselines.api import FitRequest
    module = api()
    assert hasattr(module, 'OrdinaryNeuralPredictor'), 'Native fitting adapter is missing'
    F = np.array([[float(i), 'a'] for i in range(10)], object)
    y = np.arange(10.); y[3] = np.nan
    request = FitRequest(F, y, F[:2], np.arange(2.), 7, config(), {}, tmp_path/'failed')
    model = module.OrdinaryNeuralPredictor('P03')
    with pytest.raises(ValueError, match='target'):
        model.fit(request)
    receipt = json.loads((request.artifact_dir/'attempt.json').read_text())
    assert receipt['status'] == 'input_or_dependency_failure'
    assert receipt['total_fit_wall_ns'] > 0 and receipt['checkpoint_eligible'] is False
    assert receipt['exception'] == 'ValueError'
    with pytest.raises(RuntimeError):
        model.predict(F)


def test_native_F_mean_safeguard_wins_on_constant_G_after_finite_training():
    import torch
    module = api()
    parsed = module.OrdinaryNeuralConfig.parse('P03', config('P03', 'engineering_fixture',
        max_epochs=1, patience=1, feature_schema=dict(columns=['x'], categorical=[])))
    model = torch.nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.fill_(0.); model.bias.fill_(1.)
    result, orders = module._train_model(model, parsed, np.zeros((10, 1), np.float32),
        np.zeros(10, np.float32), np.zeros((3, 1), np.float32), np.zeros(3, np.float32))
    assert result['selected_F_mean'] is True
    assert result['mean_G_mse'] == 0. and result['history'][0]['G_mse'] > 0.
    np.testing.assert_array_equal(np.sort(orders[0]), np.arange(10))


@pytest.mark.parametrize('failure,status', [
    (OSError('runtime inspection unavailable'), 'input_or_dependency_failure'),
    (ImportError('API dependency missing'), 'input_or_dependency_failure'),
    (TypeError('unsupported API input'), 'input_or_dependency_failure'),
    (MemoryError('native allocation'), 'resource_exhausted'),
    (FloatingPointError('nonfinite loss'), 'numerical_failure'),
    (KeyboardInterrupt('worker interrupted'), 'interrupted_recoverable')])
def test_native_failure_keeps_original_exception_domain_phase_and_paid_cost(failure, status, tmp_path, monkeypatch):
    from treeneuron.baselines.api import FitRequest
    module = api()
    monkeypatch.setattr(module, '_dependencies', lambda: {'import_environment': {}})
    def failed_native(*args):
        raise failure
    monkeypatch.setattr(module, '_train_model', failed_native)
    F = np.arange(20., dtype=np.float64).reshape(10, 2)
    request = FitRequest(F, np.arange(10.), F[:2], np.arange(2.), 7,
        config(feature_schema=dict(columns=['x', 'y'], categorical=[])), {}, tmp_path/'attempt')
    model = module.OrdinaryNeuralPredictor('P03')
    with pytest.raises(type(failure)) as caught:
        model.fit(request)
    assert caught.value is failure
    receipt = json.loads((request.artifact_dir/'attempt.json').read_text())
    assert receipt['status'] == status and receipt['failure_phase'] == 'native_training'
    assert receipt['total_fit_wall_ns'] > 0 and receipt['traceback']
    with pytest.raises(RuntimeError):
        model.predict(F)


@pytest.mark.parametrize('rows,columns', [(20001, 2), (10, 129)])
def test_oversized_inputs_reject_before_imports_or_preprocessing(rows, columns, tmp_path, monkeypatch):
    from treeneuron.baselines.api import FitRequest
    module = api()
    def forbidden_dependencies():
        raise AssertionError('Oversized input reached native imports')
    monkeypatch.setattr(module, '_dependencies', forbidden_dependencies)
    request = FitRequest(np.zeros((rows, columns)), np.zeros(rows),
        np.zeros((2, columns)), np.zeros(2), 7,
        config(feature_schema=dict(columns=[f'x{i}' for i in range(columns)], categorical=[])),
        {}, tmp_path/'oversized')
    with pytest.raises(ValueError, match='envelope'):
        module.OrdinaryNeuralPredictor('P03').fit(request)


def test_source_change_during_native_work_rejects_before_runtime_or_success_publication(tmp_path, monkeypatch):
    from treeneuron.baselines.api import FitRequest
    module = api()
    versions = iter([{'version': 'before'}, {'version': 'after'}])
    monkeypatch.setattr(module, '_source', lambda: next(versions))
    monkeypatch.setattr(module, '_dependencies', lambda: {'import_environment': {}})
    monkeypatch.setattr(module, '_train_model', lambda *args: ({}, np.arange(10)[None]))
    def forbidden_runtime():
        raise AssertionError('Changed source reached post-fit runtime publication')
    monkeypatch.setattr(module, '_runtime', forbidden_runtime)
    request = FitRequest(np.zeros((10, 1)), np.zeros(10), np.zeros((2, 1)), np.zeros(2), 7,
        config(feature_schema=dict(columns=['x'], categorical=[])), {}, tmp_path/'changed')
    with pytest.raises(ValueError, match='Source changed'):
        module.OrdinaryNeuralPredictor('P03').fit(request)
    receipt = json.loads((request.artifact_dir/'attempt.json').read_text())
    assert receipt['checkpoint_eligible'] is False and receipt['total_fit_wall_ns'] > 0


def test_native_threshold_subtraction_rounding_controls_patience_but_not_strict_best():
    from ordinary_neural_reference import official_threshold
    tracker = api().SelectionTracker(patience=1)
    first = float.fromhex('0x1.00de87540a5e7p-1')
    native_threshold = float.fromhex('0x1.00d7f3e8f2823p-1')
    assert official_threshold(first) == native_threshold
    assert tracker.observe(first, 1) == (True, False)
    assert tracker.observe(native_threshold, 2) == (True, True)
    assert tracker.best_epoch == 2


def test_negative_squared_loss_cannot_be_a_native_checkpoint():
    with pytest.raises(FloatingPointError):
        api().SelectionTracker(patience=2).observe(-.01, 1)
