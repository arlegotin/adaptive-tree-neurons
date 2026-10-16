"""Package score checks against independent derivatives and dense moments."""
import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from treeneuron.models.units import SelectiveAffineLayer
from treeneuron.growth.score import InvalidScoreRun, score_units


def score_case():
    with torch.random.fork_rng():
        torch.manual_seed(1602027)
        X = torch.randn(41, 5, dtype=torch.float64)
        model = nn.Sequential(SelectiveAffineLayer(5, 4), nn.ReLU(), nn.Linear(4, 1)).double().eval()
        y = torch.randn(41, dtype=torch.float64)
    return model, X, y


def score(model, X, y, **kwargs):
    return score_units(model, X, y, epsilon=.05, tau=1., adam_epsilon=1e-8,
                       batch_size=kwargs.pop('batch_size', 41),
                       unit_block_size=kwargs.pop('unit_block_size', 4), **kwargs)


def test_gate_gradient_matches_moment_formula_and_first_adam_step():
    model, X, y = score_case()
    table = score(model, X, y)
    model[0].grow(table.ids, table.contrasts)
    loss = (model(X).squeeze(1) - y).square().mean()
    gradient = torch.autograd.grad(loss, model[0].gate.weight, retain_graph=True)[0]
    bias_gradient = torch.autograd.grad(loss, model[0].gate.bias)[0]
    actual = torch.cat((gradient, bias_gradient[:, None]), 1)
    torch.testing.assert_close(table.gradients, actual, atol=1e-10, rtol=0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model[0].gate.parameters():
        parameter.requires_grad_(True)
    before = loss.item()
    opt = torch.optim.AdamW(model[0].gate.parameters(), lr=1e-7, eps=1e-8, weight_decay=0)
    (model(X).squeeze(1) - y).square().mean().backward()
    opt.step()
    observed = before - (model(X).squeeze(1) - y).square().mean().item()
    predicted = 1e-7 * table.adam_scores.sum().item() * 12
    assert observed == pytest.approx(predicted, rel=1e-4)


def test_e_uses_sum_of_row_losses():
    model, X, y = score_case()
    table = score(model, X, y)
    augmented = torch.cat((X, torch.ones(41, 1, dtype=X.dtype)), 1)
    z = model[0](X)
    e = torch.autograd.grad((model[2](z.relu()).squeeze(1) - y).square().sum(), z)[0]
    for j in range(4):
        moment = augmented.T @ (e[:, j, None] * augmented) / 41
        expected = moment @ table.contrasts[j] / 2
        torch.testing.assert_close(table.gradients[j], expected, atol=1e-10, rtol=0)
        torch.testing.assert_close(table.eigenvalues[j], torch.linalg.eigvalsh(moment), atol=1e-10, rtol=0)
        assert table.contrasts[j, table.contrasts[j].abs().argmax()] > 0
        assert (augmented @ table.contrasts[j]).square().mean().sqrt() == pytest.approx(.05 * z[:, j].square().mean().sqrt().item())
    torch.testing.assert_close(table.sgd_scores, table.gradients.square().sum(1) / 12)


@pytest.mark.parametrize('block', [1, 2, 10])
def test_score_batch_invariance(block):
    model, X, y = score_case()
    full, batched = score(model, X, y), score(model, X, y, batch_size=7, unit_block_size=block)
    for field in ('contrasts', 'gradients', 'adam_scores', 'eigenvalues', 'spectral_gaps', 'base_rms', 'projection_rms', 'gram'):
        torch.testing.assert_close(getattr(full, field), getattr(batched, field), atol=1e-10, rtol=0)


def test_positive_relu_rescaling_and_permutation():
    model, X, y = score_case()
    original = score(model, X, y)
    scaled = copy.deepcopy(model)
    scales = torch.tensor([.3, 2., 7., 1.1], dtype=X.dtype)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        scaled[0].base.weight.mul_(scales[:, None])
        scaled[0].base.bias.mul_(scales)
        scaled[2].weight.div_(scales)
    rescaled = score(scaled, X, y)
    torch.testing.assert_close(original.gradients, rescaled.gradients, atol=1e-10, rtol=0)
    torch.testing.assert_close(original.adam_scores, rescaled.adam_scores, atol=1e-10, rtol=0)
    with torch.no_grad():
        scaled[0].base.weight.copy_(model[0].base.weight[permutation])
        scaled[0].base.bias.copy_(model[0].base.bias[permutation])
        scaled[2].weight.copy_(model[2].weight[:, permutation])
    permuted = score(scaled, X, y)
    torch.testing.assert_close(permuted.adam_scores, original.adam_scores[permutation], atol=1e-10, rtol=0)


def test_signed_dominant_eigenvalue_and_tie_diagnostics():
    model = nn.Sequential(SelectiveAffineLayer(1, 1), nn.ReLU(), nn.Linear(1, 1)).double().eval()
    with torch.no_grad():
        model[0].base.weight.zero_(); model[0].base.bias.fill_(1)
        model[2].weight.fill_(1); model[2].bias.zero_()
    X = torch.tensor([[-2.], [2.]], dtype=torch.float64)
    table = score(model, X, torch.full((2,), 2., dtype=X.dtype))
    torch.testing.assert_close(table.eigenvalues, torch.tensor([[-8., -2.]], dtype=X.dtype))
    torch.testing.assert_close(table.contrasts, torch.tensor([[.025, 0.]], dtype=X.dtype))
    assert table.gradients[0, 0] == pytest.approx(-.1)
    X = torch.tensor([[-1.], [1.]], dtype=X.dtype)
    table = score(model, X, torch.full((2,), 2., dtype=X.dtype))
    assert table.spectral_gaps.item() == 0
    assert table.near_ties.item() and table.eligible.item()


def test_no_source_state_or_global_rng_changes():
    model, X, y = score_case()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    state = copy.deepcopy(model.state_dict())
    grads = [p.grad.clone() for p in model.parameters()]
    rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    score(model, X, y)
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    assert np.array_equal(np_rng[1], np.random.get_state()[1])
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(g, p.grad) for g, p in zip(grads, model.parameters()))
    assert not model.training


def test_zero_units_retained_with_reasons():
    model, X, y = score_case()
    with torch.no_grad():
        model[0].base.weight[0].zero_(); model[0].base.bias[0].zero_()
        model[2].weight.zero_()
    table = score(model, X, y)
    assert not table.eligible.any()
    assert table.status[0] == 'zero_base_rms'
    assert table.status[1:] == ('zero_score',) * 3


@pytest.mark.parametrize('change', ['train', 'dtype', 'labels', 'nonfinite', 'grown', 'activation', 'multi'])
def test_rejects_invalid_run_or_unsupported_model(change):
    model, X, y = score_case()
    if change == 'train': model.train()
    elif change == 'dtype': model.float()
    elif change == 'labels': y = y[:, None, None]
    elif change == 'nonfinite':
        with torch.no_grad(): model[2].weight.fill_(float('nan'))
    elif change == 'grown': model[0].grow(torch.empty(0, dtype=torch.int64), torch.empty(0, 6, dtype=X.dtype))
    elif change == 'activation': model[1] = nn.Sigmoid()
    elif change == 'multi': model = nn.Sequential(model, SelectiveAffineLayer(1, 1).double()).eval()
    with pytest.raises((ValueError, InvalidScoreRun)):
        score(model, X, y)


def test_wrapper_and_column_targets_without_broadcast():
    model, X, y = score_case()
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__(); self.network = model
        def forward(self, x):
            return self.network(x).squeeze(1)
    wrapped = Wrapper().eval()
    torch.testing.assert_close(score(wrapped, X, y[:, None]).gradients, score(model, X, y).gradients)


@pytest.mark.parametrize('setting', ['epsilon', 'tau', 'adam_epsilon', 'batch_size', 'unit_block_size', 'x_scalar', 'x_empty', 'x_nonfinite', 'y_dtype', 'tau_mismatch'])
def test_rejects_invalid_inputs_and_score_config(setting):
    model, X, y = score_case()
    kwargs = dict(epsilon=.05, tau=1., adam_epsilon=1e-8, batch_size=7, unit_block_size=2)
    if setting in kwargs:
        kwargs[setting] = 0
    elif setting == 'x_scalar': X = torch.tensor(1., dtype=X.dtype)
    elif setting == 'x_empty': X, y = X[:0], y[:0]
    elif setting == 'x_nonfinite': X[0, 0] = float('inf')
    elif setting == 'y_dtype': y = y.float()
    elif setting == 'tau_mismatch': kwargs['tau'] = 2.
    with pytest.raises(ValueError):
        score_units(model, X, y, **kwargs)


def test_nonfinite_statistics_are_retained_without_blocking_finite_model_loss():
    model = nn.Sequential(SelectiveAffineLayer(1, 1), nn.ReLU(), nn.Linear(1, 1)).double().eval()
    with torch.no_grad():
        model[0].base.weight.zero_(); model[0].base.bias.fill_(1)
        model[2].weight.fill_(1); model[2].bias.zero_()
    X = torch.full((2, 1), 1e155, dtype=torch.float64)
    table = score(model, X, torch.full((2,), 2., dtype=X.dtype))
    assert not table.eligible.item()
    assert table.status == ('nonfinite_statistics',)
    assert len(table.table_hash) == 64


@pytest.mark.parametrize('failure', ['prediction', 'loss'])
def test_nonfinite_predictions_or_loss_block_the_run(failure):
    model = nn.Sequential(SelectiveAffineLayer(1, 1), nn.ReLU(), nn.Linear(1, 1)).double().eval()
    with torch.no_grad():
        model[0].base.weight.fill_(1); model[0].base.bias.zero_()
        model[2].weight.fill_(1e200 if failure == 'prediction' else 1.); model[2].bias.zero_()
    X = torch.full((2, 1), 1e200, dtype=torch.float64)
    with pytest.raises(InvalidScoreRun):
        score(model, X, torch.zeros(2, dtype=X.dtype))


def test_scoring_isolated_from_wrapper_buffer_mutation_and_rng():
    model, X, y = score_case()
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__(); self.network = model
            self.register_buffer('calls', torch.zeros((), dtype=torch.float64))
        def forward(self, x):
            self.calls.add_(1)
            random.random(); np.random.random(); torch.rand(1)
            return self.network(x)
    wrapped = Wrapper().eval()
    rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
    score(wrapped, X, y)
    assert wrapped.calls.item() == 0
    assert torch.equal(rng, torch.get_rng_state()) and py_rng == random.getstate()
    assert np.array_equal(np_rng[1], np.random.get_state()[1])


def test_frozen_eval_model_inside_no_grad():
    model, X, y = score_case()
    expected = score(model, X, y)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.no_grad():
        actual = score(model, X, y)
    torch.testing.assert_close(actual.gradients, expected.gradients)


def test_table_stays_cpu_with_another_default_factory_device():
    model, X, y = score_case()
    with torch.device('meta'):
        table = score(model, X, y)
    assert all(getattr(table, name).device.type == 'cpu' for name in
               ('ids', 'contrasts', 'gradients', 'adam_scores', 'eligible', 'gram'))


@pytest.mark.parametrize('functional', [False, True])
def test_rejects_inplace_activation_that_destroys_captured_preactivation(functional):
    model, X, y = score_case()
    if functional:
        class Wrapper(nn.Module):
            def __init__(self):
                super().__init__(); self.layer, self.head = model[0], model[2]
            def forward(self, x):
                return self.head(torch.nn.functional.relu(self.layer(x), inplace=True))
        model = Wrapper().eval()
    else:
        model[1] = nn.ReLU(inplace=True)
        model.eval()
    state = copy.deepcopy(model.state_dict())
    hooks = [(dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in model.modules()]
    rng = torch.get_rng_state()
    with pytest.raises(ValueError):
        score(model, X, y)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(state[key], value) for key, value in model.state_dict().items())
    assert hooks == [(dict(module._forward_hooks), dict(module._forward_pre_hooks)) for module in model.modules()]
