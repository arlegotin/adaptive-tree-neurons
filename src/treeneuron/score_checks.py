"""Fresh-process evidence using the real package score and allocation APIs."""
from __future__ import annotations

import copy
import json
import platform

import torch
from torch import nn

from treeneuron.growth.allocate import allocate
from treeneuron.growth.score import score_units
from treeneuron.models.units import SelectiveAffineLayer


def run_checks():
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(1602029)
    n, d, h = 29, 3, 5
    X, y = torch.randn(n, d), torch.randn(n)
    model = nn.Sequential(SelectiveAffineLayer(d, h), nn.ReLU(), nn.Linear(h, 1)).eval()
    def score(candidate, batch_size=n, block=h):
        return score_units(candidate, X, y, epsilon=.05, tau=1., adam_epsilon=1e-8,
                           batch_size=batch_size, unit_block_size=block)
    table, batched = score(model), score(model, 7, 2)
    transformed = copy.deepcopy(model)
    scales = torch.tensor([.3, 2., 7., 1.1, .8])
    permutation = torch.tensor([2, 0, 4, 3, 1])
    with torch.no_grad():
        transformed[0].base.weight.mul_(scales[:, None])
        transformed[0].base.bias.mul_(scales)
        transformed[2].weight.div_(scales)
    rescaled = score(transformed)
    with torch.no_grad():
        transformed[0].base.weight.copy_(model[0].base.weight[permutation])
        transformed[0].base.bias.copy_(model[0].base.bias[permutation])
        transformed[2].weight.copy_(model[2].weight[:, permutation])
    permuted = score(transformed)
    augmented = torch.cat((X, torch.ones(n, 1)), 1)
    actual_rms = (augmented @ table.contrasts.T).square().mean(0).sqrt()
    top = allocate(table, 3, policy='adam', seed=0)
    random = allocate(table, 3, policy='random', seed=1)
    if top.effective_count != random.effective_count or top.effective_count != 3:
        raise AssertionError('Allocation count mismatch')
    random_error = (random.contrasts - table.contrasts[random.ids]).abs().max().item()
    model[0].grow(table.ids, table.contrasts)
    loss = (model(X).squeeze(1) - y).square().mean()
    weights, bias = torch.autograd.grad(loss, (model[0].gate.weight, model[0].gate.bias))
    gradient = torch.cat((weights, bias[:, None]), 1)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model[0].gate.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(model[0].gate.parameters(), lr=1e-7, eps=1e-8, weight_decay=0)
    (model(X).squeeze(1) - y).square().mean().backward()
    optimizer.step()
    observed = loss.item() - (model(X).squeeze(1) - y).square().mean().item()
    predicted = 1e-7 * table.adam_scores.sum().item() * 2 * (d + 1)
    results = {
        'gate_gradient_max_abs': (gradient - table.gradients).abs().max().item(),
        'batch_score_max_abs': (table.adam_scores - batched.adam_scores).abs().max().item(),
        'batch_contrast_max_abs': (table.contrasts - batched.contrasts).abs().max().item(),
        'rescaled_score_max_abs': (table.adam_scores - rescaled.adam_scores).abs().max().item(),
        'permuted_score_max_abs': (table.adam_scores[permutation] - permuted.adam_scores).abs().max().item(),
        'normalization_max_abs': (actual_rms - .05 * table.base_rms).abs().max().item(),
        'random_contrast_max_abs': random_error,
        'effective_count': top.effective_count,
        'predicted_loss_decrease': predicted,
        'observed_loss_decrease': observed,
        'adam_prediction_relative_error': abs(predicted - observed) / predicted,
    }
    if any(value >= 1e-10 for key, value in results.items() if key.endswith('_max_abs')):
        raise AssertionError('Package score numerical contract failed')
    if not observed > 0 or not results['adam_prediction_relative_error'] < 1e-4:
        raise AssertionError('Package score gate-only Adam prediction failed')
    return {
        'schema_version': 1, 'check': 'package_score_math', 'status': 'passed', 'seed': 1602029,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'platform': platform.platform(), 'architecture': platform.machine(),
                        'dtype': str(X.dtype), 'device': str(X.device), 'threads': torch.get_num_threads()},
        'shapes': dict(n=n, d=d, h=h),
        'constants': {'epsilon': .05, 'tau': 1., 'adam_epsilon': 1e-8, 'learning_rate': 1e-7,
                      'error_threshold': 1e-10, 'adam_relative_error_threshold': 1e-4},
        'results': results,
    }


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2, sort_keys=True, allow_nan=False))
