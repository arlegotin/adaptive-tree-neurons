"""Independent CPU float64 checks of package layer math, run in a child process."""
from __future__ import annotations

import json
import platform

import torch

from treeneuron.models.units import SelectiveAffineLayer


def run_checks():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(1602028)
    n, d, h = 13, 3, 4
    x = torch.randn(n, d, requires_grad=True)
    layer = SelectiveAffineLayer(d, h, tau=0.7)
    before = layer(x).detach()
    layer.grow(torch.tensor([3, 1]), torch.randn(2, d + 1))
    preservation = (layer(x) - before).abs().max().item()
    initial_loss = layer(x).square().mean()
    initial_loss.backward()
    grad_gate = torch.cat((layer.gate.weight.grad, layer.gate.bias.grad[:, None]), 1)
    X = torch.cat((x.detach(), torch.ones(n, 1)), 1)
    D = torch.cat((layer.contrast.weight.detach(), layer.contrast.bias.detach()[:, None]), 1)
    # d mean(z^2) / dz, independently obtained in the original base coordinates.
    e = 2 * before[:, layer.grown_ids] / h
    expected = torch.stack([X.T @ (e[:, k:k+1] * X) @ D[k] / (n * 2 * layer.tau)
                            for k in range(2)])
    gate_error = (grad_gate - expected).abs().max().item()
    contrast_norm = sum(p.grad.square().sum() for p in layer.contrast.parameters()).sqrt().item()
    gate_norm = grad_gate.norm().item()
    layer.zero_grad()
    with torch.no_grad():
        layer.gate.weight.normal_()
        layer.gate.bias.normal_()
    B = torch.cat((layer.base.weight, layer.base.bias[:, None]), 1)
    D = torch.cat((layer.contrast.weight, layer.contrast.bias[:, None]), 1)
    A = torch.cat((layer.gate.weight, layer.gate.bias[:, None]), 1)
    X = torch.cat((x, torch.ones(n, 1)), 1)
    ids = layer.grown_ids
    g = torch.sigmoid(X @ A.T / layer.tau)
    left, right = X @ (B[ids] - D).T, X @ (B[ids] + D).T
    mixture = (X @ B.T).index_copy(1, ids, (1 - g) * left + g * right)
    bypass = (X @ B.T).index_copy(1, ids, left + g * (right - left))
    actual, probe = layer(x), torch.randn(n, h)
    coordinates = (x, *layer.parameters())
    expected_gradients = torch.autograd.grad((mixture * probe).sum(), coordinates, retain_graph=True)
    actual_gradients = torch.autograd.grad((actual * probe).sum(), coordinates, retain_graph=True)
    head = torch.randn(h)
    collapsed = X @ (head @ B) + ((2 * g - 1) * (X @ D.T) * head[ids]).sum(1)
    hard = torch.where(g >= 0.5, right, left)
    soft = actual[:, ids]
    gap = torch.minimum(g, 1 - g) * (right - left).abs()
    # The activation convention has a concrete strict inequality.
    zero = SelectiveAffineLayer(1, 1)
    with torch.no_grad():
        zero.base.weight.zero_()
        zero.base.bias.zero_()
    zero.grow(torch.tensor([0]), torch.tensor([[0., 2.]]))
    post_mixture = zero(torch.zeros(1, 1)).relu().item()
    with torch.no_grad():
        zero.contrast.weight.zero_()
        zero.contrast.bias.zero_()
    zero(torch.ones(1, 1)).sum().backward()
    dead_norm = sum(p.grad.square().sum() for mod in (zero.gate, zero.contrast)
                    for p in mod.parameters()).sqrt().item()
    count_layer, count_head = SelectiveAffineLayer(16, 64), torch.nn.Linear(64, 1)
    base_count = sum(p.numel() for mod in (count_layer, count_head) for p in mod.parameters())
    count_layer.grow(torch.arange(16), torch.ones(16, 17))
    grown_count = sum(p.numel() for mod in (count_layer, count_head) for p in mod.parameters())
    results = {
        'forward_max_abs': (actual - mixture).abs().max().item(),
        'bypass_max_abs': (actual - bypass).abs().max().item(),
        'shared_gradient_max_abs': max((a-b).abs().max().item() for a, b in zip(actual_gradients, expected_gradients)),
        'collapse_max_abs': (actual @ head - collapsed).abs().max().item(),
        'growth_preservation_max_abs': preservation,
        'gate_gradient_formula_max_abs': gate_error,
        'gate_gradient_norm': gate_norm,
        'contrast_gradient_norm': contrast_norm,
        'dead_gradient_norm': dead_norm,
        'soft_hard_gap_max_abs': ((soft - hard).abs() - gap).abs().max().item(),
        'relu_bound_max_violation': ((soft.relu() - hard.relu()).abs() - gap).max().item(),
        'post_mixture_relu': post_mixture, 'leafwise_relu': 1.0,
        'base_parameter_count': base_count, 'grown_parameter_count': grown_count,
    }
    errors = ('forward_max_abs', 'bypass_max_abs', 'shared_gradient_max_abs', 'collapse_max_abs',
              'gate_gradient_formula_max_abs', 'soft_hard_gap_max_abs', 'relu_bound_max_violation')
    for key in errors:
        if not results[key] < 1e-10:
            raise AssertionError(f'{key}: {results[key]} >= 1e-10')
    if any(results[k] != 0 for k in ('growth_preservation_max_abs', 'contrast_gradient_norm', 'dead_gradient_norm', 'post_mixture_relu')):
        raise AssertionError('Expected exact insertion/dead-gradient/activation identity')
    if not gate_norm > 0 or (base_count, grown_count) != (1153, 1697):
        raise AssertionError('Gate gradient or compact parameter count failed')
    return {
        'schema_version': 1, 'check': 'package_layer_math', 'status': 'passed', 'seed': 1602028,
        'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                        'platform': platform.platform(), 'architecture': platform.machine(),
                        'dtype': str(x.dtype), 'device': str(x.device), 'threads': torch.get_num_threads()},
        'shapes': {'n': n, 'd': d, 'h': h},
        'constants': {'tau': 0.7, 'error_threshold': 1e-10}, 'results': results,
    }


if __name__ == '__main__':
    print(json.dumps(run_checks(), indent=2, sort_keys=True, allow_nan=False))
