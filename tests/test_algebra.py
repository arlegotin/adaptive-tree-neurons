"""Independent leaf-coordinate checks for the package preactivation layer."""
import pytest
import torch

from treeneuron.models.units import SelectiveAffineLayer


def fixture():
    generator = torch.Generator().manual_seed(44)
    x = torch.randn(11, 3, generator=generator, dtype=torch.float64, requires_grad=True)
    layer = SelectiveAffineLayer(3, 4, tau=0.7).double()
    with torch.no_grad():
        layer.base.weight.copy_(torch.randn(4, 3, generator=generator, dtype=torch.float64))
        layer.base.bias.copy_(torch.randn(4, generator=generator, dtype=torch.float64))
    contrasts = torch.randn(2, 4, generator=generator, dtype=torch.float64)
    layer.grow(torch.tensor([1, 3]), contrasts)
    with torch.no_grad():
        layer.gate.weight.copy_(torch.randn(2, 3, generator=generator, dtype=torch.float64))
        layer.gate.bias.copy_(torch.randn(2, generator=generator, dtype=torch.float64))
    return x, layer


def test_shared_leaf_forward_backward_and_scalar_head_collapse():
    # Catches wrong gate scale, selected row placement, and detached index updates.
    x, layer = fixture()
    B = torch.cat((layer.base.weight, layer.base.bias[:, None]), 1)
    D = torch.cat((layer.contrast.weight, layer.contrast.bias[:, None]), 1)
    A = torch.cat((layer.gate.weight, layer.gate.bias[:, None]), 1)
    X = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), 1)
    ids = layer.grown_ids
    g = torch.sigmoid(X @ A.T / layer.tau)
    left, right = X @ (B[ids] - D).T, X @ (B[ids] + D).T
    tree = (X @ B.T).index_copy(1, ids, (1 - g) * left + g * right)
    bypass = (X @ B.T).index_copy(1, ids, left + g * (right - left))
    actual = layer(x)
    probe = torch.linspace(-1, 2, actual.numel(), dtype=x.dtype).reshape_as(actual)
    coordinates = (x, *layer.parameters())
    expected_grads = torch.autograd.grad((tree * probe).sum(), coordinates, retain_graph=True)
    actual_grads = torch.autograd.grad((actual * probe).sum(), coordinates, retain_graph=True)
    assert (actual - tree).abs().max() < 1e-10
    assert (actual - bypass).abs().max() < 1e-10
    for expected, got in zip(expected_grads, actual_grads):
        assert (expected - got).abs().max() < 1e-10
    head = torch.tensor([0.2, -0.3, 1.1, 2.0], dtype=x.dtype)
    collapsed = X @ (head @ B) + (g * (X @ (2 * head[ids, None] * D).T)).sum(1) - X @ (head[ids] @ D)
    assert (actual @ head - collapsed).abs().max() < 1e-10
    assert (actual < 0).any()  # The layer must not apply ReLU.


def test_hardening_exact_discrepancy_and_relu_bound():
    x, layer = fixture()
    base, delta = layer.base(x)[:, layer.grown_ids], layer.contrast(x)
    g = torch.sigmoid(layer.gate(x) / layer.tau)
    left, right = base - delta, base + delta
    hard = torch.where(g >= 0.5, right, left)
    soft = layer(x)[:, layer.grown_ids]
    gap = torch.minimum(g, 1 - g) * (right - left).abs()
    assert ((soft - hard).abs() - gap).abs().max() < 1e-10
    assert ((soft.relu() - hard.relu()).abs() - gap).max() < 1e-10


def test_relu_applied_after_mixture_is_not_leafwise_relu():
    layer = SelectiveAffineLayer(1, 1).double()
    with torch.no_grad():
        layer.base.weight.zero_()
        layer.base.bias.zero_()
    layer.grow(torch.tensor([0]), torch.tensor([[0., 2.]], dtype=torch.float64))
    actual = layer(torch.zeros(1, 1, dtype=torch.float64)).relu().item()
    leafwise = 0.5 * max(-2., 0.) + 0.5 * max(2., 0.)
    assert actual == 0.0
    assert leafwise == 1.0


def test_gate_gradient_matches_formula_and_finite_difference():
    layer = SelectiveAffineLayer(2, 1, tau=1.7).double()
    x = torch.tensor([[1., 2.], [-2., 1.], [0.5, -1.]], dtype=torch.float64)
    X = torch.cat((x, torch.ones(3, 1, dtype=x.dtype)), 1)
    v = torch.tensor([[0.2, -0.5, 0.7]], dtype=x.dtype)
    with torch.no_grad():
        layer.base.weight.zero_()
        layer.base.bias.fill_(3.)  # All points strictly away from ReLU kinks.
    layer.grow(torch.tensor([0]), v)
    prediction = layer(x).relu()
    loss = prediction.square().mean()
    loss.backward()
    gradient = torch.cat((layer.gate.weight.grad, layer.gate.bias.grad[:, None]), 1)[0]
    e = 2 * prediction.detach()[:, 0]
    moment = X.T @ (e[:, None] * X) / len(x)
    assert (gradient - moment @ v[0] / (2 * layer.tau)).abs().max() < 1e-10
    assert torch.count_nonzero(layer.contrast.weight.grad) == 0
    assert torch.count_nonzero(layer.contrast.bias.grad) == 0
    assert gradient.norm() > 0
    step = 1e-5
    with torch.no_grad():
        layer.gate.weight[0, 0] = step
        plus = layer(x).relu().square().mean()
        layer.gate.weight[0, 0] = -step
        minus = layer(x).relu().square().mean()
    assert abs((plus - minus) / (2 * step) - gradient[0]) < 1e-10
