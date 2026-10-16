"""Standalone Appendix A CPU reference; never import into a training process.

Draw order and constants: docs/research/2026-10-16-evidence-and-machine-record.md.
This checks algebra and local descent, not predictive performance or MPS.
"""
import json
import platform

import torch


def run_checks():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(1602026)
    n, d, h = 37, 7, 5
    x = torch.randn(n, d, device="cpu", requires_grad=True)
    left = torch.randn(h, d + 1, requires_grad=True)
    right = torch.randn(h, d + 1, requires_grad=True)
    gate = torch.randn(h, d + 1, requires_grad=True)
    X = torch.cat((x, torch.ones(n, 1)), dim=1)
    g = torch.sigmoid(X @ gate.T)
    l, r = X @ left.T, X @ right.T
    tree = (1 - g) * l + g * r
    skip = l + g * (r - l)
    probe = torch.randn_like(tree)
    coordinates = (x, left, right, gate)
    tree_grads = torch.autograd.grad((tree * probe).sum(), coordinates, retain_graph=True)
    skip_grads = torch.autograd.grad((skip * probe).sum(), coordinates, retain_graph=True)
    v, c = torch.randn(h), torch.randn(())
    collapsed = X @ (v @ left) + (g * (X @ (v[:, None] * (right - left)).T)).sum(1) + c

    # Continue the same RNG stream, including the preceding probe/head draws.
    B, head, y = torch.randn(h, d + 1), torch.randn(h), torch.randn(n)
    X0 = X.detach()
    base = X0 @ B.T
    pre = base.clone().requires_grad_()
    old_pred = pre.relu() @ head
    old_loss = ((old_pred - y) ** 2).mean()
    delta = torch.autograd.grad(((old_pred - y) ** 2).sum(), pre)[0].detach()
    j = 2
    moment = X0.T @ (delta[:, j:j + 1] * X0) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(moment)
    q = eigenvectors[:, eigenvalues.abs().argmax()]
    # Appendix A intentionally uses the eigensolver's sign without normalization.
    D = (0.05 * q).detach().requires_grad_()
    A = torch.zeros(d + 1, requires_grad=True)
    grown = base.clone()
    grown[:, j] += torch.tanh(X0 @ A / 2) * (X0 @ D)
    prediction = grown.relu() @ head
    loss = ((prediction - y) ** 2).mean()
    grad_A, grad_D = torch.autograd.grad(loss, (A, D))
    step = 1e-4 / (1 + grad_A.norm())
    with torch.no_grad():
        A1 = A - step * grad_A
        after = base.clone()
        after[:, j] += torch.tanh(X0 @ A1 / 2) * (X0 @ D)
        loss_after = ((after.relu() @ head - y) ** 2).mean()

    hard = torch.where(g >= 0.5, r, l)
    gap = torch.minimum(g, 1 - g) * (r - l).abs()
    results = {
        "forward_max_abs": (tree - skip).abs().max().item(),
        "shared_gradient_max_abs": max((a - b).abs().max().item()
                                        for a, b in zip(tree_grads, skip_grads)),
        "collapse_max_abs": (tree @ v + c - collapsed).abs().max().item(),
        "growth_preservation_max_abs": (prediction - old_pred).abs().max().item(),
        "gate_gradient_formula_max_abs": (grad_A - moment @ D.detach() / 2).abs().max().item(),
        "contrast_gradient_norm": grad_D.norm().item(),
        "gate_gradient_norm": grad_A.norm().item(),
        "loss_before": old_loss.item(), "loss_after": loss_after.item(),
        "gate_step_size": step.item(),
        "soft_hard_gap_max_abs": ((tree - hard).abs() - gap).abs().max().item(),
        "relu_bound_max_violation": ((tree.relu() - hard.relu()).abs() - gap).max().item(),
    }
    for key in ("forward_max_abs", "shared_gradient_max_abs", "collapse_max_abs",
                "gate_gradient_formula_max_abs", "soft_hard_gap_max_abs",
                "relu_bound_max_violation"):
        if not results[key] < 1e-12:
            raise AssertionError(f"{key}: {results[key]} >= 1e-12")
    for key in ("growth_preservation_max_abs", "contrast_gradient_norm"):
        if results[key] != 0:
            raise AssertionError(f"{key}: expected exactly zero, got {results[key]}")
    if not results["loss_after"] < results["loss_before"]:
        raise AssertionError("Tiny gate-only step did not decrease the loss")
    return {
        "schema_version": 1, "check": "appendix_a", "status": "passed", "seed": 1602026,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "architecture": platform.machine(),
                        "dtype": str(x.dtype), "device": str(x.device),
                        "threads": torch.get_num_threads()},
        "shapes": {"n": n, "d": d, "h": h},
        "constants": {"epsilon": 0.05, "tau": 1, "unit": j,
                      "error_threshold": 1e-12},
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True, allow_nan=False))
