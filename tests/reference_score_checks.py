"""Standalone Appendix B normalized-score reference, isolated from training.

This small dense reference is independent of the future package score API.
"""
import json
import platform

import torch


def normalized_scores(X, W, head, y):
    """Apply spec §5 literally on the nondegenerate reference draw."""
    n, p = X.shape
    pre = (X @ W.T).detach().requires_grad_()
    row_losses = (pre.relu() @ head - y) ** 2
    e = torch.autograd.grad(row_losses.sum(), pre)[0].detach()
    contrasts, gradients, scores = [], [], []
    for j in range(W.shape[0]):
        moment = X.T @ (e[:, j:j + 1] * X) / n
        moment = (moment + moment.T) / 2
        eigenvalues, eigenvectors = torch.linalg.eigh(moment)
        q = eigenvectors[:, eigenvalues.abs().argmax()]
        if q[q.abs().argmax()] < 0:
            q = -q
        rms_base = pre[:, j].detach().square().mean().sqrt()
        rms_projection = (X @ q).square().mean().sqrt()
        contrast = 0.05 * rms_base * q / rms_projection.clamp_min(1e-12)
        gradient = moment @ contrast / (2 * 1.0)
        score = (gradient.square() / (gradient.abs() + 1e-8)).sum() / (2 * p)
        contrasts.append(contrast)
        gradients.append(gradient)
        scores.append(score)
    return torch.stack(contrasts), torch.stack(gradients), torch.stack(scores)


def run_checks():
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(1602027)
    n, d, h = 41, 5, 4
    X = torch.cat((torch.randn(n, d, device="cpu"), torch.ones(n, 1)), dim=1)
    W, head, y = torch.randn(h, d + 1), torch.randn(h), torch.randn(n)
    V, U, S = normalized_scores(X, W, head, y)
    scales = torch.tensor([0.3, 2.0, 7.0, 1.1])
    _, scaled_U, scaled_S = normalized_scores(X, W * scales[:, None], head / scales, y)
    permutation = torch.tensor([2, 0, 3, 1])
    _, _, permuted_S = normalized_scores(X, W[permutation], head[permutation], y)
    j = S.argmax().item()
    A = torch.zeros(d + 1, requires_grad=True)
    base = X @ W.T
    grown = base.clone()
    grown[:, j] += torch.tanh(X @ A / 2) * (X @ V[j])
    loss = ((grown.relu() @ head - y) ** 2).mean()
    optimizer = torch.optim.AdamW([A], lr=1e-6, eps=1e-8, weight_decay=0.0)
    loss.backward()
    grad_error = (A.grad - U[j]).abs().max().item()
    predicted_decrease = (1e-6 * (U[j].square() / (U[j].abs() + 1e-8)).sum()).item()
    loss_before = loss.item()
    optimizer.step()
    with torch.no_grad():
        after = base.clone()
        after[:, j] += torch.tanh(X @ A / 2) * (X @ V[j])
        loss_after = ((after.relu() @ head - y) ** 2).mean().item()
    observed_decrease = loss_before - loss_after
    relative_error = abs(observed_decrease - predicted_decrease) / predicted_decrease
    results = {
        "rescaled_gradient_max_abs": (U - scaled_U).abs().max().item(),
        "rescaled_score_max_abs": (S - scaled_S).abs().max().item(),
        "permuted_score_max_abs": (S[permutation] - permuted_S).abs().max().item(),
        "gate_gradient_formula_max_abs": grad_error,
        "contrasts": V.tolist(), "gate_gradients": U.tolist(), "adam_scores": S.tolist(),
        "gate_gradient_norm": U[j].norm().item(), "contrast_norm": V[j].norm().item(),
        "selected_unit": j, "loss_before": loss_before, "loss_after": loss_after,
        "predicted_loss_decrease": predicted_decrease,
        "observed_loss_decrease": observed_decrease,
        "adam_prediction_relative_error": relative_error,
    }
    for key in ("rescaled_gradient_max_abs", "rescaled_score_max_abs",
                "permuted_score_max_abs", "gate_gradient_formula_max_abs"):
        if not results[key] < 1e-12:
            raise AssertionError(f"{key}: {results[key]} >= 1e-12")
    if not observed_decrease > 0:
        raise AssertionError("First gate-only Adam step did not decrease the loss")
    if not relative_error < 1e-4:
        raise AssertionError(f"First Adam decrease relative error {relative_error} >= 1e-4")
    return {
        "schema_version": 1, "check": "appendix_b", "status": "passed", "seed": 1602027,
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "platform": platform.platform(), "architecture": platform.machine(),
                        "dtype": str(X.dtype), "device": str(X.device),
                        "threads": torch.get_num_threads()},
        "shapes": {"n": n, "d": d, "h": h},
        "constants": {"epsilon": 0.05, "tau": 1, "adam_epsilon": 1e-8,
                      "learning_rate": 1e-6, "weight_decay": 0.0,
                      "error_threshold": 1e-12, "adam_relative_error_threshold": 1e-4},
        "results": results,
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, sort_keys=True, allow_nan=False))
