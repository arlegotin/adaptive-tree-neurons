"""Signed first-event moment scoring on an isolated CPU float64 evaluation model."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor, nn

from treeneuron.models.units import SelectiveAffineLayer


class InvalidScoreRun(ValueError):
    """A nonfinite model, prediction, or loss blocks the whole scoring run."""


@dataclass(frozen=True)
class ScoreTable:
    """CPU diagnostic rows in original unit-ID order, including ineligible rows.

    Spectra are signed and ascending. Gaps compare the two largest absolute
    eigenvalues; near ties (gap <= 1e-12 * max(1, spectral radius)) diagnose
    unstable directions, without excluding them.
    ``gram`` is E_F[[x,1][x,1]^T], sufficient for new-direction RMS. Tensors
    remain ordinary serializable tensors; table_hash is recomputed to bind their
    actual current contents, including nonfinite diagnostic bytes.
    """

    ids: Tensor
    contrasts: Tensor
    gradients: Tensor
    adam_scores: Tensor
    sgd_scores: Tensor
    eigenvalues: Tensor
    spectral_gaps: Tensor
    base_rms: Tensor
    projection_rms: Tensor
    eligible: Tensor
    status: tuple[str, ...]
    near_ties: Tensor
    gram: Tensor
    epsilon: float
    tau: float
    adam_epsilon: float
    n_rows: int
    formula_version: str = 'signed-moment-rms-v1'

    @property
    def table_hash(self) -> str:
        digest = hashlib.sha256()
        for field in fields(self):
            value = getattr(self, field.name)
            digest.update(field.name.encode() + b'\0')
            if isinstance(value, Tensor):
                digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(json.dumps(value, allow_nan=False, sort_keys=True).encode())
            digest.update(b'\0')
        return digest.hexdigest()


@contextmanager
def _isolated_rng():
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _target(model: nn.Module) -> tuple[SelectiveAffineLayer, nn.Linear]:
    if not isinstance(model, nn.Module) or any(module.training for module in model.modules()):
        raise ValueError('Scoring requires a deterministic evaluation model')
    layers = [module for module in model.modules() if isinstance(module, SelectiveAffineLayer)]
    if len(layers) != 1:
        raise ValueError('Expected exactly one first SelectiveAffineLayer')
    layer = layers[0]
    if layer.growth_applied:
        raise ValueError('Only the first growth event is supported')
    heads = [module for module in model.modules() if isinstance(module, nn.Linear) and module is not layer.base]
    if len(heads) != 1 or heads[0].in_features != layer.h or heads[0].out_features != 1:
        raise ValueError('Expected one scalar Linear head after first-layer ReLU')
    for module in model.modules():
        if isinstance(module, nn.ReLU) and module.inplace:
            raise ValueError('Scoring requires out-of-place ReLU preactivation preservation')
        if not list(module.children()) and not isinstance(module, (nn.Linear, nn.ReLU)):
            raise ValueError('Only first-layer ReLU scalar regression is supported')
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.device.type != 'cpu' or (tensor.is_floating_point() and tensor.dtype != torch.float64) or tensor.is_complex():
            raise ValueError('All model tensors must be CPU; floating tensors must be float64')
        if not bool(torch.isfinite(tensor).all()):
            raise InvalidScoreRun('Nonfinite model tensor')
    return layer, heads[0]


def _batch_derivatives(model, layer, head, x, y):
    """Capture the actual preactivation and reject altered core computation."""
    captured, preactivation_values, head_inputs = [], [], []
    def capture(_module, args, output):
        if len(args) != 1 or not torch.equal(args[0], x):
            raise ValueError('Selective layer must consume the original fitting inputs')
        captured.append(output.requires_grad_())
        preactivation_values.append(output.detach().clone())
    def head_capture(_module, args):
        head_inputs.append(args[0])
    handles = [layer.register_forward_hook(capture), head.register_forward_pre_hook(head_capture)]
    try:
        with torch.enable_grad():
            prediction = model(x)
            if len(captured) != 1 or len(head_inputs) != 1:
                raise ValueError('Core layer and head must each execute exactly once')
            z = captured[0]
            if not isinstance(prediction, Tensor) or prediction.shape not in ((len(x),), (len(x), 1)):
                raise ValueError('Model must return scalar predictions [N] or [N,1]')
            prediction = prediction.reshape(-1)
            if prediction.dtype != torch.float64 or prediction.device.type != 'cpu':
                raise ValueError('Model predictions must be CPU float64')
            if not bool(torch.isfinite(z).all()) or not bool(torch.isfinite(prediction).all()):
                raise InvalidScoreRun('Nonfinite preactivation or prediction')
            if not torch.equal(z.detach(), preactivation_values[0]):
                raise ValueError('Downstream computation modified the captured preactivation in place')
            if not torch.equal(head_inputs[0], z.relu()):
                raise ValueError('Head must consume ReLU of the selective preactivation')
            expected = torch.nn.functional.linear(z.relu(), head.weight, head.bias).reshape(-1)
            if not torch.equal(prediction, expected):
                raise ValueError('Output must be the scalar head prediction')
            losses = (prediction - y).square()
            if not bool(torch.isfinite(losses).all()) or not bool(torch.isfinite(losses.sum())):
                raise InvalidScoreRun('Nonfinite row loss or sum loss')
            e = torch.autograd.grad(losses.sum(), z)[0]
            return z.detach(), e.detach()
    finally:
        for handle in handles:
            handle.remove()


def score_units(model: nn.Module, X_fit: Tensor, y_fit: Tensor, *, epsilon: float,
                tau: float, adam_epsilon: float, batch_size: int,
                unit_block_size: int) -> ScoreTable:
    """Score squared-error regression without changing caller state or RNGs.

    Supports Sequential or wrappers executing x -> selective -> ReLU -> scalar
    Linear, optionally squeezing its last axis. ReLU must be out of place;
    downstream mutation of the captured preactivation is rejected. Caller supplies an eval CPU64
    copy. An additional private copy protects even buffers/custom wrapper state.
    Repeated forward passes per unit block trade compute for bounded memory:
    O(batch*h + block*(d+1)^2), plus the O(h*(d+1)) result table.
    """
    for value, name in ((epsilon, 'epsilon'), (tau, 'tau'), (adam_epsilon, 'adam_epsilon')):
        _positive(value, name)
    for value in (batch_size, unit_block_size):
        if type(value) is not int or value <= 0:
            raise ValueError('Batch and unit block sizes must be positive integers')
    layer, _ = _target(model)
    for tensor in (X_fit, y_fit):
        if not isinstance(tensor, Tensor) or tensor.device.type != 'cpu' or tensor.dtype != torch.float64:
            raise ValueError('F inputs and targets must be CPU float64 tensors')
        if not bool(torch.isfinite(tensor).all()):
            raise InvalidScoreRun('Nonfinite F input or target')
    if X_fit.ndim != 2:
        raise ValueError('Expected fitting inputs [N,d]')
    n, h, p = len(X_fit), layer.h, layer.d + 1
    if X_fit.shape[1] != layer.d or n == 0 or y_fit.shape not in ((n,), (n, 1)):
        raise ValueError('Expected nonempty X [N,d] and y [N] or [N,1]')
    if float(tau) != layer.tau:
        raise ValueError('Scoring tau must match the selective layer')
    y = y_fit.detach().reshape(-1)
    with _isolated_rng():
        evaluation = copy.deepcopy(model)
        layer, head = _target(evaluation)
        gram = torch.zeros(p, p, dtype=torch.float64, device='cpu')
        for start in range(0, n, batch_size):
            x = X_fit[start:start + batch_size].detach()
            augmented = torch.cat((x, x.new_ones((len(x), 1))), 1)
            gram += augmented.T @ augmented / n
        vectors = [torch.zeros(h, p, dtype=torch.float64, device='cpu') for _ in range(3)]
        contrasts, gradients, eigenvalues = vectors
        base_rms, projection_rms, gaps, adam, sgd = [torch.zeros(h, dtype=torch.float64, device='cpu') for _ in range(5)]
        eligible, ties = torch.zeros(h, dtype=torch.bool, device='cpu'), torch.zeros(h, dtype=torch.bool, device='cpu')
        status = ['nonfinite_statistics'] * h
        for first in range(0, h, unit_block_size):
            last = min(h, first + unit_block_size)
            moments = torch.zeros(last - first, p, p, dtype=torch.float64, device='cpu')
            squares = torch.zeros(last - first, dtype=torch.float64, device='cpu')
            for start in range(0, n, batch_size):
                x = X_fit[start:start + batch_size].detach()
                z, e = _batch_derivatives(evaluation, layer, head, x, y[start:start + batch_size])
                augmented = torch.cat((x, x.new_ones((len(x), 1))), 1)
                # Per-unit GEMMs avoid any einsum contraction path that builds
                # batch-by-feature-by-feature outer products.
                for offset, j in enumerate(range(first, last)):
                    moments[offset] += augmented.T @ (e[:, j:j + 1] * augmented) / n
                squares += z[:, first:last].square().sum(0) / n
            for offset, j in enumerate(range(first, last)):
                moment = (moments[offset] + moments[offset].T) / 2
                base_rms[j] = squares[offset].sqrt()
                if not bool(torch.isfinite(moment).all()) or not bool(torch.isfinite(base_rms[j])):
                    eigenvalues[j].fill_(float('nan'))
                    continue
                try:
                    values, directions = torch.linalg.eigh(moment)
                except torch.linalg.LinAlgError:
                    status[j] = 'eigendecomposition_failed'
                    eigenvalues[j].fill_(float('nan'))
                    continue
                eigenvalues[j] = values
                magnitudes = values.abs().sort(descending=True).values
                gaps[j] = magnitudes[0] - magnitudes[1]
                ties[j] = gaps[j] <= 1e-12 * magnitudes[0].clamp_min(1)
                q = directions[:, values.abs().argmax()]
                if q[q.abs().argmax()] < 0:
                    q = -q
                projection_rms[j] = (q @ gram @ q).clamp_min(0).sqrt()
                contrasts[j] = epsilon * base_rms[j] * q / projection_rms[j].clamp_min(1e-12)
                gradients[j] = moment @ contrasts[j] / (2 * tau)
                adam[j] = (gradients[j].square() / (gradients[j].abs() + adam_epsilon)).sum() / (2 * p)
                sgd[j] = gradients[j].square().sum() / (2 * p)
                statistics = torch.cat((values, contrasts[j], gradients[j], torch.stack((base_rms[j], projection_rms[j], gaps[j], adam[j], sgd[j]))))
                if not bool(torch.isfinite(statistics).all()):
                    continue
                if base_rms[j] == 0:
                    status[j] = 'zero_base_rms'
                elif adam[j] <= 0 or sgd[j] <= 0:
                    status[j] = 'zero_score'
                else:
                    eligible[j], status[j] = True, 'eligible'
    return ScoreTable(torch.arange(h, dtype=torch.int64, device='cpu'), contrasts, gradients, adam, sgd,
                      eigenvalues, gaps, base_rms, projection_rms, eligible, tuple(status),
                      ties, gram, float(epsilon), float(tau), float(adam_epsilon), n)
