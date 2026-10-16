"""Compact affine preactivations with one function-preserving growth event."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn

from treeneuron.growth.types import GrowthManifest, ParameterMapping


class SelectiveAffineLayer(nn.Module):
    """Return [batch, h] preactivations; the caller supplies any activation.

    Only selected units own gate/contrast parameters. ``grow`` accepts CPU int64
    IDs and contrasts on the base dtype/device, with explicit bias last.
    """

    def __init__(self, d: int, h: int, tau: float = 1.0):
        super().__init__()
        if any(type(size) is not int or size <= 0 for size in (d, h)):
            raise ValueError('d and h must be positive integers')
        if isinstance(tau, bool) or not isinstance(tau, (int, float)):
            raise TypeError('tau must be a finite positive number')
        if not math.isfinite(tau) or tau <= 0:
            raise ValueError('tau must be finite and positive')
        self.d, self.h, self.tau = d, h, float(tau)
        self.base = nn.Linear(d, h)
        self.register_buffer('grown_ids', torch.empty(0, dtype=torch.int64))
        self.gate: nn.Linear | None = None
        self.contrast: nn.Linear | None = None
        self.growth_applied = False

    def forward(self, x: Tensor) -> Tensor:
        z = self.base(x)
        if self.grown_ids.numel() == 0:
            return z
        s = torch.tanh(self.gate(x) / (2.0 * self.tau))
        selected = z.index_select(1, self.grown_ids) + s * self.contrast(x)
        return z.index_copy(1, self.grown_ids, selected)

    def grow(self, ids: Tensor, contrasts: Tensor) -> GrowthManifest:
        """Validate atomically, sort ID/contrast rows together, add zero gates."""
        if self.growth_applied:
            raise RuntimeError('Only one growth event is allowed')
        if not isinstance(ids, Tensor) or not isinstance(contrasts, Tensor):
            raise TypeError('ids and contrasts must be tensors')
        if ids.dtype != torch.int64 or ids.device.type != 'cpu' or ids.ndim != 1:
            raise ValueError('ids must be a 1-D CPU int64 tensor')
        m = ids.numel()
        weight = self.base.weight
        if contrasts.shape != (m, self.d + 1):
            raise ValueError('contrasts must have shape [m, d+1]')
        if contrasts.dtype != weight.dtype or contrasts.device != weight.device:
            raise ValueError('contrasts must match the base dtype and device')
        if not bool(torch.isfinite(contrasts).all()):
            raise ValueError('contrasts must be finite')
        sorted_ids, order = ids.sort()
        if m and (sorted_ids[0] < 0 or sorted_ids[-1] >= self.h):
            raise ValueError('ids are out of range')
        if m > 1 and bool((sorted_ids[1:] == sorted_ids[:-1]).any()):
            raise ValueError('ids must be unique')
        old = dict(self.named_parameters())
        old_count = sum(parameter.numel() for parameter in old.values())
        if m:
            rows = contrasts.detach().index_select(0, order.to(weight.device))
            gate = nn.Linear(self.d, m, device=weight.device, dtype=weight.dtype)
            contrast = nn.Linear(self.d, m, device=weight.device, dtype=weight.dtype)
            with torch.no_grad():
                gate.weight.zero_()
                gate.bias.zero_()
                contrast.weight.copy_(rows[:, :self.d])
                contrast.bias.copy_(rows[:, self.d])
            self.gate, self.contrast = gate, contrast
        self.grown_ids = sorted_ids.to(weight.device).clone()
        self.growth_applied = True
        mappings = tuple(ParameterMapping(
            old_name=name, new_name=name, old_shape=tuple(parameter.shape),
            new_shape=tuple(parameter.shape),
            old_slice=tuple((0, n) for n in parameter.shape),
            new_slice=tuple((0, n) for n in parameter.shape),
        ) for name, parameter in old.items())
        added = 2 * m * (self.d + 1)
        return GrowthManifest(
            schema_version=1, d=self.d, h=self.h,
            grown_ids=tuple(sorted_ids.tolist()), growth_applied=True,
            old_parameter_count=old_count, new_parameter_count=old_count + added,
            added_parameter_count=added, old_parameter_names=tuple(old),
            new_parameter_names=tuple(name for name, _ in self.named_parameters() if name not in old),
            parameter_mappings=mappings,
        )

    def export_config(self) -> dict[str, Any]:
        """Export topology; pair with state_dict for the parameter values.

        Dtype/device are selected by the caller via ``to`` before loading state.
        Empty completed events are distinct from layers that have never grown.
        """
        return dict(schema_version=1, d=self.d, h=self.h, tau=self.tau,
                    growth_applied=self.growth_applied,
                    grown_ids=self.grown_ids.detach().cpu().tolist())

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> SelectiveAffineLayer:
        """Rebuild compact topology before loading tensors, without dormant rows."""
        expected = {'schema_version', 'd', 'h', 'tau', 'growth_applied', 'grown_ids'}
        if set(config) != expected or type(config['schema_version']) is not int or config['schema_version'] != 1:
            raise ValueError('Invalid selective layer configuration schema')
        applied, ids = config['growth_applied'], config['grown_ids']
        if type(applied) is not bool or not isinstance(ids, (list, tuple)):
            raise ValueError('Invalid growth event state')
        if any(type(j) is not int for j in ids) or (ids and not applied):
            raise ValueError('Invalid grown_ids or event state')
        if list(ids) != sorted(set(ids)):
            raise ValueError('grown_ids must be sorted and unique')
        layer = cls(config['d'], config['h'], config['tau'])
        if applied:
            layer.grow(torch.tensor(ids, dtype=torch.int64),
                       layer.base.weight.new_zeros((len(ids), layer.d + 1)))
        return layer
