"""Compact, active static gated coordinates; counts specify topology, not ID search."""
from __future__ import annotations
import math
import torch
from torch import nn


class StaticAffineLayer(nn.Module):
    def __init__(self, d, h, count, coordinates, tau):
        super().__init__()
        if (type(d) is not int or not 0 < d <= 16 or type(h) is not int or not 0 < h <= 64
                or type(count) is not int or not 0 <= count <= h
                or coordinates not in ('centered', 'direct_leaf')
                or type(tau) not in (float, int) or not math.isfinite(tau) or tau <= 0):
            raise ValueError('Invalid bounded static topology')
        self.d, self.h, self.count, self.coordinates, self.tau = d, h, count, coordinates, float(tau)
        self.ordinary = nn.Linear(d, h-count) if count < h else None
        self.gate = nn.Linear(d, count) if count else None
        if coordinates == 'centered':
            self.center = nn.Linear(d, count) if count else None
            self.contrast = nn.Linear(d, count) if count else None
        else:
            self.left = nn.Linear(d, count) if count else None
            self.right = nn.Linear(d, count) if count else None

    def forward(self, x):
        parts = []
        if self.count:
            if self.coordinates == 'centered':
                parts.append(self.center(x) + torch.tanh(self.gate(x)/(2*self.tau))*self.contrast(x))
            else:
                gate = torch.sigmoid(self.gate(x)/self.tau)
                parts.append(gate*self.left(x) + (1-gate)*self.right(x))
        if self.ordinary is not None:
            parts.append(self.ordinary(x))
        return torch.cat(parts, dim=1)

    def export_config(self):
        return dict(d=self.d, h=self.h, count=self.count, coordinates=self.coordinates, tau=self.tau)


class StaticRegressor(nn.Module):
    def __init__(self, d, h, count, coordinates, tau=1.):
        super().__init__()
        self.layer = StaticAffineLayer(d, h, count, coordinates, tau)
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(torch.relu(self.layer(x))).squeeze(-1)

    def export_config(self):
        return dict(schema_version=1, kind='static_gated_relu_scalar', layer=self.layer.export_config())

    @classmethod
    def from_config(cls, config):
        if (set(config) != {'schema_version','kind','layer'} or type(config['schema_version']) is not int
                or config['schema_version'] != 1 or config['kind'] != 'static_gated_relu_scalar'
                or set(config['layer']) != {'d','h','count','coordinates','tau'}):
            raise ValueError('Invalid static model topology')
        return cls(**config['layer'])
