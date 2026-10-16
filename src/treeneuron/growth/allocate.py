"""Fixed-count matched placement and top-ID Gaussian direction ablation."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from treeneuron.growth.score import ScoreTable


@dataclass(frozen=True)
class Allocation:
    requested_count: int
    effective_count: int
    ids: Tensor
    contrasts: Tensor
    policy: str
    seed: int
    table_hash: str
    direction_attempts: Tensor
    cost: dict[str, int]


def allocate(table: ScoreTable, requested: int, *, policy: str, seed: int) -> Allocation:
    """Select sorted IDs; ``gaussian`` uses Adam IDs and independent directions.

    Random placement shares each ID's exact table contrast. ``keep`` records the
    requested budget and zero actual growth. The seed is local and never draws
    from training RNG. A caller should pass a separately namespaced contrast seed
    for gaussian. Cost counts work requirements, not fabricated timings: full
    shared scoring has already executed; standalone random contrast work has not.
    """
    if type(requested) is not int or requested < 0:
        raise ValueError('requested must be a nonnegative integer')
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError('seed must be an integer in [0, 2**63)')
    if policy not in ('adam', 'sgd', 'random', 'keep', 'gaussian'):
        raise ValueError('Unknown allocation policy')
    # Strictly positive finite scores guard diagnostic or caller-modified tables.
    valid = (table.eligible & torch.isfinite(table.adam_scores) & (table.adam_scores > 0)
             & torch.isfinite(table.sgd_scores) & (table.sgd_scores > 0)
             & torch.isfinite(table.contrasts).all(1))
    rows = torch.where(valid)[0]
    count = 0 if policy == 'keep' else min(requested, len(rows))
    generator = torch.Generator(device='cpu').manual_seed(seed)
    if policy == 'random':
        selected = rows[torch.randperm(len(rows), generator=generator, device='cpu')[:count]]
    else:
        scores = table.sgd_scores if policy == 'sgd' else table.adam_scores
        ranked = sorted(rows.tolist(), key=lambda row: (-scores[row].item(), table.ids[row].item()))
        selected = torch.tensor(ranked[:count], dtype=torch.int64, device='cpu')
    selected = selected[table.ids[selected].argsort()]
    ids, contrasts = table.ids[selected].clone(), table.contrasts[selected].clone()
    attempts = torch.zeros(count, dtype=torch.int64, device='cpu')
    if policy == 'gaussian':
        for i, row in enumerate(selected.tolist()):
            for attempt in range(1, 101):
                attempts[i] = attempt
                q = torch.randn(table.contrasts.shape[1], dtype=torch.float64, generator=generator, device='cpu')
                norm = q.norm()
                if not bool(torch.isfinite(q).all()) or not bool(torch.isfinite(norm)) or norm <= 0:
                    continue
                q = q / norm
                if q[q.abs().argmax()] < 0:
                    q = -q
                rms = (q @ table.gram @ q).clamp_min(0).sqrt()
                candidate = table.epsilon * table.base_rms[row] * q / rms.clamp_min(1e-12)
                if bool(torch.isfinite(rms)) and rms > 0 and bool(torch.isfinite(candidate).all()) and candidate.norm() > 0:
                    contrasts[i] = candidate
                    break
            else:
                raise ValueError(f'Gaussian direction unusable for unit {ids[i].item()} after 100 attempts')
    return Allocation(requested, count, ids, contrasts, policy, seed, table.table_hash, attempts,
                      {'shared_scored_units': len(table.ids),
                       'standalone_contrast_units_required': count if policy == 'random' else 0})
