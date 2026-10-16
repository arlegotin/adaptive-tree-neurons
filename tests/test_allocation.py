"""Allocation preserves fixed count and per-ID contrasts without training RNG."""
from dataclasses import replace

import pytest
import torch

from treeneuron.growth.allocate import allocate
from test_score import score, score_case


def test_top_random_fixed_count_exact_contrasts_and_rng():
    model, X, y = score_case()
    table = score(model, X, y)
    rng = torch.get_rng_state()
    top = allocate(table, 2, policy='adam', seed=42)
    random = allocate(table, 2, policy='random', seed=42)
    expected = table.ids[torch.randperm(4, generator=torch.Generator().manual_seed(42))[:2]].sort().values
    assert torch.equal(random.ids, expected)
    for result in (top, random):
        assert result.requested_count == result.effective_count == 2
        assert torch.equal(result.contrasts, table.contrasts[result.ids])
        assert result.table_hash == table.table_hash
    assert torch.equal(rng, torch.get_rng_state())
    assert random.cost['shared_scored_units'] == 4
    assert random.cost['standalone_contrast_units_required'] == 2


def test_ties_sgd_cap_keep_and_empty():
    model, X, y = score_case()
    table = score(model, X, y)
    tied = replace(table, adam_scores=torch.ones(4, dtype=X.dtype), sgd_scores=torch.tensor([1., 4., 3., 2.], dtype=X.dtype))
    assert allocate(tied, 2, policy='adam', seed=0).ids.tolist() == [0, 1]
    assert allocate(tied, 2, policy='sgd', seed=0).ids.tolist() == [1, 2]
    assert allocate(table, 100, policy='random', seed=0).effective_count == 4
    keep = allocate(table, 2, policy='keep', seed=0)
    assert keep.effective_count == 0 and keep.requested_count == 2
    zero = replace(table, eligible=torch.zeros(4, dtype=torch.bool), adam_scores=torch.zeros(4, dtype=X.dtype), sgd_scores=torch.zeros(4, dtype=X.dtype))
    for policy in ('adam', 'sgd', 'random', 'gaussian', 'keep'):
        result = allocate(zero, 2, policy=policy, seed=0)
        assert result.ids.shape == (0,) and result.contrasts.shape == (0, 6)
    for policy in ('adam', 'random'):
        assert allocate(table, 0, policy=policy, seed=0).effective_count == 0
    with pytest.raises(ValueError): allocate(table, -1, policy='adam', seed=0)
    with pytest.raises(ValueError): allocate(table, 1, policy='unknown', seed=0)


def test_gaussian_top_ids_exact_rms_local_seed_and_hash_binding():
    model, X, y = score_case()
    table = score(model, X, y)
    top = allocate(table, 3, policy='adam', seed=1)
    rng = torch.get_rng_state()
    gaussian = allocate(table, 3, policy='gaussian', seed=9)
    again = allocate(table, 3, policy='gaussian', seed=9)
    assert torch.equal(gaussian.ids, top.ids)
    assert torch.equal(gaussian.contrasts, again.contrasts)
    assert torch.equal(rng, torch.get_rng_state())
    assert gaussian.direction_attempts.tolist() == [1, 1, 1]
    augmented = torch.cat((X, torch.ones(len(X), 1, dtype=X.dtype)), 1)
    torch.testing.assert_close((augmented @ gaussian.contrasts.T).square().mean(0).sqrt(), .05 * table.base_rms[gaussian.ids])
    for row in gaussian.contrasts:
        assert row[row.abs().argmax()] > 0
    assert not torch.equal(top.contrasts, gaussian.contrasts)
    assert replace(table, epsilon=.2).table_hash != table.table_hash
    assert replace(table, status=('changed',) + table.status[1:]).table_hash != table.table_hash
    modified = table.contrasts.clone(); modified[0, 0] += 1
    assert replace(table, contrasts=modified).table_hash != table.table_hash


def test_gaussian_redraws_zero_and_nonfinite_directions(monkeypatch):
    model, X, y = score_case()
    table = score(model, X, y)
    original = torch.randn
    calls = 0
    def bad_then_real(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return torch.full(args, 0. if calls == 1 else float('nan'), dtype=kwargs['dtype'])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'randn', bad_then_real)
    result = allocate(table, 1, policy='gaussian', seed=1)
    assert result.direction_attempts.tolist() == [3]
    augmented = torch.cat((X, torch.ones(len(X), 1, dtype=X.dtype)), 1)
    rms = (augmented @ result.contrasts.T).square().mean(0).sqrt()
    torch.testing.assert_close(rms, .05 * table.base_rms[result.ids])


def test_gaussian_unusable_fitting_geometry_rejects_safely():
    model, X, y = score_case()
    table = score(model, X, y)
    with pytest.raises(ValueError, match='100 attempts'):
        allocate(replace(table, gram=torch.zeros_like(table.gram)), 1, policy='gaussian', seed=1)


def test_allocation_stays_cpu_with_another_default_factory_device():
    model, X, y = score_case()
    table = score(model, X, y)
    with torch.device('meta'):
        for policy in ('adam', 'random', 'gaussian', 'keep'):
            result = allocate(table, 2, policy=policy, seed=0)
            assert result.ids.device.type == result.contrasts.device.type == 'cpu'
