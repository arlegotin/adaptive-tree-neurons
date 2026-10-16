"""Compact growth topology, validation and optimizer identity contracts."""
import json

import pytest
import torch

from treeneuron.growth.types import GrowthManifest
from treeneuron.models.units import SelectiveAffineLayer


def test_parameter_count():
    layer, head = SelectiveAffineLayer(16, 64), torch.nn.Linear(64, 1)
    count = lambda: sum(p.numel() for p in layer.parameters()) + sum(p.numel() for p in head.parameters())
    assert count() == 1153
    layer.grow(torch.arange(16, dtype=torch.int64), torch.ones(16, 17))
    assert count() == 1697


def test_grow_sorts_ids_with_contrasts_and_preserves_parameter_objects():
    layer = SelectiveAffineLayer(2, 4).double()
    old = dict(layer.named_parameters())
    x = torch.tensor([[1., -2.], [3., 4.]], dtype=torch.float64)
    before = layer(x).detach().clone()
    ids = torch.tensor([3, 1])
    contrasts = torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype=x.dtype)
    manifest = layer.grow(ids, contrasts)
    assert isinstance(manifest, GrowthManifest)
    assert layer.grown_ids.tolist() == [1, 3]
    assert torch.equal(layer.contrast.weight, contrasts[[1, 0], :2])
    assert torch.equal(layer.contrast.bias, contrasts[[1, 0], 2])
    assert torch.equal(layer(x), before)
    for name, parameter in old.items():
        assert dict(layer.named_parameters())[name] is parameter
    assert manifest.schema_version == 1
    assert manifest.grown_ids == (1, 3)
    assert manifest.growth_applied is True
    assert manifest.old_parameter_names == ('base.weight', 'base.bias')
    assert manifest.new_parameter_names == ('gate.weight', 'gate.bias', 'contrast.weight', 'contrast.bias')
    assert (manifest.old_parameter_count, manifest.new_parameter_count, manifest.added_parameter_count) == (12, 24, 12)
    for mapping in manifest.parameter_mappings:
        assert mapping.kind == 'identity'
        assert mapping.old_name == mapping.new_name
        original = old[mapping.old_name]
        assert mapping.old_shape == mapping.new_shape == tuple(original.shape)
        assert mapping.old_slice == mapping.new_slice == tuple((0, n) for n in original.shape)
    assert len(manifest.parameter_mappings) == 2
    assert GrowthManifest.from_dict(json.loads(json.dumps(manifest.to_dict()))) == manifest
    # Caller mutation must not corrupt stored IDs/contrasts or manifest.
    ids.zero_()
    contrasts.zero_()
    assert layer.grown_ids.tolist() == [1, 3]
    assert layer.contrast.bias.tolist() == [60., 30.]


@pytest.mark.parametrize('ids, contrasts', [
    (torch.tensor([1, 1]), torch.ones(2, 3)),
    (torch.tensor([-1]), torch.ones(1, 3)),
    (torch.tensor([4]), torch.ones(1, 3)),
    (torch.tensor([1], dtype=torch.int32), torch.ones(1, 3)),
    (torch.tensor([[1]]), torch.ones(1, 3)),
    (torch.tensor([1]), torch.ones(1, 2)),
    (torch.tensor([1]), torch.ones(1, 3, 1)),
    (torch.tensor([1]), torch.ones(1, 3, dtype=torch.float64)),
    (torch.tensor([1]), torch.tensor([[1., float('nan'), 1.]])),
    (torch.tensor([1]), torch.tensor([[1., 1., float('inf')]])),
    (torch.tensor([1]), torch.ones(1, 3, device='meta')),
    (torch.tensor([1], device='meta'), torch.ones(1, 3)),
    ([1], torch.ones(1, 3)),
    (torch.tensor([1]), [[1., 2., 3.]]),
])
def test_invalid_growth_is_atomic(ids, contrasts):
    layer = SelectiveAffineLayer(2, 4)
    with pytest.raises((ValueError, TypeError)):
        layer.grow(ids, contrasts)
    assert layer.growth_applied is False
    assert list(dict(layer.named_parameters())) == ['base.weight', 'base.bias']
    layer.grow(torch.tensor([0]), torch.ones(1, 3))


@pytest.mark.parametrize('tau', [0., -1., float('nan'), float('inf'), -float('inf'), True])
def test_invalid_tau(tau):
    with pytest.raises((ValueError, TypeError)):
        SelectiveAffineLayer(2, 4, tau=tau)


@pytest.mark.parametrize('ids', [[], [2, 0]])
def test_serialization_reconstructs_topology_and_records_empty_event(ids, tmp_path):
    layer = SelectiveAffineLayer(2, 3).double()
    manifest = layer.grow(torch.tensor(ids, dtype=torch.int64), torch.ones(len(ids), 3, dtype=torch.float64))
    assert layer.growth_applied is True
    if not ids:
        assert layer.gate is None and layer.contrast is None
        assert manifest.new_parameter_names == ()
        assert manifest.added_parameter_count == 0
    config = json.loads(json.dumps(layer.export_config()))
    restored = SelectiveAffineLayer.from_config(config).double()
    path = tmp_path / 'layer.pt'
    torch.save(layer.state_dict(), path)
    restored.load_state_dict(torch.load(path, weights_only=True))
    assert restored.export_config() == config
    x = torch.ones(2, 2, dtype=torch.float64)
    assert torch.equal(layer(x), restored(x))
    with pytest.raises(RuntimeError):
        restored.grow(torch.tensor([], dtype=torch.int64), torch.empty(0, 3, dtype=torch.float64))


def test_ungrown_serialization_retains_future_growth():
    layer = SelectiveAffineLayer.from_config(SelectiveAffineLayer(2, 3).export_config())
    assert layer.growth_applied is False
    layer.grow(torch.tensor([1]), torch.ones(1, 3))
    with pytest.raises(RuntimeError):
        layer.grow(torch.tensor([2]), torch.ones(1, 3))


@pytest.mark.parametrize('change', [
    {'schema_version': 2}, {'growth_applied': False, 'grown_ids': [1]},
    {'grown_ids': [1, 1], 'growth_applied': True},
    {'grown_ids': [3], 'growth_applied': True}, {'growth_applied': 'yes'},
    {'grown_ids': [True], 'growth_applied': True}, {'unknown': 2},
])
def test_invalid_config_rejected(change):
    config = SelectiveAffineLayer(2, 3).export_config()
    config.update(change)
    with pytest.raises((ValueError, TypeError)):
        SelectiveAffineLayer.from_config(config)


def test_zero_zero_insertion_has_dead_gradients():
    layer = SelectiveAffineLayer(2, 2).double()
    layer.grow(torch.tensor([1]), torch.zeros(1, 3, dtype=torch.float64))
    layer(torch.ones(3, 2, dtype=torch.float64)).square().sum().backward()
    for module in (layer.gate, layer.contrast):
        for parameter in module.parameters():
            assert torch.count_nonzero(parameter.grad) == 0
