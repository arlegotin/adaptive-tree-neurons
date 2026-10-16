"""One-event CPU insertion from a completed, selected AdamW parent.

No allocation randomness is drawn here. The public operation preserves old
Parameter/state objects, and returns with optimizer and schedule registered.
The narrow migration helper accepts only full identity slices; widening and
other optimizers/backends require their own verified adapters.
"""
from __future__ import annotations

import copy
import json
import math

import torch

from treeneuron.config import canonical_json
from treeneuron.growth.allocate import Allocation
from treeneuron.growth.types import GrowthManifest
from treeneuron.train import (TrainingState, capture_rng, pack_state, restore_rng,
                              validate_metadata, validate_tensor_state, validate_adam_history)

_BASE = ('base.weight', 'base.bias')
_ADDED = ('gate.weight', 'gate.bias', 'contrast.weight', 'contrast.bias')
_OLD = {'layer.base.weight', 'layer.base.bias', 'head.weight', 'head.bias'}


def _validate_optimizer(state, expected_names):
    """Check names, live objects and the shared AdamW/schedule recipe read-only."""
    optimizer, schedule, t = state.optimizer, state.scheduler, state.config.training
    if (type(optimizer) is not torch.optim.AdamW or state.device != 'cpu'
            or t.optimizer != 'adamw' or schedule.optimizer is not optimizer):
        raise ValueError('Growth requires the selected CPU AdamW training recipe')
    if (schedule.position != state.update or len(schedule.base_lrs) != len(optimizer.param_groups)
            or schedule.kind != t.scheduler or schedule.horizon != t.base_updates + t.continuation_updates
            or schedule.min_lr != t.min_learning_rate or not 0 <= schedule.position <= schedule.horizon):
        raise ValueError('Incompatible growth schedule')
    named = dict(state.model.named_parameters())
    expected_options = dict(betas=(t.adam_beta1, t.adam_beta2), eps=t.adam_epsilon,
        weight_decay=t.weight_decay, amsgrad=False, maximize=False, foreach=False,
        capturable=False, differentiable=False, fused=None, decoupled_weight_decay=True)
    seen, parameters = [], []
    for group, base in zip(optimizer.param_groups, schedule.base_lrs):
        names = group.get('param_names')
        if (not isinstance(names, list) or not names or len(names) != len(group['params'])
                or any(type(n) is not str or n not in named or named[n] is not p
                       for n, p in zip(names, group['params']))):
            raise ValueError('Unmapped or incompatible optimizer parameter names')
        options = {k: v for k, v in group.items() if k not in ('params', 'param_names', 'lr')}
        if options != expected_options or base != t.learning_rate:
            raise ValueError('Incompatible optimizer group config')
        factor = 1. if schedule.kind == 'constant' else (1 + math.cos(math.pi*schedule.position/schedule.horizon))/2
        lr = base if schedule.kind == 'constant' else schedule.min_lr + (base-schedule.min_lr)*factor
        if group['lr'] != lr:
            raise ValueError('Optimizer learning rate differs from current schedule')
        seen.extend(names)
        parameters.extend(group['params'])
    if len(seen) != len(set(seen)) or set(seen) != set(expected_names):
        raise ValueError('Unmapped or duplicate optimizer parameter names')
    if any(p not in set(parameters) for p in optimizer.state):
        raise ValueError('Unmapped optimizer state')
    validate_adam_history(state.update, state.model.export_config()['kind'],
        {name: optimizer.state.get(p, {}) for name, p in zip(seen, parameters)})
    for p in parameters:
        if p.dtype != torch.float32 or p.device.type != 'cpu' or not bool(torch.isfinite(p).all()):
            raise ValueError('Growth requires finite CPU float32 parameters')
        saved = optimizer.state.get(p, {})
        if not saved:
            continue  # Validated original update-zero lazy state only.
        if set(saved) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError('Incompatible Adam state keys')
        step = saved['step']
        if (not isinstance(step, torch.Tensor) or step.shape != torch.Size([])
                or step.dtype != torch.float32 or step.device.type != 'cpu'
                or not bool(torch.isfinite(step)) or step.item() < 0 or step.item() != int(step.item())):
            raise ValueError('Invalid Adam step')
        for key in ('exp_avg', 'exp_avg_sq'):
            value = saved[key]
            if (not isinstance(value, torch.Tensor) or value.shape != p.shape or value.dtype != p.dtype
                    or value.device != p.device or not bool(torch.isfinite(value).all())):
                raise ValueError('Incompatible Adam moment')
        if bool((saved['exp_avg_sq'] < 0).any()):
            raise ValueError('Invalid second Adam moment')


def _validate_manifest(state, manifest):
    layer = state.model.layer
    added = _ADDED if layer.grown_ids.numel() else ()
    m, d, h = layer.grown_ids.numel(), layer.d, layer.h
    if (not isinstance(manifest, GrowthManifest) or manifest.schema_version != 1
            or (manifest.d, manifest.h) != (d, h) or manifest.growth_applied is not True
            or not layer.growth_applied or manifest.grown_ids != tuple(layer.grown_ids.tolist())
            or manifest.old_parameter_names != _BASE or manifest.new_parameter_names != added
            or manifest.old_parameter_count != h*(d+1)
            or manifest.added_parameter_count != 2*m*(d+1)
            or manifest.new_parameter_count != (h+2*m)*(d+1)):
        raise ValueError('Invalid growth manifest topology, names or counts')
    named = dict(layer.named_parameters())
    if set(named) != set(_BASE + added) or len(manifest.parameter_mappings) != len(_BASE):
        raise ValueError('Unmapped manifest parameters')
    seen = set()
    for mapping in manifest.parameter_mappings:
        if mapping.old_name not in _BASE or mapping.old_name in seen or mapping.new_name != mapping.old_name:
            raise ValueError('Unmapped or duplicate manifest names')
        p = named[mapping.old_name]
        full = tuple((0, size) for size in p.shape)
        if (mapping.kind != 'identity' or mapping.old_shape != tuple(p.shape)
                or mapping.new_shape != tuple(p.shape) or mapping.old_slice != full or mapping.new_slice != full):
            raise ValueError('Only explicit full identity shape/slice mappings are supported')
        seen.add(mapping.old_name)
    for name in added:
        p = named[name]
        shape = (m, d) if name.endswith('weight') else (m,)
        if p.shape != shape or p.dtype != torch.float32 or p.device.type != 'cpu' or not bool(torch.isfinite(p).all()):
            raise ValueError('Invalid added parameter shape or values')
    if set(dict(state.model.named_parameters())) != _OLD | {'layer.' + n for n in added}:
        raise ValueError('Unsupported model parameter topology')


def migrate_optimizer(state: TrainingState, manifest: GrowthManifest) -> None:
    """Register a freshly grown layer once, preserving every old state object.

    Call immediately after layer.grow. This helper validates the full layer-local
    manifest and preserves the unchanged head explicitly. Repeated calls,
    including empty events, reject. It registers the absolute schedule too.
    insert_growth is the atomic model-plus-optimizer public boundary.
    """
    if 'optimizer_migration' in state.growth_metadata:
        raise ValueError('Optimizer migration already applied')
    _validate_manifest(state, manifest)
    _validate_optimizer(state, _OLD)
    names = ['layer.' + n for n in manifest.new_parameter_names]
    named = dict(state.model.named_parameters())
    # Allocate all new state before mutating groups or schedule.
    fresh = {named[n]: {'step': torch.tensor(0., dtype=torch.float32, device='cpu'),
                       'exp_avg': torch.zeros_like(named[n]), 'exp_avg_sq': torch.zeros_like(named[n])}
             for n in names}
    metadata = copy.deepcopy(state.growth_metadata)
    metadata['optimizer_migration'] = json.loads(canonical_json(manifest.to_dict()))
    if names:
        options = {k: copy.deepcopy(v) for k, v in state.optimizer.param_groups[0].items()
                   if k not in ('params', 'param_names')}
        state.optimizer.add_param_group({'params': [named[n] for n in names], 'param_names': names, **options})
        state.optimizer.state.update(fresh)
        state.scheduler.base_lrs.append(state.scheduler.base_lrs[0])
    state.growth_metadata = metadata


def _allocation_values(state, allocation):
    if not isinstance(allocation, Allocation):
        raise TypeError('Expected Allocation')
    a, layer = allocation, state.model.layer
    if (type(a.requested_count) is not int or type(a.effective_count) is not int
            or not 0 <= a.effective_count <= min(a.requested_count, layer.h)
            or a.policy not in ('adam', 'sgd', 'random', 'keep', 'gaussian', 'greedy')
            or (a.policy == 'keep' and a.effective_count != 0)
            or type(a.seed) is not int or not 0 <= a.seed < 2**63
            or type(a.table_hash) is not str or len(a.table_hash) != 64
            or any(c not in '0123456789abcdef' for c in a.table_hash)):
        raise ValueError('Invalid allocation counts, policy, seed or table hash')
    m = a.effective_count
    if (not isinstance(a.ids, torch.Tensor) or a.ids.dtype != torch.int64
            or a.ids.device.type != 'cpu' or a.ids.shape != (m,)
            or a.ids.tolist() != sorted(set(a.ids.tolist()))
            or (m and (a.ids[0] < 0 or a.ids[-1] >= layer.h))):
        raise ValueError('Allocation IDs must be sorted unique CPU int64 within layer')
    if (not isinstance(a.contrasts, torch.Tensor) or a.contrasts.dtype != torch.float64
            or a.contrasts.device.type != 'cpu' or a.contrasts.shape != (m, layer.d+1)
            or not bool(torch.isfinite(a.contrasts).all())):
        raise ValueError('Allocation contrasts must be finite CPU float64 [m,d+1]')
    values = a.contrasts.detach().to(device=layer.base.weight.device, dtype=layer.base.weight.dtype)
    if not bool(torch.isfinite(values).all()) or (m and bool((values.abs().amax(dim=1) == 0).any())):
        raise ValueError('Inserted contrasts must remain finite and nonzero after casting')
    if (not isinstance(a.direction_attempts, torch.Tensor) or a.direction_attempts.dtype != torch.int64
            or a.direction_attempts.device.type != 'cpu' or a.direction_attempts.shape != (m,)
            or (a.policy == 'gaussian' and bool(((a.direction_attempts < 1) | (a.direction_attempts > 100)).any()))
            or (a.policy != 'gaussian' and bool((a.direction_attempts != 0).any()))):
        raise ValueError('Invalid allocation direction attempts')
    if (not isinstance(a.cost, dict) or set(a.cost) != {'shared_scored_units', 'standalone_contrast_units_required'}
            or any(type(v) is not int or v < 0 for v in a.cost.values())
            or a.cost['shared_scored_units'] != layer.h
            or a.cost['standalone_contrast_units_required'] != (m if a.policy == 'random' else 0)):
        raise ValueError('Invalid allocation work counts')
    record = dict(requested_count=a.requested_count, effective_count=m, ids=a.ids.tolist(),
        contrasts=a.contrasts.tolist(), policy=a.policy, seed=a.seed, table_hash=a.table_hash,
        direction_attempts=a.direction_attempts.tolist(), cost=copy.deepcopy(a.cost))
    canonical_json(record)
    return values, record


def insert_growth(state: TrainingState, allocation: Allocation) -> GrowthManifest:
    """Atomically add one allocated event to a canonical G-selected CPU parent.

    fit_parent's return or its completed best checkpoint is accepted. An active
    stage or a completed last checkpoint still carrying embedded G-best rejects;
    load the completed best checkpoint first. Old state, identities, training and
    process RNGs, sampler and stage stay unchanged. No checkpoint IO is performed.
    """
    layer = state.model.layer
    if layer.growth_applied:
        raise ValueError('Only one growth insertion is allowed')
    if not state.stage.get('complete') or state.best_snapshot is not None:
        raise ValueError('Growth requires a completed selected parent; load completed best checkpoint')
    if state.growth_metadata:
        raise ValueError('Parent already carries growth metadata')
    _validate_optimizer(state, _OLD)
    snapshot = pack_state(state)
    validate_metadata(snapshot['metadata'], state.identity_hashes, state.device)
    validate_tensor_state(snapshot['tensors'])
    values, allocation_record = _allocation_values(state, allocation)
    rng = capture_rng()
    groups = list(state.optimizer.param_groups)
    old_state = dict(state.optimizer.state)
    base_lrs = list(state.scheduler.base_lrs)
    metadata = state.growth_metadata
    original = (layer.gate, layer.contrast, layer.grown_ids, layer.growth_applied)
    try:
        manifest = layer.grow(allocation.ids, values)
        migrate_optimizer(state, manifest)
        state.growth_metadata.update(schema_version=1, manifest=json.loads(canonical_json(manifest.to_dict())), allocation=allocation_record,
            inserted_contrasts=values.cpu().tolist(), topology=state.model.export_config(),
            insertion_update=state.update, scheduler=state.scheduler.state_dict())
        # Ensure the public result can be serialized under existing interfaces.
        pack_state(state)
    except BaseException:
        layer.gate, layer.contrast, layer.grown_ids, layer.growth_applied = original
        state.optimizer.param_groups[:] = groups
        state.optimizer.state.clear()
        state.optimizer.state.update(old_state)
        state.scheduler.base_lrs[:] = base_lrs
        state.growth_metadata = metadata
        raise
    finally:
        restore_rng(rng)
    return manifest
