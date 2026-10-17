"""Serial F-only float32 training, G selection, and complete-update recovery.

``continue_fit`` resumes an unfinished stage, or begins a fresh continuation
from a completed selected state. A policy's max_updates is a stage allowance,
not a new allowance on retry. Last checkpoints carry the stage's complete best
snapshot. Returned states carry every executed diagnostic but the selected
model/optimizer/scheduler/sampler/RNG state. No V observations are accessed.

Costs measure the _run attempt region, not an end-to-end fit total. Parent
model/optimizer construction, upfront metadata, backend evidence and imports
precede that region. F transform fitting is separately measured; validation is
a nested subcost. The study runner owns enclosing end-to-end fit/study totals.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .artifacts import RunPaths, append_event, finish_run, update_run_observations
from .backend import ROOT, lock_hash, runtime_hash, select_backend, validate_backend_report
from .cpu_envelope import supported_batch, requires_envelope, validate_envelope_reference
from .config import ExperimentConfig, canonical_hash, canonical_json, resolve_experiment
from .data.transforms import FittedTransform
from .data.types import DevelopmentData, ObservationSet, array_hash
from .evaluation.metrics import regression_metrics
from .models.units import SelectiveAffineLayer

BACKEND_REPORT_PATH = ROOT / 'environment' / 'report.json'


class SelectiveRegressor(nn.Module):
    """One reconstructible selective affine layer, ReLU and scalar head."""

    def __init__(self, d: int, h: int, tau: float = 1.):
        super().__init__()
        self.layer = SelectiveAffineLayer(d, h, tau)
        self.activation = nn.ReLU()
        self.head = nn.Linear(h, 1)

    def forward(self, x):
        return self.head(self.activation(self.layer(x))).squeeze(-1)

    def export_config(self):
        return {'schema_version': 1, 'kind': 'selective_relu_scalar', 'layer': self.layer.export_config()}

    @classmethod
    def from_config(cls, config):
        if set(config) != {'schema_version', 'kind', 'layer'} or config['schema_version'] != 1 or config['kind'] != 'selective_relu_scalar':
            raise ValueError('Invalid scalar model topology')
        layer = SelectiveAffineLayer.from_config(config['layer'])
        model = cls(layer.d, layer.h, layer.tau)
        model.layer = layer
        return model


@dataclass(frozen=True)
class TrainingPolicy:
    logical_batch_size: int
    physical_batch_size: int
    max_updates: int
    validation_interval: int
    patience: int
    checkpoint_selection: str = 'strict_improvement_earliest'
    evaluate_at_update_zero: bool = True
    checkpoint_interval: int = 1

    def __post_init__(self):
        for name in ('logical_batch_size', 'physical_batch_size', 'max_updates', 'validation_interval', 'patience', 'checkpoint_interval'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if self.physical_batch_size != self.logical_batch_size:
            raise ValueError('physical batch must equal logical batch; accumulation is not implemented')
        if not supported_batch(self.logical_batch_size):
            raise ValueError('Batch outside bounded backend evidence; extended CPU batches require a measured envelope')
        if self.checkpoint_selection != 'strict_improvement_earliest' or self.evaluate_at_update_zero is not True:
            raise ValueError('Require strict improvement, earliest ties, and update-zero evaluation')

    @classmethod
    def from_config(cls, config: ExperimentConfig, *, continuation: bool = False):
        t = config.training
        for name in ('base_updates', 'continuation_updates', 'validation_interval', 'patience'):
            if getattr(t, name) is None:
                raise ValueError(f'training.{name} must be supplied before training')
        return cls(t.batch_size, t.batch_size, t.continuation_updates if continuation else t.base_updates,
                   t.validation_interval, t.patience, t.checkpoint_selection, t.evaluate_at_update_zero,
                   t.checkpoint_interval)


class UpdateSchedule:
    """Absolute optimizer-update schedule; growth never restarts its horizon."""

    def __init__(self, optimizer, *, kind: str, horizon: int, min_lr: float,
                 base_lrs: list[float], position: int = 0):
        if (kind not in ('constant', 'cosine') or type(horizon) is not int or horizon <= 0
                or type(position) is not int or not 0 <= position <= horizon
                or len(base_lrs) != len(optimizer.param_groups)
                or not base_lrs or any(not math.isfinite(lr) or lr <= 0 for lr in base_lrs)
                or not math.isfinite(min_lr) or not 0 <= min_lr <= min(base_lrs)):
            raise ValueError('Invalid update schedule')
        self.optimizer, self.kind, self.horizon = optimizer, kind, horizon
        self.min_lr, self.base_lrs, self.position = min_lr, list(base_lrs), position
        self._apply()

    def _apply(self):
        factor = 1. if self.kind == 'constant' else (1 + math.cos(math.pi*self.position/self.horizon))/2
        for group, base in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = base if self.kind == 'constant' else self.min_lr + (base-self.min_lr)*factor

    def step(self):
        if self.position >= self.horizon:
            raise ValueError('Schedule horizon exhausted')
        self.position += 1
        self._apply()

    def state_dict(self):
        return dict(kind=self.kind, horizon=self.horizon, min_lr=self.min_lr,
                    base_lrs=list(self.base_lrs), position=self.position)


def capture_rng() -> dict[str, Any]:
    numpy = np.random.get_state()
    mps = {'state': None, 'reason': 'MPS RNG unavailable on this runtime'}
    if torch.backends.mps.is_available():
        mps = {'state': torch.mps.get_rng_state().clone(), 'reason': None}
    return {'python': random.getstate(),
            'numpy': {'name': numpy[0], 'keys': torch.from_numpy(numpy[1].astype(np.int64)),
                      'position': numpy[2], 'has_gauss': numpy[3], 'cached_gaussian': numpy[4]},
            'torch_cpu': torch.get_rng_state().clone(), 'torch_mps': mps}


def restore_rng(states: Mapping[str, Any]) -> None:
    random.setstate(states['python'])
    s = states['numpy']
    np.random.set_state((s['name'], s['keys'].cpu().numpy().astype(np.uint32), s['position'],
                         s['has_gauss'], s['cached_gaussian']))
    torch.set_rng_state(states['torch_cpu'].cpu())
    if states['torch_mps']['state'] is not None:
        if not torch.backends.mps.is_available():
            # CPU checkpoints made on a machine with MPS retain that RNG even if
            # sandbox access disappears; CPU training consumes no MPS RNG.
            return
        torch.mps.set_rng_state(states['torch_mps']['state'].cpu())


def source_hash() -> str:
    """Execution identity covers all package source, independently of Git HEAD."""
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _identities(data, config, preprocessing):
    return {'code': source_hash(), 'lock': lock_hash(), 'config': canonical_hash(config.to_dict()),
            'split': data.manifest.assignment_hash, 'raw': data.manifest.dataset_id,
            'transform': canonical_hash(preprocessing.to_state()), 'runtime': runtime_hash()}


def _fg_hashes(data):
    return {f'{split}.{name}': array_hash(getattr(getattr(data, split), name))
            for split in ('F', 'G') for name in ('X', 'y', 'row_id', 'group_id')}


@dataclass
class TrainingState:
    model: nn.Module
    optimizer: torch.optim.AdamW
    scheduler: UpdateSchedule
    update: int
    epoch: int
    order: torch.Tensor
    cursor: int
    sampler_generator: torch.Generator
    rng_states: dict[str, Any]
    preprocessing: FittedTransform
    identity_hashes: dict[str, str]
    config: ExperimentConfig
    backend: dict[str, Any]
    fit_target_variance: float
    data_hashes: dict[str, str]
    stage: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    growth_metadata: dict[str, Any] = field(default_factory=dict)
    best_snapshot: dict[str, Any] | None = None

    @property
    def device(self):
        return str(next(self.model.parameters()).device)

    def predict(self, X, *, batch_size: int | None = None) -> np.ndarray:
        batch = self.config.training.batch_size if batch_size is None else batch_size
        if not supported_batch(batch) or (batch>128 and 'cpu_envelope' not in self.backend):
            raise ValueError('Prediction batch requires bounded source-bound CPU evidence')
        # The finite CPU extension certifies actual tensors through max F9224.
        # Prediction chunks obey that same bound, including much larger E/T sets.
        if 'cpu_envelope' in self.backend:
            batch=min(batch,self.config.data.n_fit)
        transformed = self.preprocessing.transform_X(X)
        values = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(transformed), batch):
                x = torch.tensor(transformed[i:i+batch], dtype=torch.float32, device=self.device)
                values.append(self.model(x).cpu().numpy().astype(np.float64))
        return self.preprocessing.inverse_y(np.concatenate(values) if values else np.empty(0))

    def next_row_ids(self, rows: ObservationSet) -> np.ndarray:
        """Peek without advancing local or process RNGs, including epoch rollover."""
        if len(rows.row_id) != self.order.numel():
            raise ValueError('Sampler row count mismatch')
        order, cursor = self.order, self.cursor
        if cursor == len(order):
            generator = torch.Generator().set_state(self.sampler_generator.get_state())
            order, cursor = torch.randperm(len(order), generator=generator), 0
        batch = self.stage.get('policy', {}).get('logical_batch_size', self.config.training.batch_size)
        return rows.row_id[order[cursor:cursor+batch].numpy()].copy()


def _optimizer(model, config):
    t = config.training
    if t.optimizer != 'adamw':
        raise ValueError('sgd training is not implemented; use explicit AdamW configuration')
    names, parameters = zip(*model.named_parameters())
    return torch.optim.AdamW([{'params': list(parameters), 'param_names': list(names)}],
        lr=t.learning_rate, betas=(t.adam_beta1, t.adam_beta2), eps=t.adam_epsilon,
        weight_decay=t.weight_decay, foreach=False)


def _backend_evidence(config, data, *, causal_model=None):
    from .locators import locate
    path = locate(BACKEND_REPORT_PATH,producer=True)
    document = json.loads(path.read_text())
    report = document['backend_probes']
    for probe in report['probes'].values():
        if probe.get('status') == 'passed': validate_backend_report(probe)
    decision = select_backend(report['probes'], comparison_blocks=report['comparison_blocks'])
    if decision != report['decision']:
        raise ValueError('Backend decision differs from validated numerical evidence')
    b = config.backend
    if b.device != decision['selected_backend']:
        raise ValueError('Requested backend differs from validated selected backend recipe')
    if b.cpu_threads not in (None, 1):
        raise ValueError('Backend evidence requires one CPU thread')
    if b.interop_threads not in (None, torch.get_num_interop_threads()):
        raise ValueError('Requested interop threads differ from live runtime; configure before training')
    interop = report['probes'][b.device]['environment'].get('interop_threads')
    if interop is not None and torch.get_num_interop_threads() != interop:
        raise ValueError('Live interop threads differ from backend evidence')
    envelope = None
    if requires_envelope(config):
        envelope = dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        validate_envelope_reference(envelope,config,causal_model.export_config() if causal_model is not None else None)
    if (config.model.d > 16 or config.model.h > (95 if causal_model is not None else 64)
            or (envelope is None and (len(data.F.y)>2306 or config.training.batch_size>128))):
        raise ValueError('Workload exceeds bounded backend evidence; obtain a relevant new probe')
    if b.device == 'mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'):
        raise ValueError('MPS requires availability and fallback disabled before process import')
    workload = {'d': config.model.d, 'h': config.model.h, 'n_fit': len(data.F.y),
                'logical_batch_size': config.training.batch_size, 'physical_batch_size': config.training.batch_size}
    calibrated = workload == {'d': 16, 'h': 64, 'n_fit': 2306, 'logical_batch_size': 128, 'physical_batch_size': 128}
    if not calibrated and b.device != 'cpu':
        raise ValueError('Small diagnostic workload is supported only on CPU; remeasure other workloads')
    if any(decision['selected_batches'][key] != 128 for key in ('base_epoch', 'grown_epoch')):
        raise ValueError('Selected physical recipe requires unimplemented accumulation')
    evidence = {'device': b.device, 'dtype': 'float32', 'execution': 'eager', 'fallback': False,
            'cpu_threads': 1, 'interop_threads': torch.get_num_interop_threads(),
            'kernel_tolerance': decision['tolerance'], 'resume_tolerance': {'rtol': 0., 'atol': 0.} if b.device == 'cpu' else decision['tolerance'],
            'evidence_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'kernel_implementation_sha256': report['probes'][b.device]['implementation_sha256'],
            'workload': workload, 'performance_calibration_matches': calibrated,
            'performance_scope': decision['scope'] if calibrated else 'CPU correctness diagnostic; kernel compatibility only; throughput not calibrated'}
    if envelope is not None:
        evidence['cpu_envelope']=envelope
    if causal_model is not None:
        from .baselines.causal_evidence import attach_causal_evidence
        evidence = attach_causal_evidence(evidence, causal_model, config.training.batch_size)
    return evidence


def _validate_inputs(data, config, paths):
    config = resolve_experiment(config.to_dict())
    if config.model.depth != 1:
        raise ValueError('Only depth=1 is supported')
    TrainingPolicy.from_config(config)
    if config.training.optimizer != 'adamw':
        raise ValueError('sgd training is not implemented')
    if config.training.scheduler == 'constant' and config.training.min_learning_rate != 0:
        raise ValueError('min_learning_rate is only supported with cosine scheduling')
    if data.manifest.phase != 'exploratory':
        raise ValueError('Training requires exploratory development data')
    if (data.F.X.shape[1] != config.model.d or data.G.X.shape[1] != config.model.d
            or len(data.F.y) != config.data.n_fit or len(data.G.y) != config.data.n_growth):
        raise ValueError('F/G dimensions and counts must match resolved configuration')
    stored = json.loads((paths.directory / 'config.resolved.json').read_text())
    if stored != config.to_dict():
        raise ValueError('Run config identity mismatch')
    if config.seeds.optimizer >= 2**32:
        raise ValueError('Optimizer seed must be within NumPy legacy RNG range [0, 2**32)')
    return config


def _new_stage(state, policy, kind):
    if state.update + policy.max_updates > state.scheduler.horizon:
        raise ValueError('Stage allowance exceeds original base+continuation schedule horizon')
    state.stage = {'kind': kind, 'start_update': state.update, 'updates_completed': 0,
                   'bad_evaluations': 0, 'best_mse': None, 'policy': asdict(policy), 'complete': False}
    state.best_snapshot = None


def _tensor_copy(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: _tensor_copy(v) for k, v in value.items()}
    if isinstance(value, list): return [_tensor_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(_tensor_copy(v) for v in value)
    return copy.deepcopy(value)


def validate_adam_history(update, model_kind, records):
    """Require complete history without equating per-parameter Adam ages.

    Original ordinary/static parameters can be lazy only at selected update
    zero. Selective gates/contrasts and widening blocks are explicitly created
    with complete step-zero state, even when inserted into an update-zero parent.
    A fresh added record may therefore have age zero at a positive global update.
    """
    added_prefixes = {
        'selective_relu_scalar': ('layer.gate.', 'layer.contrast.'),
        'widened_relu_scalar': ('layer.added.', 'head.added.'),
        'static_gated_relu_scalar': (),
    }[model_kind]
    for name, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f'Invalid Adam history for {name}')
        if not record:
            if update > 0 or name.startswith(added_prefixes):
                raise ValueError(f'Missing required Adam history for {name}')
        elif set(record) != {'step', 'exp_avg', 'exp_avg_sq'}:
            raise ValueError(f'Incomplete Adam history for {name}')


def _validate_live_adam_history(state):
    validate_adam_history(state.update, state.model.export_config()['kind'],
        {name: state.optimizer.state.get(parameter, {})
         for name, parameter in state.model.named_parameters()})


def validate_saved_adam_history(meta, tensors):
    """Check named serialized history before construction, loading or writing."""
    saved = tensors['optimizer']
    if (not isinstance(saved, Mapping) or not isinstance(saved.get('param_groups'), list)
            or not isinstance(saved.get('state'), Mapping)
            or len(saved['param_groups']) != len(meta['optimizer_groups'])):
        raise ValueError('Invalid saved Adam history groups')
    records, identifiers = {}, set()
    for source, expected in zip(saved['param_groups'], meta['optimizer_groups']):
        if (not isinstance(source, Mapping) or source.get('param_names') != expected['names']
                or not isinstance(source.get('params'), list)
                or len(source['params']) != len(expected['names'])):
            raise ValueError('Saved Adam history parameter names/count mismatch')
        for name, identifier in zip(expected['names'], source['params']):
            if type(identifier) is not int or identifier < 0 or identifier in identifiers:
                raise ValueError('Invalid saved Adam history parameter identifiers')
            identifiers.add(identifier)
            records[name] = saved['state'].get(identifier, {})
    if set(saved['state']) - identifiers:
        raise ValueError('Unmapped saved Adam history')
    validate_adam_history(meta['update'], meta['model']['kind'], records)


def pack_state(state: TrainingState) -> dict[str, Any]:
    """Internal complete snapshot; no nested best state or unbounded curves."""
    _validate_live_adam_history(state)
    if 'causal_cpu_evidence' in state.backend:
        from .growth.insert import _validate_optimizer
        _validate_optimizer(state, set(dict(state.model.named_parameters())))
    optimizer = state.optimizer.state_dict()
    groups = []
    named = dict(state.model.named_parameters())
    for live, saved in zip(state.optimizer.param_groups, optimizer['param_groups']):
        names = live.get('param_names')
        if (not isinstance(names, list) or len(names) != len(live['params'])
                or any(name not in named or named[name] is not p for name, p in zip(names, live['params']))):
            raise ValueError('Optimizer requires explicit correct parameter names per group')
        groups.append({'names': names, 'options': {k: v for k, v in saved.items() if k not in ('params', 'param_names')}})
    all_names = [name for group in groups for name in group['names']]
    if len(all_names) != len(set(all_names)) or set(all_names) != set(named):
        raise ValueError('Optimizer groups must cover each named model parameter exactly once')
    metadata = {'schema_version': 1, 'config': state.config.to_dict(), 'model': state.model.export_config(),
                'identity_hashes': state.identity_hashes, 'device': state.device, 'backend': state.backend,
                'update': state.update, 'epoch': state.epoch, 'cursor': state.cursor,
                'n_fit': state.order.numel(), 'optimizer_kind': 'adamw', 'optimizer_groups': groups,
                'scheduler': state.scheduler.state_dict(), 'preprocessing': state.preprocessing.to_state(),
                'fit_target_variance': state.fit_target_variance, 'data_hashes': state.data_hashes,
                'stage': state.stage, 'growth_metadata': state.growth_metadata}
    canonical_json(metadata)
    return {'metadata': copy.deepcopy(metadata), 'tensors': _tensor_copy({
        'model': state.model.state_dict(), 'optimizer': optimizer,
        'order': state.order, 'sampler_generator': state.sampler_generator.get_state(), 'rng_states': state.rng_states})}


def validate_metadata(meta, expected_hashes, device):
    """Validate finite safe metadata and identities before tensor/optimizer load."""
    canonical_json(meta)
    required = {'schema_version', 'config', 'model', 'identity_hashes', 'device', 'backend', 'update', 'epoch',
                'cursor', 'n_fit', 'optimizer_kind', 'optimizer_groups', 'scheduler', 'preprocessing',
                'fit_target_variance', 'data_hashes', 'stage', 'growth_metadata'}
    if set(meta) != required or meta['schema_version'] != 1:
        raise ValueError('Invalid training metadata schema')
    identities = meta['identity_hashes']
    keys = {'code', 'lock', 'config', 'split', 'raw', 'transform', 'runtime'}
    if (set(identities) != keys or set(expected_hashes) != keys or identities != dict(expected_hashes)
            or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in identities.values())):
        raise ValueError('Checkpoint identity mismatch or incomplete expected identities')
    if identities['code'] != source_hash() or identities['lock'] != lock_hash() or identities['runtime'] != runtime_hash():
        raise ValueError('Checkpoint code/lock/runtime identity differs from live execution')
    config = resolve_experiment(meta['config'])
    TrainingPolicy.from_config(config)
    topology = meta['model']
    if not isinstance(topology, dict) or set(topology) != {'schema_version','kind','layer'} or not isinstance(topology['layer'], dict):
        raise ValueError('Invalid model topology')
    if requires_envelope(config):
        validate_envelope_reference(meta['backend'].get('cpu_envelope',{}),config,topology)
    if config.model.coordinates == 'widened':
        from .models.widened import WidenedRegressor
        from .baselines.causal_evidence import validate_causal_evidence
        validate_causal_evidence(meta['backend'], topology, config)
        before = capture_rng()
        try:
            candidate = WidenedRegressor.from_config(topology)
            if topology['layer'] != dict(d=config.model.d,h=config.model.h,parent_width=config.model.parent_width,tau=config.model.tau):
                raise ValueError('Widening topology differs from resolved config')
            expected_names = set(dict(candidate.named_parameters()))
        finally:
            restore_rng(before)
    elif config.model.coordinates != 'selective':
        from .models.static import StaticRegressor
        from .baselines.causal_evidence import validate_causal_evidence
        validate_causal_evidence(meta['backend'], topology, config)
        before = capture_rng()
        try:
            candidate = StaticRegressor.from_config(topology)
            if candidate.layer.export_config() != dict(d=config.model.d,h=config.model.h,
                    count=config.model.static_count,coordinates=config.model.coordinates,tau=config.model.tau):
                raise ValueError('Static topology differs from resolved config')
            expected_names = set(dict(candidate.named_parameters()))
        finally:
            restore_rng(before)
    else:
        if (not isinstance(topology, dict) or set(topology) != {'schema_version', 'kind', 'layer'}
                or type(topology['schema_version']) is not int or topology['schema_version'] != 1
                or topology['kind'] != 'selective_relu_scalar'):
            raise ValueError('Invalid model topology')
        layer = topology['layer']
        if (not isinstance(layer, dict) or set(layer) != {'schema_version', 'd', 'h', 'tau', 'growth_applied', 'grown_ids'}
                or type(layer['schema_version']) is not int or layer['schema_version'] != 1
                or type(layer['d']) is not int or type(layer['h']) is not int
                or (layer['d'], layer['h'], layer['tau']) != (config.model.d, config.model.h, config.model.tau)
                or not 0 < layer['d'] <= 16 or not 0 < layer['h'] <= (95 if 'causal_cpu_evidence' in meta['backend'] else 64)
                or type(layer['growth_applied']) is not bool or type(layer['grown_ids']) is not list
                or any(type(i) is not int or not 0 <= i < layer['h'] for i in layer['grown_ids'])
                or layer['grown_ids'] != sorted(set(layer['grown_ids']))
                or (layer['grown_ids'] and not layer['growth_applied'])):
            raise ValueError('Model topology differs from bounded config or is malformed')
        expected_names = {'layer.base.weight', 'layer.base.bias', 'head.weight', 'head.bias'}
        if layer['grown_ids']:
            expected_names.update({'layer.gate.weight', 'layer.gate.bias', 'layer.contrast.weight', 'layer.contrast.bias'})
    if 'causal_cpu_evidence' in meta['backend']:
        from .baselines.causal_evidence import validate_causal_evidence
        validate_causal_evidence(meta['backend'], topology, config)
    if canonical_hash(config.to_dict()) != identities['config']:
        raise ValueError('Checkpoint config identity mismatch')
    if device not in ('cpu', 'mps') or meta['device'] != device or meta['backend']['device'] != device:
        raise ValueError('Same-backend resume required')
    if device == 'mps' and (not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0'):
        raise ValueError('MPS resume requires available backend with fallback disabled')
    if (meta['optimizer_kind'] != 'adamw' or config.training.optimizer != 'adamw'
            or config.model.depth != 1):
        raise ValueError('Unsupported optimizer or depth')
    for key in ('update', 'epoch', 'cursor', 'n_fit'):
        if type(meta[key]) is not int or meta[key] < 0:
            raise ValueError('Invalid update/sampler metadata')
    if meta['n_fit'] != config.data.n_fit or not 0 <= meta['cursor'] <= meta['n_fit']:
        raise ValueError('Invalid sampler cursor/count')
    transform = FittedTransform.from_state(meta['preprocessing'])
    if canonical_hash(transform.to_state()) != identities['transform']:
        raise ValueError('Transform identity mismatch')
    if type(meta['fit_target_variance']) not in (int, float) or meta['fit_target_variance'] < 0:
        raise ValueError('Invalid F target variance')
    stage = meta['stage']
    if (set(stage) != {'kind', 'start_update', 'updates_completed', 'bad_evaluations', 'best_mse', 'policy', 'complete'}
            or stage['kind'] not in ('base', 'continuation') or type(stage['complete']) is not bool):
        raise ValueError('Invalid stage metadata')
    policy = TrainingPolicy(**stage['policy'])
    if policy.logical_batch_size != config.training.batch_size:
        raise ValueError('Checkpoint batch differs from resolved config')
    if policy.checkpoint_interval != config.training.checkpoint_interval:
        raise ValueError('Checkpoint cadence differs from resolved config')
    for key in ('start_update', 'updates_completed', 'bad_evaluations'):
        if type(stage[key]) is not int or stage[key] < 0: raise ValueError('Invalid stage counters')
    if stage['updates_completed'] > policy.max_updates or (not stage['complete'] and meta['update'] != stage['start_update'] + stage['updates_completed']):
        raise ValueError('Stage update/counter mismatch')
    if stage['best_mse'] is not None and (type(stage['best_mse']) not in (int, float) or stage['best_mse'] < 0):
        raise ValueError('Invalid best risk')
    schedule = meta['scheduler']
    if (set(schedule) != {'kind', 'horizon', 'min_lr', 'base_lrs', 'position'}
            or schedule['position'] != meta['update']
            or schedule['kind'] != config.training.scheduler
            or schedule['horizon'] != config.training.base_updates + config.training.continuation_updates):
        raise ValueError('Invalid schedule metadata')
    groups = meta['optimizer_groups']
    options = {'lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused', 'decoupled_weight_decay'}
    if not isinstance(groups, list) or not groups:
        raise ValueError('Invalid named optimizer groups')
    for group in groups:
        if (set(group) != {'names', 'options'} or not isinstance(group['names'], list) or not group['names']
                or any(type(n) is not str for n in group['names']) or set(group['options']) - options
                or group['options'].get('foreach') is not False):
            raise ValueError('Invalid named optimizer group metadata')
    names = [name for group in groups for name in group['names']]
    if len(names) != len(set(names)) or set(names) != expected_names:
        raise ValueError('Named optimizer groups do not match model topology')
    if (meta['backend'].get('dtype') != 'float32' or meta['backend'].get('fallback') is not False
            or meta['backend'].get('cpu_threads') != 1):
        raise ValueError('Invalid checkpoint backend recipe')
    expected_data_keys = {f'{split}.{field}' for split in ('F', 'G') for field in ('X', 'y', 'row_id', 'group_id')}
    if set(meta['data_hashes']) != expected_data_keys:
        raise ValueError('Incomplete F/G identity metadata')
    return config, transform


def validate_tensor_state(tensors):
    """Reject nonfinite saved tensors/scalars without allocating a model or RNG.

    This is shared by publication, deserialization and in-memory reconstruction.
    Walk the whole collection so model values, Adam moments/steps and any stored
    RNG scalar state obey the same finite-state boundary, including embedded best.
    """
    if not isinstance(tensors, Mapping) or set(tensors) != {
            'model', 'optimizer', 'order', 'sampler_generator', 'rng_states'}:
        raise ValueError('Invalid checkpoint tensor state schema')

    def finite(value, path):
        if isinstance(value, torch.Tensor):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f'Nonfinite checkpoint tensor: {path}')
        elif isinstance(value, Mapping):
            for key, item in value.items(): finite(item, f'{path}.{key}')
        elif isinstance(value, (tuple, list)):
            for index, item in enumerate(value): finite(item, f'{path}[{index}]')
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f'Nonfinite checkpoint scalar: {path}')

    finite(tensors, 'state')


def unpack_state(snapshot, *, expected_hashes, device):
    meta, tensors = snapshot['metadata'], snapshot['tensors']
    config, preprocessing = validate_metadata(meta, expected_hashes, device)
    validate_tensor_state(tensors)
    validate_saved_adam_history(meta, tensors)
    # Rebuilding Linear modules consumes RNG; never let it alter paired streams.
    before = capture_rng()
    try:
        if config.model.coordinates == 'selective':
            model = SelectiveRegressor.from_config(meta['model'])
        elif config.model.coordinates == 'widened':
            from .models.widened import WidenedRegressor
            model = WidenedRegressor.from_config(meta['model'])
        else:
            from .models.static import StaticRegressor
            model = StaticRegressor.from_config(meta['model'])
        model = model.to(device=device, dtype=torch.float32)
        if 'causal_cpu_evidence' in meta['backend']:
            expected_tensors = model.state_dict()
            if set(tensors['model']) != set(expected_tensors) or any(
                    not isinstance(tensors['model'][n], torch.Tensor)
                    or tensors['model'][n].shape != p.shape or tensors['model'][n].dtype != p.dtype
                    or tensors['model'][n].device.type != device for n,p in expected_tensors.items()):
                raise ValueError('Causal model tensor shape/dtype/device mismatch')
        model.load_state_dict(tensors['model'], strict=True)
        named = dict(model.named_parameters())
        names = [name for group in meta['optimizer_groups'] for name in group['names']]
        if len(set(names)) != len(names) or set(names) != set(named):
            raise ValueError('Optimizer names do not match model topology')
        groups = [{'params': [named[n] for n in g['names']], 'param_names': list(g['names']), **g['options']}
                  for g in meta['optimizer_groups']]
        optimizer = torch.optim.AdamW(groups, foreach=False)
        saved = tensors['optimizer']
        if len(saved['param_groups']) != len(groups): raise ValueError('Optimizer group mismatch')
        for source, expected in zip(saved['param_groups'], meta['optimizer_groups']):
            if source.get('param_names') != expected['names']:
                raise ValueError('Saved optimizer parameter names mismatch')
            actual = {k: v for k, v in source.items() if k not in ('params', 'param_names')}
            if canonical_json(actual) != canonical_json(expected['options']):
                raise ValueError('Saved optimizer options mismatch')
        if 'causal_cpu_evidence' in meta['backend']:
            # Adam.load_state_dict casts moments to the parameter dtype. Check
            # the saved values first so malformed float64 state cannot pass by
            # being silently converted to the required float32 recipe.
            for group in saved['param_groups']:
                if len(group['params']) != len(group['param_names']):
                    raise ValueError('Saved Adam parameter count mismatch')
                for name, identifier in zip(group['param_names'], group['params']):
                    record = saved['state'].get(identifier, {})
                    if not record:
                        continue
                    if not isinstance(record, Mapping) or set(record) != {'step','exp_avg','exp_avg_sq'}:
                        raise ValueError('Invalid saved causal Adam state')
                    for key, tensor in record.items():
                        shape = torch.Size([]) if key == 'step' else named[name].shape
                        if (not isinstance(tensor, torch.Tensor) or tensor.shape != shape
                                or tensor.dtype != torch.float32 or tensor.device.type != 'cpu'):
                            raise ValueError('Saved causal Adam shape/dtype/device mismatch')
        # CPU optimizer loading can retain tensor storage from its input. Keep
        # the live moments/step independent of the durable embedded best, even
        # when serialization deduplicated a best checkpoint's two snapshots.
        optimizer.load_state_dict(_tensor_copy(saved))
        scheduler = UpdateSchedule(optimizer, **meta['scheduler'])
        order = tensors['order']
        if (order.dtype != torch.int64 or order.shape != (meta['n_fit'],)
                or not torch.equal(order.sort().values, torch.arange(meta['n_fit']))):
            raise ValueError('Invalid saved sampler permutation')
        generator = torch.Generator().set_state(tensors['sampler_generator'])
        state = TrainingState(model, optimizer, scheduler, meta['update'], meta['epoch'], order.clone(),
            meta['cursor'], generator, _tensor_copy(tensors['rng_states']), preprocessing,
            dict(meta['identity_hashes']), config, copy.deepcopy(meta['backend']), meta['fit_target_variance'],
            copy.deepcopy(meta['data_hashes']), copy.deepcopy(meta['stage']), growth_metadata=copy.deepcopy(meta['growth_metadata']))
        if 'causal_cpu_evidence' in meta['backend']:
            from .growth.insert import _validate_optimizer
            _validate_optimizer(state, set(named))
        model.eval()
    except BaseException:
        restore_rng(before)
        raise
    restore_rng(state.rng_states)
    return state


def _observe_initial(state, data, paths, preprocess_ns=None):
    model = state.model
    updates = {
        'identity': {'code_revision': state.identity_hashes['code'], 'lock_hash': state.identity_hashes['lock']},
        'data': {'raw_hash': state.identity_hashes['raw'], 'split_hash': state.identity_hashes['split'],
                 'transform_hash': state.identity_hashes['transform'], 'seeds': state.config.seeds.__dict__,
                 'counts': {s: {'rows': len(getattr(data, s).y), 'groups': len(np.unique(getattr(data, s).group_id))}
                            for s in ('F', 'G')}},
        'method': {'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                   'total_parameters': sum(p.numel() for p in model.parameters()),
                   'backend': {'actual': state.backend},
                   'threads': {'cpu': 1, 'interop': torch.get_num_interop_threads()},
                   'weight_terms': {'layer': sum(p.numel() for p in model.layer.parameters()),
                                    'head': sum(p.numel() for p in model.head.parameters())},
                   'context_bytes': 0, 'cache_bytes': 0},
        'resources': {'timing_exclusions': []}}
    if preprocess_ns is not None:
        updates['costs'] = {'preprocess': {'nanoseconds': preprocess_ns, 'status': 'measured', 'reason': None,
            'boundary': 'F-only transform fitting; feature/target conversion is included in base or continuation; repeated G transformation is included in validation'}}
    update_run_observations(paths, updates)


def fit_parent(data: DevelopmentData, config: ExperimentConfig, paths: RunPaths, *,
               growth_metadata: dict | None = None) -> TrainingState:
    """Train on F; return earliest strict G-best complete state, leaving run open."""
    try:
        config = _validate_inputs(data, config, paths)
        policy = TrainingPolicy.from_config(config)
        start = time.perf_counter_ns()
        preprocessing = FittedTransform.fit(data.F.X, data.F.y)
        preprocess_ns = time.perf_counter_ns() - start
        random.seed(config.seeds.optimizer)
        np.random.seed(config.seeds.optimizer)
        torch.manual_seed(config.seeds.optimizer)
        if torch.backends.mps.is_available(): torch.mps.manual_seed(config.seeds.optimizer)
        if config.model.coordinates == 'widened':
            raise ValueError('Widened models require a selected parent, not fresh fit')
        if config.model.coordinates == 'selective':
            model = SelectiveRegressor(config.model.d, config.model.h, config.model.tau)
        else:
            from .models.static import StaticRegressor
            model = StaticRegressor(config.model.d, config.model.h, config.model.static_count, config.model.coordinates, config.model.tau)
        model = model.to(config.backend.device, dtype=torch.float32)
        causal = model if config.model.h > 64 or config.model.coordinates != 'selective' else None
        backend = _backend_evidence(config, data, causal_model=causal)
        optimizer = _optimizer(model, config)
        schedule = UpdateSchedule(optimizer, kind=config.training.scheduler,
            horizon=config.training.base_updates + config.training.continuation_updates,
            min_lr=config.training.min_learning_rate, base_lrs=[config.training.learning_rate])
        generator = torch.Generator().manual_seed(config.seeds.optimizer)
        state = TrainingState(model, optimizer, schedule, 0, 0,
            torch.randperm(len(data.F.y), generator=generator), 0, generator, capture_rng(),
            preprocessing, _identities(data, config, preprocessing), config, backend,
            float(np.var(data.F.y, ddof=0)), _fg_hashes(data))
        state.growth_metadata = copy.deepcopy(growth_metadata or {})
        _new_stage(state, policy, 'base')
        _observe_initial(state, data, paths, preprocess_ns)
    except BaseException as error:
        _record_failure(paths, error, update=0)
        raise
    return _run(state, data, policy, paths)


def continue_fit(state: TrainingState, data: DevelopmentData, policy: TrainingPolicy, paths: RunPaths) -> TrainingState:
    """Resume an active stage exactly, or start a continuation with update 0 eligible."""
    try:
        _validate_inputs(data, state.config, paths)
        if state.identity_hashes != _identities(data, state.config, state.preprocessing) or state.data_hashes != _fg_hashes(data):
            raise ValueError('Continuation data/code/config identity mismatch')
        if (policy.logical_batch_size != state.config.training.batch_size
                or policy.validation_interval != state.config.training.validation_interval
                or policy.patience != state.config.training.patience
                or policy.checkpoint_interval != state.config.training.checkpoint_interval):
            raise ValueError('Continuation must preserve paired batch/evaluation/patience policy')
        _validate_live_adam_history(state)
        if state.best_snapshot is not None:
            validate_saved_adam_history(state.best_snapshot['metadata'], state.best_snapshot['tensors'])
        if state.stage['complete']:
            # Recovery can land after durable completion of last.json but before
            # final best publication/return. Continue from the selected full
            # snapshot, never the later last-update weights and moments.
            if state.best_snapshot is not None:
                recovered = unpack_state(state.best_snapshot, expected_hashes=state.identity_hashes, device=state.device)
                recovered.stage = copy.deepcopy(state.stage)
                recovered.history = copy.deepcopy(state.history)
                state = recovered
            if policy.max_updates > state.config.training.continuation_updates:
                raise ValueError('Continuation allowance exceeds config')
            _new_stage(state, policy, 'continuation')
        elif state.stage['policy'] != asdict(policy):
            raise ValueError('Resume must use the original unfinished-stage policy')
        _observe_initial(state, data, paths)
    except BaseException as error:
        _record_failure(paths, error, update=state.update)
        raise
    return _run(state, data, policy, paths)


def _record_failure(paths, error, *, update):
    """Preserve a failed attempt; retry writes a new RunPaths, never reopens it."""
    status = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
    record = {'type': 'failure', 'failure_type': type(error).__name__, 'message': str(error),
              'update': update, 'operational_status': status}
    # A terminal record or an interrupted metrics publication is deliberately not
    # repaired here. Preserve the original exception and evidence on log failure.
    try:
        append_event(paths, record)
        finish_run(paths, {'failure': record}, status)
    except (OSError, ValueError) as logging_error:
        error.add_note(f'Failure ledger publication also failed: {logging_error}')


def _run(state, data, policy, paths):
    from .checkpoint import save_checkpoint, save_best_checkpoint, prune_checkpoints
    checkpoint_dir = paths.directory / 'checkpoints'
    previous_threads = torch.get_num_threads()
    started = time.perf_counter_ns()
    validation_ns = 0
    checkpoint_safe = False
    try:
        checkpoint_dir.mkdir(exist_ok=True)
        torch.set_num_threads(state.backend['cpu_threads'])
        restore_rng(state.rng_states)
        append_event(paths, {'type': 'stage_started', 'status': 'running', 'stage': state.stage['kind'],
                             'update': state.update, 'resumed_stage': state.stage['updates_completed'] > 0})
        x = torch.tensor(state.preprocessing.transform_X(data.F.X), dtype=torch.float32, device=state.device)
        y = torch.tensor(state.preprocessing.transform_y(data.F.y), dtype=torch.float32, device=state.device)
        if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
            raise FloatingPointError('Nonfinite float32 training data after conversion')

        def record(event):
            state.history.append(event)
            append_event(paths, event)

        def evaluate():
            nonlocal validation_ns
            tick = time.perf_counter_ns()
            metrics = regression_metrics(data.G.y, state.predict(data.G.X), fit_target_variance=state.fit_target_variance)
            validation_ns += time.perf_counter_ns() - tick
            improved = state.stage['best_mse'] is None or metrics['mse'] < state.stage['best_mse']
            record({'type': 'evaluation', 'split': 'G', 'update': state.update,
                    'stage_update': state.stage['updates_completed'], 'metrics': metrics, 'improved': improved})
            if improved:
                state.stage['best_mse'] = metrics['mse']
                state.stage['bad_evaluations'] = 0
                state.rng_states = capture_rng()
                state.best_snapshot = pack_state(state)
            else:
                state.stage['bad_evaluations'] += 1

        def publish():
            state.rng_states = capture_rng()
            digest = save_checkpoint(state, checkpoint_dir / 'last.json')
            if state.best_snapshot['metadata']['update'] == state.update or not (checkpoint_dir / 'best.json').exists():
                save_best_checkpoint(state, checkpoint_dir / 'best.json')
            retention = prune_checkpoints(checkpoint_dir)
            best_digest = json.loads((checkpoint_dir / 'best.json').read_text())['sha256']
            update_run_observations(paths, {'method': {
                'weight_checkpoint': 'checkpoints/best.json', 'weight_hash': best_digest,
                'serialized_bytes': retention['referenced_bytes'], 'checkpoint_retention': retention}})
            record({'type': 'checkpoint', 'update': state.update, 'sha256': digest, 'path': 'checkpoints/last.json'})

        if state.best_snapshot is None:
            evaluate()
            checkpoint_safe = True
            publish()
        while state.stage['updates_completed'] < policy.max_updates and state.stage['bad_evaluations'] < policy.patience:
            # An asynchronous interruption during an update or G selection must
            # retain the previous durable state, never a partially applied Adam step.
            checkpoint_safe = False
            if state.cursor == len(state.order):
                state.epoch += 1
                state.order = torch.randperm(len(state.order), generator=state.sampler_generator)
                state.cursor = 0
            end = min(state.cursor + policy.logical_batch_size, len(state.order))
            ids = state.order[state.cursor:end].to(state.device)
            state.model.train()
            state.optimizer.zero_grad(set_to_none=True)
            prediction = state.model(x.index_select(0, ids))
            target = y.index_select(0, ids)
            if prediction.shape != target.shape:
                raise ValueError('Prediction/target shape mismatch; broadcasting is forbidden')
            # Same per-row squared error (without 1/2) as CPU64 score derivatives.
            loss = torch.square(prediction-target).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in state.model.parameters()):
                raise FloatingPointError('Nonfinite training gradient')
            lr = state.optimizer.param_groups[0]['lr']
            state.optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in state.model.parameters()):
                raise FloatingPointError('Nonfinite model after optimizer update')
            if any(isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
                   for values in state.optimizer.state.values() for value in values.values()):
                raise FloatingPointError('Nonfinite optimizer state after update')
            state.scheduler.step()
            state.cursor = end
            state.update += 1
            state.stage['updates_completed'] += 1
            evaluation_due = (state.stage['updates_completed'] % policy.validation_interval == 0
                              or state.stage['updates_completed'] == policy.max_updates)
            record({'type': 'update', 'update': state.update, 'stage_update': state.stage['updates_completed'],
                    'loss_standardized_mse': float(loss.detach().cpu()), 'learning_rate': lr,
                    'rows': len(ids), 'row_ids': data.F.row_id[ids.cpu().numpy()].tolist()})
            if evaluation_due:
                evaluate()
            # History and durable event publication (including due G selection)
            # must return before an interruption may snapshot this update.
            checkpoint_safe = True
            if evaluation_due or state.stage['updates_completed'] % policy.checkpoint_interval == 0:
                publish()
        state.stage['complete'] = True
        executed_updates = state.stage['updates_completed']
        record({'type': 'selection', 'selected_update': state.best_snapshot['metadata']['update'],
                'executed_updates': executed_updates, 'last_update': state.update,
                'g_mse': state.stage['best_mse'], 'rule': policy.checkpoint_selection})
        save_checkpoint(state, checkpoint_dir / 'last.json')
        selected = unpack_state(state.best_snapshot, expected_hashes=state.identity_hashes, device=state.device)
        selected.stage = copy.deepcopy(state.stage)
        selected.history = copy.deepcopy(state.history)
        selected.best_snapshot = None
        digest = save_checkpoint(selected, checkpoint_dir / 'best.json')
        retention = prune_checkpoints(checkpoint_dir)
        duration = time.perf_counter_ns() - started
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)
        resources = {'rss_bytes': int(rss)}
        if state.device == 'mps':
            resources.update(mps_allocated_bytes=torch.mps.current_allocated_memory(), mps_driver_bytes=torch.mps.driver_allocated_memory())
        update_run_observations(paths, {
            'method': {'weight_checkpoint': 'checkpoints/best.json', 'weight_hash': digest,
                       'serialized_bytes': retention['referenced_bytes'], 'checkpoint_retention': retention},
            'costs': _attempt_costs(state, duration, validation_ns),
            'resources': resources})
        return selected
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt) and checkpoint_safe:
            try:
                publish()
            except BaseException as publication_error:
                # The original interruption remains primary; the preceding
                # atomic pointer remains valid if emergency publication fails.
                error.add_note(f'Interruption checkpoint publication failed: {publication_error}')
        try:
            update_run_observations(paths, {'costs': _attempt_costs(state, time.perf_counter_ns()-started, validation_ns)})
        except (OSError, ValueError) as logging_error:
            error.add_note(f'Attempt cost publication also failed: {logging_error}')
        _record_failure(paths, error, update=state.update)
        raise
    finally:
        torch.set_num_threads(previous_threads)


def _attempt_costs(state, duration, validation_ns):
    return {
        state.stage['kind']: {'nanoseconds': duration, 'status': 'measured', 'reason': None,
            'boundary': 'Attempt wall time including F conversion, G validation, checkpoint IO and ledger updates; excludes F transform fitting, initial model/optimizer construction, upfront metadata, imports and backend-evidence validation'},
        'validation': {'nanoseconds': validation_ns, 'status': 'measured', 'reason': None,
            'boundary': f"G transformation, prediction and original-unit metrics; included in {state.stage['kind']}, do not sum again"}}
