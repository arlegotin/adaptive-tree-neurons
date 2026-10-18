"""Explicit intervention at a captured actual state, without a G-best claim."""
import copy
import math

from treeneuron import train
from treeneuron.config import canonical_hash
from treeneuron.evaluation.metrics import regression_metrics
from treeneuron.experiments.paired import _json_tree
from .roles import FittingData


def validate_boundary_record(meta):
    record = meta['growth_metadata'].get('research_boundary')
    keys = {'schema', 'selection_policy', 'requested_update', 'actual_update',
            'source_snapshot_sha256', 'source_identity_hashes', 'original_stage', 'current_g_mse'}
    if (not isinstance(record, dict) or set(record) != keys
            or record['schema'] != 'research/v2-current-boundary'
            or record['selection_policy'] != 'actual_update'
            or type(record['actual_update']) is not int or not 0 <= record['actual_update'] <= meta['update']
            or type(record['requested_update']) is not int or record['requested_update'] != record['actual_update']
            or type(record['current_g_mse']) not in (float, int) or not math.isfinite(record['current_g_mse']) or record['current_g_mse'] < 0
            or type(record['source_snapshot_sha256']) is not str or len(record['source_snapshot_sha256']) != 64
            or any(c not in '0123456789abcdef' for c in record['source_snapshot_sha256'])):
        raise ValueError('Invalid declared current research boundary')
    identities = record['source_identity_hashes']
    if (not isinstance(identities, dict) or set(identities) != set(meta['identity_hashes'])
            or any(type(v) is not str or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in identities.values())
            or any(v != meta['identity_hashes'][k] for k, v in identities.items() if k != 'config')):
        raise ValueError('Current boundary source/data/runtime lineage differs')
    if identities['config'] != meta['identity_hashes']['config'] and (
            meta['model']['kind'] != 'widened_relu_scalar'
            or meta['growth_metadata'].get('parent_identity_hashes') != identities):
        raise ValueError('Current boundary config changed without declared ordinary widening lineage')
    original = record['original_stage']
    if (not isinstance(original, dict) or set(original) != set(meta['stage'])
            or original['kind'] != 'base' or original['complete'] is not False
            or type(original['start_update']) is not int or type(original['updates_completed']) is not int
            or original['start_update']+original['updates_completed'] != record['actual_update']):
        raise ValueError('Current boundary must come from the declared active parent stage')
    policy = train.TrainingPolicy(**original['policy'])
    if (original['start_update'] < 0 or not 0 <= original['updates_completed'] <= policy.max_updates
            or type(original['bad_evaluations']) is not int or original['bad_evaluations'] < 0
            or policy.logical_batch_size != meta['config']['training']['batch_size']
            or record['actual_update'] not in policy.snapshot_updates):
        raise ValueError('Invalid original current-boundary stage counters/policy')
    if meta['stage']['kind'] == 'research_boundary' and (
            meta['stage']['complete'] is not True or meta['stage']['updates_completed'] != 0
            or meta['stage']['start_update'] != record['actual_update'] or meta['update'] != record['actual_update']
            or meta['stage']['bad_evaluations'] != 0 or meta['stage']['best_mse'] != record['current_g_mse']):
        raise ValueError('Current boundary stage is inconsistent with its actual-update selection')


def current_boundary(snapshot, data, *, requested_update):
    if not isinstance(data, FittingData):
        raise ValueError('Current boundary requires an F/G-only worker view')
    if (type(requested_update) is not int or requested_update < 0
            or snapshot['metadata']['update'] != requested_update
            or snapshot['metadata']['stage']['complete'] is not False
            or snapshot['metadata']['stage']['kind'] != 'base'
            or snapshot['metadata']['growth_metadata']):
        raise ValueError('Require the exact requested active, ungrown parent snapshot')
    state = train.unpack_state(snapshot, expected_hashes=snapshot['metadata']['identity_hashes'], device='cpu')
    if (state.identity_hashes != train._identities(data, state.config, state.preprocessing)
            or state.data_hashes != train._fg_hashes(data) or state.model.layer.growth_applied):
        raise ValueError('Current boundary parent/data mismatch or previous growth')
    risk = regression_metrics(data.G.y, state.predict(data.G.X), fit_target_variance=state.fit_target_variance)['mse']
    record = dict(schema='research/v2-current-boundary', selection_policy='actual_update',
        requested_update=requested_update, actual_update=state.update,
        source_snapshot_sha256=canonical_hash(_json_tree(snapshot)),
        source_identity_hashes=copy.deepcopy(state.identity_hashes),
        original_stage=copy.deepcopy(state.stage), current_g_mse=risk)
    state.stage = dict(kind='research_boundary', start_update=state.update, updates_completed=0,
        bad_evaluations=0, best_mse=risk, policy=copy.deepcopy(state.stage['policy']), complete=True)
    state.growth_metadata = {'research_boundary': record}
    state.best_snapshot = None
    packed = train.pack_state(state)
    train.validate_metadata(packed['metadata'], state.identity_hashes, state.device)
    return state


def insert_at_current_boundary(state, allocation):
    if state.stage.get('kind') != 'research_boundary':
        raise ValueError('Research insertion requires a declared actual-update boundary')
    from treeneuron.growth.insert import _insert_at_completed_boundary
    packed = train.pack_state(state)
    train.validate_metadata(packed['metadata'], state.identity_hashes, state.device)
    return _insert_at_completed_boundary(state, allocation, allow_current_boundary=True)


def widen_at_current_boundary(state, added_units, seed):
    if state.stage.get('kind') != 'research_boundary':
        raise ValueError('Research widening requires a declared actual-update boundary')
    from treeneuron.baselines.causal import _widen_at_completed_boundary
    packed = train.pack_state(state)
    train.validate_metadata(packed['metadata'], state.identity_hashes, state.device)
    return _widen_at_completed_boundary(state, added_units, seed, allow_current_boundary=True)
