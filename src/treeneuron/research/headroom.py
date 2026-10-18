"""F-only frozen hidden-feature ridge refit with finite G selection."""
from dataclasses import dataclass
import json
import math
import time

import numpy as np

from treeneuron.config import canonical_json
from treeneuron.data.types import _readonly

HEAD_ALPHAS = (0., 1e-6, 1e-4, 1e-2, 1., 100.)


@dataclass(frozen=True)
class HeadRefit:
    alpha: float
    intercept: float
    coefficients: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    active_columns: np.ndarray
    records_json: str

    @property
    def records(self):
        return json.loads(self.records_json)['trials']

    def predict_features(self, H):
        values = np.asarray(H, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_mean) or not np.isfinite(values).all():
            raise ValueError('Frozen head requires finite matching feature columns')
        active = self.active_columns
        result = self.intercept+((values[:, active]-self.feature_mean[active])/self.feature_scale[active])@self.coefficients
        if not np.isfinite(result).all():
            raise FloatingPointError('Nonfinite frozen-head prediction')
        return result


def solve_head_features(F, yF, G, yG, *, alphas=HEAD_ALPHAS):
    f, g = np.asarray(F, dtype=np.float64), np.asarray(G, dtype=np.float64)
    yf, yg = np.asarray(yF, dtype=np.float64), np.asarray(yG, dtype=np.float64)
    if (f.ndim != 2 or g.ndim != 2 or f.shape[1] != g.shape[1] or not len(f) or not len(g)
            or yf.shape != (len(f),) or yg.shape != (len(g),)
            or any(not np.isfinite(a).all() for a in (f, g, yf, yg))
            or not alphas or len(set(alphas)) != len(alphas)
            or any(type(a) not in (float, int) or not math.isfinite(a) or a < 0 for a in alphas)):
        raise ValueError('Frozen head requires finite F/G arrays and unique nonnegative ridge alphas')
    with np.errstate(over='ignore', invalid='ignore'):
        mean = f.mean(axis=0)
        centered = f-mean
        scale = np.sqrt(np.mean(centered**2, axis=0))
    if any(not np.isfinite(a).all() for a in (mean, centered, scale)):
        raise FloatingPointError('Nonfinite frozen-head F geometry')
    active = scale > 0
    scale[~active] = 1.
    normalized = centered[:, active]/scale[active]
    query = (g[:, active]-mean[active])/scale[active]
    intercept = float(yf.mean())
    target = yf-intercept
    if any(not np.isfinite(a).all() for a in (normalized, query, target)) or not math.isfinite(intercept):
        raise FloatingPointError('Nonfinite frozen-head normalized geometry')
    root_n = math.sqrt(len(f))
    records, solutions = [], []
    for alpha in sorted(float(a) for a in alphas):
        tick = time.perf_counter_ns()
        try:
            matrix, response = normalized/root_n, target/root_n
            if alpha > 0:
                matrix = np.vstack((matrix, math.sqrt(alpha)*np.eye(normalized.shape[1])))
                response = np.concatenate((response, np.zeros(normalized.shape[1])))
            coefficient, _, rank, singular = np.linalg.lstsq(matrix, response, rcond=None)
            prediction = intercept+query@coefficient
            score = float(np.mean((prediction-yg)**2))
            if not np.isfinite(coefficient).all() or not math.isfinite(score):
                raise FloatingPointError('Nonfinite frozen-head ridge fit')
            records.append(dict(alpha=alpha, G_mse=score, rank=int(rank),
                singular_values=singular.tolist(), coefficients=coefficient.tolist(), intercept=intercept,
                elapsed_ns=time.perf_counter_ns()-tick, status='completed_eligible'))
            solutions.append(coefficient)
        except (np.linalg.LinAlgError, FloatingPointError) as error:
            error.head_records = [*records, dict(alpha=alpha, status='numerical_failure',
                elapsed_ns=time.perf_counter_ns()-tick, reason=str(error))]
            raise
    best = min(r['G_mse'] for r in records)
    tolerance = 1e-12*max(1., abs(best))
    winner = next(i for i, record in enumerate(records) if record['G_mse'] <= best+tolerance)
    return HeadRefit(records[winner]['alpha'], intercept,
        _readonly(solutions[winner], np.dtype('float64')), _readonly(mean, np.dtype('float64')),
        _readonly(scale, np.dtype('float64')), _readonly(active, np.dtype('bool')),
        canonical_json({'trials': records, 'score_units': 'provided_target_coordinates_squared',
            'selection': 'G_mse_smallest_alpha_with_1e-12_relative_floor_ties'}))


@dataclass(frozen=True)
class FrozenHeadPredictor:
    head: HeadRefit
    weights: np.ndarray
    bias: np.ndarray
    preprocessing: object
    parent_identity_json: str

    @property
    def parent_identity_hashes(self):
        return json.loads(self.parent_identity_json)

    def hidden_features(self, X):
        import torch
        values = torch.tensor(self.preprocessing.transform_X(X), dtype=torch.float32)
        with torch.no_grad():
            result = torch.nn.functional.linear(values, torch.tensor(self.weights), torch.tensor(self.bias)).relu()
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('Nonfinite frozen parent hidden features')
        return result.numpy().astype(np.float64)

    def predict(self, X):
        return self.preprocessing.inverse_y(self.head.predict_features(self.hidden_features(X)))


def refit_frozen_head(parent, data):
    from treeneuron import train
    from treeneuron.data.transforms import FittedTransform
    from .roles import FittingData
    if (not isinstance(parent, train.TrainingState) or not isinstance(data, FittingData)
            or type(parent.model) is not train.SelectiveRegressor
            or parent.model.layer.growth_applied or parent.device != 'cpu'
            or parent.identity_hashes != train._identities(data, parent.config, parent.preprocessing)
            or parent.data_hashes != train._fg_hashes(data)):
        raise ValueError('Frozen-head control requires matching F/G-only ordinary CPU parent')
    train.pack_state(parent)  # Validate the supplied parent, without changing its fit.
    transform = FittedTransform.from_state(parent.preprocessing.to_state())
    placeholder = FrozenHeadPredictor(None,
        _readonly(parent.model.layer.base.weight.detach().numpy(), np.dtype('float32')),
        _readonly(parent.model.layer.base.bias.detach().numpy(), np.dtype('float32')),
        transform, canonical_json(parent.identity_hashes))
    fitted = solve_head_features(placeholder.hidden_features(data.F.X), transform.transform_y(data.F.y),
        placeholder.hidden_features(data.G.X), transform.transform_y(data.G.y))
    return FrozenHeadPredictor(fitted, placeholder.weights, placeholder.bias, transform, placeholder.parent_identity_json)
