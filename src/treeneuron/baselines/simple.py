"""Closed-form F-only sanity controls, with strict JSON estimator snapshots.

Ridge penalizes coefficients on F-standardized features (including fixed category
codes/indicators), with an unpenalized intercept. Alpha is supplied externally;
these adapters do not inspect G or run hyperparameter selection.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np

from treeneuron.baselines.api import FitRequest
from treeneuron.data.transforms import FittedTransform, finite_vector


def _alpha(value: Any) -> float:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (float, int, np.floating, np.integer))
            or not np.isfinite(value) or value < 0):
        raise ValueError('Ridge alpha must be finite and nonnegative')
    return float(value)


class MeanPredictor:
    def __init__(self) -> None:
        self._transform: FittedTransform | None = None
        self._coefficients: np.ndarray | None = None

    def _fitted(self) -> tuple[FittedTransform, np.ndarray]:
        if self._transform is None or self._coefficients is None:
            raise RuntimeError('Predictor is not fitted')
        return self._transform, self._coefficients

    def fit(self, request: FitRequest) -> MeanPredictor:
        transform = FittedTransform.fit(request.X_fit, request.y_fit)
        # Standardized mean is zero; inverse_y restores the F mean in original units.
        self._transform, self._coefficients = transform, np.zeros(1)
        return self

    def predict(self, X: Any, *, batch_size: int | None = None) -> np.ndarray:
        transform, coefficients = self._fitted()
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError('batch_size must be a positive integer or None')
        features = transform.transform_X(X)
        # For these small CPU controls batch_size does not alter prediction semantics.
        standardized = np.full(len(features), coefficients[0])
        if isinstance(self, RidgePredictor):
            standardized += features @ coefficients[1:]
        return transform.inverse_y(standardized)

    def _state(self) -> dict[str, Any]:
        transform, coefficients = self._fitted()
        return {'version': 1, 'method': type(self).__name__, 'transform': transform.to_state(),
                'coefficients': coefficients.tolist(),
                'alpha': self.alpha if isinstance(self, RidgePredictor) else None}

    def save(self, path: Path) -> None:
        # JSON only: no executable pickle/vendor checkpoint state.
        payload = json.dumps(self._state(), allow_nan=False, sort_keys=True)
        Path(path).write_text(payload + '\n', encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> MeanPredictor:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if (type(state) is not dict or set(state) != {'version', 'method', 'transform', 'coefficients', 'alpha'}
                or type(state['version']) is not int or state['version'] != 1
                or state['method'] != cls.__name__):
            raise ValueError('Invalid predictor snapshot schema or method')
        transform = FittedTransform.from_state(state['transform'])
        coefficients = finite_vector(state['coefficients'], 'coefficients', nonempty=True)
        width = len(state['transform']['columns']) + sum(
            c.get('indicator', False) for c in state['transform']['columns'])
        if cls is RidgePredictor:
            model = cls(_alpha(state['alpha']))
            if len(coefficients) != width + 1:
                raise ValueError('Ridge coefficient shape does not match transform')
        else:
            model = cls()
            if state['alpha'] is not None or len(coefficients) != 1 or coefficients[0] != 0.:
                raise ValueError('Invalid mean estimator state')
        model._transform, model._coefficients = transform, coefficients.copy()
        return model

    def resource_summary(self) -> dict[str, Any]:
        unavailable = lambda reason: {'value': None, 'reason': reason}
        return {
            'capabilities': {
                'internal_validation': False, 'selection_labels_used': False,
                'preprocessing': 'F-only median/population scaling; fixed indicators and ordinal category codes',
                'output_units': 'original_target', 'categories': 'F-only codes; reserved unknown 0',
                'sample_weights': False, 'transductive': False, 'context': False, 'cache': False,
                'backend': 'numpy_cpu', 'dtype': 'float64', 'fallback': False,
                'batch_composition_dependent': False,
            },
            'fitted': self._transform is not None,
            'alpha': self.alpha if isinstance(self, RidgePredictor) else None,
            'total_parameters': len(self._coefficients) if self._coefficients is not None else None,
            'rss_bytes': unavailable('process memory not measured by adapter'),
            'mps_allocated_bytes': unavailable('CPU-only adapter'),
            'mps_driver_bytes': unavailable('CPU-only adapter'),
            'fit_nanoseconds': unavailable('timing belongs to execution harness'),
            'search_nanoseconds': unavailable('adapter performs no search'),
        }


class RidgePredictor(MeanPredictor):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = _alpha(alpha)

    def fit(self, request: FitRequest) -> RidgePredictor:
        transform = FittedTransform.fit(request.X_fit, request.y_fit)
        features = transform.transform_X(request.X_fit)
        target = transform.transform_y(request.y_fit)
        design = np.column_stack([np.ones(len(features)), features])
        if self.alpha:
            regularizer = np.column_stack([np.zeros(features.shape[1]),
                                           np.sqrt(self.alpha) * np.eye(features.shape[1])])
            design = np.vstack([design, regularizer])
            target = np.r_[target, np.zeros(features.shape[1])]
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        finite_vector(coefficients, 'fitted coefficients')
        self._transform, self._coefficients = transform, coefficients
        return self


class SplineRidgePredictor(MeanPredictor):
    """Additive cubic regression splines; F quantile knots, caller-selected alpha.

    Each numerical feature contributes x, x², x³ and (x-knot)₊³ independently.
    Missingness indicators enter linearly. There are no cross-feature products
    or gates. The unpenalized intercept and ridge solve use F-standardized y.
    """
    def __init__(self, alpha: float, *, n_knots: int = 5):
        super().__init__()
        self.alpha = _alpha(alpha)
        if type(n_knots) is not int or n_knots <= 0:
            raise ValueError('n_knots must be a positive integer')
        self.n_knots = n_knots
        self.knots: list[np.ndarray] = []

    def _design(self, features):
        pieces = [np.ones((len(features), 1))]
        with np.errstate(over='ignore', invalid='ignore'):
            for j, knots in enumerate(self.knots):
                x = features[:, j:j+1]
                pieces.extend((x, x*x, x*x*x, np.maximum(x-knots, 0.)**3))
            pieces.append(features[:, len(self.knots):])
            design = np.concatenate(pieces, axis=1)
        if not np.isfinite(design).all():
            raise ValueError('Spline basis overflowed')
        return design

    def fit(self, request: FitRequest) -> SplineRidgePredictor:
        transform = FittedTransform.fit(request.X_fit, request.y_fit)
        columns = transform.to_state()['columns']
        if any(c['kind'] != 'numeric' for c in columns):
            raise ValueError('SplineRidge supports numeric features only')
        features = transform.transform_X(request.X_fit)
        knots = []
        for j in range(len(columns)):
            values = features[:, j]
            k = np.unique(np.quantile(values, np.linspace(0, 1, self.n_knots+2)[1:-1]))
            knots.append(k[(k > values.min()) & (k < values.max())])
        # Build on a temporary estimator, so a failed fit preserves fitted state.
        candidate = SplineRidgePredictor(self.alpha, n_knots=self.n_knots)
        candidate.knots = knots
        design = candidate._design(features)
        target = transform.transform_y(request.y_fit)
        width = design.shape[1]-1
        if self.alpha:
            design = np.vstack((design, np.column_stack((np.zeros(width), np.sqrt(self.alpha)*np.eye(width)))))
            target = np.r_[target, np.zeros(width)]
        coefficients = finite_vector(np.linalg.lstsq(design, target, rcond=None)[0], 'spline coefficients')
        self._transform, self._coefficients, self.knots = transform, coefficients, knots
        return self

    def predict(self, X: Any, *, batch_size: int | None = None) -> np.ndarray:
        transform, coefficients = self._fitted()
        if batch_size is not None and (type(batch_size) is not int or batch_size <= 0):
            raise ValueError('batch_size must be a positive integer or None')
        return transform.inverse_y(self._design(transform.transform_X(X)) @ coefficients)

    def _state(self):
        transform, coefficients = self._fitted()
        return dict(version=1, method=type(self).__name__, transform=transform.to_state(),
                    coefficients=coefficients.tolist(), alpha=self.alpha, n_knots=self.n_knots,
                    basis='additive_cubic_truncated_power', knots=[k.tolist() for k in self.knots])

    @classmethod
    def load(cls, path: Path) -> SplineRidgePredictor:
        state = json.loads(Path(path).read_text(encoding='utf-8'))
        if (type(state) is not dict or set(state) != {'version','method','transform','coefficients','alpha','n_knots','basis','knots'}
                or type(state['version']) is not int or state['version'] != 1
                or state['method'] != cls.__name__ or state['basis'] != 'additive_cubic_truncated_power'):
            raise ValueError('Invalid spline snapshot schema, method or basis')
        model = cls(state['alpha'], n_knots=state['n_knots'])
        transform = FittedTransform.from_state(state['transform'])
        columns = transform.to_state()['columns']
        if (any(c['kind'] != 'numeric' for c in columns) or type(state['knots']) is not list
                or len(state['knots']) != len(columns)):
            raise ValueError('Spline requires one knot vector per numeric feature')
        knots = [finite_vector(k, 'knots') for k in state['knots']]
        if any(len(k) > model.n_knots or (np.diff(k) <= 0).any() for k in knots):
            raise ValueError('Spline knots must be strictly increasing and bounded in count')
        coefficients = finite_vector(state['coefficients'], 'coefficients', nonempty=True)
        width = 1 + sum(3+len(k) for k in knots) + sum(c['indicator'] for c in columns)
        if len(coefficients) != width:
            raise ValueError('Spline coefficient width differs from basis')
        model._transform, model._coefficients, model.knots = transform, coefficients.copy(), knots
        return model

    def resource_summary(self):
        summary = super().resource_summary()
        summary['alpha'] = self.alpha
        summary['n_knots'] = self.n_knots
        summary['capabilities'].update(preprocessing='F-only median/scaling and per-feature quantile knots; F missing indicators',
                                       categories='unsupported; explicit error', basis='additive cubic; no multivariable gates')
        return summary
