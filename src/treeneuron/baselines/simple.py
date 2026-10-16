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
