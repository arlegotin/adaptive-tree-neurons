"""Descriptive regression metrics; no inference or protected-label access."""
from __future__ import annotations

from typing import Any
import numpy as np

from treeneuron.data.transforms import finite_vector


def _mean_nonnegative(values: np.ndarray) -> float:
    # Dividing first avoids overflowing a sum whose arithmetic mean is finite.
    return float(np.sum(values / len(values)))


def regression_metrics(y: np.ndarray, prediction: np.ndarray, *, fit_target_variance: float,
                       groups: np.ndarray | None = None,
                       noisefree: np.ndarray | None = None) -> dict[str, Any]:
    y = finite_vector(y, 'y', nonempty=True)
    prediction = finite_vector(prediction, 'prediction')
    if prediction.shape != y.shape:
        raise ValueError('Prediction and target row counts differ')
    if (isinstance(fit_target_variance, (bool, np.bool_))
            or not isinstance(fit_target_variance, (int, float, np.integer, np.floating))
            or not np.isfinite(fit_target_variance) or fit_target_variance < 0):
        raise ValueError('F population target variance must be finite and nonnegative')
    absolute = np.abs(prediction - y)
    squared = np.square(absolute)
    if not np.isfinite(squared).all():
        raise ValueError('Regression loss overflowed')
    mse = _mean_nonnegative(squared)
    result: dict[str, Any] = {
        'n': len(y), 'mse': mse, 'rmse': float(np.sqrt(mse)), 'mae': _mean_nonnegative(absolute),
        'normalized_mse': mse / float(fit_target_variance) if fit_target_variance else None,
        'normalized_mse_reason': None if fit_target_variance else 'zero F population target variance; degenerate task',
        'absolute_error_p95': float(np.quantile(absolute, .95)), 'max_absolute_error': float(absolute.max()),
        'group_macro_mse': None, 'worst_group_mse': None, 'group_metrics': None,
    }
    if result['normalized_mse'] is not None and not np.isfinite(result['normalized_mse']):
        raise ValueError('Normalized regression loss overflowed')
    if groups is not None:
        labels = np.asarray(groups, dtype=object)
        if labels.shape != y.shape:
            raise ValueError('Groups must be a vector of stable string or integer labels')
        normalized = [v.item() if isinstance(v, np.generic) else v for v in labels]
        if not (all(type(v) is str for v in normalized) or all(type(v) is int for v in normalized)):
            raise ValueError('Groups must contain exclusively strings or exclusively integers')
        rows = []
        for label in sorted(set(normalized)):
            mask = np.array([v == label for v in normalized])
            rows.append({'group': label, 'n': int(mask.sum()), 'mse': _mean_nonnegative(squared[mask]),
                         'mae': _mean_nonnegative(absolute[mask])})
        result.update(group_metrics=rows, group_macro_mse=_mean_nonnegative(np.array([r['mse'] for r in rows])),
                      worst_group_mse=max(r['mse'] for r in rows))
    if noisefree is not None:
        target = finite_vector(noisefree, 'noisefree')
        if target.shape != y.shape:
            raise ValueError('Noise-free target row count differs')
        loss = np.square(prediction - target)
        if not np.isfinite(loss).all():
            raise ValueError('Noise-free loss overflowed')
        result['noise_free_mse'] = _mean_nonnegative(loss)
    return result
