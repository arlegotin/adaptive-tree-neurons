"""F-only preprocessing with an explicit, JSON-reconstructible feature schema.

Arrays are numerical; DataFrame columns are explicitly supplied predictors in order.
Numerical indicators exist only for columns missing in F, preserving complete
synthetic width. Later missing values in previously complete columns use the F
median without a new indicator. Categorical missing/unknown values share code 0;
known typed string/integer/boolean categories receive fixed codes starting at 1.
Project-record metadata names are rejected, never dropped. This is a guard
against passing our own records as X; source adapters must still supply their
own explicit predictor allowlists to exclude dataset-specific leakage columns.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import json
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_complex_dtype, is_numeric_dtype

_PROJECT_METADATA = frozenset({'y', 'noise_free_target', 'gates', 'teacher',
                               'reference_mean', 'reference_std', 'row_id', 'group_id', 'dates'})


def finite_vector(value: Any, name: str, *, nonempty: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iuf' or raw.ndim != 1:
        raise ValueError(f'{name} must be a real numerical vector [N]')
    result = np.asarray(raw, dtype=np.float64)
    if (nonempty and not len(result)) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be finite and nonempty when fitting/evaluating')
    return result


def _category(value: Any) -> list[Any] | None:
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return None
    if isinstance(value, (str, np.str_)):
        return ['str', str(value)]
    if isinstance(value, (bool, np.bool_)):
        return ['bool', bool(value)]
    if isinstance(value, (int, np.integer)):
        return ['int', int(value)]
    raise ValueError('Categorical values must be strings, integers, booleans or missing')


def _numerical(values: Any) -> np.ndarray:
    try:
        result = np.asarray(pd.array(values, dtype='Float64').to_numpy(dtype=float, na_value=np.nan))
    except (TypeError, ValueError) as exc:
        raise ValueError('Expected numerical values or missing values') from exc
    if np.isinf(result).any():
        raise ValueError('Infinite predictors are invalid')
    return result


def _schema(X: Any) -> tuple[str, list[str], list[str], list[Any], int]:
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        if (not names or any(type(v) is not str or not v for v in names)
                or len(set(names)) != len(names)):
            raise ValueError('Predictor names must be unique nonempty strings')
        if _PROJECT_METADATA.intersection(names):
            raise ValueError('Project metadata cannot enter predictor columns')
        if any(is_complex_dtype(dtype) for dtype in X.dtypes):
            raise ValueError('Complex predictors are unsupported')
        kinds = ['numeric' if is_numeric_dtype(dtype) and not is_bool_dtype(dtype)
                 else 'categorical' for dtype in X.dtypes]
        return 'dataframe', names, kinds, [X[name] for name in names], len(X)
    array = np.asarray(X)
    if array.ndim != 2 or not array.shape[1] or array.dtype.kind not in 'iuf':
        raise ValueError('Array predictors must be a real numerical matrix [N,D]')
    return ('array', [f'x{i}' for i in range(array.shape[1])],
            ['numeric'] * array.shape[1], [array[:, i] for i in range(array.shape[1])], len(array))


class FittedTransform:
    """Use ``fit`` or strict ``from_state`` to construct fitted preprocessing."""

    def __init__(self, state: Mapping[str, Any]):
        self._state = copy.deepcopy(dict(state))

    @classmethod
    def fit(cls, X_fit: Any, y_fit: np.ndarray) -> FittedTransform:
        y = finite_vector(y_fit, 'y_fit', nonempty=True)
        container, names, kinds, values, n = _schema(X_fit)
        if n != len(y):
            raise ValueError('F features and labels have different row counts')
        columns = []
        for name, kind, raw in zip(names, kinds, values):
            column: dict[str, Any] = {'name': name, 'kind': kind}
            if kind == 'numeric':
                values_array = _numerical(raw)
                missing = np.isnan(values_array)
                all_missing = bool(missing.all())
                fill = 0. if all_missing else float(np.median(values_array[~missing]))
                imputed = np.where(missing, fill, values_array)
                mean, scale = float(np.mean(imputed)), float(np.std(imputed, ddof=0))
                column.update(fill=fill, mean=mean, scale=scale or 1., constant=scale == 0.,
                              all_missing=all_missing, indicator=bool(missing.any()))
            else:
                categories = [_category(value) for value in raw]
                known = {json.dumps(value): value for value in categories if value is not None}
                column['categories'] = [known[key] for key in sorted(known)]
            columns.append(column)
        target_scale = float(np.std(y, ddof=0))
        return cls.from_state({'version': 1, 'container': container, 'columns': columns,
                               'indicator_policy': 'F_missing_only',
                               'previously_complete_missing_policy': 'F_median_without_indicator',
                               'target_mean': float(np.mean(y)), 'target_scale': target_scale or 1.,
                               'target_constant': target_scale == 0.})

    def transform_X(self, X: Any) -> np.ndarray:
        container, names, kinds, raw_columns, n = _schema(X)
        columns = self._state['columns']
        if (container != self._state['container'] or names != [c['name'] for c in columns]
                or kinds != [c['kind'] for c in columns]):
            raise ValueError('Predictor container, column order or kinds differ from F schema')
        output, indicators = [], []
        for column, raw in zip(columns, raw_columns):
            if column['kind'] == 'numeric':
                values = _numerical(raw)
                missing = np.isnan(values)
                output.append((np.where(missing, column['fill'], values) - column['mean']) / column['scale'])
                if column['indicator']:
                    indicators.append(missing.astype(np.float64))
            else:
                codes = {json.dumps(value): i + 1 for i, value in enumerate(column['categories'])}
                output.append(np.array([codes.get(json.dumps(_category(value)), 0) for value in raw], dtype=float))
        result = np.column_stack(output + indicators).reshape(n, len(output) + len(indicators))
        if not np.isfinite(result).all():
            raise ValueError('Feature transformation overflowed')
        return result

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        result = (finite_vector(y, 'y') - self._state['target_mean']) / self._state['target_scale']
        return finite_vector(result, 'transformed y')

    def inverse_y(self, prediction: np.ndarray) -> np.ndarray:
        result = finite_vector(prediction, 'prediction') * self._state['target_scale'] + self._state['target_mean']
        return finite_vector(result, 'original-unit prediction')

    def to_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FittedTransform:
        """Reject unknown fields and invalid/nonfinite numerical or schema state."""
        def number(value: Any, *, positive: bool = False) -> bool:
            return (type(value) in (int, float) and np.isfinite(value)
                    and (not positive or value > 0))

        try:
            state = copy.deepcopy(dict(state))
            json.dumps(state, allow_nan=False)
            if (set(state) != {'version', 'container', 'columns', 'indicator_policy',
                               'previously_complete_missing_policy',
                               'target_mean', 'target_scale', 'target_constant'}
                    or type(state['version']) is not int or state['version'] != 1
                    or state['container'] not in ('array', 'dataframe')
                    or state['indicator_policy'] != 'F_missing_only'
                    or state['previously_complete_missing_policy'] != 'F_median_without_indicator'
                    or not number(state['target_mean']) or not number(state['target_scale'], positive=True)
                    or type(state['target_constant']) is not bool
                    or (state['target_constant'] and state['target_scale'] != 1.)
                    or type(state['columns']) is not list or not state['columns']):
                raise ValueError('Invalid transform state header')
            names = []
            for column in state['columns']:
                if type(column) is not dict or type(column.get('name')) is not str or not column['name']:
                    raise ValueError('Invalid column state')
                names.append(column['name'])
                if column.get('kind') == 'numeric':
                    if (set(column) != {'name', 'kind', 'fill', 'mean', 'scale', 'constant', 'all_missing', 'indicator'}
                            or not all(number(column[key]) for key in ('fill', 'mean'))
                            or not number(column['scale'], positive=True)
                            or any(type(column[key]) is not bool for key in ('constant', 'all_missing', 'indicator'))
                            or (column['constant'] and column['scale'] != 1.)
                            or (column['all_missing'] and (column['fill'] != 0. or column['mean'] != 0.
                                                         or not column['indicator'] or not column['constant']))):
                        raise ValueError('Invalid numerical column state')
                elif column.get('kind') == 'categorical':
                    if set(column) != {'name', 'kind', 'categories'} or type(column['categories']) is not list:
                        raise ValueError('Invalid categorical column state')
                    keys = []
                    for value in column['categories']:
                        if (type(value) is not list or len(value) != 2
                                or value[0] not in ('str', 'int', 'bool')
                                or type(value[1]) is not {'str': str, 'int': int, 'bool': bool}[value[0]]):
                            raise ValueError('Invalid category token')
                        keys.append(json.dumps(value))
                    if len(keys) != len(set(keys)):
                        raise ValueError('Duplicate categories')
                else:
                    raise ValueError('Unknown column kind')
            if len(names) != len(set(names)):
                raise ValueError('Duplicate column names')
            if _PROJECT_METADATA.intersection(names):
                raise ValueError('Project metadata cannot enter saved predictor schema')
            if state['container'] == 'array' and (names != [f'x{i}' for i in range(len(names))]
                                                  or any(c['kind'] != 'numeric' for c in state['columns'])):
                raise ValueError('Invalid array schema')
        except (TypeError, KeyError, OverflowError) as exc:
            raise ValueError('Malformed transform state') from exc
        return cls(state)
