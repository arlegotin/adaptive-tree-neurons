"""Immutable data boundaries; diagnostic metadata is never a predictor column."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import numpy as np

from treeneuron.config import canonical_hash


def _readonly(value: object, dtype: np.dtype) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    # Immutable bytes prevent callers from re-enabling WRITEABLE on owned arrays.
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def array_hash(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ObservationSet:
    X: np.ndarray
    y: np.ndarray
    row_id: np.ndarray
    group_id: np.ndarray
    dates: np.ndarray | None = None
    noise_free_target: np.ndarray | None = None
    stream_id: str = ''
    feature_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        X = np.asarray(self.X)
        y = np.asarray(self.y)
        if X.ndim != 2 or y.shape != (len(X),):
            raise ValueError('Expected X [n,d] and y [n]')
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError('Synthetic predictors and labels must be finite')
        object.__setattr__(self, 'X', _readonly(X, np.dtype('float64')))
        object.__setattr__(self, 'y', _readonly(y, np.dtype('float64')))
        for name in ('row_id', 'group_id'):
            values = np.asarray(getattr(self, name))
            if values.dtype != np.int64 or values.shape != (len(X),):
                raise ValueError(f'{name} must be a one-dimensional CPU int64 array')
            object.__setattr__(self, name, _readonly(values, np.dtype('int64')))
        if len(np.unique(self.row_id)) != len(X):
            raise ValueError('Observation row IDs must be unique')
        if self.noise_free_target is not None:
            values = np.asarray(self.noise_free_target)
            if values.shape != y.shape or not np.isfinite(values).all():
                raise ValueError('Noise-free targets must be finite [n] metadata')
            object.__setattr__(self, 'noise_free_target', _readonly(values, np.dtype('float64')))
        if self.dates is not None:
            dates = np.asarray(self.dates)
            if dates.shape != y.shape:
                raise ValueError('dates must have shape [n]')
            object.__setattr__(self, 'dates', _readonly(dates, dates.dtype))
        names = tuple(self.feature_names) or tuple(f'x{i}' for i in range(X.shape[1]))
        if len(names) != X.shape[1] or len(set(names)) != len(names):
            raise ValueError('Feature allowlist must name each predictor once')
        forbidden = {'y', 'noise_free_target', 'gates', 'teacher', 'reference_mean',
                     'reference_std', 'row_id', 'group_id', 'dates'}
        if forbidden.intersection(names):
            raise ValueError('Diagnostic and identity fields cannot enter feature allowlist')
        object.__setattr__(self, 'feature_names', names)


def predictor_matrix(rows: ObservationSet, features: Sequence[str] | None = None) -> np.ndarray:
    """Select only explicitly allowlisted input columns, never object metadata."""
    if features is None:
        return rows.X
    if not features or any(name not in rows.feature_names for name in features):
        raise ValueError('Requested predictor is outside the feature allowlist')
    return rows.X[:, [rows.feature_names.index(name) for name in features]]


@dataclass(frozen=True)
class SplitManifest:
    dataset_id: str
    row_id: tuple[int, ...]
    group_id: tuple[int, ...]
    assignments: tuple[str, ...]
    teacher_id: str | None = None
    fold: int | None = None
    phase: str = 'exploratory'
    date_boundaries: tuple[tuple[str, str, str], ...] = ()
    hashes: tuple[tuple[str, str], ...] = ()
    feature_names: tuple[str, ...] = ()
    descriptive_only_reason: str | None = None
    assignment_hash: str = field(init=False)
    feature_schema_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # Accept lists at the boundary, but never retain mutable assignments.
        for name in ('row_id', 'group_id', 'assignments', 'feature_names'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('date_boundaries', 'hashes'):
            object.__setattr__(self, name, tuple(tuple(item) for item in getattr(self, name)))
        if not self.dataset_id or not (len(self.row_id) == len(self.group_id) == len(self.assignments)):
            raise ValueError('Manifest identity and assignment lengths must be valid')
        if any(type(v) is not int for v in (*self.row_id, *self.group_id)):
            raise ValueError('Manifest IDs must be integers')
        if len(set(self.row_id)) != len(self.row_id):
            raise ValueError('Manifest row IDs overlap')
        if self.phase not in ('exploratory', 'confirmatory'):
            raise ValueError('Manifest phase must be declared')
        if (any(type(name) is not str or not name for name in self.feature_names)
                or len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError('Manifest feature schema must contain unique nonempty names')
        object.__setattr__(self, 'feature_schema_hash',
                           canonical_hash({'feature_names': self.feature_names}))
        group_parts: dict[int, str] = {}
        for group, partition in zip(self.group_id, self.assignments):
            if partition not in ('F', 'G', 'V', 'T'):
                raise ValueError('Unknown manifest partition')
            if group in group_parts and group_parts[group] != partition:
                raise ValueError('A group cannot cross partitions')
            group_parts[group] = partition
        object.__setattr__(self, 'assignment_hash', canonical_hash({
            'dataset_id': self.dataset_id, 'teacher_id': self.teacher_id,
            'fold': self.fold, 'phase': self.phase,
            'row_id': self.row_id, 'group_id': self.group_id, 'assignments': self.assignments,
            'date_boundaries': self.date_boundaries,
            'feature_names': self.feature_names, 'feature_schema_hash': self.feature_schema_hash,
        }))

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True)
class DevelopmentData:
    F: ObservationSet
    G: ObservationSet
    V: ObservationSet
    teacher_id: str
    manifest: SplitManifest | None = None

    def __post_init__(self) -> None:
        observations = (self.F, self.G, self.V)
        if any(rows.feature_names != self.F.feature_names for rows in observations):
            raise ValueError('Development partitions must have identical feature schemas')
        row_ids = tuple(int(i) for rows in observations for i in rows.row_id)
        group_ids = tuple(int(i) for rows in observations for i in rows.group_id)
        assignments = tuple(p for p, rows in zip('FGV', observations) for _ in rows.y)
        hashes = tuple((f'{p}.{name}', array_hash(getattr(rows, name)))
                       for p, rows in zip('FGV', observations) for name in ('X', 'y'))
        schema_hash = canonical_hash({'feature_names': self.F.feature_names})
        dataset_id = canonical_hash({'teacher_id': self.teacher_id, 'hashes': hashes,
                                     'row_id': row_ids, 'feature_names': self.F.feature_names,
                                     'feature_schema_hash': schema_hash})
        expected = SplitManifest(dataset_id=dataset_id, teacher_id=self.teacher_id,
                                 row_id=row_ids, group_id=group_ids,
                                 assignments=assignments, hashes=hashes,
                                 feature_names=self.F.feature_names)
        if self.manifest is not None and self.manifest != expected:
            raise ValueError('Manifest does not match development observations')
        object.__setattr__(self, 'manifest', expected)
