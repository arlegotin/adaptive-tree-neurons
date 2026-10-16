"""Explicit exploratory control equations, distributions and diagnostic access.

Version 1 choices beyond the primary/original-2D specification are exploratory:
  globally_affine: .3 + u'x
  additive_sine_quadratic: .3 sin(u'x) + .3 ((v'x)^2 - 1)
  pure_noise: f=0, y=sigma*xi
  homogeneous_smooth: (1/sqrt(4)) sum_r sin(u_r'x)
  random_label_permutation: independent within-draw permutation of primary y
All except original 2D draw independent standard Gaussian inputs. Control target
units are their explicit equations (no target normalization); the fixed separate
reference records diagnostic mean/std only. The permutation's primary base uses
primary normalization. F-fitted preprocessing remains a later operation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import math
import numpy as np

from treeneuron.config import canonical_hash
from .synthetic import (
    REFERENCE_ROWS, GATE_TYPES, TeacherSpec, _integer, _rng, _unit_vectors,
    _tuples, _validate_X, _sample_parameters, _observations, make_teacher,
    sample_teacher, evaluate_teacher, derive_seed,
)
from .types import _readonly

CONTROL_VERSION = 'exploratory-controls-v1'
CONTROL_KINDS = ('globally_affine', 'additive_sine_quadratic', 'pure_noise',
                 'random_label_permutation', 'conditional_slope_2d', 'homogeneous_smooth')
EQUATIONS = {
    'globally_affine': '.3 + dot(u,x)',
    'additive_sine_quadratic': '.3*sin(dot(u,x)) + .3*(dot(v,x)**2-1)',
    'pure_noise': '0; y=sigma*independent_standard_normal',
    'random_label_permutation': 'independent_permutation(primary_normalized_f(x)+sigma*xi)',
    'conditional_slope_2d': 'x[1] if x[0]<0 else 1-x[1]',
    'homogeneous_smooth': '.5*sum(sin(dot(u[r],x)),r=0..3)',
}


@dataclass(frozen=True)
class ControlSpec:
    version: str
    kind: str
    d: int
    seed: int
    vectors: tuple[tuple[float, ...], ...]
    equation: str
    coefficients: tuple[tuple[str, float], ...]
    reference_seed: int
    reference_mean: float
    reference_std: float
    reference_rows: int = REFERENCE_ROWS
    input_distribution: str = 'independent_N(0,1)'
    target_normalization: str = 'none'
    regime: str = 'isotropic'
    nuisance_dimensions: int = 0
    rotation: tuple[tuple[float, ...], ...] = ()
    base_teacher: TeacherSpec | None = None
    phase: str = 'exploratory'

    def __post_init__(self) -> None:
        _integer(self.d, 'd', 1); _integer(self.seed, 'seed')
        _integer(self.reference_seed, 'reference_seed')
        _integer(self.nuisance_dimensions, 'nuisance_dimensions')
        if self.version != CONTROL_VERSION or self.phase != 'exploratory':
            raise ValueError('Unsupported control version or generation phase')
        if self.kind not in CONTROL_KINDS:
            raise ValueError('Unsupported control kind')
        if self.reference_rows != REFERENCE_ROWS or self.nuisance_dimensions >= self.d:
            raise ValueError('Invalid control reference count or signal dimension')
        if self.kind == 'conditional_slope_2d' and self.d != 2:
            raise ValueError('Original conditional-slope control requires d=2')
        for name in ('vectors', 'coefficients', 'rotation'):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))
        count = {'globally_affine': 1, 'additive_sine_quadratic': 2,
                 'homogeneous_smooth': 4}.get(self.kind, 0)
        if len(self.vectors) != count:
            raise ValueError('Wrong number of control vectors')
        if count:
            vectors = np.asarray(self.vectors)
            if (vectors.shape != (count, self.d) or not np.isfinite(vectors).all()
                    or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12, rtol=0)):
                raise ValueError('Control vectors must be finite unit vectors')
        if (not math.isfinite(self.reference_mean) or not math.isfinite(self.reference_std)
                or self.reference_std < 0 or any(not math.isfinite(v) for _, v in self.coefficients)):
            raise ValueError('Control reference statistics and coefficients must be finite')
        if self.kind == 'random_label_permutation' and (
                not isinstance(self.base_teacher, TeacherSpec) or self.base_teacher.d != self.d):
            raise ValueError('Permutation control requires a matching primary base teacher')

    @property
    def teacher_id(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def reference_stream_id(self) -> str:
        distribution = 'uniform-2d-v1' if self.kind == 'conditional_slope_2d' else 'gaussian-v1'
        return f'exploratory/reference/{distribution}/{self.reference_seed}/d{self.d}'

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> ControlSpec:
        data = dict(value)
        if data.get('version') != CONTROL_VERSION:
            raise ValueError('Unsupported control version')
        for name in ('vectors', 'coefficients', 'rotation'):
            data[name] = tuple(tuple(row) for row in data[name])
        if data['base_teacher'] is not None:
            data['base_teacher'] = TeacherSpec.from_dict(data['base_teacher'])
        return cls(**data)


def control_target(control: ControlSpec, X: np.ndarray) -> np.ndarray:
    """Noise-free control equation; permutation has no pointwise target."""
    values = _validate_X(X, control.d)
    c = dict(control.coefficients)
    if control.kind == 'conditional_slope_2d':
        return np.where(values[:, 0] < 0, values[:, 1], 1-values[:, 1])
    if control.kind == 'pure_noise':
        return np.zeros(len(values), dtype=np.float64)
    if control.kind == 'random_label_permutation':
        raise ValueError('Row-label permutation has no noise-free pointwise target')
    projections = np.einsum('nd,rd->nr', values, np.asarray(control.vectors))
    if control.kind == 'globally_affine':
        return c['intercept'] + c['linear'] * projections[:, 0]
    if control.kind == 'additive_sine_quadratic':
        return c['sine'] * np.sin(projections[:, 0]) + c['quadratic'] * (projections[:, 1]**2 - 1)
    if control.kind == 'homogeneous_smooth':
        return c['component_scale'] * np.sin(projections).sum(axis=1)
    raise ValueError('Unsupported control kind')


def _inputs(control: ControlSpec, n: int, seed: int, *, reference: bool = False) -> np.ndarray:
    root = 'exploratory/reference' if reference else 'exploratory/observation'
    if control.kind == 'conditional_slope_2d':
        rng = _rng(seed, f'{root}/uniform-2d-v1')
        return rng.uniform(size=(n, 2)) * np.array([2., 1.]) - np.array([1., 0.])
    namespace = f'{root}/gaussian-v1' if reference else f'{root}/inputs-v1'
    return _rng(seed, namespace).standard_normal((n, control.d))


def make_control(kind: str, seed: int, d: int = 16, *, reference_seed: int | None = None,
                 reference_rows: int = REFERENCE_ROWS, regime: str = 'isotropic',
                 nuisance_dimensions: int = 0, gate_prevalences: tuple[float, ...] = (.1, .5),
                 gate_types: tuple[str, ...] = GATE_TYPES) -> TeacherSpec | ControlSpec:
    """Build a named control or explicit primary sensitivity.

    rare/axis_aligned/rotated/nuisance are convenience names for primary teacher
    sensitivities. nuisance uses four final irrelevant coordinates by default;
    all details, including thresholds and the rotation, are stored in TeacherSpec.
    """
    _integer(seed, 'seed'); _integer(d, 'd', 1)
    kwargs = dict(reference_seed=reference_seed, reference_rows=reference_rows,
                  regime=regime, nuisance_dimensions=nuisance_dimensions,
                  gate_prevalences=gate_prevalences, gate_types=gate_types)
    if kind in ('primary', 'rare', 'axis_aligned', 'rotated', 'nuisance'):
        if kind in ('rare', 'axis_aligned', 'rotated'):
            if regime != 'isotropic' and regime != kind:
                raise ValueError('Conflicting control kind and regime')
            kwargs['regime'] = kind
        if kind == 'nuisance' and nuisance_dimensions == 0:
            kwargs['nuisance_dimensions'] = 4
        return make_teacher(seed, d, **kwargs)
    if kind not in CONTROL_KINDS:
        raise ValueError(f'Unsupported control kind: {kind}')
    if kind == 'conditional_slope_2d' and d != 2:
        raise ValueError('conditional_slope_2d requires d=2')
    if kind == 'random_label_permutation':
        base = make_teacher(seed, d, **kwargs)
    else:
        base = None
        if tuple(gate_prevalences) != (.1, .5) or tuple(gate_types) != GATE_TYPES:
            raise ValueError('This control has no gates; custom gate configuration is unsupported')
        if regime not in ('isotropic', 'axis_aligned', 'rotated'):
            raise ValueError('This control has no rare gates; requested regime is unsupported')
    _integer(nuisance_dimensions, 'nuisance_dimensions')
    if nuisance_dimensions >= d:
        raise ValueError('At least one signal dimension is required')
    if kind == 'conditional_slope_2d' and (regime != 'isotropic' or nuisance_dimensions):
        raise ValueError('Original 2D control requires its unchanged input coordinates')
    if reference_rows != REFERENCE_ROWS or type(reference_rows) is not int:
        raise ValueError('Version 1 requires exactly 65536 reference observations')
    reference_seed = (derive_seed(seed, 'exploratory/reference-root-v1')
                      if reference_seed is None else reference_seed)
    _integer(reference_seed, 'reference_seed')
    count = {'globally_affine': 1, 'additive_sine_quadratic': 2,
             'homogeneous_smooth': 4}.get(kind, 0)
    active = d - nuisance_dimensions
    vectors = _unit_vectors(_rng(seed, f'exploratory/control/{kind}-v1'), count, d, active, regime)
    rotation = ()
    if regime == 'rotated' and count:
        q, r = np.linalg.qr(_rng(seed, 'exploratory/rotation/v1').standard_normal((active, active)))
        q *= np.where(np.diag(r) >= 0, 1., -1.)
        vectors[:, :active] = vectors[:, :active] @ q
        rotation = _tuples(q)
    coefficients = {
        'globally_affine': (('intercept', .3), ('linear', 1.)),
        'additive_sine_quadratic': (('sine', .3), ('quadratic', .3)),
        'homogeneous_smooth': (('component_scale', .5),),
    }.get(kind, ())
    control = ControlSpec(version=CONTROL_VERSION, kind=kind, d=d, seed=seed,
                          vectors=_tuples(vectors), equation=EQUATIONS[kind],
                          coefficients=coefficients, reference_seed=reference_seed,
                          reference_mean=0., reference_std=0., regime=regime,
                          nuisance_dimensions=nuisance_dimensions, rotation=rotation,
                          input_distribution=('independent_Uniform(-1,1)_x_Uniform(0,1)'
                                              if kind == 'conditional_slope_2d' else 'independent_N(0,1)'),
                          target_normalization=('primary_base_only' if base else 'none'),
                          base_teacher=base)
    if base is not None:
        return replace(control, reference_mean=base.reference_mean, reference_std=base.reference_std)
    target = control_target(control, _inputs(control, REFERENCE_ROWS, reference_seed, reference=True))
    return replace(control, reference_mean=float(target.mean()), reference_std=float(target.std(ddof=0)))


def sample_control(control: ControlSpec | TeacherSpec, n: int, seed: int,
                   sigma: float | None = None) -> ObservationSet:
    """Sample a control; original 2D defaults/requires zero noise, others .1."""
    if isinstance(control, TeacherSpec):
        return sample_teacher(control, n, seed, .1 if sigma is None else sigma)
    if not isinstance(control, ControlSpec):
        raise ValueError('Expected a versioned teacher/control specification')
    sigma = (0. if control.kind == 'conditional_slope_2d' else .1) if sigma is None else sigma
    _sample_parameters(n, seed, sigma)
    if control.kind == 'conditional_slope_2d' and sigma != 0:
        raise ValueError('Original conditional_slope_2d control requires no noise')
    if control.kind == 'random_label_permutation':
        rows = sample_teacher(control.base_teacher, n, seed, sigma)
        permutation = _rng(seed, 'exploratory/observation/label-permutation-v1').permutation(n)
        return _observations(control.teacher_id, rows.X, None, rows.y[permutation], seed)
    X = _inputs(control, n, seed)
    target = control_target(control, X)
    noise = _rng(seed, 'exploratory/observation/noise-v1').standard_normal(n)
    return _observations(control.teacher_id, X, target, target + sigma * noise, seed)


@dataclass(frozen=True)
class OracleDiagnosticFeatures:
    X: np.ndarray
    feature_names: tuple[str, ...]
    information_advantaged: bool = True
    purpose: str = 'oracle_gate_least_squares_diagnostic'


def oracle_diagnostic_features(teacher: TeacherSpec, X: np.ndarray) -> OracleDiagnosticFeatures:
    """Explicit unfair diagnostic basis [1, x, (2g_r-1)*x for r=0..3]."""
    X = _validate_X(X, teacher.d)
    _, gates = evaluate_teacher(teacher, X, return_gates=True)
    columns = [np.ones((len(X), 1)), X]
    names = ['intercept', *(f'x{i}' for i in range(teacher.d))]
    for r in range(4):
        columns.append((2*gates[:, r:r+1]-1) * X)
        names.extend(f'oracle_gate_{r}_times_x{i}' for i in range(teacher.d))
    return OracleDiagnosticFeatures(_readonly(np.concatenate(columns, axis=1), np.dtype('float64')),
                                    tuple(names))
