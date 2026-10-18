"""Separate versioned custom-prevalence and correlated-input stress populations.

The original teacher is evaluated in latent coordinates z. Observed x=z@L has
population covariance L.T@L; targets and noise retain the original equations
and reference normalization. L mixes only signal coordinates. These stresses
are descriptive and cannot replace the primary confirmation population.
"""
from dataclasses import asdict, dataclass

import numpy as np

from treeneuron.config import canonical_hash
from treeneuron.data.synthetic import TeacherSpec, make_teacher, sample_teacher, evaluate_teacher, _observations
from treeneuron.data.types import _readonly
from .seeds import seed64


@dataclass(frozen=True)
class StressTeacher:
    base: TeacherSpec
    input_map_rows: tuple[tuple[float, ...], ...]
    prevalence: float | None
    condition_target: float
    version: str = 'research/v2-stress-teacher'
    kind: str = 'custom_prevalence_covariance'

    def __post_init__(self):
        if (self.version != 'research/v2-stress-teacher' or self.kind != 'custom_prevalence_covariance'
                or not isinstance(self.base, TeacherSpec)):
            raise ValueError('Unsupported stress teacher identity')
        if self.prevalence is not None and (type(self.prevalence) not in (int, float)
                                            or self.prevalence not in (.05, .2)):
            raise ValueError('Custom stress prevalence must be .05 or .2')
        if type(self.condition_target) not in (int, float) or self.condition_target not in (1., 10.):
            raise ValueError('Stress covariance condition target must be1 or10')
        if self.prevalence is not None and self.base.prevalences != (self.prevalence,)*4:
            raise ValueError('Base prevalence differs from declared stress')
        object.__setattr__(self, 'input_map_rows', tuple(tuple(float(x) for x in r) for r in self.input_map_rows))
        matrix = self.input_map
        if matrix.shape != (self.base.d, self.base.d) or not np.isfinite(matrix).all():
            raise ValueError('Stress input map must be finite square dimension d')
        if not np.isclose(np.linalg.cond(matrix.T@matrix), self.condition_target, rtol=1e-10, atol=1e-12):
            raise ValueError('Stored map differs from declared population covariance condition')
        active = self.base.d-self.base.nuisance_dimensions
        if (np.any(matrix[:active, active:] != 0) or np.any(matrix[active:, :active] != 0)
                or not np.array_equal(matrix[active:, active:], np.eye(self.base.nuisance_dimensions))):
            raise ValueError('Stress map cannot mix nuisance coordinates into the signal')

    @property
    def input_map(self):
        return _readonly(self.input_map_rows, np.dtype('float64'))

    @property
    def population_covariance(self):
        return self.input_map.T@self.input_map

    @property
    def teacher_id(self):
        return canonical_hash(self.to_dict())

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        values = dict(data)
        values['base'] = TeacherSpec.from_dict(values['base'])
        return cls(**values)


def make_stress_teacher(seed, d=16, *, prevalence=None, condition_target=1.,
                        regime='isotropic', nuisance_dimensions=0, reference_seed=None):
    if type(condition_target) not in (int, float) or condition_target not in (1., 10.):
        raise ValueError('Stress covariance condition target must be1 or10')
    if prevalence is not None and (type(prevalence) not in (int, float) or prevalence not in (.05, .2)):
        raise ValueError('Custom stress prevalence must be .05 or .2')
    if regime not in ('isotropic', 'axis_aligned', 'rotated'):
        raise ValueError('Stress uses isotropic, axis-aligned or rotated base structure')
    base = make_teacher(seed, d, reference_seed=reference_seed, regime=regime,
        nuisance_dimensions=nuisance_dimensions,
        gate_prevalences=(.1, .5) if prevalence is None else (prevalence,))
    active = d-nuisance_dimensions
    if active < 2 and condition_target != 1:
        raise ValueError('Correlated stress requires at least two signal dimensions')
    matrix = np.eye(d, dtype=np.float64)
    if condition_target != 1:
        rng = np.random.Generator(np.random.PCG64(seed64(seed, 'stress-v2', 'covariance')))
        q, r = np.linalg.qr(rng.standard_normal((active, active)))
        q *= np.where(np.diag(r) >= 0, 1., -1.)
        eigenvalues = np.geomspace(1., condition_target, active)
        matrix[:active, :active] = np.diag(np.sqrt(eigenvalues))@q.T
    return StressTeacher(base, tuple(tuple(float(x) for x in r) for r in matrix),
                         prevalence, float(condition_target))


def sample_stress(teacher, n, seed, sigma=.1):
    rows = sample_teacher(teacher.base, n, seed, sigma)
    return _observations(teacher.teacher_id, rows.X@teacher.input_map,
                         rows.noise_free_target, rows.y, seed)


def evaluate_stress(teacher, X):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != teacher.base.d or not np.isfinite(values).all():
        raise ValueError('Stress predictors must be finite [n,d]')
    latent = np.linalg.solve(teacher.input_map.T, values.T).T
    return evaluate_teacher(teacher.base, latent)
