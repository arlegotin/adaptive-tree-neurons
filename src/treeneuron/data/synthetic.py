"""Versioned independent Gaussian teachers, reference normalization and F/G/V draws.

All numerical generation/equations use CPU float64. No function in this module
loads or generates a protected test partition. Protocol validation must precede
any later evaluator's use of the low-level population sampler.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import math
import numpy as np
from scipy.special import expit, ndtri

from treeneuron.config import ExperimentConfig, canonical_hash, resolve_experiment
from .types import DevelopmentData, ObservationSet

VERSION = 'independent-regime-teacher-v1'
ROTATED_VERSION = 'axis-paired-rotated-teacher-v2'
REFERENCE_ROWS = 65536
DEFAULT_COUNTS = {'F': 2306, 'G': 577, 'V': 577}
GATE_TYPES = ('hard', 'smooth_0.2', 'smooth_1.0')
GATE_SCALES = {'hard': 0., 'smooth_0.2': .2, 'smooth_1.0': 1.}


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')


def derive_seed(seed: int, namespace: str) -> int:
    """Stable 63-bit seed from an explicit exploratory domain and integer root."""
    _integer(seed, 'seed')
    if not namespace or not namespace.startswith('exploratory/'):
        raise ValueError('Generation supports an explicit exploratory seed namespace only')
    payload = f'treeneuron-seeds-v1:{namespace}:{seed}'.encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') & ((1 << 63) - 1)


def _rng(seed: int, namespace: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(seed, namespace)))


def _unit_vectors(rng: np.random.Generator, count: int, d: int, active: int,
                  regime: str) -> np.ndarray:
    if regime == 'axis_aligned':
        vectors = np.zeros((count, d), dtype=np.float64)
        vectors[np.arange(count), rng.integers(active, size=count)] = rng.choice([-1., 1.], count)
        return vectors
    vectors = rng.standard_normal((count, active))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    result = np.zeros((count, d), dtype=np.float64)
    result[:, :active] = vectors
    return result


def _tuples(array: np.ndarray) -> tuple:
    return tuple(tuple(float(v) for v in row) for row in array)


def _rotate_inputs(X: np.ndarray, rotation: tuple, active: int) -> np.ndarray:
    """Pair row coordinates with the coefficient convention a_rot=a_axis @ Q."""
    if rotation:
        X[:, :active] = X[:, :active] @ np.asarray(rotation)
    return X


@dataclass(frozen=True)
class TeacherSpec:
    version: str
    d: int
    seed: int
    u0: tuple[float, ...]
    u: tuple[tuple[float, ...], ...]
    v: tuple[tuple[float, ...], ...]
    w: tuple[tuple[float, ...], ...]
    prevalences: tuple[float, ...]
    gate_types: tuple[str, ...]
    scales: tuple[float, ...]
    thresholds: tuple[float, ...]
    reference_seed: int
    reference_mean: float
    reference_std: float
    reference_rows: int = REFERENCE_ROWS
    gate_type_probabilities: tuple[float, ...] = (1/3, 1/3, 1/3)
    prevalence_options: tuple[float, ...] = (.1, .5)
    gate_type_options: tuple[str, ...] = GATE_TYPES
    regime: str = 'isotropic'
    nuisance_dimensions: int = 0
    rotation: tuple[tuple[float, ...], ...] = ()
    phase: str = 'exploratory'

    def __post_init__(self) -> None:
        _integer(self.d, 'd', 1); _integer(self.seed, 'seed')
        _integer(self.reference_seed, 'reference_seed')
        _integer(self.nuisance_dimensions, 'nuisance_dimensions')
        expected_version = ROTATED_VERSION if self.regime == 'rotated' else VERSION
        if self.version != expected_version or self.phase != 'exploratory':
            raise ValueError('Unsupported teacher version or generation phase')
        if self.reference_rows != REFERENCE_ROWS or self.nuisance_dimensions >= self.d:
            raise ValueError('Invalid reference count or signal dimension')
        if self.regime not in ('isotropic', 'rare', 'axis_aligned', 'rotated'):
            raise ValueError('Unsupported regime')
        for name in ('u0', 'prevalences', 'gate_types', 'scales', 'thresholds',
                     'gate_type_probabilities', 'prevalence_options', 'gate_type_options'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ('u', 'v', 'w', 'rotation'):
            object.__setattr__(self, name, tuple(tuple(row) for row in getattr(self, name)))
        if any(len(getattr(self, name)) != 4 for name in
               ('u', 'v', 'w', 'prevalences', 'gate_types', 'scales', 'thresholds')):
            raise ValueError('Exactly four independent regime components are required')
        vectors = np.asarray((self.u0, *self.u, *self.v, *self.w), dtype=np.float64)
        if (vectors.shape != (13, self.d) or not np.isfinite(vectors).all()
                or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-12, rtol=0)):
            raise ValueError('Teacher vectors must be finite unit vectors [13,d]')
        if (any(g not in GATE_TYPES for g in self.gate_types)
                or any(g not in GATE_TYPES for g in self.gate_type_options)
                or any(not math.isfinite(p) or not 0 < p < 1 for p in
                       (*self.prevalences, *self.prevalence_options))):
            raise ValueError('Invalid gate type or prevalence')
        if any(scale != GATE_SCALES[kind] for scale, kind in zip(self.scales, self.gate_types)):
            raise ValueError('Gate scales must match their declared types')
        if (not np.isfinite(self.thresholds).all() or not math.isfinite(self.reference_mean)
                or not math.isfinite(self.reference_std) or self.reference_std <= 0):
            raise ValueError('Thresholds and normalization must be finite with positive scale')
        expected_thresholds = ndtri(1-np.asarray(self.prevalences))
        if not np.allclose(self.thresholds, expected_thresholds, atol=1e-12, rtol=0):
            raise ValueError('Gate thresholds must equal Phi^-1(1-prevalence)')
        if (not self.gate_type_options or not self.prevalence_options
                or len(self.gate_type_probabilities) != len(self.gate_type_options)
                or any(p <= 0 or not math.isfinite(p) for p in self.gate_type_probabilities)
                or not math.isclose(sum(self.gate_type_probabilities), 1)):
            raise ValueError('Invalid component sampling probabilities')
        if self.nuisance_dimensions and np.any(vectors[:, -self.nuisance_dimensions:] != 0):
            raise ValueError('Teacher vectors must exclude nuisance coordinates')
        if bool(self.rotation) != (self.regime == 'rotated'):
            raise ValueError('Rotation matrix is required exactly for the rotated regime')
        if self.rotation:
            active = self.d - self.nuisance_dimensions
            matrix = np.asarray(self.rotation)
            if matrix.shape != (active, active) or not np.allclose(matrix.T @ matrix, np.eye(active)):
                raise ValueError('Rotation must be orthogonal on the signal subspace')

    @property
    def teacher_id(self) -> str:
        return canonical_hash(self.to_dict())

    @property
    def reference_stream_id(self) -> str:
        stream = f'exploratory/reference/gaussian-v1/{self.reference_seed}/d{self.d}'
        if self.rotation:
            stream += f'/{ROTATED_VERSION}/{canonical_hash({"rotation": self.rotation})}'
        return stream

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping) -> TeacherSpec:
        data = dict(value)
        for name in ('u0', 'prevalences', 'gate_types', 'scales', 'thresholds',
                     'gate_type_probabilities', 'prevalence_options', 'gate_type_options'):
            data[name] = tuple(data[name])
        for name in ('u', 'v', 'w', 'rotation'):
            data[name] = tuple(tuple(row) for row in data[name])
        return cls(**data)


def reference_inputs(teacher: TeacherSpec) -> np.ndarray:
    """Reproduce the separate fixed reference, for normalization diagnostics only."""
    if teacher.reference_rows != REFERENCE_ROWS:
        raise ValueError('Version 1 requires exactly 65536 reference observations')
    X = _rng(teacher.reference_seed, 'exploratory/reference/gaussian-v1').standard_normal(
        (REFERENCE_ROWS, teacher.d))
    return _rotate_inputs(X, teacher.rotation, teacher.d-teacher.nuisance_dimensions)


def _validate_X(X: np.ndarray, d: int) -> np.ndarray:
    values = np.asarray(X, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != d or not np.isfinite(values).all():
        raise ValueError(f'Expected finite X [n,{d}]')
    return values


def _raw_teacher(teacher: TeacherSpec, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gate_projection = np.einsum('nd,rd->nr', X, np.asarray(teacher.v))
    gates = np.empty_like(gate_projection)
    for r, (kind, scale, threshold) in enumerate(zip(teacher.gate_types, teacher.scales, teacher.thresholds)):
        gates[:, r] = ((gate_projection[:, r] >= threshold) if kind == 'hard' else
                       expit((gate_projection[:, r] - threshold) / scale))
    linear = np.einsum('nd,rd->nr', X, np.asarray(teacher.u))
    switched = np.einsum('nd,rd->nr', X, np.asarray(teacher.w))
    raw = .3 * np.sin(np.einsum('nd,d->n', X, np.asarray(teacher.u0)))
    raw += .5 * (.3 * linear + (2 * gates - 1) * switched).sum(axis=1)
    return raw, gates


def evaluate_teacher(teacher: TeacherSpec, X: np.ndarray, *, return_gates: bool = False):
    """Diagnostic population equation; returned gates must never enter fair inputs."""
    raw, gates = _raw_teacher(teacher, _validate_X(X, teacher.d))
    target = (raw - teacher.reference_mean) / teacher.reference_std
    return (target, gates) if return_gates else target


def make_teacher(seed: int, d: int = 16, *, reference_seed: int | None = None,
                 reference_rows: int = REFERENCE_ROWS, regime: str = 'isotropic',
                 nuisance_dimensions: int = 0, gate_prevalences: tuple[float, ...] = (.1, .5),
                 gate_types: tuple[str, ...] = GATE_TYPES) -> TeacherSpec:
    """Create four independent components. Sensitivities are explicit v1 choices.

    rare: all p=.01; axis_aligned: independent signed coordinate axes;
    rotated v2: the same signed-axis teacher/gates as axis_aligned, with one
    stored orthogonal map applied to coefficients and sampled/reference inputs.
    Last nuisance_dimensions coordinates are unused. Isotropic v1 is unchanged.
    """
    _integer(seed, 'seed'); _integer(d, 'd', 1)
    _integer(nuisance_dimensions, 'nuisance_dimensions')
    if nuisance_dimensions >= d:
        raise ValueError('At least one signal dimension is required')
    if reference_rows != REFERENCE_ROWS or type(reference_rows) is not int:
        raise ValueError('Version 1 requires exactly 65536 reference observations')
    if regime not in ('isotropic', 'rare', 'axis_aligned', 'rotated'):
        raise ValueError('Unsupported regime')
    if not gate_prevalences or any(type(p) not in (int, float) or not 0 < p < 1 for p in gate_prevalences):
        raise ValueError('Gate prevalences must be nonempty probabilities in (0,1)')
    if not gate_types or any(g not in GATE_TYPES for g in gate_types):
        raise ValueError('Unsupported gate type')
    if regime == 'rare' and tuple(gate_prevalences) not in ((.1, .5), (.01,)):
        raise ValueError('Rare v1 fixes prevalence at .01; custom prevalence conflicts')
    reference_seed = (derive_seed(seed, 'exploratory/reference-root-v1')
                      if reference_seed is None else reference_seed)
    _integer(reference_seed, 'reference_seed')
    rng = _rng(seed, 'exploratory/teacher/v1')
    active = d - nuisance_dimensions
    vectors = _unit_vectors(rng, 13, d, active, 'axis_aligned' if regime == 'rotated' else regime)
    rotation = ()
    if regime == 'rotated':
        q, r = np.linalg.qr(_rng(seed, 'exploratory/rotation/v1').standard_normal((active, active)))
        q *= np.where(np.diag(r) >= 0, 1., -1.)
        vectors[:, :active] = vectors[:, :active] @ q
        rotation = _tuples(q)
    options = (.01,) if regime == 'rare' else tuple(float(p) for p in gate_prevalences)
    prevalences = tuple(float(p) for p in rng.choice(options, size=4))
    types = tuple(str(g) for g in rng.choice(gate_types, size=4))
    teacher = TeacherSpec(
        version=ROTATED_VERSION if regime == 'rotated' else VERSION,
        d=d, seed=seed, u0=tuple(float(x) for x in vectors[0]),
        u=_tuples(vectors[1:5]), v=_tuples(vectors[5:9]), w=_tuples(vectors[9:13]),
        prevalences=prevalences, gate_types=types,
        scales=tuple(GATE_SCALES[g] for g in types),
        thresholds=tuple(float(ndtri(1-p)) for p in prevalences),
        reference_seed=reference_seed, reference_mean=0., reference_std=1.,
        gate_type_probabilities=tuple(1/len(gate_types) for _ in gate_types),
        prevalence_options=options, gate_type_options=tuple(gate_types),
        regime=regime, nuisance_dimensions=nuisance_dimensions, rotation=rotation)
    raw, _ = _raw_teacher(teacher, reference_inputs(teacher))
    std = float(raw.std(ddof=0))
    if not math.isfinite(std) or std <= 0:
        raise ValueError('Teacher has degenerate normalization')
    return replace(teacher, reference_mean=float(raw.mean()), reference_std=std)


def _sample_parameters(n: int, seed: int, sigma: float) -> None:
    _integer(n, 'n', 1); _integer(seed, 'seed')
    if type(sigma) not in (float, int) or not math.isfinite(sigma) or sigma < 0:
        raise ValueError('Noise sigma must be finite and nonnegative')


def _observations(teacher_id: str, X: np.ndarray, target: np.ndarray | None, y: np.ndarray,
                  seed: int) -> ObservationSet:
    stream = f'exploratory/observation/v1/{teacher_id}/{seed}'
    ids = np.array([int.from_bytes(hashlib.sha256(f'{stream}/{i}'.encode()).digest()[:8], 'big')
                    & ((1 << 63) - 1) for i in range(len(X))], dtype=np.int64)
    return ObservationSet(X=X, y=y, row_id=ids, group_id=ids,
                          noise_free_target=target, stream_id=stream)


def sample_teacher(teacher: TeacherSpec, n: int, seed: int, sigma: float = .1) -> ObservationSet:
    """Draw original inputs only; target/gates never augment X."""
    if not isinstance(teacher, TeacherSpec):
        from .controls import sample_control
        return sample_control(teacher, n, seed, sigma)
    _sample_parameters(n, seed, sigma)
    X = _rng(seed, 'exploratory/observation/inputs-v1').standard_normal((n, teacher.d))
    X = _rotate_inputs(X, teacher.rotation, teacher.d-teacher.nuisance_dimensions)
    target = evaluate_teacher(teacher, X)
    noise = _rng(seed, 'exploratory/observation/noise-v1').standard_normal(n)
    return _observations(teacher.teacher_id, X, target, target + sigma * noise, seed)


def make_development(teacher: TeacherSpec, counts: Mapping[str, int] | None = None,
                     seeds: Mapping[str, int] | None = None, sigma: float = .1) -> DevelopmentData:
    """F/G/V only. Counts and seed maps require exactly these case-sensitive keys.

    Default counts are 2306/577/577. Default phase seeds derive from the teacher's
    seed via exploratory/development/{F,G,V}-v1; an explicit seed map must contain
    distinct nonnegative integers. No protected path or T-key is accepted.
    """
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    seeds = ({p: derive_seed(teacher.seed, f'exploratory/development/{p}-v1') for p in 'FGV'}
             if seeds is None else dict(seeds))
    if set(counts) != set('FGV') or set(seeds) != set('FGV'):
        raise ValueError('Development count/seed namespace must contain exactly F, G, V')
    for p in 'FGV':
        _sample_parameters(counts[p], seeds[p], sigma)
    if len(set(seeds.values())) != 3:
        raise ValueError('Overlapping phase seed namespaces are prohibited')
    rows = {p: sample_teacher(teacher, counts[p], seeds[p], sigma) for p in 'FGV'}
    return DevelopmentData(**rows, teacher_id=teacher.teacher_id)


def make_from_config(config: ExperimentConfig):
    """Return (population, development); reject unsupported semantics explicitly.

    This boundary generates raw data. F-only standardization belongs to the later
    preprocessing module. Optimizer/allocation/contrast seeds belong to training.
    """
    config = resolve_experiment(config.to_dict())
    data, seeds = config.data, config.seeds
    if data.components != 4:
        raise ValueError('Version 1 supports exactly four components')
    if data.reference_rows != REFERENCE_ROWS:
        raise ValueError('Version 1 requires exactly 65536 reference observations')
    if data.n_test is not None:
        raise ValueError('Protected test requests require the later protected evaluator')
    kwargs = dict(reference_seed=seeds.reference, reference_rows=data.reference_rows,
                  regime=data.regime, nuisance_dimensions=data.nuisance_dimensions,
                  gate_prevalences=data.gate_prevalences, gate_types=data.gate_types)
    if data.population == 'primary':
        teacher = make_teacher(seeds.teacher, config.model.d, **kwargs)
    else:
        from .controls import make_control
        teacher = make_control(data.population, seeds.teacher, config.model.d, **kwargs)
    phase_seeds = {p: derive_seed(seeds.data, f'exploratory/development/{p}-v1') for p in 'FGV'}
    counts = dict(zip('FGV', (data.n_fit, data.n_growth, data.n_validation)))
    return teacher, make_development(teacher, counts, phase_seeds, data.noise_sigma)
