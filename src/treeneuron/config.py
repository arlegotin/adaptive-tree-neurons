"""Strict exploratory experiment schema and finite, canonical JSON identities.

Configuration describes an experiment; it never grants protected test access.
Use ``dataclasses.asdict(config)`` or ``config.to_dict()`` for ledger input.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from types import UnionType
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class ModelConfig:
    d: int = 16
    h: int = 64
    depth: int = 1
    tau: float = 1.0
    activation: Literal["relu"] = "relu"


@dataclass(frozen=True)
class TrainingConfig:
    # AdamW uses decoupled weight decay; SGD is the diagnostic alternate.
    optimizer: Literal["adamw", "sgd"] = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    sgd_momentum: float = 0.0
    batch_size: int = 128
    # Pilot outputs, deliberately not invented stopping allowances.
    base_updates: int | None = None
    continuation_updates: int | None = None
    validation_interval: int | None = None
    patience: int | None = None
    scheduler: Literal["constant", "cosine"] = "constant"
    min_learning_rate: float = 0.0
    checkpoint_selection: Literal["strict_improvement_earliest"] = "strict_improvement_earliest"
    evaluate_at_update_zero: Literal[True] = True


@dataclass(frozen=True)
class GrowthConfig:
    m_requested: int = 16
    epsilon: float = 0.05
    policy: Literal["adam", "sgd", "random", "keep"] = "adam"
    direction: Literal["eigenvector", "gaussian"] = "eigenvector"
    block_units: int = 8
    projection_rms_floor: float = 1e-12
    retain_zero_growth: Literal[True] = True


@dataclass(frozen=True)
class DataConfig:
    dataset: Literal["synthetic"] = "synthetic"
    population: Literal[
        "primary", "globally_affine", "additive_sine_quadratic", "pure_noise",
        "random_label_permutation", "conditional_slope_2d", "homogeneous_smooth",
    ] = "primary"
    noise_sigma: float = 0.1
    components: int = 4
    reference_rows: int = 65536
    gate_prevalences: tuple[float, ...] = (0.1, 0.5)
    gate_types: tuple[Literal["hard", "smooth_0.2", "smooth_1.0"], ...] = (
        "hard", "smooth_0.2", "smooth_1.0")
    regime: Literal["isotropic", "axis_aligned", "rotated", "rare"] = "isotropic"
    nuisance_dimensions: int = 0
    sample_density: float = 2.0
    row_rounding: Literal["ceiling"] = "ceiling"
    n_fit: int = 2306
    n_growth: int = 577
    n_validation: int = 577
    n_test: int | None = None
    standardize_inputs: Literal[True] = True
    standardize_targets: Literal[True] = True
    fit_transforms_on: Literal["F"] = "F"
    refit_full_development: Literal[False] = False


@dataclass(frozen=True)
class SeedsConfig:
    namespace: Literal["exploratory"] = "exploratory"
    teacher: int = 0
    data: int = 1
    optimizer: int = 2
    allocation: int = 3
    contrast: int = 4
    reference: int = 5


@dataclass(frozen=True)
class BackendConfig:
    device: Literal["cpu", "mps"] = "cpu"
    training_dtype: Literal["float32"] = "float32"
    execution: Literal["eager"] = "eager"
    score_device: Literal["cpu"] = "cpu"
    score_dtype: Literal["float64"] = "float64"
    allow_fallback: Literal[False] = False
    cpu_threads: int | None = None
    interop_threads: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    study_id: str
    phase: Literal["exploratory"] = "exploratory"
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    growth: GrowthConfig = GrowthConfig()
    data: DataConfig = DataConfig()
    seeds: SeedsConfig = SeedsConfig()
    backend: BackendConfig = BackendConfig()

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible resolved configuration."""
        return json.loads(canonical_json(asdict(self)))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON mapping keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError(f"Expected finite JSON value, got {type(value).__name__}")


def canonical_json(value: Mapping[str, Any]) -> str:
    """Sorted compact UTF-8 JSON; reject coercion, unsupported types and NaN."""
    if not isinstance(value, Mapping):
        raise ValueError("Expected a mapping")
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def canonical_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _typed(value: Any, annotation: Any, path: str) -> Any:
    origin, args = get_origin(annotation), get_args(annotation)
    if origin is UnionType:
        if value is None and type(None) in args:
            return None
        return _typed(value, next(arg for arg in args if arg is not type(None)), path)
    if origin is Literal:
        if not any(type(value) is type(choice) and value == choice for choice in args):
            raise ValueError(f"{path}: expected one of {args!r}")
        return value
    if origin is tuple:
        if not isinstance(value, (tuple, list)) or not value:
            raise ValueError(f"{path}: expected a nonempty sequence")
        return tuple(_typed(item, args[0], path) for item in value)
    if annotation is float:
        if type(value) not in (int, float):
            raise ValueError(f"{path}: expected a finite number")
        try:
            number = float(value)
        except OverflowError as error:
            raise ValueError(f"{path}: number is outside float64 range") from error
        if not math.isfinite(number):
            raise ValueError(f"{path}: expected a finite number")
        return number
    if annotation in (int, str, bool):
        if type(value) is not annotation:
            raise ValueError(f"{path}: expected {annotation.__name__}")
        return value
    return _section(annotation, value, path)


def _section(cls: type, value: Any, path: str) -> Any:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{path}: expected a string-keyed mapping")
    names = {field.name for field in fields(cls)}
    unknown = set(value) - names
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    hints = get_type_hints(cls)
    try:
        return cls(**{key: _typed(item, hints[key], f"{path}.{key}")
                      for key, item in value.items()})
    except TypeError as error:
        raise ValueError(f"{path}: missing required fields") from error


def resolve_experiment(value: Mapping[str, Any]) -> ExperimentConfig:
    """Validate and resolve a mapping; protected evaluation needs a later API."""
    canonical_json(value)  # reject invalid values even before schema validation
    if value.get("phase", "exploratory") != "exploratory":
        raise ValueError("Core supports exploratory phase only; protected protocol capability required")
    config = _section(ExperimentConfig, value, "experiment")
    if not config.study_id.strip():
        raise ValueError("study_id must be nonempty")
    # Recompute row defaults from the requested width/dimension and density.
    data = dict(value.get("data", {}))
    if config.data.population == "conditional_slope_2d":
        data.setdefault("noise_sigma", 0.0)
        if data["noise_sigma"] != 0:
            raise ValueError("conditional_slope_2d is the noise-free control; noise_sigma must be zero")
    if "n_fit" not in data:
        p0 = (config.model.d + 2) * config.model.h + 1
        try:
            fit_rows = config.data.sample_density * p0
        except OverflowError as error:
            raise ValueError("derived n_fit is outside float64 range") from error
        if not math.isfinite(fit_rows):
            raise ValueError("derived n_fit must be finite")
        data["n_fit"] = math.ceil(fit_rows)
    data.setdefault("n_growth", (data["n_fit"] + 3) // 4)
    data.setdefault("n_validation", (data["n_fit"] + 3) // 4)
    config = ExperimentConfig(**{**{f.name: getattr(config, f.name) for f in fields(config)},
                                 "data": _section(DataConfig, data, "experiment.data")})
    positive = {
        "model.d": config.model.d, "model.h": config.model.h,
        "model.depth": config.model.depth, "model.tau": config.model.tau,
        "training.learning_rate": config.training.learning_rate,
        "training.adam_epsilon": config.training.adam_epsilon,
        "training.batch_size": config.training.batch_size,
        "growth.epsilon": config.growth.epsilon, "growth.block_units": config.growth.block_units,
        "growth.projection_rms_floor": config.growth.projection_rms_floor,
        "data.components": config.data.components, "data.reference_rows": config.data.reference_rows,
        "data.sample_density": config.data.sample_density,
        "data.n_fit": config.data.n_fit, "data.n_growth": config.data.n_growth,
        "data.n_validation": config.data.n_validation,
    }
    for name in ("base_updates", "continuation_updates", "validation_interval", "patience"):
        number = getattr(config.training, name)
        if number is not None:
            positive[f"training.{name}"] = number
    for name in ("cpu_threads", "interop_threads"):
        number = getattr(config.backend, name)
        if number is not None:
            positive[f"backend.{name}"] = number
    if config.data.n_test is not None:
        positive["data.n_test"] = config.data.n_test
    for name, number in positive.items():
        if number <= 0:
            raise ValueError(f"{name} must be positive")
    nonnegative = {
        "growth.m_requested": config.growth.m_requested,
        "data.noise_sigma": config.data.noise_sigma,
        "data.nuisance_dimensions": config.data.nuisance_dimensions,
        "training.weight_decay": config.training.weight_decay,
        "training.min_learning_rate": config.training.min_learning_rate,
        **{f"seeds.{name}": getattr(config.seeds, name)
           for name in ("teacher", "data", "optimizer", "allocation", "contrast", "reference")},
    }
    for name, number in nonnegative.items():
        if number < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("adam_beta1", "adam_beta2", "sgd_momentum"):
        if not 0 <= getattr(config.training, name) < 1:
            raise ValueError(f"training.{name} must be in [0, 1)")
    if any(not 0 < p < 1 for p in config.data.gate_prevalences):
        raise ValueError("data.gate_prevalences must be in (0, 1)")
    if config.training.min_learning_rate > config.training.learning_rate:
        raise ValueError("min_learning_rate exceeds learning_rate")
    if config.data.nuisance_dimensions >= config.model.d:
        raise ValueError("nuisance_dimensions must be less than model.d")
    if config.data.population == "conditional_slope_2d" and config.model.d != 2:
        raise ValueError("conditional_slope_2d requires model.d=2")
    return config


class _UniqueLoader(yaml.SafeLoader):
    pass


def _unique_mapping(loader: _UniqueLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if type(key) is not str:
            raise ValueError("YAML mapping keys must be strings")
        if key in result:
            raise ValueError(f"duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_experiment(path: Path) -> ExperimentConfig:
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueLoader)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    return resolve_experiment(value)
