"""Contracts catch silent coercion, omitted defaults and unauthorized phases."""
from dataclasses import asdict
import math

import pytest
import yaml


def load(tmp_path, value):
    from treeneuron.config import load_experiment

    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(value))
    return load_experiment(path)


def test_resolved_defaults_and_independent_seed_namespaces(tmp_path):
    config = load(tmp_path, {"study_id": "pilot"})
    assert config.phase == "exploratory"
    assert (config.model.d, config.model.h, config.model.tau) == (16, 64, 1.0)
    assert config.growth.m_requested == 16
    assert config.growth.epsilon == 0.05
    assert config.training.adam_epsilon == 1e-8
    assert config.training.base_updates is None
    assert (config.data.n_fit, config.data.n_growth, config.data.n_validation) == (2306, 577, 577)
    assert config.data.n_test is None
    assert config.seeds.namespace == "exploratory"
    assert len({config.seeds.teacher, config.seeds.data, config.seeds.optimizer,
                config.seeds.allocation, config.seeds.contrast, config.seeds.reference}) == 6
    assert config.backend.training_dtype == "float32"
    assert config.backend.score_device == "cpu"
    assert config.backend.score_dtype == "float64"
    assert config.backend.allow_fallback is False
    assert asdict(config)["model"]["tau"] == 1.0


@pytest.mark.parametrize("value", [
    {"typo": 1}, {"model": {"typo": 1}}, {"training": {"typo": 1}},
    {"growth": {"typo": 1}}, {"data": {"typo": 1}},
    {"seeds": {"typo": 1}}, {"backend": {"typo": 1}},
    {"model": {"h": True}}, {"model": {"d": "16"}},
    {"training": {"batch_size": 2.5}}, {"backend": {"allow_fallback": "false"}},
    {"model": []}, {"seeds": {"data": -1}}, {"data": {"n_fit": 0}},
    {"growth": {"m_requested": -1}}, {"training": {"adam_epsilon": 0}},
    {"model": {"tau": 0}}, {"model": {"tau": -1}},
    {"model": {"tau": math.inf}}, {"growth": {"epsilon": math.nan}},
    {"backend": {"allow_fallback": True}}, {"backend": {"training_dtype": "float16"}},
    {"data": {"refit_full_development": True}}, {"seeds": {"namespace": "confirmatory"}},
])
def test_rejects_unknown_keys_wrong_types_and_invalid_values(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, {"study_id": "pilot", **value})


@pytest.mark.parametrize("phase", ["confirmatory", "frozen", "test"])
def test_protected_phase_rejected_by_core_runner(tmp_path, phase):
    with pytest.raises(ValueError, match="protected|exploratory|capability"):
        load(tmp_path, {"study_id": "pilot", "phase": phase})


def test_yaml_duplicate_keys_rejected(tmp_path):
    from treeneuron.config import load_experiment

    path = tmp_path / "duplicate.yaml"
    path.write_text("study_id: pilot\nmodel:\n  tau: 1\n  tau: 2\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_experiment(path)


@pytest.mark.parametrize("value", [[], None, {"study_id": ""}])
def test_requires_nonempty_study_and_mapping(tmp_path, value):
    with pytest.raises(ValueError):
        load(tmp_path, value)


def test_explicit_exploratory_controls_and_budgets_survive_resolution(tmp_path):
    config = load(tmp_path, {
        "study_id": "control", "model": {"d": 2, "h": 32, "depth": 2},
        "data": {"population": "conditional_slope_2d", "noise_sigma": 0,
                 "n_fit": 101, "n_growth": 26, "n_validation": 26, "n_test": 33},
        "training": {"optimizer": "sgd", "base_updates": 20,
                     "continuation_updates": 10, "validation_interval": 2, "patience": 3},
        "growth": {"policy": "sgd", "direction": "gaussian", "m_requested": 0},
    })
    assert config.model.depth == 2
    assert config.data.population == "conditional_slope_2d"
    assert config.training.base_updates == 20
    assert config.growth.m_requested == 0


def test_hash_is_order_independent_and_seed_sensitive():
    from treeneuron.config import canonical_hash

    assert canonical_hash({"seed": 1, "tau": 1.0}) == canonical_hash({"tau": 1.0, "seed": 1})
    assert canonical_hash({"seed": 1}) != canonical_hash({"seed": 2})
    assert canonical_hash({"training": {"learning_rate": 0.001}}) != canonical_hash({"training": {"learning_rate": 0.002}})
    assert canonical_hash({}) == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"


@pytest.mark.parametrize("value", [{"x": math.nan}, {"x": [math.inf]}, {1: "bad"}, {"x": {"a": object()}}])
def test_hash_rejects_non_json_or_nonfinite_values(value):
    from treeneuron.config import canonical_hash

    with pytest.raises(ValueError):
        canonical_hash(value)


def test_out_of_range_numeric_input_is_a_validation_error(tmp_path):
    with pytest.raises(ValueError):
        load(tmp_path, {"study_id": "pilot", "model": {"tau": 10**400}})
