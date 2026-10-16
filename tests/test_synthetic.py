"""Independent numerical fixtures catch gate, normalization and stream mistakes."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.special import expit

from treeneuron.data.synthetic import (
    make_teacher, sample_teacher, make_development, evaluate_teacher,
    reference_inputs, make_from_config,
)
from treeneuron.data.controls import (
    make_control, sample_control, control_target, oracle_diagnostic_features,
)
from treeneuron.config import resolve_experiment


def test_primary_vectors_components_and_threshold_orientation():
    teacher = make_teacher(71)
    assert teacher.d == 16
    assert len(teacher.u) == len(teacher.v) == len(teacher.w) == 4
    vectors = np.asarray((teacher.u0, *teacher.u, *teacher.v, *teacher.w))
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-14)
    assert np.linalg.matrix_rank(vectors) == 13
    assert set(teacher.prevalences) <= {0.1, 0.5}
    assert teacher.gate_type_probabilities == (1 / 3, 1 / 3, 1 / 3)
    for p, t in zip(teacher.prevalences, teacher.thresholds):
        assert t == pytest.approx(1.2815515655446004 if p == 0.1 else 0.0)
    # All components share controlled axes; arithmetic is hand-computable.
    e0 = (1.0,) + (0.0,) * 15
    fixture = replace(teacher, u0=e0, u=(e0,) * 4, v=(e0,) * 4, w=(e0,) * 4,
                      gate_types=('hard', 'smooth_0.2', 'smooth_1.0', 'hard'),
                      scales=(0.0, 0.2, 1.0, 0.0), thresholds=(0.0,) * 4,
                      reference_mean=0.0, reference_std=1.0)
    X = np.zeros((3, 16)); X[:, 0] = [-1, 0, 1]
    f, gates = evaluate_teacher(fixture, X, return_gates=True)
    np.testing.assert_allclose(gates[:, 0], [0, 1, 1])
    np.testing.assert_allclose(gates[:, 1], expit(np.array([-5, 0, 5])))
    np.testing.assert_allclose(gates[:, 2], expit(np.array([-1, 0, 1])))
    expected_positive = .3 * np.sin(1) + .5 * (1.2 + 2 + np.tanh(2.5) + np.tanh(.5))
    expected_negative = -.3 * np.sin(1) + .5 * (-1.2 + 2 + np.tanh(2.5) + np.tanh(.5))
    np.testing.assert_allclose(f, [expected_negative, 0, expected_positive])


def test_reference_has_exact_size_separate_stream_and_stored_population_normalization():
    teacher = make_teacher(4, reference_seed=19)
    reference = reference_inputs(teacher)
    assert reference.shape == (65536, 16)
    normalized = evaluate_teacher(teacher, reference)
    assert normalized.mean() == pytest.approx(0, abs=1e-14)
    assert normalized.std(ddof=0) == pytest.approx(1, abs=1e-14)
    rows = sample_teacher(teacher, 100, 19, .1)
    assert rows.stream_id != teacher.reference_stream_id
    assert not np.array_equal(reference[:100], rows.X)
    alternate = make_teacher(4, reference_seed=20)
    assert teacher.u == alternate.u
    assert teacher.reference_mean != alternate.reference_mean
    np.testing.assert_array_equal(sample_teacher(alternate, 100, 19, .1).X, rows.X)


def test_teacher_and_observation_reproducibility_and_independent_noise():
    teacher = make_teacher(71)
    assert teacher.to_dict() == make_teacher(71).to_dict()
    assert teacher.teacher_id != make_teacher(72).teacher_id
    a = sample_teacher(teacher, 20, 72, .1)
    b = sample_teacher(teacher, 20, 72, 0)
    c = sample_teacher(teacher, 20, 73, .1)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.row_id, b.row_id)
    np.testing.assert_allclose(b.y, a.noise_free_target)
    assert not np.array_equal(a.y, b.y)
    assert not np.array_equal(a.X, c.X)
    assert not set(a.row_id) & set(c.row_id)
    assert a.row_id.dtype == a.group_id.dtype == np.int64
    assert not a.X.flags.writeable
    assert not a.row_id.flags.writeable


def test_development_defaults_distinct_rows_and_manifest():
    data = make_development(make_teacher(3))
    assert tuple(len(getattr(data, p).y) for p in 'FGV') == (2306, 577, 577)
    assert len(set(np.concatenate([getattr(data, p).row_id for p in 'FGV']))) == 3460
    assert data.manifest.teacher_id == data.teacher_id
    assert len(data.manifest.row_id) == 3460
    assert len(data.manifest.assignment_hash) == 64
    assert data.manifest.assignments.count('F') == 2306
    assert isinstance(data.manifest.row_id, tuple)


def test_gate_distributions_are_componentwise_independent():
    teachers = [make_teacher(seed) for seed in range(24)]
    # Catches one shared type/prevalence drawn for all components.
    assert any(len(set(t.gate_types)) > 1 for t in teachers)
    assert any(len(set(t.prevalences)) > 1 for t in teachers)
    assert {g for t in teachers for g in t.gate_types} == {'hard', 'smooth_0.2', 'smooth_1.0'}
    assert {p for t in teachers for p in t.prevalences} == {.1, .5}


def test_config_bridge_honors_counts_reference_gates_and_data_seed():
    config = resolve_experiment({'study_id': 'demo', 'data': {
        'n_fit': 5, 'n_growth': 3, 'n_validation': 2,
        'gate_prevalences': [.1], 'gate_types': ['hard'],
    }, 'seeds': {'teacher': 7, 'reference': 19, 'data': 31}})
    teacher, data = make_from_config(config)
    assert teacher.reference_seed == 19
    assert teacher.prevalences == (.1,) * 4
    assert teacher.gate_types == ('hard',) * 4
    assert tuple(len(getattr(data, p).y) for p in 'FGV') == (5, 3, 2)
    other = replace(config, seeds=replace(config.seeds, data=32))
    teacher2, data2 = make_from_config(other)
    assert teacher2.teacher_id == teacher.teacher_id
    assert not np.array_equal(data.F.X, data2.F.X)


@pytest.mark.parametrize('changes', [{'reference_rows': 50}, {'components': 3}, {'n_test': 10}])
def test_bridge_rejects_unsupported_or_protected_requests(changes):
    config = resolve_experiment({'study_id': 'demo', 'data': changes})
    with pytest.raises(ValueError):
        make_from_config(config)


@pytest.mark.parametrize('regime', ['axis_aligned', 'rotated', 'rare'])
def test_config_exploratory_regimes_and_nuisance_are_real_coordinates(regime):
    config = resolve_experiment({'study_id': 'demo', 'data': {
        'regime': regime, 'nuisance_dimensions': 3, 'n_fit': 5,
    }})
    teacher, rows = make_from_config(config)
    assert teacher.regime == regime
    assert teacher.nuisance_dimensions == 3
    X = rows.F.X.copy(); X[:, -3:] += 1000
    np.testing.assert_array_equal(evaluate_teacher(teacher, rows.F.X), evaluate_teacher(teacher, X))
    if regime == 'rare':
        assert teacher.prevalences == (.01,) * 4
    if regime == 'axis_aligned':
        assert all(np.count_nonzero(v) == 1 for v in (teacher.u0, *teacher.u, *teacher.v, *teacher.w))


def test_original_2d_control_distribution_equation_and_no_noise():
    control = make_control('conditional_slope_2d', 5, 2)
    X = np.array([[-.5, .2], [.5, .2], [0, .2]])
    np.testing.assert_allclose(control_target(control, X), [.2, .8, .8])
    rows = sample_control(control, 5000, 7)
    assert (-1 <= rows.X[:, 0]).all() and (rows.X[:, 0] < 1).all()
    assert (0 <= rows.X[:, 1]).all() and (rows.X[:, 1] < 1).all()
    assert abs(np.corrcoef(rows.X.T)[0, 1]) < .05
    np.testing.assert_array_equal(rows.y, rows.noise_free_target)
    with pytest.raises(ValueError, match='noise'):
        sample_control(control, 5, 7, sigma=.1)


@pytest.mark.parametrize('kind', ['globally_affine', 'additive_sine_quadratic', 'pure_noise',
                                  'homogeneous_smooth', 'random_label_permutation'])
def test_controls_are_versioned_serialized_repeatable_and_sampleable(kind):
    control = make_control(kind, 17, 16)
    assert control.version and control.to_dict()['kind'] == kind
    a = sample_control(control, 40, 18, .1)
    b = sample_control(control, 40, 18, .1)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.X.shape == (40, 16)
    assert np.isfinite(a.y).all()
    assert len(set(a.row_id)) == 40


def test_explicit_control_equations_and_permutation_decoupling():
    e0 = (1.0,) + (0.0,) * 15
    affine = replace(make_control('globally_affine', 3, 16), vectors=(e0,))
    additive = replace(make_control('additive_sine_quadratic', 3, 16), vectors=(e0, e0))
    X = np.zeros((2, 16)); X[:, 0] = [0, 1]
    np.testing.assert_allclose(control_target(affine, X), [.3, 1.3])
    np.testing.assert_allclose(control_target(additive, X), [-.3, .3*np.sin(1)])
    noise = sample_control(make_control('pure_noise', 3, 16), 40, 4, .7)
    np.testing.assert_array_equal(noise.noise_free_target, np.zeros(40))
    assert noise.y.std() > .4
    shuffled = make_control('random_label_permutation', 3, 16)
    original = sample_teacher(shuffled.base_teacher, 40, 4, .1)
    permuted = sample_control(shuffled, 40, 4, .1)
    np.testing.assert_array_equal(permuted.X, original.X)
    np.testing.assert_array_equal(np.sort(permuted.y), np.sort(original.y))
    assert not np.array_equal(permuted.y, original.y)
    assert permuted.noise_free_target is None


def test_homogeneous_smooth_equation_is_declared():
    e0 = (1.0,) + (0.0,) * 15
    control = replace(make_control('homogeneous_smooth', 3, 16), vectors=(e0,) * 4)
    X = np.zeros((2, 16)); X[:, 0] = [0, 1]
    np.testing.assert_allclose(control_target(control, X), [0, 2*np.sin(1)])


def test_oracle_features_are_explicitly_information_advantaged():
    teacher = make_teacher(9)
    rows = sample_teacher(teacher, 5, 10, .1)
    oracle = oracle_diagnostic_features(teacher, rows.X)
    assert oracle.information_advantaged is True
    assert oracle.purpose == 'oracle_gate_least_squares_diagnostic'
    assert oracle.X.shape == (5, 1 + 16 + 4*16)
    assert len(oracle.feature_names) == oracle.X.shape[1]
    assert rows.X.shape == (5, 16)


@pytest.mark.parametrize('args', [dict(seed=-1), dict(seed=True), dict(seed=1, d=0),
                                dict(seed=1, reference_rows=65535)])
def test_bad_teacher_parameters_fail(args):
    with pytest.raises(ValueError):
        make_teacher(**args)


def test_invalid_control_and_observation_requests_fail():
    with pytest.raises(ValueError):
        make_control('typo', 1, 16)
    with pytest.raises(ValueError):
        make_control('conditional_slope_2d', 1, 16)
    t = make_teacher(1)
    for n, seed, sigma in [(-1, 2, .1), (1, -2, .1), (1, 2, float('nan')), (1, 2, -.1)]:
        with pytest.raises(ValueError):
            sample_teacher(t, n, seed, sigma)


def test_serialization_roundtrips_exact_teacher_identity_and_controls():
    import json
    from treeneuron.data.synthetic import TeacherSpec
    from treeneuron.data.controls import ControlSpec
    teacher = make_teacher(91, regime='rotated', nuisance_dimensions=2)
    loaded = TeacherSpec.from_dict(json.loads(json.dumps(teacher.to_dict())))
    assert teacher.teacher_id == loaded.teacher_id
    X = sample_teacher(teacher, 9, 8, 0).X
    np.testing.assert_array_equal(evaluate_teacher(teacher, X), evaluate_teacher(loaded, X))
    for kind in ['globally_affine', 'additive_sine_quadratic', 'pure_noise',
                 'random_label_permutation', 'conditional_slope_2d', 'homogeneous_smooth']:
        control = make_control(kind, 91, 2 if kind == 'conditional_slope_2d' else 16)
        copy = ControlSpec.from_dict(json.loads(json.dumps(control.to_dict())))
        assert copy.teacher_id == control.teacher_id
        np.testing.assert_array_equal(sample_control(copy, 9, 8).y, sample_control(control, 9, 8).y)


@pytest.mark.parametrize('change', [dict(d=0), dict(reference_std=0), dict(phase='confirmatory'),
                                   dict(version='unrecognized'), dict(gate_types=('bad',)*4)])
def test_malformed_teacher_specs_fail_before_sampling(change):
    with pytest.raises(ValueError):
        replace(make_teacher(1), **change)


@pytest.mark.parametrize('change', [dict(kind='unknown'), dict(d=0), dict(version='unknown'),
                                   dict(vectors=()), dict(phase='confirmatory')])
def test_malformed_control_specs_fail_before_sampling(change):
    with pytest.raises(ValueError):
        replace(make_control('globally_affine', 1, 16), **change)


def test_nonprimary_config_gates_do_not_silently_disappear():
    config = resolve_experiment({'study_id': 'demo', 'data': {
        'population': 'globally_affine', 'gate_types': ['hard'],
    }})
    with pytest.raises(ValueError, match='gate'):
        make_from_config(config)


def test_rare_prevalence_override_cannot_silently_disappear():
    with pytest.raises(ValueError, match='prevalence'):
        make_teacher(1, regime='rare', gate_prevalences=(.2,))


def test_named_primary_sensitivities_and_seed_domains():
    from treeneuron.data.synthetic import derive_seed
    domains = ['teacher/v1', 'reference/gaussian-v1', 'observation/inputs-v1',
               'observation/noise-v1', 'development/F-v1', 'development/G-v1', 'development/V-v1']
    assert len({derive_seed(1, 'exploratory/' + domain) for domain in domains}) == len(domains)
    with pytest.raises(ValueError):
        derive_seed(1, 'confirmatory/observation/inputs-v1')
    for kind in ['rare', 'rotated', 'axis_aligned', 'nuisance']:
        population = make_control(kind, 7, 16)
        assert sample_control(population, 5, 8).X.shape == (5, 16)
    assert make_control('nuisance', 7, 16).nuisance_dimensions == 4


@pytest.mark.parametrize('population,d', [('conditional_slope_2d', 2), ('random_label_permutation', 16),
                                        ('homogeneous_smooth', 16), ('globally_affine', 16)])
def test_control_config_bridge_is_repeatable_and_uses_requested_reference(population, d):
    config = resolve_experiment({'study_id': 'demo', 'model': {'d': d}, 'data': {
        'population': population, 'n_fit': 5, 'n_growth': 3, 'n_validation': 2,
    }, 'seeds': {'reference': 43}})
    population1, data1 = make_from_config(config)
    population2, data2 = make_from_config(config)
    assert population1.reference_seed == population2.reference_seed == 43
    assert population1.teacher_id == population2.teacher_id
    np.testing.assert_array_equal(data1.F.y, data2.F.y)
    if population == 'conditional_slope_2d':
        np.testing.assert_array_equal(data1.F.y, data1.F.noise_free_target)
