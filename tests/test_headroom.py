"""Frozen-head controls use F geometry and a finite G-selected ridge grid."""
import numpy as np
import pytest

from test_train import training_case, cpu_training_evidence
from test_checkpoint import assert_nested_equal


def test_head_reference_recovers_linear_signal_with_duplicate_and_constant_columns():
    from treeneuron.research.headroom import solve_head_features
    x = np.asarray([-2., -1., 0., 1., 2.])
    f = np.column_stack((x, 2*x, np.ones(5)*19))
    gx = np.asarray([-3., .5, 4.])
    g = np.column_stack((gx, 2*gx, np.ones(3)*19))
    fit = solve_head_features(f, 7+3*x, g, 7+3*gx)
    assert fit.alpha == 0
    assert len(fit.records) == 6
    assert fit.active_columns.tolist() == [True, True, False]
    assert fit.feature_mean.tolist() == [0., 0., 19.]
    np.testing.assert_allclose(fit.predict_features(g), 7+3*gx, atol=1e-12, rtol=0)
    assert fit.records[0]['rank'] == 1
    changed_query = solve_head_features(f, 7+3*x, g*100, 7+3*gx)
    np.testing.assert_array_equal(changed_query.feature_mean, fit.feature_mean)
    np.testing.assert_array_equal(changed_query.feature_scale, fit.feature_scale)
    assert not fit.feature_mean.flags.writeable


def test_all_constant_features_choose_smallest_alpha_and_unpenalized_intercept():
    from treeneuron.research.headroom import solve_head_features
    fit = solve_head_features(np.ones((5, 3)), np.full(5, 2.5),
                              np.ones((2, 3))*100, np.full(2, 2.5))
    assert fit.alpha == 0
    assert not fit.active_columns.any()
    np.testing.assert_array_equal(fit.predict_features(np.zeros((4, 3))), np.full(4, 2.5))
    assert all(r['G_mse'] == 0 for r in fit.records)
    with pytest.raises(ValueError):
        solve_head_features(np.ones((5, 3)), np.ones(5), np.ones((2, 3)), np.ones(2), alphas=(0., -1.))


def test_positive_ridge_uses_mean_loss_and_G_selection():
    from treeneuron.research.headroom import solve_head_features
    f = np.asarray([[-1.], [1.]])
    fit = solve_head_features(f, np.asarray([5., 9.]), f, np.asarray([6., 8.]))
    assert fit.alpha == 1 and fit.intercept == 7
    np.testing.assert_allclose(fit.coefficients, [1.], atol=1e-14)
    repeated = solve_head_features(np.repeat(f, 3, axis=0), np.repeat([5., 9.], 3),
                                   f, np.asarray([6., 8.]))
    assert repeated.alpha == fit.alpha
    for a, b in zip(fit.records, repeated.records, strict=True):
        np.testing.assert_allclose(a['coefficients'], b['coefficients'], atol=1e-14)
        assert a['G_mse'] == pytest.approx(b['G_mse'], abs=1e-14)


def test_nonfinite_intermediate_geometry_fails_before_selection():
    from treeneuron.research.headroom import solve_head_features
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(FloatingPointError, match='geometry'):
            solve_head_features(np.asarray([[-1e300], [1e300]]), np.asarray([-1., 1.]),
                                np.asarray([[-1.], [1.]]), np.asarray([-1., 1.]))


def test_frozen_head_worker_preserves_parent_and_never_needs_V_targets(training_case):
    from treeneuron import train
    from treeneuron.research.roles import fitting_view
    from treeneuron.research.headroom import refit_frozen_head
    data, config, paths = training_case
    parent = train.fit_parent(data, config, paths)
    before, rng = train.pack_state(parent), train.capture_rng()
    model = refit_frozen_head(parent, fitting_view(data))
    prediction = model.predict(data.V.X)
    assert prediction.shape == data.V.y.shape and np.isfinite(prediction).all()
    assert len(model.head.records) == 6
    assert model.parent_identity_hashes == parent.identity_hashes
    assert_nested_equal(train.pack_state(parent), before)
    assert_nested_equal(train.capture_rng(), rng)
