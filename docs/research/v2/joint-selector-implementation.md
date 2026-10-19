# Exact joint selector implementation

Gate A passed after the complete M2 collection and independent raw arithmetic;
see [M2 findings](../../../reports/research-v2-M2.md). WP04 is now admitted for
its bounded numerical qualification and fresh-teacher study. It has no new
predictive result yet. The M3 common competent parent and continuation policy
must be fixed before fitting new teachers; the observed ordinary training
headroom remains an explicit design constraint.

The first implementation slice is a float64 array core in `joint_selector.py`.
`prepare_joint_problem(J0, r, blocks, config)` computes original column RMS masks
once per parent/block, normalizes retained columns once and builds the exact
parent-conditioned gate Gram/residual. `gain(ids)`, `objective(ids)`,
`marginal_gain(ids, candidate)` and `greedy(eligible_ids)` share those original
scales. They cannot read data roles, models or labels outside the supplied F
arrays. Full parent/gate analytic Jacobians, immutable CandidateBlock and the
Allocation adapter follow after this numerical core passes.

Use the binding spec's Schur objective for positive parent ridge, and one
orthogonal SVD parent basis for zero parent ridge. Report retained rank/cutoff,
solve residual and conditioning; no jitter, inverse or easier eligibility
fallback. Parent rank cutoff is1e-10 times its largest singular value. Gate
ridge is strictly positive in the estimator; the completely unpenalized
counterexample is an external least-squares reference, with its same actual
gate-Jacobian geometry also tested at zero-parent/.01-gate ridge. Pure span
when the parent spans all F rows has zero additional empirical tangent gain.

Conditional greedy uses cached block Cholesky extensions. Zero local blocks
stay in the common eligible pool and count toward the same requested/effective
count, even when every gain is zero. Stable IDs break ties at the declared
relative tolerance. Gate/contrast parameter cost is unchanged by dropped
numerical columns. Temporary solved updates remain diagnostics only; primary
selection will insert the original contrasts through the existing insertion API.

Dense acceptance references independently normalize the original columns and
solve an augmented least-squares system. They must not reuse the core's Schur
matrices, scales or objective to compute expected results. For zero parent
ridge they use the same declared retained parent span. Numerical sensitivity
pairs are algebra checks, not nine extra trained arms. No M3 fit is admitted
until all Jacobian/objective/eligibility/insertion checks, source binding and
measured resource qualification pass.

The array core has now passed29focused checks after15initial missing-module
RED failures. Independent augmented solves agree across72subset/objective
cases and108conditional marginals in nine numerical ridge settings. Maximum
absolute errors are4.45e-15for objective,2.45e-15for gain and2.67e-15for marginal
gain. The parent solve residual is at most1.17e-16. The one-row penalty-sharing
gain is1/6; zero-parent-ridge/full-row-span gain is zero. Tests also cover fixed
pre-residualization scales, rank-deficient parents, zero blocks/count retention,
duplicate-gate sharing and non-tied row/ID permutation.

These checks use small matrices, without scientific teachers or model fitting.
The full prediction-Jacobian builder and actual gate-Jacobian counterexample
are now verified below. CandidateBlock/trace/Allocation integration and native
workload validation remain pending. No complete selector or trained M3 outcome
follows. See
[the numerical receipt](WP04-joint-core-receipt.json).

`prediction_jacobians` in `research/jacobian.py` accepts the actual ungrown
SelectiveRegressor, CPU float64 model-coordinate X and unchanged augmented
contrasts. It includes every currently trainable parent parameter, in named
parameter order, and rejects other layers rather than conditioning on a partial
parent. Frozen parent parameters are omitted deliberately. Analytic gate blocks
match autograd on the actual zero-gate insertion, including biases and ReLU's
zero derivative at the boundary. Model mode, parameters, gradients and RNG are
preserved. The future state adapter must verify exact F identity and apply its
existing F-fitted transform once before calling this helper.

Twelve missing-module RED checks preceded implementation. An additional malformed
layer case then failed explicitly and received a strict layer-type guard. Final
Jacobian/core verification passed43checks in.31s. The actual-Jacobian stationary
parent construction ranks duplicate singletons above the complementary singleton:
unpenalized residual MSE.00113057 versus.00280908. The duplicate pair remains at
.00113057, while the complementary pair reaches2.37e-31. Parent stationarity is
accurate to1.03e-16. The zero-parent/.01-gate estimator chooses the complementary
pair; default positive-parent ridge is instead checked against its own dense
objective, with maximum error2.50e-17 across all subsets. These are algebraic
controls, without fitted teachers or efficacy evidence. The
[Jacobian receipt](WP04-joint-jacobian-receipt.json) retains every test failure and
the constructed reference.
