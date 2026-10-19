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
arrays. Full parent/gate analytic Jacobians and the immutable candidate/native
Allocation integration are now verified on small actual models below.

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
are now verified below. Temporary solved-update diagnostics also pass the checks
below. Full native workload validation and the M3 runner remain pending. The
common M3 development recipe is now fixed in
[its decision record](M3-development-recipe.md); no trained M3 outcome follows. See
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

`research/joint_allocation.py` connects these components through explicit APIs:
`build_gate_blocks(parent,F,scored)`,
`joint_ridge_select(parent,F,candidates,config,scored=scored)` and
`selection_to_allocation(parent,F,scored,trace,seed=seed)`. The input `scored` is
the existing parent-bound `ScoredParent`, rather than a bare table whose parent
cannot be authenticated. Verify exact original F arrays/rows/groups and exclude
evaluator targets; apply the original F-fitted transforms once. Every nominal ID
remains recorded, with precisely the legacy allocation guard shared across arms.
Candidate contrasts use immutable bytes-backed float64 arrays. Selection
deep-copies the parent into float64 and changes no parent state or RNG.

The immutable serialized trace includes original contrasts, all conditional
scores/ties, solve residuals/conditioning, fixed RMS/drop masks, parent-span rank
and novelty fractions, data/source/configuration bindings, objectives and timing
scope. Known ineligible nonfinite spectra use explicit IEEE spelling strings,
with the original tensor-byte table hash retained. No scalar is imputed.
Conversion checks the parent/table/contrast/count identities and complete
conditional winner/gain chain. Native insertion accepts the distinct
`joint_ridge` policy label while retaining its dtype, optimizer, rollback and
parameter guards. It still casts the unchanged float64 contrasts into the
parent's native float32 representation, explicitly recorded by the trace.

Eight missing-module RED tests preceded implementation;51combined checks then
passed. Two further failures pinned nonfinite diagnostic serialization and the
final identity-recheck timer; three more pinned inconsistent conditional traces.
The affected growth/optimizer/Adam/state run passed192checks with one unavailable
device skip in54.56seconds, and final adapter/core/Jacobian checks passed56in7.68seconds.
These are software/geometry checks, not evidence of predictive improvement.
See [the integration receipt](WP04-joint-allocation-receipt.json).

`diagnose_local_updates(parent,F,G,scored,trace)` in `research/local_updates.py`
uses a direct augmented least-squares solve with the original RMS masks/scales.
It matches the selected regularized objective and rejects unexpected rank
truncation. The zero-parent-ridge version uses the same retained parent span and
the minimum normalized-coordinate-norm parent update on that span. It records
the full solved parent/gate vectors, singular values, rank/cutoff and objective
agreement. G cannot fit the solution or select a fraction.

All fixed fractions1/16,1/4,1 are applied to separate float64 copies, with original
contrasts, alongside predicted and actual F/G losses, ReLU-mask changes and gate
saturation. These are float64 nonlinear diagnostics, distinct from native
float32 training. Penalized gain is relative to an optimized local parent;
current-parent raw MSE and nonlinear changes are reported separately. Parameters,
optimizer, sampler and caller RNG remain unchanged. Four initial missing-module
RED tests and three missing rank/vector-record failures preceded the final
60combined checks in11.42seconds. See
[the diagnostic receipt](WP04-local-joint-update-receipt.json).

The full-size engineering operator exposed a publication integration gap before
any M3 experiment: its initial configuration used the wrong legacy field name,
and the corrected run reached the local solve but could not publish a NumPy
cutoff/integer-key diagnostic through the strict canonical writer. Both failed
attempts and costs remain retained. Explicit scalar/key serialization now passes
61 affected checks after a reproduced failing publication test. The numerical
objective and training treatment are unchanged. See the
[correction receipt](WP04-diagnostic-publication-fix-receipt.json). A fresh full-size
run is still required; neither failure establishes a predictive outcome.

The corrected h64/d16/F2306/count16 numerical workload now passes, including
full-parent conditioning, exact selected-set dense reference, fixed-fraction
nonlinear diagnostics, native insertion and closed checkpoint reload. Its
contended supervised elapsed time was7.097seconds, observed root RSS821346304bytes
(about.765GiB); this includes a30-update linear-plus-sine fixture, not a full
scientific fit or an aggregate machine peak. Independently regenerated F/G
inputs, manually assembled full Jacobian/dense ridge and restored native
predictions/moments agree. See the
[full-workload receipt](WP04-full-selector-workload-receipt.json). No primary
teacher was generated; Gate B and predictive usefulness remain unresolved.

The matched local-singleton ridge control and conditional joint placement now
enter the same native diagnostic continuation through a checked `ridge` fork.
The singleton control ranks independent gains but reports the actual selected
set objective; it never adds singleton scores and calls that a joint gain.
Traces retain the explicit strategy, common eligibility/count and original
contrasts. Native allocation metadata distinguishes `local_ridge` and
`joint_ridge`. Fixed-budget forks preserve paired streams and parent/RNG state.
Final verification passed195checks with1unavailable-device skip. The
[integration receipt](WP04-joint-fork-integration-receipt.json) records the failing
feature checks and recoveries; there is still no M3 predictive result.

The [small-oracle prediction check](../../../reports/research-v2-small-joint.md)
now passes its narrowly defined development screen, with mixed correlations and
identical selected pairs across new variants. The prospective M3 study expands
to160initial final fits,1040nominal probes and16shared parent procedures; one
conditional eight-teacher tranche is predeclared. Its CLI plan and exact shared
seeds are implemented, while public run rejects before data creation until
production workers/resource admission are ready. See the
[plan receipt](WP04-M3-plan-implementation-receipt.json). This is no M3 outcome.
