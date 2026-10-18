# P13 simple-reference adapter contract

Proposed for root review before implementation, 18 October 2026. The isolated
branch is `research-v2-p13-simple`, based on `4cc937d`. Binding requirements are
research-v2 design sections “Transfer data and split contract” and “Mandatory
native procedures, search and resources”. This slice qualifies tiny engineering
fixtures and existing-reference parity; it does not run prepared transfer data,
HPO, full practical comparisons, or Gate D.

## Owned boundary and interface

Create only `src/treeneuron/research/p13_simple.py`, P13-specific tests, and this
contract/its later receipt. Read shared code without changing it. Core27 identities,
locks, shared registries, legacy spline/NODE guards, and active scientific source
remain unchanged. No dependency installation or other agents are needed.

Proposed interfaces are `p13_configurations(feature_schema)` (29 ordered JSON
configs), `P13Predictor().fit(FitRequest)`, feature-only `predict(X, batch_size=...)`,
`save(directory)`, `load(directory, expected_manifest_sha256=...)`, and
`resource_summary()`. `request.config` carries a strict
`research/v2-p13-config` with `family='P13'`, estimator `mean`/`ridge`/`spline`, an
explicit ordered `{columns, categorical}` feature schema, and `parameters`.
Unknown fields, unsupported values and coercive types fail. Dataset semantic
schemas and reserved role fields own target/proxy exclusions; the generic adapter
preserves every explicitly declared feature name and never infers/drops columns. Mean parameters are empty; ridge parameters contain alpha only; spline
parameters contain alpha and n_knots only.

The fixed enumeration is mean first, ridge by ascending alpha, then spline by
ascending knot count and alpha. Alphas are exactly
`[1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, 1e6]`; spline n_knots are exactly `[4, 8, 16]`.
Thus mean1 + ridge7 + spline21 =29 distinct declared configs, with no filler or
replacement when a constant feature collapses realized knots. The adapter
performs one closed-form fit, never selection among configs. F is the only fit
role: neither G features nor G labels are inspected. V/E/T and model selection
belong to the external coordinator. Seed is retained but does not choose a fit.

## F-only geometry: explicit builder choices

Keep the useful numerical conventions of `baselines/simple.py` and
`data/transforms.py`. Continuous numeric columns use F median imputation,
population mean/std after imputation, unit scale for constants, and zero fill for
all-missing F columns. Add an unscaled missingness indicator only when that
column was missing in F. Later missing values in an F-complete column use its
saved F median without creating a new feature. Explicitly numeric columns accept
real integer/float values or missing values, rejecting booleans, complex values,
nonnumeric strings and infinities. Targets are finite real float64 vectors,
F-centered/scaled with population std and unit scale for constants.

Categories are declared by schema, never inferred from G or query batches. Use
sorted F-only typed tokens distinguishing strings, integers, booleans and finite
floats (float tokens retain exact hex values). Each categorical column contributes
all known F one-hot columns plus an explicit reserved unknown column, placed last.
Missing and unseen categories use that column; a literal string resembling an
unknown token remains a normal typed category. These 0/1 columns are unscaled and
enter linearly. This deliberately replaces legacy ordinal category codes only
inside the new v2 adapter, preserving the legacy categorical spline rejection.

Ridge design order is intercept, standardized continuous columns, F-created
numeric missingness indicators, then categorical one-hot blocks in schema order.
Spline design replaces each continuous column z with additive
`z, z**2, z**3, maximum(z-knot,0)**3` blocks, followed by those same linear missing
and categorical indicators. There are no interactions, gates or splines on
category codes/indicators. Knots are unique F quantiles at
`linspace(0,1,n_knots+2)[1:-1]`, with linear quantile interpolation; discard knots
at the observed F extrema. Retain requested and realized knot counts. An
all-categorical input has no spline blocks and remains a linear categorical fit.

Preserve legacy ridge's **sum-of-squared-errors + alpha times squared penalized
coefficients**, with an unpenalized intercept. Do not silently change alpha to a
mean-loss convention or rescale spline columns after basis expansion. Solve the
float64 augmented least-squares system using pinned NumPy `lstsq(rcond=None)`;
all coefficients except the intercept, including linear categorical/missingness
coefficients, receive the same ridge penalty. Mean returns the F target mean.
Predictions always invert the F target transform to original units. Nonfinite
statistics, basis overflow, solves or predictions fail rather than changing basis,
precision, training rows, alpha, or model. Failed refits preserve prior fitted state.

## Portable deployment and evidence

Save fresh, non-pickle `manifest.json` + `arrays.npz`. The strict manifest binds
resolved config, ordered schema/preprocessing/vocabularies, basis/knots and full
array inventory (dtype, shape and hashes), F input/target identities, source bytes,
core-lock bytes and numerical runtime identity. Load requires the external trusted
manifest SHA256 before accepting any NPZ bytes, then verifies exact schema,
method/config/basis consistency, array/file inventory, float64 coefficient/knot
and integer-index dtypes, geometry, finiteness and current source/runtime identity.
No refit or dtype casts occur during load; reject unknown/duplicate fields,
object arrays, unsafe paths/symlinks and inconsistent states. Paths are provenance,
not a requirement to restore into the original absolute worktree path.

Source/lock identity comes from byte reads and numerical runtime identity from
Python/platform and pinned NumPy/Pandas metadata and installed source/library
bytes. Never import `train`, Torch, NODE, sklearn or a neural/tree dependency
merely to hash identities. Reuse plain schema/frame helpers only where their
imports remain lazy; no heavy module import may be hidden behind save/load.
Use the existing pinned environment (Python3.12.13, NumPy2.5.4, Pandas3.0.6), not
floating versions or a new lock. Fit/error receipts retain original exceptions
and paid elapsed time, with publication errors secondary. Report encoded/basis
widths and allocation sizes; outer startup, process limits, search accounting,
finite transformed-workload admission and interruption/retry harness remain
separate Gate D prerequisites. No adapter timer is a standalone efficiency claim.

## Verification after root approval

Use TDD for exact29 support/schema rejection; F-only category/statistic fitting
with G sentinels; unknown/type-collision categories; numeric-only parity against
unchanged mean/ridge/spline references; independent augmented solves on tiny
well-conditioned fixtures; missing/constant/all-categorical/rank-deficient inputs;
original-unit/batch-invariant predictions and immutable inputs; failed-refit
preservation; exact same-runtime save/load predictions; externally trusted hash,
structured schema/dtype/shape/source/runtime corruption rejection; and fresh-process
fit/predict/save/load proving Torch/NODE are never imported. Any legacy reference
is read or invoked unchanged. No prepared transfer rows, searches, full workload
qualification or practical-performance claims follow from these fixtures.

Root approved contract commit `2900a65` before implementation, with the explicit
feature-name clarification above. The semantic choices are declared adapter
choices, not additional historical preregistration. Implementation evidence and
the independent review disposition are retained in [the implementation ledger](P13-implementation.md).
