# Core Task 6 implementer report

## Status and commit

Implemented in the existing main checkout from baseline `272e4ea`. Commit:
`94553f37620ea90cea70819659b1e0c896a7f4b3`
(`feat: add fitting-only preprocessing and regression sanity controls`).
Nine task-owned files, 732 inserted lines. No dependencies changed. This handoff
report is left outside the commit for the controller/reviewer. No branch or
worktree change, subagent, protected evaluator/label access, predictive
experiment, protocol freeze, plan checkbox, ledger, README or DECISIONS edit.

## Delivered APIs

- `data/transforms.py`: `FittedTransform.fit(X_fit, y_fit)`, `transform_X`,
  `transform_y`, `inverse_y`, `to_state`, `from_state`.
- `evaluation/metrics.py`: exact requested `regression_metrics` signature.
- `baselines/api.py`: exact frozen `FitRequest` fields/order and `Predictor`
  methods/signatures from design section 11.
- `baselines/simple.py`: `MeanPredictor`, `RidgePredictor(alpha)`, `fit`,
  `predict`, `save`, `resource_summary`; classmethod `load` reconstructs either
  model from a strict JSON snapshot, with no executable pickle/vendor state.
- Minimal baseline/evaluation package initializer files and three test files.

## F-only preprocessing and state

All fit statistics use only passed F arrays. Numerical arrays retain their
positional width and must be real two-dimensional matrices. Raw pandas frames
retain exact supplied column names/order and numerical-versus-categorical
kinds; switching containers, reordering/missing/extra columns or changing kinds
fails clearly. Pandas nullable numeric columns and NumPy scalar categories are
handled deliberately. Empty prediction matrices retain two-dimensional width.

Numerical missingness uses the observed F median; an all-missing F numerical
column uses fill 0, population mean 0, scale 1, constant/all-missing flags, and a
missingness indicator. Population mean/std are calculated after imputation;
constant numeric columns use scale 1 and a persisted constant flag. Indicators
are appended, in source-column order, only for numerical columns missing in F.
A previously complete numerical column missing later uses its F median without
adding an indicator. This policy is explicit in state as `F_missing_only` and
`F_median_without_indicator`, preserving the complete synthetic d=16 input width
and primary parameter counts.

Targets must be finite one-dimensional numerical [N] values, with matching F row
count. Target scaling uses F population mean/std; constant-target scale is 1.
`inverse_y` validates and restores finite original-target-unit [N] predictions.

Categorical strings/integers/booleans use type-tagged deterministic F dictionaries
with ordinal codes 1 onward and reserved 0 for unknown/missing. These are sanity
control encodings, not native vendor preprocessing or a strong practical recipe.
Numerical categorical DataFrame columns require categorical dtype to avoid
ambiguity with numerical predictors; float-valued category tokens, nested
objects, complex numbers and unsupported scalars fail rather than coerce.

State version 1 uses only finite JSON scalars, dictionaries/lists and copied data.
Unknown fields, invalid positive scales, inconsistent constant/all-missing flags,
duplicate categories/column names, bad typed category tokens, invalid array
schemas and project metadata fields are rejected. State copies cannot mutate a
reconstructed transform. No downstream labels are accepted by fitting APIs.

After clarification with the controller, raw frames reject the same nine
reserved project-record names as ObservationSet (`y`, `noise_free_target`,
`gates`, `teacher`, `reference_mean`, `reference_std`, `row_id`, `group_id`,
`dates`), including on state reconstruction. Nothing is silently dropped. This
only guards against accidentally passing our records as predictors; future raw
source adapters still require explicit source-specific predictor allowlists.

## Metrics

Metrics retain raw original-unit MSE/RMSE/MAE, row count, and F-population-variance
normalized MSE. Zero F variance yields null normalized MSE with an explicit
degenerate-task reason and raw errors retained. Negative/nonfinite/boolean
variance, wrong shapes, mismatched row counts and nonfinite labels/predictions
are rejected before broadcasting. Noise-free MSE is an explicitly separate
optional output.

Group labels must be homogeneous stable strings or integers. Mixed Python
labels are preserved until validation so integer 1 and string '1' cannot merge
through NumPy coercion. Group outputs are deterministic records with label,
count, MSE and MAE, plus equally weighted group-macro MSE and worst-group MSE.
Tail guardrails expose absolute-error p95 and maximum. Large finite squared
errors are averaged by dividing before summation, avoiding unnecessary overflow.
No inference or practical success threshold was added. Paired-effect zero is
checked by subtracting identical normalized risks; effect convention remains
control risk minus proposed risk in the later analysis harness.

## Mean and ridge controls

Both adapters fit only F, never inspect G arrays/labels, and do no HPO/checkpoint
selection. Mean predicts the F target mean. Ridge uses an augmented least-squares
system on transformed F features with an unpenalized intercept; nonnegative
finite externally supplied alpha penalizes feature coefficients. Alpha 0 uses
the stable least-squares path, including rank-deficient designs/constant targets.
Independent closed-form parity fixtures cover alpha 0, 0.7 and 100.

Predictions return finite [N] original-unit CPU float64 arrays. Batch size accepts
positive Python integers or None, does not change semantics, and does not impose
physical memory batching in these small controls. JSON snapshots store method,
version, transform, coefficients and alpha. Loads reject unknown fields/methods,
wrong coefficient width, nonfinite values and invalid alpha. Capability summaries
disclose F-only ordinal coding, no internal validation/sample weights/context/
cache/transduction, CPU float64 backend, no fallback or batch-composition effect;
unmeasured timing/memory counters have null values with explicit reasons.

## TDD evidence and verification

1. Wrote all initial tests before implementation. Ran:
   `.venv/bin/python -m pytest tests/test_transforms.py tests/test_metrics.py tests/test_simple_baselines.py -q`
   Observed expected RED: three ModuleNotFoundError collection failures for the
   absent requested modules (the task brief explicitly anticipated missing APIs).
2. Implemented requested APIs; same focused command: 47 passed.
3. Self-review added regression fixtures for mixed Python group labels and
   large finite-loss aggregation. Ran focused transforms/metrics: 2 failed,
   36 passed, reproducing silent group coercion and overflow. Fixed both;
   all three focused files: 51 passed.
4. Added reserved-project-metadata input tests after controller clarification:
   9 failed, 19 passed. Added saved-state metadata test: 1 failed, 17 passed
   alongside baseline tests. Implemented rejection in raw/saved schema paths.
   Additional snapshot mutation checks verify unknown fields, wrong coefficient
   width, invalid alpha and mismatched method rejection.
5. Final focused command: 65 passed (29 transform, 19 metric, 17 baseline tests).
6. Exactly one final full-suite run after final code changes:
   `.venv/bin/python -m pytest` -> **259 passed in 29.67s**, no warnings/failures.
   Baseline 194 + new 65 = 259.
7. Reviewed the staged production diff, task ownership and scope; staged
   `git diff --cached --check` clean. Commit succeeded with authorized git-write
   escalation after sandbox denied index.lock creation. No approval rejection.

## WP02 decision and remaining boundaries

Recommend accepting the synthetic/harness preprocessing, original-unit metrics
and B0 mean/ridge sanity portion for WP02 review. Evidence is the task commit
and 259-test full-suite result. No claim of raw transfer-data acceptance: loaders,
raw-byte/schema/license verification, source predictor allowlists and sufficient
group checks remain open for the evidence plan. The controller owns ledger and
milestone updates, so this report records the recommendation without editing
DECISIONS or claiming a completed broader WP02 gate.

No known failing tests or blocker remain. Limitations are deliberate: ordinal
category codes, restricted scalar/schema types, descriptive p95 guardrail rather
than a frozen inferential rule, unmeasured resource counters, no physical
prediction batching, and explicit source-specific leakage prevention deferred
to raw transfer adapters. Root should dispatch the planned independent reviewer.
