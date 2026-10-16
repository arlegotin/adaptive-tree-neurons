# Core Task 7 report: moment scoring and matched allocation

Implemented on current `main`, based on `c115cac`, without a branch/worktree change, subagents, dependencies, predictive comparisons, protected T access, or plan/ledger/docs edits. This report is the requested workspace handoff artifact.

Final task commit: `7d637f59a9ebd8d5232e363e0d4a483e88fb06c3` — `feat: implement normalized moment scoring and matched allocation`. This amends the unpublished intermediate task commit `82bbd83`; reviewers should review `c115cac..7d637f5`. Working tree was clean after commit. No blockers remain for Task 7.

## Public API and integration contract

- `treeneuron.growth.score.score_units(model, X_fit, y_fit, *, epsilon, tau, adam_epsilon, batch_size, unit_block_size) -> ScoreTable` implements squared-error scalar-regression scoring. Inputs must be nonempty finite CPU float64 X `[N,d]` and y `[N]` or `[N,1]`. Tau must match the layer; all three numeric constants and both integer block sizes must be positive and finite where applicable.
- Caller supplies a CPU float64 evaluation model. The scorer identifies exactly one ungrown `SelectiveAffineLayer` by module identity and one scalar `Linear(h,1)` head, supporting Sequential and ordinary wrappers, including `[N]` outputs. Every submodule must be in eval mode. Execution checks verify that the selective layer consumes the original input, executes once, and supplies an out-of-place ReLU to the scalar head. Extra target layers, previous growth (including completed empty growth), unsupported leaf modules, alternate activations, transformed outputs, and in-place ReLU are rejected. Functional downstream mutations of the captured preactivation are also rejected. Primary architecture remains ordinary out-of-place ReLU.
- Floating model parameters/buffers must be CPU64; integer/bool buffers remain their native CPU types. Nonfinite model tensors, F data, preactivations, predictions, or loss raise `InvalidScoreRun`, a `ValueError` subclass. Individual units with nonfinite statistics or unusable eigendecomposition remain in the table and are ineligible; these are distinct from blocked runs.
- Scoring uses a private deep copy after validating the caller-provided eval copy. Original mode, parameters, buffers, existing `.grad`, hooks, Python/NumPy/Torch CPU RNGs are preserved, including rejection paths. Frozen parameters and an enclosing `torch.no_grad()` are supported. All scoring/allocation tensor factories explicitly select CPU, independent of caller default factory device.
- `ScoreTable` contains CPU int64 IDs, CPU64 contrasts/gradients `[h,d+1]`, Adam/SGD scores, full signed ascending eigenvalue spectra, absolute spectral gaps, preactivation/projection RMS, boolean eligibility and near-tie flags, tuple statuses, F augmented unweighted Gram matrix, epsilon/tau/Adam epsilon, F row count, and formula version `signed-moment-rms-v1`. It retains no fitting rows or full per-unit moments. Its computed `table_hash` binds every field name, dtype, shape, tensor bytes (including nonfinite diagnostics), statuses and constants; mutating table tensor contents changes the next hash.
- `treeneuron.growth.allocate.allocate(table, requested, *, policy, seed) -> Allocation` accepts policies `adam`, `sgd`, `random`, `keep`, and documented direction-ablation alias `gaussian`. `requested` is a nonnegative integer; local seeds are integers in `[0,2**63)`. Allocation contains requested/effective counts, sorted CPU int64 IDs, corresponding CPU64 contrasts, policy, seed, actual table hash, per-selected-ID direction attempts, and explicit work-accounting counts. The later insertion caller casts contrasts to the trained layer dtype/device.

## Mathematical and allocation decisions

`e` comes from `autograd.grad(sum(row squared losses), preactivation)` on the real model. Moments accumulate divided by total F size, then are symmetrized. Each block retains only `block_units*(d+1)^2` moment entries; per-unit matrix multiplications avoid a contraction path materializing row outer products. Batch activations and a small Gram matrix add bounded workspace. The implementation recomputes forward/derivatives per unit block to avoid retaining all F derivatives; that extra real scoring compute is disclosed.

The eigenvector corresponds to the largest absolute signed eigenvalue, with the first maximum-magnitude coordinate positive. No PSD covariance routine is used for the signed moment. Contrast normalization is exactly the specified epsilon times base RMS divided by the projection RMS floored at `1e-12`. Projection RMS uses the exact F-derived augmented Gram sufficient statistic. Gradients are `M v / (2 tau)`; both scores use the prescribed `2(d+1)` denominator and actual supplied Adam epsilon.

An absolute spectral gap compares the two largest eigenvalue magnitudes. Near ties use `gap <= 1e-12 * max(1, spectral_radius)` and remain eligible when other criteria pass. The implementation makes no stable-direction claim in repeated eigenspaces. Finite zero-base-RMS units report `zero_base_rms`; finite zero-score units report `zero_score`; nonfinite statistics and failed eigendecompositions have separate statuses. Invalid units are never padded into growth.

Adam/SGD ranking uses strictly positive finite eligible scores; ties sort by original unit ID. Random placement uniformly samples without replacement from the same eligible set through a local CPU generator and uses exactly each sampled ID's stored eigenvector contrast. Counts cap at eligible units, including empty allocations. Keep preserves requested count but records zero effective growth.

Gaussian ablation uses the same Adam-ranked IDs/count, with its supplied independent local seed. Directions are Gaussian, unit-normalized, sign-normalized and RMS-scaled using the same F Gram and per-ID base RMS. Zero/nonfinite/unusable directions redraw deterministically, record attempt counts, and reject after 100 failed attempts instead of silently replacing units or shrinking the count. Non-Gaussian attempt counts are zero.

Random cost accounting reports `shared_scored_units=h` and `standalone_contrast_units_required=m_eff`. This API has actually constructed the complete shared table; the second count is a requirement for a future standalone sampled-ID path, not a claimed cheaper executed measurement. No sampled-only scorer or runtime measurement is fabricated.

## TDD and verification evidence

1. Added analytic/autograd gradient and gate-only Adam tests, dense sum-loss moments, batch-tail/block invariance, signed negative dominant spectrum, sign convention, RMS/SGD scores, ReLU-rescaling/permutation, state/RNG, zero-unit, invalid-model/input, wrapper, matched-count/contrast, local Gaussian seed/RMS/hash and empty-policy tests before implementation. Initial focused run failed collection because `growth.score` and `growth.allocate` were absent, as expected.
2. Implemented the APIs; first focused result was **20 passed**. Added invalid-input/statistics, Gaussian zero/nonfinite redraw/exhaustion and package-gate integration tests. Observed RED for scalar-input validation (TypeError instead of ValueError), absent package gate, and old scope; fixed these, then **35 passed** on the focused score/allocation and selected integration checks.
3. Self-review replaced a generic moment einsum with explicit per-unit GEMMs to bound intermediate storage. Added nonfinite prediction/loss, wrapper-buffer/RNG and frozen/no-grad coverage. Focused score/allocation/reference tests then reported **59 passed**.
4. Reproduced caller-default-device contamination: two CPU-placement regressions failed under `torch.device('meta')`. Made all score/allocation factories explicitly CPU. Final initial focused run: **39 passed in 1.17s**. Initial full suite: **299 passed in 36.32s**.
5. Final audit found that an in-place activation could overwrite the captured preactivation. Reproduced both module and functional in-place cases as RED after ensuring the model remained in eval mode. Added an explicit out-of-place restriction and a captured-preactivation value guard. Rejection tests verify original state, RNG and hook dictionaries are unchanged. Focused result: **41 passed in 1.15s**. This new correctness concern justified repeating the full suite before amending the task commit.
6. Final verification: `.venv/bin/python -m pytest -q` — **301 passed in 34.88s**, no failures. `git diff --check` passed. Only the seven task-owned source/test files were committed.

The tiny `score_checks.py` integration executes real package scoring/allocation in a fresh process; `verification.py` validates its schema/results and propagates failures through CLI status. The scope is now `reference_and_package_math_cpu_only`. CPU/MPS backend gates remain `not_implemented` and complete remains false for Task 8. Historical reference numerical implementations were untouched; existing full-suite tests continue to execute them, with no extra standalone historical rerun.

Fresh `.venv/bin/python -m treeneuron.score_checks` evidence (arm64 macOS, Python 3.12.13, Torch 2.14.1, CPU64, one thread):

| Quantity | Result |
|---|---:|
| Gate gradient maximum absolute error | `4.336808689942018e-19` |
| Batched score maximum absolute error | `4.336808689942018e-19` |
| Batched contrast maximum absolute error | `4.5102810375396984e-17` |
| Positive-rescaling score error | `8.673617379884035e-19` |
| Permutation score error | `6.505213034913027e-19` |
| Contrast RMS normalization error | `6.938893903907228e-18` |
| Random per-ID contrast error | `0.0` |
| Matched effective count | `3` |
| Predicted gate-only loss decrease | `2.8955387489426842e-09` |
| Observed gate-only loss decrease | `2.895539141789527e-09` |
| Adam prediction relative error | `1.3567314304246556e-07` |

## Limits and self-review

Only the first growth event for the specified scalar squared-error model is supported. There are no training trajectories, optimizer migration, MPS backend claims, predictive outcomes or generalization claims in this task. Wrapper support checks the actual core computation; it is not a general graph/activation adapter. In-place activations are deliberately unsupported. Repeated/near eigenvalue ties are diagnostic instability. CPU64 Gram arithmetic and eigensolvers retain normal floating-point conditioning limits. The private eval copy and repeated per-block forwards have real costs, to be measured by the later runner. Standalone random-only contrast execution remains future integration, explicitly distinguished from shared research scoring.

Self-review checked normalization against independent dense moments/autograd, signed spectra versus a hand-derived negative-eigenvalue case, exact per-ID contrast reuse, tie ordering, integer/count/seed validation, zero/ineligible paths, actual hash binding, nonfinite-run distinction, device/RNG/state protection, hook cleanup, bounded intermediate storage, and score-gate failure propagation. No unresolved Task 7 defect or blocked test remains.
