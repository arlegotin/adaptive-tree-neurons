### Spec Compliance

- ✅ Spec compliant for Core Task 7, base `c115cac` through `7d637f5`. All four required new files are present, with the permitted minimal package-score gate integration in the other three files. No missing, extra, or misunderstood task requirement was found.
- ✅ Signed moment and row-loss scaling follow design §5: `growth/score.py:144` differentiates the sum of row squared losses, `:206` divides accumulated signed moments by total fitting rows, and `:209` symmetrizes each matrix. The implementation uses the actual downstream model derivative and never substitutes a PSD covariance routine. Independent autograd/grown-gate and dense signed-moment tests are in `tests/test_score.py:29` and `:51`.
- ✅ Bounded moment intermediates follow design §5: `growth/score.py:197` allocates only the current unit block; `:205` uses one unit GEMM at a time rather than building row outer products. The small augmented Gram matrix and diagnostic outputs are retained, with no full fitting-row derivative table or all-unit moment tensor. Batch-tail and block-size agreement are covered by `tests/test_score.py:68`.
- ✅ Dominant absolute signed eigenvalue, deterministic first-maximum-coordinate sign, absolute spectral gap, recorded near ties, F-derived projection RMS with the specified floor, and both exact score formulas appear at `growth/score.py:215`, `:222`, `:224`, `:227`, `:229`, and `:230`. Negative-dominant and repeated-spectrum cases are explicitly tested at `tests/test_score.py:96`; rescaling/permutation agreement is tested at `:75`. This satisfies design §5 without claiming stable directions in repeated eigenspaces.
- ✅ Per-unit zero/nonfinite statistics remain diagnostic rows with eligibility reasons at `growth/score.py:211`, `:233`, and `:235`; whole-run nonfinite tensors, predictions, and losses are rejected at `:105`, `:132`, and `:142`. Tests distinguish these paths at `tests/test_score.py:127`, `:179`, and `:192`, as required by §5.
- ✅ The first-event scalar-head and layer contract is explicit at `growth/score.py:85`: exactly one ungrown selective layer, one compatible scalar Linear head, CPU64 floating tensors, and eval mode. Actual execution, original layer inputs, scalar output shapes, ReLU/head values, and preactivation preservation are checked at `:113` and `:123`. This scope and explicit inplace-ReLU rejection match the binding Task 7 integration context; `tests/test_score.py:139`, `:153`, and `:239` cover the main rejection/wrapper paths.
- ✅ Caller state and training RNG protection follow the brief: `growth/score.py:70` restores Python, NumPy, and Torch CPU RNG; `:183` makes the private evaluation copy inside that isolation; `:147` removes temporary hooks on failure. CPU allocations are explicit. Existing parameters, gradients, buffers, modes and hook dictionaries are exercised by `tests/test_score.py:112`, `:202`, `:220`, `:230`, and `:239`. Private-copy/repeated-forward overhead is accurately disclosed at `growth/score.py:159` and in the task report.
- ✅ Fixed-count policies match §5 and the integration vocabulary at `growth/allocate.py:34`: nonnegative counts/local seeds, positive finite eligibility, Adam/SGD ranking with ID ties, uniform random sampling without replacement, sorted selected IDs, and exact stored per-ID contrasts. `keep` preserves requested count while recording zero growth. `tests/test_allocation.py:11` and `:28` check matching counts, exact seeded selection/contrast reuse, ties, caps, zero/ineligible tables, and rejection.
- ✅ Gaussian ablation uses Adam-selected IDs and its own local seed, sign normalization and exact F-Gram projection RMS at `growth/allocate.py:57`; it records deterministic redraw attempts and fails safely after exhaustion at `:59`. `tests/test_allocation.py:47`, `:69`, and `:88` cover these requirements.
- ✅ Diagnostic identity is substantive: `growth/score.py:24` stores the required arrays/configuration and `:55` recomputes a hash from field names, dtypes, shapes, actual tensor bytes, statuses and constants. `growth/allocate.py:74` binds the actual current hash to each allocation; mutation/configuration binding is tested at `tests/test_allocation.py:47`.
- ✅ The package score gate exercises the real APIs at `score_checks.py:23`, growth/autograd and a real tiny gate-only Adam step at `:47`, and numerical/count/descent assertions at `:73`. `verification.py:43`, `:100`, `:104`, and `:147` validate and propagate this evidence; CLI failure propagation is tested at `tests/test_reference_checks.py:205`. Backend gates remain pending honestly.
- ⚠️ Downstream scope, not a Task 7 defect: `growth/allocate.py:75` records actual full shared scoring and the standalone random sampled-ID contrast requirement separately. No sampled-only execution or cheaper runtime measurement is claimed. The actual standalone path, parent checkpoint selection, training forks, insertion casting, runtime measurement, backend gates and protected evaluation remain later integration obligations under the task context.

### Strengths

- `growth/score.py:110` separates execution/derivative validation from block accumulation, and the dense/autograd tests at `tests/test_score.py:29` and `:51` compare independent mathematical quantities rather than merely repeating the implementation.
- `growth/score.py:211` and `:233` preserve unusable rows with reasons while retaining useful diagnostics; `growth/allocate.py:41` also guards positive finite selection without padding empty growth.
- `growth/allocate.py:57` treats Gaussian direction failure as a reproducible, recorded event while retaining exactly the primary selected IDs and count.
- `score_checks.py:16` adds real package evidence and `tests/test_reference_checks.py:205` verifies that a broken package check reaches CLI failure, keeping the acceptance gate meaningful.

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- None found.

#### Minor (Nice to Have)

- None identified that warrants a task finding.

### Assessment

**Task quality:** Approved.

**Reasoning:** The implementation matches the exact signed-moment, normalization, scoring and matched-allocation contract, with meaningful independent checks and explicit diagnostic/failure handling. Its restricted first-event architecture, private-copy cost, repeated-forward cost and deferred standalone random execution are documented scope limits consistent with the task brief.

### Review Evidence

- Reviewed the supplied `review-c115cac..7d637f5.diff`, all task instructions/context, global constraints, binding design §5/layer contract and the implementer report. The initial combined tool output was truncated; missing diff sections were retrieved in bounded chunks. No source edits, commits, git state changes, subagents, or T access occurred.
- Focused outside-diff check for the layer integration risk: inspected `src/treeneuron/models/units.py:20`, `:35`, and `:44` to verify positive dimensions, returned preactivation semantics, and first-event/growth dtype contracts relied on by scoring/allocation. These agree with the new implementation.
- The verification diff cut the existing schema validator mid-function. Inspected only `src/treeneuron/verification.py:55` through `:90` to confirm the new gate also receives the existing complete finite scalar/environment validation.
- No test suite or historical script was rerun. The supplied task report records the final focused **41 passed**, final full **301 passed**, clean diff check, and concrete fresh-process score metrics. Static review found no unanswered behavioral risk requiring an additional targeted execution; no reported warning/noise requires a finding.
