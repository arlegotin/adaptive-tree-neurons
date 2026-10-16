# Task 8 pre-freeze implementation review

## Spec compliance

- **Issues found; implementation needs fixes.** The authorized engineering-only phase is implemented, including the requested files and public calculation/driver interfaces. Two design-calculation gaps remain: the purported between-teacher variance bound retains finite-E sampling noise, and declared required controls beyond the hardcoded minimum are not enforced (`src/treeneuron/pilot/design.py:153`, `:292`). Neither finding requires running a genuine pilot or expanding to H3.
- The scientific-purpose restriction and deferred readiness validation are appropriate for this phase (`src/treeneuron/config.py:125`, `src/treeneuron/pilot/design.py:360`). The explicit PRE-pilot / POST-pilot contract avoids requiring genuine pilot outputs before running the pilot (`audit/implementation/evidence/2026-10-16/task-8-report.md:43`). Task9 must implement that validator and scientific schema branch before definitive execution; these authorized integration obligations are not additional defects here.
- Cannot verify from this diff: actual final physical feasibility, measured final numeric policies, genuine pilot precision, protected inference and complete independent confirmation. These are explicitly pending, and the retained output is blocked/freeze-ineligible (`pilot/design_decision.md:3`). Root supplied independent verification of the 400 indexed artifacts, receipt/runtime, 32 terminal manifests and byte-identical regeneration; I did not duplicate it.

## Strengths

- The driver fixes F/G/V/E within each teacher, separates optimizer/allocation/contrast repetitions, uses independent exploratory E and retains zero growth (`src/treeneuron/pilot/calibrate.py:41`, `:56`, `:194`; `tests/test_pilot_calibrate.py:108`). The minimal control set includes actual B2/B3/B6 fits rather than a placeholder inventory (`src/treeneuron/pilot/calibrate.py:151`).
- Repetitions remain within teacher; per-row averaging preserves shared-E correlation, and the bootstrap resamples whole teachers (`src/treeneuron/pilot/design.py:88`). Calibration stopping uses SD/runtime/endpoint precision rather than effect significance (`:173`).
- Both signs use half the absolute effect; static fallback, unresolved/zero-variance estimation, the accepted-pilot count floor and infeasible-workload blocking have concrete branches and tests (`src/treeneuron/pilot/design.py:301`; `tests/test_design.py:37`, `:47`, `:64`, `:72`).
- Costs use enclosing clocks with nested subclocks disclosed; failed attempts and fresh-parent lineage are retained. Raw regeneration checks source and a separate raw hash seal (`src/treeneuron/pilot/calibrate.py:204`, `:223`, `:264`).
- The extra power check is a conservative inference authorization policy tied to the existing 0.8 power target, not an application utility threshold. Its empirical-shape assumption, t-rejection method and coverage limitations are disclosed (`src/treeneuron/pilot/design.py:192`, `:335`). It remains a sensitivity calculation: Task9/10 must verify support for the actual primary bootstrap inference policy, as the two-stage handoff already states.
- Legacy canonical configurations omit absent pilot fields, and new pilot fields use the existing strict schema path (`src/treeneuron/config.py:160`; `tests/test_pilot_calibrate.py:24`).

## Critical

None.

## Important

### I1 — finite-E sampling noise is treated as between-teacher variance

`src/treeneuron/pilot/design.py:153` subtracts only mean within-repetition variance divided by R from the variance of observed teacher means. Because each teacher's E is finite and shared across its repetitions, the resulting quantity still includes the sampling variance of the E mean. Taking its lower bootstrap percentile does not turn it into a lower bound on the intrinsic between-teacher component. That quantity is subsequently used as the denominator of the conservative test-size rule at `:304`.

A concrete admitted model is `D_ij = mu + epsilon_ij`, identical across repetitions, with independent row noise and the same true mean for every teacher. Within-repetition variance is zero; observed teacher means have variance approximately `var(epsilon)/n_E`. With enough teachers the present bootstrap can establish a positive denominator and authorize a finite powered design, even though intrinsic between-teacher variance is zero and the specified unresolved/equivalence branch should apply. Increasing inner repetitions cannot remove this shared-E component.

Account for finite-E uncertainty when constructing the between-teacher lower bound, with an explicitly justified hierarchical decomposition or conservative bound that handles shared E and optimizer interaction. Preserve the unresolved branch when that component cannot be separated. Add a pure fixture with known zero intrinsic teacher variance and finite independent E noise; the existing shared-E test only uses identical teacher means and does not exercise this case (`tests/test_design.py:137`).

### I2 — the declared required-control set is not enforced

`src/treeneuron/pilot/design.py:292` checks that both the observed controls and `constraints.required_controls` contain B1–B6. It never checks that all controls declared required by the caller are actually present. Thus `required_controls = B1–B6 + ['additional_anchor']` with observations containing only B1–B6 passes the coverage gate and can produce a powered decision. This contradicts the selected-study missing-required-baseline blocking contract; it is independent of whether H1 currently requires any additional practical family.

Validate the declared set's type/domain and require it to be covered by observed or explicitly verified external control evidence, while retaining the B1–B6 minimum. Add the missing-extra-required-control fixture. This does not require running B7/H3 or changing the accepted existing baseline ruling.

## Minor

### M1 — repetition recommendation omits evaluation cost

`src/treeneuron/pilot/design.py:348` and `:351` optimize repetitions using only `seconds_per_repetition`, although the actual forecast at `:320` additionally charges `n_T * seconds_per_evaluation_row` for each repetition. When conservative n_T is large, the recommendation can substantially overstate the economical repetition count. Use the same complete marginal time cost in both the continuous optimum and integer comparison, or explicitly label a training-only recommendation. Actual selected repetitions remain unchanged and require fresh calibration, so this does not currently corrupt the chosen cohort.

- Carried final-triage item, not a new Task8 regression: the retained full-suite log has the established 28 vendor warnings (`audit/implementation/evidence/2026-10-16/task-8-full-tests.txt:13`). Existing Task3 row-trace/warning rulings are unchanged.

## Assessment and checks

- **Task quality: Needs fixes.** The implementation has substantive driver behavior, meaningful real pairing tests and honest phase/evidence boundaries. I1 affects the promised conservative statistical design, and I2 permits a selected-study requirement to be silently omitted.
- Reviewed BASE `93bddf9` to HEAD `9a018e5` using the provided diff in passes; no changed source files were reread separately, no Git commands, suite/CLI reruns, helper agents, training or protected work were performed. The machine artifact-index bulk was left to root's already completed independent verification.
- Exact-rule check: read spec §9, the cohort amendment and cited preflight to assess inference/floor requirements. Concrete outside-diff API risk: checked `src/treeneuron/baselines/causal.py:194` to confirm that the driver's two fresh-control fit calls use parent then continuation policies with matching F/G provenance and numeric allowances. No issue found there.
- Existing verification accepted as supplied: 152 covering tests; full suite 763 passed, 48 MPS skips, 28 known warnings in 294.35s. Their overlap is not evidence of benchmark performance and does not justify another suite run.
- New findings: **0 critical, 2 important, 1 minor**. Scientific evidence and final Task9/10/13 readiness remain pending regardless of these fixes.
