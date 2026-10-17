# Final pre-freeze H1 implementation review

Reviewed BASE `b29d360` through HEAD `b26931fda9a60ead929850e4589183e690de7e1d` on `main`, under the supplied final review brief and Superpowers code-reviewer contract. This is the single broad implementation review, before definitive measured calibration, source freeze and independent confirmation.

**Verdict: With fixes. Counts: 0 Critical, 2 Important, 3 carried Minor.** The implementation is substantial, but the authentic pilot producer cannot currently pass its own post-pilot validator, and the required high-density sensitivity is outside the executable receipt contract. This review establishes no scientific result.

## Spec compliance

The implementation preserves the main H1 contract: float32 eager training and CPU64 score/math, F-only fitting/transforms, G-selected checkpoints/growth, V-only HPO, a separately protected T capability, common parent/state/allowances, independent teachers, the confirmation-count floor, and explicit unfavorable/incomplete outcomes. The source/config/runtime commitment, fixed-cohort execution, saved-row analysis, failure inventory and prospective archive paths are implemented before the intended freeze.

Compliance is incomplete at two cross-task boundaries described below. The primary coordinate is executable within the bounded CPU envelope, but that does not make all specified calibration coordinates executable. Tasks 7/8 still correctly mark their measured scientific gates pending. Conditional H3 assets, deeper models and hard routing are not H1 implementation prerequisites.

## Strengths

- `src/treeneuron/pilot/calibrate.py:86` builds the common fitting graph from fresh G-selected parents and guarded checkpoints. The selected/random placement contrast uses common data, score table, effective added count and post-growth training opportunity; fresh ordinary/static controls and B6's keep/all-unit probes retain their distinct roles and costs. `tests/test_pilot_calibrate.py` exercises actual eight-role graphs, zero growth, E-loss reconstruction and V-access traps; `tests/test_greedy.py` covers state pairing, losing probes and failed selection.
- `src/treeneuron/protocol.py:87`, `:149` and `:252` check execution identity, accepted review bindings, source-bound resource samples, configuration/runtime/seeds, completed method/checkpoint graphs and saved E predictions. A hand-filled Markdown report or approval boolean does not grant freeze authority. The resource wrapper can legitimately be authored by the controller from genuine measured samples; an automatic wrapper emitter is not required by the plan.
- `src/treeneuron/pilot/design.py:333` retains independent-teacher counts, conservative finite-E variance treatment, the pilot-count floor, absolute planning effects for either sign, physically infeasible/undefined/estimation branches, and explicit Monte Carlo limitations. `src/treeneuron/analysis/primary.py:33` averages repetitions within teacher and never zero-fills missing pairs. The final completion state controls whether superiority can be claimed.
- `src/treeneuron/evaluation/confirm.py:198` follows the full frozen inventory with a finite symmetric retry policy, preserving failed attempts and seeds rather than substituting favorable units or stopping on the interim sign. Protected access validates before label generation and retains immutable observations/predictions for later reconstruction.
- Search retains official anchors, failed/timeout costs and immutable sampler ancestry. `src/treeneuron/evaluation/timing.py:111` and `:135` reconstruct complete measurement blocks and preserve pairing when estimating comparative timing precision; duplicate repeats do not become independent blocks. Host/resident/cold boundaries and overlapping memory counters remain explicit.
- The Task 13 fixes preserve partial failed-attempt evidence, resolve original-root receipt locators without changing scientific bytes, bind the actual imported analysis package to its checked source, and distinguish unsealed/external-unbound/bound archives. `src/treeneuron/analysis/audit.py:464` and the guarded work reduction preserve source and accounting authority; archive reanalysis does not create a fit, resume or T-access capability. The original I1–I4 and final F1 fixes remain closed within the reviewed code.
- Historical exploratory results and engineering fixtures retain their original provenance and are consistently distinguished from definitive calibration or independent predictive evidence.

## Issues

### Critical — none

### Important

**I1 — Real calibration writes mutually inconsistent pair and raw ledgers, blocking every authentic post-pilot design.**

Locations: `src/treeneuron/pilot/calibrate.py:249`, `:354`; `src/treeneuron/protocol.py:451`.

`run_repetition` publishes `pair.json` before returning. The coordinator then adds `record['repetition']` at line 356 and saves that augmented record in both the attempt and accepted-record sections of `raw.json`. It never republishes the pair record. `eligible_design` requires the parsed pair, attempt record and accepted record to be equal, so the genuine producer reaches `ValueError('Raw/pair ledger disagreement')` before it can establish a numeric design or freeze.

This is confirmed without fitting: the retained completed engineering attempt `artifacts/evidence8-engineering/pilot-75ecc7fbcde04b8587222be962a7371a` has two accepted records. For each, the sole key present only in the raw record is `repetition` (0 or 1); every shared key matches. That historical attempt has its own older source identity and is not presented as a fresh scientific run. The current producer/consumer lines independently establish the same defect.

The existing real-pair test validates the graph returned by `run_repetition`, while `tests/test_protocol.py` constructs simulated records with `repetition` already present before writing both ledgers. The coordinator tests mock the pair producer. Those separate tests therefore miss the authentic publication handoff.

Repair: establish the complete pair schema, including repetition identity, before its final publication, or explicitly finalize the same record at the coordinator boundary before sealing. Keep failed/interrupted pair retention and the strict equality check. Add one focused producer-to-consumer regression that exercises the actual publication handoff and equality requirement, including repetition 1; do not repair the fixture by manually inserting a field that the real producer still omits. Full scientific execution is unnecessary to verify this fix.

**I2 — Required calibration coverage exceeds hard-coded execution limits; issuing a new receipt alone cannot enable it.**

Locations: `src/treeneuron/train.py:260`, `:87`, `:461`; `src/treeneuron/baselines/causal_evidence.py:88`, `:152`; `src/treeneuron/backend.py:59`, `:507`. Requirement: spec `docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md:143` and `:236`; evidence plan `docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md:230`.

The named density sensitivity `8P0` requires 9,224 F rows at the primary coordinate. Training rejects more than 2,306 rows, and causal evidence validation repeats that ceiling. The exception suggests obtaining a relevant probe, but the producer and consumers encode fixed sizes: the backend calibration uses F2306/d16/h64/logical128, ordinary parent validation caps h at 64, causal models cap h at 95, and training/payload checks cap logical batch at 128. A newly measured JSON receipt cannot change those executable constraints.

This is an integration gap between the accepted bounded core and the planned calibration phase, not a request to remove safeguards. If work proceeded on the assumption that only measurements remained, completing the required high-density coordinate would force another execution-source change after the supposedly definitive source identity. It also affects the meaning of throughput selection: the current backend's doubled physical batches are divisors of fixed logical128; they are not a sweep over independently changed logical optimizer batches. Moreover, `train.py:270` rejects a selected base/grown physical recipe below128 because the production trainer does not implement accumulation.

Repair before definitive source identity: support the narrowly intended calibration envelope through a source/runtime-bound receipt contract and consistent training/checkpoint/causal validators, with relevant numerical and resource evidence for newly admitted operations. Do not simply raise constants or waive receipt checks. Alternatively, the controller must record a justified pre-test scope limitation with the omitted measurements and resulting claim boundary, and must not report the original full sensitivity requirement completed. A finite safe doubled-batch set can be justified; this finding does not require unlimited expansion, batch256 specifically, or every cross-product.

The executable scope assessment is:

| Coordinate or operation | Current implementation scope | Required action before scientific claims |
| --- | --- | --- |
| Primary d16/h64/m16, F2306, batch128 | Fits the ordinary parent and causal control envelope; ordinary upper bracket is h95. Actual source-bound runtime/resource evidence is still required. | Measure final ceilings, G policy, probe allowance, resource forecast and stability under the finalized recipe. |
| Density 0.5P0 = 577 and 2P0 = 2306 | F counts fit; the smaller workload is marked CPU correctness diagnostic, not a matching throughput calibration. | Run the selected exploratory measurements; retain their actual timing scope. |
| Density 8P0 = 9224 | Rejected by the F ceiling even with a fresh fixed-shape receipt. | Narrow validated implementation/evidence extension or explicit justified pre-test scope ruling. |
| Epsilon 0.01/0.2, noise/regime/control changes | The configuration/data paths can vary these without exceeding shape bounds, subject to the selected control's semantics. | Measure the planned representative sensitivities/controls; no full cross-product is required. |
| Smaller dimensions/widths, nuisance dimensions | Allowed only while the parent and every derived control remain in their bounds. Nuisance counts must stay below d. | Check derived ordinary/widened widths, rather than checking only the parent. Larger d/h require new supported operations. |
| Doubled training/scoring throughput batches | Backend probe explores physical divisors of logical128. The live trainer requires equal physical/logical batches, at most128; smaller live batches have correctness-only workload status. Predictor timing can vary batching but does not certify optimizer-epoch throughput. | Measure and declare the supported logical-batch choice; extend coordinated producer/consumer support if the desired measured recipe needs accumulation or larger logical batches. |
| Depth greater than one or hard-routing extension | Pilot explicitly requires a depth-one selective parent. | Conditional extension, outside required H1 acceptance; no automatic expansion. |

### Minor — explicit triage of all carried findings

**M1 — Per-visit NODE row traces grow with total training work. Retained; defer only while the long-run NODE cost/quality gate remains pending.**

Locations: `src/treeneuron/baselines/node.py:137`, `:151`, `:219`, `:238`; `tests/test_node.py:88`, `:141`.

Every visited row becomes a Python integer in a growing list. Summaries build a set from that list, and save/load preserves the entire trace. Long training therefore adds visit-proportional memory, Python work and serialized metadata to the method's measured costs. The short retained run is correctly marked quality-ineligible and establishes no NODE quality win; it does not establish that this overhead is negligible at default stopping budgets.

Recommended eventual repair: retain bounded coverage/count/order-digest evidence in production, with a bounded opt-in exact trace for the numerical replay fixture. Preserve deterministic sampler replay and full-F/remainder tests. Deferral is acceptable for the current H1 implementation gate only if long default NODE measurements/efficiency claims remain pending and the limitation is recorded. Correct it before using long NODE runs to choose quality or cost policy, or disclose and measure the instrumentation overhead explicitly. No vendor optimizer modernization is implied.

**M2 — Expected vendor and tiny-fixture warnings remain unasserted test output. Retained; defer as test-maintenance work, with narrowly scoped handling.**

Locations: `tests/test_node.py:35`, `:64`, `:82`; vendor emission sites `src/treeneuron/baselines/_node_vendor/odst.py:113` and `qhadam.py:133`.

The retained original full-suite log has 28 warnings: Torch JIT deprecation, the original `add_` overload, less-than1000-row NODE initialization, and sklearn reducing quantiles for the 37-row fixture. They are understood consequences of the archived vendor recipe and deliberately small tests; they are not evidence of invalid numerical results. Their unstructured presence nevertheless makes new warnings easier to overlook.

Prefer warning-category/message assertions at tests that own the triggering operation, with isolated import handling where warnings occur once per process. Unexpected warnings should remain visible. Do not hide all warnings globally or modify faithful vendor mathematics merely to silence the log. Deferring this does not block H1 calibration; record the known baseline and keep new warnings distinguishable.

**M3 — Advisory repetition optimum omits evaluation cost. Retained; recommend correcting before calibration.**

Locations: `src/treeneuron/pilot/design.py:398`, `:402`; main forecast at `:370`; current test `tests/test_design.py:162`.

Both the continuous optimum and discrete candidate objective charge only `seconds_per_repetition`. The main forecast correctly also charges `n_T * seconds_per_evaluation_row` per repetition. When evaluation is material, the recommendation consequently favors too many repetitions for its stated variance-per-cost objective.

Use the same effective per-repetition cost, `seconds_per_repetition + n_T * seconds_per_evaluation_row`, in both advisory calculations, and test a case in which nontrivial evaluation cost changes the recommended count. Preserve the selected actual repetitions and the rule that any expansion needs new calibration. This remains Minor because the recommendation does not change actual selected repetitions or the complete cohort forecast. A documented advisory-only deferral is possible, but the small correction is preferable before the definitive design is produced.

## Checks and evidence

- Completed one full pass over the supplied 12,494-line implementation/test/config diff. Read runtime vendor implementations, the first-party changed source/tests/configuration/dependency recipes, normative spec/plan and decision context. Recovered truncated documentation output and used targeted line excerpts for report anchors and the named inherited runtime-boundary question; did not reread the full duplicate review package or bulk raw hash/license blobs.
- Read the binding review brief/draft and code-reviewer template. Inspected current Task9-v3/Task10-v2/Task13-v1 acceptance records and the original/fix Task13 review trail. Earlier scoped findings remain resolved; this review does not reopen them as new work.
- Read the actual retained suite summaries: original **908 passed, 48 skipped, 28 warnings in 1485.54s**; fix1 affected **15 passed in185.78s**; fix2 affected **17 passed in11.52s**. These belong to their recorded source stages. They are not a fresh full-suite result for this review, and they do not cover I1's producer/consumer handoff.
- One read-only standard-library JSON comparison inspected the existing engineering `raw.json` and its two `pair.json` files. It reported only the missing repetition key, with no differing shared values. No numerical imports, fitting, prediction, scientific cohort, pytest suite, fixture recreation or protected-label access was performed. Existing evidence was sufficient; no new execution probe was needed.
- Inspected the inherited training/backend and data-dispatch boundaries only to resolve the explicitly named calibration-coverage risk. No web/literature project, helper agent, repository checkout, branch/index mutation or implementation edit was performed. HEAD remained the requested hash and the tracked working tree remained clean; this report is the only authored file.

## Recommendations

Use one combined fix dispatch for I1 and the controller's concrete I2 scope decision; include M3 if taking the recommended small repair. Adjudicate M1/M2 explicitly, with the conditions above. Add only focused tests for the repaired cross-task contracts and newly supported runtime boundaries. The supplied passed suites need no automatic repeat merely because the broad review is complete.

Any execution-source or reviewed-file change must receive the planned scoped re-review and renewed affected acceptance bindings before source-bound resource sampling and definitive calibration. Preserve all earlier raw results under their original identities. After calibration, generate the canonical numeric design, freeze the clean source/inputs, execute the whole declared independent cohort, and complete the scientific reproduction/audit/report obligations. An unchanged implementation does not need another broad code review after the study.

## Declined to judge

- Actual H1 benefit, statistical power achieved on future independent teachers, or scientific report acceptance: no genuine pilot, freeze or independent cohort exists at this stage; software fixtures cannot answer them.
- Final numeric training/HPO/timing/retry policies, current physical feasibility and final bootstrap precision: these require the authorized later source-bound measurements. Executable coverage was assessed above, not assumed from existing engineering receipts.
- Full default NODE convergence, quality and long-run efficiency: the retained three-update anchor is explicitly quality-ineligible; later measurements and the M1 disposition remain necessary.
- Mandatory practical H3 raw datasets/adapters, optional foundation assets, transfer conclusions, deeper growth and hard routing: explicitly conditional in the accepted spec/plan, not prerequisites for the present H1 scope.
- Automatic readiness/resource-wrapper generation: controller-authored wrappers linked to genuine measured evidence are allowed. Their absence alone is not an implementation omission; validators must still verify their scientific identities and quantities.
- Rechecking every duplicated raw hash inventory, archived upstream snapshot/license blob, or external provenance URL: excluded from this review package by the binding brief and independently byte-verified by the controller. The executing vendor code and declared mappings were reviewed.
- Locked scientific clean reproduction, representative interrupted-growth/pipeline reproduction, twice-regenerated scientific reports and the final prior-art claim-boundary revisit: later Task13 scientific obligations, not completed by its implementation fixtures or this review.

## Assessment

**Spec: needs the two cross-task corrections or an explicit justified I2 scope ruling. Quality: with fixes. Ready for definitive calibration/freeze: no at this HEAD.**

The core scientific boundaries and substantial negative-path tests are soundly implemented. Repair the authentic ledger handoff and resolve the finite executable calibration envelope before treating the code as final; retain explicit dispositions for the three Minor items. Counts: **0 Critical, 2 Important, 3 carried Minor**.
