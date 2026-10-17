# Longer Probe Allocation Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute a fresh paired comparison that isolates a longer validation-probe opportunity and produces audited exploratory evidence.

**Architecture:** Extend the existing greedy probe API with an opt-in normal-patience mode and the common fit graph with an opt-in second reference. A small diagnostic module owns fixed fresh samples, resource measurement, immutable member manifests and saved-prediction analysis. Historical canonical pilot/confirmation paths retain their default behavior.

**Tech Stack:** Existing frozen Python3.12/PyTorch/NumPy/SciPy stack; no dependencies.

**Spec:** [Long-probe diagnostic design](../specs/2026-10-18-long-probe-allocation-diagnostic-design.md).

## Global Constraints

- Work on current main; no worktree/branch switch, push or publication.
- Preserve fixed320 default probes and all historical artifact bytes.
- Long probes use ceiling8000, normal G patience20, shared parent/contrasts/RNG; no inherited probe weights in joint continuations.
- CPU float32 training/CPU float64 scoring; CPU1/inter-op1/allBLAS1/fallback0; serial numerical jobs, scoped caffeinate.
- F fits, G selects/ranks, V unused, E evaluates, no T/FrozenProtocol; no new score/depth/tuning.
- Fresh cost seeds261030100..105; cohort seeds261031100..105; exactly8 teachers/R2; bootstrap512000/seed261031106; no sign stopping.
- All nine final roles, both complete candidate searches, all failures and zero/tied outcomes retained and charged.
- Fresh source-bound receipt,10GiB disk reserve,6GiB RSS cap, measured forecast with1.5 multiplier; no silent shrinking or automatic retries.

## Review Focus

- Short/invalid long return: reject unfinished or unexplained termination; genuine patience completion is accepted only in the explicit mode.
- Hidden state reuse: both rankings and joint forks restart the same original parent, with matching contrasts/count and row prefix.
- Default API callers: no second reference or changed legacy graph unless explicitly requested.
- Manifest drift or missing pair: reject before fitting/analysis; retain failed member instead of dropping it.
- Metric/data leakage: original-unit predictions/F variance, row IDs and teacher-level pairing; E cannot select policies or counts.

## Task 1: Implement and review the complete diagnostic path

**Files:** Modify `src/treeneuron/baselines/greedy.py`, `src/treeneuron/pilot/calibrate.py`; create `src/treeneuron/pilot/probe_diagnostic.py`, `tests/test_probe_diagnostic.py`; extend `tests/test_greedy.py`, `tests/test_pilot_calibrate.py`; create `configs/diagnostics/long-probes-base.json` and `protocol/diagnostics/2026-10-18-long-probes.json`.

**Interfaces:** `run_greedy_probes(..., stopping_mode='fixed_updates')` accepts exactly `fixed_updates`/`patience`. `fit_repetition` and `run_repetition` receive keyword-only `reference_probe_updates: int | None = None`; None keeps eight roles, positive adds independently ranked `greedy_long` with patience mode and a fresh original-parent continuation. Public diagnostic module supplies `load_spec(path: Path) -> dict`, `measure(spec: dict, output: Path) -> dict`, `run(spec: dict, resource_decision: Path, output: Path) -> dict`, `analyze(output: Path) -> dict`, and `report(output: Path, destination: Path) -> dict`. CLI is `python -m treeneuron.pilot.probe_diagnostic measure --spec PATH --output NEW_DIR`, `run --spec PATH --resource-decision PATH --output NEW_DIR`, `analyze --output STUDY_DIR`, and `report --output STUDY_DIR --destination NEW_MD`.

- [ ] Write failing tests for explicit mode validation, genuine completed-patience acceptance, unfinished/unexplained return rejection and unchanged fixed/default behavior. Assert the parent/RNG remains unchanged; both coordinators see independent identical starts and contrasts; failures retain registrations/costs. Run only affected greedy tests and inspect actual failures.
- [ ] Implement mode handling and logged stopping evidence, leaving rank/count/trainer semantics unchanged.
- [ ] Write failing bounded real-graph integration tests: default eight roles; opt-in nine with same parent/score, warm row-prefix/count/contrast checks, two separate complete reports and finite E losses. Include zero eligibility and tied/negative rankings; do not add dummy parameters. Implement the opt-in fit graph and wrapper.
- [ ] Write diagnostic fixtures for strict spec/unknown keys, fresh distinct seed derivation, cost-versus-study separation, hashed lock/drift rejection, ordered complete/failed member accounting, guarded resource failure and no E-dependent selection. Implement small immutable reference-only ledgers and exact stated configs; no canonical calibrate/freeze invocation.
- [ ] Write analysis fixtures with deliberately unequal repetition/teacher counts and known risks; assert original targets/IDs/F variance, teacher averaging, positive-difference direction and retention of zero/negative effects. Missing/altered predictions must fail; report rendering twice must match bytes. Implement fixed paired bootstrap and human-readable report without inline JSON dumps.
- [ ] Run affected greedy/pilot/diagnostic tests, retain red/green commands/output and self-review. Commit implementation. Obtain one independent whole-change review (spec and quality); fix/review actual findings before numerical execution. This is the complete new source review before holding source fixed, not a repeat old H1 review/full suite.

## Task 2: Measure one full comparison and execute the locked cohort

**Files:** New `environment/long-probes-20261018/`, `artifacts/long-probes-20261018/`, `pilot/long-probes-20261018/`; curated provenance under `audit/diagnostics/2026-10-18-long-probes/`.

**Consumes:** Accepted Task1 APIs/configs and final source identity. **Produces:** Validated narrow receipt; complete cost member; explicit resource decision/forecast; diagnostic study lock; sixteen complete paired members or retained incomplete status.

- [ ] In a fresh CPU1/allBLAS1/fallback0 process, call existing `backend.probe_backend('cpu', seed=1602030, batch_sizes=(128,))`; require passed and validate. Wrap its exact raw bytes in the existing training aggregate shape and record source/lock/runtime/receipt. No MPS or full envelope rerun for d16/h<=95/F2306/batch128.
- [ ] Commit protocol/config/source before cost fitting. Run `measure` with separate engineering seeds, both searches/all nine roles and E evaluation. Retain enclosing/nested clocks, RSS, bytes, stop counts and failure evidence; numerical STOP after measured report/inventory.
- [ ] Root verifies complete member/inventory and publishes numeric1.5x16 cost/storage forecast against fresh actual resources. Lock the fixed8/R2 diagnostic before cohort fitting; cost-problem accuracy cannot change recipe/counts. If resources fail, record the concrete gap and resolve before fitting, not after E outcomes.
- [ ] Execute `run` serially for all sixteen prescribed repetitions; preserve source/receipt identity, same-parent pairing and both searches. Save each complete pair once and hash-link it; retain every failure, no automatic retries/sign stop. Return numerical STOP with full record/inventory.

## Task 3: Independently audit, replay and publish the diagnostic decision

**Files:** Diagnostic saved analyses/report, independent audit under new environment/artifact namespaces, `reports/long-probe-diagnostic.md`, README/DECISIONS and curated audit manifest.

**Consumes:** Entire locked fresh cohort and original saved predictions. **Produces:** Independently reconciled all-pair risks/effects, exact selected replay/restore evidence, deterministic report and evidence-supported next decision.

- [ ] Fresh independent auditor hashes inputs and recomputes all nine risks from E predictions/original observations and F population variance; averages repetitions within each of eight teachers. Reconcile all three planned long-reference differences and cheap/random effect without using saved row losses as sole input.
- [ ] Replay one representative short/long ranking and restore a long-probe checkpoint under the same accepted source. Check actual model/optimizer/scheduler/RNG/sampler/next-row/G prediction parity and original-parent joint restarts. No extra efficacy teacher, source changes, broad suite or new bootstrap fishing.
- [ ] Regenerate the report twice from saved evidence, compare deterministic bytes, keep full teacher/method/cost/stop tables and uncertainty limits. Root independently checks artifacts and publishes unchanged generated report, decision and meaningful commit. Keep previous pilot/report intact.
- [ ] Record complete and outstanding gates honestly. This diagnostic never substitutes for the original missing variance calibration/frozen independent confirmation or strong practical baseline suite. Collect new plan rulings with costs before handoff; preserve historical eighteen decisions by reference rather than silently changing them.
