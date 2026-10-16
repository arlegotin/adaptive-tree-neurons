# Selective Tree-Neuron Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the staged data/baseline gates, calibrate and lock an independent experiment, then deliver reproducible component and justified practical evidence, including negative outcomes.

**Architecture:** Extend the verified paired core with raw-data adapters, faithful baseline adapters, measured local HPO/timing, a pilot design calculator and a protected evaluator. Confirmation operates only on a validated frozen protocol; analysis aggregates whole teachers or grouped/temporal transfer predictions. Optional assets and extensions pass explicit evidence/resource gates rather than blocking the core or changing the hypothesis silently.

**Tech Stack:** The locked core stack, `openpyxl` for workbook formats, Optuna/TPE when practical HPO begins; isolated pinned environments for NODE, pytabkit/RealMLP, TabM, CPU boosting and conditional foundation baselines. Choose versions from verified local compatibility, not floating release names.

**Spec:** [2026-10-16-selective-tree-neurons-design.md](../specs/2026-10-16-selective-tree-neurons-design.md). This plan consumes the APIs from [the core plan](2026-10-16-selective-tree-neurons-core.md), not an existing implementation. Read both plans and the evidence record first.

## Global Constraints

- Work in the current branch. Do not create a worktree or switch branches unless a later user instruction changes this constraint.
- Training uses float32 eager PyTorch. Mathematical checks and score construction use CPU float64.
- Primary neural execution uses no reduced precision and no implicit MPS fallback.
- F fits model weights, transforms, spline knots, and foundation context. G selects checkpoints and growth/keep. V selects hyperparameters and the eligible-baseline champion. T labels are read only by the protected evaluator after protocol validation.
- No primary refit on F+G+V.
- H1-A is the sole initial confirmatory hypothesis.
- Zero-growth teachers remain in all results.
- Five observational datasets and one simulation, selected before comparative results.
- Once frozen, finish the entire independent cohort regardless of intermediate signs.
- Unresolved invalid pairs block a completed-cohort superiority claim; report counts and an incomplete/resource-limited study.
- Code and weight licenses are separate; preserve notices and do not commit downloadable weights.

Tasks are gated, not an instruction to run every method or extension automatically. For H1, Evidence Tasks 2, 3, 7–10 and the pre-freeze audit/report implementation from Task 13 are required after the core; Task 1 and practical Tasks 4–6/11 can follow the relevant WP03/WP07 gate. Timing in Task 7 supplies both H1 cost calibration and later practical calibration. Do not delay an informative component result for optional foundation assets. H3 cannot proceed until the raw data, faithful mandatory practical adapters and numeric transfer design are ready. Each task writes decision evidence and its next command; all boxes are future work.

## Execution order before the source freeze

Task numbers group deliverables; implementation and scientific execution have separate gates. The H1 execution/analysis/audit code must exist before its source identity is frozen. Adding even an unused package Python file changes the current checkpoint digest. Do not weaken that guard to make the nominal numbered order run.

Use two passes, with one reviewed implementation task/phase at a time:

1. Complete required causal adapters and search/timing (Tasks 2/3/7). Implement, test, review and commit the H1 code from Tasks 8/9/10/13, including confirmation, primary analysis, audit/report and CLI dispatch, using toy/mock/trap fixtures without the future T namespace. Record separated phases as `implementation accepted; evidence pending`; do not mark a numbered task complete while its actual evidence gate is pending.
2. With all relevant executable/schema/analysis/config/lock inputs stable and committed, execute the definitive Task 8 pilot/design and resource calibration. Preliminary exploratory measurements retain their original provenance. Recompute affected evidence if behavior, budgets, retry policy, timing boundaries, locks or analysis changes; do not rewrite historical hashes or resume old checkpoints under changed package contents.
3. Validate genuine numeric design/provenance and clean committed scientific inputs, then execute the real Task 9 freeze. Its implementation commit must precede freeze; a later protocol/docs-only commit may differ from the pinned commit only under the unchanged digest rule.
4. Execute the entire protected Task 10 cohort, then Task 13 audit/reproduction/report using that frozen code. Implementation/test preparation never grants protected access or substitutes for these scientific gates. Source changes required after T follow the existing amendment/fresh-confirmation rule.

Task 10 Steps1–3 and Task 13 Steps1–2 are therefore pre-freeze implementation dependencies; their protected execution and completed-result reporting stay after the valid freeze. Task 9 Step4's trap tests can run during implementation, but its real freeze waits for this prerequisite and genuine Task 8 outputs. Task 8 final measured design waits for the pre-freeze H1 implementation, avoiding a needless whole-pilot rerun solely because later modules were added.

Optional practical/extension source and lock changes are outside this frozen H1 checkout. Finish H1 cohort/audit before expanding it; preserve reconstructible source/locks/artifacts and never resume H1 checkpoints under a changed package digest. A later transfer or justified extension needs its own complete implementation, numeric design, source/provenance validation and separate freeze before its T access. This does not authorize a branch/worktree switch or change any hypothesis, data access or fairness rule.

## Review Focus

- Few groups, duplicate inputs and workbook shufflings: preserve source multiplicity while preventing row/group leakage; classify unsupported nested inference (Task 1).
- Vendor semantics/asset guards: preserve TabM member loss, NODE sparse functions, F-only initialization/context and full-F execution; unavailable models stay unavailable (Tasks 3–6).
- Degenerate pilot variance or infeasible power: choose the explicit estimation/resource branch; never divide by zero or stop when a sign is favorable (Task 8).
- Freeze drift and checkpoint-only interruptions: allow documented resume metadata while rejecting changed scientific code/config/data; no orphaned test access (Task 9).
- Missing confirmatory pairs or overlapping folds: keep failure counts and use the correct independent unit, never seed/fold pseudo-replication (Tasks 10–11).

## Files and public contracts

| Files | Responsibility |
|---|---|
| `data/acquire.py`, `data/transfer.py`, `evaluation/splits.py` | Official asset acquisition, raw schema/group keys, immutable nested splits. |
| `baselines/registry.py`, `baselines/greedy.py`, `baselines/node.py` | Capability records, restricted ANT-style probes, pinned NODE. |
| `baselines/boosting.py`, `baselines/modern_mlp.py`, `baselines/foundation.py` | Faithful practical adapters and named variants. |
| `evaluation/search.py`, `evaluation/timing.py` | Seeded native-recipe HPO and synchronized cost boundaries. |
| `pilot/calibrate.py`, `pilot/design.py` | Independent-teacher calibration and deterministic numeric design decisions. |
| `protocol.py`, `evaluation/protected.py` | Freeze hashes, access ledger, amendments and protected-label capability. |
| `evaluation/confirm.py`, `analysis/primary.py`, `analysis/transfer.py` | Fixed-cohort execution and correct unit-level inference. |
| `evaluation/extensions.py`, `analysis/audit.py`, `analysis/report.py` | Evidence-triggered extension registration, reproducibility and final claim report. |
| `configs/{data,pilot,transfer}.yaml`, `configs/vendor/` | Typed local studies and pinned original vendor recipes/mappings. |
| `locks/<baseline>/`, `provenance/`, `pilot/`, `protocol/`, `reports/`, `audit/` | Separate environments, source/asset identity and decision outputs. |

All source paths in the table are under `src/treeneuron/`; artifact directories are at repository root. Every adapter implements the core `FitRequest`/`Predictor` contract. Isolated environments use a small local worker CLI exchanging hashed JSON metadata and non-pickled arrays/artifacts; the core runner never imports incompatible packages. Errors/termination propagate into the run ledger, with costs retained. Native same-environment execution uses the same contract.

Run commands from the repository root. Adapter tests run through the core pytest process and invoke their configured pinned workers; optional dependencies are never imported into the core solely to run a test. A required untested adapter cannot be hidden by an unexpected pytest skip. Test assertion snippets below use test-owned fixtures whose setup is described in the preceding step; they do not add production APIs.

## Task 1: Acquire and split the fixed transfer suite without leakage

**Files:** Create `src/treeneuron/data/acquire.py`, `data/transfer.py`, `evaluation/splits.py`, `configs/data.yaml`, `tests/test_leakage.py`, `tests/test_splits.py`; extend CLI.

**Interfaces:** Consume the core `ObservationSet`/`SplitManifest`; produce `DatasetManifest` (ID, URLs/terms/raw hashes/schema/counts/features/target/group/date fields), `acquire_dataset(dataset_id: str, root: Path) -> DatasetManifest`, `prepare_transfer(manifest: DatasetManifest) -> ObservationSet`, `make_outer_splits(data: ObservationSet, dataset_id: str, seed: int) -> list[SplitManifest]`. CLI `prepare-data --manifest configs/data.yaml`. The preparer may parse raw tables to isolate label artifacts but cannot fit/evaluate on protected labels.

- [ ] **Step 1:** Write synthetic raw fixtures exercising all six schemas: Bike components/instant rejected; Parkinson ID/total UPDRS excluded and subjects disjoint; CCPP shuffled multisets not stacked; Concrete mixture groups span ages; Airfoil configuration groups omit frequency; Energy uses the five geometry columns and excludes cooling. Assert dimensions from actual acquisition separately from fixture sizes. Run `.venv/bin/python -m pytest tests/test_leakage.py tests/test_splits.py -q`; expect missing adapters.

```python
def test_parkinson_subjects_are_disjoint(parkinson_partition_groups):
    F, G, V, T = parkinson_partition_groups
    assert all(a.isdisjoint(b) for a, b in itertools.combinations([F, G, V, T], 2))
```

- [ ] **Step 2:** Implement official retrieval, license/metadata/hash archive, immutable row IDs and feature allowlists. Do not silently substitute bytes, deduplicate legitimate multiplicity or stack shuffled sheets. Add `openpyxl` (and only a verified reader if acquired legacy files require it) to the appropriate lock; no spreadsheet application. Test wrong bytes/schema/license-unverified states as blocking errors.
- [ ] **Step 3:** Implement the spec’s seeded-hash/greedy group balancing, five outer folds, inner 70/15/15 group allocation and at-least-two-groups rule. Test fewer than six development groups→descriptive-only, never a row fallback; duplicate inputs stay together. Bike uses the exact half-open quantile/date boundary and 24-hour embargo rules, and errors if a partition becomes empty. Labels cannot affect assignment.
- [ ] **Step 4:** Run the leakage/split tests and `prepare-data --manifest configs/data.yaml`. Expect source dimensions verified or documented discrepancies, exact per-fold row/group/date counts, disjoint groups and no test metrics/predictions. Unavailable sources block only their dataset with a pre-test amendment path; no convenient replacement. Append the remaining WP02 decision.
- [ ] **Step 5:** Commit `feat: prepare provenance-checked grouped and temporal transfer data`.

## Task 2: Register capabilities and implement the charged B6 probe control

**Files:** Create `src/treeneuron/baselines/registry.py`, `baselines/greedy.py`, `tests/test_greedy.py`; add `tests/test_adapters.py`; modify baseline CLI.

**Interfaces:** Produce `AdapterCapability` (method/version/status, preprocessing/categories/context/internal selection/transduction/cache/backend/fallback, unavailable reason), `capabilities() -> list[AdapterCapability]`, `run_greedy_probes(parent: TrainingState, table: ScoreTable, data: DevelopmentData, *, probe_updates: int, seed: int, paths: RunPaths) -> Allocation`. `verify-baselines --track causal|practical` returns a per-adapter matrix and nonzero status on an invalid required adapter; explicitly unavailable optional assets remain reasoned exclusions.

- [x] **Step 1:** Write a tiny two-unit fixture: each probe starts from identical weights/moments/row cursor, runs exactly the same F allowance, ranks using G, then joint insertion reloads the original parent. Assert every probe, including losers and keep, appears in cost/events; G-tied ranking uses unit ID; no V/T labels are accepted. Run `.venv/bin/python -m pytest tests/test_greedy.py tests/test_adapters.py -q`; expect missing implementations.

```python
def test_greedy_charges_losing_probes(probe_costs):
    assert set(probe_costs) == {"keep", "unit_0", "unit_1"}
    assert all(cost > 0 for cost in probe_costs.values())
```

- [x] **Step 2:** Implement the restricted B6 algorithm from spec §8. Select the fixed effective m by ranked G improvement among the same eligible units, even if improvements are negative; keep is a separate candidate/practical G decision. Do not sequentially train accepted probes into one accumulated parent. Label the method ANT-inspired restricted allocation, never full ANT reproduction.
- [x] **Step 3:** Add `SplineRidgePredictor` in the simple baseline module: per-feature knots fit F only, V chooses regularization through the search runner, no multivariable gates. Extend the common contract tests for `[N]` finite original-unit predictions, state save/reload and truthful null counters.
- [x] **Step 4:** Run adapter/greedy tests and `verify-baselines --track causal`. Expect B0–B6 verified, all probe costs counted and no oracle/target access. B7 closes in Task 3; do not mark complete until it is checked.
- [x] **Step 5:** Commit `feat: add costed validation-probed allocation and adapter capabilities`.

**Accepted:** B0–B6 implementation review passes; Task 3 now closes B7/full causal correctness readiness with fresh retained CLI evidence. The dedicated caller coordinator lifecycle item is fixed and independently reviewed in Task 3. This gate does not establish quality calibration or prediction superiority.

## Task 3: Pin and verify the actual NODE regression recipe

**Files:** Create `src/treeneuron/baselines/node.py`, `configs/vendor/node/`, `locks/node/`, `tests/test_node.py`; update provenance/capabilities.

**Interfaces:** Produce `NodePredictor(config: Mapping[str, Any])` under the shared adapter contract and a reproducible isolated worker entrypoint if needed. Config records source commit, experiment choice/bin functions, initialization rows, depth/ensemble/optimization/default mapping and terms.

- [x] **Step 1:** Inspect the pinned official experiment/config and `odst.py`; preserve original files/hash/notices. Write tests against independent CPU vendor evaluation for choice/bin forward and backward behavior and data-aware initialization; changing G/V values must not change F-derived thresholds/temperatures. A sigmoid replacement must fail parity. Run the NODE tests in its resolved environment; expect absent adapter/parity failure before implementation.

```python
def test_node_preserves_choice_backward(node_choice_gradients):
    adapter_gradient, vendor_gradient, recorded_tolerance = node_choice_gradients
    np.testing.assert_allclose(adapter_gradient, vendor_gradient, **recorded_tolerance)
```

- [x] **Step 2:** Implement the faithful regression adapter and lock a compatible native environment. Use a tested CPU path if custom operations lack MPS support. Do not infer the paper’s configuration from constructor defaults, claim its scores reproduced, or port unrelated legacy infrastructure.
- [x] **Step 3:** Run vendor examples/tests where available, the independent sparse-function tests and a tiny finite fit/save/reload. Record exact recipe discrepancies, update method capability and measured default cost; unresolved faithful execution is a named exclusion that narrows claims, not a swapped function.
- [x] **Step 4:** Run `.venv/bin/python -m pytest tests/test_node.py -q` and `verify-baselines --track causal`; expect B0–B7 valid or a documented explicit gate failure. Append WP03 causal acceptance before WP05; resource infeasibility may narrow the claim only through a recorded pre-test decision.
- [x] **Step 5:** Commit `feat: add provenance-pinned NODE regression adapter`.

**Accepted:** Implementation `41d0178`, evidence `bc8da2b`; independent task review approves spec and quality with no Critical/Important issues. The measured three-update full-size endpoint remains quality-ineligible. Two minor observations (row-trace bookkeeping and expected-warning capture) are retained for Task 7/final review.

## Task 4: Verify native boosting and the simpler local-affine alternative

**Files:** Create `src/treeneuron/baselines/boosting.py`, `configs/vendor/{catboost,xgboost,lightgbm}/`, corresponding `locks/` directories, `tests/test_boosting.py`.

**Interfaces:** Produce `CatBoostPredictor`, `XGBoostPredictor`, `LightGBMPredictor`, each with the shared adapter API. Export actual library parameters and runtime/thread/native-wheel identity. LightGBM config explicitly carries serial learner, `linear_tree` and F-fitted transform state.

- [ ] **Step 1:** Write tiny numerical/mixed-frame contract tests for shape, finite original-unit outputs, exact save/reload, G-only stopping and rejected unsupported categories. Require XGBoost histogram CPU, CatBoost native universal2/arm64 operation, LightGBM serial CPU. For linear-tree mode inspect coefficients after the first tree; a constant-first-tree test must not falsely require first-tree coefficients. Run adapter tests in each environment; expect absent adapters.

```python
def test_linear_tree_has_nonfirst_affine_leaves(linear_leaf_coefficients):
    assert len(linear_leaf_coefficients) > 0
    assert any(np.any(np.asarray(coefficients) != 0) for coefficients in linear_leaf_coefficients)
```

- [ ] **Step 2:** Resolve compatible stable releases, verifying the brief’s CatBoost snapshot and current LightGBM/XGBoost guidance. Archive vendor default anchors and code/package terms. Check OpenMP/native runtimes without privileged toolchain changes. Implement B10/B11/B14, passing `linear_tree` in Dataset construction and training. Retain the constant-leaf counterpart.
- [ ] **Step 3:** Implement exact initial search supports from spec §8 as typed config records, not mutable vendor defaults. Test requested/resolved settings, F-only numerical scaling for linear leaves, and default/linear higher-memory counters. No broad search yet.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_boosting.py -q`, vendor checks and faithful tiny/default fits, recording actual local time/memory and unavailable reasons. A failed native runtime remains a failure/exclusion pending a safe documented remedy; no replacement with another family.
- [ ] **Step 5:** Commit `feat: add faithful native boosting and linear-leaf controls`.

## Task 5: Preserve strong neural-baseline training semantics

**Files:** Create `src/treeneuron/baselines/modern_mlp.py`, `configs/vendor/{realmlp,tabm}/`, `locks/{realmlp,tabm}/`, `tests/test_modern_mlp.py`.

**Interfaces:** Produce `RealMLPPredictor` (RealMLP-TD regression through pytabkit) and `TabMPredictor`, both shared adapters. Produce `tabm_member_loss(predictions: Tensor, target: Tensor) -> Tensor`; retain k/embedding/default/tuning/source settings and any documented F-internal split counts.

- [ ] **Step 1:** Write the loss counterexample: members `[-1,1]`, target 0 give mean member squared loss 1, while squared error of mean is 0; assert training loss 1 and inference average 0. Test k=32, original-unit `[N]` predictions, F-only learned embedding boundaries/preprocessing and parity with official miniature examples. Run tests in chosen environments; expect absent adapters.

```python
def test_tabm_trains_member_loss():
    members, target = torch.tensor([[-1., 1.]]), torch.tensor([0.])
    assert tabm_member_loss(members, target).item() == 1.0
    assert members.mean(1).item() == 0.0
```

- [ ] **Step 2:** Implement faithful RealMLP-TD default anchors and documented search, preserving native preprocessing/embeddings; implement TabM plain and piecewise-linear embedding recipes with default LR 0.002/weight decay 0.0003 and supplied search supports after checking pinned docs. Archive originals plus explicit local mappings/reasons. Do not label a generic MLP RealMLP or remove ensemble cost.
- [ ] **Step 3:** Run vendor tests/examples, adapter save/reload and tiny fits; record effective F rows when the vendor internally splits F, ensembles/storage, actual defaults and source notices. Compare TabM output/member semantics on CPU independently before any MPS performance claim.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_modern_mlp.py -q` and practical adapter verification. Expect B8/B9 faithful paths and measured defaults, or explicit unavailable status restricting H3. Append mandatory practical-family readiness alongside Task 4’s evidence; HPO strength is still uncalibrated.
- [ ] **Step 5:** Commit `feat: add faithful RealMLP and TabM regression recipes`.

## Task 6: Gate conditional foundation baselines before acquiring weights

**Files:** Create `src/treeneuron/baselines/foundation.py`, `configs/vendor/{tabicl,tabpfn}/`, conditional `locks/` directories, `tests/test_foundation.py`; update source/asset and capability manifests.

**Interfaces:** Produce `check_asset_eligibility(method_id: str, request: FitRequest, system: Mapping[str, Any]) -> AdapterCapability`, `TabICLPredictor`, `TabPFNPredictor` only for authorized, verified local assets. Eligibility returns available/blocked/unavailable plus checkpoint ID/hash/terms, full-F row/backend/memory/disk constraints. Model load cannot run before an affirmative eligibility record.

- [ ] **Step 1:** Write fake-loader tests: no authorized terms→no download; insufficient forecast headroom→no load; full-F exceeds supported CPU guard→unavailable; T/G labels cannot enter context; smaller context/ensemble changes variant ID. Simulate a gated agreement and assert blocked status rather than automatic acceptance. Run tests without real weights; expect missing gate APIs.

```python
def test_gated_weights_are_not_downloaded(unauthorized_asset_case):
    capability, download_spy = unauthorized_asset_case
    assert capability.status == "blocked"
    download_spy.assert_not_called()
```

- [ ] **Step 2:** Verify official regression checkpoint/defaults, exact weight/code terms and supported backend/CPU row guards. The brief’s TabICLv2 filename and TabPFN-3.5 Apple version/row allowance are leads to check, not assumed installed facts. Forecast download, unpacked weights, cache and context against current 36 GiB/available disk before acquisition. Do not bypass asset gates or system guards.
- [ ] **Step 3:** Only for eligible assets, implement faithful official context construction/caching/ensemble prediction, source-hashed weights outside Git, F-label-only context and output parity for documented batching/offload. Test transductive batch-composition dependence, cached/uncached parity and separate cache-build time/bytes. Serial member execution or smaller prediction batches require parity; reduced context/ensemble/model/precision is a separately named variant.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_foundation.py -q` and eligible real miniature/default checks; record unavailable models/folds explicitly. No API-only Thinking substitute or assertion of beating an unexecuted method. Acquisition/pretraining/retained-data costs remain distinct from local fit.
- [ ] **Step 5:** Commit `feat: gate local foundation-model assets and faithful adapters`. This task can validly deliver tested gates and documented exclusions without downloading a model.

## Task 7: Implement costed HPO and calibrated timing

**Files:** Create `src/treeneuron/evaluation/search.py`, `evaluation/timing.py`, `tests/test_search.py`, `tests/test_timing.py`; add Optuna to the practical lock when first needed.

**Interfaces:** Produce `SearchBudget` (seconds, trial/order policies, G stopping, official anchors), `SearchResult` (V winner plus all trials/costs/sampler state), `search(method_id: str, data: DevelopmentData, budget: SearchBudget, seed: int, paths: RunPaths) -> SearchResult`; `TimingPolicy` (boundaries/batches/threads/warmup precision/repetitions/block order), `measure_predictor(predictor: Predictor, X: Any, policy: TimingPolicy) -> dict[str, Any]`. A tuning runner sees V but never T; `FitRequest` still contains only F/G.

- [x] **Step 1:** Write deterministic mock-clock tests requiring official defaults first and retained even when not best, all failed/rejected trials charged, immutable sampler resume, G used for checkpoints and V for HPO/champion only. Timeout during an official default records unavailable-at-budget status, not a misleading partial model win. Run search/timing tests; expect missing APIs.

```python
def test_mps_timing_synchronizes_both_boundaries(timing_trace):
    assert timing_trace == ["sync", "start", "predict", "sync", "stop"]
```

- [x] **Step 2:** Implement seeded TPE with archived state and exact search supports; serialize isolated workers, no CPU oversubscription. Count preprocess/base/score/probe/validation/continuation/compilation in method budget; data acquisition and weights acquisition separate. A static competitor can spend saved allocation cost. Run exploratory doubling budgets, expand improving ceilings/search boundaries before freeze and retain official anchors.
- [x] **Step 3:** Implement `perf_counter_ns` boundaries with MPS synchronization, fresh process import/load/first prediction, host→device→host and kernel-only warm modes, complete batching and cached/uncached context modes. Tests with an injected sync/timer must show calls on both sides of the measured region. ABBA/balanced random blocks and rolling warmup are explicit; excluded/warmup work is retained. RSS/MPS counters stay separate.
- [ ] **Step 4:** Measure on this laptop: low/common quality-budget candidates, timing variance/warmup stability, smallest safe batch reaching 95% peak throughput, median/tail repetition precision and default plateau curves. Select actual numeric policies in WP05 from these records; no global fixed time invented by this plan. Run `.venv/bin/python -m pytest tests/test_search.py tests/test_timing.py -q` and rebuild one timing summary from raw blocks.
- [x] **Step 5:** Commit `feat: account for local search costs and synchronized latency`.

**Implementation phase accepted; evidence pending WP05:** The independent review and scoped fix review close method interleaving, block-based precision and resumed-search cost lineage (`75fde42`, evidence `e90b315`). The current fix has35 passing covering tests; the preceding source passed708 full-suite tests. Step4's final quality/default-budget, training-batch and precision policies remain open until all pre-freeze H1 code exists and genuine pilot evidence supports them. Bounded timing/search receipts do not complete that scientific gate. Continue Task8 implementation, then Tasks9/10/13 implementation before the definitive pilot/freeze.

## Task 8: Calibrate variance, test size and the fixed confirmation design

**Phase status:** Implementation accepted after independent review and fix1 (`a84ec74`); scientific WP05 evidence is pending. The one-teacher engineering invocation is not the genuine variance pilot. Tasks9/10/13 must be implemented and reviewed before definitive pilot/design execution. Steps4/5 remain open until that evidence exists.

**Files:** Create `src/treeneuron/pilot/calibrate.py`, `pilot/design.py`, `configs/pilot.yaml`, `tests/test_design.py`; extend CLI `calibrate --config PATH`. Outputs: `pilot/design_decision.md`, `pilot/resource_forecast.json` and raw pilot ledgers.

**Interfaces:** Produce `PilotSummary` (`independent_teachers`, teacher means/repetition variance, paired-row variance, whole-teacher SD interval, two-batch runtimes, `h1_mean`, H1/static-control intervals, precision/power outputs), `DesignDecision` (`branch`, `K_calc: int | None`, `pilot_teachers`, `K`, `repetitions`, `n_T`, `delta_plan: float | None`, numeric policies/budgets, forecast/limitations, evidence hashes), `calibrate(config: ExperimentConfig) -> PilotSummary`, `choose_design(summary: PilotSummary, constraints: Mapping[str, Any]) -> DesignDecision`. `constraints` records actual physical feasibility and measurement precision, not a new application utility threshold.

- [x] **Step 1:** Write pure calculation fixtures for every branch: 8 teachers/2 repetitions initial minimum; SD interval ratio≤2 and runtime stability≤20% required; favorable **and unfavorable** H1 intervals yield half absolute mean; unresolved H1 falls back to static-partial versus upper-width control; both unresolved→fresh estimation cohort matching completed pilot counts; zero/undefined variance→equivalence/estimation, never division by zero. Include the low-SD/large-effect fixture where the normal approximation gives `K_calc=1`: actual `K` must be at least the completed accepted pilot teacher count. Test missing costs/required baselines block design and singleton/undefined inference cannot authorize superiority. Run `tests/test_design.py`; expect missing calculators.

```python
def test_negative_pilot_uses_absolute_planning_effect(negative_pilot, feasible_constraints):
    decision = choose_design(negative_pilot, feasible_constraints)
    assert decision.delta_plan == 0.5 * abs(negative_pilot.h1_mean)
    assert decision.K >= negative_pilot.independent_teachers >= 8
```

- [x] **Step 2:** Implement hierarchical aggregation and whole-teacher bootstrap diagnostics, doubling teacher counts (8→16→32...), fixed bootstrap seed and endpoint Monte Carlo diagnostics. Set pilot recipe across teachers rather than tuning the proposed fork per teacher. Use separate exploratory E observations for paired effect/row-loss precision, never a protected T or reusable confirmatory teacher. Include B6 probe ranking and B2/B3 attribution anchors. Do not stop variance calibration on significance.
- [x] **Step 3:** Implement exact rules from spec §9 and the [pre-freeze cohort correction](../../../protocol/amendments/2026-10-16-minimum-confirmation-cohort.md): upper 95% bound for `s_D`, normal-approximation `K_calc`, actual `K=max(K_calc, summary.independent_teachers)`, resampling power and finite-sample inference diagnostics, conservative `n_T=max(1,ceil(100*var_row/var_between))`, unresolved denominator branch, within-/between-repetition cost decision. Validate the accepted pilot count against its existing minimum; record both counts and any extra runtime/storage cost. The floor does not guarantee 95% coverage. For sole H1, use `z_0.975+z_0.8`; additional families must carry an explicit adjusted level. Publish detectable effects/precision–cost curve. Unsupported inference or an infeasible floored workload becomes a declared estimation/blocked claim before confirmation, not silent truncation.
- [ ] **Step 4:** Run exploratory sensitivity/control/timing calibration without a mandatory full cross-product. Resolve numeric parent/continuation ceilings, patience/validation interval, logical/physical batch, teacher/repetition/test counts, bootstrap iterations, retries, HPO budgets, timing precision and transfer block length where relevant. Record axes changed/boundary expansions and resource forecast, including failures. `DesignDecision` rejects absent numeric outputs for its selected study; no placeholder freeze.
- [ ] **Step 5:** Run `.venv/bin/python -m pytest tests/test_design.py -q`, `calibrate`, regenerate `pilot/design_decision.md` from raw measurements, and append WP05 evidence. Commit `feat: derive locked study sizes and budgets from local pilot evidence`. If the pilot is unresolved/resource-limited, commit its explicit design branch and limitations; do not imply powered confirmation.

## Task 9: Freeze identity and protect test access

**Files:** Create `src/treeneuron/protocol.py`, `evaluation/protected.py`, `tests/test_protocol.py`, `tests/test_protected.py`; extend CLI `freeze --design PATH` and amendment handling. Generate `protocol/frozen.yaml` only when this task’s real gate succeeds.

**Interfaces:** Produce `FrozenProtocol`, `freeze(design: DesignDecision, sources: Mapping[str, Any], output: Path) -> FrozenProtocol`, `validate_frozen(path: Path, current: Mapping[str, Any]) -> FrozenProtocol`, `ProtectedEvaluation` (access grant bound to protocol/study/run identity), `open_test(protocol: FrozenProtocol, manifest: SplitManifest, *, run_id: str) -> ProtectedEvaluation`. Only `ProtectedEvaluation` reads T labels and writes test predictions after an append-only access event; no token self-construction by adapters.

- [ ] **Step 1:** Write tests requiring freeze to reject draft/missing numeric design, source/lock/config/split/analysis drift, dirty executable files and reused confirmation namespaces; any failure leaves no test predictions/access success. Changing documentation or committing the frozen artifact alone is allowed only if execution/scientific input digests match. Resume paths require existing run linkage; repeated accidental test invocation is idempotent or a recorded error, never a new unlogged selection. Run protocol/protected tests; expect missing guards.

```python
def test_rejected_freeze_never_reads_test_labels(drift_case):
    access_error, label_reader, predictions_path = drift_case
    assert access_error is not None
    label_reader.assert_not_called()
    assert not predictions_path.exists()
```

- [ ] **Step 2:** Implement canonical hash/source-commit and execution-tree digest accounting from spec §9. Record both pinned source commit and actual checkout commit. Snapshot required configs/locks/vendor assets/splits/analysis policy. The source digest covers execution/analysis/schema inputs and rejects untracked executable overrides; it excludes the generated frozen artifact/docs/run artifacts so committing the protocol does not invalidate itself. Immutable prior freezes/amendment identities remain accessible.
- [ ] **Step 3:** Require a fully populated selected-study design; a synthetic freeze does not unlock transfer until the transfer section is separately frozen. Implement per-study access ledger and separate synthetic T generation namespace. Allow only documented operational resume fields without amendment; changed scientific behavior relabels affected old results exploratory and requires fresh confirmation. This is procedural protection, not encryption against the file owner.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_protocol.py tests/test_protected.py -q` with trap label readers and corrupted files; expect zero reads on rejection and auditable successful access only after matching freeze. Exercise the real `freeze --design pilot/design_decision.md` only after the pre-freeze implementation prerequisite above, clean committed executable inputs and genuine completed pilot outputs; inspect every frozen required field before any T access. Append WP05 final acceptance.
- [ ] **Step 5:** Commit `feat: freeze scientific identity and audit protected test access` before the real freeze. Commit the later successful frozen artifact/decision separately under the unchanged-digest rule. If numeric design or pre-freeze implementation is pending, record implementation accepted/evidence pending and commit only the tested implementation plus blocking decision; do not mark the full task complete or create a pretend frozen protocol.

## Task 10: Complete the independent H1 cohort and teacher-level analysis

**Files:** Create `src/treeneuron/evaluation/confirm.py`, `analysis/primary.py`, `tests/test_confirmation.py`, `tests/test_primary.py`; extend CLI `confirm --protocol PATH --study synthetic`.

**Interfaces:** Produce `run_confirmation(protocol: FrozenProtocol, study: str) -> Path`, `aggregate_teachers(predictions: Any, runs: Any) -> Any`, `analyze_primary(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]`. Teacher aggregation averages optimizer/allocation repetitions within each teacher/data unit, then calculates `D=R_random-R_top` and whole-teacher uncertainty. Keep unnormalized and degenerate-risk statuses explicit.

- [ ] **Step 1:** Write fixed-cohort mock tests: K×repetition units execute independent of intermediate signs; zero-growth and unfavorable units stay; interruption resumes the declared unit once; paired missing/invalid runs remain counted and block completed-cohort superiority. `test_repetitions_are_not_independent_teachers` duplicates seed rows and asserts effective sample count and bootstrap input are teacher count. Singleton and zero/undefined-variance inputs must not turn a degenerate point interval into a superiority result. Run confirmation/primary tests; expect missing implementations.

```python
def test_repetitions_are_not_independent_teachers(three_teachers_two_repetitions):
    rows = aggregate_teachers(*three_teachers_two_repetitions)
    assert len(rows) == 3
```

- [ ] **Step 2:** Implement serial protected evaluation, symmetric frozen finite retries and immutable saved predictions/metrics. Failures retain exception/resource/linkage; no new seed substituted for a losing or failed teacher. Compare measured total work with forecast. Run each fixed causal/attribution anchor with its declared allowance; no covertly reassigned unused updates.
- [ ] **Step 3:** Implement paired mean/95% whole-unit bootstrap, Student-t sensitivity, median/full effects, guardrails/noise-free diagnostics and declared Holm families only. Tests with saved toy predictions require exact metric reconstruction, same-model zero D, seed-duplication invariance of independent count and H1 sign interpretation. Bootstrap interval endpoints use the frozen reproducible precision policy.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_confirmation.py tests/test_primary.py -q`, then the entire frozen cohort. Regenerate the analysis solely from saved predictions/run identities. Acceptance is a completed audited cohort or honestly incomplete/resource-limited record, never a positive p-value. H1 positive but static/widening equal yields a placement-only conclusion; narrow negative and wide inconclusive intervals remain distinct.
- [ ] **Step 5:** Append WP06 decision and commit `feat: execute and analyze independent allocation confirmation` plus curated result/provenance indexes. Do not commit bulk T artifacts or select favorable units for the report.

## Task 11: Evaluate fixed-suite practical tradeoffs with nested selection

**Files:** Create `src/treeneuron/analysis/transfer.py`, `configs/transfer.yaml`, `tests/test_transfer.py`; extend confirmation and protected interfaces without changing existing synthetic behavior.

**Interfaces:** Produce `run_transfer_fold(protocol: FrozenProtocol, dataset_id: str, split: SplitManifest, repetition: int) -> Path`, `analyze_transfer(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]`; CLI `confirm --protocol PATH --study transfer`. The run manifest identifies per-fold F/G/V, HPO trials, V-selected champion and practical G-selected keep/grow pipeline before protected T evaluation.

- [ ] **Step 1:** Write tests for whole-group/forward-only folds, embargo, no F+G+V refit, full-F foundation context, method-native preprocessing, champion selected using V only and complete rival table. Duplicating overlapping outer folds must not increase independent dataset count. Energy must be absent from the five-observational aggregate. Failed/unavailable baseline folds stay explicit. Run `tests/test_transfer.py`; expect missing orchestration.

```python
def test_simulation_is_outside_observational_aggregate(observational_dataset_ids):
    assert len(observational_dataset_ids) == 5
    assert "energy_efficiency" not in observational_dataset_ids
```

- [ ] **Step 2:** Complete Tasks 1 and 4–7, default/quality calibration, and a numeric transfer freeze before T. Record why transfer remains informative after WP06, including negative H1 if relevant. If a dataset cannot support nested grouping, record descriptive status/pre-test amendment; no automatic dataset replacement or missing-model victory.
- [ ] **Step 3:** Implement all selection inside each outer fold: F fit, G checkpoint/growth, V HPO/champion, protected T once for each locked pipeline. Fit permitted vendor preprocessing within its documented F access. Retain raw/macro/regime/tail outcomes, complete method search costs, cold/warm/cache timing and method-specific acquisition disclosures. Run fixed repetitions serially.
- [ ] **Step 4:** Implement equal-dataset observational risk aggregate, per-dataset effect tables, grouped conditional intervals and Bike moving-date-block intervals with development-chosen length and half/double sensitivity. Show HPO/training variability separately; no universal algorithm-generalization interval or seed/fold t-test. Rebuild from saved paired predictions and raw timing blocks; run `.venv/bin/python -m pytest tests/test_transfer.py -q` and the frozen study.
- [ ] **Step 5:** Append WP07 decision and commit `feat: evaluate fixed-suite local prediction and resource tradeoffs`. State executed families/full-F availability and V-selected rival explicitly; do not import published scores into local results.

## Task 12: Register only an evidence-triggered extension

**Files:** Create `src/treeneuron/evaluation/extensions.py`, `tests/test_extensions.py`, `protocol/extensions/<id>.yaml` only for a justified extension; additional modules/tests require their own approved design/plan.

**Interfaces:** Produce `register_extension(evidence: Mapping[str, Any], protocol: Mapping[str, Any]) -> Path`; CLI `run-extension --protocol PATH` dispatches only an explicitly implemented and validated extension, otherwise errors. No generic recursive/hard/sparse NAS engine is part of this task.

- [ ] **Step 1:** After WP06/WP07, choose a branch only if spec §13 evidence predicts a discriminating measurement. Write a gate test rejecting absent parent result, vague motivation, reused confirmatory teachers/T, missing new equations/interfaces/baselines, missing resource forecast or extension implementation.

```python
def test_extension_rejects_reused_confirmation_seeds(reused_seed_extension):
    evidence, protocol = reused_seed_extension
    with pytest.raises(ValueError, match="seed"):
        register_extension(evidence, protocol)
```

- [ ] **Step 2:** Register immutable extension identity/claim/fresh seeds/analysis/fairness and capability checks. For recursive, repeated growth, pruning, temperature, sparse/hard execution or strong-recipe insertion, write a separate concrete design and implementation plan before coding that behavior. The current plan does not guess its outcome or authorize silent baseline changes.
- [ ] **Step 3:** Run `.venv/bin/python -m pytest tests/test_extensions.py -q`; expect absent implementation to fail closed and valid registered metadata to preserve parent evidence and new identity. If no branch has evidence, record “not pursued” with reason and skip extension code entirely.
- [ ] **Step 4:** Commit `feat: gate extensions on recorded evidence and fresh protocols` only if this task is needed. An extension’s actual experiment has its own review/test/commit cycle and may legitimately end negatively.

## Task 13: Audit, reproduce and write a decision report

**Files:** Create `src/treeneuron/analysis/audit.py`, `analysis/report.py`, `tests/test_audit.py`, `reports/final.md`, curated `audit/` metadata; extend CLI `audit`, `report`; update README/DECISIONS.

**Interfaces:** Produce `audit(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]`, `render_report(audit_result: Mapping[str, Any], output: Path) -> None`. Reports consume validated artifacts and protocol-defined analyses, never editable summary-score inputs. An invalid/incomplete audit emits blocked/incomplete conclusion status and reasons, not a success table.

- [ ] **Step 1:** Write tamper/missing-run fixtures: changed prediction hash, absent failed candidate, manual metric mismatch, dirty source identity, missing terms and invalid resume evidence must be detected. Include a valid negative cohort and an incomplete cohort; only the complete valid result receives a completed scientific interpretation. Run `.venv/bin/python -m pytest tests/test_audit.py -q`; expect missing audit/report APIs.

```python
def test_audit_keeps_negative_result_and_blocks_missing_pair(audit_cases):
    valid_negative, missing_pair = audit_cases
    assert valid_negative["valid"] is True
    assert missing_pair["valid"] is False
```

- [ ] **Step 2:** Implement independent original-unit metric recomputation, all-run accounting, source/lock/asset/checkpoint/protocol verification, timing boundary validation and exact claim-family checks. Render per-task regressions/exclusions/costs/uncertainty/novelty limits and failure classifications. Keep Energy simulation separate and Track C outside local measured tables.
- [ ] **Step 3:** From a clean locked environment rerun reference/kernel checks, interrupted growth/resume and at least one representative paired pipeline. Verify existing protected artifacts without silently creating a new adaptive test selection; any new confirmatory evaluation follows its protocol/access ledger. Revisit R12 and closest prior art; unresolved access preserves the novelty hold. All evidence paths/hashes must survive a fresh checkout plus separately restored allowed artifacts.
- [ ] **Step 4:** Run the relevant complete tests, `audit`, and `report`; rebuild reports from raw records twice and compare deterministic content apart from declared generation metadata. Report discrepancies before scientific claims. Include actual forecast deviations, effective counts, unavailable strongest models, achieved precision and evidence-supported continue/revise/narrow/replicate/stop decision.
- [ ] **Step 5:** Append WP09 decision and commit `docs: publish audited reproduction and research decision`. Publish nothing externally without separate authorization. A useful negative/replication result satisfies the project when the declared study and audit are complete.

## Gate and requirements coverage

| Spec requirement | Owning tasks |
|---|---|
| §1–§3 scientific scope, evidence status, platform, H0/H1/H2/H3 | Core 1/3/8/12; Evidence 8–11/13 |
| §4 compact unit/algebra/counts/activation and hard-gap diagnostic | Core 3–4/8/11 |
| §5 moment/normalization/eligibility/Adam/direction allocation | Core 7; Evidence 2/8/10 |
| §6 paired updates, optimizer, atomic resume, G keep/grow | Core 9–12; Evidence 9–11/13 |
| §7 teachers/controls/F-only transforms/group/time splits | Core 5–6; Evidence 1/8/11 |
| §8 B0–B14, licenses/native recipes/HPO/fair costs | Core 6/11–12; Evidence 2–7/11 |
| §9 pilot sizing, freeze, uncertainty, claims/failure cohort | Evidence 8–11/13 |
| §10 synchronized timing, unified memory, complete costs | Core 1/8/12; Evidence 4–8/11/13 |
| §11 strict adapters/configuration/records | Core 2/5–6/9; Evidence 1–7/9/13 |
| §12 WP gates and decision records | Every task’s acceptance/commit; README and DECISIONS |
| §13 failure/extension/report rules | Core 12; Evidence 8/10–13 |
| §14 disclosed choices and measured gate outputs | Core 2/5/12; Evidence 1/8–9 |

An unimplemented evidence-triggered extension is a declared non-goal until its gate passes. Optional unavailable baselines remain visible limits. These are intentional conditional scopes, not implied completed work.
