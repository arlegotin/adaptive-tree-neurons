# Task 7 implementation report

Status: DONE_WITH_CONCERNS for the authorized pre-freeze implementation phase, ready for independent review. Full-quality/default-plateau numerical policies remain evidence_pending_WP05; whole Task7 is not scientifically complete.

## Scope and interfaces

- `evaluation/search.py`: frozen `SearchBudget`, `SearchResult`, `search(method_id, data, budget, seed, paths)`, `ReplayTPE`, `produce_search_backend_receipt`, and `expansion_diagnostics`.
- `SearchBudget.seconds` is mandatory; max_trials is an explicit additional cap. Anchor order is fixed to official/configured first, then seeded TPE. G patience/cadence/stopping come from the immutable resolved experiment; B7 additionally requires explicit max_updates. Anchor configs and exact supports are resolved by `_recipe` and retained in trial0/sampler JSON. No user-supplied dict can relabel an arbitrary B7 architecture as the official shallow anchor.
- Implemented search paths are Mean/Ridge/SplineRidge and B1–B7. B2 lower/upper/widen, B3 partial/full centered/direct leaf, and B5 adam/sgd/gaussian recipes remain explicit. Practical pending families fail clearly. B7 currently has the faithful immutable official anchor only, with no invented unverified NODE tuning support.
- `evaluation/timing.py`: frozen `TimingPolicy`, `measure_predictor`, `measured_call`, `rebuild_summary`, hashed `write_timing`/`read_timing`, `measure_cold_predictor`, and explicit `ResidentCausalPredictor` view.

## Data, state, identity and worker contract

The search coordinator has DevelopmentData and alone computes V MSE. Workers receive a hash-wrapped JSON request and NPZ containing F/G X/y, V X, and row IDs; no V labels. The worker reconstructs and validates original F/G arrays, row/group IDs and SplitManifest commitments, including the original V hash commitment, then creates a redacted DevelopmentData view whose V object raises on access. This preserves original checkpoint identities rather than inventing zero V labels. FitRequest still contains only F/G arrays. No T is generated or read.

Every worker is a fresh serial interpreter with CPU intra-op1/inter-op1, native BLAS thread environment limits and fallback disabled before imports. The unchanged core CPU producer is invoked under that actual runtime to generate a new receipt. Existing historical14-inter-op CPU evidence is retained byte-for-byte. Worker input/response/prediction hashes, source/core-lock identity and the exact backend-receipt byte hash are verified. Unknown request/policy keys are rejected. All model snapshots are current-source; the worker protocol uses only finite JSON and non-pickle arrays. Existing causal checkpoints retain the established locally-produced weights_only tensor payload mechanism.

Optuna4.5.0 is resolved with hashed transitive dependencies in `locks/search/requirements.txt`. The ignored `environment/search-site` overlay is appended after core site-packages, so shared native dependencies retain their actual core identities. `environment.json` checks actual imported versions. License files/metadata are archived. Core `uv.lock`, NODE environment lock, numerical vendor files and backend numerical producer are unchanged. Adding package files changes source identity and requires fresh parent checkpoints.

Sampler state is immutable hashed JSON containing exact ordered ask/tell records, official anchors, failed outcomes, support, seed and source/runtime/data identities. Loading replays every proposal under pinned Optuna and verifies equality; no RNG/study pickle or mutable database. Resume creates a new run and wall budget, links the prior sampler through archived SearchBudget, and retains historical objectives for sampling without silently making historical models eligible in the new attempt. Supports and seed must match.

## Cost, availability and lifecycle

The search clock starts before data publication/sampler work and encloses startup/imports, all transforms, fresh base/parent work, score construction, all B6 keep/losing/failed probes, continuation, G checkpointing, model publication, V prediction/scoring, failed and rejected attempts, and trial/final sampler publication. It stops before terminal metrics/manifest publication. Per-trial clocks are nested diagnostics and exclude their trailing record publication; they are never added to the enclosing clock. Data and weights acquisition are outside this boundary.

Static competitors receive the same seconds and can execute additional trials with saved allocation cost, subject only to explicit max_trials. B4 currently executes and pays for the full score table; there is no claim of an implemented faster standalone sampled-ID scorer. All causal parents are fresh. B6's dedicated coordinator gets provenance before terminalization; the generic probe helper does not close the broad search run.

The coordinator rejects an otherwise completed trial if the budget expires during V scoring. Worker timeout kills/waits the process, closes known nonterminal children as interrupted, and retains original error/stdout/stderr. Native B7 endpoints with no official G report are unavailable_at_budget and cannot be a V winner. Failed/rejected/default records remain retained. Unexpected/terminal-publication failures preserve original exception type/message and known metadata; additional publication errors are attached rather than replacing the original error.

## Timing boundaries

`perf_counter_ns` encloses synchronized prediction, with MPS synchronization on both sides and inference mode. Host-to-host calls include all remainder batches, preprocessing and original-unit host output. Resident causal calls use explicitly prepared resident standardized float32 inputs and return standardized resident outputs: transformation, transfers and target inversion are outside that boundary and recorded. Unsupported resident/cache/device paths fail explicitly; CPU paths never claim MPS timing.

Fresh-process cold timing separately measures imports, safe model load and the first full host prediction. Serialization outside the boundary and launch-to-exit wall cost are retained separately. This does not evict OS disk/page caches. Warm timing retains rolling warmups, excluded work/reasons, ABBA or seeded mirrored random blocks, all actual repetitions, and median/p95 diagnostic bootstrap precision. The current diagnostic bootstrap is iid and explicitly does not correct correlation; it is not a final inferential/plateau certificate. Sampling caps or unstable warmup remain unresolved rather than passing by cap exhaustion.

Reconstruction calculates the smallest successful measured batch reaching95% of the tested peak median complete-call throughput. Native pressure guards before each tested batch preserve9GiB reserve plus512MiB; unavailable counters are explicitly uncertified. RSS high-water and MPS allocated/driver values remain separate overlapping counters, never added. Timing records bind source/core lock/runtime/input and available fitted-model identities. Predictors without exposed fitted identity get null plus reason.

## TDD and verification

The first requested command `.venv/bin/python -m pytest tests/test_search.py tests/test_timing.py -q` produced9 failures on missing search/timing APIs (0.21s). The first implementation reached9 passed (1.84s). Subsequent meaningful REDs covered real causal workers, timeout after V scoring, cold timing,1/1 receipt generation, explicit resident parity, alternative recipe schemas, retained prediction failure records, immutable search resume, expansion flags, timeout child closure and backend receipt substitution. Failures were corrected without loosening numerical tolerances. Intermediate issues included the reserved ledger status field, JSON tuple/list identity normalization, backend TorchVersion normalization at JSON publication, and receipt inter-op mismatch; no numerical producer was bypassed.

Initial focused run:28 passed in25.10s. Full-suite results and the subsequent measured warmup defect are recorded below. Tests use a real fresh backend receipt and real isolated causal fits, not receipt fixtures or an all-method historical predictive rerun. Original RED output is retained in the conversation tool trace; the filesystem report summarizes it and does not pretend to be a saved original full transcript.

## Limits and triage

This is pre-freeze machinery plus bounded engineering evidence. Full-quality/default learning curves, final low/common study budgets, robust final median/tail repetition policies and final batch decisions remain evidence_pending_WP05, after Tasks8/9/10/13 pre-freeze implementation under final identities. Exact-boundary/improving-ceiling diagnostics always leave freeze_ready=false. No full long NODE fit, full17-method inventory, independent cohort, protected T, raw transfer download or model weights were run.

Task3 M1 per-visit row-trace overhead and M2 expected vendor warnings remain retained triage because this task does not modify NODE or run long defaults. Existing numerical files and warning visibility are preserved. No efficacy claim is made from a bounded fixture. Root owns independent review, ledger and task completion checkboxes.

## Full-suite compatibility finding

The first full suite completed with17 new-search failures,689 passed,48 existing skips and28 existing warnings in254.51s. Every failure was `Search dependency mismatch: tqdm`; `tests/node_reference.py` inserts the separately pinned reference site before the appended search overlay, exposing4.67.1 against the initial search resolution4.70.1. No existing tests failed. The original log remains `artifacts/evidence7-search-timing/final-full-suite.txt`, and the initial search environment manifest is preserved separately.

Only the search requirements were re-resolved with tqdm==4.67.1, matching the existing NODE reference lock; no check was bypassed and no core/NODE lock or vendor file was edited. The combined NODE/search/timing command passes47 tests with28 unchanged warnings in26.72s. A fresh clean coordinator interpreter validates Optuna4.5.0 and imports tqdm4.67.1 from the search overlay; actual isolated causal workers also pass the combined tests. A second full-suite run is justified by this concrete newly exposed compatibility failure and lock change, rather than an unchanged repeated confirmation.

## Final implementation verification and commit

- Implementation commit: `d6248aa` (`feat: account for local search costs and synchronized latency`).
- Final full suite: `.venv/bin/python -m pytest -q` →706 passed,48 existing sandbox/MPS skips,28 existing vendor warnings in305.98s, exit0. Raw log: `artifacts/evidence7-search-timing/final-full-suite-after-compatibility.txt`.
- Combined covering command: `.venv/bin/python -m pytest tests/test_node.py tests/test_search.py tests/test_timing.py -q` →47 passed,28 existing warnings in26.72s, exit0.
- Package source SHA256: `a64b5f98ac9d533fe35df1c5b81f4582aeb11336eda5d0290fb9cad69756dd6e`.
- Core lock SHA256: `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5` (unchanged).
- Backend numerical producer SHA256: `237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e` (unchanged).
- NODE lock SHA256: `89ce6a4b559114eb759d31092d88287ff5e0cd04672bdcabd9143c3a4e56b561` (unchanged).
- Historical14-inter-op CPU receipt SHA256: `44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197` (unchanged).
- Runtime: Python3.12.13, torch2.14.1, macOS26.7.1 arm64. Actual source/lock/runtime record: `artifacts/evidence7-search-timing/implementation-identities.json`.
- `git diff --cached --check -- . ':!locks/search/licenses'` passed before commit. Archived license bytes intentionally retain their original upstream whitespace; license hashes were verified against every archived path.

## Real calibration failure and termination correction

The first actual native calibration under implementation `d6248aa` completed seven retained searches, then stalled during Ridge warmup. Inspection identified an early-stable batch below its count cap keeping the outer loop alive after another unstable batch reached its cap. The identified calibration process PID92088 alone was interrupted with SIGINT (exit130). Its search records, source identity and original stderr remain intact under `environment/evidence7-calibration` and `artifacts/evidence7-search-timing`; the interrupted attempt has a separate status record. No timing JSON had been published, and those in-memory pre-fix warmup blocks were lost; no timing result is claimed for that attempt.

Two deterministic REDs reproduce mixed stability/cap nontermination with a bounded test sentinel and interruption between blocks losing raw records (2 failed,8 deselected in1.09s). The correction excludes early-stable batches from pending warmup work and attaches accumulated timing records to exceptions arising between blocks. The reproducer publishes such failure records before re-raising. All30 search/timing tests pass in31.65s. A new full suite is justified by this concrete behavior change; its result and new source commit precede fresh measurements. The original seven searches are not resumed, and none of their checkpoints is loaded under the corrected package digest.


## Corrected-source verification and actual bounded measurements

The final full suite after the termination correction passed **708 tests,48 existing sandbox/MPS skips,28 existing vendor/short-fixture warnings in270.79s**, exit0. Its exact log is `artifacts/evidence7-search-timing/final-full-suite-after-warmup-fix.txt`. The tested fix commit is `ca7af11f97a86f43311f0ea9cbd63b3f1c97e935`, following implementation `d6248aa`. Current package SHA256 is `e2d8a6efe205468a2c77c17baeaa4b197b86c01d3e328649e50c13b6296eb214`; core lock and numerical backend producer remain unchanged.

Fresh reproduction, with actual native memory counters and MPS fallback disabled:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python audit/implementation/evidence/2026-10-16/reproduce-task7.py --output environment/evidence7-calibration-fix1 --bootstrap-watchdog-seconds 30
```

That output root now exists and is immutable evidence; a repeat must supply a new root. The30-second bootstrap watchdog is an explicit process guard, not a chosen study budget. The synthetic engineering fixture is d16/h64, F128/G32/V64, batch128, two base and continuation updates. The observed Ridge anchor enclosure was0.867764917s. Actual doubled exploratory candidates were1.735529834s and3.471059668s. There were seven searches: three Ridge, B1, B3 partial16 centered, B5 Adam, and a10ms B7 budget-exclusion check. Search enclosures were respectively0.868/1.750/3.480/3.490/3.362/3.486/0.0118s. Overruns include actual kill/wait and final bookkeeping; they are charged, never clipped away. B3 selected trial1 while retaining the official trial0. Real worker timeouts and late successful predictions remained unavailable. The B7 budget expired before worker launch, yielding no trial or winner; this is not a measured NODE fit or default-plateau result. Seventeen search/child manifests are terminal.

Ridge's two measured best objectives are equal at0.011645021168453527; this tiny two-budget observation does not establish a quality plateau. `expansion_diagnostics` retains no exact-boundary winner or improving ceiling here and always reports freeze_ready=false/evidence_pending_WP05. The next phase must expand winning boundaries/improving ceilings and collect full-quality/default curves before selecting actual WP05 policies.

The timing query contains257 rows, including remainder batches, and tests batches1/16/32/64/128. All three modes chose128 as the smallest tested batch within95% of peak median complete-call throughput; all native memory guards were observed and passed. Ridge host timing used20 repetitions per batch and reached diagnostic median/p95 precision (selected median0.611021ms/p950.657358ms). B5 causal host timing needed66 repetitions and reached precision (0.8177295ms/0.94704225ms). Causal resident timing exhausted80 repetitions: median0.1076665ms/p950.1583524ms, but selected-batch tail relative half-width0.152612 exceeded requested0.1. Its status correctly remains unresolved_at_repetition_or_warmup_cap. All warmups stabilized; raw records retain131/369/446 blocks respectively. The iid bootstrap diagnostic does not correct temporal correlation and is not final inferential evidence. No extra repetitions were run to turn the capped resident result into a pass.

Fresh Ridge cold-process timing separately records import1.176916167s, model load0.000258708s, first complete host prediction0.051584959s, process wall1.436193833s and excluded serialization0.001052750s. Its first prediction uses policy batch1; do not compare it as a cold/warm ratio against selected warm batch128. Exact cold predictions are hashed, and the tests verify save/load prediction equality. Measured batch parity was checked against each fitted model's full prediction with retained tolerances; maximum causal discrepancy was2.6036e-7. Resident timing excludes transforms, target inversion and host transfer, so its boundary differs from host timing.

All three raw timing files were reloaded through `read_timing`, which verifies the envelope and exactly reconstructs the summaries. The Ridge reconstructed summary is also separately archived. Process peak RSS was249511936/331055104/331841536bytes for Ridge host/causal host/resident, respectively; these cumulative process peaks are not model increments and are never added to MPS counters. CPU records explicitly mark MPS counters unsupported.

## Task8 runtime handoff

Use fresh receipt `environment/evidence7-calibration-fix1/search-backend.json`, SHA256 `e403a1fb2216b58557dfb414254facda09acfaa5dfbabf05a9f61ce8bca7095d`. It was produced by the unchanged actual numerical CPU producer under Python3.12.13, torch2.14.1, macOS26.7.1 arm64, float32, CPU intra-op1/inter-op1 and fallback0. Runtime SHA256 is `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68`. Core lock remains `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`; producer remains `237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e`. NODE environment hash remains `89ce6a4b559114eb759d31092d88287ff5e0cd04672bdcabd9143c3a4e56b561`.

For a fresh caller performing causal fit/load/timing, set `PYTORCH_ENABLE_MPS_FALLBACK=0` before imports and initialize `torch.set_num_threads(1); torch.set_num_interop_threads(1)` before parallel work. Set resolved experiment backend cpu_threads1/interop_threads1 and pass the receipt explicitly via `SearchBudget(backend_report=...)`. Search workers already enforce1/1 and OMP/OPENBLAS/MKL/VECLIB/NUMEXPR thread limits1 in serial fresh interpreters; their measured resource summaries record actual1/1. `TimingPolicy.cpu_threads=1` restores the caller's intra-op setting after timing, but it does not reset inter-op, so initialize the caller once. Do not reuse the historical14-inter-op receipt for these1/1 causal workers. Runtime/producer/core-lock changes require a newly validated numerical receipt; package-source changes still require fresh parents/checkpoints even if that numerical receipt remains applicable.

Historical `environment/task11-parent-backend.json` remains SHA256 `44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197`. All521 Task3 indexed files were independently rehashed and matched. The current attempt contains no old checkpoint loads, T access, frozen policy, long NODE fit or new predictive claim.

## Evidence and self-review

The curated record is `docs/research/2026-10-16-evidence7-search-timing.json`;705 exact size/SHA pairs are in `audit/implementation/evidence/2026-10-16/task-7-artifact-index.json`. They cover both source-scoped attempts, original failures, new logs, receipts, raw timing/search/checkpoint artifacts and search terms/lock files. Bulk remains local. The report is also archived beside the index. All source changes preceded final measurements; the evidence commit only adds documentation/index/decisions.

Self-review checked the worker's F/G-only label boundary, source/receipt substitutions, exact replay, charged failures/late V scoring, full B6 lifecycle in integration tests, ordinary and between-block timing failures, cold/resident boundaries, precision caps and historical byte preservation. This found and corrected the actual tqdm compatibility failure and measured warmup termination defect without weakening identity or numerical checks. An ad hoc final evidence checker initially looked for status at the manifest top level and guessed a nonexistent NODE lock filename; these verification-script mistakes were corrected to identity.status and locks/node/environment.json before successful publication. No implementation behavior or evidence output was changed to satisfy the checker.

Remaining concerns are explicit phase limits: full-quality/default budgets and plateau curves, correlation-aware final timing policy, final WP05 numeric design after Tasks8/9/10/13, and retained Task3 M1/M2 triage. Independent review and root-owned task ledger decisions are next. No further measurements are proposed by this implementation handoff.
