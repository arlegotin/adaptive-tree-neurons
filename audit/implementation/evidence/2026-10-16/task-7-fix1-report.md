# Task7 review fix1 report

Status: DONE_WITH_CONCERNS for I1–I3 implementation correction, ready for root's scoped re-review. FIX_BASE `2e921d3`; tested fix `75fde4239e0803cf7292988508bcf0276b3c3a6d`. Full-quality/default plateaus and final numeric WP05 policy remain pending. The historical initial report,705-file index and measurements were not edited.

## I1: shared method blocks

`measure_comparison(cases, X)` accepts a mapping from explicit method/mode IDs to `(predictor, TimingPolicy)` pairs. Each case has one previously chosen batch, so per-method batch calibration remains a separate `measure_predictor` operation. Cases share thread, warmup, repetition, precision, tail, order and seed controls; device, resident/host boundary, cache policy and batch may differ explicitly. Unsupported capabilities fail before execution. Method IDs are sorted deterministically for scheduling/reference choice and remain in every raw call alongside common block ID and repeat slot0/1.

The coordinator executes mirrored method passes: A/B/B/A, then B/A/A/B, or a seeded random permutation and its reverse. Every case is included twice in every complete warmup/measurement block. It retains raw order, preparation/resource exclusions, fitted identities, actual backend resources, mode boundaries and original errors. Failures abort the comparison with accumulated raw records attached; a partially executed common block cannot support paired precision. Inference/thread/synchronization/batching/cache contracts remain explicit, and caller intra-op threads are restored. Cold timing and single-model batch calibration remain separate APIs.

## I2: complete-block precision and pairing

Timing schema2 requires an explicit `repeat` slot. Reconstruction recognizes complete blocks containing successful slots0 and1; missing/failed slots do not contribute. Exact duplicate records are deduplicated, conflicting repeats are rejected, and repeated calls cannot increase the independent-block count. Bootstrap resampling draws whole two-call blocks. Median and p95 estimates retain the complete call distribution while each bootstrap draw preserves its block membership.

`rebuild_comparison(raw_blocks, policies)` intersects complete block IDs across all methods. Comparative intervals draw the same block indices for numerator and denominator, preserving shared drift and pairing. Tests with exact B=2A and strongly varying shared block durations retain ratio2 and interval[2,2]; partial blocks are excluded. Pairwise output uses the deterministic first sorted method as its reference.

Precision requires at least `max(2, ceil(min_repetitions/2), ceil(1/(1-tail_quantile)))` complete blocks and the requested relative half-width for both median and tail. Repetition caps remain call caps, so p95 requires at least20 complete blocks/40 calls. The paired ratio also has a minimum-block and width gate; comparative success requires the method and paired gates. This minimum does not guarantee nominal coverage. Between-block independence remains an explicit design assumption requiring final WP05 assessment; within-block dependence is no longer ignored or disclosed merely through a diagnostic label. Warmup instability or insufficient blocks remains unresolved. Historical schema1 records are preserved and require their original source reader; current `read_timing` rejects them instead of assigning new statistical meaning to old records.

## I3: costed resume lineage

Standalone `ReplayTPE.save/load` remains an exact sampler utility. Costed `search` now accepts only the returned `sampler.resume.json` descriptor, which hashes an immutable `search.ledger.json`. Every ledger binds its originating run ID, resolved configuration, search identity/budget, complete trial JSON records, final replay snapshot and predecessor descriptor. Resume verifies the entire ancestry, artifact hashes, configuration/identity, predecessor path, unique run IDs, ordered sampler/trial correspondence, first official anchor, statuses/objectives and nonnegative costs. It rejects raw sampler snapshots, missing ancestry, changed trial bytes, inconsistent cumulative charges and unexplained sampler history. Per-trial snapshots alone are deliberately not resumable cost evidence.

`SearchResult.trials` and `winner` still refer only to the new attempt. `prior_trials` retains full verified historical anchors/failures, and `all_trials` combines them with current records. `attempts` exposes each verified ledger and charge. `charged_ns` is a compatibility alias of `attempt_charged_ns`; `cumulative_charged_ns` sums all nonoverlapping attempt enclosures, including failed/rejected/timed-out work. Historical models cannot win the resumed attempt. A new `SearchBudget.seconds` limits the current attempt, not the lifetime budget; callers comparing total tuning cost must use cumulative charges.

The enclosing timer includes lineage verification/replay, input/sampler publication, workers and all existing fitting/selection/failure work. It stops after final sampler publication. Final cost-ledger/resume-descriptor/manifest/metrics publication is explicitly outside that enclosure, because the terminal ledger binds the observed completed duration. Nested trial costs are checked against, and never added to, their enclosing attempt. Unsupported or damaged lineage fails through the existing terminal-error path with its original reason and known metadata. Only successful operational completion publishes a costed descriptor; an attempt that ends with failed trials may still complete operationally and retain those failures for resume.

## RED/GREEN and exact verification

Initial meaningful RED command:

```sh
.venv/bin/python -m pytest tests/test_search.py tests/test_timing.py -k 'comparison_ or precision_requires or costed_resume' -q
```

Result: **4 failed,30 deselected in1.27s**. Failures reproduce missing comparison APIs, one-block iid success and missing cumulative lineage. Original output: `artifacts/evidence7-fix1/review-red.txt`.

Intermediate timing run:1 failed/12 passed because the older hand-built raw fixture lacked the newly required repeat slots. The fixture was updated to two explicit slots per block. Search lineage checks:5 passed/16 deselected in1.28s. Combined first covering run:34 passed in25.55s. The added reversed-ID random-order/JSON round-trip test initially compared an in-memory tuple with a decoded JSON list; reconstruction itself passed. Its assertion now compares JSON-normalized records, without changing production code or tolerances. The failed output remains archived.

Final required covering command:

```sh
.venv/bin/python -m pytest tests/test_search.py tests/test_timing.py -q
```

Result: **35 passed in27.14s**, exit0, retained at `artifacts/evidence7-fix1/covering-final-after-json-test.txt`. This includes real isolated Ridge/causal/B6 workers, matching numerical receipt generation, source-bound fitting, all existing timing/cost tests and the new review regressions. No unchanged708-test suite or broad baseline inventory was rerun. `git diff --cached --check` passed before the tested fix commit.

## Committed-source bounded native evidence

Package SHA256: `1979ab38867c2acabe989817ad15d59ea30ea10e5e0f30a9501edc87cc3676b5`. All retained new models/attempts were created after fix commit75fde42. Exact command:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python audit/implementation/evidence/2026-10-16/reproduce-task7-fix1.py --output environment/evidence7-review-fix1 --startup-deadline .25 --attempt-watchdog 30
```

The explicit0.25s deadline exercises unavailable-at-budget lineage, and30s is a one-trial watchdog; neither is a proposed quality budget. Synthetic data are d2/F32/G8/V16. The actual official Ridge trial timed out in a real worker, was killed/waited and retained with its TimeoutError. Attempt charge was269365083ns. The resumed fresh trial succeeded and charged902586084ns. Cumulative charge is exactly1171951167ns, with two verified attempt ledgers and the unavailable official anchor still present. Historical models were not promoted. An uncosted raw sampler was rejected, and the prior descriptor bytes stayed unchanged. Both run manifests are completed; unavailable trial status is preserved within the first operationally complete run.

Fresh Mean and Ridge models then predicted257 rows with batch64 in one shared comparison. Actual method schedules alternate Mean/Ridge/Ridge/Mean and Ridge/Mean/Mean/Ridge over blocks0–5; blocks0–1 are warmup and2–5 are measurement. All24 raw calls retain method/common-block/repeat identity. There are four complete measured blocks/eight calls per method, below the p95 requirement of20 blocks. Both warmups also remained unstable at their explicit cap. Precision correctly remains `unresolved_at_repetition_or_warmup_cap` rather than extending the bounded run until it passes.

Observed median complete-call timings were154250ns for Mean and161229ns for Ridge; p95 estimates were172997.8ns and183982.9ns. The paired median ratio was1.0452447325769854 with interval[0.990153151319806,1.1226716382148378], also ineligible for precision because of the block cap. These are bounded engineering observations with different fitted models, not final speed/quality claims. Native memory guards were observed and passed. Full raw-summary reconstruction, adding ten copies of every retained call without changing the summary, and rejecting one-block precision after twenty duplicate copies all passed.

The existing genuine CPU1/1 receipt remains applicable: `environment/evidence7-calibration-fix1/search-backend.json`, SHA256 `e403a1fb2216b58557dfb414254facda09acfaa5dfbabf05a9f61ce8bca7095d`. Producer `237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e`, core lock `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5` and runtime `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68` are unchanged. Caller settings remain fallback0 before imports, torch intra-op1/inter-op1 initialized before work; workers retain serial native thread limits1. No numerical producer/core/NODE lock/vendor edit was required.

## Append-only evidence and remaining gates

Curated fix record: `docs/research/2026-10-16-evidence7-fix1.json`. New54-file size/SHA index: `audit/implementation/evidence/2026-10-16/task-7-fix1-artifact-index.json`. Raw evidence: `environment/evidence7-review-fix1`; test/native logs: `artifacts/evidence7-fix1`. The original705 index entries all rehashed successfully, and the original index/report/curated record remain byte-identical to FIX_BASE2e921d3. Current source changes intentionally supersede only executable behavior for future runs; historical source identities, hashes, observations and reports were not rewritten.

I1–I3 are addressed for scoped re-review. Final quality/default plateaus, timing design/independence assessment and numeric WP05 policies remain pending all pre-freeze code. Existing Task3 row-trace/warning triage remains unchanged. No protected T, freeze, long NODE fit, general NAS platform, practical adapter expansion or new predictive claim was introduced. Root owns re-review and acceptance; no additional measurement is needed for this fix handoff.
