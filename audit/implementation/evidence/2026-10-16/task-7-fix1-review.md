# Task 7 fix-round 1 scoped review

## Finding verdicts

- **I1 — Balanced method blocks: ADDRESSED.** `measure_comparison` accepts explicit method/mode cases with separately chosen batches, validates shared block controls, and executes mirrored method passes within common block IDs (`src/treeneuron/evaluation/timing.py:279`, `:321`). Every successful call retains method ID and repeat slot; alternating ABBA/BAAB and seeded mirrored random order are covered by `tests/test_timing.py:160` and `:204`. Batch calibration remains a separate API. The new native reproduction records a real shared Mean/Ridge comparison and rebuilds its summary (`audit/implementation/evidence/2026-10-16/reproduce-task7-fix1.py:62`). This fixes the original absence of comparative method scheduling.

- **I2 — Complete-block precision and pairing: ADDRESSED.** `_complete_blocks` requires two successful explicit repeat slots, deduplicates identical records, rejects conflicting duplicates, and excludes incomplete blocks (`src/treeneuron/evaluation/timing.py:67`). The bootstrap now samples whole repeat pairs; the precision gate counts complete blocks rather than calls (`:87`, `:99`). Comparative reconstruction intersects complete block IDs across methods and uses identical bootstrap draws for the reference and numerator (`:135`). Both per-method and paired precision must pass, and warmup stability still gates the returned successful status (`:356`). Tests directly cover one-block rejection, duplicate invariance, partial-block exclusion, and preserved ratio intervals under shared drift (`tests/test_timing.py:179`, `:192`). Between-block independence is explicitly an assumption for the final design; the old within-block iid defect no longer exists. Historical schema1 records are rejected by the new reader rather than silently reinterpreted (`timing.py:368`).

- **I3 — Verified prior cost and anchor lineage: ADDRESSED.** Costed resume now requires a hashed descriptor and ledger chain, rather than accepting a standalone sampler snapshot. The loader verifies referenced artifact bytes, run/configuration/identity/budget ancestry, unique originating runs, ordered sampler/trial correspondence, official-anchor status, trial charges and cumulative enclosure totals (`src/treeneuron/evaluation/search.py:232`, `:240`). Results expose prior and combined trial records, verified attempts, and separate attempt/cumulative costs while keeping the winner restricted to current trials (`:86`, `:362`, `:372`). Terminal ledger/descriptor publication is explicitly excluded from the stated enclosure, consistently with terminal manifest publication (`:24`). The new regression retains a failed anchor's error and charge, checks cumulative cost, and rejects raw snapshots, changed trial bytes and missing ancestor ledgers (`tests/test_search.py:218`). The bounded native reproduction additionally exercises an actual timed-out official worker followed by a fresh successful resumed trial (`audit/implementation/evidence/2026-10-16/reproduce-task7-fix1.py:40`).

## New breakage in the fix diff

**None found.** The new APIs preserve explicit timing boundaries and failure records; incomplete blocks cannot enter paired precision, and raw sampler history cannot enter costed search without its ledger. No new Critical, Important or Minor finding.

## Out-of-scope observations

**None new.** Existing Task3 row-trace overhead and expected-warning triage remain unchanged and are not reopened by this scoped review. Full-quality/default plateaus and final numeric WP05 policy remain validly pending; approving these fixes does not complete Task7 scientifically.

## Verification and scope

- Reviewed the supplied `2e921d3..e90b315` fix diff in passes against I1–I3 and the appended fix report. No untouched-code re-review, broader repository search, helper agents, git regeneration or production edits.
- The fix report names the exact covering command, `.venv/bin/python -m pytest tests/test_search.py tests/test_timing.py -q`, and reports **35 passed in 27.14s** with the retained log path. The controller independently verified that log, all 54 new indexed artifacts and all 705 unchanged historical artifacts. No suite, CLI or additional focused test was rerun: the new tests and code resolve the original doubts.
- The bounded native evidence retains four complete measured method blocks/eight calls per method, with insufficient p95 evidence and unstable capped warmups correctly unresolved. Its resumed search retains the unavailable official anchor and reports cumulative charge **1,171,951,167 ns** (`docs/research/2026-10-16-evidence7-fix1.json:21`, `:39`). No final numeric policy or predictive claim is introduced.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** I1, I2 and I3 are closed. **Task7 implementation-phase quality: Approved**, retaining the existing scientific-phase limits and Task3 minor triage.
