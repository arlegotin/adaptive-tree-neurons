# Task13 fix round 2 — final combined work overflow (F1)

Status: **implemented and verified; ready for the same reviewer's scoped re-review**. BASE `69d2d3c4d00322e8541c17c9f0374dc9a5ec3dd5`; tested commit `d5c8828a6b4ceb48e29ec64c834b2d8335ff54d8` on existing `main`.

Read the complete `task-13-fix1-review.md`. Original I1–I4 are closed by that review; this round changes only F1. The final expression `base + outer + standalone + elapsed` was outside the invalid-work guard. Individually finite categories could therefore overflow at publication, leaving the result unclassified and omitting the current work receipt.

## Correction and boundaries

`analysis/audit.py::_finish_audit_work` now reduces the final combined work total with `math.fsum` and validates it with the existing finite/nonnegative `_number` guard inside the existing error-classification boundary, before writing either output. Overflow produces the existing blocked audit classification, an updated deterministic audit digest/snapshot, and a current work receipt containing `total_study_work_seconds=null` plus an explicit reason. The measured current invocation duration remains recorded. A valid total is published unchanged by subsequent logic.

The disjoint execution + prior outer audits + standalone analyses + current invocation boundary is unchanged. Nested analysis exclusion, exact prefix checking, missing/unknown-work handling and timing receipt schemas remain unchanged. No model, score, risk, scientific inference, frozen config, source/receipt/access guard, public API, or numerical producer changed. There is no tolerance waiver, schema migration, new ruling, new hypothesis or new evaluation authority.

Only two files changed: `src/treeneuron/analysis/audit.py` and `tests/test_audit_fix1.py` (42 insertions, 6 deletions). The adjacent synthetic regression covers outer `1e308` + standalone `1e308`, execution `1e308` + outer `1e308`, and the finite control execution `2` + outer `3` + standalone `5` + current measured duration. It checks that both the deterministic snapshot and current work receipt are actually published and bind the resulting digest. It uses tiny synthetic JSON receipts only.

## Exact verification

RED (before the production correction):

```sh
.venv/bin/python -m pytest tests/test_audit_fix1.py::test_final_work_total_cross_category_overflow_is_blocked -q --basetemp=/tmp/task13-fix2-red > /tmp/task13-fix2-red.txt 2>&1
```

**2 failed, 1 passed in 1.58s**, exit 1. Both overflows reproduced `ValueError: Expected finite JSON value, got float` escaping final receipt publication. The finite control passed.

GREEN: the same node with `--basetemp=/tmp/task13-fix2-green` and output `/tmp/task13-fix2-green.txt`: **3 passed in 1.72s**, exit 0.

Affected checks:

```sh
.venv/bin/python -m pytest tests/test_audit_fix1.py::test_final_work_total_cross_category_overflow_is_blocked tests/test_audit_fix1.py::test_work_receipt_exact_prefix_and_overflow_classification tests/test_audit_fix1.py::test_analysis_clock_writer_is_independent_of_filesystem_order tests/test_primary.py -q --basetemp=/tmp/task13-fix2-affected > /tmp/task13-fix2-affected.txt 2>&1
```

**17 passed in 11.52s**, exit 0. This includes within-prefix overflow, exact prefix/order independence, the new cross-category overflow cases, and primary aggregation checks. The saved-primary test uses the existing simulated protocol fixture; no real model fit, protected numerical cohort, complete 16-fit fixture, 155-test covering matrix or full suite was rerun. Tests were serial, used unique basetemps, and ran with fixed source. All 311 launch files were rehashed and matched after exit.

## Tested identity and retained evidence

Strict post-commit `execution_snapshot(root, HEAD)` succeeded. Its complete 311-file map equals the affected test-launch map exactly.

- Commit: `d5c8828a6b4ceb48e29ec64c834b2d8335ff54d8`.
- Execution SHA256: `15254c54ac4bf7f3547c660b9b8734dcd62c9a3efbd2d60011fab98a3f290384`.
- Package SHA256: `5d8f34f31c2b71f816a73efe934a124c7941eeee8769cbbbab7f6d065c0032da`.

`task-13-fix2-evidence/` contains the three exact raw logs, launch source map, strict tested snapshot, scoped diff, three synthetic cases' actual snapshots/receipts, and the prior-evidence preservation map. `task-13-fix2-evidence-index.json` hashes this report and every evidence file; it excludes itself to avoid recursion.

- `affected.txt` SHA256: `df6130cc0347c8e4e264c8293192c6ca9fb2420a1533df7dcfa18509050570ef`.
- `green.txt` SHA256: `d7ba8c071e5c81ad5d9c8ab565dcf88b56d8f3a0da3a365bf1fc48cd97607119`.
- `red.txt` SHA256: `4a795fc8e90ce3d17bcb465ea7642efe2ec19d525fe12617a3cee76ae3cfd621`.

All 38 recorded prior report/index/fix1 evidence files still match their pre-run hashes. Original and fix1 reports/indexes/logs were not rewritten. Their historical coverage remains assigned to their own recorded source, never this fix's source. Root-owned untracked Ruling12/Ruling13 files remain untouched and unstaged.

Self-review examined the complete two-file diff, guarded placement before both publication calls, exact final reduction, blocked/null/reason behavior, finite positive control, and source/evidence identities. `git diff --check` passed. No helper/reviewer was spawned, no branch/worktree was switched, and nothing was pushed or published.

## Remaining gates

The same reviewer must re-review F1, then root owns acceptance and the broad final review before scientific work. This round performs no genuine pilot, root freeze, scientific cohort, scientific T acquisition or final scientific report. No unresolved implementation concern is claimed within F1; independent review remains outstanding.
