# Spec compliance: approved for the narrow storage correction

- Reviewed baseline `680429908735a8077fbfab972468b9d420b7a9f1` to tested source `388c82c59e3dccb6425c55a4a3832dadf80e55fb`, using the supplied full five-file diff once. No missing, extra, or misunderstood implementation requirement was found.
- `src/treeneuron/train.py:786` changes only the recording boundary: update history receives a new mapping without `row_ids`, with schema version 1, ordered-int64 hash encoding, count, and the existing `array_hash`. All remaining fields and all events are retained. `append_event(paths, event)` still receives the original event at line 797. The change does not touch training arithmetic, optimizer, sampler, RNG, scheduling, growth decisions, stopping, or allowance policy.
- `docs/research/2026-10-17-checkpoint-row-history.md:3` accurately documents the compact representation and raw-log retention; line 17 preserves the legacy/source-bound distinction. No checkpoint-reader permission or conversion was added.
- `tests/test_history_storage.py:16` checks complete event equality with an independently constructed dtype/shape/ordered-byte digest. The tests exercise real fit, best/last persistence, continued training, minibatch-size scaling, and growth-prefix preservation. `tests/test_optimizer_growth.py:168` and `tests/test_causal_baselines.py:297` migrate recovery assertions to commitments and strengthen them with full raw ordered-row equality across attempt plus retry.

## Strengths

- The production change is small and located at the single duplication boundary (`src/treeneuron/train.py:786`). It avoids mutating the raw event and introduces no new storage framework or dependency. Keeping non-update events unchanged preserves their existing semantics.
- The storage test varies actual minibatch size on F257 and compares both compact and full raw histories (`tests/test_history_storage.py:55`); the growth test preserves the complete parent prefix (`tests/test_history_storage.py:80`). These check observable behavior rather than a standalone serialization helper.
- Retained RED evidence shows all three new regressions failing on the missing commitment representation (`environment/h1-v1/history-storage-fix/red-2.log:34`, `:91`, `:115`). The affected run records 225 passes and the nine legacy assertion failures (`targeted-1.log:502`); `migrations-green-1.log:2` records all three new tests plus nine migrated cases passing. This supports the stated aggregate 234-case coverage, not a claim of one clean 234-case run.
- The actual replay wrapper starts from the old resolved config and changes execution paths (`environment/h1-v1/history-storage-fix/reference-replay.py:31`), checks data against the original reference before fitting (`:48`), and measures the fit plus terminal publication directly (`:70`). The retained final measurement records 4000 updates, 485,540 examples, and selected update 3675.
- The retained final comparator loads only the new-source checkpoint through its reader (`environment/h1-v1/history-storage-fix/complete-reference-comparison.py:52`); the old reference is a plain tensor/metadata dictionary (`:57`). Its recursive tensor/container comparison includes type, dtype, shape, device, and bytes (`:62`); metadata allows exactly five identity/path/receipt differences (`:79`). Predictions (`:96`), all raw events except checkpoint digests (`:104`), ordered update commitments (`:116`), complete compact history (`:124`), and same-source reload (`:142`) are explicitly checked. This supports the numerical claims without a source-bound loading waiver.
- `environment/h1-v1/history-storage-fix/replay-4-completed-2/replay-measurement.json:1` records `passed`, no failures, 21 selected tensors, 4324 raw events, 161 checkpoint-digest changes, and identical old/new raw-update hashes. The report correctly distinguishes the failed native MPS-state comparison and preserved postprocessing failures from the successful sandbox comparison.
- The measured metadata reduction is supported by actual file measurements: 20,613,624 to 2,567,456 bytes, and the same parent-directory boundary falls from 31,118,960 to 13,072,965 bytes (`replay-4-completed-2/replay-measurement.json:1`). The implementation report explicitly separates single-run timing and unequal memory measurement boundaries; it does not claim universal study feasibility.
- The renewed receipt wrapper uses the existing CPU128 kernel and finite-envelope producers, checks their pass status, validates the envelope, and checks unchanged execution snapshot (`environment/h1-v1/history-storage-fix/renew-receipt.py:20`). `receipt-measurement.json:1` and the successful comparator bind the same tested commit, package hash, execution hash, and backend receipt hash.

## Findings

### Critical

None.

### Important

None.

### Minor

- `environment/h1-v1/history-storage-fix/targeted-1.log:496`: the existing `test_node_gate_failure_preserves_completed_b0_b6_checks` emits the PyTorch `torch.jit.script` deprecation warning. This prevents a pristine affected-run output but is unrelated to the storage change and is transparently retained. Track a separate update of that existing probe to a supported API; do not suppress the warning or expand this correction to fix it.

## Focused unchanged-code checks

- **Risk: compact nested history might break or alias causal/greedy forks.** One focused search of the history/row-ID sites in the two fork implementations shows `copy.deepcopy(parent.history)` in `src/treeneuron/baselines/causal.py:290` and `src/treeneuron/baselines/greedy.py:137`, with no row-ID interpretation in either file. Compact commitments remain isolated in each fork. Retained affected-test coverage includes causal and greedy paths.
- **Risk: pairing or audit might still require historical raw IDs from the checkpoint.** One focused search of the named consumers confirms raw `events.jsonl` reads and ordered-int64 hashing in `src/treeneuron/experiments/paired.py:264`, `src/treeneuron/analysis/exploratory.py:36`, and `src/treeneuron/protocol.py:294` / `:355`. The original raw event is preserved by the changed recording boundary, and the retained comparator verifies its full ordered update stream.
- No other unchanged production code was inspected. No test suite, fit, numerical producer, checkpoint load, Git operation, or helper/subagent was run by this reviewer. Only the requested review document was written.

## Cannot independently verify within this review

- Archive byte integrity and completeness were not independently rehashed by this reviewer. Root separately reports all 113 indexed files / 31,131,569 bytes rehashed with zero mismatch, index SHA256 `fc36d2db999babe1563b0ee84197e3006fd4c2207f2436a747568e085a37f166`, and an exact 315-file strict-snapshot match at the tested source/package/execution identities. The old 289-file inventory was independently verified by root before edits. Those root checks complement this review of comparison logic, result records, failure inventory, and relevant logs.
- The reported numerical executions, clocks, and process peaks were not independently rerun. Their retained wrappers and outputs support the stated scope; this review cannot certify measurements beyond those artifacts. No additional heavy validation is recommended for this correction on the evidence examined.
- The finite-envelope receipt's underlying arithmetic cells were not recomputed. Receipt generation and source/snapshot linkage are supported by the retained wrapper and measurements; root has independently checked source/snapshot and artifact integrity and still owns the acceptance-document bindings.
- Whole-H1 scientific validity, complete resource-teacher feasibility, a genuine variance pilot, freeze, independent cohort completion, protected-T evaluation, and reproduction remain outside this five-file task. This approval is not evidence that those later gates have completed.
- Root-owned README, DECISIONS, calibration archive, and Ruling16 pending edits were not reviewed and are not covered by this verdict.

## Task quality: approved

The minimal recording-boundary change satisfies the authorized storage contract, and the tests plus retained actual replay address persistence, growth/resume, raw-stream preservation, numerical equivalence, and measured storage reduction. No blocking code or specification defect was found; root reports its independent archive/source checks passed and retains ownership of the acceptance-document bindings.
