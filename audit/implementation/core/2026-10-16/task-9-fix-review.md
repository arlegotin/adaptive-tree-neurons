# Task 9 fix round 1 — d0e34a1..bc04f0a

## Finding Verdicts

- **Durable active-best checkpoint loads but cannot resume — ADDRESSED.** `src/treeneuron/checkpoint.py:95` now publishes the selected point with its own complete embedded best snapshot. Loading it preserves the original unfinished stage and allows the existing resume path to proceed without the tied-risk reevaluation that previously left `best_snapshot=None`. `src/treeneuron/train.py:488` additionally copies optimizer tensor storage before applying saved state, preventing resumed Adam updates from modifying the embedded selected moments/step when serialization shares storage. `tests/test_train.py:326` exercises selected updates 0 and 2, a later interruption, exact complete-state restoration, remaining update/evaluation counts, unchanged stage policy/start, and no subsequent G improvement. This addresses both the original crash and the aliasing exposed by its regression.

- **Public checkpoint save/load accepts nonfinite Adam state — ADDRESSED.** The shared recursive validator at `src/treeneuron/train.py:436` rejects nonfinite tensors and floating scalars. Publication validates both current and embedded-best snapshots before serialization or filesystem mutation (`src/treeneuron/checkpoint.py:111`, `:114`). Loading validates both before model reconstruction, RNG restoration or optimizer application (`src/treeneuron/checkpoint.py:162`, `:164`); direct in-memory reconstruction also validates (`src/treeneuron/train.py:464`). Tests at `tests/test_checkpoint.py:216` and `:246` cover NaN/infinity in model values, both Adam moments and step, independently for current/embedded state. They verify preserved pointer bytes and RNGs; save tests also verify no new files and successful restoration of the previous checkpoint, while load tests use internally consistent checksums and an optimizer-application sentinel.

## New Breakage in the Fix Diff

- None found. The fix retains the existing checkpoint schema and uses a nonrecursive selected snapshot. The optimizer copy is necessary isolation, and both publication and loading validate embedded state before any externally visible state transition.

## Out-of-Scope Observations

- None. Native MPS training recovery and Tasks 10–12 remain outside this scoped fix review; this verdict makes no additional backend or downstream implementation claim.

## Checks

- Read the task brief, scoped re-review instructions, appended fix report and provided fix diff. The combined tool output truncated part of the new malformed-load test, so only that missing diff segment was reread. No git commands, source crawl, source/test edits, test reruns, backend probes or subagents were used.
- Inspected the recorded RED summary in `environment/task9-fix1-red.txt`: 34 failed / 30 deselected, matching the two active-best cases and 32 finite save/load cases in the diff.
- Inspected `environment/task9-fix1-focused-before-copy.txt`: the preserved nonzero-best failure reports Adam step 6 versus expected 2, supporting the optimizer-storage isolation fix.
- Inspected final recorded output in `environment/task9-fix1-focused.txt`: **88 passed in 17.14s**; and `environment/task9-fix1-pytest.txt`: **441 passed, 48 skipped in 110.63s**. These logs contain no warnings. The report identifies the skips as sandbox MPS availability and states no source changes followed final verification. No additional unanswered narrow risk warranted running another test.

## Verdict

**Fix round: All findings addressed, no new Critical/Important breakage.** Both original Important findings are closed; Task 9 is approved within its stated scope.
