# Task 9 review — cb63fe3..d0e34a1

## Spec Compliance

- ❌ Issues found: active-stage `best.json` is loadable but cannot be resumed through the public training API; the checkpoint boundary also accepts nonfinite optimizer state. Both concern Task 9's complete, valid recovery-state contract (spec §6), rather than deferred growth or study-runner work.
- ✅ The remaining reviewed Task 9 behavior matches the brief: F-only fitting and G-only strict selection; original-unit predictions; explicit unsupported-setting rejection; full selected optimizer/scheduler/sampler/RNG restoration; durable immutable payload publication and bounded successful-version retention; a narrow observation API; and honestly scoped backend/cost provenance. Evidence is identified below.
- ⚠️ Native MPS training recovery is not established by this change. The report explicitly limits the demonstrated training parity to CPU and records inherited kernel tolerances; Task 10 growth migration, Task 11 controls, and Task 12 enclosing fit/study timing and hardware provenance remain downstream scope.

## Strengths

- `src/treeneuron/train.py:598` uses strictly lower original-unit G MSE and snapshots the complete state. `src/treeneuron/train.py:626` evaluates stage update zero, while final selection reconstructs the selected snapshot and retains all executed diagnostics. Tests at `tests/test_train.py:112` and the nonzero-best test exercise meaningful selection outcomes, rather than only final weights.
- `src/treeneuron/train.py:532` distinguishes unfinished-stage recovery from continuation and restores an embedded selected state from a completed last checkpoint. `tests/test_train.py:144` compares parameters, optimizer state, all recorded RNG states, scheduler, sampler cursor/order, predictions and next row IDs after a real non-epoch interruption; `tests/test_train.py:299` covers the completed-last crash boundary.
- `src/treeneuron/checkpoint.py:102` writes immutable payload and metadata before atomically publishing a durable pointer. `src/treeneuron/checkpoint.py:73` checks metadata/payload hashes; `src/treeneuron/train.py:343` validates identity and bounded topology before tensor reconstruction. The failure-injection, identity-drift, checksum and topology tests exercise actual filesystem or loader boundaries.
- `src/treeneuron/train.py:436` reconstructs compact topology and explicitly named parameter groups, then restores training RNG after allocation. `tests/test_checkpoint.py:126` accurately limits its claim to grown-topology/group roundtrip, without claiming Task 10 optimizer migration.
- `src/treeneuron/artifacts.py:204` atomically applies whitelisted observations, preserves immutable identities and rejects terminal/inconsistent records. `src/treeneuron/train.py:478` fills actual F/G/backend/model observations without accessing V or inventing hardware evidence. The final hardware regression covers both unavailable and independently supplied hardware hashes.
- `src/treeneuron/train.py:239` validates producer evidence and identifies smaller CPU workloads as correctness diagnostics; the primary recipe remains 128/128. `src/treeneuron/train.py:697` documents measured attempt versus nested validation boundaries, consistent with the controller's Task 12 accounting ruling.

## Issues

### Critical

- None found.

### Important

1. **A durable active-best checkpoint cannot be resumed despite loading as a valid TrainingState.** `src/treeneuron/checkpoint.py:95` publishes `state.best_snapshot` without an embedded best. That snapshot retains `stage.complete=False` and an already populated `best_mse`. `load_checkpoint` therefore returns an unfinished state with `best_snapshot=None` (`src/treeneuron/checkpoint.py:156`). On `continue_fit`, `_run` reevaluates it (`src/treeneuron/train.py:626`); the same CPU prediction ties the stored MSE, so `evaluate()` does not create a snapshot (`src/treeneuron/train.py:603`). `publish()` then indexes `None` (`src/treeneuron/train.py:617`) and terminates the retry. This contradicts the usable complete-checkpoint/recovery boundary in spec §6 and the durable-best retention requirement. Restore/bootstrap the selected snapshot without losing its selection state, or explicitly encode and reject a non-resumable checkpoint role before entering training. Add a regression that loads the active `best.json` after interruption and exercises the declared public behavior.

   **Confirmed diagnostic:** interrupt the existing tiny CPU training case after checkpoint update 2 using existing `environment/report.json`; load `best.json` with the correct identities; resume on fresh RunPaths with the original policy. Output: `active best: 0 complete=False has_snapshot=False`; `active best resume: TypeError 'NoneType' object is not subscriptable`. No input corruption or checksum manipulation was involved.

2. **The public save/load boundary accepts nonfinite Adam state as a valid checkpoint.** `src/treeneuron/checkpoint.py:102` validates metadata only before publication; `src/treeneuron/train.py:459` loads optimizer tensors without checking finiteness. The in-loop finite check protects normal training updates, but it does not protect public `save_checkpoint` calls, reconstructed states, or the embedded selected snapshot. A bad state can receive valid checksums and replace a last-valid pointer, then load successfully. This breaks the claimed valid complete-state boundary and finite-state failure behavior in the Task 9 context/spec §6. Validate model/optimizer tensor finiteness for current and embedded-best states before publishing, and validate deserialized tensor state before applying it to the reconstructed optimizer; reject without advancing the durable pointer or caller RNG state. Add direct save/load rejection tests rather than relying solely on the existing optimizer-step injection test.

   **Confirmed diagnostic:** load the valid interrupted last checkpoint, set one `exp_avg` tensor to infinity, call `save_checkpoint`, then `load_checkpoint` with unchanged correct identities. Both succeed and the returned optimizer still contains infinity. Output: `nonfinite Adam save/load accepted: True`.

### Minor

- None necessary for this task gate.

## Verification and inspection boundaries

- Read the provided diff in sequential segments once; the tool truncated part of the training fixture/assertion hunk, so that missing diff segment was reread. No changed source file was independently reread, and no git command, source edit, commit, subagent, broad test suite or new backend probe was run.
- Read raw recorded logs: `environment/task9-focused.txt` reports 53 passed; `environment/task9-pytest.txt` reports 406 passed / 48 skipped for the implementation before the final provenance fix; `environment/task9-hardware-observation-focused.txt` reports 54 passed for the final metadata correction. These outputs contain no warnings. The review does not attribute the earlier full-suite run to the final commit.
- Named cross-file risk: whether resumed runtime identity also guarantees the live backend/thread recipe. Inspected `src/treeneuron/backend.py:45`–`55` and its producer-report validator at `:120`; runtime identity is software/platform/architecture, distinct from hardware. No new hardware or cross-backend parity claim is made here.
- Named cross-file risk: whether prediction inversion can return nonfinite original-unit values. Inspected `src/treeneuron/data/transforms.py:141`; inverse transformation checks both inputs and results for finite vectors.
- Named cross-file risk: whether checkpoint/observation JSON validation actually rejects nonfinite metadata. Inspected `src/treeneuron/config.py:142`; canonical JSON uses strict conversion and `allow_nan=False`. This protects metadata, not the tensor-state issue above.
- Ran one targeted inline diagnostic, using the existing test fixture with the existing backend report and temporary artifacts, to answer the two untested recovery/serialization risks. It produced the concrete failures listed above. No repository source or test files were modified.

## Assessment

**Task quality: Needs fixes.**

The core training and ordinary last-checkpoint recovery design is sound and has substantive behavioral tests. The two public-checkpoint boundary failures above need correction before Task 9 can serve as the trusted state boundary for subsequent growth and fork work; neither requires implementing downstream tasks.
