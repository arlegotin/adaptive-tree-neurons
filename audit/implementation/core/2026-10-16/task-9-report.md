# Task 9 — deterministic training and atomic recovery

Implemented on the existing `main` checkout, starting at `cb63fe3`. No branch/worktree/subagent, protected T access, freeze, public research documentation, plan or ledger edits were made. Root owns review and public milestones. No predictive comparison or scientific conclusion is claimed.

Commits:

- `3836847c7796a7f6b38aca84379c8fc7d902d984` — `feat: add deterministic parent training and atomic recovery`
- `d0e34a1ea30194576fe5d0a25362bd1d5a71eaab` — `fix: keep runtime and hardware training provenance distinct`

Final package execution-source SHA256: `bab5b08e03faf3a42d46851e7ff197016da967dec42484772b6eef3a1484ee01`. This hashes sorted package-relative Python paths and contents, independently of Git HEAD. Git staging and commits used the expressly authorized sandbox escalation. Final working tree is clean; this report and environment logs are ignored execution evidence, not added to Git.

## Owned changes and APIs

Six committed task-owned files:

- `src/treeneuron/train.py`: explicit scalar model, policy, complete state, schedule, RNG snapshots, parent/continuation training, observation integration.
- `src/treeneuron/checkpoint.py`: immutable versions, checked metadata and tensor payloads, durable pointers, bounded successful-version retention.
- `src/treeneuron/artifacts.py`: narrow validated observation update API; strengthened existing manifest identity guard.
- `tests/test_train.py`, `tests/test_checkpoint.py`, `tests/test_artifacts.py`: real training, interruption, filesystem failure, corruption, identity, naming and observation regressions.

Public task interfaces:

```python
fit_parent(data: DevelopmentData, config: ExperimentConfig, paths: RunPaths) -> TrainingState
continue_fit(state: TrainingState, data: DevelopmentData, policy: TrainingPolicy, paths: RunPaths) -> TrainingState
save_checkpoint(state: TrainingState, path: Path) -> str
load_checkpoint(path: Path, *, expected_hashes: Mapping[str, str], device: str) -> TrainingState
update_run_observations(paths: RunPaths, observations: Mapping[str, Any]) -> None
```

`TrainingPolicy(logical_batch_size, physical_batch_size, max_updates, validation_interval, patience, checkpoint_selection='strict_improvement_earliest', evaluate_at_update_zero=True)` uses positive explicit stage allowances. `TrainingPolicy.from_config(config, continuation=False/True)` rejects unresolved base/continuation/validation/patience values. No hidden pilot defaults were introduced.

`SelectiveRegressor(d, h, tau)` is exactly `SelectiveAffineLayer -> ReLU -> Linear(h,1)`, returning `[N]`. `export_config/from_config` reconstruct compact grown topology before loading values. A real CPU64 `score_units` call confirms scorer compatibility. This is not a model registry or an implementation of Task 11 controls.

`TrainingState.predict(X, batch_size=None)` returns finite original-unit `[N]` NumPy predictions through the F-fitted inverse transform. `next_row_ids(F)` peeks without consuming sampler state, including epoch rollover. Internal `pack_state/unpack_state`, `capture_rng/restore_rng`, and `UpdateSchedule` are the narrow Task 10 reconstruction/state boundary. `save_best_checkpoint` publishes an embedded snapshot directly, without allocating a model or consuming RNGs.

## Training and complete-state semantics

Only F predictors/labels fit preprocessing and weights; only G determines checkpoint selection. Training never accesses V observations or T. It binds the already-constructed split manifest and its hashes without reading V labels. Tests replace the entire V observation object with an accessor that raises, and training still succeeds.

Immutable NumPy CPU64 data is explicitly copied to float32 tensors. Training minimizes the mean of per-row squared standardized-target errors, with no factor of one half, matching the scorer's row-loss convention. Prediction and target shapes must match exactly; `[N,1]`/`[N]` broadcasting is rejected. G metrics use original-unit predictions and actual F population variance. Constant targets retain scale one for numerics and null normalized MSE with its degeneracy reason.

AdamW honors configured LR, betas, epsilon and weight decay, using `foreach=False`. Constant and cosine schedules are implemented explicitly. The cosine horizon is the original base-plus-continuation allowance; the scheduler position equals the absolute optimizer update. A continuation retains current LR and scheduler position. SGD training, depth greater than one, constant schedules with nonzero cosine-only minimum LR, unresolved stopping settings, and physical/logical mismatch are rejected explicitly. Accumulation is intentionally not implemented; the selected primary physical/logical recipe is 128/128.

The local CPU sampler generator is separate from process RNGs. Each epoch uses its saved permutation, cursor and generator state; a short tail batch is one complete optimizer update and no rows are dropped. `epoch` is the zero-based index of the currently stored permutation; it advances when the next permutation is drawn after the cursor reaches the end. Checkpoints never include partial gradients or resume half an optimizer update.

The complete state owns:

- Model values/topology, including compact gate/contrast tensors and growth-event flags/IDs.
- AdamW state, parameter group hyperparameters, explicit stable parameter names per group, scheduler position/horizon/base LRs.
- Absolute update, epoch index, sampler order/cursor and local generator state.
- Python, NumPy legacy global, Torch CPU and available Torch MPS RNG state; unavailable MPS has a null state and reason.
- F preprocessing, actual F variance, config, backend receipt/tolerances, identity hashes and explicit F/G array/row/group hashes.
- Active/completed selection-stage metadata, full executed diagnostics, and a growth metadata mapping for Task 10.

Model reconstruction can allocate `Linear` modules and consume Torch RNG; `unpack_state` restores the saved training RNG after reconstruction. Failure during reconstruction restores the caller's prior RNG. Local allocation/contrast randomness remains Task 10's separate responsibility.

Every new stage evaluates G at stage update zero. The first strictly lower G MSE replaces the best; equal risks keep the earliest state. Snapshots contain optimizer, scheduler, sampler and RNG along with weights. Nonzero-best and update-zero tie tests verify the selected moments and sampler are from the selected update, not the last executed update. All executed curves and the final selection diagnostic remain available even when selected update precedes the last update.

`continue_fit` distinguishes unfinished-stage recovery from a new continuation. An unfinished stage retains its original allowance, relative validation intervals, patience count and embedded complete best snapshot; the policy must match. A completed selected state starts a fresh continuation with update-zero eligibility. A completed `last.json` can represent a crash after final last publication but before final selected return: the function first restores that last state's embedded G-best, then starts continuation. A dedicated regression showed the previously incorrect update 6 versus required update 0 and now passes.

Finite checks cover converted tensors, loss, gradients, updated parameters and optimizer tensors. Nonfinite Adam moments are rejected before advancing/publishing a valid update. Failures and keyboard interruptions append details and publish terminal failed/interrupted attempt records; successful parent/continuation fits leave their run operationally running, because they do not finish an enclosing multifork study. Retry uses new `RunPaths`; terminal attempts are never reopened.

## Backend and identity contract

Production `fit_parent` reads `environment/report.json`, validates real producer reports with `validate_backend_report`, recomputes the backend decision with `select_backend`, and rejects a differing recorded decision or requested backend. The actual reviewed primary selection remains CPU float32, one Torch thread, live/recorded interop count, physical/logical 128/128; score construction stays a separate CPU64 operation.

Per root's explicit diagnostic ruling, smaller CPU correctness workloads are accepted within d<=16, h<=64, F<=2306, batch<=128, with physical=logical. The actual dimensions, F rows and batches are recorded. Their receipt says kernel compatibility only and `performance_calibration_matches=False`; it makes no measured throughput claim for those workloads. Other backend/workload combinations and larger workloads require relevant evidence rather than an implicit calibration claim. This task does not change backend producer tolerances or Task 8 evidence.

Tests obtain genuine one-batch128 CPU probe evidence using the real producer in a session fixture and supply its file through the module's evidence-path seam. There is no fake passed flag, hash patch, hidden test bypass or dependency on an ignored pre-existing doctor file. No new native MPS training-resume experiment was performed in Task 9. CPU exact agreement is tested; MPS backend receipt records the inherited kernel tolerance, without a Task 9 same-backend training parity claim.

`load_checkpoint` requires the complete expected identity mapping: `code`, `lock`, `config`, `split`, `raw`, `transform`, `runtime`, each SHA256. Source/lock/runtime are additionally checked against live execution, and config/transform hashes against the reconstructed finite JSON metadata. Resume rejects a different backend. Split/config/code/lock mismatches are independently tested before optimizer loading. Operational output directory is not part of scientific identity.

The final provenance correction leaves `identity.hardware_hash` unavailable when Task 9 has not independently observed hardware; an already supplied real hardware hash is preserved. Runtime identity remains correctly named in checkpoint identities. Task 12 supplies actual doctor/hardware-report provenance. This avoids presenting a Python/Torch/OS/architecture runtime digest as a hardware observation.

## Checkpoint format, atomicity and retention

The caller passes a `.json` pointer path, normally `checkpoints/last.json` or `checkpoints/best.json`.

1. Pointer: exact schema `{schema_version:1, metadata:<checkpoint-uuid.meta.json>, sha256:<metadata digest>}`.
2. Immutable metadata: payload filename/checksum, finite strict complete-state metadata, optional active-stage best metadata, and diagnostic history.
3. Immutable `.pt` payload: current tensor state and optional complete best tensor state. The saved optimizer has explicit `param_names`; metadata independently describes named groups/options and both are cross-checked before optimizer load.

Metadata validation precedes tensor deserialization: required fields, complete identities, topology/config agreement and bounded dimensions, sorted valid growth IDs, stage counters/policy, scheduler position/horizon, sampler count/cursor, preprocessing, named optimizer coverage and backend recipe. Local tensor loading uses `torch.load(..., map_location='cpu', weights_only=True)`; this is an API for self-created local artifacts, not importing external pickle checkpoints. Restored model parameters are moved to the requested same backend before optimizer state loading.

A save writes a unique immutable payload and metadata/checksum, fsyncs both and their directory, writes/fsyncs a temporary sibling pointer, atomically replaces the pointer, then fsyncs the directory. An exception before pointer rename leaves the previous pointer byte-for-byte intact. Checksum corruption, metadata topology drift before tensor loading, and failures after payload or metadata writes are tested. Failure artifacts remain available. `save_checkpoint` returns the immutable metadata digest; `method.weight_hash` records this digest, resolved by the corresponding pointer.

Each complete update publishes last-valid; every improved best is durably published separately. On a fresh retry, an embedded best snapshot is republished locally even if the first resumed update is not an improvement. Direct snapshot publication avoids reconstruction or RNG advancement.

`published.jsonl` records only durable successful versions. Cleanup occurs only after durable publication and removes only journaled, unreferenced superseded versions. The last and best pointers retain their targets; last also embeds the complete active-stage G-best. Extra caller-owned JSON pointers can protect interruption/final artifacts. Failed/unpublished files, all events/curves/run records and losing forks are never removed. Successful normal training retains at most two payload files; failed-attempt storage is deliberately preserved rather than subjected to that bound.

Retention observations include protected pointer names, referenced bytes, retained version bytes, actual directory bytes, payload count and policy text. `method.serialized_bytes` is the bytes of referenced complete checkpoint payload+metadata versions, not a claim of weights-only model size. These measured checkpoint observations support Task 12 disk forecasts; the journal/events and failed attempts remain additional actual disk costs. This task does not forecast a whole cohort or delete any unrelated files.

## Observation ownership and timing boundaries

`update_run_observations` locks the existing manifest, validates finite JSON and whitelisted section/field updates, then atomically replaces it. It rejects run/study/phase/config/protocol/status changes, unsupported keys including nested actual-backend keys, terminal records, and inconsistent metrics-before-manifest records. Existing code/lock/data identity observations and parent linkage cannot be changed once supplied. `parent_run_id` can be filled once for later linked forks. Seeds must match immutable resolved config. Row counts must match resolved config; counts and parameter/byte/timing fields are validated.

Task 9 fills observed code/lock/split/raw/transform identities, optimizer seed configuration, actual F/G rows/groups, model counts, backend/threads/tolerances, selected checkpoint/hash, measured checkpoint storage, scoped attempt/preprocess/validation timings and process peak RSS. CPU runs leave MPS counters unavailable. Teacher normalization is explicitly unavailable to this consumer and remains generator-owned. V/T labels are not accessed to fill observations. Task 12 owns fork/study/score accounting, hardware provenance, retry/parent links and enclosing end-to-end totals.

The measured `_run` wall interval is **not an end-to-end fit total**. It includes F conversion, G validation, checkpoint publication and in-loop ledger updates. Initial model/optimizer construction, upfront metadata, imports and backend-evidence validation precede that interval. F transform fitting is separately timed. G transformation/prediction/original-unit metrics are reported as validation, explicitly nested in base/continuation and never to be summed again. Failed/interrupted attempts also publish their measured attempt/validation costs before terminal status. A final observation write necessarily follows the timing sample.

`costs.*.boundary` carries these scope/inclusion rules. `resources.timing_exclusions=[]` means no measured timing observations were discarded; it is not used to describe nested costs. Root explicitly assigned the measured enclosing end-to-end fit/study timer (including initialization and method overhead) to Task 12, rather than widening this task's measurement region or claiming that the scoped interval is a total.

This is not a cross-file transactional ledger. Metrics published before a crash in terminal-manifest replacement are preserved and rejected by the observation updater. Recovery does not overwrite that inconsistent attempt; a later runner links a new attempt. Checkpoint pointer atomicity is independent of append-only events and manifest observations.

## TDD and verification evidence

The test-driven-development and verification-before-completion skills were used; no subagent delegation occurred.

- Initial RED: `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py tests/test_artifacts.py -q` — 11 failed, 12 passed, 15 setup errors in 1.33s. Failures/errors were missing train/checkpoint/observation APIs, as the task anticipated. Preserved `environment/task9-red.txt`.
- First implementation: 37 passed, 1 failed in 4.12s. The remaining fixture requested patience different from immutable config; corrected that test to honor the paired policy. A subsequent focused run passed 38 tests.
- Expanded RED: active-stage best pointer and topology-before-tensor-loading failures; also corrected a scorer test's initially wrong keyword names against its real API. Evidence `task9-red2.txt` and `task9-green2.txt`. Expanded focused GREEN: 45 passed in 4.87s.
- Optimizer-state RED: nonfinite Adam moments allowed last-valid update 1 before a later failure. Added pre-publication optimizer tensor finite check; last-valid now remains update 0. `task9-red3.txt`; focused GREEN 50 passed in 10.04s (`task9-green3.txt`).
- Observation RED: tampered study identity and unsupported nested backend key accepted. Both rejected after the fix: artifacts 24 passed in 0.13s. `task9-red4.txt` retains the two expected failures.
- Failed-attempt cost RED: cost remained null after real training failure. Added measured interrupted/failed attempt costs and checkpoint observations. `task9-red5.txt`.
- That accounting integration exposed fresh-retry best-pointer bootstrap: focused 51 passed, 1 failed (`task9-green5.txt`). A full suite was inadvertently started before inspecting this focused result. Its complete result is explicitly **superseded**, not final: 1 failed (`test_interrupted_non_epoch_resume_matches_uninterrupted`), 404 passed, 48 skipped in 114.48s, at `environment/task9-pytest-superseded.txt`.
- Additional recovery RED: completed last-state continuation started at update 6 instead of selected update 0. `environment/task9-red6.txt`; fixed by restoring the embedded selected state before creating a new continuation stage.
- Corrected focused suite, inspected before the full run: `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py tests/test_artifacts.py -q` — **53 passed in 9.97s**, `environment/task9-focused.txt`.
- Full repository suite after recovery fixes: `.venv/bin/python -m pytest -q` — **406 passed, 48 skipped in 108.74s**, `environment/task9-pytest.txt`. MPS skips are sandbox availability skips; no native Task 9 training parity claim. This is the verified main implementation state, before the subsequent narrow hardware-provenance correction.
- After that full run, only API/cost-boundary explanatory wording was expanded to explicitly name setup exclusions; import smoke and `git diff --check` passed before the main commit. No numerical training behavior changed in that wording adjustment.
- Final hardware observation RED: targeted parent/manifest test failed because `hardware_hash` incorrectly contained a runtime digest. `environment/task9-hardware-observation-red.txt`.
- Final correction verifies both unavailable hardware evidence and preservation of an enclosing runner's supplied hardware hash. `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py tests/test_artifacts.py -q` — **54 passed in 9.45s**, `environment/task9-hardware-observation-focused.txt`. Per root direction, the prior full-suite evidence is preserved honestly and no redundant full/native measurement was run for this metadata-only fix.
- Final `git diff --check`, staged diff check and clean-working-tree check passed. No failures are omitted from this report.

Evidence SHA256:

| Artifact | SHA256 |
|---|---|
| `environment/task9-pytest.txt` | `575a72e64db47a0a717ee892f6fb35726118330074836911197915ac04a8868a` |
| `environment/task9-focused.txt` | `fac6d1081b4990c4d9a8b8b1d74839e58141aa8438148580ed5b71e7f9e789b0` |
| `environment/task9-pytest-superseded.txt` | `0958c10c1ca3c0ff3c1f9c6e43b65eea2ee9c3d9b92139ec62fd6fc3a7517b9e` |
| `environment/task9-hardware-observation-focused.txt` | `acc580e56460e273b4c23c6456048d5f2bc278a4bff6f87451d6c6a96b0184fe` |

## Remaining ownership and limits

Task 10 owns growth application, explicit optimizer state migration, causal fork RNG restoration and its through-growth backend tests. This task reconstructs grown topology/groups but does not implement or claim migration. Task 11 owns other model coordinates and ordinary widening. Task 12 owns actual teacher/fork execution, actual hardware/report linkage, enclosing end-to-end accounting, resource/disk forecasts, study completion and all scientific comparisons. No exploratory stopping rule has been frozen.

CPU interruption/resume is tested bitwise at complete optimizer updates, including non-epoch cursor, predictions, all parameters, optimizer tensors/steps, scheduler and next row IDs. Cross-backend resumption is rejected. Native MPS training recovery is not newly demonstrated here. No automatic fallback, unchecked accumulation, SGD training, expanded-depth model, larger workload calibration or external-checkpoint import is silently supported.

## Review fix round 1 — active best recovery and finite checkpoint boundaries

Addressed both Important findings in `task-9-review.md`, starting from `d0e34a1`. Separate fix commit: **`bc04f0a0f97a279db613abf6ddec361c599e03c7`**, `fix: preserve finite complete state across best checkpoint recovery`. Changed only `src/treeneuron/train.py`, `src/treeneuron/checkpoint.py`, `tests/test_train.py` and `tests/test_checkpoint.py`. No public docs/ledgers, branches/worktrees, subagents, T, freeze, backend producer source or downstream Task 10/11/12 functionality changed. Final working tree is clean.

The receiving-code-review skill was read and used to verify both findings against the current implementation before making changes. New tests reproduced all reviewed failures against the original implementation; the fix remained limited to the two checkpoint boundaries.

### Active best checkpoint semantics

An active `best.json` now contains the selected complete snapshot as both the current point and the embedded best point. The existing schema and public API signatures are unchanged. No model reconstruction, sampler draw or process RNG draw occurs during best publication. There is no recursive embedded snapshot.

Loading this checkpoint therefore produces an unfinished stage with its original start update, policy, completed-update count, patience count, best risk, model/optimizer/scheduler, sampler order/cursor/generator and RNGs. `continue_fit` follows the existing unfinished-stage path: it executes only the allowance remaining at that saved point and does not reevaluate the existing selected G point or start/reset a continuation stage. An active best saved at update 0 with max_updates=6 executes updates 1–6; one saved at update 2 executes updates 3–6. If neither improves, they return the complete original best at update 0 or 2 respectively. Later work present only in a separate last checkpoint is not silently grafted onto the earlier best point.

The nonzero-best regression exposed a second aspect of the same recovery finding: Torch CPU optimizer loading can retain the saved tensors' storage, and serialization can deduplicate the identical current/best snapshots in a best checkpoint. Without isolation, resuming live Adam updates advanced the embedded selected step from 2 to 6. `unpack_state` now copies the named optimizer state before `load_state_dict`, so current moments/steps cannot mutate the selected snapshot. This uses the existing tensor-copy helper; no extra recovery API or checkpoint role was added.

The two active-best tests construct G targets from known trajectory points, interrupt a real run at update 4, load its active best pointer at update 0 or 2, and resume with the original policy. They verify exact selected model/optimizer/scheduler/sampler/RNG equality; no RNG advancement from reconstruction; original start/policy; no duplicate selected-point evaluation; exact remaining update/evaluation lists; and final patience counters when no new G improvement occurs.

### Save/load finite-state boundary

`validate_tensor_state` is the shared internal check for complete serialized state collections. It checks the expected state collection keys and recursively rejects nonfinite tensors or floating scalars, including model values, Adam moments and steps, and stored scalar RNG state.

- `_publish_snapshot` checks current and embedded best tensor states after metadata validation but **before tensor serialization, version creation, or pointer publication**. Invalid direct public saves cannot replace the prior pointer or create extra artifacts. Both ordinary and best publication use this boundary.
- `load_checkpoint` checks both deserialized current and embedded best collections **before reconstructing a model, applying optimizer state, or restoring/consuming training RNG**. A valid checksum is insufficient to declare nonfinite state valid; a bad embedded best rejects the whole checkpoint even if current state is finite.
- `unpack_state` also validates its input collection so direct internal reconstruction obeys the same finite-state boundary. Named group checks, source/config/lock/split identity checks, checksum checks, atomic publication and bounded retention remain in force.

The save regressions inject NaN and infinity independently into model parameters, Adam `exp_avg`, `exp_avg_sq` and `step`, for current and embedded best state. They check rejection, byte-identical last-valid pointer, no extra version files, unchanged caller RNGs, and successful exact restoration of the prior valid checkpoint.

The load regressions construct deliberately malformed **pytest temporary checkpoint fixtures** whose payload and metadata checksums are internally consistent. They independently inject the same NaN/infinity cases in current and embedded best state, then require rejection before any optimizer application and with caller RNG state and pointer bytes untouched. No actual backend artifact, producer digest, source identity or scientific evidence file was patched to manufacture acceptance.

### RED, focused iteration, and final verification

All commands used the existing `.venv` and normal sandbox execution. No separate/native backend evidence refresh or doctor measurement was run; the required suites execute their usual genuine CPU test fixtures. Existing native MPS kernel evidence remains separately scoped to Task 8, and this fix makes no new native MPS training parity claim.

1. **RED**: `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py -q -k 'active_best_resume or invalid_tensor_save or valid_checksum_nonfinite' > environment/task9-fix1-red.txt 2>&1` — **34 failed, 30 deselected in 11.12s**. Active best cases hit the reviewed `NoneType` failure; invalid saves did not raise; malformed loads reached the optimizer-application sentinel.
2. First focused implementation run: `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py tests/test_artifacts.py -q` — **1 failed, 87 passed in 16.77s**. The nonzero active-best test demonstrated aliased Adam step 6 versus required selected step 2. Preserved as `environment/task9-fix1-focused-before-copy.txt`; not presented as passing evidence.
3. **Focused GREEN**, inspected before starting the full suite: `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py tests/test_artifacts.py -q > environment/task9-fix1-focused.txt 2>&1` — **88 passed in 17.14s**.
4. **One final full suite after the behavior fixes**: `.venv/bin/python -m pytest -q > environment/task9-fix1-pytest.txt 2>&1` — **441 passed, 48 skipped in 110.63s**. Skips remain sandbox MPS availability cases. This is fresh full-suite evidence for the final fix implementation; the original 406-pass/48-skip pre-fix run remains preserved under its original path and is not relabeled.
5. `git diff --check`, staged diff check and final clean-working-tree check passed. The final fix source was not changed after focused/full verification.

Final execution-source SHA256 after this fix: **`62b8201404c79fd58930c4d2f9a1b39047258274aa5d17b7b56b47f519127c00`**. Existing strict source identity still prevents loading checkpoints produced by different package contents; this fix does not introduce a cross-version checkpoint migration or bypass.

| Fix evidence | SHA256 |
|---|---|
| `environment/task9-fix1-red.txt` | `214745c794b3fdf7395dcc853e45cf06af607184e0c0693b7f57769190d84fdc` |
| `environment/task9-fix1-focused-before-copy.txt` | `55a5ad710b0d4316af62b792a8250febd85dfd81954865291d1ac456cd42e399` |
| `environment/task9-fix1-focused.txt` | `a66f415b89851c9efc255148d3110c56141fa5028607f9ab79cc136e94b43625` |
| `environment/task9-fix1-pytest.txt` | `516bf0e7c0ae3bc89e7263b45da484bef201d8a2b4bc647a216e023487ea7bed` |

Limitations/ownership are unchanged: checkpoints are self-created local artifacts, cross-backend resume is rejected, CPU exact recovery is demonstrated, and native MPS training recovery is not newly established. Task 10 still owns growth-state migration; Task 11 controls; Task 12 enclosing totals, actual hardware provenance, studies and comparisons. Root holds the public Task 9/D005 gate pending scoped re-review.
