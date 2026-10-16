# Task 10: state-preserving one-event specialization

Implementation status: implemented and verified; task commit recorded below. Baseline: `9e2d25a7b9c69500dbc8e3c177c128fd221ecd41` on current `main`. No branch/worktree/subagents/helpers/reviewers, dependency changes, protected evaluation, freeze, public documentation, ledger, or study work was performed.

## Files and API

- Added `src/treeneuron/growth/insert.py` (232 lines): `insert_growth(state: TrainingState, allocation: Allocation) -> GrowthManifest` and `migrate_optimizer(state: TrainingState, manifest: GrowthManifest) -> None`.
- Added `tests/test_optimizer_growth.py`: 38 parameterized test cases covering insertion, migration, allocation validation, preservation, checkpoint reconstruction and continuation.
- Existing `train.py`, `checkpoint.py`, layer, allocator, manifests, dependency lock and backend implementation sources were not changed. Existing checkpoint interfaces already support compact grown topology and named groups; no schema bypass or hash rewriting was needed.
- Task evidence is preserved under ignored local `environment/task10-*.txt`; this report is in the assigned local SDD directory. Root owns milestone/public evidence updates.

## Public insertion semantics

Insertion accepts a completed selected ordinary parent from `fit_parent` or the completed `best.json`. It explicitly rejects an active stage, any completed state still carrying `best_snapshot` (including a completed unselected `last.json`), a prior growth event, or existing growth metadata. It does not silently select/copy the parent: old Parameter objects must survive. To insert after completed-last recovery, load the completed selected best checkpoint first. The existing training recovery interface still owns completed-last selection for continuation.

Preflight checks the exact named model/optimizer coverage and object correspondence; shared AdamW options; live scheduled LR, base LR and absolute schedule position/horizon; old moment shapes, dtype/device, finite values and valid steps; and the existing complete checkpoint metadata/tensor boundaries. It retains the original scientific identities. It validates allocation counts, sorted unique CPU int64 IDs/ranges, CPU float64 bias-last contrast dimensions/finiteness, supported policy, allocation seed, SHA256-shaped score table identity, direction attempt counts and work counts. The consumer validates/records the supplied table hash; it does not have the score table and therefore cannot independently recompute that producer hash or repeat eligibility/scoring.

The allocation contrasts are explicitly cast to the CPU float32 parent dtype/device. A finite CPU64 contrast that overflows or becomes an all-zero row on conversion rejects before mutation. Insertion never selects new IDs or resamples directions. Both original CPU64 allocation metadata and actual inserted float32 contrasts are recorded.

The layer's existing grow implementation supplies zero gates, compact gate/contrast modules, original base objects, sorted topology and the layer-local manifest. New module construction is surrounded by capture/restore of the **process** Python, NumPy, Torch CPU and available MPS RNG state. `state.rng_states`, sampler generator, order, cursor, epoch, update, history, preprocessing, identities and completed stage remain unchanged. Allocation randomness remains producer-owned and separately seeded; its seed/policy/attempts are stored without consuming the training stream.

On success, model topology, optimizer groups/states, scheduler groups and metadata form a checkpointable state. There is no intermediate checkpoint/event publication. On an insertion exception, original layer modules/buffer/event flag, optimizer group list, old state objects, scheduler base LR list and growth metadata are restored; process RNG is restored in `finally`. A regression injects failure after migration to verify rollback of the already-added group and new state too.

## Migration and schedule semantics

The migration helper is narrow: call immediately after the existing layer `grow`, or use `insert_growth` for the complete transaction. It accepts the one-event layer-local manifest with exactly the old `base.weight`/`base.bias` and new compact gate/contrast names. It validates full identity mappings with explicit shapes and half-open slices, prefixes layer names when matching model names, and separately preserves the unchanged head. Invalid/unmapped/duplicate mappings and incompatible groups reject before optimizer mutation. Slice widening is explicitly unsupported by this helper and belongs to Task 11 adapters.

Old Parameter objects, optimizer state dictionaries and every old moment/step tensor are retained by identity, as are existing group dictionaries/options/order. Selected update-zero parents retain their lazy absent old state. The helper creates one added group when growth is nonempty, using the shared current scheduled LR and original options. It adds the **original base LR** to `scheduler.base_lrs`, leaving absolute position and horizon unchanged; subsequent cosine/constant scheduling therefore follows the existing origin rather than restarting at the decayed insertion LR.

Every new parameter receives explicit CPU float32 scalar `step=0`, `exp_avg=0` and `exp_avg_sq=0`. Its first actual Adam update advances to step 1; old states continue from their own steps. The first post-growth ordinary-parameter update agrees exactly with an independently, manually migrated reference that uses PyTorch's lazy fresh state. At zero gates, initial contrast gradients are zero and gate gradients can be nonzero. Both gate and contrast receive normal shared weight decay on that update.

An empty allocation records the completed event and its manifest, adds no module/group/moment or scheduler entry, and preserves the teacher and its continuation. Repeated helper calls are rejected using stored `optimizer_migration`, including empty events; repeated insertion rejects the layer event flag. The helper's marker stores a JSON-normalized manifest. Full insertion additionally stores `schema_version`, `manifest`, complete `allocation`, `inserted_contrasts`, insertion topology, insertion update and insertion scheduler state. `GrowthManifest.from_dict` reconstructs the manifest after JSON decoding. Existing checkpoint optimizer-group metadata carries the actual group recipe.

## Supported paths and explicit boundaries

Supported: selected CPU float32 eager scalar selective-regressor parent; existing one-event identity-mapping growth; shared AdamW with foreach false and the locked producer recipe; current constant/cosine scheduler; adam/random/sgd-ranking/gaussian/keep Allocation policies; zero growth and selected update-zero parents; exact same-backend CPU checkpoint recovery immediately before/after insertion and during continuation.

Unsupported: native MPS insertion/training recovery, CUDA or other backend, alternate optimizer (the `sgd` allocation policy only changes ranking), modified Adam options/heterogeneous recipes, widened or renamed/replaced old tensors, active training-stage insertion, repeated growth, stale source-identity checkpoints, and cross-backend recovery. CPU support is explicit in validation rather than an unverified MPS recovery claim. Existing Task 8 native kernel evidence remains separately scoped and unchanged. No new native backend probe was run.

Tests use the existing real CPU backend-evidence fixture. The exact initial coordinate is also exercised: `d=16,h=64,m_requested=16,τ=1,epsilon=0.05`, ordinary scalar count 1,153, added count 544, grown count 1,697. The coordinate test uses a controlled nonzero allocation for count/insertion verification; it is not a scored predictive study or a claimed RMS contrast experiment. Separate integration tests use actual scorer/allocator outputs for all five supported allocation policies.

## TDD, debugging and evidence

TDD, systematic-debugging and verification-before-completion skills were used; no delegation occurred.

1. Initial RED command: `.venv/bin/python -m pytest tests/test_optimizer_growth.py -q > environment/task10-red.txt 2>&1`. Exit 1; **31 failed in 8.56s**. Failures were missing `growth.insert` module/APIs, as required. Real trained-parent fixtures succeeded and selected nonzero old Adam history.
2. First implementation command: `.venv/bin/python -m pytest tests/test_optimizer_growth.py -q > environment/task10-green1.txt 2>&1`. Exit 1; **2 failed, 29 passed in 9.34s**. Both failures were manifest tuple/list representation drift after JSON recovery (`after` and `during`); model/optimizer/scheduler/RNG recovery comparisons had already passed. Root cause: `GrowthManifest.to_dict()` retains tuples, JSON changes them to lists. Stored manifests were normalized through canonical JSON before publication. This output is preserved as an intermediate failure, not GREEN evidence.
3. Initial focused GREEN command: `.venv/bin/python -m pytest tests/test_optimizer_growth.py tests/test_checkpoint.py -q > environment/task10-green2.txt 2>&1`. Exit 0; **78 passed in 20.19s**.
4. Expanded focused command: `.venv/bin/python -m pytest tests/test_optimizer_growth.py tests/test_checkpoint.py -q > environment/task10-focused.txt 2>&1`. Exit 0; **85 passed in 21.64s**. Added real allocator policy integration, lazy old-state update-zero support, and exact initial-coordinate counts/reconstruction. This result was inspected before starting the full suite.

CPU exact tolerances are `rtol=0, atol=0`. Recovery checks compare model tensors, named optimizer state/groups, schedule, selected update/epoch/cursor, sampler permutation/generator, stored RNG collections, growth metadata, next row IDs and continuation row sequence. The before/after boundaries save and reload the complete state at those points; the during boundary raises a controlled KeyboardInterrupt after the first actual continuation update and recovers the last durable checkpoint into a fresh retry run. Invalid allocation/parent/manifest cases verify no mutation, and insertion failure rollback is exercised after a successful optimizer migration.

## Review and next-task integration

Self-review checked named coverage and head exclusion from layer counts, explicit slices, moment and group object preservation, scheduler origin, JSON roundtripping, RNG capture boundaries, completed-stage ownership, selected-state requirements, empty events, malformed-input preflight and rollback. Root will dispatch independent review.

Task 11 can consume the stable selected-parent state/recipe and should implement separate explicit slice adapters for widening rather than relaxing this identity-only helper. Task 12 should load or clone the same canonical selected parent under the current package source identity, obtain allocations from the shared score table, call `insert_growth` for each serial fork, save the returned fully consistent state if required, and call `continue_fit` with the paired continuation policy and a fresh RunPaths. Completed stage is preserved so continuation starts its own update-zero G evaluation; no stale pre-growth best snapshot remains to remove the branches. Zero-growth allocation remains a complete teacher event. Study-level linkage, actual hardware provenance, timings and predictive comparison results remain Task 12 responsibilities.

No unresolved CPU correctness failure is known from focused testing. The supported-path restrictions above are intentional, not evidence for native MPS training or future controls.

## Final full verification and artifact identities

One final full-suite run after behavior stabilized: `.venv/bin/python -m pytest -q > environment/task10-pytest.txt 2>&1`. Exit 0; **479 passed, 48 skipped in 130.49s (0:02:10)**. Skips are the existing sandbox MPS availability cases; no native migration/recovery result is claimed. No production code was changed after the focused/full runs. `git diff --check` and staged `git diff --cached --check` passed before commit.

Final execution-source SHA256 (all package Python, HEAD independent): **`a57aa94f0b126a06d5012b81e9675233b46bd564b3a8609df6090e87dc632deb`**.

| Artifact | SHA256 |
|---|---|
| `src/treeneuron/growth/insert.py` | `34017547db5cfc84f56c21cc6dbb58c9e342c6e0da7101137d4b5a9b9713981e` |
| `tests/test_optimizer_growth.py` | `06efe51b32e54d00c22e5fad6fa1ce4ab051faea3c25bb5d3fc17d791f465183` |
| `environment/task10-focused.txt` | `7e9bdc9f23a5e0b6c9c089e9a50816a856b8050ec33c33eaf5a3dd65de783753` |
| `environment/task10-green1.txt` | `888ebcfe4f0cbe6230b72c3b629005445ed6f155af23da931cf58928c5777f81` |
| `environment/task10-green2.txt` | `c80eb23115e36b8d81a63d469c3b4cc1e09cf558e4109e938f483d0d51895ae4` |
| `environment/task10-pytest.txt` | `31b29e6557881db8b0c7f8c57ea4ec73b0e980a92a0e19c061d14b6a10a9cc08` |
| `environment/task10-red.txt` | `149b6cce2ffcb271e4c713e38db338e5e2b28a3d525c63e1d56fb1777223f7a6` |

## Commit

Task implementation commit: `f7416bb` — `feat: preserve optimizer state across one-event specialization`. Only the new insertion module and its tests were committed. Post-commit `git status --short` was empty. The report and detailed test artifacts remain at their assigned ignored local evidence paths for root integration.
