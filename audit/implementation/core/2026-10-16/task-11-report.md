# Task 11: honest ordinary-capacity and static-gated causal controls

Implemented on dispatch baseline `c272fc9a62752f99a5ec76abe57ad5a13447f637`, current main, without branches/worktrees, helpers/reviewers, dependency/vendor work, protected evaluation, freeze, public docs, decision or ledger edits. The assigned private report and ignored environment evidence are the only documentation/evidence writes. TDD, systematic-debugging and verification-before-completion skills were used. Root owns independent review and milestone integration.

## APIs and exact factory recipes

`treeneuron.baselines.causal` exports `build_causal(method_id, config, *, parent=None, allocation=None) -> CausalPredictor`, `ordinary_width_brackets(d, target_parameters) -> tuple[int,int]`, `widen_parent(state, added_units, seed) -> GrowthManifest`, and the imported compact `StaticRegressor`. `CausalPredictor` implements the shared `fit(FitRequest)`, finite original-unit `[N]` `predict`, `save(Path)`, and `resource_summary`; `load(Path, *, expected_hashes)` uses the existing checked checkpoint loader. Parent-based factories produce independent model, optimizer, tensor storage, schedule, sampler and RNG forks. Factory construction preserves process RNG.

Factory recipe config is finite canonical JSON, separate from FitRequest runtime context. Extra/unknown recipe keys and unsupported roles reject.

| ID | Factory config | Parent/allocation |
|---|---|---|
| B1 | `{}` | Required canonical completed selected ordinary parent; no allocation |
| B2 fresh | `{'experiment': resolved_experiment_dict, 'variant': 'lower' or 'upper', 'target_parameters': 1697}` | Fresh; parent/allocation rejected |
| B2 warm | `{'variant': 'widen', 'added_units': 30 or 31, 'seed': separately_owned_seed}` | Required ordinary parent; no allocation |
| B3 partial | `{'experiment': resolved_experiment_dict, 'variant': 'partial', 'count': 16, 'coordinates': 'centered' or 'direct_leaf'}` | Fresh, counts only; no learned IDs/search charge |
| B3 full | `{'experiment': resolved_experiment_dict, 'variant': 'full', 'coordinates': 'centered' or 'direct_leaf'}` | Fresh; count resolved to h |
| B4 | `{}` | Required parent plus actual `Allocation(policy='random')` |
| B5 | `{'variant': 'gaussian' or 'adam' or 'sgd'}` | Required parent plus exactly matching actual allocation policy |

B4/B5 reuse the reviewed insertion consumer without changing its identity-only optimizer migration. Allocation/table ownership remains producer-owned: the consumer validates/records the supplied table hash, counts, IDs, contrasts, policy, work and attempts, but cannot recompute a missing producer table. CLI verification uses the real scorer and allocator. SGD here is a ranking diagnostic, not an alternate training optimizer.

At d16, target1697 gives widths94/95 and totals1693/1711, counting every scalar head parameter. Exact target1693 gives (94,94). Static h64 counts are1153 at m0,1697 at m16,3329 at m64, in both coordinate families. Direct-leaf selected units own only gate, left and right affine tensors; unselected units own the compact ordinary block. Full direct-leaf has no ordinary/base block. Centered selected units own center/gate/contrast. Shared-tensor formulas and chain-rule gradients agree in CPU64; independent Adam/regularization trajectories are deliberately not asserted equal.

## Resolved model identities and checkpoint ownership

`ModelConfig` narrowly adds `coordinates` (`selective`, `centered`, `direct_leaf`, `widened`), `static_count`, and `parent_width`. Existing default selective semantics remain. Resolved fresh wider/static configs carry their own model identity/hash; widening updates total h/coordinates/parent_width and config hash while retaining full original parent hashes and topology. No padding, model-kind alias, arbitrary registry, or generic NAS layer was introduced.

Added model kinds are `static_gated_relu_scalar` and `widened_relu_scalar`; ordinary/selectively grown remains `selective_relu_scalar`. `TrainingState`/`fit_parent`/`continue_fit`/existing checkpoint publication and loader remain the common path. Strict topology/config agreement, names, shapes, selected CPUfloat32 recipe, source/lock/runtime/config/data identities, finite tensors, full Adam groups/state, schedule, sampler/order/cursor and RNG are retained. New-model loading also checks saved model and Adam tensor dtypes/shapes/devices before PyTorch can silently cast them. Adapter load/save checks that B labels and recipe counts/policies match the loaded topology/provenance. No executable external deserialization was added.

Actual interruption/recovery tests cover fresh ordinary h94/95, full centered/direct-leaf h64, and warm widening64→94/95, with physical=logical batch128, a short last batch, cosine schedule, durable interruption after one update, exact model/optimizer/sampler/RNG replay, next row IDs and update row sequences. Existing checkpoint/growth suites were also run. CPU comparisons for preservation and replay use rtol0/atol0. Zero-unit widening is a complete event with no added parameters/group; existing selective zero-growth behavior remains tested by Task10 tests.

## Widening representation, slices and fairness

Widening uses separate active ordinary blocks, resolving the Adam per-Parameter scalar-step issue identified by root. `layer.base` keeps the original incoming values; `head.base` keeps the old head weight/bias; `layer.added` contains new random incoming weights/bias and `head.added.weight` contains zero outgoing entries, with **no extra outgoing bias**. The output is old head contribution plus the new contribution, preserving predictions bitwise on tested CPU paths. Initial outgoing gradients can be nonzero while new incoming gradients are zero. Normal shared AdamW decay still applies.

Every old tensor has an explicit model-qualified name/shape/full-slice mapping, including `head.weight → head.base.weight` and `head.bias → head.base.bias`. Old values, moments and step counters are copied exactly into independent replacement parameters. Added tensors have explicit zero moments and step0; their first update uses fresh Adam bias correction. A separate added group uses the current scheduled LR and original base LR. The scheduler retains its absolute position, horizon and min LR. There is no restart or reduced continuation allowance. The independent widening seed does not advance training RNG, sampler state, update/epoch/cursor/history or stage. Invalid inputs and failed preflight do not mutate the parent; the fully validated candidate is installed only after numerical evidence and state checks pass.

`GrowthManifest` remains the common return structure. Its documentation now explicitly distinguishes producer scopes: selective insertion names/counts are layer-relative and exclude the caller head; widening names/counts describe the complete regressor including head and record `manifest_scope='complete_regressor_including_head'` in growth metadata. The widening manifest is stored JSON-normalized with parent identity/topology, seed and explicit Adam-step policy. The Task10 identity-only helper is not weakened or reused for widening.

## FitRequest context, original provenance, budgets and costs

The accepted adapter runtime schema is exactly `request.config = {'development_data': original_DevelopmentData}`. This object is explicit **in-process context**, not a hidden/non-JSON value in persisted scientific config. `request.X_fit/y_fit/X_select/y_select` must match original F/G arrays byte-for-byte including dtype. Missing/malformed context, changed arrays, original manifest/teacher/schema/row/group/hash mismatches, or seed mismatch reject. The manifest is reconstructed/checked from its recorded commitments; raw identity is recomputed from stored hashes and row/schema identities without reading V labels. Original F/G hashes and raw/split/transform identities are retained in TrainingState/checkpoints. No synthetic substitute IDs, fabricated empty V partition, or fabricated data-access evidence are created. Tests place an object that raises on any V access into the original DevelopmentData after construction.

`request.seed` must match resolved optimizer seed. `request.budget` is exactly `{'max_updates': resolved_stage_allowance}`. Fresh controls train the explicitly configured base allowance; parent forks use the same configured continuation policy as selective growth. Retries of an active TrainingState keep its original allowance, selection/patience and schedule. Returned selected states follow the existing strict-improvement/earliest-tie G rule, with update-zero evaluation. No V HPO or primary F+G+V refit occurs.

Task12 must set fair **total** fresh training allowances, including giving static controls an opportunity to train from the start or use saved search cost. The diagnostic CLI's two-update stages are correctness checks, not a proposed predictive study budget. The adapter never invents a base-training or search charge for a fresh static model.

`predictor.run_paths` exposes the actual run immediately after creation, including on failure, so Task12 can attach enclosing hardware provenance and finalize its own run. `resource_summary` exposes run ID/directory, actual counts, backend receipt, context limitations and nested timer semantics. Adapter fit duration includes create_run plus fit/continuation; training phase/validation/preprocessing observations are nested and must not be summed again. Factory construction/probing/scoring are outside that adapter timer; Task12's enclosing timer must include them. The adapter does not fabricate parent/scoring costs, hardware identity, memory measurements or throughput. Loaded adapters have no claimed new fit duration/run identity. In-flight stage recovery uses the existing `load_checkpoint`/`continue_fit` APIs; convenience `CausalPredictor.load` is for explicit adapter snapshots carrying method recipe metadata.

## Actual CPU evidence and retained CLI artifacts

New `probe_causal_cpu` evaluates the actual topology at its actual batch size on isolated copies. An independent per-unit-reduction affine/leaf reference checks forward values and gradients. Separate AdamW optimizers consume the two gradient paths; optimizer parameter parity is checked. Torch weights-only serialization/reconstruction restores model and optimizer, then checks an exact subsequent update and every moment/step. Receipt records topology, all parameter shapes, total count, coordinate family, actual growth count, CPUfloat32 eager recipe, source digest, errors and scope. Existing parent backend validation and selected CPU recipe remain required. Receipts are checked against the resolved model and parameter counts; new variants always set performance_calibration_matches false. Supported bounds remain d≤16, h≤95 for ordinary/widened (parent width≤64), h≤64 for static, F≤2306 and batch≤128. These are correctness paths; no new resident-memory or cross-backend throughput claim is made.

Final CLI command:

```
.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --artifacts-dir environment/task11-causal-runs --output environment/task11-causal-capabilities.json > environment/task11-cli.txt 2>&1
```

Exit0; **13 actual B1–B5 recipes passed** at d16/h64 parent, batch128, F128/G32. The report records fresh h94/95 and warm64→94/95, partial m16/full m64 for both static coordinates, B4 random and B5 gaussian/Adam/SGD with **actual m16**. Each recipe was fit, predicted in original units, saved/reloaded, and given an exact matched next optimizer update. Complete diagnostic run artifacts/checkpoint pointers are retained under `environment/task11-causal-runs`; each recipe report records its absolute pointer path and metadata SHA256. B0 mean and ridge also execute with mean/closed-form ridge checks and roundtrips. CLI says B0 mean/ridge verified, spline pending EvidenceTask2, B1–B5 verified, B6/B7 pending, duplication deferred. No package-name substitute is counted as a verified recipe.

The shared GrowthManifest documentation change changes the existing backend producer implementation digest (that helper hashes growth/types.py). Per root ruling, its contract was corrected rather than left false. `environment/task11-refresh-cpu.py` ran the actual existing CPU `probe_backend('cpu', seed=1602030, batch_sizes=(128,))` under unchanged tolerances and wrote `environment/task11-parent-backend.json`; exit0, CPU passed, selected physical/logical batch128. It is a CPU-only correctness refresh, with no native ABBA rerun or comparison claim. Its source hash records the exact source at that refresh; subsequent causal-module refinements do not change the backend producer implementation hash. Prior `environment/report.json` is preserved byte-for-byte and its SHA256 is recorded in the new receipt. The initial successful CLI run before that documentation refresh is also preserved under `task11-*-initial` filenames; final evidence uses the new receipt. The old default environment/report.json now fails the current exact producer digest check: Task12 must explicitly select the refreshed receipt (e.g. set `train.BACKEND_REPORT_PATH` in its in-process harness) or have root update the canonical machine receipt; do not silently reuse/relabel the old native report.

## TDD and test commands

All failed evidence is retained. Initial RED was actual missing module/APIs, with real trained parent fixtures succeeding. Subsequent narrowly scoped REDs exposed the packed-counter unfairness, malformed receipt/state acceptance, missing CLI, recipe-label mismatch, default dtype dependency, missing artifact path exposure, and raw Adam moment casting. The first implementation failure also exposed the strict artifact backend-field allowlist (extended by one explicit causal receipt field) and an invalid gaussian test-fixture attempt count (corrected to a real allowed attempt). No tolerance was relaxed to make a check pass.

| Command (all redirected with `> evidence_path 2>&1`) | Evidence path | Result |
|---|---|---|
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -q` | `environment/task11-red.txt` | 14 failed in 4.34s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -q` | `environment/task11-green1.txt` | 6 failed, 8 passed in 4.40s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k fresh_adam -q` | `environment/task11-widen-red.txt` | 1 failed, 14 deselected in 3.83s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py -q` | `environment/task11-green2.txt` | 32 passed in 7.17s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k 'strict_new or causal_cli' -q` | `environment/task11-boundary-red.txt` | 5 failed, 4 passed, 16 deselected in 5.06s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py -q` | `environment/task11-green3.txt` | 42 passed in 12.77s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py tests/test_checkpoint.py tests/test_optimizer_growth.py -q` | `environment/task11-focused1.txt` | 138 passed in 39.44s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k adapter_load_rejects -q` | `environment/task11-recipe-red.txt` | 3 failed, 36 deselected in 4.07s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py tests/test_checkpoint.py tests/test_optimizer_growth.py -q` | `environment/task11-focused2.txt` | 141 passed in 41.37s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k process_default -q` | `environment/task11-dtype-red.txt` | 1 failed, 39 deselected in 4.31s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py -q` | `environment/task11-final-focused.txt` | 57 passed in 16.63s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k 'actual_large_shapes and extra0' -q` | `environment/task11-artifact-red.txt` | 1 failed, 39 deselected in 3.88s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py -q` | `environment/task11-focused-final.txt` | 57 passed in 17.65s; exit0 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k moment_dtype -q` | `environment/task11-moment-dtype-red.txt` | 1 failed, 40 deselected in 3.70s; exit1 |
| `.venv/bin/python -m pytest tests/test_causal_baselines.py -k 'strict_new_boundaries or process_default or adapter_load_rejects' -q` | `environment/task11-final-boundaries.txt` | 13 passed, 28 deselected in 5.82s; exit0 |
| `.venv/bin/python -m pytest -q` | `environment/task11-pytest.txt` | 520 passed, 48 skipped in 143.46s (0:02:23); exit0 |

Additional actual commands: `.venv/bin/python environment/task11-refresh-cpu.py > environment/task11-refresh-cpu.txt 2>&1` (exit0); the final CLI command above (exit0); and the initial CLI command using `--backend-report environment/report.json` without `--artifacts-dir` (exit0, subsequently renamed output/log to `task11-causal-capabilities-initial.json` / `task11-cli-initial.txt`). Both `git diff --check` and staged `git diff --cached --check` passed.

## Self-review, unsupported capabilities and Task12 integration

Self-review checked compact active coordinates/head accounting; explicit widening counter/mapping semantics; independent forks; original provenance/context separation; no V/T reads; finite scalar prediction contract; schedule origin/shared continuation updates; selected/active state ownership; exact CPU serialization/replay; recipe/topology matching; receipt shape/source validity; honest timing/performance labels; retained failures; and narrow shared-file changes.

Unsupported/deferred: native MPS/CUDA causal controls or recovery, alternate training optimizers, width>95/d>16/F>2306/batch>128, multiple growth events, duplication/symmetry control, B0 spline (EvidenceTask2), B6 validation-probed allocation and B7 official NODE (evidence plan), cross-process serialization of the runtime DevelopmentData context, practical vendor baselines, and predictive/throughput/superiority claims. B5 cheap causal diagnostics beyond the implemented gaussian/Adam/SGD ranking adapters remain evidence-plan work where requested. This task does not tune a budget or establish new growth gains.

For Task12: train/load one canonical completed selected ordinary parent under the final source identity; pass each fresh control an explicit finite resolved experiment; pass the original DevelopmentData as runtime context; use independent serial factories with their separately namespaced event seeds; supply original real allocations for B4/B5; preserve actual count0 cases and honest30/31 ordinary brackets. Set total fresh budgets fairly rather than inheriting a smaller allowance by accident. Pair continuation update/row/checkpoint policy, record complete parent/probe/scoring/HPO/fit costs in an enclosing timer without double counting, attach actual hardware provenance via run_paths, and finalize terminal attempts only once. For in-flight interruption use the existing strict training checkpoint APIs and fresh retry run identity. Make no throughput/native recovery claim from these CPU receipts.

## Final verification, hashes and commit

One final full-suite run after behavior stabilized: **520 passed, 48 skipped in 143.46s (0:02:23)**, exit0. Skips are existing MPS availability cases; no native causal training/recovery result is claimed. Final targeted boundaries: **13 passed, 28 deselected in 5.82s**. No production code changed after the final CLI/full-suite runs.

Final execution-source SHA256 (all package Python paths/content, HEAD independent): **`9ba92d00f5f2572913fc58cf08500286056de1044edb716bc66102b6ae8b424d`**.

`environment/task11-artifact-hashes.json` records SHA256 for all **230** task source/test files, top-level evidence artifacts and retained diagnostic run/checkpoint files. Hash of the index itself: **`b7509664972f71788c81c0f2a4db6202db0571dea6c02a3073106049121ce3bb`**. Checkpoint descriptors also bind their tensor payload hashes; the CLI report lists each final pointer and metadata digest.

| Artifact | SHA256 |
|---|---|
| `environment/task11-artifact-red.txt` | `7529856501238cf11a48d340ed39ce0b580c4f202208fcf85af1bef9a289ab3c` |
| `environment/task11-boundary-red.txt` | `ddf081b5dd41255404231492bda4c7f30a663f7231bd4df0ae33b67a09e399f3` |
| `environment/task11-causal-capabilities-initial.json` | `79273adda8fee36cd3f68d20efddd97e94e457520f5b3932ee0668f36afec33e` |
| `environment/task11-causal-capabilities.json` | `baea6b2fb4bcbed7de0d7cd1ac0e22818493659ae66dd3bce0e3b95ccdeb16c6` |
| `environment/task11-cli-initial.txt` | `79273adda8fee36cd3f68d20efddd97e94e457520f5b3932ee0668f36afec33e` |
| `environment/task11-cli.txt` | `baea6b2fb4bcbed7de0d7cd1ac0e22818493659ae66dd3bce0e3b95ccdeb16c6` |
| `environment/task11-dtype-red.txt` | `8febb7a87bebfd3f1d1960bdf96825c3b1fd6996bf1e2d28d50ad83946bfc6b1` |
| `environment/task11-final-boundaries.txt` | `e018a7389331a57ba713c6c72da61212721725eb1c7e7f7e5f056777d3d084c6` |
| `environment/task11-final-focused.txt` | `5c1d41266b9e8a3658956926e5707f41f85ec9b56173cfaf07a75c9af076ef6e` |
| `environment/task11-focused-final.txt` | `619f12ee330cb66ad39ec812cd5abe185dcd10953f9b611f9e8a0375a3509dfb` |
| `environment/task11-focused1.txt` | `64cb61ce8f710b823e0d135a8db22be647c59f26e541b1ae7020a40d453bdc11` |
| `environment/task11-focused2.txt` | `1368230293ca073e20e906f49c890fd90d8ae280f92547fbf60d5f8d7d7041dd` |
| `environment/task11-green1.txt` | `1fc7c6f7268ba0575f0bc49bb9c7297afa1e661d0391886307001298fd41a301` |
| `environment/task11-green2.txt` | `2e05c59c02653b1f32c017fda4f6a75e33d6081a7ca0d8d435d1cc159bd31ac6` |
| `environment/task11-green3.txt` | `56c59e095fb01b764216aeec2fa22045748d00c8d69eff16a3e9ff723d906a46` |
| `environment/task11-moment-dtype-red.txt` | `3af1413dae9b940e39a6c73e5357c223df263f24612266b045e078e65dd53243` |
| `environment/task11-parent-backend.json` | `44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197` |
| `environment/task11-pytest.txt` | `ef6325439ad93ae5fdabe1169f0d27343cd57446a415e5bb11acd248d5c23a28` |
| `environment/task11-recipe-red.txt` | `e30046b93f400a629d095ba986e0f14845a1aa65fd27ff33fec1bd03e49f8339` |
| `environment/task11-red.txt` | `0087f1a725cb66272c31c69bc28284357cf5aa31c51598110afc1378c53ce07a` |
| `environment/task11-refresh-cpu.py` | `2356760c15ec8159b32c785e86bae4424835ac1ef3e1223d6305392272e7224f` |
| `environment/task11-refresh-cpu.txt` | `0c64762167e66ab558b08621e8d26b3ac7afecb2a449e8f80263f464ad87f3d6` |
| `environment/task11-widen-red.txt` | `03cad06007d6d9cd2af06d600556fcab356c100eeaf910cbd0cca3d55b4d24c4` |
| `src/treeneuron/artifacts.py` | `0e5ad93cbe6e2fd8b76c6ab379af2ca687fa6a9e026c39e4bcf620118bd41d0c` |
| `src/treeneuron/baselines/causal.py` | `defe02b63a0cd35fb89352533b125a73bfa2c1f6606b55fb7ea4b555db682e46` |
| `src/treeneuron/baselines/causal_checks.py` | `077c0f77b8b70e424b89cc4fcbdfa716bb27cbf0abe315cca8b74c160bc18702` |
| `src/treeneuron/baselines/causal_evidence.py` | `8f2a2f3f770c3e0f4e5b576773d96f745d04165235dec0f39722cc99e80e6858` |
| `src/treeneuron/cli.py` | `73ee85fdcb371bfe2537f7d335040f97b9164716773269e48194e161762a4966` |
| `src/treeneuron/config.py` | `465640246578fc335e0e1926c6aa1a5b7519490fe3f13975f415763e5569fe96` |
| `src/treeneuron/growth/types.py` | `9d0305c4fbe42d36366424a52c3ed67316d3d0727daa3a2c28a10f1113f2cb50` |
| `src/treeneuron/models/static.py` | `83e8b2b7b5405b8b9285b91e8ad9b8c63134dda077ac7096ffbe5095bed8b428` |
| `src/treeneuron/models/widened.py` | `c75e86da320a1ab9d7daca8c827f5d55e5471896a4e82fc5a24aedd0469d98ba` |
| `src/treeneuron/train.py` | `3f6936748da4e92bdcdc7b96ff9be422b6127c6f070ebb7892e9e32b6edff874` |
| `tests/test_causal_baselines.py` | `d8cb1f123d74f1805380ea822514e5d324b59a115bf77cdd5af06bd9186d8279` |

Task commit: **`d8c5591` — `feat: add matched ordinary and static gated causal controls`**. Only the 11 listed task-owned source/test files were committed. Post-commit `git status --short` is empty. The assigned private report, exact failure/success logs, refreshed CPU receipt and retained diagnostic run artifacts remain ignored local evidence for root integration. Post-commit validation confirmed the final source digest and every one of the 230 artifact hashes unchanged. Independent review belongs to root; no helper/reviewer was spawned.

## Fix round 1: reject missing trained Adam history

The independent review found that missing or empty trained Adam records passed reconstruction as if they were valid lazy state. Reproduction confirmed that both deserialization and the live optimizer preflight could silently reset a trained parameter's Adam age/moments. This amendment supersedes the original boundary-completeness claim; the original report, CLI runs, logs and index above remain historical evidence for `d8c5591`, not relabeled evidence for this fix.

Added shared `validate_adam_history(update, model_kind, records)` and `validate_saved_adam_history(meta, tensors)` in train.py. Original ordinary/static tensors may have absent/empty lazy records only at selected update0. Every tensor must have a complete record at a positive selected update. Selective gate/contrast and widened added blocks require their producer-initialized complete records even at update0; an explicit complete **step0** added record remains valid beside inherited records at positive ages. No validator equates every Parameter's step to global update, changes a step/moment, or manufactures missing history. Named serialized group/identifier binding is validated before history lookup so missing entries cannot be hidden by a malformed binding.

The check runs in `pack_state`, the live growth/widening optimizer preflight, continuation preflight, and `unpack_state` before optimizer application/model reconstruction or restoration of caller RNG. Both current and embedded-best snapshots are checked at public checkpoint save/load boundaries before publication or any optimizer load; completed-last selection and in-flight continuation therefore cannot defer discovery of a corrupt best snapshot until after training. Existing finite/type/device/source/recipe checks remain. The replacement-object widening representation is unchanged, per root's clarified ruling; all old values/moment+step values, distinct ages, schedule and update semantics are retained.

Added `tests/test_adam_history.py` with **49** focused cases: deleted and empty trained records for ordinary, centered/direct-leaf static, old/new widening and old/new selective-growth parameters; rejection before optimizer loading with unchanged caller RNG; rejection by public save without replacing an existing pointer; growth/widening/continuation preflight; selected-update0 original lazy states; explicit new step0 records; exact valid roundtrip/next update; distinct inherited/fresh ages; and malformed embedded-best save/load (including recomputed checksums to reach semantic validation). Seven legitimate lazy/mixed-age cases passed in the initial RED run. An older checkpoint-only test manually created a growth group without its fresh records; its fixture now explicitly initializes zero moments and step0, matching the producer contract, while preserving its roundtrip assertions.

Only train.py, checkpoint.py, growth/insert.py, the existing checkpoint fixture and the new focused test file changed. No backend-hashed producer source changed. The existing `environment/task11-parent-backend.json` still matches `backend.implementation_hash()`; no CPU/native probe refresh, native ABBA, dependency, plan, public-doc, ledger, helper/reviewer, T/freeze or Task12 work occurred.

Exact verification commands and retained results:

| Command | Evidence | Result |
|---|---|---|
| `.venv/bin/python -m pytest tests/test_adam_history.py -q` | `environment/task11-fix1-red.txt` | Exit1; **42 failed, 7 passed in 13.97s**, confirming missing-history acceptance |
| `.venv/bin/python -m pytest tests/test_adam_history.py tests/test_checkpoint.py tests/test_optimizer_growth.py -q` | `environment/task11-fix1-green1.txt` | Exit1; **1 failed, 133 passed in 34.70s**; only the manually created growth group's missing-state fixture failed |
| `.venv/bin/python -m pytest tests/test_adam_history.py tests/test_causal_baselines.py tests/test_checkpoint.py tests/test_optimizer_growth.py -q` | `environment/task11-fix1-focused.txt` | Exit0; **175 passed in 51.94s** |
| `.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --artifacts-dir environment/task11-fix1-causal-runs --output environment/task11-fix1-causal-capabilities.json` | `environment/task11-fix1-cli.txt`, new capabilities JSON and retained fix1 runs | Exit0; **13 B1–B5 recipes**, B0 mean/ridge verified; previous pending/deferred capabilities unchanged |
| `.venv/bin/python environment/task11-fix1-preservation.py` | `environment/task11-fix1-preservation.txt` | Exit0; original **219 environment evidence files** and old index unchanged, existing CPU receipt applicable, new CLI/source identity verified |
| `.venv/bin/python -m pytest -q` | `environment/task11-fix1-pytest.txt` | Exit0; **569 passed, 48 skipped in 150.55s (0:02:30)**; one final full-suite run, existing MPS availability skips |

Every command above redirected stdout/stderr with `> evidence_path 2>&1`. All RED/intermediate failure logs were retained. Final CLI checkpoints and recipes use new `task11-fix1-*` paths and the new source identity; no historical checkpoint/source/evidence digest was patched. Original index SHA256 remains `b7509664972f71788c81c0f2a4db6202db0571dea6c02a3073106049121ce3bb`. Its source entries describe the old commit; its 219 environment artifacts remain byte-for-byte intact.

Self-review checked that lazy permission depends on selected state/topology rather than executed-attempt counts, that static contrast parameters remain eligible for original update0 lazy initialization, that explicit added records are mandatory without forcing equal ages, that the intermediate migration preflight checks only the registered old group, and that embedded-best checks precede caller RNG restoration/optimizer application. No production source changed after focused GREEN and final CLI/full-suite execution began. Existing CPU-only, shape/batch, runtime-context and pending-capability limitations remain; no new performance or predictive claim is made.

Final fix source SHA256: **`d4e6102013d8e3eb5d2618590d9479bae3783e11d89629ea27cfe62b17066601`**.

Final fix1 evidence index: `environment/task11-fix1-artifact-hashes.json`, **209 files**, SHA256 **`685c57969966273e42d20a01aa37dc82ffe80fd74e3e5f0af6099c1621ec279c`**. It covers the five changed source/test files, all new top-level fix1 evidence and every retained fix1 diagnostic run/checkpoint file. Both unstaged/staged whitespace checks pass before commit.

| Fix1 artifact | SHA256 |
|---|---|
| `environment/task11-fix1-causal-capabilities.json` | `cb44cde6c2b294d15822798b93e05d38a31b9f705829b683408cc0a1947a5a88` |
| `environment/task11-fix1-cli.txt` | `cb44cde6c2b294d15822798b93e05d38a31b9f705829b683408cc0a1947a5a88` |
| `environment/task11-fix1-focused.txt` | `b0447313b1130f3376d24f6a79c333b475004ec097ee9d49ac793c239ddc57fc` |
| `environment/task11-fix1-green1.txt` | `7b80b06cbc35b24e90cc1aa825ddbdeaffd3a33ef0a9f85a4cf0081195a1a0ad` |
| `environment/task11-fix1-preservation.py` | `e1fe1f07654514401290d5772ac1de6f91301487bbb9ba584d5b91d8951b98b1` |
| `environment/task11-fix1-preservation.txt` | `58907fd47b16951e9aa5d2aad6e776819ca41bdcaff8fe2b02a5b04fdcee8eab` |
| `environment/task11-fix1-pytest.txt` | `1423bc19a9fa35b4178eca5ba074a3aa5839dbdc02481f6031f4c0b13ce45b4b` |
| `environment/task11-fix1-red.txt` | `6ef3d589f0b517b3f4a9d68d8321ad0b7be9858602b9f656aaa4c871a9e92edb` |
| `src/treeneuron/checkpoint.py` | `7a2061b48245efb9158274edf0be68e77d418e3394fc2751f173c6253e633606` |
| `src/treeneuron/growth/insert.py` | `4202564f10a3081b1ef9862e6cef9a4bba5f72a0781e4acca7600ed2d5b6b412` |
| `src/treeneuron/train.py` | `da94a2a6fae4a874a97615dd07ab4441ff0cc730dbcc984dcda3c95218fbef0e` |
| `tests/test_adam_history.py` | `7b10b3fbcb6081627572c1a36a9a484dfe749f78d56534df9ee904de491799ee` |
| `tests/test_checkpoint.py` | `01a560b895904244c4f2a7b4df84548feb878e1a837a009ee21ecf1053702cad` |

Fix commit: **`75e7742` — `fix: reject incomplete trained Adam checkpoint history`**. Five task-owned files were committed; post-commit working tree is clean. Source digest and all **209** fix1 index entries were rechecked after commit, and all **219** original environment evidence files remained unchanged. No production changes followed the successful focused/full/CLI checks. Ready for the same reviewer's scoped re-review; no additional reviewer was spawned.
