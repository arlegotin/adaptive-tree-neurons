# Decision record

## D000 — 16 October 2026: specification and gated implementation handoff

**Inputs:** User-supplied research proposal, cut off 16 October 2026; empty repository on current branch `main`; local read-only hardware/tool discovery. No existing project code or repository guidance was present.

**Evidence:** [Machine/source record](docs/research/2026-10-16-evidence-and-machine-record.md); [builder check and plan review](docs/superpowers/reviews/2026-10-16-selective-tree-neurons-review.md). Actual memory is 38,654,705,664 bytes (36 GiB), logical CPUs 14, OS macOS 26.7.1, native arm64, free disk approximately 21 GiB. Default Python is 3.14.6 without Torch. The supplied numerical checks remain historical Linux evidence; they were not executed locally during planning.

**Decision:** Adopt [the specification](docs/superpowers/specs/2026-10-16-selective-tree-neurons-design.md) and split execution into [a core plan](docs/superpowers/plans/2026-10-16-selective-tree-neurons-core.md) and [a gated evidence plan](docs/superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md). Preserve H1 as the sole initial confirmatory superiority hypothesis. A valid negative or inconclusive outcome satisfies the research purpose when its declared study and audit are complete. Use the current branch; no worktree or branch switch.

**Handoff choices:** The specification explicitly identifies author-chosen gate-type probabilities, positive-control distribution/noise, smooth gate orientation, Energy geometry grouping, date boundary rounding, all-missing fill, growth input conventions and checkpoint/tie rules. Frozen-artifact/documentation commits may differ from the pinned source commit only when execution/scientific input digests match. These are visible draft implementation decisions, not experimental findings or claims of user preference.

**State:** Planning documents written; no WP is complete. No dependencies installed, datasets/weights acquired, predictive models trained, confirmatory teachers/tests generated, or protocol frozen. No scientific result is claimed. The fresh builder review resolved one missing control definition. Plan self-review checks and their outcomes are recorded separately.

**Unresolved execution gates:** Resolve native compatible Python/PyTorch and core lock; test CPU/MPS operations and reference algebra; measure current RAM pressure/storage and resource forecasts; verify consequential current sources/licenses/assets, including R12’s unresolved exact overlap. Runtime budgets, numerical tolerances and confirmatory counts must come from pilot measurements, never guessed to complete the document.

**Next step:** Execute Core Task 1 when implementation begins. Its first real discovery command is `uname -m`; subsequent commands/acceptance are in that task. `python -m treeneuron.cli doctor --output environment/report.json` is a proposed interface that Task 1 must implement, not an existing command. Start no comparative claim before the prerequisites, mathematics and harness gates pass.

Future completed work packages append inputs/hashes, evidence, decision, blockers and next command here. Do not relabel this planning entry as WP00 completion.

## D001 — 16 October 2026: native core prerequisites established

**Inputs/evidence:** Core Task 1 on current `main`; pre-acquisition forecast reserved 5 GiB installation/cache headroom against ~21 GiB free. `uv` 0.11.29 installed managed native CPython 3.12.13 inside `.uv-python`, with `.uv-cache` and `.venv` inside the checkout. No system Python changes. Actual directory sizes: interpreter 70 MiB, cache 763 MiB, environment 754 MiB (filesystem sharing means these are not necessarily additive physical usage); free disk still ~21 GiB. Initial RAM reservation 9 GiB. Current swap used ~6206.62 MiB; measure pressure/swap deltas before work.

**Decision:** Resolve stable packages from the registry under Python 3.12 rather than guessing a Torch minimum or following the stale version selector. Lock SHA-256 `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`: Torch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, pandas 3.0.6, scikit-learn 1.9.1, PyYAML 6.0.3, development pytest 9.1.1. These are the spec core packages, with resolver-selected transitive dependencies and setuptools build backend. Native compiled wheel tags are cp312 arm64; pytest is pure Python. License expressions/metadata, bundled notices and their hashes are recorded in `provenance/sources.json`; separate data/weight terms are not inferred.

**Verification:** Expected TDD RED: four tests failed for missing `treeneuron` APIs/CLI. Doctor file tests subsequently passed (4/4); required full suite verification is recorded in the task report. The implemented CLI produces finite JSON, retains denied counters/reasons/return codes, redacts serial/UUIDs, records installed package imports, lock/source digests and fallback settings, and exits nonzero for failed indispensable CPU imports. Runtime reports are ignored: `environment/report.json` (sandboxed), `environment/report-native.json` (permitted native observation). Both observe arm64/Python 3.12.13; CPU imports succeed. Sandboxed sysctl reads fail and MPS availability is false; native observations give 36 GiB memory/14 CPUs and Torch MPS built/available true. Native fallback was explicitly `0`; recommended working set 30,150,672,384 bytes, allocated 0 bytes, driver 475,136 bytes. These counters overlap unified RAM and are not summed. No scientific MPS kernels or parity tests ran; WP00 remains open until Task 8.

**Source status:** Official R12 PDF retrieval succeeded; SHA-256 `f1628581fc83f79a4c32d4013299e56f2cf3693eb50821cb355b8c2c5926cd65`. OpenReview still challenges access. Scoped text/algorithm inspection is recorded in `protocol/prior_art_delta.md`; exact overlap and broad novelty stay provisional. No weights/data, vendor code copying, agreement acceptance, or predictive experiment occurred.

**Next step:** Core Task 2 (strict experiment configuration/run ledger), then numerical gates in plan order. No confirmatory or practical-comparison claim is unlocked.

## D002 — 16 October 2026: strict records and local reference algebra verified

**Inputs/evidence:** Reviewed Core Task 2 commits `bcd0da7`/`baf30ed` and Core Task 3 commit `a75197a`, using the locked native Python 3.12.13 / Torch 2.14.1 environment. Configuration rejects unknown keys, coercion, nonfinite values and protected phases; resolved AdamW semantics and the noise-free conditional-slope control were corrected during review. The run ledger preserves identities, null/reason observations, exclusive terminal records and atomic file publication. Multi-file crash reconciliation remains later recovery work.

**Local math:** Fresh CPU float64 subprocesses, one thread, seeds 1602026/1602027 reproduced the preserved A/B recipes. A forward error `8.882e-16`, shared-coordinate gradient error `1.332e-15`, gate-gradient error `2.776e-17`, insertion change `0.0`, initial contrast-gradient norm `0.0` and gate-gradient norm `0.4451935795`. Loss decreased `18.7354818901 → 18.7354681759` in the tiny gate-only step. B analytic-gradient error `5.551e-17`, rescaling/permutation score errors below `2.1e-17`, first-Adam predicted/observed decrease `8.941316578e-7 / 8.941315848e-7`, relative error `8.168e-8`. All required reference assertions passed.

**Artifact identities:** Ignored but regenerable `environment/reference-a.json` SHA-256 `2bfbdbc3fb4e4ae70e24c2320917d080857707a5aadc111d1ab74c0a395e561b`; `reference-b.json` `20f06dc44a4626dcd230a9bd844838c19e165914bf34105d940d600306997faa`; aggregate `reference-math.json` `0a7a01d36448e064179b410429069d459e47e2fd127365b11c2788bec66d6099`. Tracked reference script hashes: A `e14dc49d6654223033a511ad1605df0c2b7ff5f70715b3f2d9be31f8e772fb22`, B `f75f7308970aeace6ab614ffc8bf0736db4e698a7bc784b9f5557701fb3810e2`.

**Decision/limits:** Reference algebra is locally consistent. Core package layer/score math and CPU/MPS kernel gates remain unimplemented and explicitly pending in the CLI. No predictive model or allocation benchmark ran; WP01 is not complete. Task reviews found no blocking issue after fixes. Carry the reference-output schema hardening recommendation into Task 4 and the failed-OS-probe stdout diagnostic recommendation into Task 8. Training/run integration must populate actual manifest observations rather than leave initial placeholders stale.

**Verification/next command:** Core Task 2 full suite 62 passed; Core Task 3 focused 11/full 73 passed. Reproduce references with `.venv/bin/python -m treeneuron.cli verify-math --output environment/reference-math.json`. Next implementation: Core Task 4, compact selective affine preactivation and package math verification.

## D003 — 16 October 2026: compact layer, development data and F-only sanity pipeline verified

**Inputs/evidence:** Reviewed Core Task4 commit `a5790ff` and Task5 commits `b877fc7`/`272e4ea`. Full suites grew from 121 to 194 passing tests. Compact preactivation layer identities/gradients have maximum error about `3.6e-15` on the tested CPU float64 cases; insertion preserves predictions exactly. Topology export reconstructs compact gate/contrast tensors before state loading; dtype/device remain explicit caller choices. One-event/empty-event validation and stable base parameter identity pass. Optimizer migration and MPS kernel gates remain later work.

**Data:** Four independent regime components, separate teacher/reference/input/noise domains, 65,536-reference normalization, F/G/V-only development rows and explicit predictor allowlists are implemented. Ordered feature schema contributes to dataset and assignment identity. Primary isotropic and axis teacher/observation bytes are preserved through the review fixes. Generated arrays are immutable complete NumPy float64; later neural execution copies/converts to float32. These synthetic types are not raw missing/categorical transfer tables.

**Rulings I made before predictive experiments:**

- Interpret additive sine/quadratic as a sum of univariate functions of observed coordinates, `.3*(u·sin(x)+v·(x²−1))`. A random-projection sum introduces observed-coordinate interactions and would disadvantage the required per-feature spline control. If wrong, this exploratory control equation/version and its baselines need rework; the primary population is unchanged.
- Define rotated robustness as a stored orthogonal transform of the same axis-aligned teacher, including paired coefficients, reference inputs and observation coordinates/noise. Rotating an isotropic coefficient population cannot isolate axis alignment. If wrong, this exploratory coordinate/version must be relabeled; the primary population is unchanged.
- Reject project-reserved metadata names in raw DataFrame preprocessing, matching the synthetic observation boundary, with an explicit error. This prevents accidental use of our own record schema; source-specific predictor allowlists still belong to dataset adapters. No columns are silently dropped and arbitrary target names are not inferred. If wrong, narrow or rename this generic validation; source allowlists and primary numerical inputs are unchanged.

**Version/acceptance boundaries:** Controls use `exploratory-controls-v2`; rotated primary teachers use `axis-paired-rotated-teacher-v2`. Additive+rotated is rejected because arbitrary coordinate rotation destroys observed-coordinate additivity. Same-seed axis/rotated primary labels differ by at most `3.6e-15` in the reported cases (zero relative tolerance, `1e-13` test bound). Serialized equations, coefficient metadata, distributions and threshold/prevalence consistency are validated. Oracle-gate affine diagnostic features omit the sine component, so this is not an exact Bayes oracle. Neither generation nor algebra demonstrates allocation or predictive advantage.

**F-only sanity pipeline:** Reviewed Core Task6 commit `94553f3`; 65 focused / 259 full-suite tests pass without warnings. Transform state stores F medians/population mean/std, zero-scale and all-missing flags, exact schema, target inversion and typed ordinal categories with reserved unknown/missing code0. Missingness indicators are appended only for columns missing in F; later missing values in previously complete numerical columns use F medians without changing width. Complete synthetic inputs therefore retain d=16. Ordinal categories are a sanity-control choice, not a faithful strong-baseline recipe. Strict finite JSON restores mean/ridge state; both use F alone, ridge leaves its intercept unpenalized, and predictions return finite original-unit [N] arrays. Metrics retain raw and F-variance-normalized errors, null/reason for zero target variance, group-macro/worst-group and tail summaries. Source-specific leakage guards and raw transfer acceptance remain unimplemented.

**State/next step:** No predictive model, protected T or frozen protocol exists. The synthetic/preprocessing/B0 portion of WP02 passes its reviewed tests; full grouped/temporal transfer acceptance remains open. Tasks7/8 close package score/backend gates; WP00/WP01 are not complete. Next implementation is Core Task7, followed by `.venv/bin/python -m treeneuron.cli verify-math` with its real package-score gate. No allocation or practical superiority claim is unlocked.

## D004 — 16 October 2026: package score and native backend gates

**Package score evidence:** Reviewed Task7 commit `7d637f5`; 41 focused and 301 full-suite tests pass. Real package checks report gate-gradient error `4.336808689942018e-19`, batched-contrast error `4.5102810375396984e-17`, positive-rescaling score error `8.673617379884035e-19`, exact per-ID random contrast agreement, and first gate-only Adam predicted/observed decrease `2.8955387489426842e-9 / 2.895539141789527e-9` (relative error `1.3567314304246556e-7`). Signed moments are accumulated in bounded blocks from summed row-loss derivatives, with spectra/gaps/RMS and per-unit eligibility retained. This verifies implementation consistency, not prediction or allocation quality.

**Supported scope/cost:** One ordinary first selective layer, out-of-place ReLU, scalar linear head and squared error. In-place activations explicitly reject after a self-audit regression; the primary architecture uses out-of-place ReLU. Scoring works on a private CPU64 evaluation copy and recomputes forwards per unit block, real overhead to be measured. The shared-table random control uses exact unit-specific contrasts. Its `standalone_contrast_units_required` is a work requirement, not an executed cheaper runtime; practical random-cost claims require a measured sampled-ID path or an explicit unmeasured limit.

**Native backend evidence:** Task8 commits `2f03c82` / `a4f4cdc`; final native focused tests 109 passed, full sandbox suite 365 passed with 48 reasoned MPS skips. Actual CPU/MPS float32 GEMM, gates, ReLU, index-select/copy forward/backward, layer/head gradients, AdamW `foreach=False`, serialization and resumed update pass against CPU64 references. Three seeds and magnitudes `{0.1,1,10}` validate fixed tolerances `atol=7.62939453125e-6, rtol=3.0517578125e-5`; maximum normalized error is below 0.076 on both backends. Within-backend serialization/resumed-update errors are zero in these tiny checks. Full training-state/growth recovery remains unimplemented. Native subprocesses set fallback0 before Torch import; scoring never puts float64 tensors on MPS. The sandbox's unavailable MPS and denied counters are separately retained.

**Calibration/selection:** Actual d16/h64/F2306 ABBA CPU→MPS→MPS→CPU blocks give median complete-path costs **36.103167 ms CPU / 66.896271 ms MPS**. Select CPU with one Torch CPU thread (observed interop14), physical/logical batch128 and CPU64 scoring batch128. The path times base epoch, model CPU64 copy/score, matched insertion and grown epoch directly, with synchronized MPS timers and retained warmups/samples. It uses resident data, fresh optimizers per phase and score unit blocks16; cold import/initial transfers and causal optimizer migration are outside this calibration. This is not a full-study runtime forecast or general CPU speed claim. Accumulation versus unsplit maximum parameter difference is `2.98e-8`. Warmup/timing stability uses a recorded 20% calibration precision with capped-instability status; all four direct complete-path records meet it. CPU block1 base/b128 and MPS block2 base/b64 reached the 11-sample cap without isolated-stage timing stability; those observations remain flagged and included. WP05 must remeasure actual pipeline precision rather than treating these stage medians as precise latency estimates. Later workload/precision/shape/thread changes require remeasurement.

**Resources/artifact identities:** Native reports record normal pressure, AC, 25.56–26.64GiB coarse pressure-derived headroom and no swap increase in final blocks. The guard reserves9GiB plus512MiB first-use allowance; it was raised from128MiB after observing about250MiB MPS RSS growth. No actual reserve violation occurred. Superseded reports are retained under `environment/task8-before-guard-*`. RSS/MPS allocated/driver counters overlap unified memory and remain separate. Final native report SHA256 `60f5cc801909f2b83b1963c814432f5e1bc8a3e37a3ebdf649aa8e8b239b18dc`; complete math report `d8d8c5a99911c7429e37489f89ac0fcb304a6a30d7238631c0ece0a1614e7162`. Source digest `0f054cb12e1ace3a9f46fc8edbb452d3a89b2e48c58c92fc6c4bb202a84cab51` and lock/runtime digests bind the evidence; docs-only commits do not invalidate unchanged execution source. A [curated index](docs/research/2026-10-16-core-gate-evidence.json) preserves the scalar record and hashes without bulk runtime artifacts. Failed-probe stdout diagnostics are fixed and tested. Review found that imported batch labels could bypass the live physical-batch contract; the fix applies shared positive-Python-int/doubling/divisor-of128 bounds to live inputs, stage rows and selected/complete-path recipes, and records the actual measured recipe. Source-bound native evidence was regenerated; the prior report remains archived under `task8-before-batch-validation-*`.

**Decision/next step:** **WP00/WP01 core gates pass after task and scoped fix reviews.** Core prerequisites and implementation algebra/kernel gates meet their tested numerical contracts; R12 full text has been acquired but exact score/growth novelty remains provisional as recorded in prior_art_delta. No held-out predictive/allocation result, protected T or frozen protocol exists. Next is Core Task9, deterministic parent/continuation training with complete G-best state and atomic recovery; it must populate actual manifest observations. Reproduce complete math with `.venv/bin/python -m treeneuron.cli verify-math --backend-report environment/report.json --output environment/task8-verify-math.json`; generate fresh native doctor evidence with fallback0 if source/lock/runtime changes. Later WP02/WP03 practical data/baseline scope and WP04 paired experiment remain open.

**Ruling for later checkpoint implementation:** Retain durable last-valid and G-best payloads plus required interruption/final artifacts; permit cleanup only of unreferenced superseded within-run payloads after durable new pointers are published. Keeping every evaluation checkpoint could exhaust local disk during cohorts. Complete curves/events, failed and losing run records, and selected/final models remain intact. If wrong, an intermediate weight snapshot requires deterministic replay; no reported run outcome is removed. This is an implementation requirement for Task9, not a claim that recovery/retention has passed yet.

## D005 — 16 October 2026: complete CPU training-state recovery

**Research purpose:** The next comparison asks whether choosing neurons by score helps more than random placement. Both forks must inherit the same learned weights, optimizer history and subsequent data stream. Complete recovery makes that comparison attributable to allocation rather than an accidental training advantage.

**Implementation/evidence:** Reviewed Task9 commits `3836847`, `d0e34a1`, `bc04f0a`; final focused tests **88 passed**, full suite **441 passed / 48 sandbox MPS skips**. Complete strict G-best selection includes stage update zero and restores weights, named optimizer groups/moments, scheduler, sampler cursor/order and RNGs. CPU interruption/resume is exact in the tested cases. Active best checkpoints at updates0/2 execute the original remaining allowance; live optimizer tensors cannot mutate embedded selected state. Public save/load rejects nonfinite current or embedded state before publication or optimizer/RNG changes. Immutable checksum-bound payloads publish atomic last-valid/best pointers with bounded successful-version retention; source/config/lock/split/runtime identity checks remain active. Review closed two Important checkpoint defects; retained RED and intermediate failing logs document the fixes. The [curated recovery evidence](docs/research/2026-10-16-training-recovery-evidence.json) records hashes and limits. No native MPS complete training recovery or through-growth optimizer migration is established by this milestone.

**Cost/provenance boundary:** The measured training-attempt region includes F conversion, G evaluation and checkpoint/event IO; F-transform fitting is separate and validation is nested. Initial model/optimizer construction and backend-evidence validation are outside that interval. Task12 must measure enclosing end-to-end fit/study cost including method-specific setup. Runtime identity is distinct from hardware provenance; unknown hardware observations retain a reason until an enclosing runner supplies actual evidence.

**Ruling:** Permit explicitly recorded small CPU diagnostic configurations with physical=logical<=128 for correctness tests, while recording actual dimensions, rows, batches and performance-calibration scope mismatch. Small sampler/resume fixtures are required by the training plan and must not be disguised as the measured primary128 workload. If wrong, rerun those diagnostic checks at the full128 recipe; primary experiment settings and calibration measurements are unchanged. Larger/other-backend/workload paths need relevant parity/probe evidence rather than inherited throughput claims.

**Source preparation:** A [source-only NODE memo](docs/research/2026-10-16-node-recipe-notes.md) pins official regression notebooks, sparse functions, F-only initialization/preprocessing, checkpoint-weight averaging and notices. QHAdam1.1.0 is an immutable dependency candidate with inspected defaults, not an executed historical reproduction or compatibility result. No NODE adapter, package installation or predictive evaluation has occurred.

**Decision/next step:** Task9 recovery gate passes after scoped review. Core Task10 now inserts growth while preserving old optimizer/scheduler state and paired update streams; ordinary/static gated controls and the first exploratory comparison follow. No held-out allocation/generalization result, protected T or frozen protocol exists. Broader raw transfer/baseline, pilot, confirmation and audit gates remain open.

## D006 — 16 October 2026: one-event growth preserves training state

**Research purpose:** Score-selected and random placement must inherit identical learned functions and optimizer history. At insertion the added gates start at zero and predictions are preserved; later differences can be attributed to the declared allocation/direction policies under paired training.

**Implementation/evidence:** Reviewed Task10 commit `f7416bb`, no findings. Focused tests **85 passed**; full suite **479 passed / 48 sandbox MPS skips**. Compact insertion retains old base/head Parameter objects, moment tensors, steps, sampler/order/RNG and scheduler clock. New Adam state starts at zero with the current scheduled LR and the original future schedule. Explicit name/shape/slice validation and rollback prevent half-migrated state. Empty allocations retain the teacher. Checkpoint recovery immediately before/after insertion and during continuation agrees exactly on the tested CPU path. At d16/h64/m16, total parameters change **1,153 → 1,697** (544 added) with exact initial prediction preservation. Actual scorer/allocator policies integrate; the count fixture itself uses a controlled allocation and is not a predictive experiment. The [growth-state evidence](docs/research/2026-10-16-growth-state-evidence.json) records final hashes and limits.

**Scope/limits:** Insertion requires a completed selected ordinary CPU float32 AdamW parent. Active stages and completed last checkpoints still carrying embedded G-best reject; use the completed selected best first. The migration helper accepts full identity slices; ordinary widening needs a separate explicit adapter. Native MPS insertion/training recovery is unsupported and not inferred from earlier kernel tests. Mathematical/state integrity does not establish allocation or prediction advantage. No protected T or frozen protocol exists.

**Decision/next step:** Task10 state-integrity gate passes. Core Task11 now supplies ordinary capacity/widening and static gated controls; Task12 executes the first complete exploratory G/V comparison. B6/B7, cost calibration, pilot, freeze, independent confirmation and audit remain future evidence gates.

## D007 — 16 October 2026: causal controls and source-freeze sequencing

**Research purpose/evidence:** Reviewed Task11 commits `d8c5591` / `75e7742` add continued-parent, nearest ordinary parameter brackets, ordinary warm widening, compact centered/direct-leaf static gates and random/direction/ranking adapters. Full suite **569 passed / 48 sandbox MPS skips**; focused fix suite175 passed, and the actual CLI exercised **13 causal recipes** with retained fit/predict/save/reload/next-update artifacts. Ordinary widths 94/95 have1693/1711 parameters around the1697 target; static partial/full counts are1697/3329. Shared-coordinate formulas agree; independent Adam trajectories are not claimed equal. B0 mean/ridge are verified; spline, B6/B7 and secondary duplication remain explicitly pending/deferred. The [curated control evidence](docs/research/2026-10-16-causal-control-evidence.json) records final scalar results, recipe topology and hashes. These tiny diagnostic fits are not a comparative predictive study.

**Recovery/provenance:** Review exposed missing or empty trained Adam records being accepted as lazy state. The fix rejects incomplete trained/added histories before public current/embedded checkpoint save/load and growth/continuation preflight. Original update-zero lazy records remain valid; distinct inherited/fresh ages are preserved without manufacturing history. Actual CPU forward/backward/optimizer/reload receipts cover required wider/static topologies. Runtime DevelopmentData context is explicit and F/G/manifest identities are checked without V labels; persisted scientific configs remain finite JSON. New variants make no calibrated throughput claim. The shared manifest documentation now distinguishes layer-local selective counts from complete-regressor widening counts. That changes the producer digest: the current applicable CPU-only parent receipt is `environment/task11-parent-backend.json` (SHA256 `44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197`). Historical native `environment/report.json` remains byte-for-byte preserved; it is not accepted by the new live exact-hash check. Current runs must explicitly select applicable evidence. No new native MPS control recovery or GPU comparison is claimed.

**Ruling6:** Use separate compact added incoming/outgoing blocks for warm ordinary growth, with explicit name/full-slice mappings and fresh Adam step0. A packed enlarged tensor has only one scalar Adam age and cannot retain both histories. I initially also required old object identity. If wrong, rerun B2 as a separately labeled packed-counter variant with its counter rule declared; primary H1 is unchanged.

**Ruling7:** Implement, test, review and commit H1 execution/analysis/schema/CLI/audit code from Evidence Tasks8/9/10/13 before definitive pilot/design validation and real freeze; actual protected cohort/audit remain afterward. A [source-only dependency audit](docs/research/2026-10-16-freeze-order-audit.md) confirmed that the literal plan froze before later Python additions, which would correctly invalidate protected-source and package-wide checkpoint identities. The evidence plan now preserves task numbers while separating implementation accepted/evidence pending from completed scientific gates. If wrong, rework pre-freeze orchestration and regenerate an unaccessed protocol; scientific changes after T still require amendment and fresh independent confirmation. No source guard, hypothesis, information access or required evidence is weakened.

**Ruling8, clarification of6:** Exact old predictions/parameter values, moment/step values, optimizer age, schedule and update semantics are required. Explicit mapped replacement Parameters installed as an independently validated atomic candidate satisfy B2; pointer identity was overspecified. If wrong, implement and rerun an identity-preserving widening installation/variant. Primary H1 is unchanged.

**Decision/next step:** Task11 passes after scoped fix review. Core Task12 now implements and executes the primary exploratory G/V paired study plus the separate conditional-slope positive control. No held-out allocation advantage, protected T, frozen protocol or independent confirmation exists. Required B6/B7, calibrated design, protected cohort and audit remain future gates.

## D008 — 16 October 2026: exploratory checkpoint cost and first paired study

**Engineering evidence:** The first committed Task12 study (`2afd9cb`, execution source `b3a24ea14987ee1608c7b434549489eb5a34e8d4416749f3040d751da8881eea`) was stopped for cost before any allocation fork. Its retained directory is `artifacts/exploratory/475e0835ed6c4e3090a7650fb51c6dab`: terminal interrupted, 253.604 seconds, 1,701 complete update events and a durable update-1701 checkpoint. The latest measured throughput implied roughly 2h28m for the 42,000-update maximum study allowance. Parent G MSE had improved to 0.23450, but no allocation comparison existed at the stop. A diagnostic on a private copy measured median compute-only update 0.470ms versus snapshot publication/pruning 149.24ms. Profiled canonical JSON validation of growing histories dominated; this is persistence overhead, not evidence against the neural mechanism.

**Ruling9:** Retain and stop the first exploratory study as an interrupted engineering diagnostic after measured throughput projected roughly 2h28m for 42000 updates; introduce an explicitly configured and hashed durable-checkpoint cadence with snapshots at G selection, growth, stage-end and interruption boundaries, retaining every update event and identical model-selection/training rules across forks, then run fresh studies under committed source — every-update full-history snapshotting is not required by the spec and creates substantial avoidable artifact I/O before a larger pilot; no allocation outcome motivated the stop — if wrong, replay at most the declared cadence from the last valid complete-update snapshot or rerun the affected exploratory comparison under cadence1; no protected data have been accessed.

**Cadence semantics:** `training.checkpoint_interval` defaults to 1; both exploratory configs declare 25, matching G evaluation cadence. Every G evaluation and stage boundary still publishes; growth's stage-update-zero evaluation publishes before its first optimizer update. An orderly interruption after complete update bookkeeping publishes that state. An interruption inside Adam or G bookkeeping retains the preceding valid atomic pointer and its exact sampler/RNG/optimizer state; uncommitted progress is replayed. Complete per-update events remain. Config/hash/state checks reject a changed cadence on resume. A fresh 200-update same-data/seed comparison took 6.269s at cadence1 and 0.528s at cadence25, with exactly matching update rows/losses, G evaluations, selected update and predictions. This small diagnostic is not a full-study speed claim. The interrupted study remains source-pinned and is never resumed under new source.

**Research result:** Does the score place conditional growth better than random at equal capacity? The first primary teacher gives negative evidence: top G/V MSE **0.166548/0.147531**, random **0.159741/0.143840**, both at **1,697 parameters**. Keep/sham gives **0.183698/0.160662** at **1,153 parameters**, with exactly identical predictions. Warm ordinary widths 94/95 (1,693/1,711 parameters) give V **0.156438/0.158814**, fresh ordinary brackets **0.174201/0.162524**. Gaussian directions give **0.145013**, SGD ranking **0.149247**; static partial/full centered gives **0.189883/0.199565**, direct-leaf **0.183686/0.177849**. Random placement has the lowest descriptive V MSE among this executed inventory. Conditional growth remains interesting on this teacher; the score has not demonstrated a placement advantage. No losing branch is removed.

**Scope and checks:** Each complete study uses one teacher and optimizer seed, G-selected checkpoints and descriptive V, with no T generated/accessed. Eight serial warm forks independently load the same complete parent; inherited parameters/Adam state, initial predictions, contrasts, RNG/sampler/schedule and executed row streams are checked. The separate d2/noise0/m2 positive control is a different mechanism diagnostic, not another primary-population replicate: keep/top/random V MSE is approximately 0.001338, with no useful extra-growth advantage. Keep’s learned left/right slopes are +0.98609/−0.96362 against the intended +1/−1. Parent standardized training-loss means fall 0.65325→0.04690 across first/last 100 updates; keep continues 0.04678→0.02667. Full-centered static gates (641 parameters) give V 0.00119771, compared with 0.00131570 for the 269-parameter fresh ordinary controls. The local fresh-Adam gate-step forecast matches observed tiny-step descent; this algebra check does not imply long-run allocation superiority.

**Measured cost and limits:** The primary study used 39,950 of at most 42,000 physical updates and took 611.655 s enclosing generation, hardware observations, setup/probes, training, predictions and artifact work (excluding imports and final hash-index publication). Its shared parent cost 14.942 s, shared event 0.0524 s including scoring 0.0380 s; top/random additional regions 39.711/39.798 s. The separate 2D study used 42,000 updates and 713.470 s. Both contain 17 method records and pass rebuilt pairing/prediction checks. Manifest creation→finish UTC intervals, including final index publication, are 612.115/713.852 s; these overlap the monotonic study timers and are not added. Exact process-launch→exit duration is unmeasured. These are realized paired-study costs; a standalone sampled-ID scorer remains explicitly unmeasured. Nested validation is not added twice. RSS is process-lifetime high-water, not incremental branch memory. All actual runtime/hardware/config/source identities, full controls and scalar results are linked in the [curated evidence](docs/research/2026-10-16-exploratory-paired-evidence.json).

**Prospective allowance amendment:** Top/Gaussian/SGD still improved at the final primary G evaluation. Parent/keep/random and ordinary controls also made several strict improvements in the last 20 evaluations; a false final-point improvement flag is not a plateau certificate. Preserve the exact executed `explore.yaml` and `explore-positive-control.yaml` at 2,000 + 2,000 updates. Separate `explore-next-allowance.yaml` and `explore-positive-control-next-allowance.yaml` declare **unexecuted, calibration-pending** 4,000 + 4,000 allowances symmetrically for all forks and fresh controls, without changing batch, patience, score or selection rules. No repeated inventory search, variance pilot, top-win threshold or convergence claim is made. Definitive ceilings require later calibration before WP05.

**Next gate:** Task 12 implementation and complete exploratory evidence go to independent root review. Mechanism/harness acceptance does not require the score to win. Required spline/B6/B7, search/timing and all pre-freeze H1 execution/analysis/schema/audit code precede the definitive H1 pilot and real source freeze; protected confirmation and audit remain later. Broader grouped/temporal raw data and practical adapters gate the separate transfer study. No practical or population superiority claim is unlocked.

**Scoped review corrections (I1/I2):** Review found that an interruption could publish an update before its event bookkeeping completed, and that known coordinator/child metadata remained unset in generic manifests. Commit `61cf0f6` keeps snapshots unsafe until update/G bookkeeping returns and adds strict study-aware observation fields. Final G/V artifacts and metrics now precede child completion; prediction failures retain a failed child. The final suite passes **596 tests / 48 MPS skips**, with the causal CLI rerun. A committed small CPU integration study completes all 17 methods and verifies all 22 coordinator/child manifests. This is engineering verification, not another predictive comparison. The [fix evidence](docs/research/2026-10-16-exploratory-fix1-evidence.json) records commands, identities and hashes. An [append-only correction sidecar](docs/research/2026-10-16-exploratory-manifest-corrections-fix1.json) binds 46 historical manifests to their original hashes and supplies only observations derivable from existing artifacts. The original studies, their source identities, curated evidence and 1,026-file index remain intact. No large inventory was rerun, no old checkpoint was loaded under new source, and no scientific allowance/selection rule changed. Independent scoped rereview remains the next gate; minor review items remain in whole-core triage.


## D009 — 16 October 2026: accepted exploratory task and estimable confirmation design

**Reviewed milestone:** Core Task12 and its scoped fix are accepted. The review found an interruption window before update-event bookkeeping and known observations left as placeholders in run manifests. Commits `61cf0f6` / `97e3414` fix both; focused checks115 passed, full suite596 passed /48 sandbox MPS skips, and causal CLI checks passed. A separate small17-method engineering execution verifies22 populated coordinator/child manifests and correct terminal prediction links. The [fix evidence](docs/research/2026-10-16-exploratory-fix1-evidence.json) records current source `6223991050f82a7e40c35e20112c893ea852988a3abcebc1e4c8ce60077d1d5d`. The [46-manifest correction overlay](docs/research/2026-10-16-exploratory-manifest-corrections-fix1.json) is append-only and derives known metadata from hashed historical records. Original predictive studies and measured times remain pinned to `46d537c` / source `ec52c0…`; no old checkpoint is resumed under new source. The numerical findings in D008 stand at their stated exploratory scope. Whole-core review is still pending, including two deferred robustness items in stream-audit duplication and global-setting restoration on final I/O failure.

**Ruling10:** Retain the normal-approximation teacher count as `K_calc`, then require `K=max(K_calc, K_pilot)` using the completed accepted variance-calibration pilot's independent-teacher count and its existing minimum of eight. Reject singleton or zero/undefined-variance inference as support for superiority; retain finite-sample/resampling diagnostics and the estimation/resource-limited branches. The literal calculation can produce one teacher, yielding a point bootstrap interval and undefined Student-t sensitivity. The study-specific floor closes that defect without inventing an application effect threshold. If wrong, the cohort spends extra measured runtime/storage relative to `K_calc` and may still require an estimation-only interpretation; no fixed floor guarantees95% coverage. No protected data have been accessed. The [preflight](docs/research/2026-10-16-h1-design-preflight.md) and [dated pre-freeze amendment](protocol/amendments/2026-10-16-minimum-confirmation-cohort.md) preserve the reasoning. The specification and future EvidenceTask8/10 calculation/validation requirements now state this rule.

**Next action:** Whole-core review, then required H1 adapters (spline/B6 and NODE), search/timing, and the complete pre-freeze implementation phases. Definitive pilot/design, real freeze, the independent cohort and audit follow in that order. Raw transfer data and practical adapters remain gates for a separate transfer study. Task12 acceptance is acceptance of a working experiment; it does not establish a beneficial allocation score or complete the independent research objective.

## D010 — 16 October 2026: whole-core acceptance and evidence handoff

**Review outcome:** The whole-core review found three Important and three Minor issues. One combined fix wave (`a3561c5`, evidence `df21950`) closes all six: unsupported growth selectors now reject before execution; ineligible score diagnostics export as finite records; child provenance precedes training failures; paired stream metadata binds to events/counts; process settings restore after publication failures; and the declared F-derived diagnostics cover both coordinate families. The scoped review reports all findings addressed and no new breakage. The [reviewed handoff](docs/research/2026-10-16-reviewed-core-handoff.md) links the archived reports, exact interfaces and ten chronological rulings. No further final fix wave was needed.

**Verification and provenance:** 224 focused tests and 629 full-suite tests pass, with 48 sandbox MPS skips; 13 implemented causal recipes pass. A separate tiny run with 17 methods verifies 22 manifests and 21 pretraining registrations, taking 5.779797208 seconds within its declared boundary. Package source is `39f697d80eba928e5bbcce019beed912c358a20e3b62995d71c9577517e6a032`, lock `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`. The [new evidence](docs/research/2026-10-16-core-final-fix-evidence.json) has SHA256 `b4193dda2d2ddde6e4617a80d7f982cec42c0a2e1baff6030b624df69e34868e`; root verified all 494 indexed files. Kernel/scoring/allocation producers are unchanged, so the genuine existing CPU receipt remains applicable. Historical predictive studies, the interrupted attempt and earlier engineering runs retain their original sources and hashes. The primary result remains top worse than random on one exploratory teacher; there is no new predictive claim.

**Decision/next action:** Accept Core Tasks 1–12 and continue the user-authorized research work on current `main`. Full causal-baseline readiness still requires spline/B6 and B7. Implement Evidence Task 2's capability matrix, spline ridge and costed independent single-unit probes; its causal verification remains explicitly phased until Task 3 closes NODE. Search/timing and all H1 execution/analysis/audit code precede definitive pilot/design and real freeze. No protected T, frozen protocol, completed independent cohort or WP09 scientific audit exists. Transfer/practical work remains separately gated. Next proposed verification after Task 2 implementation is `.venv/bin/python -m treeneuron.cli verify-baselines --track causal` with an applicable explicit backend receipt and newly named evidence output.

## D011 — 16 October 2026: restricted validation probes and additive baseline contracts

**Implemented:** Evidence Task2 adds the restricted ANT-inspired B6 allocation control, additive spline ridge and a truthful adapter capability matrix. Each eligible single-unit probe and continued keep starts independently from the same complete G-selected parent, uses the same short F allowance and G checkpoint policy, and retains its setup/training/checkpoint cost. G ranks the probes with unit-ID tie breaking; the requested effective growth count stays fixed even if every probe loses to keep. Joint insertion uses a fresh original parent and normal continuation. Failures retain incomplete/not-attempted candidates and prevent exhaustive-selection claims. Spline knots and coefficients fit F only; V-based alpha search remains Task7. No V/T labels enter either fitting path.

**Verification:** Implementation commit `dca45c0` passes 189 focused tests (1 existing MPS skip) and the single final full suite: **657 passed, 48 sandbox/MPS skips in202.92s**. The retained causal CLI verifies B0 mean/ridge/spline and14 neural recipes through B6. Its separate B6 fixture is d16/h2, F128/G32, CPU eager float32 with CPU64 scores, logical=physical batch128, two parent/probe/joint updates. All three probes (keep and two units) are retained; their enclosing region measured0.422686709s, including setup/checkpoint work and excluding prior parent fit/shared scoring and final report publication. This is a tiny correctness receipt, not throughput, predictive efficacy, calibration or full ANT reproduction evidence. The [curated record](docs/research/2026-10-16-evidence2-capabilities.json) links exact outputs and a329-file hash index; bulk checkpoints remain local.

**Provenance and limits:** Package source `e0a45e52199338996ac9573233adb2ca9b7d342d930c683e379d7bb0b90e1e35`; the lock and backend producer hashes are unchanged, so the genuine existing CPU receipt remains applicable. Every final CLI parent/checkpoint was created under the new source. Original predictive and engineering evidence hashes remain unchanged; no old checkpoint was loaded under a new source. The original one-teacher comparison therefore remains the scientific observation, with no added superiority claim.

**Next gate:** Root-owned independent Task2 review, then Task3's faithful pinned NODE implementation and verification. B7 remains explicitly pending and the CLI reports `track_ready=false` despite B0–B6 implementation verification. Practical verification exits nonzero with mandatory B8/B9/B10/B11/B14 pending and optional B12/B13 reasoned unavailable. Full WP03/causal readiness, H1 cost calibration/pilot, real freeze and protected confirmation remain future gates. The next engineering command after NODE implementation is `.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --artifacts-dir environment/evidence3-causal-runs --output environment/evidence3-causal.json`, subject to rechecking backend-producer identity at that time.

**Review acceptance:** The [Task 2 review](audit/implementation/evidence/2026-10-16/task-2-review.md) approves the B0–B6 implementation with no Critical or Important findings. Root independently verified all 329 indexed files, current source and preserved evidence hashes. One Minor lifecycle issue remains: the dedicated CLI probe coordinator is left running although its probe report and children are complete. Task 3's verification integration will close this caller-owned run after observations, without closing a broader search coordinator inside the generic probe helper. The original Task 2 receipt remains unchanged. Task 2 implementation is accepted; its full-readiness step waits for B7 in Task 3. No new efficacy or confirmation claim is added.

## D012 — 16 October 2026: pinned NODE causal gate and bounded full-size cost

**Implemented:** Evidence Task3 adds the faithful local `year_node_shallow` NODE adapter at implementation commit `41d0178`:2048 trees, one layer, depth6, three channels, entmax15/entmoid15 and channel0 tree mean. Original NODE/QHOptim source, hashes and MIT notices are archived; QHAdam1.1.0 executes unchanged on native CPU. F alone fits noisy quantiles/target scaling and the ordered initialization prefix. Full F is streamed. G uses newest-five parameter snapshots with native report100/strict patience5000 semantics, restores raw state for training and selects matching model/optimizer state. Checkpoints retain initialization/preprocessing and bind source/lock/runtime. The dedicated causal CLI probe coordinator now terminalizes on success or failure without changing the generic greedy helper's ownership.

**Verification:** The single final full suite passes **678 tests,48 existing sandbox/MPS skips**,249.77s. The required NODE command passes19 tests; independent archived vendor entries exercise forward/backward, initialization, QHAdam and checkpoint reduction semantics. The fresh retained causal CLI verifies B0–B7 and all14 existing B1–B6 neural recipes with `track_ready=true`. The [curated evidence](docs/research/2026-10-16-evidence3-node.json) links a521-file hash index and the [implementation report](audit/implementation/evidence/2026-10-16/task-3-report.md). Source is `6146d731db228c5a9495f19262139622dec462b9707208476f84400342710fd7`; core lock/backend producer remain unchanged and the prior genuine matching CPU receipt was not rewritten. Existing predictive evidence retains its original provenance.

**Measured scope:** Exact full-size anchor on synthetic F1025/G1024/d16 initializes from1000 ordered F rows, then processes batches512/512/1: initialization1.004s, full-F epoch6.448s, prediction1024 rows1.275s, dedicated process peak RSS10,002,169,856bytes. Worker body10.087s/startup-inclusive11.936s; explicit endpoint save/reload0.339/0.049s, exact reload predictions. These are one-run engineering costs, not primaryF2306, default convergence, throughput superiority or quality calibration. The three-update budget ends before report100: `selection_status=unavailable_at_budget`, `official_selection_completed=false`, `quality_selection_eligible=false`. The unselected endpoint cannot be a quality winner.

**Decision and next gate:** The executable WP03 causal correctness/readiness gate passes, subject to root independent review. Task7 must measure full default quality/stopping budgets and handle unavailable-at-budget methods explicitly before subsequent H1 gates. Historical sklearn/QHOptim resolutions are unknown; this is a pinned modern local recipe, not published YEAR-score reproduction. Notebook UI/logging/temporary checkpoint I/O is replaced by in-memory recent snapshots and explicit endpoint persistence. No YEAR/split/weight assets, T labels, freeze, pilot, new predictive inventory or independent confirmation were used. Full default5000+ training was not requested or executed here.

**Review acceptance:** The [Task 3 review](audit/implementation/evidence/2026-10-16/task-3-review.md) approves spec compliance and code quality with zero Critical/Important issues. Root independently verified all521 indexed size/SHA pairs and the actual test log. Accept Tasks2/3 causal correctness readiness; retain the bounded endpoint's quality exclusion. Two minor observations remain visible for Task7/final triage: per-visit Python row traces can inflate long-run cost, and expected vendor/short-fixture warnings should be captured narrowly in tests. Neither changes the original numerical files or earlier evidence. Continue Task7 search/timing, then pre-freeze pilot/design, protected evaluation and audit implementation; no independent scientific claim is unlocked.

## D013 — 16 October 2026: isolated exploratory search and timing boundaries

**Implementation decision:** Task7 searches implemented numeric H1 adapters through serial fresh workers, with F/G labels and V features in hashed non-pickle arrays. A redacted, validated development view preserves the original split commitments without handing V labels to fitting. The coordinator alone scores V. Configured/official anchors run first and remain in the record; failed, rejected and timed-out work is charged. A NODE endpoint before its first official G report is unavailable at budget and cannot win. Static controls retain the same enclosing seconds and can spend saved allocation work on additional trials. B4 currently pays for the full score table; this does not implement or claim a cheaper standalone sampled-ID scorer.

**Reproducibility decision:** Optuna4.5.0 and its resolved dependencies are a hash-locked overlay under `locks/search`, appended after the verified native core environment. Neither the core lock nor NODE lock is changed. Immutable finite-JSON sampler archives replay every ordered ask/tell under the pinned implementation and reject changed source/runtime/data/support identities. Resume creates a new run and budget; historical sampler outcomes inform proposals, while historical model artifacts cannot silently become the new attempt's winner. Every new package source requires fresh parents/checkpoints.

**Runtime and cost decision:** Search workers use one CPU intra-op and one inter-op thread, with native BLAS thread limits and MPS fallback disabled before imports. A fresh receipt runs the existing numerical CPU producer under that actual1/1 runtime; the historical14-inter-op receipt remains unchanged. The search enclosure starts before input/sampler publication and includes imports, all parent/base/transform/scoring/probe/continuation/checkpoint work, V scoring and failures. It ends after final sampler publication, before terminal metrics/manifest publication; nested durations are never added. Acquisition is outside. A late result is unavailable even when the underlying fit succeeded.

**Timing decision:** Warm host timing includes all rows, remainder batches, transforms and host output. Explicit causal resident timing excludes transforms/transfers/target inversion and returns standardized float32 resident output. Fresh-process cold timing separates imports, load and first host prediction, retaining serialization and process-wall costs separately; it does not claim OS cache eviction. ABBA or seeded balanced random blocks, rolling warmup, median/tail diagnostic precision, excluded work and separate overlapping RSS/MPS counters remain in hashed raw records. The smallest successful measured batch reaching95% of peak median complete-call throughput is an engineering choice, subject to observed memory guards and pending final precision policy.

**Phase boundary:** This completes machinery only after independent review. Bounded CPU calibration records will accompany the implementation; no full-quality/default plateau or numeric WP05 budget is selected here. Final design follows all pre-freeze H1 Tasks8/9/10/13 implementation under the final committed identities. No protected T, real freeze, complete independent cohort or practical transfer inventory is authorized by these diagnostics. Task3 row-trace overhead and expected warnings remain visible triage; no long NODE fit or vendor numerical edit is made. Next: one final full test suite, commit tested code, then fresh bounded measurements and timing-summary reconstruction before root review.

**Implementation verification:** The final combined NODE/search/timing check passes47 tests with28 existing vendor warnings. The full suite passes706 tests,48 existing sandbox/MPS skips and28 existing warnings in305.98s. Its preceding full attempt exposed only a strict tqdm version conflict between the existing NODE reference site and the new overlay; the search dependency was re-resolved to the existing4.67.1 pin, with the failed log preserved. No source guard, core/NODE lock, numerical tolerance or historical receipt was weakened. Current package source is `a64b5f98ac9d533fe35df1c5b81f4582aeb11336eda5d0290fb9cad69756dd6e`.

**Measured defect and correction:** The first native bounded calibration under `d6248aa` completed seven search records, then exposed a warmup nontermination when an early-stable batch remained below its cap while another exhausted its cap unstable. Only that identified process was interrupted; its outputs and original source remain preserved, with no published timing result. Two deterministic REDs justify the minimal loop-eligibility fix and retention of raw timing on interruption between blocks. The corrected search/timing checks pass30 tests; the final full suite passes708 tests,48 existing skips and28 existing warnings in270.79s. Corrected source is `e2d8a6efe205468a2c77c17baeaa4b197b86c01d3e328649e50c13b6296eb214`. Fresh measurements must use a new directory and fresh parents, without resuming those old checkpoints.

**Bounded evidence:** Fresh `ca7af11` parents and a genuine1/1 numerical CPU receipt completed seven small engineering searches and three timing modes. Ridge anchor cost0.867764917s generated exploratory doubled budgets1.735529834/3.471059668s; no full-quality budget was selected. All three timing modes selected tested batch128. Ridge host20 repetitions and causal host66 reached diagnostic precision; causal resident80 remained unresolved at its cap. All raw summaries rebuilt exactly,17 manifests are terminal, and all521 historical Task3 index entries still match. The [curated record](docs/research/2026-10-16-evidence7-search-timing.json),705-file index and [report](audit/implementation/evidence/2026-10-16/task-7-report.md) retain the failed original attempt and final evidence. This is ready for independent implementation review; final WP05 quality/default policies and the scientific objective remain pending.

**Task7 review correction (I1–I3):** Independent review identified three implementation gaps: method order was not counterbalanced across predictors, iid calls could falsely pass precision within one block, and resumed tuning lacked full cost/anchor ancestry. The fix adds a separate method/mode comparison coordinator with retained shared mirrored blocks; complete repeat pairs are the bootstrap unit and paired comparisons share resampling draws. Precision requires multiple complete blocks and a tail-derived block floor; duplicate slots add no information. Timing schema2 makes this meaning explicit while historical schema1 evidence remains source-scoped. Costed resume now requires immutable run/config/budget/identity/trial-ledger ancestry; standalone replay snapshots cannot enter a costed search. Results retain current and historical trials plus separate attempt/cumulative charges, with historical models ineligible to win. Terminal ledger/descriptor publication is explicitly outside the measured enclosure, like terminal manifest publication. Four targeted REDs reproduce the findings; the final covering command `python -m pytest tests/test_search.py tests/test_timing.py -q` passes35 tests in27.14s. No unchanged whole-suite rerun or full-quality/default calibration is required by these scoped corrections. Fresh bounded evidence follows the tested fix commit; original reports/index/receipts remain unchanged.

**Task7 fix1 bounded receipt:** Under committed75fde42/source1979ab38…, the official Ridge worker timed out and charged0.269365083s; a fresh resumed trial succeeded and charged0.902586084s. Verified cumulative cost is1.171951167s, retaining the failed anchor and rejecting an uncosted replay snapshot. One Mean/Ridge comparison retained alternating shared method blocks; four measured blocks and unstable capped warmups correctly remain precision-ineligible. Raw reconstruction and within-block duplication invariance pass. The [append-only fix evidence](docs/research/2026-10-16-evidence7-fix1.json) and [fix report](audit/implementation/evidence/2026-10-16/task-7-fix1-report.md) bind54 new indexed files; all original705 entries and original rollups remain unchanged. This addresses review I1–I3 for scoped re-review, without selecting final WP05 policies.

**Task7 implementation review accepted:** The [initial review](audit/implementation/evidence/2026-10-16/task-7-review.md) found I1–I3; the [scoped review](audit/implementation/evidence/2026-10-16/task-7-fix1-review.md) closes all three with no new breakage or out-of-scope finding. Root independently verified54 new indexed files, all705 unchanged historical entries and the actual35-pass covering log. Accept the implementation phase at `75fde42`/`e90b315`; retain Step4's real numeric WP05 evidence gate. Historical iid timing flags remain scoped to their original reader and cannot certify final block precision. Continue Task8's tested pilot/design code, then pre-freeze protection, confirmation/analysis and audit/report code from Tasks9/10/13. Only after those implementations are reviewed and committed may definitive pilot/design and a real freeze proceed. No new prediction result, protected test access or independent confirmation exists; the one-teacher findings in D008 remain unchanged.

## D014 — 16 October 2026: pilot/design implementation, scientific evidence pending

**Independent units and roles:** The pilot coordinator fixes one teacher's F/G/V/E data across inner optimizer/allocation repetitions, then averages the paired E row losses within teacher. Whole teachers are the bootstrap unit; reused E rows do not become independent observations across repetitions. Every repetition trains a fresh complete G-selected parent and reconstructs matched keep/top/random/warm-width/B6 forks through guarded checkpoints. Fresh B2 lower/upper brackets and B3 centered static-partial controls have the same two-stage training opportunity. The prescribed fallback is static-partial versus fresh upper-width ordinary. B6 retains keep and every eligible unit probe, including losers/failures, before a fresh joint continuation. No automatic full method inventory or V-selected per-teacher proposed recipe is introduced.

**Numeric decisions:** Whole-teacher SD precision (upper/lower ≤2), two-batch runtime stability (≤20%) and endpoint Monte Carlo precision govern calibration, never the favored effect sign. Favorable and unfavorable resolved effects both use half their absolute mean. The calculator archives the normal count and applies `K=max(K_calc, completed_pilot_teachers)` after the existing eight-teacher/two-repetition minimum. Conservative paired-row and between-teacher variance bounds supply the fixed test-size formula. Unresolved variance/effects, unsupported finite-sample inference, absent numeric policies/costs/controls and infeasible floored workloads produce explicit calibration/estimation/blocked branches. Conditional empirical resampling power uses Student-t rejection as a reported sensitivity diagnostic; it does not prove bootstrap coverage. Cost/precision curves and the within-/between-repetition cost recommendation are recorded without silently changing completed repetitions or reducing the floored cohort.

**Phase and provenance:** `configs/pilot.yaml` is explicitly a one-teacher/two-repetition tiny engineering invocation. Its values are not WP05 outputs. All attempts, nested cost boundaries, raw E differences/predictions, checkpoints and failures are retained; new invocations use fresh parents. Final raw ledgers are independently hash-sealed; regeneration rejects changed raw bytes or package source. The numerical producer, core/NODE locks and genuine Task7 CPU1/1 receipt are unchanged. The current schema rejects scientific execution and every decision remains freeze-ineligible. Tasks9/10/13 must first supply a reviewed readiness path binding implementation commits/current execution source, locks, actual receipt/runtime, complete measured pilot ledgers and numeric policies. Only then may the same statistical machinery be used for a genuine pilot and validated freeze; no protected observations or final numeric design have been created here.

**Implementation verification and bounded evidence:** Tested implementation commit `3819660` passes152 covering checks and the single final full suite:763 passed,48 existing skips,28 known warnings in294.35s. The subsequent source `72d46eac…` engineering invocation ran one tiny teacher/two repetitions in4.764845708s under the unchanged actual CPU1/1 receipt. Both paired audits and all required minimal controls completed;32 manifests are terminal. All four derived pilot outputs rebuild byte-for-byte from sealed raw measurements. The [400-file index](audit/implementation/evidence/2026-10-16/task-8-artifact-index.json), [curated evidence](docs/research/2026-10-16-evidence8-pilot-design.json) and [report](audit/implementation/evidence/2026-10-16/task-8-report.md) preserve provenance. `pilot/design_decision.md` is explicitly blocked/freeze-ineligible, with no invented final costs/policies/counts. Independent implementation review is next; genuine WP05 evidence remains pending. PRE-pilot readiness must check code/runtime/config/seed plan and the pilot's engineering resource forecast; POST-pilot freeze eligibility separately checks completed measured pilot ledgers, final policies and actual floored-cohort feasibility.


**Task8 independent implementation acceptance:** The [initial review](audit/implementation/evidence/2026-10-16/task-8-review.md) identified two Important gaps. Fix1 `a84ec74` now subtracts a conservative crossed optimizer/shared-E nuisance allowance when bounding intrinsic between-teacher variance, and enforces every validated declared control against the full observed intersection. Conditional iid/crossed-effects assumptions and bootstrap coverage limitations remain explicit; unresolved intrinsic variation selects finite-row estimation. The [scoped review](audit/implementation/evidence/2026-10-16/task-8-fix1-review.md) closes both findings with no new breakage. Covering tests pass70 in10.14s; the unchanged763-test suite and engineering CLI were not repeated. The [rollup](audit/implementation/evidence/2026-10-16/task-8-index.json) binds reports, test logs and original400-file index. Original runtime artifacts remain unchanged/source-scoped. Accept implementation only; final numeric WP05 design remains blocked. Minor M1 (evaluation cost absent from the repetition recommendation) stays visible for final triage. Next: Task9 protocol/protected-access and staged readiness implementation, followed by Tasks10/13 implementation before any genuine pilot/freeze. No new scientific outcome or novelty claim.

## D015 — 16 October 2026: protocol protection implementation; scientific freeze pending

**Pre-freeze status:** Task9 implements scientific identity and protected access with simulated temporary repositories, sealed toy ledgers and trap label readers. Implementation acceptance remains pending independent review. No root `protocol/frozen.yaml`, genuine pilot, protected scientific observations, confirmation cohort or WP05 completion is produced. The historical engineering Markdown/report/index and their source identities remain unchanged.

**Separate readiness stages:** Scientific-purpose configuration now names a required readiness artifact. Before the pilot starts, that artifact must bind committed accepted Task9/10/13 reviews and reviewed files, the current clean execution/config/lock/vendor tree, the actual numerical runtime receipt, a common recipe and independent exploratory seed plan. Pilot time/storage/memory forecasts derive from linked measured resource samples and respect declared caps, reserved memory and retained-failure policy. This stage does not require future pilot records or final K. After execution, the freeze gate validates the complete scientific raw/child artifact seal, completed batches and controls, reconstructs paired E losses from saved predictions and F variance, recomputes the accepted numeric design, and checks the entire floored confirmation commitment. Existing statistical formulas and limitations are unchanged.

**Identity and access:** Canonical JSON hashes record the pinned implementation commit and actual checkout separately. Dirty or untracked executable/scientific inputs reject; documentation and generated protocol commits may differ when scientific digests match. A strict adjacent canonical companion binds exact Markdown bytes to raw seals and verified design inputs; prose is never parsed for numeric authority. Freezes preserve immutable history and reject reused confirmation identities. Scientific amendments preserve the old hash, relabel affected old results exploratory and require fresh confirmation. Resume permits only a logged identical existing run link and narrow verbosity settings.

**Protected lifecycle:** The complete eight-role causal/control set is fixed before access; top versus random remains sole confirmatory H1-A and other outputs remain descriptive. Task10 must register the actual selected model artifacts, teacher/development/T commitments and observed F-target variance. The protected evaluator validates these identities, gives predictors label-free inputs, records access before private T generation, and seals predictions/targets for later authorized analysis. Completed repeats return the same sealed result; interrupted label access cannot automatically reread T. `load_evaluation` validates previously authorized saved rows without prediction or regeneration. These are procedural guards for cooperating code, not encryption against the file owner. Transfer remains locked pending its separate design.

**Next integration:** Root review and actual committed acceptance records are still required. Tasks10/13 must supply confirmation orchestration, analysis/audit consumers and their reviewed evidence before definitive pilot execution. The new guard APIs do not authorize passing T labels into the existing generic pilot driver. Final numeric readiness and the real freeze remain pending.

**Verification:** The stable full suite at source `f051bd96e395e79626ab7d9683dbbc3d299581a669f8c24b05e57de0f1adc56e` passes833 tests with48 existing skips and28 known warnings in372.67s. A subsequent narrow readiness correction binds measured resource samples to pilot repetitions, E rows, probe updates and bootstrap iterations as well as model/training/data/backend settings. Four direct REDs reproduce workload reuse; the final protocol/protected/config/calibration coverage passes109 tests in104.16s with no warnings. Final package source is `3a56d0f139a28cfd8ca54eca73ba13538ed8da8a575c7fb35e9d1b9be3b7e0fb`. Root authorized scoped coverage for this identity-only follow-up, so no unchanged full suite was repeated. The original absolute source root and inherited absolute child locators remain an explicit Task13 portability prerequisite before genuine freeze; scientific hashes must remain binding.


**17 October 2026 — Task9 implementation accepted:** The [scoped review](audit/implementation/evidence/2026-10-17/task-9-fix1-review.md) closes all three Important findings: index/worktree agreement, actual pilot control/checkpoint/B6-loser lineage, and declared teacher/development recipe consistency. No new breakage or open Critical/Important findings remain. Fix3df4e078 passes92 covering tests in431.87s, without rerunning the preceding833-test full suite; source stages remain explicit in the [evidence index](audit/implementation/evidence/2026-10-17/task-9-index.json). The committed acceptance record covers actual reviewed files and reports, not scientific readiness. A root snapshot correctly rejected five redundant generated editable-install metadata files in src; they were preserved intact under environment/install-metadata, existing installed package metadata/version remained available, and the clean execution snapshot passed without weakening the guard.

Continue pre-freeze Task10 confirmation preparation/orchestration/analysis, then Task13 audit/report and portable archival integration. Only after those implementations and integration reviews are committed may genuine measured pilot/design, a real freeze and the entire independent cohort proceed. The historical engineering report stays blocked. No new prediction result or test access occurred. The first scored-growth loss to random placement remains visible in the [plain-language exploratory interpretation](docs/research/2026-10-17-exploratory-interpretation.md); its contrast-similarity diagnostic is post-hoc, F-only and descriptive.

## D016 — 17 October 2026: pre-freeze confirmation implementation

**Status:** Task 10 code and bounded verification only; scientific WP06 remains pending independent review, Task 13 implementation/review, committed final source, a genuine measured pilot/design, and the actual freeze. No root frozen artifact or genuine future cohort was generated.

**Decision:** Prepare fresh teacher/F/G/V artifacts and committed T row IDs after numeric design but before freeze. Preserve strict source/runtime/receipt identities and the original teacher/optimizer commitments. Map the nominal optimizer seed once to the existing 32-bit NumPy boundary and bind distinct versioned allocation/contrast streams from its full value. Fit the existing eight-role graph with F/G only; register its actual G-selected saved models before protected X-only prediction and private T scoring. Finish every declared teacher/repetition irrespective of signs, retain zero-growth and failed units, and permit only finite frozen same-unit retries. A pre-label inference retry reuses registered models; incomplete label access cannot be repaired by a new seed, model or replacement teacher.

**Analysis:** Reconstruct original-unit and F-variance-normalized risk from sealed rows. Average repetitions within teacher/data units before whole-teacher bootstrap and Student-t sensitivity. Missing/invalid repetitions are never zero-filled; incomplete, singleton, zero-variance, unsupported or unresolved Monte Carlo precision cannot establish completed-cohort superiority. H1 remains random minus scored placement, with a placement-only interpretation; other fixed controls remain descriptive unless explicitly covered by a frozen optional family. Preparation, execution and analysis work have separate measured boundaries, with disjoint totals only where measurements exist.

**Pre-test clarification:** Root Ruling 11 in [the accumulated rulings](audit/implementation/evidence/2026-10-17/rulings-for-final.md) specifies eight descriptive slices: below/above probability .5 for each of the four teacher gates, using the actual hard/smooth teacher semantics. Diagnostic masks are generated inside protected evaluation and never offered to fitting/prediction; empty slices are unavailable. Default Holm families are empty; a planning family count cannot invent new claims. This resolves a previously unspecified schema before any real test data, without adding a new hypothesis or population. If wrong, revised descriptive policy requires a versioned schema/analysis; changed confirmatory claims after T require fresh confirmation and time/storage.

**Next:** Root reviews the Task 10 diff and renews affected Task 9/8 acceptance bindings. Task 13 must then supply portable locators and explicit archived reanalysis while preserving strict source guards. The detailed task report records test boundaries, exact validation output, remaining limitations and handoff contracts.


**Task10 implementation accepted — 17 October 2026:** The [independent review](audit/implementation/evidence/2026-10-17/task-10/task-10-review.md) approves spec compliance and quality with no new Critical, Important or Minor findings; inherited warning triage remains visible. Commit `b40cee4` passes 890 tests with 48 existing skips and 28 known warnings in 900.20s, exit 0, on unchanged source `18cb7f06…`; root verified all 19 retained log hashes, the report and 302-file execution snapshot. The [rollup](audit/implementation/evidence/2026-10-17/task-10-index.json) preserves the original worker bundle and actual review. Root renewed Task9 acceptance against its genuinely reviewed integration changes and recorded Task10 acceptance; neither record grants scientific readiness.

The three CannotVerify items are resolved as phase obligations: genuine study execution stays at WP05/WP06, portable live/archival lookup stays in the next Task13 code gate, and the controller-owned acceptance records now bind actual review artifacts/current reviewed files. Ruling11 fixes descriptive gate-half slices before any real test access; [its exact line and cost](audit/implementation/evidence/2026-10-17/rulings-for-final.md) extend the ten original rulings. The bounded real fixture fits one teacher/two repetitions/all eight roles and retains other missing units, so its analysis is correctly incomplete. No predictive benchmark result, genuine pilot, frozen scientific protocol or independent confirmation was added.

Next: implement/review Task13 audit/report and portable archive/reproduction paths, then complete the broad H1 implementation review and known-Minor triage before definitive calibration/source freeze. Run the entire fixed cohort and actual scientific audit afterward. The first exploratory scored-growth loss to random placement remains unchanged.

## D017 — 17 October 2026: pre-freeze audit and portable evidence implementation

**Status:** Task 13 implementation, bounded fixtures and review preparation. WP09
scientific execution is pending the definitive pilot/design, source freeze, full
fixed cohort and actual clean reproduction. No completed scientific result or
external publication is authorized by this entry.

**Decision:** Audit the frozen inventory and all attempts before interpreting the
sign of an effect. Reconstruct F population variance and original-unit regression
metrics from authorized saved rows; retain unfavorable, failed and zero-growth
units. Check full selected/last/probe checkpoint evidence, parameter terms,
original access/resume lifecycle, cost boundaries and exact declared analyses.
A complete negative result can be valid; incomplete or corrupt evidence cannot
support completed-cohort superiority. Deterministic report content is separate
from per-invocation work metadata. H1 remains a placement component claim.

**Portability:** Translate historical absolute and producer-relative locators to
byte-identical restored paths while preserving frozen config and receipt hashes.
Record original and current checkout identities separately. Execution, lock,
source, checkpoint and protected-access guards remain binding. No ignore-source
flag or archived training capability is added.

**Archive boundary:** [Root Ruling 12](audit/implementation/evidence/2026-10-17/ruling-12-archive-sealing.md)
approves prospective sealing while live evidence is valid, before amendment or
source drift. The immutable copy preserves the original access-ledger prefix;
later evidence-read appends do not invalidate it. Reanalysis verifies every copied
byte and reconstructs only sealed original rows under a separately recorded clean
analysis source. Every affected claim is exploratory; new T/fit/predict/resume is
unavailable. There are no genuine legacy protected studies requiring retroactive
sealing. A missing pre-amendment seal cannot be repaired by silently restoring live
eligibility. If this boundary proves insufficient, recovery needs the retained
original evidence plus a reviewed migration, or a fresh independent confirmation,
with corresponding engineering, time and storage cost.

**Next:** Root-owned scoped spec/quality review and renewed shared-file acceptance,
then the final broad H1 implementation review/known-Minor triage before definitive
calibration. Actual scientific reproduction/audit/report and closest-prior-art
revisit follow the entire fixed cohort. The R12 access obstacle is resolved;
bounded inspection still does not prove novelty. No optional practical or new
claim family is implied by the H1 tooling.

**Pre-freeze numerical check:** The unchanged seed42 toy recipe exposed B6's
near-zero gate-gradient sensitivity in the independent float32 leaf reference.
The exact failure was reconstructed and preserved; float64 agreed to 4.1e-15.
The probe now checks each first AdamW update against its own gradient formula and
a derived componentwise conditioning/rounding allowance. Forward/gradient
criteria, measured discrepancy, receipt ceiling and exact replay stay binding.
Actual training, seeds and model equations are unchanged. The preserved toy
regression and derivation are in `tests/fixtures/causal_seed42_probe.md`.

**17 October 2026 — Task13 implementation accepted:** The independent review and
same-reviewer scoped checks close the four original Important findings and the
follow-up final work-total overflow. The [public evidence index](audit/implementation/evidence/2026-10-17/task-13-index.json)
preserves the original 908-pass full suite, the fix1 affected 15-pass result and
the fix2 affected 17-pass result under their distinct source identities. No
unchanged full suite was repeated. Final implementation is `d5c8828`; reviewed
shared-file acceptance records for Tasks9/10 are renewed and Task13 acceptance is
published. Failed partial graphs, original-root receipts, actual executing analysis
bytes and completion-time archive eligibility are now checked. Root Rulings12/13
are committed separately from the historical Ruling11 record.

This accepts code and bounded fixtures, not a scientific result. Next is the one
broad H1 implementation review, including the three carried Minor items and
executable calibration coverage, followed by genuine measured calibration,
numeric design, freeze, the entire independent cohort and scientific reproduction
and audit. The initial score-versus-random loss remains the only primary
exploratory predictive comparison; no new accuracy claim is made.

## D018 — 17 October 2026: final H1 implementation accepted; scientific execution next

The single broad review found two Important integration gaps and retained three
Minor observations. The one combined fix at `dfed0d2` corrects actual pilot
pair/attempt publication, admits the required finite CPU density/logical-batch
range through measured source-bound evidence, and includes evaluation cost in
the advisory repetition calculation. The original reviewer closes I1/I2/M3 with
zero new Critical, Important or Minor findings. [The complete review/evidence
index](audit/implementation/evidence/2026-10-17/h1-final-index.json) renews
Task9-v4, Task10-v3 and Task13-v2 bindings without rewriting historical records.

The full-suite stage produced 958 passes, 48 skips, 2 failures, 9 setup errors
and 28 known warnings. The failures/errors exposed a historical receipt copied
by relocation tests and a duplicate Torch inter-op setter. Those exact paths
were corrected; final affected coverage passed 190 tests with 47 optional skips.
The final package is `5c66c556…`, execution map `ccb5447a…` (314 files). Fresh
native evidence covers 40 numerical cells, 162 timing cells, 24 scoring-parity
checks and 8 high-density parent/prediction reloads. It is source-specific
verification, not a clean final full-suite claim or a predictive result.

The finite CPU range is F577/2306/9224, primary d16/h64/m16 with derived
controls, and logical=physical batches128 through16384 on the measured doubling
grid. Diagnostic32768 adds no actual rows beyond9224. Final timing variation is
retained; there is no universal optimum or stable less-than5% plateau claim.
A diagnosed near-zero ReLU branch discrepancy led to a reference-only,
rounding-certified derivative check with unchanged training, seeds and gradient
tolerances. Unsupported nonlinear ambiguities remain failed attempts; no seed
substitution or hidden tolerance change is authorized.

**Ruling14:** Defer NODE's per-visit trace and scoped warning cleanup to the
conditional long-run NODE/practical phase; H1's fitted cohort does not use NODE.
If wrong, adapter instrumentation/warning assertions and measurements must be
reworked before that comparison can be available. The [exact ruling and
cost](audit/implementation/evidence/2026-10-17/ruling-14-conditional-node-instrumentation.md)
preserve the short anchor's quality-ineligible status.

**Ruling15:** Keep actual H1 resident timing within explicitly asserted and
recorded supported chunks no larger than min(128,F). The direct resident view
does not gain the extended host predictor's enforced bound. If wrong, add a
reviewed capability/guard and redo affected timing/source-bound evidence before
a broader claim; source changes after T require amendment/fresh confirmation.
The [exact scope](audit/implementation/evidence/2026-10-17/ruling-15-resident-timing-scope.md)
leaves the model and protected host predictions unchanged.

**Next:** Measure common batching/stopping/sensitivity/timing policies and a
complete current-recipe resource teacher, validate readiness, run genuine
independent-teacher variance calibration, derive/freeze the numeric design,
execute the entire fixed cohort regardless of sign, and complete actual locked
reproduction, interrupted growth/resume, saved-row audit, repeated report rebuild
and closest-prior-art claim review. The old single-teacher result remains
keep0.16066/top0.14753/random0.14384; no independent accuracy evidence exists yet.


## D019 — 17 October 2026: measured calibration exposes duplicated checkpoint histories

Six parent-only development fits compared batches 128/256/512 with learning
rates 0.003/0.005 on the same exploratory problem. Their normalized G errors
range from 0.07099 to 0.09644. Several curves remain truncated; these measurements
do not select a final recipe or provide another selective-versus-random result.
The [calibration archive](audit/implementation/evidence/2026-10-17/calibration-phase-a/index.json)
preserves all six measurements/configs, failures, timing boundaries and an exact
prechange numerical reference. Root independently verified the complete ignored
inventory: 289 files, 1,718,434,895bytes.

The current checkpoint history repeats the complete ordered minibatch row IDs
already preserved in each stage's raw event log. Copying that prefix into every
one of 65 validation probes makes even the smaller-batch recipes project more
than 85 GB of duplicate metadata over 16 pilot and 16 confirmation teachers, before
other outputs. This is a measured-prefix projection, not a completed study cost.
The original full-batch parent alone retained 551.8 MB, mostly checkpoint JSON.

[Ruling16](audit/implementation/evidence/2026-10-17/ruling-16-compact-checkpoint-row-history.md)
authorizes replacing only those redundant historical ID copies with explicit
versioned row counts and the existing ordered-int64 SHA256 commitment. Every full
raw stage log, ordered ID, failure and sampler/RNG state remains retained. The
scoped correction at `388c82c` passed independent spec/quality review. Its actual
4,000-update replay matches all 21 selected tensors, complete numerical metadata,
F/G/E predictions and all 4,000 ordered row/loss/LR commitments exactly. The
retained metadata sum fell from 20,613,624 to 2,567,456 bytes; the same fit boundary
fell from 47.640 to 14.144 seconds in one observation. New-source recovery, raw-event
audits and applicable CPU kernel/envelope receipts pass. The [review and verified
evidence index](audit/implementation/evidence/2026-10-17/h1-history-storage-index.json)
renews Task 9-v5/10-v4/13-v3 bindings; prior records remain unchanged.
Checkpoint-only full historical inspection will require the retained stage logs;
any numerical/audit discrepancy blocks acceptance and further scientific use.

No complete resource teacher, genuine variance pilot, protocol freeze or protected
test result exists. The throughput-only batch choice still needs reconciliation
with complete-study feasibility; neither a batch-rule waiver nor a new scientific
claim is implied. After this accepted storage correction, resume measured recipe/resource
calibration and the independent study. All prior artifacts stay unchanged.


## D020 — 17 October 2026: pre-freeze batch selection accounts for the whole study

The [fresh 4096/0.005 parent](audit/implementation/evidence/2026-10-17/calibration-phase-b-parent/index.json)
exhausted patience at 3,375 updates and selected 2,875, with normalized G MSE
0.0730621. Its 14.885-second fit retained 155.2 MB of raw events. The matched
128/0.005 development parent gave descriptive G MSE 0.0709911, a 14.144-second
fit and 10.3 MB of raw events. These single-problem observations do not establish
statistical equivalence or a population quality ranking. Root independently
rehashed all 30 source-bound files / 159,084,344 bytes, with no mismatch.

[Ruling 17](audit/implementation/evidence/2026-10-17/ruling-17-complete-study-batch-policy.md)
and the [pre-freeze amendment](protocol/amendments/2026-10-17-development-batch-policy.md)
replace the initial throughput-only 95% batch heuristic with complete-pipeline
learning, time and retention. Use common batch 128 for the next finite
calibration, while preserving the measured 4096 throughput result. All causal
forks and fresh controls receive the same batch and allowance. No raw history,
failure or losing probe is removed. A conditional ceiling exceeding disk is
not used as proof that the realized graph is impossible.

Next test symmetric 8,000 parent/continuation ceilings, unchanged learning rate
0.005, G/checkpoint interval 25 and patience 20, on the predeclared development
teachers and representative controls. Ceilings remain candidates until learning
curves are checked. No choice may use top-minus-random sign. Complete matching
resource evidence and actual pilot-derived sample sizes remain required before
protected access. If the batch decision is wrong, claims remain limited to the
chosen recipe; another batch needs separately budgeted calibration and fresh
independent confirmation, with more time and storage. No batch-independent
allocation advantage is claimed.


## D021 — 17 October 2026: common-recipe development checks remain mixed

The two predeclared primary problems completed under the shared 128/0.005,
8,000+8,000 candidate recipe. All 18 stages exhausted patience 20 before their
ceilings; insertion and unchanged-model shams preserve predictions exactly,
and both sample-stream pairing audits pass. Root independently rehashed all
282 retained files / 157,930,582 bytes, with no mismatch.

On A, exploratory normalized E risk is 0.0755812 for selected growth,
0.0757393 for random growth and 0.0762721 for keeping the parent. On B, all
three are 0.0425751; the upper ordinary network gives 0.0336228. The
[complete table and limitations](docs/research/2026-10-17-common-recipe-exploratory.md)
and [source-specific archive](audit/implementation/evidence/2026-10-17/calibration-common-recipe/index.json)
retain every control, including unfavorable static and ordinary results.
These are two exploratory teachers with one repetition each, not a genuine
variance pilot or independent confirmation. Do not pool different recipes or
replace the registered metric with whichever result is favorable.

An auxiliary wrapper's continuation-example count included an inherited history
tail. Original records remain retained, with a direct raw-stage summary
superseding that diagnostic count. Model updates, predictions, selected
checkpoints and pairing were unaffected. No production/source change was needed.

Continue finite one-factor sensitivities, negative controls and measured
precision/probe calibration. The primary recipe stays fixed during those checks;
no growth-sign-driven tuning is allowed. Complete matching all-eight-role,
R>=2 resource evidence and strict readiness still precede the genuine pilot,
numeric design, full independent cohort and actual reproduction/audit.


## D022 — 17 October 2026: finite controls complete; no placement advantage established

Completed the predeclared 13 finite development coordinates, 15 restricted
probe trials, prediction timing and conditional numerical precision checks.
Root read the full report and rehashed all 1,169 retained files / 663,296,577
bytes with no mismatch. The [plain results](docs/research/2026-10-17-finite-development-controls.md)
and [original evidence](audit/implementation/evidence/2026-10-17/calibration-finite-controls/index.json)
retain favorable and unfavorable outcomes. All 53 fitting stages stopped by
patience before their ceilings. These reused/single-repetition development
coordinates are not independent confirmatory units.

Keep the shared primary recipe unchanged: outcomes are mixed, with random
placement ahead on several controls. Use the measured 320-update finite B6
probe opportunity, charging every candidate and loser; convergence and a stable
all-unit ranking are unestablished. Use E=262,144 only as the bounded pilot
workload: it fails the two-teacher engineering H1 standard-error yardstick.
Independent-teacher variance must determine final nT or an honest limited
estimation/resource branch. Conditional block bootstrap and artificial-shift
power calibrate numerical work, not population inference. No precision waiver
or new hypothesis is adopted.

Run exactly one complete engineering teacher with two repetitions, all eight
comparison methods and every greedy probe under recipe hash
4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a.
Measure actual complete costs before choosing resource constraints and strict
scientific readiness. Genuine independent-teacher calibration, a numeric frozen
design, the complete independent cohort and reproduction/audit remain pending.


## D023 — 17 October 2026: complete resource sample accepted; strict pilot readiness passes

The measured full engineering teacher completed R2/all eight roles/all 130
probes without failure: 485.014 seconds, 907,887,243 producer bytes and peak RSS
1,042,415,616 bytes. Root rehashed all 1,966 retained files / 916,297,448 bytes,
read the complete report and accepted the actual sample. The existing strict
readiness validator passed at the unchanged 315-file execution map, runtime
receipt and committed Task9v5/10v4/13v3 reviews. This is permission to run the
genuine exploratory variance pilot, not a scientific result or freeze.

[Numeric policies, uncertainty and original evidence](docs/research/2026-10-17-scientific-pilot-readiness.md)
record the fresh independent seed namespace, fixed recipe, 8-to-16 teachers,
two repetitions, complete greedy costs and four-hour pilot allowance. Root
reserves 10 GiB disk plus the forecast pilot footprint before setting the
shared storage cap, uses observed free-page headroom with an additional 2 GiB
reserve, and retains failures with zero automatic retries. Nonlinear cumulative
ledger costs and one-teacher resource uncertainty remain explicit. Population
inference support is not inferred from conditional numerical calibration.
The bounded E workload has unresolved precision; genuine variance governs final
nT or the honest estimation/resource branch. Do not call an unresolved capped
pilot accepted confirmation or evidence of no effect.

One pre-fit validation call failed because the YAML reader resolved JSON's
exponent as a string. Retained failure evidence and an actual YAML rendering
show exactly equal resolved config values; readiness then passed without source
or policy changes. Launch the scientific pilot only after committing this
record, hold execution source fixed, and continue through the resulting design,
full eligible independent cohort and reproduction/audit.


## D024 — 17 October 2026: closest-paper revisit narrows the contribution

The independent source comparison and root inspection of the consequential R07
Appendix B.2 passage confirm that random neuron selection with informed per-unit
directions is already an allocation control. Related spectral splitting,
function-preserving duplication and gated limits are established. The acquired
R12 full architecture/training/parsing text resolves the previous access blocker.

The [bounded comparison](docs/research/2026-10-17-prior-art-revisit.md) did not
locate the exact combined centered insertion, residual-moment direction, RMS
scaling and fresh-Adam ranking in R07/R08/R12. This is no priority proof or code
survey. Frame any result as empirical characterization of a specified procedure
in established model families. Root verified all six indexed source files;
no code, weights or agreements were acquired through this comparison. The
revisit used no pilot outcomes and changed no experiment or scientific gate.


## D025 — 17 October 2026: preserve the pause timeout; complete the exact missing cohort members

The original genuine pilot stopped at its unchanged elapsed-time guard after
the user explicitly paused and resumed the live process. Root read the complete
terminal report, independently verified all 29,003 files / 12,219,004,537 bytes
and compared every completed pair against the sealed raw attempt ledger:
30 completed pairs, 15 R2 teachers, zero model-pair failures, exact missing set
index15/R0 and R1. Only the first eight-teacher batch reached canonical summary;
no terminal design or protected test exists.

The [retained evidence and descriptive outcomes](docs/research/2026-10-17-genuine-pilot-interruption.md)
include unfavorable guided-versus-random and static-versus-ordinary averages.
The interruption occurred during a between-pair publication. Original enclosing
clocks include the documented suspension; the next pair's clock excludes it.
No original clock, config, source, budget, raw ledger or seal was changed.

Ruling: Complete only the two never-started members of the already prescribed
16-teacher/R2 cohort in a separately identified continuation, with identical
inputs/functions/recipe and every control/probe, retaining the failed original
and withholding canonical freeze eligibility — explicit user continuation and
verified unchanged scientific work make a full restart unnecessary — what it
costs if wrong: the joined bundle needs reviewed provenance/resumption support
or a fresh valid cohort before protected confirmation, with added time/storage.
This is chronological [Ruling18](audit/implementation/evidence/2026-10-17/ruling-18-user-pause-continuation.md).

The [prospective amendment](protocol/amendments/2026-10-17-user-pause-continuation.md)
charges new enclosing work against the original four-hour allowance outside
the documented suspension, checks actual resources and grants no additional
teachers, retuning, MC change, freeze or T access. Complete the linked exploratory
record and then actual WP09. Preserve all eighteen rulings in the final handoff.


## D026 — 17 October 2026: complete the linked pilot; retain an unresolved allocation result

All 16 prescribed independent teachers now have two complete repetitions, eight
roles and all 2,080 finite greedy probes, with zero scientific-pair failures.
Root independently rehashed all 1,995 continuation files / 741,223,772 bytes,
all 32 member bindings and saved-row means; original raw, seal and inventory
remain unchanged. Two pre-fit/saved-wrapper setup mistakes are retained and
charged separately. No original completed pair was retried.

The [complete exploratory results](docs/research/2026-10-17-complete-exploratory-pilot.md)
give random-minus-guided normalized E MSE −0.000536435. The original 8k bootstrap
spans zero. The prospectively declared fixed 512k saved-scalar calculation first
reproduced 8k to 2.17e−19, then gave [−0.001205030, +0.000156050]. Root read
the full script/report and verified all 20 files / 75,962 bytes and original
input bindings. Its computational precision passes, while SD ratio 2.367 and
intrinsic variance remain unresolved. More bootstrap draws supply no independent
teachers or coverage guarantee. Original `calibrate`, failed producer and missing
canonical runtime batch remain binding; no freeze/T is authorized.

Narrow the claim to the completed diagnostic rather than infer beneficial
placement, equivalence, architecture novelty or practical competitiveness.
The finite pilot cap is a chosen workload, not proof of physical impossibility.
Any future expansion needs a prospective design; no deeper growth or transfer
study is triggered by these outcomes. Execute actual WP09 now: independent raw
prediction/F-variance recomputation, fresh frozen-lock full representative replay,
durable interrupted-growth recovery and deterministic report regeneration. This
is evidence validation, not another efficacy teacher or a new source review.


## D027 — 18 October 2026 local / 17 October UTC: diagnostic reproduction accepted; narrow the research claim

Actual WP09 is complete for the exploratory diagnostic. A fresh separate
offline frozen-lock environment reproduced both reference scripts and the
complete eight-role/65-probe engineering pipeline. Model topology/tensors,
every Adam moment/age/group, scheduler, complete RNG/sampler/cursor, next rows
and F/G/E predictions match exactly. Parent/probe states and all 11 stage row
streams/losses/evaluations also match. A real durable interruption 25 updates
after growth, followed by a linked 525-update retry, matches the uninterrupted
550-update continuation exactly without additional allowance.

An independent calculation from actual saved predictions and original F/E
labels/row identities verified all 32 pairs, 256 risks and hierarchical
variances. Maximum risk rounding discrepancy was 1.39e−17; paired rows and
teacher effects agree exactly. Root read the calculation/replay scripts and
full report, rehashed all 1,030 new files / 481,577,468 bytes, and verified the
unchanged 315-file execution map and package identity. No scientific or numerical
WP09 failure occurred. The intentional interruption and a corrected pre-render
quote error are retained separately from scientific pairs.

The [report](reports/final.md) is generated from hash-bound compact records,
with every teacher/method, inference limit, failure, cost and finite-probe
association retained. A readability-only revision preserves the full technical
report and replaces inline JSON dumps with tables. Two independent renders
match byte-for-byte. Root additionally verified the published standard-library
command from the curated location, yielding the identical report SHA
4b26c13d7d171f466d2305a471ac7c8781e2dd7eadcdad918fb72473cbd11d38.
[Evidence and reproduction inputs](audit/implementation/evidence/2026-10-18/wp09/index.json)
retain exact commands, original inventories and independent root checks.

Decision: **narrow and revise**. The specified score has no clear placement
advantage at the achieved exploratory resolution; ordinary capacity remains
competitive under the shared recipe. This is a reproducible inconclusive
diagnostic, not equivalence, a powered rejection, practical superiority or
model-family novelty. SD/intrinsic-variance calibration and canonical runtime
evidence remain unresolved, so no accepted final n_T/K, freeze or independent
confirmation is manufactured. The finite pilot cap is not a physical impossibility
claim. Any future expansion needs a new prospective design; deeper growth,
transfer and hard execution are not triggered by this result.

The [current design decision](pilot/design_decision.md) and
[cost record](pilot/resource_forecast.json) replace stale engineering placeholders
while preserving all original bulk records. Work remains on the requested main
branch. No further numerical work is running. All eighteen chronological
rulings and their costs remain part of the final human handoff.


## D028 — 18 October 2026: execute a separate longer-probe allocation diagnostic

The user approved the next focused experiment: test whether longer validation
trials reveal placement gains missed by the cheap score and the short reference.
The [design](docs/superpowers/specs/2026-10-18-long-probe-allocation-diagnostic-design.md)
and [plan](docs/superpowers/plans/2026-10-18-long-probe-allocation-diagnostic.md)
specify one measured cost comparison, then eight fresh problems/R2 with both
short and long exhaustive searches, matched final capacity/continuation and all
nine final roles. Long trials use normal8000 ceiling/patience20; short320 retains
its fixed default. Source changes require fresh parents and a narrow current
CPU128 receipt. A fresh independent builder check found no unresolved question.

The [planning evidence](audit/diagnostics/2026-10-18-long-probes/planning/index.json)
records the smallest backward-compatible API change and resource boundary.
Cost and cohort use separate fresh seeds; full search cost, failures, source,
raw predictions and all unfavorable/zero outcomes remain retained. Counts and
policies do not depend on E outcomes. This is an exploratory diagnostic with
descriptive paired uncertainty, not the original missing frozen confirmation
or proof of practical superiority. Old results and eighteen rulings stay intact.
Implementation/review, one full cost measurement and the fixed cohort/audit
will now execute on current main without another permission checkpoint.

## D029 — 18 October 2026: accept the reviewed diagnostic implementation

The opt-in longer-probe path is implemented at `b2a9937`. The existing eight-role
and fixed-320 paths remain defaults; the new nine-role path uses separate
same-parent short/long searches and checks original weights, Adam state,
contrasts and minibatch pairing. The affected three test files pass all 55
checks, including a bounded real measure/run/analysis/report pipeline. The old
full suite was not repeated. An independent whole-change review approved the
source with no Critical or Important findings. [Implementation evidence](audit/diagnostics/2026-10-18-long-probes/implementation/index.json)
retains the diff, review, exact red/green logs and disclosed setup corrections.

One small accounting distinction remains explicit: the internal clock excludes
final summary/seal publication. The next measurement therefore also records the
complete outer command and post-publication bytes; its broader forecast will
inform the prospective resource decision. No predictive conclusion, completed
production receipt, cost comparison or cohort is claimed by these tests.
Source is now held fixed for the fresh CPU receipt and engineering cost run.

## D030 — 18 October 2026: measured longer-probe cost; lock the fixed exploratory cohort

The first complete cost comparison passed both65-trial searches and all nine
final roles, with no model failure/retry. It took504.277314250 seconds through
process exit, retained744,987,498 bytes and peaked at733,855,744 bytes RSS.
Root independently verified1,791 new files/745,280,545 bytes, all seals and
unchanged execution bytes. [Cost evidence](audit/diagnostics/2026-10-18-long-probes/cost/index.json)
includes exact raw receipts, commands, stopping counts, original records and
root checks. No cost-problem accuracy determined the design.

The [resource decision](pilot/long-probes-20261018/resource-decision.json) uses
16 comparisons ×1.5:12,102.655542 seconds, with one-sample uncertainty and no
automatic time kill. Logical storage projects17,879,699,952 bytes. A conservative
physical-disk budget uses the larger measured positive free-space delta,
projecting21,364,703,232 bytes plus10GiB reserve. Actual free space38,716,370,944
bytes leaves6,614,249,472 bytes beyond that budget and reserve. Native memory
availability stayed64%, swapouts did not increase, and RSS stayed below6GiB.
The [hashed diagnostic declaration](protocol/diagnostics/2026-10-18-long-probes-lock.json)
fixes eight fresh teachers/R2, seeds, all nine roles, analysis, failure policy and
execution identities before cohort fitting. It is not a FrozenProtocol or
confirmation. The original unresolved pilot/strong-baseline gates stay intact.

Operational ruling: retain the successfully validated CPU receipt despite the
OS timing wrapper's later exit1, then measure the first cost comparison with
Python elapsed/RSS counters. The sandbox denied `sysctl kern.clockrate` inside
`/usr/bin/time -l`; no model fit had started. Root separately validated all15
CPU operations. Supplementary OS timing counters are unavailable; full Python
outer elapsed, child RSS, native pressure/swap observations and post-publication
bytes are retained. The earlier failure and read-only checker corrections stay
visible; no numerical retry or old-artifact rewrite occurred.

Proceed serially with the complete fixed cohort under the committed source,
then independently audit raw predictions and replay/restoration. Do not choose
continuation, counts or exclusions from intermediate results.
## D031 — 18 October 2026: audited longer-search diagnostic; narrow allocation claim

The fixed fresh cohort completed all eight independent teachers/R2, all 144
final role instances and both exhaustive searches per pair, with zero numerical
failures or retries. All 1,040 short trials reached320; all 1,040 long trials
ended by normal patience with no8000 ceiling hits. The complete outer workload
took7804.403474250s, retained11496978848 logical bytes and peaked at947650560B
RSS. Nested short/extra-long search clocks are2140.461725334s/4169.761685458s;
they are not added to enclosing time. The prospectively committed cost/resource
gate was respected. [Cohort evidence](audit/diagnostics/2026-10-18-long-probes/cohort/index.json)
retains the entire file-hash inventory and original invocation/observations.

Independent prediction arithmetic agrees exactly with760 reviewed-analysis
values. Original F/G/E identities, F population variance and all144 model risks
were verified; repetitions are averaged within teacher. The prospectively fixed
first pair replays both rankings exactly:130 trials/53300 raw updates/2275 G
evaluations/133 complete states, including an actual long-last restore. All96
warm forks bind the original parent and raw shared minibatch prefix. The
[independent audit](audit/diagnostics/2026-10-18-long-probes/reproduction/index.json)
preserves two checker failures/corrections and their overhead; completed fits
were retained, with no numerical retry. Root independently rehashed the full
cost/cohort/audit inventories, held execution source and four historical files.

The [unchanged generated report](reports/long-probe-diagnostic.md) matches both
independent renders, SHA256
`9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.
Its internal retained-byte count precedes the final486-byte publication tail;
the external inventory/timer retains the complete boundary. No source/config
changes occurred during cost, cohort or audit execution.

Long selection averages normalized E MSE0.0633780485 versus random0.0638495407
and short selection0.0631248538. Random minus long is+0.0004714922, descriptive
whole-teacher bootstrap95% interval[−0.0010365806,+0.0022851931], with3 positive
and5 negative teacher means. Short minus long is−0.0002531947. Cheap-score
placement again has no clear advantage. Long selection compares favorably with
the fresh ordinary upper bracket, but warm ordinary widening remains close;
this does not establish general or practical architectural superiority.

Decision: **narrow the current allocation claim**. Longer search did not reveal
a clear placement advantage worth its extra cost in this fixed recipe. This is
an exploratory inconclusive result, not equivalence, a powered negative result,
nominal-coverage assurance or priority claim. The
[plain-language result](docs/research/2026-10-18-long-probe-diagnostic.md) explains
the next possible timing question without authorizing another study. No depth,
new score, transfer or automatic cohort is triggered. Canonical variance/runtime
calibration, accepted confirmation counts, true freeze, independent confirmation
and strong practical comparisons remain unmet. Old pilot/report bytes and all
eighteen historical rulings remain unchanged. The only new controller ruling is
D030's Python timing-wrapper recovery and its missing supplementary OS counters.
The focused plan is complete after publication verification and its local commit;
the larger original evidence program is not complete. Work stays on main.

## D032 — 18 October 2026: public research release

The owner explicitly requested public preparation and publication. The release
keeps the scientific source, tests, configurations, core lock and original
reports unchanged from research handoff `7af00e8`. Preparation commits `a7e692c`
and `b1f04bc` add a human README using the requested write-like-me skill, concrete
results and limits, runnable layer example, reproduction guidance, citation,
contribution notes and retained licensing/provenance.

Original software uses AGPL-3.0-only; original prose and result presentation use
CC BY 4.0, matching the owner's recent research repositories. Vendored
NODE/entmax/QHOptim retain MIT. The distribution expression includes all three
terms after independent review; all eight legal/notice files and canonical
license bytes were checked. CITATION.cff validates against official CFF 1.2.0,
identifies Artem Legotin/software 0.1.0 and invents no DOI or paper.

The clean-clone quick start passed 89 tests/1skip, both standalone references,
the actual layer example and CLI help. A first core-only full attempt retained
928 passes / 48 skips / 74 failures in 1920.73s. All failures were setup-related:49
interop-thread mismatches and 25 optional reference/search dependency cases.
Existing hash-pinned overlays plus explicit Torch thread initialization resolved
all 74; an expanded affected/core/train/checkpoint run passed 228 / 1 skip in 82.27s.
The two results remain distinct; no second full-suite pass is claimed. The
[publication record](provenance/publication/2026-10-18/index.json) preserves both
outputs and the corrected setup without changing experiment code or outcomes.

The complete reachable history was scanned. All 71 initial candidates were
contextually inspected SHA-256 experiment identities, with 68 independently
recomputed. Exactly 59 immutable historical fingerprints are dispositioned; no
whole-rule/path/hex-pattern suppression exists. Staged and final committed scans
reported no new findings. The bounded review found no credential blocker.
Selected synthetic fixtures, owner identity and historical local paths remain
disclosed; full research archives and environments remain external.

Prepared main was pushed while private, then
[arlegotin/adaptive-tree-neurons](https://github.com/arlegotin/adaptive-tree-neurons)
was made public. Anonymous API/raw requests verified public visibility, main
commit `b1f04bc`, exact README bytes, citation and recognized AGPL license. The
[publication receipt](provenance/publication/2026-10-18/public-verification.json)
records that initial public source commit; subsequent receipt-only changes do
not alter it. Public presentation emphasizes the working primitive and reusable
matched experiment while retaining the inconclusive placement result. Original
confirmation and practical-baseline gates remain unmet.

## 2026-10-18 — Authorized research v2 continuation

The user supplied a complete finite research handoff and authorized specification,
planning and execution. Its versioned spec and plan are under `docs/superpowers/`,
with durable progress/gate/recovery records in `docs/research/v2/execution.md`.
Historical STOP decisions and reported outcomes remain unchanged in their scope.

WP00 used a fresh frozen-core Python3.12.13 environment: both standalone numerical
references passed, focused software checks89 passed/1 skipped, and independent
public long-cohort scalar means agreed within1.39e-17. No missing historical raw
archive was authenticated. WP01 introduced training metadata2, explicit parameter
birth maps and prepublication/load shape/dtype/nonnegative-moment/age validation.
Two material review findings were fixed with failing regressions first;220 targeted
checks passed and19 valid-run arrays remained byte-identical. The interrupted
broad suite is retained separately and is not a whole-suite pass.

WP02 is in progress: strict finite M0/M1 plans, role-safe views, planted controls,
snapshot callbacks, native tiny fixtures and resource-admission guards exist.
Full diagnostic matrices, Gate A, the proposed joint selector, native practical
suite and confirmation remain unexecuted. Full-matrix resource measurements and
remaining runner recovery qualification precede scientific admission. This is
software progress and a research contract, not a new positive allocation result.
