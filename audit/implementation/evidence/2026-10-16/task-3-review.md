### Spec Compliance

- ✅ Spec compliant for Evidence Task 3 and the carried dedicated-coordinator fix. The named shallow anchor, explicit QHAdam resolution, F-only preprocessing/initialization, G snapshot selection, native identity checks and bounded full-size worker are implemented in `src/treeneuron/baselines/node.py:27`, `src/treeneuron/baselines/node.py:102`, `src/treeneuron/baselines/node_checks.py:37` and `src/treeneuron/baselines/node_worker.py:15`.
- ✅ The three-update anchor is explicitly selection-unavailable and quality-ineligible; its successful execution does not claim a default plateau or predictive winner (`src/treeneuron/baselines/node.py:213`, `src/treeneuron/baselines/node_worker.py:39`, `configs/vendor/node/README.md:27`). Deferring longer calibration to Task 7 complies with this dispatch.
- ✅ B0–B6 verification is preserved when B7 fails, and only the caller-owned dedicated greedy coordinator is terminalized (`src/treeneuron/baselines/causal_checks.py:99`, `src/treeneuron/baselines/causal_checks.py:135`, `tests/test_causal_baselines.py:345`, `tests/test_causal_baselines.py:359`).
- ⚠️ Cannot establish historical upstream dependency versions, default convergence, or full-cohort scientific readiness from this diff; these are explicitly unclaimed/deferred (`configs/vendor/node/README.md:39`, `docs/research/2026-10-16-evidence3-node.json:265`). Root's final WP03 acceptance and subsequent Task 7 calibration remain separate gates (`DECISIONS.md:174`).

### Strengths

- Numerical source fidelity is unusually reviewable: the four unchanged runtime files are byte-identical to their archived originals; QHAdam changes only its relative import, and the extracted `check_numpy` AST is unchanged. The six mappings and hashes are explicit (`configs/vendor/node/runtime-files.json:1`, `configs/vendor/node/compatibility.patch:1`). Review-time checks reconstructed these files from the diff and verified all six mappings, all fourteen source hashes and all seventeen wheel-member hashes.
- Independent reference imports use archived upstream modules rather than adapter wrappers; tests exercise nonzero sparse gradients, support boundaries, the real selector axis, model initialization/head and QHAdam updates (`tests/node_reference.py:7`, `tests/test_node.py:36`, `tests/test_node.py:52`, `tests/test_node.py:65`, `tests/test_node.py:119`). The sigmoid counterexample and CPU64 known-value/derivative check add useful behavioral anchors (`tests/test_node.py:52`, `tests/test_node.py:244`).
- F/G separation is structural and tested by changing G while requiring identical initial state and quantiles (`src/treeneuron/baselines/node.py:120`, `tests/test_node.py:82`). Training includes remainder batches and records actual prefix/coverage rather than inventing 1000 initialization rows (`src/treeneuron/baselines/node.py:129`, `src/treeneuron/baselines/node.py:145`).
- Newest-first sequential averaging matches the archived Trainer reduction, raw weights are restored before subsequent optimization, and the selected optimizer corresponds to the selected report's newest snapshot, as upstream does (`src/treeneuron/baselines/node.py:64`, `src/treeneuron/baselines/node.py:153`, `configs/vendor/node/upstream/lib/trainer.py:92`, `tests/test_node.py:190`, `tests/test_node.py:222`). Independent raw-update replay and fresh-reload next-update checks cover consequential state behavior (`tests/test_node.py:131`, `tests/test_node.py:165`).
- Endpoint serialization uses nonpickled arrays with both archive and metadata integrity checks, enforces source/lock/runtime identity, and refuses uninitialized persisted models (`src/treeneuron/baselines/node.py:226`, `src/treeneuron/baselines/node.py:244`). The separate full-anchor worker reports its actual workload, process-memory scope and startup/save/reload boundaries (`src/treeneuron/baselines/node_worker.py:15`, `docs/research/2026-10-16-evidence3-node.json:95`).

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- None.

#### Minor (Nice to Have)

- M1 — `src/treeneuron/baselines/node.py:151`, `src/treeneuron/baselines/node.py:219`, `src/treeneuron/baselines/node.py:238`: every training-row visit becomes a retained Python integer and is serialized into JSON. This is harmless for the measured three-step worker but grows with updates × batch size; tens of thousands of default updates can add hundreds of MB of bookkeeping, plus serialization copies, to calibration costs. Before long Task 7 runs, prefer a total counter, bounded per-row coverage state and a reproducible sampler seed/digest; retain the full trace only for explicit small replay fixtures, or account for its overhead explicitly.
- M2 — `tests/test_node.py:20`, `tests/test_node.py:82`, `artifacts/evidence3-node/required-node-tests.txt:2`: the required test run leaves 28 expected warnings uncaptured. Inspection confirms short-F initialization/quantile-clamping warnings and original TorchScript/QHAdam deprecations, rather than numerical failures. Capture/assert these narrowly in tests so unexpected warnings remain visible; preserve the vendor numerical files and retained original execution logs. This is test-output hygiene, not a request to modernize vendor semantics or block the verified CPU path.

### Assessment

- **Task quality: Approved.** Counts: 0 Critical, 0 Important, 2 Minor.
- **Reasoning:** The implementation preserves the requested numerical recipe and observable checkpoint behavior, with real independent archived-source execution and explicit limits on the full-size cost receipt. The remaining issues concern long-run bookkeeping cost and expected-warning handling, not the correctness or eligibility of the retained Task 3 evidence.
- **Review checks:** Reviewed implementation/tests/configuration and relevant upstream notebook/trainer/kernel hunks in passes from `review-232ba2c..bc8da2b.diff`; did not run git, a test suite, or the CLI. The first combined archive/notebook output was truncated by embedded image data; reread only the affected textual hunks with embedded images omitted. Mechanically compared archived/runtime file bodies and hashes from the diff as described above.
- **Focused outside check:** Named risk: moving the B7 call outside the artifact-directory context might use an already-deleted temporary directory. Inspected only the missing unchanged context at `src/treeneuron/baselines/causal_checks.py:42`; it uses `nullcontext` around a deliberately retained `mkdtemp` directory, so that risk is absent.
- **Focused evidence check:** Read `artifacts/evidence3-node/required-node-tests.txt:1` to classify the specific warning-noise concern; it records 19 passed and 28 warnings. No runtime doubt required an additional test. Root separately verified all 521 indexed artifact size/SHA pairs and the retained 678-pass/48-skip full-suite output; this review did not duplicate that audit.
