# Task 10 independent review

## Spec compliance

- **PASS for the authorized pre-freeze implementation phase.** Reviewed `9fb5be61d094eb3c7b21a4200cfe4fc8186b735c..b40cee4d705a03b90044ac36b023803df69503e2` against the brief, global constraints, accepted dependency context, execution instructions and reported Ruling 11. No missing, extra or misunderstood implementation requirement requiring a fix was found.
- All four requested new files and the confirmation CLI are present. The additional preparation CLI, protected diagnostics, strict protocol extensions and fitting extraction implement the expressly authorized dependency contracts: `src/treeneuron/evaluation/confirm.py:43`, `src/treeneuron/analysis/primary.py:143`, `src/treeneuron/cli.py:37`, `tests/test_confirmation.py:1`, `tests/test_primary.py:1`.
- The decision accurately leaves scientific WP06 execution pending rather than claiming that bounded tests completed an independent cohort: `DECISIONS.md:234`.
- **Cannot verify / intentionally deferred:** a genuine measured pilot/design, actual frozen cohort, completed independent scientific evaluation, and scientific cost/coverage claims are outside this code gate. The tiny integration test deliberately completes only one teacher's two repetitions and retains the other declared units as failures: `tests/test_confirmation.py:190`.
- **Cannot verify / intentionally deferred:** Task 13 portability and explicit archival reanalysis after source changes. Current consumers retain the original live-root and source checks, including the fixed study directory and child checkpoint locators: `src/treeneuron/analysis/primary.py:146`, `src/treeneuron/analysis/primary.py:177`, `src/treeneuron/evaluation/confirm.py:117`. This review does not accept archive portability or relax those checks.
- **Cannot verify from this diff:** renewed formal acceptance bindings for changed Task 8/9 files. The controller must issue them from actual integration review evidence; the implementation correctly does not manufacture its own acceptance record (`DECISIONS.md:244`).

## Strengths

- **Preparation and fitting boundaries are concrete.** `prepare_cohort` validates the numeric design, uses the committed per-unit teacher/reference/data seeds, writes new teacher/F/G/V artifacts and only T row-ID commitments, and refuses an existing destination. `load_development` checks the retained arrays against the frozen manifest. The generator rejects test-count requests and derives F/G/V phase seeds from the declared data seed: `src/treeneuron/evaluation/confirm.py:43`, `src/treeneuron/evaluation/confirm.py:89`, `src/treeneuron/data/synthetic.py:313`. Preparation provenance relies on this prescribed producer and frozen references; it does not claim to reconstruct arbitrary edited pre-freeze inputs from seeds.
- **The eight-role extraction preserves the actual training graph.** Fitting receives development data only and saves actual selected models; the exploratory wrapper retains its separate E prediction/loss work. Registration is preceded by graph validation and a metadata/weight comparison to each role's G-selected checkpoint: `src/treeneuron/pilot/calibrate.py:85`, `src/treeneuron/pilot/calibrate.py:220`, `src/treeneuron/evaluation/confirm.py:108`, `src/treeneuron/evaluation/confirm.py:153`. The added real fitting test covers both zero and positive growth and rejects a substituted model reference (`tests/test_confirmation.py:46`).
- **Seeds and retries preserve the experimental unit.** Nominal optimizer commitments are mapped once at the legacy boundary; allocation and contrast receive separate versioned streams. The coordinator traverses the exact declared teacher/repetition inventory without a sign-dependent stopping condition. Fitting retries retain the same unit/seeds, registered models are reused for pre-label prediction retries, incomplete label access blocks further access, and sealed-evidence read failure terminates that run: `src/treeneuron/evaluation/confirm.py:28`, `src/treeneuron/evaluation/confirm.py:173`. Tests exercise fixed completion, finite failures, interruption, inference-only retry and sealed-read failure (`tests/test_confirmation.py:112`, `:135`, `:154`, `:246`, `:266`).
- **Saved evidence drives the analysis.** Analysis verifies frozen run/seed/data provenance, the parent recipe, the selected checkpoint graph and registered model references before loading protected rows. It reconstructs risks rather than accepting stored summary scores: `src/treeneuron/analysis/primary.py:143`. The real toy integration checks exact reconstructed D, deterministic report identity, and rejection of changed optimizer provenance (`tests/test_confirmation.py:190`).
- **Independent counts and inference safeguards are explicit.** Duplicate run identities deduplicate, conflicting identities reject, repetitions average within teacher/data units, and missing/invalid pairs retain the teacher without zero filling. Whole-teacher bootstrap, Student-t sensitivity, full effects, median, endpoint precision and supported-inference status are reported. Singleton/zero variance and incomplete cohorts cannot establish completed-cohort superiority: `src/treeneuron/analysis/primary.py:32`, `src/treeneuron/analysis/primary.py:92`, `src/treeneuron/analysis/primary.py:192`; tests at `tests/test_primary.py:24`, `:49`, `:56` cover the principal degeneracies and sign interpretation.
- **Diagnostic and inferential claims remain scoped.** Gate masks are computed inside protected evaluation after prediction and label-access recording. The versioned policy fixes the descriptive gate semantics; optional Holm adjustment uses only explicit named families. Other controls remain descriptive and H1 superiority receives a placement-only interpretation: `src/treeneuron/evaluation/protected.py:200`, `src/treeneuron/evaluation/protected.py:212`, `src/treeneuron/protocol.py:563`, `src/treeneuron/analysis/primary.py:197`.
- **Cost and report provenance distinguish measured from unavailable work.** Preparation and execution clocks are disjoint, child clocks remain nested, unknown interruption durations prevent a fabricated combined total, and variable reanalysis timing is kept outside deterministic content-addressed reports: `src/treeneuron/protocol.py:589`, `src/treeneuron/evaluation/confirm.py:269`, `src/treeneuron/analysis/primary.py:226`.

## Issues

### Critical

- None.

### Important

- None.

### Minor

- **M1 — Existing warning debt, not introduced by Task 10.** `.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/task-10-logs/task10-full.txt:15` retains 28 known warnings: Torch JIT/add_ deprecations, tiny-data NODE initialization notices and sklearn quantile-count notices. The full suite is therefore not warning-free. Keep this in the already deferred final NODE/warning triage; expected fixture warnings should eventually be asserted narrowly and deprecated calls addressed where ownership permits. This is not a new Task 10 regression or a reason to rerun the suite here.

## Checks and review limits

- Read the supplied full diff once in three contiguous sections; all ten changed files were covered. No git commands, suite reruns, source/index/branch changes, or subagents were used. The only output write is this requested review report.
- **Named risk: declared generation provenance versus retained development bytes.** Inspected the generator's configuration-to-teacher/F/G/V mapping in `src/treeneuron/data/synthetic.py:313`. The supplied protocol hunk ended mid-function, so also read the remainder of `_scientific_sources` and adjacent `fit_variance` in `src/treeneuron/protocol.py:605`; this was the permitted cut-off-function exception, not a second changed-file review. The producer uses the declared recipe, while manifest/file references bind the retained bytes consumed by fitting.
- **Named risk: metric reconstruction and degeneracy semantics.** Read `src/treeneuron/evaluation/metrics.py:18` to confirm the existing original-unit/group/tail/noise-free metrics, shape/finite checks, and explicit zero-F-variance result used by the new aggregation.
- **Named risk: compatibility with the frozen endpoint precision policy.** Checked the existing endpoint/statistical-policy definitions in `src/treeneuron/pilot/design.py:201` and `:411`. The new summary uses the same ten-disjoint-chunk endpoint standard-error convention and consumes the frozen draws/seed/alpha/tolerance/support fields. An additional guessed `pilot/power.py` path did not exist; no broader search was performed.
- Read the retained complete full-suite log; it ends **890 passed, 48 skipped, 28 warnings in 900.20s**. The controller supplied normal exit 0 and final-source binding; no duplicate suite was run. The report's substantive behavioral claims were assessed against the diff rather than inferred solely from this passing count.

## Assessment

**Task quality: Approved for pre-freeze implementation.**

**Counts:** 0 Critical, 0 Important, 0 newly introduced Minor; 1 inherited Minor warning obligation retained above.

The implementation supplies the required preparation, fixed-cohort protected execution and saved-row teacher-level analysis with meaningful bounded coverage. Scientific acceptance, Task 13 archive integration and renewed controller-owned acceptance records remain separate required gates.
