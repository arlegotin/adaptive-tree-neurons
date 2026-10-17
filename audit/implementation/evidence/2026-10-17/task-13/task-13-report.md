# Task 13 — pre-freeze audit/report and portable evidence implementation

Status: scoped implementation complete; stable verification passed; independent review and acceptance pending. This file is an implementation handoff, not scientific WP09 acceptance. No genuine pilot, real root freeze, independent scientific cohort, new benchmark result or completed scientific report has been produced.

## Scope and APIs

`analysis.audit.audit(study_dir: Path, protocol: FrozenProtocol) -> dict[str, Any]` validates live frozen source/design/input identity and the exact fixed cohort/attempt inventory, recomputes F population variance and raw original-unit metrics, audits original protected access and resume records, reads every completed role's selected and last checkpoints and B6 loser checkpoints through the existing strict complete-state loader, and reconstructs the whole-teacher analysis. Loaded training states are discarded and process RNG is restored. It performs no training, prediction or T generation. Missing, failed, unfavorable and zero-growth units remain visible. Integrity failures yield blocked output; incomplete evidence cannot become completed-cohort superiority. A complete negative cohort remains a valid scientific interpretation.

`analysis.report.render_report(audit_result: Mapping, output: Path) -> None` checks the audit content digest and emits deterministic Markdown. Reports include counts, teacher effects, original-unit regression metrics, regime availability, control contrasts/family adjustments, failures, exact measured costs and uncertainty/precision. Invalid/incomplete results get reasons and no success table. The CLI reconstructs an audit from artifacts; it has no editable-summary-score input.

`seal_archive(study_dir, protocol, output) -> Path` prospectively validates live evidence and copies an immutable inventory before amendment/source drift. Incomplete but structurally valid evidence may be sealed; blocked evidence cannot. The output contains `archive.json`, repository-relative bytes below `files/`, and a separate `copy-work.json`. Source/readiness/review files, prepared F/G/V and teacher artifacts, sealed pilot inputs, fitting/checkpoint graphs, model payloads, original protected links/predictions/access-ledger prefix and deterministic/work reports are retained. Every copied byte is rehashed before publication.

`audit_archive(bundle, *, analysis_root=None) -> dict` is the separately versioned `archived-saved-evidence/v1` path. It verifies the copied inventory and original authorization, reconstructs from saved rows, compares reconstructed content with the sealed live analysis, records the original execution identity and the current clean analysis commit/execution/package identity, and makes all affected claims exploratory. It never returns a `FrozenProtocol`, issues a grant, fits/refits, invokes a predictor, generates T features/labels or resumes old-source training. Later ordinary live `evidence_read` events do not modify the copied ledger prefix.

`summarize_cohort` extracts the existing pure Task 10 aggregation/inference boundary without changing estimands, teacher grouping, sign, bootstrap/t sensitivity, optional Holm policy or Ruling 11 descriptive masks. `load_evaluation` still validates the live source before saved reads and logs `evidence_read`; its internal sealed-row reader additionally checks registration/grant/access/completion order and identities, exact F variance and operational resume evidence.

## Portable identity and compatibility

`locators.py` translates only original-root/producer-relative locations to the actual restored repository root. A context-local mapping leaves frozen config strings, hashes and artifact bytes untouched. Stage-relative paths retain their enclosing-run semantics. Frozen protocol/history and scientific execution digests remain binding; dirty, untracked or changed executable/lock inputs still reject. The actual protocol must live under the current checkout's `protocol/` directory. Ordinary confirmation uses a scoped serial producer working directory so unchanged repository-relative receipt paths resolve correctly; it restores the caller's CWD afterward. The numerical producer, train implementation, core/NODE locks and genuine CPU1/1 receipt have not been changed.

Public `FrozenProtocol` fields and existing source/cohort/confirmation schemas remain compatible. New APIs do not add ignore-source or archive-mode execution switches. The archive format is prospective v1 only. There are no genuine old protected studies needing retroactive migration. A missing seal after amendment cannot be repaired by silently restoring source or deleting an amendment. Root Ruling 12 governs this boundary and its recovery/rework cost; it is referenced from D017 and README. Ordinary source acceptance bindings for shared Task 9/10 files require root renewal after actual independent review.

## Accounting and deterministic content

The audit reconstructs the authoritative unit/repetition sequence and finite attempt indexes/phases, checks evaluation-only retries reuse the same pair/path, retains failure/interruption reasons, detects unaccounted attempt directories, and requires completed prediction references at the original protected location. Every selected graph retains the B6 coordinator and all candidate losers; selected/last/probe descriptors, tensors, topology, Adam state, config/source/lock/runtime identities, parameter terms and timing terms are checked. Model pointer/weight correspondence remains the accepted Task 10 graph check.

Preparation and active confirmation invocation time are disjoint. Child/control clocks are diagnostic nested clocks and are never summed into the enclosing total. Missing preparation or hard-stop durations yield null combined time and a reason. Frozen forecast deviations are recomputed from linked receipts. The execution storage sample is explicitly identified as taken before the final ledger write; subsequent report/evidence-read files are not mistaken for that old sample. RSS and MPS unified-memory counters are not added as disjoint usage.

Deterministic results have `schema_version`, `mode`, `valid`, `status`, `reasons`, protocol/checkout identity, `primary`, `accounting`, `limits`, `conclusion` and `audit_sha256`. Primary content retains Task 10's content-addressed report. `audit-starts/UUID.json` marks an audit invocation; immutable `audit-work/UUID.json` records its duration and hashes of nested primary-analysis work receipts. Its total counts enclosing audits once plus standalone analyses, excluding nested analyses; an unmatched start or invalid work record leaves the total unknown. Audit snapshots are `audit-HASH.json`. Archive copy, archive analysis and report rendering have separately bounded `copy-work.json`, `archive-work/UUID.json` and `report-work/UUID.json`; their varying durations are outside deterministic scientific content.

## CLI and phase boundaries

- `audit --protocol PATH --study-dir PATH [--output JSON]` reconstructs live evidence; invalid/incomplete returns exit 1.
- `report --mode audited --protocol PATH --study-dir PATH --output MARKDOWN` reconstructs first and renders afterward.
- `seal-archive --protocol PATH --study-dir PATH --output DIRECTORY` copies still-valid prospective evidence.
- `audit --archive DIRECTORY [--output JSON]` and `report --mode archived --archive DIRECTORY --output MARKDOWN` use only the copied inventory under current clean analysis source.
- Existing exploratory report/confirmation/preparation/amendment commands remain supported.

`reports/final.md` is an explicit blocked pre-study status. H1 placement evidence never establishes H3 practical competitiveness. Unavailable strongest practical models remain limits, Energy is separate simulation, and Track C references are excluded from local measured tables. R12 full-text retrieval was resolved in WP00; the bounded ArborCell comparison is provisional, and actual closest-prior-art review remains at the eventual claim boundary. No copyrighted PDF is copied into the new public artifacts.

## Bounded verification, not scientific evidence

Existing protocol fixtures provide a temporary clean Git repository and simulated pilot/readiness graph. The real incomplete integration prepares 32 toy units, fits one teacher × two repetitions × all eight roles with actual models/checkpoints, and deliberately retains the other 31 teachers' 124 failed attempts. Only pytest's existing numerical receipt and interop setting are substituted; no fresh genuine CPU1/1 receipt or actual scientific pilot is created. This test checks original-unit recomputation, two identical audits/Markdown renders, work accounting, corruption traps, actual restored saved-model reads with the original root unavailable, terminal confirmation reopening with no fitting, prospective copy, live evidence-read append independence, amendment/source drift and archived reanalysis with no-new-T/fit/predict traps.

A separate pure raw-row negative/incomplete fixture checks teacher averaging and sign. The complete bounded negative audit fixture passed in the stable full suite: it uses eight units to respect the design object's minimum count, an explicitly substituted toy numeric-design gate, actual eight-role fitting graphs, and controlled adverse predictor outputs solely to guarantee a negative sign. Those outputs are not model-performance evidence. The test retains all actual model/linkage/checkpoint/access validation and exercises missing B6 candidate rejection and completed archive relabeling.

The checkpoint test checks actual fitted saved/last states, missing parameter terms and invalid sampler resume state. Protected metric/resume tests check changed scalar summaries and omitted terms against raw rows, and reject changed resume IDs. CLI missing-protocol paths produce blocked reports. All tests and source changes are serial; source is unchanged during every test invocation. No helper, reviewer, branch/worktree switch, push or external publication was used.

## Observed RED/GREEN record so far

Commands use `.venv/bin/python -m pytest`; complete outputs are retained as `/tmp/task13-*.txt` for root archival.

| Log | Selected tests | Observed result |
| --- | --- | --- |
| task13-red.txt | tests/test_audit.py -q (initial 3 cases) | 3 failed, 11.35s: relocated original-root rejection and missing audit/report APIs |
| task13-portable-green.txt | test_live_frozen_identity_survives_restored_checkout | 1 passed, 10.20s |
| task13-conclusion-red.txt | test_negative_complete_rows_and_missing_pair_have_different_conclusions | 1 failed, 1.05s: missing pure cohort reconstruction API |
| task13-initial-green.txt | tests/test_audit.py tests/test_primary.py -q | 15 passed, 27.80s |
| task13-integration-red.txt | real incomplete integration and metric/resume traps | 1 failed, 1 passed, 104.48s: moving file-count field broke deterministic reports |
| task13-checkpoint-cli-red.txt | checkpoint inventory and CLI cases | 2 failed, 4.70s: missing checkpoint inventory API and audit CLI |
| task13-checkpoint-cli-green.txt | same cases | 2 passed, 8.79s |
| task13-integration-green.txt | real incomplete integration | 1 passed, 130.71s |
| task13-work-relocation-red.txt | real integration with work receipts | 1 failed, 95.52s: missing outer audit-work receipts |
| task13-audit-green.txt | tests/test_audit.py -q | 8 passed, 192.53s |
| task13-negative-green.txt | complete negative fixture, first setup | 1 failed, 9.31s: fixture K=3 rejected by production minimum K=8; corrected fixture, no weakened count guard |
| task13-native-error-red.txt | malformed native evidence-read classification | 1 failed, 8.63s: native RuntimeError escaped blocked result |

Final focused/full-suite output, source/commit identities, self-review and next command forms follow below. No passing/full-completion assertion was based on a running command.

## Narrow pre-freeze numerical-check correction

The complete negative fixture's original seed42 run retained 15/16 valid runs;
unit6/rep1 failed twice with identical first-Adam comparison error. The audit
correctly remained incomplete. Original focused output: 1 failed, 1 passed in
282.90s (`/tmp/task13-final-focused.txt`). The failed log is preserved under
`/tmp/task13-seed42-diagnostic/original-pytest-failure.txt`. An unrelated Python3.13
pytest in `/Users/artemlegotin/Documents/git_alter/FAM-Jet` pruned the shared default
pytest temp root. It was not stopped or altered. All later tests use dedicated
`--basetemp` directories. The original tensors could not be retained, so root
authorized reconstruction only of the exact original unit6/rep1 streams/recipe.
No seed was changed and no different-unit graph was substituted.

The isolated rebuild reproduced the identical exception in 13.19s, package hash
`a6495f78e8ea46785559ae5e277262834124a053d0521ade8b304c595e5ceeb5`:
`causal_evidence.py:69`, B6/greedy `layer.gate.weight[0,11]`. Float32 actual/reference
gradients were -1.400608429236172e-8 / -1.3866156223230064e-8, giving first-update
difference 2.442335244268179e-6. Float64 from the exact same operands agreed to
4.065432886168896e-15. Full traceback, config/seeds, float32/float64 operands,
model and diagnostic JSON are in `/tmp/task13-seed42-diagnostic/`. The exact model
is the regression fixture `tests/fixtures/causal_seed42_probe.pt`; its accompanying
Markdown records provenance, operand values and the conditioning derivation.

Root authorized a narrow correctness-check repair before freeze, without a new
hypothesis or ruling. Actual training/model/score/seed behavior is untouched.
Float32 forward/gradient tolerances remain unchanged. Each first AdamW update is
verified independently against its own retained gradient's bias-corrected formula
in float64. A componentwise allowance is derived from `g/(abs(g)+eps)`, plus a
conservative gamma_20 floating-point rounding bound on the two update terms.
This replaces the unjustified flat cross-reference updated-parameter tolerance.
The observed optimizer error remains reported, the existing overall receipt
limit stays 1e-4, and exact next-update serialization checks remain intact. An
injected common wrong update must fail the own-gradient formula check.

Amendment integration now accepts optional `archive` / CLI `--archive`, verifies
strict archive keys, copied inventory, original frozen/history identity and live
audit digest, and records either `sealed_saved_evidence_only` with its digest or
`unavailable_no_archive_supplied` with null digest. It does not restore live
eligibility or rewrite the original protocol/amendment. Actual archive→amend
coverage checks damaged inventory, unknown archive keys and another original
protocol before the positive binding.

Additional self-review closed an outer-ledger accounting gap: unknown outer
statuses reject; a completed ledger requires every declared run complete; an
honest running/interrupted/incomplete publication remains incomplete even with
all inner rows. Known attempt clocks may not exceed enclosing invocation clocks
apart from sum-rounding tolerance (relative 1e-12, absolute 1ns). Hard interruption
with a missing enclosing clock remains explicitly unknown and is not treated as
zero. Four direct ledger/raw-row RED cases failed as expected in 1.16s before the
fix; no new numerical cohort was used for these checks.

## Final focused checks and stable-suite launch

All commands below use `.venv/bin/python -m pytest` and explicit unique basetemp
locations after the shared-temp pruning incident. No executable file changes
occurred during any active test command.

| Exact selection (plus `-q --basetemp=PATH`) | Result | Log |
| --- | --- | --- |
| `/tmp/task13-seed42-diagnostic/test_rebuild.py`, PATH `/tmp/task13-seed42-diagnostic/pytest` | expected original failure, 13.19s | `/tmp/task13-seed42-diagnostic/rebuild-log.txt` |
| `tests/test_audit.py::test_amendment_rejects_missing_supplied_archive`, PATH `/tmp/task13-amend-missing-red` | 1 expected missing API failure, 6.91s | `/tmp/task13-amend-missing-red.txt` |
| `tests/test_causal_precision.py`, PATH `/tmp/task13-precision-red` | 2 failed, 0.48s; initial test import typo, corrected before real RED | `/tmp/task13-precision-red.txt` |
| `tests/test_causal_precision.py`, PATH `/tmp/task13-precision-red2` | 2 failed, 1.13s; exact original parameter comparison and missing own-gradient check | `/tmp/task13-precision-red2.txt` |
| `tests/test_causal_precision.py tests/test_audit.py::test_amendment_explicitly_records_absent_archive_eligibility tests/test_audit.py::test_amendment_rejects_missing_supplied_archive`, PATH `/tmp/task13-precision-amend-green` | 4 passed, 14.37s | `/tmp/task13-precision-amend-green.txt` |
| `tests/test_causal_baselines.py tests/test_causal_precision.py tests/test_audit.py::test_real_incomplete_audit_relocation_archive_and_tamper`, PATH `/tmp/task13-covering` | 46 passed, 1 inherited NODE warning, 179.96s | `/tmp/task13-covering.txt` |
| `tests/test_audit.py::test_outer_ledger_status_and_enclosing_work_are_authoritative tests/test_audit.py::test_nonterminal_publication_does_not_claim_complete`, PATH `/tmp/task13-accounting-red` | 4 failed, 1.16s; intended status rejection/completion failures | `/tmp/task13-accounting-red.txt` |
| same two accounting selections plus `tests/test_primary.py`, PATH `/tmp/task13-accounting-green` | 15 passed, 8.69s | `/tmp/task13-accounting-green.txt` |

Stable command: `.venv/bin/python -m pytest -q --basetemp=/tmp/task13-stable-full > /tmp/task13-stable-full.txt 2>&1`.
The launch manifest `/tmp/task13-stable-source.json` records BASE
`110a33f3a6412e7f07237fde76580fe0b9018a54`, 313 execution/test/fixture file hashes,
and package SHA256 `0a372fd8bc4c9a4e7179c15480fd90ee27f413ce9ca70409d5af4de1f3d7bd7d`.
Completion, post-run comparison and commit identity are recorded below only after
observing the command result.

## Self-review and later scientific gate

Self-review checked the requested interfaces, ordinary source/lock/runtime guards,
all-row accounting, original-unit reconstruction, prospective archive authority,
restored source/graph/receipt locator handling and deterministic content versus
variable work. It found and corrected outer ledger status/clock accounting; root
reviewed the independently established numerical-check defect before its narrow
repair. A completed test run does not renew Task9/10/13 acceptance automatically.
Root still owns fresh independent spec/quality review, shared-file acceptance
renewal, and the final broad H1 review including carried NODE row-trace cost and
Task8 recommendation-cost Minors. No helper/reviewer was used by this implementer.

Later execution must use fresh reviewed source, real thread/BLAS receipt and
measured resource/readiness inputs. The old blocked design is not executable
scientific authority. After those gates, the command forms are:

```sh
.venv/bin/python -m treeneuron.cli calibrate --config GENERATED_SCIENTIFIC_CONFIG
.venv/bin/python -m treeneuron.cli prepare-cohort --design GENERATED_NUMERIC_DESIGN --output FRESH_COHORT_DIRECTORY --cohort-id FRESH_ID --root-seed REGISTERED_SEED --publish GENERATED_PREPARED_DESIGN
.venv/bin/python -m treeneuron.cli freeze --design GENERATED_PREPARED_DESIGN --output protocol/frozen.yaml
.venv/bin/python -m treeneuron.cli confirm --protocol protocol/frozen.yaml --study synthetic
.venv/bin/python -m treeneuron.cli audit --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output audit/SCIENTIFIC_AUDIT.json
.venv/bin/python -m treeneuron.cli report --mode audited --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output reports/final.md
.venv/bin/python -m treeneuron.cli seal-archive --protocol protocol/frozen.yaml --study-dir protocol/confirmation/PROTOCOL_HASH --output audit/archives/FRESH_STUDY
```

These uppercase values are future registered artifacts, not supplied parameters
or permission to freeze now. Actual clean locked reproduction, interrupted-growth
checks and a representative paired pipeline are still required at the eventual
scientific reproduction boundary. Deterministic report generation must be repeated
there against the same authorized saved evidence; no new adaptive T evaluation.
Before an amendment, seal while original live source is valid, then use
`amend --protocol protocol/frozen.yaml --archive audit/archives/FRESH_STUDY --reason ... --replacement-study-id NEW_ID`.
Archived report command:
`report --mode archived --archive audit/archives/FRESH_STUDY --output reports/archived.md`.

## Stable verification result

The exact stable command above exited **0**: **908 passed, 48 skipped, 28 warnings
in 1485.54s (24:45)**. All 28 warnings are the inherited NODE/vendor warnings:
TorchScript deprecation, small-sample data-aware initialization, quantile count
reduction and deprecated optimizer overload. The 48 existing skips remain limits;
this is not blanket native-backend coverage. Full output SHA256:
`5fcf8cd6f9d4417159cedb1cf976095358f281b4324ac36f187d5c0f0b60431a`.

Post-run comparison verified all **313** launch-manifest execution/test/fixture
files unchanged and the exact package SHA256 still
`0a372fd8bc4c9a4e7179c15480fd90ee27f413ce9ca70409d5af4de1f3d7bd7d`.
`git diff --check` passed; branch remained `main`. The numerical producer is still
`237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e`;
core `uv.lock` SHA256 is still
`13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`;
genuine backend receipt
`environment/evidence7-calibration-fix1/search-backend.json` remains
`e403a1fb2216b58557dfb414254facda09acfaa5dfbabf05a9f61ce8bca7095d`.
No receipt regeneration or hash waiver was used. Package changes intentionally
invalidate old-source checkpoints; the exact seed42 tensor regression is a
plain bounded model fixture, not a checkpoint-resume exception.

The stable suite completed all original seed42 unit/repetition streams (16/16),
accepted its controlled-output complete negative audit, rejected the missing B6
probe, and successfully relabeled the sealed complete archive exploratory. The
actual incomplete fixture passed restored-checkout model audits with the original
root unavailable, terminal no-fit reopening, repeated deterministic reports,
work accounting, pre-amendment archive capture, binding/rejection checks and
saved-only archived reanalysis. No scientific result follows from these tests.

Only non-executable handoff/report prose was updated after the suite. No further
test run is needed to reconfirm unchanged executable content. Root's untracked
Ruling12 file is deliberately excluded from the implementation commit. The
scientific final report remains explicitly blocked pending the later gates.

## Commit and final handoff

Implementation commit: **`3b5095df5a5b16916ec5a103d128dea6100ce207`**
(`feat: audit frozen studies and preserve portable evidence`), on `main` over BASE
`110a33f3a6412e7f07237fde76580fe0b9018a54`. It contains 16 scoped files, including
the explicit blocked `reports/final.md`. Cached diff whitespace checks passed.
The Git-index sandbox required approved escalation; no approval-review rejection
remains. The general `*.pt` ignore rule required explicit force-add of the
11,246-byte toy model fixture, and no other ignored artifact was staged.

The broad launch manifest contained 313 files and excluded that then-ignored
binary. Its bytes were separately verified against the pre-fix diagnostic SHA256
`34e2d6653f14764df60ada5cdd5f4730b5c6123268d1174470df209c191b92a1`.
After commit, the production `execution_snapshot(root, HEAD)` passed strictly
with **309 files**, including that tracked binary, and execution SHA256
`b34e61a33f782ffa0bce87b61946cc872c5911bf2ca376c36476d48adef58826`.
The count is 313 minus five audit/environment curator scripts excluded by the
production execution-file policy, plus the tracked model fixture. No covered
bytes changed. Full committed source inventory is
`/tmp/task13-committed-source.json`; package SHA256 remains
`0a372fd8bc4c9a4e7179c15480fd90ee27f413ce9ca70409d5af4de1f3d7bd7d`.

Final `git status --short` contains only root-owned
`audit/implementation/evidence/2026-10-17/ruling-12-archive-sealing.md` untracked;
the implementation tree/index is clean. No branch/worktree change, push or
external publication occurred. The handoff index
`.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/task-13-evidence-index.json`
lists exact retained logs, source inventories, report and isolated numerical
diagnostic inputs with hashes for root's later public archive. Larger bounded
fixture trees remain in their explicit `/tmp/task13-*` locations.

Remaining concerns are explicit phase limits, not claimed completions: prospective
archives require a pre-amendment seal; legacy unsealed studies have no waiver;
the original failure tensors were pruned and the exact failure was reconstructed;
48 skipped tests and inherited warning/Minor debt remain; toy data/controlled
predictions establish implementation behavior only. Independent scoped review,
renewed Task9/10 bindings, final broad H1 review and real scientific execution
remain root-owned gates. No genuine frozen protocol or scientific result exists.
