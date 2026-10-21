# Fresh whole-branch source review before final qualification

Review read-only from base `230e6575149c9fee440ca301fa3782a85751bac8` to
head `783cb2d85d99abbd8e99b156ab7a5ed5ca69266c`, on branch
`research-v2-continuation`. Do not edit source, tests, index, HEAD, environment,
dependencies or the running native study. Do not spawn another agent. Do not
run actual benchmark/native qualification fits or inspect actual protected T.
Small isolated fixture/reproduction checks are welcome; do not manufacture a
successful actual native qualification gate.

This is the one fresh whole-branch review required by Superpowers inline
execution. Evaluate implemented software against the spec, plan, recorded
rulings and a reasonable user's expectations. It precedes the final executable
freeze and expensive native qualification/HPO; it is not a claim that the
scientific programme or release is complete.

## Binding materials and identities

- `docs/superpowers/specs/2026-10-18-research-v2-design.md`
- `docs/superpowers/plans/2026-10-18-research-v2.md`
- Durable ledger copy: `audit/research-v2/final-review/execution-ledger-at-783cb2d.md`;
  inspect its `Ruling:` entries when judging deviations.
- Full generated review package: `.superpowers/sdd/2026-10-18-research-v2/review-230e657..783cb2d.diff`.
  Durable compressed copy: `audit/research-v2/final-review/review-230e657..783cb2d.diff.gz`.
  Most volume is repeated evidence; inspect current first-party code and meaningful
  tests deliberately rather than loading16MB of generated artifacts at once.
- Changed execution-file hash inventory:
  `audit/research-v2/final-review/execution-change-inventory-783cb2d.json`.
- Actual committed execution snapshot:
  `artifacts/research-v2/WP09/production-execution-snapshot-783cb2d.json`.
  It passed the existing Git/index/working-tree checks on780 execution files.
  Package SHA256 `30fcf368a21cfe412b5c74bace4442c09a89f580a0101662daf421e23dd4658e`;
  core SHA256 `eeeeb85725e9da424f35a25af96a8837ad8c555eb8b557abb10143f5e2097962`.
- Current evidence/claim summary: `audit/research-v2/current-study/claim-ledger.json`
  and `reports/research-v2.md`.
- Latest source/software contract:
  `docs/research/v2/production-readiness.md`, `practical-cli.md`,
  `WP09-production-readiness-implementation-receipt.json` and
  `artifacts/research-v2/WP10/production-readiness-software/observation.json`.

## Review Focus (verbatim from plan)

- Malformed current or embedded-best state must fail before publishing a new pointer, without silently casting tensors.
- Zero-gradient, rank-deficient and full-row-span cases must remain visible with correct count/ties and no false span claim.
- Interrupted queues and G-best restoration must preserve model state while retaining losing attempts and consumed costs.
- Small-group and temporal splits must obey minima/24-clock-hour embargo and primary-only development barriers.
- Missing native anchors or predictions must block the corresponding claim and never become adverse predictive scores.

## What was implemented

State hardening, the role-safe diagnostic runner, planted/negative controls,
actual snapshots/stream treatments, exact parent-conditioned joint ridge
selection, independent development reconstruction, fixed transfer data/splits,
native adapters/qualification/supports, source-bound search and paid failures,
primary168/global freeze/secondary672, all840 model binding, full cost/protected
prediction/label/audit/analysis/timing/report paths, CLI and final source/software/
review readiness. Legacy scientific schemas, guards and reports are retained.

Scientific gates are real: M3 used its one8-to16-teacher expansion and failed
Gate B; no M3lambda correction or M4/M5/M6/C0/C1/C2 was admitted. Ordinary
continuation averaged15.6% lower E risk than joint growth, with no reliable
placement gain over random. The separate strong-backbone screen also failed.
Current practical P00 is explicitly the unchanged legacy cheap-score procedure,
not a post-hoc promoted successful joint selector. No new broad novelty is claimed.

The user replaced the two-hour fit cutoff with quality-first native stopping.
Native fits/searches have no wall cap;10/30/120-minute views remain descriptive.
They want exact models/data/effective batches, one-worker resource adaptation,
honest retries and results. No20GiB-free-RAM prerequisite is assumed.

## Evidence boundaries and current stage

Four actual supervised profiles under this core passed751 cases/44 distinct
test files: search257, protection197, pipeline159, reporting138. All four real
loaders reopened process/JUnit/source proofs. These are software fixtures,
including known tiny model fits, not actual benchmark outcomes. Prior failures
and source snapshots are retained. The historical whole repository run was
2217passed/42failed/369setup errors/6skips; later seven native profiles passed202
checks under their own earlier source. Seven current legacy numerical-producer
setup errors from host memory pressure remain disclosed. Do not infer a
monolithic current whole-suite green.

Actual final-source native qualification (all120 anchors/258 supports), the840
production searches, full public positive protected chain and final comparison
are UNRUN. Readiness code now exists and refuses caller Boolean shortcuts;
no genuine readiness seal exists before real qualification and this review.
These acknowledged experimental prerequisites are not themselves missing-code
findings. Flag code that cannot fulfill them or misreports them.

The older held native queue continues independently under exact sourcec33a84e
in `/private/tmp/atn-research-v2-node-recovery`; do not touch it or its shared
dependencies.118/120 initial engineering fits complete, CCPP then Bike remain.
Do not run broad/heavy concurrent tests that disturb its worker or other apps.

## Specific integration risks to examine

Review actual positive-path interfaces and sequencing, not only early negative
gates: dataclass/reference shapes, output path types, default namespaces, resume,
all failed calls/invocations and complete raw/metric/cost proof joins. Recent
fixtures found argument shadowing after successful confirmation and JSON-only
guards applied to directory/Markdown outputs; these were fixed. Also examine
whether nested validation accidentally causes impractical repeated global
work despite the permitted immutable in-call genuine context. No persistent
qualification cache, source migration or fake successful native loader was
admitted. Distinguish a demonstrated feasibility issue from hypothetical timing.

Check exact ridge algebra/scaling/eligibility, state semantics, native recipe
fidelity and data roles; verify scientific report wording and independent units.
Practical inference is conditional on five fixed observational holdouts, with
Energy, dependent folds and prefixes separate. CCPP exchangeability is unverified
and Parkinsons has few clusters. Dedicated/precision-qualified efficiency and
dataset-population superiority are not claimed. Explicit blocked/retired
development reporting can have `cases:[]`; it never substitutes for a failed
protected proof or relabels T as E/V.

## Response

Follow the Superpowers code-reviewer template: strengths; concrete Critical,
Important and Minor findings with file:line, effect, reproduction/evidence and
suggested correction; recommendations; verdict. Before the verdict, list every
behavior considered but declined to judge and why (empty if none). State your
coverage and limits honestly; do not claim full semantic reading of every
generated artifact. Grade by actual user impact, including behavior the spec
does not enumerate. Do not fix code. The executor will re-grade, ledger rulings,
make one verified material fix pass and retain deferred minors.
