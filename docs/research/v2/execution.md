# Research v2 execution ledger

Read this with `docs/superpowers/specs/2026-10-18-research-v2-design.md` and
`docs/superpowers/plans/2026-10-18-research-v2.md`. This is the durable resume
entry point. Update it after every task, gate, failure recovery and source change.
Unchecked work is not completed, and prior investigation findings are not new runs.

## Intake

- User authorized specification, plan and execution of the full finite handoff.
- Starting HEAD230e6575149c9fee440ca301fa3782a85751bac8, clean `main`.
- `origin/HEAD` missing; no scientific path diff against pinned commit.
- Normal checkout, no linked worktree or submodule detected. Isolated branch or
  worktree creation is authorized by WP00; avoid a second environment sharing
  stale distribution metadata.
- Existing `.venv` and `environment/` are historical infrastructure; not qualified
  for new work until measured. No current test or new efficacy result yet.

## Task and gate status

| Package | Status | Required next evidence |
|---|---|---|
|WP00|qualified|97228c6; references passed;89 passed/1 skipped; arithmetic error1.39e-17; tracked WP00 receipt|
|WP01|verified|220 targeted passed;19 reference arrays byte-identical; WP01-receipt.json; broad integration still pending|
|WP02|in progress|first native resource case completed8 roles/65 probes; independent arithmetic/replay verified; full M0/M1/admission pending|
|WP03|diagnostics in progress|boundary/prefix/head helper and history verified; finite M2 plan reviewed; controller/resource admission and Gate A pending|
|WP04|pending Gate A|joint selector numerical and trained evidence; Gate B|
|WP05|not admitted|measured geometry/newborn trigger|
|WP06|not admitted|measured retention/family ceiling; at most one redesign|
|WP07|data prepared|six raw pins and30 splits audited;41 affected passed; all native qualification/Gate D still pending|
|WP08|pending|all production evaluation code tested;168 primary searches; Gate E|
|WP09|pending|variance-only calibration/simulation; Gate F|
|WP10|pending Gate F|frozen complete execution; Gate G|
|WP11|pending|independent audit and faithful final report|

## Decisions and deviations

- The supplied handoff is the authorized scientific design. No additional design
  approval round is needed to execute the user's explicit request. Implementation
  details follow existing Python/Torch conventions. Native execution in this
  session is the default; independent reviewers are used where skills require it.
- No practical full-suite or allocation-success claim has been established.
- Builder review returned three material ambiguities. Resolved in the spec before
  new outcomes: planted controls use supplied ungrown parent/fresh Adam/zero gates;
  portfolio chooses per split/repetition; qualitative development gates now have
  explicit2% opportunity/repeatability and interval/correlation rules. These are
  implementation choices, not user-supplied historical preregistration.
- Authorized branch creation initially hit the filesystem sandbox; the same
  operation succeeded through automatic escalation. Current branch is
  `research-v2-continuation`; no rejection or user approval was required.
- Task1 complete: commit97228c6, fresh Python3.12.13 frozen core; references and
  89-pass/1-skip focused suite; actual CPU receipt; scalar reaggregation only.
  First DNS failure retained in command receipts. Fresh setuptools-generated
  source egg-info duplicate was archived; installed dist-info remains.
- Task2 RED: all15 semantic corruption assertions failed because old code accepted
  corrupt state; legitimate roundtrip passed (`WP01/red.log`). Before reference
  selected update6/cursor8, source089b03d13e1950b8e0c1ff54a147a121cf78db66acaaa2dcedd9ebf29afc3543.
  New validation is isolated in `state_validation.py`; shape rules do not construct
  modules or consume RNG. Training metadata version2 rejects schema1 under this
  new source; exact old loading uses old source. Original age equals global update;
  birth maps supply exact new ages. The reviewer identified the initial missing-
  birth fallback as a real bypass; it was removed after a failing regression.
  Local manual additions can register birth only with complete age0/zero-moment
  records, while deserialization requires the explicit map. Incumbents require
  birth0. Explicit freeze/applied-count policies remain Task6.
- Task2 independent review found a second material case: process-default float64
  makes PyTorch lazily create float64 Adam step scalars despite float32 parameters.
  A RED regression reproduced it. Training now explicitly initializes fresh Adam
  slots to the declared float32 recipe immediately before their first applied
  update, preserving lazy update0 behavior and same numerical update equations.
- The first broad suite was deliberately interrupted for these material fixes:
  115 passed in242.90s, KeyboardInterrupt retained. It is not a full-suite pass.
  Targeted state/training/causal verification is running on the corrected source;
  broad integration verification follows when production source stabilizes.
- Task3 preparatory contract tests:7 RED failures, missing research package and
  snapshot API (`WP02/red.log`). No research efficacy fit has started.

## Resume procedure

1. Read spec, plan, this ledger and most recent source/runtime record.
2. Check Git status and active processes; do not duplicate running work.
3. Reconcile scientific source digest before reusing receipts/checkpoints.
4. Read the next task brief and its prerequisites, then implement/test or execute.
5. Retain failures, terminal counts, raw artifacts and exact commands. Update this
   ledger and the plan checkboxes before context is lost.

Artifacts belong under `artifacts/research-v2/`, qualification declarations under
`docs/research/v2/` and `locks/research-v2/`. No production T created/accessed yet.

Task2 review dispositions: fixed missing-birth contract and process-default float64
Adam scalar findings using failing regressions before changes. Deferred minor:
new corruption tests do not directly assert optimizer-load call count/RNG preservation;
existing coverage and preconstruction ordering cover that boundary. Schema1 loading
is deliberately restricted to its original source; Task6 freezing and arbitrary
moment-history authentication remain separate. No additional reviewer was dispatched.

WP02 implementation in progress (not Gate A evidence): strict M0/M1 plan expansion,
v2 namespace/32-bit collision handling, F/G-only views removing noiseless metadata,
actual-update snapshot callbacks with isolated tensors/RNG, exact raw planted-control
transport and unfitted parent, native reference/control workers and raw score/prediction
artifacts are implemented. Tiny native paths and combined state/training checks passed
(46 tests before latest stream/recovery additions). Plans expanded to64 M0 final roles
plus520 nominal probes and256 M1 roles; initial plan artifacts bind their earlier
source revision and must not be silently overwritten after code changes.

Production resource admission is enforced before creating a matrix output. The
checkout had about19GiB free when checked; no full-budget scientific matrix has been
started. An asynchronous question asks the available compute/storage envelope for
the full practical program; independent implementation continues meanwhile.

Native legacy pair recovery uses the spec-permitted one deterministic same-seed
infrastructure rerun with original artifacts retained. Exact per-stage trainer
recovery remains tested; the outer legacy multi-role graph does not pretend to
support an unimplemented checkpoint-stage resume. Cumulative cost precision,
second-retry limits and concurrent locking received material review fixes. Actual
250/1000 snapshots are integrated with M0, using atomic checked checkpoint pointers.
Control fitting now receives an F/G-only view and observation-free parameters,
rather than full development/E objects. Execution identity also binds core lock,
actual dependency versions/runtime and backend receipt bytes. Native measurements
must match source, execution, training and workload hashes; asserted measurements
cannot admit a matrix. Post-review affected integration:87 passed in49.77s.
Full production qualification remains pending. No source freeze or T access occurred.

The runner reviewer also noted incomplete WP02 work: native replay classification,
custom prevalence/covariance constructors, additional collision/zero/tie coverage,
resource forecasts/admission and full M0/M1 execution. These remain required;
foundation tests do not make WP02 complete. The fixed first M0 pair resource
recipe is stored in `scripts/measure_reference.py`; it retains all8 final roles,
the parent and all65 nominal short trials, separately from a full cohort.

WP02 diagnostic completion is being integrated from isolated checkout
`/private/tmp/atn-research-v2-next`, branch `research-v2-next`, to preserve the
running native qualification's unchanged source in the original checkout.
Added replay classification with ordered raw-event verification, custom
prevalence/covariance constructors under a separate identity, and explicit
32-bit collision/current-scorer zero/tie tests. Review reproduced a certificate
bug accepting contradictory streams; all seven structural counterexamples and
the independent G-match-count regression first failed, then passed after repair.
The latest combined runner/diagnostic suite passed35 tests in14.04s. Native tiny
patience replay is covered; no new stress fit or placement efficacy is claimed.
The eligible-zero-joint-gain cases remain Task5 tests, not a relaxation of the
current scorer's eligibility. Full M0/M1, measured admission and Gate A remain
pending. Original foundation and failed fixture-setup attempts remain retained.

The user confirmed best-effort execution on the shared computer, with resource
contention expected. Use one numerical worker; recheck memory/disk before
admission and preserve/retry operational interruptions under the finite recovery
rules. Timing with competing workloads is qualification evidence, not a formal
efficiency comparison. The full practical program is still an unexecuted ceiling.

WP07 data slice in progress in the isolated checkout: all six official raw pins
matched. Root independently rehashed116 retained files with zero mismatches;
the compact tracked acquisition receipt binds the54,899-byte manifest at
`0ef5199e064f079b0802a31a4e022fd469d1661fb56835f8d4ef3638bc0db4cf`.
Twelve initial DNS failures and successful request receipts remain. The original
two acquisition scripts are retained under docs/research/v2/scripts/acquire_uci
for clean acquisition; no raw labels were parsed by that workstream.

Implemented semantic table allowlists, target-free stored group/chronological
splits, F-only numeric/category encoding, separated label files and a primary-only
transfer store. Five RED missing-module tests now pass (latest0.22s), covering
target/proxy exclusions,12-group Energy minima, missing-hour embargo, reserved
unknown categories and label barriers. A separate hash-pinned data overlay has
openpyxl3.1.5/xlrd2.0.2/et-xmlfile2.0.0. Independent data-contract review and actual
preparation are pending. All14 native practical families, their qualification,
protected evaluator release and secondary-development release remain unimplemented.

Data-contract review is complete. A public mutable manifest could change the
secondary `primary` flag; the RED regression reproduced the bypass. Validated
metadata now stays private behind defensive copies, and access independently
checks the dataset's contractual primary fold. Expected file hashes are protected
by the same boundary. The coordinator receives aligned V labels/rows/groups/dates
without exposing them to workers. Duplicate configuration keys reject before
output creation; exact reader versions/import locations and duplicate distribution
checks precede preparation. No material review finding remains unaddressed;
actual six-table preparation and all native-adapter qualification remain pending.

Actual preparation completed under commit12d8086, using the isolated checkout's
source/data overlay from the original repository cwd. Output:
`artifacts/research-v2/WP07/prepared`; immutable suite/30 split manifests and
all role arrays are retained. Rows/groups matched the exact contract:
Airfoil1503/106, Concrete1030/428, CCPP9568/9527, Parkinsons5875/42,
Bike17379/731, Energy768/12. Original raw targets were parsed solely for
partition serialization; no transfer fit, selection or protected prediction ran.

An independent array auditor opened only feature/identity packs, rehashed every
retained file, and checked row/group disjointness, two-group F/G/V minima and
Bike chronology/24-hour embargo across all30 splits. It did not open protected
target packs. Audit SHA256:
`06c76aaf87be1f09967ca3c3b2b342bdc0d3b6f28126cf5b5e97b94bd1bc33c4`.
Prepared allocation8488KiB. The compact tracked WP07-preparation-receipt.json
binds source, suite, all realized row/group counts and dependency import identities.
The latest affected suite passed41 tests in14.35s; initial RED/review failures
and recovery logs remain under WP07/data-contract-tests. This completes data
acquisition/preparation only. Native adapters, qualification and Gate D remain
pending; secondary F/G/V and all T labels remain blocked through the store API.

WP03 state primitives now distinguish actual-current intervention from G-best
selection. A named research_boundary records its active parent snapshot, exact
requested update/current G risk and source/data lineage. Legacy insertion and
widening reject this boundary; explicit research wrappers reuse the same atomic
migrations. Tensor/Adam/sampler-generator/RNG equality and rollback through both
wrappers are tested. A fixed-budget policy disables patience explicitly while
retaining update0/strict G selection and the absolute schedule.

Review reproduced capture-plan drift on interrupted fixed-budget runs. Versioned
capture requirements now persist in TrainingPolicy; omitted/altered plans reject
on active resume. Canonical policy comparison also fixes tuple/list differences
after JSON roundtrip. The first affected integration run passed215 with one
missing optional NODE requests-overlay failure; the pinned overlay was restored.
The next integration passed257 with three probe-policy representation failures.
After the canonical comparison repair, all44 focused recovery/probe/runner checks
passed in57.71s. Final boundary/semantic checks passed24 in12.06s. This is not a
monolithic full-suite pass. All initial failures remain retained.

M2 dispatcher/finite matrices, prefix G-best capture, ridge-head diagnostics,
resource admission and Gate A still remain. The plain-language research state
is in status.md. In the first old-source M0 qualification case, all65 short
probes were stage-zero; keep selected its parent; top/random joint growth
improved G. Longer joint budgets confound interaction with horizon, and V/E
results are not yet available. This development observation changes no declared
seed, gate or matrix. The original checkout remains at source5c55fa9 while its
native eight-role case finishes; do not merge package edits into it mid-run.

WP03 state source committed as2db543b. A fresh real CPU receipt was generated for
that package, then the pre-hardening normal-path reference was rerun: all19
model/Adam/prediction/update arrays are exactly equal, with identical NPZ SHA256
`8c072d32850a63e1512e0fa6858363fe3ab5d7df3300914db1e469799ff6bb77`.
WP03-state-receipt.json binds the source, actual receipt, review disposition,
ten retained RED/failure/recovery logs and precise verification levels. Default
training behavior is preserved on this reference; no new efficacy result follows.

### WP03 frozen-head algebra, 18 October14:30 UTC

Implemented the six-alpha F-hidden-feature ridge helper and original-unit
predictor. Independent read-only review found no material algebra, role-access or
parent/RNG defect. Its requested positive-alpha fixture selects alpha1,
intercept7/coefficient1; duplicating F rows preserves every solution/G score,
checking the mean-loss penalty convention. A separate RED test exposed silent
overflow in finite-but-extreme F geometry; the helper now rejects nonfinite
intermediates before selection. Final headroom/boundary verification passed11
in8.04s; initial3 missing-module failures, first9-test pass and overflow RED are
retained under the main checkout's artifacts/research-v2/WP03/headroom-tests/.
No diagnostic fits or Gate A evidence were produced. Controller persistence,
full enclosing cost accounting and the M2 dispatcher remain pending.

### WP03 nested-prefix observer and checkpoint profile

Implemented capture-schema2, fixed-budget G-best prefix callbacks and a durable
prefix sink. All three initial prefix tests were RED. Independent review exposed
two artifact-level defects: accepting a later state under an earlier prefix and
same-object replay after interrupted record publication. Both reproduced RED,
then were fixed with semantic trajectory binding, atomic immutable records and
disk-first replay. The reviewer found no remaining material process-recovery
issue; its directory-fsync hardening was also applied. Final integration passed51
in27.83s; the final durability-only rerun result is recorded in the retained log.
Six logs remain under artifacts/research-v2/WP03/prefix-tests/ in the main checkout.
No M2 trajectory or Gate A outcome has been produced yet.

A checkpoint-only profile under the live MAIN5c55fa9 source compared ten
save/prune publications of the same selected parent with its8571 history events
versus an empty-history bookkeeping control. Elapsed was1.91556s versus.214942s;
canonical JSON recursion/encoding of history dominated this bounded profile.
The profiler never fitted a model or evaluated target labels. This is contended
diagnostic timing, not a formal efficiency result or proof of an achieved fix.
Input pointer/hash, source identity, pstats and outputs are retained under
artifacts/research-v2/WP03/checkpoint-profile/. Before larger matrices, implement
and verify append-only committed history references as specified next.

### First native qualification case completed

The MAIN5c55fa9 case completed successfully in5613.23429s (93.55minutes),
484895492 retained bytes and562511872 peak RSS bytes. Eight final roles and65
short probes completed. Every short probe selected stage0. Raw replay audit
verified500 matching unchanged keep updates and21 G evaluations. Initial manual
audit invocation passed a plain training dict and raised AttributeError; the
corrected SimpleNamespace call completed without changing the run or its inputs.
Independent NumPy reconstruction of all16 V/E MSEs from saved original-unit
predictions agreed within6.94e-18. Independent-arithmetic SHA256 is
f099d31789b4a5061cc905ca2a3e744e282dc923e134f2bb33f15572ffd075f8.

E NMSE: keep.0491281399, cheap.0489107267, random.0503192884,
short-selected.0501154130, warm upper.0491281399, fresh lower.0453586104,
fresh upper.0531436149, static partial.1015898980. The fresh lower ordinary
network also had the best V risk; cheap placement was worse than random on V.
This is exactly one teacher/repeat, exploratory development, not completed M0,
Gate A, strong-baseline evidence or a placement/generalization verdict. All raw
arrays/predictions/curves/models/measurements remain under
artifacts/research-v2/WP02/M0-native-qualification/. The old-source live-run
restriction is lifted; preserve its exact source for authentication/replay before
integrating newer source. Full measured source/run receipts are still required
before admitting larger matrices under changed code.

### Checkpoint journal implementation verified

Committed asbb08732, packageSHA256
f36bc2a49d6d866da39fafbef582c2472161334c6d9f5ef23bfa4155402ed191.
Native history now stores immutable canonical records behind an atomic persistent
tail and publishes committed JSONL prefix/count/SHA references. Full history checks
precede model reconstruction. Interrupted append tails remain retained; recovery
uses a new journal rather than truncating them. Retention checks only existing
versions against a validated, bounded32-directory publication-name cache, while
keeping protected/failed/unpublished files and all publication records.

Review findings each reproduced RED: partial append mutation, repeated full
prefix reads, expired-name rescans, unterminated publication tails. Repairs and
tests are detailed in history-storage.md and WP03-history-receipt.json. Broad
affected integration passed319 with one invocation-environment failure (missing
PYTORCH_ENABLE_MPS_FALLBACK=0); that unchanged diagnostic passed in10.37s under
the correct environment. Final retention/history/semantic integration passed106,
and final LF/cache history/checkpoint/cadence verification passed78 in24.39s.
This is not a monolithic green full-suite claim.

A fresh actual CPU receipt and the original numerical reference under final code
again produced exactly the same19 arrays and NPZ SHA8c072d32850a63e1512e0fa6858363fe3ab5d7df3300914db1e469799ff6bb77.
Equivalent8589-record history publication fixtures took.20427s journal versus
.97994s inline for ten saves/prunes. Ten retention checks with4000 synthetic
expired entries took.02737s including cold validation, previously1.38205s.
These are bounded, contended infrastructure diagnostics, not predictive/efficiency
claims. Full same-case native requalification remains next.

The exact old measured source is preserved in detached worktree
/private/tmp/atn-research-v2-measured-reference at5c55fa9. New implementation stays
in /private/tmp/atn-research-v2-next until integration; main's old live-run source
need no longer be held. M2-execution.md records the builder-reviewed finite DAG,
nominal/conditional counts, paired ablation reuse and bounded fallback/extension
rules before any M2 fitting. M2 config exists, but dispatcher/run remain pending.
native-tree-qualification.md records official package/source/wheel candidates and
API traps; no native tree adapter or quality anchor is qualified yet.

### Integrated source and same-case journal requalification

Main research-v2-continuation fast-forwarded to8f4e163 after the old run finished;
packageSHA remainsf36bc2a49d6d866da39fafbef582c2472161334c6d9f5ef23bfa4155402ed191.
New edits continue in isolated /private/tmp/atn-research-v2-next; do not change main
package source while the following run is active. The M0-journal-reference config
retains M0-reference's study/seed labels and every numerical setting, changing
only the actual backend receipt to research-v2-journal-qualified-cpu.json. Thus it
reruns the same first teacher/repetition, not a new independent teacher.

Run command from main, with all numerical thread variables1 and MPS fallback0:
qualified Python docs/research/v2/scripts/measure_reference.py --config
configs/research/v2/M0-journal-reference.json --output
artifacts/research-v2/WP02/M0-journal-qualification. Retain its result/status and
compare every original-unit V/E prediction and selected/executed update against
M0-native-qualification before calling it a compatible storage change. Full M0/M1
and all M2 efficacy work remain pending; this is an engineering replay. The old
qualification recipe/run was session54673, source5c55fa9, log
artifacts/research-v2/WP02/M0-native-qualification.log, actual CPU receipt
environment/research-v2-WP02-cpu.json; it completed normally, and its source lives
in the detached measured-reference worktree. Do not relaunch either output root.

### WP03 finite M2 planning

Implemented plan expansion only:1176 initial nominal procedures across M2a–e,
288 conditional extension procedures,320 conditional fallback cells and104 total
prerequisite records including conditional parents. Exact counts, paired seed
namespaces, nominal small subsets and F-bound candidate slots are retained.
Five recipes are fully resolved, late/current snapshot policies explicit, and
spec/M2 contract hashes plus structured gate/selection rules bind the plan.

Independent builder review found five material omissions in the first artifact:
incomplete anchor validation, unresolved ordinary recipe names, missing admission
bindings, incomplete machine-readable selection settings and a premature paid-cost
assertion. The three corresponding failing tests reproduced these omissions;
all were fixed, and final static review found no remaining material planning issue.
Five final plan/public CLI tests passed in3.41s. The combined plan/native-runner
integration then passed25 in15.33s. Logs are under artifacts/research-v2/WP03/plan-tests/;
M2-plan-candidate.json is a development plan artifact with execution_implemented
false and no qualified M2 backend receipt. No M2 fitting or Gate A result exists.
Public run explicitly rejects before creating an output until the controller and
source-bound resource admission are implemented; plan support does not claim run
capability. Further code stays in isolated worktree while main's same-case native
qualification runs as session49572 under unchanged sourcef36bc2a4.

### Same-case native storage equivalence completed

Session49572 completed normally in981.287996666s, with484993527 retained bytes and
538722304 peak RSS bytes. The old case took5613.234290042s, retained484895492
bytes, and recorded562511872 peak RSS bytes. These are contended single-case
observations; retained storage is approximately equal. The original receipt's
transaction estimates were2606368 and193266; their journal omission is corrected
below. Conservative reconstructed bounds are2606368 and2606902 bytes.

Correction before any production resource admission: that recorded transaction
estimator counted tensors/metadata but omitted complete history journals created
in a new export directory. It therefore cannot establish reduced maximum disk
headroom. Original receipts remain unchanged; WP03-native-equivalence-receipt.json
adds independently reconstructed conservative bounds including journals. Both
measurement recipes now include the largest complete journal as well as payload,
metadata and publication allowance. Numerical equivalence/timing observations
are unaffected. The old193266 estimate must not qualify journal-run capacity.

Independent raw readers verified20 observation arrays,16 V/E prediction vectors
plus4 reference arrays,18 ScoreTable arrays,77 native stages,40375 ordered
updates,1705 G evaluations, all allocations/selections, and162 byte-identical
authenticated tensor payload pairs covering8488 leaves. Both pure replay checks
verified500 updates and21 G evaluations. Final audit has zero failures/mismatches.
The first audit harness's terminal-event/metadata assumptions failed; attempt-0
is retained. Its correction changed only the audit reader, never training data
or source. Result SHA256:
4beeda33b62bf7e04107d986b7d81916699336f927a26b403e3edc578b6bef5a.

WP03-native-equivalence-receipt.json retains the disposition and hashes. The two
tracked scripts/audit_native_storage.py and scripts/audit_native_replay.py are
byte-identical copies of the executed readers; follow the receipt's fresh
sibling-directory copy instructions rather than executing these copies in place.
No additional independent teacher, protected access, Gate A result or predictive
benefit follows. Main source need no longer be held for the audit; implementation
continues in the isolated worktree before source-bound qualification/integration.

### WP07 optional tree dependency and import qualification

The isolated tree overlay installed all12 hash-pinned wheels offline while
preserving27 core distribution identities. Independent audits verified2921
overlay RECORD entries before/after. Fresh one-thread CatBoost1.2.10,
XGBoost3.4.1 and LightGBM4.7.0 imports passed without importing Torch. XGBoost and
LightGBM load both sklearn's bundled and Homebrew's LLVM OpenMP copies; imports
showed no crash/warning, and both threadpool limits were1. This observation
requires bounded fitting fixtures, not an invented incompatibility verdict.

tree-lock-resolution.md and tree-import-qualification.md retain versions, source
pins, wheel/native-library hashes, acquisition/auditor failures, offline install
command and actual import receipts under artifacts/research-v2/WP07/tree-imports/.
Native API/category/stopping/save-load fixtures are next on synthetic F/G-only
data. No transfer fitting/search or quality anchor is qualified; Gate D remains
pending for every mandatory family.

### WP03 immutable diagnostic forks and candidate artifacts

Implemented diagnostics.py and opportunity_inputs.py worker primitives described
in opportunity-implementation.md. Four first native fork tests passed in5.11s.
Review found incorrect configured-ceiling provenance, an aliased parent identity
record, unscored eligibility presented as empty, and result-publication errors
masking training/interruption failures. Reporting/failure cases reproduced RED
and were fixed. The combined boundary/prefix/head/fork/input integration passed
28 in18.32s. A final synthetic sampled-pool test exercises the more-than-eight
eligible-candidate branch without extra fitting. WP03-forks-receipt.json retains
the scope, initial invocation errors, RED cases and immutable log hashes.

This does not implement the outer M2 queue, admission/recovery, local-to-finite
diagnostics, full cost accounting or Gate A analysis. M2 public run remains
disabled. Continue with those controller pieces; do not infer a scientific result
from worker acceptance. Main's former source-hold and raw comparison are finished;
new package code needs its own source/runtime qualification before production.

### WP03 parent prerequisite artifacts

Added parent_artifacts.py with explicit trained/unfitted and patience/fixed-budget
treatments, exact actual versus G-selected checkpoint loading, and unavailable
states distinct from corruption. Initial two tests passed in3.00s. Review exposed
the configured receipt being ignored in favor of a module global and missing
manifest validation on the unavailable path; both reproduced RED and were fixed.
The final static review found no material issue. Expanded integration passed31
with one new test-fixture omission (required G interval/patience); supplying those
declared fields made the unchanged-code targeted test pass in2.93s. This is a
targeted recovery, not a second monolithic32-pass claim. The added fixture also
proves planted prerequisites remain unfitted at update0 and receipt state is
restored on interruption. WP03-parent-receipt.json retains all logs and outcomes.
Outer M2 execution/admission/recovery and Gate A remain pending.

### Controlled-task native resource qualification prepared

Preparing the fixed M1 instance0/repeat0 gamma2/sigma.1 measurement, using the
existing five-role F/G-only control worker and a new source-bound CPU receipt.
M1-native-qualification.json changes only the backend receipt locator from the
full M1 config, retaining every study/seed/numerical setting. The recipe is
scripts/measure_positive_control.py; the destination must be fresh.

Before reuse, source inspection found the control worker's report-publication
error could mask a fit failure and skip restoring the caller's backend receipt.
A mocked full-disk publication after a numerical failure reproduced RED. The
failure path now retains the original exception and always restores the receipt;
the numerical training path is unchanged. Targeted plus runner integration passed
21 in13.57s. Logs control-publication-red/green are retained under WP03/parent-tests.
No planted fit or new control-sensitivity result exists yet at this preparation
entry. The full M1 matrix and Gate A remain pending regardless of this one cell.

### WP07 direct native tree fixtures completed

Eight fixed numeric/categorical cells,16 tiny fits across CatBoost/XGBoost and
constant/linear LightGBM passed. They checked F/G coverage, F-only encodings,
unseen categories, finite predictions, native stopping/best selection and exact
save/load predictions. Every opposed-signal stopping control kept round1 after
executing9; XGBoost's ordinary categorical cell reached128 rounds and is explicitly
not a quality anchor. Actual LightGBM linear leaves were inspected. Independent
metric reconstruction agreed within2.22e-16. All27 core and12 overlay identities
remained unchanged; observed dual LLVM runtime processes did not crash/warn.

The initial monitor launch failed on sandbox-denied process inspection. Review
found that failure could leave a child unmonitored and omit terminal receipts.
The original successful16-fit artifact bytes remain unchanged and bound to their
executed helper; hardened process ownership now reaps on monitor/interruption
errors and preserves the original exception with command/cost receipts. Workers
enforce source/dependency hashes before imports, and phase-aware runtime context
clarifies old auditor-only flags. Nine harmless tests passed, repeated by root.
No numerical refits were needed; numerical function ASTs were checked unchanged.

WP07-tree-receipt.json and tree-fixture-qualification.md retain exact script/lock/
artifact identities, independent checks and all failures. Production adapters,
full native eligible anchors, native-training interruption/recovery and broader
workloads remain pending; Gate D/full-suite comparisons are not qualified.

### First planted-control qualification completed and audited

Main b11bbff, package70f12afcd00edcd2c5b48850f5ae77f6b0512f97d979ca8a7191cd88c8f36515,
completed the predeclared M1 instance0/repeat0 gamma2/sigma.1 five-role case as
session7956. Actual source-bound backend receipt creation passed first; the
containing run took209.39096375s, retained26361012 bytes and recorded407666688
peak RSS bytes. Conservative checkpoint transaction bound including the complete
history journal was2070967 bytes. No formal timing/cost advantage is claimed.

All five roles completed; selected/executed continuation updates were keep
975/1475, top1800/2300, random3850/4350, known-ID/current-contrast975/1475 and
known-ID/true-contrast400/900. E NMSEs in that order were.05453974275,
.04185875327,.04230658200,.04149013687,.04168284494; sampled noise.03973173748.
All growth arms improved on ordinary continuation on this constructed cell;
cheap-versus-random improvement was only about1% on E. Oracles are privileged.
Cheap selected duplicate units0/1, random4/6, and both oracles0/2. Similar final
errors despite these differences motivate the already planned control-sensitivity
and small joint-pair tests; they do not establish population-wide interchangeability.

Independent raw NumPy/JSON audit passed282 checks plus32 exact-array checks,
zero failures, all ten V/E metrics exact to saved arithmetic. It authenticated
teacher coefficients/noiseless equations, FGVE seeds/IDs, F transforms, allocations
and selected/executed raw-stage counters without model restoration or fitting.
Result SHA9844302b817d4905a8a0d5dbc8b5ab1f472db484bdbcffd1c1b568ee91546969.
The teacher-before-observation ordering is established from the committed recipe
and flag, not independent filesystem timestamp evidence. Full disposition and
artifact hashes are retained in WP02-positive-control-receipt.json. The tracked
audit_positive_control.py is a byte-identical copy of the executed reader; follow
the receipt's fresh-directory replay/source instructions.

This is one diagnostic instance/repeat. It supplies neither a primary-population
placement win nor full M1/Gate A completion. No T access occurred. Main package
hold for this completed run is lifted; exact loading uses its recorded b11 source.
Further implementation continues in the isolated worktree until qualified.

### WP03 frozen-head retained model

Added head_artifacts.py and a real-parent roundtrip test. Hidden-feature head
predictions reconstruct exactly without refitting, with immutable parent/RNG.
Initial artifact+head tests passed6 in5.59s. A structured wrong-runtime payload
reproduced RED and was rejected after identity hardening. Review then found
parseable but contradictory six-solve selection evidence could load; missing
trials, invalid alpha and altered retained coefficients reproduced RED. The
loader now verifies the full finite solve population, declared G tie rule and
winning parameters. Final artifact+head tests passed6 in4.86s. All logs are bound
in WP03-head-artifact-receipt.json. Outer M2 controller/cost/admission/recovery and
Gate A remain pending; this is software acceptance only.

### Initial M2 queue graph and recoverable prerequisite ledger

The initial M2 inventory now compiles to2,472 immutable prerequisite/fit/evaluator
jobs representing exactly1,176 nominal procedures and8 G-only pair-oracle
evaluations. No fitting or public M2 execution is enabled by this implementation.
Independent review reproduced and helped close exact-inventory reuse, second
orphan-closure, unsafe-output publication and pair-oracle aggregation defects.
Every fix had a failing temporary-handler test before the correction. The final
queue/graph/input/plan integration passed22 in5.97s; all failures/recoveries remain
in WP03-queue-receipt.json. Ineligible nominal pairs stay visible; any eligible
pair failure makes the oracle missing. Source/runtime checks, one same-seed retry,
append-only paid failures and unknown worker-loss timing bounds remain explicit.
Actual handlers, conditional admissions, resource qualification and Gate A are
still pending. This is software evidence, not a placement result.

### Retained M2 data and independent worker/evaluator access

Exact native primary arrays and planted-control arrays now round-trip through
role-separated storage. The fit loader opens F/G arrays and validation features
only; V/E labels and all noiseless targets remain evaluator-only. Predeclared
planted coefficients are saved before observation generation. One tiny native
parent/fork integration uses those retained arrays and evaluates afterward. This
is software verification, not an additional scientific teacher or M2 outcome.
Independent review found retained-config/raw-ID and V-feature geometry gaps. All
four malformed-artifact cases reproduced RED and were fixed; final affected
integration passed22 in12.96s. Earlier fixture-import failure and recovery are
retained in WP03-data-artifact-receipt.json. Outer handlers, resource admission,
conditional execution and Gate A remain pending.

### Native M2 worker composition verified on tiny fixtures

Individual data/parent/score/binding/fork/head/oracle/evaluator handlers now work
through the immutable queue. This verifies state/data/prefix/model reuse and
separate G selection versus V/E evaluation; it adds no scientific teacher. A
final affected suite passed28 in9.25s. Independent review found final model/data
binding and unauthenticated altered-plan cost accounting gaps; both reproduced
RED and were fixed, alongside explicit frozen-job and numerical-domain checks.
WP03-worker-receipt.json retains every test result. Local calibration, conditional
admission, measured resource forecasts and Gate A remain pending.

### WP07 native tree wrappers integrated

Reviewed P08–P11 wrappers landed from84e1030 (main integration831536e). The builder
retained46 tiny engineering fits across failed/initial/final attempts; the final
eight-cell parity run used20 fits and matched direct official/save-load
predictions exactly. Its scoped73 checks passed, with all material review fixes
proved RED→GREEN. Main integration's import-isolated contract checks passed32
in0.43s. All403 raw builder evidence files were copied byte-for-byte into the
main artifact namespace with zero mismatch; original source identities remain
unchanged. Full native anchors, hard resource harness/recovery, remaining families
and Gate D are still pending. No prepared transfer data or protected labels were
used, and this does not establish practical superiority.

### Independent local-score calibration implemented

F-only CPU-float64 autograd, separate1e-7 fresh-Adam gate steps, mask changes and
64 fixed-seed diagnostic minibatches now measure the legacy score's precise
first-step target. Full gradients, row indices, signed full-F predictions and
undefined ranks are retained without G/V/E-based selection. A zero eligible pool
is explicitly insensitive. Independent review found a caller no-grad failure;
it reproduced RED and the helper now enables/restores autograd explicitly. Final
local/worker/data/graph/queue/native-contract integration passed62 in10.16s. No
new scientific teacher or primary fit is counted by these fixtures. Optional
single-eligible/all-tied rank fixtures and a redundant precision field remain
deferred, with CPU float64 stated in code/docs. The receipt preserves every RED
and GREEN result. Production dispatch/resources, conditional execution and Gate A
remain pending.

### Fixed-source M1 continuation qualification started

An isolated detached checkout at132b250 now protects the control experiment from
implementation edits. A newly produced source-bound CPU receipt qualified its
runtime, and the existing measurement recipe is repeating the exact fixed
instance0/repeat0 gamma2/sigma.1 case under that source. This is source/resource
qualification of an already inspected teacher, not a new independent result.
The active output is WP02/M1-source-132b250-positive. One heavy numerical worker
uses one thread; root continues implementation in a different checkout. Larger
control workload measurement and a full resource forecast still precede any
full M1 matrix. No protected data have been created or opened.

### Full-size negative-control resource recipe prepared

The next qualification is fixed before outcomes: M1 globally_affine teacher0,
repeat0, keep/top/random. scripts/measure_negative_control.py resolves that exact
case, retains raw arrays/state/predictions and produces separate parent and
continuation measurements with complete-journal transaction bounds. Selection
and nested-timer arithmetic had2 RED cases before the recipe existed and2 GREEN
checks afterward. No negative-control fit has run yet. It will use the same
isolated132b250 source/runtime as the positive qualification; its independently
recorded script hash identifies the new measurement recipe. One heavy worker
means it starts only after the positive case finishes. Full M1 admission remains
dependent on both workloads' actual resource evidence and a free-space forecast.

### Fixed-source positive qualification completed; full-size negative case active

The132b250 positive case completed in203.405494333s with410,632,192-byte peak
RSS,26,364,620 retained bytes and a complete-journal checkpoint bound recorded
in its measurement. Independent raw comparison found52 arrays byte-identical to
the prior b11bbff case (observations, predictions and full score table), with
identical selected/executed counters for all five roles. This repeats the same
teacher and adds zero independent scientific problems. The first comparison
invoked system Python without NumPy; the unchanged raw comparison passed under
the qualified interpreter and retains that harness error in its receipt.

The fixed globally-affine teacher0/repeat0 keep/top/random qualification then
started under the same isolated source/runtime. Active session77008 writes
WP02/M1-source-132b250-negative; root implementation remains separate. The exact
argv/cwd/environment, receipt hashes and pending resource-admission state are
in WP02-M1-source-132b250-receipt.json. Full M1 has not been admitted, Gate A is
still pending, and no protected data have been accessed.

### Full-size negative qualification completed

The132b250 globally-affine case completed in68.645324625s,436,633,600-byte
peak RSS and28,720,835 retained bytes; its transaction bound is983,263 bytes.
The parent selected update1250 after1750 executed. Keep/top/random each executed
500 continuation updates and selected stage0, with identical E NMSE
0.012289263582213952 (V0.013025476424453803). This is one exploratory negative
control, not proof of calibrated no-op behavior across populations. It produced
both full-size parent and continuation workload measurements. Full M1 admission
is still pending a complete forecast and audit of remaining-space behavior on
resume; no production matrix has started. Current raw results/measurement hashes
are included in WP02-M1-source-132b250-receipt.json.

### Remaining-capacity recovery defect fixed before full M1

Review confirmed that the old research validator re-required its whole original
storage forecast before reading the resume ledger. New immutable per-transaction
allocations now credit only whole terminal groups under the authenticated lease,
with full unfinished/rerun allocations and a rolling reserve only for untouched
pending groups. The initial admission is pinned by content/raw SHA; separate
source/plan/admission/event-hash readiness records retain success and insufficient
space without changing paid attempts. Additional review found admission
replacement and double-reserved retries; all defects reproduced RED before fixes.
Seven focused checks and the45-test affected integration passed (15.01s for the
latter). resource-recovery.md and WP02-resource-recovery-receipt.json specify the
contract and all evidence. Training algebra/optimizer/stopping are unchanged.
No production admission had been issued. The132b250 qualifications remain under
their actual source; final-source backend/resource qualification is required
before full M1. No protected data or production matrix have been accessed.

### Full fixed M1 admitted and launched under604cffb

Final-source positive/negative qualification completed in209.299543875s and
74.719578625s. Positive raw52-array/five-role counter comparison with132b250 is
exact; no new independent teacher. Source package SHA256 is
5095a5d3bfc8d58bc863e0334550f122378664663fe265533e0441713c7216c4.
The fixed plan is cfe48a305c3c721eae7afe058ffe9af8f27c00dd7321be63359a3999abd6859d.
Independent admission review verified64 transactions/256 roles, source/runtime,
three profile hashes and complete arithmetic. Actual live admission passed with
16,940,695,552 free bytes versus15,782,326,102 additional forecast bytes and
536,870,912 free floor. Forecast time14.01/18.94h is an uncalibrated stage-speed
projection, assuming all update ceilings; storage includes a rolling retry.

The reviewed proposal was copied byte-exact, exclusively, to
artifacts/research-v2/WP02/M1-604cffb-admission.json. One worker is active via
session10775, cwd /private/tmp/atn-research-v2-M1-run, detached604cffb. Its config is
configs/research/v2/M1-source-604cffb.json in that worktree, with qualified receipt
environment/research-v2-control-604cffb-cpu.json. Source/locks must stay unchanged.
Full argv/environment/pid/log/terminal status live under
artifacts/research-v2/WP02/M1-604cffb-launch. Scientific output is
artifacts/research-v2/M1-controls-604cffb; read its immutable ledger for progress.
If the process stops, inspect failure and lease before the same CLI with --resume;
do not create a different seed/config or overwrite admission. Both source config
and full plan are bound in the retained qualification/forecast assets. The
tracked WP02-M1-source-604cffb-receipt.json summarizes their hashes and tests.

The new forecast recipe had2 RED tests before implementation and4 GREEN checks
including the existing negative measurement recipe. Final invocation initially
named a nonexistent negative test module; no tests ran. Corrected unchanged test
selection passed4 in0.05s; all logs retained. A documentation updater initially
looked for an execution-history heading in the shorter status file and stopped;
no scientific files changed. Corrected explicit append completed. Full M1/Gate A
and efficacy remain pending. Root continues independent M2 implementation; no T
creation/access.

### Bounded M2 conditional graph and reviewed fallback gates

The bounded graph now precommits3,853 jobs/1,784 nominal procedures, including all
initial slots,16 fallback parents,320 fallback fork slots, and the single288-fit
primary extension. Initial graph remains available unchanged. Conditional
extension worker, measured production dispatch/resources and final Gate A are
still unimplemented; public M2 run stays disabled. No M2 fitting occurred.

Missing-snapshot gates now execute through the frozen queue worker. Available
originals suppress fallback coordinates; original numerical/dependency failures
remain missing. Independent review found that an internally contradictory report
could authorize fallback despite reaching its claimed missing update. A concrete
8000-executed/missing1000 case reproduced RED. The correction verifies the raw
committed capture, requested/captured/unavailable partition and actual execution
count before admitting training. Changed commitments and inconsistent partitions
also reject. Final affected tests passed40 in13.61s under one-thread treatment;
initial graph/gate RED, worker integration RED, and review RED are retained.

Review also resolved primary-only extension dependencies and capacity-arm scope.
M2-execution.md now hash-binds the pre-outcome contrast/reference/bootstrap and
original/fallback acquisition policy. Early/new-stream controls have limited
legacy recipe scope; warm-only opportunity supports ordinary capacity. M1
true-contrast sensitivity cannot be replaced by the M2d G-selected pair oracle.
The machine-readable WP03-bounded-graph-receipt.json retains test/review details.
Full M1 continues independently in detached604cffb; root source edits do not
modify that scientific run.

### V-only screening, fixed references and bounded extension implemented

The source-resolved M2 plan now includes24 fixed growth/capacity contrasts and
complete numerical decision policy. Opportunity summary averages R2 before
V-reference family choice, preserves one original-or-required-fallback acquisition
for every matched coordinate, keeps all fixed teachers, and reports ordinary
capacity separately from top/random growth. Only late/continued has the five
ordinary-recipe reference family; early/new-stream recipe scope stays limited.
The extension worker uses only the448 declared primary evaluator inputs, never E.
The separate M1 helper enforces gamma2/sigma.1 true-contrast oracle/random across
four instances/R2; it cannot borrow the M2d trained-pair oracle.

Independent review found an inclusive2% floating-point boundary affecting both
counts and interval-trigger endpoints, plus misleading extension_required after
all eight teachers. All facets reproduced RED before correction. Counts now use
the risk inequality; interval endpoints use the same bootstrap draws on
risk-space differences centered at zero/margin, retaining both centered and
original descriptive intervals. The eight-teacher summary marks exhaustion and
never requests another extension. No new utility threshold or tolerance was added.
Final affected suite passed55 in15.08s; full repo suite is not claimed.

Earlier first-GREEN attempts exposed two test-fixture boundaries: a numpy scalar
instead of serialized JSONfloat and a manually rounded.93 versus the actual
1-.07 input. Both fixtures were corrected at the native boundary without weakening
validation/tolerances or changing production arithmetic. All original RED/GREEN
logs are retained by WP03-opportunity-analysis-receipt.json. Production M2 dispatch,
measured resources and final Gate A still remain; no M2 scientific fit, trained
selector, modern quality comparison or protected-data access is claimed.

### Reviewed MLP-D/ResNet-D wrappers integrated

Builder a56ae089 integrated as c9d7a7d. The ordinary-neural adapter pins the named
MLP-D/ResNet-D anchors, exact author quantile/layout arithmetic, categorical
unknown handling and distinct strict-G-best versus relative-patience semantics.
F-only target scaling is explicitly a protocol adaptation from the original
wrapper. All three material review findings (exceptional anchor/search support,
actual native-image reload authentication, negative-risk/fallback-state binding)
and the exact source-defined Skorch subtraction boundary reproduced RED and
were corrected. Builder scoped verification passed115 in88.71s, including four
fresh tiny native-reference/reload parity cases and all15 mutation checks per
case. These three-epoch cases are quality-ineligible.

Root integration passed47 contract/training checks in2.31s. All652 regular raw
builder evidence files (70,190,929 bytes including final inventory) copied
byte-exact, zero mismatches. The first copy attempt stopped before writing on
pytest's62 internal absolute navigation aliases. Recovery copied regular files
and retained an explicit alias map without following them; no scientific payload
was omitted. The failed initial test redirection consequently ran no test, then
the scoped command ran normally. Original builder package identity remains
44f94a1243b8901c2700a93a6cc590dc6098638e73430ecff4ea5596be13f194;
integrated package is f20c7b393f90aa83d06456022e7c772b1ea9c3698341d35f8c1e43632405236b.
See ordinary-neural-integration-receipt.json for raw copy/test hashes.

Full native anchors, hard worker time/memory enforcement, exact training recovery,
HPO and Gate D remain pending. No transfer predictive fitting, protected labels
or shared dependency mutation occurred. The parallel native work now begins
TabM recipe/API contract qualification in another isolated worktree before any
implementation. M1 remains pinned to its separate604cffb source throughout.

### Independent M1 raw-prediction arithmetic helper ready

control_audit.audit_control_study now reads an externally committed M1 plan and
authenticated event ledger under a read-only shared lease; an active worker
blocks the audit. It checks the selected native transaction/pointer FILE hash,
row/target alignment and original F variance, then independently recomputes seven
V/E scalars with stdlib accumulation. All missing/ineligible/failed roles remain
explicit. Standalone sums are not reported as physical cohort cost. The helper
does not regenerate teachers or rerun/validate model tensor predictions or the
checkpoint metadata chain; its evidence level is retained raw prediction
arithmetic. Invoking run receipts must separately bind the auditor source.
The active604cffb cohort has not been audited or interpreted with this helper.

Independent review reproduced a material row-identity defect: equal but short,
duplicate, matrix or float ID arrays passed alignment. All four cases went RED
before per-side shape/length/type/uniqueness and exact dtype/equality checks.
Final affected suite passed42 in12.83s with explicit intra/interop1. An initial
integration command omitted interop setup and correctly failed one backend guard
(41 passed); the unchanged correctly configured harness passed. All logs retained
in WP02-control-audit-receipt.json. Two minor scope notes are deferred: extreme
finite-mean accumulation overflow outside bounded M1 and helper-level auditor
identity/explicit metadata-chain flag (the invoking manifest and stated evidence
level carry that scope). No production result or full-suite pass is claimed.

### Interim completed M1 cells; no change to the attempt list

A frozen165-event prefix reaggregated the first two fully completed40-fit cells:
gamma.5/sigma0 and gamma.5/sigma.1, each4base instances/R2/five roles. Its canonical
hash chain verified. Mean V/E values and exact prefix/raw record hashes are in
M1-interim-cells-receipt.json. No production raw audit ran while the worker lease
was active; these are descriptive event arithmetic only. The noiseless setting
shows much lower error from growth and especially privileged generating IDs with
current contrasts; adding noise narrows the gap. Automatic cheap selection remains
unresolved versus random. These coordinates do not substitute for the frozen
M1 gamma2/sigma.1 sensitivity rule or M2/Gate A. The same four base instances recur
across coordinates; nothing is counted as8 independent teachers. No seed, method,
capacity, schedule, budget or remaining attempt changed after inspection. The
full pinned604cffb cohort continues; no T access or practical baseline result.

Gate A pure decision partial implementation (baseae2d298): independent material
review reproduced three contradictions RED→GREEN. The final affected suite passed
54 tests in6.75s. Both control and opportunity summaries are reconstructed from
retained risks; the frozen extension node must agree. No actual M2 gate decision
or admission exists. Production authenticated collection/resources remain next;
[receipt](WP03-gate-A-receipt.json) retains failures and the exact scope.

WP03 authenticated collection partial implementation: full frozen3853-job graph
binding and read-only raw file/status collection are fixture-tested. Paid failure
histories survive successful retries, including their safe raw inventories.
Worker-loss upper bounds stay explicit in cumulative charged time. Review
findings reproduced RED→GREEN; final affected suite55passed in7.86s. M2 producer
metrics are not an independent arithmetic audit. Public M2 dispatch, measured
admission and actual Gate A remain pending; see
[WP03-collection-receipt.json](WP03-collection-receipt.json).

P07 TabM engineering implementation integrated from reviewedf5378d5 as e89f7c6.
Builder107checks passed in84.38s, including4tiny official-code parity fixtures,
exact retained predictions/history/row orders/reloads,14mutation rejections per
case, and the reviewed pre-import bytecode-policy correction. Copied641regular
evidence files19,734,430bytes and the byte-identical private overlay into the main
workspace; no core install. Root57contract/collection checks passed in6.29s,
recorded in [tabm-integration-receipt.json](tabm-integration-receipt.json). Builder
and integrated whole-package source identities remain distinct. Full native
quality anchors, workload/deadline/recovery qualification, HPO and Gate D remain
pending. These checks add zero independent teachers or predictive benchmark results.

WP03 resource preparation now enumerates53 exact shape/treatment classes for
all3853jobs/10,908,000 nominal+conditional update ceilings. Pure forecast and
remaining-storage arithmetic passed41affected checks in9.46s after a reviewed
retry-reserve contradiction was reproduced RED→GREEN. These tests use software
profiles; no native profile, actual forecast or production admission is claimed.
See [M2-resource-preparation.md](M2-resource-preparation.md). Next is native
profile collection/authentication and production dispatch; no Gate A yet.

M1 final development-table helper is prepared, not executed on the live cohort.
It validates all256slots and audited risks, averages R2 within each of four base
instances, labels privileged oracles, and preserves incomplete cells/undefined
log effects. Analytic28checks passed in0.24s after8missing-API RED failures. See
[M1-final-analysis.md](M1-final-analysis.md). The live worker's lease still blocks
raw audit; no full-cohort result or Gate A is being claimed.

Math/software scope note:32fixed-count nonzero-gate relabeling fixtures passed
forward and all mapped gradient checks (max8.88e-16,1,697parameters each). Zero
training updates/independent teachers; no fixed-parent allocation equivalence or
optimizer trajectory claim. [fixed-count-capacity.md](fixed-count-capacity.md)
explains why finite learning/regularization/cost, rather than a different
unrestricted family at the same count, is the placement estimand. No gate changed.
