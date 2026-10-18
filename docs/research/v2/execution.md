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
