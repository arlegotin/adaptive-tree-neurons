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
|WP02|in progress|replay/stress/collision/tie foundations reviewed;35 focused passed; first native resource case running; full M0/M1/admission pending|
|WP03|state primitives in progress|actual-current boundary/fixed-budget control; recovery-plan review fix; M2 dispatcher/prefix/headroom and Gate A pending|
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
