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
|WP02|in progress|87 integration checks passed; native resource qualification, replay/stress controls and full M0/M1 still pending|
|WP03|pending|complete admitted M1/M2 evidence; Gate A|
|WP04|pending Gate A|joint selector numerical and trained evidence; Gate B|
|WP05|not admitted|measured geometry/newborn trigger|
|WP06|not admitted|measured retention/family ceiling; at most one redesign|
|WP07|pending WP00|official assets, sealed splits/native qualifications; Gate D|
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
