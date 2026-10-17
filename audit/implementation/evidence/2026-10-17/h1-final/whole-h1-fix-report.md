# Combined final H1 implementation fix report

Base: `b26931fda9a60ead929850e4589183e690de7e1d`, current `main` only.
This is the single combined implementation fix wave for final-review I1, I2,
and M3. No branch/worktree switch, push, genuine variance pilot, freeze,
scientific protected T access, confirmatory cohort, predictive benchmark, or optional
H3 work was performed. Isolated tests exercise toy freeze/protected-access/cohort
machinery without creating scientific evidence. Historical artifacts/configs/raw/receipts retain their
original identities. Final verification and commit identities are recorded below.

## Original finding dispositions

**I1 corrected:** the coordinator completes `record['repetition']`, republishes
the complete actual pair record, and only then inserts that same record into
the attempt and accepted ledgers. The strict protocol equality check is
unchanged. Failed/interrupted attempts still remain failures, and publication
failure cannot enter an accepted batch. A real one-teacher/two-repetition
engineering run now compares actual `pair.json`, attempt records and accepted
records for both repetition0 and repetition1. No manually repaired pair
fixture is used. The authentic RED showed only the missing repetition key;
73 ledger/advisory/runtime tests later passed together.

**I2 corrected within a finite measured CPU envelope:** the existing MPS and
physical-divisor diagnostic remains separate. The added
`treeneuron.cpu_envelope.probe_cpu_envelope()` produces source/lock/runtime
bound CPU operation and resource evidence for F577/2306/9224 and logical
batches128/256/512/1024/2048/4096/8192/16384. Requested32768 is a diagnostic
saturation point only. Logical equals physical; no accumulation was added.
Training, causal evidence, artifact metadata, checkpoint reload and prediction
boundaries consume the same receipt and actual-row contract. Larger logical
requests saturate at the actual F size; prediction chunks also obey F.
Other dimension, width, depth, optimizer, fallback and coordinate limits
remain bounded. See the exact APIs/schema/derivation in
`docs/research/2026-10-17-h1-cpu-envelope.md`.

**M3 corrected:** both the continuous optimum and discrete variance-per-cost
objective charge `seconds_per_repetition + n_T*seconds_per_evaluation_row`.
A material-evaluation regression changes the advisory recommendation from
more than100 repetitions to2. Actual selected repetitions, forecast accounting,
count floor and the requirement for new calibration remain unchanged.

**M1/M2 deferred explicitly by root Ruling14:** H1's fitted cohort excludes NODE;
the short NODE anchor remains quality-ineligible. Before long NODE runs inform
quality/cost policy, replace visit-proportional tracing with bounded coverage/
order evidence, or explicitly disclose and measure its overhead. Vendor/tiny
fixture warnings remain visible; future handling must be local category/message
assertions rather than global suppression or vendor-math modernization. No
practical-quality or efficiency claim follows from this fix or its tests.
Root-owned `audit/implementation/evidence/2026-10-17/ruling-14-conditional-node-instrumentation.md`
was left unstaged.

## Finite envelope and why it stops

The initial source-fixed CPU engineering exploration used F9224,d16,h64,m0/16,
one warmup and three measured optimizer epochs. Requested128→2048 kept
improving: base8.89→1.01ms and grown16.59→2.73ms. That boundary was reported to
root instead of called a plateau. Root authorized four further bounded probes.
Requested4096/8192/16384/32768 gave base.83/.65/.50/.50ms and
grown2.36/2.15/1.85/1.82ms. The last two execute the identical actual9224-row,
one-update operation. Root approved128..16384, with32768 retained as redundant
saturation evidence. Original raw samples/resources and scripts remain in
`whole-h1-fix-evidence/logical-batch-exploration*`.

The final producer has40 numerical cells (five representative topologies ×eight
logical batches),162 timing cells (three F sizes ×nine requested batches ×six
phases), and24 CPU64 scoring-parity checks. Representative topologies are
parent64, grown16, ordinary95, widened64→95, and centered static16. Derived
ordinary/widened widths64..95 and effective counts0..16 retain actual-topology
causal checks at live fitting/reload boundaries. Each timing cell records
requested/actual/tail rows, updates, one discarded warmup, five raw samples,
median throughput and process peak RSS. It does not silently substitute
physical divisors of fixed logical128 for a logical optimizer-batch sweep.

Timing scope is explicit: model reconstruction, optimizer construction and
one F epoch, or CPU64 model copy plus scoring. It excludes checkpoint IO,
G selection and the full paired graph. It does not choose a study recipe.
The later common-recipe calibration must choose one common batch and stopping
policy from actual measured fitting work. No universal throughput optimum or
unloaded-laptop performance is claimed. Native read-only resource observation
found normal pressure,36GiB RAM,81% coarse free percentage, AC100%, about3GiB
swap used and about62GiB disk available; an unrelated CPU-intensive process
remained active and was neither stopped nor modified. Sandbox counters that
were unavailable remain explicitly unavailable in earlier exploratory output.

## Diagnosed numerical-reference correction

An added CPU64 derivative comparison initially failed for widened95 with
requested16384 rows. The exact retained tensors exposed one ReLU mask change
at row1417/unit66: z64=−1.0459109800464716e−8 and z32=+3.725290298461914e−9.
Forward prediction normalized error was.0173; added-weight derivative error
was10.67. Clipping those exact tensors to reachable9224 rows preserved the
mask difference and increased the derivative error to18.947. The independent
float32 per-row reference also failed on those same9224 rows (normalized
added-weight error24.94). All unconditional failures and tensors remain saved;
no seed substitution, tolerance increase or relabelling erased them.

Root authorized a reference-only branch-conditioned remedy. For ordinary
and widened affine rows, an independently derived gamma_(2d+2) interval,
including binary64-center error and an underflow allowance, must contain both
computed preactivations and zero. The independent reference then evaluates
the actual float32 branch VJP while retaining its own forward values. Forward,
loss and gradient tolerances are unchanged. Raw unconditional errors and all
ambiguous indices/margins/intervals remain in version2 causal evidence.
Wrong masks outside the interval, incorrect backward rules and incorrect
AdamW steps reject. Own-gradient first-AdamW formula and exact subsequent
optimizer replay remain mandatory. Gated/contrast-row mask ambiguities have
no affine-only waiver: without a composed interval they fail closed.
The training model/forward/backward, scorer, optimizer, input, seed and dtype
were not altered by this reference-only conditioning.

`tests/fixtures/h1_widened_relu_rounding.pt` preserves the original full16384
rows and exact model tensors. Regressions use its reachable9224-row prefix,
including the actual offending row, for both CPU32 and CPU64 references.
`failed-operation-diagnosis.jsonl`, `reachable-operation-diagnosis.jsonl`,
`retained-reference-diagnosis.json`, and `envelope-failed-1.json` retain the
original unsuccessful measurements and diagnosis. This is a correction of
an invalid derivative-equivalence oracle at a nondifferentiable point, not a
claim that unconditional gradients agreed.

## Runtime integration

The first authentic handoff test exposed a separate launcher mismatch: the
legacy fresh backend child reported default inter-op14 while the authorized
pilot launcher requires1. The backend child now explicitly sets inter-op1;
its RED and GREEN are retained. Old inter-op14 receipts remain unchanged.
The causal verification CLI also configures its owned serial inter-op runtime;
a real CLI regression exposed and verified that integration. Other in-process
callers continue to satisfy the explicit runtime contract.

Extended checkpoints store an exact receipt path/hash under `backend.cpu_envelope`.
Reload validates package source, lock, runtime, operation topology and live
CPU1/inter-op1/BLAS1/fallback0. Missing/modified receipts, unsupported shapes or
batches, incorrect tails, source/runtime drift and failed operation/loss/
resource evidence reject. Existing relocation translates the receipt path
without rewriting scientific bytes. New source makes old source-bound evidence
historical; none was overwritten or asserted to authorize the new envelope.

## Verification, commands, and retained failures

All numerical processes were serialized. Test processes used a fresh interpreter,
CPU1/inter-op1, all five BLAS environment variables1 and fallback0. Explicit
unique `/tmp/h1-final-fix-*` basetemps avoided shared pytest retention cleanup.
Every run held package source fixed. Raw logs live in `whole-h1-fix-evidence/`.

- `red-1.log`: M3's incorrect141 recommendation and I2's batch256 rejection;
  the first I1 attempt stopped at the independently diagnosed inter-op mismatch.
- `red-interop.log`: actual child inter-op14 versus required1.
- `red-handoff.log`: authentic missing repetition key; runtime regression GREEN.
- `red-envelope.log`: new producer API absent before implementation.
- `green-1.log`:78 passed;18 setup errors share the new unconditional CPU64
  widened-reference failure. This failed stage is not reported as a passing
  numerical envelope.
- `ledger-advisory-green.log`:73 passed in15.08s.
- `red-relu.log` / `green-relu.log`: retained failing-reference regressions,
  wrong-mask/backward rejection, and conditioned-Adam checks; initial GREEN5.
- `green-envelope-2.log`:22 passed;8 failures from a test fixture missing the
  required study_id, corrected in the test rather than weakening production.
- `green-envelope-3.log`: all8 actual high-density per-batch parent/growth/
  static/ordinary/widened fit-and-checkpoint paths passed in31.34s.
- `red-runtime.log`:4 REDs for live runtime and loss-evidence rejection.
- `cross-contract.log`:220 passed,1 CLI inter-op failure,1 known vendor warning.
- `cli-reference-green.log`:8 passed in15.79s after the owned CLI correction.
- `red-operation-identity.log`:5 REDs for causal schema/device/shapes/count/seed;
  the producer/consumer now validates those cells explicitly.
- `full-suite.log`: single complete suite; final outcome below.

The full-suite command is retained verbatim in `full-suite-command.sh`.
`suite-launch-manifest.json` records314 execution/test/config file hashes,
package SHA25675a705dc5a9be2cb248f58f75b781066922d2bd3d2ca5a3a6fdb7a63d1f6b091,
base Git identity and the command. Its own SHA256 is
90cb58d0895067995ecdd5995907d8210d9057f215e3074f8058772bc85eb421.
The final source comparison, native receipts/parents, full-suite result,
commit and execution identity are appended after completion.

## Complete-suite outcome and narrowly scoped follow-up

The single complete suite exited with **958 passed,48 skipped,2 failed,9 setup
errors,28 warnings in1662.40s**. It is not represented as a clean complete-suite
pass. `suite-exit-manifest-check.json` confirms all314 launch files remained
byte-identical throughout. The48 skips are optional coverage; the28 warnings
remain visible, including deferred NODE/vendor warnings.

Both failures were `test_receipt_io_survives_unavailable_original_root[True/False]`:
the test copied the historical `environment/evidence7-calibration-fix1/search-backend.json`
and the current producer correctly rejected its old implementation hash. The
regression now uses the existing actual fresh `cpu_training_evidence` fixture.
The old receipt and strict source validation remain unchanged. All9 setup errors
shared `test_search.py`'s actual search-backend fixture: its fresh launcher had
already configured inter-op1, and the new backend setter attempted a second
PyTorch assignment. The setter now acts only when the value differs from1;
a non1 runtime that cannot be configured still raises normally. No mathematical
or timing operation changed in this follow-up.

A separate narrow supported-policy mismatch RED is retained in `red-policy.log`:
a real batch4 checkpoint accepted stage-policy logical=physical256 although its
resolved config remained batch4. `validate_metadata` now requires equality before
reconstruction. No batch constants, receipt authority or tolerances were loosened.

Only `backend.py`, `train.py`, `test_audit_fix1.py`, and `test_checkpoint.py` changed
from the complete-suite snapshot. The final affected check is documented verbatim
in `affected-final-command.sh`; its launch manifest binds package SHA256
`5c66c55693825e7dec51f0e59c74bd995da98b54c36f34acc445039d9ef85495`.
It covers checkpoint/train/CPU-envelope/backend/search and the two authentic
relocation cases. There is no automatic second complete-suite run.

## Final affected/native evidence and commit

Final affected GREEN: **190 passed,47 optional skips in87.78s**.
`affected-final-exit-check.json` confirms all314 recorded files stayed unchanged.
Together this resolves the exact11 full-suite failure/error contexts and the
new batch-policy RED. The earlier complete suite remains explicitly a failed
source-specific stage; no claim of a second clean complete suite is made.

`native-final-command.sh` invokes `make-final-evidence.py` in a fresh authorized
native CPU1/inter-op1/BLAS1/fallback0 process. It passed40 numerical cells,
162 timed cells,24 scoring parity checks and8 fresh F9224 parent/checkpoint
reloads, one for every admitted logical batch. Each parent used one requested
update on random engineering data, with no scientific teacher/cohort/E/T.
Separate affected tests cover every batch with parent,selective growth,ordinary,
widened and static fit/checkpoint paths. All final native package bytes match
the affected launch manifest and committed package; `native-final-summary.json`
records no drift. Native receipts are immutable files in `native-final/`, and
`parents.json` records exact run/checkpoint locators and hashes.

Peak probe RSS was495,583,232 bytes. Final native requested16384/32768 medians
were base0.708/0.752ms,grown2.185/2.307ms,score21.434/21.331ms. Both requests
execute exactly9224 actual rows, including the same full-size tail. The final
ambient variation exceeds5% for base/grown, unlike the earlier bounded
exploration; all five raw samples per cell are retained. This does not establish
a universal optimum or a stable statistical plateau.32768 remains diagnostic
only because it adds no reachable rows or different optimizer operation.
Final resource observations reported normal pressure,81% coarse free,AC100%
and unchanged approximately3GiB swap used; resource counters are observations,
not additive independent memory allocations.

Commit on main: `dfed0d24fce9c53860b1742c31d74337c27cb6cb` — Fix H1 ledger handoff and validate finite CPU calibration envelope.

Strict `execution_snapshot` succeeded against that commit, index and working
tree, covering314 committed execution files. The only remaining untracked file
is root-owned Ruling14. `git diff --check` passed; baseline diffs for configs,
uv.lock,artifacts,environment and locks are empty. The exact regression tensor
fixture was explicitly committed despite the generic binary ignore rule.

- package_sha256: `5c66c55693825e7dec51f0e59c74bd995da98b54c36f34acc445039d9ef85495`
- execution_sha256: `ccb5447a6bec01c78b2448a8812bcc32d2ae4b64b8603cba72bfcf719ce3cdd5`
- kernel_producer_sha256: `6f702a34dda4b260fbef75d800ef183fcfeb1f716f266b8895bef50d07189401`
- lock_sha256: `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`
- runtime_sha256: `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68`
- regression_tensor_sha256: `437121d680aeb561772c8630776340b699201886c4b0a7a7119493f7667dfaf2`

Final receipt hashes:

- `native-final/backend.json`: `c2b12cb30d050da4ae9a1419c3f3e53d5179dfa810a9dd9bf0458ea7d850df30`
- `native-final/envelope.json`: `4834f28d877fdf132bee2d501a14795a3072531bbaa4d9829648be091c7e079d`
- `native-final/kernel.json`: `182a9309f902dd07d0dbf30cce21ca5b1a595dd167997ff6feb4d244b76fc0fa`
- `native-final/launch.json`: `58cd817e048b1ca9cee13824eb22afcae1ec92c3a94dce097b21091e69e1df11`
- `native-final/parents.json`: `3d152fa18a0940cb1095ebee23d340caede47d174deedb89a165aa94bd4e18fc`

`final-identities.json` and `final-execution-snapshot.json` contain the full
machine-readable identities. `hash-index.json` indexes the retained raw logs,
scripts, failed evidence, final receipts and engineering run/checkpoint files.
The source-specific full/affected command files and manifests let review
separate prior RED/GREEN stages from final applicable evidence.

Implementation work is complete for the scoped I1/I2/M3 fixes. This report does
not grant scientific calibration/freeze readiness or review acceptance. M1/M2
retain root Ruling14's explicit conditional obligations. The same reviewer's
one scoped re-review and root's residual adjudication remain the next handoff.
