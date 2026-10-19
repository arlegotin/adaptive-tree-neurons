# Practical gated-family adapter implementation contract

This is a Task7 implementation addendum to the binding research-v2 spec/plan.
It does not admit a new selector or change any scientific gate. P00/P01/P02/P05
still lack production adapters; the code described here is to be implemented and
qualified before any practical search. GateA remains pending.

Keep the legacy training/checkpoint, static and widening guards intact. Their
finite d16/h64–95 receipts cannot certify categorical expansion, h128 or the
larger transfer F partitions. Add separate `research/gated_config.py`,
`gated_models.py` and `gated_native.py` with their own schema, source/runtime
checks, finite workload contract and engineering-ineligible execution mode.
Reuse the implemented `SelectiveRegressor`/`SelectiveAffineLayer`, unchanged
F-only scorer/allocation and absolute `UpdateSchedule`/RNG helpers. Extend model
wrappers for static/warm capacity without pretending the legacy bounded wrappers
were already qualified. The legacy guards must continue to reject their old
out-of-envelope requests.

P00 initially implements only the historical fresh-Adam top placement, P01 its
informed-direction random counterpart. Both score the same F-derived contrasts,
use the same eligibility/count, parent initialization and complete G-selected
parent state, and pay their complete standalone parent/scoring/continuation work.
No joint selector, new direction, stream restart, newborn treatment or pruning
branch is admitted here. A later measured gate must explicitly admit such a
strategy before its configuration becomes valid.

Use strict `research/v2-gated-native-config` records with family, mode,
feature_schema and parameters. P02 also requires a lower/upper `variant`; P05
requires centered_partial, centered_full, direct_partial or direct_full. Unknown
fields fail. Native anchors have no parameter overrides. Default parent width64,
growth fraction1/4, epsilon.05, tau1, AdamW LR.005/WD0/betas.9/.999/eps1e-8,
batch128, absolute constant schedule, G every25/patience20 and8000updates per
stage reproduce the historical treatment's semantic recipe. P05 has two ordinary
training stages with no growth, matching the historical fresh-control anchor.
This is no claim of historical array/platform replay or a quality-competitive
recipe before tuning.

Search support is the previously declared finite support: P00/P01/P02 parent
width64/128, fraction1/8,1/4,1/2; LR[1e-4,1e-2], WD0or[1e-7,1e-2],
constant/cosine and batch128/256. P05 widths32/64/128, the four static variants,
dropout0/.1/.2 and the same bounded optimizer/schedule choices; partial static
count is width/4, full count is width. P00/P01 receive identical support apart
from placement. Cosine uses the unchanged16000absolute-update horizon; restoring
G-best restores its matching schedule position. Engineering fixtures alone may
use width4/8 and explicitly small1–25-update stage/evaluation/patience limits.
None of these supports is selected from V outcomes; no production search has run.

Fit numerical means/scales, target normalization and categorical vocabularies
on F only. Reuse the P13 typed vocabulary/explicit unknown-column preprocessing
helpers, recording transformed dimension and exact state. There are no splines
in these models. Continuous columns remain standardized; categories become full
one-hot indicators including an unknown column. This differs from the native
embedding/quantile recipes of competing neural families, which retain their own
preprocessing. Initial engineering bounds are F/G≤20000, encoded dimension≤128,
parent width≤128 and resulting ordinary width≤256. The worker's raw-feature
bound is separate. Bounds grant no quality qualification without actual shape,
gradient, complete-row, resource, checkpoint and recovery evidence.

P02 grows an ordinary residual sum: keep the original parent parameter objects
and Adam slots, add independent ReLU affine units with a zero output head. The
exact lower/upper widths bracket the selected grown model's total parameter
count, using its actual common eligible count. Existing parameters and absolute
Adam/schedule ages remain unchanged; new tensors start at age0. The new module
must preserve predictions exactly at insertion. For d16/h64/m16 the parameter
totals remain1153/1697for parent/grown and1693/1711for widths94/95.

Static wrappers retain the implemented preactivation semantics: mix affine
leaves first, then ReLU. Centered and direct-leaf coordinates have distinct
trainable parameters/conditioning and native random initialization. A centered
zero gate is used for growth, not silently imposed on static initialization.
Optional static dropout follows ReLU. A zero-headed warm block and each static
parameterization require independent forward/gradient and parameter-count checks,
including encoded dimensions above16.

The training stage must select the strict earliest G-best complete state,
including optimizer, sampler/order/cursor, absolute scheduler, RNG and update
counters. Update0 is eligible. Restoring a chosen state must leave the outer paid
attempt/history log intact. Newborn birth/applied-update ages and nonnegative
Adam second moments are validated. Identical replay, gate/contrast learnability
and function preservation are separate tests. Record selected and executed
updates, all candidate contrasts/scores/eligibility, model/parameter identities,
gradient/participation evidence, row-stream commitments and costs. Retained
inference snapshots use closed JSON/NPZ with exact dtype/shape/source checks;
unsupported native training recovery must fail clearly and use only an admitted
same-seed operational rerun, never an implicit optimizer restart.

Implementation order is config and mathematical model checks, then stage/state
and F/G-only adapter checks, then closed snapshot/worker integration and measured
native workload qualification. This addendum makes implementation choices within
the authorized spec; it does not replace GateA/B/D, confirm a useful model or
authorize V/T access. Production integration remains incomplete until every
owning contract and the final-source qualification pass.


The configuration/model core and in-memory complete-state stage engine are now
implemented. The stage selects strict original-unit G MSE with float64 target
values, supports absolute constant/cosine schedules and restores model, Adam,
parameter births, complete RNG, sampler order/cursor/epoch and selected counters.
Full row-index update events remain external and are not rewound on selection.
One-process replay and corruption boundaries passed154affected checks with one
unavailable-MPS skip. This does not qualify persistent interruption recovery,
the complete adapter, closed inference snapshots or worker/native workloads.
See [stage evidence](WP07-gated-stage-receipt.json). Native search/calibration and
all scientific gates remain pending.
