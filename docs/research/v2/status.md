# What the research is trying to establish

Start with a small neural network. Give selected neurons a learned switch between
two affine rules, preserving predictions when that flexibility is inserted.
Then ask whether growth helps, whether a chooser beats random placement, and
whether the complete procedure is competitive with strong existing methods.

The present verdict is cautiously encouraging for useful capacity, disappointing
for the current chooser, and unresolved for practical competitiveness.

The first full new control study has finished:256 planned fits, no missing or
failed result. Independent arithmetic from512 retained V/E prediction vectors
agrees with the producer's3,584 scalar comparisons to2.22e-16. The
[complete M1 report](../../../reports/research-v2-M1.md) includes every coordinate,
negative control, oracle and limitation. This is development evidence; no
protected T was created or evaluated.

The positive tasks deliberately plant gated structure in an h8/count2 problem.
The same four base instances recur across four magnitude/noise settings, with
R2 averaged before instances. They are not16 independent problems. Mean E NMSE:

| Gate magnitude / noise | Ordinary continuation | Cheap growth | Random growth | True-information oracle |
|---|---:|---:|---:|---:|
| .5 / 0 | .00212917 | .0000174730 | .0000131730 | .0000000825200 |
| .5 / .1 | .04354531 | .04172836 | .04163478 | .04127422 |
| 2 / 0 | .01540438 | .000975103 | .000438801 | .0000588626 |
| 2 / .1 | .05534798 | .04429366 | .04209095 | .04149124 |

Growth learns useful structure here. At magnitude2/noise.1, random reduces error
about24% versus ordinary continuation, while cheap growth reduces it about20%.
Random has lower mean E error in every displayed coordinate. This comparison
does not establish superiority over matched ordinary extra capacity or tuned
models. The oracle uses privileged generating information.

The predeclared sensitivity check uses V: true-information placement improves
over random by1.54%,3.08%,9.95% and−0.94% across the four instances. Exactly2of4
exceed2%, meeting that control screen. A chooser has a measurable opportunity in
this constructed setting; our present chooser has not shown that it captures it.
One instance has about33% higher V error with cheap placement than with random.

Pure noise and shuffled-label controls select stage zero in all8 paired runs per
population; their saved V/E predictions equal keep exactly. Affine controls mostly
remain unchanged. The additive smooth control benefits from extra capacity on
average, with random better than cheap, which also motivates ordinary-capacity
controls. These small samples do not establish a general false-positive rate.

The earlier fresh main-family qualification case remains one teacher/repeat.
Exploratory E NMSE was .04912814 keep, .04891073 cheap, .05031929 random and
.04535861 for the best fresh ordinary bracket. Its independent engineering replay
matched predictions, scores, allocations,40,375 row updates and1,705 G events.
All65 short probes selected stage zero. A fresh ordinary model did best in that
case; neither the full M0 cohort nor a primary-population placement win follows.

The normal-path insertion and optimizer-history checks pass. A separate replay
finding matters: restoring the complete earlier G-best state can make unchanged
continuation repeat an already unsuccessful tail. That no-op does not prove
ordinary training has exhausted useful headroom. Changing timing or minibatch
streams therefore needs matched controls rather than assuming earlier growth helps.

M2 is running: matched timing/stream, ordinary-capacity/recipe and small pair-oracle
diagnostics. Native resource qualification passed all53workload classes; the
fixed M2 scientific queue is now running in an isolated source checkout with
live disk/memory pauses. Its full conservative storage projection is227GiB
against about12GiB free at admission, so eventual full completion is not promised.
Gate A remains pending. The
proposed complementary-group selector has not been trained or confirmed. At a
fixed growth count, unrestricted model families can be relabeled; the placement
question concerns learning from the current parent within finite training/cost.

Strong baseline adapters have passed tiny official-reference checks, and full
native-recipe execution is now being measured. A CatBoost anchor completed on one
main-family development dataset; all24tree anchors completed on the designated
primary transfer splits. These are execution/resource checks, with no V labels
or protected outcomes used. The modern predictive comparison remains unrun. Prospective
variance/power software is prepared on artificial fixtures; no real calibration
or protocol freeze has occurred.

The complete M1 run took about2h23m30s of contended wall time and retained1.55GiB.
One substantial training worker ran at a time. These are observations, not an
efficiency claim. [Execution ledger](execution.md), [reproduction](reproduce.md),
[binding spec](../../superpowers/specs/2026-10-18-research-v2-design.md) and
[implementation plan](../../superpowers/plans/2026-10-18-research-v2.md) retain the
exact state, evidence and next steps.

As of19October,12:32UTC, the main diagnostic has2,421eligible completed queue
jobs,496ineligible entries and one running fit in the preset extension. Queue
jobs include preparation/evaluation and are not independent problems. No
scientific failure has been recorded at that snapshot. About7.22GiB disk was
free; live guards may pause the unchanged queue. The last native memory-pressure
reading was level2, so nine extra legacy backend setup checks remain deferred
with their guard intact. No full-repository green result is claimed.

The benchmark infrastructure now checks paid anchor/failure/recovery accounting,
exact resumed TPE choices, late-result exclusion and own-process deadline/resource
outcomes. The independent raw-result auditor agrees with saved predictions from
48earlier engineering vectors. These are reliability improvements, not new
predictive successes. No modern-baseline benchmark or protected confirmation has
run, and the new joint chooser still awaits Gate A.

An immutable pre-calibration method identity is now fixture-tested. It fixes the
future analysis policy and its random seeds before calibration data, but grants
no calibration or protected-label access. No real calibration, final protocol
freeze or protected confirmation has occurred. The remaining readiness and
evaluation code must be completed before production searches or calibration.

A qualified storage pass compressed908published logs from finished jobs while
preserving their exact readable hashes and all recorded evidence. It reported
about1.28GiB less per-file allocation and left models/data/checkpoints unchanged.
The experiment continues under its original source and resource guards; this
changes storage representation and supplies no new predictive result.

The checkpoint-history storage pass also kept every original readable hash.
I then found and corrected a reader-concurrency gap: future history maintenance
requires the producer to be inactive. The completed pass had no recorded
scientific failure; all4,376maintained/verified file commitments matched afterward.
The study's source, predictions and decision rules have not changed.

The RealMLP comparison adapter now records the actual rows used during native
initialization. Its64-check engineering suite preserves official predictions,
saved-model reload and RNG parity while checking F-only row boundaries. This
improves baseline reliability; it supplies no new predictive result or native
quality qualification. Full practical searches have not begun.

The paid search queue now also runs real engineering workers, retaining failed
attempts, complete file hashes and exact clean-pause resume. Its148-check affected
suite passed. It reads no validation labels and admits no benchmark winner until
native qualification is complete. At that earlier snapshot the main diagnostic had reached the third small
pair-control problem. It has since finished those controls and entered its
predeclared four-to-eight-teacher extension; the first four did not settle its
screening rule. The complete scientific verdict remains pending.

One stronger baseline, RealMLP, has now completed its full256-epoch recipe through
the supervised pipeline on a small synthetic fixture. All300 training rows were
visited across the run, and saved-model reload reproduced predictions exactly.
Recorded worker memory stayed around459MiB. This verifies that fixture's recipe
execution; it is not a benchmark comparison or a result against our grown model.

The first full CatBoost worker run exposed a snapshot-format integration error
after training had finished. Four native regressions reproduced it; the corrected
worker passed87affected checks and one same-seed recovery completed10000rounds
with exact saved-model reload. Official inference on the original partial model
and recovered model agreed exactly. Both attempts and their costs are retained;
this fixes our harness and gives no predictive verdict. Those24anchor checks used primary training/stopping data and validation features only. See
[native anchor execution](native-tree-full-shape.md).

All24tree anchors completed their native recipes, reopened saved models exactly,
and passed file/primary-split/query-row audits. Full256-epoch RealMLP anchors
completed on all6datasets after one explicit larger-memory, same-seed recovery
each for CCPP and Bike. Both initial operational failures and paid costs remain
retained. The supervisor cleanup race exposed by Bike was reproduced and fixed.

The12MLP/ResNet and12TabM plain/embedded primary-anchor checks also completed
without timeout or retry. Their saved predictions reopened exactly and every F
row was covered each epoch. This makes54distinct primary native-anchor
completions from56attempts across the tree/ordinary/RealMLP/TabM families. These
are contended execution checks, with no V labels or T accessed, no comparative
accuracy claim and no production search. See
[native neural checks](native-neural-primary-anchors.md).

The proposed neurons' own practical adapter is being extended to the actual
transformed tabular shapes. Its configuration/model core passed109affected checks
with one unavailable-MPS skip, including exact gate insertion, inherited Adam
state, ordinary widening and independent static forward references. The in-memory stage engine now also passes154affected checks with one MPS skip:
it restores the whole selected training state and records original-unit G loss.
The complete adapter, closed saved-state handling, worker integration and measured
workload qualification are still pending. This work leaves the historical model guards
intact and does not admit the new joint chooser before Gate A.
