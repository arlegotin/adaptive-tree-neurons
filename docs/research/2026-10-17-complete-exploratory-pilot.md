# Complete exploratory pilot: the placement score has no clear advantage

The planned diagnostic now contains **16 independently generated regression
problems, each trained twice**, all eight methods and all 2,080 finite greedy
probes. No scientific pair failed. Predictions were evaluated on independently
generated exploratory E observations. These are pilot results, not protected
confirmation or a tabular benchmark.

The core question is whether a cheap training-derived score chooses better
places for two affine specialists than random placement. The observed average
favors random placement:

| Method | Mean normalized E MSE |
|---|---:|
| Continue the ordinary parent | 0.0623326 |
| Score-guided growth | 0.0611880 |
| Random placement, matched capacity | 0.0606515 |
| Greedy validation-probed allocation | 0.0605594 |
| Ordinary function-preserving widening | 0.0606308 |
| Fresh ordinary lower parameter bracket | 0.0614538 |
| Fresh ordinary upper parameter bracket | 0.0604880 |
| Static partial gated model | 0.0892495 |

Define benefit as random risk minus guided risk; positive favors the score.
Its teacher-averaged value is **−0.000536435**. The original 8,000-draw
whole-teacher bootstrap interval is [−0.001176854, +0.000147291]. A prospectively
declared fixed 512,000-draw saved-data calculation gives
**[−0.001205030, +0.000156050]**. It first reproduced the actual 8,000-draw
calculation to within 2.17e−19, then used the same seed and endpoint method.
Student-t sensitivity is [−0.001298944, +0.000226073]. Repetitions are averaged
within teachers; neither rows nor optimizer seeds become independent problems.

The interval still spans regression and a small improvement. This does not
establish a positive allocation effect, equivalence, or a powered rejection of
every useful effect. Ordinary capacity is competitive in this recipe. The
static partial model was worse than its upper ordinary bracket, but this is
one shared causal recipe, not a tuned comparison of all gated architectures.

The larger saved bootstrap resolves computational endpoint precision: maximum
chunk endpoint SE is 0.0000168885, below the unchanged 0.0000510286 gate. It
does not add independent problems. The SD interval ratio remains 2.367, above
the required 2, and the intrinsic between-teacher variance lower bound remains
zero/unresolved. The original design branch stays `calibrate`; final test size,
powered cohort size and freeze eligibility remain unresolved. The empirically
chosen pilot cap is not proof that more computation is physically impossible.

The user-requested overnight pause caused the original producer's elapsed-time
timeout after 30 completed pairs. A prospective operational amendment completed
only the two never-started pairs, with identical data arrays, seeds, code and
recipe. The original failed raw ledger, seal, clocks and missing second canonical
runtime batch remain intact. The linked 32-pair diagnostic is explicitly
noncanonical and cannot authorize protected test access.

The measured producer footprint is 12.953 GB, with original peak RSS 3.667 GB.
Original elapsed time outside the documented suspension plus continuation
through saved diagnostics was 7,606.0 seconds, including I/O and waits; it is
not CPU utilization or a clean end-to-end speed benchmark. Original enclosing
and suspended clocks, failed setup wrappers, nested method costs and cumulative
ledger rewrites are retained separately.

[Linked original evidence and root checks](../../audit/implementation/evidence/2026-10-17/complete-linked-pilot/index.json)
bind all prescribed members and both original inventories. The
[saved-MC diagnostic](../../audit/implementation/evidence/2026-10-17/saved-mc-diagnostic/index.json)
retains both draw counts and the original blocked branch. Actual independent
raw-prediction recomputation, clean locked replay, interrupted-growth recovery
and a report rebuilt twice are the remaining WP09 checks. Strong practical
baselines, transfer datasets, deeper growth and hard execution remain conditional.

The appropriate direction is to **narrow the claim and examine the allocation
rule**. The specialists implement the intended behavior, while their scoring
rule has supplied no clear benefit here. This result does not justify adding
depth or announcing a new model family. The [prior-art comparison](2026-10-17-prior-art-revisit.md)
also limits any contribution to empirical characterization of a specified
procedure in established gated and growth families.
