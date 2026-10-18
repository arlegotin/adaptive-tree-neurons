# Cost analysis preparation

Cost accounting keeps the measured containing cohort clock separate from the
standalone procedure's cost. Timers have explicit parent clocks and prerequisite
edges. Shared parent work is charged once to each standalone procedure that
requires it; nested timers are not added to their containing timer. Dependencies
of nested scoring work also propagate to the procedure. The same physical parent
therefore can legitimately occur in several attributed procedure costs, while the
physical cohort pays it once. Unattributed cohort overhead remains visible.

The timer tree rejects missing/cyclic identities, child clocks outside their
container, unfinished prerequisites, explicit containing-plus-child charges and
overlapping sequential standalone roots. Parallel worker attribution requires a
separate contract. These pure calculations do not certify complete native or
whole-pipeline measurement; no such qualification is asserted.

Paired timing arithmetic requires at least30blocks and the fixed randomized
method order reconstructed from the seed. It resamples the same blocks for both
methods and computes1−mean(selected cost)/mean(baseline cost), rather than averaging
per-block savings fractions. It retains20,000default paired bootstrap draws,
index commitments, raw-input commitments and separate95%percentile intervals for
fractional and absolute time saving. Chunk size preserves draws and commitments.

The numerical condition requires the lower savings bound strictly above20%, a
positive lower bound for paired time saving, and a supplied quality risk-ratio
upper bound strictly below1.02. Equality fails. Passing this arithmetic condition
authorizes no claim: frozen quality evidence, native CPU float32/batch1 timing
including preprocessing, randomized fresh-process measurements, method-specific
imports and fixed power/cache/compile settings remain pending. Conditional
percentile timing intervals do not establish universal coverage.

Seventeen missing-interface REDs became GREEN. A further regression exposed an
omitted parent cost when only the nested scorer declared that dependency:55was
charged instead of85. Propagating contained work's dependency closure corrected
it. The final cost/teacher/practical/workload/design suite passed105tests in1.66s.
All timers and timing blocks were artificial. scripts/check_cost_analysis.py
retains an explicit artificial example and all20,000paired draws; it measures no
real model and makes no efficiency claim.
