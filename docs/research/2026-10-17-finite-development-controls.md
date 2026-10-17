# Development controls and remaining uncertainty

The growth mechanism works, but the early results do not establish that its
placement score is better than random selection. The finite checks below show
both opportunities and limits. They helped choose an executable common training
recipe; they are not independent confirmation.

The [retained evidence index](../../audit/implementation/evidence/2026-10-17/calibration-finite-controls/index.json)
links the original report, exact coordinates/seeds, full learning curves, probe
records and numerical measurements. Root independently rehashed the complete
1,169-file, 663,296,577-byte ignored inventory without a mismatch. The public
archive preserves selected original bytes; the full raw runs remain retained.

| Exploratory condition | Continue parent | Selected growth | Random growth |
|---|---:|---:|---:|
| Fewer fitting observations, F=577 |0.264226|0.226240|0.236890|
| More fitting observations, F=9,224 |0.035041|0.035041|0.035041|
| Smaller contrast, epsilon=0.01 |0.076272|0.076488|0.075671|
| Larger contrast, epsilon=0.2 |0.076272|0.075631|0.075212|
| Affine target |0.012955|0.012955|0.012955|
| Smooth additive target |0.117701|0.117701|0.114575|
| Noise only |0.965377|0.965377|0.965377|
| Shuffled labels |0.980296|0.980296|0.980296|
| Homogeneous smooth target |0.036815|0.036815|0.036815|
| Axis-aligned regimes |0.063165|0.063165|0.063165|
| Rotated regimes |0.057530|0.056385|0.055131|
| Four irrelevant input dimensions |0.075383|0.075383|0.075383|
| Rare regime, prevalence 0.01 |0.086543|0.078585|0.074990|

Lower is better. Values are squared error divided by fitting-target variance,
on 4,096 separate exploratory evaluation observations for each coordinate.
These 13 conditions reuse some teachers and contain only one training/allocation
repetition each. They must not be counted as 13 independent estimates of the
primary effect. The axis/rotation comparison uses paired generator seeds; the
contrast checks reuse problem A. The larger population inference remains open.

All 53 fitting stages stopped by the common patience rule before the 8,000
update ceiling. Insertion and sample-stream pairing checks passed; the noise
sham returned exactly the continued-parent predictions. The common primary
recipe remains batch 128, AdamW learning rate 0.005 with zero weight decay,
16 inputs, width 64, 16 grown units, contrast epsilon 0.05, G/checkpoint interval
25 and patience 20. Losing conditions did not trigger a replacement recipe.

The score wins on the sparse-data draw, while random placement does better on
the additive, rotated, rare-regime and both alternative-contrast draws. Many
models choose their original parent checkpoint. That combination is reason to
measure independent-teacher variability, rather than infer either a reliable
advantage or universal failure from these examples.

A short trial can also miss a useful split. On four predeclared candidate units
from the same parent, 20- and 80-update probes selected unchanged predictions;
at 320 updates, three candidates selected improvements at update 225. The
320-update horizon establishes an opportunity within that allowance, not
convergence or reliable ranking across all 64 units. The expensive greedy
control will pay for the unchanged-parent trial and every candidate, including
losers. It receives the same finite trial allowance for each candidate.

Evaluation precision is unresolved. On one fixed trained problem, increasing
the exploratory evaluation sample to 262,144 rows still missed the declared
engineering standard-error yardstick for selected versus random placement.
That row count is a bounded workload for the upcoming genuine pilot, not a
validated final test size. Final test rows must follow independent-teacher
variance estimates or an explicitly limited estimation/resource decision.
Conditional row resampling and artificial-shift power checks only calibrate
computation; they do not supply independent teachers or a population interval.

Prediction timing showed no demonstrated speed advantage: at batch 128, the
host selected/random ratio was 1.0066 with interval [0.9997, 1.0110]. All four
host/resident cases reached the declared 5% precision rule after 27 complete
paired blocks. Resident chunks stayed within the tested 128-row bound.

The next gate is a complete two-repetition engineering teacher with all eight
methods and all greedy probes. Its measured fit, evaluation, publication,
storage and memory costs will support an explicit resource forecast. A genuine
variance pilot, frozen independent cohort and reproduction still follow. No
practical-superiority or model-family novelty claim follows from these checks.
