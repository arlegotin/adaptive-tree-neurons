# Two further exploratory comparisons

The new checks give mixed results. Selected growth has a tiny lead over random
growth on problem A. On problem B, both return the unchanged parent predictions,
and a wider ordinary network does better. This does not establish a reliable
benefit from choosing which neurons grow.

The [machine-readable results](2026-10-17-common-recipe-exploratory.json) and
[retained evidence index](../../audit/implementation/evidence/2026-10-17/calibration-common-recipe/index.json)
record all controls and source identities. Root independently rehashed the
complete 282-file inventory, including every retained run and prediction.

| Method | Problem A error | Problem B error |
|---|---:|---:|
|Continued parent|0.076272|0.042575|
|Selected growth|0.075581|0.042575|
|Random growth|0.075739|0.042575|
|Fresh wider ordinary network|0.094315|0.033623|
|Static partial gated network|0.132770|0.047559|

Lower is better. These are normalized mean squared errors on separate
exploratory evaluation samples of 4,096 observations per problem. Each problem
has one optimizer/allocation repetition. The selected-minus-random comparison
has no population interval or confirmatory interpretation. The original study
used a different training recipe and V observations; do not pool its error
numbers with this table.

All methods share the calibrated candidate recipe: batch 128, AdamW learning
rate 0.005, zero weight decay, up to 8,000 parent and 8,000 continuation updates,
G selection/checkpoints every 25 updates and patience 20. All 18 stages across
the two problems stopped under that patience rule before their ceilings.
The primary models have 16 inputs, width 64 and 16 grown units. Parameter counts
are 1,153 for the parent, 1,697 for grown/static models and 1,711 for the upper
ordinary bracket. Fresh ordinary/static controls receive both training stages.

The unchanged-model sham reproduces the parent exactly on both problems.
Insertion preserves initial predictions exactly, and the paired sample-stream
audits pass. Complete raw logs, losing controls and selected checkpoints remain
retained. An auxiliary wrapper miscounted continuation examples by including an
inherited history tail; the saved raw-stage summary corrects that field without
changing training, predictions or decisions.

These checks help settle training and cost settings before the independent
study. Required next evidence includes finite sensitivities, complete measured
resource costs, a genuine variance pilot, a frozen design, the full independent
cohort and reproduction. The conclusion can still be positive, negative or
inconclusive. No practical superiority or architectural novelty is established.
