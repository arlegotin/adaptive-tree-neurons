# Complete paired practical comparisons

`compare_practical_families` consumes complete normalized group/day losses for
all fourteen families and both search repetitions. It compares P00 against
P01–P13 and a champion whose family IDs were already selected on V, separately
for each dataset and repetition. It cannot authenticate that selection or the
models. It never chooses a family from these supplied held-out losses.

All comparisons use the same bootstrap group/day indices and both repetitions
share those indices. Risks are averaged across repetitions before taking log
ratios, then the five fixed observational tasks receive equal weight. Energy
remains descriptive and outside the aggregate and multiplicity family. Missing
families fail; undefined log risks make the full planned inference unavailable
without dropping resamples or reducing thirteen hypotheses to an easier family.

The primary V-champion contrast retains the existing two-sided 95% percentile
interval and 2% useful-risk margin. It also reports positive aggregate effects
in both repetitions, the count of nonnegative tasks and every degradation over
5%. Three- and fourteen-day Bike block runs cannot substitute for the primary
seven-day result. Dependence sensitivities and qualification still need the
complete evaluator; arithmetic flags authorize no claim.

The previously unspecified individual-family test law is now declared before
any protected outcome: for point effect `b` and bootstrap effects `b_star`, use
centered errors `b_star-b` and one-sided probability
`(1 + count(error >= b-margin))/(B+1)`. This is a conditional bootstrap location
approximation, not an exact permutation test. Ties count against rejection;
probabilities cannot become zero. Strict Holm correction uses all thirteen
comparisons at alpha .025. The primary percentile procedure is unchanged and
distinct from this individual-family procedure; no favorable choice between
them is permitted after outcomes.

Compatible lower regions use the already documented
[Guilbaud construction](holm-compatible-bounds.md), with the marginal bootstrap
law replacing the teacher t distribution. Invert that exact discrete law:
`k=ceil(level*(B+1))-2`, with critical error `sorted_errors[B-k-1]`. The marginal
lower endpoint is closed. Insufficient replication to resolve alpha/13 makes
the family unavailable. Partial rejection gives an open margin endpoint for a
rejected effect, without claiming a measured gap above the margin. All-rejected
families use the common weakest unadjusted marginal lower bound, intersected
with the above-margin region.

The primary paper's assumptions and equations 18/22 were rechecked in the
[accessible source](https://www.aub.edu/SHARP/PublishingImages/Pages/publications/Simultaneous-Confidence-Regions.pdf).
Its result requires valid marginal probabilities and confidence regions. It
does not establish the validity of this bootstrap approximation for few clusters
or temporal dependence. Parkinsons' small subject count and CCPP's unverified
exchangeability remain material limits. No population superiority or exact
finite-sample coverage is asserted.

The affected six-module suite passed **116 checks in 1.98 seconds**. Fourteen
initial missing-interface failures were retained. A later equality fixture used
rounded floating inputs that lost an intended tie by one ULP; the diagnosed
fixture now uses exactly representable integers, without adding a tolerance or
changing the production test law. Four additional boundary checks passed the
implemented behavior on their first run. See the
[receipt](WP08-practical-comparisons-receipt.json) for commands and raw synthetic
loss/draw commitments. No new models, real labels or T were used.

Production queue/readiness, global semantic protocol, secondary release,
protected evaluation and canonical practical-family report integration remain
unfinished. This component supplies arithmetic for those stages.
