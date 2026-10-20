# Compatible secondary-family bounds

This implements Task10's existing requirement for Holm tests and compatible
one-sided confidence bounds. It leaves the primary teacher t interval, margins,
power gate and protected access unchanged. No production analysis has run.

Use Guilbaud(2008), Theorem1/equation18 and Section3.2 Choice2/equation22,
with equal Holm weights and the common log-risk usefulness margin. For a partly
rejected family, a rejected effect has an **open** lower endpoint at the margin;
an unrejected effect uses the smaller of that margin and its marginal t bound
at alpha divided by the number unrejected. If all reject, use the maximum of
the margin and the minimum unadjusted one-sided t lower bound across the family.
This can be less informative than separate intervals. A rejected effect's open
margin endpoint must never be printed as a measured positive gap above the margin.

Do not apply each observed Holm rank's alpha to an ordinary interval and call
those intervals simultaneous. The construction requires valid marginal t
inference; it gives no distribution-free guarantee for nonnormal teacher effects.
The frozen procedure's simulation/assumption gates remain required. Missing or
zero-variance contrasts leave the whole planned family unavailable.

Primary source: [Guilbaud, 2008](https://doi.org/10.1002/bimj.200710449),
[accessible paper](https://www.aub.edu/SHARP/PublishingImages/Pages/publications/Simultaneous-Confidence-Regions.pdf).
The publisher record and paper's relevant method/proof sections were checked
before implementation. This is an implementation choice within the authorized
analysis contract, before real calibration or protected outcomes.

The implementation preserves the existing strict `p < alpha` decision. Its
marginal t intervals therefore include their lower endpoint; rejection regions
exclude the margin. This distinguishes a nonrejected equality boundary from an
open above-margin assertion. The common all-rejected bound includes its endpoint
when it is numerically above the margin. The paper's usual non-strict testing
convention is adapted explicitly rather than mixing endpoint conventions.


Canonical report projection is implemented. The provisional method policy records
the construction and endpoint rule and binds the inference source before data.
The renderer validates the complete family and preserves open versus included
endpoints without relaxing the primary confidence gate. Raw-artifact/protected
protocol authentication remain pending. Separate [practical bootstrap-family
arithmetic](practical-comparisons.md) and [report projection](practical-family-report.md)
now use their own marginal law, units and dependence assumptions; the teacher
procedure is unchanged.
