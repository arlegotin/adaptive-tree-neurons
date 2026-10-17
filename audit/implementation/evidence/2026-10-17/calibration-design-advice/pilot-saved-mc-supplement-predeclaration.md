# Prespecified optional saved-data Monte Carlo diagnostic

Written while the fixed sixteen-teacher pilot is incomplete (23/32 repetitions
last observed). This does not change that pilot, its eight-thousand-draw recipe,
its precision threshold, raw evidence, stop policy or canonical eligibility.
No numerical work occurs until the worker returns explicit STOP and root audits
the complete report. This diagnostic is additional explanatory saved-data work;
it is not a substitute frozen protocol or power/coverage validation.

If the terminal pilot reports endpoint MC imprecision, run exactly one separate
512,000-draw whole-teacher bootstrap using the original bootstrap seed261022106,
the same95% percentile intervals and the same ten-disjoint-chunk endpoint SE
method over all H1/static/SD/component endpoints. Do not search seeds, selectively
refine endpoints, change the tolerance, or choose a draw count based on effect
sign. The larger fixed draw count provides a computational margin relative to
the first-eight observed endpoint error and declared tolerance; it guarantees
neither a passing numerical diagnostic nor adequate population information.

Use a bounded saved-scalar calculation after independently checking the terminal
teacher/repetition scalar means and within-/row variances. Bind all original
input hashes plus analysis script/config/outputs. Reproduce the original8,000
calculation first as an explicit derivation check; preserve both results and all
failure evidence. Use teacher means/static means/within variances/row variances
and row counts; avoid decoding giant duplicate raw arrays when those sufficient
scalar summaries can be bound and independently checked. No package/code/source
or accepted environment modification, training, new data or T access.

Report original and supplemental H1/static/SD/component intervals, SD ratio,
all endpoint SEs and max, count/seed, runtime, memory, limitations and exact
numerical agreement with the original. If the independent reproduction of the
original calculation disagrees beyond declared floating rounding, stop and report
that discrepancy before the larger calculation. Keep the original canonical
calibrate/blocked branch unchanged. A supplemental stable SD interval wider than
the calibration ratio remains sampling uncertainty, which extra MC cannot fix.

Root must still adjudicate scientific value/resources for additional independent
evidence and complete WP09; this note authorizes no cohorts or eligibility waiver.
No new chronological ruling: it preserves the original statistical policy and
adds a transparently labeled diagnostic without changing any decision gate.
