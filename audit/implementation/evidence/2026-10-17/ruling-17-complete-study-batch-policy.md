# Ruling 17 — select the common batch using complete-study feasibility

Ruling: Replace the throughput-only 95% minibatch-selection heuristic for this
pre-freeze H1 study with selection using development learning curves and measured
complete-pipeline time and retention; use a common batch of 128 for the next
finite calibration, and retain the original 4096 epoch-throughput result as a
diagnostic — the actual 128 and 4096 parents have similar descriptive G minima
and single-run fitting times, while their full raw logs occupy about 10.3 MB
and 155.2 MB respectively; retaining every method, probe, teacher and failure
is part of the research requirement — what it costs if wrong: the allocation
effect may depend on minibatch optimization, so the conclusion must name this
recipe; testing the throughput-selected batch would require a separately
budgeted, amended calibration and fresh independent cohort, with additional
time and storage. The current study cannot claim batch-independent superiority.

This is an explicit change to the initial engineering heuristic before any
genuine variance pilot, frozen protocol or protected test access. It changes no
model equation, primary population, information access, precision, score,
metric or paired fairness rule. Every causal fork and fresh capacity control
uses the same chosen batch and allowance. Full ordered raw logs remain retained;
the checkpoint storage correction is not expanded into another format change.

The actual same-problem parent observations are:

| Batch / learning rate | Selected / executed updates | G normalized MSE | Fit seconds | Raw event bytes |
|---|---:|---:|---:|---:|
|128 / 0.005|3675 / 4000|0.0709911|14.1435|10,295,035|
|4096 / 0.005|2875 / 3375|0.0730621|14.8851|155,181,441|

These are single calibration-problem observations, not statistical equivalence
or unseen-data superiority. The 128 parent still needs its stopping allowance
checked. The 4096 parent exhausted patience. A conditional full-graph ceiling
exceeding disk is not proof that the realized graph is impossible; this ruling
uses conservative planning and the measured retention tradeoff rather than
presenting that ceiling as a completed resource sample.

Next calibrate a finite common 128/0.005 recipe with symmetric 8000 parent and
continuation ceilings, unchanged G/checkpoint interval 25 and patience 20, on
the declared development teachers and representative controls. Those ceilings
are candidates, not a convergence claim. Do not shorten a still-improving run
to meet storage or select a recipe using the top-minus-random effect. A full
matching R>=2/all-eight-role/B6 resource teacher and strict readiness remain
required before the genuine pilot; numeric K and test size still come from
that pilot. If the full workload is infeasible, use the declared estimation or
resource-limited branch and report its attainable precision.
