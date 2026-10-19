# Small-oracle check of the new placement scores

The new scores identify the two planted generating units on the four existing
M2d control instances. This is a useful diagnostic lead, with mixed finite
outcomes and no independent M3 result. All four new score variants choose the
same pair, so their quality difference cannot establish a benefit from joint
selection or parent conditioning.

This reuses the completed M2d exhaustive fit table: four known h8/count2 planted
bases, two training repeats, gamma2/noise.1, legacy weight decay0 and native
patience stopping. It contains296nominal entries,232eligible fits, with the dead
unit and all nominal ineligible subsets retained. No model was newly fit.
Historical checkpoints were read under their original pinned source. New
selection used only exported F Jacobians/residuals before joining old G/V/E
outcomes. This differs from the stronger prospective M3 training treatment.

| Selector / diagnostic | Mean V NMSE | Mean E NMSE |
|---|---:|---:|
| Original cheap score | 0.04500629 | 0.04622444 |
| Matched informed-direction random | 0.04583136 | 0.04713066 |
| Local-singleton ridge | 0.04346333 | 0.04510079 |
| Conditional joint ridge | 0.04346333 | 0.04510079 |
| Unconditioned joint ridge | 0.04346333 | 0.04510079 |
| Zero-parent-ridge span | 0.04346333 | 0.04510079 |
| Full-horizon G-singleton ranking | 0.04358374 | 0.04533922 |
| Exhaustive G-pair oracle | 0.04197731 | 0.04435032 |

The old cheap score chooses duplicate units0/1; each new variant chooses0/2.
Joint beats random in E on three bases and loses on one. It beats cheap on two
and loses on two. The mean is a descriptive result on these known controls;
it is neither a reliable population effect nor evidence against strong baselines.
The exhaustive oracle has extra G information and pays every pair trial.

The binding prediction screen compares predicted joint gains against the sum
of independently predicted singleton gains, using finite V pair gains after
averaging the two repeats within each base. The joint predictor has higher
Spearman correlation on three of four defined bases:

| Base | Joint predictor | Singleton-sum predictor |
|---|---:|---:|
| 0 | 0.81632 | 0.76026 |
| 1 | 0.69557 | 0.65624 |
| 2 | -0.42420 | -0.13194 |
| 3 | -0.39557 | -0.46406 |

This passes the small-control prediction screen. Two bases have negative
correlations for both predictors, showing a substantial local-to-finite mismatch.
Passing this screen does not pass Gate B, which also needs fresh primary-teacher
benefit over matched random at qualified complete cost. No ridge/direction
settings were tuned here.

All actual gate blocks match autograd exactly in this export. Independent dense
subset objectives agree within4.17e-16. A separate audit rebuilt all selected
method metrics from original committed predictions with zero discrepancy and
checked the V rank correlations using manual average-tie ranks. It read66unique
selected V/E vectors and176unique V rank vectors; those sets overlap. All old
fit/probe costs remain retained; this analysis does not make the oracle cheap.

See the [receipt and exact-file inventory](../docs/research/v2/WP04-small-oracle-reanalysis-receipt.json),
[M2 report](research-v2-M2.md), and
[locked next-study treatment](../docs/research/v2/M3-development-recipe.md).
No protected T was created or read.
