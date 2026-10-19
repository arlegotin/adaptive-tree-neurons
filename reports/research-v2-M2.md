# M2: opportunity exists, but ordinary training raises the bar

The predeclared Gate A screen passes and admits one bounded joint-selector
study. This is development evidence, with no confirmed placement advantage.
Current cheap placement has higher mean V and E error than informed-direction
random placement in every displayed timing/treatment cell. Ordinary widening
and improved ordinary training are stronger leads. Early gated growth is
usually harmful here.

Eight independent primary synthetic teachers have two paired repetitions each.
The initial four triggered the single predeclared four-to-eight extension:
the descriptive fixed-arm interval covered zero and the 2% screening margin.
No further M2 teacher expansion is allowed. Four separate small planted base
instances support the pair-oracle diagnostic; they are not additional draws
from the primary population.

All 3,853 planned queue entries are terminal: 2,989 eligible and 864 ineligible.
Those counts include preparation, evaluation, aliases and nominal skips.
They are not 3,853 fitted models or independent problems. Of 1,792 planned
evaluation jobs, 1,408 have eligible saved metrics; 384 are explicitly ineligible.
The independent audit checked 5,472 prediction vectors and 65,664 scalar
comparisons, with maximum discrepancy 5.55e-17. No protected T was created
or evaluated.

## Timing, stream and capacity

Lower NMSE is better. Original-unit squared error is divided by each fit's
F target variance (ddof0); average R2 within each teacher, then weight teachers
equally. G selects checkpoints; V and E are development evaluations.
All original 250/1000/late snapshots were available; fixed-parent fallbacks
were not needed. New-stream means immediate F permutation/cursor replacement,
with old Adam state and absolute schedule retained, identically in every arm.
Growth uses existing F-derived contrasts, count 16, epsilon .05 and tau 1.
Ordinary lower/upper widths bracket the added parameter count.

### V NMSE

| Snapshot / stream | Keep | Cheap | Random | Ordinary lower | Ordinary upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| actual_250 / continued | 0.04978214 | 0.07494751 | 0.07003676 | 0.04705616 | 0.04876021 |
| actual_250 / new | 0.04876953 | 0.06990301 | 0.06979158 | 0.04838832 | 0.04602518 |
| actual_1000 / continued | 0.04978214 | 0.06334426 | 0.05980103 | 0.04939287 | 0.05021837 |
| actual_1000 / new | 0.04800584 | 0.06264055 | 0.05937629 | 0.04914660 | 0.04961933 |
| late_G_best / continued | 0.04978214 | 0.04902641 | 0.04855727 | 0.04976719 | 0.04962220 |
| late_G_best / new | 0.04836545 | 0.04941805 | 0.04887658 | 0.04909107 | 0.04748334 |

### E NMSE

| Snapshot / stream | Keep | Cheap | Random | Ordinary lower | Ordinary upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| actual_250 / continued | 0.05190017 | 0.07811001 | 0.07357066 | 0.04928228 | 0.05020242 |
| actual_250 / new | 0.04966652 | 0.07332558 | 0.07328191 | 0.04973051 | 0.04757494 |
| actual_1000 / continued | 0.05190017 | 0.06547459 | 0.06256124 | 0.05145057 | 0.05308340 |
| actual_1000 / new | 0.04905545 | 0.06502447 | 0.06264481 | 0.05064380 | 0.05081464 |
| late_G_best / continued | 0.05190017 | 0.05085810 | 0.04990736 | 0.05188278 | 0.05113280 |
| late_G_best / new | 0.05023155 | 0.05078308 | 0.05008620 | 0.05025111 | 0.04876863 |

Gate A finds at least 2% relative V improvement on two primary teachers for
a gated-growth coordinate, meeting its screening rule. The maximum for an
ordinary-capacity coordinate is five teachers. These are different settings,
not a paired proof that either family wins. The best observed fixed screen
coordinate is actual 250/new-stream/ordinary-upper: mean teacher-relative
benefit 4.8831%, descriptive bootstrap interval [−1.1220%, 13.3740%]. Its arm was
chosen on these development teachers; that interval is not confirmation.
The late/continued screening contrasts additionally use the V-best of keep
and the five ordinary recipes below as a diagnostic floor. Other screen
contrasts use matched keep. Raw mean tables above do not apply that floor.

## Score calibration and finite probes

On four teachers × R2, analytic birth gate gradients match autograd to 3.42e-18;
contrast gradients are zero at birth. The maximum small-step first-Adam
relative prediction error is 2.01e-6. Per-parent median minibatch/full-F rank
correlations range .851–.933, while 2.91%–13.16% of gate-only minibatch directions
predict a full-F loss increase. These source-bound F diagnostics validate a
local score, not its eventual predictive usefulness.

The 300-update singleton diagnostic has 512 unit probes plus eight keep probes.
501 of 512 units select stage zero. Six of eight score–G-gain correlations are
undefined because gains are constant; the two defined Spearman values are
+.13299 and −.14523. Undefined correlations remain undefined.

## Actual training horizon

The 72 fixed 8,000-update trajectories use four teachers × R2 and keep plus eight
F-score-stratified singleton slots. These are G-selected prefix predictors
from the same complete trajectories, without patience stopping. This differs
from the deployable patience 20 treatment. Lower E NMSE is better.

| Slot | Update 0 | Through 300 | Through 1200 | Through 8000 |
| --- | ---: | ---: | ---: | ---: |
| keep | 0.05159380 | 0.05159380 | 0.05019387 | 0.04321783 |
| sampled-0 | 0.05159380 | 0.05159380 | 0.04969395 | 0.04365491 |
| sampled-1 | 0.05159380 | 0.05159380 | 0.05013680 | 0.04346587 |
| sampled-2 | 0.05159380 | 0.05159380 | 0.05081790 | 0.04406350 |
| sampled-3 | 0.05159380 | 0.05159380 | 0.04974340 | 0.04347905 |
| top-0 | 0.05159380 | 0.05159380 | 0.05017486 | 0.04409969 |
| top-1 | 0.05159380 | 0.05159380 | 0.05056719 | 0.04522963 |
| top-2 | 0.05159380 | 0.05159380 | 0.05050750 | 0.04433528 |
| top-3 | 0.05159380 | 0.05159380 | 0.04985993 | 0.04406096 |

Ordinary keep improves mean E NMSE from .05159380 to .04321783, a 16.2344%
reduction. It has lower final mean V and E risk than every sampled singleton
slot. This establishes ordinary headroom beyond the replayed losing tail
in this diagnostic. It does not prove global convergence or rule out useful
joint additions. The subset is score-stratified, not a 64-unit random census.

## Small exhaustive joint control

The h8/count 2 planted control retains dead unit 7: seven eligible singletons,
21 eligible pairs and keep give 232 fitted trajectories across four bases × R2.
All 28 nominal pairs per repeat remain counted; ineligible pairs are explicit.
The pair oracle selects on G, with V/E unused for that choice.

| Base / repeat | Singleton-ranked pair | G-selected pair | E risk, singleton pair | E risk, G pair |
| --- | --- | --- | ---: | ---: |
| 0 / 0 | [0, 1] | [2, 6] | 0.04667132 | 0.04600383 |
| 0 / 1 | [0, 1] | [0, 6] | 0.04635646 | 0.04622789 |
| 1 / 0 | [2, 6] | [3, 6] | 0.04150842 | 0.03769974 |
| 1 / 1 | [2, 6] | [2, 6] | 0.04063309 | 0.04063309 |
| 2 / 0 | [1, 2] | [0, 3] | 0.04066834 | 0.04071284 |
| 2 / 1 | [0, 1] | [0, 3] | 0.04043211 | 0.04034468 |
| 3 / 0 | [3, 5] | [0, 1] | 0.05318341 | 0.05157466 |
| 3 / 1 | [3, 6] | [0, 1] | 0.05326063 | 0.05160582 |

G-selected joint pairs differ in 7 of 8 repeats. Mean E risk is 0.04533922
for singleton-ranked pairs and0.04435032 for the exhaustive G-selected pairs.
Seven repeats improve V; one ties. Six improve E, one ties and one worsens.
This is finite evidence of singleton/joint disagreement on the planted task,
with an expensive oracle. It does not establish a useful cheap selector on
the primary population. No independently trained parent weights were spliced.

## Ordinary recipe and head controls

Each recipe has G-selected retained weights. These are diagnostic recipes,
not the modern native practical benchmark. Same teacher/R2 aggregation.

| Recipe | V NMSE | E NMSE |
| --- | ---: | ---: |
| constant_001 | 0.07076305 | 0.07541432 |
| cosine_005 | 0.05122356 | 0.05346545 |
| legacy_005 | 0.04978214 | 0.05190017 |
| wd_1e3 | 0.04730260 | 0.04781628 |
| wd_1e4 | 0.04867694 | 0.05037639 |
| F-only ridge head, G-selected alpha | 0.04908294 | 0.05049748 |

The .001 weight-decay ordinary recipe has the lowest global mean V error
among the five declared neural recipes. The longer fixed-horizon keep
diagnostic and the recipe grid use different teacher counts/treatments;
their means are not a direct before/after comparison. Both raise the bar
for attributing gains to informed gated placement.

## Cost, provenance and limits

The enclosing scientific process took 69,415.103434250 s (19 h 16 m 55 s). Summed
charged job time is 69,114.187297110 s; these timers are nested and must not
be added. Collection plus independent arithmetic took 227.861691833 s
separately. Measurements were contended, include evidence I/O, and are not
a dedicated hardware benchmark or a modern quality–cost comparison.
Standalone prerequisite charges remain retained in the collector; shared
physical work must not be summed repeatedly across methods.

Producer:e1b1018 / package e322da64b518f29a14bde3385bfcf6d683993b717ed4c1fc13a7a5440a1f9a47.
Independent auditor:b848726 / package 6e81099c339b803ab6ed216e4f6987d3d2d238065b474a5703477d3b7ef345f7.
The authenticated queue endpoint is
`0c68ffaf65dc850924c4760e280e39e90cb59225076950edbb972cbf537541d2`.
Queue/file/row/array commitments and independent raw V/E metric arithmetic
were checked. The independent audit did not regenerate teachers, rerun model
predictions or authenticate every checkpoint tensor transition. Additional
F/G calibration and selection counters are authenticated producer records.
This distinction remains separate from normal-path software tests.

The [compact descriptive input](../audit/research-v2/M2/description.json)
retains every displayed coordinate, all teacher risks and additional source
bindings. Its [renderer](../docs/research/v2/scripts/report_m2_opportunity.py)
rebuilds this report without training. Full local raw predictions/checkpoints
are external to the tracked compact input; a clone can reproduce this
arithmetic account, not independently authenticate absent raw assets.
See the [execution receipt](../docs/research/v2/WP03-M2-complete-receipt.json).

Next: validate the exact parent-conditioned joint ridge estimator against
dense augmented solves and exhaustive small subsets, then run only its
bounded fresh-teacher comparison. Freeze a competent common parent and
training treatment before those new teachers; current cheap/random and
ordinary controls must receive that same treatment. A passing screening
rule permits the experiment; it does not predict that it will succeed.
