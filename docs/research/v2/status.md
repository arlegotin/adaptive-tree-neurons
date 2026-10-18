# What the research is trying to establish

A grown neuron can blend between two affine rules instead of using one. The
question is whether choosing where to add this flexibility improves prediction
enough to justify the complete training and selection cost.

The idea remains plausible. Historical runs show that added capacity sometimes
helps under the specified recipe, but the cheap placement score has not reliably
beaten matched random placement. This is a limitation of the present evidence,
not a demonstrated failure of every gated architecture. Strong native modern
baselines have not yet received the planned predictive comparison.

The main findings guiding this continuation are:

- Function-preserving insertion and ordinary optimizer-history transfer work on
  the tested normal path. Manipulated checkpoint validation gaps were hardened;
  this has not been credited as a predictive improvement.
- Patience selection restores the whole earlier training state. Unchanged
  deterministic continuation can repeat an already unsuccessful tail, so a no-op
  keep arm does not show that ordinary training has exhausted useful headroom.
- Good single additions can overlap. A useful selection procedure may need to
  choose complementary additions as a group. The joint selector is still a
  proposed method, and implementation/efficacy are gated on measured opportunity.

The fresh first M0 qualification case is development evidence. Its parent
selected update3700 after executing4200. All65 short320-update singleton/keep
probes selected their starting checkpoint. The longer unchanged keep arm also
selected the parent, with500 identical updates and21 identical G evaluations
verified from the retained raw event streams. All eight final roles completed.

On the independent exploratory E rows, normalized MSE was:

| Procedure | Error; lower is better |
|---|---:|
| Unchanged parent/keep | .04912814 |
| Cheap-score growth | .04891073 |
| Matched random growth | .05031929 |
| Short singleton-selected growth | .05011541 |
| Warm ordinary widening | .04912814 |
| Fresh ordinary lower bracket | .04535861 |
| Fresh ordinary upper bracket | .05314361 |
| Static partial gated | .10158990 |

The cheap-score result was close to keep; a fresh ordinary lower-bracket network
performed best. V also favored that ordinary network and did not favor cheap
placement over random. One teacher/repeat cannot establish a population effect,
and E is now development information. Independent raw-prediction arithmetic
agreed within6.94e-18. This case took93.55minutes of contended wall time and retained
about485MB; those are resource observations, not an efficiency claim. The full
four-teacher/two-repeat M0 cohort and positive-control sensitivity remain pending.

The same case was rerun after checkpoint bookkeeping changes. An independent
raw-artifact audit found exact agreement for every prediction, score, allocation
and selection, all40,375 updates and1,705 G evaluations, and all162 authenticated
checkpoint/export tensor payloads. Wall time was16.35minutes in the rerun versus
93.55minutes originally; both observations were subject to machine contention.
Retained bytes remained approximately485MB. This verifies engineering
compatibility, not an additional teacher or a predictive improvement.

A first planted-control case has also completed: instance0/repeat0 at gamma2,
noise sigma.1, with a deliberately unfitted h8 parent and count2. Here the known
target contains useful gated structure. Exploratory E normalized errors were:

| Procedure | Error; lower is better |
|---|---:|
| Ordinary continuation | .05453974 |
| Cheap-score growth | .04185875 |
| Matched random growth | .04230658 |
| Known generating IDs, current F-derived contrasts | .04149014 |
| Known generating IDs and true contrasts | .04168284 |
| Sampled noise diagnostic | .03973174 |

All growth roles reduced E error by about22–24% versus ordinary continuation in
this case. Cheap placement was only about1% better than random on E. Independent
NumPy/raw-array reconstruction reproduced every V/E metric exactly, with282
checks and32 exact array checks. The two oracle roles use privileged information.
This supports learnable useful capacity under this constructed recipe; it does
not establish placement superiority, advantage over matched ordinary extra
capacity or tuned models, or generalization across independent problems.
This single qualification cell took209.39s and retained26.36MB; timings are
contended observations. Full M1/M2 and Gate A remain pending.

Cheap placement chose the duplicate units0/1; random chose4/6, while the planted
generating IDs were0/2. Their similar final errors suggest this constructed case
may offer many workable placements. That is a hypothesis about control
sensitivity, not evidence that location never matters; the fixed multi-instance
and small-pair comparisons must measure it.

The next scientific gate asks whether the controls expose useful, learnable
capacity on multiple independent problems. Matched timing, minibatch-stream,
ordinary-capacity and recipe controls separate training gains from placement
gains. A negative gate closes or narrows the tested branch; it does not authorize
more tuning until a favorable seed appears.

Current execution state and commands live in [execution.md](execution.md),
[reproduce.md](reproduce.md), the [binding spec](../../superpowers/specs/2026-10-18-research-v2-design.md)
and [implementation plan](../../superpowers/plans/2026-10-18-research-v2.md).

The first globally affine negative control also completed. Keep, cheap growth
and random growth each selected their unchanged starting predictions after 500
continuation updates: E NMSE .01228926 in all three arms. This is one case, not
population evidence of calibrated no-op behavior.

The full fixed M1 control cohort is now admitted and running under unchanged
scientific source604cffb: 64 transactions, 160 planted-positive and 96 negative
final fits. The positive coordinates share four base instances across magnitudes
and noise; paired repeats do not create independent problems. A final-source
qualification repeated the inspected positive case with all 52 raw arrays and
five role counters exactly unchanged, adding zero independent teachers.

One numerical training worker runs at a time. Measured speed projections at all
update ceilings are 14.01/18.94 worker-hours, with 14.698GiB forecast storage
including a rolling retry reserve plus a 0.5GiB free-space floor. Those are
uncalibrated projections from two measured cases, not promised completion times
or efficiency evidence. The actual live capacity check passed. Full M1, M2,
Gate A, trained joint selection and modern practical comparison remain pending.
No protected T has been created or evaluated. The tracked
[run receipt](WP02-M1-source-604cffb-receipt.json) binds the measurements and launch.
