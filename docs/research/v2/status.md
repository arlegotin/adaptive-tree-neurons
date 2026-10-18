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
selected the parent. Joint grown/ordinary final arms are still running; this
single case is neither a cohort result nor a placement verdict.

The next scientific gate asks whether the controls expose useful, learnable
capacity on multiple independent problems. Matched timing, minibatch-stream,
ordinary-capacity and recipe controls separate training gains from placement
gains. A negative gate closes or narrows the tested branch; it does not authorize
more tuning until a favorable seed appears.

Current execution state and commands live in [execution.md](execution.md),
[reproduce.md](reproduce.md), the [binding spec](../../superpowers/specs/2026-10-18-research-v2-design.md)
and [implementation plan](../../superpowers/plans/2026-10-18-research-v2.md).
