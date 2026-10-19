# Results-first native execution

The user asked on19October why training stopped after two hours and explicitly
prioritized results over speed. The handoff supplied a120-minute comparison
budget; applying it to native qualification stopped a healthy NODE run. The
prospective policy now lets native qualification and the quality view reach their
declared stopping rule. Resource-bounded results remain separately identified.

The earlier Energy/seed17 bounded-memory run used about1.49GiB sampled peak RSS
and stopped at7200seconds. Its updates are unknown and it published no model or
training checkpoint. Those hours remain in the cost record. A fresh same-data,
same-configuration/seed attempt is required; it cannot recover missing weights.

Implementation decisions are the builder's calls under the user's instruction:

- Preserve full2048trees/depth6/effective batch512, QHAdam and native G100,
  rolling-five selection/patience5000. Preserve bounded-memory32-row execution.
- Save initial and completed report-boundary training state, at most100applied
  updates apart. Resume the raw model and moments, sampler/permutation/cursor,
  NumPy/Python/Torch RNG, rolling raw checkpoints, selected averaged model and
  selected optimizer/window, strict G-best value/step, F transform and identities.
- Bind source, runtime, configuration, F/G and seed. Explicitly restore native
  initialization flags. Check model dtype/shape/finiteness and nonnegative QHAdam
  second moments/ages before publishing or loading. Never treat an incomplete
  checkpoint as an eligible final model.
- Keep row/update history in immutable external segments. Each state commits its
  prefix; a resumed attempt reads that prefix and leaves later uncommitted files
  intact. Publication writes immutable payload/metadata before an atomic pointer.
  Retain all operational checkpoints initially; measure storage and compact only
  with a declared, verified retention policy if needed.
- A separate explicit native-stopping route has no time deadline. Keep owned
  process supervision, RSS/disk admission and interruption receipts. One worker
  runs at a time. No hidden data subsampling, native-recipe shortening or new seed.

Execution steps for Task7, before the new long run:

- [ ] Write and observe failing tests for native-stopping budgets, exact interrupted
  continuation across a non-epoch boundary, wrong identities/dtype/negative
  moments, raw versus averaged weights, and incomplete atomic publication.
- [ ] Add a focused training-checkpoint module and NODE training hooks. Reuse
  safe closed-array encoding; keep prediction snapshots and legacy capped paths.
- [ ] Prove uninterrupted and resumed rows, raw/selected weights, moments, RNG,
  sampler and G records agree on actual native tiny geometry, ignoring measured
  elapsed times. Run affected endpoint, worker and process-supervision checks.
- [ ] Add and verify deadline-free owned supervision and a checkpointed native
  qualification entry point. Preserve all attempts and independently reopen the
  final model/predictions. This route grants no V/T access or quality win.
- [ ] Run an actual full2048-tree shape recovery proof before admitting the long
  Energy run. Record peak RSS, checkpoint bytes, source and exact results.
- [ ] Pin the new executable source, forecast disk use, and restart Energy/seed17
  under native stopping with periodic checkpoints and RAM/disk guards. Continue
  selector implementation while that single worker runs.

Quality-first HPO needs explicit controller/inventory semantics before production:
finite32attempt supports remain; every attempted native configuration may finish;
time prefixes admit only models completed by their deadline. Full-quality and
matched-budget winners and costs are distinct. No current engineering queue is
retroactively promoted into that production procedure.
