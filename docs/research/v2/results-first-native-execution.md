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
  Retain the latest two committed tensor payloads; after atomic publication,
  remove only older committed operational payloads, recording their hashes and
  the superseding publication. Keep every manifest, row-history segment and
  failed transaction. Test this bound before admission: retaining every full
 2048-tree tensor payload would otherwise consume the remaining disk.
- A separate explicit native-stopping route has no time deadline. Keep owned
  process supervision, RSS/disk admission and interruption receipts. One worker
  runs at a time. No hidden data subsampling, native-recipe shortening or new seed.

Execution steps for Task7, before the new long run:

- [x] Write and observe failing tests for native-stopping budgets, exact interrupted
  continuation across a non-epoch boundary, wrong identities/dtype/negative
  moments, raw versus averaged weights, and incomplete atomic publication.
- [x] Add a focused training-checkpoint module and NODE training hooks. Reuse
  safe closed-array encoding; keep prediction snapshots and legacy capped paths.
- [x] Prove uninterrupted and resumed rows, raw/selected weights, moments, RNG,
  sampler and G records agree on actual native tiny geometry, ignoring measured
  elapsed times. Run affected endpoint, worker and process-supervision checks.
- [x] Add and verify deadline-free owned supervision and a checkpointed native
  qualification entry point. Preserve all attempts and independently reopen the
  final model/predictions. This route grants no V/T access or quality win.
- [x] Run an actual full2048-tree shape recovery proof before admitting the long
  Energy run. Record peak RSS, checkpoint bytes, source and exact results.
- [x] Pin the new executable source, forecast disk use, and restart Energy/seed17
  under native stopping with periodic checkpoints and RAM/disk guards. Continue
  selector implementation while that single worker runs.

The new route is `research.method_worker --training-checkpoints PATH` with an
explicit `native_stopping:true` budget; `--resume` requires matching stored state.
Only P12 accepts this route. Explicit update budgets up to128 are engineering
proofs, never native completion. Other workers and existing capped controllers
retain their prior contracts. The native launch operator is
`docs/research/v2/scripts/run_node_native_recovery.py`; its arguments pin the
prepared manifest and held source. It stages only F/G and V features, checks
resource admission and uses owned-process supervision without a deadline.

Final affected verification passed106checks with79retained native warnings in
248.74seconds. A separate real process-supervision run passed33checks in2.04s.
Initial missing features, an incorrect test optimizer-slot assumption and all
subsequent RED/GREEN stages remain in
[the implementation receipt](WP07-NODE-training-recovery-implementation-receipt.json).
The tiny exact-recovery result does not qualify full2048-tree recovery or native
patience completion. The full-size operator has now passed under held commit
`c33a84e`:105reference updates, interruption after102applied updates, restoration
from checkpoint100 and continuation to105match raw/selected weights, moments,
RNG, sampler, rows and predictions exactly. The owned process took219.565seconds
for all three fits and closure, peaking at1,267,253,248sampled RSS bytes
(about1.18GiB). Independent audit regenerated the fixture inputs, reopened both
models, checked exact state equality, resource commitments and process cleanup.
Its separate cost is retained; the105-update fixture is quality-ineligible.

The Energyprimary data/seed17 full native fit completed in that held checkout:
5701updates/selected700/57Gchecks,7362.48seconds and1.64GiBsampled peak RSS.
A separate audit verified the native stop and exactly replayed saved predictions.
Operational payloads retain at most two committed states plus one transaction;
full row histories remain once. Full six-dataset P12/final-source qualification
and full-suite claims remain pending. See
[the completion receipt](WP07-NODE-uncapped-completion-receipt.json).

The next single native NODE worker now runs Airfoil primary fold0/seed17 from
the same held source, with the same full architecture, native stop,32-row
execution, periodic exact recovery,4GiBguard and no wall deadline. Its primary
operator also supports grouped fold0 and Bike last temporal fold4 explicitly;
no secondary-fold experimentation is admitted. This launch has no quality result.
See WP07-NODE-Airfoil-launch-receipt.json.

Quality-first HPO needs explicit controller/inventory semantics before production:
finite32attempt supports remain; every attempted native configuration may finish;
time prefixes admit only models completed by their deadline. Full-quality and
matched-budget winners and costs are distinct. No current engineering queue is
retroactively promoted into that production procedure.

Prospective quality-search scope: at most32attempted configurations per family
(including failures/retries; P13ends after its29finite configurations), with fixed
supports/anchors/seeds and each attempted native fit allowed its declared
stopping rule. There is no wall-clock quality cutoff. The10/30/120minute
completion prefixes remain descriptive views of that same sequential search.
Their winners cannot include a trial completed later. The full-quality portfolio
pays every P02–P13search; its cost is measured, with no inherited24-worker-hour
ceiling or equal-cost claim. This results-first policy must enter the final
prospective quality estimand, workload forecast and freeze before any production
search. The earlier120minute primary contract is superseded for the quality
view only, under the user’s explicit instruction. Margins, fixed supports, paired
repeats and test-access barriers remain unchanged.

The controller now implements explicit `deadline_ns=None` for that finite quality
view, with exact nil-budget journal replay and separate prefix selection. Legacy
integer-deadline outputs retain their prior semantics. Three new controller
cases failed before implementation; a legacy capped-bridge refusal also failed
first. Final65affected checks passed in36.09seconds. An initial sandbox run
retained63passes/two owned-RSS probe failures; native recovery passed without
weakening the guard. The existing engineering bridge explicitly rejects
quality plans. A qualified deadline-free production worker bridge, full-quality
primary inventory/freeze and actual workload admission remain pending; no
production HPO or real quality selection has run. See
WP08-native-quality-controller-receipt.json.
