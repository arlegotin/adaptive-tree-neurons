# Repeatable native-anchor qualification

**20October amendment:** explicit `native_stopping:true` now selects the
results-first qualification route, with no wall deadline and the same finite
anchor inventory. NODE always uses its recoverable training checkpoints. Legacy
configurations without this field retain their original capped behavior and
identities; they are not a substitute for native-completed qualification. See
[the execution contract](results-first-native-execution.md).

Implement the WP07 `research.cli qualify` command using the already tested
method worker, native support compiler, owned-process supervisor and immutable
work queue. This replaces one-off operational launchers for future qualification;
existing executions and receipts keep their original identities.

The strict `research/v2-native-qualification` configuration contains `study_id`,
`source_sha256`, `locks_sha256`, `interpreter`, `datasets`, `families`, and
`overlays`. `datasets` maps a nonempty subset of the six fixed dataset names to
absolute sealed roots and external manifest byte hashes. `families` is an ordered
nonempty unique subset of P00–P13; canonical order is required. `overlays` maps
selected families to lists of relative directories within the current checkout's
`environment`. Installed dependency identities remain checked by each adapter.
Source and lock digests are externally supplied and rechecked throughout execution.
The lock digest covers ordinary files under `locks/research-v2` plus `uv.lock`.
Unknown fields and changed configuration, manifests, source or locks fail.
The optional `native_stopping` field must be the literal `true` when present;
numeric, string, null and false values fail rather than select an ambiguous mode.

`qualification_plan(config_path)` expands every declared native anchor for each
selected family/dataset, always using its primary split. The complete inventory
has 20 anchors × 6 datasets = 120 jobs; a subset is explicitly incomplete.
P03's h64 coordinate retains its native preprocessing/stopping. P12 always has
`bounded_memory_v1`. This command runs no HPO and reads no V labels or T.
`plan --config` supports the same configuration without starting fitting.

`qualify(config_path, output, resume=False)` uses one sequential worker, seed 17,
the complete native procedure and sampled 4 GiB RSS. Explicit native-stopping
mode has no wall deadline; legacy mode retains 7,200 seconds per anchor.
Recheck the live memory proxy (5.5 GiB) and disk (1.5 GiB) before every job.
Pause before launch if unavailable; `--resume` requires the identical immutable
plan. Preserve native stopping, every failure, source/dependency identity,
closed model/predictions and all costs. Materialize only F/G and V-feature files
through checked hard links or disclosed cross-device copies. Pay staging and
translation in the enclosing trial budget. No partial timed-out fit is promoted.

Native mode uses `native_fit_v2` directly, without a fake one-trial capped
search. Before its one allowed resource/interruption retry, authenticate the
previous queue failure and the entire retained raw attempt inventory. Only a
closed, externally committed native result can supply a checkpoint-copy
reference. Copy NODE checkpoints into independent files in a fresh attempt;
other methods explicitly restart the same request/seed. Changed previous bytes
fail before another child starts. All attempts and costs remain recorded. An
unknown closure is not permission to copy or silently replace training state.

The result records completed/ineligible/failed/paused statuses and coverage. It
does not grant Gate D, native quality, production search, secondary development
or protected access. Those require the remaining independent workload/range/
recovery and final common-source qualification. An all-120 execution is a native
anchor execution inventory, not a predictive comparison.

Files: `research/qualify.py`, the existing CLI dispatcher, focused tests, and
fresh output manifests. API `load_qualification_config(path)` validates inputs;
`qualification_plan(path)` returns a work-queue plan;
`qualify(path, output, resume=False)` executes or pauses it. Test planning/counts,
strict drift rejection, resource pause/resume, CLI dispatch and one real native
P13 primary-split job before admitting a future full common-source queue.

Fixture verification observed 12 missing-module failures, then 12 passing checks.
Four malformed path/CLI cases and one missing-feature case exposed actionable
error handling gaps; strict type/schema checks corrected them. One combined run
retained 65 passes and a legacy caller-thread precondition failure. The documented
Torch thread setup resolved that case without changing its guard. The final
affected run passed 67 checks in 18.12 s. This includes planning all 120 jobs,
resource pause, identical-plan resume, terminal-job preservation, CLI dispatch,
and the existing research runner/work-queue/planner checks. The mocked queue
completion fixture fits no model. Actual CLI native execution is still pending.
See [the preparation receipt](WP07-native-qualification-runner-receipt.json).

The real CLI check subsequently completed at `c9cf9d9`/`7e9105f6…`: plan then
qualify ran the native P13 mean coordinate on Energy's primary F320 and 128 query
rows. An inactive independent audit authenticated the queue and process records,
recomputed the F mean with zero prediction error, and reopened the saved model
with exact predictions. The worker imported no Torch and was reaped. Sampled
peak RSS was 85,131,264 bytes. Fresh qualification CLI wall time was 2.737855750 s,
containing a 0.942460666 s worker process; the separate audit body took 0.511829708 s.
This repeats an already executed coordinate to verify command wiring; it adds no
distinct anchor or independent teacher. No V/T labels, HPO, quality comparison or
Gate D promotion. See [the real-command receipt](WP07-native-qualification-real-CLI-receipt.json).

The explicit native mode has now been checked on real synthetic sealed-role
fixtures, including a native P13 worker, a forced owned-RSS interruption and
same-request/seed retry, and rejection of changed old evidence. These fixtures
do not authenticate official dataset acquisition or qualify a new full native
anchor. See WP07-native-qualification-uncapped-receipt.json for the complete
RED/GREEN record. Production HPO, complete support/workload qualification and
Gate D remain pending.
