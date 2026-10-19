# Repeatable native-anchor qualification

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

`qualification_plan(config_path)` expands every declared native anchor for each
selected family/dataset, always using its primary split. The complete inventory
has 20 anchors × 6 datasets = 120 jobs; a subset is explicitly incomplete.
P03's h64 coordinate retains its native preprocessing/stopping. P12 always has
`bounded_memory_v1`. This command runs no HPO and reads no V labels or T.
`plan --config` supports the same configuration without starting fitting.

`qualify(config_path, output, resume=False)` uses one sequential worker, seed 17,
the complete native procedure, 7,200 seconds per anchor and sampled 4 GiB RSS.
Recheck the live memory proxy (5.5 GiB) and disk (1.5 GiB) before every job.
Pause before launch if unavailable; `--resume` requires the identical immutable
plan. Preserve native stopping, every failure, source/dependency identity,
closed model/predictions and all costs. Materialize only F/G and V-feature files
through checked hard links or disclosed cross-device copies. Pay staging and
translation in the enclosing trial budget. No partial timed-out fit is promoted.

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
