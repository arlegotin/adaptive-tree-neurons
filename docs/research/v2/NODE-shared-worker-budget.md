# NODE shared-worker time budget

The registered P12 shared worker supplies `max_seconds`; the original v2 NODE
adapter accepts only `max_updates`, so its native anchor currently fails before
training. Add a distinct time-budget path for native/search configurations. Keep
the existing explicit update-budget path and engineering fixture bound unchanged.

Native time-budget fits run the same QHAdam, full-F shuffled batches, report100,
rolling5 selection and strict patience5000. They have no added update ceiling.
An absolute monotonic deadline includes configuration/preprocessing/training;
check it between phases/updates and after G evaluation. Overrun raises TimeoutError
and retains a failed/capped receipt. The outer owned-process supervisor still
provides the hard deadline/RSS/cleanup boundary, including noninterruptible kernels.
The shared worker must classify a timed-out fit as timeout, not dependency failure.
No partial timed-out model is promoted or substituted for a completed native fit.

A complete time-budget endpoint records `max_updates=null`; accepting that state
requires native/search mode and proof the recorded trace stopped exactly at
native patience. Legacy finite-update snapshots retain their existing checks.
The request budget is retained in the fit receipt. This is prediction-state
serialization, not interrupted optimizer recovery or historical snapshot migration.

Tests will check the budget parser, expired native shared-worker request before
native allocation, unchanged bounded native-reference trajectories, actual small
NODE loop patience with no update cap, deadline failure and endpoint rejection.
These do not qualify the full2048-tree anchor. No transfer V/T labels, search or
new predictive result is admitted by this compatibility repair.

Verification retained10initial RED cases, followed by28passing budget/worker
checks in4.20s. One further RED showed that a deadline expiring inside an
optimizer update could still start a G evaluation; a post-update check now
prevents that work and retains the applied-update event. Final affected tests
passed153checks with44native warnings in192.79s. They include the unchanged
bounded direct-official comparisons, rolling selection, closed snapshots,
structured corruption and worker/P13/transform checks. The actual small NODE
loop uses a declared fixture recipe and predetermined G predictions to test
strict patience, not to claim native quality. The full2048-tree timed anchor
and complete no-update-ceiling snapshot qualification remain pending. No real
transfer labels, protected outcomes or native-quality eligibility were opened.
See [the receipt](WP07-NODE-worker-budget-receipt.json).

The first full Energy native attempt used03eb296/ca2ad724, passed live admission,
and failed in dependencies before any optimizer update because fresh Torch
inter-op threads were unset. The native guard remained intact. All34retained
files, source/role commitments,3.620184250s charged job time,3.562527542s nested
process time and the reaped child are bound in
[the initial attempt receipt](WP07-NODE-native-anchor-initial-receipt.json).
It receives no predictive loss or eligibility.

A missing helper regression and a separate real fresh-worker wiring regression
both failed. The process-owned worker now explicitly sets Torch intra/inter-op
threads before the neural factory and records their actual values. It does so
only after strict source/input validation; simple/tree methods remain Torch-free.
Adapter caller-state guards are unchanged.39affected checks passed39.88s,
including eight actual own-family workers and two fresh-process regressions;
four optional tree cases skipped because this checkout has no pinned tree
overlay (the isolated qualification checkout retains it). This is separate from
the earlier153-check budget run. The same Energy data/configuration/seed17 is
admitted for one operational rerun under a new explicit source. Full native
qualification remains pending until that run and its inactive audit succeed.
