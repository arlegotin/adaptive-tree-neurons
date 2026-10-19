# Native tree anchor execution qualification

The corrected worker completed the full CatBoost anchor on one retained primary
synthetic development problem: F2306/G577 and577query features,16predictors,
fit seed17. It executed10000 rounds and retained round9994. Reaching the native
maximum is a completed bounded recipe, not proof of convergence. No V labels or
T were read and no comparative generalization score was computed.

The initial attempt completed training/save but failed at worker snapshot reopen.
Its21.793681958s paid time and partial model remain retained. Four native worker
regressions reproduced the missing `manifest.json` assumption; tree snapshots use
`state.json`. The correction leaves native training and serialization unchanged.
The87-check affected suite passed with6optional snapshot-mutation skips. The
[correction receipt](WP07-tree-worker-snapshot-correction.json) binds those facts.

One same-data/configuration/seed infrastructure rerun under commit
`702479eb80d7f2a03dacd140c8d75286ab9fdf5f` and scientific Python digest
`17cbebfcacf71ba776eef8d763b3b544a842582b5d3fdbb5c32f1e26d2895bad`
completed with exact closed reload. Independent pinned official inference on the
old partial model and new model produced identical577query predictions, also
matching the worker. That is inference parity, not independent full-training
reproduction. Both native and worker source identities remain explicit.

The rerun charged22.092602042s; failed plus recovered searches charged
43.886284000s. Native training was19.672747625s inside the rerun, whose launcher
body was22.147426959s. These nested clocks must not be added together. The sampled
root RSS peak was223674368bytes; the worker lifetime peak was204062720bytes.
Those are distinct measurements, neither strict aggregate descendant memory nor
dedicated deployment timing. Both owned children were reaped. See the
[full-shape receipt](WP07-P08-native-primary-shape-receipt.json).

Raw attempts, commands, source/input hashes, native configuration/history,
snapshot files and official-inference proof are retained under
`artifacts/research-v2/WP07/P08-native-primary-shape-b0e4c8e/` and
`artifacts/research-v2/WP07/P08-native-primary-shape-recovery-702479e/`.
These local raw artifacts are required for exact replay of this diagnostic;
their presence in a clean public checkout is not asserted. The original M2 data
files were independently byte/array authenticated and repackaged for this worker;
no older checkpoint was loaded under a changed source identity.

## Actual transfer anchor qualification in progress

A separate immutable queue now contains24engineering anchors: CatBoost, XGBoost,
constant-leaf LightGBM and linear-leaf LightGBM on all6prepared datasets. Only
outer0is admitted for nontemporal data; Bike uses outer4. The existing
`TransferStore.development` access boundary releases F/G labels and V features;
secondary development, V labels, T features and T labels are not released here.
The original preparation audit and all6manifest hashes are externally bound.

Each anchor retains its full native configuration, has a300s enclosing measured
engineering deadline and a1GiB sampled root RSS stop. This shorter qualification
deadline does not replace the prospective common120-minute practical search
budget. A native anchor that times out remains an engineering cap, not a scored
predictive defeat. One worker runs at a time beside the independent M2 worker.
Before every anchor, measured memory and disk proxies must exceed1.75GiB and1GiB;
unknown or insufficient availability pauses the immutable queue. Proxies are
not allocation guarantees. No alternative seed, shortened recipe or automatic
failure deletion is admitted.

The parked execution checkout is `/private/tmp/atn-research-v2-next` at702479e;
its source must remain unchanged while this queue runs or resumes. Local launch,
plan, access records, per-job resources, complete native snapshots/failures and
paid clocks are under
`artifacts/research-v2/WP07/tree-primary-anchors-702479e/`. A resource-only pause
resumes the identical launcher with `--resume`; infrastructure faults need their
recorded diagnosis before a same-seed recovery. Source changes require a new
identity and disposition of affected work.

At19October11:45UTC,4anchors had completed as engineering-ineligible and the
Bike CatBoost anchor was running. A complete terminal audit is still required.
All worker quality/protected flags and GateD remain false. This is native recipe,
shape and resource qualification; it is neither the168primary practical searches
nor a modern predictive benchmark, and it cannot select a winner. Missing native
recovery, production registry/readiness and remaining mandatory adapters still
block a complete practical claim.
