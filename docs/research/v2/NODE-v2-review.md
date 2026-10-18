# NODE-v2 independent review and correction

18 October 2026. Reviewer `/root/diagnostic_review` read held commit
`6c518a6ca77ea61b364af11496583f90cefb38c7` in the isolated P12 branch.
The review authenticated137 source/lock hashes, four owned hashes, seven unchanged
pins,19 logs and the821-file/44,107,588-byte raw inventory. The reviewer performed
no fits, tests, native imports, edits or additional delegation. Root approved one
combined RED→GREEN correction pass; no second review is used.

## P2: construction defaults controlled initial native draws

The held `_build_model` called `.cpu().float()` after DenseBlock constructors.
Pinned ODST creates response/feature-selection tensors without explicit dtype or
device, then applies native random initializers. A caller float64 default therefore
changes initialization draws before the cast; a non-CPU default can consume an
unmanaged device generator or fail when copying a meta tensor. Seeding/restoring
only the CPU generator did not close this boundary.

Two regressions use float64 and meta caller defaults. Both first reached a trapped
DenseBlock constructor instead of rejecting before draws. The correction rejects
unsupported caller defaults before seeding/preprocessing/native construction.
It reads the same pinned Torch device-context metadata as `get_default_device`,
without that function's possible off-device empty allocation to resolve an index.
The valid constructor additionally runs under scoped CPU device context. No caller
dtype/default-device policy is modified. The two newly executed Torch source files
are included in runtime file commitments. These tests require no constructor call,
unchanged caller policy/NumPy/Python/CPU RNG and previous predictor, and a paid
failed receipt with zero completed updates.

## P2: same-step optimizer copies could disagree

When G selects the final executed report, selected/raw optimizer states are copies
of the same newest raw QHAdam member. The selected model can be a rolling average,
while those optimizer slots remain the raw member's slots. Held validation checked
the two states' ages and geometry independently but did not require equality.

Two regressions alter only one selected first/second moment in a12-update,
selected-at12 endpoint, preserving finite float32 shape, valid age, nonnegative
second moments, NPZ/file inventories and a newly trusted external manifest hash.
Both first loaded successfully. Validation now compares canonical typed optimizer
metadata and every array's exact dtype/shape/byte identity when selected and final
steps coincide. It rejects both resealed edits in the same-step optimizer domain.
This enforces retained endpoint consistency; it does not authorize training resume.

The combined reviewed regressions produced **4 failed,43 deselected in20.90s**,
then **4 passed,43 deselected in16.13s**. Original logs and every raw artifact remain
retained. The final affected run and final source/artifact commitments are recorded
separately in `NODE-v2-evidence/final-receipt.json`: **141 passed,44 warnings in175.15s**, including47 NODE cases.

## Deferred minor and scope

Terminal Python MT state currently bounds the final index but does not require an
exact integer type; Torch RNG state checks byte geometry without asking a native
generator to validate its structure. This metadata hardening is deferred. Loaded
terminal RNG evidence is never installed for prediction; caller RNG is saved and
restored independently. The artifact remains prediction restoration only, with no
mid-fit interruption/resume claim.

No other material native/preprocessing/selection/serialization issue was reported.
Native vendor kernels, legacy guards and frozen core dependencies remain unchanged.
Full2048 anchor, finite transformed workloads, hard resource/recovery harness,
prepared transfer fits, search/HPO, registry integration and GateD remain pending.
