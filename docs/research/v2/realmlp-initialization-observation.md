# RealMLP initialization observation contract

The epoch-row observer starts after initialization and cannot by itself
report initialization-only RAM-limited row selection. This slice now observes that
separate native policy. It does not activate native quality eligibility, alter
the256-epoch anchor, or change any initialization/optimizer/sampler equation.

Use a scoped fresh-worker hook around pinned NNCreator.create_model,
Fitter.fit_transform_subsample and DictDataset.get_sub_dataset. Native methods
perform every draw and transformation. Observe the actual indices passed by the
native default RAM sampler, with their input-local and composed F-table positions.
Do not reconstruct a random draw or call it twice. The creator's F selection must
equal the declared contiguous F positions; G positions are forbidden. Sequential
fitter outputs preserve the row mapping except for observed native subsets.
Unexpected row selectors/lineage fail qualification rather than guessing.

Record each fitter's type/scope, call tree, input/output counts, actual RAM budget,
mapping hashes, status and any selected indices. Keep the initial full F mapping
once, not duplicated for every no-subset call. These are positions within the F
table; the coordinator separately retains raw observation IDs. This scope covers
the named native RAM sampler, not every possible parameter-initialization draw.

Hooks restore their exact original methods on success/failure and reject nested
or concurrent installation. A standalone engineering fixture may bind an F-only
DictDataset to force native RAM subsampling on a tiny matrix. It is labeled
separately from observing the real creator. Native parity must compare unchanged
versus observed tensor outputs/parameters and post-call RNG states, including a
sequential nested subset, no subset, exception cleanup and complete tiny-model
initialization. Full workload/quality/resource/recovery gates remain pending.

## Executed evidence and integration

Five missing-module assertions failed before implementation. The first native
fixture then exposed two harness mistakes: an omitted private overlay, followed
by a repeated already-set Torch interop setting. Both failure logs remain. The
corrected component passed5 checks in8.23s. A separate adapter integration check
failed on the missing initialization summary, then passed in21.83s with exact
direct-official predictions and fresh-process reload.

The final affected command passed64 checks in516.71s. It includes the native
graph cases, prior saved-state/runtime rejection checks and native sampler
support. This is a focused suite, not a full-repository pass. One existing PLR/
SELU case still has the declared native numerical failure before any optimizer
update; its passing software check verifies the retained failure record. That
failure is not replaced by a successful model or removed from the search support.

The component directly compares plain versus observed pinned native execution:
actual32-to8 selection, sequential32-to8-to4 selection, no selection, and a
controlled fitter exception. Output tensors and post-call RNG states match
exactly; hooks restore after success and failure. A complete one-epoch native
model also matches all22 stored tensors and RNG state exactly. A real creator
request containing a G row rejects before the first data-dependent fitter.

The real tiny model records26 nested fitter calls and no RAM subsampling at the
default1GiB setting. Forced tiny-RAM fitter references demonstrate capture when
the native policy activates; they are not full-architecture resource evidence.
Both input-local indices and composed F positions are retained. Successful
adapter summaries now contain initialization_observation; failure summaries also
retain it. Context status failed records failure of the enclosing fit and does
not, by itself, identify initialization as the cause.

The exact package digest and per-file/raw evidence are in
[WP07-initialization-observation-receipt.json](WP07-initialization-observation-receipt.json).
The source snapshot was taken during the affected run and agrees with the
source-bound native attempt records. The two early failed fixture temporary
directories expired before raw copying; their complete failure stdout is retained.
Successful component and final affected raw artifacts were copied and hashed.

Full256-epoch anchors, actual transfer/resource qualification, hard-worker and
recovery integration, production eligibility, HPO, deployment and GateD remain
pending. Every RealMLP mode still returns quality_eligible=False and
checkpoint_eligible=False. No prepared transfer experiment or protected T was
used. The running M2 source stays fixed and does not import this module.
