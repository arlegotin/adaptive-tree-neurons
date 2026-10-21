# Fresh independent review response

Reviewer: `/root/final_source_review`, `gpt-6-astra`, `xhigh`.
Reviewed: `230e6575149c9fee440ca301fa3782a85751bac8` to
`783cb2d85d99abbd8e99b156ab7a5ed5ca69266c`.
Received21 October2026. The response below is retained from the reviewer's final message.

## Strengths

- The exact joint selector preserves zero-gradient candidates and deterministic ties, and distinguishes ridge improvement from geometric span. The focused selector and transfer tests passed **36/36**.
- Model and Adam tensor validation checks shapes, dtypes, finiteness, second moments, and parameter ages before loading can cast them. G-best restoration preserves the outer attempt history and consumed work.
- The sampled split and access paths enforce group minima, the24-clock-hour temporal embargo, and primary-only development barriers. Missing qualification or prediction evidence blocks the relevant claim.
- Scientific reporting retains the failed M3 Gate B, labels P00 as the legacy procedure, and separates software evidence from unrun native and protected results. Native stopping and descriptive budget prefixes follow the recorded ruling.

## Issues

Critical: none found. Minor: none retained.

### R1 — Important: incomplete snapshot validation

Incomplete snapshot validation allows publication of an unusable current checkpoint or a corrupt embedded best.

Locations at the reviewed commit: `checkpoint.py:159`, `checkpoint.py:225`,
`train.py:653`, `train.py:682`.

Publication validates metadata, model tensors and Adam history, but sampler permutation, sampler-generator state and agreement between serialized optimizer options and metadata are checked only during `unpack_state`. The loader unpacks only the current snapshot; embedded-best state passes the same incomplete validators and is attached without these checks.

I reproduced this using a tiny, manually constructed state with **zero training updates**, the actual `_publish_snapshot` writer and `load_checkpoint`, and temporary files. Each case began with a valid pointer, then corrupted either current or embedded-best state:

| Mutation | Current state | Embedded best |
|---|---|---|
| Sampler order converted to float64 | Pointer replaced; load rejects permutation | Pointer replaced; load accepts |
| Sampler order replaced by duplicate zeros | Pointer replaced; load rejects permutation | Pointer replaced; load accepts |
| Generator state replaced by one uint8 byte | Pointer replaced; load rejects generator state size | Pointer replaced; load accepts |
| Serialized optimizer learning rate changed without matching metadata | Pointer replaced; load rejects optimizer options | Pointer replaced; load accepts |

This violates the explicit before-publication requirement. A malformed current state replaces a working recovery pointer with one the normal loader cannot use. Malformed best state remains latent until restoration.

Recommended correction: use one complete, side-effect-free snapshot validator before publication and before reconstruction, applying it to both snapshots. Move sampler and optimizer-option checks into that boundary and validate generator state without changing global RNG. Add regressions asserting that every demonstrated corruption rejects before pointer replacement, preserves RNG, and rejects when placed in embedded-best state. Preserve valid exact replay.

### R2 — Important: nested validation amplification

Nested public validation multiplies complete primary and native audits across the all-840 path.

Primary location at the reviewed commit: `qualified_practical_models_v2.py:120`.
The source establishes this call chain:

1. Complete-model reconstruction audits all672 secondary searches.
2. Each secondary audit loads the global barrier in `_frozen_declaration`, then loads it again when constructing `ReleasedTransferStore`: `secondary_native_search_v2.py:240`, `released_transfer_v2.py:125`.
3. Each global-barrier load reconstructs the qualified primary inventory: `practical_global_freeze_v2.py:97`, `practical_global_freeze_v2.py:203`.
4. Each primary reconstruction audits all168 primary searches: `qualified_primary_models_v2.py:83`.
5. Every primary-search audit opens the complete native registry, whose reader inspects all120 anchors and258 supports: `qualified_search_audit_v2.py:247`, `native_qualification_registry_v2.py:181`.

Consequently, one successful complete-model reconstruction necessarily performs at least `672 × 2 × 168 =225,792` primary-search re-audits. Those imply at least27,095,040 anchor inspections and58,254,336 support inspections, before additional final guards and repeated downstream protocol/cost/report loads. This is a conservative source-derived bound; further `_unchanged` and `compile_again` calls add work.

Secondary candidate callbacks also reconstruct the global barrier before and after each attempt (`secondary_native_search_v2.py:132`). That work enters the paid callback clock and can consume descriptive prefix budgets.

The demonstrated defect is the multiplicative validation structure at the required workload size. **I did not measure production wall time or claim a timed native failure.**

Recommended correction: establish a genuine immutable private validation context inside each actual public invocation and reuse its already validated prerequisites throughout nested operations. Retain source/input guards before and after the invocation and checks for each selected input. The recorded ruling permits this. Avoid persistent caches, caller-supplied qualification flags, or reduced native evidence requirements. Add structural call-count checks at representative inventory sizes to prevent repeated whole-program reconstruction.

## Recommendations

Address R1 and R2 in the material fix pass before final source freeze. Validate checkpoint rejection and replay directly, then verify bounded validation composition while preserving all authority and source-change guards. Any execution-source change requires fresh source-bound software evidence; earlier receipts remain historical evidence.

## Coverage and limits

Reviewed range: `230e6575149c9fee440ca301fa3782a85751bac8` → `783cb2d85d99abbd8e99b156ab7a5ed5ca69266c`.

I reviewed the request package, binding spec, plan, relevant ledger rulings, readiness documentation and scientific report. Source passes covered checkpoint/state reconstruction, joint selection/Jacobians, diagnostic forks, split/access controls, queues and paid searches, qualification and global/model barriers, and the practical protocol, prediction, evaluation, audit, cost, deployment, report and CLI composition.

This was a whole-branch architectural and targeted semantic review, **not a line-by-line certification of all374 changed source/test files or generated artifacts**.

Fresh checks comprised the eight isolated checkpoint reproductions and `tests/test_joint_selector.py`, `tests/test_transfer_contract.py`:36passed in0.46s.

I did not rerun the supplied751-case profiles or claim a current whole-suite green. No native fits, protected T-label reads, fake successful qualification gates, source/index/environment edits, or worker operations occurred. HEAD remained783cb2d; final status showed only the supplied untracked final-review directory.

## Declined to judge

- Actual final-source native qualification, all840 searches and protected positive results: expressly unrun prerequisites; their absence alone is not a software defect.
- Production wall time and memory feasibility in numerical terms: no heavy run was authorized. R2 establishes call multiplicity, not measured runtime.
- Independent reproduction of historical scientific results and every native recipe: this review did not rerun scientific studies or requalify vendor methods.
- Predictive superiority, dataset-population generalization and dedicated precision-qualified efficiency: these claims are expressly withheld.
- Additional M3 remedies or M4/M5/M6/C0/C1/C2 implementation: the recorded failed gate did not admit those branches.
- Held c33 worker outcome: independently running under its own source and explicitly outside this review.

## Assessment

Ready to merge? **With fixes.**

R1 and R2 should be corrected before freezing the execution source for expensive qualification. The reviewed code has substantial state, role and evidence safeguards, but publication still accepts malformed recovery state and the positive practical composition repeats global validation at a problematic scale. Native scientific readiness remains a separate, unrun requirement.
