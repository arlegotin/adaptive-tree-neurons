Spec compliance: Issues found — the new causal reconstruction path accepts incomplete trained Adam state, violating the strict recovery requirement.
Code quality: Needs fixes.

Strengths

- `src/treeneuron/baselines/causal.py:33` counts the complete scalar head when bracketing ordinary capacity; `src/treeneuron/models/static.py:26` implements compact active centered/direct-leaf coordinates. `tests/test_causal_baselines.py:196` checks shared-coordinate forward and chain-rule equivalence without claiming equal independent Adam trajectories.
- `src/treeneuron/baselines/causal.py:55` maps old names/full slices, copies old values/moments/steps, registers separate new parameters with zero Adam age, preserves the absolute schedule, and restores training RNG. This satisfies the sixth ruling as chronologically clarified by root: old values, moment/step values, optimizer age, schedule, and update semantics must survive; the explicit mapping permits replacement Parameters. Lines 71 and 97 do build replacement old Parameters and clone their optimizer tensors; the report explicitly describes this. Independent factory forks appropriately do not alias the source parent.
- `src/treeneuron/baselines/causal.py:127` validates original F/G arrays and manifest commitments without accessing V, and line 225 enforces finite original-unit scalar predictions. Runtime DevelopmentData is explicitly separate from persisted scientific recipe configuration.
- `src/treeneuron/baselines/causal_evidence.py:47` probes actual topology using an independent affine reduction path, gradients, Adam updates, and exact serialized next-update replay. Line 102 validates source/topology/shape/count receipts. Added controls explicitly disclaim calibrated performance.
- `tests/test_causal_baselines.py:233` exercises durable interrupted recovery for fresh widths 94/95, both full static coordinates, and warm widths 94/95 with a short batch and cosine schedule. `src/treeneuron/baselines/causal_checks.py:21` executes concrete B0 mean/ridge and 13 B1–B5 recipes, with B0 spline, B6/B7, and duplication explicitly pending/deferred.

Critical findings

- None.

Important findings

- `src/treeneuron/train.py:547` — Missing or empty saved Adam records are unconditionally accepted as lazy state. The later check at line 572 delegates to `growth.insert._validate_optimizer`, which likewise permits missing state without checking whether that parameter has already trained. Consequently, a corrupted/incomplete causal checkpoint can pass reconstruction while silently discarding an old parameter's moments and Adam age. A focused mutation of a retained B2 widened diagnostic checkpoint at selected update 4 removed one old record whose step was 4; `unpack_state` accepted it and returned 6 Adam records instead of 7. The next optimizer step initializes that parameter from step zero, defeating strict recovery and the widening control's declared state preservation. Require complete records for parameters whose state must exist, allowing lazy absence only where the saved update/growth history actually permits it; added widening tensors are explicitly initialized even at step zero. Add a focused rejection test for deleted and empty trained records while retaining valid update-zero behavior.

Minor findings

- None.

Cannot verify / integration boundaries

- Fair total fresh/continuation allowances, enclosing cost/hardware provenance, comparative G/V results, native MPS controls, throughput, and protected evaluation/freeze remain Task12/evidence-plan work; this change does not establish them. The old native receipt is historical; integration must explicitly select the refreshed CPU receipt.
- Existing shared training selection semantics and full protected-evaluation architecture are not re-audited beyond the supplied hunks; they span prior tasks. No T access or predictive study is introduced here.

Checks and scope

- Reviewed supplied `review-c272fc9..d8c5591.diff`, base c272fc9 / head d8c5591, in bounded chunks. Tool-output truncation required recovering only the clipped widened constructor/train boundary; no separate changed-source reads or git reruns.
- Named cross-file risk: the new reconstruction path's call to existing optimizer validation might accept absent/malformed state. Checked only `src/treeneuron/growth/insert.py`'s optimizer/manifest contract for that risk. Its unconditional lazy-state allowance corroborates the finding.
- Ran one narrow in-memory checkpoint mutation using a retained CLI artifact: loaded B2 widen, packed state, removed the first Adam record, and called unpack_state. Result: selected_update=4, removed_step=4.0, original_state_count=7, loaded_state_count=6; no suite or numerical probe rerun and no evidence artifacts mutated.
- Read `environment/task11-pytest.txt`: 520 passed, 48 skipped in 143.46s, with no warnings shown. Read final boundary log: 13 passed, 28 deselected in 5.82s. CPU refresh log reports CPU selected with physical/logical batch128; no native throughput claim inferred.
- Root independently verified all 230 artifact-index hashes, source digest `9ba92d00f5f2572913fc58cf08500286056de1044edb716bc66102b6ae8b424d`, final CLI success/13 recipes, and preservation of the historical native receipt. Index SHA256: `b7509664972f71788c81c0f2a4db6202db0571dea6c02a3073106049121ce3bb`.

Assessment: Needs fixes for incomplete Adam-state acceptance. Capacity, coordinate, F/G provenance, explicit pending capabilities and bounded CPU evidence integration otherwise match the task's implemented scope.
