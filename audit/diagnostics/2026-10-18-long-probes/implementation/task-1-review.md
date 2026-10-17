### Spec Compliance

- ✅ The reviewed source implements the requested diagnostic path; no blocking spec defect found. The supplied package covers all eight files named in the brief. The shipped coordinates and namespaces match the approved recipe (`configs/diagnostics/long-probes-base.json:1`, `protocol/diagnostics/2026-10-18-long-probes.json:1`). The fixed/default probe mode remains the default, while the new mode validates completed-stage evidence (`src/treeneuron/baselines/greedy.py:60`, `:164`). The opt-in graph retains the original parent, adds the ninth role, and checks the long final fork's original weights and optimizer state (`src/treeneuron/pilot/calibrate.py:147`, `:190`).
- ✅ The diagnostic entry point retains ordered membership and failed attempts, separates cost and study samples, binds execution provenance and wrapper arguments, and uses the fixed resource policy (`src/treeneuron/pilot/probe_diagnostic.py:60`, `:169`, `:230`, `:298`, `:322`). Analysis recomputes original-unit risks from saved predictions and original observations, weights teachers equally after repetition averaging, and uses paired whole-teacher resampling (`src/treeneuron/pilot/probe_diagnostic.py:429`, `:439`).
- ⚠️ Cannot verify from this source diff: an actual fresh final-source CPU128 receipt, serial production launches, completed cost/cohort evidence, representative ranking replay, exact restored state/prediction parity, or historical artifact byte preservation outside the changed-file package. These are later execution/evidence checks, not missing executed outcomes in Task 1. The source provides the relevant hooks and artifacts (`src/treeneuron/pilot/probe_diagnostic.py:119`, `:132`, `:219`, `:341`).
- ⚠️ The new bounded integration test exercises requested growth counts zero and one, rather than explicitly forcing an empty eligible score table (`tests/test_pilot_calibrate.py:210`). The changed tests also do not add an explicit tied long-ranking fixture. Existing unchanged coverage was not crawled; the controller should retain the prior rank/count edge-case evidence alongside this change's test evidence.

### Strengths

- The short and long coordinators share a canonical complete-parent-state binding, original score table and per-ID contrasts. The joint long continuation explicitly reloads the original parent, verifies original parameters/Adam state, and participates in the existing warm-fork stream audit (`src/treeneuron/baselines/greedy.py:111`, `src/treeneuron/pilot/calibrate.py:162`, `:190`, `:226`).
- Stopping metadata is checked against the declared policy and retained in both trial reports and the selected checkpoint; the new checkpoint/manifest test checks their growth identity agrees (`src/treeneuron/baselines/greedy.py:164`, `tests/test_greedy.py:264`).
- The cumulative ledger stores references and hashes, while full pair artifacts remain separately retained. F/G/E observations are saved without accessing V arrays; analysis regenerates originals and rejects missing or altered evidence (`src/treeneuron/pilot/probe_diagnostic.py:207`, `:219`, `:409`, `:439`).
- Tests include a real bounded cost-to-study-to-analysis path, unequal retained repetition counts with known zero/negative teacher effects, deliberate resource failures, and deterministic report rendering (`tests/test_probe_diagnostic.py:95`, `:151`, `:188`).

### Issues

#### Critical (Must Fix)

- None found.

#### Important (Should Fix)

- None found.

#### Minor (Nice to Have)

- `src/treeneuron/pilot/probe_diagnostic.py:289`, `:292`, `:307`, `:318`: the enclosing duration and retained-byte snapshot are captured before the final member-ledger rewrite/seal and the resource-decision/seal publication, yet the boundary describes publication and these values drive the forecast. All substantial numerical work and per-member artifact publication are charged, so this is a small accounting-precision gap rather than a numerical validity defect. The implementer report discloses the exclusion. In the later cost-evidence task, retain an outer wall-clock measurement and post-publication byte inventory and explicitly distinguish them from the internal boundary; alternatively include a separate measured publication tail in the source accounting.

### Focused Checks and Evidence

- Reviewed the supplied `5b0483e..b2a9937` diff in one sequential pass; did not derive another Git diff, mutate source/Git state, or dispatch agents. The only written artifact is this expressly requested review.
- Named cross-interface risk: the new patience acceptance might misinterpret a selected checkpoint's stage counters. Checked only `src/treeneuron/train.py:719`, `:800`, `:877`: the trainer records completed execution counters and patience evidence before copying that stage into the G-selected state. The acceptance logic is consistent with that contract.
- Named cross-interface risk: the existing pairing auditor might enumerate only the old warm methods and omit `greedy_long`. Checked only `src/treeneuron/experiments/paired.py:60`, `:76`: it discovers every method with a parent binding and checks its sampler/RNG/schedule/policy and executed common prefix, so the additional warm role is included.
- The implementer report retains behavioral RED evidence and a pristine final affected-file GREEN result, `55 passed in 36.45s`. Earlier setup failures and subsequent corrections are explicitly documented. No suite or numerical experiment was rerun; the two unanswered interface doubts were resolved by focused source inspection.

### Assessment

**Task quality:** Approved

**Reasoning:** The source preserves the scientific comparison, default path, original-parent initialization, complete-search accounting and teacher-level analysis. No source fix is required before the next execution gate; carry the disclosed final-publication accounting distinction into the measured cost evidence and complete the separate receipt/replay/cohort acceptance checks.
