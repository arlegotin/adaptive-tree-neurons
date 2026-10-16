### Spec Compliance

- ❌ Issues found: artifact revalidation omits the physical-batch input contract, allowing an out-of-contract selected recipe through the backend gate (`src/treeneuron/backend.py:137`, `src/treeneuron/backend.py:190`; Important issue below).
- ✅ The implemented live path covers the requested operations, actual compact-layer output/loss/input and parameter gradients, eager float32 AdamW, checkpoint reconstruction and resumed update against CPU64, across three seeds and three magnitudes (`src/treeneuron/backend.py:263`, `src/treeneuron/backend.py:304`; spec §§2/4/5 and Task8 steps1–2).
- ✅ The live calibration preserves logical128, includes the two-row tail, compares accumulation with unsplit updates, times actual epochs/scoring and the complete growth path, and chooses stage batches at the 95% threshold (`src/treeneuron/backend.py:405`, `src/treeneuron/backend.py:445`, `src/treeneuron/backend.py:490`; spec §10 and Task8 step3).
- ⚠️ Public WP00/WP01 decisions are deliberately pending root review, not missing implementation work. `DECISIONS.md` still records the gates as open; append the reviewed artifact identities and preserve the R12 hold only after the issue below is fixed. Causal optimizer migration/fork/resume and complete training-trajectory costs belong to Tasks9/10, rather than this warm dataset-resident calibration (`src/treeneuron/backend.py:405`, `src/treeneuron/backend.py:490`; Task8 context).

### Strengths

- Fresh subprocess probes set fallback0 before Torch import and avoid changing parent RNG/dtype/thread/environment state; the child entry point rejects an already imported Torch process (`src/treeneuron/backend.py:59`, `src/treeneuron/backend.py:513`; `tests/test_backend.py:29`). CPU64 scoring and its copy boundary are explicit, with no float64-MPS or implicit training fallback path (`src/treeneuron/backend.py:263`, `src/treeneuron/backend.py:490`).
- Complete-path selection uses balanced CPU/MPS/MPS/CPU blocks and direct measurements rather than summing stage medians (`src/treeneuron/backend.py:154`, `src/treeneuron/backend.py:490`). The archived final blocks all have stable warmup and timing records. Recorded CPU33.906667ms/MPS57.0170415ms select CPU and physical128 for all phases while logical128 remains fixed (`environment/report.json`, `backend_probes.decision`).
- Numeric validation uses explicit requirements that remain active under optimized Python, binds execution-source/lock/runtime identity, rejects failed operation evidence, and requires exact serialization/resume equality (`src/treeneuron/backend.py:93`; `tests/test_backend.py:159`). Real fault injection verifies that an indispensable indexing exception fails the probe instead of mixing execution (`tests/test_backend.py:215`).
- The resource guard includes measured first-use overhead above the initial9GiB reserve; RSS, MPS counters and unified-memory overlap are represented separately (`src/treeneuron/backend.py:252`; `tests/test_backend.py:241`). The superseded128MiB-allowance records remain archived; actual observed headroom stayed24+GiB, so there is no evidence of a reserve violation. Failed diagnostic stdout now retains a sanitized explanation (`src/treeneuron/doctor.py:40`).

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `src/treeneuron/backend.py:137` — Artifact validation checks that stage row labels equal `report['batch_sizes']` and that the selected label meets the throughput rule, but never checks that the batch list contains positive integers forming a doubling sequence of divisors of logical128. The live `probe_backend` enforces that contract (`src/treeneuron/backend.py:70`), while imported evidence bypasses it. A focused check copied the real final CPU artifact, consistently relabeled its four batch sizes to `[1000,2000,4000,8000]`, and selected8000 in all three stages without changing any measurements. `validate_backend_report` accepted it and `select_backend` returned physical8000 with logical128 (`src/treeneuron/backend.py:190`). Consequently `verify-math --backend-report` can open the gate and expose an untested, out-of-contract integration recipe (`src/treeneuron/verification.py:158`). **Basis:** Task8 public API/input contract, context requirement to accept only actual successfully validated probe artifacts/schema, and spec §10’s safe tested physical batches with fixed logical batching. Enforce the same nonempty positive-integer/doubling/divisor contract during artifact validation, preferably through one shared helper, and regress consistent relabeling plus invalid types/values. The final recorded native measurements are valid; this finding concerns their import/revalidation boundary.

#### Minor (Nice to Have)

- None.

### Assessment

- **Task quality: Needs fixes.** Live numerical/device/timing/resource work is supported by the diff and recorded evidence, but the reusable artifact gate must reject out-of-contract batches before WP00/WP01 can be declared complete.
- **Checks performed:** Read the supplied diff once, task brief/context/report, global constraints, binding spec §§2/4/5/10 and WP00/01, and decision record. No changed source file was reread separately and no broader code crawl was performed.
- **Evidence checked:** Read native focused output (70passed, no skips) and sandbox full-suite output (323passed/16skipped), and verified all nine final/superseded artifact hashes listed in the implementer report. Inspected final JSON statuses, decisions, ABBA block costs and warmup/timing stability; all agree with the report. No suite was rerun.
- **One focused unanswered-risk check:** Tested whether the artifact validator enforces the live physical-batch contract, using a copy of the actual CPU report in memory. Observed acceptance of selected physical8000/logical128; no source, artifact or branch state was modified by that check.
