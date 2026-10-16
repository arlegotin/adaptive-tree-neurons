# Core Task 4 report — 2026-10-16

Implemented on existing `main`, baseline `1897ae3`. No branch/worktree change, subagents, plan/ledger edits, training, predictive experiments, package scoring, or backend validation. Read the Task 4 brief, global constraints, spec §4, TDD/writing-good-tests and verification-before-completion instructions.

## Layer API and behavior

- `treeneuron.models.units.SelectiveAffineLayer(d, h, tau=1.0)` owns `base: Linear(d,h)` and an int64 `grown_ids` buffer. Forward returns preactivations `[B,h]`, with no ReLU. Its selected correction is `tanh(gate(x)/(2*tau))*contrast(x)` through differentiable `index_select/index_copy`.
- `grow(ids, contrasts) -> GrowthManifest` permits exactly one completed event. IDs must be a 1-D CPU int64 tensor. Contrasts must be finite `[m,d+1]` with base dtype/device and explicit final bias coordinate. Sorting permutes IDs and rows together. Validation precedes topology mutation, so a rejected event can be corrected and retried.
- There are no dormant specialist parameters. `gate`/`contrast` are `None` before growth or after an empty event; nonempty growth adds exactly two `Linear(d,m)` modules. Gates start exactly zero; contrast tensors are copied, detached from caller storage. Base parameter objects remain identical.
- Empty growth sets `growth_applied=True` and prohibits a later event without introducing parameters. A zero contrast is accepted for the required dead-insertion negative control; it does not imply a scoring eligibility decision.
- `export_config()` exports exactly `schema_version=1`, `d`, `h`, `tau`, `growth_applied`, and sorted `grown_ids`. `from_config(config)` checks schema/event/ID consistency and reconstructs topology before `load_state_dict`. Save both config and state dict. Caller applies `.to(dtype/device)` before loading to preserve desired precision/device; this topology config does not serialize dtype/device. Empty completed and never-grown configurations remain distinct.

## Migration record, version 1

`treeneuron.growth.types.GrowthManifest` is a frozen dataclass containing:

- `schema_version`, `d`, `h`, tuple `grown_ids`, `growth_applied`.
- Layer scalar counts `old_parameter_count`, `new_parameter_count` (total after event), `added_parameter_count` (difference). A caller-owned head is excluded.
- `old_parameter_names`: the pre-event layer names (`base.weight`, `base.bias`). `new_parameter_names`: only newly introduced names (`gate.weight`, `gate.bias`, `contrast.weight`, `contrast.bias`); empty for zero growth.
- Tuple `parameter_mappings`, each a frozen `ParameterMapping` containing `kind`, `old_name`, `new_name`, `old_shape`, `new_shape`, `old_slice`, `new_slice`. Names are relative to this layer. Slices are half-open `(start,stop)` for each axis. Current growth has full-tensor identity maps only, preserving base objects and old optimizer state; new tensors have no old mapping and need new optimizer state. The representation supports explicit slice maps for a later widening task without implementing widening here.
- `to_dict()` / `from_dict()` support JSON round trips and restore immutable tuple fields, rejecting unknown schema versions. This record stores metadata, not tensors or optimizer state; optimizer migration is a later task. Counts and names are covered against actual layer parameters.

## Verification integration and Task 3 carry

`src/treeneuron/layer_checks.py` runs actual package forwards/backwards against independently built affine-leaf mixture and bypass expressions using shared tensors, not package helpers for expected results. It checks centered insertion, analytic gate gradient, zero initial contrast gradient, nonzero gate gradient, dead zero–zero insertion, scalar-head collapse, hardening discrepancy, ReLU bound/convention, and exact scalar-model counts. It executes in a fresh CPU float64 subprocess, one Torch thread, seed 1602028, N=13/d=3/h=4, tau=0.7; collection preserves caller dtype, RNG and threads.

`collect_math_report()` now includes the real child report/SHA/returncode under `gates.package_layer_math`. Scope is `reference_and_package_layer_cpu_only`, `complete=false`. `reference_checks_passed` retains its A/B-only meaning; new `implemented_checks_passed` requires references and layer checks and controls CLI success. Package score, CPU backend and MPS backend gates remain explicitly `not_implemented`.

Reference validation now checks required schema version/check identity/seed/shapes/constants, CPU float64 environment, nonempty runtime/platform identity, all actual A/B numeric result fields, and the Appendix B array dimensions. It rejects partial success, mismatched identity/environment, missing or wrongly shaped results, nonfinite JSON and violated numerical/descent contracts. Reports from subprocess failure never pass. This does not implement the package score or Adam migration gates.

## TDD and verification evidence

1. Added `tests/test_algebra.py` and `tests/test_growth.py` first. `.venv/bin/python -m pytest tests/test_algebra.py tests/test_growth.py -q` was RED: two missing-module collection errors (`treeneuron.models`, `treeneuron.growth`), the expected missing API boundary in the task brief.
2. Implemented the layer and manifest. Same command GREEN: **37 passed in 0.63s**. Includes shared-coordinate forward/backward checks, scalar collapse, hardening/ReLU convention, exact centered preservation, constructed nonstationary gate and dead insertion gradients, central finite differences away from ReLU kinks (error <1e-10), parameter counts, sorting/caller-storage isolation, validation atomicity, invalid tau, second-event rejection, and JSON/config/state-dict round trips including empty events.
3. Added integration/schema regressions first: **12 failed, 10 passed** against missing API/scope behavior. Isolated the precise existing Task 3 bug before fixing it: `-k partial_success` RED **2 failed, 20 deselected in 0.08s** because both status-only and incomplete Appendix A JSON incorrectly returned `passed`.
4. Implemented fresh-process layer verification, required A/B schemas and CLI gate propagation. `.venv/bin/python -m pytest tests/test_algebra.py tests/test_growth.py tests/test_reference_checks.py -q`: GREEN **59 passed in 15.71s**. Schema mutations copy real A/B reports before corruption, preventing fixture pollution.
5. Final full suite, once after changes: `.venv/bin/python -m pytest -q`: **121 passed in 20.08s**, no skips, warnings or failures. `git diff --check` passed. No subsequent production/test edits.
6. `.venv/bin/python -m treeneuron.cli verify-math --output /tmp/treeneuron-task4-math.json`: exit 0, implemented checks passed, complete=false. Copied the report unchanged to ignored `environment/package-layer-math.json` for local evidence.

## Actual package layer numbers

Native arm64 Python 3.12.13 / Torch 2.14.1, CPU float64, one Torch thread.

| Check | Result |
|---|---:|
| Forward max absolute error | 3.552713678800501e-15 |
| Bypass max absolute error | 2.6645352591003757e-15 |
| Shared-coordinate gradient max absolute error | 2.4424906541753444e-15 |
| Scalar-head collapse max absolute error | 3.552713678800501e-15 |
| Centered insertion max absolute error | 0.0 |
| Gate gradient analytic max absolute error | 8.326672684688674e-17 |
| Initial gate gradient norm | 0.8719377874983594 |
| Initial contrast / dead insertion gradient norms | 0.0 / 0.0 |
| Soft-hard discrepancy identity max absolute error | 3.209238430557093e-15 |
| ReLU bound max violation | 4.969982758673552e-16 |
| Post-mixture / leafwise ReLU counterexample | 0.0 / 1.0 |
| Base / grown scalar-model parameter count (d=16,h=64,m=16) | 1153 / 1697 |

All discrepancies meet <1e-10, and required centered zeros/counts are exact. This supports package layer algebra and growth contracts only. It does not establish float32/backend tolerances, MPS support, optimizer migration/resume, package scoring, generalization, or performance. No blocker observed. Reference scripts remain checkout-based; an installation without them fails honestly. No protocol/frozen artifact or WP completion record was created. This ignored report is local as with previous task reports.

## Evidence SHA-256

| File | SHA-256 |
|---|---|
| environment/package-layer-math.json | `1700281ac72dbbf89cc72768b423abe4d9979448f8b3c7c3ad42f7c298ce4bac` |
| src/treeneuron/models/units.py | `15a97c20c5dec6856e2918c8a2bd4832e3bd9d17448b7cbdec255468733765e3` |
| src/treeneuron/growth/types.py | `bf3024d403b17cb510b3dba86b62e11639059c17612684681d1ea5ed3f9ad115` |
| src/treeneuron/layer_checks.py | `c342d27db7c007ecd9d6e5533eb0437494a8c10ba94af6627a3eb652809c049a` |

## Commit

`a5790ff` — `feat: add compact function-preserving selective affine layer` (10 task-owned files, 686 insertions / 13 deletions). Initial staging was denied at `.git/index.lock`; the authorized escalated staging and commit succeeded. The ignored local report and evidence JSON are intentionally outside the commit. Working tree was checked after commit.
