# Core Task 8 report: actual CPU/MPS kernels and batching gate

Implemented on current `main`, based on `7d637f59a9ebd8d5232e363e0d4a483e88fb06c3`. Final task commit: `2f03c821ad6456a8d16cb2f0d61c24fe29f8bf7e`. Root owns the public WP00/WP01 decisions, plan ledgers and documentation. No branch/worktree change, subagents, dependency/system/runtime changes, watermark overrides, protected T access, freeze, predictive comparison or forecast was performed.

## Result and stable Task 9 integration surface

Both real CPU and MPS passed mandatory eager float32 operations, CPU64-reference parity, indexed autograd, `AdamW(foreach=False)`, checkpoint reconstruction and exact resumed-update parity. Final native ABBA evidence selects **cpu** for the measured d16/h64/F2306 workload: CPU **33.9066670 ms**, MPS **57.0170415 ms** per directly timed base-epoch -> CPU64 copy/score -> matched growth -> grown-epoch path. These are medians of two block medians. Backend selection follows measured complete-path cost; MPS availability alone is never a pass or selection criterion.

Selected physical batches: `{"base_epoch": 128, "grown_epoch": 128, "score": 128}`. Logical training batch stays **128**; scoring batch is an independent CPU64 memory/throughput choice. Each 2306-row epoch has **19** updates, including a final **2**-row logical batch. Microbatch losses use `sum(square losses)/actual logical rows`; accumulate gradients then perform one AdamW step. Maximum accumulated-versus-unsplit parameter difference across base/grown epochs and tested batches is `2.9802322387695312e-08`, below the fixed `1e-5` gate.

Public APIs:

- `treeneuron.doctor.probe_backend(device: str, *, seed: int, batch_sizes: Sequence[int]) -> dict[str, Any]` (implemented in the focused `backend.py` helper, re-exported by doctor). Supported devices are CPU/MPS; invalid devices return an explicit failed report. Batches must double and divide logical128; the bounded default is `(16,32,64,128)`.
- `treeneuron.backend.collect_backend_report(*, seed=1602030, batch_sizes=DEFAULT_BATCHES)` returns `schema_version`, `status`, `probes`, `comparison_blocks`, and `decision`. `decision` is the small stable integration surface: selected training/scoring backend, selected physical batches, logical batch128, float32 tolerance, measured whole-method costs, ABBA order/block costs, exclusions and workload scope. Read actual probe `environment.threads` (**1**) when launching the selected CPU recipe; doctor's untimed parent Torch thread default is a separate observation.
- `validate_backend_report(report)` rejects unknown/incomplete schemas, operation/trial omissions, failed numeric errors, invalid batch choices, source/lock/runtime drift, fallback not disabled and malformed complete-path timing. Checks remain active under `python -O`.
- `select_backend(probes, *, comparison_blocks=None)` selects one complete training backend; indispensable unsupported paths exclude the whole backend. Mandatory CPU failure raises an error. Scoring always returns to explicit CPU64, including signed eigendecomposition.
- `collect_math_report(*, backend_report=None)` runs actual reference, package-layer and package-score checks plus validated backend evidence. `complete` and `implemented_checks_passed` require all mandatory gates. Unavailable MPS may select a validated CPU path; it stays unavailable rather than passed. `verify-math --backend-report <doctor.json>` revalidates the full existing artifact and recomputes the decision, rather than trusting a stored status or decision.

## Import/device and timing boundaries

Each probe runs in a fresh subprocess with `PYTORCH_ENABLE_MPS_FALLBACK=0` set **before Torch import**. The probe rejects an already imported Torch process at its child entry point. Parent RNG, default dtype, threads and environment are preserved; fixture and aggregate tests check actual calls. Native focused tests also start with fallback0. No float64 tensor is sent to MPS; scoring copies the model to CPU first and only then casts double. No selection-matrix fallback was necessary: actual `index_select` and `index_copy` forward/backward passed directly.

Training and optimizer probes use float32 eager execution. The reference uses the same original float32-generated tensors/parameters converted to CPU64, at seeds1602030/1602031/1602032 and input magnitudes0.1/1/10. Kernel cases check GEMM, tanh, sigmoid, ReLU and indexing outputs plus input backward; the actual compact grown layer additionally checks preactivation, scalar prediction, loss, parameter/input gradients, first AdamW update and resumed second update against CPU64. Optimizer constants in these synthetic probes are lr0.001, weight_decay0.01, default eps1e-8, `foreach=False`. Serialization and resumed-update rows separately require exact within-backend equality (zero error).

Calibration uses primary-sized d16/h64/F2306 synthetic tensors, an ungrown base epoch and a grown16-unit epoch, actual `score_units` with unit_block_size16, actual Adam allocation and matched insertion. Complete-path timing executes those stages in order; it is **direct timing**, not a sum of unrelated stage estimates. Fresh AdamW state per phase is explicitly part of this calibration boundary; migration of a causal parent's optimizer history belongs to Task9. Individual epoch timings include model reconstruction/device copying and fresh optimizer creation. F tensors are already resident on the training backend and CPU64 fitting inputs already exist for scoring; initial dataset transfer/import/load and first initialization are outside these warm timing regions. This is not a cold-start or full training-trajectory estimate.

`perf_counter_ns` brackets every timed region. MPS synchronizes immediately before and after the region, and around the separately recorded model-to-CPU64 copy boundary. Warmups are retained as discarded work: 3..9 repetitions, stopping when last3 range/median<=20%; then5..11 timed repetitions until last5 range/median<=20%, with capped instability explicitly retained. Final native stage and complete-path timing records all met that stability criterion. CPU/MPS comparison uses sequential **CPU -> MPS -> MPS -> CPU** fresh-process blocks. All block reports, warmups, sample arrays, copy boundaries, UTC and resources are archived. There were no failed native benchmark attempts or timing exclusions in the final calibration.

## Final measured stage and batch costs

Each cell is median milliseconds; all tested batches were safe. Throughput is `2306e9/median_ns`, recorded per row in JSON. The selected stage batch is the smallest tested safe batch attaining>=95% of that stage's maximum median throughput; all final blocks selected128 for all three stages.

| Block/backend | Phase | b16 ms | b32 ms | b64 ms | b128 ms |
|---|---|---:|---:|---:|---:|
| 1/cpu | base_epoch | 9.406375 | 5.792209 | 3.675000 | 2.586583 |
| 1/cpu | grown_epoch | 18.091291 | 10.393625 | 6.598625 | 4.531792 |
| 1/cpu | score | 125.200875 | 68.693917 | 41.200000 | 25.711917 |
| 2/mps | base_epoch | 34.447208 | 21.336500 | 14.260708 | 10.890708 |
| 2/mps | grown_epoch | 63.447542 | 38.567542 | 26.239666 | 19.072792 |
| 2/mps | score | 128.376166 | 71.057750 | 42.372042 | 28.584416 |
| 3/mps | base_epoch | 34.660458 | 23.299084 | 14.286834 | 11.165417 |
| 3/mps | grown_epoch | 65.430292 | 39.657500 | 26.635500 | 19.509542 |
| 3/mps | score | 127.960792 | 71.020500 | 43.493625 | 28.366250 |
| 4/cpu | base_epoch | 9.592667 | 5.600709 | 3.755875 | 2.565417 |
| 4/cpu | grown_epoch | 18.548750 | 10.370666 | 6.565250 | 4.568875 |
| 4/cpu | score | 125.441583 | 69.307292 | 41.411208 | 25.137416 |

| Block/backend | UTC start | Whole path ms | CPU64 model-copy median ms | Samples / discarded warmups |
|---|---|---:|---:|---:|
| 1/cpu | 2026-10-16T14:54:34.898429+00:00 | 34.582500 | 0.257792 | 65 / 42 |
| 2/mps | 2026-10-16T14:54:39.741341+00:00 | 57.525208 | 2.634688 | 65 / 39 |
| 3/mps | 2026-10-16T14:54:46.802883+00:00 | 56.508875 | 2.667792 | 67 / 41 |
| 4/cpu | 2026-10-16T14:54:54.040108+00:00 | 33.230834 | 0.256188 | 67 / 41 |

## Tolerance and actual CPU64-reference errors

The fixed pre-native binary32 budget is `atol=64*eps=7.62939453125e-6`, `rtol=256*eps=3.0517578125e-5`, eps=`2^-23`. This budget covers short d<=16/h<=64 reductions and chained gradients, with an absolute bound near reference zeros. Nine seed/magnitude pairs per operation empirically validate the bound against CPU64, including a CPU32 baseline; no MPS-specific allowance or result-dependent loosening was applied. Acceptance is elementwise `abs(actual-reference) <= atol+rtol*abs(reference)`; normalized error is the maximum ratio of the two sides. Table values aggregate all final ABBA trials by backend. Raw relative error uses `max(abs(reference),1e-12)`; saturated sigmoid/tanh gradients can have relative error1 at negligible absolute magnitude, so the recorded normalized metric controls acceptance. No tolerance hides a failing or systematic mismatch.

| Operation | CPU max abs | MPS max abs | CPU max relative | MPS max relative | CPU max normalized | MPS max normalized |
|---|---:|---:|---:|---:|---:|---:|
| adamw_resumed_step | 1.16022221e-07 | 1.0857164e-07 | 7.05836549e-06 | 8.08761276e-06 | 0.0111415888 | 0.0104261111 |
| adamw_step | 1.11522574e-07 | 1.04071993e-07 | 1.22577667e-05 | 7.02252984e-06 | 0.0107328544 | 0.0100158158 |
| gemm | 1.45424181e-06 | 1.45424181e-06 | 0.000110451323 | 0.000110451323 | 0.05229777 | 0.05229777 |
| index_copy | 4.76837158e-07 | 4.76837158e-07 | 2.06683272e-07 | 2.06683272e-07 | 0.00246434417 | 0.00246434417 |
| index_select | 3.97364298e-08 | 3.97364298e-08 | 7.47225104e-08 | 7.47225104e-08 | 0.00136522776 | 0.00136522776 |
| input_gradient | 1.48697354e-08 | 1.19741492e-08 | 0.000138697452 | 0.000138697452 | 0.00169272274 | 0.00135015349 |
| layer_output | 3.02088725e-06 | 2.94011514e-06 | 0.00036498379 | 0.00034514196 | 0.0698584408 | 0.0690920315 |
| loss | 3.98102288e-07 | 2.65120825e-07 | 1.08640842e-07 | 1.08640842e-07 | 0.00313095408 | 0.00313095408 |
| parameter_gradients | 3.48856429e-06 | 2.85752638e-06 | 0.00187215883 | 0.000902711765 | 0.0668623562 | 0.0753428326 |
| prediction | 1.04361075e-06 | 8.83940928e-07 | 1.09984651e-05 | 1.09984651e-05 | 0.0168343237 | 0.0312642072 |
| relu | 9.71814862e-09 | 9.71814862e-09 | 7.7683177e-08 | 7.7683177e-08 | 0.000809349548 | 0.000809349548 |
| resumed_update | 0 | 0 | 0 | 0 | 0 | 0 |
| serialization | 0 | 0 | 0 | 0 | 0 | 0 |
| sigmoid | 8.4100381e-08 | 8.4100381e-08 | 1 | 1 | 0.0023795424 | 0.0028454256 |
| tanh | 5.81805778e-08 | 1.17212567e-07 | 1 | 1 | 0.00152516901 | 0.00344505396 |

## Resources, reserve and superseded calibration

The final native report observes Apple M3 Max/36GiB unified RAM/14 logical CPUs, AC power, native Python3.12.13/Torch2.14.1. Probe CPU thread count is1, interop count is separately recorded. RSS, peak RSS, MPS allocated/driver/recommended working-set limit are separate overlapping counters; they are never added as independent RAM. Current memory pressure and swap plus raw `vm_stat` observations appear at every batch attempt and before/after each backend block. Unavailable/denied counters are null with reasons, never zero substitutes. `ps` RSS is sanitized to the probe's own numeric counter only; serial/UUID/UDID diagnostics remain redacted.

The initial reserve remains **9GiB**. The resource guard requires normal observable macOS pressure and current pressure-derived headroom of at least9GiB+512MiB. Headroom is explicitly a coarse `memory_pressure` free percentage times physical memory, not the count of free VM pages. The512MiB allowance follows the observed ~250MiB first-use MPS RSS growth and exceeds twice that observed growth; it is a bounded-workload guard, not a general allocator limit. Shapes/batches never grow from installed-RAM or free-disk budgets. Restricted counters cannot certify native headroom; the separate sandbox artifact states that limitation, while the actual native report supplies reserve evidence.

| Block/backend | RSS before/after MiB | Peak RSS MiB | Headroom before/after GiB | Swap delta bytes | MPS allocated/driver/recommended after bytes |
|---|---:|---:|---:|---:|---|
| 1/cpu | 196.953 / 281.703 | 281.703 | 25.920 / 25.920 | 0 | 0 / 475136 / 30150672384 |
| 2/mps | 198.109 / 435.281 | 435.281 | 25.920 / 25.920 | 0 | 268288 / 17514496 / 30150672384 |
| 3/mps | 197.969 / 436.281 | 436.281 | 25.920 / 25.920 | 0 | 268288 / 17514496 / 30150672384 |
| 4/cpu | 197.328 / 282.031 | 282.031 | 25.920 / 25.920 | 0 | 0 / 475136 / 30150672384 |

All final native block pressure levels were1(normal); final swap did not increase. Initial higher historical swap use was recorded and was not treated as current pressure on its own. Power observations report AC. No thermal control or system setting was changed; external interference cannot be eliminated, and this bounded calibration is not a thermal/scaling study.

The first native calibration used a128MiB raw-tensor allowance while actual first-use MPS RSS grew~250MiB. There was **no actual9GiB reserve violation**: observed headroom stayed24+GiB. Self-review nevertheless found the allowance insufficient for a future near-reserve run. A regression first failed because9GiB+256MiB was incorrectly allowed; the guard now requires512MiB above reserve. This justified refreshed native focused tests, native doctor/verify-math artifacts and a repeated final full suite. The earlier measurements remain auditable as `task8-before-guard-*`; they are **superseded**, not deleted or used for the final backend selection. Earlier complete-path aggregate medians were CPU33.9822915ms/MPS57.1670210ms.

## TDD, verification and diagnostic regression

- Initial RED: missing probe/backend module plus failed-command stdout regression:2 failed,8 errors. Implemented actual probes: first sandbox GREEN6passed/4reasoned MPSskips; first native focused10passed in13.71s.
- Integration RED/GREEN: old pending backend scope and absent validation were observed failing; real validated gate integration then passed3focused tests. Status-only fake success is rejected by `verify-math`.
- Additional observed RED cycles covered missing direct whole-path timing, missing current RSS fields, validation erased by `python -O`, dropped resource-failure observations, absent ABBA selection and lock/runtime digest checks. Each was corrected with focused GREEN runs. The final reserve regression also failed before the512MiB change.
- Meaningful fault injection makes the actual `index_copy` operation raise `NotImplementedError`: the backend fails explicitly, marks the operation failed, preserves the reason and leaves later operations not_run. A validated CPU report plus failed indispensable MPS operation selects whole-method CPU; missing valid CPU raises. No backend is silently mixed.
- Final native focused command: `env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/test_backend.py tests/test_growth.py -q > environment/task8-native-focused.txt` — **70passed in12.11s**, real CPU and MPS, no skips. This is distinct native device verification, not sandbox availability evidence.
- Final native doctor command: `env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m treeneuron.cli doctor --output environment/report.json` — exit0, both actual backends passed, balanced block evidence above.
- Final required math command: `.venv/bin/python -m treeneuron.cli verify-math --backend-report environment/report.json --output environment/task8-verify-math.json > environment/task8-verify-math.stdout.json` — exit0; `complete=true`, reference_cpu/package_layer_math/package_score_math/cpu_backend/mps_backend all passed. Existing native backend evidence is fully revalidated against current source, lock and runtime; CPU64 reference/package checks run in fresh interpreters.
- Sandbox system observation command: `env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -c 'from pathlib import Path; from treeneuron.doctor import collect_system_report,write_report; write_report(collect_system_report(),Path("environment/task8-sandbox-system.json"))'`. This intentionally runs no kernels; it records MPS unavailable and denied sysctl counters, with kernel_gate not_run. Native gate evidence stays in the separate native doctor artifact.
- Final full suite command: `.venv/bin/python -m pytest -q > environment/task8-pytest.txt` — **323 passed, 16 skipped in 86.75s (0:01:26)**. Sandbox MPS tests explicitly skip with process-unavailability reasons; CPU is mandatory. Earlier pre-guard full suite322passed/16skipped in91.30s remains archived.
- `git diff --check` passed. The carried Task1 minor is fixed: failed commands retain sanitized stdout in their reason when stderr is empty, preserving denied return codes and identity redaction. The focused diagnostic regression exercises that branch.

## Artifact hashes and drift boundary

Execution digest binds `backend.py`, actual layer, actual scorer, allocator and growth types. Lock and runtime digests bind uv.lock plus Python/Torch version/platform/architecture; fallback and dtype/thread contracts are validated separately. Digests do not depend on mutable HEAD, so a later documentation-only milestone commit does not invalidate identical execution source. Actual execution/input drift is rejected. JSON evidence includes all original trials and timing/resource samples, and writes reject NaN/Infinity.

Final source digest: `2b8cd095322ccd1b0e32e4fe31132dc1b366e0cf6150fac8fa113929b375171a`.
Lock SHA256: `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`.
Runtime digest: `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68`.

| Artifact | SHA256 |
|---|---|
| `environment/report.json` | `7687520cd03031de1090a980481836fd0314e28d91a09626b0da16ae934ed415` |
| `environment/task8-verify-math.json` | `82f9302d8ae8040ab07e3e142cf4c9db6009056ed5055b3f55a54f978c1918cd` |
| `environment/task8-verify-math.stdout.json` | `82f9302d8ae8040ab07e3e142cf4c9db6009056ed5055b3f55a54f978c1918cd` |
| `environment/task8-native-focused.txt` | `3f302ce884dc5231459855245090308656e3735b6fca23fb00fe6a2e5a28d6fc` |
| `environment/task8-pytest.txt` | `f7766619590f2b69ee27130bb6728c0285d1091811d7f71c131840a299ee0464` |
| `environment/task8-sandbox-system.json` | `fdbc61404e9a108793dafd6df35e6273c912150e282549112563a1ea4dfaf65c` |
| `environment/task8-before-guard-report.json` | `9da4d82b18b1f35d157a041d53ff9407ff82f113f2634d3c579804e8889a01a2` |
| `environment/task8-before-guard-verify-math.json` | `d2f1f77289bd1d6815836886298bb0b59a1bf2dc29db6ee747195fc070423485` |
| `environment/task8-before-guard-pytest.txt` | `fe18bea04a52ce391ccc994cf119b4630b2c3da1806694a70e6f820ea40ff360` |

## Limits and handoff

These are infrastructure/kernel measurements on one bounded synthetic workload, not predictive results or a laptop-wide forecast. Larger layers/data, changed precision/optimizer/threads, physical batches outside the tested range or additional operations require new parity/resource/throughput evidence. The selected CPU path does not imply a general CPU advantage. Warm dataset-resident timings exclude cold import/initial transfer and do not cover causal optimizer-state migration across growth. No watermark, fallback or float64-MPS workaround was used.

Task9 should consume `backend_probes.decision` from the final doctor report, retain logical128 and the explicit CPU64 scoring boundary, and implement state migration/fork/resume semantics independently. Revalidate the artifact after executable changes rather than copying a passed flag. Root must review and append the public WP00/WP01 decisions, preserving the R12 novelty hold and all remaining scientific/protocol gates. No public WP decision or plan ledger was changed by this task.


## Review fix round 1: strict batch-artifact contract (supersedes previous final evidence)

Addressed the single Important finding in `task-8-review.md` on BASE`2f03c82`. Separate fix commit: **`a4f4cdc79521d0749a28b0b5e9cb3b543d991ef1`**. Only `src/treeneuron/backend.py` and `tests/test_backend.py` changed. No public WP00/WP01 decisions, plan ledgers, scientific protocol, branch/worktree, dependencies, system settings, protected T or predictive comparisons changed. Root retains the public decisions pending scoped re-review.

The original live-probe validation required physical batch divisors of logical128, while imported artifacts checked label consistency without enforcing that contract. An actual valid four-batch fixture was consistently relabeled to `[1000,2000,4000,8000]`, including stage rows, selected recipe and (when present) whole-path recipe, with its implementation digest unchanged. The regression demonstrated acceptance before the fix. It now fails validation and cannot produce a selected CPU recipe. No source-drift check or stored artifact digest was weakened or patched.

`_validate_physical_batches(batch_sizes, *, logical_batch_size=128)` is now the shared contract for public `probe_backend`, the isolated child, imported `report.batch_sizes`, and each stage's physical batch labels. It requires a nonempty non-string Sequence of positive **Python int** values, each<=logical128 and dividing128, with every next value exactly twice its predecessor. Logical batch must itself be a Python int128. Bool, float, duplicate, negative/zero, nondivisor, oversized, empty and malformed labels are rejected.

`_validate_batch_recipe(recipe, *, batch_sizes, logical_batch_size)` requires exactly `base_epoch`, `grown_epoch`, `score`, validates each physical value with the same contract and requires membership in tested batches. Both the selected recipe and complete-path context pass this check. Stage rows must match the validated batch sequence and selected values must still satisfy the original95% median-throughput rule.

The complete-path measurement adds explicit `calibration.whole_method.physical_batches` and `.logical_batch_size`. The measured callback consumes that exact copied recipe; imported evidence requires it to match the validated selected recipe. The existing `whole_method.batch_size=None` sentinel is enforced because this region contains three phase-specific batch sizes. Public API signatures and `backend_probes.decision` remain unchanged; existing source binding intentionally requires regenerating old evidence after executable changes. Tests run against genuine fresh reports, with no digest patching.

### TDD and verification commands

- RED1: `.venv/bin/python -m pytest tests/test_backend.py -q -k 'consistent_out_of_contract or batch_contract_applies' > environment/task8-batch-validation-red.txt` — **18failed,8passed,26skipped,35deselected in6.98s**. The failures include consistent huge labels, float/bool labels and complete-path contexts accepted before the fix. CPU execution is actual; MPS is reasoned unavailable in the sandbox.
- RED2: `.venv/bin/python -m pytest tests/test_backend.py -q -k 'live_probe_rejects or recipes_require' > environment/task8-batch-validation-red-extra.txt` — **14failed,2passed,6skipped,87deselected in6.06s**. These cover malformed live input and exact recipe keys.
- GREEN focused: `.venv/bin/python -m pytest tests/test_backend.py -q > environment/task8-batch-validation-focused.txt` — **62passed,47skipped in8.44s**.
- Necessary native focused verification: `env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m pytest tests/test_backend.py -q > environment/task8-native-focused.txt` — **109passed in16.38s**, real CPU and MPS, no skips. Unchanged `test_growth.py` native evidence remains archived from the original task; fresh backend probes exercise the actual layer and indexed autograd again.
- Native doctor: `env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m treeneuron.cli doctor --output environment/report.json` — exit0, actual CPU/MPS reference kernels and ABBA calibration regenerated under the new source digest.
- Required math revalidation: `.venv/bin/python -m treeneuron.cli verify-math --backend-report environment/report.json --output environment/task8-verify-math.json > environment/task8-verify-math.stdout.json` — exit0; **complete=true**, all five gates passed. Actual CPU64 references/package checks run; backend evidence is revalidated from the newly executed artifact.
- Final full suite: `.venv/bin/python -m pytest -q > environment/task8-pytest.txt` — **365 passed, 48 skipped in 99.96s (0:01:39)**. CPU mandatory; sandbox MPS cases skip explicitly. Native MPS verification is the separate109-test execution above.
- `git diff --check` passed. The source/lock/runtime integrity rules remain active, including optimized-Python validation.

### Refreshed measurements and precision limits

The refreshed native evidence is the current gate/selection evidence. Selected backend **cpu**: CPU **36.103167000 ms**, MPS **66.896271000 ms**, using median ABBA complete-path block costs. Selected physical batches remain `{"base_epoch": 128, "grown_epoch": 128, "score": 128}` with logical128 and CPU64 scoring. Accumulation error remains `2.9802322387695312e-08`. All four direct complete-path blocks met the declared stability criterion.

| ABBA block/backend | UTC | Direct whole path ms | Whole timing stable | Base/grown/score selected batch |
|---|---|---:|---|---|
| 1/cpu | 2026-10-16T15:06:03.925516+00:00 | 36.616792 | True | 128/128/128 |
| 2/mps | 2026-10-16T15:06:09.642157+00:00 | 65.874667 | True | 128/128/128 |
| 3/mps | 2026-10-16T15:06:18.688307+00:00 | 67.917875 | True | 128/128/128 |
| 4/cpu | 2026-10-16T15:06:26.958157+00:00 | 35.589542 | True | 128/128/128 |

| Block/backend | Stage | b16 ms | b32 ms | b64 ms | b128 ms |
|---|---|---:|---:|---:|---:|
| 1/cpu | base_epoch | 9.752042 | 5.907500 | 4.111313 | 2.640084 |
| 1/cpu | grown_epoch | 19.911417 | 12.010250 | 7.003083 | 5.065771 |
| 1/cpu | score | 131.719625 | 72.603750 | 43.803500 | 27.252833 |
| 2/mps | base_epoch | 41.070541 | 25.312125 | 19.755791 | 13.124354 |
| 2/mps | grown_epoch | 83.318667 | 47.313750 | 33.320333 | 23.189625 |
| 2/mps | score | 136.518416 | 75.008375 | 44.531209 | 28.921750 |
| 3/mps | base_epoch | 38.700792 | 25.861834 | 18.551375 | 13.308000 |
| 3/mps | grown_epoch | 74.729500 | 43.919791 | 30.081250 | 22.718000 |
| 3/mps | score | 135.044458 | 74.475250 | 45.201792 | 29.184042 |
| 4/cpu | base_epoch | 10.287416 | 6.076417 | 3.915625 | 2.649666 |
| 4/cpu | grown_epoch | 19.075375 | 10.777542 | 6.972208 | 5.163625 |
| 4/cpu | score | 132.803458 | 73.987708 | 42.583291 | 27.120042 |

Individual stage precision caveats, retained without exclusions or reruns to obtain favorable stability:
- Block1/cpu, `base_epoch`, physical128: warmup_stable=True, timing_stable=False, 11 timed samples; selected stage batch=128. Last-five range/median criterion was not met before the sample cap. The measured median still enters the original95% rule; this precision limit is separate from stable direct complete-path selection.
- Block2/mps, `base_epoch`, physical64: warmup_stable=True, timing_stable=False, 11 timed samples; selected stage batch=128. Last-five range/median criterion was not met before the sample cap. The measured median still enters the original95% rule; this precision limit is separate from stable direct complete-path selection.

These capped stage observations limit isolated-stage precision and require later WP05 remeasurement; they do not represent a failed kernel/reference gate or a reserve violation. All recorded kernel/error trial tables are **exactly unchanged** from the prior native artifact, including the fifteen operations, nine seed/magnitude trials, CPU64 references, fixed `atol=7.62939453125e-6`/`rtol=3.0517578125e-5`, and exact serialization/resumed-update equality. There was no tolerance change. Timing changes are fresh observations, not a new scientific method comparison or forecast.

Resource/fallback boundaries remain unchanged: fallback0 before Torch import, CPU64 score/eigendecomposition, fixed bounded batches,9GiB reserve plus512MiB runtime allowance, no watermark changes. Final native before/after pressure levels are all1(normal). Headroom, RSS/MPS overlapping counters and swap records are retained per block below; no memory totals are formed by adding overlapping counters.

| Block/backend | Headroom before/after GiB | RSS after MiB | Swap delta bytes |
|---|---:|---:|---:|
| 1/cpu | 26.280 / 25.920 | 282.578 | -8388608 |
| 2/mps | 25.560 / 26.280 | 437.344 | 0 |
| 3/mps | 26.640 / 26.640 | 437.156 | 0 |
| 4/cpu | 26.640 / 26.640 | 283.125 | 0 |

### Refreshed source/artifact identities and retained history

The current primary artifact paths in the earlier report now hold refreshed evidence. Their earlier hashes/timings are superseded by this section and remain preserved under the before-batch-validation archive names; earlier pre-guard archives also remain untouched.

Current execution-source digest: `0f054cb12e1ace3a9f46fc8edbb452d3a89b2e48c58c92fc6c4bb202a84cab51`.
Lock digest (unchanged): `13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5`.
Runtime digest (unchanged): `af9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68`.

| Artifact | SHA256 |
|---|---|
| `environment/report.json` | `60f5cc801909f2b83b1963c814432f5e1bc8a3e37a3ebdf649aa8e8b239b18dc` |
| `environment/task8-verify-math.json` | `d8d8c5a99911c7429e37489f89ac0fcb304a6a30d7238631c0ece0a1614e7162` |
| `environment/task8-verify-math.stdout.json` | `d8d8c5a99911c7429e37489f89ac0fcb304a6a30d7238631c0ece0a1614e7162` |
| `environment/task8-native-focused.txt` | `2ae8a7cdfc5139248d6be65356014138c1f768b17c9b15137ed14ed4cc446551` |
| `environment/task8-pytest.txt` | `a1326af17cf5eb324b2825a036ecd83809c700a69bf5c6f13e114794a9922200` |
| `environment/task8-batch-validation-focused.txt` | `33adab12c020555de52bcfe19df548c9b99259956f3d7b4e9585efd45a7c6c24` |
| `environment/task8-batch-validation-red.txt` | `dbf4e83f0be04e8c31842c6d1ed0bfdcf53d35929e2e568e58c6c5aad0d751f8` |
| `environment/task8-batch-validation-red-extra.txt` | `cdeb49d0f87e26429d209bd1717b991743deadc865b9b722d290a9e006e1e760` |
| `environment/task8-before-batch-validation-doctor.json` | `7687520cd03031de1090a980481836fd0314e28d91a09626b0da16ae934ed415` |
| `environment/task8-before-batch-validation-verify-math.json` | `82f9302d8ae8040ab07e3e142cf4c9db6009056ed5055b3f55a54f978c1918cd` |
| `environment/task8-before-batch-validation-verify-math.stdout.json` | `82f9302d8ae8040ab07e3e142cf4c9db6009056ed5055b3f55a54f978c1918cd` |
| `environment/task8-before-batch-validation-native-focused.txt` | `3f302ce884dc5231459855245090308656e3735b6fca23fb00fe6a2e5a28d6fc` |
| `environment/task8-before-batch-validation-pytest.txt` | `f7766619590f2b69ee27130bb6728c0285d1091811d7f71c131840a299ee0464` |
| `environment/task8-before-batch-validation-sandbox-system.json` | `fdbc61404e9a108793dafd6df35e6273c912150e282549112563a1ea4dfaf65c` |
| `.superpowers/sdd/2026-10-16-selective-tree-neurons-core/task-8-before-batch-validation-report.md` | `d37ad63fbf20d6347c90c9369615d0ded9925e24d9b685b181718df6f76a6b98` |

The prior native kernel findings remain sound and auditable. Older source-bound artifacts intentionally fail revalidation after the fix; consume the regenerated doctor artifact rather than editing hashes. Task9 still needs independent optimizer migration/fork/resume integration and later workload/resource measurements. Root's scoped re-review and public WP00/WP01 decisions remain outstanding.
