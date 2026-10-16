# Selective Tree-Neuron Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement WP00–WP04: a locally verified, checkpointable, paired one-event allocation experiment, without making a predictive superiority claim.

**Architecture:** A small `treeneuron` Python package owns the compact selective affine layer, CPU-float64 score, synthetic data, F-only transforms, training/checkpoint engine and sequential causal forks. A strict run ledger records evidence and costs. Practical vendor adapters, full raw-data acquisition, calibration, freezing and confirmation follow the separate evidence plan only when their gates are reached.

**Tech Stack:** Native arm64 Python (first try 3.12), PyTorch, NumPy, SciPy, pandas, scikit-learn, PyYAML, pytest; `uv` for resolved core environment; standard-library argparse for CLI. No distributed framework.

**Spec:** [2026-10-16-selective-tree-neurons-design.md](../specs/2026-10-16-selective-tree-neurons-design.md). Read it and [the evidence record](../../research/2026-10-16-evidence-and-machine-record.md) before executing. Continue with [the evidence plan](2026-10-16-selective-tree-neurons-evidence.md) after Task 12.

## Global Constraints

- Work in the current branch. Do not create a worktree or switch branches unless a later user instruction changes this constraint.
- Training uses float32 eager PyTorch. Mathematical checks and score construction use CPU float64.
- Primary neural execution uses no reduced precision and no implicit MPS fallback.
- No dropout, BatchNorm, learned normalization, embeddings, or ensemble in the causal experiment.
- Initial synthetic coordinate: `d=16,h=64,m_requested=16,τ=1,epsilon=0.05`.
- IDs are a 1-D CPU int64 tensor; contrasts have shape `[m,d+1]`, the base parameter dtype/device, explicit bias in the last coordinate, and finite values.
- T labels are read only by the protected evaluator after protocol validation.
- No primary refit on F+G+V.
- Zero-growth teachers remain in all results.
- RSS, MPS allocated/driver memory overlap in unified memory; record separately, never sum as disjoint usage.
- No `frozen.yaml`, confirmatory result or completed-WP record is created by this documentation task.

All checkboxes start unchecked because these are future implementation steps. Update `DECISIONS.md` at every WP gate with evidence paths, hashes, decision, blockers and next command. Commit only task-owned files; preserve unrelated work. Tests in this plan are requirements to implement, not claimed existing fixtures or executed results.

## Review Focus

- Zero eligible units: return a valid unchanged paired model and record requested versus actual count, never discard the teacher (Tasks 7, 10, 12).
- Incorrect per-example scaling: scoring must agree across batch sizes, including a short last batch (Task 7).
- Growth during interruption: old moments, new step counters and sampler position must survive an atomic checkpoint and resume (Tasks 9–10).
- Nonfinite/constant/all-missing fitting columns or constant targets: deterministic F-only handling and undefined normalized-risk status (Task 6).
- Hidden access or drift: exploratory runs cannot read protected T labels, and code/config/split mismatch cannot silently resume (Tasks 2, 5, 9, 12).

## File map and interfaces

Create only each task’s files when that task begins. `src/treeneuron/__init__.py` and namespace `__init__.py` files are minimal packaging, not a general plugin system.

| Files | Responsibility |
|---|---|
| `pyproject.toml`, `uv.lock`, `.gitignore` | Core install/lock; exclude raw data, runtime artifacts, caches, weights and local environments. |
| `src/treeneuron/cli.py`, `doctor.py` | Command dispatch; local hardware/software/kernel evidence. |
| `config.py`, `artifacts.py` | Strict resolved configuration, canonical hashing, immutable run directories and ledger writes. |
| `models/units.py`, `growth/types.py` | Compact preactivation layer, topology export, migration manifest. |
| `data/types.py`, `data/synthetic.py`, `data/controls.py` | Typed observations, deterministic teacher populations and controls. |
| `data/transforms.py`, `evaluation/metrics.py` | F-only preprocessing and original-unit/normalized risk. |
| `growth/score.py`, `growth/allocate.py` | Float64 moment construction, contrasts, eligibility and allocation. |
| `train.py`, `checkpoint.py`, `growth/insert.py` | Deterministic updates, G checkpoints, atomic recovery and optimizer migration. |
| `baselines/api.py`, `baselines/simple.py`, `baselines/causal.py` | Predictor contract; B0 and B1–B5 factory/adapters. |
| `experiments/paired.py`, `analysis/exploratory.py` | Sequential paired runner and explicitly exploratory summaries. |
| `configs/explore.yaml`, `protocol/draft.yaml` | Resolved exploratory starter recipe and unfreezable draft protocol. |
| `tests/reference_math_checks.py`, `tests/reference_score_checks.py` | Standalone A/B reproduction JSON scripts. |
| `tests/test_*.py` named in tasks | Numerical, systems, access and integration contracts. |

Paths below are relative to repository root. Standard run command is `.venv/bin/python -m treeneuron.cli ...`; tests use `.venv/bin/python -m pytest ...`. Commands shown without the interpreter prefix in the spec are equivalent once the environment is activated.

Assertion snippets use test-owned fixtures whose setup is described in the task; they pin consequential behavior rather than supply full fixture implementations.

## Task 1: Resolve a native core and record WP00 prerequisites

**Files:** Create `pyproject.toml`, `uv.lock`, `.gitignore`, `src/treeneuron/__init__.py`, `src/treeneuron/cli.py`, `src/treeneuron/doctor.py`, `tests/test_doctor.py`, `provenance/sources.json`, `protocol/prior_art_delta.md`. Modify `README.md`, `DECISIONS.md` when the task is verified.

**Interfaces:** Produce `collect_system_report() -> dict[str, Any]`, `write_report(report: dict[str, Any], output: Path) -> None`, and CLI `doctor --output PATH`. Report schema version, UTC, architecture/OS/memory/CPU/disk/power/tool/package observations, command errors, Torch/MPS build/availability, fallback environment, lock hash and source status. Missing counters are null with reasons.

- [ ] **Step 1:** Re-run the spec §2/§10 discovery commands; check stable primary package docs and R12’s official/author routes. Record source URLs/retrieval/scope, code/asset terms, access failures, and an install/disk forecast before acquisition. Do not download weights or accept an agreement. Select native Python/PyTorch compatibility, first trying Python 3.12.
- [ ] **Step 2:** Bootstrap only the chosen local venv and pytest so a failing test can run. Write `test_doctor_preserves_unavailable_fields` with an injected failed probe: assert the counter is `None`, reason/return code retained, report JSON finite/parseable, no serial/UUID output, and an arm64 observation is not replaced by an assumption. Run `.venv/bin/python -m pytest tests/test_doctor.py -q`; expect a missing doctor API failure.

```python
def test_doctor_preserves_unavailable_fields(tmp_path, unavailable_report):
    output = tmp_path / "report.json"
    write_report(unavailable_report, output)
    assert json.loads(output.read_text()) == unavailable_report
```

- [ ] **Step 3:** Implement packaging, doctor and CLI; declare only the spec’s core packages, resolve `uv.lock`, then `uv sync --frozen`. Do not force a guessed version. Pin source/license metadata and write provisional novelty status in `protocol/prior_art_delta.md`. Probe failures remain data; indispensable CPU import failures return a nonzero command status.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_doctor.py -q` and `.venv/bin/python -m treeneuron.cli doctor --output environment/report.json`; expect passing tests and a report with actual native process/package values. MPS availability alone does not complete WP00; Task 8 closes the kernel gate. Keep runtime reports outside Git unless explicitly curated/redacted as evidence.
- [ ] **Step 5:** Append the prerequisite decision and commit the packaging/doctor/provenance files: `chore: establish native core environment and prerequisite record`. Include exact selected dependencies and reasons, remaining R12 access hold, and any disk/backend limitation.

## Task 2: Define strict configuration, hashes and append-only run records

**Files:** Create `src/treeneuron/config.py`, `src/treeneuron/artifacts.py`, `tests/test_config.py`, `tests/test_artifacts.py`, `protocol/draft.yaml`.

**Interfaces:** Produce `load_experiment(path: Path) -> ExperimentConfig`, `canonical_hash(value: Mapping[str, Any]) -> str`, `create_run(root: Path, resolved: Mapping[str, Any]) -> RunPaths`, `append_event(paths: RunPaths, event: Mapping[str, Any]) -> None`, `finish_run(paths: RunPaths, metrics: Mapping[str, Any], status: str) -> None`. `RunPaths` contains `run_id: str` and `directory: Path`. `ExperimentConfig` exposes `phase`, `study_id`, `model`, `training`, `growth`, `data`, `seeds`, `backend`; nested fields implement spec §4–§7 values and explicit resolved defaults.

- [ ] **Step 1:** Write tests asserting unknown YAML keys/nonfinite floats/invalid τ reject; canonical hashes ignore mapping order but change on seed or training value; duplicate run ID cannot overwrite; unavailable counters serialize as null with reason; interrupted runs keep existing events. `test_protected_phase_rejected_by_core_runner` requires a protocol capability for confirmatory phase. Run the two test files; expect missing APIs.

```python
def test_hash_is_order_independent_and_seed_sensitive():
    assert canonical_hash({"seed": 1, "tau": 1.0}) == canonical_hash({"tau": 1.0, "seed": 1})
    assert canonical_hash({"seed": 1}) != canonical_hash({"seed": 2})
```

- [ ] **Step 2:** Implement strict nested validation, canonical sorted finite JSON and SHA-256, unique run paths, atomic manifest/metrics writes and append-only events. Do not implement fake freeze authority: the core supports exploratory phase only until Evidence Task 9. Define run statuses `created/running/interrupted/failed/completed` separately from scientific failure classes. Reject reused paths.
- [ ] **Step 3:** Implement schema-versioned manifests with all spec §11 field groups; unmeasured fields use null/reason, never made-up zero. Draft protocol states `status: draft`, has no usable freeze hash, and cannot unlock T. Save resolved config before training begins.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_config.py tests/test_artifacts.py -q`; expect all pass, including the no-overwrite/access assertions. Inspect one generated manifest in a temporary directory for exact boundaries and seed fields.
- [ ] **Step 5:** Commit `feat: add strict experiment configuration and run ledger`.

## Task 3: Reproduce the historical reference algebra on this CPU

**Files:** Create `tests/reference_math_checks.py`, `tests/reference_score_checks.py`, `tests/test_reference_checks.py`; modify CLI.

**Interfaces:** Standalone processes print JSON; CLI `verify-math --output PATH` will aggregate these checks and, once Tasks 4/7/8 exist, package/backend checks. Use [the preserved recipes](../../research/2026-10-16-evidence-and-machine-record.md#reproduction-recipes-for-the-two-reference-checks) for random draw order and constants.

- [ ] **Step 1:** Write subprocess tests checking required error/norm/loss/environment fields, float64 CPU, one thread, seeds 1602026/1602027, and assertion failures propagating nonzero exit. Run `.venv/bin/python -m pytest tests/test_reference_checks.py -q`; expect missing scripts.

```python
def test_reference_centered_insertion(reference_a_json):
    assert reference_a_json["results"]["growth_preservation_max_abs"] == 0.0
    assert reference_a_json["results"]["gate_gradient_formula_max_abs"] < 1e-12
```

- [ ] **Step 2:** Implement A/B exactly from the recipes: A identity/gradient/gap thresholds `<1e-12`, exact centered preservation/zero initial contrast gradient, descending tiny step; B invariance/gradient thresholds `<1e-12`, positive observed first Adam decrease and relative approximation error `<1e-4`. Use fresh processes so dtype/RNG state cannot leak into training.
- [ ] **Step 3:** Run both scripts independently and the subprocess tests. Expect assertions pass and local JSON; compare to historical numbers by tolerance, not exact eigensolver-dependent values. If they fail, diagnose algebra/dependency/numerical causes before any prediction experiment.
- [ ] **Step 4:** Save local report hashes/environment in the decision record, explicitly distinguished from historical Linux evidence.
- [ ] **Step 5:** Commit `test: reproduce reference algebra and normalized score checks`.

## Task 4: Implement compact one-event selective affine preactivations

**Files:** Create `src/treeneuron/models/units.py`, `src/treeneuron/growth/types.py`, `tests/test_algebra.py`, `tests/test_growth.py`.

**Interfaces:** Produce the spec’s `SelectiveAffineLayer`, `grow(ids: Tensor, contrasts: Tensor) -> GrowthManifest`, `export_config() -> dict[str, Any]`, and `from_config(config: Mapping[str, Any]) -> SelectiveAffineLayer`. `GrowthManifest` stores version, IDs/counts, event state, `old_parameter_names`, `new_parameter_names`, and explicit old→new named identity/slice mappings. Layer forward returns preactivations, never ReLU.

- [ ] **Step 1:** Write tests with shared tensors for stump/gated forward and gradient mapping, scalar-head collapse, hardening identity/ReLU bound, and leaf-wise versus post-mixture ReLU inequality. `test_parameter_count` asserts 1,153 base and 1,697 after 16 growths for `d=16,h=64` including head. `test_grow_sorts_ids_with_contrasts` checks unsorted IDs retain the intended row mapping. Run the two files; expect import/API failure.

```python
def test_parameter_count():
    layer, head = SelectiveAffineLayer(16, 64), torch.nn.Linear(64, 1)
    count = lambda: sum(p.numel() for p in layer.parameters()) + sum(p.numel() for p in head.parameters())
    assert count() == 1153
    layer.grow(torch.arange(16, dtype=torch.int64), torch.ones(16, 17))
    assert count() == 1697
```

- [ ] **Step 2:** Add parametrized rejection tests for duplicate/out-of-range IDs, non-int64/shape/device contrasts, nonfinite values, invalid τ and second growth. Empty growth must record the event with no dormant parameters; serialization reconstructs topology before load. Test zero–zero dead gradients and nonzero-contrast/zero-gate prediction preservation with nonzero constructed gate gradient.
- [ ] **Step 3:** Implement compact base/gate/contrast modules and `index_select/index_copy` logic from spec §4. Keep stable base parameter objects. Export topology fully, preserving `growth_applied` even for empty events. Use float64 reference tests away from ReLU kinks for gradient/finite-difference checks.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_algebra.py tests/test_growth.py -q`; expect all float64 identity/gradient discrepancies `<1e-10`, exact CPU centered preservation, and correct parameter counts. Backend-specific float32 tolerances are added in Task 8.
- [ ] **Step 5:** Commit `feat: add compact function-preserving selective affine layer`.

## Task 5: Generate independent teachers and named control problems

**Files:** Create `src/treeneuron/data/types.py`, `src/treeneuron/data/synthetic.py`, `src/treeneuron/data/controls.py`, `tests/test_synthetic.py`, `tests/test_access.py`.

**Interfaces:** Produce `TeacherSpec` (version, dimension, vectors, prevalences/types/scales/thresholds, reference seed/mean/std), `ObservationSet` (`X`, `y`, `row_id`, `group_id`, optional `dates` and noise-free target), `DevelopmentData` (`F`, `G`, `V`, `teacher_id`), and shared `SplitManifest` (immutable row/group assignments, dataset/teacher ID, fold/phase, optional date boundaries, hashes, descriptive-only reason). Produce `make_teacher(seed: int, d: int = 16) -> TeacherSpec`, `sample_teacher(teacher: TeacherSpec, n: int, seed: int, sigma: float) -> ObservationSet`, `make_development(teacher: TeacherSpec, counts: Mapping[str, int], seeds: Mapping[str, int], sigma: float) -> DevelopmentData`. `generate_protected_test` is reserved for the protected evaluation module; normal loaders do not accept a T path. Control factory: `make_control(kind: str, seed: int, d: int) -> TeacherSpec | ControlSpec`, with `ControlSpec` serialized/versioned by the same module.

- [ ] **Step 1:** Write tests asserting deterministic independent seed streams, distinct row IDs, 65,536 reference samples disjoint from observation streams, stored normalization, exactly four independent regime components, and the explicit gate probabilities/threshold orientation in spec §7. `test_features_exclude_oracle_information` asserts X contains only sampled inputs; teacher state/noise-free targets are metadata, never extra predictors. Run the tests; expect missing APIs.

```python
def test_features_exclude_oracle_information():
    rows = sample_teacher(make_teacher(71), n=5, seed=72, sigma=0.1)
    assert rows.X.shape == (5, 16)
    assert rows.y.shape == (5,)
```

- [ ] **Step 2:** Implement the primary teacher and counts (`F=2306,G=577,V=577`); keep reference/teacher/data namespaces separate. Generate only F/G/V in the exploratory runner. Store enough teacher state to reproduce f and gates for diagnostic analysis without exposing them through `ObservationSet.X`.
- [ ] **Step 3:** Implement/version globally affine, additive sine/quadratic, pure noise, row-label permutation, homogeneous smooth, rare/rotation/nuisance controls and the original 2-D target. The 2-D sampler uses independent `Uniform(-1,1)` and `Uniform(0,1)`, no noise; test `(-0.5,0.2)→0.2`, `(0.5,0.2)→0.8`. Configs declare exact coefficients/distributions for the other controls; builder choices stay exploratory until frozen. Oracle gates are available only to an explicitly labeled diagnostic factory.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_synthetic.py tests/test_access.py -q`; expect repeatable generation, no protected T access and no oracle columns. Include a malformed/overlapping phase-seed namespace rejection.
- [ ] **Step 5:** Commit `feat: add independent synthetic teachers and control populations`.

## Task 6: Fit preprocessing on F and establish metric/B0 sanity

**Files:** Create `src/treeneuron/data/transforms.py`, `src/treeneuron/evaluation/metrics.py`, `src/treeneuron/baselines/api.py`, `src/treeneuron/baselines/simple.py`, `tests/test_transforms.py`, `tests/test_metrics.py`, `tests/test_simple_baselines.py`.

**Interfaces:** Produce `FittedTransform.fit(X_fit: Any, y_fit: np.ndarray) -> FittedTransform`, `.transform_X(X: Any) -> np.ndarray`, `.transform_y(y: np.ndarray) -> np.ndarray`, `.inverse_y(prediction: np.ndarray) -> np.ndarray`, `.to_state() -> dict[str, Any]`, `.from_state(state: Mapping[str, Any]) -> FittedTransform`; `regression_metrics(y: np.ndarray, prediction: np.ndarray, *, fit_target_variance: float, groups: np.ndarray | None = None, noisefree: np.ndarray | None = None) -> dict[str, Any]`. Define exact `FitRequest` and `Predictor` from spec §11. Implement `MeanPredictor` and `RidgePredictor(alpha: float)` against that contract; HPO, not adapter fit, later chooses alpha on V.

- [ ] **Step 1:** Write tests for population scale, constant columns→scale 1, all-missing F columns→fill 0 plus indicator/manifest, unseen categories→reserved code, nonfinite labels→error, output shape `[N]` in original units. Changing G/V/T labels must not alter F transform state; test a copy with mutated protected labels without reading them in fitting. Run these three files; expect missing APIs.
- [ ] **Step 2:** Write metric assertions: `y=[1,3],pred=[2,2],var_F=4` gives MSE 1, normalized MSE 0.25, RMSE/MAE 1; identical predictions give paired effect zero; zero target variance gives null normalized risk with a reason and raw MSE retained. Unequal group sizes expose a row/macro difference; reject nonfinite/incorrect-shape predictions rather than broadcast.

```python
def test_normalized_risk_uses_fitting_population_variance():
    result = regression_metrics(np.array([1., 3.]), np.array([2., 2.]), fit_target_variance=4.)
    assert result["mse"] == 1.0
    assert result["normalized_mse"] == 0.25
```

- [ ] **Step 3:** Implement F-only transforms, original-unit metrics, mean, and ridge; persist transformations. Test ridge on an augmented small numerical design against an independent closed-form solution with unpenalized intercept. Spline-feature ridge can follow in the evidence plan; it is not required for WP04.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_transforms.py tests/test_metrics.py tests/test_simple_baselines.py -q`; expect all pass including zero variance, label access and closed-form parity.
- [ ] **Step 5:** Append WP02 synthetic/harness decision and commit `feat: add fitting-only preprocessing and regression sanity controls`. Raw transfer-data acceptance remains explicitly open for the evidence plan.

## Task 7: Compute the moment score and allocate a fixed count

**Files:** Create `src/treeneuron/growth/score.py`, `src/treeneuron/growth/allocate.py`, `tests/test_score.py`, `tests/test_allocation.py`.

**Interfaces:** `ScoreTable` stores unit IDs, `[h,d+1]` contrasts/gradients, `[h]` Adam/SGD scores, spectra/gaps/RMS and per-unit eligibility/status. Produce `score_units(model: nn.Module, X_fit: Tensor, y_fit: Tensor, *, epsilon: float, tau: float, adam_epsilon: float, batch_size: int, unit_block_size: int) -> ScoreTable`, `allocate(table: ScoreTable, requested: int, *, policy: str, seed: int) -> Allocation`. `Allocation` stores `requested_count: int`, `effective_count: int`, sorted CPU-int64 `ids`, `contrasts`, `policy`, `seed` and `table_hash`. Scoring accepts a CPU-float64 evaluation copy of the model only, preserving training RNG/state.

- [ ] **Step 1:** Write tests for analytic gate gradient versus autograd, first gate-only Adam step prediction, sign convention, permutation and positive-ReLU-rescaling agreement `<1e-10`. `test_score_batch_invariance` compares full data to batches of 7 with a short tail and multiple unit block sizes. `test_e_uses_sum_of_row_losses` catches the extra `1/n` failure. Run both files; expect missing APIs.

```python
def test_gate_gradient_matches_moment_formula(score_case):
    analytic, autograd_gradient = score_case
    assert (analytic - autograd_gradient).abs().max().item() < 1e-10
```

- [ ] **Step 2:** Implement the exact spec §5 moment, symmetrization, eigenvector, RMS normalization, scores and eligibility. Never call a PSD covariance routine. Ties of positive scores sort by ID; log eigenvalue ties without claiming stable directions in repeated eigenspaces. Record zero/nonfinite statistics instead of selecting them. Bound moment memory by block size.
- [ ] **Step 3:** Implement top/random/SGD policies and top-ID Gaussian-direction ablation. Tests require equal `m_eff`, uniform sampling without replacement via the seeded generator, exact per-ID contrasts shared between top/random, and all-zero/ineligible inputs yielding empty allocations. Negative requested count rejects. Random-only standalone cost records sampled contrast work separately from shared research scoring.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_score.py tests/test_allocation.py -q`; expect all pass. Run both reference scripts again only if this task changes their numerical implementation. Record blocked nonfinite runs separately from individual ineligible units.
- [ ] **Step 5:** Commit `feat: implement normalized moment scoring and matched allocation`.

## Task 8: Close the actual CPU/MPS operation and batching gate

**Files:** Modify `src/treeneuron/doctor.py`, CLI, `tests/test_growth.py`; create `tests/test_backend.py`.

**Interfaces:** Produce `probe_backend(device: str, *, seed: int, batch_sizes: Sequence[int]) -> dict[str, Any]` and CLI doctor report section `backend_probes`. Report selected whole-method backend, fallback disabled status, operation/error table, float32 tolerances with CPU-float64 reference errors, working set/RSS/pressure/swap availability and epoch/scoring costs.

- [ ] **Step 1:** Write conditional tests for float32 GEMM, tanh/sigmoid, ReLU, index-select/copy forward/backward, AdamW `foreach=False`, serialization and a resumed update. CPU is mandatory; unavailable MPS is a reasoned skip, not a pass. A deliberately unsupported path must select CPU or error, never silently mix execution. Run `.venv/bin/python -m pytest tests/test_backend.py -q`; expect missing probe functionality.

```python
def test_index_copy_backward_matches_reference(index_copy_gradients):
    actual, reference, recorded_tolerance = index_copy_gradients
    torch.testing.assert_close(actual, reference, **recorded_tolerance)
```

- [ ] **Step 2:** Implement explicit probes with `PYTORCH_ENABLE_MPS_FALLBACK` disabled before process import. Derive and record float32 tolerance from CPU-float64 references across seeds and magnitudes; justify any MPS-specific allowance before predictive runs. CPU float64 eigendecomposition stays off MPS. If indexing needs a selection-matrix fallback, verify gradients and record/timing-name it separately.
- [ ] **Step 3:** Measure complete epoch/score passes at safe doubled batches, using synchronized timers. Select the smallest safe tested batch reaching at least 95% of maximum median throughput; record physical/logical distinction and the 9 GiB initial reserve/current pressure. Do not allocate to a budget inferred solely from installed RAM or free disk. No forecast is measured until this command runs.
- [ ] **Step 4:** Run backend tests and `doctor --output environment/report.json`; expect mandatory operations valid on the selected backend, explicit unsupported paths/exclusions and recorded tolerances. Append final WP00 and WP01 decisions after CPU reference/layer/score tests pass; retain R12 novelty hold if unresolved.
- [ ] **Step 5:** Commit `feat: verify local backend kernels and calibration boundaries`.

## Task 9: Train a parent and recover it atomically

**Files:** Create `src/treeneuron/train.py`, `src/treeneuron/checkpoint.py`, `tests/test_train.py`, `tests/test_checkpoint.py`.

**Interfaces:** Produce `TrainingState` (model, optimizer, scheduler, update/epoch, sampler order/cursor, named RNG states, preprocessing, identity hashes), `TrainingPolicy` (logical/physical batch, max updates, evaluation interval, G patience, strict-improvement rule), `fit_parent(data: DevelopmentData, config: ExperimentConfig, paths: RunPaths) -> TrainingState`, `continue_fit(state: TrainingState, data: DevelopmentData, policy: TrainingPolicy, paths: RunPaths) -> TrainingState`, `save_checkpoint(state: TrainingState, path: Path) -> str`, `load_checkpoint(path: Path, *, expected_hashes: Mapping[str, str], device: str) -> TrainingState`.

- [ ] **Step 1:** Write tests for earliest G-best selection with its complete optimizer/sampler/RNG state, initial update-zero G eligibility for every continuation, no V/T use, deterministic sampler cursor at a non-epoch checkpoint, RNG reconstruction, original-unit prediction inversion, and finite-loss failure logging. `test_failed_atomic_write_keeps_last_valid` injects failure before rename; `test_resume_rejects_identity_drift` changes split/config/code/lock hash independently. Run the two test files; expect missing APIs.

```python
def test_resumed_parameters_match_uninterrupted(cpu_resumed, cpu_uninterrupted):
    assert cpu_resumed.update == cpu_uninterrupted.update
    for name, tensor in cpu_uninterrupted.model.state_dict().items():
        torch.testing.assert_close(cpu_resumed.model.state_dict()[name], tensor, rtol=0, atol=0)
```

- [ ] **Step 2:** Implement logical-update training/evaluation with explicit loss semantics, AdamW and schedule configuration, G-only checkpoint selection and all required state. Return the complete G-best checkpoint, never best weights plus last-state moments. Store checkpoints at complete updates only; checksums and atomic last-valid pointer protect prior state. Catch/log failure details before propagating; do not mark a partially trained run completed.
- [ ] **Step 3:** Implement same-backend resume and a controlled interruption test comparing uninterrupted predictions, parameters, optimizer tensors/steps, scheduler and next row IDs. Use exact deterministic CPU agreement where supported and the recorded backend tolerance otherwise. No cross-backend resumption parity claim.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_train.py tests/test_checkpoint.py -q`; expect parent checkpoint selection and recovery pass, including mid-epoch interruption and corrupted-checksum rejection.
- [ ] **Step 5:** Commit `feat: add deterministic parent training and atomic recovery`.

## Task 10: Insert growth without corrupting optimizer or update streams

**Files:** Create `src/treeneuron/growth/insert.py`, `tests/test_optimizer_growth.py`; modify checkpoint/train only through their existing interfaces.

**Interfaces:** Produce `insert_growth(state: TrainingState, allocation: Allocation) -> GrowthManifest`, `migrate_optimizer(state: TrainingState, manifest: GrowthManifest) -> None`. Use named state and explicit slices, not optimizer positional IDs. Base/head parameter objects and moments are unchanged; new parameter groups use current scheduled LR/shared recipe and zero moments/step.

- [ ] **Step 1:** Write `test_growth_keeps_old_moments_and_predictions`, asserting exact old moment tensors/steps, prediction preservation and parameter counts; `test_new_gate_step_starts_at_zero` checks the first new Adam step; `test_empty_allocation_preserves_run` checks valid zero-growth event; `test_resume_through_growth_matches_uninterrupted` interrupts immediately before and after insertion. Run `.venv/bin/python -m pytest tests/test_optimizer_growth.py -q`; expect missing APIs.

```python
def test_growth_keeps_old_moments_and_predictions(growth_case):
    before, after, old_moments, migrated_moments = growth_case
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name in old_moments:
        assert torch.equal(old_moments[name], migrated_moments[name])
```

- [ ] **Step 2:** Implement insertion, manifest-based migration, scheduler-group extension and checkpoint reconstruction. No resetting old state, learning-rate restart or reinitializing existing weights. Capture training versus allocation RNGs separately; attaching groups cannot shift paired row order.
- [ ] **Step 3:** Add a negative test for unmapped/duplicate parameter names, incompatible optimizer group config, and second insertion; reject before mutating durable state. Test the first post-growth ordinary-parameter update against a manually migrated reference.
- [ ] **Step 4:** Run optimizer-growth plus checkpoint tests on CPU and available selected MPS backend; expect old-state integrity, correct zero/new-step semantics and resume agreement within the recorded same-backend tolerance.
- [ ] **Step 5:** Commit `feat: preserve optimizer state across one-event specialization`.

## Task 11: Implement honest causal capacity and static-gated controls

**Files:** Create `src/treeneuron/baselines/causal.py`, `tests/test_causal_baselines.py`; extend simple baselines only where needed.

**Interfaces:** Produce `build_causal(method_id: str, config: Mapping[str, Any], *, parent: TrainingState | None = None, allocation: Allocation | None = None) -> Predictor`, using B1–B5 IDs from the spec; `ordinary_width_brackets(d: int, target_parameters: int) -> tuple[int, int]`; `widen_parent(state: TrainingState, added_units: int, seed: int) -> GrowthManifest`. Factory returns fresh static models or independent loaded parent forks according to the method’s declared role.

- [ ] **Step 1:** Write tests for widths 94/95 and totals 1693/1711 around target 1697, counted head parameters, zero-outgoing widening prediction preservation/nonzero outgoing gradient, old moment slice mapping and static partial/full counts. Direct-leaf versus centered coordinates must be labeled separately; shared-coordinate formulas agree but independent Adam training is not asserted equal. Run the causal test file; expect missing factory/controls.

```python
def test_nearest_ordinary_parameter_brackets():
    assert ordinary_width_brackets(16, 1697) == (94, 95)
```

- [ ] **Step 2:** Implement continued parent, lower/upper fresh ordinary MLPs, ordinary warm-start widening, static partial/full gated models and B4/B5 adapters. Static partial takes counts, no learned-ID-search overhead. Ensure widening gets the same update/restart policy as selective growth and that its new incoming weights need not receive gradients on the first zero-outgoing step.
- [ ] **Step 3:** Add adapter tests rejecting ensemble-axis/standardized output leakage. If implementing the secondary duplication control now, define halved head/state mapping and explicit symmetry treatment with preservation tests; otherwise mark it deferred in the capability matrix rather than imply it ran.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest tests/test_causal_baselines.py tests/test_simple_baselines.py -q` and add CLI `verify-baselines --track causal`; expect B0–B5 verified and B6/B7 explicitly pending for the evidence plan. No package name substitutes for a tested recipe.
- [ ] **Step 5:** Commit `feat: add matched ordinary and static gated causal controls`.

## Task 12: Execute and report the minimum complete exploratory experiment

**Files:** Create `src/treeneuron/experiments/paired.py`, `src/treeneuron/analysis/exploratory.py`, `configs/explore.yaml`, `tests/test_paired.py`; extend CLI/README/DECISIONS.

**Interfaces:** Produce `run_paired(config: ExperimentConfig) -> Path` returning the study artifact directory, `summarize_exploratory(study_dir: Path) -> dict[str, Any]`; CLI `run --config PATH` and exploratory report mode. The manifest links every fork to its parent, shared observations/stream, score table and actual growth size. No T predictions are generated.

- [ ] **Step 1:** Write integration tests requiring independent sequential fork loads, equal requested/effective counts, per-ID contrast equality, paired row IDs/update allowance, keep and losing branches always retained, sham identical predictions→zero differences, and a zero-score teacher retained with zero growth. Deliberately mutate a sampler cursor and assert the pairing audit fails. `test_exploration_never_loads_T` injects a protected-label loader that raises if called. Run `tests/test_paired.py`; expect missing runner.

```python
def test_zero_score_teacher_is_retained(zero_teacher_allocations):
    top, random = zero_teacher_allocations
    assert top.requested_count == random.requested_count == 16
    assert top.effective_count == random.effective_count == 0
```

- [ ] **Step 2:** Implement serial parent→score→top/random/keep/random-direction forks, plus B0, static partial/full and ordinary brackets/widening. Collect predictions/curves on G and V only; use G for checkpoints and report V descriptively without selecting a confirmatory claim. Log gate/leaf/gradient/spectrum diagnostics and predicted-versus-observed tiny gate-step descent. Standalone versus shared scoring cost boundaries must be distinguishable.
- [ ] **Step 3:** Write an explicitly exploratory starter config: primary `d=16,h=64,m=16,tau=1,epsilon=0.05,sigma=0.1,F=2306,G=V=577`; AdamW LR `1e-3`, epsilon `1e-8`, weight decay `0`; provisional parent/continuation ceilings 2000 updates each, G evaluation every 25 updates, patience 20 evaluations, logical batch 128. These are builder-chosen starting values, not calibrated limits or frozen choices. Device/physical batching uses Task 8’s safe output; never change logical batch for just one fork. Namespace new exploratory seeds separately from references and future confirmation.
- [ ] **Step 4:** Run `.venv/bin/python -m pytest -q`, `verify-math`, `verify-baselines --track causal`, and `run --config configs/explore.yaml`. Execute a separate 2-D positive-control config with the same complete pipeline. Expect complete finite records, valid pairing/insertion/state checks, and evidence of train-loss learning on a positive control; neither top-over-random nor a fixed accuracy threshold is required. If ceilings still truncate improving curves, retain the run and revise exploratory allowances before WP05. Preserve all failed attempts.
- [ ] **Step 5:** Rebuild the exploratory summary from artifacts, append WP04’s decision (including negative evidence) and remaining B6/B7/data/practical gates, then commit `feat: run reproducible paired one-event growth experiments`. Do not commit bulk checkpoints/data; commit code/configs and a curated evidence index with hashes. Proceed only to the next justified evidence task.

## Core completion and handoff

WP04 is complete only after actual local A/B, kernel, layer/score, optimizer/resume, leakage/metric and baseline tests pass and the complete exploratory paired experiment is recorded. Acceptance means the mechanism and harness work, not that allocation helps. WP00–WP04 scientific decisions and limitations must be readable from `DECISIONS.md` without this chat.

Read the evidence plan next. It completes the staged WP02/WP03 practical scope, then calibrates and freezes the design. Do not implement repeated growth, recursive routing, hard dispatch or pretrained downloads merely because this core works.
