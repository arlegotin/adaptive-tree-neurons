# Core Task 5 report — 2026-10-16

Implemented on existing `main`, baseline `a5790ff91aeb2c652cbb751b8754a8147da40357`. No branch/worktree change, subagents, plan/ledger edits, protected T generation/access, predictive fitting, or frozen protocol artifacts. Read Task 5 brief, global constraints, design §7, and TDD/verification skills.

## Public APIs and runner integration

`treeneuron.data.synthetic`:

- `make_teacher(seed, d=16, *, reference_seed=None, reference_rows=65536, regime="isotropic", nuisance_dimensions=0, gate_prevalences=(.1,.5), gate_types=("hard","smooth_0.2","smooth_1.0")) -> TeacherSpec`.
- `sample_teacher(teacher, n, seed, sigma=.1) -> ObservationSet`. Also dispatches ControlSpec to its control sampler; use `sample_control` directly for control-specific default noise.
- `make_development(teacher, counts=None, seeds=None, sigma=.1) -> DevelopmentData`. Both mappings, when supplied, require exactly the case-sensitive keys **`F`, `G`, `V`**. Count values are positive Python integers; seed values are distinct nonnegative Python integers. Default counts are **F=2306, G=577, V=577**. Rejects extra/missing keys, repeated seeds, invalid integers, and any `test_path` argument. Explicit original-2D development requires `sigma=0`.
- `make_from_config(config: ExperimentConfig) -> (TeacherSpec | ControlSpec, DevelopmentData)` is the small bridge for later runner use. It revalidates the Task 2 schema; uses `model.d`, `data.population`, noise/counts/regime/nuisance/gate choices, `seeds.teacher`, `seeds.data`, and **explicit `seeds.reference`**. No new config fields were invented. `components != 4`, `reference_rows != 65536`, and any non-None `n_test` raise ValueError. Unsupported control/gate/regime combinations raise rather than silently dropping requested values. Sample-density/rounding have already resolved the counts in the config module. Input/target standardization is explicitly deferred to the F-only preprocessing module; returned rows are original generated inputs/target units. Optimizer/allocation/contrast seed consumers remain training tasks.
- `evaluate_teacher(teacher, X, return_gates=False)` reproduces the normalized noise-free equation. `return_gates=True` is diagnostic metadata access only.
- `reference_inputs(teacher)` reproduces exactly 65,536 separately seeded Gaussian inputs for normalization diagnostics. It accepts a primary TeacherSpec, not a control spec or observation stream.
- `derive_seed(seed, namespace)` supports declared `exploratory/` seed namespaces only.

`TeacherSpec` contains immutable/versioned vectors, per-component prevalence/type/scale/threshold, uniform gate-type sampling probabilities/options, teacher/reference seeds, fixed reference count, stored raw reference mean/population std, regime/nuisance/rotation metadata, and phase. `to_dict()`/`from_dict()` preserve JSON identities and reproduce equations. `teacher_id` is SHA-256 of canonical metadata. Invalid version/phase/shapes/unit vectors/types/scales/reference statistics are rejected before sampling.

`treeneuron.data.types`:

- `ObservationSet(X, y, row_id, group_id, dates=None, noise_free_target=None, stream_id="", feature_names=())`. Complete synthetic arrays are immutable **NumPy CPU float64**; row/group IDs are immutable **NumPy CPU int64**, not Torch tensors. A training adapter must copy/convert to eager Torch float32. Arrays use immutable bytes backing so write protection cannot be reversed on the stored view. Synthetic groups are singleton row groups. Noise-free targets remain optional diagnostic metadata; dates are optional aligned metadata.
- `DevelopmentData(F,G,V,teacher_id,manifest=None)` checks row/group separation and identical feature schemas, constructs a manifest, and rejects mismatching supplied manifests.
- `SplitManifest` freezes row/group IDs, per-row assignments, dataset/teacher ID, optional fold/date boundaries, phase, data hashes and optional descriptive-only reason. It rejects duplicate row IDs or a group assigned across partitions. Its `assignment_hash` covers dataset identity, immutable assignments, fold/phase and date boundaries. T is permitted as future manifest metadata but no T observation is generated/read here.
- `predictor_matrix(rows, features=None)` selects only `feature_names`; generators set these to `x0,...,x{d-1}`. Teacher state, IDs, dates, labels, normalization and noise-free targets cannot be requested through this allowlist. Teacher gates never augment X.

## Exact seed and identity choices

Seed derivation version is `treeneuron-seeds-v1`. For root integer `s` and namespace `n`, encode UTF-8 `treeneuron-seeds-v1:{n}:{s}`, SHA-256, take the first eight bytes big-endian, mask to 63 bits. Feed this to NumPy **PCG64**. Distinct semantic domains permit the same numeric root without sharing a generator. Streams do not use global NumPy/Torch RNG state.

- Teacher vectors/options: `exploratory/teacher/v1` with teacher root.
- Default reference root, only when no explicit reference seed was supplied: derive teacher root under `exploratory/reference-root-v1`.
- Gaussian reference inputs: `exploratory/reference/gaussian-v1` with reference root. Reference identity is `exploratory/reference/gaussian-v1/{reference_seed}/d{d}`. Reference inputs are never observation rows.
- Gaussian observation inputs: `exploratory/observation/inputs-v1` with per-partition observation seed.
- Observation noise: `exploratory/observation/noise-v1` with the same per-partition root but independent domain.
- Default `make_development` partition seeds derive **teacher.seed** under `exploratory/development/F-v1`, `/G-v1`, `/V-v1`.
- **Config bridge** partition seeds instead derive **config.seeds.data** under those same F/G/V domains. Therefore config data seed changes observations while preserving teacher identity; reference seed changes normalization while preserving input draws for a fixed observation seed.
- Rotation matrix: `exploratory/rotation/v1` with teacher root.
- Nonprimary control vectors: `exploratory/control/{kind}-v1` with teacher root.
- Original 2D reference/input domains: `exploratory/reference/uniform-2d-v1` and `exploratory/observation/uniform-2d-v1`.
- Label permutation: `exploratory/observation/label-permutation-v1` with observation seed. Permutations are independent across partition seeds and independent from the input/noise domains; labels never cross partitions.

An observation stream ID is `exploratory/observation/v1/{teacher_id}/{observation_seed}`. Row i's ID is first-eight-byte/big-endian/63-bit-masked SHA-256 of `{stream_id}/{i}`. `group_id=row_id` for independent synthetic rows. Different observation seeds have distinct sampled rows/IDs in tests; uniqueness/partition overlap checks fail closed on any ID collision. Count changes preserve the prefix IDs/draws. Reusing a numeric seed for teacher/reference/observations is safe because their domains differ; reusing a numeric seed across F/G/V is rejected.

## Primary equation and exploratory choices

Version `independent-regime-teacher-v1` implements exactly four components, independent unit-normal Gaussian vectors u0,u_r,v_r,w_r, independent component prevalence draws uniformly from {.1,.5}, independent gate-type draws uniformly from hard/smooth .2/smooth 1, thresholds Phi^-1(1-p), hard `>=` orientation, and smooth sigmoid((projection-threshold)/scale). It implements the specified .3 sine plus .5 sum of [.3 linear + (2g-1) switched linear], and stores raw mean/population std on exactly 65,536 independent reference rows. Labels add independent sigma*standard-normal noise after reference normalization.

These exploratory sensitivity choices are explicit/versioned, not confirmatory defaults:

- `rare`: all component threshold-region prevalences .01. Default primary prevalence options are replaced by this named coordinate; a conflicting custom option set is rejected. Explicit `(.01,)` is accepted.
- `axis_aligned`: each of the 13 vectors independently chooses a uniform signal coordinate and independent sign.
- `rotated`: Gaussian unit vectors followed by a common separately seeded QR orthogonal matrix restricted to the signal subspace; QR diagonal signs are fixed nonnegative. Matrix is stored. This rotates a Gaussian teacher, not an axis-aligned paired teacher.
- `nuisance_dimensions=k`: final k Gaussian coordinates have exactly zero coefficients in every teacher/control vector. `make_control("nuisance",...)` uses k=4 if not supplied; k must be less than d.
- Any positive d is available as an exploratory coordinate; the primary default is exactly d=16. Sampling remains Gaussian in all coordinates, including nuisances.

## Control APIs and equations

`treeneuron.data.controls.make_control(kind, seed, d=16, **same_teacher_coordinate_keywords) -> TeacherSpec | ControlSpec` accepts six mandatory control names below, plus `primary`, `rare`, `axis_aligned`, `rotated`, and `nuisance` returning TeacherSpec. `ControlSpec` uses `exploratory-controls-v1`, explicit equation/coefficient/distribution/normalization metadata, reference statistics and optional primary base state; JSON round trips restore nested specs. Control vector counts and invalid versions/phases/kinds are validated.

`sample_control(control,n,seed,sigma=None)` returns the same ObservationSet contract. Default sigma is .1 except original 2D, where it is zero and nonzero noise is rejected. `control_target(control,X)` exposes the explicit pointwise equation; row-label permutation raises because it has no conditional noise-free pointwise function. All Gaussian control vectors are unit vectors. Exact choices:

| kind | noise-free target / label rule | sampled inputs |
|---|---|---|
| `globally_affine` | .3 + u'x | independent N(0,I_d) |
| `additive_sine_quadratic` | .3 sin(u'x) + .3((v'x)^2-1) | independent N(0,I_d) |
| `pure_noise` | f=0; y=sigma*xi, diagnostic f retained as zero | independent N(0,I_d) |
| `homogeneous_smooth` | .5 sum over four independent sin(u_r'x) components | independent N(0,I_d) |
| `random_label_permutation` | independently permute the matching primary teacher's noisy labels within each sampled partition; noise_free_target=None | matching independent primary Gaussian draws |
| `conditional_slope_2d` | x2 if x1<0, otherwise 1-x2; exactly no noise | independent Uniform(-1,1) and Uniform(0,1), requires d=2 |

Nonpermuted control target units are the table's exact equations, with `target_normalization="none"`. Their separate 65,536-row reference is diagnostic only, recording actual empirical mean/std and consuming the requested reference seed; it does not alter target equations. A zero reference target std is valid for pure noise. The permutation records its primary base's raw reference normalization and `target_normalization="primary_base_only"`. F-only model preprocessing is still required later. Original-2D points (-.5,.2),(.5,.2),(0,.2) give .2,.8,.8. Non-gated controls reject custom gate options and rare-gate regimes; original 2D also rejects regime/nuisance alterations.

`oracle_diagnostic_features(teacher,X)` is a separate named diagnostic factory returning an `OracleDiagnosticFeatures` record with `information_advantaged=True`, `purpose="oracle_gate_least_squares_diagnostic"`, and basis `[1,x,(2g_r-1)x for r=0..3]`. It returns d+4d+1 columns (81 at d16), separately names oracle features, does not fit least squares, and never modifies ObservationSet.X. It must never enter the fair-rival path.

## TDD and verification evidence

1. Wrote synthetic/access tests before implementation. `.venv/bin/python -m pytest tests/test_synthetic.py tests/test_access.py -q` RED: two expected missing-module import collection errors for the absent data package APIs.
2. Implemented primary/types/control modules: same focused command GREEN **35 passed in .89s**.
3. Added serialization/config/invariant cases before fixes. RED **12 failed, 42 passed in 1.40s**: malformed teacher/control specs did not reject, rare custom prevalence was silently replaced, and inconsistent F/G/V feature schemas were accepted. JSON round trips/config controls/namespace tests already passed.
4. Added invariant validation, conflicting-rare rejection and schema agreement. Focused GREEN **54 passed in 1.21s**.
5. Full fresh project suite `.venv/bin/python -m pytest -q`: **175 passed in 22.76s**, no skips/warnings/failures. `git diff --check` exit 0. No subsequent production/test edits.
6. Manual config-bridge evidence draw (no fitting, T, or scoring): Python 3.12.13 arm64, NumPy 2.5.4, SciPy 1.18.1; config defaults and study_id `task5-evidence`. Counts 2306/577/577. Teacher ID `6491eae8fe7cdb07709300641a79337d4a25202725ebfff08d388816186149c4`; reference seed=5, mean=.11450709125337387, std=1.087194454411077. Data-root=1 derived F/G/V seeds: `8184935959063033473`, `4520633044469760199`, `3332795942548724832`. Dataset ID `8d5ea8202c5a57a200872fe5a9acfa893d63a775a1196b35ab471d17e07dc19e`; assignment SHA `798fcc02baaab670198813e0f5b0fff8c0e226014920bf82ce5b4e61033d5eb0`. F.X array hash `c0ffc20ec4cf3183c1021b86c433bfdac9e21cec2987e5569fa3f4b2af5888ad`; F.y `9af60cd4621cf29f33ec1bda07fafe4a2ce975ab6ae93667cb28568a7dfd2369`. Array hashes prefix dtype and shape before contiguous bytes.

Tests cover exact hard/smooth orientation and hand-computed raw equation, normalization on the reproduced fixed reference, independent noise/input/reference domains, all four independent component choices, row IDs and schema immutability, normal/invalid configs, all controls/pointwise equations, serialized identities, development-only access, feature allowlist, oracle labeling, and malformed/overlapping seed maps. No scientific benefit, optimizer/model fit, CPU/MPS training tolerance or protected-evaluation capability is claimed. Reproducibility evidence is for the stated NumPy/SciPy environment; cross-version QR/libm byte identity is not promised. No blocker observed.

## File SHA-256

| file | SHA-256 |
|---|---|
| src/treeneuron/data/types.py | `2858e78b6106a01979f6060a1a23cd4d9bf6ef1ce74f7e8caf8b6c34be024a5a` |
| src/treeneuron/data/synthetic.py | `7e4ed604113a4177b8f256dd3e536b9371cfbf02e2c51dd8f455ea0df94433bb` |
| src/treeneuron/data/controls.py | `3b53b61af03d7b3795ebb423a387fda9d6ee8e190085da0687d04386603b06e8` |
| tests/test_synthetic.py | `0a4a919b066997fc42d5288cdcd82c571fe1fc8fceeaa5de842560b3c4c2e2ad` |
| tests/test_access.py | `3936f38f70810d30499d8c852a97e363ba8fc49d1d65ee71e29704ac1cb9209c` |

## Commit

`b877fc7` — `feat: add independent synthetic teachers and control populations` (five task-owned files, 1,082 insertions). Authorized escalated staging/commit succeeded. `git diff --cached --check` passed before commit; working tree clean afterward. This ignored local report follows prior task reports and is intentionally outside the commit.

# Task 5 review fix round 1 — baseline b877fc7

This section supersedes the initial report's additive-control equation, rotated-teacher interpretation, control version and dataset/assignment hashes. No actual predictive experiment or protected evaluation had occurred. Implemented the controller's scientific rulings in the existing branch without new agents, plan/ledger edits, protected T access, preprocessing, missing-data handling or Torch-adapter work.

## Corrected behavior and version choices

1. **Observed-coordinate additivity.** `additive_sine_quadratic` now computes exactly `.3*(sum_j u_j*sin(x_j) + sum_j v_j*(x_j**2-1))`, with independent unit coefficient vectors u,v and complete Gaussian inputs. Applying sine/quadratic before the weighted sums removes mixed observed-coordinate interactions. A dense two-coordinate fixture tests the mixed finite difference is zero at absolute tolerance 1e-14; the previous implementation produced -0.28371898593468325. The equation metadata is `.3*(dot(u,sin(x)) + dot(v,x**2-1))`.
2. **Paired rotation.** A rotated teacher is now the **same axis-aligned base teacher** for the same teacher/reference seeds and options. The 13 base vectors, component prevalences/types/scales/thresholds use exactly the axis-aligned RNG draws. A separately seeded orthogonal Q rotates each coefficient row `a_rot=a_axis @ Q`; Gaussian reference rows and observation rows use the matching `x_rot=x_axis @ Q` transform. Nuisance coordinates remain unchanged and excluded from coefficients. The inverse relation is `f_rot(x_rot)=f_axis(x_rot @ Q.T)` on the active coordinates; normalization uses the paired transformed reference. Observation noise uses its original independent domain and is unchanged. Rotation matrix Q is stored. Same-seed labels agree to floating-point arithmetic tolerance, not a claimed exact byte equality.
3. **Versions/capabilities.** Isotropic/axis/rare primary versions stay `independent-regime-teacher-v1`; their equations/streams/outputs are unchanged. Rotated teachers now require `axis-paired-rotated-teacher-v2`; a v1 artifact claiming rotated semantics is rejected. All ControlSpec records now require `exploratory-controls-v2`. Controls store validated `supported_regimes` and `coordinate_semantics` fields: rotated controls use `axis_paired_orthogonal_v2`, other controls use `direct_observed_coordinates_v1`. Additive supports only `("isotropic","axis_aligned")`; original 2D only `("isotropic",)`; projection/noise controls retain isotropic/axis/rotated; permutation also retains its previously supported rare primary base. **Additive+rotated is explicitly rejected** because arbitrary coordinate rotation would destroy observed-coordinate additivity; original 2D retains its exact independent uniform input distributions/zero-noise equation and rejects rotation. Already-supported affine/homogeneous/permutation rotations pair coefficients, reference inputs and observation inputs consistently. No new population/regime combination was added. Control v1 metadata is intentionally rejected rather than interpreted under the new equation/coordinate semantics.
4. **Ordered schema in identity.** SplitManifest stores immutable ordered `feature_names` and `feature_schema_hash=canonical_hash({"feature_names":names})`. Dataset identity includes both names/hash in addition to teacher/data/row identity. Assignment hash includes names/hash explicitly. DevelopmentData creates the matching schema manifest and rejects an older supplied manifest after all F/G/V feature names are consistently renamed. Renaming or reordering a schema now changes dataset/schema/assignment hashes without changing numeric arrays. Generic standalone manifests may omit the schema when no feature matrix is attached; generated development manifests always bind it.
5. **Load/construction validation.** ControlSpec enforces exact required coefficient names/values for its version, exact equation/input-distribution/normalization strings, capability/coordinate consistency, orthogonal rotation presence exactly for a rotated regime, nuisance-vector exclusion, original-2D nuisance exclusion, and permutation/base metadata agreement. Malformed metadata now fails during construction/load rather than causing a later missing-coefficient KeyError or silently contradicting the equation. TeacherSpec validates thresholds against `Phi^-1(1-prevalence)` with zero relative tolerance and absolute tolerance 1e-12. The controlled gate-equation test fixture now declares .5 prevalences matching its zero thresholds.

The public generation/config-bridge/count APIs and all seed-domain derivation formulas are unchanged. Rotated reference stream identifiers now additionally bind the rotation version/matrix hash, while underlying axis Gaussian/reference/observation/noise roots remain paired. Isotropic primary teacher identity and F.X/F.y byte hashes in the initial report remain exact; only its development dataset/manifest identities change because they now bind schema. The additive control is an intentional new equation under controls v2; its previous target bytes are not retained.

## TDD / numerical / full-suite evidence

- First regression run before fixes: `.venv/bin/python -m pytest tests/test_synthetic.py tests/test_access.py -q` **15 failed, 55 passed in 1.24s**. Failures covered dense additivity, all six coefficient/equation/distribution/normalization corruptions, threshold/prevalence disagreement, both paired-rotation nuisance settings, affine/homogeneous/permutation rotation, additive-rotation rejection, and schema rename identity.
- First GREEN: **70 passed in 1.23s** after implementation.
- Follow-up compatibility/load tests exposed **2 failed, 71 passed in 1.37s**: new capability validation inadvertently excluded the existing rare permutation base, and original-2D loading accepted false nuisance metadata. Restored rare-base capability and rejected false nuisance declarations (also for dense affine vectors).
- Final focused suite after stricter zero-relative-tolerance paired-label assertions: **73 passed in 1.35s**.
- Final full suite: **194 passed in 18.09s**, no skips/warnings/failures. `git diff --check` passed. No subsequent production/test edits after this final run.
- Direct numerical evidence, seed=51/reference=52/observation=53: paired primary max absolute signal/label difference **3.552713678800501e-15** with no nuisance, **1.5543122344752192e-15** with three nuisance coordinates (N=200). Paired-control max label differences at N=50: affine **2.220446049250313e-15**, homogeneous **8.881784197001252e-16**, permutation **1.3322676295501878e-15**. Tests require absolute 1e-13 with rtol=0.
- Independently loaded `b877fc7`'s synthetic module from `git show` and compared axis-aligned seed51/reference52/N200/observation53/sigma.1 to the fixed implementation. Teacher ID/X/y hashes were **exactly equal**: teacher `2fce705068f0efc9a72e15c03835dec11b3ec39d7811e210b6530c154b311293`; X `74f95367e57f1c2d8b525cbc56b95e1c5220d65870965069a56af0e4f5a17099`; y `3d2c6cc48669b917c1166f617a0ae88b3158016a6392fc2f368966fae2639ff6`. The isotropic default teacher/F.X/F.y baseline hashes are also asserted unchanged in the test suite.
- Corrected default config development identity for study `task5-evidence`: dataset `700e34e8ab837adc77784a4d34a292b891d12d5bc69ee461870035070815e97e`; assignment `f54318180625a73a23ea367648a9eae8223ff23c62eea67f060aef42cb4fab10`; ordered feature schema `a49a80e3cbcbf9796cfda7f69115f92093d746c116a93415c1fc7f7824f8b125`.

## Fix-round file SHA-256

| file | SHA-256 |
|---|---|
| src/treeneuron/data/types.py | `0e37b3aa2f5a35f6efe54a4cdb54adcf2e43b5216333fd96df1287362054c148` |
| src/treeneuron/data/synthetic.py | `980b91841713263117ae6a90a28617682a4f7e69d61e42e28bc071f9973084e3` |
| src/treeneuron/data/controls.py | `1af8cfc407264c94c6f1e96c074ac1ea7b85733c55f61dc745f7481947f2f5a2` |
| tests/test_synthetic.py | `d54cd8e4b2888e38ad01bb4b53397523bd7c833bb7cd86de7c47f5ec4326f061` |
| tests/test_access.py | `0ca7bc4af9c451fcf63e738e07e540953abffc461b1ce9be6f7f2030a26369a6` |

Fix commit: `272e4ea` — `fix: enforce additive controls and paired rotation data contracts` (five task-owned files; 268 insertions / 29 deletions); authorized escalated staging/commit succeeded, cached diff check passed, working tree clean afterward. The ignored local report remains outside the implementation commit. Future raw/missing transfer data types and Torch copies remain downstream obligations. No scientific outcome or protected evaluation claim is made.
