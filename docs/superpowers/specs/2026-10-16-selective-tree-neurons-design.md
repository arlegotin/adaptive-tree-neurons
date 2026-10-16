# Selectively growing tree-neurons: proof-of-concept specification

Date: 16 October 2026. Status: implementation handoff draft; no predictive experiment has run. Requirements source: the user-supplied research proposal, cut off on the same date. Reference identifiers and evidence status are preserved in [the evidence record](../../research/2026-10-16-evidence-and-machine-record.md). The accompanying plans implement this specification; they do not authorize changing a scientific claim after observing its test results.

## 1. Intent and success

Build a reproducible local experiment answering: at a fixed added-capacity budget, can a training-derived score choose which learned ordinary neurons should acquire conditional affine specialists better than uniformly random placement?

The primary deliverable is evidence about capacity allocation, including a defensible negative or inconclusive result. A working neural layer, fitting a piecewise function, fewer named neurons, or one dataset win is insufficient. Intended users are researchers and builders of compact predictors; no production need or application-specific utility threshold has been established.

An affine-leaf soft stump is an established hierarchical mixture/gated family (R01–R08), not a new computational primitive. The exact score/insertion combination has provisional differentiation only. R12’s full text was inaccessible in the supplied research; unresolved overlap blocks broad novelty claims, not empirical diagnosis.

### Goals

- Verify the algebra, initialization, optimizer migration, backend, data handling, and evaluation before interpreting predictive outcomes.
- Isolate placement from contrast initialization, added parameters, training time, and an existing static gated model.
- Execute one first-layer growth event in scalar regression, using paired forks and independent synthetic teachers.
- Measure complete local costs and uncertainty; protect F/G/V/T information boundaries.
- Proceed to a fixed transfer suite only through an explicit scientific and baseline-feasibility gate.
- Preserve every failed, unfavorable, interrupted, and excluded run with reasons; regenerate conclusions from raw records.

### Non-goals and unacceptable outcomes

The core does not implement recursive trees, repeated growth, pruning, learned temperature, sparse/hard execution, custom Metal kernels, distributed architecture search, or biological/semantic interpretability. These require fresh evidence-triggered extension protocols. Soft routing computes both specialists and makes no conditional-execution claim. Broad novelty, universal tabular superiority, clinical effectiveness, and freedom to operate are outside scope.

Unacceptable outcomes include tuning against T, silently weakening a baseline, counting seeds/folds as independent tasks, omitting failed searches from cost, deleting losing runs, presenting Linux checks as M3 measurements, or deepening a model solely to rescue an unfavorable stump result.

## 2. Platform, authority, and evidence boundaries

Work in the current branch. Do not create a worktree or switch branches unless a later user instruction changes this constraint. This planning request produces documentation and commits only. Package installation, source acquisition, model implementation, and experiments belong to execution of the plan.

Observed laptop: M3 Max, 36 GiB physical memory, 14 logical CPUs, native arm64, macOS 26.7.1, about 21 GiB available disk, AC power. Native Python 3.14.6, `uv`, Git, and Xcode are available; default Python has no Torch. These observations are recorded with command provenance; current usable RAM, system load, MPS support, and future disk headroom remain to be measured.

Core execution uses native arm64 Python/PyTorch with a resolved lock. First try Python 3.12 where supported, then choose the newest mutually compatible stable combination after checking the actual OS and package support. Do not assume a universal baseline environment or a PyTorch minimum from mutable documentation. Keep incompatible baselines isolated; never downgrade the core to ANT’s Python 2-era stack. Do not update macOS, remove environments, install privileged system components, or accept unauthorized asset agreements to make a method fit.

Training uses float32 eager PyTorch. Mathematical checks and score construction use CPU float64. Primary neural execution uses no reduced precision and no implicit MPS fallback. Validate MPS with fallback disabled; unsupported kernels select a faithful whole-method CPU path or a parity-tested equivalent implementation. CPU execution answers H1 but changes the hardware-performance scope. `foreach=False` is the conservative AdamW path.

WP00 rechecks hardware, package metadata, source terms, and storage before acquisition. Reserve at least `max(8 GiB, 0.25 × physical RAM)` initially: 9 GiB on the observed machine. Budget from current headroom and the MPS recommended working set; preserve normal allocator safeguards. Forecast install/download/unpacked/checkpoint/cache storage before optional environments or weights. Never delete user data to create headroom. Code and weight licenses are separate; preserve notices and do not commit downloadable weights.

## 3. Scientific contract

| Hypothesis | Scope and main contrast | Interpretation |
|---|---|---|
| H0-E | Shared-coordinate tree versus affine-bypass gated expression | Disagreement indicates a bug/unequal coordinates, never expressive novelty. |
| H1-A, sole initial confirmatory hypothesis | Top-ranked versus uniformly random allocation, identical parent, actual growth count, per-unit contrast construction, data stream, and continuation allowance | Positive paired teacher-level interval supports placement on the registered synthetic population only. |
| H2-C | Grown method versus ordinary capacity/growth and static partial/full gated families | Required attribution; superiority claims need a predeclared multiplicity family, otherwise descriptive. |
| H3-T | Practical procedure versus V-selected strong locally executed competitor on the fixed transfer suite | Separate finite-suite assessment with complete search/fit/inference costs; no universal ranking. |
| H4-H, extension | Faithful hard execution versus soft execution with declared deployment tolerance | Actual latency and prediction discrepancy must be measured; cannot follow from H1–H3. |

Terminology: depth counts routing decisions; a stump has one gate and two leaves. The selected tree-neuron produces a scalar **preactivation**, followed by ordinary ReLU. Growth adds active parameters; allocation chooses unit IDs at fixed added capacity. Parameter, update, time, and memory budgets are distinct. Unit IDs in one symmetric dense layer do not discover a different static architecture up to permutation; any placement effect depends on inherited functions and trajectory.

## 4. Model and algebra

Let `X=[x,1]`, `b_j=w_jᵀX`, `d_j=v_jᵀX`, `g_j=sigmoid(a_jᵀX/τ)`, and `s_j=2g_j-1=tanh(a_jᵀX/(2τ))`. The selected unit is

```text
z_j = b_j + s_j d_j
h_j = ReLU(z_j)
leaves L_j = b_j - d_j, R_j = b_j + d_j
```

An unselected unit is `ReLU(b_j)`. The main model is `x → SelectiveAffineLayer(d,h) → ReLU → Linear(h,1)`. Initial synthetic coordinate: `d=16,h=64,m_requested=16,τ=1,epsilon=0.05`. Sensitivity contrast magnitudes are 0.01 and 0.2. No dropout, BatchNorm, learned normalization, embeddings, or ensemble in the causal experiment.

For arbitrary affine leaves, `T=(1-g)L+gR=L+g(R-L)`. Compare the two formulas using the same tensors and probe gradients. Independently optimizing `(L,R)` and `(base,delta)` with AdamW is not the same optimizer or regularization problem. Without intervening activation, a scalar head collapses all base affine terms into one affine term; this collapse does not transfer unchanged through ReLU. `ReLU((1-g)L+gR)` and `(1-g)ReLU(L)+gReLU(R)` are distinct models; identity and leaf-wise activation are labeled ablations only.

Initialize every added gate weight/bias to zero and its contrast to a nonzero stored `v_j`; predictions are preserved. Initially gate gradients can be nonzero and contrast gradients are zero. Both factors zero is the required dead-insertion negative check; it does not invalidate every duplicated-expert initialization.

Layer parameters: `P=(h+2m)(d+1)`. One-hidden-layer scalar model: `P0=h(d+2)+1`; growth adds `2m(d+1)`. At the initial coordinate: base 1,153, added 544, total 1,697. Ordinary parameter brackets are width 94 (1,693) and width 95 (1,711). Include heads and every later layer in any expanded model. Do not pad with unused parameters.

An ordinary affine unit evaluates one projection; a soft stump three; a hard stump still approximately two plus dispatch. A full depth-D affine-leaf binary tree has `(2^(D+1)-1)(d+1)` parameters; constant leaves have a different count. Record operation estimates separately from timing.

Hardening uses `R` when `g>=0.5`, otherwise `L`. Its exact discrepancy is `abs(Tsoft-Thard)=min(g,1-g)*abs(R-L)`. ReLU cannot increase it. Entropy alone is insufficient; leaf disagreement matters. A downstream Lipschitz estimate is not a task-risk guarantee.

### Layer interface

`SelectiveAffineLayer(d: int, h: int, tau: float = 1.0)` owns `base: Linear(d,h)`, sorted `grown_ids: int64[m]` as a buffer, and compact `gate: Linear(d,m)` / `contrast: Linear(d,m)` only when `m>0`. It returns preactivations `[B,h]`; the caller owns ReLU. No dormant per-unit trees.

```python
z = self.base(x)
if self.grown_ids.numel() == 0:
    return z
s = torch.tanh(self.gate(x) / (2.0 * self.tau))
selected = z.index_select(1, self.grown_ids) + s * self.contrast(x)
return z.index_copy(1, self.grown_ids, selected)
```

`grow(ids: Tensor, contrasts: Tensor) -> GrowthManifest` is one event only. IDs are a 1-D CPU int64 tensor; contrasts have shape `[m,d+1]`, the base parameter dtype/device, explicit bias in the last coordinate, and finite values. Sort IDs and corresponding contrast rows together. Reject duplicates, invalid ranges/shapes/dtypes/devices, nonfinite values, nonpositive/nonfinite τ, and a second event. Empty growth records a completed zero-growth event with no new modules. Export configuration reconstructs dimensions, τ, event state, and IDs before loading tensors. The manifest names old/new parameters and identity/slice mappings for optimizer migration. This device/input convention is an implementation choice in this handoff.

## 5. Score and allocation contract

At the G-selected parent checkpoint, evaluate F deterministically. For each row, `e_ij=∂ell_i/∂z_ij`, including the whole downstream derivative and PyTorch’s zero ReLU derivative at exactly zero. Differentiate a **sum of row losses**, not a mean with an uncorrected extra batch factor. Accumulate CPU float64 moments, bounded by a unit/block memory limit:

```text
M_j = (1/n_F) Σ_i e_ij X_i X_iᵀ
M_j ← (M_j + M_jᵀ)/2
q_j = eigenvector with largest absolute eigenvalue
sign(q_j): its first largest-magnitude coordinate is positive
v_j = epsilon × RMS_F(b_j) × q_j / max(RMS_F(X q_j), 1e-12)
u_j = M_j v_j / (2 tau)
S_Adam,j = Σ_k u_jk² / (abs(u_jk) + adam_epsilon) / [2(d+1)]
S_SGD,j = ||u_j||² / [2(d+1)]
```

Adam ranking uses the actual optimizer epsilon, initially `1e-8`. For new zero-moment gates its numerator predicts first-step gate-only descent divided by added parameters, omitting a common learning rate. Old-model updates, weight decay, and higher-order terms are outside that local prediction; compare it to a tiny gate-only step with the parent/contrast fixed. It is a train-loss heuristic, not an optimal-split or generalization theorem. The eigenvector criterion and RMS/Adam changes are different optimization problems.

Record preactivation/projection RMS, eigenvalues, absolute spectral gap, contrast, gradient, both scores, and eligibility reason. Near spectral ties are recorded instability, not automatic exclusion. A unit with zero preactivation RMS, zero score, or a nonfinite statistic is ineligible and retained in the diagnostic table. A nonfinite model/loss is an invalid run. Sort positive-score ties by original unit ID. Set `m_eff=min(m_requested, eligible_count)` **before** forking; top and random use the same count. Zero-growth teachers remain in all results. Never create zero-score branches to obtain headroom.

Top selects largest Adam scores. Random selects a uniformly sampled subset without replacement from the same eligible set, using its own allocation seed and **the same unit-specific eigenvector contrast**. The direction ablation uses the selected top IDs with independent Gaussian directions, sign-normalized and RMS-scaled by the same rule; redraw a zero/nonfinite direction deterministically with logged attempt count. SGD ranking is an explicit alternate allocation policy. Positive ReLU rescaling with reciprocal outgoing-head rescaling and unit permutation must preserve scores numerically; optimizer trajectories are not claimed invariant.

Dense work is approximately `O(n_F h(d+1)^2)`, storage `O(block_units(d+1)^2)`. CPU float64 scoring from an MPS parent is an explicit scoring phase, not hidden fallback during training. A matrix-free, randomized, clipped, or low-rank score is a named future variant requiring ranking parity checks.

## 6. Training, forks, and recovery

Train the ordinary parent on F and select a checkpoint on G. Restore its complete best-checkpoint state, including optimizer/scheduler/sampler/RNGs, rather than combining best weights with the last update’s state. Evaluate every continuation on G at update zero as well as at the shared intervals, so the unchanged checkpoint remains an eligible result for all forks. Freeze causal hyperparameters on exploratory teachers; no per-confirmatory-teacher HPO. Save model, optimizer, scheduler, every RNG, data order and sampler cursor, preprocessing, update count, code/lock/split/config hashes. Base stopping rule, continuation allowance, validation interval, and patience become numeric WP05 outputs based on exploratory curves; confirmation rejects absent values. G ties choose the earliest checkpoint under strict loss improvement, an implementation choice that avoids ambiguous resume behavior.

From the same complete parent state run serial forks: top, random, keep/continued parent, and top IDs with random contrast. Add static gated and ordinary-capacity/growth controls. Pair post-growth row orders and **logical optimizer update counts**, keep batch size/checkpoint rules identical, and restore each fork’s initial RNG state. Record separate seeds for allocation/contrast randomness so allocation does not consume the training stream. Inference must be preserved immediately before the first post-growth update. Do not run forks concurrently on the laptop.

Preserve old optimizer states by stable name and explicit tensor mapping. New gates/contrasts receive zero Adam moments and their own zero step state; initialize lazily only if the optimizer’s semantics agree. Their initial learning rate is the current scheduled rate for the corresponding old group; weight decay and subsequent schedule follow the shared recipe. Do not restart the old schedule or reset old moments. Record group hyperparameters and scheduler position. Widening requires an explicit slice map; duplication additionally requires declared symmetry handling. Test behavior through the growth boundary on both available backends.

Checkpoint writes use temporary sibling files, checksum, atomic rename, and a durable last-valid pointer. Resume checks topology/config/code/lock/split identity before optimizer loading; mismatches fail without overwriting artifacts. Checkpoints include Python/NumPy/Torch CPU and available MPS RNG states and the growth manifest/contrasts. The uninterrupted and interrupted/resumed same-backend run must agree within the recorded deterministic tolerance; cross-backend bitwise equality is not promised.

Retain OOM and interruption events. Do not resume mid-update. A changed physical batch uses verified accumulation to preserve the logical update or restarts the whole affected paired comparison. Prediction batching/serial ensemble execution can be a faithful fallback after parity checks; smaller context, ensemble, precision, or model size is a differently named baseline.

Always report the causal top/random outcomes even if keep wins. A practical algorithm may let G select keep versus grow; that choice is itself tested by the outer evaluation and included in cost.

## 7. Data and information boundaries

F fits model weights, transforms, spline knots, and foundation context. G selects checkpoints and growth/keep. V selects hyperparameters and the eligible-baseline champion. T labels are read only by the protected evaluator after protocol validation. No primary refit on F+G+V. A full-development refit is a separately declared later analysis. Internal vendor splits of F are logged. Document any transductive use of T features and test prediction-batch composition dependence.

For causal numerical models, F-fitted population mean/std standardize inputs; zero feature scale becomes 1 and enters a constant-column manifest. Numerical missingness uses F medians and missingness indicators; an entirely missing F column uses fill 0 plus an explicit all-missing flag, a handoff implementation choice. For synthetic complete data this has no effect. Targets use F-only mean/population scale, and predictions return to original units. A constant target uses scale 1 for numerical fitting, but normalized risk is undefined. Reject missing/nonfinite labels. Categorical unknowns use a reserved category; neural encoders fit F only; CatBoost gets raw categories. Vendor-native transformations remain intact in the practical track.

### Synthetic population

Independent teacher/data/optimizer/allocation seeds are separately namespaced by exploratory or confirmatory phase. For each teacher draw inputs `x~N(0,I_16)`, independent unit Gaussian vectors `u0` and `u_r,v_r,w_r`, `r=1..4`. For each component independently choose prevalence from `{0.1,0.5}` and gate type uniformly from hard threshold, smooth scale 0.2, smooth scale 1.0. This uniform type distribution is the handoff’s explicit interpretation of the brief’s unspecified sampling probabilities. The hard gate is `1[v_rᵀx>=Phi^-1(1-p)]`; smooth is `sigmoid((v_rᵀx-threshold)/scale)`. The prevalence is exact for the threshold region; smooth gate mean need not equal p.

```text
f_raw(x) = 0.3 sin(u0ᵀx)
           + (1/sqrt(4)) Σ_r [0.3 u_rᵀx + (2g_r(x)-1) w_rᵀx]
```

Normalize f by its population mean/std estimated on 65,536 separately seeded reference observations, stored with the teacher and never reused as fitting/selection/test rows. Then `y=f(x)+sigma*xi`, `xi~N(0,1)`, primary `sigma=0.1`. Teacher gates and normalization parameters are never predictor features. Store noise-free f for diagnostic risk.

Primary `P0=1,153,n_F=2P0=2,306,n_G=n_V=ceil(n_F/4)=577,h=64,m=16`. Sample-density sensitivities are `{0.5,2,8}P0`, with integer rounding declared in configs; use ceiling for fractional rows. Noise-free/high-noise, rare regimes, axis-aligned/rotated regimes, nuisance dimensions, and h/depth are exploratory coordinates. Mandatory controls: globally affine, additive sine/quadratic, pure noise, independent random-label permutation, the original 2-D conditional-slope target, and homogeneous-complexity smooth components. Oracle-gate least squares is information-advantaged diagnosis, never a fair rival. Nested gates appear only in a depth-2 extension.

The original 2-D target is `f(x1,x2)=x2` when `x1<0`, and `1-x2` otherwise. For the executable positive control, independently sample `x1~Uniform(-1,1)` and `x2~Uniform(0,1)` and add no noise. The equation comes from the brief; this input distribution/noise choice is a handoff decision because the brief did not supply one. Keep this control separate from the primary Gaussian teacher population. A finite soft gate approximates its jump; exact discontinuity fitting is not an acceptance requirement. Use the normal F-fitted preprocessing and original-unit metrics.

Fresh teacher plus independent dataset is the H1 independent unit. Average optimizer/allocation repetitions inside each unit. F/G/V/T observations are independently drawn conditional on the teacher. H1 hyperparameters stay fixed across teachers.

Choose one confirmatory T size using exploratory paired row-loss variance and between-teacher variance: `max(1,ceil(100*var_row_difference/var_between_teacher))`, using conservative pilot estimates. If the denominator is unresolved/zero, use the estimation/equivalence branch rather than inventing infinite precision. Freeze the count before generating confirmation; never enlarge tests selectively after seeing effects.

### Fixed transfer suite

Five observational datasets and one simulation, selected before comparative results. Verify exact raw bytes/dimensions/licenses, record SHA-256 and immutable row/group IDs. The supplied research reports CC BY 4.0 for all six; verify at acquisition. Retry official routes if unavailable, then block the dataset rather than substitute silently.

| Dataset | Schema / target | Leakage and grouping contract |
|---|---|---|
| Airfoil, UCI 291 | 1,503 rows, 5 inputs; sound pressure in dB | Group angle, chord, speed, displacement thickness; omit frequency from group key. Verify numeric values and actual groups. |
| Concrete, UCI 165 | 1,030 rows, 8 inputs; strength in MPa | Group seven ingredient values across age; keep age as a feature. |
| Energy, UCI 242 | 768 simulated cases, 8 inputs; heating load | Exclude cooling load. Group geometry by relative compactness, surface area, wall area, roof area, and overall height; exclude orientation/glazing from key. Verify redundancy/group counts from raw schema; report separately. |
| CCPP, UCI 294 | 9,568 rows, 4 inputs; MW | Verify the five workbook sheets’ row multisets; use one copy, never stack. Group duplicate inputs; no temporal claim without timestamps. |
| Parkinsons, UCI 189 | 5,875 rows, 42 people; motor UPDRS | Features: age, sex, test time, 16 voice measures. Exclude subject ID and total UPDRS; group by subject. Interpolated targets; no clinical claim. |
| Bike, UCI 275 | Hourly 17,379 rows; cnt | Drop casual, registered, instant; dates define chronology. Use documented calendar/weather categories and numerical weather variables; no future aggregates. Observed-weather retrospective prediction only. |

For non-temporal tasks, outer five folds assign whole groups by seeded hash and greedy row balancing using no labels. Each outer development complement is split approximately 70/15/15 F/G/V using group balancing, at least two groups each. With fewer than six development groups, report descriptively or amend before test access; never switch to rows. Exact duplicate inputs stay together, preserving multiplicity unless a source establishes accidental duplication. Numerical group keys use source values without learned rounding.

For Bike, sort unique dates, and define five consecutive T windows bounded by date-index quantiles 50%,60%,70%,80%,90%,100%. Use floor of `q*number_of_dates` as the half-open boundary index, with the final boundary equal to the date count; store exact dates. Each training prefix precedes its T window; chronological F/G/V nominal fractions are 70/15/15. Remove the 24 hours immediately preceding every G/V/T start from the earlier partition. Require nonempty partitions after embargo. No future-fold fitting for an earlier prediction.

T features/labels remain in separate artifacts; preparation may inspect schema, group keys, dates and counts without fitting or evaluating on T. Enforce access procedurally and log it; local owner access is not cryptographically prevented. Transforms must not change when only T labels change.

## 8. Comparison families and tuning

Track A is paired causal attribution: same transforms, parent, updates, label access and checkpoint rule; exact counts or honest ordinary brackets. Track B is practical local comparison: faithful method-native recipes, complete costs, common measured wall-time budgets plus quality–cost curves. Track C contains nonlocal/published results only; never mix their scores into a controlled local table.

| ID | Required implementation / role |
|---|---|
| B0 | Mean and V-tuned ridge; verify ridge against closed form. Add F-knotted per-feature spline ridge as a cheap additive control. |
| B1 | Continued ordinary parent: same architecture and allowance. |
| B2 | Fresh wider MLP at nearest lower/upper parameter brackets; function-preserving widening of parent with random incoming weights and zero outgoing weights. Secondary duplication copies a unit, halves outgoing weights, and declares state/symmetry handling. |
| B3 | Static partial and full affine-bypass gated networks, centered and direct-leaf coordinates labeled. Static partial needs only unit counts, no ID-search charge. |
| B4 | Uniform random eligible IDs, same eigenvector contrasts and actual count; primary H1 rival. |
| B5 | Top IDs with random normalized contrasts; Adam/SGD score variants; cheap outgoing/gradient-norm diagnostics when investigating causes. |
| B6 | Restricted ANT-inspired validation-probed allocation: from the same parent, try each eligible single-unit insertion with identical F probe allowance, rank G improvement, select m jointly into a fresh parent, then normal continuation; include keep and count every loser. This is not unrestricted ANT reproduction. |
| B7 | Official NODE pinned source/config; preserve configured choice/bin functions and F-only data-aware initialization; faithful CPU is acceptable. |
| B8 | RealMLP-TD regression through pinned pytabkit, native preprocessing/embeddings/default anchors and documented tuning. |
| B9 | TabM default and piecewise-linear embedding recipe, `k=32`; mean member loss during fitting, mean prediction at inference. |
| B10 | Native CPU CatBoost; verify supplied 1.2.10 snapshot/wheel and log requested/resolved parameters. |
| B11 | Pinned compatible XGBoost CPU histogram; record OpenMP and threads. |
| B12 | Local TabICLv2 regressor, verify supplied `tabicl-regressor-v2-20260212.ckpt`, exact weight terms, official ensemble/cache defaults and parity. |
| B13 | Local open-weight TabPFN-3.5 base, explicit hash/license; no API-only substitute. Verify full-F backend support and current CPU row guard; do not bypass it. |
| B14 | LightGBM CPU serial constant-leaf and `linear_tree=true`, set in native Dataset and train config; inspect non-first-tree linear coefficients. F scaling, memory, OpenMP verification. |

Mandatory controls for WP04: B0–B5. B6/B7 correctness and cost are completed before WP05 design freeze; B6 supplies the pilot ranking diagnostic. B8–B11/B14 are mandatory practical families before broad local competitiveness claims; B12/B13 are conditional on assets, terms, full-F and resource support. A failed default or an unavailable family remains explicitly unavailable with claim restrictions. Do not replace native sparse functions or silently shrink context/ensemble and keep the same name.

Preserve vendor configs/source URLs/retrieval dates/revisions under `configs/vendor/`; source/asset provenance under `provenance/`. Each adapter has a faithful tiny fit, finite shape checks, vendor tests/examples where available, and logged differences. No published score is reproduced merely by loading its package.

Practical HPO uses seeded TPE, archived sampler state, and official defaults as initial trials/always reported anchors. Initial custom neural search: learning rate log-uniform `[1e-4,5e-3]`; weight decay includes zero and log-uniform `[1e-6,1e-1]`; depth 1–4; widths 32–512 on a multiplicative grid. Share Track A optimizer choices across forks.

TabM anchors: `k=32`, default AdamW learning rate 0.002, weight decay 0.0003; plain and numerical-embedding searches; widths 64–1024 in steps of 16 and learning rate `[1e-4,5e-3]` as supplied, verified against pinned vendor docs. CatBoost support: depth `{4,6,8,10}`, learning rate `[0.01,0.2]`, L2 `[1,30]`. XGBoost: depth `{2,4,6,8}`, learning rate `[0.01,0.2]`, child weight `[0.1,100]`, row/column fractions `[0.6,1]`, L2 `[1e-3,100]`. LightGBM: leaves `{4,8,16,32,64}`, rate `[0.01,0.2]`, minimum leaf rows `{5,10,20,50}`, linear lambda zero plus `[1e-6,10]`. Positive rates/penalties are log-uniform, fractions uniform, listed depths/leaves categorical. Squared error, no encoding fitted outside F. G selects tree count; raise exploratory ceilings if improving at the boundary.

Double exploratory search allowances to inspect marginal validation gains/noise; expand winning boundary configurations before freeze. WP05 records common numeric low/quality-oriented local wall-time budgets from complete default/plateau measurements. Include preprocessing, scoring, rejected candidates, HPO, validation, fit and compilation; static methods can spend saved search cost. For standalone random selection charge only sampled contrast computation when possible, separate from shared causal scoring. Data downloads are separate; method-specific weight acquisition is separate cold cost. Pretraining compute is disclosed unavailable, not charged as local fit; weights/context/cache storage and data dependence are disclosed.

## 9. Pilot, freeze, confirmation, and inference

Separate exploratory and confirmatory seed namespaces. Calibration may draw a separate exploratory evaluation set E for pilot paired effects/row-loss variance; E is never a protected T, never reused for confirmation, and never used for fitting or checkpointing. A draft schema may exist, but `protocol/frozen.yaml` is created only by a successful freeze after numeric design outputs exist. A protected-test command refuses a missing/invalid freeze or drift in scientific config, execution/analysis source, locks, splits, or source/assets. Resume-only operational fields are explicitly allowlisted and logged. Hash canonical JSON (sorted keys, finite scalar values, explicit resolved defaults), not YAML formatting. Record UTC and SHA-256.

Pin the source commit and a digest of all execution/analysis source, vendor revisions, config schemas and scientific inputs. Record the actual checkout commit at run time too. A later commit containing only the frozen artifact or documentation may differ from the pinned commit if the execution/input digests still match; this avoids making the freeze invalid merely by committing it. Changed or dirty executable code is rejected. This revision-accounting rule is a handoff implementation choice, not permission for scientific drift.

The freeze includes hypotheses/contrasts, exact population/primary coordinate, independent units, data/source hashes, exclusions, metrics, grouping/preprocessing, model configs/revisions, HPO algorithm/budgets, seeds/counts, batch/stopping/checkpoint rules, retries/failures, timing/interval method, and analysis revision. After T access, scientific changes require a dated amendment, exploratory relabeling of the affected old claim, and fresh independent confirmation. Never overwrite old protocol hashes.

### Numeric design rule

Start the exploratory variance pilot with 8 independent teachers and at least 2 optimizer/random-placement repetitions per teacher; add teachers in batches doubling the completed teacher count. Calibrate until the whole-teacher-bootstrap 95% interval of paired-effect standard deviation has upper/lower ratio at most 2 and projected per-teacher runtime is stable within 20% across the last two batches. Archive diagnostics. Do not stop on significance. If variance is effectively zero, diagnose equivalence; more optimizer repetitions are justified only by the within-/between-teacher cost tradeoff.

If pilot H1 paired 95% interval excludes zero, `delta_plan=0.5*abs(mean_H1)`, even for a negative mean. Otherwise, if static-partial GLU versus upper-parameter-bracket ordinary MLP has a pilot interval excluding zero, use half that absolute mean. Use the upper 95% whole-teacher-bootstrap bound on paired-effect SD as `s_D`. Plan `K=ceil((z_0.975+z_0.8)^2*s_D^2/delta_plan^2)` for sole H1, compare resampling-based power, and publish attainable precision/MDE and cost. Broader confirmatory families use the declared adjusted level.

If neither effect resolves, run a fresh independent **estimation** cohort with the same completed pilot teacher/repetition counts; report attainable interval width, not superiority power. Unresolved/zero between-teacher variance also triggers an explicit equivalent-model/estimation assessment and finite declared test-size precision target. No infinite division or arbitrary superiority promise. If calculated work is physically impractical, record local forecast and attainable resolution before confirmation; retain a declared underpowered estimation study or block the larger claim. `pilot/design_decision.md` is a blocking artifact with the actual calculation and branch. Once frozen, finish the entire independent cohort regardless of intermediate signs.

### Metrics and uncertainty

Primary risk `R=mean_T((prediction-y)^2)/var_F(y)` uses population F variance. For zero variance, normalized risk is undefined; show raw MSE and classify the task as degenerate. Pair effect `D=R_control-R_proposed`; positive favors proposal. Report raw MSE/RMSE/MAE in original units, noise-free synthetic approximation risk separately, row and group-macro error, prespecified worst group/regime slices, and tail absolute error.

H1 analysis averages repetitions within teacher, then estimates paired mean and whole-teacher-bootstrap 95% interval; also report Student-t sensitivity, median, full effect distribution and effective counts. Bootstrap seeds and enough iterations for negligible endpoint Monte Carlo error are chosen on development/pilot precision and frozen. A positive two-sided interval supports H1; an entirely negative interval contradicts benefit in that regime; a wide crossing interval is inconclusive. Extra superiority contrasts require declared Holm families or exploratory labels. No invented practical utility threshold.

Transfer is a fixed suite: per-task paired effects plus an equally weighted aggregate of the five observational task risks; Energy separate. Outer folds combine covered-row predictions and are not independent datasets. Conditional intervals resample whole test groups, retaining pairing; Bike uses moving date blocks chosen from development residual dependence, with half/double sensitivity. Report HPO/training repetition variability separately. These are not universal population-of-datasets intervals. V chooses the baseline champion; report the entire local table and evaluate champion selection on T without picking rivals using T.

Runs that fail remain in the cohort ledger. Freeze a symmetric finite retry policy from diagnosed exploratory failures before confirmation; never silently drop paired units or fill missing effects with zero. Unresolved invalid pairs block a completed-cohort superiority claim; report counts and an incomplete/resource-limited study. A resumed valid run links to the failed attempt.

## 10. Cost, timing, and resources

Record method-specific total search-and-fit seconds, post-selection fit, preprocessing, score/probes, first prediction, warm latency at batch 1 and representative batches, throughput, model bytes, context/cache bytes, trainable/total parameters and operation estimates. CPU thread counts and failed candidates count. Count shared parent work once for the harness and disclose attributable standalone costs for each method.

Use `perf_counter_ns`; synchronize MPS immediately before and after timed MPS regions. Separate fresh-process import/load/first prediction, host→device→host warm batch-1, device-resident kernel-only, and complete batched pipeline. Include context build/cached versus uncached modes and transductive batch semantics. Calibration selects timing repetitions for useful median/tail precision; warm up until the predeclared rolling precision is stable and log discarded work. Interleave ABBA or randomized balanced method blocks; document demonstrated thermal/system exclusions and retain excluded observations.

Measure CPU/MPS one-step output/loss/gradient parity from CPU float64 reference before performance use. Benchmark epochs and scoring at doubled batches. Select the smallest tested safe batch with at least 95% of maximum median throughput; freeze **logical** batch before comparison. Do not oversubscribe concurrent trials. Wall-time forecasts come from this laptop; none is supplied in the research.

First memory model: float32 AdamW weights+grad+two moments approximately `16P` bytes, plus activations, library/allocator/workspace overhead, data copies, preprocessing, checkpoints and process. Moment storage if all retained is `8h(d+1)^2` bytes; block accumulation bounds it. Foundation weights/context/ensembles/caches need a separate measured model. RSS, MPS allocated/driver memory overlap in unified memory; record separately, never sum as disjoint usage. Record memory pressure and swap deltas. No watermark disabling.

## 11. Interfaces and artifact contracts

Use a small Python package `treeneuron` with `python -m treeneuron.cli`. Proposed commands below do not exist until implemented. Public APIs are project adapters, not vendor-native signatures.

```python
@dataclass(frozen=True)
class FitRequest:
    X_fit: Any
    y_fit: np.ndarray                 # finite [N], original units
    X_select: Any                    # G only
    y_select: np.ndarray
    seed: int
    config: dict[str, Any]
    budget: dict[str, Any]
    artifact_dir: Path

class Predictor(Protocol):
    def fit(self, request: FitRequest) -> "Predictor": ...
    def predict(self, X: Any, *, batch_size: int | None = None) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...
    def resource_summary(self) -> dict[str, Any]: ...
```

Predictions are finite `[N]` in original units; ensemble axes/standardized outputs must not escape. HPO alone reads V labels; protected evaluation alone reads T labels. Capabilities disclose internal validation, transforms, weights/categories/context, transduction/caching, backend/fallback, and unavailable counters. Unsupported semantics raise recorded errors, not coercion.

Configuration is human YAML with typed strict validation and canonical JSON hashes. Unknown keys and nonfinite numbers are errors. Each run owns an immutable ID/directory and writes `config.resolved.json`, `manifest.json`, append-only `events.jsonl`, `metrics.json`, checkpoints, and permitted predictions. Manifest groups:

- Identity: run/parent/study/phase/status/failure class, code/adapter revision, protocol/lock/hardware/growth hashes.
- Data: dataset/raw/split/transform hashes, independent seeds, teacher normalization, F/G/V/T row/group counts.
- Method: resolved config, trainable/total counts, serialized bytes, backend/dtype/threads/fallback, weight checkpoint/hash/terms, context/cache bytes.
- Costs: search/preprocess/base/score/probe/continuation/validation/first/warm prediction nanoseconds, boundaries and measured/estimated status; acquisition separate.
- Resources: RSS, MPS allocated/driver, pressure, swap deltas, UTC timestamps, interruption/retry links, timing exclusions.

Unavailable counters are `null` with reasons. Prediction records include run/fold/immutable row/group/date IDs, actual/predicted target, split, and synthetic noise-free target. A feature allowlist prevents metadata, target-derived columns or IDs from entering inputs. T predictions/labels are written only after logged authorized access.

## 12. Work packages and gates

Implement the infrastructure required by the next gate, not a general NAS platform. The core plan covers WP00–WP04; the evidence plan covers the remaining gated work. WP03 deliberately stages causal adapters before practical ones so no large optional dependency stack blocks the mechanism diagnosis.

| WP | Deliverable and command | Acceptance / branch |
|---|---|---|
| 00 | `doctor`, pinned core, environment/provenance, `protocol/prior_art_delta.md` | CPU tests run; MPS kernels pass or CPU selected; licenses known; R12 overlap held provisional if inaccessible. Exact prior implementation becomes replication/refinement. |
| 01 | `verify-math`, compact layer, score; local A/B reproduction | Float64 identity/gradient errors `<1e-10`; insertion within justified backend tolerance; constructed nonstationary gate gradient nonzero. Failure blocks prediction comparisons. |
| 02 | Synthetic generators, F-only transforms, B0, split/metric/freeze access boundaries; later raw-suite adapters | Sham gives zero; oracle/ridge sanity, hidden gates, disjoint groups, target-column rejection, no T dependence. Bad data/insufficient groups block or narrow task. |
| 03 | `verify-baselines --track causal`, then practical adapter matrix | B0–B5 before WP04; B6/B7 before WP05; practical defaults/tested faithful paths before H3. Unavailable models limit claims. |
| 04 | `run --config configs/explore.yaml`: complete paired growth experiment, diagnostics and costs | Positive-control mechanism learns, forks correctly paired, insertion/state preserved. Acceptance never requires proposed victory. |
| 05 | `calibrate`; numeric design/forecast; `freeze` | Another worker can execute without choosing counts/hypotheses/budgets after T. Negative/estimation designs allowed; numeric missing outputs block freeze. |
| 06 | `confirm --study synthetic` | Full independent cohort and audited teacher statistics, irrespective of sign. Capacity/static equality narrows attribution. |
| 07 | `confirm --study transfer`, faithful practical defaults/HPO, complete timing | Frozen suite, credible local competitors, champion selected on V, full cost table; no implicit win over unavailable baselines. Normally follows WP06; a recorded informative question can justify transfer after negative H1. |
| 08 | `run-extension --protocol ...`, only for recorded unresolved evidence | New equations/baselines/protocol/forecast/fresh confirmation. No automatic depth, kernel or dataset escalation. |
| 09 | `audit`, `report`, clean representative rerun and independent metric recomputation | Regenerated reports account for every run, source/claim revisit, resume validated; negative outcomes are valid deliverables. |

At every completed WP append `DECISIONS.md`: inputs/hashes, evidence, decision, blockers, next command. The repository documents, not chat memory, are the execution state.

## 13. Validation and decision rules

Required mathematical tests cover shared-tensor forward/backward equivalence, affine collapse, hardening discrepancy, ReLU convention, parameter counts, empty/invalid growth, function preservation, zero–zero dead insertion, score analytic/finite-difference gradients away from kinks, batch accumulation scaling, permutation/rescaling/sign rules, and first Adam gate-only descent. Required systems tests cover actual indexing/backprop, native adapters, state migration, interrupted resume, atomic writes, drift rejection, split leakage, F-only transforms, test access and metric reconstruction. Float32 tolerances are established from CPU-reference error records before predictive use, recorded in the hardware report, and frozen; no post-result loosening.

Log gate mean/variance/quantiles/saturation, `E[g(1-g)]`, leaf disagreement, gradient/input norms, per-unit spectra/scores and synthetic regime occupancy. Probe score–improvement correlation on independent probe data; outlier/clipped/leave-group-out diagnostics and outgoing-weight explanations are separately named exploratory variants. Soft occupancy is not executed-branch count, and plotting a tree is not interpretability evidence.

Failure classes: `implementation_invalid`, `measurement_invalid`, `proxy_unsuitable`, `resource_limited`, `inconclusive`, `mechanism_without_advantage`, `evidence_against_hypothesis`, `qualified_improvement`. Every class needs recorded evidence. Invalid implementation/measurement blocks scientific conclusions. A wide interval is not equivalence. A small statistically resolved effect is described with absolute costs and regressions, without an invented utility threshold. A Pareto claim requires all named dimensions; better accuracy with slower inference is a tradeoff.

Evidence-triggered options: expensive scoring → parity-tested streaming/approximation; oracle helps but learned gate fails → optimization diagnosis; compositional teacher exposes stump limitation → depth 2; broadly useful healthy branches → registered repeated growth; redundant branches → F-fitted affine-surrogate pruning selected on G; small disagreement-weighted hardening gap plus measured headroom → hard CPU prototype before kernels; advantage erased by RealMLP/TabM → report boundary, then separately confirm insertion into the strong recipe if justified.

Final report states a continue/revise/narrow/replicate/stop decision against the registered question. Component wording identifies the teacher population, effect/interval, matching and costs within an established gated family. Practical wording additionally names the observational suite, full-F locally run baseline set, V-selected champion, group/time protocol and measured tradeoffs. Negative wording identifies controls and achieved resolution. Retain novelty and scaling limits. Rebuild reports without manual score editing or selective deletion.

## 14. Handoff choices and deferred decisions

The brief supplied the scientific question, equations, core architecture, suite, information rules, comparison families and statistical design. This handoff chooses package/file boundaries, compact growth input device rules, deterministic tie handling, the explicit uniform synthetic gate-type draw, smooth-threshold orientation, positive-control sampling/noise, Energy geometry key, all-missing fill behavior, date-quantile rounding and earliest-checkpoint ties. These choices are visible for review and become protocol inputs; none is a claimed user preference or empirical finding.

Numeric runtime budgets, stopping allowances, repetition/test/teacher counts, float32 tolerances, exact dependency locks, baseline assets and license eligibility are **measured gate outputs**, not open-ended implementation discretion. WP05 freeze cannot pass without them. No `frozen.yaml`, confirmatory result or completed-WP record is created by this documentation task.
