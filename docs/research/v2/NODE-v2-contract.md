# Proposed P12 NODE-v2 adapter contract

18 October 2026. Contract only; implementation waits for root review. Branch
`research-v2-p12-node`, base `7f781cc7787707000f368be4917d851df359a052`, in managed
worktree0dbd. The completed `research-v2-p13-simple` branch/evidence is preserved.
Binding scope: [Task7](../../superpowers/plans/2026-10-18-research-v2.md#task-7-wp07-transfer-data-native-adapters-and-gate-d)
and [native procedures](../../superpowers/specs/2026-10-18-research-v2-design.md#mandatory-native-procedures-search-and-resources).

## Owned boundary and source

Create only `research/node_v2.py`, P12-specific tests/docs/evidence and
`locks/research-v2/node-v2/` versioned identity records. Read existing FitRequest,
plain schema/frame/token helpers, vendor kernels and archived official sources.
Do not change legacy `baselines/node.py`, vendor files, old NODE locks/guards,
core27/uv.lock, shared registries, main or active604cffb scientific source. No
installs, prepared transfer fits, HPO, full2048-tree fit or GateD execution.

Reuse the existing archived NODE commit
`3bae6a8a63f0205683270b6d566d9cfa659403e4`, its shallow YEAR notebook and numerical
kernels; QHOptim1.1.0 commit `0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea` and wheel
SHA256 `cdd9660b8310ef7749733db03fe372bf5453861ba9cb0aa31db59f6127b3ae00`.
The contract-stage byte check verified all14 archived source records and all6
runtime/original pairs. Kernels remain unchanged; existing QHAdam relative-import
and unused utility extraction changes remain explicit in compatibility.patch.
Original NODE/QHOptim/entmax notices remain in place. Historical sklearn/QHAdam
dependency resolutions and published YEAR-score reproduction remain unknown.

Use existing Python3.12.13/native arm64 CPU, Torch2.14.1, NumPy2.5.4,
SciPy1.18.1, sklearn1.9.1, Pandas3.0.6. The new lock references actual archive,
vendor, source/core-lock bytes; distribution RECORD pins, executed numerical
Python/native-library bytes and actual import origins. It must identify its
verified file coverage. Capture source/runtime before numerical work and compare
again before success publication. Importing the adapter/config/identity helpers
must not import Torch/NODE/train; lazily import needed numerical backends at
fit/load, and never use `train.source_hash` merely to hash source bytes.

## Public interface and exact support

Use strict `NodeV2Predictor().fit(FitRequest)`, feature-only
`predict(X,batch_size=...)`, `save(directory)`,
`load(directory,expected_manifest_sha256=trusted_digest)` and `resource_summary()`.
Config fields are schema `research/v2-node-config`, family `P12`, mode,
explicit ordered `{columns,categorical}` feature_schema and parameters. Reject
unknown/reserved keys, coercive types, unsupported coordinates and implicit
schedule/device overrides. Request budget supplies a positive max_updates
resource ceiling; a ceiling cannot certify native stopping or quality.

| Coordinate | native_anchor | search |
|---|---|---|
| trees |2048|512/1024/2048|
| depth |6|4/6|
| LR multiplier |1|.5/1/2|
| layers/tree_dim |1/3|fixed1/3|
| batch/report/patience/latest |512/100/5000/5|same fixed native values|
| init prefix/validation chunk |min(1000,F)/1024|same|

Anchor parameters are empty; search parameters declare trees/depth/LR multiplier.
Effective QHAdam: LR .001 times multiplier, betas(.95,.998), nus(.7,1),
eps1e-8, WD0, decouple_weight_decayFalse. Preserve explicit entmax15 selectors
on feature axis0 and entmoid15 bins, original response/selection initialization,
threshold beta/cutoff1/1, no input dropout, and channel0 mean over all trees.
No Adam/sigmoid/head/row-dropping substitution.

Separate `engineering_fixture` permits trees16/32, depth2/4, batch8/16/32,
report1/2/100 and patience1/2/5000, with max_updates<=128 and no architecture or
optimizer-function changes. Every such fixture is completed_ineligible even
when it exercises a native report. This permits short rolling-five and patience
fixtures plus one small actual100-update report, without running a full anchor.
Pure scheduling tests cover the native5000 strict boundary without5001 fits.

## F-only geometry and declared v2 adaptations

FitRequest supplies F/G only. Explicit dataset schemas own target/proxy
exclusions; preserve declared feature names rather than blacklist arbitrary names.
No V/T data are accepted by fit. F fits every transform, category vocabulary,
target statistic and ordered initialization prefix; only G selects reports.

Builder proposal: continuous numeric columns retain raw scale, with F-median
imputation (all-missing fill0) and an unscaled indicator only if missing in F.
Reject bool/complex/string/infinite numeric values and conversion overflow.
Categories use sorted F-only typed tokens distinguishing str/int/bool/finite-float,
all known one-hot columns and one reserved unknown column last; missing/unseen
use unknown. Feature order is raw numeric, F-created missing indicators, then
categorical blocks in schema order. This adds a v2 category/missing path without
weakening legacy finite-numeric guards.

Builder proposal: convert this encoded matrix explicitly to float32, then apply
the native noisy normal-quantile transform to **all encoded columns**, including
one-hot/unknown and missing indicators. Do not standardize numerics first or
quietly leave indicators outside the native transformer. Fit a noisy F copy with
noise amplitude `.001 / maximum(F_std,.001)`, global seeded NumPy draws and the
source's in-place float32 operation order; transform original encoded F/G/query
without query noise. Pin QuantileTransformer n_quantiles1000,
output_distribution normal, ignore_implicit_zerosFalse, subsample10000,
random_state=request.seed, copyTrue; record realized min(1000,F) quantiles.
These are explicit modern defaults, not recovered historical dependency versions.

Builder proposal: F target mean/population std are float64; constant F uses unit
scale and records the adaptation, instead of dividing by zero. Normalize F/G
labels to native float32 before loss/G MSE. Prediction first obtains native
float32 scalar output, then explicitly promotes to float64 for F-scale/mean
inversion. This is original-unit output, not a published notebook prediction API.
Numeric-only fixtures compare against direct official arithmetic with these
same declared dtype boundaries; categories are separately adapted fixtures.

Seed NumPy/Python/Torch, fit quantiles, construct the native model, then trigger
initialization once on ordered transformed F[:min(1000,F)]. Retain initialization
row count/order and prefix commitment. Warn/record short F. Never initialize on
G/query/load; validate initialized uint8 flags and their Python cache before
prediction. Restore caller RNG on all success/error/interrupt/load paths, and
preserve prior fitted state on failed refits. Do not change caller thread policy;
fixtures run sequentially in one-thread processes with package source held.

## Training, selection and complete retained state

Shuffle one NumPy index array each epoch, include incomplete final batches and
all F rows, and use mean MSE on the aggregated scalar output. No F prefix fit,
G feature fitting, artificial update0 report or extra G report at a budget cap.
Every100 updates, retain raw state, average newest-first sequentially over up to
five raw report snapshots, evaluate native normalized float32 G MSE in chunks,
and retain the earliest strict improvement. Restore raw parameters and unchanged
QHAdam state before the next update. Test `step > best_step + 5000` every update.

Replacing filesystem-ctime temporary checkpoints with ordered report-step records
is an explicit storage adaptation. Floating model states use native newest-first
sum/count. Initialization uint8 flags must all agree and retain dtype rather than
be averaged and silently cast. Retained averaged checkpoint optimizer metadata
belongs to its newest raw member, as upstream; it does not mean optimization
continued from averaged weights. Preserve the raw endpoint/optimizer separately.

A cap before any report returns only an explicitly unselected engineering
endpoint. A cap after reports can retain a valid selected prediction state, but
cannot claim native patience completion. Record selected/executed updates,
window member steps, finite G history, stop reason, full-F row coverage, seeds,
source/runtime and F/G commitments. No fixture or adapter return establishes
native_full_anchor_qualified or quality_eligible; GateD remains external.

Portable state includes schema/config/resolved recipe, typed F geometry,
quantile arrays/references/parameters, target stats, initialized selected model,
raw endpoint/QHAdam slots, rolling raw windows and selected-window evidence,
report history, sampler cursor/permutation and terminal RNG commitments/state.
Retain enough selected-window tensors to recompute the saved average. Store all
native arrays/tensors with explicit exact dtype/shape/name/finite commitments;
validate optimizer-slot geometry and integer flags without implicit load casts.
This is a complete auditable prediction/endpoint artifact, not an implementation
of mid-fit interruption/resume or cross-source checkpoint migration.

## Serialization, errors and bounded qualification

Use fresh-directory JSON+nonpickled NPZ, externally trusted raw manifest SHA256
checked before NPZ, exact file/array inventories and source/runtime/lock binding.
Reject duplicates/unknown fields, malformed scalar/role/update evidence, object
arrays, unsafe file paths/symlinks, tensor/basis/dtype mismatches and nonfinite
state. Recreate the fixed native model/QuantileTransformer without refitting,
restore validated initialized state before first forward, and never evaluate
serialized expressions or arbitrary object pickles. Same-runtime/same-chunk
save/load predictions must be exact; different chunk layouts and independent
float32 native parity use declared rtol1e-6/atol1e-7 on normalized outputs.

Direct reference fixtures independently import unchanged archived DenseBlock,
entmax/entmoid and official wheel QHAdam, not the legacy/new adapter. Verify all
source/import bytes first. For data/trainer files whose broad imports require
uninstalled category_encoders/tensorboardX, execute exact hash-verified source
bodies for preprocessing/averaging under inert data/logging/I/O facades, declared
up front. This is numerical-path parity, not untouched whole-package/notebook
execution. Existing pinned requests/tqdm reference overlay may be read only;
missing dependencies fail visibly, with no install or alternative math fallback.

After approval, TDD config/role/transform/initialization/full-row/gradient/rolling5/
strict-patience/selected-versus-raw/eligibility boundaries; tiny direct native
prediction, optimizer and save/load parity; structured corruption; original error
and receipt-publication failure preservation; and source/runtime drift. Retain
all attempted configs, failures, interruptions, logs and source revisions; no
hidden retry. Fit receipts include paid phase/total timing; outer process startup,
validation/publication/inference and failed attempt costs remain separately paid.

Finite transformed-workload memory/admission, hard deadlines/RSS/concurrency,
verified interruption/retry harness, native2048 feasibility, practical HPO,
registry integration and GateD remain pending. No new scientific/practical
quality or efficiency claim follows from these small engineering fixtures.
