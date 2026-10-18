# P07 TabM adapter contract for review

This file retains the contract reviewed before implementation. Its initial
status was primary-source/static wheel qualification only; root subsequently
approved implementation below. Current implementation and executed checks are
recorded in [tabm-qualification.md](tabm-qualification.md). Prepared transfer
data, V/T labels, full native quality anchors and HPO remain outside this slice.

The isolated branch is research-v2-tabm, based on
a56ae0897d62adb04eabbf11bc09dae24ca18e2f, in
/Users/artemlegotin/.codex-s/worktrees/37a4/adaptive-tree-neurons.
The finalized ordinary-neural branch/evidence and shared main remain untouched.

## Pinned implementation and environment

[Official TabM commit](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4)
declares version0.0.3. The source module, README, notebook, pyproject and Apache2.0
license are retained byte-exact with upstream Git blob SHA1 and SHA256. The pinned
commit fixes gradient clipping in the notebook; that corrected call is binding.
The notebook is read as JSON/source, never executed as a whole.

Use rtdl_num_embeddings0.0.12, matching the pinned author's own uv constraint.
Its [official v0.0.12 tag](https://github.com/yandex-research/rtdl-num-embeddings/tree/7381c5a4ed01253fbef9ea8cfed137f907265552)
resolves to commit7381c5a4ed01253fbef9ea8cfed137f907265552. The package's
Apache2.0 license is binding; the repository root has a different license and is
not substituted for the package license.

Two PyPI universal wheels were acquired and inspected without installation:
- tabm-0.0.3-py3-none-any.whl,17347bytes,
  SHA2566cabed3cb436fef5e8a45f46097fd745f51b3e16be16d555b9af16cffda7f944.
- rtdl_num_embeddings-0.0.12-py3-none-any.whl,13293bytes,
  SHA25687fd61270118915cf40888f2164f3a1f0353edf702f15fd91e71a147d23c930d.

Their module bytes exactly match the pinned official Git blobs. They expand to
115966bytes total before installer metadata. The complete extra runtime graph is
these two packages; Torch>=1.12,<3 and typing_extensions>=4.6,<5 are already
satisfied by the frozen core's2.14.1 and4.16.0. Numerical embeddings additionally
attempt optional sklearn.tree and tqdm imports; sklearn1.9.1 is in the core and
tqdm is not required for unsupervised bins/verboseFalse. No dev extras are needed.
The author's development constraints use older Torch/NumPy; declared runtime
compatibility is not yet import/numerical qualification on this macOS core.

Exact sources/licenses/wheel hashes and METADATA hashes:
[static source lock](../../../locks/research-v2/tabm/source-lock.json).
Eight retained assets total170512bytes. Wheels/metadata inspection receipts are
under ignored artifacts/research-v2/WP07/tabm-source. Once this contract is
accepted, install only the two hash-locked wheels with --no-deps into a fresh
environment/research-v2-tabm-site overlay. Preserve all27 core distributions,
check every overlay RECORD byte and actual module/native-library origins.
Neither core.json nor core uv.lock may change.

## Exact native anchors and training

The actual TabM.make factory defaults depend on whether embeddings are supplied:
- Plain: n_blocks3,d_block512,dropout.1,ReLU,k32,arch_type tabm,
  start_scaling_init random-signs.
- Piecewise-linear: n_blocks2,d_block512,dropout.1,ReLU,k32,arch_type tabm,
  start_scaling_init normal. Construct PiecewiseLinearEmbeddings from
  compute_bins(F_numeric_transformed,n_bins=48),d_embedding16,activationFalse,
  versionB. Keep native chunk initialization and parameterization.

Call the pinned official factory; do not port a weaker substitute architecture.
Both anchors use AdamW LR.002/WD.0003 and native defaults, shared batches,
batch256, clipping global gradient norm1 after backward/before step, no scheduler,
CPU float32, AMPFalse and compileFalse. Each epoch uses torch.randperm(F_count)
and split(256), with complete coverage and no dropped rows.

The pinned notebook computes mean member MSE: flatten predictions across row and
member, repeat_interleave each row target32 times, then MSE. Optimizing MSE after
averaging members is a different objective. Validation/query predictions rescale
each float32 member to original target coordinates first, then average members
in float32, matching the notebook's operation order. Native stopping maximizes
negative original-unit G RMSE, using strict improvement and first-best ties.
The practical coordinator may independently compute its declared V/T metrics.

Native patience16 means remaining_patience starts16 and stops when it becomes
negative:17 consecutive misses, not16. The native epoch ceiling is1000000000,
with the separately paid external resource deadline; no artificial1000-epoch cap
is introduced. Initial checkpoint epoch-1 has score minus infinity; epoch0 is
the first evaluated selectable checkpoint. Report native zero-based epoch and
executed epoch count separately. Tiny fixtures have an explicit at-most3-epoch
engineering mode and are always quality-ineligible. A deadline does not certify
native stopping/convergence. TabM has no Skorch F-mean safeguard; do not transplant
that policy. Nonfinite training, gradients, parameters or G scores fail visibly.

The notebook's before-training and per-epoch test evaluations are excluded.
FitRequest supplies only F/G; V is prediction features only and T is evaluator
only. Direct parity is official factory/preprocessing/loss/optimizer invocation
with the explicit protocol boundaries, not untouched notebook reproduction.

## F-only preprocessing and declared boundary adaptations

The native notebook uses C-contiguous float32 numeric inputs and absolute
Gaussian noise from default_rng(0).normal(0,1e-5,shape), cast to float32 before
addition. This differs from P03's inverse-std.001 noise and float64 promotion.
Fit QuantileTransformer on noisy F only: normal output, subsample1000000000,
n_quantiles=max(min(nF//30,1000),10). Preserve the native random_stateNone;
with the finite F ceiling below subsample, no subsampling draws are used.
Transform original unnoised F/G/query values. The model/sampler seed and fixed
noise RNG are separate recorded namespaces.

Cast original F targets to float32; native F mean/std(ddof0) are float32
reductions converted with item(). Standardize training targets in float32.
Keep original float32 G labels for native original-unit RMSE. Return the final
float32 original-unit mean prediction as a finite vector (coordinator conversion
to float64 does not restore discarded precision).

Proposed explicit policies requiring this contract's review:
- Drop F-constant numeric columns before quantile fitting, for both variants;
  retain raw-column mask, values and reason. This handles the native compute_bins
  rejection of constant columns and the suite's possible F-constant year input.
  Query variation never changes that F-fitted mask. This is a declared
  preprocessing adaptation, with direct parity using the same F-only mask.
- Retain low-distinct-count nonconstant columns. Native quantile bin edges use
  torch.quantile followed by unique; record each actual edge/bin count without
  forcing48 distinct bins or adding jitter after preprocessing.
- If n_bins>=nF, retain an explicit construction/input failure; never silently
  lower bins or subsample F. Anchor48 therefore requires nF>48. A piecewise
  variant with no surviving numeric columns is ineligible, not silently plain.
  Plain supports categorical-only inputs; no surviving inputs is ineligible.
- Fit typed category vocabularies on F only. Add one reserved unknown index per
  category and pass those K+1 cardinalities to TabM's native one-hot encoder.
  Unseen queries map to K. Preserve the factory's corresponding random weights;
  do not invent zero embeddings or zero the unknown column. Retain expanded
  dimensions/parameter cost; this is the protocol's explicit category boundary.
- For an exactly constant F target, use scale1 and retain the event; the
  unguarded notebook would divide by zero. All nonconstant cases preserve native
  float32 reductions. Overflow/nonfinite input conversions remain explicit
  failures rather than fallback predictions.

## Search, ownership and acceptance

Two native anchors (plain then piecewise) count as the first two attempts of one
P07 family, within its existing32-attempt/120-minute ceiling. Later variant choice
is explicit and paid. Frozen support: blocks2/3/4,width64..1024 step16,
LR log[1e-4,5e-3], WD zero or log[1e-4,.1], dropout zero or uniform[0,.5],
piecewise bins2..128 and embedding8..32 step4; k32 remains fixed. The pinned
README allows plain blocks1..5 and embedded1..4; our2/3/4 restriction is the
handoff's bounded search choice. No older-paper embedding ranges or extra
architecture/factorization/ensemble choices enter this study.

Proposed owned files only:
- New src/treeneuron/research/tabm_native.py and its contract/native fixture tests.
- locks/research-v2/tabm source, two-wheel overlay requirements/audit/receipts.
- docs/research/v2/tabm-adapter-contract.md and final qualification receipt/docs.

Reuse existing read-only FitRequest and plain schema/frame helpers. Do not edit
ordinary_neural.py, native_trees.py, shared registries, legacy guards, core locks,
execution.md or finalized a56ae089 evidence.

After approval: TDD role/transform/loss/stopping/config boundaries; import/RECORD/
native-runtime checks; tiny one-thread direct factory/plain+piecewise numeric and
categorical parity; saved prediction-state/config/bins/source/dependency/runtime
binding with external trusted digest; corruption and failure-domain tests.
Preserve every attempt and source revision, with one native worker at a time.
Full native anchors, practical HPO/registry integration, hard deadlines/RSS,
interrupted-training recovery and Gate D remain pending. No new predictive
quality or placement claim follows from this qualification.

## Root contract disposition before implementation

Root approved contract SHA25676094b82ada02083758c317e3b3a82ea5f9b080dfc3f7dd90ae507cd958d11da
for scoped implementation. Constant-column detection compares actual float32 F
numeric values and records dropped values, mask and precision. Scale1 applies
only when actual float32 F targets are all equal. Nonconstant float32 targets
whose std underflows to zero, or nonfinite/overflow statistics, fail explicitly.
The target/numeric/category boundary adaptations are declared in direct parity.
The canonical FitRequest seed directly seeds Torch model/sampler RNG; it is not
shifted by the notebook demonstration's fixed +2 offset. Fixed preprocessing
noise seed0 remains independent. No whole-notebook reproduction is claimed.
All other stated native policies and scope remain binding. Implementation uses
TDD, one tiny native process at a time, and independent review before commit.

Independent review clarified one additional worker requirement: disable Python
bytecode writes before native imports (PYTHONDONTWRITEBYTECODE=1 or python -B).
Otherwise imports can create unowned overlay __pycache__ files after the first
audit, causing a later load to fail. The adapter refuses this condition before
any native import; it never changes the interpreter policy or accepts untracked
overlay files. All executed native workers already used this setting.
