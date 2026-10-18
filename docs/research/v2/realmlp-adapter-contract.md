# P06 RealMLP-TD adapter contract for review

Historical preimplementation status at the original reviewed contract (SHA256
fb7e9b05cb0d9bf8b5d158b3c4b238518462aa4dabc100baec899aed8e8adfb4):
static official-source, dependency and wheel availability preparation.
No RealMLP adapter implementation, package installation, native import, fit,
prepared transfer data, V/T labels, HPO or full quality anchor has run. Root must
review this contract and dependency report before installation or implementation.

The isolated branch is research-v2-realmlp, based on
b394127b24bc0e4946797c8635b97aba79c060ee, in managed worktree
/Users/artemlegotin/.codex-s/worktrees/3e5f/adaptive-tree-neurons. Finalized TabM
f5378d5 and its evidence/overlay, shared source and qualified core are unchanged.
The existing multi-stage spec/plan remain binding; this is the P06 WP07 slice.

## Official identity, exact native scope and reading coverage

Use [official PyTabKit commit c126ea51187c5080b91f28d352481dbd3b2194b0](https://github.com/dholzmueller/pytabkit/tree/c126ea51187c5080b91f28d352481dbd3b2194b0),
which declares version 1.7.3 and package Apache-2.0. Its PyPI universal wheel is
364,005 bytes, SHA256
1589f281e99d4a6965a83f954d9f78b038fc633a4246d450a1a3599952d4d841.
All 89 Python files shared by the wheel and pinned Git tree match their Git blob
identities exactly. Every wheel RECORD entry was checked without installation.
The package license file SHA256 is
b40930bbcf80744c86c46a12bc9da056641d722716c378f5659b9e555ef833e1.
Retain upstream per-file notices, including the AUC-Mu module's MIT notice.
This is source/license provenance, not legal analysis.

The source lock retains 38 official assets, 684,782 bytes, with Git blob SHA1,
SHA256, path and commit URL. Reading focused on the regression factory, parameter
sampler, sklearn/conversion boundary, NN creator/fitter, preprocessing, categorical
handling, output normalization/clipping, optimizer/factors, schedules, loaders,
validation/checkpoints and imports. Whole-wheel byte equality is broader than
semantic reading; unused benchmark/classification/alternative-model code has not
been fully audited. Do not describe all 89 files as semantically reviewed.

The relevant primary paper is
[Better by Default, v3](https://arxiv.org/html/2407.04491v3), especially Table A.1,
Appendix A.1 and Table C.14. An initial v4 HTML request returned HTTP404; v3 was
available. The pinned implementation resolves native keyword/probability details;
paper/source differences below remain explicit. This is not author end-to-end
benchmark reproduction.

## Native regression anchor

Obtain the resolved configuration through RealMLP_TD_Regressor.get_config and
its official NNAlgInterface factory, rather than transplanting base LR into a
different parameterization. The pinned DefaultParams.RealMLP_TD_REG anchor is:

```python
dict(hidden_sizes=[256, 256, 256], max_one_hot_cat_size=9, embedding_size=8,
     weight_param='ntk', weight_init_mode='std', bias_init_mode='he+5',
     bias_lr_factor=.1, act='mish', use_parametric_act=True, act_lr_factor=.1,
     wd=.02, wd_sched='flat_cos', bias_wd_factor=0.,
     block_str='w-b-a-d', p_drop=.15, p_drop_sched='flat_cos',
     add_front_scale=True, scale_lr_factor=6.,
     tfms=['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
     num_emb_type='pbld', plr_sigma=.1, plr_hidden_1=16, plr_hidden_2=4,
     plr_lr_factor=.1, clamp_output=True, normalize_output=True,
     lr=.2, n_epochs=256, lr_sched='coslog4', opt='adam', sq_mom=.95)
```

Implicit native settings must also be resolved and retained: batch256,
predict_batch1024, Adam beta1=.9/epsilon1e-8, use_best_epoch=True,
use_last_best_epoch=True, use_early_stopping=False, adjust_bs=False, one model,
n_ens=1, no CV ensemble/refit and CPU float32. Parameterize public factory with
n_cv=1,n_refit=0,n_repeats=1, explicit CPU/threads/seed; the wrapper bypasses only
its F+G converter boundary below. Record the actual full resolved dictionary,
native model tree, tensor shapes, factors, schedules and library defaults.

The official opt='adam' path wraps torch.optim.Adam and separately applies
decoupled weight decay plus native HyperparamManager/Variable hyper factors.
Preserve that code exactly, including the order/factor application; a bare
AdamW replacement is not an established equivalent. Bias, activation, numerical
embedding and scaling parameters retain their distinct factors. Preserve native
data-driven initialization and initialization-only RAM-limited subsampling policy,
recording actual indices if activated. Training data must not be subsampled to
hide a memory limit. Native NTP naming is weight_param='ntk'.

Run the complete native 256-epoch schedule with no patience shortcut. Validation
at each epoch uses pinned native original-coordinate G RMSE after output normalization
and evaluation-time F-min/max clipping. Native last-best uses <=, retaining the
latest tied best epoch, unlike prior adapters' strict first-best. Native epoch
recording is one-based after each training epoch; zero is not silently added as
an eligible checkpoint. Restore the native selected model via official callbacks.
No F-mean rescue or extra clipping/target transform is added by the adapter.

The native loader uses drop_last=True and actual batch=min(256,nF), with
adjust_bs=False. For non-divisible F, a shuffled tail is omitted in that epoch;
it is not a fixed hidden training subset. Preserve native batch_randperm and
record admitted/omitted row IDs every epoch plus cumulative row exposure. Root
explicitly accepted this interpretation: all F rows are admitted to the native
loader, and actual exposure/any unvisited rows are reported; do not force
epochwise full coverage, replace seeds or extend training to conceal omissions.

Engineering mode may use at most three explicitly declared epochs and remains
quality-ineligible. Its changed schedule horizon is recorded, not presented as
the native quality recipe. A deadline/cap before full native treatment remains a
paid incomplete/ineligible outcome; it cannot become a converged anchor.

## Exact F/G-safe wrapper and preprocessing

The public sklearn fit(X,y,X_val=G,y_val=G) concatenates F+G before fitting its
ToDictDatasetConverter (sklearn_base.py lines201–243), so using that wrapper
unchanged would learn G category dictionaries. Instead:

1. Validate the explicit semantic feature schema, finite numeric/target
   conversion and typed category policy. Canonicalize category scalar types
   without reading labels; prevent string/int/bool conflation and preserve order.
2. Fit the official ToDictDatasetConverter on F features only. Transform G and
   feature-only queries with that fitted converter; retain vocabulary/schema.
   Its native OrdinalEncoder maps known values to1..K and unseen/missing to0.
   Reuse F tensor_infos/cardinalities in every DictDataset.
3. Assemble a native F+G DictDataset with raw float32 targets. Build explicit
   SplitIdxs(train_idxs=arange(nF)[None],val_idxs=arange(nF,nF+nG)[None],
   test_idxs=None,split_seed=seed32,sub_split_seeds=[native_subseed],split_id=0).
   Only F and G labels exist in that worker. No randomized internal split/refit.
4. Use the official factory's NNAlgInterface.fit on that dataset/indices with
   InterfaceResources(n_threads=1,gpu_devices=[]), native Logger and fresh
   artifact-local temporary paths. Preserve all NNFactory/NNCreator processing.
   Query NNAlgInterface.predict with only converter-transformed query features.

Native sklearn seed derivation is split_seed=seed32 and sub_split_seed from
np.random.RandomState(seed32).randint(0,2**31-1). NNCreator applies native
combine_seeds(split_seed,ensemble_index) for shared data initialization and
combine_seeds(sub_split_seed,ensemble_index) for individual initialization.
Derive a dedicated collision-checked32-bit protocol seed; retain both levels,
complete sampler RNG/order and source. Do not substitute a direct Torch seed and
claim native seed mapping. Fresh method workers isolate native global state.

Fitting the converter on F is the explicit protocol boundary adaptation; direct
parity uses that same adaptation and official native model/trainer. Merely passing
X_val separately to the public sklearn wrapper does not provide it.

NNCreator.create_model already fits all data-dependent transforms and initial
values on the F subdataset. split_off_dynamic leaves only nontrainable transforms
requiring no tensor values in the static stage; their metadata now comes from F.
Median center and robust scale therefore fit F. Native robust scaling uses IQR,
fallback half-range when IQR is zero, epsilon1e-30 and factor0 for truly constant
columns. Preserve numeric constants and native smooth clipping; do not transplant
P03/TabM quantile preprocessing or their constant-column dropping rules.
PBLD numerical embeddings exclude one-hot categorical channels through the native
factory's own ordering. Keep native binary one-hot compression, thresholds and
8-dimensional embeddings for larger categorical cardinalities. Unknown/missing ordinal code zero is passed through the pinned one-hot special
cases; it is not forced to a custom all-zero representation. The default normal embedding initializer
retains random unknown-row weights. Do not force zero embeddings or grow the
dictionary from G/query values.

Despite the variable name trainval_ds, the pinned NNCreator assigns it the F
subdataset before NormalizeOutputFactory and ClampOutputFactory run. Hence mean,
population std, min/max clipping bounds and supervised initialization already
fit F. Preserve native float32 normalization (std+1e-30 during training), inverse
scaling and evaluation-only min/max clipping. Constant targets use this native
behavior; do not transplant TabM scale1. Nonfinite statistics/loss/gradients/
parameters/predictions fail visibly. Finite numerical differences are measured,
not silently repaired by a fallback model. G selects checkpoints only.

## Frozen HPO distribution

P06 tries the full TD anchor first, within its one32-attempt/120-minute family
budget. Subsequent configurations use the pinned official
RealMLPParamSampler(is_classification=False,hpo_space_name='default'). It starts
from the same TD regression dictionary and draws in this exact order:

- numerical embedding: uniform ['none','pbld','pl','plr'];
- front scaling: True/False with probabilities .6/.4;
- LR: exp(uniform(log(.02),log(.3)));
- dropout: 0/.15/.3 with probabilities .3/.5/.2;
- WD: uniform choice 0/.02;
- plr_sigma: exp(uniform(log(.05),log(.5)));
- activation: uniform ['relu','selu','mish'];
- hidden sizes: [256]*3/[64]*5/[512] with probabilities .6/.2/.2.

The paper's regression sigma entry is blank, while the pinned default sampler
samples it for both tasks. Follow and disclose the pinned sampler; no label
smoothing is drawn for regression. Do not select newer alt*/TabArena distributions,
retune embedding dimensions, enable early stopping, change epochs or expand
supports based on V outcomes. Parameters that have no effect for a selected
embedding variant remain visible/paid. HPO sampler seed is separate from fit seed;
neither is a candidate chosen by score. Actual HPO implementation/execution is
outside this preparation slice.

## Private dependencies and recovery

See [realmlp-dependencies.md](realmlp-dependencies.md) and
locks/research-v2/realmlp/dependency-availability.json. Core-constrained resolution
has37packages:22already satisfied in the frozen core and15missing private wheels.
The other5core distributions remain present and unchanged. All27core distribution
identities were rechecked during static preparation. The complete active metadata
closure has54edges, including fsspec[http], with compatible pinned versions.
Wheel availability is not actual import/numerical/API compatibility.

After root review, installation used a fresh RealMLP-only overlay with
--no-index --no-deps --require-hashes and requirements-overlay.txt. It must not
reuse the TabM/tree overlays or install optional models/bench/hpo extras. Do not
update shared Torch/NumPy/sklearn to make an import succeed. Retain any failure
and propose a separately identified compatible private runtime if unavoidable.
The first cross-target resolution failed because generic aarch64-apple-darwin
defaulted to macOS13 while frozen Torch2.14.1 has a macOS14+ wheel. Retained native
macOS26 resolution succeeds without any version/platform fiction.

## Implementation ownership and qualification gates after approval

Own only a new research/realmlp_native.py or dedicated module package, its new
tests/reference fixtures, locks/research-v2/realmlp and dedicated docs/receipts.
Do not edit existing native adapters, legacy registries, core/uv locks,
execution.md, root source or prior finalized evidence.

Start with TDD strict configuration/F-G label boundaries and complete installed
RECORD/module-origin/actual-native-runtime authentication. Require fresh numerical
threads1, bytecode disabled and artifact-local caches; record upstream KMP
environment transitions without patching/preloading/suppressing libraries.
Qualify numeric, mixed low/high-cardinality and unknown-category tiny native
fixtures against direct pinned factory/trainer calls. Include non-divisible F
loader exposure and last-best tie/constant-target controls; no full anchors yet.

Portable prediction snapshots are a required engineering gate, not an assumed
native feature: native Layer/Variable/Context trees include non-state_dict
callables, factors and transform topology. A tensor file alone is insufficient.
Plan a closed, versioned native-layer constructor registry and JSON structural
manifest plus strict NPZ tensors, source/dependency/runtime/recipe/transform/
selection binding and independent trusted manifest digest. Preserve every
inference-relevant scalar/buffer/callable identity; no generic eval/pickle or
quiet reduced model. Qualify each admitted native default/HPO graph before
declaring snapshot support. If faithful reconstruction needs retained F tensors
or another native artifact, record its storage/load costs and require a named
reviewed representation before proceeding; do not invent an untested portable
format. Formal deployment timing stays pending until this is solved. Native
training interruption recovery and hard process deadline/RSS enforcement also
remain explicit pending work, never inferred from a successful tiny fit.

Full256-epoch native anchor eligibility, actual transfer workload qualification,
HPO/registry integration, deployment measurements, recovery and Gate D remain
pending. An engineering-only implementation can finish with those limits; it
cannot supply the full mandatory-suite superiority claim. No predictive or
placement result follows from this static preparation.

## Approved implementation correction: native regression selection metric

The paper describes MSE validation. The pinned native regression factory instead
uses `val_metric_name='rmse'`, while the training loss separately accepts
`train_metric_name='mse'`. Preserve those exact source keys, 256 epochs, and native
latest-best `<=` selection. Although mathematical MSE and RMSE rankings are
monotone, float32 ties need not coincide; selecting on a reconstructed MSE history
would change the declared procedure. Log both raw G MSE and native G RMSE and bind
the selected epoch to native RMSE. External HPO V selection remains the declared
group/day-weighted normalized MSE.

The first numeric and categorical tiny engineering fits failed at their first
validation because the adapter passed the training alias `mse` to `Metrics.apply`,
which does not define that validation key. Both failures and paid execution remain
in `artifacts/research-v2/WP07/realmlp/04-native-first`; no seed, epoch, dependency
or tolerance was changed to recover. The native implementation sanitizes
nonfinite predictions before metric calculation. An observation callback must
check every cached raw G prediction before the module's validation epoch hook;
nonfinite evaluations fail explicitly even if native sanitization would hide them.
The approved original contract is retained in the evidence root. This correction
was approved by root after inspecting the pinned source.

## Reviewed implementation boundary additions

The independent review and correction evidence are in realmlp-qualification.md.
Actual missing categories remain np.nan at the official converter boundary;
typed nonmissing values retain canonical scalar prefixes. F category arrays and
a separate F-missing flag are bound in the closed snapshot/carrier. No missing
token is learned, and no G/query category enters the vocabulary. Fit publication
is transactional; retained executable/runtime and exact preprocessing functions/
order/static partition are checked at load. Original reviewed contract bytes
remain in the immutable evidence root, with every correction assigned a new
source identity. Initialization-only RAM-subsample indices remain unobserved in
this slice and require qualification before a workload activating that policy.
