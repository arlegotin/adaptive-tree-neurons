# Official NODE regression recipe research for EvidenceTask3

Research only, retrieved 2026-10-16. This memo supports EvidenceTask3 and design-spec B7; it does not implement an adapter, satisfy parity/acceptance gates, reproduce a score, or establish compatibility. Only official repository/API source was inspected. No third-party packages were installed or imported, no Torch/model code was executed, and no dataset or weights were downloaded. Source responses were read in memory and hashed; original vendor files have **not** been archived in the project yet.

The binding requirements are [Evidence Task 3](../superpowers/plans/2026-10-16-selective-tree-neurons-evidence.md#task-3-pin-and-verify-the-actual-node-regression-recipe) and [design comparison families/B7](../superpowers/specs/2026-10-16-selective-tree-neurons-design.md#8-comparison-families-and-tuning). Future execution must retain the selected sparse functions, F-only initialization and full-F fitting, and explicitly report any unavailable faithful path.

## Immutable source anchor and selectable experiment

The live official `master` API returned **`3bae6a8a63f0205683270b6d566d9cfa659403e4`** at **2026-10-16T15:26:44Z**. The commit is dated 2019-12-29T09:48:37Z, message `Create LICENSE.md`, tree `e6694b8dc0bebc25b1ff01bf8e4c739137f1b042`. Verify against the [immutable commit](https://github.com/Qwicen/node/commit/3bae6a8a63f0205683270b6d566d9cfa659403e4), [commit API](https://api.github.com/repos/Qwicen/node/commits/3bae6a8a63f0205683270b6d566d9cfa659403e4), and [immutable recursive tree](https://api.github.com/repos/Qwicen/node/git/trees/3bae6a8a63f0205683270b6d566d9cfa659403e4?recursive=1). Branch-head discovery used the mutable `/commits/master` endpoint; all recipe links below pin the inspected commit.

There are two official YEAR regression notebooks. A concrete selectable B7 anchor is **`year_node_shallow.ipynb` at this commit**, preserving its full configuration. This is a proposed named anchor for the later task, not a claim that it is the paper-wide default or has a known local cost. The second notebook is a separate legitimate anchor, with consequential differences:

| Setting | `year_node_shallow` | `year_node_8layers` |
|---|---:|---:|
| Trees per layer | 2048 | 128 |
| Layers | 1 | 8 |
| Total trees | 2048 | 1024 |
| Depth / response channels | 6 / 3 | 6 / 3 |
| Explicit choice / bin | `entmax15` / `entmoid15` | `entmax15` / `entmoid15` |
| Initialization rows, ordered training prefix | 1000 | 5000 |
| Training batch | 512 | 1024 |
| Validation chunk | 1024 | 16384 |

Both use YEAR, seed 1337, noisy normal-quantile features, training-target standardization, QHAdam with `nus=(0.7,1.0)` and `betas=(0.95,0.998)`, MSE, five recent checkpoints, reports every 100 optimizer steps, and strict stopping when `step > best_step + 5000`. They load the best validation checkpoint for test evaluation and rescale normalized MSE by target variance. Sources: [shallow notebook, cells 1–8, zero-based](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/notebooks/year_node_shallow.ipynb), [eight-layer notebook, cells 1–8](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/notebooks/year_node_8layers.ipynb).

Exact model-selection excerpt from shallow cell 2 (omit notebook presentation and device scaffolding when creating a worker):

```python
lib.DenseBlock(in_features, 2048, num_layers=1, tree_dim=3, depth=6, flatten_output=False,
               choice_function=lib.entmax15, bin_function=lib.entmoid15)
lib.Lambda(lambda x: x[..., 0].mean(dim=-1))
```

These expressions are consecutive modules inside the notebook's `nn.Sequential`. The notebook does **not** specify learning rate, epsilon, weight decay, QHAdam version, or most sklearn transformer arguments. Those cannot be called experiment-selected values. The floating QHAdam dependency leaves their historical resolution unresolved.

## Constructor semantics versus experiment semantics

`ODST` defaults are `depth=6`, `tree_dim=1`, flattened output, **`sparsemax`/`sparsemoid`**, normal response initialization, uniform selection-logit initialization, and `threshold_init_beta=threshold_init_cutoff=1.0`. The YEAR experiments override channels, flattening and sparse functions. Selectors have shape `[features, trees, depth]`; thresholds/temperatures `[trees, depth]`; responses `[trees, channels, 2**depth]`. Each tree uses one split per depth, shared across its nodes and response channels; responses remain per tree/channel/leaf. Forward selects features on dimension 0, evaluates complementary signed bins, multiplies depth matches into leaf weights, then sums responses. Two-dimensional input yields `[batch, trees, channels]` when unflattened. Initialization computes selected feature values, draws per-tree/depth percentiles from `100*Beta(beta,beta)`, and sets thresholds via NumPy percentiles. Temperature is the percentile of absolute threshold distance at `100*min(1,cutoff)`, divided by `max(1,cutoff)`; stored log-temperature adds `1e-6` before log. Follow this code rather than extrapolating the sparsemoid-focused docstring. Fewer than 1000 initialization rows emits a warning. [ODST constructor/forward/initialization, lines 10–123](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/odst.py#L10).

`DenseBlock` concatenates each layer's outputs onto its input and returns all generated features. Defaults leave feature history uncapped and input dropout zero. Its unflattened result is `[batch, layers*trees_per_layer, channels]`; the head selects channel 0 and averages **all trees across all layers**, yielding `[batch]`. Channels 1–2 therefore provide downstream features in the deep recipe; reducing `tree_dim` changes that recipe. Preserve even the shallow recipe's unused response channels. [DenseBlock, lines 6–33](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/arch.py#L6).

The selected functions are custom autograd `Entmax15Function` and `Entmoid15`; the bin's forward/backward helpers are TorchScript. Entmax sorts and computes the alpha-1.5 threshold, then squares its positive part; backward uses the square-root support-weighted correction. Entmoid is the binary alpha-1.5 counterpart, with exact saturation and its supplied backward. The file explicitly flags its entmax threshold-support calculation as differing from the referenced algorithm. Retain **this implementation** for vendor parity, including support-boundary conventions; a sigmoid or merely similarly named package function is insufficient. Separately, sparsemoid is the clipped affine function, not logistic sigmoid. `ModuleWithInit.__call__` initializes on the first actual input; its persisted uint8 flag and Python cache need faithful save/reload treatment. The source credits Ben Peters/Vlad Niculae for sparsemax and references deep-spin/entmax for entmax. [nn_utils, lines 34–203](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/nn_utils.py#L34).

## Preprocessing, fitting and stopping details

`Dataset` seeds NumPy, Torch and Python. The YEAR loader uses the final 51630 rows as test data and external Dropbox training/validation indices by default. Its label dtype uses removed `np.int`. Quantile fitting copies training features and adds Gaussian noise with per-feature amplitude `quantile_noise / maximum(training_std, quantile_noise)`; this differs from the prose docstring's multiplication description. A `QuantileTransformer` with normal output and seed 1337 fits that noisy copy, then transforms original train/valid/test features without adding output noise. Unspecified transformer defaults are sklearn-version dependent. Feature normalization defaults false. [Dataset, lines 21–83; YEAR, lines 246–284](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/data.py#L21).

The trainer computes MSE on the **aggregated scalar prediction** and backpropagates its mean; this is not per-tree/member MSE. Validation switches to eval mode and evaluates prediction chunks without gradients. `average_checkpoints` takes the newest five temporary snapshots by filesystem creation time and averages **model state values**, retaining optimizer/step metadata from the first loaded snapshot; it is parameter averaging, not prediction averaging. The notebook evaluates this averaged state, saves it when validation improves, then restores the latest unaveraged training checkpoint before continuing. At early reports fewer than five snapshots may exist. Do not replace this with evaluation of the latest raw model or continued optimization from averaged weights. [Trainer, lines 17–34, 62–115, 128–136](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/trainer.py#L17).

Minibatches shuffle a NumPy index array each epoch, include the final incomplete batch by default, and run indefinitely until notebook stopping. Chunking concatenates model outputs without a new prediction aggregation. `check_numpy` detaches and copies tensors to CPU. The notebook neither requires nor justifies dropping F rows for training. [Utilities, lines 42–88](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/utils.py#L42).

For this project's F/G/V/T protocol, the following are **adapter requirements/inferences**, not claims that the upstream notebooks know those split names: fit feature transforms and target scaling on F only; initialize all layers from the declared ordered F prefix only; record row IDs/order and random states. Use the authorized validation split for stopping without fitting a transform or triggering initialization on G/V/T. Changing G/V values must leave F-derived thresholds and temperatures unchanged. Initialization on a prefix does not permit fitting on that prefix: normal optimization must cover full F, with complete cost accounting. If F has fewer than the anchor's requested initialization rows, record the actual count and discrepancy before execution. Persist preprocessing state, target mean/std, source/config hashes and initialized model state; reload into a fresh model before its first forward. Test that prediction/save/reload cannot trigger data-aware fitting on evaluation rows.

## Environment and parity risks requiring later execution

The supplied installed-core context is Python 3.12 / Torch 2.14.1 / native arm64 CPU; this research did not import or independently verify those packages. The upstream README describes Ubuntu 16.04-era testing and permits CPU execution, while warning of high resource use. That is no evidence for present arm64 or core-stack compatibility. [Pinned README](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/README.md).

`requirements.txt` combines broad lower bounds with `catboost==0.12.2`, `xgboost==0.81`, and a mutable `facebookresearch/qhoptim/archive/master.zip`. Installing it wholesale is neither an immutable lock nor a demonstrated Python 3.12/arm64 solution. Avoid importing upstream `lib` blindly: its initializer imports every module, pulling in data/training dependencies. [Requirements](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/requirements.txt), [package initializer](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/lib/__init__.py).

Consequential unresolved checks for EvidenceTask3:

- Resolve an immutable QHAdam implementation/version and **its** effective default learning rate/epsilon/weight decay; record the choice as dependency resolution because NODE does not pin it. Check old optimizer API assumptions against the selected Torch version.
- Lock sklearn/NumPy versions and explicitly record transformer effective parameters. Quantile subsampling/default changes, removed NumPy aliases, float precision and percentile behavior can change initialization or preprocessing. Fixing loader compatibility does not establish numerical parity.
- Test the supplied custom autograd/TorchScript functions on CPU, forward **and backward**, including zero/support boundaries, tied logits, saturated bins, noncontiguous tensors and dimensions used by actual selector tensors. No compiled extension appears in the inspected sparse-function path, but successful modern execution remains unverified.
- Independently evaluate the pinned vendor implementation before evaluating any port. Test initialized thresholds/temperatures, prediction shapes, finite full-F tiny fitting and fresh-model save/reload. The core stack is not assumed to be the reference worker.
- Exercise the exact checkpoint averaging/restoration rule, integer initialization state, modern checkpoint loading, and CPU map-location behavior. Preserve train/eval semantics. Explicitly select CPU; MPS parity was not investigated.
- Measure faithful anchor cost before any resource decision. A single smaller smoke configuration cannot establish full anchor feasibility; smaller trees, fewer layers, altered sparse functions or dropped F rows need separate names and cannot silently inherit B7 claims.

A resolved **isolated CPU worker** is the preferred first attempt when legacy dependencies conflict with the installed core. A faithful minimal port may extract only ODST, DenseBlock, sparse autograd/init utilities, QHAdam and the required regression training/preprocessing behavior, retaining original files/notices and separately documenting every compatibility patch. It must pass independent vendor parity in the resolved environment. Neither approach was implemented or selected here; failure to resolve/verify either remains an explicit B7 exclusion/gate failure.

## Source and asset terms; retrieval scope

The pinned repository has an MIT license, copyright 2019 Sergey Popov. When copying substantial source, retain the complete copyright and permission notice, and preserve the supplied warranty disclaimer by including the license. Keep the sparse-function author/source attributions with copied code. The NODE license alone does not establish terms for every third-party dependency or externally hosted dataset. Upstream deep-spin/entmax licensing, the eventual QHAdam revision/license and its notices, and other resolved dependency notices still require a separate dependency inventory. [Pinned LICENSE.md](https://github.com/Qwicen/node/blob/3bae6a8a63f0205683270b6d566d9cfa659403e4/LICENSE.md).

YEAR data and split indices are referenced by Dropbox URLs in `fetch_YEAR`; their availability, byte hashes and redistribution terms were **not checked**. No learned weights are referenced by these two recipes. The repository image and classification notebook were not inspected or copied. Do not treat source-code MIT terms as dataset permission, and do not imply that an untested download URL establishes asset availability.

The GitHub tree inventory, README, license, requirements, `lib/__init__.py`, `odst.py`, `nn_utils.py`, `arch.py`, `trainer.py`, `utils.py`, `data.py`, and both YEAR notebook code cells were inspected. Notebook output images were not analyzed. `lib/data.py` and the pinned deep notebook were initially unavailable through the browser cache; both were subsequently retrieved successfully as official raw GitHub responses using read-only requests. No unresolved source-access block remains for the listed recipe files. The paper and other repositories were not used to infer missing settings. Future Task3 still needs original-file archival, worker lock, dependency/asset terms, actual execution/parity and all mandated review/acceptance gates.

SHA256 values below were computed directly from the full raw HTTP response bytes for `https://raw.githubusercontent.com/Qwicen/node/3bae6a8a63f0205683270b6d566d9cfa659403e4/<path>` on the retrieval date. They are archival verification anchors, not evidence that copies already exist locally.

| Path | SHA256 |
|---|---|
| `README.md` | `d4bf222d99d3fb42cb26fbc37fb23490fd60c9a4465b0636ae3dcfd43da3e011` |
| `LICENSE.md` | `60b6519a046458e6901cf233eb0d403ae56f499aa6fc008e95fceb75eea4edd4` |
| `requirements.txt` | `e40f34e2ec7ce8f2bf68bb203d1c82d31ed2ee794f7860753f88c1988ca1293f` |
| `lib/__init__.py` | `16cd879e8b3a6354615dc32fe3335cd4597ab4ad206975f9483fd50a2aa98753` |
| `lib/odst.py` | `878fa48cbf3c319dc9909400efe0ed601dae180754ae30997aaf14c94e59b81b` |
| `lib/nn_utils.py` | `91fcac1db376552fef6ec9555cbfefbe468036d8a7d1e1d836d8d5aca8d52a4a` |
| `lib/arch.py` | `02f9d08cfadc96b3d77c04dbe6720be7a5e165892949944c366439c018e50541` |
| `lib/trainer.py` | `68db1c8af0f4a71619ba41aca580eea75329b0d231802a1a1c8f469f515549e7` |
| `lib/utils.py` | `fac08661980a8097f8e4e7e6fd3745be19bd91db17def866b6023c9d6127354c` |
| `lib/data.py` | `cbe0accfd0ce06bf640af27e7f3112338801383f1f92083449853b16c2d21808` |
| `notebooks/year_node_shallow.ipynb` | `28e2dd2fbd4122b282d6cd072772d42f67e4fa40caac9e85c707293e4d16ab0b` |
| `notebooks/year_node_8layers.ipynb` | `a781e8d4642a593d68b1a74d695f10acaf49c693d820248f7dcb8d5269a3d0a1` |

## Follow-up: bounded QHAdam dependency candidate, 2026-10-16

Official metadata/source inspection at approximately 15:31–15:34 UTC identifies **`qhoptim==1.1.0` as a reproducible candidate to attempt later**, rather than a demonstrated compatible dependency. The [version-specific PyPI JSON](https://pypi.org/pypi/qhoptim/1.1.0/json) lists non-yanked wheel/source releases uploaded 2019-10-02. The wheel is pure Python (`py3-none-any`); its metadata declares neither `Requires-Python` nor `Requires-Dist`, despite the optimizer importing Torch. These metadata omissions do not certify Python 3.12, arm64 or Torch support. The wheel was retrieved and inspected **in memory only**, never installed, imported, executed or written to disk.

The immutable source candidate is [commit `0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea`](https://github.com/facebookresearch/qhoptim/commit/0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea), dated 2019-09-26, which changes the version to 1.1.0. Its optimizer, version file and license match the PyPI 1.1.0 wheel members **byte for byte**; this comparison does not establish that every repository file equals the built distribution. The inspected current archived-repository head is `e81dea3f2765780cf4fbb90b87b22ba7604b8625`; its optimizer has the same source hash. The official tags API exposes no `1.1.0` release tag, so record the commit and artifact hash rather than inventing one. [Pinned version file](https://github.com/facebookresearch/qhoptim/blob/0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea/qhoptim/version.py), [immutable head commit](https://github.com/facebookresearch/qhoptim/commit/e81dea3f2765780cf4fbb90b87b22ba7604b8625).

| QHAdam argument | 1.1.0 constructor default | Effective candidate for the NODE notebooks | Provenance |
|---|---|---|---|
| `lr` | `1e-3` | `1e-3` | Resolved dependency default |
| `eps` | `1e-8` | `1e-8` | Resolved dependency default |
| `weight_decay` | `0.0` | `0.0` | Resolved dependency default |
| `decouple_weight_decay` | `False` | `False` | Resolved dependency default |
| `nus` | `(1.0, 1.0)` | `(0.7, 1.0)` | Explicit NODE notebook override |
| `betas` | `(0.9, 0.999)` | `(0.95, 0.998)` | Explicit NODE notebook override |

Source: [QHAdam constructor, lines 49–78](https://github.com/facebookresearch/qhoptim/blob/0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea/qhoptim/pyt/qhadam.py#L49), with the previously linked NODE notebook cell 3 overrides. These resolve a **new, explicit dependency choice**. They do not identify the moving-`master` checkout used by historical NODE runs, and the PyPI release postdates the saved notebook output dates. Preserve that uncertainty even if later execution succeeds.

Source inspection highlights legacy positional `add_` and `addcdiv_` calls, `.data` updates, Python-scalar bias-correction state and rejection of sparse gradient tensors. [QHAdam step, lines 80–149](https://github.com/facebookresearch/qhoptim/blob/0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea/qhoptim/pyt/qhadam.py#L80). Torch 2.14 documents keyword-only scalar arguments for [Tensor.add_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.add_.html) and [Tensor.addcdiv_](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.addcdiv_.html). However, the official `v2.14.1` tag resolves to **`5c4886908584029761b579af026dcfb627c84070`**, whose [deprecated-overload declarations, lines 7–8 and 34–35](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/tools/autograd/deprecated.yaml#L7) still map both legacy signatures. Its [Optimizer constructor, line 396](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/optim/optimizer.py#L396) still accepts `(params, defaults)`. Thus the inspected source gives a reason to **attempt unchanged eager CPU execution**, not a source-level basis for declaring those calls broken or proactively rewriting them. Actual binding behavior, warnings, numeric updates and checkpoint-state restoration remain unexecuted.

For the later worker, explicitly log every effective argument above and retain QHAdam's supplied update/bias-correction equations; substituting Adam/AdamW is not a dependency resolution. If actual execution requires scalar-argument compatibility patches, archive the original source and patch separately and verify multi-step parameters and optimizer state against an independently runnable pinned reference. Sparse selector **outputs** alone do not establish the storage format of parameter gradients; check that during future parity execution. Package imports, full NODE training, serialization and TorchScript compatibility remain separate unresolved checks. No current-core package or lock was changed here.

QHOptim's [pinned MIT license](https://github.com/facebookresearch/qhoptim/blob/0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea/LICENSE) credits Facebook, Inc. and its affiliates. Preserve its complete copyright/permission notice and disclaimer when copying substantial code, and retain the source-file license headers. The inspected wheel contains that same license both in `qhoptim-1.1.0.dist-info/LICENSE` and under `source_docs/qhoptim/LICENSE`. This resolves this candidate dependency's source notice; other dependencies and YEAR asset terms remain unresolved.

| Follow-up artifact or inspected member | SHA256 and status |
|---|---|
| [Official 1.1.0 wheel](https://files.pythonhosted.org/packages/bc/e6/b3fb8a75287252ae697e7b8ab092d54eceaf3fd76f766ebefc3c2e82108e/qhoptim-1.1.0-py3-none-any.whl) | `cdd9660b8310ef7749733db03fe372bf5453861ba9cb0aa31db59f6127b3ae00`; computed from response bytes, matches PyPI |
| [Official 1.1.0 source archive](https://files.pythonhosted.org/packages/b9/9e/f18a794c5a3f6abc3cbf9d450e956395234e75a1586f8065258eadb68ca7/qhoptim-1.1.0.tar.gz) | `e8578e2465d25f1c423a66d329a2c85b6bdfeae14cb95df183734ba0227fb1ca`; PyPI-reported only, archive not retrieved |
| `qhoptim/pyt/qhadam.py` at source candidate / wheel | `ab29409cfa20fcd576ba239440ee3a651dbe734d68743cd3f768727bbc3aeb2e`; computed, equal |
| `qhoptim/version.py` at source candidate / wheel | `a6b0552b8565919aa903b817841f1d34a8c9a6ec4786eeebdbcbc4fc0e6ff728`; computed, equal |
| QHOptim `LICENSE` at source candidate / wheel | `52412d7bc7ce4157ea628bbaacb8829e0a9cb3c58f57f99176126bc8cf2bfc85`; computed, equal |
| `qhoptim-1.1.0.dist-info/METADATA` in wheel | `b8cfffd31027114ba0c74a56bba5fc43c28bbf5785ef15e2aab983629cc23df1`; computed, matches PyPI metadata digest |
| PyTorch 2.14.1 `tools/autograd/deprecated.yaml` | `51bb5a8f35a8f3d8dfd50e6cc3c24d248653e6cfa20c16a09eb189b61b434b64`; computed from official tag-resolved source |
| PyTorch 2.14.1 `torch/optim/optimizer.py` | `ebc2511e0a07b18bbfba44f61667ddaeee47f23cdc856a43234d89007010fe86`; computed from official tag-resolved source |

The follow-up inspected official PyPI metadata and wheel contents, QHOptim GitHub commit/file histories and the constructor/update/license/version/package metadata, plus the two relevant official Torch API pages and two source files. It did not inspect or execute upstream tests, other optimizer implementations, all import dependencies, build tooling or any model/data assets. This narrows the earlier missing-defaults lead without completing EvidenceTask3's environment, parity or execution gates.
