# Third-party notices

The project licenses cover original project material. They do not relicense
dependencies, upstream snapshots, external assets, cited publications or their
authors' work. Original notices remain in place.

## NODE, entmax and QHOptim

The optional NODE adapter includes restricted, attributed compatibility copies
and upstream source snapshots:

| Component | Upstream | License and retained notice |
| --- | --- | --- |
| NODE / ODST | [Qwicen/node](https://github.com/Qwicen/node), copyright 2019 Sergey Popov | MIT; [LICENSE-NODE.md](src/treeneuron/baselines/_node_vendor/LICENSE-NODE.md) |
| entmax / entmoid | [deep-spin/entmax](https://github.com/deep-spin/entmax), copyright 2019 DeepSPIN | MIT; [LICENSE-ENTMAX](src/treeneuron/baselines/_node_vendor/LICENSE-ENTMAX) |
| QHOptim / QHAdam | [facebookresearch/qhoptim](https://github.com/facebookresearch/qhoptim), copyright Facebook, Inc. and affiliates | MIT; [LICENSE-QHOPTIM](src/treeneuron/baselines/_node_vendor/LICENSE-QHOPTIM) |

Runtime copies are under [src/treeneuron/baselines/_node_vendor](src/treeneuron/baselines/_node_vendor).
The [pinned source records](configs/vendor/node/sources.json),
[compatibility notes](configs/vendor/node/README.md), patch and runtime-file
mapping identify exact sources and local adaptations. Archived notebook, source
and QHOptim package-member snapshots under `configs/vendor/node/upstream/`
retain their upstream licenses. They are reference snapshots, not original
project code. Attribution does not claim a reproduced published YEAR score.

## Research v2 native adapters and source references

The continuation adds wrappers around upstream tabular implementations. The
wrappers are original project software; the implementations and retained source
references keep their own licenses:

| Component | Upstream and terms | Retained provenance |
| --- | --- | --- |
| RTDL-style MLP/ResNet and quantile preprocessing | [yandex-research/rtdl-revisiting-models](https://github.com/yandex-research/rtdl-revisiting-models), Apache-2.0 | [Ordinary-neural source pins and notices](locks/research-v2/ordinary-neural) |
| PyTabKit / RealMLP | [dholzmueller/pytabkit](https://github.com/dholzmueller/pytabkit), Apache-2.0 with the nested AUC Mu MIT exception below | [Exact source lock and licensed snapshots](locks/research-v2/realmlp) |
| TabM and numerical embeddings | [yandex-research/tabm](https://github.com/yandex-research/tabm) and [yandex-research/rtdl-num-embeddings](https://github.com/yandex-research/rtdl-num-embeddings), Apache-2.0 | [Source lock, snapshots and license texts](locks/research-v2/tabm) |
| skorch dependency | [skorch-dev/skorch](https://github.com/skorch-dev/skorch), BSD-3-Clause | [Retained license](locks/research-v2/ordinary-neural/skorch-LICENSE.txt) |
| CatBoost, XGBoost and LightGBM | Official installed packages under their respective terms; no package binaries bundled | [Pinned tree resolution](locks/research-v2/tree-resolution.json) and [command receipts](locks/research-v2/tree-command-receipts.json) |

The v2 NODE wrapper uses the existing MIT runtime material described above;
its [separate source lock](locks/research-v2/node-v2/source-lock.json) records
the versioned qualification path. Adapter execution does not imply reproduction
of an upstream paper's benchmark. License documents, upstream text snapshots
and installed-file inventories are attribution/provenance records, not a blanket
license grant for external packages or an installed environment.

The retained [AUC Mu source](locks/research-v2/realmlp/sources/pytabkit/models/training/auc_mu.py.txt)
credits Ross Kleiman, copyright 2019, and declares MIT. Its original
[MIT notice](licenses/AUC-Mu-MIT.txt) is retained separately from PyTabKit's
Apache license, copied unchanged from
[kleimanr/auc_mu at `8586753`](https://github.com/kleimanr/auc_mu/blob/8586753cb225424c63310983b10858a30cde769a/LICENSE.md).
This is an attributed upstream source reference; it is not a new project
contribution or a reassignment of its license.

## Dependencies and license archives

PyTorch, NumPy, SciPy, pandas, scikit-learn, PyYAML, pytest, optional Optuna and
their transitive dependencies retain their own licenses and bundled-component
notices. The [core source manifest](provenance/sources.json), `locks/node/` and
`locks/search/` retain installed-distribution metadata and license texts.
License archives include notices for components bundled upstream; inclusion of a
notice does not mean this repository vendors that component's binary or source.
The public release excludes installed environments and dependency binaries.

## Data, models and cited research

The controlled mechanism studies use this project's generated synthetic inputs.
Small selected engineering arrays and checkpoint fixtures are versioned; full
training archives and final-study checkpoints are external. No pretrained
third-party model weights are bundled.

The native-anchor development survey also fits official UCI Airfoil, Concrete,
Combined Cycle Power Plant, Parkinsons Telemonitoring, Bike Sharing and Energy
Efficiency data. Acquisition/license records, exact raw hashes and split
contracts are described in the [v2 reproduction guide](docs/research/v2/reproduce.md).
Raw datasets are not bundled in this repository. Their dataset-specific terms
remain with the original providers. The prepared test labels remain sealed;
completed validation fits are distinct from the unfinished tuned/test benchmark.

Cited papers, external packages, optional pretrained models and weights retain
their own terms. A discussed or planned method is not thereby an executed
comparison or licensed for redistribution. Repeat source/license checks when
acquiring a new asset.

## Writing assistance

README editing used the requested
[write-like-me](https://github.com/arlegotin/write-like-me) skill. Its source and
reference material are not bundled. Implementation, experiments and documentation
used AI assistance under the project owner's direction.
