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

## Dependencies and license archives

PyTorch, NumPy, SciPy, pandas, scikit-learn, PyYAML, pytest, optional Optuna and
their transitive dependencies retain their own licenses and bundled-component
notices. The [core source manifest](provenance/sources.json), `locks/node/` and
`locks/search/` retain installed-distribution metadata and license texts.
License archives include notices for components bundled upstream; inclusion of a
notice does not mean this repository vendors that component's binary or source.
The public release excludes installed environments and dependency binaries.

## Data, models and cited research

The reported predictive studies use this project's generated synthetic inputs.
Small selected engineering arrays and checkpoint fixtures are versioned; full
training archives and final-study checkpoints are external. No pretrained
third-party model weights are bundled.

The design discusses UCI datasets, tabular packages, pretrained models and prior
neural-tree/gating research. Their papers, code, datasets and weights retain
their own terms. Planned baselines and datasets are not represented as executed
or licensed for redistribution by this project. Source/license checks must be
repeated when acquiring any new asset.

## Writing assistance

README editing used the requested
[write-like-me](https://github.com/arlegotin/write-like-me) skill. Its source and
reference material are not bundled. Implementation, experiments and documentation
used AI assistance under the project owner's direction.
