# RealMLP private dependency availability

Historical initial status: downloaded and statically verified before root
review. The approved private overlay has since been installed and engineering-
tested; the current execution section below distinguishes that evidence from
full native quality qualification. Prepared transfer data and V/T stay untouched.

PyTabKit1.7.3 declares Torch>=2,NumPy>=1.25,pandas>=2,sklearn>=1.3,
torchmetrics>=1.2.1,pytorch_lightning>=2 and psutil>=5. The frozen core already
satisfies numerical/data requirements. Its PyYAML is needed by native utilities
and Lightning. Do not install the broad models/bench/hpo extras: tree methods,
skorch, probmetrics/numba, AutoGluon, Dask, Ray and external search engines are not
needed for this bounded native regression path. Some are lazily imported only
when their corresponding methods/metrics/serialization are actually called.

Core-constrained binary-only resolution on the qualified native Python3.12.13,
macOS26.7.1 arm64 environment produced37active packages, including22existing core
packages and these15new packages:

| Private package | Exact candidate version |
|---|---:|
| aiohappyeyeballs | 2.7.1 |
| aiohttp | 3.14.5 |
| aiosignal | 1.4.0 |
| attrs | 26.1.0 |
| frozenlist | 1.8.0 |
| idna | 3.20 |
| lightning-utilities | 0.15.3 |
| multidict | 7.1.0 |
| propcache | 0.5.4 |
| psutil | 7.2.2 |
| pytabkit | 1.7.3 |
| pytorch-lightning | 2.6.6 |
| torchmetrics | 1.9.0 |
| tqdm | 4.70.1 |
| yarl | 1.25.1 |

The complete active Requires-Dist closure was independently checked from wheel
metadata and existing core metadata,54edges including fsspec's HTTP extra.
The15selected compatible wheels total3,455,651download bytes and12,809,269expanded
bytes before installer metadata. No source build is required by these candidates.
Wheel names, exact SHA256, PyPI asset URLs, METADATA/license hashes, per-wheel
RECORD verification and dependency edges are retained in
locks/research-v2/realmlp/dependency-availability.json. Requirements-overlay.txt
pins only these15actual wheels; requirements-resolved.txt retains full resolution
and cross-platform distribution hashes. The27package core constraint file is
read-only evidence, not a request to reinstall or replace core packages.

PyTabKit's89Python files exactly match the pinned official Git blobs. Third-party
libraries' API/numerical compatibility is still untested. The native code prefers
lightning.pytorch when installed; this private graph deliberately installs only
pytorch_lightning, selecting the pinned code's explicit fallback. Require exactly
the27core plus15private distributions, reject duplicate identities and authenticate
actual imported modules/native images before allowing a fit. Bytecode writes
must be disabled to keep the audited overlay immutable.

Initial resolution with --python-platform aarch64-apple-darwin selected a macOS13
target, incompatible with frozen Torch2.14.1's macOS14+ wheel. The full error is
retained in artifacts/research-v2/WP07/realmlp-source/resolution.log. Retrying with
the actual qualified interpreter/native platform succeeded in767ms; no package
constraint was relaxed. resolution-native.log preserves that result. An initial
arXiv v4 HTML lookup returned404; the v3primary paper was accessible. No package
download failure occurred. Availability receipts/metadata/wheels remain under
artifacts/research-v2/WP07/realmlp-source.

Prospective installation after review, into a new overlay only:

```sh
uv pip install --python /absolute/qualified/python --no-index --no-deps \
  --require-hashes --find-links artifacts/research-v2/WP07/realmlp-source \
  --target environment/research-v2-realmlp-site \
  -r locks/research-v2/realmlp/requirements-overlay.txt
```

Then retain/audit generated installer files and RECORDs, all42distribution
identities, actual module/native library origins and one-thread runtime treatment.
Do not claim this has happened. If the native Lightning/Torch stack fails, retain
the error and propose a new explicit compatibility candidate/lock; do not silently
downgrade shared core, substitute a bare MLP or import a different Lightning tree.
Tiny parity, full anchors and Gate D remain pending.

## Current approved engineering execution

The preceding prospective status records the preinstallation review. Root then
approved contract fb7e9b05cb0d9bf8b5d158b3c4b238518462aa4dabc100baec899aed8e8adfb4.
All15 wheels were installed offline with the exact no-deps/hash-locked command
into the isolated RealMLP overlay; the qualified shared core stayed unchanged.
`installed-files.json` binds1055 private RECORD entries and every generated
metadata byte. Fresh workers authenticate all42 distributions, selected module
origins and actual core/native images. They reject bytecode writes before import.
Pinned sklearn/threadpoolctl's own KMP environment transition is retained;
observed execution under that upstream suppression is not proof of unsuppressed
OpenMP compatibility.

The final metric-correction run has20 passing scoped tests in35.99s, including
exact native-direct predictions for numeric and mixed low/high-cardinality
3-epoch fixtures. Initial wrong-MSE-key fit failures and metric/guard RED tests
remain immutable. Correct validation uses pinned native RMSE/latest-best<=;
training still uses native MSE, with no dependency addition or schedule change.
Both fixtures select epoch2. Native drop_last admits256 of300 rows per epoch;
actual per-epoch omissions and cumulative exposure are retained. The short
changed-horizon fixtures are quality-ineligible. Source warnings remain in
worker logs (TorchScript/Lightning deprecations and available-MPS/CPU notices);
no backend was switched to silence them.

Evidence is in artifacts/research-v2/WP07/realmlp, with progress.json and numbered
RED/GREEN directories; source/download/install receipts are in sibling
realmlp-source. At that intermediate receipt, snapshots were still proposed; the later
closed inference implementation and reviewed corrections are documented in
realmlp-qualification.md. Full256-epoch anchors, real workloads, HPO, hard
deadlines/RSS and interrupted native recovery remain pending GateD. No prepared
transfer data, V/T labels, paid API or full quality anchor was accessed.

The final reviewed implementation is recorded in realmlp-qualification.md:
52 affected checks passed, all27core identities remain unchanged, and no
additional package/version or shared installation was introduced during fixes.
All native fixtures remain short engineering runs; full256-epoch anchors and
GateD remain pending.
