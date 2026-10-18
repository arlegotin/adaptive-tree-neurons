# Native tree dependency lock resolution

On 18 October 2026, WP07 resolved the complete dependency closure for CatBoost
1.2.10, XGBoost 3.4.1 and LightGBM 4.7.0 against the unchanged WP00 core. All 12
additional compatible wheels were downloaded and verified in scratch. Nothing
was installed, no baseline was imported or fitted, and no prepared data was read.
Gate D remains pending adapter/import/native-anchor qualification.

This section records the resolution phase. Subsequently the parent executed the
shown offline installation into the isolated worktree overlay and the fresh
method-specific import checks passed. See
[tree-import-qualification.md](tree-import-qualification.md) for the installed
byte inventory, actual native libraries, retained failures and pending model
fixtures. The intended-installation wording below describes the earlier phase;
do not rerun it into the existing destination.

The target is CPython 3.12.13, macOS 26.7.1, arm64. Resolution used uv
0.11.29 (`901092ee1`, aarch64-apple-darwin), official PyPI only, binary wheels
only, no prereleases, and an upload cutoff of `2026-10-18T23:59:59Z`. The actual
selected wheel filenames and hashes, rather than a generic version constraint,
define the optional overlay.

## Files and scope

| File | Purpose |
|---|---|
| [tree-requirements.in](../../../locks/research-v2/tree-requirements.in) | Three exact direct release candidates |
| [tree-core-constraints.txt](../../../locks/research-v2/tree-core-constraints.txt) | All 27 installed frozen core versions; none may be replaced |
| [tree-resolved-all.txt](../../../locks/research-v2/tree-resolved-all.txt) | Complete 19-package uv resolution with generated hashes and actual command |
| [tree-requirements.txt](../../../locks/research-v2/tree-requirements.txt) | Installable 12-package overlay, one verified target-wheel hash per package |
| [tree-resolution.json](../../../locks/research-v2/tree-resolution.json) | Actual wheel metadata, dependency edges, licenses, source pins, core identities and byte forecasts |
| [tree-command-receipts.json](../../../locks/research-v2/tree-command-receipts.json) | Resolver failures/recovery, dry-run result and independent checks |
| [tree-resolve.py](../../../locks/research-v2/tree-resolve.py) | Reacquire/verify exact resolved wheel versions in fresh scratch without extraction or installation |

The overlay is complete **together with the qualified core**, rather than a
standalone environment lock. Do not install `tree-resolved-all.txt` over that
core: seven dependencies are intentionally reused, not duplicated in the overlay.
The narrow overlay lock SHA256 is
`1298155fa06b633689632fc29b9fdbad6df30ab9c5ac8ac5f60cf0d4efc4fd93`.

The reused distributions are narwhals 2.27.1, NumPy 2.5.4, packaging 26.3, pandas
3.0.6, python-dateutil 2.9.0.post0, SciPy 1.18.1 and six 1.17.0. The core record
and metadata hashes matched before/after acquisition, and a separate post-run
check found no core identity change. Important core module import origins are
recorded without importing the numerical/baseline modules. The 12 wheel archives
contain no `.pth` startup files and no member overlapping any existing core
distribution's RECORD paths.

## Additional wheels

| Distribution | Selected compatible wheel | SHA256 |
|---|---|---|
| CatBoost 1.2.10 | cp312-cp312-macosx_11_0_universal2 | b27115d5b443048f710001c8ac666892dfe03498492310b00466203c91cc30a5 |
| contourpy 1.4.0 | cp312-cp312-macosx_11_0_arm64 | c76f3a5318164db7d9401132fc1b5364b784c613c93fa506e3b5e6d1bf353ec8 |
| cycler 0.12.1 | py3-none-any | 85cef7cff222d8644161529808465972e51340599459b8ac3ccbac5a854e0d30 |
| fonttools 4.67.0 | cp312-cp312-macosx_10_13_universal2 | fcb9743140419410161acfe7ec205fb0a8a703acfccb85b586becb5a97c047c9 |
| graphviz 0.21 | py3-none-any | 54f33de9f4f911d7e84e4191749cac8cc5653f815b06738c54db9a15ab8b1e42 |
| kiwisolver 1.5.1 | cp312-cp312-macosx_11_0_arm64 | 1798e83840c3f627246104c4d8a9639c60fa068adf9ce92b61791781fa8a68c1 |
| LightGBM 4.7.0 | py3-none-macosx_12_0_arm64 | 129535462686f274df179133643118c5c5c5667167fe6c3a28d955f0b3c8e868 |
| matplotlib 3.11.2 | cp312-cp312-macosx_11_0_arm64 | ef31985c4dedb5f1424e1aec6849a47dd37689cb7fa3c20b1b82187f26806261 |
| Pillow 13.0.0 | cp312-cp312-macosx_11_0_arm64 | fc481f742ccda20e9ed760b8f978dff50f2a4e9bc253ac167fb0a05b0783ef52 |
| plotly 7.1.0 | py3-none-any | dbb7fa18afce40d0a8e80d1bf162eceb3faa0ce5a77fe741ad09a74cf78f53f3 |
| pyparsing 3.3.3 | py3-none-any | ece8c00a69cf01b45d0b1dedabb469c90d8caf996d4fda40f147627a122849a4 |
| XGBoost 3.4.1 | py3-none-macosx_12_0_arm64 | a7afd7dbace0951c93aa85ffe046e54bc40893f5b51cd3e7991eb157bf9c7c7c |

All active `Requires-Dist` markers in the actual downloaded wheels and reused
core dependency metadata were evaluated for this interpreter. Every active edge
is satisfied by the resolved version in either the frozen core or overlay.
Linux-only NCCL and optional widget, plotting, dataframe and distributed extras
were not admitted. CatBoost's ordinary declared graphviz/matplotlib/plotly
dependencies are retained even though qualification will not plot anything.

## Acquisition, failures and verification

Raw files remain under
`/private/tmp/atn-tree-lock-20261018-hin0ywhi/`. This absolute path is an execution
locator, not a clean-checkout prerequisite. Its `acquisition-recovery/wheels/`
directory contains all 12 final verified wheels; acquisition events retain URLs,
timestamps, HTTP status, size, digest and retry identity.

1. `uv pip download --help` showed that this installed uv has no download
   subcommand. Acquisition therefore used Python HTTPS reads of official release
   metadata and wheel URLs, not an invented uv interface.
2. The first exact resolver command failed after uv's three retries because
   sandbox DNS could not resolve PyPI. The unchanged command succeeded with
   authorized normal network access. Both command records and the first error
   text are retained in `tree-command-receipts.json`.
3. The first wheel acquisition verified every download, dependency edge and core
   identity, then failed while serializing matplotlib's multiline/non-ASCII
   license header. A focused reproduction isolated the legacy email parser's
   `Header` object. Converting that field to its full Unicode string fixed receipt
   generation without dropping license content. A fresh acquisition completed;
   an independent comparison confirmed identical wheel hashes across both sets.
4. An offline uv installation **dry run** selected exactly the 12 intended wheels
   from the local wheelhouse. uv created only an empty target's `.lock` file.
   The wrapper initially asserted that no target file at all would exist; the
   corrected independent check confirmed no distribution or module was installed.
   The uv command itself exited zero, and its complete output is retained.
5. Independent checks rehashed all 12 final wheels and the narrow lock, checked
   script syntax, core RECORD identities and wheel/core path collisions. These
   establish a complete install plan, not successful native imports or fitting.

The resolution has no source-build fallback and does not update `uv.lock` or core
packages. A changed wheel hash, missing compatible binary, moved direct source
tag, incompatible dependency, duplicate core identity or changed core RECORD
stops verification. Preserve a failed attempt and reconcile a new explicit lock
instead of ignoring that check.

## Resource forecast and intended installation

The selected wheels total **60,238,911 bytes** (57.45 MiB). Their ZIP members total
**205,266,732 uncompressed bytes** (195.76 MiB). A planning allowance of 15% plus
16 MiB gives **252,833,957 bytes** (241.12 MiB) for the overlay. Allow another
**505,667,915 bytes** (482.24 MiB) for an installation transaction/cache with two
unpacked copies and margins, excluding retained wheel archives. These are byte
forecasts from the actual archives, not measured installed size or worker RAM.
No bytecode compilation is requested. Retaining both acquisition sets currently
uses approximately 120.5 MB of wheel payload; preserve their failure evidence.
Check free disk again before installing because other processes share this disk.

The following is the concrete **next installation command; it has not been
executed**. Run from the implementation checkout, use the qualified interpreter,
and require a fresh overlay destination. This exact option combination was
validated with `--dry-run` against the local wheels:

```sh
ATN_PY="/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python"
uv pip install --python "$ATN_PY" \
  --target environment/research-v2-tree-site \
  --no-deps --require-hashes --only-binary :all: \
  --no-index --offline \
  --find-links /private/tmp/atn-tree-lock-20261018-hin0ywhi/acquisition-recovery/wheels \
  --cache-dir /private/tmp/atn-tree-lock-20261018-hin0ywhi/install-cache \
  --no-config -r locks/research-v2/tree-requirements.txt
```

`--no-deps` is intentional: the complete dependency closure was checked, and the
seven reused core distributions must not be copied into a shadowing target.
After installation, run new-process import/metadata/native-library checks with
the overlay on `PYTHONPATH`, repeat the core-origin/duplicate-distribution checks,
then perform the native qualification fixtures. This lock alone authorizes no
claim of native compatibility or anchor eligibility.

If the scratch wheels are unavailable, the retained downloader can reconstruct
them without running the resolver again:

```sh
PYTHONDONTWRITEBYTECODE=1 "$ATN_PY" locks/research-v2/tree-resolve.py \
  --output /private/tmp/atn-tree-wheels-fresh
```

Choose a new absent output path. The script re-verifies release source pins and
actual wheel metadata, preserves retries, and regenerates the lock/receipt. Run
it only during pre-freeze dependency preparation: it writes the declared tree
lock outputs. A frozen scientific run uses its retained immutable identities and
must not regenerate receipts silently. Requalification under a different core
requires a separately recorded runtime identity.

## Source and license provenance

Release metadata and download hashes came from official versioned PyPI records:
[CatBoost 1.2.10](https://pypi.org/pypi/catboost/1.2.10/json),
[XGBoost 3.4.1](https://pypi.org/pypi/xgboost/3.4.1/json), and
[LightGBM 4.7.0](https://pypi.org/pypi/lightgbm/4.7.0/json). Every additional
dependency has its own versioned PyPI URL, raw metadata digest, project URLs and
actual wheel license-file names/digests in `tree-resolution.json`. These are
provenance records, not a legal analysis.

The three direct official source tags were verified through GitHub's API:
CatBoost `v1.2.10` peels to
[`b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084`](https://github.com/catboost/catboost/tree/b1bd2a6d77219e82a1acfcedfccb8e6f6c1ee084),
XGBoost `v3.4.1` is
[`6fe8c547bdd21c73e4555d85b087d9260595d30d`](https://github.com/dmlc/xgboost/tree/6fe8c547bdd21c73e4555d85b087d9260595d30d), and
LightGBM `v4.7.0` is
[`8f7036f03627054d5a54a6f965b13f4b9ff2cb63`](https://github.com/lightgbm-org/LightGBM/tree/8f7036f03627054d5a54a6f965b13f4b9ff2cb63).
The existing Homebrew libomp 20.1.7 library was hashed read-only and remains
unchanged; its path/hash are in the receipt. Its presence is not proof that a
future baseline process loads that exact library. Preserve the method-specific
native-library qualification required by [the adapter brief](native-tree-qualification.md).
