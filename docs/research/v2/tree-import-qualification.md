# Native tree import and dependency qualification

On 18 October 2026, fresh method-specific processes successfully imported CatBoost
1.2.10, XGBoost 3.4.1 and LightGBM 4.7.0 from the exact offline-installed tree
overlay. None imported Torch. The complete installed dependency byte audit
passed, with no duplicate distributions and all 27 frozen core identities
unchanged. No model was constructed or fitted, and no prepared data or labels
were accessed. Native API/model fixtures and eligible anchors remain pending;
Gate D has not passed.

XGBoost and LightGBM each actually loaded two distinct LLVM OpenMP libraries.
Both reported one thread, and neither process crashed or emitted a warning.
This finding requires explicit disposition during the forthcoming bounded native
fixtures, interruption/recovery and resource checks. It is not a demonstrated
fitting defect or an automatic permanent blocker. No library was hidden, patched,
preloaded, replaced or uninstalled, and `KMP_DUPLICATE_LIB_OK` was removed from
caller-supplied child environments. Later adapter checks established that this
does not mean the flag remains absent: pinned `sklearn/__init__.py:56` and
`threadpoolctl.py:48` both use `os.environ.setdefault` to set it to `True`.
XGBoost/LightGBM import sklearn before inspection; CatBoost leaves it unset until
the threadpoolctl inspector imports. Thus no-warning/no-crash observations here
were made under upstream import behavior, and do not assure compatibility without
that suppression. No library or environment is patched to remove or hide it.

## Immutable receipts

All raw evidence is under the main checkout's
`artifacts/research-v2/WP07/tree-imports/`, an ignored artifact root:

- `index.json` identifies the retained initial audit failure, recovery suite and
  independent audit.
- `attempt-0/` retains the first launch, error and precise diagnosis.
- `recovery-1/environment-before.json` and `environment-after.json` contain all
  installed distribution identities and complete per-file overlay RECORD checks.
- `recovery-1/{catboost,xgboost,lightgbm}/` each retains its exact command, process
  log, import origins/versions, actual dyld images, library hashes, `otool -L`
  output, threadpool counts, import-body time and complete process wall time.
- `recovery-1/suite.json` retains the conservative raw `qualification_incomplete`
  flag for the multiple-runtime review; all three child exit codes are zero.
- `independent-audit.json` independently checks receipt/log hashes, unchanged
  distributions, Torch absence and the installed source cause of sklearn imports.
- `disposition.json` distinguishes the successful dependency/import result from
  pending native qualification and records that no permanent blocker was declared.
- `parent-offline-install.log` retains the parent's successful offline installation
  log without claiming this import task installed anything.

Key SHA256 identities:

```text
recovery-1/suite.json
  d0534403993eabff3a0fbb5a6b29d7793aad7accf407862432e49240c101126b
independent-audit.json
  8abe1d47716f81efed936e0abcc920648fdb130d2294971239b4232a5042cfe4
locks/research-v2/tree-imports.py
  9d2c645c5f3418959b6ed012b8a86a1084383eb012683e6be064da40e6a27ed2
```

The scratch and machine paths in these receipts are execution locators. Their
actual files/hashes and environment are retained explicitly; an unavailable old
locator is not itself a portable qualification receipt for a new environment.

## Installed identities and byte checks

The environment contained exactly 39 uniquely named distributions: the 27 frozen
core packages plus 12 overlay packages. Each overlay package came from the
[single-wheel lock](../../../locks/research-v2/tree-requirements.txt) and its
[verified acquisition manifest](../../../locks/research-v2/tree-resolution.json).

Both before and after imports, the auditor checked **2,921 installed RECORD
entries**, covering **205,272,152 bytes** of installed files. Of these entries,
2,909 carry declared SHA256/size values; the 12 RECORD files themselves have no
declared self-checksum and were independently hashed in the receipt. All declared
hashes/sizes matched, every path remained within the overlay, actual installed
METADATA matched the verified wheel's raw bytes, and no unowned file was present
apart from the installer's `.lock`. The overlay and core inventories were exactly
unchanged across imports.

Core checks compare exact installed name/version/path and the WP00 RECORD digest,
plus before/after raw metadata digests and module import origins. This is not a
new rehash of every file in every core distribution. The newly loaded relevant
native libraries were separately byte-hashed below. The core `uv.lock` and its
earlier runtime receipt were unchanged.

The first auditor incorrectly compared Graphviz's raw wheel METADATA digest with
`importlib.metadata.read_text()` output, which normalizes CRLF to LF. Its 12,306
installed bytes matched the wheel exactly; normalized text has 12,040 bytes
because of 266 CRLF sequences. The retained failing check was an audit-script
defect, not package corruption. Switching the digest input to the actual file
bytes resolved it. No package was reinstalled or altered. The corrected complete
audit and all three imports ran in a fresh recovery directory.

## Import results and actual libraries

All module origins are below
`/private/tmp/atn-research-v2-next/environment/research-v2-tree-site/`:

| Method | Version | Relative module origin | Torch imported | Complete process wall | Peak process RSS |
|---|---|---|---|---:|---:|
| CatBoost | 1.2.10 | `catboost/__init__.py` | No | 1.535 s | 102,940,672 bytes |
| XGBoost | 3.4.1 | `xgboost/__init__.py` | No | 3.753 s | 159,547,392 bytes |
| LightGBM | 4.7.0 | `lightgbm/__init__.py` | No | 1.317 s | 165,920,768 bytes |

These are single import observations while other work could contend for the
machine. They include the qualification process and native inspection; they are
not repeated cold-start benchmarks, fit forecasts or method-efficiency claims.
macOS `ru_maxrss` was recorded in bytes. Import-body durations are separately
retained in raw records.

Relevant native libraries actually loaded:

| Library | SHA256 |
|---|---|
| `catboost/_catboost.so` | 5b0fffd2f3fd9aa3f9bfbf3b1c7e28bd067c475dd8247adce5cfdc50be60b6ec |
| `xgboost/lib/libxgboost.dylib` | 5aef338ede7b84ae7d3405a306614de1d326ccaad033a2f14242aedd421359bc |
| `lightgbm/lib/lib_lightgbm.dylib` | bc392db609d97730a9ed2acec7a56529b356c03dfad39bebf689523fed7dff18 |
| Core sklearn `.dylibs/libomp.dylib` | 7fe04d06d824817d39e4c99ec44e837fd6b5f0f75f2cc3e77a384c298f3f1709 |
| Homebrew libomp 20.1.7 | 956f48d4d7abf8edc01ed0111694aa7df0fe00ca6824790188af95daaacb8502 |

CatBoost loaded no dynamic OpenMP image. XGBoost and LightGBM each loaded both:

```text
/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/lib/python3.12/site-packages/sklearn/.dylibs/libomp.dylib
/opt/homebrew/Cellar/libomp/20.1.7/lib/libomp.dylib
```

The former contains 678,720 bytes and the latter 740,896 bytes. Both identify as
LLVM OpenMP, both were visible to dyld and threadpoolctl, and both reported a
maximum thread count of one. This is not the documented Intel/LLVM runtime mix.
The primary threadpoolctl guidance advises caution even with distinct copies of
one implementation, while distinguishing the observed severe Intel/LLVM
incompatibility. No such incompatibility was established in these import-only
checks. See [the upstream runtime discussion](https://github.com/joblib/threadpoolctl/blob/master/multiple_openmp.md).

Installed `xgboost/compat.py` and `lightgbm/compat.py` both import available
scikit-learn compatibility classes during package import. LightGBM intentionally
imports sklearn first. Those ordinary native package imports load the core's
bundled runtime as well as the tree package's Homebrew-linked runtime; choosing
the native training API later does not by itself prevent this import path.
The exact source line locations and file hashes are retained in the audit. See
the pinned official [XGBoost compatibility source](https://github.com/dmlc/xgboost/blob/6fe8c547bdd21c73e4555d85b087d9260595d30d/python-package/xgboost/compat.py)
and [LightGBM compatibility source](https://github.com/lightgbm-org/LightGBM/blob/8f7036f03627054d5a54a6f965b13f4b9ff2cb63/python-package/lightgbm/compat.py).

The later [adapter transition checks](native-tree-adapters.md) retain actual
caller/native-import/runtime-inspection values and verify both source files
against their locked installed RECORDs. The unchanged core source SHA256 values
are `66a78bb27706626c4cc8ef424a5479f6301204baacd85986e8002c16b751fef1`
for sklearn initialization and
`e5f119e394193e206655bec4b993ef0bb40b0fbb2f22c92bd5b2d70a622bb518`
for threadpoolctl. Earlier raw import records are preserved; this is an explicit
qualification of their environment wording.

All dyld image paths are retained, with byte hashes for readable non-system
images. Apple shared-cache/system image paths are explicitly marked when their
on-disk bytes were not hashed. Relevant native/OpenMP libraries have complete
`otool -L` output; linked names were not substituted for actual loaded paths.

## Replay and next qualification

The import auditor is
[tree-imports.py](../../../locks/research-v2/tree-imports.py). It requires a fresh
output directory and the exact core-plus-overlay inventory. A replay uses the
qualified interpreter and overlay without importing the repository's model code:

```sh
ATN_PY="/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python"
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTHONPATH="$PWD/environment/research-v2-tree-site" \
"$ATN_PY" locks/research-v2/tree-imports.py \
  --overlay "$PWD/environment/research-v2-tree-site" \
  --output /private/tmp/atn-tree-imports-fresh
```

Each child gets its own artifact-local `MPLCONFIGDIR` and `XDG_CACHE_HOME`, an Agg
backend, disabled bytecode writes, one-thread numerical settings and no inherited
`KMP_DUPLICATE_LIB_OK`. Raw failed and successful attempts are retained rather
than overwritten. The raw auditor exits nonzero if imports fail or if it finds
multiple OpenMP paths requiring review; inspect that explicit reason rather than
calling every nonzero result an import failure.

Next work remains the bounded F/G-only native model/API/categorical/stopping/
save-load/recovery fixtures, disposition of the observed runtime combination,
then complete eligible anchor fits and resource qualification from
[the native recipe brief](native-tree-qualification.md). A new isolated runtime,
if needed, receives its own lock and receipt. These successful imports provide
no predictive comparison, convergence result or evidence about adaptive-neuron
quality.
