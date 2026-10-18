# Ordinary MLP-D and ResNet-D native recipe

This scoped WP07 implementation starts from
`5ab3ae2f37f99ebbbdf1620599e18b293002b78d` on
`research-v2-ordinary-neural`, in the managed isolated worktree
`/Users/artemlegotin/.codex-s/worktrees/75a0/adaptive-tree-neurons`.
It implements P03/P04 without editing shared source, existing dependency locks or
legacy guards. Only tiny synthetic F/G fixtures are admitted. Full quality anchors,
HPO, prepared transfer data, V/T labels and performance claims remain outside scope.

## Resolved primary sources

- Better by Default, arXiv2407.04491v3, Appendix Tables C.4/C.6:
  <https://arxiv.org/html/2407.04491v3>. These are configurations adapted from
  McElfresh et al., not universal RTDL defaults.
- Original RTDL revisiting source, commit
  `e3ed46cac38568785289d8fa16b8cfa585bde27e`:
  <https://github.com/yandex-research/rtdl-revisiting-models/tree/e3ed46cac38568785289d8fa16b8cfa585bde27e>.
- Current author PyTabKit source, commit
  `c126ea51187c5080b91f28d352481dbd3b2194b0`:
  <https://github.com/dholzmueller/pytabkit/tree/c126ea51187c5080b91f28d352481dbd3b2194b0>.
  `models/sklearn/default_params.py` supplies the TabZilla-derived defaults;
  `models/nn_models/rtdl_resnet.py` supplies the actual architectures;
  `models/utils.py:TabrQuantileTransformer` supplies noisy quantile preprocessing.
  Exact source bytes, licenses, SHA256s and extraction coverage are retained
  under a new `locks/research-v2/ordinary-neural/` source lock.

The official tables and current author architecture settings agree: MLP hidden
widths[128,256,128], ReLU/dropout.1; ResNet two width128 preactivation residual
blocks, expansion2, BatchNorm/ReLU, hidden dropout.25/residual dropout.1 and final
normalization/activation/head. Both use categorical embedding8, PyTorch native
initialization, AdamW LR.001/WD.01, batch128, at most1000 epochs, G each epoch,
patience20/best-G restoration and no scheduler.

## Explicit source differences and decisions

Use the pinned **author PyTabKit TabrQuantileTransformer** numerical procedure,
the implementation selected by its MLP-D/ResNet-D `quantile_tabr` defaults. Fit
only F numerical features, after an explicit C-contiguous float32 input conversion. Preserve
`n_quantiles=max(min(nF//30,1000),10)`, normal output, integer subsample1000000000,
noise.001 divided by `maximum(F_std,noise)`, `default_rng(0)` and its float64 noise
promotion. Transform original unnoised F/G/query rows using that retained fit.
Quantile preprocessing randomness is the author's fixed0, separate from the
declared model/sampler seed. Do not change inverse-std noise to std-scaled noise.

Original RTDL uses `subsample=1e9` as a float, rejected by the current sklearn
integer validator, and adds noise in-place, changing float32 rounding. We are
not silently mixing that legacy path with the selected current author path.
The selected author implementation already uses an integer subsample, so it
requires no float-to-integer compatibility patch.

The current author Skorch interface fits target StandardScaler before splitting
F/G. That conflicts with the binding F-only protocol. This adapter explicitly
fits target mean/population scale on **F only**, using float64 target statistics
and float32 training targets, preserving original-unit inverse
prediction. This is a disclosed boundary adaptation, not untouched end-to-end
author-wrapper reproduction. Model architecture/preprocessing parity means the
pinned official definitions invoked with this explicit F/G boundary.

Category dictionaries/cardinalities fit F only. Reserve one additional unseen
index per feature; the author architecture's final category embedding starts at
zero. Numerical and categorical query changes cannot alter preprocessing state.
All actual dtype transitions, model/sampler RNG and F statistics are retained.

The author Skorch1.1.0 callback distinguishes relative1e-4 progress for patience
from strict G-best checkpoint selection. Both rules are preserved. The F-mean
predictor is a retained G-selected safeguard when every finite neural checkpoint
is worse. Nonfinite training/gradients/parameters fail; they are never rescued as
a completed mean model. Skorch source is pinned at
`6008085ce0a8118828b7559c40c51189b70cb28f`; its BSD license is retained.

Training uses the native PyTorch shuffled DataLoader with the declared model/
sampler seed, full F coverage each epoch, batch128/256, zero loader workers and
no dropped rows. Global Torch RNG is isolated and restored outside each fit.
G evaluation uses its own unshuffled DataLoader and consumes the native loader
RNG stream. G losses are row-weighted native squared loss in F-standardized
target coordinates. Selection occurs after epochs1 onward. There is no
update-zero checkpoint. A ResNet workload whose last training batch has one row
fails explicitly because native BatchNorm cannot train it; changing this policy
requires new qualification.

## Proposed implementation contract

The new lazy `research/ordinary_neural.py` adapter uses existing FitRequest and
Predictor interfaces, strict versioned family/mode/feature schema/config, finite
P03/P04 supports and explicit engineering limits. Torch imports only when needed.
Reference fixtures invoke hash-pinned official architecture and transformer
definitions independently, without running the author's top-level benchmark or
accessing test-per-epoch paths. No extra packages are installed into the core.

It saves native NPZ tensor state plus portable JSON preprocessing/recipe/role/update
metadata; require an external trusted state digest when loading. Check exact
source/core identity, requested versus resolved architecture/optimizer settings,
dtype/shape/finite buffers, selected/executed epochs and mode/eligibility. Preserve
failure domain/phase, interruption and paid costs. A bounded fixture is never a
quality-eligible native anchor. Hard process budgets and verified interrupted
training recovery remain separately qualified harness work.

`native_anchor` fixes the full published1000-epoch/patience20 budget. `search`
accepts only the finite P03/P04 supports from the v2 spec, including the injected
nonuniform anchor and historical one-layer coordinate. GELU is a declared P03
search variant; exact author architecture parity applies to ReLU anchors.
Generic search defaults are three width256 MLP layers or ResNet hidden dropout.1.
The exceptional nonuniform MLP dimensions and ResNet hidden dropout.25 are
admitted only when the complete resolved architecture/optimizer recipe equals
the published anchor; they cannot create extra HPO cross-combinations. A fixture
with no overrides uses the published anchor with its explicit small epoch budget.
`engineering_fixture` permits at most8 epochs/patience4 and always reports
`completed_ineligible`/`checkpoint_eligible=False` unless an operational failure
overrides that status. Every returned model currently has `quality_eligible=False`
and `native_full_anchor_qualified=False`: neither an adapter nor a tiny fixture
supplies Gate D. A full native fit may have a valid G checkpoint before its
external workload/quality qualification exists.

The finite static safety envelope is F10–20000, G1–20000, at most128 raw
features,100000 total category slots,10M parameters, CPU float32. It is an input
guard, not a measured claim of full-workload feasibility. The synchronous adapter
records a `max_seconds` overrun after fitting; hard deadlines/RSS/concurrency,
fresh-process startup and interrupted-training recovery need the outer worker
harness. Its containing fresh-process timer must include dependency/source
validation, preprocessing, fit, prediction and publication. Internal fit timing
ends before the terminal receipt; snapshot serialization and inference are
separately charged by that harness.

Loading requires an externally trusted manifest SHA256 before opening native
payloads. The loader checks source/core identity; mode/config/resolved optimizer
and stopping history; F-transform/target-transform/selected-tensor bindings; exact
tensor names/shapes/dtypes/finiteness; BatchNorm variance and selected-epoch age.
No executable model pickle is loaded. The files are portable prediction state;
cross-runtime/source migration and exact training resume are not claimed.

## Execution plan and current state

1. Finish source acquisition/hash/license lock and run a small baseline check.
2. Write failing F-only transform/category/config and lazy-import tests; implement
   those boundaries. Write failing snapshot/optimizer/stopping tests first.
3. Run tiny numeric/categorical official architecture/preprocessor/gradient/update
   parity and save/load fixtures sequentially on one thread. Retain failures and
   immutable source/config/data/cost receipts.
4. Request independent review, fix material findings with RED→GREEN evidence,
   update this document/receipt, and commit only owned new files. Preserve the
   worktree and ignored evidence for integration. No execution.md edits.

The first22 boundary checks went RED→GREEN. Exact author preprocessing then
caught a real float32 memory-layout reduction difference: retaining the Pandas
column-extraction layout changed noisy quantiles by at most4.57e-8 in the fixed
fixture. Explicit C-contiguous native numeric input restored exact equality;
the test tolerance was not relaxed.

The first complete pass had35 checks green, including four fresh processes
(P03/P04 × numeric/categorical), three epochs on F160/G33, exact direct-author
forward/gradient/update/training-history/prediction parity and exact reloads.
Additional copied-checkpoint tests exposed accepted target/feature-transform,
same-dtype weight and optimizer-default edits. The retained RED evidence led to
explicit fitted-state bindings. The resulting42-check pass was green in64.13s.
Three final source/resource boundary tests then went RED→GREEN;40 related
boundary/update checks passed in1.84s. The final complete pass/review and immutable
receipt are recorded below. Initial evidence is never overwritten.

Evidence is under ignored
`artifacts/research-v2/WP07/ordinary-neural/` in this worktree. `06-executed-source`
and `09-executed-source` retain the exact earlier executed module/test bytes;
their snapshots remain tied to those source identities. Later validation must
not silently load them as new-source studies.

Every native fixture observed both Torch's and sklearn's LLVM OpenMP libraries.
Caller KMP_DUPLICATE_LIB_OK was unset; pinned sklearn set it to`True` on import.
Actual paths/hashes, import transitions and one-thread runtime counts are retained.
No library was patched/preloaded and the adapter did not set/remove that flag.
No observed crash or warning under this upstream policy proves compatibility
without suppression. The full workload/resource qualification remains pending.

## Independent review and final verification

Independent read-only review found three material P2 issues in the pre-correction
module (`b6986d9191fa6b9ecdd76134d3b9f7379323942f018275e96c482f6c27276e98`):
exceptional anchor coordinates expanded generic HPO support; snapshot loading did
not reauthenticate the actual linked native images; and fallback metadata could
accept negative squared risks or change the selected prediction scalar.
No-fit copies reproduced every issue. Those probes supplied a recomputed trusted
digest for the copied manifest: they exposed internal consistency gaps, not a
bypass of the original external trusted digest. The originals remain unchanged.

The correction enforces the complete-anchor/generic-support union, authenticates
current native image origins/RECORD bytes/thread counts on load and compares the
stable retained identity, rejects invalid G/fallback risks and nonfinite fallback
outputs, and binds the chosen predictor including its mean scalar. Seven cheap
support/stopping/risk tests went RED→GREEN; five no-fit snapshot/runtime probes
retained their accepted-corruption results before the fix. Final fresh fixtures
reject all15 named mutation/authentication cases, including positive finite
fallback replacement and an explicit runtime-authentication failure.

A separate source check found an exact native callback boundary: for first loss
`0x1.00de87540a5e7p-1`, multiplication by`1-1e-4` gives threshold
`0x1.00d7f3e8f2824p-1`, one ULP above the author's subtraction result
`0x1.00d7f3e8f2823p-1`. The tracker now uses the exact pinned subtraction
arithmetic. A source-extracted Skorch method supplies the independent test
reference. The boundary test preserves strict checkpoint improvement while
correctly recording a patience miss. Earlier three-epoch fixture results do not
establish that this boundary occurred in those runs.

The affected final suite passed **115 checks in88.71s**:52 new neural checks and63
related simple-baseline/config regressions. All four fresh numeric/categorical
P03/P04 cases matched direct pinned-author predictions and G histories exactly,
and their reloaded predictions were exact. Each fit used F160/G33 and at most
three epochs. The complete repository suite was not run under this explicitly
scoped tiny-fixture task; no whole-suite claim is made. Root accepted one reviewed
correction pass with these tests; a second review was not required.

The machine-readable [qualification receipt](ordinary-neural-adapter-receipt.json)
binds exact source, source-lock, case and log hashes. Final immutable evidence is
`artifacts/research-v2/WP07/ordinary-neural/16-cases` and `16-executed-source`.
Use a fresh output directory to repeat the same bounded checks:

```sh
ATN_PY=/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$ATN_PY" -m pytest -q tests/test_ordinary_neural_contract.py tests/test_ordinary_neural_training.py tests/test_ordinary_neural_native.py tests/test_simple_baselines.py tests/test_config.py --basetemp=artifacts/research-v2/WP07/ordinary-neural/fresh-cases
```

The existing qualified interpreter is required; no dependency was installed or
changed. Fresh native subprocesses have60-second engineering timeouts, artifact
local cache directories and a256MiB disk transaction guard. The supervisor retains
argv/environment/error/elapsed receipts and uses subprocess.run's child reaping.
This is still not a production16GiB/120-minute resource qualification. Gate D,
full native anchors, complete worker budgeting/recovery and practical HPO remain
pending. Source changes on integration require new source-bound receipts; old
snapshots are not silently migrated.

The pinned author `pytabkit-rtdl_resnet.py.txt` retains seven upstream trailing
whitespace lines so its Git blob identity remains exact. The authored module,
tests and documents pass the scoped staged whitespace check. Raw-source fidelity
takes precedence over rewriting those retained upstream bytes for formatting.
