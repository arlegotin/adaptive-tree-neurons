# P07 TabM engineering qualification

Status: independently reviewed implementation and final tiny native parity
checks passed. This does not qualify a full native quality anchor,
practical workload or Gate D. No prepared transfer data, V/T labels, HPO,
foundation weights, paid services or shared environment changes were used.

The reviewed [contract](tabm-adapter-contract.md) fixes both official anchors and
the exact protocol adaptations. Branch research-v2-tabm started at
a56ae0897d62adb04eabbf11bc09dae24ca18e2f in the isolated managed37a4 worktree.
The root's running M1 source and earlier finalized ordinary-neural evidence were
left unchanged. Only the new TabM module/tests/docs/locks belong to this change.

## Pinned native procedure and dependencies

The adapter invokes official TabM0.0.3 at
[28e47ae301c92ec37787dde1ce923a0793f405b4](https://github.com/yandex-research/tabm/tree/28e47ae301c92ec37787dde1ce923a0793f405b4),
with numerical embeddings0.0.12 at
[7381c5a4ed01253fbef9ea8cfed137f907265552](https://github.com/yandex-research/rtdl-num-embeddings/tree/7381c5a4ed01253fbef9ea8cfed137f907265552).
Their wheel module bytes match those Git blobs exactly. Source/license/notebook
assets, hashes, complete dependency requirements and exact installed-file bytes
are in locks/research-v2/tabm. Both packages use their package Apache2.0 license.
The embeddings repository's different root license is not substituted.

Two universal wheels total30640bytes and expand to115966bytes before installer
metadata. They were installed offline, with no dependencies, into this worktree's
environment/research-v2-tabm-site. The pinned27-distribution core was unchanged.
The worker checks every core version/path/RECORD digest, every overlay installed
file/RECORD, metadata and actual imported module origins before native use.
It authenticates loaded Torch/OpenMP images against owning core RECORDs and
retains real threadpool counts. Core byte coverage is selected imported/native
files plus all RECORD digests, not a rehash of every installed core byte.

The installation command is reproducible with retained wheels:

```sh
uv pip install --python /absolute/qualified/python --no-index --no-deps \
  --require-hashes --find-links artifacts/research-v2/WP07/tabm-source \
  --target environment/research-v2-tabm-site \
  -r locks/research-v2/tabm/requirements.txt
```

The installed-files lock includes installer metadata and therefore binds this
specific verified installation. A fresh installation whose metadata differs
needs its own verified runtime identity; do not rewrite a retained receipt or
silently load an older snapshot under new source/runtime bytes.
Native workers must start with PYTHONDONTWRITEBYTECODE=1 or python -B. This
pre-import requirement protects the exact owned-file overlay audit; the adapter
refuses enabled bytecode writes instead of changing policy or allowing new files.

Plain native defaults are3blocks,width512,start scaling random-signs; the
piecewise anchor has2blocks,width512,start scaling normal,48 F-only bins,
embedding16,activationFalse,versionB. Both keep k32,AdamW.002/WD.0003,batch256,
gradient clipping1,no scheduler and native17-miss patience. Native epoch ceiling
is1000000000 with paid external deadline; it is not silently replaced by1000.
Engineering fixtures explicitly cap3epochs and remain quality-ineligible.

Training minimizes mean member MSE. G/query predictions inverse-transform each
float32 member before their float32 mean. Strict best original-unit G RMSE
selects the first best epoch; there is no update-zero selection or F-mean rescue.
Each epoch uses native torch.randperm/split, preserving all F rows. Source,
exact permutation hashes/arrays, selected/executed epochs, defaults, target
coordinates and complete G-only history are retained.

Native preprocessing is float32 F input plus absolute1e-5 float32 noise from
default_rng(0), with the author's quantile rule/random_stateNone. Actual float32
F-constant numeric columns are dropped with their mask/value/precision retained;
query variation cannot change the mask. Targets use native float32 F mean/std;
scale1 is allowed only for an actually constant float32 F vector. Underflow on
nonconstant targets or overflow/nonfinite statistics fails. Typed F-only K+1
category vocabularies reserve unknowns while retaining native random weights.
These are explicit protocol boundary adaptations. Low-distinct columns keep
native unique quantile edges; n_bins>=nF and no-numeric piecewise inputs fail
visibly, with no silent range/subsample/variant change. Categorical-only plain
inputs pass x_num=None as required by the official API.

## Executed verification and retained failures

Raw outputs are under artifacts/research-v2/WP07/tabm and the source/wheel
acquisition receipts under artifacts/research-v2/WP07/tabm-source. Every native
fixture ran in its own fresh process, sequentially with numerical threads1 and
artifact-local cache paths. F96/G23/query11 are fixed synthetic engineering
fixtures. Their mode uses2blocks,width64,k32,3epochs; it is distinct from full
native512-wide quality anchors. Direct parity calls the pinned official factory,
preprocessor, member loss, optimizer and selection operations with our explicit
F/G boundaries. It is not untouched whole-notebook reproduction; notebook test
evaluations are excluded.

Retained sequence:

- 01:29 expected RED failures for the missing adapter.
- 02:28pass/1test-schema mismatch. The bins-boundary test had declared a
  categorical feature but passed no category cardinality; its fixture was fixed.
- 03:29pass/5expected RED checks for missing native fit/snapshot support.
- 04:30pass/4native parity failures with at most4.77e-7 prediction discrepancy.
  This was investigated without relaxing tolerance or changing seeds.
- 05:an independent no-training forward proved the actual native head emits
  column-major member tensors. NumPy concatenate preserved that layout, whereas
  the notebook's torch.cat creates contiguous member rows. Equal member values
  then had a4.77e-7 float32 mean difference due to reduction order.
- 06:a focused RED stride regression preserved that cause. The adapter now uses
  exact torch.cat→cpu→numpy before inverse/member averaging.
- 07:35pass in83.51s; all four plain/piecewise × numeric/category fixtures had
  exact predictions, histories, row orders and reloads, plus13mutations rejected.
- 08:7pass/1RED categorical-only native input boundary. Native expects None,
  not an empty numeric tensor; training and query calls were corrected.
- 09:a no-tests launch error named nonexistent test_baselines.py. It is retained;
  no fit ran. The correct existing file is test_simple_baselines.py.
- 10:105pass/1toy-helper failure in84.91s. Four official fixtures, reloads,
 14mutation rejections and real categorical-only native forwards passed. A toy
  Parameter-view output retained requires_grad under no_grad, exposing that
  concatenation had been outside the evaluation context. The entire
  concatenation/NumPy conversion now stays inside no_grad, matching the notebook.
- 11:102contract/simple-baseline/config checks pass in2.09s after that correction.
- 12:independent review found that bytecode disabling had only been a test-worker
  convention. A no-fit RED worker trapped the attempted Torch import before any
  overlay mutation. The adapter now refuses bytecode-enabled workers before
  importing native packages. All previous executed workers already disabled it.

The independent reviewer found no second material issue after the remaining
static/lock checks; the contract's opening status was also marked historical.
Root accepted the single RED→GREEN correction pass without a second review.
Run13-reviewed-final passed107checks in84.38s:40TabM contract,4native parity,
and63existing simple-baseline/config checks. All four final-source fixtures had
exact direct-reference predictions, complete history/row-order agreement,
exact reloads and14rejections each. No warnings were reported by pytest.

The executed module SHA256 is
450476815517e2d483c24c2986a1ee660765d3e838f4117917494ec57e8da2f0;
the complete scientific source/package SHA256 is
2dce0892d85f584aa9dc5e27bfcd0b57c2ac29889dc763b4b87853cfdc04ffaf.
Final module/test/reference byte copies, per-worker argv/environment/cost,
source/dependencies, raw safe fixture arrays/features, full fit/snapshot/mutation
receipts and JUnit/log outputs are in13-reviewed-final. Its execution-summary.json
binds the final script/module hashes. The raw evidence inventory at
artifacts/research-v2/WP07/tabm/evidence-inventory-final.json covers both tabm
and tabm-source roots; pytest convenience symlinks are excluded from counts.
Integration into another source tree creates a different whole-package identity;
these executed receipts retain this identity rather than being relabeled.

The four native fixtures'14rejections cover trusted digest, source, LR edit,
engineering→search promotion, dtype/weight edits, preprocessing/target/bin edits,
negative G risk, optimizer defaults, recorded runtime changes, actual runtime
authentication failure, and a version-matching module stub selected from outside
the authenticated overlay. Rehashed copies test internal semantic binding;
production callers still must supply an independent trusted manifest SHA256.
Original snapshots remain unchanged. Saved NPZ tensors are read without pickle,
with strict dtype/name/shape/finiteness and constructor-buffer checks. The
prediction-state digest binds weights, transforms, bins and selected epoch.

Import/runtime observations retain two distinct LLVM OpenMP images:
Torch libomp SHA2566256bee09e93c28d71c65711cc69224d69994c6965648b628b70a22772fe98d4,
sklearn libomp SHA2567fe04d06d824817d39e4c99ec44e837fd6b5f0f75f2cc3e77a384c298f3f1709.
Both reported one thread. Caller KMP_DUPLICATE_LIB_OK was absent; pinned upstream
sklearn/threadpoolctl set it toTrue. The adapter never patches, preloads, sets or
removes that policy. Successful tiny execution under upstream suppression is
not an assurance of unsuppressed compatibility. Full paths, actual native images
and owning RECORD identities are in the per-fit receipts.

The observed tiny fits took about3.2s including imports/authentication and about
.5s native training, with process-lifetime peak RSS roughly423–428MiB in run07.
These are engineering observations while other work used the machine, not a
full-anchor forecast or fair quality–cost benchmark. Repeated snapshot/runtime
corruption checks dominate the83s test-suite wall time and are separately paid.
Failure injection verifies domain/phase, original exception, consumed time and
atomic attempt receipt for input/API, import/I/O, numerical, memory and interrupt
cases. No hidden retry or replacement optimizer seed occurs.

## Usage and remaining Gate D work

TabMPredictor implements FitRequest/Predictor using only F/G labels and
feature-only predict(X). Versioned config fixes familyP07, variantplain/piecewise,
modenative_anchor/search/engineering_fixture, explicit feature schema and bounded
parameters. Native anchors take no overrides. The frozen search support is
validated separately from architecture defaults; unknown settings fail. There
is no registry/HPO change to legacy procedures in this slice.

save writes portable JSON+NPZ prediction state; load requires an external trusted
manifest SHA256 and the same qualified source/dependencies/actual native images.
Full optimizer/sampler training interruption recovery is not implemented.
max_seconds is an observed post-fit budget here, never hard deadline enforcement.
Production needs a separately qualified child resource harness. Source changes
create a new digest and cannot silently consume old snapshots as new evidence.

Pending: full native plain+embedded anchor eligibility; actual
transformed transfer workload/RAM/time qualification; common budget/deadline/RSS
and interrupted-training recovery; primary-only F/G/V search integration and
remaining mandatory families. The bounded32-attempt P07family must charge plain
then embedded anchors as its first two attempts, not give each a separate budget.
No predictive/placement/practical superiority claim follows from these checks.
The repository's full test suite was not run: root explicitly limited this
parallel task to tiny fixtures and scoped regression checks alongside M1.
