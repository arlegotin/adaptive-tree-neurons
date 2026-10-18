# Bounded native tree fixture qualification

This WP07 engineering exercise uses the official pinned CatBoost 1.2.10,
XGBoost 3.4.1 and LightGBM 4.7.0 packages directly. It does not implement or qualify
the production adapters, run the native full-quality anchors, compare predictive
methods, or inspect prepared transfer data. Gate D remains pending.

The finite plan is eight cells: four native families (CatBoost, XGBoost histogram,
LightGBM constant leaves, LightGBM linear leaves), each with numeric and mixed
categorical inputs. Every cell gets a fresh one-thread process and two fits,
ordinary G selection and an intentionally opposed-signal G stopping stress.
Thus the plan has 16 fits, maximum 128 rounds/trees each and patience eight.
The reversed G task is an engineering control that forces the stopping and
selected-model path to be exercised; its losses are not generalization evidence.

The predeclared shared synthetic fixture has 192 F and 64 G rows. Inputs are three
independent Uniform[-1,1] numerical variables and balanced/shuffled categories
`a/b/c`. Every seventh G category becomes an unseen token (10 rows). Targets are
`2+1.5*x1-.75*x2+.4*x3+1.25*relu(x1-.25)+.65*x1*x2+category_effect+noise`, with
effects `a=-1,b=.5,c=1.75,unseen=.2` and Gaussian noise sigma .1. The numeric variant
omits the category predictor but retains the same labels. Stopping-stress G
labels are `4-clean_G+same_noise_G`. Observation and native-training seeds derive
from root 2026101802 and labels `WP07-tree-fixture/synthetic-v1/observations` or
`native-training`, using the contract's canonical SHA256 seed function. Methods
share their declared data and training seed; no seed is selected for an outcome.

Before any fitting, the script writes an immutable manifest with every case,
resolved seed, parameter dictionary, executable/dependency identities, raw F/G
array hashes, retry/resource policy and maximum work. Generation-only noiseless
targets/noise are excluded from the fit-worker archive. There is no V/T role.

F-only category vocabularies are tested explicitly. CatBoost receives strings
and maps unknowns to `__ATN_UNKNOWN__`. XGBoost and LightGBM receive native
categorical columns with the F vocabulary and explicit missing unknown values.
Only LightGBM linear-leaf continuous inputs are standardized, using F means and
ddof=0 standard deviations. Query distribution/category changes must leave those
F transforms unchanged. The helper self-check first failed on absent fixture
implementation, then passed; a pandas future-warning prompted explicit
unknown-to-missing conversion before native fits were admitted.

The tiny configurations deliberately differ from full native quality anchors:
CatBoost depth4/LR.1/MVS.8/L2=3; XGBoost depth3/eta.1/hist/max_bin64;
LightGBM leaves7/min_leaf8/LR.1/max_bin31/L2=.01/linear_lambda=.01. Native squared
error objectives, native G stopping and best-state selection remain intact.
Their complete settings are in the manifest and individual native receipts.

Acceptance checks cover all passed F/G row counts and target arrays, finite
original-unit prediction shapes, native selected/executed rounds, selected G
loss recomputation, exact save/load predictions, and category metadata after
loading. The reversed G control must stop below the cap and retain an earlier
model. XGBoost uses a fresh best-saving callback and a G QuantileDMatrix referencing
F. LightGBM receives `linear_tree` at Dataset construction; the ordinary linear
fit must show actual nonzero leaf coefficients after its constant first tree,
and categorical indices must never enter those leaf regressions. Gradients are
not an applicable check for these native tree procedures.

Admission requires at least 256 MiB free disk and a 512 MiB free+inactive RAM
proxy from `vm_stat`. Workers run sequentially, with a two-GiB RSS guard and
90-second cell ceiling. One identical seed/config infrastructure retry is
allowed; numerical/assertion/API failures are retained without automatic retry
or changed seeds. All failures and costs remain visible. Matplotlib/cache writes
go to artifact-local directories, bytecode writes are disabled, and OpenMP
conflict suppression/preloading is not used.

The executable is [qualify_native_trees.py](scripts/qualify_native_trees.py). It
reuses the read-only import auditor to record method-specific versions/origins,
actual loaded native/OpenMP paths/hashes, `otool -L` dependencies, thread counts
and Torch import status before/after fitting. Exact argv, environment overrides,
imports/startup/body timings, process RSS, native model files and predictions
are retained. The 27 core and 12 overlay identities are checked before/after
the complete fixture suite.

## Executed result

All eight fixed cells and all **16 tiny native fits passed**. The successful
recovery used the same executable, arrays, cases, parameters and seeds as the
original admitted plan. The observation seed was `3121097753693322812`; the native
training seed was `735836459`. No outcome-triggered seed/configuration changes
or further fits were made.

| Native family | Inputs | Ordinary retained / executed rounds | Stopping-stress retained / executed | Nonzero affine leaves in ordinary fit |
|---|---|---:|---:|---:|
| CatBoost | Numeric | 30 / 38 | 1 / 9 | Not applicable |
| CatBoost | Categorical | 81 / 89 | 1 / 9 | Not applicable |
| XGBoost | Numeric | 39 / 47 | 1 / 9 | Not applicable |
| XGBoost | Categorical | 123 / 128 | 1 / 9 | Not applicable |
| LightGBM constant | Numeric | 15 / 23 | 1 / 9 | 0 |
| LightGBM constant | Categorical | 97 / 105 | 1 / 9 | 0 |
| LightGBM linear | Numeric | 16 / 24 | 1 / 9 | 105 |
| LightGBM linear | Categorical | 62 / 70 | 1 / 9 | 427 |

Every saved model produced exactly equal G predictions after native save/load.
Every cell passed all 192 F and 64 G input rows to the native dataset/Pool and
checked the returned target arrays; CatBoost's declared native MVS sampling
remains an internal training choice. All F/G prediction shapes were correct and
finite. All categorical cells handled the 10 G tokens absent from the F
vocabulary. Both linear-leaf fits had a constant first tree, actual nonzero
later leaf coefficients, and no categorical index in the linear regressions.

The independent audit reconstructed all 16 G RMSE values from saved row-aligned
predictions and synthetic labels. Maximum disagreement with the reported applied
metric was `2.22e-16`; maximum disagreement with the native selected G metric was
`1.44e-8`, inside the prospectively fixed relative/absolute-floor `1e-6` tolerance.
That distinction retains the native float32 label/arithmetic boundary rather
than claiming byte equality of two different metric implementations.

XGBoost's ordinary categorical fit reached its 128-round fixture cap and retained
round123. It is not evidence of convergence. All opposed-signal controls selected
the first tree and stopped after nine executed rounds, so the selected-versus-
executed model distinction was actually exercised rather than inferred from
configuration strings.

The two LLVM OpenMP copies reported by the earlier import audit remained visible
after XGBoost/LightGBM fitting, each at one thread. No warning, crash, nonfinite
prediction or save/load discrepancy occurred in these bounded cases. CatBoost
loaded no dynamic OpenMP image, and no worker imported Torch. These observations
support this bounded engineering qualification; they do not establish arbitrary
long-run, multi-thread, interruption or large-workload compatibility. The full
native anchors and recovery/resource qualification remain required.

The largest recorded worker peak RSS was **187,039,744 bytes**, below the two-GiB
guard. The successful coordinator launch took 10.158275 seconds; including the
failed infrastructure launch, recorded launch time was **10.783437 seconds**.
These single contended engineering timings include import/native inspection,
serialization, monitoring and fitting. They are not method-efficiency evidence.
All before/after installed package identities matched, including all 27 core
distributions and the 12 exact overlay distributions.

## Failure, recovery and evidence

The first coordinator launch was interrupted when the sandbox denied `/bin/ps`,
which the RSS monitor required immediately after child launch. No completed
native fit/result was recorded, and a subsequent process inspection found no
remaining worker. Review also established a coordinator lifecycle defect: that
exception path lacked `finally` cleanup and a per-child command receipt. The
later absence of a worker does not make that lifecycle safe. The first
source/manifest, guard, process log, launch timings and infrastructure
disposition remain retained. The single allowed identical
infrastructure retry ran with normal authorized process-inspection access in a
fresh output directory. The guard was preserved, and no library or environment
was patched/preloaded to hide the earlier dual-runtime finding.

Evidence is under the main checkout's ignored directory
`artifacts/research-v2/WP07/tree-fixtures/`:

- `index.json` records both launches, the one infrastructure retry and exact scope.
- The root `manifest.json`, `source.py`, first cell directory and
  `launch-0-disposition.json` retain the interrupted original attempt.
- `recovery-1/manifest.json` and `source.py` bind the successful fixed execution.
- `recovery-1/case-*/attempt-0/` retains resource admission, exact command/log,
  encoding, two native model/results/prediction sets and before/after native
  runtime records.
- `independent-audit.json` reconstructs metrics and verifies hashes, role arrays,
  all stopping-stress outcomes and actual affine-leaf contents independently.
- `self-checks/` retains the initial missing-implementation failure, intermediate
  pandas warning and final clean helper checks.
- `lifecycle-repair/` retains the review-driven failure reproductions, repaired
  helper and verification source snapshots, harmless-child test results,
  dependency/artifact rechecks and append-only audit-context clarification.

Key SHA256 identities:

```text
executed fixture script
  150106235122286a7cf39d0194c5c35de5a424b7d3390a48286f94ba592df450
recovery-1/manifest.json
  df85cf76ec093b225196fd37b47106454d3c877469d3e9bf5cf730b308ef9a7f
recovery-1/suite.json
  ed459c7eaac05ddb9c7b4d9e25b562401fa406145ff54a48a1afc9474670e354
independent-audit.json
  d0addda7a9ff821cc41bde9f6212b90ac8691295645c9743099307819dcdfddc
hardened helper (not used for the 16 completed native fits)
  fea9c465bf2e25a6a92922a6b46d29b01b54430834bedf2d2f422dfab4525a3f
lifecycle verification helper
  6151a87cc77ebcf0755c3c2e1d6f13705bc2c08df40f19a1ae1935cf85a38670
```

## Lifecycle and audit-context correction

The coordinator now places every launched child under `finally` cleanup:
terminate, wait, then kill and wait if necessary. It publishes the child command,
terminal status, error and elapsed cost before re-raising the original monitor
exception or interrupt. It records a cleanup failure without replacing the
original exception. Monitoring failure does not silently launch a replacement
child.

Test-first reproductions used harmless sleeping Python children and a mocked
RSS monitor. Both `PermissionError` and `KeyboardInterrupt` initially left the
child unreaped and lacked the required receipts. After the correction, **nine
checks pass**: both failures preserve the exact exception object, reap the child
before fallback test cleanup, and publish error/command/cost records; six changed
hash bindings reject before dynamic auditor import; and the after-fit audit
wrapper reports its actual process phase. These checks execute no native fits.
The separate test cleanup protects even an intentionally failing reproduction;
it is not what makes the repaired supervisor assertions pass.

Workers now check the manifested helper, interpreter binary/path, import auditor,
tree requirements, core lock and fixture hashes before dynamically importing the
auditor or loading observations. Future manifests explicitly bind the executable
binary and path; the original manifest did not bind the executable binary hash.
The original manifested auditor, tree-lock, core-lock and fixture hashes all
still match. The nine checks are reproducible in a fresh output directory:

```sh
ATN_PY="/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$ATN_PY" \
  docs/research/v2/scripts/check_tree_fixture_lifecycle.py \
  --output /private/tmp/atn-tree-fixture-lifecycle-fresh
```

The reused import-only auditor writes `models_constructed=false` and
`fits_executed=false` to describe its own call. In the old `after-fitting` records
those fields do **not** describe prior fixture work in the same process: two fits
had completed. All raw records are preserved unchanged. The append-only
`lifecycle-repair/legacy-runtime-context-disposition.json` binds all 16 before/
after record hashes to their phase and true completed-fit count. The future
wrapper retains the raw auditor call separately and publishes phase-aware
process flags, completed-fit count and raw-record digest.

The repair audit rechecked all **160 original artifacts / 6,665,179 bytes** against
the prior inventory, all original dependency bindings, all 27 core and 12 overlay
identities, and their installed RECORDs. The fixture generation, seed function,
encoding, parameters and native fit/save/load functions have identical normalized
Python ASTs to the original executed snapshot. Thus the 16 numerical results
remain evidence under the original `150106...` helper identity; **no native fits
were rerun**. The hardened helper has the separate `fea9c4...` identity above.
The full repair evidence and reasoned no-refit disposition are in
`lifecycle-repair/disposition.json`. The old inventory and all original receipts
remain unchanged.

Reproduction uses a fresh output directory, the qualified core interpreter and
the exact offline-installed overlay. Process-inspection access must be available
for the RSS guard; use the normal authorized execution path if a sandbox denies
it. The script never weakens that guard as a fallback. The following commands use
the hardened helper and therefore create a fresh execution identity; the saved
original `source.py` records the helper used for the completed results:

```sh
ATN_PY="/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$ATN_PY" \
  docs/research/v2/scripts/qualify_native_trees.py --self-check
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTHONPATH="$PWD/environment/research-v2-tree-site" \
"$ATN_PY" docs/research/v2/scripts/qualify_native_trees.py \
  --overlay "$PWD/environment/research-v2-tree-site" \
  --output /private/tmp/atn-native-tree-fixtures-fresh
```

This completes the requested bounded direct-native fixture exercise. Production
adapter parity, native full-quality anchor eligibility, interrupted native
training recovery, wider workload envelopes and practical comparisons are still
pending. No result here changes the evidence about informed neuron placement.
