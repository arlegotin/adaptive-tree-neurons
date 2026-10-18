# Native tree adapter implementation

This scoped WP07 continuation implements P08 CatBoost, P09 XGBoost histogram,
P10 LightGBM constant leaves and P11 LightGBM linear leaves. The approved research
design and [native qualification brief](native-tree-qualification.md) define the
recipes. This document records implementation decisions, checks and remaining
qualification; it does not announce benchmark results.

The implementation is isolated on `research-v2-native-adapters`, starting at
`419855e1ee88b98a4736ef01d903df582f9b7b31`, in
`/private/tmp/atn-native-adapters-next`. The existing qualified interpreter and
read-only optional tree overlay are reused. No legacy registry/adapter/guard,
shared worktree, core dependency or prepared transfer partition is changed.

## Contract and decisions

`src/treeneuron/research/native_trees.py` supplies `NativeTreePredictor(family)`,
using the existing `FitRequest`/`Predictor` interface. Its strict v2 config contains
family, execution mode, explicit ordered feature schema/category names and native
parameters. The adapter receives F/G labels only; prediction receives features
only. It does not search, choose seeds, refit on other roles or read a data store.

F fits numerical scaling for linear LightGBM only and every categorical
vocabulary. Raw category values have deterministic type-tagged tokens; unknown
queries map to a reserved CatBoost string or explicit missing native category in
XGBoost/LightGBM. Internal feature names are stable `f0`, `f1`, ... so arbitrary
semantic column names cannot violate native feature-name rules. Nonfinite
numerical values and missing categories are rejected in this initial finite
no-missing-data workload. Targets stay in original units. Query rows never change
F statistics or vocabulary.

Modes distinguish native anchors/search from engineering fixtures. Anchor limits
are CatBoost 10,000/200 patience, XGBoost 10,000/100, LightGBM constant 10,000/100
and linear 5,000/100. Bounded fixtures use at most128 rounds/patience8 and remain
quality-ineligible. Search accepts only the design's finite native supports;
unknown settings and native budget overrides fail. CatBoost retains native
automatic anchor LR and records all resolved parameters. Its fixed CPU regression
bootstrap policy is MVS/subsample.8/random-strength1; HPO must retain and qualify
that policy before freeze. XGBoost uses F QuantileDMatrix and F-referenced G,
with a fresh best-saving G callback. LightGBM constructs fresh datasets with
`linear_tree` present at construction, uses F-referenced G and retains best G.

Each fit binds source/package, core/tree locks, exact installed dependencies and
native module/library bytes; imports remain method-specific and lazy. Immutable
native model plus strict JSON snapshots retain F transforms, config/seed,
dependency/source identity, selected/executed rounds, G history and model hash.
Loading requires the trusted raw `state.json` SHA256 retained by the coordinator,
then checks source/dependency bindings and native recipe/stopping semantics before
native model loading. The internal JSON self-hash alone does not authenticate a
rehashed edit. Actual imported module/native-library origins and installed RECORD
bytes must belong to the locked package, rather than merely having the right
version string. A saved model supplies exact same-runtime predictions; no
executable pickle is used.

Import transition checks reject caller-supplied `KMP_DUPLICATE_LIB_OK`, retain its
actual value after the native import and after runtime inspection, and verify
sklearn/threadpoolctl source bytes against the locked core RECORDs. Pinned
sklearn and threadpoolctl both set the flag using `setdefault`. CatBoost leaves
it unset until the inspector imports; XGBoost/LightGBM import sklearn earlier.
The adapter preserves this upstream behavior and does not patch or preload
libraries. No observed crash/warning under that policy is a claim about
unsuppressed compatibility. Before-fit and completed-fit records distinguish the
two phases rather than silently rewriting their dependency evidence.

The adapter measures fitting/import/preprocessing/inspection time up to its final
receipt publication and process peak RSS, records native
max-round versus patience termination and retains failures/interrupts. A caller
may supply a wall-time ceiling; exceeding it makes the returned checkpoint
ineligible and retains paid time. Hard deadlines/RSS admission and child reaping
belong to the isolated execution harness; this synchronous adapter does not
pretend to provide transactional interrupted-native training resume. Full-quality
anchor/workload/operational qualification and Gate D remain external prerequisites.
The enclosing worker timer includes final receipt, snapshot and test I/O; the
adapter's timer is sampled before its own final JSON publication. Failures retain
their original exception, phase, traceback and measured cost, separating numerical,
resource, input/API and infrastructure/dependency domains. Unknown native errors
require diagnosis and cannot trigger an automatic seed replacement.

A minimal native config is:

```json
{"schema":"research/v2-native-tree-config","family":"P11","mode":"native_anchor","feature_schema":{"columns":["x","kind"],"categorical":["kind"]},"parameters":{}}
```

F/G arrays, seed, budget and fresh artifact directory are passed through
`FitRequest`. After `model.save(path)`, record `sha256((path/'state.json').read_bytes())`
in the external immutable selected-model record. Restore with
`NativeTreePredictor.load(path, expected_state_sha256=that_recorded_digest)`;
do not replace the trusted digest with one recalculated from an edited snapshot.

## Implementation plan

1. Write failing schema/preprocessing tests: F-only scaling/vocabulary, unseen
   categories, finite data, immutable inputs, feature order and support rejection.
   Implement strict config and the preprocessing state.
2. Write failing lazy-import, dependency/snapshot binding and failure-receipt tests.
   Implement native fit/predict/save/load and measured status records.
3. In fresh method-specific one-thread processes, compare tiny numeric/categorical
   wrapper predictions with independent direct official invocation; verify native
   G stopping, selected/executed counts, linear coefficients and save/load. Use
   synthetic F/G only and retain all attempts. No full anchors or HPO are run.
4. Run related core regression tests, request an independent review, address
   findings with failing tests first, then commit only this module, its related
   tests and documentation. Record exact source/test/receipt hashes and limitations.

## Review focus

- A reordered, extra or missing input feature must fail before prediction.
- F category values that resemble the unknown token must remain distinguishable.
- Changed model/config/source/dependency bytes must fail before model loading.
- A tiny capped fit must not become an eligible native quality result.
- A native fit failure or interrupt must retain its elapsed cost and original error.

## Validation and remaining work

Baseline on the isolated starting revision: `tests/test_simple_baselines.py`,
17 passed. Contract TDD first produced23 missing-module failures, then23 passes;
the next two missing-native-fit checks failed before implementation and passed.
The final contract set also checks rejection of caller-supplied suppression.

The first three native runs stopped at the first CatBoost cell. They retained
two import-context failures (first the upstream flag, then the inspector's later
transition), and the pinned CatBoost loader's `n_features_in_=0` wrapper behavior.
The saved native `feature_names_` accurately contains all features, so loading
now verifies persisted native feature names/topology. All attempts and their
source identities remain retained; no seed/config was changed to seek success.

The fourth native run and the final post-review fifth run passed all eight
numeric/categorical cells, comparing the
wrapper against independent literal direct official calls with exact prediction
equality. Exact save/load predictions, four corrupted-snapshot rejections per
cell, full192F/64G row coverage, unknown-category handling and categorical G-stop
stress all passed. Both linear LightGBM variants have actual nonzero affine-leaf
coefficients, a constant first tree and no categorical leaf-regression feature.

| Family | Numeric selected/executed | Categorical selected/executed |
|---|---:|---:|
| P08 CatBoost | 48/52 | 64/64 |
| P09 XGBoost | 30/34 | 64/64 |
| P10 LightGBM constant | 30/34 | 64/64 |
| P11 LightGBM linear | 22/26 | 64/64 |

All capped models remain engineering-only; cap attainment is not convergence.
No method-specific worker imported Torch or another tree package. The largest
recorded final adapter-fit process peak RSS was174,866,432 bytes. The final native
test run took22.80seconds including fresh-process imports/audits/direct calls;
this is not method-efficiency evidence. There were20 tiny fits in each passing
run, plus six retained fits from earlier failed engineering runs:46 total. There
were no native full-quality anchors.

Independent review found three material defects before commit: a rehashed recipe
edit/fixture promotion could pass loading; an external version-matching module
could be mislabeled with the official package identity; and post-start API/
infrastructure failures were all classified as numerical. The targeted first
reproduction produced8 failures/28 passes without any fitting. The fixes passed
36 targeted checks. Final expanded verification passed **73 scoped checks** in
4.39seconds, including six no-fit snapshot/origin probes,32 contract checks and
35 related legacy/transfer-contract checks. The two additional probes isolate
saved native-config consistency and the impossible30-selected/34-executed history
under a promoted patience100 recipe, even after its simpler parameter mismatch
is removed. The raw trusted-hash guard is tested separately from semantic checks.
No material review finding remains unaddressed. No second independent review or
full-project test suite is claimed.

The no-fit mutation probes make explicitly forged copies, updating their source
field only to isolate recipe validation from the independent source guard after
this repair changes source. They are not migrations, newly authenticated models
or quality results under the changed source. Original snapshots remain unchanged.

Final source SHA256:

```text
native_trees.py
  7798bf9db36ebb4a4c36b29141efb4aff964a2f04026588281c6c12bd1d18455
scientific package plus bound locks
  c2ed50988d17dd3f7cfa92db1ad489b200dd5bb24742e2dd377a3039307851e1
retained-array audit
  85cc0d33d109c933a3fb1373a13e5130fad7f0ae62579e1b54e1cbe6205ea2a9
```

The retained-array audit reconstructs all eight applied G risks from original
F/G arrays and row-aligned predictions, verifies snapshot/model/input hashes and
exact direct/restored prediction equality, and checks the test/status arithmetic.
Maximum discrepancy with the native selected G metric was8.69e-9; applied metric
reconstruction matched to1e-12. It imports or fits no native model. The compact
[receipt](native-tree-adapter-receipt.json) retains the exact evidence bindings.

Receipts are under this worktree's ignored
`artifacts/research-v2/WP07/native-adapters/`: `attempt-0` through `attempt-4`,
Junit results, per-cell admission/argv/cost logs, F/G arrays, native snapshots,
predictions, source/dependency/config manifests and verified summaries;
`review-red`, `review-green`, `review-verified`, Junit results and
`retained-audit.json` retain the review/fix evidence. Exact final passing source
and test bytes are copied into `attempt-4/source-*`. These
are a new fixture execution identity, separate from the earlier16 direct-native
fits and their unchanged receipts.

Run contract tests with the qualified interpreter. Native fixtures are explicit
opt-in and sequential, with fresh method-specific processes, a120second process
timeout, at least256MiB disk and512MiB free+inactive RAM admission proxy. The
standard subprocess timeout path kills and reaps its child. Use fresh outputs:

```sh
ATN_PY="/Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.venv-research-v2/bin/python"
PYTHONDONTWRITEBYTECODE=1 "$ATN_PY" -m pytest -q tests/test_native_tree_contract.py
ATN_TREE_OVERLAY=/private/tmp/atn-research-v2-next/environment/research-v2-tree-site \
ATN_TREE_ADAPTER_TEST_OUTPUT=/private/tmp/atn-tree-adapter-fixtures-fresh \
PYTHONDONTWRITEBYTECODE=1 "$ATN_PY" -m pytest -q tests/test_native_tree_parity.py
```

Prepared transfer data, V/T labels, native full anchors, HPO and heavy jobs are
outside this scoped implementation. Full-suite execution is not claimed: the
parent explicitly limited this parallel task to tiny isolated qualification.
