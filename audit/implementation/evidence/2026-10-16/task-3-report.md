# Evidence Task 3 implementation report — DONE_WITH_CONCERNS

Scope: current main, base232ba2c; no worktree, T access, freeze, scientific pilot,
new predictive inventory, YEAR data, or model weight downloads. No child agents.
Implementation owns B7 plus the expressly carried M1 dedicated coordinator fix.

## Source, environment, terms

NODE commit3bae6a8a63f0205683270b6d566d9cfa659403e4: all12 prior memo response
hashes verified and archived in configs/vendor/node/upstream. QHOptim1.1.0
commit0e0dc6053c4f7f95bc26cfeb336c6471fea4c5ea; wheel
cdd9660b8310ef7749733db03fe372bf5453861ba9cb0aa31db59f6127b3ae00 and source
ab29409cfa20fcd576ba239440ee3a651dbe734d68743cd3f768727bbc3aeb2e verified.
Sources.json includes14 immutable source/license records; the wheel-member
manifest independently records original archived package members.

Native Python3.12.13, arm64, Torch2.14.1, NumPy2.5.4, SciPy1.18.1,
sklearn1.9.1. Core uv.lock unchanged. locks/node/environment.json pins the
baseline native resolution; fit/load enforce exact environment. Reference-only
requests/tqdm with transitive packages live in environment/node-reference-site,
reinstallable via hash-locked reference-requirements.txt. No blanket legacy
requirements installation or core downgrade. The unchanged vendor reference
was executed BEFORE adapter implementation: DenseBlock forward/backward,
entmax/entmoid and actual QHAdam step passed. TorchScript and legacy add_
deprecation warnings are retained; no numerical/optimizer compatibility patch.

Runtime nn_utils.py/odst.py/arch.py and param_conv.py are byte-identical original
files. QHAdam changes only its local param_conv import. utils.py retains the
unchanged check_numpy function and its two imports, removing unrelated download
utilities. compatibility.patch and runtime-files.json record the exact changes.
Original NODE/QHOptim MIT notices and sparse attribution are preserved. Installed
numeric/reference dependency license files are inventoried/copied; current
DeepSPIN MIT attribution is separately pinned, with the exact earlier incorporated
entmax revision expressly unknown. No dataset/split/weight terms are inferred
from source MIT. Initial sandbox DNS failed; authorized scoped escalation fetched
only immutable source/licenses and exact-version dependency metadata/packages.

## Interfaces and recipe

NodePredictor(config) implements FitRequest fit/predict/save/load/resource_summary.
Full default config is year_node_shallow:2048x1/depth6/channels3, entmax15 feature
axis0/entmoid15, channel0 tree mean. Float32 CPU only, unsupported settings reject.
F noisy quantiles use the exact upstream divide-by-max-std formula, explicitly
resolved modern sklearn defaults, F target mean/std and F ordered prefix1000.
Actual shorter prefixes and training coverage are reported. Full F is streamed,
including remainder batches. G only selects latest-five parameter snapshots.
Reports100, strict stop step>beststep+5000, newest-first sequential sum/count;
restore latest raw weights for ongoing training, selected averaged weights and
matching QHAdam optimizer at final selection. Raw model+optimizer stay separately
available. No V fitting or prediction-time initialization.

An explicit max_updates budget is required. Ending before report100 exposes
selection_status=unavailable_at_budget, official_selection_completed=false,
quality_selection_eligible=false. Engineering fixtures are separately named and
never quality-eligible for B7. A completed selection does not certify a quality
plateau. Native notebook historical dependency versions remain unresolved, and
the present local recipe makes no published YEAR-score reproduction claim.

Checkpoints use hashed JSON metadata and hashed nonpickled NPZ arrays, bind exact
source/core lock/NODE lock/runtime, and reject uninitialized persisted state.
Quantiles, target scale, selected/raw models and optimizers, coverage/history are
restored. Row order is positional F order with actual prefix feature-byte hash;
no external row identifiers are fabricated. Fresh reload preserves prediction
and the next raw optimizer update.

node_checks.verify_node executes separately imported immutable upstream numerical
entry points, checks source hashes, then launches node_worker as a dedicated
native CPU subprocess. Worker is reproducible synthetic F1025/G1024/d16, exact
full-size anchor, init1000, batches512/512/1, one complete epoch, prediction1024,
JSON/NPZ save and fresh reload. Native default G selection is deliberately NOT
reached at budget3. Costs do not extrapolate to primaryF2306 or real convergence.
Registry declares B7 implemented; causal CLI marks it verified only after this
actual gate. Previously successful B0–B6 remain verified on a B7 gate failure;
track_ready and implemented_checks_passed remain false on that failure.

M1: collect_causal_report owns and terminalizes only its dedicated greedy probe
coordinator after report observation (completed), or on producer/report failure
(failed). Generic run_greedy_probes is unchanged, preserving broader search owner
semantics. Tests cover actual success receipt and injected failure terminalization.

## RED/GREEN record

All command logs are retained under artifacts/evidence3-node. Initial command
`.venv/bin/python -m pytest tests/test_node.py -q`:9 expected absent-adapter
assertion failures (red.txt). Initial implementation:8 passed/1 fixture failed;
the originally linear noncontiguous tensor did not reach sparsity. Fixed the
fixture without changing tolerance; final tensor covers both zero support and
nonzero gradients. CPU32 tolerance rtol1e-6/atol1e-7 was declared before execution.
CPU64 entmoid values/derivatives use1e-12.

Additional observed REDs: registry/CLI pending status (2); restored training
counts (1); M1 coordinator lifecycle (1); JSON metadata corruption (1); native
environment mismatch (1); newest-first sequential checkpoint reduction (1);
selected checkpoint optimizer alignment (1); preservation of B0–B6 verification
when B7 fails (1). Logs retain all failed/superseded attempts with their roles.
The first lifecycle test incorrectly asserted all caller manifests terminal;
it was narrowed to the dedicated owned coordinator, matching the explicit M1
scope rather than changing generic parent/search lifecycle behavior.

Tests independently execute archived original kernels and QHAdam, replay training
from F initialization with original vendor model/optimizer, execute the unchanged
archived Trainer.average_checkpoints method body (isolated from notebook logging
imports), reject a sigmoid counterexample, check sparse support/ties/noncontiguous
actual selector axis/backward, F/G isolation, full coverage, original-unit output,
checkpoint bytes/metadata/source/runtime/initialized flags, latest-five state/raw
restoration, strict patience, dense gradients, and next optimizer update on reload.

Covering command: `.venv/bin/python -m pytest tests/test_node.py tests/test_adapters.py tests/test_causal_baselines.py tests/test_greedy.py tests/test_simple_baselines.py -q`
=>106 passed,28 warnings,46.43s. Final focused command tests NODE and both new
failure paths =>21 passed,28 warnings,9.76s. No numerical tolerances loosened.

Final full-suite, retained CLI, identities, costs and self-review follow below.

Additional implementation boundary: recent report snapshots are kept in memory
and ordered by step; the notebook uses filesystem-ctime discovery and temporary
checkpoint files. Notebook logging/UI and temporary checkpoint I/O are not
ported. The mathematical newest-first parameter sum/count and raw restoration
are preserved. Three-step default-cost observations exclude any G report and
its averaging/selection work; explicit endpoint save/reload and startup are
reported separately. Long-running snapshot memory/quality costs remain Task7.
The reference hash lock was recreated offline in /tmp/evidence3-reference-repro:
6 exact packages installed. A sandbox cache-access failure and scoped successful
retry are both retained. The original20,343-byte QHOptim wheel is also archived
locally at artifacts/evidence3-node/acquisition with its known response hash.

## Final verification and retained evidence

Status DONE_WITH_CONCERNS (scope limits below; no blocked correctness requirement).
Implementation commit41d0178cf1747ff1ee2ab44159c9dc39662210bc, package source
6146d731db228c5a9495f19262139622dec462b9707208476f84400342710fd7.
One final `.venv/bin/python -m pytest -q`: **678 passed,48 skipped,28 warnings,
249.77s**. The48 existing skips are sandbox/MPS-dependent; no B7 test is skipped.
Exact required `.venv/bin/python -m pytest tests/test_node.py -q` after the
implementation commit: **19 passed,28 warnings,2.28s**. `git diff --check` clean.
All vendor manifests were checked against14 original source hashes and6 runtime
copy records. No tolerance changes were used to obtain a pass.

Required retained CLI, exit0:

```
.venv/bin/python -m treeneuron.cli verify-baselines --track causal --backend-report environment/task11-parent-backend.json --artifacts-dir environment/evidence3-node-causal-runs --output environment/evidence3-node-causal.json
```

B0 mean/ridge/spline,14 neural B1–B6 recipes and actual B7 gate pass;
implemented_checks_passed=true and track_ready=true. Root independent review
remains pending; this is the executable WP03 causal gate, not quality/confirmation.
Every new parent/checkpoint uses the current package source. Dedicated coordinator
`environment/evidence3-node-causal-runs/4fa37922b24c49539683877f37610992`
has manifest status completed and terminal metrics. Failure-side terminalization
is separately covered by the focused injected-producer-failure test.

Core backend producer237035c85e84b706facd7a7054a159f62dcff50e00f3ca3a5fdae0a11ca4fd7e,
lock13e693201df7368be782d81f8fc386e6c291da2efb819a97a0524f9563cb19b5,
runtimeaf9618d1f31b7f55bd414c489f0cf65b3f96547df5b9c57128207a50b7871b68
still exactly match the genuine prior CPU receipt. Its SHA is
44899df8ec13349ee74b37511b25bfcda3aeace9a0640449e05e51669ce60197;
it was not rewritten. That receipt supports matching core B0–B6 paths only;
NODE's native worker and original vendor checks have separate execution evidence.
NODE environment lock SHA89ce6a4b559114eb759d31092d88287ff5e0cd04672bdcabd9143c3a4e56b561.
Historical evidence2 index resolves all329 entries:325 remain unchanged; four
intentional current code/test updates are verified against their original Git
base bytes (causal_checks.py,registry.py,test_adapters.py,test_causal_baselines.py).
All17 previously tracked research/audit documents remain byte-identical to232ba2c.
Original source guards and old predictive results/provenance are untouched.

### Actual full-size anchor costs (one retained run)

Full2048x1/depth6/channels3 anchor, synthetic F1025/G1024/d16, seed1337,
F-prefix1000, full-F batches512/512/1. Structural reload verifies response
[2048,3,64], selectors[16,2048,6],614400 trainable and769 nontrainable parameters.
Initialization1.004014s; preprocessing0.016368s; training steps
4.388721/2.042690/0.015904s; full F epoch6.447524s; fit7.974388s;
prediction of1024 rows1.275407s. Dedicated process peak RSS10,002,169,856bytes
includes imports/init/train/predict/save/reload and is not incremental model memory.
Worker measured body10.087284s, startup-inclusive subprocess11.935681s.
Endpoint save0.338711s, fresh reload0.048626s, reload prediction difference0.
Endpoint JSON+NPZ9,743,514bytes; NPZ SHA
2e7e22c90a44303106eae629a8c3666b441c1578976873e57dd2e63378ea95fc.

Cost-worker scope starts its body timer after synthetic array generation and
excludes imports, which are included in the enclosing subprocess timer. The
fit timer starts after input/identity validation; per-step timers enclose forward,
loss, backward and QHAdam update; training timer also includes sampler and loop
bookkeeping. Final checkpoint/prediction timers are separate and overlap worker
body time. No overlapping timers are added into a claimed total.

Budget3<report100 means no official G-selected averaged checkpoint exists.
selection_status=unavailable_at_budget; official_selection_completed=false;
quality_selection_eligible=false. The endpoint establishes full-size execution
and cost, not a usable quality winner. There is no primaryF2306 timing, default
5000+ stopping run, quality plateau, real-data convergence or predictive forecast.
No longer training was launched. The earlier development worker's differing
costs were preliminary single-run observations, not substituted for this receipt.

Curated record: docs/research/2026-10-16-evidence3-node.json,
SHA980c1f450d09e285ed14e68cf3d034e521a9d734d497085bfbb620544e22b9de.
521-file index: audit/implementation/evidence/2026-10-16/task-3-artifact-index.json,
SHA008b37858dcbcafdd6f3ecac8452f8b226e484b74b0c43c496cdce93b68ec052.
Index binds original sources/licenses, runtime/lock/tests, retained failure logs,
final CLI parents/checkpoints and worker output; rollups/index exclude themselves
to avoid circular hashes. Bulk engineering artifacts stay local, not in Git.

## Self-review, concerns, next gate

Requirements checked against the brief: original hashes and terms retained;
actual unchanged CPU vendor attempted first; faithful functions, selector axis,
unused channels, QHAdam update/state, F transforms/init, full-F stream, checkpoint
averaging/restoration and initialized reload verified; no optional-unavailable
skip hides B7; full-size cost measured; all B0–B6 tests preserved; M1 owner closure
verified success/failure; baseline-specific environment identity and fresh source
parents enforced. The self-review found and corrected reduction-order, selected
optimizer alignment, persisted coverage and metadata-integrity defects before
final suite/commit. No source edit followed the final full suite.

Remaining limits are deliberate and recorded: native arm64 CPU checkout only;
modern explicitly resolved dependencies rather than unknown historical versions;
exact older DeepSPIN revision unidentified; no missing/categorical input recipe;
no published YEAR-score reproduction; no default quality/plateau or primaryF2306
cost calibration. Snapshot persistence replaces notebook temporary filesystem/UI
infrastructure. Save/load guarantees prediction plus tested next raw optimizer
update, not a resumable epoch/cursor API. These are not hidden substitutions.
No protected T, freeze, pilot, new predictive dataset inventory or independent
H1 confirmation occurred. Root owns review/ledger acceptance. Next authorized
gate is Task7 cost/default-budget/search calibration after that review; it must
treat a default unable to reach required G selection at budget as unavailable.
