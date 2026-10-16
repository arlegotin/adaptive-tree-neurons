# Faithful local shallow NODE recipe

`upstream/` archives immutable official files. `sources.json` verifies all twelve
NODE response hashes from the prior memo plus QHAdam source and license. The
hash-verified QHOptim wheel's original package members and metadata are archived
separately; the wheel hash and member hashes are in `locks/node/qhoptim-wheel.json`.
Numerical runtime copies retain original attributions and MIT notices. See
`compatibility.patch` and `runtime-files.json` for exact extraction/import changes.

The named `year_node_shallow` anchor is 2048 trees, one layer, depth 6, three
channels, entmax15 selectors on feature axis 0, entmoid15 bins, and mean of
channel 0 across trees. Initialization uses the ordered first min(1000, F) rows.
Features use F-fitted normal quantiles with the exact noisy-fit formula; target
mean/std come from F. No G/V values fit transforms or initialization. Only finite
numerical inputs are supported. Short F inputs disclose their actual prefix.

Training streams all F rows in shuffled epochs, including incomplete batches.
QHAdam 1.1.0 preserves nus=(.7,1), betas=(.95,.998), with explicit resolved
lr=.001, eps=1e-8, weight_decay=0, decouple_weight_decay=False. Every 100 updates,
newest-five model snapshots are averaged using newest-first sequential sum/count,
evaluated on G in chunks of 1024, and saved if improved. Latest raw parameters
are restored for further optimization. Stop uses strict step > best_step+5000.
Final prediction uses the best averaged state and its optimizer; the latest raw
model and optimizer are retained separately. Integer initialization state is
validated and retains its dtype (upstream averages it then casts on load).

An explicit `FitRequest.budget.max_updates` limits local execution. If it ends
before a G report, `selection_status=unavailable_at_budget`,
`official_selection_completed=false`, `quality_selection_eligible=false`.
The resulting raw endpoint is useful for bounded engineering execution only.
Task 7 must calibrate default quality/stopping budgets. A successful correctness
and cost gate does not make that endpoint a quality winner.

`engineering_fixture` permits explicitly named smaller scale/cadence tests.
They cannot claim full-anchor quality eligibility. The complete 2048-tree cost
worker uses synthetic F1025/G1024/d16, initialization1000, batches512/512/1 and
one complete epoch. It does not measure primary F2306 or any dataset convergence.

Historical sklearn/QHOptim versions remain unknown. Effective QuantileTransformer
parameters are pinned to the tested modern environment (subsample10000). The
old source runs natively with TorchScript/add_ deprecation warnings, without
math patches or Adam/sigmoid substitutions. No published YEAR score is reproduced.
Code MIT permission is not a claim about YEAR/split-index/weight asset terms.

The local adapter keeps the latest five report snapshots in memory, orders them
by report step, and explicitly saves JSON/NPZ endpoints. This replaces the
notebook's filesystem-ctime discovery, logging/UI and temporary checkpoint I/O;
it preserves the numerical newest-first averaging and raw-state restoration.
The bounded three-step cost run does not measure G-report snapshot averaging,
selection-time checkpoint work, or long-running training memory. Its final
endpoint save/reload and process startup have separately recorded boundaries.
