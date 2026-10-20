# Research v2: current result and next work

Updated 20 October 2026. This page summarizes the current state; the
[execution ledger](execution.md) retains earlier snapshots, failures, source
changes and exact evidence boundaries.
The [cross-study report](../../../reports/research-v2.md) links the scientific
conclusions to their [evidence and counterevidence ledger](../../../audit/research-v2/current-study/claim-ledger.json).

## The idea and present verdict

Give selected neurons a switch between two affine rules. Insert it without
changing the network’s predictions, then train the extra flexibility. The
research question is whether choosing the locations intelligently helps more
than random locations or competent ordinary training.

**The mechanism works, but the tested growth chooser has not earned its added
complexity.** Useful planted-task learning survives independent checks. A
reliable placement or practical advantage remains unsupported. No protected
confirmation, broad tabular superiority or automatic inference saving is claimed.

## Completed scientific results

| Study | Actual scope | Result and disposition |
|---|---|---|
| [M1 controls](../../../reports/research-v2-M1.md) | 256 fits; planted structure and four negative-control populations | Growth learns the planted gates. Cheap placement does not reliably beat random. The four planted base instances recur across settings; they are not sixteen independent teachers. |
| [M2 diagnostics](../../../reports/research-v2-M2.md) | Eight primary teachers, paired repeats; timing, stream, ordinary-training and small-set controls | Substantial ordinary training headroom. Gate A passed a development screen, admitting one bounded joint-selector test. |
| [M3 joint selector](../../../reports/research-v2-M3.md) | Sixteen fresh primary teachers, two repeats, ten roles: 320 final models | Ordinary continuation beat joint growth on all sixteen teacher averages, about 15.6% lower arithmetic mean E error. Joint growth averaged 1.25% higher error than matched random. Gate B closed negative. |
| [B0 native backbones](../../../reports/research-v2-B0.md) | Four fresh primary teachers, two repeats, nine fixed native recipes: 72 fits | V selected the small ordinary coordinate and RealMLP-TD within their respective families. Ordinary had lower E error on all four teacher averages. Zero of four met the required 5% backbone improvement, versus a required three. Screen closed negative. |

All these outcomes are development evidence. Models train on F and select
checkpoints on G; V selects permitted configurations. Exploratory E is never
reused as protected T. Repeats and prediction rows do not increase the number
of independent teachers.

M3’s exact local ridge objective passes direct-versus-Schur numerical checks.
Its solved temporary small steps improve loss, but full local steps can saturate
gates and badly miss the nonlinear loss. These are temporary diagnostic updates,
not proof of harmful saturation in actual Adam training. The discrepancy supplies
no measured trigger for an extra direction/newborn grid.

M3 exhausted its one predeclared precision extension. No regularization
correction, direction/newborn grid or pruning redesign was admitted. B0 admits
no stronger-parent/residual follow-up. No C0 calibration or confirmation teacher
root has been created. A negative screen does not prove every architecture or
future growth method must fail; it closes the specified finite branch.

## Strong-baseline work

The [original native-anchor survey](../../../reports/research-v2-primary-anchors.md)
contains 115 completed fits from a fixed 120-cell inventory, on six designated
primary transfer splits with one seed. Existing methods lead the grown historical
anchor on all five observational validation tasks. A fully static gated model
leads Energy, a separate small-group simulation. These are V development scores,
not a tuned or protected-test benchmark. P13 is the mean anchor in this survey.

The [fixed NODE follow-up](../../../reports/research-v2-native-anchors-follow-up.md)
adds Airfoil: V NMSE 0.107493, versus grown 0.092186 and RealMLP 0.073302.
All twenty Airfoil anchors remain visible. The [Concrete tranche](../../../reports/research-v2-native-anchors-concrete.md)
adds NODE at 0.161727, versus grown 0.209199 and CatBoost 0.129860. All twenty
Concrete coordinates are visible. The [Parkinsons tranche](../../../reports/research-v2-native-anchors-parkinsons.md)
adds NODE at 1.311088, versus grown 1.246646 and CatBoost 0.738719. Its same-state
continuation after the disk-full interruption finished normally; native stopping
and selected predictions passed the original-source audit. All twenty coordinates,
both attempts and the unavailable original terminal cost tail remain visible.
Native engineering completions now number 118 of 120. CCPP is running and Bike
follows through the [serial controller](native-storage-continuation.md). The
original survey, earlier tranches and [interruption receipt](WP07-NODE-storage-interruption-2026-10-20-receipt.json)
remain unchanged. No complete total-cost claim is made for interrupted Parkinsons.

The user prioritizes completed results over the original two-hour budget.
Native anchors therefore run to their normal stopping rule with **no wall-time
cap**. Time-budget prefixes are a separate comparison. The current NODE worker
has a 4 GiB owned-process RSS guard, live admission checks and saved recovery
state. Airfoil took about 5¼ hours at about 1.83 GiB sampled root RSS; Concrete
about six hours at 1.66 GiB; Energy completed just beyond two hours. These
contended measurements are not deployment
benchmarks. [Execution amendment](results-first-native-execution.md).

Actual [memory probes](../../../reports/research-v2-resource-probes.md) completed
three full-size engineering geometries, including the largest embedded TabM
Bike case at about 3 GiB sampled worker RSS. Its initial 2 GiB stop and identical
request/seed retry are retained. These checks do not require 20 GiB free RAM;
live available-memory estimates are heuristic and do not guarantee admission.
No dataset, model size or effective batch was reduced for these results.

## Verification and remaining work

Independent raw-prediction arithmetic checked M3’s 640 V/E vectors and 7,680
metrics to 5.55e-17. B0’s 288 F/G/V/E vectors and 1,440 metrics agree to 1.11e-16;
all 72 F variances also agree. Those checks authenticate available arrays and
records, not independent retraining. Full raw execution assets remain local;
compact public evidence, scripts and source identities have explicit limits.

The complete software run at `61221f4` returned 2,217 passes, 42 failures,
369 setup errors and six skips. Its [full case inventory](../../../audit/research-v2/software/whole-suite-61221f4/summary.json)
retains memory-pressure failures, process-state contamination and unresolved
details. Seven separate native profiles at `4f19a05` passed 202 checks. Neither
is a final-source whole-suite success. Two reproduced launcher-isolation defects
were corrected in test helpers: 57 affected checks and 27 main checkout checks
passed. Fourteen fake-model timing/control checks also passed with declared
counter fixtures; the real hardware-dependent training test remains unchanged.
The [follow-up receipt](WP11-software-follow-up-receipt.json) preserves the
original failures and exact scope. Production guards remain intact.

The [recorded native training reader](native-training-observation.md) now checks
all fourteen mandatory families, with native stage/epoch/round records and
raw process observations. RealMLP's dropped partial batches and the tree
adapters' unexposed internal traces remain explicit. The latest combined run
passed 57 cases. NODE tensors are checked in chunks, with exact sampler order
and QHAdam state checks; the reader does not repeat optimization or recovery.
Earlier failures and fixes remain in the component receipts. The complete
actual qualification registry is still pending; this is not production model eligibility.

The [managed preparation reader](data-preparation-execution.md) also passed 57
separate data/process checks. Its worker prepared copies of all six official raw
datasets, fit no models and sealed every role; the reader kept protected archives
opaque. This supplies preparation provenance for the future registry, not a
new predictive result or a replacement production dataset.

The [full-size NODE recovery proof](native-recovery-qualification.md) passed
40 affected checks with retained process and state evidence. Restoring update
100 and continuing through 105 matches the uninterrupted saved training state,
including weights, optimizer, RNG and sampler, apart from elapsed-time fields.
This is an engineering capability; it grants no predictive or production
qualification. Earlier test failures and a sandbox-blocked retry are retained.

The [registry component](native-qualification-registry.md) now joins the fixed
120 anchors, 258 support exercises, seven software profiles, official preparation
and recovery proofs. Its combined partial-component checks passed 62 cases,
with one tree-overlay check skipped; that check then passed separately with the
pinned overlay. A real access-log test preserves immutable preparation while
checking every working-copy role file. The complete actual registry is unrun,
so this grants no Gate D or search eligibility.

The [generic candidate reader](native-inspection-reader.md) passed 54 affected
checks on actual small grouped and temporal native records. It reopens searched
configurations as well as anchors, checks saved replay evidence and recomputes
primary V arithmetic. It grants no selection eligibility. The separate
[registry-qualified callback component](qualified-native-trial.md) passed 70
affected checks, including real partial candidate evidence and rejection of a
false registry. Its complete qualified-success execution remains unrun; full
registry qualification, paid-journal and primary-model/freeze connections are
still required.

The [primary declaration compiler](primary-search-declaration.md) passed 78
affected checks. It binds exact study job/support/runtime/model/HPO namespaces;
a pre-production compatibility repair keeps paired model seeds within the
unchanged native tree guard. No completed scientific seed or result changes.
The [search auditor component](qualified-search-audit.md) then passed 93 scoped
checks, including actual partial candidate, failure, replay and paid-retry
records. Complete public qualified success awaits the actual final-source
registry. [Callback preparation](primary-native-preparation.md) then passed 61
scoped checks, including byte-only role staging and validation-label exclusion.
Its actual qualified-factory path is unrun. The [cooperative root supervisor](cooperative-process-ownership.md)
passed 71 affected checks, including actual nested cleanup and disk-publication
failures. It supplies a cold-root clock and no native/production authority. Its
[clock-origin extension](WP08-search-cold-origin-receipt.json) passed 110 affected
checks, paying startup and declared prior closing tails before prefix eligibility.
The [cold primary worker body](primary-search-worker.md) passed 132 affected
checks, including actual unqualified controller pause/resume and public rejection
of false qualification. Complete qualified execution remains unrun. The measured
production launcher, full external-cost reconciliation and primary-model/protocol
barriers remain unfinished. The [exact model-barrier component](qualified-primary-models.md)
passed 74 affected checks; complete execution on 168 genuine qualified searches
remains unrun and authorizes no secondary or protected access.
[Registry output boundaries](native-registry-output-paths.md) then passed 105
affected checks, protecting the registry itself, all working/proof inputs and
the interpreter before future output creation.

The [primary invocation preflight](primary-invocation-preflight.md) passed 138
affected checks. It authenticates actual qualification before plan reads or
output, protects all invocation paths and preserves paid parent setup time in
the cold-worker clock. Its complete qualified-success path remains unrun;
measured resource admission, production launch and complete cost are still
required. No protected or secondary access is authorized.

The [resource forecast component](native-resource-forecast.md) passed 85 affected
checks. It separates native anchor durations from short-probe memory/storage
measurements, includes external model files, and preserves no-cap native
training with advisory RAM/disk pauses. Its reducer fixtures are unqualified
metadata, not a measured forecast for this machine. Actual qualified forecasting,
per-candidate live admission and production launch remain unrun/unfinished.

The [search pause hook](search-resource-pauses.md) passed 112 affected checks.
It can close a paid resource pause before a proposal or fit, preserving TPE and
same-state recovery on resume. Actual forecast/host-observation binding in the
production worker and launcher remains pending; the hook alone grants no
qualification or execution authority.

Next: finish the fixed native NODE inventory, audit each completion and publish
all fixed follow-up statuses. Preserve negative mechanism results and close only
the branches their gates authorize. The native/support/workload qualification
registry's complete actual execution, production practical HPO, complete
protocol/evaluator integration and final source freeze remain unfinished.
There is no permission to treat engineering
flags as production eligibility or to open T while those prerequisites are absent.

The [reproduction guide](reproduce.md), [binding spec](../../superpowers/specs/2026-10-18-research-v2-design.md)
and [implementation plan](../../superpowers/plans/2026-10-18-research-v2.md)
separate implemented components from unrun studies. Historical source identities
and publication results are preserved.
