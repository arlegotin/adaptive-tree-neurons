# Research v2: current result and next work

Updated 20 October 2026. This page summarizes the current state; the
[execution ledger](execution.md) retains earlier snapshots, failures, source
changes and exact evidence boundaries.

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
All twenty Airfoil anchors remain visible. Native engineering completions now
number 116 of 120. Concrete is running, followed serially by Parkinsons, CCPP
and Bike. The original survey and earlier receipts remain unchanged.

The user prioritizes completed results over the original two-hour budget.
Native anchors therefore run to their normal stopping rule with **no wall-time
cap**. Time-budget prefixes are a separate comparison. The current NODE worker
has a 4 GiB owned-process RSS guard, live admission checks and saved recovery
state. Airfoil took about 5¼ hours at about 1.83 GiB sampled root RSS; Energy
completed just beyond two hours. These contended measurements are not deployment
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

Next: finish the fixed native NODE inventory, audit each completion and publish
all fixed follow-up statuses. Preserve negative mechanism results and close only
the branches their gates authorize. The native/support/workload qualification
registry, production practical HPO, complete protocol/evaluator integration and
final source freeze remain unfinished. There is no permission to treat engineering
flags as production eligibility or to open T while those prerequisites are absent.

The [reproduction guide](reproduce.md), [binding spec](../../superpowers/specs/2026-10-18-research-v2-design.md)
and [implementation plan](../../superpowers/plans/2026-10-18-research-v2.md)
separate implemented components from unrun studies. Historical source identities
and publication results are preserved.
