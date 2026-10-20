# Native backbones on four fresh primary synthetic problems

The stronger-backbone screen was **negative: 0 of 4 teachers met the 5% improvement rule**, against a required 3 of 4. The small 64-unit ordinary coordinate won the validation-selected one-layer family in all eight repetitions, and RealMLP-TD won the backbone family. The ordinary procedure had lower exploratory E error on all four teacher averages. No B0 expansion or residual-growth branch is admitted.

This is development evidence on the named primary d16 synthetic construction. It is not a new placement result, a full tuning benchmark, a general ranking of neural methods or protected confirmation. M3/Gate B remains negative. The main grown model is not an arm of this screen.

Exactly four fresh teachers, two paired repeats and nine fixed recipes produced 72 complete native fits; all 76 queue jobs completed without failure/recovery. F2306/G577/V577/E65536 and sigma 0.1 were fixed before generation. Models fit F, used G for native stopping, and the two finite families chose configurations on V only. No seed was selected. E is permanently development information.

## Complete configuration means

Original-unit MSE divided by the teacher’s F variance (ddof=0), averaged over its two repeats and four teachers; lower is better.

| Configuration | Family | V NMSE | E NMSE |
|---|---|---:|---:|
| O0_h64_historical | one_layer | 0.063702 | 0.073050 |
| O1_h64_native_optimizer | one_layer | 0.127566 | 0.135477 |
| O2_h128_native_optimizer | one_layer | 0.103083 | 0.108535 |
| O3_h256_native_optimizer | one_layer | 0.104225 | 0.112539 |
| O4_h512_native_optimizer | one_layer | 0.107880 | 0.117294 |
| B0_MLP_D | backbone | 0.116603 | 0.123689 |
| B1_ResNet_D | backbone | 0.110775 | 0.119600 |
| B2_RealMLP_TD | backbone | 0.084422 | 0.092325 |
| B3_TabM_piecewise | backbone | 0.150952 | 0.161685 |

O0 uses [64]/ReLU/LR .005/WD 0/dropout 0 under native P03 quantile preprocessing and stopping. O1–O4 use widths 64/128/256/512 with LR .001/WD .01/dropout .1. Thus O0 versus O1 changes an optimization/regularization bundle; this does not isolate learning rate, dropout or decay. MLP-D, ResNet-D, RealMLP-TD and embedded TabM retain their published native anchors, preprocessing, losses and schedules. No short engineering recipe substituted for native fitting.

## Validation-selected family comparison

Choose one configuration inside each family on V for each repeat, average the two E risks within teacher, then compare. Five one-layer and four backbone fits are paid per teacher/repeat; this is not equal portfolio cost. The teacher, not 72 fits or 262,144 E rows across teachers, is the independent unit.

| Teacher | Selected one-layer E NMSE | Selected backbone E NMSE | Backbone error change | Meets 5% reduction |
|---|---:|---:|---:|---|
| 0 | 0.036266 | 0.037952 | +4.7% | No |
| 1 | 0.069742 | 0.091740 | +31.5% | No |
| 2 | 0.112977 | 0.140520 | +24.4% | No |
| 3 | 0.073214 | 0.099088 | +35.3% | No |

The negative screen supplies no measured one-layer-family ceiling requiring a stronger parent in this regime. It does not prove that all tuned/backbone alternatives fail. In particular, the historical competent F-standardized continuation used in M3 was not directly rerun on these teachers, and the native one-layer grid uses quantile preprocessing. The predeclared standardized-ordinary and residual-structure safeguards remain prerequisites for any separately justified M6 proposal; this screen did not open that follow-up.

The earlier real-data anchor survey favored existing methods over P00 on all five observational tasks. That remains separate evidence. Do not pool its tasks/teachers with B0 or compare B0 and M3 means as paired before/after improvements. Four development teachers give limited population precision; no confirmatory interval, power or state-of-the-art claim is made.

## Execution and independent checks

The queue charged 1480.25 seconds of enclosing job time (about 24.67 minutes), including data/preparation/native/query work inside those jobs. Outer imports/publication and the earlier software qualification are separate. Another native NODE study and unrelated processes competed for resources; these are not dedicated-hardware or deployment timings.

Independent NumPy arithmetic checked 288 aligned F/G/V/E vectors and 1440 metrics against the producer’s fsum calculations; maximum absolute metric discrepancy was 1.11e-16. All 72 F variances also agreed within the declared numerical tolerance. The read-only queue lease, source/config/seed/role/native stopping and closed-state records were authenticated. Original inference/training receipts are trusted: no independent retraining or new model inference is claimed.

The producer core is commit 80afc0d (the full commit is in its reconciliation record), package `8a422995743bb121d3daa6263d47b334ca8a20c9047255206696a420cbaeaa57`; the plan hash is `c88a00977904d28756038aceb047f3942c57b3a039ac26c3d48fbc96b2a49cea`. The independently checked raw arithmetic hash is `cd401dc768ca6aa496f6e7c611b5b22af8fbde34968a473d60838524c8cd48c0`. Native worker quality/registry flags remain false: this is a controlled development screen while final common-source/support/practical qualification is unfinished. No protected T was created or read.

All four new teacher IDs/seeds are excluded from future confirmation. Local raw evidence and separately hashed operators are retained at `artifacts/research-v2/WP07/B0-native-primary-backbones-80afc0d/`. Rebuild with its read-only `audit.py` and this `report.py`; full arrays/checkpoints are retained execution assets, not promised in a clean checkout. [Declaration](../docs/research/v2/B0-native-primary-backbones.md), [execution](../docs/research/v2/B0-execution.md).
