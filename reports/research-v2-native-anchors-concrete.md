# Native anchor follow-up: Concrete NODE

Full native NODE has primary Concrete validation NMSE **0.161727166**, versus the grown cheap-placement anchor **0.209198922** and CatBoost **0.129859518**. Lower is better. NODE’s point error is 22.69% lower than grown; CatBoost remains lower than both. This is one fixed native setting/split/seed, not a tuned or protected-test comparison.

The [predeclared five-cell follow-up](../docs/research/v2/native-anchor-development-appendix.md) includes every completed cell regardless of validation outcome. This immutable tranche adds Concrete and rechecks the already scored Airfoil value exactly. It preserves the original 115-score survey and earlier Airfoil tranche. Native engineering completions now number 117 of 120; Parkinsons is running, with CCPP and Bike following serially. Energy was in the original survey.

| Fixed follow-up cell | Status | V NMSE |
|---|---|---:|
| airfoil | completed native development score | 0.107493331 |
| concrete | completed native development score | 0.161727166 |
| ccpp | pending native completion | — |
| parkinsons | pending native completion | — |
| bike | pending native completion | — |

All twenty Concrete anchor coordinates remain visible. Existing values come from the unchanged original survey; only P12 receives a newly available score. P13 is the mean anchor, rather than its full 29-configuration reference family.

| Family / anchor | Original V NMSE | New follow-up V NMSE |
|---|---:|---:|
| P00 / native_anchor | 0.209198922 | — |
| P01 / native_anchor | 0.202016216 | — |
| P02 / native_lower | 0.209275885 | — |
| P02 / native_upper | 0.201355504 | — |
| P03 / native_anchor | 0.222729208 | — |
| P03 / historical_width_lr_coordinate_under_native_preprocessing | 0.305127625 | — |
| P04 / native_anchor | 0.212364620 | — |
| P05 / native_centered_partial | 0.259638537 | — |
| P05 / native_centered_full | 0.244151279 | — |
| P05 / native_direct_partial | 0.245393119 | — |
| P05 / native_direct_full | 0.220629529 | — |
| P06 / native_anchor | 0.242288321 | — |
| P07 / native_plain | 0.232539011 | — |
| P07 / native_piecewise | 0.169020228 | — |
| P08 / native_anchor | 0.129859518 | — |
| P09 / native_anchor | 0.200201475 | — |
| P10 / native_anchor | 0.168680257 | — |
| P11 / native_anchor | 0.214310160 | — |
| P12 / native_anchor | unavailable | 0.161727166 |
| P13 / mean | 0.883176917 | — |

The native procedure with 2,048 trees, depth 6 and three output channels executed 19,801 updates, selected update 14,800 and completed 198 G evaluations. Native patience and all 5,712,812 applied row visits were independently checked from recorded traces. The audit replays selected predictions exactly; it does not repeat training, RNG transitions or weight updates.

Containing execution took 21712.649 seconds, about 6.03 hours, with 1,785,741,312 bytes sampled peak root-worker RSS (1.66 GiB). It used 516,096 trainable and 516,865 total parameters on eight encoded features and 577 F rows. Audit time was 10.091 seconds and is separate. These contended timers are not deployment performance or an efficiency comparison. No wall cap, data subsampling, reduced tree count or reduced native effective batch was used.

Primary V contains 123 rows in 86 mixture groups; group-weighted normalized MSE uses the F target variance with ddof=0. A separate raw fsum calculation agrees exactly on all seven risk metrics and F variance. Row-weighted RMSE is 5.973941; group-weighted RMSE is 6.700050. No repeat interval or independent dataset-population claim follows.

The producer remains at held source `c33a84e` / package `b680a26b...a65cc`. Engineering eligibility flags remain false; no model migration or extra fitting occurred for this scoring. The current coordinator’s source and before/after byte bindings are retained. T stays sealed, secondary development remains unavailable, and M3’s negative Gate B stays closed.

[All 20 Concrete points](../audit/research-v2/native-anchor-follow-up/concrete.csv), [complete five-cell snapshot](../audit/research-v2/native-anchor-follow-up/concrete-summary.json), [completion/scoring receipt](../docs/research/v2/WP07-NODE-Concrete-completion-and-V-receipt.json), [unchanged Airfoil tranche](research-v2-native-anchors-follow-up.md). Full common-source native qualification, HPO, global freeze and protected evaluation remain unfinished.
