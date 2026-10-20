# Native anchor follow-up: Airfoil NODE

The completed full native NODE anchor has primary Airfoil V NMSE **0.107493331**. The grown cheap-placement anchor is **0.092186325**, while the lowest earlier native-anchor point is RealMLP-TD at **0.073302238**. Lower is better. This adds one completed baseline to the development picture; it does not establish a tuned-family or protected test comparison.

The [prospective decision](../docs/research/v2/native-anchor-development-appendix.md) fixed all five originally unavailable P12 cells before this new V read. The original120-cell survey/115scores are unchanged. Airfoil is now scored; Concrete, CCPP, Parkinsons and Bike remain pending. Energy was already included in the original survey. Every later completed cell will be disclosed regardless of V.

| Fixed follow-up cell | Status | V NMSE |
|---|---|---:|
| airfoil | completed native development score | 0.107493331 |
| concrete | pending native completion | — |
| ccpp | pending native completion | — |
| parkinsons | pending native completion | — |
| bike | pending native completion | — |

All twenty Airfoil anchor points are shown below. Existing values are copied from the immutable original survey; only P12 gains a new value in this follow-up. P13 is the mean anchor, rather than its full29-configuration family.

| Family / anchor | Original V NMSE | New follow-up V NMSE |
|---|---:|---:|
| P00 / native_anchor | 0.092186325 | — |
| P01 / native_anchor | 0.109896163 | — |
| P02 / native_lower | 0.124015943 | — |
| P02 / native_upper | 0.143294240 | — |
| P03 / native_anchor | 0.128266898 | — |
| P03 / historical_width_lr_coordinate_under_native_preprocessing | 0.168831373 | — |
| P04 / native_anchor | 0.133495460 | — |
| P05 / native_centered_partial | 0.096119065 | — |
| P05 / native_centered_full | 0.120279864 | — |
| P05 / native_direct_partial | 0.112963871 | — |
| P05 / native_direct_full | 0.117464570 | — |
| P06 / native_anchor | 0.073302238 | — |
| P07 / native_plain | 0.088256035 | — |
| P07 / native_piecewise | 0.090272856 | — |
| P08 / native_anchor | 0.118554031 | — |
| P09 / native_anchor | 0.141720454 | — |
| P10 / native_anchor | 0.133238150 | — |
| P11 / native_anchor | 1.206300205 | — |
| P12 / native_anchor | unavailable | 0.107493331 |
| P13 / mean | 0.917734973 | — |

This is one fit seed17 on one primary group split, with185V rows in15groups. It supplies no repeat uncertainty, independent-task replication, test-best selection or full-suite dominance. The native producer stays at held commit`c33a84e`/package`b680a26b...a65cc`; its engineering eligibility flags remain false. No model was migrated into the current package or newly trained for scoring. The negative sixteen-teacher M3 Gate B remains closed.

The full2048-tree/depth6/tree_dim3 native run executed12,101updates, selected7,100and completed121G evaluations, stopping by its native patience rule. It took18,923.475828875seconds (about5¼hours) with1,962,049,536bytes sampled peak worker RSS, about1.83GiB. This is a contended operational runtime and worker-only memory measurement. There was no wall deadline, data subsampling or reduced tree count. Its separate8.328095500-second audit verified saved predictions exactly and checked native row coverage/rolling selection/stopping; training was not independently reexecuted.

The logged primary V scorer and a separate raw fsum reconstruction agree exactly on all seven metrics and F variance. New scoring/reconstruction took0.076362958seconds, not model training. T stayed sealed. A separate unconditional metadata review confirms terminal status, roles and hashes; the original executed scoring operator is retained, including a conditional assertion whose fallback checked roles while the prior completion audit already bound the actual valid status. No score or inclusion changed. A report-only None-metrics error was retained and corrected without reopening labels.

Source/data/score/cost bindings are in the [completion/follow-up receipt](../docs/research/v2/WP07-NODE-Airfoil-completion-and-V-receipt.json), with [all20Airfoil points](../audit/research-v2/native-anchor-follow-up/airfoil.csv) and [allfivefollow-upstatuses](../audit/research-v2/native-anchor-follow-up/summary.json). Full common-source native qualification, HPO, global freeze and protected evaluation remain unfinished.
