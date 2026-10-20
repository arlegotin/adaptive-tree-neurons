# Native anchor follow-up: Parkinsons NODE

NODE’s primary Parkinsons validation NMSE is **1.311088052**, versus grown **1.246645683** and CatBoost **0.738718662**. Lower is better. NODE has 5.17% higher point error than grown; CatBoost remains substantially lower than both. This is one fixed setting, split and seed, with no tuned or protected-test comparison.

The predeclared five-cell follow-up discloses every completed cell. This tranche adds Parkinsons and rechecks Airfoil and Concrete exactly. Earlier tranches and the original 115-score survey remain unchanged. There are now 118 of 120 native engineering completions; CCPP is running and Bike follows serially.

| Fixed follow-up cell | Status | V NMSE |
|---|---|---:|
| airfoil | completed native development score | 0.107493331 |
| concrete | completed native development score | 0.161727166 |
| ccpp | pending native completion | — |
| parkinsons | completed native development score | 1.311088052 |
| bike | pending native completion | — |

All twenty original Parkinsons anchors remain visible. P13 is the mean anchor, rather than the full reference search.

| Family / anchor | Original V NMSE | New follow-up V NMSE |
|---|---:|---:|
| P00 / native_anchor | 1.246645683 | — |
| P01 / native_anchor | 1.246645683 | — |
| P02 / native_lower | 1.246645683 | — |
| P02 / native_upper | 1.246645683 | — |
| P03 / native_anchor | 1.827357374 | — |
| P03 / historical_width_lr_coordinate_under_native_preprocessing | 1.677368979 | — |
| P04 / native_anchor | 0.938104799 | — |
| P05 / native_centered_partial | 1.359541049 | — |
| P05 / native_centered_full | 1.307316248 | — |
| P05 / native_direct_partial | 1.371212002 | — |
| P05 / native_direct_full | 1.267868525 | — |
| P06 / native_anchor | 1.043741092 | — |
| P07 / native_plain | 1.137401712 | — |
| P07 / native_piecewise | 1.404445391 | — |
| P08 / native_anchor | 0.738718662 | — |
| P09 / native_anchor | 0.784079677 | — |
| P10 / native_anchor | 1.441330265 | — |
| P11 / native_anchor | 1.440820823 | — |
| P12 / native_anchor | unavailable | 1.311088052 |
| P13 / mean | 1.516624219 | — |

Full native NODE (2,048 trees, depth six, three output channels, effective batch 512) executed 5,801 updates and selected update 800, with 58 G evaluations. Its trace, rolling checkpoint selection and first native patience endpoint were independently checked. Selected predictions replay exactly. Training, RNG transitions and weight updates were not independently reexecuted.

The original attempt hit a shared disk-full error. Its published update-4,000 checkpoint passed the complete original-source loader before a same-state, same-seed continuation. The successful continuation took 1.068 hours with 1.775 GiB sampled root RSS; the original attempt has at least 2.443 hours in retained resource samples. Its missing terminal cost tail remains unavailable. These are separate attempt costs; no complete total-cost or efficiency claim follows. Audit time (9.635 seconds) is separate. No wall cap, data subsampling or model-size reduction was used.

F has 3,352 rows and 21 encoded features. The selected model has 675,840 trainable / 676,609 total parameters. V has 742 rows from only 6 subjects. Group-weighted NMSE divides by F target variance (ddof=0). An independent fsum calculation agrees exactly on seven risk metrics and F variance. Group-weighted RMSE is 9.304465; row-weighted RMSE is 9.145428. Six subjects do not support broad population confidence from this one development split.

Producer source remains `c33a84e` / package `b680a26b...a65cc`. Scoring used coordinator `5356edf` / `65553d01...b8` without fitting or model inference. Engineering eligibility remains false. T and secondary-fold development stay sealed; M3’s negative Gate B stays closed.

[All twenty points](../audit/research-v2/native-anchor-follow-up/parkinsons.csv), [complete scalar snapshot](../audit/research-v2/native-anchor-follow-up/parkinsons-summary.json), [completion/scoring receipt](../docs/research/v2/WP07-NODE-Parkinsons-completion-and-V-receipt.json), [storage recovery](../docs/research/v2/native-storage-continuation.md). Full common-source qualification, HPO and protected evaluation remain unfinished.
