# Task 3 independent audit — numerical STOP

The complete fixed diagnostic has independently reconciled saved-prediction risks, exact selected replay/restore evidence and two identical generated reports. Final input-preservation verification and the complete audit inventory are separate machine-readable gates at the paths below. Root owns final curation/publication and the decision commit. No tracked source/config/lock, original cohort/cost record or historical report was edited.

## Findings and claim boundary

Longer selection has no clear placement advantage over random or the short selector at this diagnostic resolution. Random minus long is positive in three teacher means and negative in five; its median is negative and both descriptive intervals cross zero. Short minus long has a negative mean, one zero teacher and two negative teachers; its intervals also cross zero. Ordinary upper minus long is positive for all eight teacher means, while individual repetition signs vary. The cheap score has no clear advantage over random. This supports narrowing this allocation procedure or investigating timing/training under a new prospective design, without automatically authorizing another cohort.

All original-unit MSE/RMSE/MAE and F-population-variance-normalized risks were reconstructed from the actual NPZ predictions and regenerated original F/G/E targets, predictors and row IDs. Saved row-loss arrays and the reviewed analyzer were not inputs to this calculation. Repetitions are averaged within teacher, then all eight teachers equally; zero and negative effects remain. The only bootstrap uses 512000 paired whole-teacher draws, seed261031106, with the locked difference ordering. Student-t intervals remain descriptive sensitivity.

| Method | Original MSE | Mean RMSE | MAE | F-normalized MSE |
|---|---:|---:|---:|---:|
| keep | 0.0660175567858 | 0.255457681163 | 0.180600843945 | 0.0648668733259 |
| top | 0.0653452715027 | 0.254213776128 | 0.179739286266 | 0.0642050866253 |
| random | 0.064992093445 | 0.253424058207 | 0.1795074125 | 0.0638495407085 |
| greedy | 0.0642589251362 | 0.252103963847 | 0.178639450273 | 0.0631248538468 |
| greedy_long | 0.0645332573578 | 0.252598060693 | 0.179272803875 | 0.0633780485196 |
| widen_upper | 0.0651892748382 | 0.253677183043 | 0.179420363071 | 0.0640707018718 |
| ordinary_lower | 0.0685598740116 | 0.259238068202 | 0.185301087109 | 0.0672781714185 |
| ordinary_upper | 0.0706635088689 | 0.263550828875 | 0.188047360878 | 0.0693750468477 |
| static_partial | 0.102258539658 | 0.316065607015 | 0.227983166906 | 0.100520355135 |

| Paired difference (positive favors grown) | Mean | Median | Bootstrap 95% | Student-t 95% | Negative / zero teachers |
|---|---:|---:|---|---|---:|
| random_minus_greedy_long | 0.000471492188819 | -0.000237666404781 | [-0.001036580561557831, 0.002285193076203225] | [-0.001692681524133126, 0.0026356659017720695] | 5 / 0 |
| greedy_short_minus_greedy_long | -0.000253194672846 | 0.00016146009986 | [-0.0009664923934574713, 0.0003316261670924794] | [-0.001100746195762579, 0.0005943568500704406] | 2 / 1 |
| ordinary_upper_minus_greedy_long | 0.0059969983281 | 0.0049172408364 | [0.002534402906064537, 0.009878283716729153] | [0.0012172342858234137, 0.010776762370382734] | 0 / 0 |
| random_minus_top | -0.000355545916887 | 0 | [-0.0018856813734340256, 0.0010121020424563236] | [-0.0022412908788022614, 0.0015301990450281053] | 3 / 3 |

## Complete teacher distributions and repeat variability

The following are equal-R2 teacher means. Full four-metric per-pair risks and both repetition effects are retained in `artifacts/long-probes-20261018/audit/independent.json`; no teacher or method is omitted.

| Teacher | keep | top | random | greedy | greedy_long | widen_upper | ordinary_lower | ordinary_upper | static_partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.08777864587 | 0.08777864587 | 0.08777864587 | 0.08777864587 | 0.08982016292 | 0.08777864587 | 0.1186921242 | 0.1031138936 | 0.1662627925 |
| 1 | 0.05142696703 | 0.0501823945 | 0.05083910958 | 0.0506545838 | 0.05232555605 | 0.04811846587 | 0.05223848975 | 0.05866855238 | 0.06837035203 |
| 2 | 0.06531192257 | 0.06531192257 | 0.06531192257 | 0.06059953315 | 0.05995637594 | 0.06531192257 | 0.05569540852 | 0.06126714343 | 0.1091943391 |
| 3 | 0.07262731237 | 0.07262731237 | 0.06772786045 | 0.06938952449 | 0.06892524991 | 0.07262731237 | 0.07010639095 | 0.06938099451 | 0.09446908045 |
| 4 | 0.055883418 | 0.05490888559 | 0.05325577265 | 0.05373860166 | 0.0535863355 | 0.05257058899 | 0.05813851196 | 0.05425348382 | 0.06640822916 |
| 5 | 0.05005743635 | 0.05005743635 | 0.05005743635 | 0.0502022063 | 0.0502022063 | 0.05030939489 | 0.05718284548 | 0.05729779828 | 0.07266356381 |
| 6 | 0.08388165714 | 0.08080646847 | 0.08388165714 | 0.08066810821 | 0.08041152828 | 0.08388165714 | 0.07270971809 | 0.09573005014 | 0.1304472071 |
| 7 | 0.05196762729 | 0.05196762729 | 0.05194392108 | 0.05196762729 | 0.05179697325 | 0.05196762729 | 0.0534618824 | 0.0552884586 | 0.09634727689 |

| Teacher | random_minus_greedy_long | greedy_short_minus_greedy_long | ordinary_upper_minus_greedy_long | random_minus_top |
|---|---:|---:|---:|---:|
| 0 | -0.002041517049 | -0.002041517049 | 0.01329373071 | 0 |
| 1 | -0.001486446477 | -0.001670972258 | 0.006342996323 | 0.0006567150719 |
| 2 | 0.005355546627 | 0.0006431572125 | 0.001310767486 | 0 |
| 3 | -0.001197389466 | 0.0004642745796 | 0.0004557445996 | -0.004899451919 |
| 4 | -0.0003305628538 | 0.0001522661589 | 0.0006671483147 | -0.001653112942 |
| 5 | -0.0001447699557 | 0 | 0.007095591976 | 0 |
| 6 | 0.00347012886 | 0.0002565799328 | 0.01531852187 | 0.00307518867 |
| 7 | 0.0001469478249 | 0.0001706540408 | 0.003491485349 | -2.370621585e-05 |

Within-teacher sample SD of the two repetition effects (distinct from the analyzer’s pooled repetition SD):

| Teacher | random_minus_greedy_long | greedy_short_minus_greedy_long | ordinary_upper_minus_greedy_long | random_minus_top |
|---|---:|---:|---:|---:|
| 0 | 0.002887141099 | 0.002887141099 | 0.01646426385 | 0 |
| 1 | 0.002102152767 | 0.002363111629 | 0.01027650832 | 0.002591447324 |
| 2 | 0.004615164644 | 0.0009095616526 | 0.01142765555 | 0 |
| 3 | 0.006634758115 | 0.0006565834071 | 0.001961781625 | 0.001399251185 |
| 4 | 0.000312517643 | 0.002519755734 | 0.003334053461 | 0.0004150998193 |
| 5 | 0.0002047356348 | 0 | 0.0043964684 | 0 |
| 6 | 0.004907503298 | 0.0003628588208 | 0.02917785466 | 0.004348973524 |
| 7 | 0.0005438735565 | 0.0005103479046 | 0.0138069365 | 3.352565197e-05 |

| Teacher / repetition | random_minus_greedy_long | greedy_short_minus_greedy_long | ordinary_upper_minus_greedy_long | random_minus_top |
|---|---:|---:|---:|---:|
| 0 / 0 | 0 | 0 | 0.02493572332 | 0 |
| 0 / 1 | -0.004083034099 | -0.004083034099 | 0.001651738093 | 0 |
| 1 / 0 | 0 | 0 | 0.01360958504 | 0.002489145048 |
| 1 / 1 | -0.002972892953 | -0.003341944516 | -0.0009235923974 | -0.001175714904 |
| 2 / 0 | 0.002092132411 | 0 | 0.009391340216 | 0 |
| 2 / 1 | 0.008618960843 | 0.001286314425 | -0.006769805243 | 0 |
| 3 / 0 | 0.003494092989 | 0.0009285491591 | -0.0009314444905 | -0.003910031918 |
| 3 / 1 | -0.005888871921 | 0 | 0.00184293369 | -0.005888871921 |
| 4 / 0 | -0.0001095795092 | 0.001934002525 | -0.001690383496 | -0.001946632839 |
| 4 / 1 | -0.0005515461984 | -0.001629470208 | 0.003024680126 | -0.001359593045 |
| 5 / 0 | 0 | 0 | 0.003986819357 | 0 |
| 5 / 1 | -0.0002895399115 | 0 | 0.01020436459 | 0 |
| 6 / 0 | 0.006940257721 | 0.0005131598655 | -0.005313337026 | 0.00615037734 |
| 6 / 1 | 0 | 0 | 0.03595038076 | 0 |
| 7 / 0 | 0.0005315245049 | 0.0005315245049 | 0.01325446378 | 0 |
| 7 / 1 | -0.000237628855 | -0.0001902164233 | -0.006271493076 | -4.741243171e-05 |

## Verification and fixed replay

- Initial independent verification: all 28359 sealed cohort/environment files, 11498051579 bytes; inventory SHA256 `900b5cfeaa6d3d67cc12a7b895506fea6951c83bed76e79b195921a5cf79a5fd`.
- Eight regenerated teachers, sixteen paired repetitions, 144 final role instances, 262144 E rows per role and 37748736 saved prediction values; exact original F/G/E arrays and row IDs. Full F variance, all nine methods and within-teacher method SD remain in the independent record.
- Production reconciliation: 760 numeric values, maximum discrepancy 0.0; all nine methods/four metrics, pair risks/F variances, all four contrasts, complete teacher/repetition effects, paired bootstrap percentiles, medians, zero/negative counts and Student-t sensitivity agree.
- Six original-parent warm forks in each of sixteen pairs (96 total): independently loaded full parent-state commitments; exact optimizer/sampler/RNG/schedule/policy bindings, original score/per-ID contrasts and actual raw executed-row common prefixes. Original-parent first-batch IDs and standardized loss match keep/top/random/short/long in all sixteen pairs, including epoch rollover.
- Prospectively fixed teacher0/R0; short and long complete rankings each replayed once from the original selected parent, original serialized ScoreTable, contrasts and seed. No outcome-based representative choice or extra efficacy unit.
- 130 trials and 133 complete state comparisons: two coordinator parents, every probe best, and the first eligible long trial (`unit_0`) actual durable last checkpoint. Actual model topology/tensors, all Adam moments/ages/groups, scheduler, complete RNG, sampler/order/cursor, next rows, G predictions and durable best snapshots agree exactly.
- 53300 ordered raw update events/losses, 2275 actual G evaluations and 2275 checkpoint event coordinates agree. Short endpoint320 and long25-update grid remain separate complete procedures. Descriptor filenames, coordinator UUIDs and their checkpoint locator hashes differ legitimately: only explicitly identified/validated coordinator/parent-descriptor fields and checkpoint-history descriptor hashes are normalized; learned values are untouched. Real pointers/payloads/descriptor hashes are retained in replay records and inventory.

## Deterministic reviewed reports

The unchanged reviewed `treeneuron.pilot.probe_diagnostic` module performed `analyze`, then `report` twice to new exclusive destinations. Each call regenerates analysis from the same saved cohort; all three `analysis.json` bytes are identical. Neither generated Markdown file was hand edited. Both retain complete method/effect, candidate improvement, selected-ID/rank-change, cost and stop evidence. The separate audit tables above add explicit within-teacher variability.

- `artifacts/long-probes-20261018/audit/report-1.md`: 246328 bytes; SHA256 `9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.
- `artifacts/long-probes-20261018/audit/report-2.md`: 246328 bytes; SHA256 `9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.
- Derived `artifacts/long-probes-20261018/cohort/analysis.json`: SHA256 `c78b287708a3e3226dee3a6b8f556b111202f6a1dc36d776b30d5dbeb64b49c0`.

## Costs, stopping and checker failures

The original cohort cost is outer7804.403474250s and inner7802.112551083s. Nested short search2140.461725334s and extra long search4169.761685458s are not added to enclosing time. Final raw cohort size is11496978848B; the analyzer reports the ledger’s earlier11496978362B boundary (486B final publication growth). Actual cohort peak RSS947650560B. All1040 short trials reached320; all1040 long trials ended by patience, with zero8000 ceiling hits. No failed scientific member, excluded losing candidate or numerical retry exists.

Audit per-process timers start before scientific imports and include IO/comparisons; explicit launcher-to-exit outer measurements cover reconciliation/report generation. They are separate stages, not a fabricated single enclosing clock. All five BLAS thread settings1/fallback0 precede scientific imports; torchCPU1/inter-op1; serial numerical work; process-scoped caffeinate; no dependencies, MPS/envelope or broad tests. Resource observations and actual RSS/elapsed fields are retained.

| Audit stage | Seconds | Peak RSS bytes |
|---|---:|---:|
| independent | 24.635506625 | 612777984 |
| replay-v1 retained failure | 179.029360917 | 735379456 |
| replay-v2 reuse/long/restore | 299.928512250 | 725762048 |
| reconciliation-v1 retained failure | 2.851459209 | 469303296 |
| reconciliation-v2 all analysis/renders | 124.495500875 | 1293041664 |
| reconciliation-v2 outer launch/publication | 124.767431708 | 1293041664 |

Two audit-checker failures were retained without changing raw evidence or repeating numerical fits. (1) After the complete short coordinator, history commitment passed a list to `canonical_hash`, which requires a mapping. The v2 checker wraps history in a mapping and reuses that existing short output read-only. (2) The added first-minibatch checker assumed a nonterminal parent cursor; teacher3/R1 had cursor2306=n_fit and correctly rolls into a cloned-generator permutation. The v2 checker handles this boundary; all sixteen pairs then pass. Both v1 scripts/reports/stderr/outer costs and small reproduction records remain in the complete inventory. Zero automatic numerical retries; two manual checker corrections.

## Evidence, preservation, decisions and remaining gates

- Independent arithmetic: `artifacts/long-probes-20261018/audit/independent.json`; its script/logs under `environment/long-probes-20261018/audit/`.
- Successful replay/restore: `artifacts/long-probes-20261018/audit/replay-v2.json`; all real raw replays under `artifacts/long-probes-20261018/audit/ranking-replays/`; failed checker v1 remains `replay.json`.
- Successful reconciliation: `artifacts/long-probes-20261018/audit/reconciliation-v2.json`; both renderings named above; failed checker v1 remains `reconciliation.json`.
- Final original-input/tracked/cost preservation: `artifacts/long-probes-20261018/audit/preservation.json`. Complete stopped audit inventory (including failures, scripts, stdout/stderr, all raw replay files, derived analysis and this report): `environment/long-probes-20261018/audit/inventory.json` and seal. It excludes only itself and seal to avoid a self-hash cycle.

New rulings/costs: retain the prospective teacher0/R0 representative and first eligible long-last checkpoint; charge both complete replay searches and failed checker overhead; correct audit serialization/epoch-boundary assumptions only in new checker versions; publish the unchanged deterministic report through root; recommend narrowing/timing investigation with no automatic follow-up. The prior eighteen chronological rulings and costs remain preserved by reference to DECISIONS D027 and the historical final report; none is silently revised.

Complete here: independent actual-prediction arithmetic, all planned paired contrasts/uncertainty, exact representative rankings/restore, same-parent joint bindings and deterministic unchanged-source reports. Final preservation/inventory must show completed before root acceptance; root still owns curation, publication and meaningful commit. Remaining scientific gates are unchanged: missing canonical SD/intrinsic-variance calibration/runtime evidence, accepted independent-confirmation counts, true freeze/independent confirmation and strong practical baseline suite. No T access/generation, refit, depth, transfer, new score, calibration, freeze or confirmatory claim.
