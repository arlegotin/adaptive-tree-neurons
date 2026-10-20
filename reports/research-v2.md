# Research v2: negative growth result and unfinished practical comparison

The tested growth chooser has not delivered a reproducible advantage over matched random placement. In the completed sixteen-problem joint-selector study, competent ordinary continuation performed better on every problem average. A separate native-backbone screen supplies no reason to move this synthetic study to a stronger parent. Those finite mechanism branches close negative.

The broader practical comparison remains unfinished. Existing strong anchors lead the grown model on all five observational validation splits examined so far, but these development scores do not establish a tuned or protected-test ranking. Native baseline runs continue without a two-hour wall cap. This is a current development report, not a completed confirmation or full-program release.

## What the evidence supports

| Question | Current evidence | Decision |
|---|---|---|
| Can we insert affine switches without changing predictions? | Algebra, gradients, native insertion/state and replay checks support the implemented one-growth setting. | Supported software invariant. |
| Can added gates learn useful structure? | Planted M1 tasks improve under growth, including random growth. | Supported on those controls; ordinary extra-capacity superiority does not follow. |
| Does informed placement beat random? | M3 did not pass its final development gate; the historical cheap score also lacks a reliable lead. | Tested selector branch retired; no calibration or confirmation admitted. |
| Does growth beat competent ordinary continuation? | Keep wins all sixteen M3 teacher-average E comparisons with joint growth. | Negative for this treatment/population; no universal impossibility claim. |
| Is a stronger parent a measured remedy? | B0’s V-selected ordinary family wins all four teacher-average E comparisons. | B0 screen failed; no residual follow-up admitted. |
| Is the complete method practically superior or cheaper? | Single-seed native-anchor V survey only; remaining native/source/support/search/test work is incomplete. | Unresolved. No quality–cost win. |

## Main synthetic result: M3

The fixed d16/h64/m16 primary population used F2306/G577/V577/E65536, sigma 0.1 and current contrasts (epsilon .05, tau 1). Sixteen fresh teachers each had two paired training repetitions and ten final roles: 320 final models. Repeats were averaged within teacher first. All 3,089 queue entries were terminal; they include preparation and 2,080 short probe/keep trials and are not 3,089 final models. There were no failed or missing planned entries.

The competent parent used AdamW LR .005, weight decay .001 and the normal full-state G-best selection. All warm arms received a fixed 8,000-update continuation with strict G selection at update zero and every 25 updates. This explicitly supplies additional training opportunity after the earlier replay finding. Fresh upper models have different inheritance and total optimization history; their comparison is not a placement-only estimate.

| Final role | Mean V NMSE | Mean exploratory E NMSE |
|---|---:|---:|
| keep | 0.05115488 | 0.05077368 |
| warm_upper | 0.05580169 | 0.05479102 |
| fresh_upper | 0.05840021 | 0.05867650 |
| random | 0.05965374 | 0.05941502 |
| local_singleton | 0.06041583 | 0.05983059 |
| short_singleton | 0.05972584 | 0.05986615 |
| joint | 0.05984766 | 0.06015804 |
| pure_span | 0.06121578 | 0.06037839 |
| unconditioned | 0.06099278 | 0.06044164 |
| top | 0.06159369 | 0.06090511 |

Ordinary continuation has 15.6% lower arithmetic mean E error than joint growth. Joint growth has 1.25% higher arithmetic mean E error than matched random. The teacher log-risk comparison remains uncertain: the descriptive E interval for log(random/joint) is approximately [−.0381, +.0115], with mean −.0133. It is not a confirmatory interval. Keep wins all sixteen teacher E averages; no tested gated role clears a 2% E gain over competent keep on any teacher average.

The exact joint ridge implementation agrees with its dense reference. At a temporary solved-step fraction of 1/16, predicted F loss reduction averaged about 6.27%, actual F reduction 5.06%, and actual G reduction 0.66%. At a full solved step, the local model predicted 57.94% F improvement while actual F loss increased about 149.38%; about 61.75% of diagnostic gate activations saturated. These are solved parent-plus-gate diagnostic steps, not observed Adam gate-only trajectories. They do not establish an actual newborn/saturation trigger or a successful alternative training recipe.

Gate B closed negative after the single predeclared eight-to-sixteen precision extension. The optional regularization correction was declined, with zero correction fits; no direction/newborn grid, overgrow–prune or residual redesign followed. A hypothetical remedy remains untested rather than being assigned a favorable or unfavorable result. [Complete report](research-v2-M3.md), [gate disposition](../audit/research-v2/M3/gate-B.json).

## Native backbones on fresh synthetic teachers: B0

Four new teachers, two paired repeats and nine fixed native recipes produced 72 complete fits. Five one-layer coordinates competed within their family on V; MLP-D, ResNet-D, RealMLP-TD and embedded TabM competed within the backbone family. The small h64/LR .005/WD 0/dropout 0 coordinate won the ordinary family in all eight repeats, and RealMLP-TD won the backbone family. No best seed was selected.

After repeat averaging, backbone E error was 4.7%, 31.5%, 24.4% and 35.3% higher on the four teachers. Zero of four achieved the fixed 5% improvement screen, against a required three. No expansion or stronger-parent/residual-structure follow-up is admitted. Four teachers, not 72 fits, are the independent units.

Native quantile preprocessing and fixed presets are part of this comparison. The grid does not isolate optimizer, dropout or weight decay individually and is not full HPO. The competent F-standardized M3 continuation was not rerun on these teachers. B0 cannot establish a universal one-layer ceiling or a ranking across all tabular tasks; its means are not paired before/after improvements over M3. [Report and all nine recipes](research-v2-B0.md), [pre-data declaration](../docs/research/v2/B0-native-primary-backbones.md).

## Native practical anchors: development only

The original fixed survey contains 115 completed cells out of 120 planned, using one seed on the designated primary V splits. The separate fixed follow-up adds the completed Airfoil NODE anchor, bringing native engineering completions to 116. Concrete is running, with Parkinsons, CCPP and Bike queued serially. No missing prediction is scored as a model loss. The original survey and follow-up remain separate immutable records.

| Observational primary V split | Grown cheap | Matched random | Lowest observed existing anchor | Existing-anchor NMSE |
|---|---:|---:|---|---:|
| airfoil | 0.092186 | 0.109896 | P06 / native_anchor | 0.073302 |
| concrete | 0.209199 | 0.202016 | P08 / native_anchor | 0.129860 |
| ccpp | 0.049448 | 0.049448 | P08 / native_anchor | 0.027205 |
| parkinsons | 1.246646 | 1.246646 | P08 / native_anchor | 0.738719 |
| bike | 0.370625 | 0.370625 | P07 / native_piecewise | 0.169430 |

Best available V anchors are descriptive, selected on the same V data; they are not protected-test champions or full tuned-family winners. P13 here is mean only, not its complete ridge/spline search. The grown model beats random on Airfoil, reverses on Concrete and ties on the other four dataset coordinates. Airfoil’s later full NODE V NMSE is .107493, above grown .092186 and RealMLP .073302; this preserves both favorable and unfavorable comparisons.

Energy remains separate simulation transfer. A fully static gated anchor has V NMSE .009856, the lowest observed there. Only two geometry groups are in primary V, from twelve in the whole table. This is a useful representation lead with a different capacity/initialization scope, not evidence for adaptive placement or observational population superiority. [All 120 original cells](research-v2-primary-anchors.md), [fixed NODE follow-up](research-v2-native-anchors-follow-up.md).

All fourteen mandatory families remain in the inventory: grown, matched random, warm ordinary growth, MLP, ResNet, static gating, RealMLP-TD, TabM, CatBoost, XGBoost, constant and linear LightGBM, NODE and simple references. Final common-source/software/support/workload qualification, the fixed primary HPO searches, global model/source/protocol freeze, secondary robustness searches and T evaluation remain unfinished. Earlier native runs do not silently migrate into current-source production eligibility.

## Verification, costs and claim boundaries

| Evidence level | Established here | Limit |
|---|---|---|
| Software/numerical reference | Insertion, gradients, optimizer/state checks and exact local ridge references; source-specific test receipts. | No implication of predictive benefit; whole-program software review is still underway. |
| Public arithmetic/report reproduction | Compact M3 tables and B0 canonical report inputs/operators are versioned; B0 report and four tables replay byte for byte with the standard library. | Saved scalars do not authenticate missing raw history. |
| New raw prediction arithmetic | M3: 640 V/E vectors, 7,680 metrics, max discrepancy 5.55e-17. B0: 288 F/G/V/E vectors, 1,440 metrics plus 72 F variances, max 1.11e-16. Original survey: 115 vectors, 805 metrics, exact agreement. | Available source/byte records are authenticated at the named audit scope; training and inference are not independently repeated by an arithmetic audit. |
| Protected confirmation | None. | Synthetic T/root and C0 were not created; prepared transfer T remains sealed. Recorded API-access audits do not prove cryptographic filesystem nonaccess. |

The complete repository software run at `61221f4` finished with 2,217 passed,
42 failed, 369 setup errors and six skips. This is **not a green suite**. Of the
411 failing/error cases, 333 explicitly or through their retained doctor receipt
hit the legacy memory-pressure guard, 59 reported duplicate NODE dependency
metadata, and eleven inherited a caller OpenMP setting. Three aggregate backend
failures lack their original detailed probe records. An isolated timing check
reproduced resource-guard suppression before any fake-model prediction. All
original names and confidence limits
are retained in the [failure inventory](../audit/research-v2/software/whole-suite-61221f4/summary.json).
Seven separate native software profiles at `4f19a05` passed 202 checks with no
skips. They cover all fourteen families, but do not supply final-source workload
qualification or erase the broad-run failures. Native guards and historical
expected results remain intact. [Verification receipt](../docs/research/v2/WP11-whole-suite-verification-receipt.json).

Two test-launcher isolation corrections passed 57 affected checks and 27 main
checkout integration checks. Fake-model timing tests now supply explicitly
labeled guard-counter inputs; fourteen checks passed, including rejection under
simulated pressure. The actual hardware-dependent training test and production
guard remain unchanged. These scoped recoveries do not replace the original
whole-suite result. [Follow-up receipt](../docs/research/v2/WP11-software-follow-up-receipt.json).

Raw arrays, recoverable states and append-only ledgers are retained locally with source-bound inventories. Historical missing raw archives remain missing. Reproduction does not depend on remembering a conversation; [commands, exact identities and portability limits](../docs/research/v2/reproduce.md) are versioned. Each study’s held source is immutable, and old reports and expected historical hashes remain intact.

The user removed the two-hour native quality limit. Current native runs use normal stopping, one serial NODE worker, live resource admission, a 4 GiB owned-root RSS guard and saved recovery state. Completed Airfoil NODE took about 5¼ hours at 1.83 GiB sampled root RSS. The largest measured TabM engineering shape used about 3 GiB; an initial 2 GiB stop and exact-request retry are retained. Available-memory proxies are heuristic and the checks are not whole-machine RAM guarantees. No model, F sample or effective batch was shrunk. [Memory evidence](research-v2-resource-probes.md).

M3’s containing scientific run took about 2½ hours; B0’s enclosing jobs about 24.67 minutes. These contended clocks, candidate work and shared infrastructure are not a formal deployment or search-cost win. Soft gating evaluates both branches and supplies no automatic conditional-compute saving. Formal quality–cost and hard-routing claims remain unsupported.

Function-preserving growth, parent-tangent geometry and overgrow/prune ideas have close prior art, including Net2Net, TINY, Gauss–Newton morphisms and Firefly. The tested contribution is this parameterization’s precisely checked estimator and empirical evaluation, including its negative result. No first-growth, first-residual-aware or general-new-principle claim is made. [Research design and prior-art boundaries](../docs/superpowers/specs/2026-10-18-research-v2-design.md).

## Disposition and continuation

The mechanism result is a scoped negative development result. The primary selector and B0 remedy screens are closed; their finite allowances are exhausted or failed, rather than extended until favorable. A future method would need a named new falsifiable question, a new finite design and fresh affected protected evidence.

The remaining fixed native baseline inventory continues. Production practical comparison and protected confirmation remain pending their explicit prerequisites; neither is called complete here. This report is a curated cross-study account backed by the linked canonical inputs and receipts. Complete production protocol/audit/report integration and final whole-branch review are still outstanding. [Current status](../docs/research/v2/status.md), [execution record](../docs/research/v2/execution.md), [implementation plan](../docs/superpowers/plans/2026-10-18-research-v2.md).
