# Selective tree neurons: exploratory evidence and reproduction

The experiment does not establish that score-selected growth improves population risk. Across **16 independent teachers**, the mean H1 effect was **-0.00053643537353422854**, with a separately authorized 512,000-draw diagnostic interval of **[-0.001205029917807508, 0.00015605032904539594]**. The interval includes zero. The result is **exploratory and inconclusive**; the recorded design branch remains **calibrate**, without a freeze or protected-test confirmation.

H1 is normalized E risk(random) minus risk(top): a positive value favors score-selected growth. Static is risk(ordinary_upper) minus risk(static_partial): positive favors static partial. Risks use original-unit squared errors divided by the F target population variance (ddof=0). These saved diagnostics favor ordinary_upper over static_partial. Software reproduction succeeded; it adds no independent efficacy teacher and establishes no practical or architectural novelty.

## Evidence and decision

|Item|Retained status|
|---|---|
|Independent units|16 teachers; two optimizer repetitions per teacher; 262,144 shared E rows per teacher|
|Completed evidence|32 pairs, 256 role results, all 2,080 finite 320-update B6 probes; zero failed scientific pairs|
|Original producer|Failed after the user-requested pause; original raw ledger and seal unchanged|
|Continuation|Only teacher15/R0 and R1, the two never-started pairs; no completed pair retried|
|Canonical runtime evidence|Only the first eight-teacher batch accepted; second canonical runtime batch remains missing|
|Current design|calibrate; no final n_T, K_calc, planning effect, powered forecast or valid frozen protocol|
|Protected testing|No T generation/access, freeze or independent confirmation|

Repetitions and E rows remain paired within teachers. Repetitions are averaged within each teacher before the 16-teacher mean; the independent sample size is 16. All zero, tied and unfavorable outcomes remain included. The linked 16-teacher saved analysis is explicitly noncanonical.

|Diagnostic|Original 8,000 draws|Fixed 512,000 draws|
|---|---|---|
|H1 mean|-0.00053643537353422854|-0.00053643537353422854|
|H1 95% interval|[-0.0011768535168753332, 0.00014729129063262555]|[-0.001205029917807508, 0.00015605032904539594]|
|Static mean|-0.028761487521350156|-0.028761487521350156|
|Static 95% interval|[-0.036348131830166962, -0.021328100731346396]|[-0.036443426299342691, -0.021177012219863362]|
|Teacher SD interval|[0.00080910121071381261, 0.0019013079900325955]|[0.00080598569826168974, 0.0019077792680622309]|
|Maximum endpoint MC SE|0.00011265854071582265|1.6888484929210412e-05|
|Computational MC gate|Failed|Passed|
|Intrinsic-variance lower bound|0|0|
|Paired-row variance upper bound|0.0025833628781975265|0.0026009352397041142|

The unchanged endpoint tolerance is 5.1028567992981689e-05. The supplement's SD interval ratio remains 2.3670137971143101, above 2; the intrinsic-variance lower bound remains zero. The original 8k branch and failed computational gate are preserved. More draws reduced computational noise without increasing independent-teacher information or filling the missing runtime batch. Both calculations used seed 261022106; the supplement was prospectively authorized and ran exactly once. This report performs no resampling.

Student-t sensitivity interval: [-0.001298943959108825, 0.00022607321204036793]. Empirical whole-teacher bootstrap and t sensitivity do not establish nominal population coverage or unseen-tail behavior. The paired nuisance decomposition is conservative in expectation and remains unresolved. The complete original/supplement endpoints, component intervals, eight endpoint SEs, ten chunk vectors and index/RNG commitments remain in `joined-summary.json` and `supplement-512000.json`; original 8k endpoint details are also retained in the accepted MC bundle's `parity-8000.json`.

|512k endpoint|Monte Carlo SE|
|---|---|
|sd_lower|1.3243637880556849e-06|
|sd_upper|1.1189565762028015e-06|
|h1_lower|1.2966068861126319e-06|
|h1_upper|1.1605836287092872e-06|
|static_lower|1.6888484929210412e-05|
|static_upper|1.1773296024141737e-05|
|observed_teacher_variance_lower|2.1336743760982794e-09|
|nuisance_mean_upper|3.1489356523826838e-09|

## All eight method risks

Mean normalized E risk, averaging R2 within each teacher and then all 16 teachers:

|Role|Mean risk|
|---|---|
|keep|0.062332565434205067|
|top|0.061187969375958058|
|random|0.060651534002423818|
|greedy|0.060559400445550707|
|widen_upper|0.060630784736927852|
|ordinary_lower|0.061453763660644811|
|ordinary_upper|0.06048798904779086|
|static_partial|0.089249476569141006|

The complete 32-by-eight risk table, F variances, ordered input identities, per-pair effects and all outcomes are retained in `raw-audit.json`.

## Complete teacher distribution

|Teacher|H1 mean|Static mean|Within-R H1 variance|Paired-row H1 variance|
|---|---|---|---|---|
|0|-0.002800296086141644|-0.016345579802396055|1.5683316340120418e-05|0.0015362181445512115|
|1|-0.00013234732552138337|-0.0014209722827106916|6.5291467991630597e-07|0.00037818410143890095|
|2|0|-0.059161809406166055|0|0|
|3|-0.00015426320878011762|-0.026519000728957822|4.7594275166276314e-08|0.00037574444053704848|
|4|0|-0.044594384852121458|0|0|
|5|0.00088113620670922431|-0.048608932495724053|1.365132494950454e-06|0.0042175787354900539|
|6|0.002895707442848513|-0.046387723757908138|6.2058493729887093e-06|0.008500863029054017|
|7|0|-0.030713280841831159|0|0|
|8|-0.0015868806448172384|-0.015070944231914126|1.1725890429411894e-06|0.0027686562772887193|
|9|0|-0.030332460199494676|0|0|
|10|-0.0010689148862335219|-0.024253783153081236|2.2851580680232461e-06|0.00050355471103305668|
|11|-0.00073530385497930933|-0.011977992893790979|1.5930443710594556e-07|0.00040361593424651627|
|12|-0.0024159156195692434|-0.02365031764178141|2.0544091990088326e-06|0.0018679194500207292|
|13|0|-0.038522869564253603|0|0|
|14|-0.00067690428787300233|-0.0074089769824729288|9.1639882988171282e-07|0.00022572934307739102|
|15|-0.0027889837121899352|-0.035214771506998109|1.5556860293721503e-05|0.0017047728266288437|

Paired-row variance is computed after averaging repetitions on each E row, retaining their correlation. Within-R variance uses repetition means. No row or repetition is treated as an independent teacher.

## Independent saved-prediction audit

The audit read actual E predictions and the original F_y, E_y and ordered E row IDs for every pair. Original targets, row arrays, dtypes, source membership and consumed-file hashes matched. It independently recomputed finite original-unit losses and F population variance, all eight risks, random-minus-top and ordinary_upper-minus-static_partial row differences, teacher means and both hierarchical variance summaries. Saved h1_rows was a reconciliation target, not the sole calculation input. Reads were bounded per pair; the giant cumulative raw ledger was not decoded.

|Reconciliation target|Maximum absolute difference|
|---|---|
|h1_rows|0|
|static_rows|0|
|role_risks|1.3877787807814457e-17|
|teacher_h1|0|
|teacher_static|0|
|within_variance|0|
|row_variance|0|
|summary_means|0|

Saved score versus finite G-probe improvement association is descriptive within each fit. Every eligible unit is retained; constant cases are null and tied Spearman ranks are averaged. A 320-update probe does not establish final convergence or held-out generalization. Full eligibility, ties, growth and nonpositive-probe records remain in `raw-audit.json`.

|Teacher|R|Eligible|Pearson|Spearman|Nonpositive improvements|Growth|Near ties|
|---|---|---|---|---|---|---|---|
|0|0|64|null|null|64|16|0|
|0|1|64|null|null|64|16|0|
|1|0|64|0.053843724607656331|0.13602470532588865|60|16|0|
|1|1|64|0.3557945851645416|0.29878511286183962|60|16|0|
|2|0|64|null|null|64|16|0|
|2|1|64|null|null|64|16|0|
|3|0|64|0.18514486572286454|0.22769304454842815|62|16|0|
|3|1|64|null|null|64|16|0|
|4|0|64|null|null|64|16|0|
|4|1|64|null|null|64|16|0|
|5|0|64|0.33717572919516808|0.24016017963011405|57|16|0|
|5|1|64|0.26218579653533497|0.32345666919248522|44|16|0|
|6|0|64|0.06026913260666459|0.21303501999944663|62|16|0|
|6|1|64|null|null|64|16|0|
|7|0|64|0.29389728565541612|0.25354644822575295|59|16|0|
|7|1|64|null|null|64|16|0|
|8|0|64|-0.041636385692572539|0.089646446390066206|60|16|0|
|8|1|64|null|null|64|16|0|
|9|0|64|0.050451744157029653|0.23091715715399388|59|16|0|
|9|1|64|null|null|64|16|0|
|10|0|64|0.27809035320591408|0.18755389558175622|63|16|0|
|10|1|64|null|null|64|16|0|
|11|0|64|0.1740417277424105|0.27405341123706095|30|16|0|
|11|1|64|null|null|64|16|0|
|12|0|64|0.032287207489239911|0.11935247900657214|63|16|0|
|12|1|64|0.64541027878577018|0.16736717389614822|49|16|0|
|13|0|64|null|null|64|16|0|
|13|1|64|null|null|64|16|0|
|14|0|64|0.032931826765721949|0.15345318729416416|63|16|0|
|14|1|64|null|null|64|16|0|
|15|0|64|0.21125571773917048|-0.010763236155857583|35|16|0|
|15|1|64|-0.043458748287978952|0.037510779116351242|63|16|0|

## Clean locked reproduction and durable recovery

A disposable checkout at commit `7a63114972e0b1230b2e33c3e881a6c55b4e92a0` used a newly created, separate .venv and Python 3.12.13. The exact cached installation command was `/Users/artemlegotin/.local/bin/uv sync --offline --frozen --cache-dir /Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.uv-cache --python /Users/artemlegotin/Documents/git_alter/adaptive-tree-neurons/.uv-python/cpython-3.12.13-macos-aarch64-none/bin/python3.12` (exit 0). Generated editable egg-info was preserved and hashed outside execution source. The strict source guard was not waived. Imports resolved to `/private/tmp/h1-wp09-clean-7a63114/src/treeneuron/train.py`; actual dependencies, source map, command/environment records and install metadata are retained in `clean-readout.json` and `clean-restoration.json`.

Execution315 SHA `07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c`, package SHA `d8e5c0c6549f3b1948ee86b39a2d3234ebe5230e7bb09c13ebfc4351c2cf891c`, and original receipt SHA `4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42` remained unchanged. The engineering full config key was `154d7675190c5658dc60cbbfb536976701b1a08ace3f41705b45da58d5ccd1e0`. All 12 regenerated F/G/V/E arrays matched before fitting. This recipe's teacher0/R0 is a same-seed engineering check, with no added efficacy unit.

Separate fresh processes ran Appendices A and B, so their float64 default did not contaminate training. Training used CPU float32, scoring CPU float64, CPU1/inter-op1, all five BLAS settings1 and fallback0. Numerical jobs were serial and long workers had process-scoped caffeinate assertions. Native/sandbox MPS visibility matched the original CPU reference; no RNG field was dropped.

|Check|Scope|Result|
|---|---|---|
|Appendix A|Fresh standalone reference process|Passed|
|Appendix B|Separate fresh standalone reference process|Passed|
|Full paired replay|All eight roles and all 65 keep/unit probes|Completed|
|Scores and allocations|Full score table, selected IDs/contrasts and all probe outcomes|Exact|
|Eight selected role states|Topology, model tensors, all Adam moments/ages/groups, scheduler, full RNG/sampler/cursor/next rows; F/G/E predictions|Exact; maximum difference 0|
|Parent and all 65 probe states|Complete numerical states, next rows and G predictions|Exact; maximum difference 0|
|All 11 role stages|Executed row hashes, losses, selected updates and G evaluations|Exact|
|Actual growth recovery|Durable interruption, reload unfinished stage, linked retry|Exact; maximum difference 0|

|Role|Selected global update|Executed continuation updates|Selected G MSE|State/predictions|
|---|---|---|---|---|
|keep|5300|500|0.059498702777318314|Exact|
|top|5350|550|0.05903996065828776|Exact|
|random|5300|500|0.059498702777318314|Exact|
|greedy|5300|500|0.059498702777318314|Exact|
|widen_upper|5300|500|0.059498702777318314|Exact|
|ordinary_lower|5475|500|0.072973276560277439|Exact|
|ordinary_upper|3375|500|0.073794995526099944|Exact|
|static_partial|1100|500|0.11985769204516263|Exact|

The real checkpoint event was appended before a controlled KeyboardInterrupt at global update 5325, after growth insertion at 5300. The interrupted attempt remains retained. `checkpoints/last.json` was reloaded with unchanged expected hashes while its stage was unfinished. A linked retry continued the original allowance without extra updates: 25 interrupted-attempt updates plus 525 retry updates exactly matched the uninterrupted 550 update events, including full row IDs, loss and learning rate. Selected full state and F/G/E predictions matched the uninterrupted clean top continuation.

Durable last-pointer SHA: `3b2f15808874ce6ee4c289cb04543e99d957b93459eadd929bc3621000cc7062`. The interrupted/retry paths, every state comparison, exact next-row hashes, topology and full sampler/RNG fields remain in `clean-readout.json` and `saved-state-comparison.json`. Incidental IDs, timestamps, clocks and serialized checkpoint hashes were excluded from numerical equality targets.

## Actual costs and retained failures

|WP09 work|Perf seconds|Peak RSS bytes|
|---|---|---|
|Independent raw prediction audit|4.9259654590277933|384106496|
|Clean replay/recovery process, enclosing|302.39783216605429|812630016|
|Full paired replay, nested|271.40308629203355|Included in process peak|
|Actual growth recovery, nested|4.9543214170262218|Included in process peak|
|Saved parent/probe/stage comparison|11.418686792021617|438616064|
|Appendix A, separate process|8.4058058750233613|Not separately recorded|
|Appendix B, separate process|3.9405535419937223|Not separately recorded|

The clean worker's UTC span was 2026-10-17T21:55:06Z to 2026-10-17T21:59:49Z. Nested times must not be added to enclosing costs. Restored reference subset: 968 files / 507474141 logical bytes; initial clean physical du was `1305972	/private/tmp/h1-wp09-clean-7a63114` KiB. Initial disk free was 40117219328 bytes and a 10 GiB reserve was verified. The accepted technical WP09 new-only inventory contains 1,030 files / 481,577,468 bytes. This presentation revision has its own new-only inventory and does not change those original bytes.

Historical costs remain separate and unchanged. The original producer enclosing 47,057.135737208 perf seconds includes a documented 40,020.992005584-second controller suspension; derived outside-suspension time7,036.143731624 seconds is not CPU utilization. Original sleep-affected pair clocks remain retained. The two continuation pairs cost251.615493833 and176.384032458 wrapper seconds. Two operational setup failures preceded numerical work: tuple-versus-JSON-list metadata comparison (21.683983625 seconds) and str-versus-Path API argument (1.295538625 seconds). They caused no completed-pair retry or completed bootstrap. Original-plus-continuation indexed retention was12,970,486,988 bytes; actual producer footprint12,953,209,885 bytes. The saved MC supplement worker used .635482459 seconds and347,013,120 bytes peak RSS. The one canonical runtime batch measured406.949557520875 seconds/teacher; no valid powered full-study forecast or cost curve exists.

No unexpected scientific or numerical WP09 failure occurred. The controlled KeyboardInterrupt is an intentional retained recovery event. One report-renderer AST preflight found an unterminated string before rendering; its failed script/error were retained, and only the closing quote was corrected. No numerical calculation was repeated for that correction. This v2 revision changes presentation only and preserves the accepted original technical report, its two outputs and its inventory.

## Scope, limitations and saved details

Completed: independent raw risk/variance/source audit; all prescribed roles and probes with unfavorable/zero/tied cases retained; finite-probe association where defined; clean frozen environment/source/import checks; isolated A/B references; actual full engineering replay; durable growth interruption/reload/linked recovery; exact complete-state and row-stream comparisons; saved-input report regeneration and hash inventories. Absent: independent powered confirmation, valid freeze, protected T, transfer/depth/hard-routing evidence, practical superiority and architectural novelty. These absences are limitations, not hypothesis rejection. Existing closest-prior-art limits still apply.

The original full scientific config key9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073 remains distinct from scientific recipe4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a. The operational continuation, joined E diagnostics, fixed saved-MC supplement and engineering reproduction retain their separate lineages and scopes. No changed statistical gate, extra science, inference eligibility or experimental authority follows from report formatting.

All detailed records below are supplied unchanged in `report-inputs/`. Their exact bytes are verified before rendering:

|Saved record|Detailed content|SHA256|
|---|---|---|
|raw-audit.json|32 pair risks, 16 teacher summaries, input hashes, growth/ties, probe associations and audit differences|`e89778c48de957dc3fb1ea1ec3c401728420fe5666f19f2aa855a57030ef4dea`|
|clean-readout.json|Environment/dependencies, source315, exact role states, full recovery identities and timings|`3331c1f565a927e67aa14466be61a6ac2c02f1e44a76362ca4f35694f1472702`|
|clean-restoration.json|Restored file hashes/bytes, offline lock command, metadata relocation and disk accounting|`de221c8076135e7a253fbacd3e36c98dfdead44532d959da1f3a4767914f09cc`|
|appendix-invocations.json|Exact fresh Appendix A/B commands, statuses and durations|`1a93345a9abbb047dc91a835abb6007e252f97028d4c1cd8c94a832c39daf443`|
|saved-state-comparison.json|Parent/all65 probe state comparisons, 11 stage streams and recovery update events|`2a1cebf4fe689bddd4b900af50a056094a43db723dac51c2cfde6df1383c6205`|
|joined-summary.json|Unchanged 8k summary, uncertainty/decomposition and canonical-status limits|`351c2578bde77e225a06478e212ac4a0b2bf60313a7f1daa535b6df5ea578fa1`|
|joined-design.json|Original calibrate branch, missing runtime batch, unresolved design/forecast|`fd6844d27846b871259f5b54f5785fe57a6d99adac885e12e1df5be0eae55c2a`|
|supplement-512000.json|Fixed supplement endpoints, SEs/chunks, component intervals and RNG/index commitments|`dd5781c2b2538dfe80362c02bc771abcda1a1a14f94931b72e4ac92c1f7cbbbd`|

To regenerate this reader report, place `render_report.py` beside `report-inputs/` and run `python render_report.py OUTPUT.md`. The renderer uses only the Python standard library and the saved hash-bound inputs; it neither imports numerical packages nor runs experiments. Two independent renders are byte compared, with no generated timestamps. The accepted original technical report remains `.superpowers/sdd/2026-10-16-selective-tree-neurons-evidence/scientific-wp09-report.md`. **Numerical STOP.**
