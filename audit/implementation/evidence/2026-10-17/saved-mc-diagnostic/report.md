# Prespecified saved-scalar Monte Carlo supplement — STOP

The original joined8,000-draw calculation was reproduced from the accepted16-teacher scalars within **2.168404344971009e−19**, below the declared1e−12 absolute tolerance. Its endpoint MC value and variance/decomposition quantities matched exactly. Only after that gate passed, **exactly one512,000-draw bootstrap** ran with the original seed261022106 and the original ten-chunk endpoint method. No fit, new observation, seed/count search, source change, power calculation, replay, freeze, or T access occurred. Numerical work is stopped.

The supplement's computational endpoint MC gate passes: maximum endpoint SE is **.000016888484929210412**, below the unchanged **.00005102856799298169** gate. The uncertainty across independent teachers remains unresolved: SD interval ratio is **2.36701379711431**, above2; H1's interval spans zero; the intrinsic-variance lower bound remains0. The original8,000 output, its failed MC gate, source `calibrate` branch, failed original producer and missing second canonical runtime batch are unchanged. The supplement creates no canonical or freeze eligibility.

The predeclaration was recorded before the fixed cohort finished, at23 completed pairs. Its SHA256 is `20f7d30d536ba1b33e939973e4ef9db0db639ebe197e353980312e7dc54e71fd`. The actual command was `.venv/bin/python environment/h1-v2/pilot-saved-mc-supplement/launch.py`; it started and ended at native integer-second timestamp2026-10-17T21:50:37Z. Fresh worker66498 used a separate CPU NumPy process with all five BLAS thread settings1 and fallback0. Scoped `caffeinate -i -w66498`, PID66499, exited normally with the worker. The launcher, script, analysis config, environment, all input hashes, exact sampling-index commitments, RNG states, endpoint chunk vectors and outputs are retained.

The immutable artifact bundle is `artifacts/h1-v2/pilot-saved-mc-supplement/892290e2c5914ef0815b9edd6cca6c3d/`; scripts/invocation/readout are under `environment/h1-v2/pilot-saved-mc-supplement/`. Inputs were accepted `joined-scalars-and-risks.json` SHA `7324942c0ee22b18b3fe76e22ba96c14be2b19d602e096c480ef5bda2a9bbff4`, actual joined8k summary SHA `351c2578bde77e225a06478e212ac4a0b2bf60313a7f1daa535b6df5ea578fa1`, design SHA `fd6844d27846b871259f5b54f5785fe57a6d99adac885e12e1df5be0eae55c2a`, member manifest SHA `7f42ae967496a7e6a36095175617567454b7384bd74a81121d9be23681f7ab92`, and root's independent continuation verification. Source order0–15, R2/N262144, source teacher means, within-teacher H1 variances and paired-row variances were checked explicitly. No giant raw or row arrays were decoded.

Full config key `9152a59bb3180da35f780b44a78d9eee59f7efa3e1af9677e4acf307a1bbb073`, scientific recipe `4579ba87ffc6be1678b935c5297e849e0a6a53adb0e055ecedea84596ab29c6a`, execution map `07366c506b309cce841f08d54f25d718e246f75daea5696d6542c300cfadaf6c`, and runtime receipt `4cfb4983fbe8d152decf7e621dc79b2e77b7f4e3825cb379534ce6a4e05e9e42` are unchanged. The source summarizer file was checked against SHA `0c61e83d1e986433953270225833843ce236061582d4cfc6dd329d3580eb842e`. The analysis used Python3.12.13/NumPy2.5.4. Original/continuation inventories were bound to their accepted hashes without rereading the12GB scientific archive.

|Diagnostic|Original8k reproduction|Fixed512k supplement|
|---|---|---|
|H1 mean|−.0005364353735342285|unchanged|
|H1 interval|[−.0011768535168753332,+.00014729129063262555]|[−.001205029917807508,+.00015605032904539594]|
|Static mean|−.028761487521350156|unchanged|
|Static interval|[−.03634813183016696,−.021328100731346396]|[−.03644342629934269,−.021177012219863362]|
|SD interval|[.0008091012107138126,.0019013079900325955]|[.0008059856982616897,.001907779268062231]|
|SD interval ratio|2.349901303886576|2.36701379711431|
|Maximum endpoint MC SE|.00011265854071582265|.000016888484929210412|
|Computational MC gate|failed|passed|
|Intrinsic variance lower|0|0|
|Paired-row variance upper|.0025833628781975265|.002600935239704114|
|Observed teacher variance lower|.0000006546447807956678|.0000006496129458024406|
|Nuisance mean upper|.000002807509524797399|.0000028139663069881487|

H1 is normalized risk(random)−risk(top), positive favoring top. Static is risk(ordinary_upper)−risk(static_partial), positive favoring static. The H1 result remains mixed and uncertain; these saved diagnostics favor the ordinary baseline over static partial. Empirical whole-teacher bootstrap/t sensitivity do not validate nominal population coverage or unseen tails. More MC draws reduce computational noise without increasing independent-teacher information.

All eight original source endpoint SEs are reported below. Their maximum remains the source-defined gate statistic. The row-variance upper bound is reproduced/reported but was not added to that eight-entry vector; this preserves the source method.

|Endpoint|8k SE|512k SE|
|---|---:|---:|
|SD lower|6.4677865859751984e−6|1.324363788055685e−6|
|SD upper|6.119717453606740e−6|1.1189565762028015e−6|
|H1 lower|4.460204449823887e−6|1.2966068861126319e−6|
|H1 upper|9.756775026625548e−6|1.1605836287092872e−6|
|Static lower|1.1265854071582265e−4|1.6888484929210412e−5|
|Static upper|9.482236441152830e−5|1.1773296024141737e−5|
|Observed teacher variance lower|1.0452282834538485e−8|2.1336743760982794e−9|
|Nuisance mean upper|1.9507986135480803e−8|3.1489356523826838e−9|

The decomposition retained the same conservative nuisance allowance `within/R + row_variance/N`. Observed teacher variance2.047667065908811e−6, optimizer-repetition variance2.881220439614037e−6 and mean nuisance allowance1.445970545857149e−6 are unchanged. Additional two-sided component quantiles from these same fixed draws are informational, without extra gates:

|Component95% interval|8k|512k|
|---|---|---|
|Observed teacher variance|[6.546447807956678e−7,3.614972078613604e−6]|[6.496129458024406e−7,3.6396217356492288e−6]|
|Nuisance mean|[3.8373463202564787e−7,2.807509524797399e−6]|[3.63050971180757e−7,2.8139663069881487e−6]|
|Mean paired-row variance|[.0005107262728112522,.0025833628781975265]|[.0005079567032184324,.002600935239704114]|

`parity-comparison.json` records every comparison to the source8k output: H1/static means/intervals, SD, endpoint MC, repetition/row/intrinsic variance, observed variance, nuisance vectors/bounds and t sensitivity. All differences are0 except H1/SD interval rounding of at most2.168404344971009e−19. `parity-8000.json` and `supplement-512000.json` retain all endpoints, eight SEs, all ten chunk vectors and full RNG/index commitments. The512k index-array SHA is `b94d3c25772f171a6adcaa00d2c0be2ebdb13d4d481b25b75dda6d6c1ff6e824`; the manifest confirms one supplement call and two total calls including parity. No further resampling occurred.

Outer invocation cost was.695944791 perf seconds; worker enclosing.635482459 seconds. The nested8k/512k computations were.004522750/.165463916 seconds and must not be added to the enclosing costs. Actual peak RSS was347,013,120 bytes, below6,448,447,488. Final native availability62%, swapouts4,786,629 unchanged, free disk40,643,784,704 bytes. Conservative accepted original-plus-continuation indexed retention was12,970,486,988 bytes; the guard additionally reserved1MiB for new output and10GiB free disk. No guard fired.

The elapsed guard continued from the first continuation wrapper's514707018090000ns clock, without reset: its final current span was1,035.552869916 seconds, below the accepted7,363.856268376-second allowance. This conservatively includes intervening adjudication gaps as well as work. Original user-pause and sleep-affected clocks remain immutable and separate; no source clock or budget was patched. This short saved-only measurement does not establish future walltime stability.

The script SHA is `893c99ec57908470c970d22706ef5923587191abaa45593b8109dd2d6bb12fe7`; analysis-config SHA `2d0b85242011c0adf087fa5a3ba13a499f6e67877af5f3b0f2e7ed44e16d58d0`. The new-only inventory at `environment/h1-v2/pilot-saved-mc-supplement/inventory.json` binds the new scripts, config, invocation, readout, parity/supplement outputs, manifest and this report, with a complete byte/SHA reread. No original output was overwritten. Root validates these diagnostics before authorizing any further work.
