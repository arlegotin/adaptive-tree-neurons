# Longer validation probes for neuron allocation

The user approved a focused diagnostic: give the existing validation-probed
selector longer candidate training, compare it with the cheap score, random
placement and ordinary capacity, measure one full comparison first, and then
execute a small fixed set of fresh problems. Work stays on current `main`, uses
subagents and meaningful commits. This description makes routine implementation
and resource choices concrete; it changes no historical result.

## Question and claim boundary

Does a validation-probed selector with the normal continuation stopping rule
find useful placements that the fixed 320-update reference misses? A larger
opportunity is not guaranteed to improve held-out prediction. This restricted
single-unit greedy selector is neither an oracle nor full ANT reproduction.
Success can justify investigating a cheaper score; failure weakens only this
allocation procedure in this recipe. The old pilot remains inconclusive.

This is a new exploratory diagnostic, never confirmation or canonical variance
calibration. Do not create a frozen protocol or access/generate T. F fits weights
and transformations, G selects checkpoints and ranks probes, V stays unused,
and fresh E observations evaluate completed procedures. E outcomes never select
budgets, candidates, exclusions or additional runs. No refit, tuning, deeper
growth, new initialization or modified score is included.

## Fixed scientific coordinates

Keep the previous d16/h64/m16 Gaussian regime population, F2306/G577/V577,
E262144, noise0.1, contrast epsilon0.05, tau1, CPU float32 eager training and
CPU float64 scoring. AdamW lr0.005/wd0, batch128, base and final continuation
ceilings8000, G/checkpoint interval25, patience20 and earliest strict G best.
Fresh ordinary controls receive the original two-stage allowance.

For each independent problem/repetition, fit ONE fresh parent and ONE score
table. Run separate exhaustive short and long probe coordinators from that
same original parent, with every eligible unit plus keep, identical per-ID
contrasts, allocation seed, old optimizer states and minibatch stream:

- Short: fixed320 updates; retain the old rejection of an unexpectedly shortened
  trial. This default and legacy caller behavior stay unchanged.
- Long: ceiling8000, normal G patience20; accept a genuinely completed stage
  stopped by patience or ceiling. Record allowance, executed/selected updates,
  stopping reason and ceiling status. Incomplete or unexplained short returns
  fail. Never change the trainer's global stopping semantics.

Keep actual normal evaluation semantics: the fixed short stage has an endpoint
evaluation at320, while an uninterrupted long stage uses its25-update grid and
own endpoint. These are complete procedure comparisons, not a claim that the
long selector's evaluated checkpoints strictly contain the short selector's.
Retain actual G curves so this limitation can be inspected.

Rank each complete set using existing G loss ordering/ID tie break, fixed
m_eff=min(16,eligible count), without a positive-improvement filter. Independently
reload the ORIGINAL parent for each joint final continuation. A probe's learned
weights never become joint initialization. Both selectors use their original
moment contrasts. Retain zero-growth problems and all failed/losing candidates.

Retain the eight existing final roles and add `greedy_long`: keep, top, random,
greedy (short), greedy_long, widen_upper, ordinary_lower, ordinary_upper and
static_partial. Explicitly audit the extra warm fork's parent identity, count,
contrasts, optimizer/sampler state and shared executed-row prefix. Existing
parameter brackets remain 1693/1711 around 1697 at m_eff16; calculate actual
counts when eligibility differs. Search cost is extra and fully charged, not
described as an equal-cost practical comparison.

## Fresh samples and ordering

First execute one separate engineering cost problem, repetition0, with all nine
roles and both exhaustive searches. Its root teacher/data/optimizer/allocation/
contrast/reference seeds are261030100/101/102/103/104/105. Preserve E predictions
and all costs; no cost-problem accuracy decides the recipe or cohort size.

Then execute exactly eight fresh independent teachers, two paired repetitions
each, ordered teacher0..7/R0..1. Root seeds are261031100/101/102/103/104/105;
existing teacher_inputs/repetition_config derive per-teacher/inner seeds.
These namespaces are separate from cost, earlier calibration and the old pilot.
Eight/R2 is a bounded diagnostic coordinate, not a power or coverage promise.
Never stop or add teachers based on observed effect signs. A resource failure
retains the incomplete fixed cohort and blocks a complete-result claim.

Before cohort fitting, save the measured cost forecast and a hashed study lock
binding exact resolved config, seeds/counts, mode/ceilings, metrics, source/lock/
receipt, analysis and failure policies. This is a diagnostic lock, NOT a
FrozenProtocol. Existing PilotConfig engineering fields may be reused solely
for data/probe helpers; this entry point never calls canonical `calibrate` or
claims genuine variance-pilot readiness.

## Resources and provenance

Use a fresh validated narrow CPU128 receipt after source changes. Do not relabel
old checkpoints/receipts or renew unrelated MPS/envelope matrices. Fit all new
parents under the final committed implementation. Source stays fixed throughout
measurement and cohort. Serialize numerical jobs with CPU1/inter-op1, all five
BLAS settings1, fallback0 and process-scoped caffeinate. No new dependencies.

Measure the complete cost comparison (generation, fit, scoring, both searches,
all controls, evaluation and publication), nested clocks separately, peak RSS
and all retained bytes. Forecast 16 repetitions from that sample with a1.5
safety factor and publish the one-sample uncertainty. Fresh disk must cover
that projection plus10GiB reserve; process RSS cap6GiB, severe-pressure detection
requires <=10% coarse availability plus new swapouts. Check resources between
atomic work units; preserve failures. Elapsed forecasts are not automatic time
kills, and documented user pauses remain explicit clock observations. Zero
automatic retries. Do not silently reduce context, batch, roles or teacher count.

Store pair files once and a small hash-bound member ledger; do not repeatedly
embed full pair/row arrays in a cumulative raw ledger. Bind wrapper arguments
as well as base config, source, numerical receipt and code commit. Checkpoints,
raw update logs, failed attempts, predictions and source identities remain
available; bulk artifacts stay outside Git. Old artifacts remain untouched.

## Analysis and decision

Compute original-unit MSE/RMSE/MAE and F-variance-normalized E MSE from saved
predictions. Primary diagnostic difference is R_random-R_greedy_long. Also
report R_greedy_short-R_greedy_long, R_ordinary_upper-R_greedy_long and the old
cheap-score/random difference. Positive favors the named grown method.
Average repetitions within teacher, then teachers equally. Report all nine
method means, all eight teacher effects, median, per-repetition variability,
zero/tied cases, selected IDs, overlap/rank changes, candidate improvements,
executed updates and ceiling hits. No selective aggregation by winning task.

Use one fixed512000 whole-teacher bootstrap, seed261031106, retaining pairing
across differences; percentile95% intervals and Student-t sensitivity are
descriptive. No superiority labels, equivalence margin, nominal-coverage or
powered-confirmation claim. More rows/repetitions/bootstrap draws do not become
additional independent teachers. Report total extra search cost and limitations
of single-unit ranking, validation reuse and finite stopping.

Independently audit all saved E targets/row IDs against original observations,
recompute F variance/risks/teacher effects, and check same-parent state/row-stream
bindings. Replay one representative short/long ranking under the same source
and restore one long-probe checkpoint; verify exact CPU state/prediction parity.
Regenerate the final diagnostic report twice deterministically. Acceptance is
complete valid evidence and an honest decision, never a positive result.

If longer selection improves random but ordinary capacity matches it, retain a
component-only opportunity. If it improves both, investigate a cheap approximation
on fresh data. If it supplies no clear gain, investigate growth timing/training
or narrow/stop this allocation procedure. None automatically triggers depth,
transfer, a new score or another cohort.
