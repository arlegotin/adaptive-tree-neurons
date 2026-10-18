# Longer search does not establish a placement advantage

The conditional neuron works, but choosing which neurons acquire specialists
remains unproven. This follow-up gave an expensive selector more training time
to test whether the cheap score was missing useful placements. Longer search
did not establish an advantage over random placement or shorter search. That
narrows the current procedure; it does not show that all conditional networks
or all growth schedules are ineffective.

The [complete generated report](../../reports/long-probe-diagnostic.md) retains
every candidate, selection, stopping result and teacher effect. The
[independent audit](../../audit/diagnostics/2026-10-18-long-probes/reproduction/coordination/task-3-report.md)
adds per-teacher risks, repeat variability and exact replay evidence.

## What was tested

Eight fresh independent synthetic regression problems each received two paired
training repetitions. One learned parent and its contrast directions were
shared by all warm-started methods. Each selector chose 16 of 64 eligible
neurons, giving 1,697 parameters; ordinary width brackets had 1,693 and 1,711.
Short search trained every single-neuron candidate for 320 updates. Long search
gave each candidate the normal 8,000-update ceiling with validation patience.
Both searched every eligible neuron and a keep candidate, then restarted the
joint continuation from the original parent rather than retaining probe weights.

F supplied fitting labels and preprocessing statistics; G selected checkpoints
and ranked candidates. V stayed unused. Independently generated E observations
evaluated completed procedures. The recipe, eight-problem count, two repeats,
seeds and analysis were locked before cohort fitting. No protected T was
generated or accessed. No result-dependent extension or numerical retry occurred.

## Results

Lower normalized evaluation MSE is better. Each row first averages the two
repetitions within a problem, then gives all eight problems equal weight.

| Method | Mean normalized E MSE |
|---|---:|
| Continued parent | 0.0648669 |
| Cheap score | 0.0642051 |
| Random placement | 0.0638495 |
| Short validation search | 0.0631249 |
| Long validation search | 0.0633780 |
| Ordinary warm widening, upper bracket | 0.0640707 |
| Fresh ordinary network, lower bracket | 0.0672782 |
| Fresh ordinary network, upper bracket | 0.0693750 |
| Static partial gated network | 0.1005204 |

| Paired difference, positive favors the named grown method | Mean | Descriptive teacher-bootstrap 95% interval |
|---|---:|---|
| Random minus long search | +0.0004715 | [−0.0010366, +0.0022852] |
| Short minus long search | −0.0002532 | [−0.0009665, +0.0003316] |
| Fresh ordinary upper bracket minus long search | +0.0059970 | [+0.0025344, +0.0098783] |
| Random minus cheap score | −0.0003555 | [−0.0018857, +0.0010121] |

Long search beat random on three teacher means and lost on five. Its average
benefit is positive, but its median is negative and the interval spans gains
and losses. Short search has a slightly better overall mean. The cheap score
again has no clear placement advantage. These intervals are descriptive:
eight problems do not establish nominal coverage, equivalence, a powered
rejection or superiority.

Long search has a lower mean risk than the fresh ordinary upper-bracket model,
with positive paired effects on all eight teacher means. Ordinary warm widening
is much closer. The fresh-model comparison therefore does not establish an
advantage over ordinary capacity in general. Static gating's poor result is
also limited to this shared recipe; it is not a tuned practical benchmark.

## Cost and verification

The complete cohort took **7,804.403 seconds (2 hours 10 minutes)** from launch
through final publication. Its internal clock was 7,802.113 seconds. Nested
short search took 2,140.462 seconds; the additional long searches took
4,169.762 seconds, about 70 minutes. These clocks are included in the complete
elapsed time and must not be added to it.

All 1,040 short trials reached their 320-update limit. All 1,040 long trials
stopped by patience, with no 8,000-update ceiling hits. The 16 paired members
and all 144 final model instances completed without scientific failures or
retries. The raw cohort retained 11,496,978,848 logical bytes; peak process RSS
was 947,650,560 bytes. The machine remained within the prospectively declared
disk and memory limits. This comparison charges exhaustive search and does not
claim an equal-cost practical advantage.

An independent script recomputed all four error metrics from 37,748,736 saved
prediction values and regenerated original observations, using F population
variance for normalization. It agreed exactly with all 760 compared analyzer
values. Checks covered the original-parent bindings and raw minibatch prefixes
of all 96 warm forks, including epoch rollover.

The prospectively chosen first problem/repetition replayed both complete
rankings: 130 trials, 53,300 updates, 2,275 validation evaluations and 133 full
saved states matched exactly. This includes model tensors, Adam state,
scheduler, RNGs, sampler, next rows, predictions and an actual long-probe last
checkpoint restoration. Only documented run identifiers and checkpoint locator
fields were normalized. Two independent report builds produced identical
246,328-byte files, SHA-256
`9baa18c96a55a0023adf469984b8f5bebf948d1985a73675fbde115dc5e5335b`.

The generated report retains the internal ledger's byte boundary, 486 bytes
below the final outer inventory. Both boundaries are preserved explicitly.
Two audit-checker mistakes were corrected in new checker versions: hashing
history required a mapping wrapper, and a first-batch check needed epoch
rollover. The completed short replay was reused; no model fits were repeated.
Failed checkers, their costs and the original evidence remain archived.

The original OS timing wrapper was sandbox-denied after the CPU receipt had
passed. A Python timer and resource counters measured subsequent commands;
supplementary OS timing counters are unavailable. The valid receipt was retained
and no receipt/model rerun was introduced. This is the new plan's sole recorded
ruling; the earlier eighteen rulings remain unchanged.

## Decision and remaining work

**Narrow the current allocation claim.** This experiment supplies no clear
evidence that longer candidate training solves the placement problem. It does
not justify a more elaborate score, deeper trees or transfer benchmarking by
itself. If research continues, growth timing is a focused next question: useful
placements may differ earlier in training. That explanation is untested and
requires a new prospective diagnostic; no further cohort is automatically
authorized by this result. Stopping this allocation procedure is also a valid
decision.

Canonical variance calibration, accepted independent-confirmation counts, the
true protocol freeze, independent confirmation and the strong practical
baseline suite remain unfinished. This follow-up does not replace those gates.
The historical pilot and report were preserved byte-for-byte. The unchanged
implementation's 55 affected tests passed before the source lock; the old broad
suite was not repeated and is not represented as clean.

The [cohort inventory](../../audit/diagnostics/2026-10-18-long-probes/cohort/index.json)
and [reproduction inventory](../../audit/diagnostics/2026-10-18-long-probes/reproduction/index.json)
bind execution, scripts, failures, arithmetic, replay, resource and preservation
evidence. Bulk predictions and checkpoints remain outside Git.
