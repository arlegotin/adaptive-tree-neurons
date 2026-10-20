# M3: joint placement did not earn added capacity

The exact parent-conditioned joint selector did **not** pass Gate B. On sixteen
fresh primary synthetic teachers, continuing the competent ordinary parent had
lower exploratory evaluation error than joint growth on **all sixteen** teacher
averages. Warm ordinary widening also beat joint growth on all sixteen. The
joint selector did not beat informed-direction random placement on average.
This closes the tested selector/treatment branch as a negative development
result. It does not establish that every gated architecture is unhelpful.

## Complete experiment and evidence

Each teacher has two paired training repetitions. The study completed all ten
final roles: **320 final models**, 32 shared parents and 2,080 short probe entries
(including keep). All **3,089 queue entries** completed eligible, without a
missing or failed entry. Queue entries also include data preparation, scoring,
selection and evaluation; they are not independent models or problems.

The predeclared initial eight-teacher V interval spanned both zero and the 2%
target, triggering its sole permitted extension to sixteen. No further teachers
were added. Independent arithmetic reconstructed 640 original-unit V/E prediction
vectors and 7,680 scalar comparisons, with maximum discrepancy **5.55e−17**.
There was no protected T creation/access or confirmation.

The [recipe](../docs/research/v2/M3-development-recipe.md) was locked before these
teachers: d16/h64/count16, current contrasts ε=.05/τ=1, AdamW .005/WD=.001,
batch128. The parent used G25/patience20; every warm arm then executed the same
fixed 8,000-update continuation without patience termination, restoring the
strict G-best complete state from G0/25/…/8000. Fresh width95 used one fixed
8,000-update stage, with different initialization and inheritance. Short probes
executed 300 updates each and discarded their trained weights. Joint ridge used
λ0=λg=.01, full parent conditioning, original fixed column-RMS scales and common
eligibility/count. Direction, timing and treatment were held fixed across
placement comparisons.

Producer: commit `53bd029752a2ef033a1932367854eb6637bfe539`, package
`e78e027ffd7b0c1f4f4596b9e7ae8eee6c6a55e0f92e297a5d93e519cf1391a9`.
Independent auditor: `654a2cc`, package
`4dbf2141ca1215b4edc12aaad8d8fa9934f3b10fcf99cbc59fb87335ebc360b6`.
The [completion receipt](../docs/research/v2/WP04-M3-completion-receipt.json)
binds the exact queue, final ledger, process and audit files. The audit checks
retained bytes/arrays and metric arithmetic; it does not retrain the teachers or
authenticate every checkpoint tensor transition independently.

## All ten procedures

Risk is original-unit MSE divided by F target variance (ddof0). Average the two
repetitions within each teacher, then give each teacher equal weight. Lower is
better. V is development configuration evidence; E is exploratory evaluation.

| Procedure | Mean E NMSE | Mean V NMSE | Stage-zero selections / 32 |
|---|---:|---:|---:|
| Ordinary keep | **.05077368** | **.05115488** | 1 |
| Warm ordinary upper, width95 | .05479102 | .05580169 | 3 |
| Fresh ordinary upper, width95 | .05867650 | .05840021 | 0 |
| Informed-direction random | .05941502 | .05965374 | 11 |
| Local-singleton ridge | .05983059 | .06041583 | 13 |
| Short singleton search | .05986615 | .05972584 | 8 |
| Parent-conditioned joint ridge | .06015804 | .05984766 | 12 |
| Unpenalized-parent additional span | .06037839 | .06121578 | 11 |
| Unconditioned joint ridge | .06044164 | .06099278 | 13 |
| Historical cheap score | .06090511 | .06159369 | 17 |

The ratio of arithmetic mean E risks gives **15.60% lower error for keep than
joint**, and **1.25% higher error for joint than random**. These are different
summaries from the equal-teacher log effects below. Among all seven tested gated
roles, none improved a teacher's mean E risk by 2% over keep. Random alone beat
keep on one teacher by less than 2%. Some V effects were favorable; a post-hoc
best-arm-per-teacher choice is not a new validated procedure.

| Teacher | Keep E | Warm upper E | Random E | Joint E | log(random / joint) |
|---|---:|---:|---:|---:|---:|
|0|.04560383|.04346733|.04809475|.04892342|−.01708|
|1|.07652542|.08580708|.08880011|.09069012|−.02106|
|2|.02842029|.02792518|.03088592|.03053993|+.01127|
|3|.07036805|.08427957|.09566859|.08818984|+.08140|
|4|.07093204|.07783173|.09488745|.10385292|−.09028|
|5|.06071834|.06590635|.06737749|.06659764|+.01164|
|6|.02717223|.02841176|.02919821|.02913979|+.00200|
|7|.02684813|.02656097|.02880897|.02897485|−.00574|
|8|.04822467|.05317528|.06663930|.07123972|−.06676|
|9|.04507445|.04626225|.04739421|.04739421|.00000|
|10|.07231727|.08320403|.08828888|.08895477|−.00751|
|11|.02807906|.02709762|.02772369|.03104091|−.11302|
|12|.04586202|.04618568|.04839741|.04839741|.00000|
|13|.05267602|.05536594|.05926218|.05700933|+.03876|
|14|.06249041|.06653180|.06906821|.07025871|−.01709|
|15|.05106670|.05864368|.06014496|.06132510|−.01943|

[All 160 teacher/procedure V and E means](../audit/research-v2/M3/teacher-risks.csv)
are retained with the [machine-readable summary](../audit/research-v2/M3/development-summary.json).

## Placement uncertainty and the ordinary comparison

For each teacher take log(mean-R2 reference risk / mean-R2 joint risk); positive
favors joint. The table gives descriptive Student-t intervals over sixteen
teacher effects. These development intervals follow the declared precision
extension; they are not frozen confirmatory error guarantees. Whole-teacher
20,000-resample bootstrap sensitivity and both V/E effects for every comparator
are [retained](../audit/research-v2/M3/descriptive-comparisons.json).

| E reference over joint | Mean log benefit | Descriptive 95% t interval | Teacher signs + / 0 / − |
|---|---:|---:|---:|
| Random | −.013307 | [−.038086, +.011472] | 5 / 2 / 9 |
| Local-singleton ridge | −.007838 | [−.021903, +.006227] | 6 / 2 / 8 |
| Ordinary keep | −.146150 | [−.204006, −.088294] | 0 / 0 / 16 |
| Warm ordinary upper | −.088385 | [−.134158, −.042612] | 0 / 0 / 16 |

The placement advancement rule uses V, where mean log(random/joint) is also
negative, **−.004848**, with descriptive t interval [−.034695,+.025000]. It fails
the prewritten positive-mean requirement. The study supports neither a useful
placement win nor an advantage for this grown procedure over competent keep.
It does not formally rule out every population effect smaller than 2%.

## What the local model explains, and what it does not

All 128 saved temporary-update diagnostics remain visible: four ridge variants
on each of 32 parents, with every prescribed fraction 1/16, 1/4 and 1. These
temporarily apply both solved parent and gate changes; their gains cannot be
credited solely to new gates. They are separate from the actual Adam training.
The [review](../audit/research-v2/M3/local-review.json) summarizes saved producer
diagnostics; this collection does not independently repeat nonlinear inference.

For the default joint selector, average teacher relative changes after averaging
the two repetition losses are:

| Temporary fraction | Predicted F reduction | Actual F reduction | Actual G reduction | G improves, teachers | Mean G gate saturation |
|---|---:|---:|---:|---:|---:|
| 1/16 | 6.27% | 5.06% | .66% | 16 / 16 | 1.07% |
| 1/4 | 22.95% | 8.68% | .06% | 5 / 16 | 21.06% |
| 1 | 57.94% | **−149.38%** | **−59.13%** | 0 / 16 | 61.75% |

Small temporary steps expose a real local signal. At full scale the fitted local
model overpredicts F improvement, while even its linearized prediction worsens
G on all sixteen averages. Nonlinear saturation and activation changes compound
that mismatch. This is evidence against treating full local objective reduction
as finite predictive improvement; it does **not** prove that the actual Adam
trajectory saturated in the same way or that a new learning-rate remedy works.
No fraction was selected on G or used to update a fitted parent.

The direct augmented versus Schur objective discrepancy was at most 3.05e−16
in those saved solves. Parent numerical ranks were 1,083–1,089, below F2306;
the pure-span score was not forced to zero by full-row rank. Default joint
selected-block novelty fractions averaged .316, with range .159–.527: there
were directions outside the parent tangent span. That does not make them useful
for prediction. The zero-parent-ridge arm also lost to ordinary keep.

Joint and local-singleton selected sets differed on all 32 parents, sharing
10.03 of 16 IDs on average. The exact joint objective preferred the joint set
on every parent. At the smallest temporary fraction its actual G loss beat the
local-singleton alternative on only 14 of 32 parents. Changed allocation and a
better F objective therefore did not supply reliable held-out ranking. These
two-set diagnostics are not a new exhaustive subset oracle or population test.

## Resources and branch decision

The containing scientific process took **8,997.63 seconds (about 2h30m)**,
including contended evidence I/O, with sampled peak owned-root RSS **.975GiB**.
It had no wall deadline. The separate independent audit process took169.24seconds
at about.374GiBsampled peak. Charged job clocks sum to8,860.24seconds inside
the containing process, including134.16seconds of diagnostic/control-only work.
Do not add those nested timers to the containing wall time or sum shared
standalone costs into physical cohort cost.

Mean standalone experimental fit costs were23.99seconds keep,27.27random,
28.45joint,35.20warm upper and96.72short singleton. They include each procedure's
actual prerequisite data/parent/scoring/search work once. They are contended
experimental clocks, not dedicated deployment benchmarks or an efficiency claim.
Byte-preserving storage maintenance retains every original evidence byte;
no slower or losing candidate was removed.

**Gate B closes negative.** The one uncertainty expansion is exhausted. The
optional nine-pair regularization correction is **not admitted**: there is no
measured useful gated-capacity advantage beyond competent continuation in this
regime, and temporary parent-plus-gate local improvement does not isolate such
an advantage. No correction fits were run; their hypothetical results are unknown.
Direction/newborn grids are also unadmitted: temporary diagnostic saturation
does not establish deficient participation in the actual continuation. Neither
pruning nor a residual redesign has its required measured trigger yet. See the
[exact disposition](../audit/research-v2/M3/gate-B.json).

Native baseline qualification continues, including full-recipe bounded-memory
NODE without a wall cap. The unchanged historical procedure can remain an
explicit reference for a practical comparison; this result admits no new
selector to calibration/confirmation. The strong transfer benchmark, global
freeze and protected confirmation remain unfinished. Missing comparisons are
unavailable evidence, not predictive defeats. The infrastructure and planted
control success remain useful; a broad practical advantage remains unestablished.
