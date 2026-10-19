# Adaptive tree neurons research v2 design

Status: authorized implementation specification, 18 October 2026. The user
supplied the research handoff and explicitly requested specification, planning,
and execution. This document records that contract; it does not report success.
The execution ledger is `docs/research/v2/execution.md`. Read this document,
the implementation plan, and that ledger before continuing after a context reset.

## Intent and evidence boundary

**M3 development treatment amendment,19October (builder decision):** before any
M3 teacher generation, fix the common parent/continuation treatment in
`docs/research/v2/M3-development-recipe.md`. It uses the M2-selected global
weight-decay1e-3 ordinary recipe and late parent, followed by fixed8000-update
continuation without patience termination in every warm arm, with strict G-best
selection on the original25-update grid. The purpose is to expose the ordinary
headroom measured in M2 rather than crediting growth for replaying a short losing
tail. Existing contrasts/count/ridge defaults and the finite M3 counts, expansion
and correction rules remain. This is a prospective controlled development
treatment, not a claim that the combined recipe is optimal or a protected freeze.

**19 October user priority amendment:** the user explicitly prioritizes completed
scientific results over a two-hour deadline. Native qualification and the quality
view must finish the declared native stopping procedure, without an arbitrary
wall-time cap. Keep the finite configuration supports and attempt ceilings,
paired information access, memory/storage guards, and all paid costs. The
10/30/120-minute prefixes remain a separate resource-bounded view; an unfinished
native model is unavailable at a prefix, never a predictive loss. Quality-first
searches finish their finite attempted configurations; their different total
times must be reported. Update forecasts before admission. No production HPO,
calibration, global freeze or protected T preceded this amendment.

Long NODE fits require periodic, atomic training checkpoints before restarting.
Resume the raw model, optimizer, sampler and RNG, with G-selection/rolling-window
state, under matching source/runtime/configuration/data identities. Never resume
from averaged prediction weights. External append-only row/update records carry
their own committed prefixes; checkpoints must not duplicate an ever-growing
history. A resource interruption may resume the same seed/state and retains every
attempt's costs. The earlier capped Energy fit has no saved model and cannot be
resumed; its new checkpointed, uncapped run starts from the same seed and data.
This amendment changes prospective resource policy, not historical receipts or
the fixed usefulness margins, multiplicity rules or data-access boundaries.

Determine whether adaptive affine-stump capacity, particularly informed placement,
provides reproducible prediction or quality–cost advantages over a finite strong
suite. Preserve useful infrastructure, change scientific hypotheses when evidence
warrants, and accept negative, estimation-only, narrowed, and retired outcomes.
Do not tune indefinitely until a result wins.

Reference source is `230e6575149c9fee440ca301fa3782a85751bac8`. At intake the
checkout is clean at that commit on `main`; `refs/remotes/origin/HEAD` is absent.
Historical reports, schemas, locks, and STOP decisions retain their original
scope. The new work is authorized under `research/v2` and `confirmation/v2`.
Never reset user work, rewrite historical evidence, weaken source/receipt guards,
substitute weak native baselines, or silently score missing predictions as losses.

Established historical findings (not freshly executed here):

- Exact zero-gate insertion, gate gradients, parameter registration, incumbent
  Adam history, and the documented first-step score have independent support.
- Cheap placement has no established advantage over informed-direction random.
  Old 16-teacher/R2 mean random-minus-top NMSE is -0.0005364354, whole-teacher
  interval [-0.0012050299, 0.0001560503]. Long-search 8-teacher/R2 benefit over
  random is 0.0004714922, interval [-0.0010365806, 0.0022851931]; short-minus-long
  is -0.0002531947, interval [-0.0009664924, 0.0003316262].
- Some growth arms beat particular fresh ordinary recipes. Static partial gating
  underperformed the common recipe. Neither establishes a native modern-suite win.
- A patience-stopped parent restores G-best complete state. Unchanged deterministic
  continuation can replay the losing tail; a prior diagnostic verified 500 updates.
  This is conditional on patience stopping and is not an optimizer-reset defect.
- Singleton ranking omits cross-block interactions and discards learned probe
  weights before joint restart. An actual-Jacobian counterexample proves a general
  limitation, not its empirical causal contribution.
- Earlier growth was worse on one diagnostic. Longer probes changed rankings
  without resolved endpoint gains. Covariance-aware directions improved a local
  objective only. None is an established predictive remedy.
- Manipulated checkpoints exposed accepted age-zero incumbent Adam state, negative
  finite second moments, and silent float64-to-float32 model casts. No normal-run
  corruption or efficacy attribution was established.
- Full historical raw archives and investigation-only E00–E05 assets are absent
  from a clean checkout. Available scalar arithmetic is not raw authentication.
  Historical Mac/Linux F.y differences remain unbounded without original bytes.

Historical reading anchors: `REPRODUCE.md`, `DECISIONS.md`, `reports/final.md`,
`reports/long-probe-diagnostic.md`, `docs/research/2026-10-17-complete-exploratory-pilot.md`,
`docs/research/2026-10-18-long-probe-diagnostic.md`, and the publication audit.
New receipts must distinguish software correctness, scalar reproduction, raw
authentication, and fresh numerical research. Do not claim a monolithic suite
pass from targeted runs or import recoveries.

## Mathematical model and preserved behavior

For augmented X=[x,1], p=d+1,

    f(x) = beta0 + sum_j beta_j ReLU(b_j^T X
                + 1[j in S] tanh(a_j^T X/(2 tau)) c_j^T X).

ReLU follows the mixture. This is one stump per selected neuron, one hidden
layer and one growth event in the accepted legacy implementation. At insertion
a=0, c!=0, so predictions are preserved. With M_j=mean(2(f-y) beta_j
1[b_j^T X>0] XX^T), g_j=M_j c_j/(2 tau); contrast gradients are initially zero.
Dead units, zero head weights, and residual cancellation can yield zero gates.
For d16/h64/m16 the parent has 1153 parameters and growth 1697; ordinary widths
94/95 have 1693/1711. Soft gates evaluate both branches; no dispatch saving exists.

Current contrast: symmetrize M, choose deterministic-sign largest-absolute
eigenvector v, set c=epsilon*r*v/max(sqrt(v^T C v),1e-12), where
r=sqrt(mean((b^T X)^2)), C=mean(XX^T), epsilon=.05, tau=1. Current Adam score is
sum(g_k²/(abs(g_k)+Adam_epsilon))/(2p). It predicts the first full-F gate-only
step, not selected finite all-parameter growth. Do not alter this anchor.

Legacy training: AdamW .005, WD0, batch128, constant schedule, parent and
continuation ceilings8000, G every25, patience20, strict earliest best including
update0. G-best restoration includes optimizer/sampler/schedule/RNG. Preserve
the historical short320 grid for reconstruction. New horizon grids use
0/25/... at prefixes300/1200/8000 and fixed-budget trajectories without patience.

## Architecture and ownership

Keep the legacy CLI/config/protocol and source-bound safeguards intact. Add
`src/treeneuron/research/` for strict versioned configuration, immutable attempt
lists, snapshot/treatment handling, role access, ledgers, selector, native adapter
qualification, search, protocol, confirmation and reporting. Reuse insertion,
training, checkpoint publication, data transforms and numerical references where
their semantics match. Add new source-bound workload receipts for new dimensions,
batches, depths and architectures; do not delete finite legacy envelope guards.

Commands: `run --config --output`, `plan --config --output`,
`prepare-data --config --output`, `qualify --config --output`,
`freeze --config --output`, `confirm --protocol --output`,
`report --study --output`. Unknown fields/kinds and unimplemented prerequisites
fail explicitly. Fresh outputs cannot overwrite prior runs. Resume checks the
immutable source/config/data/queue identity. No command is advertised as working
until its owning work package is implemented and verified.

`research/v2` resolves study/source, data/splits, parent recipe/snapshots,
direction/placement/count/treatment, budget, seeds, analysis and artifact root.
Hash ties, failures, stopping, fallback, HPO and refit rules as recipe semantics.
The builder may decide internal decomposition and use existing Python/Torch/NumPy/
SciPy infrastructure. Algorithm, data access, finite matrices and claim thresholds
are fixed here. Optional branches require measured admission, not convenience.

## State, snapshots and recovery

Validate shape, declared dtype, finiteness, nonnegative second moments, complete
named Adam slots, incumbent/birth/applied-update ages in both current and best
states before publication. Reject silent casts. Legacy unfrozen incumbents have
Adam age equal to selected global update; future freezing needs explicit applied
counts and frozen intervals rather than equating every age to global time.
Only named verified migrations may import old state; otherwise run anew.

Snapshot callbacks capture actual states at absolute250/1000 during training and
the late selected state. A lower cap returning an earlier best is not a snapshot.
Missing early states stay unavailable; optional fixed-parent-budget runs are new
declared treatments. Record selected/executed absolute and stage updates separately.
The new-stream intervention immediately regenerates F permutation from its paired
seed and sets cursor0 while retaining Adam/global time/absolute LR. Save discarded
order/cursor, phase event and new sampler state. Reseeding alone is insufficient.

Publish immutable tensors/metadata, then atomic pointer. Recover model, optimizer,
schedule, sampler, RNG, preprocessing and treatment-controller together. G-best
restoration never rolls back the append-only outer attempt ledger or paid costs.
Store row streams once by checked reference, not repeatedly inside checkpoints.
Allowed statuses: completed_eligible, completed_ineligible, timeout,
resource_exhausted, numerical_failure, input_or_dependency_failure,
interrupted_recoverable, irrecoverable_missing. One deterministic same-state/seed
infrastructure rerun is allowed with both attempts and costs retained. Numerical
failures cannot be hidden by replacement seeds. Missing planned predictions block
a complete primary claim; paired subsets are descriptive.

## Data roles and seed identities

F fits weights/transforms/covariance/Jacobians/embeddings/vocabularies. G selects
checkpoints and declared structural probes. V labels belong only to the search
coordinator for complete-procedure selection. E is exploratory synthetic analysis.
T belongs only to the frozen evaluator after source/method/data/analysis freeze.
Use forbidden-label sentinels. Fit all preprocessing on F. Retain F-fit/G-selected
models; do not silently refit F+G+V. Retrieval context is F only. Evaluator-only
noiseless targets and sampled noise never enter fitting workers.

Primary synthetic population is exact current `make_teacher` d16 family, sigma.1,
h64/m16, F2306/G577/V577, E65536; protected T262144 is generated only after freeze.
Serialize resolved generator defaults, coefficients, normalizers, seeds, actual
arrays and hashes. Independent unit is teacher, not repeat or E/T row.
Exclude all historical/development teachers from confirmation, including D1
261018411 and D2 918401–918404. D1 reference/data/optimizer/allocation/contrast
seeds were261018412–261018416. D2 seeds s used data s+10000, optimizer+20000,
allocation+30000, contrast+40000, reference+50000 and E+60000.

Development roots: mechanism2026101801, HPO2026101802, splits2026101803,
variance calibration2026101804. Define

    seed64(root,*labels) = int.from_bytes(sha256(json.dumps(
        [root,*labels], separators=(',',':'), ensure_ascii=True).encode('ascii'))
        .digest()[:8], 'big') & ((1<<63)-1).

Teacher seeds omit method/repeat, observation seeds omit method, paired parent/
continuation seeds omit arm. Separate allocation/direction/HPO namespaces.
Derive separately labeled32-bit seeds for libraries; detect unintended collisions,
add the smallest deterministic counter and retain it. Confirmation root is newly
generated after choice, committed by hash before protected teacher creation,
revealed with archive. Check actual teacher identities for overlap.

Practical native-quality jobs also fix model randomness before production:
labels are `(study_id,'practical',dataset,'outer',str(fold),'repeat',str(repeat),
'fit')` under root2026101802. Retain the canonical63-bit value and allocate its
library32 value after all60fixed HPO library seeds, excluding those values and
earlier model labels. Store each collision counter. A dataset/fold/repeat shares
this model seed across every family and trial; seeds are never HPO candidates.
Bind it through the search plan and selected-model inventory. These paired
labels do not imply identical initial tensors across different architectures.

## Controls

Exact algebra control: actual gate Jacobians with duplicate and complementary
units; residual orthogonal to full ordinary-parent tangent. Enumerate subsets.
Include a case where singleton ranking is correct as well as its failure case.

Trainable control: d2/h8/m2, uniform[-1,1]^2, F512/G256/V256/E16384. Raw base rows
[x1,x2,bias] are [(1,0,2),(1,0,2),(-1,0,0),(0,1,2),(0,-1,2),(1,1,3),
(-1,1,3),(0,0,-2)]. Heads [.5,.5,1,.2,-.2,.1,-.1,.1], bias0. Four predeclared
instances multiply heads by uniform[.95,1.05] factors, sharing factors for0/1.
Generating IDs0/2: c0=(0,.5,0), c2=(.5,0,0), a0=gamma(1,0,0),
a2=gamma(0,1,0), tau1, gamma{.5,2}, sigma{0,.1}. Store coefficients before data.
Unit7 is deliberately dead; nominal ineligible candidates/subsets stay recorded.
Transport every raw affine q to standardized x by q.features*=scale_x and
q.bias+=dot(q_raw.features,mu_x); beta/=scale_y,
beta0=(beta0_raw-mu_y)/scale_y. Verify original-unit predictions independently.
Fitted arms use current F-derived contrasts, including the generating-ID oracle;
true-contrast oracle is separately information advantaged. Prior quadrature
teacher-parent MSE .0021171995449/.0247845970420 verifies signal only, not fitting.

Builder clarification before fitting: M1-positive/M2d start from the supplied
ungrown raw parent transported through the F-fitted transform, with fresh age-zero
Adam and all inserted gates zero, including both oracle arms. There is no prior
ordinary fitting stage for these controls. This preserves the explicitly planted
allocation question; ordinary adaptation during common continuation remains free.

Negative controls: affine, additive smooth, pure noise, shuffled labels. Synthetic
stress: axis-aligned/rotated structure, nuisance dimensions, separate custom
prevalence{.05,.2}, covariance conditions{1,10}, F{577,2306,9224}. Preserve legacy
`rare` prevalence.01 identity. Only primary population initially confirmatory.

## Exact joint ridge selector

Primary comparison changes IDs only: retain current contrasts/count/parent/
continuation. Residual r=y_F-f_F; full trainable-parent prediction Jacobian J0;
gate block Psi_j(i,:)=beta_j 1[b_j^T X_i>0](c_j^T X_i)X_i^T/(2tau).

    Q(S)=min_uv mean(||r-J0u-Psi_Sv||²)
          +lambda0||D0u||²+lambdag||Dgv||²
    Gain(S)=Q(empty)-Q(S).

D entries are original column RMS, computed once from J0 and each original block.
Drop threshold=max(1e-15,1e-12*block_max_RMS), retaining IDs/cost/eligibility.
Never rescale after residualization or standardize residual again. Normalize
columns once and apply isotropic ridge, not D penalty again. Default lambda0=
lambdag=.01; numerical sensitivity grid{1e-4,1e-2,1}² is not nine trained arms.

With A=J0'J0/n+lambda0D0², B=J0'Psi/n, C=Psi'Psi/n+lambdagDg²,
d=J0'r/n, e=Psi'r/n, H=C-B'A^-1B, s=e-B'A^-1d, Gain=s'H^-1s.
Use Cholesky/QR/SVD solves, never inverse. Compare direct augmented regularized
least squares. Squaring a soft ridge residualizer is incorrect. Log any jitter,
truncation, condition, solve residual, fallback, and stable-reference difference.
Nonfinite or materially negative gains fail; no easier candidate set fallback.

For lambda0=0, form one SVD parent basis (cutoff1e-10*largest singular value)
from originally normalized J0. Project r/Psi orthogonally once, keeping original
gate scales. Direct reference uses that same retained basis. Full row rank gives
zero additional tangent dimension, not proof of no generalization capacity.
Penalty-sharing example J0=Psi=r=[1], both lambdas1: Q0=1/2, Qjoint=1/3,
gain1/6; lambda0=0 gain0. Report parent-span novelty fraction. Positive gate ridge
can reward duplicate-gate penalty sharing even when parent ridge0.

Greedy conditional selection maximizes Gain(S+j)-Gain(S), with block Schur updates:
s_tilde=s_j-H_jS solve(H_SS,s_S), H_tilde=H_jj-H_jS solve(H_SS,H_Sj).
Delta=s_tilde' solve(H_tilde,s_tilde). Stable unit IDs resolve ties within
1e-12*max(1,abs(best)). This is not a global optimum/submodularity claim.
All arms use current scorer eligibility and m_effective=min(requested,e), even
when all scores are zero. Zero local blocks stay in the eligible pool. Count
selection/no-growth is a different procedure. Enumerate all eligible h8 pairs
(nominal28; dead7 means at most21 eligible pairs) with skipped nominal records.

Interfaces: frozen CandidateBlock(unit_id,contrast CPU float64,eligible,reason,
added_parameter_count); JointSelectionConfig(count16,parent_ridge.01,
gate_ridge.01,condition_on_parent=True,column_scaling='rms',solver='exact_float64',
tie_relative_tolerance=1e-12). build_gate_blocks(parent,F,score_table),
joint_ridge_select(parent,F,candidates,config)->SelectionTrace. Trace retains
all conditional scores, IDs, unchanged contrasts, source/data/config/parent
hashes, masks/scales/ranks/conditions, novelty, objective diagnostics, costs and
fallbacks. Checked conversion to legacy Allocation preserves insertion ownership.
Selector cannot read G/V/T. Do not apply the solved update in the primary arm.

Acceptance: autograd/analytic float64 Jacobians; Schur/direct empty, singleton,
duplicate-parent/gate, complementary/random subsets agree rtol1e-8/atol1e-10 on
normalized small cases; counterexample ordering unpenalized and lambda0=0/gate.01;
default ridge ordering follows its own dense reference. Check non-tied permutation
invariance and ties. Temporary solved updates at fractions{1/16,1/4,1} diagnose
F/G prediction, ReLU changes and saturation without G-selecting the fraction.
h64 full parent+gate Jacobian approximately2241 columns,41MB doubles; Gram40MB
before temporaries. Measure actual resources. Any sketch/last-layer approximation
is a new named algorithm with rank/seed/error/cost and exact-small validation.

## Conditional interventions and attribution

RMS-SGD direction solves M'Mv=mu Cv on covariance positive eigenspace cutoff
1e-10*max, with exact contrast RMS rho=epsilon*r. Adam-aware direction maximizes
sum(g²/(abs(g)+Adam_epsilon)) on whitened RMS sphere: starts=current, generalized
SGD, two seeded;25 projected steps at.05, best F objective. Exhaustive sign
reference p<=6; p17 is bounded optimization, not a global optimum. Cross
{current,RMS-SGD,Adam-aware} with {random,compatible-Adam-singleton,joint}. Keep
epsilon.05/tau1 unless measured saturation/scale failure admits9 pairs
epsilon{.01,.05,.2}*tau{.5,1,2} with both placement arms and paid separate matrix.

Only measured deficient participation admits LR multipliers{1,3,10}*parent
freeze{0,100}; random and ordinary widening controls get matched treatment.
Multipliers belong in scheduler groups. Frozen parent has no gradients/Adam
updates, not merely LR0 with advancing moments. Record applied ages, motion,
gradients, moments, update/weight ratios, logits, saturation>.99 and contributions.
Direction/treatment gains shared by random are credited to that factor.

At most one admitted redesign. Overgrow/prune: all eligible gates,300 jointly
trained updates, one-pass G ablation harm ranking, fixed common effective count,
stable ties. Compare retained-selected, restarted-same-IDs/current-contrasts and
retained-random from common overgrown state; also both ordinary lower/upper
parameter brackets with all three arms. Slice tensors/Adam slots consistently.
Charge candidate work to every standalone arm. This is pruning_selection, not
pre-insertion placement. Never merge independent singleton-trained parents.

Residual branch requires measured competent-one-layer ceiling and qualified
ResNet: same frozen parent/raw transformed inputs,16-unit gated residual with
zero output head versus exact ordinary lower/upper parameter brackets. Frozen
parent and optional corrective fitting are separate. Verify head gradient then
internal gradients. C2 confirms residual_capacity separately; it cannot replace
C1 or automatically enlarge P00/P01. Ordinary residual equality retires gating.
Recursive growth and hard dispatch are deferred.

## Finite experiment matrix and gates

Each plan expands all attempts before fitting, records unavailable/failed cells,
measured P50/P90 time/disk/RAM, free space and maximum checkpoint transaction.
Different matrices use fresh teachers except explicitly reused ablations.

| Matrix | Fixed work (initial maxima) |
|---|---|
| M0 |4 teachers*R2*8 roles=64 final fits: keep/top/random/short320/warm95/fresh94/fresh95/static partial;520 short trials|
| M1 positive |4 bases*2gamma*2sigma*R2*5 roles=160 fits; keep/top/random/generating-current-oracle/true-contrast-oracle; joint32 fits only after WP04|
| M1 negative |4 populations*4 teachers*R2*3 roles=96|
| M2a |4 teachers*R2*3 actual snapshots*2 streams*5 roles=240; keep/top/random/warm94/warm95; one expansion to8 teachers|
| M2b |4*R2*65 fixed300 singletons/keep=520|
| M2c |4*R2*(up to8 candidates+keep)=72 fixed8000 trajectories, prefixes300/1200/8000; top min(4,e), seeded remaining min(4,e-top)|
| M2d |4 fresh h8/gamma2/sigma.1 teachers*R2*(keep+eligible singles+pairs); nominal296, e<=7 gives<=232 fitted trajectories|
| M2e |4*R2*5 ordinary recipes=40 neural fits;48 ridge solves,8 retained heads|
| M3 |8*R2*10 roles=160 final fits and1040 short300 trials; one uncertainty expansion to16 teachers|
| M3lambda |9 pairs*4 reused M3 teachers*R2=72 entries; default8 cacheable; if changed global pair,8 fresh*R2*4 roles=64 validation fits; no expansion|
| M4 |8*R2*(3 directions*3 placements+keep)=160|
| M5 |8*R2*(12 growth+6 ordinary+keep)=304|
| M6 prune |8*R2*9=144 finals plus48 common candidate fits|
| M6 residual |8*R2*6=96 residual fits plus16 parent procedures|
| C0 |16 teachers*R2 initially; one variance/runtime-driven extension to32|
| C1 |K32–128*R2*frozen exact roles, independent T262144|
| C2 |separate C0/C1 root,3 residual roles plus2K strong-parent procedures|

M2e ordinary recipes: .005 constant legacy, .001 constant, .005 cosine horizon
16000/min0, .005 constant WD1e-4, WD1e-3. Ridge head: F-centered/RMS hidden columns,
drop zero columns, unpenalized intercept, mean-square objective plus alpha||w||²;
alpha{0,1e-6,1e-4,1e-2,1,100}, alpha0 stable least squares, G chooses smallest-alpha
ties, V/E once. M3 roles: keep/cheap/random/short300/local-singleton/joint/
unconditioned-joint/pure-span/warm95/fresh95; ridge.01 fixed.

Gate A after WP03: useful joint capacity on >=half positive-control seeds and
opportunity on >one independent primary teacher. Four discordant/noisy primary
teachers admit next four once regardless of favorable sign. Insensitive oracle
pauses efficacy interpretation; retain original cells. No opportunity with working
controls retires population/treatment or admits a specifically measured ceiling.
Earlier growth worsened V is rejected; common recipe gain is not placement.

Gate B: exact selector must improve held-out joint-gain prediction and repeatable
benefit over random on multiple teachers at qualified cost. Main M3 may expand
8->16 only if frozen exploration interval cannot distinguish2% from no gain.
Algebra-pass/local-prediction-fail permits only M3lambda: global pair minimizes
equal-teacher V NMSE averaged over repeats, ties tolerance above and ascending
(lambda0,lambdag). If default wins, close correction immediately. Otherwise
8 fresh/R2 keep/random/defaultjoint/correctedjoint, no expansion/replacement pair.
Unresolved result closes branch negative/inconclusive. Penalty-sharing-only gains
need ordinary regularization controls and cannot establish new tangent directions.

Gate C: justified retention/pruning or residual branch, at most one redesign,
8->16 development expansion once. Selected pruning must beat random or establish
useful retention; residual must beat ordinary residual. Attribute accordingly.
Gate D admits native anchors only after role/runtime/resource qualification.
Gate E chooses at most one P00 and a labeled secondary alternative before freeze.
Gate F passes design/power/resources or freezes estimation/retirement.
Gate G interprets final lower>delta useful, upper<delta rules out useful margin,
overlap inconclusive. No retuning on T or appended teachers after peeking.

Builder operational definitions, fixed before new outcomes (development screening,
not new confirmatory tests): average R2 V NMSE within teacher/base-instance first.
Positive-control sensitivity means true-contrast oracle improves >=2% over random
at gamma2/sigma.1 on >=2/4 predeclared instances; also report generating-current
oracle separately. Primary opportunity means at least one fixed growth/capacity
arm improves >=2% over the best same-snapshot/treatment competent ordinary keep
reference on >=2 independent primary teachers. If fewer than2 first-four teachers
qualify, or the whole-teacher95% descriptive interval for the best fixed contrast
overlaps both zero and2%, execute the next four once; no further Gate-A sampling.
Gate B/C repeatability means positive equal-teacher mean V log benefit and >=2%
benefit on >=2 teachers without >5% degradation on a majority of teachers.
Improved held-out joint prediction means greater paired Spearman correlation
between predicted set gain and V finite gain than local-singleton ranking on a
majority of defined small-oracle teachers, with undefined/all-tied cases reported
and not counted as successes. Require >=2 defined teachers; otherwise inconclusive.
Main M3 uncertainty expansion triggers only when its fixed joint-versus-random
whole-teacher95% descriptive interval spans both zero and log(1/.98).
Unqualified cost explosion means exceeding the qualified worker memory/storage
envelope or inability to complete without weakening controls. Per19October user
amendment, elapsed time beyond120minutes alone does not retire a quality branch;
expose measured Pareto position and do not claim efficiency from quality.
These conservative builder choices make qualitative gates executable and must
remain labeled as choices made at implementation, not historical preregistration.

## Transfer data and split contract

Acquire official UCI zip https://archive.ics.uci.edu/static/public/ID/slug.zip;
check extracted raw SHA256 before use. Changed bytes retained separately.

| Dataset ID/slug | Raw / hash | Schema and grouping |
|---|---|---|
|291 airfoil+self+noise|airfoil_self_noise.dat / 74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3|1503 rows,5 numeric; sound pressure;106 groups angle/chord/velocity/displacement thickness, excluding frequency|
|165 concrete+compressive+strength|Concrete_Data.xls / 710076c66b9ca3f8050e7942f3dcbdbe04013534daeb0077ffd3079a52d8e0c4|1030,8; strength;428 mixture groups seven ingredients excluding age (age predictor)|
|294 combined+cycle+power+plant|CCPP/Folds5x2_pp.xlsx Sheet1 only / ccd490981db2a2f079963b3d9f0aea30d9d338900a0285428dfc6385396f4651|9568,4; PE;9527 exact-input groups; five sheets same multiset|
|189 parkinsons+telemonitoring|parkinsons_updrs.data / f2c7d5025dec4e92e7feae367a5f7ccf58789a10ac6b54bdf15976c599f9dd39|5875,19; motor_UPDRS;42 subjects; exclude subject# and total_UPDRS; keep age/sex/test_time/16 voice|
|275 bike+sharing+dataset|hour.csv / e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f|17379,12;cnt;731 dates; exclude instant/casual/registered/dteday features|
|242 energy+efficiency|ENB2012_data.xlsx / 0089fcffc1415e41e2ff63730cca5280efe54fc43d722dfde1b0aaa808e35dc4|768,8;Y1 excludeY2;12 X1–X5 geometry groups; simulation separate|

Remove all-blank Energy spreadsheet rows/columns only. Raw datasets have no
observed missing predictors. Airfoil angle numeric. Categories: Energy orientation/
glazing distribution; Bike season/month/hour/weekday/weather/holiday/workday;
Parkinson sex. Bike year ordered numeric; dates metadata. Record encoded dimensions.
No-category methods use F one-hot vocabulary with explicit unknown column;
splines continuous only plus categorical indicators. Add hash-pinned xlrd/openpyxl
in separate data environment. Never bypass legacy spline/NODE category guards.

Five group outer folds: sort groups descending size plus seeded tie key, assign
greedily balanced rows. Within development split whole groups F/G/V target70/15/15,
min2 groups each. Canonical seed root2026101803 with dataset/outer-v1 labels.
Lexicographic assignment minimizes (remaining-minimum violation, projected squared
capacity deviation, current group count, seeded priority); deterministic repair
moves smallest possible group from above-minimum role with least deviation cost.
Fail if unsatisfiable. Store assignments/tie keys/moves/actual counts. Energy's
9–10 development groups necessarily lower F fraction; test explicitly.

Bike expanding T boundaries2012-01-01/03-14/05-26/08-07/10-19, ending next boundary
or after12-31. Preceding development dates chronological70/15/15; exclude first
24 actual clock hours at G/V/T starts. Preserve165 absent timestamps. Later dates
never enter earlier fitting; repeated vectors across dates allowed.

Primary outer fold0 for grouped datasets and last Bike block. Before global
freeze only primary F/G/V may inform methods/search/budgets; secondary folds'
development can contain primary T and must stay unopened. Complete168 primary
searches, freeze procedure and primary model hashes, then run672 secondary under
unchanged supports with no feedback. Other folds are descriptive, not independent
datasets. Group/day-weighted MSE primary, rows within group/day equal; report
row-weighted RMSE/MAE also. Five observational datasets primary; Energy separate.

## Mandatory native procedures, search and resources

P00 proposed; P01 exact matched random (same support except selection); P02 warm
ordinary growth; P03 tuned MLP; P04 ResNet; P05 static centered partial/full and
direct leaf; P06 RealMLP-TD official pytabkit; P07 TabM plain/embedded k32; P08
CatBoost; P09 XGBoost hist; P10 LightGBM constant; P11 LightGBM linear; P12 NODE;
P13 mean/ridge/spline. Optional FT-Transformer/TabR/TabICLv2/TabPFN-3.5 core/Fast
qualified before T by assets/hardware, not ranking. Missing mandatory native
anchor blocks full-suite superiority; optional unavailable limits wording.

Pin official commit/tag/wheel hashes/native recipe/full resolved settings/weights/
license/import location; no floating main. Investigation pinned TabM
28e47ae301c92ec37787dde1ce923a0793f405b4 and GradMax
c1f3d1ff366d794d06d3c77a6ac92e8eccfd90c3 only; resolve others, do not invent versions.
Use official primary documentation when qualifying current APIs.

Each quality-first family sequential search has <=32 attempted configs, official
anchor first, two fixed HPO seeds (not seed candidates), and no wall-time cap on
its native fits. Eligible completed retained models at10/30/120-minute prefixes
form the separate resource-bounded view; completing
after prefix is ineligible there. Paid failures/caps/startup/preprocess/parent/
scoring/probes/V evaluation included. Native minimum schedule must not be weakened
to fit a prefix: retain it as unavailable there, and finish for the quality view.
Search supports fixed before168 primary jobs; no V-driven range changes/restarts.
840 total jobs=14*6*5*2,168 primary/672 secondary; maximum26880 attempts
(primary5376). The earlier1680/336worker-hour ceilings belonged to the superseded
120-minute policy and no longer cap quality work. Measure anchor P50/P90 time,
disk/RAM and schedule first. Default memory16GiB/worker, same CPU thread allocation;
GPU track separate with hardware/precision/power and cost when available.

Exact semantic search supports:

- P00/P01/P02: h{64,128}, growth fractions{1/8,1/4,1/2} only after qualification;
  admitted direction/treatment only; historical h64/.005/epsilon.05/tau1 anchor;
  LR log[1e-4,1e-2], WD zero or log[1e-7,1e-2], constant/cosine,batch128/256;
  exact ordinary lower/upper parameter brackets.
- P03: MLP-D [128,256,128]/ReLU/dropout.1/default init; historical one-layer;
  depth1/2/3,uniform width64/128/256/512,ReLU/GELU,dropout0/.1/.2,
  LR log[1e-4,1e-2],WD zero or log[1e-7,1e-2],batch128/256.
- P04: ResNet-D two width128 blocks,expansion2,ReLU/BatchNorm,hidden dropout.25,
  residual.1; search blocks2/3/4,width128/256/512,dropouts0/.1/.2, same LR/WD/batch.
  P03/P04 anchors AdamW .001/WD.01,batch128,max1000epochs,G each epoch/patience20,
  best restoration,no scheduler,native RTDL F-quantile/noise transform,RNG fixed,
  category embeddings8. These anchors are Appendix C.4/C.6 MLP-D/ResNet-D in
  RealMLP, adapted from McElfresh, not universal RTDL defaults.
- P05: partial/full centered/direct leaves,h32/64/128 qualified; ordinary bounded
  neural LR/WD/dropout/schedule where applicable; retain init and parameter totals.
- P06: native [256]*3,PBLD,parametric Mish/NTP,AdamW beta2.95,batch256/full256
  epochs,LR.2 native factors/coslog4,scheduled dropout.15/WD.02,target scale/clip,
  last-best G. Pin actual Table C.14 distribution/probabilities/dependencies:
  LR log[.02,.3],dropout0/.15/.3,WD0/.02,layers[256]*3/[64]*5/[512],
  ReLU/SELU/Mish,native embeddings/scaling. Factory extraction prerequisite.
- P07: plain TabM.make and F-only48-bin embedding16/versionB/activationFalse,k32,
  AdamW.002/WD.0003,batch256,clip1,patience16. Search blocks2/3/4,width64–1024
  step16,LR log[1e-4,5e-3],WD zero or log[1e-4,.1],dropout zero or U[0,.5],
  bins2–128,embedding8–32 step4. Mean member training losses; validation mean
  prediction. No test-per-epoch notebook logging.
- P08: native RMSE,max10000/G stop200,depth4/6/8/10,LR log[.01,.3],
  L2 log[1,100]; random-strength/bootstrap/dependent knobs fixed native anchor.
- P09: hist squared error,max10000/G stop100,depth3/5/7/9,eta log[.01,.3],
  child weight log[1,100],subsample/columns U[.6,1],lambda log[1e-3,100],
  alpha zero or log[1e-4,10]; correct best-iteration API.
- P10: max10000/G stop100,leaves15/31/63/127,LR log[.01,.2],min rows5/10/20/50/100,
  feature/bagging U[.6,1],bagging_freq1,L2 zero or log[1e-3,100].
- P11: linear_tree at dataset construction,F scaling,max5000/G stop100,
  leaves7/15/31,LR log[.01,.2],min rows20/50/100,L2/linear_lambda zero or
  log[1e-3,100]; verify pinned category/missing/monotonic limitations.
- P12: native2048trees/depth6/one layer/tree_dim3,QHAdam.001,betas(.95,.998),
  nus(.7,1),eps1e-8,WD0,batch512,G100,patience5000updates,rolling5 native
  averaging/selection/data-aware initialization/normalization/entmax.
  v2-only variants trees512/1024/2048,depth4/6,LR multiplier.5/1/2; legacy guards
  unchanged. Three-update engineering run is quality-ineligible.
- P13: mean1,ridge7 alpha10^k k{-6,-4,-2,0,2,4,6},cubic spline knots4/8/16*7=21;
 29 configurations exhaustive, no duplicate filler.

Optional TabICLv2 ensembles8/32; exact TabPFN3.5 core/Fast separately; full-F context,
precision/cache/assets/pretraining provenance, any subsampling separately labeled.
No paid APIs required. Qualify one direct official fixture prediction/checkpoint;
lazy method imports keep ridge deployment independent of Torch/NODE startup.

Charge shared prerequisites once per standalone method and once physically in
cohort, never add nested timers to enclosing wall time. Cache identity includes
source/data/recipe and disclosed saved physical work. Portfolio P02–P13 selection
uses V only and pays all12 searches (up to24h/split/repeat), not one2h search.
V metric group/day-weighted normalized MSE; numerical tie tolerance1e-12*max(1,
abs(best)); earliest eligible trial/fixed lowest family ID. Separate policy-HPO
contrast from fixed-treatment placement mechanism.
Choose the portfolio family separately for each split/search repetition using
that repetition's V scores, then average the two retained T risks before the
primary log ratio. Do not pool V scores to select one family across repetitions.

## Prospective inference and freeze

Synthetic risk=original-unit MSE/F target variance ddof0. Average R2 within
teacher before B=log(meanRisk_random/meanRisk_selected). Useful margin
delta=log(1/.98)=.0202027073; design alternative deltaA=log(1/.95)=.0512932944.
Primary two-sided95% Student-t interval over independent teacher effects, K>=32;
useful win lower>delta (one-sided alpha.025). Whole-teacher bootstrap sensitivity,
all signs/medians/tails/no-op rates; undefined correlations stay undefined.
Secondary competent keep/warm ordinary/fresh ordinary family uses Holm-adjusted
one-sided teacher effects/compatible lower bounds. Informed-placement causal
wording gates on primary; pipeline gains may be reported separately.

Before calibration lock provisional method and variance-only script. Initial16
teachers/R2;10000 bootstrap SDs. s_U=max(normal-theory one-sided90% SD bound
sqrt((Kcal-1)s²/chi2.ppf(.10,Kcal-1)), bootstrap90th SD percentile). Extend once
to32 if bootstrap90% SD upper/lower>2,lower0,or runtime P90/P50>2, independent of
effect sign. All-zero/unidentifiable effects -> retire or estimation, not invented
precision. Starting K=ceil(((z.975+z.90)*s_U/(deltaA-delta))²), round up grid
32/40/.../128. At each feasible K simulate100000 cohorts in chunks from centered
empirical scaled to s_U,matched normal,matched t5. At alternative binomial95%
lower power>=.90; at margin null upper rejection<=.03. Increase fixed grid only.
If no validated/resource-feasible K, freeze estimation-only with attainable
precision; no alternative analysis fishing or changed margins. No N looks after T.

Practical primary: V-selected baseline champion P02–P13 versus P00; average risks
over two search repeats before log ratio; equally weight five observational dataset
log ratios (Energy separate). 20000 paired group bootstrap resamples; Bike7-day
moving blocks. Same indices across methods/repeats, independently resample each
dataset for aggregate. Conditional test-sampling uncertainty for fixed fitted
models, not broad dataset-population or retraining inference. Leave-one-group-out
and Bike3/7/14 sensitivity; Parkinson8–9 subjects fragile,Energy2–3 geometries
descriptive ranges only,CCPP exchangeability unverified (no time metadata).
Useful finite-suite win requires aggregate lower>delta,positive aggregate each
search repeat,all task regressions disclosed; report >=4/5 nonnegative and any
>5% degradation. Failed robustness -> heterogeneous aggregate wording. Individual
mandatory family claims Holm adjusted; portfolio win is not every-family dominance.
If dependence qualifications fail, report finite observed quality and qualified
intervals, no population-confirmed superiority or dropped unfavorable tasks.

Quality–cost is separate. Same-budget quality requires same useful-risk criterion.
Matched-quality risk-ratio upper<1.02 and cost savings lower95% bound>.20; paired
timing interval must exclude zero. At least30 paired randomized deployment timing
blocks, CPU float32 batch1 including preprocessing primary,batch1024 secondary,
fresh cold import/load and resident timings separate; disclose latency,throughput,
RSS,serialized bytes,operations. Hard routing claims need actual sparse dispatch.

Implement and fixture-test ALL production WP09–WP11 protocol/confirmation/metric/
cost/audit/report code BEFORE production WP08 primary searches or C0 calibration.
Then run under final executable revision; Gate F binds SAME source,procedure,
primary model hashes,adapter/data locks,attempts,hypotheses,missingness and budgets.
Generated outputs are declared artifacts, not scientific source. Later source
changes create new identity and explicit affected-run disposition. T prediction
once per retained model in sealed evaluator; immutable access events. Same-runtime
checkpoint replay of one predeclared role is engineering, not a new teacher.

## Work packages, release and recovery

WP00 qualify source/runtime/inventory/arithmetic; WP01 narrow state hardening and
exact valid replay; WP02 runner/snapshots/controls/roles; WP03 opportunity/Gate A;
WP04 exact joint/Gate B; WP05 diagnosis-admitted bounded factors; WP06 one redesign/
Gate C; WP07 data/native qualification/Gate D (independent after WP00); WP08 fair
search/Gate E; WP09 variance/design/freeze/Gate F; WP10 frozen execution/Gate G;
WP11 independent audit/report/release. Every package records purpose,evidence,
dependencies,files/change,verification,commands/artifacts,decision,integration.

Run records include argv/cwd/env, Git/package/patch/config/recipe hashes, package
wheel/import/runtime/hardware/thread provenance, raw/transformed data/role/group
hashes, fitted transforms, RNG/sampler/HPO lineages, topology/contrasts/eligibility/
scores/spectra/rank, named births/applied ages, all attempted statuses and costs,
selected/executed updates, original-unit predictions, raw uncertainty inputs,
protocol/access/review/claim/blocker records. Inventory tracked path/bytes/SHA256/
kind/owner/semantic sections/execution/consequential claims. Structural scan is
not semantic review. Keep absent raw-artifact map explicit.

Recovery: preserve dirty source, reconcile affected claims; missing historical raw
permits arithmetic/new work; platform changes need clean hashed lock and real
receipts, never forged Mac identity; duplicate dist-info -> recreate clean env;
fixed synthetic-byte mismatch -> same-runtime old/current comparison and retained
arrays, no changed historical expected hash; changed native API -> pinned official
adapter/anchor qualification; missing weights -> optional unqualified; interruption
-> exact same queue/state and paid cost; RAM/disk -> lower concurrency/new qualified
track, no hidden F subsampling; leakage -> invalidate study and fresh protected
evidence. TINY author code was unavailable: paper-level derived comparison must
not be called author-code replay.

Canonical report recomputes raw prediction metrics independently, statuses/counts,
teacher/group units, intervals/multiplicity/costs. Include all attempted branches,
regressions, native qualifications and strongest alternative explanations. Three
final reviews: evidence/causality, validity, clean-checkout execution. Historical
publication remains. New `reports/research-v2.md` and reproducibility docs support
completed, estimation, retired, and blocked studies honestly. Each consequential
claim maps to report/source/recomputed quantity/new execution/derivation/literature/
hypothesis with supported/uncertain/contradicted and counterevidence.

Closest work: Net2Net/Network Morphism (preservation), SSD/GradMax (spectral/gradient
growth), Firefly (joint overgrowth/pruning), TINY arXiv2405.19816v2 (parent tangent/
covariance), Gauss–Newton morphisms arXiv2411.05855 (curvature/nonadditivity), LAG
arXiv2406.02469 (horizon), staged/variance-transfer growth (optimization), ANT (leaf
split/deepen/keep), NODE, RTDL, RealMLP arXiv2407.04491, TabM arXiv2410.24210,
TabR, TabArena. Residual learning has Cascade-Correlation/boosting/GrowNet prior
art. Defensible prospective novelty is the exact estimator/constraint/cost and
finite evidence for this parameterization, not first adaptive/residual-aware growth.
Future literature/API assertions require primary-source verification. The release
may validly be a useful component, a scoped win, a null result, or inconclusive.

## WP02 implementation resolutions before stress fitting

The separately named `research/v2-stress-teacher` wraps the unchanged native
teacher. Custom prevalence fixes all four component probabilities at .05 or .2;
it does not change the legacy `rare` constructor. For covariance condition1 or10,
draw latent inputs z using the native sampler and publish x=z@L. Targets/noise
and the base teacher's separate reference normalization remain in z coordinates.
For condition10, draw a signal-subspace QR basis from
`seed64(seed,'stress-v2','covariance')`, fix QR signs by nonnegative R diagonal,
and set L_signal=diag(sqrt(geomspace(1,10,active)))@Q.T. Nuisance coordinates
retain an identity block and never mix into signal. Condition1 uses L=I. Store L
and the complete base teacher; observed population covariance is L.T@L. These
are descriptive stress populations; no fitted stress outcome informed this rule.

A replay certificate requires ordered raw native events from stage start to
terminal selection: unique contiguous updates, complete minibatch IDs/loss/LR,
G cadence, matching executed count, and earliest strict G-best. Missing or
contradictory evidence is incomplete, not a proof. Compare update fields and G
metrics independently. Cap stopping alone never establishes the patience replay
premise; even verified replay never establishes exhausted training headroom.

User resource direction (18 October): use best judgment on this shared computer;
other processes may change disk/RAM/CPU availability. Default to one training
worker, recheck resources before admitted batches, retain interruptions and paid
work, and use the declared same-state/same-seed infrastructure recovery. Do not
interpret contended timing as a dedicated-worker efficiency result or relax
scientific budgets, seed pairing, or success gates to accommodate contention.

### NODE bounded-memory execution amendment (19 October)

The user says20GiBfree RAM is unlikely and requests the same scientific comparison
within available RAM. Builder choice: add an explicit `execution` configuration
value `bounded_memory_v1` to NODE-v2, leaving absent-field legacy behavior intact.
This is an execution identity, never a V-selected hyperparameter or an automatic
fallback. Keep the native2048trees/depth6/tree_dim3, effective512-row minibatches,
F-prefix initialization on up to1000rows, QHAdam settings, row order, report100,
rolling5, and strict5000-update patience. Vendor sources remain unchanged.

Initialize the native single ODST layer on the entire original prefix and set its
native initialization flags without computing the discarded full-prefix output.
Verify initialized tensors and RNG exactly against the native first call. Split
each training minibatch into32-row chunks; each chunk contributes its squared
error sum divided by the original minibatch count, including short final chunks.
Accumulate all gradients, then take exactly one QHAdam step. Evaluation uses
at most32rows per forward, including an externally requested larger query batch.
No training rows, trees, channels, precision or candidate checkpoints are removed.

Numerical acceptance is CPUfloat32 rtol1e-5/atol1e-6 for predictions, gradients,
QHAdam tensors and short finite trajectories against the full native calculation;
initialization/RNG/order/optimizer ages remain exact. Also check CPUfloat64
gradient algebra at rtol1e-10/atol1e-12. Retain actual maximum errors and any
unstable/failed cases. Chunking changes floating-point reduction order, so this
qualifies equivalent arithmetic within declared tolerances, not byte-identical
long training or guaranteed identical checkpoint decisions. A completed native
fit remains necessary for native eligibility; a bounded update/resource proof
alone cannot establish quality. All old stops/costs remain visible.

First measure a disposable full2048-tree engineering workload under a4GiBRSS
limit with a live5.5GiBmemory proxy and1.5GiBdisk floor; only after it passes
admit the unchanged Energyprimary/seed17 native anchor at the same resource
limits and7200s. This is a newly verified execution track, superseding waiting
for20.5GiB for this branch. Do not rerun the old unchunked4GiBtrack. Preserve
frozen M2 and its held auditor; use a separate held source for NODE qualification.
