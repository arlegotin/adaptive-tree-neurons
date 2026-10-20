# First native-anchor development scores

This is an early **primary validation (V)** survey of existing native fits,
not the tuned transfer benchmark or a protected generalization test. It scores
115 of the120fixed anchor/dataset cells, with one fit seed17per cell. The five
unfinished NODE cells remain unavailable. No new model was trained, no seed was
chosen, and no T or secondary-fold development data was accessed.

The adaptive historical anchor beats its informed-direction random counterpart
on Airfoil, loses on Concrete and ties on CCPP, Parkinsons, Bike and Energy V
risk. Existing baseline anchors have lower V error on each of the five named
observational tasks. The static fully gated model has the lowest observed V
error on Energy, the separately reported small geometry simulation. These are
development leads and counterevidence, not established procedure rankings.

## Procedure and verification

The [survey decision](../docs/research/v2/primary-anchor-development-survey.md)
was committed before V-label scoring, and a byte-bound120-cell plan fixed the
115available predictions before outcomes were opened. Only primary outer0and
Bike outer4are included. Every original source, native configuration, seed,
closed-worker result, F/G/V-feature commitment and prediction remains visible.
Original resource failures and same-seed recoveries remain retained. Earlier
workers remain engineering-ineligible; offline arithmetic does not grant final
source/support qualification or practical-search eligibility.

F/G labels were available to native fitting; query features contained no V
targets. The logged coordinator now reads primary V labels. The metric is
original-unit MSE, equally weighting held-out groups (Bike days), divided by F
target variance with ddof0. All transforms were F-fit. Final models are their
retained F-fit/G-selected states; there is no F+G+V refit.

An independent reconstruction using original-unit errors, `math.fsum`, equal
group/day means and F variance agreed **exactly** on805metrics across115saved
prediction vectors. That is arithmetic and retained-byte evidence, not full
independent retraining. The [receipt](../docs/research/v2/WP07-primary-anchor-development-survey-receipt.json)
binds the plan, both arithmetic paths, prior source/qualification records and
the retained planning-format failure. All120statuses and row-weighted RMSE/MAE
are in the [complete CSV](../audit/research-v2/primary-anchor-development/scores.csv)
and [source-bound summary](../audit/research-v2/primary-anchor-development/summary.json).

## All anchors, same fixed primary splits

Lower V NMSE is better. Airfoil, Concrete, CCPP, Parkinsons and Bike are the five
observational tasks. Energy is simulation transfer with few geometry groups;
it is not pooled as a sixth observational dataset. An unavailable value is not
a predictive loss. P13 here is the mean anchor, not the full mean/ridge/spline
family. MLP-D and ResNet-D use their published/native anchors; h64 retains native
preprocessing/stopping rather than replaying the historical common recipe.

| Fixed anchor | Airfoil | Concrete | CCPP | Parkinsons | Bike | Energy simulation |
|---|---:|---:|---:|---:|---:|---:|
| Adaptive cheap P00 | .092186 | .209199 | .049448 | 1.246646 | .370625 | .079480 |
| Matched random P01 | .109896 | .202016 | .049448 | 1.246646 | .370625 | .079480 |
| Warm ordinary lower P02 | .124016 | .209276 | .049448 | 1.246646 | .345939 | .079480 |
| Warm ordinary upper P02 | .143294 | .201356 | .049448 | 1.246646 | .344501 | .079480 |
| MLP-D P03 | .128267 | .222729 | .043258 | 1.827357 | .203762 | .712354 |
| MLP h64/.005 P03 | .168831 | .305128 | .048485 | 1.677369 | .332955 | .022386 |
| ResNet-D P04 | .133495 | .212365 | .047167 | .938105 | .282016 | .038359 |
| Static centered partial P05 | .096119 | .259639 | .048075 | 1.359541 | .439146 | .288530 |
| Static centered full P05 | .120280 | .244151 | .047888 | 1.307316 | .312102 | **.009856** |
| Static direct partial P05 | .112964 | .245393 | .048642 | 1.371212 | .413241 | .093626 |
| Static direct full P05 | .117465 | .220630 | .048285 | 1.267869 | .374964 | .043453 |
| RealMLP-TD P06 | **.073302** | .242288 | .040641 | 1.043741 | .461231 | .303634 |
| TabM plain P07 | .088256 | .232539 | .039444 | 1.137402 | .209022 | .032626 |
| TabM embedded P07 | .090273 | .169020 | .032624 | 1.404445 | **.169430** | .197989 |
| CatBoost P08 | .118554 | **.129860** | **.027205** | **.738719** | .298609 | 1.827882 |
| XGBoost P09 | .141720 | .200201 | .031234 | .784080 | .380772 | .066489 |
| LightGBM constant P10 | .133238 | .168680 | .029635 | 1.441330 | .231001 | 6.561001 |
| LightGBM linear P11 | 1.206300 | .214310 | .029106 | 1.440821 | .370376 | 6.513854 |
| Full native NODE P12 | unavailable | unavailable | unavailable | unavailable | unavailable | 1.078700 |
| F mean P13 | .917735 | .883177 | .958839 | 1.516624 | 3.825311 | 3.184889 |

Bold identifies the lowest observed available anchor V score per column. It does
not designate a protected-test winner or a tuned-family champion. The full NODE
anchor on Energy reached native stopping under the explicit bounded-memory
execution; it is included despite its unfavorable V risk. The unfinished NODE
cases are not scored as defeats. Unusually large losses and every regression
remain visible; no dataset or weak anchor is removed after seeing scores.

## Implications and limits

The Airfoil adaptive-versus-random difference is a useful narrow lead, but one
split/seed cannot establish a reproducible placement effect. Concrete reverses
the direction and four coordinates tie. The [M3 primary synthetic experiment](research-v2-M3.md)
still closes its joint-selector branch as negative; this survey does not reopen
its exhausted sampling allowance or retrospectively select another M3 method.

Energy suggests that the gated representation can be well suited to a particular
structured simulation. Fully static gating spends capacity differently from
selective growth and has a different initialization/training path. Its favorable
V score cannot establish an adaptive-placement advantage, equal-cost dominance
or broad tabular superiority. There are only twelve geometry groups in the
whole dataset and two V groups in this primary split.

Native existing methods remain serious competitors; a weak common training
recipe would misrepresent them. Conversely, native defaults are not uniformly
strong on every grouped or future split, and these are not tuned procedure
results. No HPO, repeated search seeds, matched-cost comparison, uncertainty
claim or baseline portfolio test is supplied by this survey. No residual or
pruning redesign is automatically admitted: residual conditional structure and
ordinary-component controls remain unmeasured.

The full mandatory comparison still needs the remaining NODE executions,
final-source/workload/support qualification, completed production coordination
and pre-freeze code, the fixed primary searches, global freeze and untouched T
evaluation. Results-first execution retains native stopping without a wall cap;
it does not waive these evidence requirements or select better seeds. This
survey is permanent development evidence for that continuation.
