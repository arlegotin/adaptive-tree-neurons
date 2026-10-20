# Recorded native training and workload observations

`research.native_workload_v2.inspect_native_workload` consumes the eight exact
arguments accepted by `audit_native_generation`. It reopens that managed native
fit, its primary F/G and feature-only V roles, and its externally committed
closed output inventory. It reads no V labels or protected arrays. It does not
fit a model, solve coefficients or make new predictions.

The current implementation covers **all fourteen mandatory families, P00–P13**.
The complete native qualification registry remains unfinished. Implemented
record readers alone do not qualify the real-data workloads.

The reader captures mutable proposal inputs, hashes metadata buffers before
decoding, and rechecks the full generation/inventory at exit. It joins the
original process start and owned spawned PID/group, then independently recomputes
sample count and peak from the committed raw RSS stream. A genuine zero-RSS exit
sample is allowed when a positive observed peak exists. Understated peaks,
incorrect ownership and changed source/input/proof bytes cannot supply a passed
observation. Samples measure the root worker; descendants and unobserved peaks
remain outside that measurement.

For the four own-model families it joins the complete F/G preprocessing and
transformed-input records, checks actual native topology and finite float32
parameters/int64 growth buffers, and reopens both training stages. Applied rows,
partial batches and complete observed epochs must agree with the sampler phase.
Recorded gradient norms, absolute learning-rate schedule, strict G-best flags,
patience/cap stops and selected/executed counters must agree. It checks recorded
optimization and selected tensors; it does not independently rerun neural updates
or establish persistent interrupted-training resume.

For P13 it checks the declared native configuration and composite source identity,
replays F-only preprocessing/design arithmetic and identities, verifies F-derived
spline knots and actual encoded/design dimensions, and checks the complete closed
coefficient/knot payload. Payload dtype, shape, finiteness, archive allocation,
metadata commitments and worker geometry must agree. This arithmetic uses no
least-squares solve. It checks recorded full-F context; the original managed fit
and closed reload remain trusted execution evidence, not independently repeated
optimization. Iterative gradient checks are explicitly inapplicable to this
closed-form family.

The result reports recorded facts and retains `native_workload_qualified`,
`quality_eligible` and `T_accessed` as false. A tiny synthetic Energy-schema role
fixture carrying the known raw digest is **not official acquisition evidence**.
Final qualification must still join actual raw/prepared data, final-source native
software profiles, all 120 full native anchors, support geometry, workload and
recovery evidence, and the complete production candidate/protocol identity.
Earlier held-source models and their flags remain unchanged.

For MLP, ResNet and both TabM variants, the reader streams retained epoch orders
one epoch at a time and verifies complete F permutations and their history hashes.
It checks recorded gradient bounds, finite selected tensors and native stopping
and checkpoint selection. Ordinary neural epochs start at one; TabM epochs start
at zero. TabM trains mean member MSE and selects with native averaged-prediction
RMSE. Its committed raw targets use float32; ordinary neural uses float64.
Existing fitted feature transforms are reopened for input arithmetic without
refitting their quantiles. Scalar F-target statistics are recomputed using the
native convention. The original quantile fit and optimizer execution remain
trusted managed evidence. This does not establish persistent training recovery.

RealMLP's reader preserves all 256 native epochs and its dropped final partial
batch. It checks admitted and omitted F rows, cumulative visits, native last-best
selection, initialization mapping records, original float32 input commitments
and the closed inference graph's finite tensors/recipe. Its recorded evidence has
no gradient-norm trajectory: nonfinite-gradient rejection remains a trusted native
source hook. The reader binds the extracted official factory parameters without
re-extracting them in the coordinator.

The four native tree readers check full F-derived scaling/vocabularies, F/G input
commitments, transformed dimensions, native selected/executed round histories
and stopping rules, and the closed native model bytes. Native internal gradients
and sampled row visits are not exposed by these adapters. They are labeled as
unexposed rather than treated as independently checked traces.

Every implemented tensor-payload reader checks NPY headers and declared/actual
byte lengths before NumPy materialization. A small archive with an inflated shape
cannot authorize an oversized allocation.

NODE's reader decodes portable metadata and streams numeric tensor commitments
one chunk at a time. It materializes only bounded transform/sampler arrays,
preserving the full model/data. It checks complete F/G identities, native
initialization prefix, recorded rows/gradients, native G reports and rolling
windows, strict selection and patience, exact saved sampler order, QHAdam ages
and nonnegative second moments. It also joins the prediction-state binding.
Actual optimizer updates, rolling-weight averaging and RNG recovery are not
independently repeated. Persistent training recovery still requires its separate
qualified checkpoint evidence. A short geometry probe does not substitute for
a complete native procedure.

Implementation verification uses real closed fits and controlled re-committed
content faults, not hand-filled success records. The exact receipt distinguishes
those software fixtures from any later actual-data/source qualification.
The [P13 component receipt](WP07-native-training-P13-observation-receipt.json)
records 23 passing reader/generation checks, including all three actual estimator
formats, after the retained missing-interface, payload and typed-identity failures.
No full repository pass or other-family implementation follows from that count.
The [own-model receipt](WP07-native-training-own-models-receipt.json) records the
later 32-case combined pass. Actual full native procedures used F325/G130 and
eighteen encoded features, including 69-row partial batches. Consistently
recommitted duplicate rows, false G-best flags, negative gradient norms, wrong
tensor dtype and a boolean update counter are rejected. The final counter case
exposed a reader gap before the typed checks were added. These manipulated
software fixtures do not establish corruption in ordinary or historical runs.
The [epoch receipt](WP07-native-training-epochs-receipt.json) records a later
40-case combined pass, including four full native epoch procedures and
consistently recommitted duplicate-row, negative-gradient, boolean-epoch and
false-selected-epoch faults. This count includes the earlier components' cases;
it is not an additional forty independent studies.
The [external-family receipt](WP07-native-training-external-receipt.json) retains
five missing-family failures, five passing full native procedures, and the
subsequent 48-pass/two-failure combined run. Both failures came from a corruption
test helper assuming every snapshot used `manifest.json`; tree snapshots use
`state.json`. After correcting only that helper, fifteen affected checks passed,
including both tree content faults. No whole-suite pass follows from this split
verification. The inflated-header check also failed before its allocation guard
and passed afterward without allocating the declared oversized array.
The [NODE/all-family receipt](WP07-native-training-NODE-receipt.json) records the
final 57-case combined pass. The NODE parser fixture ran its actual native stop
at 9,001 updates, selecting update 4,000. Its declared search geometry is 512
trees/depth four with ten F rows; configured batch 512 emits native ten-row
partial batches there. This is not the 2,048-tree real-data anchor. One full fit
supplies the seven final NODE checks; controlled mutations restore its exact
original bytes and inventory after each case. Saved sampler-order, unknown-field
and receipt-counter gaps failed before correction. These manipulated records
do not establish damage in normal training or historical studies.
