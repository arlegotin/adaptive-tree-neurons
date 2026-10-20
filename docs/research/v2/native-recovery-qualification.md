# Closed full-size NODE recovery qualification

`research.native_recovery_proof_v2.qualify_fullsize_node_recovery` supervises the
existing fixed recovery operator with the qualified interpreter, one numerical
thread, an owned RSS guard and no wall-time deadline. The operator is
`docs/research/v2/scripts/check_node_training_recovery.py`; its bytes, source and
core/runtime identities are bound before and after execution.

This is a numerical proof with 2,048 trees, depth six, three output channels,
16 features, F1025/G65 and native configured batch512 under the declared
32-row memory policy. It runs 105 reference updates, interrupts the second fit
before update103 after102applied updates, restores checkpoint100, and executes
five more updates. That is212applied updates plus one additional unapplied
gradient attempt across three paid fit paths. The 105-update engineering horizon
is separate from native-quality training, which retains its uncapped stopping
rule. The saved checkpoint100 is inside a sampler epoch.

`load_fullsize_node_recovery(reference)` authenticates the closed managed
process, source/operator, inputs, actual core/import files and three fit receipts.
It streams the numeric arrays and checks actual full model/optimizer geometry,
float32 parameters, uint8 initialization flags, native bin codes, moment ages and
nonnegative second moments. It reconstructs only the small, predeclared
engineering F/G arrays. It constructs no model and runs no fit, prediction or
quantile refit.

The reader independently re-encodes and compares the saved reference and resumed
states, including raw and selected weights/moments, RNG, sampler, windows and
row history. Only per-update elapsed nanoseconds are removed. Exact prediction
and closed-reload assertions remain trusted execution of the pinned operator;
state equality is checked separately from those assertions. The reader does not
independently repeat the numerical updates.

The result qualifies this recovery capability. It grants no full native workload,
quality comparison, Gate D, HPO or protected-evaluation authority. Old c33 proof
and model bytes remain unchanged; they do not migrate into this current source.
The eventual registry still needs actual final-source data/software/native
anchors/supports and these recovery references.

Verification passed40recovery/allocation/process checks. One actual numerical
proof supplies six recovery cases; controlled mutations restore the complete
original inventory. Altered state digests and recovery counters reject. Paired
int64 initialization flags and a contradictory managed interpreter were accepted
before the typed-geometry/runtime joins were added, then rejected. Those
manipulated records do not establish damage in ordinary runs. Exact logs,
resources and source identities are in the
[component receipt](WP07-native-recovery-proof-receipt.json).
