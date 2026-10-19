# NODE within the shared machine's RAM

**19October user amendment:** complete native results take priority over the
two-hour budget. The timed attempt below is retained historical evidence. The
next fit uses native stopping with periodic raw-training checkpoints; see
[the prospective implementation contract](results-first-native-execution.md).

The user requested completing the same comparison without waiting for20GiBfree
RAM. This adds the explicit NODE-v2 configuration field
`execution: bounded_memory_v1`. It is a fixed execution policy, not a model-search
choice or a silent fallback. An absent field preserves the legacy calculation.

The pinned ODST forward constructs a routing tensor with shape
`[rows, trees, depth, leaves]`. At2048trees/depth6/64leaves/float32, one such
tensor occupies2.93GiBfor1000rows,1.5GiBfor512rows, and96MiBfor32rows. Backward
needs additional intermediates. The previous4GiBstop is retained; the size
calculation explains a memory pressure mechanism without claiming its exact
allocation caused every byte of the observed peak.

The new policy preserves all trees/channels, precision, F-fitted transforms,
initialization rows, effective512-row training batch, row order, QHAdam settings,
G cadence, rolling selection and native patience. It changes three allocations:

- Run the unchanged native initializer on the complete original F-prefix and
  set the same native flags, omitting the subsequently discarded routing output.
- Backpropagate32-row chunks, each squared-error sum divided by the complete
  original minibatch count. Accumulate all gradients before one optimizer update.
  This also handles short final chunks and short epoch-ending minibatches.
- Predict at most32rows per forward, including when a caller requests a larger
  external prediction batch.

No vendor file changes. Initialization tensors and RNG must match exactly.
CPUfloat32 comparison tolerances are rtol1e-5/atol1e-6; float64 gradient algebra
uses rtol1e-10/atol1e-12. The first seven checks passed in17.55s. A measured
small native-model case had maximum gradient/moment error4.6566e-10,
parameter error1.4901e-8, and exactly matching predictions after one update.
Float64 gradient error was4.44e-16. These are fixture results, not guarantees
of byte-identical long nonlinear trajectories or identical selected checkpoints.
The affected native/reference/snapshot/worker/support regression run passed
111checks with50native warnings in180.12s. It includes the unchanged direct
official comparisons. An initial command named a nonexistent worker test file;
it ran no tests, was retained, and was corrected. Actual full-shape resource
measurement below has now completed. See [the receipt](WP07-NODE-bounded-memory-receipt.json).

Admission sequence: pass affected regression checks; commit the new source;
hold a separate NODE checkout (the M2 auditor remains unchanged); measure a
disposable2048-tree/depth6/d16/F1025 workload with full1000-row initialization,
three updates of512/512/1rows, and closed prediction reload under sampled4GiBRSS.
Only after that succeeds admit the Energyprimary native fit at the same data,
seed17,7200s and native stopping. The live floor is5.5GiBmemory proxy plus1.5GiB
disk. This is a new qualified execution track; the unchunked4GiBtrack is not
retried. A three-update proof remains quality-ineligible. Keep every failure,
timeout, prior cost and resource observation. No Vlabels or protected T are
needed for these engineering checks.

The full-shape proof completed under held source `ea3993a`/`c2639fc8…`.
It retained615,169total/614,400trainable parameters and all specified rows/updates.
The supervisor sampled179RSS observations, with a maximum1,177,665,536bytes
(1.097GiB), and reaped the child after19.212825542s. The training phase itself
took2.804748084s; these nested clocks are not added. Independent checks verified
all supervisor commitments, regenerated the artificial inputs, checked full-F
coverage and reopened the closed model with exact predictions. The initial
stdin audit took6.205387208s. A retained executable audit then repeated those
checks solely to provide a durable replay command; both costs remain visible.
This remains a capped three-update engineering result.

The full Energyprimary/seed17 native fit ran in the separate held NODE checkout
with `bounded_memory_v1`,7200s, sampled4GiBRSS and unchanged native stopping.
It reached the two-hour time cap. All65,666RSS samples stayed below4GiB; sampled
peak was1,604,878,336bytes (about1.49GiB). Thus the observed memory barrier was
removed, while native completion remains unqualified. The child was terminated,
reaped and its process group verified absent. No successful fit receipt or
prediction model was published; executed updates are unknown, not zero. This
timeout is no predictive loss or general NODE infeasibility. See
[the full-shape receipt](WP07-NODE-bounded-full-shape-receipt.json).

The practical P12 support compiler now includes this same explicit policy in
its native anchor and every search configuration. The HPO coordinates remain
trees/depth/LR multiplier; memory execution is fixed outside that space.
One missing-policy regression failed first. All34affected support/compiled-worker
checks then passed14.25s with native RSS observation. An earlier sandbox run
retained33passes/one failure because `/bin/ps` was blocked before either owned
child started; native permission resolved it without a code or guard change.
No production search or qualification is granted. The timed-out native attempt
retains its original separate source; see [the compiler receipt](WP08-NODE-fixed-execution-policy-receipt.json).

The independent inactive audit authenticated the queue, source, original
Energy F320/G128/query128 bytes, full native recipe and all process/resource
commitments. Its containing job cost was7200.104335625s, enclosing paid search
7200.071415251s and process7200.022782625s; do not add them. Earlier failed attempts
and proof/qualification costs remain separate. At the declared120-minute budget
this Energy anchor is unavailable; the full mandatory-suite claim remains
blocked pending qualified missing comparisons. A larger common budget is a
possible separately declared recovery before protected data, with an updated
whole-program forecast. No repeat of this same capped run is currently admitted.
See [the timeout receipt](WP07-NODE-bounded-native-timeout-receipt.json).
