# NODE within the shared machine's RAM

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
measurement remains pending. See [the receipt](WP07-NODE-bounded-memory-receipt.json).

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
