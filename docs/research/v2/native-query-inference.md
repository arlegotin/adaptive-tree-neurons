# Complete bounded feature-only inference

`research.native_query_v2.predict_native_query` owns a fresh native prediction
worker with an RSS guard, one numerical thread and no wall deadline. It loads
an immutable closed model and predicts **every** supplied feature row in fixed
1024-row chunks, retaining original row/group IDs and optional timestamps.
The request supports up to262,144rows and128raw columns within its explicit
archive allocation bound. It performs no fit or parameter update.

The request binds the current scientific source, model metadata, semantic
feature schema, feature archive and chunk treatment. Exact archive fields admit
only float64 X and identifiers; labels, extra request fields, object arrays,
changed hashes and stale source reject before native adapter import. Each
prediction chunk must have a consistent float32/float64 dtype, matching shape
and finite values. The worker and supervisor recheck request/query/model bytes,
including the complete snapshot inventory, before publishing a successful outer
record. Private imports remain method-specific. The caller retains all startup,
loading, preprocessing, inference, failure and publication costs.

This supplies inference, not training lineage or qualification. The worker
records`fitting_performed=False`, `labels_read=False` and all quality/protected
authorization flags false. A protected evaluator must independently validate
its frozen protocol and access events before giving the worker authorized
features. This helper cannot create or unlock protected T, choose a model or
compute risk. Its archive bound is not full workload/memory qualification.

Existing deployment timing measures batch1/1024prefixes, and prediction replay
checks a supplied existing vector. Neither was used to pretend that a complete
E65536prediction had been produced. The new component is needed for the bounded
[B0 native-backbone screen](B0-native-primary-backbones.md) and later evaluation.

Eight missing-module failures preceded eight passing cases in3.41seconds. Three
controlled late mutations then failed to reject: a staged query, metadata or
weight file could change after the child exited. The supervisor now rechecks
all those inputs before outer success. The47-case affected suite passed in
51.43seconds with four retained native-vendor warnings. A first affected command
named an absent test file and ran no tests; its failure log remains.

The positive large-query check uses an actual native P13 model on a synthetic
fixture and65536feature-only rows, verifying all64chunks, complete output IDs,
constant-model values, exact preservation of original model artifacts and no
Torch import. A separate nonconstant function verifies chunk boundaries and
per-row output order. Forbidden-label/object/source guards and an actual
one-byte RSS stop are retained. Other adapter replay/candidate regressions were
included; all families' full-size large-query qualification remains required.
No real teacher, native anchor, HPO, quality score or protected outcome was added
by these software checks. See [the receipt](WP10-native-query-inference-receipt.json).
