# Independent closed-model prediction replay

This WP10 implementation slice checks a retained model and its feature-only
query in a fresh method-specific process. It consumes externally committed
request, model metadata, query and prediction files; it never fits, opens
F/G/V/T labels, chooses a model or grants native qualification. The eventual
production evaluator must additionally authenticate generation, stopping,
qualification, protocol and complete paid attempt history.

The strict `research/v2-prediction-replay-request` names its engineering purpose,
family, exact scientific source, semantic feature schema and relative snapshot,
query and saved-prediction references. All references stay below the request's
directory without symlinks. Snapshot metadata uses the native state/manifest
format. Recheck every snapshot file and request/query/prediction commitment
after replay. Existing native closed loaders retain their source/recipe guards.

Query archives contain exactly original-coordinate float64 X, unique int64
row_id, aligned int64 group_id and optional increasing actual timestamps.
Predictions contain exactly the matching identities and finite float32/float64
prediction. Preserve raw precision and row order; do not reuse deployment's
float32 casting or its minimum1024-row timing workload. Reject mismatches before
method import. The model then predicts once with its native default batching;
require identical dtype and values to the retained output. A difference is a
failed replay, never permission to refit or substitute a tolerance or seed.

The entry point configures the one-thread native treatment after inputs pass,
imports only the declared adapter, retains model/input commitments and a
terminal success/failure record, and refuses existing output. It measures its
body; the outer supervisor must own full process wall time, deadline, RSS and
reaping. All quality/protected/model-generation authority stays false. The
first verification uses artificial closed P13, P00 and P12 models only, plus
malformed/forbidden-label/identity/dtype/path/drift sentinels. No real transfer
V labels or protected outcomes are needed.

Executed verification:28missing-component RED checks preceded28passes in37.24s.
The first combined run passed87checks and exposed one older deployment-fixture
failure: an in-process P00 fit inherited sklearn's extra OpenMP library after a
NODE test. Its fresh loader correctly rejected that different native image set.
The diagnostic retained two saved pools versus one fresh pool and the extra
sklearn libomp image. P00 fixture fitting now happens in a fresh process, matching
production. Native runtime guards and model code were preserved.

The final affected run reversed the fixture order to exercise this boundary and
passed88checks in75.51s, with six retained upstream NODE warnings. It includes
fresh P13 grouped/temporal, P00 and P12 replay; exact output dtype/value rejection;
forbidden label/identity/path/hash checks; and late input/snapshot drift. The
replay queries have37rows in float64, so deployment's1024-row/float32 workload
cannot conceal a precision change here. This is focused software verification,
not a full-repository pass or a new predictive result. The
[receipt](WP10-independent-prediction-replay-receipt.json) binds source, commands,
all failures, diagnostic, fixtures and the final raw inventory.
