# Pre-freeze transfer access evidence

WP09 needs to check recorded data access before admitting its final freeze. Add
`research/transfer_access_v2.py` with
`audit_pre_freeze_access(root, expected_manifest_sha256, expected_access_sha256=None)`
and a read-only shared-lease context `primary_access_snapshot` using the same
arguments. This component authenticates metadata and API access records only.
It never opens an observation archive or a target array, releases secondary
development data, or authorizes protected evaluation.

Require an externally pinned ordinary manifest and its internal canonical hash,
the fixed primary fold (Bike4, others0), five consistent split hashes/flags, and
the exact existing access-event schema/hash chain. Admit only the declared primary
F/G labels, V features/labels and V group/day identities. Reject secondary folds,
T/unknown roles, malformed/duplicate fields, wrong manifest identity, partial
lines and changed expected access bytes. Preserve all files. A busy append lease
fails rather than reading a partial record. Hold a shared lease for the complete
context so a later freeze can combine its checks with publication. Recheck the
manifest and log before releasing that context.

An absent access log is an unleased absence snapshot, not proof that nothing was
read. An existing valid log proves only what the audited API recorded. Return its
byte hash, chain endpoint, event/role counts, raw-identity flag and lease/absence
status. Always retain `protected_access_authorized=False`,
`freeze_readiness_qualified=False` and an explicit statement that unlogged direct
filesystem access is not authenticated. The complete readiness, primary model
hashes and final protocol remain separate prerequisites.

Verification uses artificial sealed Energy metadata, actual existing API log
appends, forbidden-NPZ sentinels, corruption/secondary/T cases, shared-lease and
post-snapshot append checks. A later read-only audit of the six real access logs
may retain these same limited facts without reading any predictive outcome.

Verification completed16missing-module failures, then46passing checks in1.25s.
Two additional malformed-dataset cases exposed a TypeError before the intended
metadata rejection; a string-type check corrected it. The final affected run
passed49checks in1.26s, including Bike's primary fold4, shared-lease exclusion,
external-frontier drift and existing transfer/scoring fixtures.

A retained read-only command audited135recorded API events across the six real
prepared datasets: Airfoil35, Concrete19, CCPP20, Parkinsons19, Bike20 and Energy22.
Every recorded event declared only primary F/G labels and V features. This audit
opened no observation archive, V label or T label. It establishes valid recorded
history at that endpoint; it does not prove the absence of unlogged access,
count135fits, or complete any readiness gate. The active native fit and M2
producer retain their separate source identities. See
[the receipt](WP09-pre-freeze-transfer-access-receipt.json).
