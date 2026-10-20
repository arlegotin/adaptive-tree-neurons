# Binding live admission to a qualified primary worker

Resource-bound episodes use the explicit
`research/v2-primary-search-resource-episode-request` schema and an immutable
forecast file reference. Missing or invalid references fail; a resource-bound
request cannot silently fall back to the earlier ungated component schema.
The older schema remains a separate engineering component and supplies no
production resource admission.

The cached forecast reader recomputes actual current-source qualification,
source/runtime/support/split bindings and measured resource envelopes before
opening the cache. Exact cache bytes, fields, policies and observations must
match. Only the stored forecast-preparation duration is allowed to differ from
the current revalidation duration. That duration is an observation, not a cost
authority: the current parent/worker clock pays the entire revalidation again.
Custom margins remain at least 1 GiB and are recomputed rather than trusted.

Invocation preflight protects the forecast path and validates its contents
before output. The actual worker likewise validates its resource context before
creating its worker output. It copies that bound context and feeds a fresh host
observation into the pre-proposal admission hook on every trial. The journal
records the forecast reference/hash, job identity and complete live decision.
Disk/RAM shortage closes a resource pause through the existing exact replay
contract; no model slot, new seed, shortened fit or changed dataset is invented.

The affected five-module suite passed **98 checks in 5.42 seconds**. Eight initial
failures covered missing APIs/bootstrap/gate support; three already-valid
negative schema checks passed. The first affected run then passed 27 checks.
An actual cold-controller fixture reads the host's resources and denies an
explicitly unqualified, deliberately impossible synthetic storage envelope,
closing with zero fits. The impossible envelope tests control flow; its numbers
are not a measured native forecast. Pure metadata tests grant no qualification.
No complete actual-qualified public success was stubbed or executed.

The [receipt](WP08-primary-resource-binding-receipt.json) retains the evidence.
Production caller launch, complete outer-cost reconciliation, final executable
readiness/global procedure freeze and protected evaluation remain unfinished.
This binding does not authorize secondary folds or T. Existing scientific
results and held-source NODE jobs are unchanged.
