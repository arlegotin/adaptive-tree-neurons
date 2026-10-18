# Resource admission and exact continuation

Production M0/M1 admission uses immutable native measurements plus one frozen
allocation per full `(cell,teacher,repeat)` transaction, identified by the
canonical hash of those three named fields. `case_retained_bytes` must cover
exactly the expanded cases. `retry_reserve_bytes` must cover the largest case;
the initial `forecast_remaining_bytes` covers every case, that rolling reserve,
and any extra fixed overhead. This is an operational forecast, not a statistical
guarantee about future load. All measured source/runtime/workload gates remain.

A new run checks its full forecast before creating output. Under the study lease,
it publishes `resource-admission-binding.json`, preserving the original admission
content and raw SHA with source/plan identities. Each readiness decision has a
new immutable `resource-availability-N.json`, tied to the admission digest and
current ledger event hash. No readiness record changes paid scientific attempts.

On resume the runner first acquires the OS lease and authenticates the saved
plan and complete event chain. The validator reauthenticates it, rejects changed
admission content, and releases only whole terminal-transaction allocations.
An unfinished transaction retains its complete allocation even if some roles
finished, partial files occupy disk, or its first worker was interrupted. No
arbitrary output byte or completed-role count buys capacity credit. Terminal
failures keep their scientific status; releasing their future allocation cannot
manufacture a missing prediction or a successful primary comparison.

An interrupted transaction's full allocation pays its one remaining rerun. Only
untouched pending transactions can still require an additional retry reserve;
reserve their largest allocation. Once every transaction is terminal, worker and
retry allocations are released while fixed publication overhead/free-space
margin remain. This avoids either demanding the original whole-study space
again or reserving an unauthorized third attempt.

Insufficient space at resume publishes a separate failed readiness decision and
leaves attempts/costs intact. A result-publication error cannot replace the
original resource error. An interrupted initial publication can establish its
binding only if the authenticated ledger has no paid events. A missing original
binding after work has begun is an error, never an inferred migration.

This changes the new research admission contract, not historical schemas or the
numerical training recipe. No production M0/M1 admission record had been issued
before the change. The existing132b250 qualification measurements remain bound to
that source. Obtain a fresh backend/measurement identity for the final source
before admitting a full matrix; never relabel the old receipt.
