# Retaining the primary access history at freeze

Native qualification and model auditors may legitimately recheck primary V
after a global freeze. Comparing their later complete log hash with the earlier
hash would invalidate the study even when its models and rules stayed fixed.
`frozen_primary_access_prefix` instead retains the exact byte/event boundary at
freeze and requires that prefix to remain unchanged.

The complete current log must still contain only valid primary-role events.
Later secondary folds or T roles cannot enter that channel, even if their
records are rehashed. The future released-data API needs its own protocol-bound
journal. Legacy `TransferStore` role guards remain unchanged.

An existing log has a shared lease during capture or revalidation. A previously
absent log is explicitly an absence snapshot and grants no lease or permission;
the helper creates no file. New valid primary events may follow that empty
prefix. Integer counts, folds and boolean flags have strict types, rather than
relying on Python's equality between booleans and integers.

This authenticates recorded API history only. It opens no arrays, grants no
secondary/protected access and cannot prove absence of unlogged filesystem
access. The actual protocol must validate native/model prerequisites before
holding log leases, avoiding an audit that tries to append while holding its
own shared lease. The [primary global barrier](practical-global-barrier.md)
now uses that ordering and grants no final-test permission. Its actual qualified
execution and the full protected protocol remain pending.

The affected four-module suite passed **67 checks in 10.64 seconds**. Thirteen
missing-interface failures preceded the first 64-check run. Three further
failures exposed boolean/integer equality accepting altered metadata; strict
type checks corrected them. The [receipt](WP09-transfer-access-prefix-receipt.json)
retains both failures and the final source. A read-only toy-log capture reopens
the unchanged first event after a later primary event, without role arrays,
native fits or T.
