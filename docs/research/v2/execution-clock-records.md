# Explicit execution clock records

Caller completions and serial-program invocations now retain monotonic origins,
ends and observed clock domains. The domain binds the clock implementation,
hashed host/system identity and kernel boot identity. A failed kernel query is
recorded as unknown. Different or unknown domains cannot support comparisons of
absolute timestamps; separately observed elapsed durations remain usable.

Readers validate the stored domain without requiring the machine's current boot
to match. A resumed program may therefore retain prior closed costs after a
reboot. Program readers check the source, immutable plan, exact receipt bytes,
interval arithmetic and committed journal prefix. Caller domains are also bound
to their linked start record. Neither reader starts a worker or opens labels.

These are new `*-clock-1` record schemas. Earlier receipts remain unchanged under
their original source; there is no automatic migration. The recorded boundaries
still exclude their own final publication/return. They do **not** prove complete
cold-process or pipeline cost, native qualification, production readiness or T
permission. The owned outer program envelope and cost auditor remain required.

The affected eleven-module command passed **181 checks in 19.94 seconds**. It
covers real closed caller/program records, altered hashes/source/bounds/domains,
driver failure, journal growth, reboot-independent readback, import isolation
and existing cold-worker/resource/recovery behavior. Two existing recovery tests
initially blocked the new bounded Darwin kernel query along with model workers;
their guards now permit only that exact OS query and continue forbidding every
replacement worker. The first missing-module failures and regex warning are
retained in the [receipt](WP10-clock-records-receipt.json).

Two software-only captures retain actual prespawn resource denials on the same
job, with no model worker, fit or real transfer labels. The sandbox capture
truthfully has unknown domains. The separately permitted kernel-query capture
has observed matching domains and verifies both callers are contained in their
respective program intervals. Its independent caller sum is 67,125,458 ns,
exactly matching the queue's known total. No native/global success is fabricated;
actual complete-qualified program execution remains unrun. Scientific results
and the ongoing held-source native baseline treatment are unchanged.
