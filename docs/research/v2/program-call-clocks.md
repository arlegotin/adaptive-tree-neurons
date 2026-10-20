# Full method-call observations

The containing serial program now measures each `execute_episode` call through
its actual return or raised exception. This includes the caller's final
publication and return, which its inner completion record excludes. The new
source/program/job/episode/domain-bound record lives under `call-clocks`, outside
the caller's fresh output. Its own publication and queue bookkeeping remain
common overhead contained in the outer program observation.

The read-only helper checks exact local identity, bounds, bytes, exception
disposition and any retained caller proof. It compares interval containment only
for observed matching clock domains. A failure before caller output records no
closed caller; a hard crash without this record remains missing. Recovery after
a driver exception can reopen a closed caller without spawning a replacement or
overwriting its call observation. No native, complete-cost or T authority is
granted by this helper. The later cost auditor must require all genuine records.

The affected thirteen-module command passed **214 checks in 30.15 seconds**.
The missing-module failures and a first implementation error—assuming the
atomic writer created parent directories—are retained in the
[receipt](WP10-program-call-clock-receipt.json). The new directory is now created
as part of publication after the call interval ends.

A software-only capture contains one delayed return and exceptions before and
after a real resource-denied caller. The delayed return ends30,839,208ns after
the inner completion boundary, covering the deliberately added25ms and final
caller bookkeeping. The before-caller exception has no invented closure. No
model worker, fit, real transfer labels or T is used; no native/global success
is fabricated. Actual complete-qualified execution and the pre-T cost join,
final protected evaluator/audit, deployment, CLI/readiness and branch review
remain before production.

Containing program invocations now also commit all full-call file hashes under
the new `*-invocation-clock-2` schema. This closes the byte chain from a call's
return measurement through the invocation and owned worker body. The separate
binding regression passed215affected checks after its initial failure. Actual
complete cost qualification is still unfinished.
